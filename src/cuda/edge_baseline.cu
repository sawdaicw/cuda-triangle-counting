#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include "../common/edge_binary.hpp"
#include "../common/profile_json.hpp"
#include "../common/preprocess.hpp"
#include "../common/warp_tasks.hpp"
#include <memory>

#ifndef TC_EDGE_MODE
#define TC_EDGE_MODE 0
#endif
static_assert(TC_EDGE_MODE >= 0 && TC_EDGE_MODE <= 2, "Unknown edge backend");
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>

namespace {
constexpr int block_threads = 256;
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template<class T> struct Buffer {
    T* data = nullptr;
    const std::size_t bytes;
    explicit Buffer(std::size_t count) : bytes(std::max<std::size_t>(count, 1) * sizeof(T)) {
        check(cudaMalloc(reinterpret_cast<void**>(&data), bytes));
    }
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
struct Event {
    cudaEvent_t data{};
    Event() { check(cudaEventCreate(&data)); }
    ~Event() { cudaEventDestroy(data); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};
struct Options { std::string input; int repeats = 10, warmup = 2, blocks = 0, threshold = 32; bool verify = false, profile_preprocess = false, profile_gpu = false; tc::PreprocessOptions preprocess; };
int integer(const std::string& text, int low, int high) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Expected a nonnegative decimal integer: " + text);
    const auto value = std::stoull(text);
    if (value < static_cast<unsigned>(low) || value > static_cast<unsigned>(high))
        throw std::runtime_error("Option integer out of range: " + text);
    return static_cast<int>(value);
}
Options parse(int argc, char** argv) {
    if (argc < 2) throw std::runtime_error("Usage: " + std::string(argv[0]) + " INPUT.txt [--verify] [--profile-preprocess] [--repeat N] [--warmup N] [--blocks N] (warp/split: --profile-gpu; split: --threshold N)");
    Options o; o.input = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--verify") o.verify = true;
        else if (flag == "--profile-preprocess") o.profile_preprocess = true;
        else if (flag == "--profile-gpu" && TC_EDGE_MODE != 0) o.profile_gpu = true;
        else if (flag == "--threshold" && TC_EDGE_MODE == 2) {
            if (++i == argc) throw std::runtime_error("Missing value for " + flag);
            o.threshold = integer(argv[i], 1, std::numeric_limits<int>::max());
        }
        else if (flag == "--repeat" || flag == "--warmup" || flag == "--blocks") {
            if (++i == argc) throw std::runtime_error("Missing value for " + flag);
            if (flag == "--repeat") o.repeats = integer(argv[i], 1, 10000);
            if (flag == "--warmup") o.warmup = integer(argv[i], 0, 10000);
            if (flag == "--blocks") o.blocks = integer(argv[i], 1, std::numeric_limits<int>::max());
        } else if (!tc::consume_preprocess_option(flag, i, argc, argv, o.preprocess)) throw std::runtime_error("Unknown option: " + flag);
    }
    return o;
}
// Each edge is assigned to exactly one thread, using a 64-bit grid-stride loop.
// Every thread must reach BlockReduce, including threads with zero edge tasks.
__global__ void count_edges(tc::Offset m, const tc::Offset* offsets,
                            const tc::Vertex* neighbors, const tc::Vertex* sources,
                            tc::Count* block_sums) {
    const tc::Offset tid = static_cast<tc::Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
    const tc::Offset stride = static_cast<tc::Offset>(gridDim.x) * blockDim.x;
    tc::Count local = 0;
    for (tc::Offset e = tid; e < m; e += stride) {
        const tc::Vertex u = sources[e], v = neighbors[e];
        local += tc::intersect_binary(neighbors, offsets[u], offsets[static_cast<std::size_t>(u) + 1],
                                      offsets[v], offsets[static_cast<std::size_t>(v) + 1]);
    }
    using Reduce = cub::BlockReduce<tc::Count, block_threads>;
    __shared__ typename Reduce::TempStorage temp;
    const tc::Count sum = Reduce(temp).Sum(local);
    if (threadIdx.x == 0) block_sums[blockIdx.x] = sum;
}
#if TC_EDGE_MODE != 0
#include "warp_kernels.cuh"
#endif
struct Sample {
    float count_reduce_ms;
    double d2h_ms, execution_wall_ms;
    tc::Count count;
    float light_queue_ms = 0, warp_count_ms = 0, reduce_ms = 0;
};
std::string quote(const std::string& text) {
    std::ostringstream out; out << '"';
    for (const unsigned char c : text) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else out << c;
    }
    out << '"'; return out.str();
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        tc::PreprocessProfile profile; tc::PreprocessReport report;
        const auto begin = Clock::now();
        const auto graph = tc::read_graph_configured(options.input, options.preprocess, options.profile_preprocess ? &profile : nullptr, report);
        const auto graph_ready = Clock::now();
        const auto sources = tc::edge_sources(graph);
        const auto metadata_ready = Clock::now();
        check(cudaSetDevice(0));
        cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, 0));
        int runtime = 0, driver = 0;
        check(cudaRuntimeGetVersion(&runtime)); check(cudaDriverGetVersion(&driver));
        constexpr int tasks_per_block = TC_EDGE_MODE == 0 ? block_threads : block_threads / 32;
        const auto needed = std::max<tc::Offset>(1, (graph.neighbors.size() + tasks_per_block - 1) / tasks_per_block);
        const int blocks = options.blocks ? options.blocks : static_cast<int>(
            std::min<tc::Offset>(needed, static_cast<tc::Offset>(prop.multiProcessorCount) * 8));
        if (blocks > prop.maxGridSize[0]) throw std::runtime_error("--blocks exceeds device grid limit");
        Buffer<tc::Offset> d_offsets(graph.offsets.size());
        Buffer<tc::Vertex> d_neighbors(graph.neighbors.size()), d_sources(sources.size());
        const std::size_t sum_count = static_cast<std::size_t>(blocks) * (TC_EDGE_MODE == 2 ? 2 : 1);
        // Keep CUB's item count in its supported signed-int range.
        if (sum_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("Too many block partial sums");
        Buffer<tc::Count> d_sums(sum_count), d_result(1);
        std::unique_ptr<Buffer<tc::Offset>> queue, queue_size;
        if constexpr (TC_EDGE_MODE == 2) {
            queue = std::make_unique<Buffer<tc::Offset>>(graph.neighbors.size());
            queue_size = std::make_unique<Buffer<tc::Offset>>(1);
        }
        std::size_t scratch_bytes = 0;
        check(cub::DeviceReduce::Sum(nullptr, scratch_bytes, d_sums.data, d_result.data, static_cast<int>(sum_count)));
        Buffer<unsigned char> scratch(scratch_bytes);
        Event start, stop;
        std::unique_ptr<Event> after_light, after_count;
        if (options.profile_gpu) {
            after_count = std::make_unique<Event>();
            if constexpr (TC_EDGE_MODE == 2) after_light = std::make_unique<Event>();
        }
        const auto setup_ready = Clock::now();
        check(cudaMemcpy(d_offsets.data, graph.offsets.data(), d_offsets.bytes, cudaMemcpyHostToDevice));
        if (!graph.neighbors.empty()) {
            check(cudaMemcpy(d_neighbors.data, graph.neighbors.data(), graph.neighbors.size() * sizeof(tc::Vertex), cudaMemcpyHostToDevice));
            check(cudaMemcpy(d_sources.data, sources.data(), sources.size() * sizeof(tc::Vertex), cudaMemcpyHostToDevice));
        }
        const auto uploaded = Clock::now();
        auto run = [&]() {
            const auto run_begin = Clock::now();
            check(cudaEventRecord(start.data));
#if TC_EDGE_MODE == 0
            {
                count_edges<<<blocks, block_threads>>>(graph.neighbors.size(), d_offsets.data, d_neighbors.data, d_sources.data, d_sums.data);
                check(cudaGetLastError());
            }
#elif TC_EDGE_MODE == 1
            {
                count_warp_edges<<<blocks, block_threads>>>(graph.neighbors.size(), d_offsets.data, d_neighbors.data, d_sources.data, d_sums.data);
                check(cudaGetLastError());
                if (after_count) check(cudaEventRecord(after_count->data));
            }
#else
            {
                // Reset and rebuild the queue on EVERY run, including warmups.
                check(cudaMemsetAsync(queue_size->data, 0, sizeof(tc::Offset)));
                count_light_queue_heavy<<<blocks, block_threads>>>(graph.neighbors.size(), d_offsets.data, d_neighbors.data, d_sources.data,
                    d_sums.data, options.threshold, queue->data, queue_size->data);
                check(cudaGetLastError());
                if (after_light) check(cudaEventRecord(after_light->data));
                // Same default stream: queue writes finish before this kernel.
                // Always launch, even for an empty queue; no host count round-trip.
                count_warp_edges<<<blocks, block_threads>>>(graph.neighbors.size(), d_offsets.data, d_neighbors.data, d_sources.data,
                    d_sums.data + blocks, queue->data, queue_size->data);
                check(cudaGetLastError());
                if (after_count) check(cudaEventRecord(after_count->data));
            }
#endif
            check(cub::DeviceReduce::Sum(scratch.data, scratch_bytes, d_sums.data, d_result.data, static_cast<int>(sum_count)));
            check(cudaEventRecord(stop.data));
            check(cudaEventSynchronize(stop.data));
            float elapsed = 0; check(cudaEventElapsedTime(&elapsed, start.data, stop.data));
            float light_ms = 0, warp_ms = 0, reduce_ms = 0;
            if (options.profile_gpu) {
                if constexpr (TC_EDGE_MODE == 2) {
                    check(cudaEventElapsedTime(&light_ms, start.data, after_light->data));
                    check(cudaEventElapsedTime(&warp_ms, after_light->data, after_count->data));
                } else check(cudaEventElapsedTime(&warp_ms, start.data, after_count->data));
                check(cudaEventElapsedTime(&reduce_ms, after_count->data, stop.data));
            }
            const auto download = Clock::now();
            tc::Count count = 0;
            check(cudaMemcpy(&count, d_result.data, sizeof(count), cudaMemcpyDeviceToHost));
            const auto done = Clock::now();
            return Sample{elapsed, ms(download, done), ms(run_begin, done), count, light_ms, warp_ms, reduce_ms};
        };
        // A first complete one-shot result, before any benchmark warmup or reuse.
        const Sample first = run();
        const auto first_ready = Clock::now();
        for (int i = 0; i < options.warmup; ++i)
            if (run().count != first.count) throw std::runtime_error("Unstable warmup count");
        std::vector<double> samples, light_samples, warp_samples, reduce_samples;
        double exec_sum = 0;
        for (int i = 0; i < options.repeats; ++i) {
            const auto sample = run();
            if (sample.count != first.count) throw std::runtime_error("Unstable repeated count");
            samples.push_back(sample.count_reduce_ms);
            if (options.profile_gpu) {
                light_samples.push_back(sample.light_queue_ms);
                warp_samples.push_back(sample.warp_count_ms);
                reduce_samples.push_back(sample.reduce_ms);
            }
            exec_sum += sample.execution_wall_ms;
        }
        auto sorted = samples; std::sort(sorted.begin(), sorted.end());
        const auto k = sorted.size() / 2;
        const double median = sorted.size() % 2 ? sorted[k] : (sorted[k - 1] + sorted[k]) / 2;
        // Diagnostic readback is outside first_total_ms and GPU timing samples.
        const auto stats_begin = Clock::now();
        tc::Offset heavy_edges = 0;
        if constexpr (TC_EDGE_MODE == 2)
            check(cudaMemcpy(&heavy_edges, queue_size->data, sizeof(heavy_edges), cudaMemcpyDeviceToHost));
        if (heavy_edges > graph.neighbors.size()) throw std::runtime_error("Heavy queue overflow");
        const double stats_ms = ms(stats_begin, Clock::now());
        const auto verify_begin = Clock::now();
        if (options.preprocess.check_csr) tc::check_csr_against_legacy(options.input, graph);
        if (options.verify && first.count != tc::count_serial(graph.vertices(), graph.offsets.data(), graph.neighbors.data()))
            throw std::runtime_error("GPU baseline / CPU merge count mismatch");
        const double verify_ms = ms(verify_begin, Clock::now());
        const std::size_t queue_bytes = queue ? queue->bytes + queue_size->bytes : 0;
        const std::size_t bytes = d_offsets.bytes + d_neighbors.bytes + d_sources.bytes + d_sums.bytes + d_result.bytes + scratch.bytes + queue_bytes;
        constexpr const char* backend = TC_EDGE_MODE == 0 ? "gpu_edge_binary" :
            (TC_EDGE_MODE == 1 ? "gpu_warp_binary" : "gpu_split_binary");
        std::cout << std::setprecision(10)
            << "{\"backend\":" << quote(backend) << ",\"input\":" << quote(options.input)
            << ",\"device\":" << quote(prop.name) << ",\"compute_capability\":" << quote(std::to_string(prop.major) + "." + std::to_string(prop.minor))
            << ",\"cuda_runtime_version\":" << runtime << ",\"cuda_driver_version\":" << driver
            << ",\"vertices\":" << graph.vertices() << ",\"undirected_edges\":" << graph.neighbors.size()
            << ",\"triangles\":" << first.count << ",\"blocks\":" << blocks << ",\"threads_per_block\":" << block_threads
            << ",\"device_buffer_bytes\":" << bytes << ",\"cub_scratch_bytes\":" << scratch.bytes
            << ",\"preprocess_ms\":" << ms(begin, graph_ready) << ",\"task_metadata_ms\":" << ms(graph_ready, metadata_ready)
            << ",\"cuda_setup_ms\":" << ms(metadata_ready, setup_ready) << ",\"h2d_ms\":" << ms(setup_ready, uploaded)
            << ",\"first_count_reduce_ms\":" << first.count_reduce_ms << ",\"first_d2h_ms\":" << first.d2h_ms
            << ",\"first_execution_wall_ms\":" << first.execution_wall_ms << ",\"first_total_ms\":" << ms(begin, first_ready)
            << ",\"warmup\":" << options.warmup << ",\"repeat\":" << options.repeats
            << ",\"count_reduce_ms_samples\":[";
        for (std::size_t i = 0; i < samples.size(); ++i) std::cout << (i ? "," : "") << samples[i];
        std::cout << "],\"count_reduce_ms_min\":" << sorted.front() << ",\"count_reduce_ms_median\":" << median
            << ",\"count_reduce_ms_mean\":" << std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size()
            << ",\"repeated_execution_wall_ms_sum\":" << exec_sum
            << ",\"verified\":" << (options.verify ? "true" : "false") << ",\"verify_ms\":" << verify_ms;
        if constexpr (TC_EDGE_MODE != 0)
            std::cout << ",\"warps_per_block\":" << block_threads / 32
                      << ",\"gpu_stage_profiled\":" << (options.profile_gpu ? "true" : "false");
        if (options.profile_gpu) {
            std::cout << ",\"first_stages_ms\":{\"light_count_queue_ms\":" << first.light_queue_ms
                << ",\"warp_count_ms\":" << first.warp_count_ms << ",\"global_reduce_ms\":" << first.reduce_ms << "}";
            const auto write_samples = [&](const char* name, const std::vector<double>& values) {
                std::cout << "," << quote(name) << ":[";
                for (std::size_t i = 0; i < values.size(); ++i) std::cout << (i ? "," : "") << values[i];
                std::cout << "]";
            };
            write_samples("light_count_queue_ms_samples", light_samples);
            write_samples("warp_count_ms_samples", warp_samples);
            write_samples("global_reduce_ms_samples", reduce_samples);
        }
        if constexpr (TC_EDGE_MODE == 2) {
            std::cout << ",\"threshold\":" << options.threshold << ",\"heavy_edges\":" << heavy_edges
                << ",\"light_edges_including_empty\":" << graph.neighbors.size() - heavy_edges
                << ",\"heavy_edge_fraction\":" << (graph.neighbors.empty() ? 0.0 : static_cast<double>(heavy_edges) / graph.neighbors.size())
                << ",\"queue_capacity_edges\":" << graph.neighbors.size() << ",\"queue_buffer_bytes\":" << queue_bytes
                << ",\"stats_readback_ms\":" << stats_ms;
        }
        if (options.profile_preprocess) {
            std::cout << ",\"preprocess_profile\":";
            tc::write_preprocess_profile(std::cout, profile, ms(begin, graph_ready));
        }
        std::cout << ",\"preprocess_options\":";
        tc::write_preprocess_options(std::cout, options.preprocess, report, options.profile_preprocess);
        std::cout << "}\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
