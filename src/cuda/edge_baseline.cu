#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include "../common/edge_binary.hpp"
#include "../common/profile_json.hpp"
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
struct Options { std::string input; int repeats = 10, warmup = 2, blocks = 0; bool verify = false, profile_preprocess = false; };
int integer(const std::string& text, int low, int high) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Expected a nonnegative decimal integer: " + text);
    const auto value = std::stoull(text);
    if (value < static_cast<unsigned>(low) || value > static_cast<unsigned>(high))
        throw std::runtime_error("Option integer out of range: " + text);
    return static_cast<int>(value);
}
Options parse(int argc, char** argv) {
    if (argc < 2) throw std::runtime_error("Usage: tc_gpu_edge_baseline INPUT.txt [--verify] [--profile-preprocess] [--repeat N] [--warmup N] [--blocks N]");
    Options o; o.input = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--verify") o.verify = true;
        else if (flag == "--profile-preprocess") o.profile_preprocess = true;
        else if (flag == "--repeat" || flag == "--warmup" || flag == "--blocks") {
            if (++i == argc) throw std::runtime_error("Missing value for " + flag);
            if (flag == "--repeat") o.repeats = integer(argv[i], 1, 10000);
            if (flag == "--warmup") o.warmup = integer(argv[i], 0, 10000);
            if (flag == "--blocks") o.blocks = integer(argv[i], 1, std::numeric_limits<int>::max());
        } else throw std::runtime_error("Unknown option: " + flag);
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
struct Sample { float count_reduce_ms; double d2h_ms, execution_wall_ms; tc::Count count; };
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
        tc::PreprocessProfile profile;
        const auto begin = Clock::now();
        const auto graph = tc::read_graph(options.input, options.profile_preprocess ? &profile : nullptr);
        const auto graph_ready = Clock::now();
        const auto sources = tc::edge_sources(graph);
        const auto metadata_ready = Clock::now();
        check(cudaSetDevice(0));
        cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, 0));
        int runtime = 0, driver = 0;
        check(cudaRuntimeGetVersion(&runtime)); check(cudaDriverGetVersion(&driver));
        const auto needed = std::max<tc::Offset>(1, (graph.neighbors.size() + block_threads - 1) / block_threads);
        const int blocks = options.blocks ? options.blocks : static_cast<int>(
            std::min<tc::Offset>(needed, static_cast<tc::Offset>(prop.multiProcessorCount) * 8));
        if (blocks > prop.maxGridSize[0]) throw std::runtime_error("--blocks exceeds device grid limit");
        Buffer<tc::Offset> d_offsets(graph.offsets.size());
        Buffer<tc::Vertex> d_neighbors(graph.neighbors.size()), d_sources(sources.size());
        Buffer<tc::Count> d_sums(blocks), d_result(1);
        std::size_t scratch_bytes = 0;
        check(cub::DeviceReduce::Sum(nullptr, scratch_bytes, d_sums.data, d_result.data, blocks));
        Buffer<unsigned char> scratch(scratch_bytes);
        Event start, stop;
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
            count_edges<<<blocks, block_threads>>>(graph.neighbors.size(), d_offsets.data, d_neighbors.data, d_sources.data, d_sums.data);
            check(cudaGetLastError());
            check(cub::DeviceReduce::Sum(scratch.data, scratch_bytes, d_sums.data, d_result.data, blocks));
            check(cudaEventRecord(stop.data));
            check(cudaEventSynchronize(stop.data));
            float elapsed = 0; check(cudaEventElapsedTime(&elapsed, start.data, stop.data));
            const auto download = Clock::now();
            tc::Count count = 0;
            check(cudaMemcpy(&count, d_result.data, sizeof(count), cudaMemcpyDeviceToHost));
            const auto done = Clock::now();
            return Sample{elapsed, ms(download, done), ms(run_begin, done), count};
        };
        // A first complete one-shot result, before any benchmark warmup or reuse.
        const Sample first = run();
        const auto first_ready = Clock::now();
        for (int i = 0; i < options.warmup; ++i)
            if (run().count != first.count) throw std::runtime_error("Unstable warmup count");
        std::vector<double> samples;
        double exec_sum = 0;
        for (int i = 0; i < options.repeats; ++i) {
            const auto sample = run();
            if (sample.count != first.count) throw std::runtime_error("Unstable repeated count");
            samples.push_back(sample.count_reduce_ms);
            exec_sum += sample.execution_wall_ms;
        }
        auto sorted = samples; std::sort(sorted.begin(), sorted.end());
        const auto k = sorted.size() / 2;
        const double median = sorted.size() % 2 ? sorted[k] : (sorted[k - 1] + sorted[k]) / 2;
        const auto verify_begin = Clock::now();
        if (options.verify && first.count != tc::count_serial(graph.vertices(), graph.offsets.data(), graph.neighbors.data()))
            throw std::runtime_error("GPU baseline / CPU merge count mismatch");
        const double verify_ms = ms(verify_begin, Clock::now());
        const std::size_t bytes = d_offsets.bytes + d_neighbors.bytes + d_sources.bytes + d_sums.bytes + d_result.bytes + scratch.bytes;
        std::cout << std::setprecision(10)
            << "{\"backend\":\"gpu_edge_binary\",\"input\":" << quote(options.input)
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
        if (options.profile_preprocess) {
            std::cout << ",\"preprocess_profile\":";
            tc::write_preprocess_profile(std::cout, profile, ms(begin, graph_ready));
        }
        std::cout << "}\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
