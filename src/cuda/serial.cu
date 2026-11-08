#include "../common/graph.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

inline void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(std::size_t count) {
        check(cudaMalloc(reinterpret_cast<void**>(&data), std::max<std::size_t>(count, 1) * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
struct Event {
    cudaEvent_t value{};
    Event() { check(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};
__global__ void serial_kernel(std::size_t n, const tc::Offset* offsets,
                              const tc::Vertex* neighbors, tc::Count* result) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *result = tc::count_serial(n, offsets, neighbors);
}
int main(int argc, char** argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--verify")) {
        std::cerr << "Usage: tc_gpu_serial INPUT.txt [--verify]\n";
        return 1;
    }
    try {
        using Clock = std::chrono::steady_clock;
        const auto begin = Clock::now();
        const auto graph = tc::read_graph(argv[1]);
        const auto ready = Clock::now();
        // CUDA_VISIBLE_DEVICES selects the physical GPU; use logical device 0.
        check(cudaSetDevice(0));
        check(cudaFree(nullptr));
        DeviceBuffer<tc::Offset> offsets(graph.offsets.size());
        DeviceBuffer<tc::Vertex> neighbors(graph.neighbors.size());
        DeviceBuffer<tc::Count> result(1);
        Event start, stop;
        const auto upload_begin = Clock::now();
        check(cudaMemcpy(offsets.data, graph.offsets.data(), graph.offsets.size() * sizeof(tc::Offset), cudaMemcpyHostToDevice));
        if (!graph.neighbors.empty())
            check(cudaMemcpy(neighbors.data, graph.neighbors.data(), graph.neighbors.size() * sizeof(tc::Vertex), cudaMemcpyHostToDevice));
        const auto upload_end = Clock::now();
        check(cudaEventRecord(start.value));
        serial_kernel<<<1, 1>>>(graph.vertices(), offsets.data, neighbors.data, result.data);
        check(cudaGetLastError());
        check(cudaEventRecord(stop.value));
        check(cudaEventSynchronize(stop.value));
        float kernel_ms = 0;
        check(cudaEventElapsedTime(&kernel_ms, start.value, stop.value));
        tc::Count count = 0;
        const auto download_begin = Clock::now();
        check(cudaMemcpy(&count, result.data, sizeof(count), cudaMemcpyDeviceToHost));
        const auto end = Clock::now();
        const bool verify = argc == 3;
        if (verify && count != tc::count_serial(graph.vertices(), graph.offsets.data(), graph.neighbors.data()))
            throw std::runtime_error("GPU/CPU count mismatch");
        const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        const auto device_bytes = graph.offsets.size() * sizeof(tc::Offset)
            + std::max<std::size_t>(graph.neighbors.size(), 1) * sizeof(tc::Vertex) + sizeof(tc::Count);
        std::cout << "{\"backend\":\"gpu_serial\",\"blocks\":1,\"threads_per_block\":1,\"vertices\":" << graph.vertices()
                  << ",\"undirected_edges\":" << graph.neighbors.size() << ",\"triangles\":" << count
                  << ",\"device_buffer_bytes\":" << device_bytes
                  << ",\"preprocess_ms\":" << ms(begin, ready)
                  << ",\"h2d_ms\":" << ms(upload_begin, upload_end)
                  << ",\"kernel_ms\":" << kernel_ms << ",\"d2h_ms\":" << ms(download_begin, end)
                  << ",\"end_to_end_ms\":" << ms(begin, end)
                  << ",\"verified\":" << (verify ? "true" : "false") << "}\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
