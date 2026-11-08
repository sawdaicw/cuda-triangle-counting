#include "../common/graph.hpp"
#include <chrono>
#include <iostream>
int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "Usage: tc_cpu_serial INPUT.txt\n"; return 1; }
    try {
        const auto begin = std::chrono::steady_clock::now();
        const auto graph = tc::read_graph(argv[1]);
        const auto ready = std::chrono::steady_clock::now();
        const auto count = tc::count_serial(graph.vertices(), graph.offsets.data(), graph.neighbors.data());
        const auto end = std::chrono::steady_clock::now();
        std::cout << "{\"backend\":\"cpu_serial\",\"vertices\":" << graph.vertices()
                  << ",\"undirected_edges\":" << graph.neighbors.size() << ",\"triangles\":" << count
                  << ",\"preprocess_ms\":" << std::chrono::duration<double, std::milli>(ready - begin).count()
                  << ",\"count_ms\":" << std::chrono::duration<double, std::milli>(end - ready).count() << "}\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
