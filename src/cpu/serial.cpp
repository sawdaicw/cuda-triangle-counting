#include "../common/graph.hpp"
#include "../common/profile_json.hpp"
#include <iomanip>
#include <chrono>
#include <iostream>
int main(int argc, char** argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--profile-preprocess")) { std::cerr << "Usage: tc_cpu_serial INPUT.txt [--profile-preprocess]\n"; return 1; }
    try {
        const bool profiling = argc == 3;
        tc::PreprocessProfile profile;
        const auto begin = std::chrono::steady_clock::now();
        const auto graph = tc::read_graph(argv[1], profiling ? &profile : nullptr);
        const auto ready = std::chrono::steady_clock::now();
        const auto count = tc::count_serial(graph.vertices(), graph.offsets.data(), graph.neighbors.data());
        const auto end = std::chrono::steady_clock::now();
        std::cout << std::setprecision(10) << "{\"backend\":\"cpu_serial\",\"vertices\":" << graph.vertices()
                  << ",\"undirected_edges\":" << graph.neighbors.size() << ",\"triangles\":" << count
                  << ",\"preprocess_ms\":" << std::chrono::duration<double, std::milli>(ready - begin).count()
                  << ",\"count_ms\":" << std::chrono::duration<double, std::milli>(end - ready).count();
        if (profiling) {
            std::cout << ",\"preprocess_profile\":";
            tc::write_preprocess_profile(std::cout, profile, std::chrono::duration<double, std::milli>(ready - begin).count());
        }
        std::cout << "}\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
