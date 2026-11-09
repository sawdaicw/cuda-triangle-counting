#include "../common/preprocess.hpp"
#include "../common/profile_json.hpp"
#include <iomanip>
#include <iostream>
int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error("Usage: tc_cpu_serial INPUT.txt [--profile-preprocess] [--parser legacy|fast|omp] [--mapping fused|serial|omp] [--sort serial|omp] [--threads N] [--check-csr]");
        bool profiling = false;
        tc::PreprocessOptions options;
        for (int i = 2; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "--profile-preprocess") profiling = true;
            else if (!tc::consume_preprocess_option(flag, i, argc, argv, options)) throw std::runtime_error("Unknown option: " + flag);
        }
        tc::PreprocessProfile profile; tc::PreprocessReport report;
        const auto begin = std::chrono::steady_clock::now();
        const auto graph = tc::read_graph_configured(argv[1], options, profiling ? &profile : nullptr, report);
        const auto ready = std::chrono::steady_clock::now();
        const auto count = tc::count_serial(graph.vertices(), graph.offsets.data(), graph.neighbors.data());
        const auto end = std::chrono::steady_clock::now();
        if (options.check_csr) tc::check_csr_against_legacy(argv[1], graph);
        std::cout << std::setprecision(10) << "{\"backend\":\"cpu_serial\",\"vertices\":" << graph.vertices()
                  << ",\"undirected_edges\":" << graph.neighbors.size() << ",\"triangles\":" << count
                  << ",\"preprocess_ms\":" << std::chrono::duration<double, std::milli>(ready - begin).count()
                  << ",\"count_ms\":" << std::chrono::duration<double, std::milli>(end - ready).count();
        if (profiling) {
            std::cout << ",\"preprocess_profile\":";
            tc::write_preprocess_profile(std::cout, profile, std::chrono::duration<double, std::milli>(ready - begin).count());
        }
        std::cout << ",\"preprocess_options\":";
        tc::write_preprocess_options(std::cout, options, report, profiling);
        std::cout << "}\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
