#include "../src/common/preprocess.hpp"
#include <iostream>
#include <cmath>
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Usage: tc_preprocess_equivalence INPUT.txt");
        const auto reference = tc::read_graph(argv[1]);
        std::size_t checks = 0;
        for (const auto& parser : {"legacy", "fast", "omp"})
        for (const auto& mapping : {"fused", "serial", "omp"})
        for (const auto& sorting : {"serial", "omp"})
        for (const int threads : {1, 3, 8}) {
            tc::PreprocessOptions options; options.parser = parser; options.mapping = mapping;
            options.sorting = sorting; options.threads = threads;
            tc::PreprocessProfile profile; tc::PreprocessReport report;
            const auto graph = tc::read_graph_configured(argv[1], options, &profile, report);
            if (reference.offsets != graph.offsets || reference.neighbors != graph.neighbors)
                throw std::runtime_error("CSR mismatch: " + options.parser + "/" + options.mapping + "/" + options.sorting + "/" + std::to_string(threads));
            if (!std::isfinite(profile.accounted_ms()) || profile.accounted_ms() < 0)
                throw std::runtime_error("Invalid profile time");
            ++checks;
        }
        std::cout << "PASS: " << checks << " configurations produce exactly identical offsets and neighbors\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
