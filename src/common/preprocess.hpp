#pragma once
#include "graph.hpp"
#include <ostream>

namespace tc {
struct PreprocessOptions {
    std::string parser = "legacy"; // legacy / fast / omp
    std::string mapping = "fused"; // fused / serial / omp
    std::string sorting = "serial"; // serial / omp
    int threads = 1;
    bool check_csr = false;
};
struct PreprocessReport {
    int parser_threads = 1, mapping_threads = 1, sorting_threads = 1;
    std::size_t input_buffer_bytes = 0;
    double file_read_ms = 0, partition_ms = 0, parse_collect_ms = 0, merge_collect_ms = 0;
};
// Returns false for an option owned by the caller; may consume argv[++index].
bool consume_preprocess_option(const std::string& flag, int& index, int argc, char** argv, PreprocessOptions& options);
Graph read_graph_configured(const std::string& path, const PreprocessOptions& options,
                            PreprocessProfile* profile, PreprocessReport& report);
void check_csr_against_legacy(const std::string& path, const Graph& graph);
void write_preprocess_options(std::ostream& out, const PreprocessOptions& options,
                              const PreprocessReport& report, bool profiled);
} // namespace tc
