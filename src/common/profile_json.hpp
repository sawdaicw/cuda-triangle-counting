#pragma once
#include "graph.hpp"
#include <ostream>

namespace tc {
// The outer timer also covers function entry/exit and temporary array cleanup.
// Do not silently drop those costs or assume stage sums equal the whole call.
inline void write_preprocess_profile(std::ostream& out, const PreprocessProfile& p, double outer_ms) {
    out << "{\"read_parse_normalize_ms\":" << p.read_parse_normalize_ms
        << ",\"vertex_sort_unique_ms\":" << p.vertex_sort_unique_ms
        << ",\"edge_sort_unique_ms\":" << p.edge_sort_unique_ms
        << ",\"remap_degree_ms\":" << p.remap_degree_ms
        << ",\"orient_row_counts_ms\":" << p.orient_row_counts_ms
        << ",\"csr_scan_ms\":" << p.csr_scan_ms
        << ",\"csr_fill_ms\":" << p.csr_fill_ms
        << ",\"neighbor_sort_ms\":" << p.neighbor_sort_ms
        << ",\"accounted_ms\":" << p.accounted_ms()
        << ",\"unattributed_ms\":" << outer_ms - p.accounted_ms() << '}';
}
} // namespace tc
