#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tc {
using Vertex = std::uint32_t;
using Offset = unsigned long long;
using Count = unsigned long long;
static_assert(sizeof(Offset) == 8, "64-bit offsets required");
struct Graph {
    std::vector<Offset> offsets;
    std::vector<Vertex> neighbors;
    std::size_t vertices() const { return offsets.size() - 1; }
};

// Optional diagnostic timings, not a different preprocessing algorithm.
struct PreprocessProfile {
    double read_parse_normalize_ms = 0;
    double vertex_sort_unique_ms = 0;
    double edge_sort_unique_ms = 0;
    double remap_degree_ms = 0;
    double orient_row_counts_ms = 0;
    double csr_scan_ms = 0;
    double csr_fill_ms = 0;
    double neighbor_sort_ms = 0;
    double accounted_ms() const {
        return read_parse_normalize_ms + vertex_sort_unique_ms + edge_sort_unique_ms
            + remap_degree_ms + orient_row_counts_ms + csr_scan_ms + csr_fill_ms + neighbor_sort_ms;
    }
};

// IDs are sorted before remapping, so dense IDs preserve original ID tie breaks.
// Vertices appearing only in self-loops remain present, like the Python reference.
inline Graph read_graph(const std::string& path, PreprocessProfile* profile = nullptr) {
    using ProfileClock = std::chrono::steady_clock;
    if (profile) *profile = {};
    auto previous = profile ? ProfileClock::now() : ProfileClock::time_point{};
    const auto mark = [&](double PreprocessProfile::*field) {
        if (!profile) return;
        const auto now = ProfileClock::now();
        profile->*field = std::chrono::duration<double, std::milli>(now - previous).count();
        previous = now;
    };
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open input: " + path);
    std::vector<std::pair<std::int64_t, std::int64_t>> edges;
    std::vector<std::int64_t> ids;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#' || line[first] == '%') continue;
        std::istringstream stream(line);
        std::int64_t u, v;
        std::string extra;
        if (!(stream >> u >> v) || (stream >> extra))
            throw std::runtime_error("Line " + std::to_string(line_number) + ": expected two signed 64-bit integer endpoints (plain text, no header)");
        ids.push_back(u);
        ids.push_back(v);
        if (u != v) edges.emplace_back(std::min(u, v), std::max(u, v));
    }
    if (input.bad()) throw std::runtime_error("Error reading input");
    mark(&PreprocessProfile::read_parse_normalize_ms);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    if (ids.size() > std::numeric_limits<Vertex>::max())
        throw std::runtime_error("Too many vertices for 32-bit vertex indices");
    mark(&PreprocessProfile::vertex_sort_unique_ms);
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    mark(&PreprocessProfile::edge_sort_unique_ms);
    std::vector<std::pair<Vertex, Vertex>> dense;
    dense.reserve(edges.size());
    std::vector<Offset> degree(ids.size(), 0);
    for (const auto& edge : edges) {
        const Vertex u = static_cast<Vertex>(std::lower_bound(ids.begin(), ids.end(), edge.first) - ids.begin());
        const Vertex v = static_cast<Vertex>(std::lower_bound(ids.begin(), ids.end(), edge.second) - ids.begin());
        dense.emplace_back(u, v);
        ++degree[u]; ++degree[v];
    }
    mark(&PreprocessProfile::remap_degree_ms);
    Graph graph;
    graph.offsets.assign(ids.size() + 1, 0);
    for (auto& edge : dense) {
        auto& u = edge.first;
        auto& v = edge.second;
        if (degree[u] > degree[v] || (degree[u] == degree[v] && u > v)) std::swap(u, v);
        ++graph.offsets[static_cast<std::size_t>(u) + 1];
    }
    mark(&PreprocessProfile::orient_row_counts_ms);
    for (std::size_t i = 1; i < graph.offsets.size(); ++i) graph.offsets[i] += graph.offsets[i - 1];
    mark(&PreprocessProfile::csr_scan_ms);
    graph.neighbors.resize(dense.size());
    auto cursor = graph.offsets;
    for (const auto& edge : dense) graph.neighbors[cursor[edge.first]++] = edge.second;
    mark(&PreprocessProfile::csr_fill_ms);
    // Orientation uses (degree,id); intersection uses numeric ID order.
    for (std::size_t u = 0; u < ids.size(); ++u)
        std::sort(graph.neighbors.begin() + graph.offsets[u], graph.neighbors.begin() + graph.offsets[u + 1]);
    mark(&PreprocessProfile::neighbor_sort_ms);
    return graph;
}

#ifdef __CUDACC__
#define TC_HD __host__ __device__
#else
#define TC_HD
#endif
// Identical serial counting logic is compiled for the host and the device.
TC_HD inline Count count_serial(std::size_t n, const Offset* offsets, const Vertex* neighbors) {
    Count total = 0;
    for (std::size_t u = 0; u < n; ++u) {
        for (Offset edge = offsets[u]; edge < offsets[u + 1]; ++edge) {
            const Vertex v = neighbors[edge];
            Offset a = offsets[u], b = offsets[v];
            while (a < offsets[u + 1] && b < offsets[static_cast<std::size_t>(v) + 1]) {
                if (neighbors[a] < neighbors[b]) ++a;
                else if (neighbors[b] < neighbors[a]) ++b;
                else { ++total; ++a; ++b; }
            }
        }
    }
    return total;
}
#undef TC_HD
} // namespace tc
