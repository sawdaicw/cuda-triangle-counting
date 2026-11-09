#include "../src/common/edge_binary.hpp"
#include <iostream>
#include <cmath>
#include <random>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
int main(int argc, char** argv) {
    try {
        if (argc == 2) {
            tc::PreprocessProfile profile;
            profile.read_parse_normalize_ms = 999999;
            const auto plain = tc::read_graph(argv[1]);
            const auto measured = tc::read_graph(argv[1], &profile);
            require(plain.offsets == measured.offsets && plain.neighbors == measured.neighbors,
                    "profiling changed exact CSR arrays");
            require(std::isfinite(profile.accounted_ms()) && profile.accounted_ms() >= 0,
                    "invalid preprocessing time");
            // Reusing a profile object resets it; fields are not accumulated.
            profile.neighbor_sort_ms = 999999;
            const auto again = tc::read_graph(argv[1], &profile);
            require(again.offsets == plain.offsets && again.neighbors == plain.neighbors,
                    "reused profiling changed CSR");
            require(profile.neighbor_sort_ms < 999999, "profile was not overwritten");
        }
        // Check every slice pair of a sorted array (including empty, identical,
        // disjoint, and very unequal slices) against direct enumeration.
        const std::vector<tc::Vertex> items{0, 3, 5, 8, 10, 18, 23, 90};
        for (std::size_t a = 0; a <= items.size(); ++a)
        for (std::size_t ae = a; ae <= items.size(); ++ae)
        for (std::size_t b = 0; b <= items.size(); ++b)
        for (std::size_t be = b; be <= items.size(); ++be) {
            tc::Count expected = 0;
            for (auto i = a; i < ae; ++i) for (auto j = b; j < be; ++j)
                expected += items[i] == items[j];
            require(tc::intersect_binary(items.data(), a, ae, b, be) == expected, "slice intersection mismatch");
        }
        // Build 100 random graphs independently, orient, then simulate exactly
        // the baseline's grid-stride ownership for different thread counts.
        for (unsigned seed = 0; seed < 100; ++seed) {
            constexpr std::size_t n = 24;
            std::mt19937 rng(seed);
            bool adjacency[n][n]{};
            std::vector<unsigned> degree(n, 0);
            for (std::size_t u = 0; u < n; ++u) for (std::size_t v = u + 1; v < n; ++v)
                if (rng() % 5 == 0) { adjacency[u][v] = adjacency[v][u] = true; ++degree[u]; ++degree[v]; }
            tc::Count oracle = 0;
            for (std::size_t u = 0; u < n; ++u) for (std::size_t v = u + 1; v < n; ++v)
                for (std::size_t w = v + 1; w < n; ++w) oracle += adjacency[u][v] && adjacency[u][w] && adjacency[v][w];
            tc::Graph graph; graph.offsets.push_back(0);
            for (std::size_t u = 0; u < n; ++u) {
                for (std::size_t v = 0; v < n; ++v)
                    if (adjacency[u][v] && (degree[u] < degree[v] || (degree[u] == degree[v] && u < v)))
                        graph.neighbors.push_back(static_cast<tc::Vertex>(v));
                graph.offsets.push_back(graph.neighbors.size());
            }
            const auto sources = tc::edge_sources(graph);
            require(sources.size() == graph.neighbors.size(), "metadata size mismatch");
            for (const tc::Offset threads : {1, 2, 8, 31, 32, 255, 256, 512}) {
                tc::Count total = 0;
                std::vector<unsigned> visits(sources.size(), 0);
                for (tc::Offset tid = 0; tid < threads; ++tid)
                    for (tc::Offset e = tid; e < sources.size(); e += threads) {
                        ++visits[e];
                        const auto u = sources[e], v = graph.neighbors[e];
                        require(e >= graph.offsets[u] && e < graph.offsets[u + 1], "wrong source metadata");
                        total += tc::intersect_binary(graph.neighbors.data(), graph.offsets[u], graph.offsets[u + 1],
                                                     graph.offsets[v], graph.offsets[v + 1]);
                    }
                require(total == oracle, "binary/grid-stride count mismatch");
                require(std::all_of(visits.begin(), visits.end(), [](auto count) { return count == 1; }), "edge lost or duplicated");
            }
        }
        std::cout << "PASS: binary intersections, source metadata, and grid-stride ownership on 100 random graphs\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
