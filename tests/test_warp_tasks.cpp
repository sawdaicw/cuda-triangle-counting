#include "../src/common/warp_tasks.hpp"
#include <iostream>
#include <random>

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void check_graph(const tc::Graph& graph, tc::Count oracle) {
    const auto sources = tc::edge_sources(graph);
    const auto m = sources.size();
    for (unsigned blocks : {1, 2, 3}) {
        const tc::Offset warps = blocks * 8;
        tc::Count fixed = 0;
        std::vector<unsigned> visits(m, 0);
        for (tc::Offset warp = 0; warp < warps; ++warp)
            for (tc::Offset e = warp; e < m; e += warps) {
                ++visits[e];
                const auto u = sources[e], v = graph.neighbors[e];
                for (unsigned lane = 0; lane < 32; ++lane)
                    fixed += tc::intersect_binary_lane(graph.neighbors.data(), graph.offsets[u], graph.offsets[u+1],
                                                       graph.offsets[v], graph.offsets[v+1], lane);
            }
        require(fixed == oracle, "fixed warp vs independent oracle");
        require(std::all_of(visits.begin(), visits.end(), [](unsigned n){ return n == 1; }), "warp task coverage");
        for (tc::Offset threshold : {1, 2, 16, 32, 64, 1000000}) {
            std::vector<tc::Offset> queue;
            tc::Count split = 0;
            for (tc::Offset tid = 0; tid < blocks * 256; ++tid)
                for (tc::Offset e = tid; e < m; e += blocks * 256) {
                    const auto u = sources[e], v = graph.neighbors[e];
                    const auto size = tc::shorter_forward_size(graph.offsets.data(), u, v);
                    if (size >= threshold) queue.push_back(e);
                    else split += tc::intersect_binary(graph.neighbors.data(), graph.offsets[u], graph.offsets[u+1],
                                                       graph.offsets[v], graph.offsets[v+1]);
                }
            // Atomic insertion order on GPU is unspecified: test another order.
            std::reverse(queue.begin(), queue.end());
            for (tc::Offset warp = 0; warp < warps; ++warp)
                for (tc::Offset q = warp; q < queue.size(); q += warps) {
                    const auto e = queue[q]; const auto u = sources[e], v = graph.neighbors[e];
                    for (unsigned lane = 0; lane < 32; ++lane)
                        split += tc::intersect_binary_lane(graph.neighbors.data(), graph.offsets[u], graph.offsets[u+1],
                                                           graph.offsets[v], graph.offsets[v+1], lane);
                }
            require(split == oracle, "split vs independent oracle");
        }
    }
}
int main() {
    try {
        const std::vector<tc::Vertex> items{0, 3, 5, 8, 10, 18, 23, 90};
        for (std::size_t a=0; a<=items.size(); ++a)
        for (std::size_t ae=a; ae<=items.size(); ++ae)
        for (std::size_t b=0; b<=items.size(); ++b)
        for (std::size_t be=b; be<=items.size(); ++be) {
            tc::Count expected = 0, actual = 0;
            for (auto i=a; i<ae; ++i) for (auto j=b; j<be; ++j) expected += items[i] == items[j];
            for (unsigned lane=0; lane<32; ++lane)
                actual += tc::intersect_binary_lane(items.data(), a, ae, b, be, lane);
            require(actual == expected, "lane partition intersection");
        }
        check_graph(tc::Graph{{0}, {}}, 0);
        // Independent adjacency/triple enumeration; includes a dense K96 so
        // short rows cross 31/32/33 and 63/64/65 lane and threshold boundaries.
        for (unsigned seed=0; seed<101; ++seed) {
            const unsigned n = seed == 100 ? 96 : 24;
            std::mt19937 rng(seed);
            std::vector<std::vector<bool>> adj(n, std::vector<bool>(n));
            std::vector<unsigned> degree(n);
            for (unsigned u=0; u<n; ++u) for (unsigned v=u+1; v<n; ++v)
                if (seed == 100 || rng()%5 == 0) { adj[u][v]=adj[v][u]=true; ++degree[u]; ++degree[v]; }
            tc::Count oracle = 0;
            for (unsigned u=0; u<n; ++u) for (unsigned v=u+1; v<n; ++v)
                for (unsigned w=v+1; w<n; ++w) oracle += adj[u][v] && adj[u][w] && adj[v][w];
            tc::Graph graph{{0}, {}};
            for (unsigned u=0; u<n; ++u) {
                for (unsigned v=0; v<n; ++v)
                    if (adj[u][v] && (degree[u]<degree[v] || (degree[u]==degree[v] && u<v))) graph.neighbors.push_back(v);
                graph.offsets.push_back(graph.neighbors.size());
            }
            check_graph(graph, oracle);
        }
        std::cout << "PASS: lane slices, warp stride, split thresholds and queue ordering vs independent enumeration (101 graphs + empty)\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
