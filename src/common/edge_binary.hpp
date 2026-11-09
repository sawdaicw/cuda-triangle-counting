#pragma once
#include "graph.hpp"

namespace tc {
// The two buffers are sorted and contain no duplicates. This also serves as
// a host-testable implementation of the exact intersection used by the GPU.
#ifdef __CUDACC__
#define TC_HD __host__ __device__
#else
#define TC_HD
#endif
TC_HD inline Count intersect_binary(const Vertex* neighbors,
                                    Offset a_begin, Offset a_end,
                                    Offset b_begin, Offset b_end) {
    if (a_begin == a_end || b_begin == b_end) return 0;
    if (a_end - a_begin > b_end - b_begin) {
        const Offset begin = a_begin, end = a_end;
        a_begin = b_begin; a_end = b_end;
        b_begin = begin; b_end = end;
    }
    Count total = 0;
    for (Offset i = a_begin; i < a_end; ++i) {
        const Vertex key = neighbors[i];
        Offset lo = b_begin, hi = b_end;
        while (lo < hi) {
            const Offset mid = lo + (hi - lo) / 2;
            if (neighbors[mid] == key) { ++total; break; }
            if (neighbors[mid] < key) lo = mid + 1;
            else hi = mid;
        }
    }
    return total;
}
#undef TC_HD

// neighbors[e] is already the destination of oriented edge e. Store only its
// source as extra task metadata; do not filter, sort, or duplicate tasks.
inline std::vector<Vertex> edge_sources(const Graph& graph) {
    std::vector<Vertex> sources(graph.neighbors.size());
    for (std::size_t u = 0; u < graph.vertices(); ++u)
        std::fill(sources.begin() + graph.offsets[u],
                  sources.begin() + graph.offsets[u + 1], static_cast<Vertex>(u));
    return sources;
}
} // namespace tc
