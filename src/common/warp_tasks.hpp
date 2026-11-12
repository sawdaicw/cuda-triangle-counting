#pragma once
#include "edge_binary.hpp"

namespace tc {
#ifdef __CUDACC__
#define TC_WARP_HD __host__ __device__
#else
#define TC_WARP_HD
#endif
// Threshold uses the SHORTER forward row, not original undirected degree.
TC_WARP_HD inline Offset shorter_forward_size(const Offset* offsets, Vertex u, Vertex v) {
    const Offset a = offsets[static_cast<std::size_t>(u) + 1] - offsets[u];
    const Offset b = offsets[static_cast<std::size_t>(v) + 1] - offsets[v];
    return a < b ? a : b;
}
// One lane's disjoint share of the short row. Summing lanes 0..width-1
// equals intersect_binary exactly; no padding, overlap, or approximate counts.
TC_WARP_HD inline Count intersect_binary_lane(const Vertex* neighbors,
                                              Offset a_begin, Offset a_end,
                                              Offset b_begin, Offset b_end,
                                              unsigned lane, unsigned width = 32) {
    if (a_begin == a_end || b_begin == b_end) return 0;
    if (a_end - a_begin > b_end - b_begin) {
        const Offset begin = a_begin, end = a_end;
        a_begin = b_begin; a_end = b_end;
        b_begin = begin; b_end = end;
    }
    Count total = 0;
    for (Offset i = a_begin + lane; i < a_end; i += width) {
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
#undef TC_WARP_HD
} // namespace tc
