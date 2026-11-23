#pragma once
// All 32 lanes participate in every shuffle/ballot, including invalid tail
// lanes. Paths are warp-uniform; all block threads eventually reach CUB.
__device__ tc::Offset query_warp_sum(tc::Offset value) {
    for (int delta = 16; delta; delta /= 2)
        value += __shfl_down_sync(0xffffffffU, value, delta);
    return __shfl_sync(0xffffffffU, value, 0);
}
__device__ tc::Offset query_warp_max(tc::Offset value) {
    for (int delta = 16; delta; delta /= 2) {
        const auto other = __shfl_down_sync(0xffffffffU, value, delta);
        value = value > other ? value : other;
    }
    return __shfl_sync(0xffffffffU, value, 0);
}
// mode 0: lane-per-edge control; 1: always pool; 2: adaptive (default).
// diagnostics: [batches, fast-path batches, scored scalar batches, pool batches,
//               nonempty edges, pooled queries, all queries, all scores, sum batch max scores]. Only a separate untimed pass
// instantiates Diagnostics=true. Timed kernels have no diagnostic atomics.
template<bool Diagnostics>
__global__ void count_query_batches(tc::Offset m, const tc::Offset* offsets,
                                   const tc::Vertex* neighbors, const tc::Vertex* sources,
                                   tc::Count* block_sums, int mode, tc::QueryPolicy policy,
                                   tc::Offset* diagnostics = nullptr) {
    const unsigned lane = threadIdx.x % 32;
    const tc::Offset tid = static_cast<tc::Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
    const tc::Offset first = tid / 32 * 32;
    const tc::Offset stride = static_cast<tc::Offset>(gridDim.x) * blockDim.x;
    tc::Count local = 0;
    for (tc::Offset base = first; base < m; base += stride) {
        const tc::Offset e = base + lane;
        tc::Offset ab = 0, ae = 0, bb = 0, be = 0;
        if (e < m) {
            const auto u = sources[e], v = neighbors[e];
            ab = offsets[u]; ae = offsets[static_cast<tc::Offset>(u) + 1];
            bb = offsets[v]; be = offsets[static_cast<tc::Offset>(v) + 1];
            if (ae - ab > be - bb) {
                const auto t0 = ab, t1 = ae; ab = bb; ae = be; bb = t0; be = t1;
            }
        }
        const tc::Offset a = ae - ab, b = be - bb;
        bool pool = mode == 1;
        bool fast = false;
        tc::Offset queries = 0;
        if (mode == 2) {
            // One ballot suffices for the common all-short fast path.
            fast = __ballot_sync(0xffffffffU, a > policy.small) == 0;
            if (!fast) {
                const auto score = tc::query_score(a, b);
                const auto maximum = query_warp_max(score);
                const auto sum = query_warp_sum(score);
                queries = query_warp_sum(a);
                // max_short is only used for the already-completed small gate.
                pool = tc::use_query_pool(static_cast<tc::Offset>(policy.small) + 1,
                                         maximum, sum, queries, policy);
            }
        }
        if constexpr (Diagnostics) {
            const auto all_queries = query_warp_sum(a);
            const auto score = tc::query_score(a, b);
            const auto all_scores = query_warp_sum(score);
            const auto max_score = query_warp_max(score);
            const auto nonempty = __popc(__ballot_sync(0xffffffffU, a != 0));
            if (pool && mode == 1) queries = query_warp_sum(a);
            if (lane == 0) {
                atomicAdd(diagnostics, 1ULL);
                atomicAdd(diagnostics + 1, fast ? 1ULL : 0ULL);
                atomicAdd(diagnostics + 2, !pool && !fast ? 1ULL : 0ULL);
                atomicAdd(diagnostics + 3, pool ? 1ULL : 0ULL);
                atomicAdd(diagnostics + 4, static_cast<tc::Offset>(nonempty));
                atomicAdd(diagnostics + 5, pool ? queries : 0ULL);
                atomicAdd(diagnostics + 6, all_queries);
                atomicAdd(diagnostics + 7, all_scores);
                atomicAdd(diagnostics + 8, max_score);
            }
        }
        if (!pool) {
            // Retain exact baseline query order, including zero-work lanes.
            local += tc::intersect_binary(neighbors, ab, ae, bb, be);
        } else {
            tc::Offset end = a;
            for (unsigned delta = 1; delta < 32; delta *= 2) {
                const auto prior = __shfl_up_sync(0xffffffffU, end, delta);
                if (lane >= delta) end += prior;
            }
            const auto total = __shfl_sync(0xffffffffU, end, 31);
            // Uniform rounds: even lanes q>=total execute metadata shuffles.
            // This is required by full-mask shuffle semantics on Volta+.
            for (tc::Offset qbase = 0; qbase < total; qbase += 32) {
                const auto q = qbase + lane;
                unsigned lo = 0, hi = 32;
                for (unsigned step = 0; step < 6; ++step) {
                    const unsigned mid = lo + (hi - lo) / 2;
                    const auto limit = __shfl_sync(0xffffffffU, end, mid < 32 ? mid : 31);
                    if (lo < hi) {
                        if (limit <= q) lo = mid + 1;
                        else hi = mid;
                    }
                }
                const unsigned owner = lo < 32 ? lo : 31;
                const auto previous = __shfl_sync(0xffffffffU, end, owner ? owner - 1 : 0);
                const auto start = owner ? previous : 0ULL;
                const auto owner_ab = __shfl_sync(0xffffffffU, ab, owner);
                auto left = __shfl_sync(0xffffffffU, bb, owner);
                auto right = __shfl_sync(0xffffffffU, be, owner);
                if (q < total) {
                    const auto key = neighbors[owner_ab + q - start];
                    while (left < right) {
                        const auto mid = left + (right - left) / 2;
                        if (neighbors[mid] == key) { ++local; break; }
                        if (neighbors[mid] < key) left = mid + 1;
                        else right = mid;
                    }
                }
            }
        }
    }
    using Reduce = cub::BlockReduce<tc::Count, block_threads>;
    __shared__ typename Reduce::TempStorage temp;
    const auto sum = Reduce(temp).Sum(local);
    if (threadIdx.x == 0) block_sums[blockIdx.x] = sum;
}
