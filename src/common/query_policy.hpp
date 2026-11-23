#pragma once
#include "graph.hpp"

#ifdef __CUDACC__
#define TC_QUERY_HD __host__ __device__
#else
#define TC_QUERY_HD
#endif
namespace tc {
// Score units, NOT calibrated GPU cycles. CLI bounds each parameter at 65535.
struct QueryPolicy {
    unsigned small = 4;
    unsigned overhead = 64;
    unsigned per_round = 2;
    unsigned margin = 32;
};
TC_QUERY_HD inline unsigned query_depth(Offset longer) {
#ifdef __CUDA_ARCH__
    return longer ? 64U - static_cast<unsigned>(__clzll(longer)) : 0U;
#else
    unsigned depth = 0;
    while (longer) { ++depth; longer >>= 1; }
    return depth;
#endif
}
TC_QUERY_HD inline Offset query_score(Offset shorter, Offset longer) {
    return shorter * query_depth(longer);
}
// For a valid simple graph with 32-bit vertex IDs, each row has <= 2^32-1
// entries. Thus scores, 32-lane sums and all products below fit in uint64.
TC_QUERY_HD inline bool use_query_pool(Offset max_short, Offset max_score,
                                      Offset sum_score, Offset queries,
                                      QueryPolicy policy) {
    if (max_short <= policy.small || queries == 0) return false;
    const Offset rounds = queries / 32 + (queries % 32 != 0);
    const Offset cost = static_cast<Offset>(policy.overhead) + policy.margin
                      + static_cast<Offset>(policy.per_round) * rounds;
    // 32 * (max - mean) > 32 * cost, with no floating-point rounding.
    return max_score * 32 > sum_score + cost * 32;
}
} // namespace tc
#undef TC_QUERY_HD
