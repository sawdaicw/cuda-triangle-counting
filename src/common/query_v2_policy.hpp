#pragma once
#include "query_policy.hpp"

#ifdef __CUDACC__
#define TC_QUERY_V2_HD __host__ __device__
#else
#define TC_QUERY_V2_HD
#endif
namespace tc {
// Uncalibrated comparison-work proxies, never interpreted as GPU cycles.
// CLI: small/setup/locate/budget in [0,65535], margin_percent in [0,1000].
struct QueryV2Policy {
    unsigned small = 4;
    unsigned setup = 16;
    unsigned locate = 8;
    unsigned margin_percent = 20;
};
// R=0 disables pooling; R=32 reproduces the original budget decision.
TC_QUERY_V2_HD inline bool query_v2_pool_for_active(unsigned remaining, unsigned limit) {
    return remaining != 0 && remaining <= limit;
}
TC_QUERY_V2_HD inline Offset query_v2_rounds(Offset queries) {
    return queries / 32 + (queries % 32 != 0);
}
TC_QUERY_V2_HD inline Offset query_v2_pool_cost(Offset queries, unsigned max_depth,
                                              QueryV2Policy policy) {
    return static_cast<Offset>(policy.setup) + query_v2_rounds(queries)
         * (static_cast<Offset>(policy.locate) + max_depth);
}
TC_QUERY_V2_HD inline bool query_v2_use_pool(Offset max_work, Offset queries,
                                           unsigned max_depth, QueryV2Policy policy) {
    if (queries == 0) return false;
    // With 32-bit simple-graph rows, max_depth<=32, Q<2^37. These products
    // fit in 64 bits even at the CLI's largest allowed coefficients.
    return 100ULL * max_work > (100ULL + policy.margin_percent)
                                * query_v2_pool_cost(queries, max_depth, policy);
}
TC_QUERY_V2_HD inline Count query_v2_lookup(const Vertex* neighbors, Vertex key,
                                           Offset begin, Offset end) {
    while (begin < end) {
        const auto mid = begin + (end - begin) / 2;
        if (neighbors[mid] == key) return 1;
        if (neighbors[mid] < key) begin = mid + 1;
        else end = mid;
    }
    return 0;
}
} // namespace tc
#undef TC_QUERY_V2_HD
