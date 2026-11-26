#pragma once
#include "query_v2_policy.hpp"
#ifdef __CUDACC__
#define TC_GLOBAL_HD __host__ __device__
#else
#define TC_GLOBAL_HD
#endif
namespace tc {
// Precondition: q<prefix[m]. Repeated prefixes skip zero-query edges.
TC_GLOBAL_HD inline Offset global_query_owner(const Offset* prefix,Offset m,Offset q) {
    Offset lo=0,hi=m;
    while(lo<hi) {
        const auto mid=lo+(hi-lo)/2;
        if(prefix[mid+1]<=q) lo=mid+1;else hi=mid;
    }
    return lo;
}
// Locate once per contiguous chunk, then advance through edge boundaries.
TC_GLOBAL_HD inline Count global_query_chunk(const Offset* prefix,Offset m,
        const Offset* offsets,const Vertex* neighbors,const Vertex* sources,
        Offset q,Offset end) {
    if(q>=end) return 0;
    auto edge=global_query_owner(prefix,m,q);
    Count count=0;
    while(q<end) {
        const auto u=sources[edge],v=neighbors[edge];
        Offset ab=offsets[u],ae=offsets[static_cast<Offset>(u)+1];
        Offset bb=offsets[v],be=offsets[static_cast<Offset>(v)+1];
        if(ae-ab>be-bb) {const auto t0=ab,t1=ae;ab=bb;ae=be;bb=t0;be=t1;}
        const auto stop=end<prefix[edge+1]?end:prefix[edge+1];
        for(;q<stop;++q) count+=query_v2_lookup(neighbors,neighbors[ab+q-prefix[edge]],bb,be);
        if(q<end) {++edge;while(prefix[edge+1]<=q) ++edge;}
    }
    return count;
}
} // namespace tc
#undef TC_GLOBAL_HD
