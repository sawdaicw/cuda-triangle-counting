#pragma once
// Included only for TC_EDGE_MODE=4. Mode is a template parameter: the timed
// scalar control contains no pool or policy branch. Diagnostics uses separate
// instantiations in an extra pass, never atomic operations in timed kernels.
__device__ __forceinline__ tc::Offset query_v2_sum(tc::Offset value) {
    for (int delta=16;delta;delta/=2) value+=__shfl_down_sync(0xffffffffU,value,delta);
    return __shfl_sync(0xffffffffU,value,0);
}
__device__ __forceinline__ tc::Offset query_v2_max(tc::Offset value) {
    for (int delta=16;delta;delta/=2) {
        const auto other=__shfl_down_sync(0xffffffffU,value,delta);
        value=value>other?value:other;
    }
    return __shfl_sync(0xffffffffU,value,0);
}
// All lanes enter/leave together. Zero-length rows and invalid tail queries
// still participate in every full-mask shuffle; only valid q accesses memory.
__device__ __forceinline__ tc::Count query_v2_pool(const tc::Vertex* neighbors,
        tc::Offset ab, tc::Offset ae, tc::Offset bb, tc::Offset be, unsigned lane) {
    tc::Offset end=ae-ab;
    for (unsigned delta=1;delta<32;delta*=2) {
        const auto previous=__shfl_up_sync(0xffffffffU,end,delta);
        if (lane>=delta) end+=previous;
    }
    const auto total=__shfl_sync(0xffffffffU,end,31);
    tc::Count result=0;
    for (tc::Offset base=0;base<total;base+=32) {
        const auto q=base+lane;
        unsigned lo=0,hi=32;
        for (unsigned step=0;step<6;++step) {
            const unsigned mid=lo+(hi-lo)/2;
            const auto limit=__shfl_sync(0xffffffffU,end,mid<32?mid:31);
            if (lo<hi) { if (limit<=q) lo=mid+1; else hi=mid; }
        }
        const unsigned owner=lo<32?lo:31;
        const auto previous=__shfl_sync(0xffffffffU,end,owner?owner-1:0);
        const auto start=owner?previous:0ULL;
        const auto owner_ab=__shfl_sync(0xffffffffU,ab,owner);
        const auto owner_bb=__shfl_sync(0xffffffffU,bb,owner);
        const auto owner_be=__shfl_sync(0xffffffffU,be,owner);
        if (q<total)
            result+=tc::query_v2_lookup(neighbors,neighbors[owner_ab+q-start],owner_bb,owner_be);
    }
    return result;
}
// Mode 0 scalar, 1 always pool, 2 execute a prefix then pool the remainder,
// 3 compare estimated costs; 4 prefix budget then gate by remaining lane count.
template<int Mode, bool Diagnostics>
__global__ void count_query_v2(tc::Offset m, const tc::Offset* offsets,
        const tc::Vertex* neighbors, const tc::Vertex* sources, tc::Count* block_sums,
        unsigned budget, tc::QueryV2Policy policy, unsigned remaining_limit, tc::Offset* stats=nullptr) {
    static_assert(Mode>=0 && Mode<=4,"unknown query v2 mode");
    const unsigned lane=threadIdx.x%32;
    const tc::Offset tid=static_cast<tc::Offset>(blockIdx.x)*blockDim.x+threadIdx.x;
    const tc::Offset first=tid/32*32;
    const tc::Offset stride=static_cast<tc::Offset>(gridDim.x)*blockDim.x;
    tc::Count local=0;
    for (tc::Offset base=first;base<m;base+=stride) {
        const auto e=base+lane;
        tc::Offset ab=0,ae=0,bb=0,be=0;
        if (e<m) {
            const auto u=sources[e],v=neighbors[e];
            ab=offsets[u];ae=offsets[static_cast<tc::Offset>(u)+1];
            bb=offsets[v];be=offsets[static_cast<tc::Offset>(v)+1];
            if (ae-ab>be-bb) {
                const auto t0=ab,t1=ae;ab=bb;ae=be;bb=t0;be=t1;
            }
        }
        const auto original_a=ae-ab,b=be-bb;
        tc::Offset prefix=0;
        bool pool=Mode==1,fast=false;
        unsigned active_remaining=0;
        if constexpr (Mode==2 || Mode==4) {
            prefix=original_a<budget?original_a:static_cast<tc::Offset>(budget);
            const auto prefix_end=ab+prefix;
            for (auto j=ab;j<prefix_end;++j)
                local+=tc::query_v2_lookup(neighbors,neighbors[j],bb,be);
            ab=prefix_end;
            const unsigned remaining_mask=__ballot_sync(0xffffffffU,ab<ae);
            fast=remaining_mask==0; // All queries finished in the prefix.
            if constexpr (Mode==4) {
                active_remaining=__popc(remaining_mask);
                pool=tc::query_v2_pool_for_active(active_remaining,remaining_limit);
            } else pool=remaining_mask!=0;
        } else if constexpr (Mode==3) {
            fast=__ballot_sync(0xffffffffU,original_a>policy.small)==0;
            if (!fast) {
                const unsigned depth=original_a?tc::query_depth(b):0;
                const auto max_work=query_v2_max(original_a*depth);
                const auto queries=query_v2_sum(original_a);
                const auto max_depth=query_v2_max(depth);
                pool=tc::query_v2_use_pool(max_work,queries,static_cast<unsigned>(max_depth),policy);
            }
        }
        if constexpr (Diagnostics) {
            const auto queries=query_v2_sum(original_a);
            const auto score=tc::query_score(original_a,b);
            const auto scores=query_v2_sum(score),max_score=query_v2_max(score);
            const auto prefixes=query_v2_sum(prefix),remaining=query_v2_sum(ae-ab);
            const auto nonempty=__popc(__ballot_sync(0xffffffffU,original_a!=0));
            if (lane==0) {
                atomicAdd(stats,1ULL);
                atomicAdd(stats+1,fast?1ULL:0ULL);
                atomicAdd(stats+2,!pool&&!fast?1ULL:0ULL);
                atomicAdd(stats+3,pool?1ULL:0ULL);
                atomicAdd(stats+4,static_cast<tc::Offset>(nonempty));
                atomicAdd(stats+5,pool?remaining:0ULL);
                atomicAdd(stats+6,queries);
                atomicAdd(stats+7,scores);
                atomicAdd(stats+8,max_score);
                atomicAdd(stats+9,prefixes);
                atomicAdd(stats+10,remaining);
                if constexpr (Mode==4) atomicAdd(stats+11+active_remaining,1ULL);
            }
        }
        if constexpr (Mode==0) {
            local+=tc::intersect_binary(neighbors,ab,ae,bb,be);
        } else if constexpr (Mode==1) {
            local+=query_v2_pool(neighbors,ab,ae,bb,be,lane);
        } else if constexpr (Mode==2) {
            if (pool) local+=query_v2_pool(neighbors,ab,ae,bb,be,lane);
        } else {
            if (pool) local+=query_v2_pool(neighbors,ab,ae,bb,be,lane);
            else local+=tc::intersect_binary(neighbors,ab,ae,bb,be);
        }
    }
    using Reduce=cub::BlockReduce<tc::Count,block_threads>;
    __shared__ typename Reduce::TempStorage temp;
    const auto sum=Reduce(temp).Sum(local);
    if (threadIdx.x==0) block_sums[blockIdx.x]=sum;
}
// Host dispatch occurs before the kernel executes; no runtime Mode branch in
// device code. Keep this declaration inside the same preprocessing guard.
template<bool Diagnostics>
void launch_query_v2(int blocks, tc::Offset m, const tc::Offset* offsets,
        const tc::Vertex* neighbors, const tc::Vertex* sources, tc::Count* sums,
        int mode, unsigned budget, tc::QueryV2Policy policy, unsigned remaining_limit, tc::Offset* stats=nullptr) {
#define TC_QUERY_V2_LAUNCH(M) count_query_v2<M, Diagnostics><<<blocks,block_threads>>>(m,offsets,neighbors,sources,sums,budget,policy,remaining_limit,stats)
    switch (mode) {
        case 0: TC_QUERY_V2_LAUNCH(0); break;
        case 1: TC_QUERY_V2_LAUNCH(1); break;
        case 2: TC_QUERY_V2_LAUNCH(2); break;
        case 3: TC_QUERY_V2_LAUNCH(3); break;
        case 4: TC_QUERY_V2_LAUNCH(4); break;
        default: throw std::runtime_error("Unknown query v2 mode");
    }
#undef TC_QUERY_V2_LAUNCH
}
