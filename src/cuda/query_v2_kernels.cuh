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
// Reuse the actual common LONG row, not merely the source vertex. Empty lanes
// are ignored. Mixed targets retain the existing virtual query pool.
// Mode 5: common-row binary lookup; Mode 6: bounded warp-private shared hash.
// Dense ids are < UINT32_MAX (the graph reader rejects larger vertex counts).
constexpr unsigned query_hash_slots = 1024;
constexpr tc::Vertex query_hash_empty = 0xffffffffU;
__device__ __forceinline__ unsigned query_hash_slot(tc::Vertex key) {
    return (key * 2654435761U) & (query_hash_slots-1);
}
template<bool Hash, bool Strict=false, bool Deferred=false>
__device__ __forceinline__ tc::Count query_v2_reuse(const tc::Vertex* neighbors,
        tc::Offset ab, tc::Offset ae, tc::Offset bb, tc::Offset be, unsigned lane,
        tc::Vertex* table, bool diagnostics, tc::Offset* stats,
        tc::Offset* queue=nullptr, tc::Offset* queue_size=nullptr, tc::Offset batch_base=0) {
    const unsigned mask=__ballot_sync(0xffffffffU,ab<ae);
    if (!mask) return 0;
    const unsigned leader=__ffs(mask)-1;
    const auto common_bb=__shfl_sync(0xffffffffU,bb,leader);
    const auto common_be=__shfl_sync(0xffffffffU,be,leader);
    const bool common=__all_sync(0xffffffffU,ab==ae || (bb==common_bb && be==common_be));
    if (!common) return query_v2_pool(neighbors,ab,ae,bb,be,lane);
    tc::Offset end=ae-ab;
    for (unsigned delta=1;delta<32;delta*=2) {
        const auto previous=__shfl_up_sync(0xffffffffU,end,delta);
        if (lane>=delta) end+=previous;
    }
    const auto total=__shfl_sync(0xffffffffU,end,31), length=common_be-common_bb;
    // At most 50% occupancy; table is completely rebuilt inside timed work.
    // This is an initial reuse gate, not a claim about optimal hardware costs.
    const bool use_hash=(Hash || Deferred) && length>=(Strict?64ULL:16ULL) && length<=query_hash_slots/2
        && total>=(Strict?8ULL:2ULL)*length;
    if (diagnostics && lane==0) {
        atomicAdd(stats+11,1ULL);
        atomicAdd(stats+12,use_hash?1ULL:0ULL);
        atomicAdd(stats+13,use_hash?total:0ULL);
        atomicAdd(stats+14,use_hash?length:0ULL);
    }
    // All lanes have made the same decision. A batch is enqueued once, and
    // contributes zero in this pass; its complete query set is counted later.
    if constexpr (Deferred) {
        if (use_hash) {
            if (lane==0) {
                const auto slot=atomicAdd(queue_size,1ULL);
                queue[slot]=batch_base;
            }
            return 0;
        }
    }
    if constexpr (Hash) {
        if (use_hash) {
            for (unsigned i=lane;i<query_hash_slots;i+=32) table[i]=query_hash_empty;
            __syncwarp();
            for (tc::Offset i=common_bb+lane;i<common_be;i+=32) {
                const auto key=neighbors[i];
                auto slot=query_hash_slot(key);
                while (atomicCAS(table+slot,query_hash_empty,key)!=query_hash_empty)
                    slot=(slot+1)&(query_hash_slots-1);
            }
            __syncwarp();
        }
    }
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
        if (q<total) {
            const auto key=neighbors[owner_ab+q-start];
            if (use_hash) {
                auto slot=query_hash_slot(key);
                for (unsigned probe=0;probe<query_hash_slots;++probe) {
                    const auto item=table[slot];
                    if (item==key) { ++result;break; }
                    if (item==query_hash_empty) break;
                    slot=(slot+1)&(query_hash_slots-1);
                }
            } else result+=tc::query_v2_lookup(neighbors,key,common_bb,common_be);
        }
    }
    // All readers finish before any lane clears this warp's table next batch.
    __syncwarp();
    return result;
}
// Mode 0 scalar, 1 always pool, 2 execute a prefix then pool the remainder,
// 3 compare estimated costs; 4 prefix budget then gate by remaining lane count.
// 7 short gate + reuse; 8 fused shared hash; 9 short gate + reuse + deferred
// hash batches. Mode 9 reserves NO hash table, only the CUB reduction storage.
template<int Mode, bool Diagnostics>
__global__ void count_query_v2(tc::Offset m, const tc::Offset* offsets,
        const tc::Vertex* neighbors, const tc::Vertex* sources, tc::Count* block_sums,
        unsigned budget, tc::QueryV2Policy policy, unsigned remaining_limit, tc::Offset* stats=nullptr,
        tc::Offset* queue=nullptr, tc::Offset* queue_size=nullptr) {
    static_assert(Mode>=0 && Mode<=9,"unknown query v2 mode");
    const unsigned lane=threadIdx.x%32;
    const tc::Offset tid=static_cast<tc::Offset>(blockIdx.x)*blockDim.x+threadIdx.x;
    const tc::Offset first=tid/32*32;
    const tc::Offset stride=static_cast<tc::Offset>(gridDim.x)*blockDim.x;
    __shared__ tc::Vertex hash_storage[(Mode==6 || Mode==8) ? (block_threads/32)*query_hash_slots : 1];
    tc::Vertex* table=hash_storage+((Mode==6 || Mode==8) ? (threadIdx.x/32)*query_hash_slots : 0);
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
        bool pool=Mode==1 || Mode==5 || Mode==6 || Mode==7 || Mode==8 || Mode==9,fast=false;
        unsigned active_remaining=0;
        if constexpr (Mode==7 || Mode==8 || Mode==9) {
            fast=__all_sync(0xffffffffU,original_a<=policy.small);
            pool=!fast;
        }
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
        } else if constexpr (Mode==7 || Mode==8 || Mode==9) {
            if (fast) local+=tc::intersect_binary(neighbors,ab,ae,bb,be);
            else local+=query_v2_reuse<Mode==8,true,Mode==9>(
                neighbors,ab,ae,bb,be,lane,table,Diagnostics,stats,queue,queue_size,base);
        } else if constexpr (Mode==5 || Mode==6) {
            local+=query_v2_reuse<Mode==6>(neighbors,ab,ae,bb,be,lane,table,Diagnostics,stats);
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
// Each queue entry is the first CSR edge index of one 32-edge batch.
// Capacity ceil(m/32) is sufficient: producer enumerates each batch exactly once.
__global__ void count_queued_hash_batches(tc::Offset m, const tc::Offset* offsets,
        const tc::Vertex* neighbors, const tc::Vertex* sources, tc::Count* block_sums,
        const tc::Offset* queue, const tc::Offset* queue_size) {
    const unsigned lane=threadIdx.x%32;
    const tc::Offset tid=static_cast<tc::Offset>(blockIdx.x)*blockDim.x+threadIdx.x;
    const tc::Offset warp=tid/32;
    const tc::Offset stride=static_cast<tc::Offset>(gridDim.x)*(blockDim.x/32);
    const auto tasks=*queue_size;
    __shared__ tc::Vertex tables[(block_threads/32)*query_hash_slots];
    auto* table=tables+(threadIdx.x/32)*query_hash_slots;
    tc::Count local=0;
    for (tc::Offset task=warp;task<tasks;task+=stride) {
        const auto e=queue[task]+lane;
        tc::Offset ab=0,ae=0,bb=0,be=0;
        if (e<m) {
            const auto u=sources[e],v=neighbors[e];
            ab=offsets[u];ae=offsets[static_cast<tc::Offset>(u)+1];
            bb=offsets[v];be=offsets[static_cast<tc::Offset>(v)+1];
            if (ae-ab>be-bb) {
                const auto t0=ab,t1=ae;ab=bb;ae=be;bb=t0;be=t1;
            }
        }
        local+=query_v2_reuse<true,true>(neighbors,ab,ae,bb,be,lane,table,false,nullptr);
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
        int mode, unsigned budget, tc::QueryV2Policy policy, unsigned remaining_limit, tc::Offset* stats=nullptr,
        tc::Offset* queue=nullptr, tc::Offset* queue_size=nullptr) {
#define TC_QUERY_V2_LAUNCH(M) count_query_v2<M, Diagnostics><<<blocks,block_threads>>>(m,offsets,neighbors,sources,sums,budget,policy,remaining_limit,stats,queue,queue_size)
    switch (mode) {
        case 0: TC_QUERY_V2_LAUNCH(0); break;
        case 1: TC_QUERY_V2_LAUNCH(1); break;
        case 2: TC_QUERY_V2_LAUNCH(2); break;
        case 3: TC_QUERY_V2_LAUNCH(3); break;
        case 4: TC_QUERY_V2_LAUNCH(4); break;
        case 5: TC_QUERY_V2_LAUNCH(5); break;
        case 6: TC_QUERY_V2_LAUNCH(6); break;
        case 7: TC_QUERY_V2_LAUNCH(7); break;
        case 8: TC_QUERY_V2_LAUNCH(8); break;
        case 9:
            if (!queue || !queue_size) throw std::runtime_error("hash-queue requires allocated queue buffers");
            TC_QUERY_V2_LAUNCH(9); break;
        default: throw std::runtime_error("Unknown query v2 mode");
    }
#undef TC_QUERY_V2_LAUNCH
}
