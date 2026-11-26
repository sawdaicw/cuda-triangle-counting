#pragma once
namespace tc_schedule {
constexpr int threads=256;
struct Segment { tc::Offset edge,start; };
static_assert(sizeof(Segment)==16,"Segment memory accounting");
__device__ inline double warp_max(double v) {
    for(int d=16;d;d/=2) {const auto x=__shfl_down_sync(0xffffffff,v,d);v=v>x?v:x;}return v;
}
__global__ void feature_blocks(tc::Offset m,const tc::Offset* offsets,const tc::Vertex* neighbors,
        const tc::Vertex* sources,tc::ScheduleStats* out,unsigned length) {
    const auto tid=static_cast<tc::Offset>(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto lane=threadIdx.x%32;const auto base=tid-lane;
    const auto stride=static_cast<tc::Offset>(gridDim.x)*blockDim.x;
    tc::ScheduleStats f;
    for(auto batch=base;batch<m;batch+=stride) {
        double cost=0;
        const auto e=batch+lane;
        if(e<m) {
            const auto r=tc::schedule_lists(e,offsets,neighbors,sources);
            const auto a=r.ae-r.ab,b=r.be-r.bb;
            cost=static_cast<double>(a)*tc::query_depth(b);
            f.queries+=a;f.work+=cost;f.max_queries=f.max_queries>a?f.max_queries:a;
            f.max_cost=f.max_cost>cost?f.max_cost:cost;
            for(unsigned p=0;p<tc::schedule_policies;++p) if(tc::split_worthwhile(a,b,length,p)) {
                f.tasks[p]+=tc::segment_count(a,length);++f.split_edges[p];
            }
        }
        const auto maximum=warp_max(cost);
        if(lane==0) f.warp_work+=maximum;
    }
    if(lane==0) f.tail_work=f.warp_work;
    using Reduce=cub::BlockReduce<tc::ScheduleStats,threads>;
    __shared__ typename Reduce::TempStorage temp;
    const auto sum=Reduce(temp).Reduce(f,tc::MergeScheduleStats{});
    if(threadIdx.x==0) out[blockIdx.x]=sum;
}
template<bool Ordered> __global__ void count_edges(tc::Offset m,const tc::Offset* offsets,
        const tc::Vertex* neighbors,const tc::Vertex* sources,const tc::Offset* order,tc::Count* sums) {
    const auto tid=static_cast<tc::Offset>(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto stride=static_cast<tc::Offset>(gridDim.x)*blockDim.x;
    tc::Count local=0;
    for(auto i=tid;i<m;i+=stride) {
        const auto e=Ordered?order[i]:i;const auto r=tc::schedule_lists(e,offsets,neighbors,sources);
        local+=tc::count_segment(neighbors,r,0,r.ae-r.ab);
    }
    using Reduce=cub::BlockReduce<tc::Count,threads>;__shared__ typename Reduce::TempStorage temp;
    const auto sum=Reduce(temp).Sum(local);if(threadIdx.x==0) sums[blockIdx.x]=sum;
}
__device__ inline unsigned bucket(tc::Offset e,const tc::Offset* offsets,const tc::Vertex* neighbors,const tc::Vertex* sources) {
    const auto r=tc::schedule_lists(e,offsets,neighbors,sources);
    return tc::query_depth(tc::query_score(r.ae-r.ab,r.be-r.bb));
}
__global__ void histogram(tc::Offset m,const tc::Offset* offsets,const tc::Vertex* neighbors,
        const tc::Vertex* sources,unsigned* bins) {
    __shared__ unsigned local[64];if(threadIdx.x<64) local[threadIdx.x]=0;__syncthreads();
    const auto tid=static_cast<tc::Offset>(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto stride=static_cast<tc::Offset>(gridDim.x)*blockDim.x;
    for(auto e=tid;e<m;e+=stride) atomicAdd(local+bucket(e,offsets,neighbors,sources),1U);
    __syncthreads();if(threadIdx.x<64) atomicAdd(bins+threadIdx.x,local[threadIdx.x]);
}
__global__ void scatter(tc::Offset m,const tc::Offset* offsets,const tc::Vertex* neighbors,
        const tc::Vertex* sources,unsigned* cursor,tc::Offset* order) {
    const auto tid=static_cast<tc::Offset>(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto stride=static_cast<tc::Offset>(gridDim.x)*blockDim.x;
    for(auto e=tid;e<m;e+=stride) order[atomicAdd(cursor+bucket(e,offsets,neighbors,sources),1U)]=e;
}
__global__ void classify_count(tc::Offset m,const tc::Offset* offsets,const tc::Vertex* neighbors,
        const tc::Vertex* sources,Segment* queue,tc::Offset capacity,tc::Offset* reserved,
        tc::Count* sums,unsigned length,unsigned policy) {
    const auto tid=static_cast<tc::Offset>(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto stride=static_cast<tc::Offset>(gridDim.x)*blockDim.x;
    tc::Count local=0;
    for(auto e=tid;e<m;e+=stride) {
        const auto r=tc::schedule_lists(e,offsets,neighbors,sources);const auto a=r.ae-r.ab;
        if(!tc::split_worthwhile(a,r.be-r.bb,length,policy)) {local+=tc::count_segment(neighbors,r,0,a);continue;}
        const auto n=tc::segment_count(a,length);const auto start=atomicAdd(reserved,n);
        for(tc::Offset s=0;s<n;++s) {
            const auto q=s*length;
            if(start<capacity && s<capacity-start) queue[start+s]={e,q};
            else local+=tc::count_segment(neighbors,r,q,tc::segment_end(q,a,length));
        }
    }
    using Reduce=cub::BlockReduce<tc::Count,threads>;__shared__ typename Reduce::TempStorage temp;
    const auto sum=Reduce(temp).Sum(local);if(threadIdx.x==0) sums[blockIdx.x]=sum;
}
__global__ void count_queue(const tc::Offset* offsets,const tc::Vertex* neighbors,const tc::Vertex* sources,
        const Segment* queue,tc::Offset capacity,const tc::Offset* reserved,tc::Count* sums,unsigned length) {
    const auto tid=static_cast<tc::Offset>(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto stride=static_cast<tc::Offset>(gridDim.x)*blockDim.x;
    const auto n=*reserved<capacity?*reserved:capacity;tc::Count local=0;
    for(auto i=tid;i<n;i+=stride) {const auto task=queue[i];const auto r=tc::schedule_lists(task.edge,offsets,neighbors,sources);
        local+=tc::count_segment(neighbors,r,task.start,tc::segment_end(task.start,r.ae-r.ab,length));}
    using Reduce=cub::BlockReduce<tc::Count,threads>;__shared__ typename Reduce::TempStorage temp;
    const auto sum=Reduce(temp).Sum(local);if(threadIdx.x==0) sums[blockIdx.x]=sum;
}
} // namespace tc_schedule
