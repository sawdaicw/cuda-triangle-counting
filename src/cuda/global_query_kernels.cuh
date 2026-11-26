#pragma once
__global__ void build_global_query_lengths(tc::Offset m,const tc::Offset* offsets,
        const tc::Vertex* neighbors,const tc::Vertex* sources,tc::Offset* prefix) {
    const tc::Offset tid=static_cast<tc::Offset>(blockIdx.x)*blockDim.x+threadIdx.x;
    const tc::Offset stride=static_cast<tc::Offset>(gridDim.x)*blockDim.x;
    for(auto e=tid;e<m;e+=stride) prefix[e]=tc::shorter_forward_size(offsets,sources[e],neighbors[e]);
    if(tid==0) prefix[m]=0; // scan m+1 values so prefix[m] stores total Q.
}
__global__ void count_global_queries(tc::Offset m,const tc::Offset* prefix,
        const tc::Offset* offsets,const tc::Vertex* neighbors,const tc::Vertex* sources,
        tc::Count* sums,unsigned chunk) {
    const tc::Offset tid=static_cast<tc::Offset>(blockIdx.x)*blockDim.x+threadIdx.x;
    const tc::Offset stride=static_cast<tc::Offset>(gridDim.x)*blockDim.x*chunk;
    const auto total=prefix[m];
    tc::Count local=0;
    for(tc::Offset q=tid*chunk;q<total;q+=stride) {
        const auto end=total-q<chunk?total:q+chunk;
        local+=tc::global_query_chunk(prefix,m,offsets,neighbors,sources,q,end);
    }
    using Reduce=cub::BlockReduce<tc::Count,block_threads>;
    __shared__ typename Reduce::TempStorage temp;
    const auto sum=Reduce(temp).Sum(local);
    if(threadIdx.x==0) sums[blockIdx.x]=sum;
}
