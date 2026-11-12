#pragma once

// Included inside edge_baseline.cu's anonymous namespace. All 256 threads
// reach BlockReduce, including lanes without queries and warps without edges.
__global__ void count_warp_edges(tc::Offset m, const tc::Offset* offsets,
                                const tc::Vertex* neighbors, const tc::Vertex* sources,
                                tc::Count* block_sums, const tc::Offset* queue = nullptr,
                                const tc::Offset* queue_size = nullptr) {
    const tc::Offset tid = static_cast<tc::Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x % 32;
    const tc::Offset warp = tid / 32;
    const tc::Offset warp_stride = static_cast<tc::Offset>(gridDim.x) * (blockDim.x / 32);
    const tc::Offset tasks = queue_size ? *queue_size : m;
    tc::Count local = 0;
    for (tc::Offset task = warp; task < tasks; task += warp_stride) {
        const tc::Offset e = queue ? queue[task] : task;
        const tc::Vertex u = sources[e], v = neighbors[e];
        local += tc::intersect_binary_lane(neighbors, offsets[u], offsets[static_cast<std::size_t>(u) + 1],
                                          offsets[v], offsets[static_cast<std::size_t>(v) + 1], lane);
    }
    // Sum lane-local counts only once per block, not once per edge.
    using Reduce = cub::BlockReduce<tc::Count, block_threads>;
    __shared__ typename Reduce::TempStorage temp;
    const tc::Count sum = Reduce(temp).Sum(local);
    if (threadIdx.x == 0) block_sums[blockIdx.x] = sum;
}

__global__ void count_light_queue_heavy(tc::Offset m, const tc::Offset* offsets,
                                      const tc::Vertex* neighbors, const tc::Vertex* sources,
                                      tc::Count* block_sums, tc::Offset threshold,
                                      tc::Offset* queue, tc::Offset* queue_size) {
    const tc::Offset tid = static_cast<tc::Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
    const tc::Offset stride = static_cast<tc::Offset>(gridDim.x) * blockDim.x;
    tc::Count local = 0;
    for (tc::Offset e = tid; e < m; e += stride) {
        const tc::Vertex u = sources[e], v = neighbors[e];
        const tc::Offset shorter = tc::shorter_forward_size(offsets, u, v);
        if (shorter >= threshold) {
            // Queue capacity is m, and each edge is enqueued at most once.
            const tc::Offset slot = atomicAdd(queue_size, 1ULL);
            queue[slot] = e;
        } else if (shorter != 0) {
            local += tc::intersect_binary(neighbors, offsets[u], offsets[static_cast<std::size_t>(u) + 1],
                                          offsets[v], offsets[static_cast<std::size_t>(v) + 1]);
        }
    }
    using Reduce = cub::BlockReduce<tc::Count, block_threads>;
    __shared__ typename Reduce::TempStorage temp;
    const tc::Count sum = Reduce(temp).Sum(local);
    if (threadIdx.x == 0) block_sums[blockIdx.x] = sum;
}
