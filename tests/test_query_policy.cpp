#include "../src/common/query_policy.hpp"
#include "../src/common/edge_binary.hpp"
#include <array>
#include <iostream>
#include <numeric>
#include <random>

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
// Mirror the six-round warp owner search; compare against STL upper_bound.
unsigned owner_search(const std::array<tc::Offset,32>& ends, tc::Offset q) {
    unsigned lo=0, hi=32;
    for (unsigned step=0; step<6; ++step) {
        const unsigned mid=lo+(hi-lo)/2;
        const auto limit=ends[mid<32?mid:31];
        if (lo<hi) { if (limit<=q) lo=mid+1; else hi=mid; }
    }
    return lo;
}
void check_mapping(const std::array<tc::Offset,32>& sizes) {
    std::array<tc::Offset,32> ends{};
    std::partial_sum(sizes.begin(),sizes.end(),ends.begin());
    const auto total=ends.back();
    std::vector<unsigned> visits(static_cast<std::size_t>(total));
    for (tc::Offset base=0;base<total;base+=32) {
        for (unsigned lane=0;lane<32;++lane) {
            const auto q=base+lane;
            const auto owner=owner_search(ends,q);
            require(owner==static_cast<unsigned>(std::upper_bound(ends.begin(),ends.end(),q)-ends.begin()),"owner upper_bound mismatch");
            if (q<total) {
                require(owner<32,"valid query must have owner");
                const auto start=owner?ends[owner-1]:0ULL;
                require(q-start<sizes[owner],"local query index");
                ++visits[q];
            }
        }
    }
    require(std::all_of(visits.begin(),visits.end(),[](unsigned n){return n==1;}),"exact query coverage");
}
// Host model of the new paths, validated against independent adjacency triples.
// This cannot validate CUDA masks, synchronization or device memory accesses.
tc::Count model_count(const tc::Graph& graph, unsigned mode, tc::QueryPolicy policy) {
    const auto sources=tc::edge_sources(graph);
    tc::Count count=0;
    for (std::size_t base=0;base<sources.size();base+=32) {
        std::array<tc::Offset,32> ab{},ae{},bb{},be{},ends{};
        tc::Offset maximum=0,sum=0,queries=0,max_short=0;
        for (unsigned lane=0;lane<32;++lane) {
            const auto e=base+lane;
            if (e<sources.size()) {
                const auto u=sources[e],v=graph.neighbors[e];
                ab[lane]=graph.offsets[u];ae[lane]=graph.offsets[u+1];
                bb[lane]=graph.offsets[v];be[lane]=graph.offsets[v+1];
                if (ae[lane]-ab[lane]>be[lane]-bb[lane]) {
                    std::swap(ab[lane],bb[lane]);std::swap(ae[lane],be[lane]);
                }
            }
            const auto a=ae[lane]-ab[lane],b=be[lane]-bb[lane];
            const auto score=tc::query_score(a,b);
            maximum=std::max(maximum,score);sum+=score;queries+=a;
            max_short=std::max(max_short,a);ends[lane]=queries;
        }
        const bool pool=mode==1 || (mode==2 && tc::use_query_pool(max_short,maximum,sum,queries,policy));
        if (!pool) {
            for (unsigned lane=0;lane<32;++lane)
                count+=tc::intersect_binary(graph.neighbors.data(),ab[lane],ae[lane],bb[lane],be[lane]);
        } else {
            for (tc::Offset q=0;q<queries;++q) {
                const auto owner=owner_search(ends,q);
                const auto start=owner?ends[owner-1]:0ULL;
                const auto key=graph.neighbors[ab[owner]+q-start];
                count+=std::binary_search(graph.neighbors.begin()+bb[owner],graph.neighbors.begin()+be[owner],key);
            }
        }
    }
    return count;
}
void check_graphs() {
    for (unsigned seed=0;seed<82;++seed) {
        const unsigned n=seed==80?67:(seed==81?0:24);
        std::mt19937 rng(seed);
        std::vector<std::vector<bool>> adj(n,std::vector<bool>(n));
        std::vector<unsigned> degree(n);
        for (unsigned u=0;u<n;++u) for (unsigned v=u+1;v<n;++v)
            if (seed==80 || rng()%5==0) { adj[u][v]=adj[v][u]=true;++degree[u];++degree[v]; }
        tc::Count oracle=0;
        for (unsigned u=0;u<n;++u) for (unsigned v=u+1;v<n;++v)
            for (unsigned w=v+1;w<n;++w) oracle+=adj[u][v]&&adj[u][w]&&adj[v][w];
        tc::Graph graph{{0},{}};
        for (unsigned u=0;u<n;++u) {
            for (unsigned v=0;v<n;++v)
                if (adj[u][v] && std::make_pair(degree[u],u)<std::make_pair(degree[v],v)) graph.neighbors.push_back(v);
            graph.offsets.push_back(graph.neighbors.size());
        }
        for (unsigned mode=0;mode<3;++mode)
            for (tc::QueryPolicy policy : {tc::QueryPolicy{},tc::QueryPolicy{0,0,0,0},tc::QueryPolicy{65535,65535,65535,65535}})
                require(model_count(graph,mode,policy)==oracle,"adaptive model vs independent graph oracle");
    }
}

int main() {
    try {
        for (unsigned bits=1;bits<=32;++bits) {
            const auto power=1ULL<<bits;
            require(tc::query_depth(power-1)==bits,"depth below power");
            require(tc::query_depth(power)==bits+1,"depth at power");
        }
        require(tc::query_depth(0)==0,"empty depth");
        check_graphs();
        const tc::QueryPolicy policy;
        require(!tc::use_query_pool(1,1,32,32,policy),"tiny fast path");
        require(!tc::use_query_pool(128,1024,32768,4096,policy),"balanced heavy batch");
        require(tc::use_query_pool(128,1024,1272,159,policy),"skewed batch");
        require(!tc::use_query_pool(0,0,0,0,policy),"empty batch");
        // Full-mask scan emulation, including tails and repeated prefix ends.
        std::mt19937 rng(20261123);
        for (unsigned test=0;test<512;++test) {
            std::array<tc::Offset,32> sizes{};
            for (unsigned lane=0;lane<test%33;++lane) sizes[lane]=rng()%130;
            auto scan=sizes;
            for (unsigned delta=1;delta<32;delta*=2) {
                const auto previous=scan;
                for (unsigned lane=delta;lane<32;++lane) scan[lane]+=previous[lane-delta];
            }
            std::array<tc::Offset,32> expected{};
            std::partial_sum(sizes.begin(),sizes.end(),expected.begin());
            require(scan==expected,"warp inclusive scan");
            check_mapping(sizes);
        }
        for (unsigned owner=0;owner<32;++owner) {
            std::array<tc::Offset,32> sizes{}; sizes[owner]=129; check_mapping(sizes);
        }
        // Prefixes exceeding 32 bits without allocating billions of queries.
        std::array<tc::Offset,32> ends{};
        for (unsigned lane=0;lane<32;++lane) ends[lane]=(lane+1ULL)*(1ULL<<31);
        for (unsigned lane=0;lane<32;++lane) {
            require(owner_search(ends,ends[lane]-1)==lane,"64-bit prefix owner");
            require(owner_search(ends,ends[lane])==lane+1,"64-bit boundary owner");
        }
        const auto a=(1ULL<<32)-1;
        const auto score=tc::query_score(a,a);
        require(score==a*32,"64-bit score");
        require(!tc::use_query_pool(a,score,score*32,a*32,{65535,65535,65535,65535}),"bounded large inputs");
        std::cout << "PASS: policy gates, 64-bit bounds, 544 query mappings, warp scan emulation, 82 graphs x 9 mode/policy combinations\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
