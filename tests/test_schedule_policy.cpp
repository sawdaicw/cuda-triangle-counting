#include "../src/common/schedule_policy.hpp"
#include <cassert>
#include <iostream>
#include <random>
#include <numeric>
#include <set>
#include <cstdio>
int main() {
    for(unsigned length:{1U,2U,32U,128U,65535U}) for(tc::Offset a:{0ULL,1ULL,31ULL,32ULL,33ULL,127ULL,128ULL,129ULL,255ULL,256ULL,257ULL,1000ULL}) {
        const auto n=tc::segment_count(a,length);tc::Offset at=0;
        for(tc::Offset s=0;s<n;++s) {const auto start=s*length,end=tc::segment_end(start,a,length);assert(start==at&&end>=start&&end<=a);at=end;}
        assert(at==a);
        if(a<2ULL*length) for(unsigned p=0;p<3;++p) assert(!tc::split_worthwhile(a,1024,length,p));
    }
    std::mt19937 rng(42);std::size_t cases=0;
    for(unsigned n:{0U,3U,16U,40U,67U}) for(unsigned seed=0;seed<6;++seed) {
        std::vector<std::set<unsigned>> adj(n);
        for(unsigned u=0;u<n;++u) for(unsigned v=u+1;v<n;++v) if(seed==0||rng()%5==0) {adj[u].insert(v);adj[v].insert(u);}
        tc::Graph g;g.offsets.assign(n+1,0);
        for(unsigned u=0;u<n;++u) {for(auto v:adj[u]) if(std::make_pair(adj[u].size(),u)<std::make_pair(adj[v].size(),v)) g.neighbors.push_back(v);g.offsets[u+1]=g.neighbors.size();}
        const auto sources=tc::edge_sources(g);tc::Count expected=0;
        for(unsigned u=0;u<n;++u) for(unsigned v=u+1;v<n;++v) for(unsigned w=v+1;w<n;++w) expected+=adj[u].count(v)&&adj[u].count(w)&&adj[v].count(w);
        for(unsigned length:{1U,8U,32U}) for(unsigned policy=0;policy<3;++policy) for(tc::Offset cap:{0ULL,1ULL,7ULL,100000ULL}) {
            struct Task{tc::Offset e,q;};std::vector<Task> queue;tc::Offset reserved=0;tc::Count count=0;
            std::vector<tc::Offset> order(g.neighbors.size());std::iota(order.begin(),order.end(),0);std::shuffle(order.begin(),order.end(),rng);
            for(auto e:order) {
                const auto r=tc::schedule_lists(e,g.offsets.data(),g.neighbors.data(),sources.data());const auto a=r.ae-r.ab;
                if(!tc::split_worthwhile(a,r.be-r.bb,length,policy)) {count+=tc::count_segment(g.neighbors.data(),r,0,a);continue;}
                const auto ns=tc::segment_count(a,length),slot=reserved;reserved+=ns;
                for(tc::Offset s=0;s<ns;++s) {
                    if(slot<cap&&s<cap-slot) queue.push_back({e,s*length});
                    else count+=tc::count_segment(g.neighbors.data(),r,s*length,tc::segment_end(s*length,a,length));
                }
            }
            assert(queue.size()==std::min(reserved,cap));
            for(auto task:queue) {const auto r=tc::schedule_lists(task.e,g.offsets.data(),g.neighbors.data(),sources.data());count+=tc::count_segment(g.neighbors.data(),r,task.q,tc::segment_end(task.q,r.ae-r.ab,length));}
            assert(count==expected);++cases;
        }
    }
    assert(tc::schedule_choose({1.,.8,.9},.01,.1)==1);
    assert(tc::schedule_choose({1.,.95,.9},.05,.1)==0);
    assert(tc::schedule_choose({0.,0.,0.},0.,0.)==0);
    std::cout<<"PASS: segment coverage/tails, bounded-queue fallback, triplet oracle "<<cases<<" cases and conservative choice\n";
}
