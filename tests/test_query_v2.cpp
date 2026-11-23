#include "../src/common/query_v2_policy.hpp"
#include "../src/common/edge_binary.hpp"
#include <array>
#include <iostream>
#include <random>

void require(bool ok,const char* what) { if(!ok) throw std::runtime_error(what); }
struct Task { tc::Offset ab=0,ae=0,bb=0,be=0; };
// CPU model validates prefix+remainder coverage and exact triangle counts.
// CUDA masks, device register allocation and memory safety require GPU tests.
tc::Count model(const tc::Graph& g,int mode,unsigned budget,tc::QueryV2Policy policy) {
    const auto sources=tc::edge_sources(g);
    tc::Count result=0;
    for(std::size_t base=0;base<sources.size();base+=32) {
        std::array<Task,32> tasks{};
        std::array<tc::Offset,32> original{},done{},ends{};
        std::array<std::vector<unsigned>,32> visits;
        tc::Offset queries=0,max_work=0,max_short=0;
        unsigned max_depth=0;
        for(unsigned lane=0;lane<32;++lane) {
            auto& t=tasks[lane];const auto e=base+lane;
            if(e<sources.size()) {
                const auto u=sources[e],v=g.neighbors[e];
                t={g.offsets[u],g.offsets[u+1],g.offsets[v],g.offsets[v+1]};
                if(t.ae-t.ab>t.be-t.bb) {std::swap(t.ab,t.bb);std::swap(t.ae,t.be);}
            }
            original[lane]=t.ab;
            const auto a=t.ae-t.ab,b=t.be-t.bb;
            visits[lane].resize(a);
            queries+=a;max_short=std::max(max_short,a);
            const auto depth=a?tc::query_depth(b):0;
            max_depth=std::max(max_depth,depth);max_work=std::max(max_work,a*depth);
        }
        if(mode==2) {
            for(unsigned lane=0;lane<32;++lane) {
                auto& t=tasks[lane];done[lane]=std::min(t.ae-t.ab,static_cast<tc::Offset>(budget));
                for(tc::Offset j=0;j<done[lane];++j) {
                    ++visits[lane][j];result+=tc::query_v2_lookup(g.neighbors.data(),g.neighbors[t.ab+j],t.bb,t.be);
                }
                t.ab+=done[lane];
            }
        }
        tc::Offset remaining=0;
        for(unsigned lane=0;lane<32;++lane) {remaining+=tasks[lane].ae-tasks[lane].ab;ends[lane]=remaining;}
        const bool pool=mode==1 || (mode==2 && remaining!=0) ||
            (mode==3 && max_short>policy.small && tc::query_v2_use_pool(max_work,queries,max_depth,policy));
        if(pool) {
            // Lane-strided queries, including a partial last 32-query round.
            for(unsigned lane=0;lane<32;++lane) for(tc::Offset q=lane;q<remaining;q+=32) {
                const auto owner=std::upper_bound(ends.begin(),ends.end(),q)-ends.begin();
                const auto start=owner?ends[owner-1]:0ULL;
                const auto j=tasks[owner].ab+q-start;
                ++visits[owner][j-original[owner]];
                result+=tc::query_v2_lookup(g.neighbors.data(),g.neighbors[j],tasks[owner].bb,tasks[owner].be);
            }
        } else if(mode!=2) {
            for(unsigned lane=0;lane<32;++lane) {
                const auto& t=tasks[lane];
                for(auto j=t.ab;j<t.ae;++j) {
                    ++visits[lane][j-original[lane]];
                    result+=tc::query_v2_lookup(g.neighbors.data(),g.neighbors[j],t.bb,t.be);
                }
            }
        }
        for(const auto& row:visits)
            require(std::all_of(row.begin(),row.end(),[](unsigned n){return n==1;}),"query duplicated or omitted");
    }
    return result;
}
int main() {
    try {
        const tc::QueryV2Policy policy;
        require(!tc::query_v2_use_pool(1024,4096,8,policy),"balanced heavy batch should remain scalar");
        require(tc::query_v2_use_pool(1024,159,8,policy),"skewed batch should pool");
        require(!tc::query_v2_use_pool(0,0,0,policy),"empty batch");
        require(!tc::query_v2_use_pool(30,32,1,policy),"strict margin equality");
        require(tc::query_v2_use_pool(31,32,1,policy),"strict margin above equality");
        const auto max_row=(1ULL<<32)-1,queries=32*max_row;
        const tc::QueryV2Policy largest{65535,65535,65535,1000};
        const auto cost=tc::query_v2_pool_cost(queries,32,largest);
        const unsigned __int128 wide=static_cast<unsigned __int128>(largest.setup)
            + static_cast<unsigned __int128>(tc::query_v2_rounds(queries))*(largest.locate+32ULL);
        require(cost==wide && static_cast<unsigned __int128>(cost)*1100< (static_cast<unsigned __int128>(1)<<64),"cost 64-bit bound");
        std::size_t configurations=0;
        for(unsigned seed=0;seed<84;++seed) {
            const unsigned n=seed==80?67:(seed==81?0:24);
            std::mt19937 rng(seed);
            std::vector<std::vector<bool>> adj(n,std::vector<bool>(n));
            std::vector<unsigned> degree(n);
            for(unsigned u=0;u<n;++u) for(unsigned v=u+1;v<n;++v) {
                const bool edge=seed==80 || (seed==82?u==0:(seed==83?(u<12&&v>=12):rng()%5==0));
                if(edge) {adj[u][v]=adj[v][u]=true;++degree[u];++degree[v];}
            }
            tc::Count oracle=0;
            for(unsigned u=0;u<n;++u) for(unsigned v=u+1;v<n;++v) for(unsigned w=v+1;w<n;++w)
                oracle+=adj[u][v]&&adj[u][w]&&adj[v][w];
            tc::Graph graph{{0},{}};
            for(unsigned u=0;u<n;++u) {
                for(unsigned v=0;v<n;++v)
                    if(adj[u][v]&&std::make_pair(degree[u],u)<std::make_pair(degree[v],v)) graph.neighbors.push_back(v);
                graph.offsets.push_back(graph.neighbors.size());
            }
            for(int mode=0;mode<4;++mode) {
                const std::vector<unsigned> budgets=mode==2?std::vector<unsigned>{0,1,4,8,10,16,65535}:std::vector<unsigned>{8};
                const std::vector<tc::QueryV2Policy> policies=mode==3?
                    std::vector<tc::QueryV2Policy>{policy,{0,0,0,0},largest}:std::vector<tc::QueryV2Policy>{policy};
                for(auto budget:budgets) for(auto p:policies) {
                    require(model(graph,mode,budget,p)==oracle,"query v2 model vs independent triple oracle");++configurations;
                }
            }
        }
        std::cout<<"PASS: v2 policy/cost bounds, exact query coverage and 84 graphs / "<<configurations<<" mode-budget-policy configurations\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
