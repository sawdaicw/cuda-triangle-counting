#include "../src/common/global_queries.hpp"
#include "../src/common/edge_binary.hpp"
#include <iostream>
#include <random>
void require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
int main(){try{
 const tc::Offset p[]={0,0,1,1,(1ULL<<33),(1ULL<<33)};
 require(tc::global_query_owner(p,5,0)==1,"zero prefix");
 require(tc::global_query_owner(p,5,1)==3,"interior zero");
 require(tc::global_query_owner(p,5,(1ULL<<33)-1)==3,"64bit");
 unsigned cases=0;
 for(unsigned seed=0;seed<82;++seed){
  unsigned n=seed==80?48:(seed==81?0:20);std::mt19937 rng(seed);
  std::vector<std::vector<bool>> adj(n,std::vector<bool>(n));std::vector<unsigned> degree(n);
  for(unsigned u=0;u<n;++u)for(unsigned v=u+1;v<n;++v)if(seed==80||rng()%4==0){adj[u][v]=adj[v][u]=true;++degree[u];++degree[v];}
  tc::Count oracle=0;for(unsigned u=0;u<n;++u)for(unsigned v=u+1;v<n;++v)for(unsigned w=v+1;w<n;++w)oracle+=adj[u][v]&&adj[u][w]&&adj[v][w];
  tc::Graph g{{0},{}};
  for(unsigned u=0;u<n;++u){for(unsigned v=0;v<n;++v)if(adj[u][v]&&std::make_pair(degree[u],u)<std::make_pair(degree[v],v))g.neighbors.push_back(v);g.offsets.push_back(g.neighbors.size());}
  auto sources=tc::edge_sources(g);auto m=sources.size();std::vector<tc::Offset> prefix(m+1);
  for(std::size_t e=0;e<m;++e){auto u=sources[e],v=g.neighbors[e];prefix[e+1]=prefix[e]+std::min(g.offsets[u+1]-g.offsets[u],g.offsets[v+1]-g.offsets[v]);}
  for(unsigned b:{1,2,4,8,16,32,65535})for(unsigned workers:{1,7,256}){
   std::vector<unsigned> visits(prefix.back());tc::Count count=0;
   for(unsigned tid=0;tid<workers;++tid)for(tc::Offset q=static_cast<tc::Offset>(tid)*b;q<prefix.back();q+=static_cast<tc::Offset>(workers)*b){auto end=std::min(prefix.back(),q+b);count+=tc::global_query_chunk(prefix.data(),m,g.offsets.data(),g.neighbors.data(),sources.data(),q,end);for(auto x=q;x<end;++x)++visits[x];}
   require(count==oracle,"triplet oracle");require(std::all_of(visits.begin(),visits.end(),[](unsigned n){return n==1;}),"coverage");++cases;
  }
 }
 std::cout<<"PASS: global query owner, chunks, oracle and coverage: "<<cases<<" cases\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
