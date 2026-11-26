#pragma once
#include "query_policy.hpp"
#include "edge_binary.hpp"
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#ifdef __CUDACC__
#define TC_SCHED_HD __host__ __device__
#else
#define TC_SCHED_HD
#endif
namespace tc {
constexpr unsigned schedule_policies = 3;
constexpr unsigned schedule_basis_size = 5;
// Proxy-work units, NOT ns/cycles. Calibration selects a policy from measured costs.
struct SegmentPolicy { double kappa, edge_cost, task_cost, margin; };
TC_SCHED_HD inline SegmentPolicy segment_policy(unsigned p) {
    return p == 0 ? SegmentPolicy{1.,64.,8.,.10} :
           p == 1 ? SegmentPolicy{1.5,256.,16.,.15} : SegmentPolicy{2.,1024.,32.,.20};
}
TC_SCHED_HD inline Offset segment_count(Offset queries,unsigned length) {
    // Merge every small tail: final segment is [L,2L); q=L+1 stays one task.
    const auto s=queries/length; return s ? s : 1;
}
TC_SCHED_HD inline Offset segment_end(Offset start,Offset queries,unsigned length) {
    return queries-start < 2ULL*length ? queries : start+length;
}
TC_SCHED_HD inline bool split_worthwhile(Offset a,Offset b,unsigned length,unsigned policy) {
    const auto segments=segment_count(a,length); if(segments<2) return false;
    const auto p=segment_policy(policy); const auto depth=query_depth(b);
    const auto last=a-(segments-1)*length;
    const double scalar=static_cast<double>(a)*depth;
    const double parallel=p.kappa*static_cast<double>(last)*depth+p.edge_cost+p.task_cost*segments;
    return scalar > (1.+p.margin)*parallel;
}
struct EdgeLists { Offset ab,ae,bb,be; };
TC_SCHED_HD inline EdgeLists schedule_lists(Offset e,const Offset* offsets,const Vertex* neighbors,const Vertex* sources) {
    const auto u=sources[e],v=neighbors[e];
    EdgeLists r{offsets[u],offsets[static_cast<Offset>(u)+1],offsets[v],offsets[static_cast<Offset>(v)+1]};
    if(r.ae-r.ab>r.be-r.bb) {const auto ab=r.ab,ae=r.ae;r.ab=r.bb;r.ae=r.be;r.bb=ab;r.be=ae;}
    return r;
}
TC_SCHED_HD inline Count count_segment(const Vertex* neighbors,EdgeLists r,Offset start,Offset end) {
    Count count=0;
    for(auto i=start;i<end;++i) {
        const auto key=neighbors[r.ab+i];auto lo=r.bb,hi=r.be;
        while(lo<hi) {const auto mid=lo+(hi-lo)/2;
            if(neighbors[mid]==key) {++count;break;}
            if(neighbors[mid]<key) lo=mid+1;else hi=mid;
        }
    }
    return count;
}
struct ScheduleStats {
    Offset queries=0,max_queries=0;
    double work=0,warp_work=0,tail_work=0,max_cost=0;
    Offset tasks[schedule_policies]={},split_edges[schedule_policies]={};
};
struct MergeScheduleStats {
    TC_SCHED_HD ScheduleStats operator()(ScheduleStats a,const ScheduleStats& b) const {
        a.queries+=b.queries;a.work+=b.work;a.warp_work+=b.warp_work;
        a.max_queries=a.max_queries>b.max_queries?a.max_queries:b.max_queries;
        a.tail_work=a.tail_work>b.tail_work?a.tail_work:b.tail_work;
        a.max_cost=a.max_cost>b.max_cost?a.max_cost:b.max_cost;
        for(unsigned p=0;p<schedule_policies;++p) {a.tasks[p]+=b.tasks[p];a.split_edges[p]+=b.split_edges[p];}
        return a;
    }
};
inline std::array<double,schedule_basis_size> schedule_basis(Offset m,const ScheduleStats& f) {
    return {1.,m/1e6,f.work/1e6,f.warp_work/1e6,f.tail_work/1e3};
}
struct CheapScheduleStats {double work=0,max_cost=0,queries=0;};
inline CheapScheduleStats cheap_schedule_stats(const Graph& g,const std::vector<Vertex>& sources) {
    CheapScheduleStats f;const Offset m=g.neighbors.size(),k=std::min<Offset>(m,256);
    for(Offset i=0;i<k;++i) {
        const Offset lo=i*m/k,hi=(i+1)*m/k;
        const Offset mixed=(i+42)*11400714819323198485ULL;
        const Offset e=lo+mixed%(hi-lo);
        const auto r=schedule_lists(e,g.offsets.data(),g.neighbors.data(),sources.data());
        const auto a=r.ae-r.ab;const double c=static_cast<double>(a)*query_depth(r.be-r.bb);
        f.work+=c;f.queries+=a;f.max_cost=std::max(f.max_cost,c);
    }
    if(k) {f.work*=static_cast<double>(m)/k;f.queries*=static_cast<double>(m)/k;}return f;
}
inline std::array<double,4> cheap_schedule_basis(Offset m,const CheapScheduleStats& f) {return {1.,m/1e6,f.work/1e6,f.max_cost/1e3};}
struct ScheduleModel {
    std::string device;
    int major=0,minor=0,sms=0,runtime=0,driver=0,blocks=0;
    unsigned length=128,queue_mib=64,policy=0;
    Offset coarse_edges=0;
    double margin=.1,cheap_gate_ms=0;
    std::array<double,4> cheap_coefficients{};
    std::array<std::array<double,schedule_basis_size>,3> coefficients{};
};
inline ScheduleModel read_schedule_model(const std::string& path) {
    std::ifstream in(path); ScheduleModel m;std::string magic;
    if(!(in>>magic) || magic!="TC_SCHEDULE_MODEL_V1") throw std::runtime_error("Invalid schedule model header");
    if(!(in>>std::quoted(m.device)>>m.major>>m.minor>>m.sms>>m.runtime>>m.driver>>m.blocks
         >>m.length>>m.queue_mib>>m.policy>>m.coarse_edges>>m.margin>>m.cheap_gate_ms)) throw std::runtime_error("Truncated schedule model");
    if(m.length<1 || m.length>65535 || m.queue_mib<1 || m.queue_mib>4096 || m.policy>=schedule_policies ||
       !std::isfinite(m.margin) || m.margin<0 || m.margin>1 || m.blocks<1 || !std::isfinite(m.cheap_gate_ms) || m.cheap_gate_ms<0 || m.cheap_gate_ms>1000) throw std::runtime_error("Invalid schedule model controls");
    for(auto& v:m.cheap_coefficients) if(!(in>>v)||!std::isfinite(v)||v<0) throw std::runtime_error("Invalid cheap coefficient");
    for(auto& row:m.coefficients) for(auto& v:row)
        if(!(in>>v) || !std::isfinite(v) || v<0) throw std::runtime_error("Invalid model coefficient");
    std::string extra;if(in>>extra) throw std::runtime_error("Extra model fields");
    return m;
}
inline std::array<double,3> schedule_predict(const ScheduleModel& model,Offset m,const ScheduleStats& stats) {
    auto x=schedule_basis(m,stats);std::array<double,3> times{};
    for(unsigned s=0;s<3;++s) for(unsigned i=0;i<schedule_basis_size;++i) times[s]+=x[i]*model.coefficients[s][i];
    return times;
}
inline unsigned schedule_choose(const std::array<double,3>& predicted,double feature_wall_ms,double margin) {
    unsigned best=0;
    for(unsigned s=1;s<3;++s) if(predicted[s]<predicted[best]) best=s;
    if(predicted[0]-predicted[best] <= feature_wall_ms+margin*predicted[0]) return 0;
    return best;
}
} // namespace tc
#undef TC_SCHED_HD
