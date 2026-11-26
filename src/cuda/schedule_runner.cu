#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include "../common/schedule_policy.hpp"
#include "../common/preprocess.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#ifndef TC_SCHEDULE_CALIBRATE
#define TC_SCHEDULE_CALIBRATE 0
#endif
#include "schedule_kernels.cuh"
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point a,Clock::time_point b) {return std::chrono::duration<double,std::milli>(b-a).count();}
void check(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct Buffer {
    T* data=nullptr;std::size_t bytes;
    explicit Buffer(std::size_t n):bytes(std::max<std::size_t>(1,n)*sizeof(T)) {check(cudaMalloc(reinterpret_cast<void**>(&data),bytes));}
    ~Buffer(){if(data) cudaFree(data);}Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
};
struct Event {cudaEvent_t data{};Event(){check(cudaEventCreate(&data));}~Event(){cudaEventDestroy(data);}};
std::string quote(const std::string& s) {
    std::ostringstream out;out<<'"';for(unsigned char c:s) {
        if(c=='"'||c=='\\') out<<'\\'<<c;
        else if(c<32) out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;
        else out<<c;
    }out<<'"';return out.str();
}
int number(const std::string& s,int low,int high) {
    if(s.empty()||s.find_first_not_of("0123456789")!=std::string::npos) throw std::runtime_error("Expected integer: "+s);
    auto v=std::stoull(s);if(v<static_cast<unsigned>(low)||v>static_cast<unsigned>(high)) throw std::runtime_error("Integer out of range: "+s);
    return static_cast<int>(v);
}
struct Options {
    std::string input,model,strategy="auto";int blocks=864,warmup=3,repeat=15,rounds=3,seed=42;
    unsigned length=128,queue_mib=64,policy=0;bool verify=false,controls_explicit=false;
    tc::PreprocessOptions prep;
};
Options parse(int argc,char** argv) {
    if(argc<2) throw std::runtime_error("Usage: tc_schedule_{calibrate,auto} INPUT [--model FILE] [--strategy auto|baseline|bucket|segment] [--verify] [--blocks N] [--warmup N] [--repeat N] [--rounds N] [--segment-length N] [--queue-mib N] [--gate-policy 0|1|2] [OpenMP options]");
    Options o;o.input=argv[1];o.prep.parser=o.prep.mapping=o.prep.sorting="omp";o.prep.threads=12;
    for(int i=2;i<argc;++i) {const std::string flag=argv[i];
        if(flag=="--verify") {o.verify=true;continue;}
        if(tc::consume_preprocess_option(flag,i,argc,argv,o.prep)) continue;
        if(i+1==argc) throw std::runtime_error("Missing value for "+flag);
        const std::string value=argv[++i];
        if(flag=="--model") o.model=value;
        else if(flag=="--strategy") o.strategy=value;
        else if(flag=="--blocks") o.blocks=number(value,1,1048576);
        else if(flag=="--warmup") o.warmup=number(value,0,10000);
        else if(flag=="--repeat") o.repeat=number(value,1,10000);
        else if(flag=="--rounds") o.rounds=number(value,1,10000);
        else if(flag=="--seed") o.seed=number(value,0,2147483647);
        else if(flag=="--segment-length") {o.length=number(value,1,65535);o.controls_explicit=true;}
        else if(flag=="--queue-mib") {o.queue_mib=number(value,1,4096);o.controls_explicit=true;}
        else if(flag=="--gate-policy") {o.policy=number(value,0,2);o.controls_explicit=true;}
        else throw std::runtime_error("Unknown option: "+flag);
    }
    if(o.strategy!="auto"&&o.strategy!="baseline"&&o.strategy!="bucket"&&o.strategy!="segment") throw std::runtime_error("Invalid strategy");
    if(TC_SCHEDULE_CALIBRATE && (!o.model.empty()||o.strategy!="auto")) throw std::runtime_error("Calibration runs all fixed candidates; no model/strategy option");
    if(!TC_SCHEDULE_CALIBRATE && o.strategy=="auto" && o.model.empty()) throw std::runtime_error("Auto selection requires a fitted model; forced strategies are available without one");
    if(!o.model.empty() && o.controls_explicit) throw std::runtime_error("Segment controls come from the frozen model; do not override");
    return o;
}
void json_stats(const tc::ScheduleStats& f,tc::Offset m,int blocks) {
    const double lane=f.warp_work?f.work/(32*f.warp_work):1;
    const double warp=f.tail_work?f.warp_work/(blocks*8.*f.tail_work):1;
    std::cout<<"{\"m\":"<<m<<",\"queries\":"<<f.queries<<",\"work\":"<<f.work<<",\"warp_work\":"<<f.warp_work
        <<",\"tail_work\":"<<f.tail_work<<",\"max_cost\":"<<f.max_cost<<",\"max_queries\":"<<f.max_queries
        <<",\"lane_eff\":"<<lane<<",\"warp_eff\":"<<warp<<",\"segment_tasks\":[";
    for(unsigned p=0;p<3;++p) std::cout<<(p?",":"")<<f.tasks[p];
    std::cout<<"],\"split_edges\":[";for(unsigned p=0;p<3;++p) std::cout<<(p?",":"")<<f.split_edges[p];std::cout<<"]}";
}
const char* name(unsigned s) {return s==0?"baseline":s==1?"bucket":"segment";}
struct Sample {double gpu=0,prepare=0,count=0,reduce=0,d2h=0;tc::Count count_value=0;};
struct DeviceGraph {
    Buffer<tc::Offset> offsets;
    Buffer<tc::Vertex> neighbors,sources;
    Buffer<tc::Count> sums,result;
    std::unique_ptr<Buffer<tc::ScheduleStats>> feature_parts,feature_result;
    std::size_t reduce_scratch=0,feature_scratch=0,scan_scratch=0;
    std::unique_ptr<Buffer<unsigned char>> scratch;
    std::unique_ptr<Buffer<tc::Offset>> order;
    std::unique_ptr<Buffer<unsigned>> bins,cursor;
    std::unique_ptr<Buffer<tc_schedule::Segment>> queue;
    std::unique_ptr<Buffer<tc::Offset>> reserved;
    tc::Offset capacity=0;int blocks;
    DeviceGraph(const tc::Graph& g,int blocks_):offsets(g.offsets.size()),neighbors(g.neighbors.size()),sources(g.neighbors.size()),
        sums(2ULL*blocks_),result(1),blocks(blocks_) {
        check(cub::DeviceReduce::Sum(nullptr,reduce_scratch,sums.data,result.data,blocks*2));
        tc::ScheduleStats* stats_dummy=nullptr;
        check(cub::DeviceReduce::Reduce(nullptr,feature_scratch,stats_dummy,stats_dummy,blocks,tc::MergeScheduleStats{},tc::ScheduleStats{}));
        unsigned* dummy=nullptr;check(cub::DeviceScan::ExclusiveSum(nullptr,scan_scratch,dummy,dummy,64));
        scratch=std::make_unique<Buffer<unsigned char>>(std::max({reduce_scratch,feature_scratch,scan_scratch}));
    }
    std::size_t bytes() const {
        return offsets.bytes+neighbors.bytes+sources.bytes+sums.bytes+result.bytes+(feature_parts?feature_parts->bytes+feature_result->bytes:0)+scratch->bytes+
            (order?order->bytes:0)+(bins?bins->bytes:0)+(cursor?cursor->bytes:0)+(queue?queue->bytes:0)+(reserved?reserved->bytes:0);
    }
    void prepare_features() {feature_parts=std::make_unique<Buffer<tc::ScheduleStats>>(blocks);feature_result=std::make_unique<Buffer<tc::ScheduleStats>>(1);}
    void prepare(unsigned strategy,tc::Offset m,tc::Offset tasks,unsigned queue_mib) {
        if(strategy==1&&!order) {order=std::make_unique<Buffer<tc::Offset>>(m);bins=std::make_unique<Buffer<unsigned>>(64);cursor=std::make_unique<Buffer<unsigned>>(64);}
        if(strategy==2&&!queue) {
            capacity=std::min(tasks,static_cast<tc::Offset>(queue_mib)*1024*1024/sizeof(tc_schedule::Segment));
            queue=std::make_unique<Buffer<tc_schedule::Segment>>(capacity);reserved=std::make_unique<Buffer<tc::Offset>>(1);
        }
    }
    Sample run(unsigned strategy,unsigned policy,tc::Offset m,tc::Offset tasks,unsigned length) {
        Event start,prepared,counted,stop;check(cudaEventRecord(start.data));
        if(strategy==1) {
            check(cudaMemset(bins->data,0,bins->bytes));
            tc_schedule::histogram<<<blocks,tc_schedule::threads>>>(m,offsets.data,neighbors.data,sources.data,bins->data);check(cudaGetLastError());
            check(cub::DeviceScan::ExclusiveSum(scratch->data,scan_scratch,bins->data,cursor->data,64));
            tc_schedule::scatter<<<blocks,tc_schedule::threads>>>(m,offsets.data,neighbors.data,sources.data,cursor->data,order->data);check(cudaGetLastError());
        } else if(strategy==2) check(cudaMemset(reserved->data,0,sizeof(tc::Offset)));
        check(cudaEventRecord(prepared.data));
        if(strategy==0) tc_schedule::count_edges<false><<<blocks,tc_schedule::threads>>>(m,offsets.data,neighbors.data,sources.data,nullptr,sums.data);
        else if(strategy==1) tc_schedule::count_edges<true><<<blocks,tc_schedule::threads>>>(m,offsets.data,neighbors.data,sources.data,order->data,sums.data);
        else {
            tc_schedule::classify_count<<<blocks,tc_schedule::threads>>>(m,offsets.data,neighbors.data,sources.data,queue->data,capacity,reserved->data,sums.data,length,policy);check(cudaGetLastError());
            if(tasks) tc_schedule::count_queue<<<blocks,tc_schedule::threads>>>(offsets.data,neighbors.data,sources.data,queue->data,capacity,reserved->data,sums.data+blocks,length);
        }
        check(cudaGetLastError());check(cudaEventRecord(counted.data));
        const int partials=strategy==2&&tasks?blocks*2:blocks;
        check(cub::DeviceReduce::Sum(scratch->data,reduce_scratch,sums.data,result.data,partials));
        check(cudaEventRecord(stop.data));check(cudaEventSynchronize(stop.data));
        float total=0,prep=0,count=0,reduce=0;
        check(cudaEventElapsedTime(&total,start.data,stop.data));check(cudaEventElapsedTime(&prep,start.data,prepared.data));
        check(cudaEventElapsedTime(&count,prepared.data,counted.data));check(cudaEventElapsedTime(&reduce,counted.data,stop.data));
        Sample r;r.gpu=total;r.prepare=prep;r.count=count;r.reduce=reduce;
        const auto begin=Clock::now();check(cudaMemcpy(&r.count_value,result.data,sizeof(r.count_value),cudaMemcpyDeviceToHost));r.d2h=elapsed(begin,Clock::now());return r;
    }
};
} // namespace
int main(int argc,char** argv) {
    try {
        auto o=parse(argc,argv);const auto begin=Clock::now();std::unique_ptr<tc::ScheduleModel> model;
        if(!o.model.empty()) {model=std::make_unique<tc::ScheduleModel>(tc::read_schedule_model(o.model));o.length=model->length;o.queue_mib=model->queue_mib;o.policy=model->policy;}
        const auto model_ready=Clock::now();tc::PreprocessReport prep_report;
        const auto graph=tc::read_graph_configured(o.input,o.prep,nullptr,prep_report);const auto graph_ready=Clock::now();
        const auto src=tc::edge_sources(graph);const auto metadata_ready=Clock::now();
        const auto m=static_cast<tc::Offset>(graph.neighbors.size());
        if(m>static_cast<tc::Offset>(std::numeric_limits<int>::max())) throw std::runtime_error("Schedule runner requires m<=INT_MAX for bucket cursors");
        check(cudaSetDevice(0));cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));int runtime=0,driver=0;
        check(cudaRuntimeGetVersion(&runtime));check(cudaDriverGetVersion(&driver));
        if(o.blocks>prop.maxGridSize[0]) throw std::runtime_error("Grid exceeds device limit");
        if(model && (model->device!=prop.name || model->major!=prop.major || model->minor!=prop.minor || model->sms!=prop.multiProcessorCount ||
            model->runtime!=runtime || model->driver!=driver || model->blocks!=o.blocks)) throw std::runtime_error("Frozen model GPU/runtime/grid mismatch; recalibrate for this configuration");
        DeviceGraph d(graph,o.blocks);const auto setup_ready=Clock::now();
        check(cudaMemcpy(d.offsets.data,graph.offsets.data(),d.offsets.bytes,cudaMemcpyHostToDevice));
        if(m) {check(cudaMemcpy(d.neighbors.data,graph.neighbors.data(),graph.neighbors.size()*sizeof(tc::Vertex),cudaMemcpyHostToDevice));check(cudaMemcpy(d.sources.data,src.data(),src.size()*sizeof(tc::Vertex),cudaMemcpyHostToDevice));}
        const auto transferred=Clock::now();tc::ScheduleStats stats;double feature_gpu=0,feature_wall=0,decision_wall=0;bool have_features=false;
        bool coarse=!TC_SCHEDULE_CALIBRATE && o.strategy=="auto" && m<model->coarse_edges;
        tc::CheapScheduleStats cheap;double cheap_wall=0,cheap_prediction=0;bool cheap_used=false;
        if(TC_SCHEDULE_CALIBRATE || (o.strategy=="auto"&&!coarse)) {
            const auto cheap_begin=Clock::now();cheap=tc::cheap_schedule_stats(graph,src);cheap_used=true;
            if(model) {const auto x=tc::cheap_schedule_basis(m,cheap);for(unsigned i=0;i<4;++i) cheap_prediction+=x[i]*model->cheap_coefficients[i];
                if(model->cheap_gate_ms>0 && cheap_prediction<=model->cheap_gate_ms) coarse=true;}
            cheap_wall=elapsed(cheap_begin,Clock::now());
        }
        if(TC_SCHEDULE_CALIBRATE || o.strategy=="segment" || (o.strategy=="auto"&&!coarse)) {
            const auto feature_begin=Clock::now();d.prepare_features();Event start,stop;check(cudaEventRecord(start.data));
            tc_schedule::feature_blocks<<<o.blocks,tc_schedule::threads>>>(m,d.offsets.data,d.neighbors.data,d.sources.data,d.feature_parts->data,o.length);check(cudaGetLastError());
            check(cub::DeviceReduce::Reduce(d.scratch->data,d.feature_scratch,d.feature_parts->data,d.feature_result->data,o.blocks,tc::MergeScheduleStats{},tc::ScheduleStats{}));
            check(cudaEventRecord(stop.data));check(cudaEventSynchronize(stop.data));float f=0;check(cudaEventElapsedTime(&f,start.data,stop.data));feature_gpu=f;
            check(cudaMemcpy(&stats,d.feature_result->data,sizeof(stats),cudaMemcpyDeviceToHost));feature_wall=elapsed(feature_begin,Clock::now());have_features=true;
        }
        unsigned selected=0;std::array<double,3> predictions{};
        const auto decision_begin=Clock::now();
        if(!TC_SCHEDULE_CALIBRATE) {
            if(o.strategy=="auto"&&!coarse) {predictions=tc::schedule_predict(*model,m,stats);if(!stats.tasks[o.policy]) predictions[2]=std::max(predictions[0],predictions[1])+1;selected=tc::schedule_choose(predictions,feature_wall+cheap_wall,model->margin);if(selected==2&&!stats.tasks[o.policy]) selected=0;}
            else if(o.strategy=="bucket") selected=1;else if(o.strategy=="segment") selected=2;
        }
        decision_wall=elapsed(decision_begin,Clock::now());
        std::ostringstream body;body<<std::setprecision(12);
        tc::Count answer=0;
        if(TC_SCHEDULE_CALIBRATE) {
            d.prepare(1,m,0,o.queue_mib);d.prepare(2,m,*std::max_element(stats.tasks,stats.tasks+3),o.queue_mib);
            std::mt19937 random(o.seed);bool first_output=true,first_answer=true;
            body<<",\"rounds\":"<<o.rounds<<",\"measurements\":[";
            for(int round=0;round<o.rounds;++round) {
                std::array<unsigned,5> variants{0,1,2,3,4};std::shuffle(variants.begin(),variants.end(),random);
                for(auto v:variants) {const unsigned strategy=v<2?v:2,policy=v<2?0:v-2;
                    for(int i=0;i<o.warmup;++i) d.run(strategy,policy,m,stats.tasks[policy],o.length);
                    std::vector<Sample> samples;std::vector<double> times;
                    for(int i=0;i<o.repeat;++i) {const auto s=d.run(strategy,policy,m,stats.tasks[policy],o.length);
                        if(first_answer) {answer=s.count_value;first_answer=false;}if(s.count_value!=answer) throw std::runtime_error("Candidate/repeat count mismatch");samples.push_back(s);times.push_back(s.gpu);}
                    auto sorted=times;std::sort(sorted.begin(),sorted.end());const auto mid=sorted.size()/2;
                    const auto median=sorted.size()%2?sorted[mid]:(sorted[mid-1]+sorted[mid])/2;
                    body<<(first_output?"":",")<<"{\"strategy\":"<<quote(name(strategy))<<",\"policy\":"<<policy<<",\"round\":"<<round
                        <<",\"gpu_ms\":"<<median<<",\"prepare_ms_mean\":"<<std::accumulate(samples.begin(),samples.end(),0.,[](double a,const Sample& b){return a+b.prepare;})/samples.size()
                        <<",\"count_ms_mean\":"<<std::accumulate(samples.begin(),samples.end(),0.,[](double a,const Sample& b){return a+b.count;})/samples.size()
                        <<",\"reduce_ms_mean\":"<<std::accumulate(samples.begin(),samples.end(),0.,[](double a,const Sample& b){return a+b.reduce;})/samples.size()<<",\"gpu_ms_samples\":[";
                    for(std::size_t i=0;i<times.size();++i) body<<(i?",":"")<<times[i];body<<"]}";first_output=false;
                }
            }
            body<<"]";
        } else {
            const auto allocation_begin=Clock::now();d.prepare(selected,m,stats.tasks[o.policy],o.queue_mib);const auto allocation_ms=elapsed(allocation_begin,Clock::now());
            const auto first=d.run(selected,o.policy,m,stats.tasks[o.policy],o.length);const auto first_done=Clock::now();answer=first.count_value;
            for(int i=0;i<o.warmup;++i) d.run(selected,o.policy,m,stats.tasks[o.policy],o.length);
            std::vector<double> samples;
            for(int i=0;i<o.repeat;++i) {const auto s=d.run(selected,o.policy,m,stats.tasks[o.policy],o.length);if(s.count_value!=answer) throw std::runtime_error("Repeat count mismatch");samples.push_back(s.gpu);}
            body<<",\"requested_strategy\":"<<quote(o.strategy)<<",\"selected_strategy\":"<<quote(name(selected))<<",\"coarse_bypass\":"<<(coarse?"true":"false")
                <<",\"candidate_allocation_ms\":"<<allocation_ms<<",\"first_total_ms\":"<<elapsed(begin,first_done)
                <<",\"first_candidate_gpu_ms\":"<<first.gpu<<",\"first_prepare_ms\":"<<first.prepare<<",\"first_count_ms\":"<<first.count
                <<",\"first_reduce_ms\":"<<first.reduce<<",\"first_d2h_ms\":"<<first.d2h
                <<",\"first_execution_with_selection_wall_ms\":"<<elapsed(transferred,first_done)<<",\"predicted_gpu_ms\":[";
            for(unsigned s=0;s<3;++s) body<<(s?",":"")<<predictions[s];body<<"],\"candidate_gpu_ms_samples\":[";
            for(std::size_t i=0;i<samples.size();++i) body<<(i?",":"")<<samples[i];body<<"]";
        }
        const auto verify_begin=Clock::now();
        if(o.verify && answer!=tc::count_serial(graph.vertices(),graph.offsets.data(),graph.neighbors.data())) throw std::runtime_error("Schedule / CPU merge count mismatch");
        tc::Offset reserved=0;if(d.reserved) check(cudaMemcpy(&reserved,d.reserved->data,sizeof(reserved),cudaMemcpyDeviceToHost));
        if(!TC_SCHEDULE_CALIBRATE && selected==2 && reserved!=stats.tasks[o.policy]) throw std::runtime_error("Feature / queue reservation mismatch");
        const auto verify_ms=elapsed(verify_begin,Clock::now());
        std::cout<<std::setprecision(12)<<"{\"schema\":\"schedule-v1\",\"kind\":"<<quote(TC_SCHEDULE_CALIBRATE?"calibration":"execution")
            <<",\"input\":"<<quote(o.input)<<",\"device\":"<<quote(prop.name)<<",\"major\":"<<prop.major<<",\"minor\":"<<prop.minor
            <<",\"sms\":"<<prop.multiProcessorCount<<",\"cuda_runtime_version\":"<<runtime<<",\"cuda_driver_version\":"<<driver
            <<",\"blocks\":"<<o.blocks<<",\"threads_per_block\":256,\"vertices\":"<<graph.vertices()<<",\"undirected_edges\":"<<m
            <<",\"segment_length\":"<<o.length<<",\"queue_mib\":"<<o.queue_mib<<",\"gate_policy\":"<<o.policy
            <<",\"model_load_ms\":"<<elapsed(begin,model_ready)<<",\"preprocess_ms\":"<<elapsed(model_ready,graph_ready)<<",\"task_metadata_ms\":"<<elapsed(graph_ready,metadata_ready)
            <<",\"cuda_setup_ms\":"<<elapsed(metadata_ready,setup_ready)<<",\"h2d_ms\":"<<elapsed(setup_ready,transferred)
            <<",\"cheap_wall_ms\":"<<cheap_wall<<",\"cheap_prediction_ms\":"<<cheap_prediction<<",\"cheap_used\":"<<(cheap_used?"true":"false")
            <<",\"cheap_work\":"<<cheap.work<<",\"cheap_max_cost\":"<<cheap.max_cost<<",\"cheap_queries\":"<<cheap.queries
            <<",\"feature_gpu_ms\":"<<feature_gpu<<",\"feature_wall_ms\":"<<feature_wall<<",\"decision_wall_ms\":"<<decision_wall
            <<",\"features\":";
        if(have_features) json_stats(stats,m,o.blocks);else std::cout<<"null";
        std::cout<<",\"warmup\":"<<o.warmup<<",\"repeat\":"<<o.repeat<<",\"seed\":"<<o.seed;
        std::cout<<body.str();
        std::cout<<",\"triangles\":"<<answer<<",\"verified\":"<<(o.verify?"true":"false")<<",\"verify_ms\":"<<verify_ms
            <<",\"device_buffer_bytes\":"<<d.bytes()<<",\"queue_capacity_tasks\":"<<d.capacity<<",\"last_reserved_tasks\":"<<reserved<<",\"last_overflow_tasks\":"<<(reserved>d.capacity?reserved-d.capacity:0)
            <<",\"preprocess_options\":";tc::write_preprocess_options(std::cout,o.prep,prep_report,false);std::cout<<"}\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
