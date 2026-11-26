"""Host C++ semantic check with CUDA launch syntax erased, NOT nvcc validation."""
import re,subprocess,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CUDA_STUB=r'''
#pragma once
#include <cstddef>
#define __host__
#define __device__
#define __global__
#define __shared__ static
#define __forceinline__ inline
struct dim3 {unsigned x=1,y=1,z=1;};
inline dim3 blockIdx,threadIdx,blockDim,gridDim;
inline void __syncthreads(){}
template<class T> T __shfl_down_sync(unsigned,T v,int){return v;}
inline int __clzll(unsigned long long v){return __builtin_clzll(v);}
template<class T> T atomicAdd(T* p,T v){T old=*p;*p+=v;return old;}
using cudaError_t=int;using cudaEvent_t=void*;constexpr int cudaSuccess=0;
constexpr int cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
struct cudaDeviceProp {char name[256]="STUB";int major=8,minor=0,multiProcessorCount=108,maxGridSize[3]={2147483647,1,1};};
inline int cudaMalloc(void**,std::size_t){return 0;}inline int cudaFree(void*){return 0;}
inline int cudaEventCreate(cudaEvent_t*){return 0;}inline int cudaEventDestroy(cudaEvent_t){return 0;}
inline int cudaEventRecord(cudaEvent_t){return 0;}inline int cudaEventSynchronize(cudaEvent_t){return 0;}
inline int cudaEventElapsedTime(float*,cudaEvent_t,cudaEvent_t){return 0;}
inline const char* cudaGetErrorString(int){return "stub";}inline int cudaGetLastError(){return 0;}
inline int cudaSetDevice(int){return 0;}inline int cudaGetDeviceProperties(cudaDeviceProp*,int){return 0;}
inline int cudaRuntimeGetVersion(int*){return 0;}inline int cudaDriverGetVersion(int*){return 0;}
inline int cudaMemcpy(void*,const void*,std::size_t,int){return 0;}inline int cudaMemset(void*,int,std::size_t){return 0;}
'''
CUB_STUB=r'''
#pragma once
namespace cub {
template<class T,int N> struct BlockReduce {struct TempStorage{};BlockReduce(TempStorage&){} T Sum(T x){return x;}template<class F>T Reduce(T x,F f){return f(x,T{});}};
struct DeviceReduce {template<class I,class O>static int Sum(void*,std::size_t&,I,O,int){return 0;}
template<class I,class O,class F,class T>static int Reduce(void*,std::size_t&,I,O,int,F f,T init){(void)f(init,init);return 0;}};
struct DeviceScan {template<class I,class O>static int ExclusiveSum(void*,std::size_t&,I,O,int){return 0;}};
}
'''
class CudaSemanticCheck(unittest.TestCase):
 def test_both_schedule_entrypoints(self):
  with tempfile.TemporaryDirectory() as temp:
   out=Path(temp);(out/'cuda_runtime.h').write_text(CUDA_STUB)
   for file in ['cub/block/block_reduce.cuh','cub/device/device_reduce.cuh','cub/device/device_scan.cuh']:
    p=out/file;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('#include "cub_stub.h"\n')
   (out/'cub_stub.h').write_text(CUB_STUB)
   # Keep relative project includes intact; inline the one header with launches.
   source=(ROOT/'src/cuda/schedule_runner.cu').read_text().replace('#include "schedule_kernels.cuh"',(ROOT/'src/cuda/schedule_kernels.cuh').read_text())
   source=source.replace('"../common/','"'+str(ROOT/'src/common')+'/')
   source=source.replace('#pragma once','');source=re.sub(r'<<<.*?>>>','',source,flags=re.S);target=out/'runner.cpp';target.write_text(source)
   for mode in [0,1]:
    result=subprocess.run(['g++','-std=c++17','-Werror','-fsyntax-only','-I',str(out),f'-DTC_SCHEDULE_CALIBRATE={mode}',str(target)],text=True,capture_output=True)
    self.assertEqual(result.returncode,0,result.stderr)
if __name__=='__main__':unittest.main()
