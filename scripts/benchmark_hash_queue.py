"""All 14 graphs, fixed 3-way comparison, then separate queue/resource diagnostics."""
import sys
import benchmark_poster as bench
from hash_queue_diagnostics import collect


def main():
    include_fused='--include-fused' in sys.argv
    if include_fused:sys.argv.remove('--include-fused')
    skip_diagnostics='--skip-diagnostics' in sys.argv
    if skip_diagnostics:sys.argv.remove('--skip-diagnostics')
    gate=['--small','8','--bypass-edges','262144']
    bench.VARIANTS={
        'edge':('tc_gpu_edge_baseline',[]),
        'gate_reuse':('tc_gpu_query_v2',['--query-mode','reuse-gated',*gate]),
        'hash_queue':('tc_gpu_query_v2',['--query-mode','hash-queue',*gate])}
    if include_fused:
        bench.VARIANTS['gate_hash']=('tc_gpu_query_v2',['--query-mode','hash-gated',*gate])
    out=bench.main()
    if not skip_diagnostics:collect(out)


if __name__=='__main__':main()
