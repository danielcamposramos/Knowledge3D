# Warm 35% Sovereign No-NumPy Live Monitor

**Date:** 2026-03-24  
**Benchmark:** warm 35% validation launched via `scripts/run_enriched_benchmarks.py`  
**Sampling window:** approximately 60 seconds  
**Sample source:** `/tmp/k3d_live_metrics_03.24.2026.csv`

## Process

- benchmark PID: `1643507`
- process RSS: `5,030,776 KB` (~`4.80 GB`)
- process CPU: steady at `146%`
- hottest thread CPU: steady at `146%`

## GPU

- GPU util average: `0.00%`
- GPU util max: `0%`
- GPU memory used: steady at `1206 MB`
- benchmark process GPU memory: steady at `1190 MB`
- GPU power draw range: `37.94W` to `38.22W`

## Raw Samples

```csv
ts,pid,cpu_pct,rss_kb,gpu_util,gpu_mem_mb,gpu_power_w,proc_gpu_mem_mb,top_thread_cpu
2026-03-24T10:35:34,1643507,146,5030776,0,1206,38.05,1190,146
2026-03-24T10:35:39,1643507,146,5030776,0,1206,38.05,1190,146
2026-03-24T10:35:44,1643507,146,5030776,0,1206,38.08,1190,146
2026-03-24T10:35:50,1643507,146,5030776,0,1206,38.22,1190,146
2026-03-24T10:35:55,1643507,146,5030776,0,1206,38.01,1190,146
2026-03-24T10:36:00,1643507,146,5030776,0,1206,37.94,1190,146
2026-03-24T10:36:05,1643507,146,5030776,0,1206,37.95,1190,146
2026-03-24T10:36:10,1643507,146,5030776,0,1206,38.16,1190,146
2026-03-24T10:36:15,1643507,146,5030776,0,1206,38.01,1190,146
2026-03-24T10:36:20,1643507,146,5030776,0,1206,38.00,1190,146
2026-03-24T10:36:26,1643507,146,5030776,0,1206,37.94,1190,146
2026-03-24T10:36:31,1643507,146,5030776,0,1206,37.98,1190,146
```

## Interpretation

This run is still strongly CPU-orchestrated:

- one hot thread dominated the full window
- GPU memory is allocated and stable, but kernels are not sustaining visible load
- the system is behaving like a host-driven control loop with light device residency, not yet like a continuously active GPU-native game world

This is consistent with the Phase D.1 findings:

- Python still owns per-question iteration
- Jarvis dispatch is still orchestrated in Python
- multiple host↔device sync points still exist per question
