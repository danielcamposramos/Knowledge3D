# D.3b Live Monitor (Corrected Python PID)

**Date:** 2026-03-26  
**Benchmark session:** `full-bdb1ac3cefdd`  
**Python PID:** `1626267`  
**Raw CSV:** `/tmp/k3d_phaseD3b_gpu_nav_monitor_pythonpid_30s_03.26.csv`

30-second corrected capture against the real Python benchmark process:

- GPU util avg/min/max: `16.67% / 0.00% / 100.00%`
- GPU memory avg/max: `1947 MB / 1947 MB`
- process GPU memory avg/max: `1930 MB / 1930 MB`
- GPU power avg/max: `41.98 W / 44.69 W`
- process CPU avg/max: `103.00% / 103.00%`
- RSS avg/max: `4424146 KB / 4424148 KB`

Important note:

- The earlier wrapper-PID capture files under `/tmp/k3d_phaseD3b_gpu_nav_monitor_2min_03.26.csv` and the derived wrapper stats should be treated as **discarded** for per-process attribution.
- This corrected file is the trustworthy utilization sample for the live D.3b run.
