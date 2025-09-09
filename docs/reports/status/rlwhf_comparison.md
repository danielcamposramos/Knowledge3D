# RLWHF Comparison (Offline Benchmarks)

Default: docs/reports/status/chat_benchmark_offline_default.json

RLWHF: docs/reports/status/chat_benchmark_offline_rlwhf.json

- llm: latency_avg 3201.8 → 1425.1 (Δ -1776.7); p50 2362.4 → 1440.2 (Δ -922.2); sim 0.2 → 0.2 (Δ 0.0)
- llm_rag: latency_avg 5013.5 → 1481.5 (Δ -3532.0); p50 5891.2 → 1460.4 (Δ -4430.8); sim 0.7 → 0.6 (Δ -0.1)
- k3d: latency_avg 178.0 → 160.3 (Δ -17.7); p50 52.3 → 55.5 (Δ 3.3); sim 0.8 → 0.8 (Δ 0.0)
