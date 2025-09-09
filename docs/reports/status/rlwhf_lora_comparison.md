# RLWHF Comparison (Offline Benchmarks)

Default: docs/reports/status/chat_benchmark_offline_default.json

RLWHF: docs/reports/status/chat_benchmark_offline_rlwhf_lora_q4_t128.json

- llm: latency_avg 3201.8 → 6238.4 (Δ 3036.5); p50 2362.4 → 3762.8 (Δ 1400.5); sim 0.2 → 0.2 (Δ 0.0)
- llm_rag: latency_avg 5013.5 → 13172.5 (Δ 8159.0); p50 5891.2 → 13242.7 (Δ 7351.5); sim 0.7 → 0.6 (Δ -0.1)
- k3d: latency_avg 178.0 → 168.1 (Δ -9.9); p50 52.3 → 58.0 (Δ 5.8); sim 0.8 → 0.8 (Δ 0.0)
