# Live Run Notes — 2025-09-09

- Ports: Reserved `8787` for ComfyUI. The K3D live script now avoids this port by default and only kills prior K3D servers via PID files.
  - Usage: `scripts/run_live_benchmark.sh` or `K3D_LIVE_PORTS="8791 8793 8797" scripts/run_live_benchmark.sh`.
- WebSockets: Pinned `websockets==10.4` for server/client compatibility and to remove v15 handshake/deprecation issues.
  - Changes:
    - `scripts/k3d_env.sh` now installs `websockets==10.4` in all bootstrap modes.
    - `envs/k3d-rapids.yml` and `envs/k3d-cpu.yml` include `websockets==10.4` under `pip:`.
    - `knowledge3d/tools/register_galaxy.py` uses a longer `open_timeout=30`.
- Live server handshake: On Debian 13 with `websockets 15.0.1`, the WS client timed out during the opening handshake even when TCP `LISTEN` was present.
  - Simple echo servers worked, indicating `websockets` itself was fine; the issue appears in the live server startup/handshake window under v15.
  - Mitigation: Added a readiness wait and client open timeout; pinned to `10.4`; provided offline fallback for end‑to‑end runs.
- Offline fallback: If live registration fails, the run script executes the offline benchmark and still builds an RLWHF dataset and retrains the Answer Ranker.
  - Artifacts:
    - `docs/reports/status/chat_benchmark_live.json`
    - `docs/reports/status/chat_benchmark_live.md`
    - `docs/reports/training/rlwhf_dataset.jsonl`
    - `docs/reports/status/rlwhf_summary.json`
    - `../Knowledge3D.local/models/answer_ranker.pkl`
- Results snapshot (offline, TinyLlama 1.1B):
  - K3D: avg_latency≈162.8 ms, p50≈51.0 ms, sim≈0.776
  - LLM: avg_latency≈3292.7 ms, p50≈2450.0 ms, sim≈0.190
  - LLM+RAG: avg_latency≈5292.9 ms, p50≈5714.0 ms, sim≈0.731

Next steps:
- Re‑test live server with pinned `websockets==10.4`.
- If needed, add a minimal `/healthz` WS response on connect to speed up readiness probes.
