#!/usr/bin/env bash
set -euo pipefail

# Live K3D benchmark + logs + RLWHF training (single-head)
# - Kills prior server, starts on free port, registers Galaxy, runs live chat benchmark,
#   publishes session logs, builds RLWHF dataset, retrains answer ranker.

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PATH="$HOME/miniconda3/bin:$PATH"

echo "[1/8] Killing prior K3D live_server (safe)…"
# Only kill our own previously launched servers via PID files; do not touch other apps (e.g., ComfyUI on 8787)
for pf in "$ROOT_DIR"/../Knowledge3D.local/datasets/live_server_*.pid; do
  [[ -f "$pf" ]] || continue
  if [[ -r "$pf" ]]; then
    PID=$(cat "$pf" || true)
    if [[ -n "${PID:-}" ]]; then
      kill -9 "$PID" 2>/dev/null || true
    fi
  fi
  rm -f "$pf" || true
done
sleep 1

echo "[2/8] Starting live_server on free port..."
# Allow custom port list via env K3D_LIVE_PORTS; default excludes 8787 (reserved for ComfyUI)
PORTS=${K3D_LIVE_PORTS:-"8788 8791 8793 8795 8797 8799"}
echo "using-ports: $PORTS"
PORT=""; for p in $PORTS; do
  if ! ss -ltnp | rg "$p" >/dev/null; then
    export K3D_LIVE_PORT=$p
    nohup "$ROOT_DIR/scripts/k3d_env.sh" run python -m knowledge3d.bridge.live_server \
      > "$ROOT_DIR/../Knowledge3D.local/datasets/live_server_${p}.log" 2>&1 & echo $! > "$ROOT_DIR/../Knowledge3D.local/datasets/live_server_${p}.pid"
    # Wait for TCP LISTEN, then probe WS healthz to confirm readiness
    for i in 1 2 3 4 5 6 7 8 9 10; do
      sleep 1
      if ss -ltnp | rg ":$p\b" >/dev/null; then
        cat > /tmp/k3d_ws_healthz.py <<'PY'
import asyncio, json, sys
import websockets
async def main(url):
    try:
        async with websockets.connect(url, open_timeout=10) as ws:
            await ws.send(json.dumps({"type":"event","event":{"kind":"healthz"}}))
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                obj = json.loads(msg)
                if obj.get("type") == "system":
                    print("OK")
                else:
                    print("ERR unexpected", obj.get("type"))
                    sys.exit(1)
            except Exception as e:
                print("ERR", e); sys.exit(1)
    except Exception as e:
        print("ERR", e); sys.exit(1)
asyncio.run(main(sys.argv[1]))
PY
        if "$ROOT_DIR/scripts/k3d_env.sh" run python /tmp/k3d_ws_healthz.py "ws://127.0.0.1:${p}" >/dev/null 2>&1; then
          PORT=$p; break
        fi
      fi
    done
    if [[ -n "$PORT" ]]; then break; fi
  fi
done
if [[ -z "$PORT" ]]; then echo "[ERR] no free port"; exit 1; fi
echo "server-port:$PORT"

echo "[3/8] Registering Galaxy to ws://127.0.0.1:${PORT}..."
LIVE_OK=1
if ! "$ROOT_DIR/scripts/k3d_env.sh" run timeout 180s \
  python -m knowledge3d.tools.register_galaxy --gltf "$ROOT_DIR/viewer/public/galaxy.cross.glb" --url "ws://127.0.0.1:${PORT}"; then
  echo "[WARN] Live server registration failed. Falling back to offline benchmark."
  LIVE_OK=0
fi

echo "[4/8] Running chat benchmark (live or offline)..."
if [[ "$LIVE_OK" == "1" ]]; then
  "$ROOT_DIR/scripts/k3d_env.sh" run timeout 900s \
    python -m knowledge3d.tools.benchmark_chat --gltf "$ROOT_DIR/viewer/public/galaxy.cross.glb" \
    --url "ws://127.0.0.1:${PORT}" --queries 20 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --out "$ROOT_DIR/docs/reports/status/chat_benchmark_live.json"
else
  # Offline path writes to the same JSON path for continuity
  "$ROOT_DIR/scripts/k3d_env.sh" run \
    python -m knowledge3d.tools.benchmark_offline --gltf "$ROOT_DIR/viewer/public/galaxy.cross.glb" \
    --queries 20 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --out-json "$ROOT_DIR/docs/reports/status/chat_benchmark_live.json" \
    --out-md "$ROOT_DIR/docs/reports/status/chat_benchmark_live.md"
fi

echo "[5/8] Publishing live session logs to repo..."
"$ROOT_DIR/scripts/k3d_env.sh" run python -m knowledge3d.tools.publish_local_artifacts

echo "[6/8] Building RLWHF dataset..."
if [[ "$LIVE_OK" == "1" ]]; then
  "$ROOT_DIR/scripts/k3d_env.sh" run python -m knowledge3d.tools.build_rlwhf_dataset \
    --logs "$ROOT_DIR/../Knowledge3D.local/logs" \
    --out "$ROOT_DIR/docs/reports/training/rlwhf_dataset.jsonl" \
    --summary "$ROOT_DIR/docs/reports/status/rlwhf_summary.json"
else
  # Build RLWHF dataset from offline benchmark results
  "$ROOT_DIR/scripts/k3d_env.sh" run python -m knowledge3d.tools.rlwhf_from_offline_benchmark \
    --gltf "$ROOT_DIR/viewer/public/galaxy.cross.glb" \
    --bench "$ROOT_DIR/docs/reports/status/chat_benchmark_live.json" \
    --out "$ROOT_DIR/docs/reports/training/rlwhf_dataset.jsonl"
  echo '{"note":"offline RLWHF (no live logs)"}' > "$ROOT_DIR/docs/reports/status/rlwhf_summary.json"
fi

echo "[7/8] Training Answer Ranker from RLWHF dataset..."
if [[ -s "$ROOT_DIR/docs/reports/training/rlwhf_dataset.jsonl" ]]; then
  "$ROOT_DIR/scripts/k3d_env.sh" run python -m knowledge3d.models.answer_ranker \
    --dataset "$ROOT_DIR/docs/reports/training/rlwhf_dataset.jsonl" \
    --out "$ROOT_DIR/../Knowledge3D.local/models/answer_ranker.pkl"
fi

echo "[8/8] Done. Summaries:"
ls -la "$ROOT_DIR/docs/reports/logs" | tail -n 8 || true
sed -n '1,140p' "$ROOT_DIR/docs/reports/status/chat_benchmark_live.json" 2>/dev/null || echo 'live benchmark json not produced yet'
sed -n '1,120p' "$ROOT_DIR/docs/reports/status/rlwhf_summary.json" 2>/dev/null || echo 'rlwhf_summary missing'

echo "OK"
