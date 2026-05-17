# Qwen3-Embedding-0.6B on Phenom — Qdrant Wiring Deployment Plan

**Date:** 2026-04-18
**Author:** Claude (architecture lane)
**Status:** SPEC — hand to Codex for execution
**Scope:** Ingestion-path only. Hot path is NOT touched. RTX 3070 stays sovereign.

---

## Route Decision: Direct Qdrant→Ollama vs. Sidecar

**Decision: Sidecar (FastAPI shim on Phenom). Direct is not available.**

Qdrant self-hosted (Docker) does NOT support remote embedder configuration at the
server level. "Qdrant Cloud Inference" (announced 2025) is a cloud-only feature.
Self-hosted Qdrant accepts pre-computed vectors only — it never calls an external
HTTP endpoint to generate them.

Therefore, the embedding call lives in the ingestion scripts, which must call
Ollama directly. However, we wrap the Ollama `/api/embed` endpoint in a thin
FastAPI sidecar on the Phenom to:

1. Expose an OpenAI-compatible `/v1/embeddings` endpoint (useful for future tooling)
2. Centralise the Matryoshka truncation logic in one place (see §8 — Ollama does
   NOT yet support a `dimensions` parameter natively; truncation must be done
   client-side after receiving the full 4096-dim vector)
3. Give a single health endpoint (`/health`) to monitor from the main workstation
4. Enforce the hard-fail policy: if Ollama returns an error, HTTP 503 propagates
   immediately — no silent CPU fallback

**Sidecar runs on Phenom (`192.168.0.60:8080`). Ingestion scripts call the sidecar.**

> Note on Ollama native MRL: As of Ollama issue #11213 (open as of 2026-04), Ollama
> does not accept a `dimensions` request parameter for Matryoshka truncation.
> Truncation to sub-4096 tiers happens by slicing the returned 4096-dim vector and
> re-normalising client-side (or sidecar-side). This matches what inferless and
> QwenLM recommend for llama.cpp-backed serving.

---

## Top-3 Risks

1. **Ollama MRL truncation is not native** — Ollama returns the full 4096-dim vector.
   Tier selection (64, 128, 256, …) requires client-side L2-norm slice. If the sidecar
   is bypassed and scripts call Ollama directly without truncation, 4096-dim vectors
   land in the wrong tier slot, silently corrupting search quality.
   *Mitigation:* ALL embedding calls route through the sidecar. Sidecar is the only
   entity that performs truncation. Scripts never call Ollama directly.

2. **MCP servers (`k3d-knowledge:8501`, `k3d-ptx:8503`) use the old `fast-all-minilm-l6-v2`
   named vector** — after migration the primary search vector is renamed (e.g.
   `qwen3-1024`). MCP server image env var `VECTOR_NAME` must be updated and containers
   restarted before the old collection alias is deleted, or every `qdrant-find` call
   will return zero results.
   *Mitigation:* Alias swap and MCP restart are a single atomic step in the migration
   procedure below.

3. **RTX 970 VRAM headroom (4 GB)** — Qwen3-Embedding-0.6B GGUF Q4_K_M ≈ 0.6 GB.
   Comfortable on the 970. Risk is if Daniel pulls a larger model variant or runs
   concurrent embedding batches that spike VRAM. Ollama queues requests; it will not
   crash, but throughput drops to sequential.
   *Mitigation:* Pull only the tagged `qwen3-embedding:0.6b` (Q4_K_M). Do not pull 4B
   or 8B variants on the Phenom.

---

## 1. Phenom Host Setup

Run all commands via SSH on `192.168.0.60` (Phenom II x6, Debian, RTX 970).

### 1.1 NVIDIA Driver Prerequisite Check

```bash
# On Phenom (ssh daniel@192.168.0.60)
nvidia-smi
# Must show RTX 970. If missing, install driver first:
# sudo apt install -y nvidia-driver-535 nvidia-cuda-toolkit
# sudo reboot
```

### 1.2 Install Ollama (official shell installer)

```bash
# On Phenom
curl -fsSL https://ollama.com/install.sh | sh
# Installs to /usr/local/bin/ollama
# Creates systemd unit: ollama.service
# Default: listens on 127.0.0.1:11434 (we will fix this next)
```

### 1.3 Configure Ollama to Listen on All Interfaces

The default Ollama systemd unit binds only to localhost. Override it so the main
workstation can reach it over the LAN.

```bash
# On Phenom — create systemd override directory and drop-in
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null <<'EOF'
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
EOF

sudo systemctl daemon-reload
sudo systemctl restart ollama
sudo systemctl enable ollama

# Verify
systemctl status ollama | grep -E "Active|OLLAMA_HOST"
curl -s http://localhost:11434/api/tags | python3 -m json.tool | head -5
```

### 1.4 Pull the Embedding Model

```bash
# On Phenom
ollama pull qwen3-embedding:0.6b
# Expected: ~550 MB download, Q4_K_M quantisation
# Verify
ollama list | grep qwen3-embedding
```

### 1.5 Open UFW Port — LAN Only

```bash
# On Phenom — open 11434 to 192.168.0.0/24 only, not to the internet
sudo ufw allow from 192.168.0.0/24 to any port 11434 proto tcp comment "k3d ollama embedder LAN"
sudo ufw status numbered | grep 11434
```

Also open port 8080 for the sidecar service (§4):

```bash
sudo ufw allow from 192.168.0.0/24 to any port 8080 proto tcp comment "k3d embedder sidecar LAN"
```

### 1.6 Smoke-Test from Main Workstation

```bash
# On main workstation (192.168.0.4)
curl -s http://192.168.0.60:11434/api/tags | python3 -c "import sys,json; d=json.load(sys.stdin); print([m['name'] for m in d['models']])"
# Expected output: ['qwen3-embedding:0.6b']

# Test embedding endpoint directly (full 4096 dims expected)
curl -s http://192.168.0.60:11434/api/embed \
  -d '{"model":"qwen3-embedding:0.6b","input":["hello world"]}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d['embeddings'][0]))"
# Expected: 4096
```

---

## 2. Sidecar Service on Phenom

The sidecar wraps Ollama's `/api/embed`, adds L2-norm Matryoshka truncation, and
exposes an OpenAI-compatible `/v1/embeddings` endpoint. It also provides `/health`
for monitoring.

Deploy at: `/opt/k3d-embedder/embedder_sidecar.py` on Phenom.

### 2.1 Install Python Dependencies on Phenom

```bash
# On Phenom
sudo apt install -y python3-pip python3-venv
python3 -m venv /opt/k3d-embedder/venv
/opt/k3d-embedder/venv/bin/pip install fastapi uvicorn httpx
```

### 2.2 Sidecar Script

Write `/opt/k3d-embedder/embedder_sidecar.py` on Phenom with the following content.
(This is the implementation for Codex to deploy — exact text matters for the
truncation logic and hard-fail behaviour.)

```python
#!/usr/bin/env python3
"""
K3D Embedder Sidecar — Phenom (192.168.0.60:8080)

Wraps Ollama /api/embed with:
- Matryoshka truncation + L2 renormalisation for any requested tier
- OpenAI-compatible /v1/embeddings endpoint
- Hard-fail HTTP 503 when Ollama is unreachable (no CPU fallback, ever)
- /health endpoint for monitoring
"""
import os
import math
import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Union

OLLAMA_HOST = os.environ.get("OLLAMA_EMBEDDER_HOST", "127.0.0.1:11434")
MODEL_NAME  = os.environ.get("QDRANT_EMBEDDER_MODEL", "qwen3-embedding:0.6b")
OLLAMA_URL  = f"http://{OLLAMA_HOST}/api/embed"
FULL_DIM    = 4096

VALID_TIERS = {64, 128, 256, 512, 768, 1024, 1536, 4096}

app = FastAPI(title="K3D Embedder Sidecar", version="1.0")


def _l2_truncate(vec: List[float], dim: int) -> List[float]:
    """Matryoshka prefix truncation + L2 renormalisation."""
    t = vec[:dim]
    norm = math.sqrt(sum(x * x for x in t)) or 1.0
    return [x / norm for x in t]


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = MODEL_NAME


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"http://{OLLAMA_HOST}/api/tags")
        if r.status_code != 200:
            raise HTTPException(status_code=503, detail="Ollama unhealthy")
        return {"status": "ok", "ollama": r.status_code}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Ollama unreachable: {exc}")


@app.post("/v1/embeddings")
async def create_embeddings(
    request: EmbeddingRequest,
    truncate_dim: Optional[int] = Query(None, description="Matryoshka tier; must be in {64,128,256,512,768,1024,1536,4096}"),
):
    if truncate_dim is not None and truncate_dim not in VALID_TIERS:
        raise HTTPException(
            status_code=400,
            detail=f"truncate_dim {truncate_dim} not in valid tiers {sorted(VALID_TIERS)}",
        )

    inputs = [request.input] if isinstance(request.input, str) else request.input
    payload = {"model": request.model, "input": inputs}

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(OLLAMA_URL, json=payload)
    except httpx.ConnectError as exc:
        raise HTTPException(status_code=503, detail=f"Ollama unreachable at {OLLAMA_HOST}: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Embedding request failed: {exc}")

    if r.status_code != 200:
        raise HTTPException(status_code=503, detail=f"Ollama returned {r.status_code}: {r.text}")

    result = r.json()
    raw_embeddings: List[List[float]] = result.get("embeddings", [])
    if not raw_embeddings:
        raise HTTPException(status_code=503, detail="Ollama returned empty embeddings list")

    dim = truncate_dim if truncate_dim is not None else FULL_DIM
    processed = [_l2_truncate(e, dim) for e in raw_embeddings]

    return {
        "object": "list",
        "model": request.model,
        "data": [
            {"object": "embedding", "index": i, "embedding": emb}
            for i, emb in enumerate(processed)
        ],
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### 2.3 Systemd Unit for Sidecar

Write `/etc/systemd/system/k3d-embedder.service` on Phenom:

```ini
[Unit]
Description=K3D Embedder Sidecar (Qwen3-Embedding Matryoshka shim)
After=network.target ollama.service
Requires=ollama.service

[Service]
Type=simple
User=daniel
WorkingDirectory=/opt/k3d-embedder
Environment="OLLAMA_EMBEDDER_HOST=127.0.0.1:11434"
Environment="QDRANT_EMBEDDER_MODEL=qwen3-embedding:0.6b"
ExecStart=/opt/k3d-embedder/venv/bin/uvicorn embedder_sidecar:app --host 0.0.0.0 --port 8080 --workers 2
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
# On Phenom
sudo systemctl daemon-reload
sudo systemctl enable k3d-embedder
sudo systemctl start k3d-embedder
systemctl status k3d-embedder
```

---

## 3. Collection Schema Migration

### 3.1 Current State

All three collections use a single named vector `fast-all-minilm-l6-v2` at 384 dims.

| Collection | Points (approx) | Vector name |
|---|---|---|
| `k3d_specifications` | 1,319 | `fast-all-minilm-l6-v2` |
| `k3d_ptx` | 11,298 | `fast-all-minilm-l6-v2` |
| `k3d_canonical` | ~80 | `fast-all-minilm-l6-v2` |

### 3.2 Target Schema — Named Vectors at 8 Matryoshka Tiers

**Primary query vector** for MCP servers: `qwen3-1024` (1024 dims — best quality/speed
tradeoff for spec search; can upgrade to 1536 later without full migration).

**All 8 tiers stored** to support future tier-adaptive queries:
`qwen3-64`, `qwen3-128`, `qwen3-256`, `qwen3-512`, `qwen3-768`,
`qwen3-1024`, `qwen3-1536`, `qwen3-4096`.

All tiers use cosine distance.

### 3.3 Concrete Qdrant REST — Recreate Collection (k3d_specifications example)

Replace `<COLLECTION>` with each of `k3d_specifications_v2`, `k3d_ptx_v2`,
`k3d_canonical_v2` (the `_v2` suffix is the migration scratch collection).

```bash
# On main workstation — create the v2 collection with named vectors
curl -s -X PUT "http://192.168.0.4:6333/collections/k3d_specifications_v2" \
  -H "Content-Type: application/json" \
  -H "api-key: @20Cooool58" \
  -d '{
    "vectors": {
      "qwen3-64":   {"size": 64,   "distance": "Cosine"},
      "qwen3-128":  {"size": 128,  "distance": "Cosine"},
      "qwen3-256":  {"size": 256,  "distance": "Cosine"},
      "qwen3-512":  {"size": 512,  "distance": "Cosine"},
      "qwen3-768":  {"size": 768,  "distance": "Cosine"},
      "qwen3-1024": {"size": 1024, "distance": "Cosine"},
      "qwen3-1536": {"size": 1536, "distance": "Cosine"},
      "qwen3-4096": {"size": 4096, "distance": "Cosine"}
    }
  }'
```

Run the same command for `k3d_ptx_v2` and `k3d_canonical_v2`.

### 3.4 Payload Indexes (re-create on v2 collections)

```bash
# k3d_specifications_v2
for field in spec_name content_type; do
  curl -s -X PUT "http://192.168.0.4:6333/collections/k3d_specifications_v2/index" \
    -H "Content-Type: application/json" \
    -H "api-key: @20Cooool58" \
    -d "{\"field_name\": \"$field\", \"field_schema\": \"keyword\"}"
done

# k3d_ptx_v2
for field in source version; do
  curl -s -X PUT "http://192.168.0.4:6333/collections/k3d_ptx_v2/index" \
    -H "Content-Type: application/json" \
    -H "api-key: @20Cooool58" \
    -d "{\"field_name\": \"$field\", \"field_schema\": \"keyword\"}"
done

# k3d_canonical_v2
for field in kind key; do
  curl -s -X PUT "http://192.168.0.4:6333/collections/k3d_canonical_v2/index" \
    -H "Content-Type: application/json" \
    -H "api-key: @20Cooool58" \
    -d "{\"field_name\": \"$field\", \"field_schema\": \"keyword\"}"
done
```

---

## 4. Migration Strategy — Zero Downtime via Aliases

Qdrant supports collection aliases. MCP servers (`k3d-knowledge` and `k3d-ptx`) are
pointed at the collection name, not a fixed alias — so we first create aliases,
then populate v2, then atomically swap.

### 4.1 Create Aliases for Current Collections

```bash
# Step 1: alias the live collections so MCP servers can be redirected
curl -s -X POST "http://192.168.0.4:6333/collections/aliases" \
  -H "Content-Type: application/json" \
  -H "api-key: @20Cooool58" \
  -d '{
    "actions": [
      {"create_alias": {"collection_name": "k3d_specifications", "alias_name": "k3d_specifications_live"}},
      {"create_alias": {"collection_name": "k3d_ptx",            "alias_name": "k3d_ptx_live"}},
      {"create_alias": {"collection_name": "k3d_canonical",      "alias_name": "k3d_canonical_live"}}
    ]
  }'
```

### 4.2 Re-ingest into v2 Collections

Run the updated ingesters (§6) with `INGEST_TARGET=v2` against the v2 collections.
Ingestion is CPU/network bound on the main workstation; the Phenom handles embedding.
Old collections remain live and searchable during this time.

```bash
# On main workstation — example for specs
QDRANT_COLLECTION=k3d_specifications_v2 \
OLLAMA_EMBEDDER_HOST=192.168.0.60:11434 \
QDRANT_EMBEDDER_MODEL=qwen3-embedding:0.6b \
  python scripts/ingest_specs_to_qdrant.py

# Repeat for ptx and canonical
```

### 4.3 Verify v2 Point Count Matches Original

```bash
for col in k3d_specifications_v2 k3d_ptx_v2 k3d_canonical_v2; do
  count=$(curl -s "http://192.168.0.4:6333/collections/$col" \
    -H "api-key: @20Cooool58" | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['points_count'])")
  echo "$col: $count"
done
```

Expected approximate minimums: `k3d_specifications_v2` ≥ 1,319; `k3d_ptx_v2` ≥ 11,298;
`k3d_canonical_v2` ≥ 80.

### 4.4 Atomic Alias Swap + MCP Config Update

**Do steps A, B, C in one maintenance window (< 2 minutes total).**

**A. Update MCP server env vars** — change `VECTOR_NAME` from `fast-all-minilm-l6-v2`
to `qwen3-1024` in both MCP containers (or their environment files).

**B. Atomic alias swap:**

```bash
curl -s -X POST "http://192.168.0.4:6333/collections/aliases" \
  -H "Content-Type: application/json" \
  -H "api-key: @20Cooool58" \
  -d '{
    "actions": [
      {"delete_alias": {"alias_name": "k3d_specifications_live"}},
      {"create_alias": {"collection_name": "k3d_specifications_v2", "alias_name": "k3d_specifications_live"}},
      {"delete_alias": {"alias_name": "k3d_ptx_live"}},
      {"create_alias": {"collection_name": "k3d_ptx_v2",            "alias_name": "k3d_ptx_live"}},
      {"delete_alias": {"alias_name": "k3d_canonical_live"}},
      {"create_alias": {"collection_name": "k3d_canonical_v2",      "alias_name": "k3d_canonical_live"}}
    ]
  }'
```

**C. Restart MCP containers:**

```bash
docker restart k3d-knowledge-mcp k3d-ptx-mcp
```

### 4.5 Verify Search Works After Swap

```bash
# Test k3d-knowledge MCP is returning results
curl -s -X POST "http://192.168.0.4:8501/qdrant-find" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the composed head pipeline?"}' | python3 -m json.tool | head -20
```

### 4.6 Delete Old Collections (After 24h Soak)

```bash
# Only after confirming MCP works correctly for 24 hours
for col in k3d_specifications k3d_ptx k3d_canonical; do
  curl -s -X DELETE "http://192.168.0.4:6333/collections/$col" \
    -H "api-key: @20Cooool58"
  echo "Deleted $col"
done
```

---

## 5. Environment Variable Conventions

### 5.1 Variables to Add to `scripts/k3d_env.sh`

Add the following block after the existing `CUDA_VISIBLE_DEVICES` export (after line 88):

```bash
# Ingestion-path embedder — Phenom host (192.168.0.60) with RTX 970
# Never used in sovereign hot path. Ingestion scripts only.
export OLLAMA_EMBEDDER_HOST=${OLLAMA_EMBEDDER_HOST:-192.168.0.60:11434}
export QDRANT_EMBEDDER_MODEL=${QDRANT_EMBEDDER_MODEL:-qwen3-embedding:0.6b}
export K3D_EMBEDDER_SIDECAR=${K3D_EMBEDDER_SIDECAR:-http://192.168.0.60:8080}
export QDRANT_URL=${QDRANT_URL:-http://localhost:6333}
```

### 5.2 Variable Reference

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_EMBEDDER_HOST` | `192.168.0.60:11434` | Phenom Ollama direct (used by sidecar's own env) |
| `QDRANT_EMBEDDER_MODEL` | `qwen3-embedding:0.6b` | Model tag for all embedding calls |
| `K3D_EMBEDDER_SIDECAR` | `http://192.168.0.60:8080` | Sidecar URL (ingestion scripts call this) |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant on main workstation |

**Ingestion scripts should call `K3D_EMBEDDER_SIDECAR` (the sidecar), NOT `OLLAMA_EMBEDDER_HOST`
directly.** The sidecar is the only entity that performs Matryoshka truncation.

---

## 6. Ingester Patch Targets

All files below must have their `fastembed.TextEmbedding` or `SentenceTransformer`
embedding calls replaced with HTTP calls to the sidecar at `K3D_EMBEDDER_SIDECAR`.

### 6.1 FastEmbed — Primary Targets (Qdrant ingesters)

These three files are the critical path and must be patched first:

| File | Line | Current call |
|---|---|---|
| `scripts/ingest_ptx_corpus.py` | 12 | `from fastembed import TextEmbedding` |
| `scripts/ingest_ptx_corpus.py` | 106 | `embedder = TextEmbedding(model_name=MODEL)` |
| `scripts/ingest_ptx_corpus.py` | 128 | `vecs = [vec.tolist() for vec in embedder.embed(...)]` |
| `scripts/ingest_specs_to_qdrant.py` | 22 | `from fastembed import TextEmbedding` |
| `scripts/ingest_specs_to_qdrant.py` | 184 | `embedder = TextEmbedding(model_name=EMBEDDING_MODEL)` |
| `scripts/ingest_specs_to_qdrant.py` | 205 | `embeddings = list(embedder.embed(texts_to_embed))` |
| `scripts/ingest_canonical_to_qdrant.py` | 15 | `from fastembed import TextEmbedding` |
| `scripts/ingest_canonical_to_qdrant.py` | 100 | `embedder = TextEmbedding(model_name=EMBEDDING_MODEL)` |
| `scripts/ingest_canonical_to_qdrant.py` | 104 | `vector = next(embedder.embed([document]))` |
| `knowledge3d/ingestion/canonical_lookup.py` | 178 | `from fastembed import TextEmbedding` |

### 6.2 SentenceTransformer — Secondary (ingestion-path tools, not Qdrant ingesters)

These are ingestion-path helpers that embed text for JSONL or Galaxy population.
They do not write to Qdrant directly but should be migrated for consistency:

| File | Line | Current call |
|---|---|---|
| `knowledge3d/ingestion/language/text_pipeline.py` | 90 | `from sentence_transformers import SentenceTransformer` |
| `knowledge3d/tools/training_pipelines/build_galaxy.py` | 83 | `from sentence_transformers import SentenceTransformer` |
| `knowledge3d/tools/training_pipelines/build_rlwhf_dataset.py` | 71 | `from sentence_transformers import SentenceTransformer` |
| `knowledge3d/tools/training_pipelines/house_memory.py` | 284 | `from sentence_transformers import SentenceTransformer` |
| `knowledge3d/tools/training_pipelines/ingest_open3d.py` | 93 | `from sentence_transformers import SentenceTransformer` |
| `knowledge3d/tools/training_pipelines/build_algorithmic_thinking.py` | 97 | `from sentence_transformers import SentenceTransformer` |
| `knowledge3d/tools/training_pipelines/gen_rlwhf_exaone.py` | 105 | `from sentence_transformers import SentenceTransformer` |
| `knowledge3d/tools/training_pipelines/rlwhf_from_glb.py` | 53 | `from sentence_transformers import SentenceTransformer` |
| `knowledge3d/tools/training_pipelines/rlwhf_from_offline_benchmark.py` | 55 | `from sentence_transformers import SentenceTransformer` |
| `knowledge3d/tools/training_pipelines/ingest_rl_open.py` | 99 | `from sentence_transformers import SentenceTransformer` |
| `knowledge3d/tools/embed_text_sharded.py` | 107 | `from sentence_transformers import SentenceTransformer` |
| `knowledge3d/skills/spatial_text.py` | 197 | `from sentence_transformers import SentenceTransformer` |
| `knowledge3d/models/answer_ranker.py` | 59, 116 | `from sentence_transformers import SentenceTransformer` |
| `knowledge3d/tools/evaluator_scripts/eval_rlwhf_policy.py` | 66 | `from sentence_transformers import SentenceTransformer` |
| `knowledge3d/tools/evaluator_scripts/benchmark_offline.py` | 76 | `from sentence_transformers import SentenceTransformer` |
| `knowledge3d/tools/evaluator_scripts/benchmark_chat.py` | 100 | `from sentence_transformers import SentenceTransformer` |
| `knowledge3d/tools/evaluator_scripts/eval_honesty_reward.py` | 68 | `from sentence_transformers import SentenceTransformer` |
| `knowledge3d/cranium/ptx_runtime/sovereign_multi_modal_embedder.py` | 10 | `from sentence_transformers import SentenceTransformer` |
| `knowledge3d/cranium/ptx_runtime/multi_modal_world_generator.py` | 10 | `from sentence_transformers import SentenceTransformer` |
| `k3dgen/__main__.py` | 119 | `from sentence_transformers import SentenceTransformer` |

> **Note:** `sovereign_multi_modal_embedder.py` and `multi_modal_world_generator.py` live
> under `cranium/ptx_runtime/` — these must be audited for sovereignty violations
> separately. SentenceTransformer in cranium is a sovereign violation regardless of
> this migration.

### 6.3 Hardcoded localhost:11434 in Ingestion Scripts

These call the main workstation Ollama (chat/vision models), not the embedder. They
are NOT embedding calls and should NOT be redirected to the Phenom. Leave them in place
but add a comment.

| File | Line | Note |
|---|---|---|
| `scripts/enrich_foundational_drawing_with_vision.py` | 95, 103 | Vision model (LLaVA etc.) — NOT an embedder |

### 6.4 Replacement Pattern for Codex

Each `fastembed.TextEmbedding` call should be replaced with a helper like this
(add as `knowledge3d/ingestion/embedder_client.py`):

```python
"""Ingestion-path embedder client — calls Phenom sidecar, never falls back."""
import os
import sys
from typing import List
import httpx

SIDECAR_URL = os.environ.get("K3D_EMBEDDER_SIDECAR", "http://192.168.0.60:8080")
MODEL_NAME  = os.environ.get("QDRANT_EMBEDDER_MODEL", "qwen3-embedding:0.6b")


def embed_texts(texts: List[str], truncate_dim: int = 1024) -> List[List[float]]:
    """
    Embed a batch of texts via the Phenom sidecar.
    truncate_dim: Matryoshka tier — must be in {64,128,256,512,768,1024,1536,4096}.
    Raises SystemExit if sidecar is unreachable (no fallback, per K3D policy).
    """
    try:
        resp = httpx.post(
            f"{SIDECAR_URL}/v1/embeddings",
            json={"input": texts, "model": MODEL_NAME},
            params={"truncate_dim": truncate_dim},
            timeout=120.0,
        )
        resp.raise_for_status()
    except httpx.ConnectError as exc:
        print(f"FATAL: Embedder sidecar unreachable at {SIDECAR_URL}: {exc}", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPStatusError as exc:
        print(f"FATAL: Sidecar HTTP {exc.response.status_code}: {exc.response.text}", file=sys.stderr)
        sys.exit(1)

    data = resp.json()["data"]
    return [item["embedding"] for item in data]
```

**Ingestion scripts call `embed_texts(batch, truncate_dim=1024)` for the primary tier.**
To populate ALL 8 tiers per point, call once at 4096 then truncate locally (faster than
8 round-trips):

```python
from knowledge3d.ingestion.embedder_client import embed_texts

# Embed at full dim once
full_vecs = embed_texts(texts, truncate_dim=4096)

# Build named-vector dict per point
TIERS = [64, 128, 256, 512, 768, 1024, 1536, 4096]

def make_named_vectors(full_vec: List[float]) -> dict:
    import math
    result = {}
    for dim in TIERS:
        t = full_vec[:dim]
        norm = math.sqrt(sum(x*x for x in t)) or 1.0
        result[f"qwen3-{dim}"] = [x/norm for x in t]
    return result
```

---

## 7. Acceptance Gates

Run these in order. Each must pass before proceeding to the next.

```bash
# GATE 1 — Ollama up and model loaded on Phenom
curl -s http://192.168.0.60:11434/api/tags \
  | python3 -c "import sys,json; models=[m['name'] for m in json.load(sys.stdin)['models']]; assert 'qwen3-embedding:0.6b' in models, 'MODEL MISSING'; print('PASS: model loaded')"

# GATE 2 — Ollama /api/embed returns 4096 floats
curl -s http://192.168.0.60:11434/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-embedding:0.6b","input":["sovereign gpu path"]}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); dim=len(d['embeddings'][0]); assert dim==4096, f'expected 4096 got {dim}'; print(f'PASS: {dim} dims')"

# GATE 3 — Sidecar /health returns ok
curl -sf http://192.168.0.60:8080/health \
  | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='ok'; print('PASS: sidecar health ok')"

# GATE 4 — Sidecar truncates correctly at tier 1024
curl -s http://192.168.0.60:8080/v1/embeddings?truncate_dim=1024 \
  -H "Content-Type: application/json" \
  -d '{"input":["What is the composed head pipeline?"]}' \
  | python3 -c "
import sys, json, math
d = json.load(sys.stdin)
emb = d['data'][0]['embedding']
dim = len(emb)
norm = math.sqrt(sum(x*x for x in emb))
assert dim == 1024, f'expected 1024 got {dim}'
assert abs(norm - 1.0) < 1e-3, f'not unit norm: {norm}'
print(f'PASS: 1024 dims, unit norm={norm:.6f}')
"

# GATE 5 — Sidecar hard-fails on bad tier
curl -s -o /dev/null -w "%{http_code}" \
  http://192.168.0.60:8080/v1/embeddings?truncate_dim=999 \
  -H "Content-Type: application/json" \
  -d '{"input":["test"]}' \
  | python3 -c "import sys; code=sys.stdin.read().strip(); assert code=='400', f'expected 400 got {code}'; print('PASS: invalid tier rejected')"

# GATE 6 — k3d_specifications_v2 collection exists with named vectors
curl -s "http://192.168.0.4:6333/collections/k3d_specifications_v2" \
  -H "api-key: @20Cooool58" \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)['result']
vecs = list(d['config']['params']['vectors'].keys())
expected = {'qwen3-64','qwen3-128','qwen3-256','qwen3-512','qwen3-768','qwen3-1024','qwen3-1536','qwen3-4096'}
assert expected == set(vecs), f'missing: {expected - set(vecs)}'
print(f'PASS: named vectors {sorted(vecs)}')
"

# GATE 7 — One full ingest round-trip (single doc, both tiers)
python3 - <<'PYEOF'
import os, math
os.environ.setdefault('K3D_EMBEDDER_SIDECAR', 'http://192.168.0.60:8080')
os.environ.setdefault('QDRANT_EMBEDDER_MODEL', 'qwen3-embedding:0.6b')

# Lazy import (sidecar must be running)
import httpx
from qdrant_client import QdrantClient, models

sidecar = os.environ['K3D_EMBEDDER_SIDECAR']
resp = httpx.post(f"{sidecar}/v1/embeddings?truncate_dim=4096",
                  json={"input": ["K3D Galaxy Universe memory architecture"]}, timeout=30)
assert resp.status_code == 200, f"sidecar error: {resp.text}"
full_vec = resp.json()['data'][0]['embedding']
assert len(full_vec) == 4096

def trunc(v, d):
    t = v[:d]; n = math.sqrt(sum(x*x for x in t)) or 1.0; return [x/n for x in t]

client = QdrantClient(url="http://localhost:6333", api_key="@20Cooool58")
client.upsert("k3d_specifications_v2", points=[
    models.PointStruct(
        id="00000000-0000-0000-0000-000000000001",
        vector={f"qwen3-{d}": trunc(full_vec, d) for d in [64,128,256,512,768,1024,1536,4096]},
        payload={"text": "acceptance gate test", "spec_name": "test"},
    )
])
result = client.query_points("k3d_specifications_v2",
    query=trunc(full_vec, 1024), using="qwen3-1024", limit=1)
assert result.points[0].id == "00000000-0000-0000-0000-000000000001"
print("PASS: round-trip upsert + query successful")
PYEOF
```

---

## 8. Health Monitoring

### 8.1 One-Liner Alert — Phenom Ollama Watchdog

Add to crontab on the main workstation (`crontab -e`):

```cron
*/5 * * * * curl -sf http://192.168.0.60:8080/health > /dev/null 2>&1 || \
  echo "[$(date)] FATAL: Phenom embedder sidecar DOWN — ingestion will fail" \
  >> /K3D/Knowledge3D.local/logs/embedder_health.log
```

### 8.2 Ingestion Start Check

Every ingestion script should run a preflight before processing any data.
Add as the first call in each ingester's `main()`:

```python
def _preflight_embedder():
    import httpx, sys
    sidecar = os.environ.get("K3D_EMBEDDER_SIDECAR", "http://192.168.0.60:8080")
    try:
        r = httpx.get(f"{sidecar}/health", timeout=5.0)
        if r.status_code != 200:
            print(f"FATAL: Embedder sidecar unhealthy ({r.status_code}): {r.text}", file=sys.stderr)
            sys.exit(1)
    except Exception as exc:
        print(f"FATAL: Embedder sidecar unreachable at {sidecar}: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Embedder sidecar OK at {sidecar}")
```

### 8.3 Systemd Status One-Liner (run from main workstation via SSH)

```bash
ssh daniel@192.168.0.60 "systemctl is-active ollama k3d-embedder"
# Expected: active\nactive
```

---

## 9. Deployment Order Summary for Codex

1. SSH to Phenom, install Ollama (`§1.2`)
2. Configure systemd override for `OLLAMA_HOST=0.0.0.0:11434` (`§1.3`)
3. Pull `qwen3-embedding:0.6b` (`§1.4`)
4. Open UFW ports 11434 and 8080 (`§1.5`)
5. Deploy sidecar script + dependencies + systemd unit (`§2`)
6. Run Gates 1–5 on main workstation (`§7`)
7. Create v2 collections on Qdrant (`§3.3`)
8. Create payload indexes on v2 collections (`§3.4`)
9. Add env vars to `scripts/k3d_env.sh` (`§5.1`)
10. Patch `scripts/ingest_specs_to_qdrant.py`, `scripts/ingest_ptx_corpus.py`,
    `scripts/ingest_canonical_to_qdrant.py` to call sidecar (`§6.4`)
11. Re-ingest all three collections into v2 (`§4.2`)
12. Verify v2 point counts (`§4.3`)
13. Run Gate 6 + Gate 7 (`§7`)
14. Atomic alias swap + MCP container restart (`§4.4`)
15. Verify Gate 7 searches against live alias (`§4.5`)
16. After 24h soak: delete old collections (`§4.6`)
17. Audit `cranium/ptx_runtime/sovereign_multi_modal_embedder.py` separately
    (SentenceTransformer in cranium = sovereignty violation, separate fix required)

---

*Spec complete. Codex executes. Claude does not run deploy commands.*
