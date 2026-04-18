# PTX Knowledge Base + MCP Wiring Spec

**Author:** Claude (Architecture Partner)
**Date:** 2026-04-16
**For:** Codex
**Goal:** Give K3D agents first-class access to CUDA/PTX reference material via Qdrant + a dedicated MCP, and stop leaking tokens to a local planner model when a cloud sibling already exists.

---

## 0. What Ships (three artifacts, one PR)

1. **`scripts/ingest_ptx_corpus.py`** — idempotent ingester that loads the PTX PDF/JSON corpus at `/mnt/arquivos/0 ChatGPTs/DataBase/Programming Languages/PTX/` into a new Qdrant collection `k3d_ptx` using the existing `all-MiniLM-L6-v2` (384-dim) embedding model.
2. **Planner-model swap** in the docker-mounted `/home/daniel/.claude/ollama_specialists.py` at line 40: `PLANNER = "qwen3.5:latest"` → `PLANNER = "qwen3.5:397b-cloud"`. Restart `k3d-ollama-mcp`.
3. **New MCP container `k3d-ptx-mcp`** — reuses the existing `k3d-qdrant-mcp:latest` image unchanged, just overrides `COLLECTION_NAME=k3d_ptx` + `TOOL_FIND_DESCRIPTION` + port `8503`. Register it in `/home/daniel/.claude.json` under `mcpServers`.

No new Dockerfiles. No new MCP server code. The existing `mcp_server_qdrant` image is collection-agnostic by env.

---

## 1. Environment (as discovered — do NOT re-derive)

- Qdrant: docker container `qdrant` at `host.docker.internal:6333`, api key `@20Cooool58`.
- Existing MCP containers:
  - `k3d-knowledge-mcp` → port 8501, image `k3d-qdrant-mcp:latest`, collection `k3d_specifications`.
  - `k3d-ollama-mcp` → port 8502, image `k3d-ollama-mcp:latest`, mounts `/home/daniel/.claude/ollama_specialists.py` → `/app/ollama_specialists.py`.
- MCP registry: `/home/daniel/.claude.json` (`mcpServers` object, `streamableHttp` transport).
- PTX corpus directory listing:
  ```
  CUDA_C_Programming_Guide.pdf
  Inline_PTX_Assembly.pdf           Inline_PTX_Assembly.pdf.json
  ptx_isa_8.5.pdf                   ptx_isa_8.5.pdf.json
  ptx_isa_8.7.pdf
  ptx_isa_9.0.pdf                   ptx_isa_9.0.pdf.json
  "PTX ISA - ptx_isa_9.0.pdf"       "PTX ISA - ptx_isa_9.0.pdf.json"
  ```
  Four files have pre-parsed `.json` siblings — **reuse them**. The two without (`CUDA_C_Programming_Guide.pdf`, `ptx_isa_8.7.pdf`) get live-parsed by `pypdf`.

---

## 2. Step 1 — PTX Ingester

### 2.1 File: `scripts/ingest_ptx_corpus.py`

Must be runnable standalone on the `k3d-rapids` or any env with `qdrant-client`, `sentence-transformers`, `pypdf` available. Uses **the same embedding model the existing `k3d-qdrant-mcp` image uses** so the same image can serve the new collection with no code changes: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, cosine).

### 2.2 Behaviour contract

- Creates collection `k3d_ptx` (384-dim, cosine) if missing. No-op if present.
- For each `.pdf` in the corpus dir:
  - If a sibling `<name>.pdf.json` exists and parses as a list of `{"text": ...}` dicts, use it. Else extract with `pypdf`.
  - Concatenate pages → split by section heading regex `r'^\s*(\d+(\.\d+)*)\.?\s+[A-Z]'`.
  - Further split any chunk >1800 chars at whitespace; continuation chunks get `.cont` suffix on their `section` payload.
- Point IDs are deterministic `uuid.uuid5(NAMESPACE_URL, f"{source}:{section}:{chunk_idx}")` → re-runs upsert, no duplicates.
- Payload schema: `{source, section, chunk_idx, text, version}` where `version` is derived from filename (`ptx_isa_9.0`, `ptx_isa_8.7`, `ptx_isa_8.5`, `inline_ptx_asm`, `cuda_c_guide`).
- Batch upserts of 64 points.
- Logs one line per file with chunk count and elapsed seconds.

### 2.3 Starter skeleton (ollama-drafted; Codex — complete the two `# TODO` regions, do not ship stubs)

```python
# scripts/ingest_ptx_corpus.py
import os, json, uuid, re, logging, time
from pathlib import Path
from typing import List, Dict, Any
import pypdf
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

COLLECTION = "k3d_ptx"
DIM = 384
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
API_KEY    = os.getenv("QDRANT_API_KEY", "@20Cooool58")
CORPUS     = Path("/mnt/arquivos/0 ChatGPTs/DataBase/Programming Languages/PTX/")
CHUNK      = 1800
BATCH      = 64
SEC_RE     = re.compile(r'^\s*(\d+(\.\d+)*)\.?\s+[A-Z]')
NS         = uuid.NAMESPACE_URL

VERSION_MAP = {
    "ptx_isa_9.0": "ptx_isa_9.0",
    "PTX ISA - ptx_isa_9.0": "ptx_isa_9.0_annotated",
    "ptx_isa_8.7": "ptx_isa_8.7",
    "ptx_isa_8.5": "ptx_isa_8.5",
    "Inline_PTX_Assembly": "inline_ptx_asm",
    "CUDA_C_Programming_Guide": "cuda_c_guide",
}

def load_pages(pdf: Path) -> List[str]:
    js = pdf.with_suffix(pdf.suffix + ".json")
    if js.exists():
        data = json.loads(js.read_text(encoding="utf-8"))
        if isinstance(data, list) and all(isinstance(p, dict) and "text" in p for p in data):
            return [p["text"] for p in data if p["text"].strip()]
    return [p.extract_text() or "" for p in pypdf.PdfReader(str(pdf)).pages]

def chunk_sections(text: str) -> List[Dict[str, Any]]:
    # TODO(codex): implement section split per §2.2. Return list of {"section": "...", "chunk_idx": N, "text": "..."}.
    ...

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    model  = SentenceTransformer(MODEL)
    client = QdrantClient(url=QDRANT_URL, api_key=API_KEY)
    if COLLECTION not in {c.name for c in client.get_collections().collections}:
        client.create_collection(COLLECTION, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))

    for pdf in sorted(CORPUS.glob("*.pdf")):
        t0 = time.time()
        stem = pdf.stem
        version = VERSION_MAP.get(stem, stem.lower().replace(" ", "_"))
        text = "\n".join(load_pages(pdf))
        chunks = chunk_sections(text)
        points = []
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i:i+BATCH]
            vecs  = model.encode([c["text"] for c in batch], normalize_embeddings=True).tolist()
            # TODO(codex): build PointStruct list with deterministic uuid5 id and payload per §2.2, upsert batch.
            ...
        logging.info("%-40s chunks=%d elapsed=%.1fs", pdf.name, len(chunks), time.time() - t0)

if __name__ == "__main__":
    main()
```

### 2.4 Acceptance

- `python3 scripts/ingest_ptx_corpus.py` completes with no stubs hit.
- `curl -s -H 'api-key: @20Cooool58' http://localhost:6333/collections/k3d_ptx | jq '.result.points_count'` > 0.
- Re-running the script does not grow `points_count` (idempotent upsert proof).

---

## 3. Step 2 — Swap planner model to cloud

**File:** `/home/daniel/.claude/ollama_specialists.py` (bind-mounted into `k3d-ollama-mcp`).

**Edit line 40 only:**

```diff
-PLANNER      = "qwen3.5:latest"                # local Qwen3.5 — default planner/router
+PLANNER      = "qwen3.5:397b-cloud"            # cloud Qwen3.5 397B — default planner/router
```

Leave `SUMMARIZER`, `EXTRACTOR`, `CODER` untouched.

**Restart the container** so the file re-imports:

```
docker restart k3d-ollama-mcp
```

**Smoke test:**
```
curl -s -X POST http://localhost:8502/mcp/tools/call \
  -H 'Content-Type: application/json' \
  -d '{"name":"plan_task","arguments":{"task":"hello"}}' | head -c 200
```
Response should be a numbered plan, not a connection error. Latency jumps from local-ms to cloud-second range — that's expected and is the point.

---

## 4. Step 3 — k3d-ptx MCP container

### 4.1 No Dockerfile — reuse `k3d-qdrant-mcp:latest`

The image's entrypoint is literally `python3 -c "from mcp_server_qdrant.server import mcp; mcp.run(transport='streamable-http', host='0.0.0.0', port=8000)"`. It picks collection + description up from env. So:

```bash
docker run -d \
  --name k3d-ptx-mcp \
  --restart unless-stopped \
  --add-host=host.docker.internal:host-gateway \
  -p 8503:8000 \
  -e QDRANT_URL=http://host.docker.internal:6333 \
  -e QDRANT_API_KEY='@20Cooool58' \
  -e COLLECTION_NAME=k3d_ptx \
  -e EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
  -e TOOL_STORE_DESCRIPTION='Store CUDA/PTX reference excerpts into the PTX knowledge base.' \
  -e TOOL_FIND_DESCRIPTION='Search CUDA/PTX reference material. Use BEFORE writing PTX/CUDA code or debugging kernel issues — covers PTX ISA 8.5/8.7/9.0, Inline PTX Assembly, and the CUDA C Programming Guide.' \
  k3d-qdrant-mcp:latest
```

Check: `docker ps | grep k3d-ptx-mcp` shows it up on 8503, and `curl -s http://localhost:8503/mcp/` returns a non-empty MCP handshake.

### 4.2 Register in MCP config

Edit `/home/daniel/.claude.json`, inside `"mcpServers": { ... }`, **add a third entry** alongside `k3d-knowledge` and `ollama-specialists`:

```json
"k3d-ptx": {
  "type": "streamableHttp",
  "url": "http://localhost:8503/mcp/",
  "disabled": false,
  "autoApprove": [
    "qdrant-find",
    "qdrant-store"
  ],
  "timeout": 60
}
```

Do **not** change `k3d-knowledge` or `ollama-specialists`. Do not delete or reorder.

### 4.3 Persistence note

If Daniel's daemons / systemd manage docker lifecycle, also add an equivalent entry wherever `k3d-knowledge-mcp` is declared (e.g. a `docker-compose.yml` under `deploy/` or `k3d-docker/` — search for `k3d-knowledge-mcp` with Grep; if you find a compose file, mirror the service block with the env overrides in §4.1). If no compose file governs these containers (they were launched by ad-hoc `docker run`), save the launch command at `deploy/docker/k3d-ptx-mcp.run.sh` with `chmod +x`.

---

## 5. Proof-of-work checklist for the report

1. `scripts/ingest_ptx_corpus.py` lands with every `# TODO(codex)` replaced by real code (grep proof).
2. `points_count` on `k3d_ptx` > 0 and stable across two consecutive ingester runs.
3. `/home/daniel/.claude/ollama_specialists.py` diff = single-line planner swap.
4. `docker ps` lists `k3d-ptx-mcp` on port 8503.
5. `/home/daniel/.claude.json` has the new `k3d-ptx` server entry, no other fields touched.
6. A `plan_task` call returns via the cloud planner (latency ≥ ~1s, proves cloud route).
7. A `mcp__k3d-ptx__qdrant-find("shared memory atomicAdd PTX ISA")` returns hits with source fields referring to the ingested PDFs.

---

## 6. Anti-drift reminders (read before coding)

- **No stubs.** If the ingester's `chunk_sections` feels hard, call `mcp__ollama-specialists__ask_coder` with the actual regex and chunk-size contract in §2.2. Do **not** leave `...` in committed code.
- **No new MCP server code.** `mcp_server_qdrant` already does `qdrant-find`/`qdrant-store`. You are only configuring a second instance of it.
- **No image rebuild.** Same `k3d-qdrant-mcp:latest` image, different env.
- **Do not touch** `k3d-knowledge-mcp`, `k3d-ollama-mcp`, `k3d_specifications` collection, or any other env vars of the existing containers.
- **MCP timeout for kimi_swarm / deep ollama calls stays at 240000 ms** per Daniel's prior directive.
- **Claude = architecture, not implementation.** This file IS the spec. Codex writes the code.

---

## 7. Handoff line to paste at the top of your report

> "PTX corpus ingested into Qdrant collection `k3d_ptx` (N points) via `scripts/ingest_ptx_corpus.py`. Planner swap `qwen3.5:latest → qwen3.5:397b-cloud` live and smoke-tested. `k3d-ptx-mcp` container up on :8503, registered in `~/.claude.json`, `qdrant-find` returning PTX hits end-to-end."

If you cannot write that line truthfully, you are not done.
