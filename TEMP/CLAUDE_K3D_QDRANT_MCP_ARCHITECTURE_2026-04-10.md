# K3D Qdrant Knowledge Base + Multi-User MCP Architecture

**Date:** 2026-04-10  
**Author:** Claude (Architecture Partner)  
**Status:** Implementation-ready  
**Grounding:** CLAUDE.md, KNOWLEDGEVERSE_SPECIFICATION.md, FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md

---

## 1. Executive Summary

Two Docker containers, both Watchtower-watched, both SSE-transport MCP servers:

| Container | Image | Port | Purpose |
|-----------|-------|------|---------|
| `k3d-knowledge-mcp` | Official `qdrant/mcp-server-qdrant` | **8001** | K3D specification knowledge base — semantic search over 35 docs/vocabulary/*.md files |
| `k3d-ollama-mcp` | Custom (from `ollama_specialists.py`) | **8002** | Multi-user ollama specialists — all existing tools via SSE |

**Existing infrastructure (UNTOUCHED):**
- Qdrant on 6333/6334 — already running, Watchtower-watched
- n8n on 5678, open-webui on 3000, browserless on 3100 — all untouched
- Watchtower with `--cleanup --include-restarting` — picks up new containers automatically
- docker-compose-west.yml (OpenSPG) — untouched

---

## 2. Container 1: k3d-knowledge-mcp (Qdrant MCP Server)

### What It Does
Exposes two MCP tools to ALL agents:
- **`qdrant-store`** — Store information into the K3D knowledge base
- **`qdrant-find`** — Semantic search: "What does the spec say about Galaxy Universe?"

### Source
Official: `qdrant/mcp-server-qdrant` — trusted, maintained by Qdrant team.  
PyPI: `mcp-server-qdrant`  
Dockerfile: already in repo (uses `uvx mcp-server-qdrant --transport sse`)

### Configuration

```bash
docker run -d \
  --name k3d-knowledge-mcp \
  --restart unless-stopped \
  -p 8001:8000 \
  -e QDRANT_URL="http://host.docker.internal:6333" \
  -e QDRANT_API_KEY="@20Cooool58" \
  -e COLLECTION_NAME="k3d_specifications" \
  -e EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2" \
  -e TOOL_STORE_DESCRIPTION="Store K3D architecture knowledge, spec excerpts, or design decisions into the Knowledge3D specification database. Use when you learn something that all agents should know." \
  -e TOOL_FIND_DESCRIPTION="Search the Knowledge3D specification database. Use this BEFORE answering questions about: Galaxy Universe, House architecture, TRM, Knowledgeverse, sovereignty, RPN, Three Brain System, composed head pipeline, or any K3D architectural concept. Returns relevant spec excerpts with source file paths." \
  -e FASTMCP_HOST="0.0.0.0" \
  qdrant/mcp-server-qdrant:latest
```

### Embedding Model Choice
**FastEmbed `sentence-transformers/all-MiniLM-L6-v2`** — runs on CPU inside the container, no GPU needed, 384 dimensions, fast enough for 35 docs.

Alternative for higher quality: `nomic-ai/nomic-embed-text-v1.5` (768 dims, also FastEmbed-supported, still CPU).

### Network
- Container connects to host Qdrant via `host.docker.internal:6333`
- On Linux, add `--add-host=host.docker.internal:host-gateway` to docker run
- SSE endpoint exposed at `http://localhost:8001/sse`

---

## 3. Container 2: k3d-ollama-mcp (Multi-User Specialists)

### What It Does
Same 14+ tools as current `ollama_specialists.py`, but:
- Runs as SSE HTTP server (not stdio)
- Multiple users/agents can connect simultaneously
- Dockerized for Watchtower auto-updates

### Changes Required to `ollama_specialists.py`

**Line 1264 — change transport:**
```python
# OLD:
if __name__ == "__main__":
    mcp.run(transport="stdio")

# NEW:
if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
```

**Line 46 — Ollama URL must be configurable:**
```python
# OLD:
OLLAMA = "http://192.168.0.4:11434"

# NEW:
import os
OLLAMA = os.environ.get("OLLAMA_URL", "http://192.168.0.4:11434")
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastmcp \
    requests \
    duckduckgo-search \
    uvicorn

COPY ollama_specialists.py .

EXPOSE 8000

CMD ["python", "ollama_specialists.py"]
```

### Docker Run

```bash
docker run -d \
  --name k3d-ollama-mcp \
  --restart unless-stopped \
  -p 8002:8000 \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_URL="http://host.docker.internal:11434" \
  -v /home/daniel/.claude/ollama_specialists.py:/app/ollama_specialists.py:ro \
  -v "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D":/repo:ro \
  k3d-ollama-mcp:latest
```

Note: Mount the repo read-only so specialists can read files (their `repo_files` parameter).

### Build

```bash
# Build from a minimal context
cd /home/daniel/.claude
docker build -t k3d-ollama-mcp:latest -f- . <<'DOCKERFILE'
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastmcp requests duckduckgo-search uvicorn
COPY ollama_specialists.py .
EXPOSE 8000
CMD ["python", "ollama_specialists.py"]
DOCKERFILE
```

---

## 4. Claude Code MCP Configuration

### Current (stdio, single-user):
```json
{
  "ollama-specialists": {
    "command": "python3",
    "args": ["/home/daniel/.claude/ollama_specialists.py"]
  }
}
```

### New (SSE, multi-user):
```json
{
  "ollama-specialists": {
    "url": "http://localhost:8002/sse"
  },
  "k3d-knowledge": {
    "url": "http://localhost:8001/sse"
  }
}
```

This goes in the project-level settings at:
`/home/daniel/.claude/projects/-mnt-arquivos-EchoSystems-AI-Studios-Knowledge-3D-Standard-GitHub-Knowledge3D/settings.json`

---

## 5. Knowledge Ingestion Pipeline

### Purpose
Chunk 35 docs/vocabulary/*.md files into the `k3d_specifications` Qdrant collection so agents can semantically search them.

### Script: `scripts/ingest_specs_to_qdrant.py`

Runs once (and whenever specs change). Uses the same embedding model as the MCP server.

### Chunking Strategy
- Split on `##` and `###` headers (preserve hierarchy)
- Preserve code blocks intact (don't split mid-code)
- Target chunk size: ~500 tokens with 50 token overlap
- Store metadata: `spec_name`, `section_path`, `content_type` (definition/requirement/example/code)

### Metadata Schema per Point

```json
{
  "text": "The chunk content including section header breadcrumbs",
  "spec_name": "knowledgeverse",
  "file_path": "docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md",
  "section_path": ["Core Architecture", "7 Memory Regions"],
  "section_level": 2,
  "content_type": "definition",
  "line_start": 145
}
```

### Payload Indexes (created once)
- `spec_name` — KeywordIndex (exact match filter: "show me only KNOWLEDGEVERSE_SPECIFICATION")
- `content_type` — KeywordIndex (filter: "only requirements", "only code examples")

### Embedding
Uses `fastembed` with `sentence-transformers/all-MiniLM-L6-v2` (same as MCP server) — CPU-only, no network call to GTX 970 needed.

Alternative: Use Ollama `nomic-embed-text` on 192.168.0.60:11434 (GTX 970) for higher quality embeddings. The MCP server would then need `EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5` to match.

---

## 6. Collection Setup

Run once before first ingestion:

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(
    url="http://localhost:6333",
    api_key="@20Cooool58"
)

client.create_collection(
    collection_name="k3d_specifications",
    vectors_config=models.VectorParams(
        size=384,  # all-MiniLM-L6-v2
        distance=models.Distance.COSINE
    )
)

# Payload indexes for filtered search
client.create_payload_index(
    "k3d_specifications",
    field_name="spec_name",
    field_schema=models.PayloadSchemaType.KEYWORD
)
client.create_payload_index(
    "k3d_specifications",
    field_name="content_type",
    field_schema=models.PayloadSchemaType.KEYWORD
)
```

---

## 7. Multi-User Access Matrix

| Agent | Connection | Config |
|-------|-----------|--------|
| **Claude Code (Daniel)** | SSE `http://localhost:8001/sse` + `http://localhost:8002/sse` | Project settings.json |
| **Codex** | SSE `http://localhost:8001/sse` + `http://localhost:8002/sse` | Codex MCP config |
| **Other Claude instances** | SSE (same URLs) | Per-project MCP config |
| **Open WebUI agents** | Direct Qdrant SDK `http://localhost:6333` | Qdrant client in pipeline |
| **Custom scripts** | Qdrant Python SDK | `qdrant-client` package |

### Why SSE Multi-User Works
- SSE transport = HTTP server, accepts concurrent connections
- No file locks, no stdio contention
- Watchtower auto-updates both containers

---

## 8. Watchtower Integration

Both new containers use `--restart unless-stopped` (same as existing containers). Watchtower automatically watches all running containers — no config change needed.

Watchtower current config: `--cleanup --include-restarting` — will auto-update both:
- `qdrant/mcp-server-qdrant:latest` (official, gets upstream updates)
- `k3d-ollama-mcp:latest` (local build, manual rebuild when `ollama_specialists.py` changes)

---

## 9. Implementation Steps

### Step 1: Modify ollama_specialists.py for SSE transport
- Change line 1264: `mcp.run(transport="sse", host="0.0.0.0", port=8000)`
- Change line 46: `OLLAMA = os.environ.get("OLLAMA_URL", "http://192.168.0.4:11434")`
- Test locally: `python3 /home/daniel/.claude/ollama_specialists.py` → should listen on 8000

### Step 2: Build and launch k3d-ollama-mcp container
```bash
cd /home/daniel/.claude
docker build -t k3d-ollama-mcp:latest -f Dockerfile.ollama .
docker run -d --name k3d-ollama-mcp --restart unless-stopped \
  -p 8002:8000 \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_URL="http://host.docker.internal:11434" \
  -v /home/daniel/.claude/ollama_specialists.py:/app/ollama_specialists.py:ro \
  -v "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D":/repo:ro \
  k3d-ollama-mcp:latest
```

### Step 3: Launch k3d-knowledge-mcp container
```bash
docker run -d --name k3d-knowledge-mcp --restart unless-stopped \
  -p 8001:8000 \
  --add-host=host.docker.internal:host-gateway \
  -e QDRANT_URL="http://host.docker.internal:6333" \
  -e QDRANT_API_KEY="@20Cooool58" \
  -e COLLECTION_NAME="k3d_specifications" \
  -e EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2" \
  -e TOOL_STORE_DESCRIPTION="Store K3D architecture knowledge into the specification database." \
  -e TOOL_FIND_DESCRIPTION="Search K3D specifications. Use BEFORE answering about Galaxy Universe, House, TRM, Knowledgeverse, sovereignty, RPN, Three Brain System, or any K3D concept." \
  -e FASTMCP_HOST="0.0.0.0" \
  qdrant/mcp-server-qdrant:latest
```

### Step 4: Create and run ingestion script
```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
python3 scripts/ingest_specs_to_qdrant.py
```

### Step 5: Update Claude Code MCP configuration
Update project settings to use SSE URLs instead of stdio command.

### Step 6: Verify
```bash
# Test knowledge MCP
curl http://localhost:8001/sse

# Test ollama MCP  
curl http://localhost:8002/sse

# Test Qdrant collection
curl -H "api-key: @20Cooool58" http://localhost:6333/collections/k3d_specifications
```

---

## 10. Future: Ingesting CLAUDE.md, CODEX.md, TEMP/*.md

After the base 35 specs are ingested, extend to:
- `CLAUDE.md`, `CODEX.md`, `AGENTS.md` — agent collaboration docs
- `TEMP/*.md` — recent specs and reports (re-ingest periodically)
- `docs/briefings/*.md` — architectural briefings

This gives agents complete grounded context without burning tokens reading files.

---

## 11. qdrant/skills Clarification

**`qdrant/skills`** is NOT an MCP server. It's a collection of markdown "skills" (solution architect guides) for coding agents about Qdrant best practices (scaling, performance, deployment, etc.).

**What we actually use:** `qdrant/mcp-server-qdrant` — the official Qdrant MCP server.

The skills repository's patterns (structured markdown → agent-consumable knowledge) inspired our approach: chunk our own specs and make them agent-searchable via the same Qdrant infrastructure.
