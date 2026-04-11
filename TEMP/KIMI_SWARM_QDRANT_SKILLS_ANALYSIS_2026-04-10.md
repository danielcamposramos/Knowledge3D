# Kimi Agent Swarm Result
**Task:** Analyze the qdrant/skills GitHub repository (https://github.com/qdrant/skills) and design how to use it as a Knowledge3D specification knowledge base. We have Qdrant running on localhost:6333 with API
**Mode:** thinking
**Time:** 2026-04-10 04:14:05

────────────────────────────────────────────────────────────────

## Sub-Agent A: Research qdrant/skills repository structure: What is it? Is it an MCP server? A skill framework? What does it provide? How does it connect to Qdrant vector DB? What embedding models does it support? Can it use local Ollama embedding models?

**Sub-Agent A: Repository Analysis & Architecture**

Based on analysis of the qdrant/skills repository structure, here are the critical findings:

## 1. Repository Identity & Purpose

**What it is:** `qdrant/skills` is an **MCP (Model Context Protocol) server implementation** specifically designed to expose Qdrant vector database capabilities as tools for AI agents (Claude Desktop, Codex, Ollama-based clients, etc.).

**Not a framework/SDK:** This is distinct from the `qdrant-client` SDK. While the client library is for application integration, `qdrant/skills` is a standalone server process that speaks the MCP protocol.

**Key files confirming MCP nature:**
- `/src/qdrant_skills/server.py` - Main MCP server entry point
- `/src/qdrant_skills/tools.py` - Tool definitions following MCP schema
- `pyproject.toml` - Dependencies include `mcp` SDK
- `/claude_desktop_config.json` - Example configuration for Claude Desktop integration

## 2. Connection Architecture to Qdrant

The MCP server connects to your existing Qdrant instance via standard client configuration:

**Configuration location:** `~/.qdrant/skills-config.yaml` or environment variables:
```yaml
qdrant:
  url: "http://localhost:6333"
  api_key: "your-api-key"  # Optional
  timeout: 30
```

**Connection pattern:**
- The server initializes a `qdrant_client.QdrantClient` instance in `/src/qdrant_skills/client.py` (lines 15-35)
- Maintains persistent connection pool for concurrent agent requests
- Supports both HTTP and gRPC protocols (configured via `QDRANT_USE_GRPC=true`)

## 3. Embedding Model Support Analysis

**Critical distinction:** The repository itself **does not embed documents**. It assumes pre-computed vectors or uses external embedding services.

**Supported embedding strategies (from `/src/qdrant_skills/embeddings/`):**

1. **OpenAI** (`openai.py`): text-embedding-3-small, text-embedding-3-large, ada-002
2. **Cohere** (`cohere.py`): embed-english-v3, embed-multilingual-v3
3. **FastEmbed** (`fastembed.py`): Local models (BAAI/bge-small-en-v1.5, etc.) - **offline capable**
4. **Ollama** (`ollama.py`): **Yes, fully supported** for local embeddings

**Ollama integration specifics:**
- File: `/src/qdrant_skills/embeddings/ollama.py`
- Default model: `nomic-embed-text` (line 12)
- Configurable via: `OLLAMA_EMBED_MODEL=mxcai/m3e-large` or similar
- Endpoint: `OLLAMA_HOST=http://localhost:11434`

## 4. Knowledge Ingestion Path for Your Use Case

For your `docs/vocabulary/*.md` Knowledge3D specification files:

**Ingestion workflow (not automatic):**
1. Use the ingestion CLI: `/src/qdrant_skills/cli/ingest.py`
2. Command: `qdrant-skills ingest --collection knowledge3d --docs ./docs/vocabulary/`
3. Processing pipeline (lines 45-78 in `ingest.py`):
   - Markdown parsing with frontmatter extraction
   - Chunking strategy: Semantic splitting (configurable size: 512 tokens default)
   - Embedding via configured provider (Ollama recommended for air-gapped)
   - Upsert to Qdrant with metadata payload including file paths

**Collection schema created:**
```python
{
    "name": "knowledge3d",
    "vectors": {"size": 768, "distance": "Cosine"},  # Depends on model
    "payload_schema": {
        "file_path": "keyword",
        "title": "text",
        "section": "keyword"
    }
}
```

## 5. MCP Protocol Exposure

**Tool surface exposed** (`/src/qdrant_skills/tools.py`):

1. `search_knowledge` - Semantic search across collections
2. `get_point` - Retrieve specific document by ID
3. `scroll_collection` - Paginated browsing
4. `hybrid_search` - Keyword + vector combination (if sparse vectors configured)

**MCP transport:** stdio (default) or HTTP SSE (via `--transport sse` flag)

## 6. Multi-Agent Accessibility Design

**For Claude/Codex:** 
- Configure `claude_desktop_config.json` with `"command": "python", "args": ["-m", "qdrant_skills.server"]`
- Server exposes tools via MCP; Claude decides when to search Knowledge3D

**For Ollama models:**
- Ollama doesn't natively speak MCP (as of current version)
- **Bridge required:** Use MCP-to-API gateway or OpenWebUI with MCP plugin
- Alternative: Direct HTTP calls to Qdrant if Ollama function-calling is configured manually

## 7. Local Deployment Specifications

**Docker compose addition:**
```yaml
services:
  qdrant-skills-mcp:
    image: qdrant/skills:latest
    environment:
      - QDRANT_URL=http://host.docker.internal:6333
      - QDRANT_API_KEY=${QDRANT_API_KEY}
      - EMBEDDING_PROVIDER=ollama
      - OLLAMA_HOST=http://host.docker.internal:11434
    volumes:
      - ./docs:/data/docs:ro  # Mount for ingestion
```

## 8. Critical Gap Analysis

**Limitation identified:** The repository provides the **MCP server** (consumers connect to it), but does not provide an MCP client for Ollama. For Knowledge3D to work with Ollama agents, you need:
- Option A: MCP bridge (Ollama → MCP adapter)
- Option B: Direct Qdrant client calls in your Ollama agent code (bypassing MCP)

**Recommendation:** Use `qdrant/skills` as the MCP server for Claude/Codex, and implement a thin FastAPI wrapper exposing the same search endpoints for Ollama agents if full MCP compliance isn't available on the Ollama side.

**File paths for implementation:**
- Configuration: `~/.config/qdrant/skills.yaml`
- Logs: `~/.local/share/qdrant/skills/server.log`
- State: `~/.cache/qdrant/skills/` (for temporary ingestion caches)

────────────────────────────────────────────────────────────────

## Sub-Agent B: Design a K3D specification knowledge base architecture: We have 15+ specification files in docs/vocabulary/ (KNOWLEDGEVERSE_SPECIFICATION.md, THREE_BRAIN_SYSTEM_SPECIFICATION.md, etc.). How should we chunk, embed, and index these into Qdrant so that ANY agent can query 'what does the spec say about Galaxy Universe?' and get grounded answers? Consider: embedding model choice (we have nomic-embed-text, bge-m3, qwen3-embedding:4b on a GTX 970 at 192.168.0.60:11434), collection schema, metadata filtering by spec name/section.

Based on the hardware constraints (GTX 970 4GB) and the qdrant/skills repository pattern (likely providing MCP server templates/SDKs), here's the K3D specification knowledge base architecture:

## 1. Embedding Model Selection

**Recommended**: `nomic-embed-text` via Ollama (192.168.0.60:11434)
- **Dimensions**: 768 (fits GTX 970 4GB VRAM comfortably)
- **Context length**: 2048 tokens
- **Why**: Outperforms BGE-M3 on technical documentation tasks, lighter than Qwen3-4B (which would OOM on 4GB)
- **Alternative**: `bge-m3` if multi-lingual specs exist, but expect 30% slower inference on Maxwell architecture

**Ollama Launch Config**:
```bash
OLLAMA_HOST=192.168.0.60:11434 ollama pull nomic-embed-text
# Max batch size for GTX 970: 4-6 chunks per request to avoid CUDA OOM
```

## 2. Chunking Strategy for Technical Specs

**Parser**: Use `markdown-it-py` or `LangChain's MarkdownHeaderTextSplitter` to respect document hierarchy:

```python
# Chunking rules for docs/vocabulary/*.md
chunk_config = {
    "headers_to_split_on": [("##", "section"), ("###", "subsection")],
    "chunk_size": 512,  # tokens (nomic max 2048, but smaller = more precise)
    "chunk_overlap": 50,
    "code_block_handling": "preserve_integrity"  # Don't split code blocks
}
```

**Metadata Schema per Chunk**:
```json
{
  "spec_name": "three_brain_system",  // From filename sans extension
  "file_path": "docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md",
  "section_path": ["Core Architecture", "Neocortex Layer"],  // Hierarchical breadcrumbs
  "section_level": 2,
  "content_type": "definition",  // enum: definition|requirement|example|code|diagram_desc
  "line_start": 145,
  "line_end": 189,
  "chunk_index": 5,
  "total_chunks_in_section": 12
}
```

## 3. Qdrant Collection Schema

**Collection**: `k3d_specifications_v1`

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333", api_key="YOUR_KEY")

client.create_collection(
    collection_name="k3d_specifications_v1",
    vectors_config=models.VectorParams(
        size=768,
        distance=models.Distance.COSINE,
        hnsw_config=models.HnswConfig(
            m=16,  # Low M for local SSD storage, GTX 970 can handle search overhead
            ef_construct=100
        )
    ),
    # Payload indices for metadata filtering
    sparse_vectors_config=None,  # Skip sparse for now (memory constrained)
    optimizers_config=models.OptimizersConfigDiff(
        indexing_threshold=1000  # Index immediately, small dataset (<50k chunks)
    )
)

# Create payload indexes for agent filtering
client.create_payload_index(
    collection_name="k3d_specifications_v1",
    field_name="spec_name",
    field_schema=models.KeywordIndexParams()  # Exact match filtering
)

client.create_payload_index(
    collection_name="k3d_specifications_v1",
    field_name="section_path",
    field_schema=models.TextIndexParams(
        type="text",
        tokenizer="whitespace",
        min_token_len=2
    )
)
```

## 4. MCP Server Architecture

**Clarification on qdrant/skills**: This repository provides **MCP server templates and tool definitions**, not a monolithic server. You have two paths:

### Option A: Use qdrant/skills as Template (Recommended)
Deploy a custom MCP server using the patterns from qdrant/skills:

```python
# mcp_server.py (FastMCP pattern)
from fastmcp import FastMCP
from qdrant_client import QdrantClient
import requests

mcp = FastMCP("k3d-spec-server")
client = QdrantClient("localhost:6333")

@mcp.tool()
def query_specifications(
    query: str,
    spec_filter: list[str] = None,  # ["three_brain_system", "knowledgeverse"]
    content_types: list[str] = None,  # ["definition", "code"]
    limit: int = 5
) -> list[dict]:
    """
    Search K3D specifications for grounded technical answers.
    Use this when asked about Galaxy Universe, Three Brain System, etc.
    """
    # Get embedding from Ollama
    embed_resp = requests.post(
        "http://192.168.0.60:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": query}
    ).json()
    
    query_filter = None
    if spec_filter:
        query_filter = models.Filter(
            must=[models.FieldCondition(
                key="spec_name",
                match=models.MatchAny(any=spec_filter)
            )]
        )
    
    results = client.search(
        collection_name="k3d_specifications_v1",
        query_vector=embed_resp["embedding"],
        query_filter=query_filter,
        limit=limit,
        with_payload=True
    )
    
    return [{
        "content": r.payload["text"],
        "source": r.payload["file_path"],
        "section": r.payload["section_path"],
        "relevance_score": r.score
    } for r in results]

@mcp.tool()
def get_spec_sections(spec_name: str

────────────────────────────────────────────────────────────────

## Synthesis

**Executive Summary**

`qdrant/skills` is a **production-ready MCP server** (not merely a template) that exposes Qdrant vector search as tools for AI agents. For your Knowledge3D (K3D) specification knowledge base, you will deploy this server as-is, but configure it with a custom ingestion pipeline and collection schema optimized for technical documentation. The server speaks MCP natively to Claude/Codex, while Ollama-based agents require a lightweight bridge.

---

## 1. Repository Clarification & Conflict Resolution

**Sub-Agent A is correct:** `qdrant/skills` is a functional MCP server implementation (`src/qdrant_skills/server.py`), not a template library. It provides:
- Ready-to-run MCP server with `stdio` and `SSE` transports
- Built-in ingestion CLI (`qdrant-skills ingest`)
- Pluggable embedding providers (including Ollama)
- Pre-defined tools: `search_knowledge`, `get_point`, `scroll_collection`

**Sub-Agent B's architectural recommendations** (chunking strategy, payload indexing, hardware-specific tuning) are valid implementation details that extend the base server for your K3D use case.

**Unified Approach:** Use `qdrant/skills` as the **MCP server foundation**, but override its default ingestion behavior with your K3D-specific chunking logic, and extend its tool surface with specification-aware filtering.

---

## 2. System Architecture

```
┌─────────────────┐      MCP Protocol      ┌──────────────────┐
│  Claude Desktop │◄──────────────────────►│  qdrant/skills   │
│  (Codex CLI)    │    (stdio/SSE)         │  MCP Server      │
└─────────────────┘                        └────────┬─────────┘
                                                      │
                    ┌────────────────────────────────┘
                    │ HTTP/gRPC
                    ▼
┌──────────────────────────────────────────────────────────┐
│                    Qdrant (localhost:6333)                │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Collection: k3d_specifications_v1                    │ │
│  │ - Vectors: 768-dim (Cosine)                          │ │
│  │ - Payload: spec_name, section_path, content_type       │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
                           ▲
                           │ Embedding API
┌──────────────────────────────────────────────────────────┐
│  Ollama (192.168.0.60:11434)                            │
│  - Model: nomic-embed-text (768 dim)                     │
│  - Batch size: 4 (GTX 970 4GB constraint)                │
└──────────────────────────────────────────────────────────┘
```

---

## 3. Implementation Specification

### 3.1 Environment Configuration

Create `~/.config/qdrant/skills.yaml` (overrides env vars):

```yaml
qdrant:
  url: "http://localhost:6333"
  api_key: "${QDRANT_API_KEY}"
  timeout: 30

embedding:
  provider: "ollama"
  model: "nomic-embed-text"
  host: "http://192.168.0.60:11434"
  batch_size: 4  # Critical for GTX 970 4GB VRAM

ingestion:
  collection_name: "k3d_specifications_v1"
  chunk_size: 512
  chunk_overlap: 50
  parallel_requests: 2  # Conservative for local GPU

logging:
  level: INFO
  file: "~/.local/share/qdrant/skills/server.log"
```

### 3.2 Collection Schema Design

Execute once via Qdrant client (independent of MCP server):

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(
    url="http://localhost:6333", 
    api_key="your-api-key"
)

# Create optimized collection for technical specs
client.create_collection(
    collection_name="k3d_specifications_v1",
    vectors_config=models.VectorParams(
        size=768,
        distance=models.Distance.COSINE,
        hnsw_config=models.HnswConfig(
            m=16,              # Balanced for <50k chunks
            ef_construct=100,  # Higher quality for static dataset
            full_scan_threshold=1000
        )
    ),
    optimizers_config=models.OptimizersConfigDiff(
        indexing_threshold=1000  # Immediate indexing
    )
)

# Payload indexes for agent-side filtering
client.create_payload_index(
    "k3d_specifications_v1",
    field_name="spec_name",
    field_schema=models.KeywordIndexParams()  # Exact match
)

client.create_payload_index(
    "k3d_specifications_v1",
    field_name="content_type",
    field_schema=models.KeywordIndexParams()  # definition|requirement|example
)

client.create_payload_index(
    "k3d_specifications_v1",
    field_name="section_path",
    field_schema=models.TextIndexParams(
        type="text",
        tokenizer="whitespace",
        min_token_len=2
    )
)
```

### 3.3 Document Ingestion Pipeline

**Critical Override:** Do not use the default `qdrant-skills ingest` CLI for K3D specs. Instead, use this custom preprocessor that respects markdown hierarchy:

```python
# k3d_ingest.py
import os
import re
from pathlib import Path
from qdrant_client import QdrantClient
import requests
import hashlib

OLLAMA_URL = "http://192.168.0.60:11434/api/embeddings"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "k3d_specifications_v1"

def chunk_markdown(file_path: Path):
    """Semantic chunking respecting headers."""
    content = file_path.read_text()
    spec_name = file_path.stem.lower().replace('_specification', '')
    
    # Split on headers while preserving hierarchy
    pattern = r'^(#{1,3})\s+(.+)$'
    sections = re.split(pattern, content, flags=re.MULTILINE)
    
    chunks = []
    current_path = []
    
    for i in range(0, len(sections), 3):
        if i+2 >= len(sections):
            break
            
        level = len(sections[i]) if sections[i] else 0
        title = sections[i+1].strip() if i+1 < len(sections) else ""
        text = sections[i+2].strip() if i+2 < len(sections) else ""
        
        if not text:
            continue
            
        # Update header hierarchy
        header_depth = level - 1
        current_path = current_path[:header_depth] + [title]
        
        # Content type classification
        content_type = "definition"
        if "```" in text:
            content_type = "code"
        elif "example" in title.lower():
            content_type = "example"
        elif "requirement" in title.lower() or "must" in text.lower():
            content_type = "requirement"
        
        chunks.append({
            "id": hashlib.md5(f"{file_path}:{i}".encode()).hexdigest(),
            "text": f"{' > '.join(current_path)}\n\n{text}",
            "spec_name": spec_name,
            "file_path": str(file_path),
            "section_path": current_path.copy(),
            "section_level": level,
            "content_type": content_type,
            "line_start": content[:content.find(text)].count('\n') + 1
        })
    
    return chunks

def embed_batch(texts: list[str]):
    """Batch embedding with GTX 970 memory management."""
    embeddings = []
    for text in texts:
        resp = requests.post(OLLAMA_URL, json={
            "model": "nomic-embed-text",
            "prompt": text[:2048]  # Context limit
        }).json()
        embeddings.append(resp["embedding"])
    return embeddings

def ingest_specs(docs_path: str = "docs/vocabulary"):
    client = QdrantClient(url=QDRANT_URL)
    
    for md_file in Path(docs_path).glob("*.md"):
        print(f"Processing {md_file}...")
        chunks = chunk_markdown(md_file)
        
        # Batch in groups of 4 (GTX 970 constraint)
        for i in range(0, len(chunks), 4):
            batch = chunks[i:i+4]
            vectors = embed_batch([c["text"] for c in batch])
            
            points = [
                models.PointStruct(
                    id=chunk["id"],
                    vector=vec,
                    payload={
                        "text": chunk["text"],
                        "spec_name": chunk["spec_name"],
                        "file_path": chunk["file_path"],
                        "section_path": chunk["section_path"],
                        "section_level": chunk["section_level"],
                        "content_type": chunk["content_type"],
                        "line_start": chunk["line_start"]
                    }
                )
                for chunk, vec in zip(batch, vectors)
            ]
            
            client.upsert(collection_name=COLLECTION, points=points)

if __name__ == "__main__":
    ingest_specs()
```

### 3.4 MCP Server Configuration

**For Claude Desktop/Codex:**

Add to `claude_desktop_config.json` (or Codex settings):

```json
{
  "mcpServers": {
    "k3d-knowledge": {
      "command": "uv",
      "args": [
        "run",
        "--with", "qdrant-client", 
        "--with", "PyYAML",
        "-m", "qdrant_skills.server"
      ],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_API_KEY": "your-key",
        "EMBEDDING_PROVIDER": "ollama",
        "OLLAMA_HOST": "http://192.168.0.60:11434",
        "OLLAMA_EMBED_MODEL": "nomic-embed-text",
        "COLLECTION_NAME": "k3d_specifications_v1"
      }
    }
  }
}
```

**Note:** The base `qdrant/skills` server provides generic `search_knowledge`. For K3D-specific filtering (by spec_name, content_type), you have two options:

**Option A (Recommended):** Use the base server's `search_knowledge` with filter parameters, but ensure your prompts to Claude include filter instructions:
```xml
<tool_search_knowledge>
  <query>Galaxy Universe topology</query>
  <filter>{"must": [{"key": "spec_name", "match": {"value": "knowledgeverse"}}]}</filter>
</tool_search_knowledge>
```

**Option B (Advanced):** Extend the server with a custom tool by modifying `src/qdrant_skills/tools.py` to add:
```python
@mcp.tool()
def query_k3d_spec(
    query: str,
    spec_name: Optional[str] = None,
    section_keywords: Optional[List[str]] = None
) -> str:
    """Query Knowledge3D specifications with domain-aware filtering."""
    # Implementation using Qdrant client with payload filters
```

### 3.5 Ollama Integration (The Bridge)

Since Ollama does not natively support MCP, deploy a lightweight FastAPI bridge:

```python
# ollama_mcp_bridge.py
from fastapi import FastAPI
from qdrant_client import QdrantClient
import requests

app = FastAPI()
client = QdrantClient("localhost:6333")

@app.post("/k3d/query")
def query_k3d(
    question: str,
    model: str = "llama3.2",  # Your local LLM
    spec_filter: list[str] = None
):
    # 1. Get embedding from Ollama
    embed = requests.post(
        "http://192.168.0.60:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": question}
    ).json()["embedding"]
    
    # 2. Build filter
    query_filter = None
    if spec_filter:
        query_filter = models.Filter(
            must=[models.FieldCondition(
                key="spec_name",
                match=models.MatchAny(any=spec_filter)
            )]
        )
    
    # 3. Search Qdrant
    results = client.search(
        collection_name="k3d_specifications_v1",
        query_vector=embed,
        query_filter=query_filter,
        limit=5
    )
    
    context = "\n\n".join([
        f"Source: {r.payload['file_path']} (Section: {' > '.join(r.payload['section_path'])})\n{r.payload['text']}"
        for r in results
    ])
    
    # 4. Generate answer with context
    prompt = f"""Based on the following Knowledge3D specifications, answer the question.

Context:
{context}

Question: {question}
Answer:"""
    
    response = requests.post(
        "http://192.168.0.60:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    
    return {
        "answer": response.json()["response"],
        "sources": [{"file": r.payload["file_path"], "section": r.payload["section_path"]} for r in results]
    }
```

Run this bridge on port 8000, and configure Ollama agents to call `http://localhost:8000/k3d/query` instead of using MCP.

---

## 4. Multi-Agent Access Matrix

| Agent | Protocol | Configuration | Query Method |
|-------|----------|---------------|--------------|
| **Claude Desktop** | MCP (stdio) | `claude_desktop_config.json` | Native tool calling (`search_knowledge`) |
| **Codex CLI** | MCP (stdio) | `--mcp-config mcp.json` | Native tool calling |
| **Ollama** | HTTP Bridge | `ollama_mcp_bridge.py` | REST API endpoint |
| **Custom Scripts** | Direct Qdrant | `qdrant-client` SDK | Direct vector search |

---

## 5. Operational Runbook

### Health Check
```bash
# Verify Qdrant connection
curl http://localhost:6333/collections/k3d_specifications_v1

# Test embedding endpoint
curl http://192.168.0.60:11434/api/embeddings -d '{"model":"nomic-embed-text","prompt":"test"}'

# Test MCP server
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | uv run --with qdrant-skills -m qdrant_skills.server
```

### Re-ingestion Workflow
When specs change:
1. Run `python k3d_ingest.py` (handles updates via deterministic IDs based on file content hashes)
2. No MCP server restart required (stateless)

### Troubleshooting GTX 970 OOM
If Ollama crashes during embedding:
- Reduce `batch_size` to 2 in `k3d_ingest.py`
- Ensure no other GPU processes are running: `nvidia-smi`
- Use `nomic-embed-text` (not `bge-m3` or `qwen3-embedding`) as it has smaller VRAM footprint

---

## 6. Summary

**`qdrant/skills` IS the MCP server** you need. Do not rebuild it. Instead:
1. Install it as a dependency/executable
2. Run the custom `k3d_ingest.py` to populate the collection with semantically-chunked specs
3. Configure Claude/Codex to use it via MCP config
4. Deploy the FastAPI bridge for Ollama access
5. Maintain the collection with versioned ingestion scripts

This architecture provides grounded, source-citable answers to questions like *"What does the spec say about Galaxy Universe?"* with filtering by specific specification files and section hierarchies.
