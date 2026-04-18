PTX corpus ingested into Qdrant collection `k3d_ptx` (11298 points) via `scripts/ingest_ptx_corpus.py`. Planner swap `qwen3.5:latest -> qwen3.5:397b-cloud` live and smoke-tested. `k3d-ptx-mcp` container up on :8503, registered in `~/.claude.json`, `qdrant-find` returning PTX hits end-to-end.

What changed:
- Added [scripts/ingest_ptx_corpus.py](/K3D/GitHub/Knowledge3D/scripts/ingest_ptx_corpus.py)
- Added [k3d-ptx-mcp.run.sh](/K3D/GitHub/Knowledge3D/deploy/docker/k3d-ptx-mcp.run.sh)
- Swapped planner in `/home/daniel/.claude/ollama_specialists.py`
- Added `k3d-ptx` and aligned all three MCP servers in:
  - `/home/daniel/.codex/config.toml`
  - `/home/daniel/.claude.json`
  - `/home/daniel/.config/Code/User/mcp.json`
  - `/home/daniel/.config/Code/User/globalStorage/rooveterinaryinc.roo-cline/settings/mcp_settings.json`

Proof:
- `k3d_ptx` points_count after ingest: `11298`
- `k3d_ptx` points_count after consecutive rerun: `11298`
- Cloud planner smoke via FastMCP client:
  - tool: `plan_task`
  - elapsed: `6.29s`
- PTX MCP smoke via FastMCP client:
  - tool: `qdrant-find`
  - query: `shared memory atomicAdd PTX ISA`
  - returned PTX ISA hits from the ingested corpus

Operational note:
- The generic `k3d-qdrant-mcp:latest` image defaulted to legacy SSE when launched ad hoc.
- The deploy helper now overrides the command to match the working `k3d-knowledge-mcp` transport:
  - `mcp.run(transport='streamable-http', host='0.0.0.0', port=8000)`
- The `k3d_ptx` collection had to use the named vector `fast-all-minilm-l6-v2` and include a `document` payload field because `mcp-server-qdrant` expects both.
