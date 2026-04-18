#!/usr/bin/env bash
set -euo pipefail

docker rm -f k3d-ptx-mcp >/dev/null 2>&1 || true

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
  k3d-qdrant-mcp:latest \
  python3 -c "from mcp_server_qdrant.server import mcp; mcp.run(transport='streamable-http', host='0.0.0.0', port=8000)"
