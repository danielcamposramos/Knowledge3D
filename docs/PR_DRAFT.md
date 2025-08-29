# PR: AI-Native Extras, Spatial Doors, OSI Routing, and HR/MR + RPN Standardization

## Summary
- Add AI-native fields to generator and viewer with per-node masks.
- Implement SpatialAddress + OSI routing primitives; wire live `/open` door command.
- Enrich viewer demos (ai_demo.glb, math_house.gltf) with doors and AI-state cues.
- Document HR/MR paradigm and RPN logic standard; update specs and README.

## Changes
- spec/k3d_node_schema.json: add `embedding_version`, `ai_interaction_protocol`, `ai_state_flags`.
- spec/glTF_K3D_extension.md: document AI fields + `ai_state_flags_mask`.
- spec/AI_RPN_standard.md: RPN logic standard.
- docs/HR_MR_STANDARD.md: dual-code principles and artifacts.
- docs/DUAL_CODE.md: existing; used as MR runbook reference.
- k3dgen/__main__.py: `--ai-*` flags, per-node mask via `--ai-new-info-indices`.
- k3dgen/ai_native.py: helper to embed AI-native node data.
- knowledge3d/spatial/address.py: `SpatialAddress` encode/decode/partition.
- knowledge3d/spatial/osi.py: Physical/DataLink/Network/Transport scaffolds; BFS routing.
- knowledge3d/bridge/live_server.py: event ingestion (graph, doors, explain), `/open` door command with restriction to registered doors.
- viewer/src/loadK3D.ts: read AI flags + mask; expose info.ai.
- viewer/src/address.ts: spatial address helper; exposed globally.
- viewer/src/agent.ts: include spatial address in explain traces.
- viewer/src/main.ts: per-node mask coloring; door coloring; emits dataset_graph + doors to server; handles `open` command.
- viewer/public/ai_demo.glb: demo with per-node `has_new_information` (generated).
- viewer/public/math_house.gltf: add door metadata + AI mask + protocol.
- README.md: document generator flags, `/open`, HR/MR and RPN references.

## Tests
- Python: `pytest -q` → 9 passed (1 warning unrelated).
- Viewer: `node ./node_modules/jest/bin/jest.js --runInBand` → 3 suites passed.

## Usage
- Generate demo: `python3 -m k3dgen examples/sample_vectors.csv --gltf examples/ai_demo.glb --k 2 --reducer pca --ai-protocol direct_vector_manipulation --ai-active --ai-new-info-indices 1,3 && cp examples/ai_demo.glb viewer/public/ai_demo.glb`.
- Run viewer: `cd viewer && npm run dev` (select `ai-demo`).
- Live bridge: `python -m knowledge3d.bridge.live_server`.
- Doors: In viewer chat, `/open two` or `/open k3d://0,0,0:0@0,0,0?label=two`.

## Notes
- The door registry restricts `/open` to known `metadata.type === 'door'` labels when provided by the client.
- RPN remains the minimal, deterministic logic layer for local similarity and signals.
