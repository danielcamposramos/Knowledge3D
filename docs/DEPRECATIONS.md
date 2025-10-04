# Deprecations — Cranium Core v3.0

This project is transitioning to the Cranium Core v3.0 pipeline where all knowledge and runtime bindings live inside embedded glTF/GLB with `meshes[*].primitives[*].extras.k3d` and direct buffer views.

As part of this transition, the following modules, patterns, and examples are **DEPRECATED**. They will remain available during the migration window, but new work should target the embedded glTF path only.

## Deprecated Data Formats

### Sidecar `.k3d` Files
- **Deprecated modules**: `k3dgen/house.py` (sidecar storage/access helpers)
- **Deprecated examples**: `examples/my_house_generator.py`, all `examples/*.k3d`, `viewer/public/*.k3d`
- **Reason**: Separates geometry from semantics; breaks the dual-client paradigm where AI clients need direct buffer access
- **Replacement**: Use embedded glTF with `extras.k3d` and bufferViews; see `spec/glTF_K3D_extension.md` and `knowledge3d/tools/phase0_export_glb.py`
- **Migration**: Run `python -m knowledge3d.tools.convert_sidecar_to_embedded --input <file>.gltf` (see `docs/MIGRATION_V3.md`)

### Base64 Embedding Payloads
- **Deprecated modules**: `k3dgen/ai_native.py` (AI-native extras with `embedding_b64`)
- **Reason**: Inefficient encoding; 33% overhead vs binary buffers; breaks GPU-native PTX access
- **Replacement**: Attach embeddings in binary `BufferView` referenced by `extras.k3d["embeddingsView"]` with `embeddingDims`

## Deprecated Cognitive Patterns

### External LLM Wrappers
- **Deprecated modules**: `knowledge3d/skills/llm.py` (transformers/llama_cpp backends as primary path)
- **Reason**: Violates the unified Cranium Core mandate; introduces latency; prevents PTX integration
- **Valid use**: Ollama integration for teacher models (exaone3.5, qwen3) during training only
- **Replacement**: Use the single unified fused head in `knowledge3d/cranium/fused_head.py` for all inference
- **Environment flag**: `K3D_CORE_HEAD=1` (default in Phase B+)

### CPU Fallback Paths
- **Deprecated patterns**: Any code path that falls back to CPU when CUDA is available
- **Reason**: K3D is GPU-native by design; CPU fallbacks hide performance regressions and break PTX kernel assumptions
- **Policy**: `AdaptedFusedHead` raises `RuntimeError` if CUDA unavailable; this is intentional
- **Exception**: Development testing on CPU-only machines is allowed, but core runs require GPU

### Wrapper-Style TTS
- **Deprecated patterns**: External subprocess calls to TTS engines (espeak, festival, etc.)
- **Reason**: Breaks embodied cognition (avatar voice should emerge from Cranium Core)
- **Replacement**: First-class TTS head in Phase D (waveform generation from latent acoustic tokens)
- **Current status**: Planned (see `docs/CRANIUM_CORE.md`)

## Migration Timeline

- **Phase A (Complete)**: Embedded glTF format stabilized; PTX kernels operational
- **Phase B (Active)**: Tablet UX consumes embedded format; legacy examples remain functional with warnings
- **Phase C (Q1 2026)**: Remove sidecar `.k3d` support from loaders; migration tools provided
- **Phase D (Q2 2026)**: External LLM wrappers removed from core paths; Ollama teacher integration remains

## Notes

- The sidecar `.k3d` format remains documented for historical reference in `spec/k3d_node_schema_legacy.json`
- New exporters must ensure direct buffer access for the AI client with `extras.k3d.direct_buffer_access = true` and provide `embeddingDims`
- See `docs/MIGRATION_V3.md` for step-by-step conversion guides
- Report migration blockers at https://github.com/danielcamposramos/Knowledge3D/issues

