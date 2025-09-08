# Scene Generation Inspiration (Reference Only)

Intent
- Clarify “world model” references: here, it means text→3D scene generation systems that produce navigable environments (not just navigation policies). These are inspirational inputs — K3D remains memory‑first: persistent knowledge in GLB + `extras.k3d` and a small logic layer.

Key References
- HunyuanWorld (Tencent): room/scene generation with strong priors. See `docs/HUNYUANWORLD_INTEGRATION.md` for how we plan to adapt outputs.
- Stable DreamFusion (ashawkey/stable-dreamfusion): https://github.com/ashawkey/stable-dreamfusion
  - Family of approaches turning text into NeRFs/meshes via score distillation.
  - We will not vendor weights. When used locally, export meshes as GLB and inject `extras.k3d` (ids, vectors/embeddings, metadata) via an adapter.

K3D Alignment
- Memory‑first: Generated rooms/objects must be converted to standard K3D GLB with `primitive.extras.k3d` so both human and AI clients see the same object as dual representations (geometry + semantics).
- Small logic: The agent’s cranium uses tiny models (intent/RSSM/RPN). Heavy scene generators are optional producers that update the House, not monolithic brains.

Adapter Sketch (future)
- CLI `knowledge3d.tools.scene_adapter` (planned subcommands):
  - `from-hunyuan` — run inference and convert scene → GLB (+ `extras.k3d`)
  - `from-dreamfusion` — convert mesh/NeRF exports → GLB (+ `extras.k3d`)
  - `inject-k3d` — take an existing GLTF and add minimal `extras.k3d` using `POSITION` as `vectorsView` and default metadata

Scope & Priority
- Near‑term: keep this as research inspiration only; do not block MVP on generator availability.
- Mid‑term: add a minimal adapter path that converts at least one open generator’s output into our GLB format.

