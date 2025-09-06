# Temporal LOD (GLM‑4.5 Alpha Channel)

Summary
- Adds a temporal “alpha channel” to K3D’s embedded payload so time becomes a navigable dimension without breaking spatial consistency.
- Human clients map alpha to visual transparency; AI clients treat alpha as a recency/validity weight.

Why
- Spatial Consistency: Older knowledge stays in place; only its salience fades.
- AI‑Client Friendly: Alpha is a first‑class scalar for temporal reasoning.
- Human‑Intuitive: Transparency is a direct metaphor for “no longer current”.
- Efficient: A small numeric field added to the existing payload.

Spec
- See `spec/glTF_K3D_extension.md` under “Embedded Variant → Temporal fields”.
- Fields: `temporal.alpha` (global 0..1), `temporal.alphaMask` (per‑node 0..1 list), optional `temporal.versions`.

Viewer Behavior (MVP)
- Global alpha maps to `PointsMaterial.opacity` (Three.js), enabling fade of the entire set.
- Per‑node alphaMask dims vertex colors (`rgb *= alpha[i]`) as a proxy for per‑vertex opacity.
- Future: custom shader for true per‑vertex alpha.

Generator
- `k3dgen` CLI embeds temporal fields:
  - `--temporal-alpha <float>` — sets `temporal.alpha`.
  - `--temporal-alpha-mask idx:value,idx:value,...` — sets `temporal.alphaMask` overrides.
  - Missing indices default to 1.0; entries without `:value` use global alpha or 0.5.

Temporal LOD Strategy
- Spatial LOD: lower detail for distant objects (existing viewer LOD/Grid culling).
- Temporal LOD: lower “validity” for outdated information (alpha fade).
- Combined: prioritize rendering and reasoning by distance × recency.

Credits
- Concept co‑developed with GLM‑4.5 (Fullstack). Integrated here as the GLM‑4.5 Temporal Alpha channel.

