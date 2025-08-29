# K3D Agent Protocol (Extras)

Location: `meshes[*].primitives[*].extras.k3d`

Core fields
- `ids: string[]` — Node identifiers (aligned with vectors/embeddings order)
- `vectorsView: number` — BufferView index for Float32 POSITIONs (3D)
- `embeddingsView: number` — BufferView index for embeddings (Float32 or Float16)
- `embeddingDims: number` — Embedding width (e.g., 384)
- `embeddingPrecision: "f32" | "f16"`
- `metadata: Array<object>` — Per-node metadata; `metadata[i].label` is friendly text; `metadata[i].type` may be `"door"`
- `neighbors: string[][]` — K nearest neighbors per node, as node ids

AI fields
- `ai_interaction_protocol: string`
  - `"direct_vector_manipulation"` — Agent can write/update vectors/embeddings
  - `"semantic_query"` — Agent queries only; no writes
  - `"spatial_reasoning"` — Agent traverses/derives paths; read-only by default
- `ai_state_flags: { is_active?: boolean; is_traversable?: boolean; has_new_information?: boolean }`
  - Global hints for the primitive
- `ai_state_flags_mask: { has_new_information?: boolean[] }`
  - Per-node boolean mask for guidance; `true` marks points with newly available information

Behavioral guidance (suggested)
- Path planning: use neighbor lists to route between labels; report cosine similarities per hop
- Exploration: prefer `has_new_information == true` when present; treat `metadata.type == "door"` as an anchor point
- Safety: refrain from writes unless protocol allows; honor `is_traversable == false`

See also
- `spec/glTF_K3D_extension.md`
- `spec/k3d_node_schema.json`
- `docs/CARE_PROTOCOL.md`

