# glTF K3D Data (Draft)

K3D supports two representations for aligning knowledge with 3D geometry:

1) Legacy sidecar via `K3D_nodes` extension: a glTF scene links to an external `.k3d` JSON.
2) Embedded variant: all K3D node data (IDs, vectors, embeddings, metadata) is stored directly in `primitive.extras.k3d`.

## Sidecar Properties (Legacy)

| Property               | Type   | Description |
|------------------------|--------|-------------|
| `uri`                  | string | Relative or absolute URI of the `.k3d` file containing node entries that follow `k3d_node_schema.json`. |
| `schema`               | string | Optional URI to a JSON schema describing the `.k3d` node format. Defaults to `k3d_node_schema.json` when omitted. |
| `nodeProperty`         | string | JSON pointer indicating where each glTF **node** stores a corresponding K3D `id`. For scenes with distinct objects. Defaults to `extras.k3dId`. |
| `primitiveIdsProperty` | string | JSON pointer indicating where a mesh **primitive** stores a list of K3D `id`s. For point clouds or batched geometries. |

## Example (Point Cloud)

```json
{
  "extensionsUsed": ["K3D_nodes"],
  "extensions": {
    "K3D_nodes": {
      "uri": "points.k3d",
      "schema": "../spec/k3d_node_schema.json",
      "primitiveIdsProperty": "extras.k3dIds"
    }
  },
  "meshes": [{
    "primitives": [{
      "attributes": { "POSITION": 1 },
      "extras": {
        "k3dIds": ["node-1", "node-2", "node-3"]
      }
    }]
  }]
}
```

## Embedded Variant (Recommended)

Data is embedded within the mesh primitive to keep geometry and semantics in a single asset.

Location: `meshes[n].primitives[m].extras.k3d`

Schema (informal):

```json
{
  "ids": ["node-1", "node-2", "node-3"],
  "vectorsView": 0,
  "embeddingsView": 1,
  "embeddingDims": 768,
  "metadata": [{"label": "..."}, {"label": "..."}, {"label": "..."}],
  "neighbors": [["...", "..."], ["..."], []],
  "temporal": {
    "alpha": 1.0,
    "alphaMask": [1.0, 0.6, 0.3],
    "versions": [
      [
        {"version": 1, "valid_until": "2023-12-31T23:59:59Z"},
        {"version": 2, "valid_until": null}
      ],
      [
        {"version": 1, "valid_until": null}
      ],
      []
    ]
  }
}
```

Notes:
- `vectorsView` is a `bufferView` index of packed Float32 triples (x,y,z for each node).
- `embeddingsView` is a `bufferView` index of packed Float32 embeddings concatenated row-wise; `embeddingDims` gives the per-node dimension.
- `embeddingPrecision`: optional, one of `"f32"|"f16"`. When `f16`, the embeddings bufferView stores IEEE754 half-precision floats (2 bytes); readers should decode to float32 for processing.
- `primitive.extras.k3dIds` mirrors the `ids` array for simple readers.

Temporal fields
- `temporal.alpha` is a global 0..1 visibility weight for the entire primitive. Viewers may map this to material opacity.
- `temporal.alphaMask` is an optional per-node array of 0..1 weights aligned with `ids`. For large sets, future versions may support a `bufferView` instead (not yet standardized here).
- `temporal.versions` is an optional array of per-node version lists. Each inner list contains objects with `version` and `valid_until` (ISO 8601 or null). Embedding deltas are intentionally omitted at this layer; agents should query external provenance if needed.

AI-native optional fields

- `ai_interaction_protocol` (string): One of `direct_vector_manipulation`, `semantic_query`, or `spatial_reasoning`. Specifies how AI agents are expected to interact with this payload.
- `ai_state_flags` (object): Dynamic flags for AI runtime. Suggested keys: `is_active` (bool), `is_traversable` (bool), `has_new_information` (bool).
- `ai_state_flags_mask` (object): Optional per-node boolean masks with same length as `ids`. Currently defined key: `has_new_information: boolean[]`. When present, it overrides global `ai_state_flags.has_new_information` for coloring/alerts.
- For large per-node embeddings, an additional compact form may be present per node entry or for the whole set: `embedding_b64` with `{ data, dtype, dims, endianness }` for efficient transfer.
 - `metadata.type` may be `door`, `diary_book`, or `diary_page` for AI-native affordances.
 - `metadata.address` (string) for `door` nodes: `k3d://rx,ry,rz:port@x,y,z?label=...` or an asset path (e.g., `/houses/<id>/memory_house.gltf`).
 - `metadata.parent` (string) for `diary_page`: the id of its parent `diary_book`.
 - `metadata.embedding32` (number[32]) for `diary_page`: native content; when present, readers may reflect this into the packed `embeddings` buffer row for the page.

## Rationale

Separating K3D metadata into a sidecar keeps the glTF payload small while
allowing rich graph relationships defined in
[`k3d_node_schema.json`](k3d_node_schema.json).

Temporal metadata follows the K3D dual-representation principle. The alpha channel communicates recency to both human clients (as transparency) and AI clients (as an explicit weight), enabling temporal Level of Detail (LOD): older information fades while remaining spatially coherent.

## Compatibility

Readers that ignore `K3D_nodes` or the `extras.k3d` payload continue to operate on standard glTF content. The embedded payload uses `extras` and remains valid glTF 2.0.
