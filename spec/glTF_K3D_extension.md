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
  "vectors": [[x, y, z], [x, y, z], [x, y, z]],
  "embeddings": [[...], [...], [...]],
  "metadata": [{"label": "..."}, {"label": "..."}, {"label": "..."}],
  "neighbors": [["...", "..."], ["..."], []]
}
```

Back‑compat helper: `primitive.extras.k3dIds` mirrors the `ids` array for readers that only expect a flat id list.

## Rationale

Separating K3D metadata into a sidecar keeps the glTF payload small while
allowing rich graph relationships defined in
[`k3d_node_schema.json`](k3d_node_schema.json).

## Compatibility

Readers that ignore `K3D_nodes` or the `extras.k3d` payload continue to operate on standard glTF content. The embedded payload uses `extras` and remains valid glTF 2.0.
