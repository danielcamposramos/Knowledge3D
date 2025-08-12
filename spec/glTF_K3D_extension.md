# glTF K3D Extension (Draft)

The `K3D_nodes` extension links a glTF scene to external `.k3d` data so that
graph‑based knowledge can be aligned with 3D geometry. Each record in the
`.k3d` file must validate against
[`k3d_node_schema.json`](k3d_node_schema.json).

## Properties

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

## Rationale

Separating K3D metadata into a sidecar keeps the glTF payload small while
allowing rich graph relationships defined in
[`k3d_node_schema.json`](k3d_node_schema.json).

## Compatibility

Readers that ignore `K3D_nodes` continue to operate on standard glTF content.
The extension follows the existing `extensionsUsed`/`extensionsRequired`
mechanism and is compatible with other extensions such as
`KHR_draco_mesh_compression`.

