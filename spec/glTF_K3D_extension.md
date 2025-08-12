# glTF K3D Extension (Draft)

The `K3D_nodes` extension links a glTF scene to a K3D sidecar file (`.k3d`). This allows graph-based knowledge to be aligned with 3D geometry, where each node in the glTF scene corresponds to a knowledge node in the `.k3d` file.

The structure of the `.k3d` file is defined by [`k3d_file.schema.json`](k3d_file.schema.json), and each entry within its `nodes` array must validate against [`k3d_node_schema.json`](k3d_node_schema.json).

## Properties

| Property      | Type   | Description |
|---------------|--------|-------------|
| `uri`         | string | Relative or absolute URI of the `.k3d` metadata file. |
| `nodeProperty`| string | JSON Pointer indicating where in the glTF node definition the corresponding K3D node `id` is stored. Defaults to `extras.k3dId`. |

## Example

This example shows a glTF file referencing a `.k3d` sidecar file. The scene contains two nodes, each representing a different knowledge point. Both nodes instance the same mesh (e.g., a simple point or sphere) but have different translations and link to different K3D nodes via their `extras.k3dId`.

```json
{
  "asset": { "version": "2.0" },
  "extensionsUsed": ["K3D_nodes"],
  "extensions": {
    "K3D_nodes": {
      "uri": "my_knowledge_graph.k3d",
      "nodeProperty": "extras.k3dId"
    }
  },
  "scenes": [{ "nodes": [0, 1] }],
  "nodes": [
    {
      "mesh": 0,
      "translation": [1.2, 0.5, -2.1],
      "extras": { "k3dId": "concept-alpha" }
    },
    {
      "mesh": 0,
      "translation": [-0.8, 1.5, -1.8],
      "extras": { "k3dId": "concept-beta" }
    }
  ],
  "meshes": [
    {
      "primitives": [{ "attributes": { "POSITION": 0 } }]
    }
  ]
}
```

## Rationale

Separating K3D metadata into a sidecar file keeps the primary glTF payload small and focused on geometry. This architecture allows rich graph relationships and high-dimensional data to be managed in the `.k3d` file while leveraging standard glTF for efficient 3D rendering.

Using one glTF node for each K3D node ensures that every knowledge point can be individually selected, transformed, and interacted with inside a standard glTF viewer.

## Compatibility

Readers that do not support the `K3D_nodes` extension can still render the glTF scene as usual, though they will not be able to access the associated knowledge graph data. The extension follows the standard `extensionsUsed`/`extensionsRequired` mechanism and is compatible with other extensions.
