# Legacy Examples

The examples in this folder that reference `.k3d` sidecar files are legacy and deprecated as of Cranium Core v3.0. New work should use embedded glTF/GLB with `meshes[*].primitives[*].extras.k3d` and binary bufferViews for vectors/embeddings.

Affected example artifacts:
- examples/my_house.gltf (+ my_house.k3d)
- examples/sample_output*.gltf (+ .k3d)
- examples/solar_system.gltf (+ .k3d)

See `docs/DEPRECATIONS.md` and `spec/glTF_K3D_extension.md` for migration guidance.

