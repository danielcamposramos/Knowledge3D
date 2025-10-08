# DEPRECATED: legacy pre-PTX script; kept for reference. Outputs belong in Knowledge3D.local/old_attempts.
import struct
from pygltflib import GLTF2


def main():
    m = GLTF2().load_binary('hello_tetrahedron.glb')
    # Read primitive extras.k3d and the vectorsView
    prim = m.meshes[0].primitives[0]
    k3d = prim.extras.get('k3d', {}) if prim.extras else {}
    vectors_view = k3d.get('vectorsView', 0)
    bv = m.bufferViews[vectors_view]
    blob = m.binary_blob()
    start = bv.byteOffset or 0
    end = start + bv.byteLength
    data = blob[start:end]
    vertices = struct.unpack('<12f', data)
    print('AI sees vertices:', vertices)
    print('embeddingDims:', k3d.get('embeddingDims'))


if __name__ == '__main__':
    main()

