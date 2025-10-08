# DEPRECATED: legacy pre-PTX script; kept for reference. Outputs belong in Knowledge3D.local/old_attempts.
import struct
from pygltflib import GLTF2


def main():
    m = GLTF2().load_binary('library_room.glb')
    prim = m.meshes[0].primitives[0]
    k3d = prim.extras.get('k3d', {})
    vectors_view = k3d.get('vectorsView')
    emb_view = k3d.get('embeddingsView')
    blob = m.binary_blob()
    bv_pos = m.bufferViews[vectors_view]
    data_pos = blob[(bv_pos.byteOffset or 0): (bv_pos.byteOffset or 0) + bv_pos.byteLength]
    # Print first vertex of first book
    v0 = struct.unpack('<3f', data_pos[:12])
    print('Book[0] first vertex:', v0)
    print('embeddingDims:', k3d.get('embeddingDims'))
    bv_emb = m.bufferViews[emb_view]
    print('embedding bytes:', bv_emb.byteLength)


if __name__ == '__main__':
    main()

