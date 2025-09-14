import argparse
from pygltflib import GLTF2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", default="viewer/public/knowledge_garden.glb")
    args = ap.parse_args()
    m = GLTF2().load_binary(args.glb)
    prim = m.meshes[0].primitives[0]
    k3d = prim.extras.get('k3d', {}) if prim.extras else {}
    print({
        'realm': k3d.get('memory_realm'),
        'nodes': k3d.get('object', {}).get('nodes'),
        'vectorsView': k3d.get('vectorsView'),
        'embeddingsView': k3d.get('embeddingsView'),
        'similarityView': k3d.get('similarityView'),
    })


if __name__ == '__main__':
    main()

