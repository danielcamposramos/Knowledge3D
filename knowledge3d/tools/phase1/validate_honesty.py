import argparse
import json
from pygltflib import GLTF2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", default="library_room.glb")
    args = ap.parse_args()
    m = GLTF2().load_binary(args.glb)
    n = 0
    bad = 0
    for mesh in m.meshes or []:
        for prim in mesh.primitives or []:
            k3d = (prim.extras or {}).get("k3d") if prim.extras else None
            if not isinstance(k3d, dict):
                continue
            rays = k3d.get("rays") or []
            for r in rays:
                n += 1
                if "honesty" not in r:
                    bad += 1
    print(json.dumps({"rays": n, "missing_honesty": bad}))


if __name__ == "__main__":
    main()

