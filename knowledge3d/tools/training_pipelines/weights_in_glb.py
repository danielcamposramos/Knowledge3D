from __future__ import annotations

"""Pack .pt weights into a House GLB as bufferViews with a k3d appliance map.

Usage:
  PYTHONPATH=. python -m knowledge3d.tools.weights_in_glb \
    --glb viewer/public/houses/default/memory_house.glb \
    --pt viewer/public/house/house_rpn_policy.pt \
    --appliance fused_rpn_policy

Notes:
  - Stores tensors in the GLB BIN chunk, appending to existing data.
  - Writes a mapping in glTF extras.k3d.appliances[appliance].tensors[].
  - Overwrites existing entries of the same appliance.
  - Creates a .bak backup next to the GLB before writing.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np  # type: ignore
import torch


def _flatten_state_dict(obj) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    if isinstance(obj, dict):
        # If this looks like {'embed': sd, 'gru': sd, 'out': sd}
        has_sub = any(isinstance(v, dict) for v in obj.values())
        if has_sub and not any(torch.is_tensor(v) for v in obj.values()):
            for prefix, sub in obj.items():
                if not isinstance(sub, dict):
                    continue
                for k, v in sub.items():
                    if torch.is_tensor(v):
                        out[f"{prefix}.{k}"] = v
        else:
            for k, v in obj.items():
                if torch.is_tensor(v):
                    out[str(k)] = v
    return out


def pack_pt_into_glb(glb_path: Path, pt_path: Path, appliance: str) -> None:
    from pygltflib import GLTF2, BufferView  # type: ignore

    if not glb_path.exists():
        raise FileNotFoundError(glb_path)
    state = torch.load(str(pt_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    flat = _flatten_state_dict(state)
    if not flat:
        raise RuntimeError("no tensors found in checkpoint")

    g = GLTF2().load(str(glb_path))
    try:
        blob = g.binary_blob()
    except Exception:
        blob = None
    if blob is None:
        blob = b""
    offset = len(blob)
    views: List[Tuple[str, int, Tuple[int, ...], str]] = []  # (name,bvi,shape,dtype)
    for name, ten in flat.items():
        arr = ten.detach().cpu().numpy()
        dtype = str(arr.dtype)
        data = arr.tobytes()
        bvi = len(g.bufferViews or [])
        if g.bufferViews is None:
            g.bufferViews = []
        g.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(data)))
        views.append((name, bvi, tuple(arr.shape), dtype))
        offset += len(data)
        blob += data

    # Update buffer length and extras map
    if g.buffers and len(g.buffers) >= 1:
        g.buffers[0].byteLength = len(blob)
    else:
        raise RuntimeError("GLB missing primary buffer")

    meta = getattr(g, "extras", None)
    if not isinstance(meta, dict):
        meta = {}
        g.extras = meta
    k3d = meta.get("k3d")
    if not isinstance(k3d, dict):
        k3d = {}
        meta["k3d"] = k3d
    apps = k3d.get("appliances")
    if not isinstance(apps, dict):
        apps = {}
        k3d["appliances"] = apps
    apps[appliance] = {
        "tensors": [
            {"name": name, "bufferView": bvi, "shape": list(shape), "dtype": dtype}
            for (name, bvi, shape, dtype) in views
        ]
    }

    # Backup, write
    bak = glb_path.with_suffix(glb_path.suffix + ".bak")
    try:
        shutil.copy2(glb_path, bak)
    except Exception:
        pass
    g.set_binary_blob(blob)
    g.save_binary(str(glb_path))


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Pack PyTorch weights (.pt) into House GLB")
    ap.add_argument("--glb", required=True, type=str, help="Path to memory_house.glb")
    ap.add_argument("--pt", required=True, type=str, help="Checkpoint .pt path")
    ap.add_argument("--appliance", required=True, type=str, help="Appliance name (e.g., fused_rpn_policy)")
    args = ap.parse_args()
    pack_pt_into_glb(Path(args.glb), Path(args.pt), args.appliance)
    print(f"Packed {args.pt} into {args.glb} as appliance '{args.appliance}'")


if __name__ == "__main__":
    main()

