from __future__ import annotations

"""Utilities to store and load model weights inside a House GLB.

Appliance layout in GLTF extras (top-level):

  extras: {
    "k3d": {
      "appliances": {
        "<appliance_name>": {
          "tensors": [
            {"name": "module.param", "bufferView": 7, "shape": [..], "dtype": "float32"},
            ...
          ]
        }
      }
    }
  }

Binary data for each tensor is stored in the GLB single BIN chunk and referenced
by bufferViews.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np  # type: ignore

try:
    import torch
except Exception:
    torch = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required for GLB appliance weight loading")


def _house_glb_path() -> Path:
    import os
    root = Path(__file__).resolve().parents[2]
    hid = (os.getenv("K3D_HOUSE_ID", "").strip() or "default")
    return root / "viewer" / "public" / "houses" / hid / "memory_house.glb"


def _dtype_from_str(s: str) -> np.dtype:
    s = (s or "float32").lower()
    if s in ("float16", "fp16", "half"):
        return np.float16
    if s in ("bfloat16", "bf16"):
        # Store/load via float32 for simplicity; bf16 not widely supported in numpy
        return np.float32
    if s in ("float64", "fp64", "double"):
        return np.float64
    if s in ("int64", "long"):
        return np.int64
    if s in ("int32", "int"):
        return np.int32
    if s in ("int16",):
        return np.int16
    if s in ("uint8", "byte"):
        return np.uint8
    return np.float32


def load_appliance_weights_from_glb(
    appliance: str,
    *,
    glb_path: Optional[Path] = None,
    device: Optional["torch.device"] = None,
) -> Optional[Dict[str, "torch.Tensor"]]:
    """Load weights for an appliance from the active House GLB.

    Returns a dict name -> tensor (on requested device). If appliance not found,
    returns None.
    """
    _require_torch()
    try:
        from pygltflib import GLTF2  # type: ignore
    except Exception:
        return None

    path = glb_path or _house_glb_path()
    if not path.exists():
        return None
    try:
        gltf = GLTF2().load(str(path))
    except Exception:
        return None

    meta = getattr(gltf, "extras", None) or {}
    if not isinstance(meta, dict):
        return None
    k3d = meta.get("k3d") if isinstance(meta.get("k3d"), dict) else None
    if not k3d:
        return None
    apps = k3d.get("appliances") if isinstance(k3d.get("appliances"), dict) else None
    if not apps or appliance not in apps:
        return None
    entry = apps[appliance]
    tensors = entry.get("tensors") if isinstance(entry, dict) else None
    if not isinstance(tensors, list) or not tensors:
        return None
    try:
        blob = gltf.binary_blob()
    except Exception:
        blob = None
    if blob is None:
        return None

    out: Dict[str, "torch.Tensor"] = {}
    for t in tensors:
        if not isinstance(t, dict):
            continue
        name = t.get("name")
        bvi = t.get("bufferView")
        shape = t.get("shape")
        dtype_s = t.get("dtype", "float32")
        if name is None or bvi is None or shape is None:
            continue
        try:
            bvi = int(bvi)
        except Exception:
            continue
        if bvi < 0 or bvi >= len(gltf.bufferViews):
            continue
        bv = gltf.bufferViews[bvi]
        off = int(getattr(bv, "byteOffset", 0) or 0)
        ln = int(getattr(bv, "byteLength", 0) or 0)
        if ln <= 0 or off < 0 or off + ln > len(blob):
            continue
        raw = memoryview(blob)[off : off + ln]
        arr = np.frombuffer(raw, dtype=_dtype_from_str(str(dtype_s)))
        try:
            arr = arr.reshape(tuple(int(x) for x in shape))
        except Exception:
            # If reshape fails, keep flat
            pass
        ten = torch.tensor(arr)
        if device is not None:
            ten = ten.to(device)
        out[str(name)] = ten

    return out if out else None


def apply_partial_state(module: torch.nn.Module, weight_map: Dict[str, torch.Tensor]) -> None:
    """Load any matching keys from weight_map into module.state_dict.

    Silently skips missing/mismatched shapes to allow partial loads.
    """
    _require_torch()
    try:
        sd = module.state_dict()
        updates: Dict[str, "torch.Tensor"] = {}
        for k, v in weight_map.items():
            if k in sd and tuple(sd[k].shape) == tuple(v.shape):
                try:
                    updates[k] = v.type_as(sd[k])
                except Exception:
                    updates[k] = v
        if updates:
            sd.update(updates)
            module.load_state_dict(sd)
    except Exception:
        pass
