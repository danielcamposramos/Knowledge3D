import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from pygltflib import GLTF2

from .registry import load_registry, save_registry, append_error, GARDEN_DIR
from .error_shell import make_error_shell_glb


class TreeLoader:
    def __init__(self, registry_path: Optional[str] = None) -> None:
        self.registry_path = Path(registry_path) if registry_path else (GARDEN_DIR / 'garden_registry.json')
        self._reg = None

    def load_registry(self) -> Dict[str, Any]:
        self._reg = load_registry()
        return self._reg

    def save_registry(self) -> None:
        if self._reg is not None:
            save_registry(self._reg)

    def map_error_to_code(self, exc: Exception) -> float:
        import struct
        if isinstance(exc, FileNotFoundError):
            return 404.0
        # pygltflib raises generic exceptions on parse; map to 500
        try:
            import json
        except Exception:
            pass
        return 500.0

    def extract_mesh(self, model: GLTF2) -> Dict[str, Any]:
        # Return minimal mesh info: vectorsView, embeddingsView, indices accessor, mode
        prim = model.meshes[0].primitives[0]
        k3d = prim.extras.get('k3d', {}) if prim.extras else {}
        return {
            'vectorsView': int(k3d.get('vectorsView', 0)),
            'embeddingsView': int(k3d.get('embeddingsView', 0)),
            'indices': int(prim.indices) if prim.indices is not None else None,
            'mode': int(prim.mode or 4),
        }

    def load_tree(self, tree_id: str) -> Tuple[Optional[GLTF2], str]:
        reg = self._reg or self.load_registry()
        ent = next((t for t in reg['trees'] if t.get('tree_id') == tree_id), None)
        if not ent:
            return None, 'tree_not_in_registry'
        p = Path(ent['filepath'])
        try:
            model = GLTF2().load_binary(str(p))
            ent['load_status'] = 'success'
            ent['error_code'] = None
            ent['last_loaded'] = __import__('datetime').datetime.utcnow().isoformat() + 'Z'
            self.save_registry()
            return model, 'success'
        except Exception as e:
            code = self.map_error_to_code(e)
            ent['load_status'] = 'error'
            ent['error_code'] = code
            ent['last_loaded'] = __import__('datetime').datetime.utcnow().isoformat() + 'Z'
            self.save_registry()
            append_error(f"{tree_id}: {code} ({type(e).__name__}) {str(e)[:140]}")
            err_dir = GARDEN_DIR / 'trees_err'
            err_path = err_dir / f"{tree_id}_error.glb"
            make_error_shell_glb(err_path, error_code=code)
            return GLTF2().load_binary(str(err_path)), 'error'

