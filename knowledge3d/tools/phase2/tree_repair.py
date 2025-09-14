import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from .registry import TREES_DIR, load_registry, save_registry, append_error
from .grow_tree import _generate_tree_py
from ..cranium.embedding_generator import DynamicEmbeddingGenerator
from .export_tree import exportTreeToGLB


class TreeRepair:
    def __init__(self) -> None:
        self.gen = DynamicEmbeddingGenerator()

    def auto_repair(self, tree_id: str) -> bool:
        reg = load_registry()
        ent = next((t for t in reg['trees'] if t.get('tree_id') == tree_id), None)
        if not ent:
            return False
        domain = ent.get('domain') or ent.get('sector') or 'Unknown'
        emb = self.gen.generate(domain, 'text')
        if emb.shape[0] > 72:
            emb = emb.copy(); emb[72] = 1.0
        tree = _generate_tree_py(emb)
        out = TREES_DIR / f"{tree_id}.glb"
        ok = exportTreeToGLB(tree, str(out), domain=domain, tree_id=tree_id, source_ref=domain, checksum=None)
        if ok:
            ent['load_status'] = 'success'
            ent['error_code'] = None
            save_registry(reg)
            return True
        append_error(f"repair_failed {tree_id}")
        return False

    def manual_repair(self, tree_id: str, new_filepath: str) -> bool:
        try:
            dst = TREES_DIR / f"{tree_id}.glb"
            Path(new_filepath).resolve(strict=True)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(new_filepath, dst)
            reg = load_registry()
            ent = next((t for t in reg['trees'] if t.get('tree_id') == tree_id), None)
            if ent:
                ent['filepath'] = str(dst)
                ent['load_status'] = 'success'
                ent['error_code'] = None
                save_registry(reg)
            return True
        except Exception as e:
            append_error(f"manual_repair_failed {tree_id}: {type(e).__name__} {e}")
            return False

