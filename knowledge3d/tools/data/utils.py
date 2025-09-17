from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path


def ensure_symlink(src: Path, dst: Path) -> None:
    """Create/refresh `dst` so it points to `src` via symlink."""
    src = Path(src)
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        try:
            if dst.resolve() == src.resolve():
                return
        except Exception:
            pass
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink(missing_ok=True)
    dst.symlink_to(src, target_is_directory=src.is_dir())


def hash_copy(src: Path, dst: Path) -> None:
    """Copy a text file but add a short hash suffix to avoid name clashes."""
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if src.suffix.lower() in {".md", ".txt", ".json"}:
        text = src.read_text(encoding="utf-8", errors="ignore")
        digest = hashlib.sha1(text.encode("utf-8").strip()).hexdigest()[:8]
        dst = dst.with_name(f"{dst.stem}_{digest}{dst.suffix}")
        if dst.exists():
            return
        dst.write_text(text, encoding="utf-8")
    else:
        shutil.copy2(src, dst)


__all__ = ["ensure_symlink", "hash_copy"]
