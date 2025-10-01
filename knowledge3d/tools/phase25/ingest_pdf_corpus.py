from __future__ import annotations

"""Ingest PDF/JSON corpora from external roots and update House manifest.

Scans roots up to a given depth for PDF/JSON files. For PDFs, attempts to
generate page previews (PNG) via PyMuPDF (fitz) if available, skipping likely
index/copyright pages. Writes entries to House manifest with 'prompt',
'source_path', and optional 'preview' paths for training/consistency.

Usage:
  PYTHONPATH=. python -m knowledge3d.tools.phase25.ingest_pdf_corpus \
    --roots \
      "/mnt/arquivos/0 ChatGPTs/DataBase/Encyclopedias,/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries" \
    --max-depth 4 --limit 100
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PUBLIC = ROOT / "viewer/public"
MATERIAL_DIR = PUBLIC / "house/materialized_objects"
MANIFEST = MATERIAL_DIR / "manifest.json"


def _iter_files(root: Path, exts: Tuple[str, ...], max_depth: int) -> Iterable[Path]:
    base_depth = len(root.parts)
    for p in root.rglob('*'):
        try:
            if not p.is_file():
                continue
            if p.suffix.lower() not in exts:
                continue
            if len(p.parts) - base_depth > max_depth:
                continue
            yield p
        except Exception:
            continue


def _skip_page_text(s: str) -> bool:
    t = (s or "").strip().lower()
    if not t:
        return False
    banned = (
        "copyright", "all rights reserved", "isbn", "printed", "publisher",
        "table of contents", "contents", "index", "acknowledgments", "foreword",
    )
    return any(k in t for k in banned)


def _pdf_previews(pdf: Path, out_dir: Path, limit_pages: int = 6) -> List[Path]:
    out: List[Path] = []
    try:
        import fitz  # type: ignore
    except Exception:
        return out
    try:
        doc = fitz.open(pdf.as_posix())
    except Exception:
        return out
    candidates = list(range(min(limit_pages, doc.page_count)))
    tail = list(range(max(0, doc.page_count - limit_pages), doc.page_count))
    pages = sorted(set(candidates + tail))
    for i in pages:
        try:
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            if _skip_page_text(text):
                continue
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"p{i+1:03d}.png"
            pix.save(out_path.as_posix())
            out.append(out_path)
        except Exception:
            continue
    return out


def _update_manifest(entries: List[dict]) -> None:
    obj = {"shapes": [], "rays": []}
    if MANIFEST.exists():
        try:
            obj = json.loads(MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            obj = {"shapes": [], "rays": []}
    shapes = obj.get("shapes") if isinstance(obj, dict) else []
    if not isinstance(shapes, list):
        shapes = []
    existing = {s.get("source_path") for s in shapes if isinstance(s, dict)}
    for e in entries:
        if e.get("source_path") not in existing:
            shapes.append(e)
    obj["shapes"] = shapes
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def run(roots: List[Path], max_depth: int, limit: int) -> None:
    entries: List[dict] = []
    seen = 0
    for rt in roots:
        for f in _iter_files(rt, (".pdf", ".json"), max_depth):
            if f.suffix.lower() == ".pdf":
                previews = _pdf_previews(f, MATERIAL_DIR / "docs" / f.stem)
                preview_rel = None
                if previews:
                    preview_rel = "/" + str(previews[0].relative_to(PUBLIC)).replace(os.sep, "/")
                entries.append({
                    "name": f.stem,
                    "prompt": f"doc {f.stem}",
                    "type": "external_document",
                    "source_path": f.as_posix(),
                    "preview": preview_rel,
                })
            else:  # JSON
                entries.append({
                    "name": f.stem,
                    "prompt": f"json {f.stem}",
                    "type": "external_json",
                    "source_path": f.as_posix(),
                })
            seen += 1
            if limit and seen >= limit:
                break
        if limit and seen >= limit:
            break
    _update_manifest(entries)
    print(f"Ingested {len(entries)} entries into manifest.")


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Ingest external PDFs/JSONs into House manifest")
    ap.add_argument("--roots", type=str, required=True, help="Comma-separated directories")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()
    roots = [Path(r.strip()) for r in args.roots.split(",") if r.strip()]
    run(roots, int(args.max_depth), int(args.limit))


if __name__ == "__main__":
    main()

