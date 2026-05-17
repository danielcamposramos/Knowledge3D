#!/usr/bin/env python3
"""Import Encapsulate CST/CRT artifacts into K3D galaxies."""

from __future__ import annotations

import argparse
from pathlib import Path

from knowledge3d.ingestion.encapsulate_importer import EncapsulateImporter


def _iter_cst_paths(cst_path: Path | None, cst_dir: Path | None) -> list[Path]:
    if cst_path is not None:
        return [cst_path]
    if cst_dir is None:
        return []
    return sorted(cst_dir.rglob("*.csts.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cst", type=Path, help="Single .csts.json file")
    parser.add_argument("--crt", type=Path, help="Optional .crts.json file (single CST mode)")
    parser.add_argument("--cst-dir", type=Path, help="Directory containing .csts.json files")
    parser.add_argument("--storage-root", type=Path, default=Path("/K3D/Knowledge3D.local"))
    parser.add_argument("--namespace", default="christoph_encapsulate")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = _iter_cst_paths(args.cst, args.cst_dir)
    if not paths:
        raise SystemExit("No CST inputs found. Use --cst or --cst-dir.")

    importer = EncapsulateImporter(storage_root=args.storage_root)
    total_entries = 0
    total_capsules = 0
    total_symlinks = 0

    for cst_path in paths:
        crt_path = args.crt
        if crt_path is None:
            candidate = Path(str(cst_path).replace(".csts.json", ".crts.json"))
            crt_path = candidate if candidate.exists() else None
        result = importer.import_capsule_source_tree(
            cst_path=cst_path,
            crt_path=crt_path,
            namespace=args.namespace,
            dry_run=bool(args.dry_run),
        )
        total_entries += int(result.get("entries_created", 0))
        total_capsules += int(result.get("capsules_processed", 0))
        total_symlinks += int(result.get("symlink_entries_created", 0))
        print(
            f"[import] cst={cst_path} crt={crt_path} "
            f"capsules={result.get('capsules_processed', 0)} "
            f"entries={result.get('entries_created', 0)} "
            f"symlinks={result.get('symlink_entries_created', 0)}"
        )

    print(
        f"[summary] files={len(paths)} capsules={total_capsules} "
        f"entries={total_entries} symlinks={total_symlinks} dry_run={bool(args.dry_run)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

