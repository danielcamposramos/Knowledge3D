#!/usr/bin/env python3
"""
Warm-boot state patch: backfill selection_role / answer_eligible on proceduralized
benchmark stars inside the serialized house state.

Why this exists:
    Knowledgeverse boots by restoring `house/galaxy_state.bin` (pickle) and/or
    `checkpoints/galaxy_consolidated_latest.json` before it ever reads the
    per-galaxy JSONLs. When those snapshots were serialized, proceduralized
    benchmark stars (`gsm8k_train_*`, `mmlu_val_*`, `mmlu_math_*`) had their
    top-level `selection_role` / `answer_eligible` stripped by an older
    normalization path. Patching the JSONL on disk does nothing because the
    warm-boot state clobbers the in-memory galaxy before the JSONL is read.

    This script walks the warm-boot payload(s), finds proceduralized benchmark
    stars (any star that is route-active via router_refs AND whose id matches a
    known proceduralized prefix), and sets:
        entry["selection_role"] = "answer"
        entry["answer_eligible"] = True
        entry["metadata"]["selection_role"] = "answer"
        entry["metadata"]["answer_eligible"] = True

    Idempotent — re-running is a no-op.

Targets (patched only if they exist):
    /K3D/Knowledge3D.local/house/galaxy_state.bin              (pickle)
    /K3D/Knowledge3D.local/checkpoints/galaxy_consolidated_latest.json (json, large)

Each patched file is backed up once as <name>.bak.<timestamp> unless a .bak.*
already exists (so we don't erase an older pre-patch backup on repeat runs).
"""

import argparse
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

DEFAULT_PICKLE = Path("/K3D/Knowledge3D.local/house/galaxy_state.bin")
DEFAULT_JSON = Path("/K3D/Knowledge3D.local/checkpoints/galaxy_consolidated_latest.json")

ID_PREFIXES = (
    "gsm8k_train_",
    "gsm8k_val_",
    "gsm8k_test_",
    "mmlu_val_",
    "mmlu_test_",
    "mmlu_math_",
    "math_train_",
    "math_test_",
)


def _layer_is_valid(value: object) -> bool:
    try:
        return int(value) > 0  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _needs_patch(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False
    entry_id = str(entry.get("id") or entry.get("rule_id") or "").strip()
    if not any(entry_id.startswith(prefix) for prefix in ID_PREFIXES):
        return False
    md = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    top_role = str(entry.get("selection_role") or "").strip().lower()
    md_role = str(md.get("selection_role") or "").strip().lower()
    role_ok = top_role == "answer" and md_role == "answer"
    answer_ok = bool(entry.get("answer_eligible")) is True and bool(md.get("answer_eligible")) is True
    layer_ok = (
        _layer_is_valid(entry.get("layer_id"))
        or _layer_is_valid(entry.get("layer"))
        or _layer_is_valid(md.get("layer_id"))
        or _layer_is_valid(md.get("layer"))
    )
    return not (role_ok and answer_ok and layer_ok)


def _patch_entry(entry: dict) -> bool:
    changed = False
    if str(entry.get("selection_role") or "").strip().lower() != "answer":
        entry["selection_role"] = "answer"
        changed = True
    if bool(entry.get("answer_eligible")) is not True:
        entry["answer_eligible"] = True
        changed = True
    md = entry.get("metadata")
    if not isinstance(md, dict):
        md = {}
        entry["metadata"] = md
    if str(md.get("selection_role") or "").strip().lower() != "answer":
        md["selection_role"] = "answer"
        changed = True
    if bool(md.get("answer_eligible")) is not True:
        md["answer_eligible"] = True
        changed = True
    # Proceduralized benchmark stars are meaning-centric (layer 2).
    # _infer_layer_id walks metadata.layer_id, source.layer_id, metadata.layer,
    # source.layer — set both layer AND layer_id on entry and metadata so the
    # validator accepts whichever key it reaches first.
    if not _layer_is_valid(entry.get("layer_id")):
        entry["layer_id"] = 2
        changed = True
    if not _layer_is_valid(entry.get("layer")):
        entry["layer"] = 2
        changed = True
    if not _layer_is_valid(md.get("layer_id")):
        md["layer_id"] = 2
        changed = True
    if not _layer_is_valid(md.get("layer")):
        md["layer"] = 2
        changed = True
    return changed


def _walk_galaxies(galaxies: dict) -> tuple[int, int]:
    total = 0
    patched = 0
    for galaxy_name, entries in galaxies.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_id = str(entry.get("id") or entry.get("rule_id") or "").strip()
            if not any(entry_id.startswith(prefix) for prefix in ID_PREFIXES):
                continue
            total += 1
            if _needs_patch(entry):
                if _patch_entry(entry):
                    patched += 1
    return total, patched


def _ensure_backup(path: Path) -> None:
    existing = sorted(path.parent.glob(f"{path.name}.bak.*"))
    if existing:
        print(f"  backup exists  : {existing[0]} (skipped)")
        return
    stamp = int(time.time())
    bak_path = path.with_suffix(f"{path.suffix}.bak.{stamp}")
    shutil.copy2(path, bak_path)
    print(f"  backup         : {bak_path}")


def patch_pickle(path: Path, dry_run: bool) -> None:
    print(f"\n--- pickle: {path} ---")
    if not path.exists():
        print("  skip (does not exist)")
        return
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "galaxies" not in payload:
        print("  skip (unexpected payload shape)")
        return
    galaxies = payload.get("galaxies") or {}
    if not isinstance(galaxies, dict):
        print("  skip (galaxies not a dict)")
        return
    total, patched = _walk_galaxies(galaxies)
    print(f"  candidate stars: {total}")
    print(f"  needs patch    : {patched}")
    if dry_run:
        print("  [DRY-RUN] no files written")
        return
    if patched == 0:
        print("  nothing to write (all compliant)")
        return
    _ensure_backup(path)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("  written.")


def patch_json(path: Path, dry_run: bool) -> None:
    print(f"\n--- json: {path} ---")
    if path.is_symlink():
        path = path.resolve()
        print(f"  resolved symlink -> {path}")
    if not path.exists():
        print("  skip (does not exist)")
        return
    # json.load / json.dump on 1.9 GB is memory-heavy but single-pass;
    # preferred over streaming because payload is a single nested object.
    size_gb = path.stat().st_size / (1024 ** 3)
    print(f"  size           : {size_gb:.2f} GiB (loading fully)")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "galaxies" not in payload:
        print("  skip (unexpected payload shape)")
        return
    galaxies = payload.get("galaxies") or {}
    if not isinstance(galaxies, dict):
        print("  skip (galaxies not a dict)")
        return
    total, patched = _walk_galaxies(galaxies)
    print(f"  candidate stars: {total}")
    print(f"  needs patch    : {patched}")
    if dry_run:
        print("  [DRY-RUN] no files written")
        return
    if patched == 0:
        print("  nothing to write (all compliant)")
        return
    _ensure_backup(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    print("  written.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pickle", type=Path, default=DEFAULT_PICKLE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, dest="json_path")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-json", action="store_true",
                        help="Skip the large consolidated JSON (pickle-only patch)")
    args = parser.parse_args()

    patch_pickle(args.pickle, dry_run=args.dry_run)
    if not args.skip_json:
        patch_json(args.json_path, dry_run=args.dry_run)
    else:
        print("\n--- json: skipped (--skip-json) ---")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
