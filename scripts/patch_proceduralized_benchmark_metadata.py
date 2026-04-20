#!/usr/bin/env python3
"""
Ingestion-path fixup: tag proceduralized benchmark galaxies as role=answer.

Reads one or more JSONL galaxy files and ensures every star that is missing
`metadata.selection_role` or `metadata.answer_eligible` gets them set to the
correct values for proceduralized answer artifacts:
    selection_role  = "answer"
    answer_eligible = true

These stars carry router_refs at the top level, which makes them route-active
in the validator. Without selection_role/answer_eligible the daemon boot fails
at sovereign_hot_path.py:1651 with `sovereign_build_metadata_invalid`.

Idempotent: re-running produces byte-identical output.
Backs up each file once as <name>.bak.<timestamp> before writing.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

KNOWN_ROLES = {"router", "executor", "validator", "answer", "anti_pattern"}

DEFAULT_PATHS = [
    "/K3D/Knowledge3D.local/galaxies/proceduralized_gsm8k_train_10.jsonl",
    "/K3D/Knowledge3D.local/galaxies/proceduralized_mmlu_val_10.jsonl",
]


def _needs_patch(metadata: dict) -> tuple[bool, bool]:
    """Return (needs_selection_role, needs_answer_eligible)."""
    sr = str(metadata.get("selection_role") or "").strip().lower()
    needs_sr = sr not in KNOWN_ROLES
    needs_ae = "answer_eligible" not in metadata
    return needs_sr, needs_ae


def patch_file(path: Path, dry_run: bool) -> None:
    """Patch a single JSONL file in place."""
    lines = path.read_text(encoding="utf-8").splitlines()
    total = 0
    patched = 0
    already_compliant = 0
    out_lines: list[str] = []

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            out_lines.append(raw_line)
            continue
        total += 1
        star = json.loads(stripped)
        metadata = star.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            star["metadata"] = metadata

        needs_sr, needs_ae = _needs_patch(metadata)

        if needs_sr or needs_ae:
            patched += 1
            if dry_run:
                star_id = star.get("id", f"line_{total}")
                changes = []
                if needs_sr:
                    changes.append("selection_role -> answer")
                if needs_ae:
                    changes.append("answer_eligible -> true")
                print(f"  [DRY-RUN] {star_id}: would set {', '.join(changes)}")
            else:
                if needs_sr:
                    metadata["selection_role"] = "answer"
                if needs_ae:
                    metadata["answer_eligible"] = True
        else:
            already_compliant += 1

        out_lines.append(json.dumps(star, sort_keys=False, ensure_ascii=False))

    print(f"  path           : {path}")
    print(f"  total stars    : {total}")
    print(f"  patched        : {patched}")
    print(f"  already-ok     : {already_compliant}")

    if dry_run:
        print("  [DRY-RUN] no files written")
        return

    if patched == 0:
        print("  nothing to write (all compliant)")
        return

    # Backup once — skip if .bak already exists
    bak_candidates = sorted(path.parent.glob(f"{path.name}.bak.*"))
    if not bak_candidates:
        timestamp = int(time.time())
        bak_path = path.with_suffix(f".jsonl.bak.{timestamp}")
        shutil.copy2(path, bak_path)
        print(f"  backup         : {bak_path}")
    else:
        print(f"  backup exists  : {bak_candidates[0]} (skipped)")

    output = "\n".join(out_lines) + "\n"
    path.write_text(output, encoding="utf-8")
    print("  written.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tag proceduralized benchmark galaxy stars as selection_role=answer"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="JSONL file paths to patch (defaults to known proceduralized benchmark galaxies)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended changes without writing files",
    )
    args = parser.parse_args()

    target_paths = [Path(p) for p in args.paths] if args.paths else [Path(p) for p in DEFAULT_PATHS]

    any_missing = False
    for p in target_paths:
        if not p.exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            any_missing = True
    if any_missing:
        sys.exit(1)

    for p in target_paths:
        print(f"\n--- {p.name} ---")
        patch_file(p, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
