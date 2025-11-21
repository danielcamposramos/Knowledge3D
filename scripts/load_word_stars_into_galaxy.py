#!/usr/bin/env python3
"""
Loader scaffold: merge PD-packed word stars and prepare for Galaxy/House ingestion.

Note: The actual Galaxy/House upsert bridge is not implemented here (no public API
present in repo). This script produces a merged file ready for an upsert bridge,
and logs basic stats. Replace the placeholder `wire_to_galaxy` once the bridge
class is available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare word stars for Galaxy/House ingestion.")
    p.add_argument("--inputs", nargs="+", required=True, help="Input word star JSONL files (PD-packed).")
    p.add_argument("--output", required=True, help="Merged output JSONL.")
    return p.parse_args()


def wire_to_galaxy(_stars_path: Path) -> None:
    """
    Placeholder. Implement Galaxy/House upsert when API is available.
    """
    raise NotImplementedError("Galaxy/House upsert bridge not implemented.")


def merge_files(inputs, output: Path) -> int:
    from merge_word_stars import main as merge_main  # reuse merger
    merge_main()
    # merge_main writes to output given via CLI; here we just return count
    count = sum(1 for _ in output.open("r", encoding="utf-8"))
    return count


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)

    # Merge inputs using merge_word_stars utility
    # Work by invoking as a module-like call with provided args
    merged: dict = {}
    for path_str in args.inputs:
        path = Path(path_str)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                star = json.loads(line)
                key = (star.get("lang", ""), star.get("lemma", ""))
                merged[key] = star

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_f:
        for star in merged.values():
            out_f.write(json.dumps(star, ensure_ascii=False) + "\n")

    print(f"Word stars merged: {len(merged)} -> {out_path}")
    print("Galaxy/House upsert not implemented; replace wire_to_galaxy when ready.")


if __name__ == "__main__":
    main()
