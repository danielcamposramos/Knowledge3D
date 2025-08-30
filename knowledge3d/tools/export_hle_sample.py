"""
Export a small sample of HLE questions from Hugging Face to local JSON files
under ../Knowledge3D.local/datasets/exams/hle-sample and build/update exams_index.json.

Requires: pip install datasets

Usage:
  python3 -m knowledge3d.tools.export_hle_sample --count 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Export HLE sample to local datasets")
    ap.add_argument("--count", type=int, default=50)
    args = ap.parse_args()
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise SystemExit("Missing dependency: datasets. pip install datasets") from exc

    repo = Path(__file__).resolve().parents[2]
    root = repo.parent / f"{repo.name}.local" / "datasets" / "exams"
    out_dir = root / "hle-sample"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("cais/hle", split="test")
    total = min(int(args.count), len(ds))
    items: List[dict] = []
    for i in range(total):
        rec = ds[i]
        # Normalize to our display format
        obj = {
            "id": str(rec.get("id") or f"hle-{i}"),
            "title": str(rec.get("title") or rec.get("subject") or f"HLE {i}"),
            "prompt": str(rec.get("question_text") or rec.get("question") or ""),
        }
        # choices and answer_idx if available
        choices = rec.get("choices") or rec.get("options")
        if isinstance(choices, list) and choices:
            obj["choices"] = [str(c) for c in choices]
            # normalize answer index
            ans = rec.get("answer_idx")
            if isinstance(ans, int):
                obj["answer_idx"] = ans
            else:
                # try mapping answer string to index
                ans_text = str(rec.get("answer") or rec.get("correct_choice") or "").strip()
                try:
                    obj["answer_idx"] = [c.strip() for c in obj["choices"]].index(ans_text)
                except ValueError:
                    pass
        else:
            # free-text answer if present
            ans = rec.get("answer") or rec.get("short_answer")
            if isinstance(ans, str) and ans.strip():
                obj["answer"] = ans.strip()
        # Write file
        fname = f"hle_{obj['id']}.json"
        (out_dir / fname).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        items.append({
            "id": obj["id"],
            "source": "HLE",
            "title": obj["title"],
            "url": f"/exams/hle-sample/{fname}",
            "kind": "hle",
        })

    # Update root exams index
    idx_path = root.parent / "exams_index.json"
    existing: List[dict] = []
    if idx_path.exists():
        try:
            existing = json.loads(idx_path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    # Filter out old HLE entries and extend
    existing = [e for e in existing if e.get("source") != "HLE"] + items
    idx_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(items)} HLE items -> {out_dir}")
    print(f"Updated index -> {idx_path}")


if __name__ == "__main__":
    main()

