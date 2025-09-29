"""Wikipedia sweep to sanity-check non-math routing of the fused head.

Usage
  conda run -n k3d-cranium env PYTHONPATH=. python -m knowledge3d.tools.wiki_sweep_evaluator \
      --max-lines 0 --summarize

Behavior
- Ensures a local corpus exists (builds via fetch_wiki_corpus if missing).
- Runs the fused head across many Wikipedia lines.
- Reports how often answers include math/RPN traces (should be ~0 for text).
- Saves aggregate stats and a small sample to logs/wiki_sweep_report.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

from knowledge3d.tools.fetch_wiki_corpus import OUT_TXT as _OUT_TXT, DEFAULT_TOPICS
from knowledge3d.tools.fetch_wiki_corpus import fetch_plain, iter_lines
from knowledge3d.cranium.fused_head import AdaptedFusedHead  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs"
REPORT = LOGS / "wiki_sweep_report.json"


def ensure_corpus(topics: List[str]) -> Path:
    if _OUT_TXT.exists() and _OUT_TXT.stat().st_size > 0:
        return _OUT_TXT
    # Build quickly from default topics
    _OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with _OUT_TXT.open("w", encoding="utf-8") as fh:
        for t in topics:
            try:
                text = fetch_plain(t)
            except Exception:
                continue
            for ln in iter_lines(text):
                s = ln.strip()
                if len(s) < 80 or len(s) > 600:
                    continue
                if s in seen:
                    continue
                fh.write(s + "\n")
                seen.add(s)
    return _OUT_TXT


def run(max_lines: int = 0, summarize: bool = False, zero_emb: bool = True) -> Dict[str, object]:
    # Avoid NVRTC text-modality initialisation during large text-only sweeps
    os.environ.setdefault("K3D_DISABLE_TEXT_MODALITY", "1")
    path = ensure_corpus(DEFAULT_TOPICS)
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if max_lines and max_lines > 0:
        lines = lines[: int(max_lines)]

    fh = AdaptedFusedHead()

    total = 0
    nonempty = 0
    boxed = 0
    rpn_tag = 0
    honesty_note = 0
    samples: List[Dict[str, str]] = []

    for i, src in enumerate(lines):
        q = f"Summarize: {src}" if summarize else src
        emb = ([0.0] * 2048) if zero_emb else ([0.0] * 2048)
        pred = fh.predict(q, emb)
        total += 1
        if str(pred or "").strip():
            nonempty += 1
        if "\\boxed{" in str(pred):
            boxed += 1
        if "Tags: [logic, rpn" in str(pred):
            rpn_tag += 1
        if "Honesty note:" in str(pred):
            honesty_note += 1
        if len(samples) < 50 and random.random() < 0.2:
            samples.append({"q": q[:240], "pred": str(pred)[:480]})

    result = {
        "total": total,
        "nonempty": nonempty,
        "boxed_count": boxed,
        "rpn_tag_count": rpn_tag,
        "honesty_note_count": honesty_note,
        "nonempty_rate": (nonempty / total if total else 0.0),
        "rpn_rate": (rpn_tag / total if total else 0.0),
        "boxed_rate": (boxed / total if total else 0.0),
        "samples": samples,
    }
    LOGS.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wikipedia sweep complete → {REPORT}")
    print(f"  total={total} nonempty={nonempty} rpn_rate={result['rpn_rate']:.3f}")
    return result


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Run a Wikipedia sweep to check non-math routing")
    ap.add_argument("--max-lines", type=int, default=0, help="Cap number of lines (0=unlimited)")
    ap.add_argument("--summarize", action="store_true", help="Prefix with 'Summarize:' to exercise summary path")
    ap.add_argument("--no-zero-emb", action="store_true", help="Use non-zero embeddings (not recommended for large sweeps)")
    args = ap.parse_args()
    run(max_lines=int(args.max_lines), summarize=bool(args.summarize), zero_emb=not args.no_zero_emb)


if __name__ == "__main__":  # pragma: no cover
    main()
