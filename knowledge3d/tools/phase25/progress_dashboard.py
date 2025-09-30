from __future__ import annotations

"""Aggregate progress logs into simple dashboard summaries.

Reads docs/benchmarks/progress_log.json and writes:
- docs/benchmarks/progress_summary.json (latest stats per trainer)
- docs/benchmarks/progress_summary.md (human-readable summary)
"""

import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[3]
LOG = ROOT / 'docs/benchmarks/progress_log.json'


def main() -> None:  # pragma: no cover
    if not LOG.exists():
        print('No progress_log.json found.')
        return
    try:
        data: List[Dict] = json.loads(LOG.read_text(encoding='utf-8'))
    except Exception:
        print('Invalid progress_log.json')
        return
    latest: Dict[str, Dict] = {}
    for rec in data:
        tr = str(rec.get('trainer') or 'unknown')
        if tr not in latest or float(rec.get('ts', 0)) > float(latest[tr].get('ts', 0)):
            latest[tr] = rec
    # Write JSON
    out_json = ROOT / 'docs/benchmarks/progress_summary.json'
    out_json.write_text(json.dumps({'latest': latest}, ensure_ascii=False, indent=2), encoding='utf-8')
    # Write MD
    lines = ["# Training Progress Summary\n"]
    for tr, rec in latest.items():
        if tr == 'multi':
            lines.append(f"- Multi-trainer: epoch {rec.get('epoch')} — numeric={rec.get('numeric_trained')} rpn={rec.get('rpn_trained')}")
        else:
            lines.append(f"- {tr.capitalize()}: epoch {rec.get('epoch')} — avg_loss={rec.get('avg_loss')}")
    out_md = ROOT / 'docs/benchmarks/progress_summary.md'
    out_md.write_text("\n".join(lines) + "\n", encoding='utf-8')
    print('Wrote progress summaries.')


if __name__ == '__main__':
    main()

