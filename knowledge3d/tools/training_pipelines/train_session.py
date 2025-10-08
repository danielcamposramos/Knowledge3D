"""
Run a lightweight training/eval session on a K3D GLB.

Steps
- Reflection: summarize structure and hubs
- Tasks: generate GOTO/DOOR tasks with BFS baseline
- Report: write a markdown summary with quick metrics

Usage
  python -m knowledge3d.tools.train_session \
    --gltf viewer/public/ai_books_basic.full.umap.doors.glb \
    --pairs 128 --door 64 \
    --out-dir docs/reports/training
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .reflect_glb import reflect  # type: ignore
from .eval_tasks import generate_tasks  # type: ignore


def summarize_tasks(tasks: dict) -> dict:
    goto = tasks.get("goto_tasks", []) or []
    door = tasks.get("door_tasks", []) or []
    def _summ(items):
        n = len(items)
        succ = sum(1 for it in items if it.get("exists"))
        lens = sorted([it.get("path_len") for it in items if isinstance(it.get("path_len"), int)])
        med = lens[len(lens)//2] if lens else None
        return {"count": n, "success": succ, "success_rate": (succ/n if n else 0.0), "median_hops": med}
    return {"goto": _summ(goto), "door": _summ(door)}


def main() -> None:
    p = argparse.ArgumentParser(description="Train/eval session runner for K3D GLB")
    p.add_argument("--gltf", required=True)
    p.add_argument("--pairs", type=int, default=128)
    p.add_argument("--door", type=int, default=64)
    p.add_argument("--out-dir", default="docs/reports/training")
    args = p.parse_args()

    gltf = Path(args.gltf)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reflection
    stats = reflect(gltf)
    # Tasks
    tasks = generate_tasks(gltf, args.pairs, args.door)
    summary = summarize_tasks(tasks)

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    md = [
        f"# Training Session — {ts}",
        f"- GLB: `{gltf}`",
        f"- Nodes: {stats['nodes']}  Avg degree: {stats['avg_degree']:.2f}",
        f"- Doors: {stats['doors']}  Guided nodes: {stats['mask_true']}",
        f"- Hubs: {', '.join(stats['top_hubs']) if stats['top_hubs'] else '—'}",
        "",
        "## Task Summary",
        f"- GOTO: {summary['goto']['success']}/{summary['goto']['count']} ok  (rate={(summary['goto']['success_rate']*100):.1f}%)  median hops={summary['goto']['median_hops']}",
        f"- DOOR: {summary['door']['success']}/{summary['door']['count']} ok  (rate={(summary['door']['success_rate']*100):.1f}%)  median hops={summary['door']['median_hops']}",
        "",
    ]
    out_md = out_dir / f"session-{ts}.md"
    out_md.write_text("\n".join(md), encoding="utf-8")

    # Persist tasks JSON alongside
    out_tasks = out_dir / f"tasks-{ts}.json"
    out_tasks.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    # Also write a viewer-fetchable scoreboard JSON
    viewer_pub = Path('viewer/public/training')
    try:
      viewer_pub.mkdir(parents=True, exist_ok=True)
      latest = {
        'ts': ts,
        'nodes': stats['nodes'], 'avg_degree': stats['avg_degree'],
        'doors': stats['doors'], 'guided': stats['mask_true'],
        'hubs': stats['top_hubs'],
        'goto': summary['goto'], 'door': summary['door'],
      }
      (viewer_pub / 'latest.json').write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
      pass
    print(f"Wrote {out_md}\nWrote {out_tasks}\nUpdated viewer/public/training/latest.json (if possible)")


if __name__ == "__main__":
    main()
