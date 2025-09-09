from __future__ import annotations

"""
Build RLWHF dataset from session logs.

Pairs chat_response messages with adjacent feedback events (if any) and
computes scalar rewards under the Honesty+Feedback policy:
  rating: good -> +1.0; partial -> +0.5; bad -> -0.25
  honesty bump: if answer admits "don't know" and context weak -> +0.5

Outputs JSONL rows with fields: {query, answer, contexts, rating, reward, gold}

Usage:
  scripts/k3d_env.sh run python -m knowledge3d.tools.build_rlwhf_dataset \
    --logs ../Knowledge3D.local/logs \
    --out docs/reports/training/rlwhf_dataset.jsonl \
    --summary docs/reports/status/rlwhf_summary.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def iter_logs(log_dir: Path):
    for p in sorted(log_dir.glob("session-*.jsonl")):
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue
        except OSError:
            continue


def honesty_bonus(ans: str, contexts: List[str]) -> float:
    t = (ans or "").lower().strip()
    if any(s in t for s in ["i don't know", "i do not know", "unsure", "not sure", "i'm not certain"]):
        # When no explicit feedback is present, honesty earns partial credit
        return 0.5
    return 0.0


def main() -> None:  # pragma: no cover
    p = argparse.ArgumentParser(description="Build RLWHF dataset from logs")
    p.add_argument("--logs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--summary", required=True)
    args = p.parse_args()
    log_dir = Path(args.logs)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(iter_logs(log_dir))
    out_rows: List[dict] = []
    # For simplicity, pair each chat_response with the next feedback event on same channel (if exists)
    by_ch: Dict[str, List[dict]] = {}
    for r in rows:
        ch = str(r.get("channel") or "#general")
        by_ch.setdefault(ch, []).append(r)
    counts = {"good":0, "partial":0, "bad":0, "none":0}
    total_reward = 0.0
    for ch, seq in by_ch.items():
        for i, rec in enumerate(seq):
            if rec.get("type") != "chat_response":
                continue
            resp = rec.get("response") or {}
            if resp.get("type") != "chat_response":
                continue
            q = str(rec.get("text") or "")
            a = str(resp.get("message") or "")
            ctx = rec.get("context") or {}
            hist = ctx.get("history_tail") or []
            contexts = [str(h.get("message") or "") for h in hist]
            rating: Optional[str] = None
            gold: Optional[str] = None
            # scan forward for feedback
            for j in range(i+1, min(i+6, len(seq))):
                r2 = seq[j]
                if r2.get("type") == "feedback":
                    rating = str(r2.get("rating") or "")
                    gold = str(r2.get("gold") or "") or None
                    break
            reward = 0.0
            if rating in ("good", "partial", "bad"):
                counts[rating] += 1
                reward += {"good":1.0, "partial":0.5, "bad":-0.25}[rating]
            else:
                counts["none"] += 1
                reward += honesty_bonus(a, contexts)
            total_reward += reward
            out_rows.append({
                "channel": ch,
                "query": q,
                "answer": a,
                "contexts": contexts,
                "rating": rating,
                "reward": reward,
                "gold": gold,
            })
    with out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = {
        "messages": len(out_rows),
        "avg_reward": (total_reward/len(out_rows) if out_rows else 0.0),
        "sum_reward": total_reward,
        "counts": counts,
        "out": str(out),
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()

