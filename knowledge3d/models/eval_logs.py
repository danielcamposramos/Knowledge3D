from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_pairs(log_dir: Path) -> Tuple[List[str], List[str]]:
    gold: Dict[str, str] = {}
    pred: Dict[str, str] = {}
    for p in sorted(log_dir.glob("session-*.jsonl")):
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    t = rec.get("type")
                    if t == "chat_response":
                        text = str(rec.get("text") or "")
                        resp = rec.get("response") or {}
                        act_type = resp.get("type")
                        action = resp.get("action")
                        if act_type in {"navigation", "exploration", "interaction"} and action:
                            gold[text] = str(action)
                    elif t == "model_prediction":
                        text = str(rec.get("text") or "")
                        action = rec.get("pred_action")
                        if action:
                            pred[text] = str(action)
        except OSError:
            continue
    keys = [k for k in gold.keys() if k in pred]
    y_true = [gold[k] for k in keys]
    y_pred = [pred[k] for k in keys]
    return y_true, y_pred


def confusion(y_true: List[str], y_pred: List[str]) -> Dict[str, Dict[str, int]]:
    labels = sorted(set(y_true) | set(y_pred))
    mat: Dict[str, Dict[str, int]] = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(y_true, y_pred):
        mat[g][p] += 1
    return mat


def main():  # pragma: no cover
    import argparse
    p = argparse.ArgumentParser(description="Evaluate model predictions vs. gold actions from logs")
    p.add_argument("--logs", required=True, help="Logs directory (../Knowledge3D.local/logs)")
    a = p.parse_args()
    y_true, y_pred = load_pairs(Path(a.logs))
    print(json.dumps({"pairs": len(y_true), "labels": sorted(set(y_true) | set(y_pred)), "confusion": confusion(y_true, y_pred)}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()

