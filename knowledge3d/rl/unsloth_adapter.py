"""
Unsloth memory-efficient RL adapter (early integration).

This module wires a lightweight RL loop that can fine-tune a policy on
K3D chat/navigation intents with small memory footprint, inspired by
https://docs.unsloth.ai/basics/memory-efficient-rl .

Design
- Optional dependency: 'unsloth' or 'trl' (Transformers Reinforcement Learning) + torch
- Dataset: built from live session logs (../Knowledge3D.local/logs)
  - Each sample has (state: text, action: label), optional reward (e.g., success on routing, sim score)
- Objective: Improve intent classification/reaction under low VRAM by
  using parameter-efficient fine-tune and small batch RL updates.

CLI
  python -m knowledge3d.rl.unsloth_adapter train --logs ../Knowledge3D.local/logs \
    --out ../Knowledge3D.local/models/intent_rl --steps 1000

Notes
- This is a thin scaffold. For production RL, adopt Unsloth APIs or TRL PPO, add
  reward shaping (e.g., shorter hops, higher sim), and enforce safety gates (Faith Engine).
 - Debian containment guard: requires Conda or Docker unless K3D_ALLOW_NATIVE=1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

try:
    # Guard unsafe runs on Debian without containment
    from ..utils.env_guard import enforce_containment  # type: ignore
    enforce_containment("RL/Unsloth training")
except Exception:
    pass

LABELS = [
    "teleport", "move", "goto", "follow", "orbit",
    "show", "find_related", "expand", "hide",
    "touch", "talk", "give",
]


def build_basic_dataset(log_dir: Path) -> Tuple[List[str], List[str], List[float]]:
    X: List[str] = []
    y: List[str] = []
    r: List[float] = []
    for p in sorted(log_dir.glob("session-*.jsonl")):
        try:
            for ln in p.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                rec = json.loads(ln)
                if rec.get("type") == "chat_response":
                    text = str(rec.get("text") or "")
                    resp = rec.get("response") or {}
                    act_type = resp.get("type"); action = resp.get("action")
                    if act_type in {"navigation", "exploration", "interaction"} and action in LABELS:
                        X.append(text)
                        y.append(action)
                        # weak reward: +1 for actionable, else 0
                        r.append(1.0)
                if rec.get("type") == "goto_resolution":
                    # boost reward on high label similarity
                    sc = float(rec.get("score") or 0.0)
                    X.append(str(rec.get("query") or ""))
                    y.append("goto")
                    r.append(max(0.0, min(1.0, sc)))
        except Exception:
            continue
    return X, y, r


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Unsloth-like memory-efficient RL scaffold for K3D")
    ap.add_argument("command", choices=["train", "dump"])  # 'dump' outputs a JSON dataset preview
    ap.add_argument("--logs", default=str((Path(__file__).resolve().parents[2].parent / (Path(__file__).resolve().parents[2].name + ".local") / "logs")))
    ap.add_argument("--out", default=str((Path(__file__).resolve().parents[2].parent / (Path(__file__).resolve().parents[2].name + ".local") / "models" / "intent_rl")))
    ap.add_argument("--steps", type=int, default=1000)
    args = ap.parse_args()
    logs = Path(args.logs)
    out = Path(args.out)
    X, y, rew = build_basic_dataset(logs)
    if args.command == "dump":
        out.parent.mkdir(parents=True, exist_ok=True)
        (out.with_suffix('.dataset.json')).write_text(json.dumps({"X": X[:256], "y": y[:256], "r": rew[:256]}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote preview -> {out.with_suffix('.dataset.json')}")
        return
    # Train: prefer TRL PPO if available; otherwise save dataset for external trainer
    try:
        import torch  # type: ignore
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        # Minimal policy fine-tune with rewards as sample weights
        label2id = {lbl: i for i, lbl in enumerate(LABELS)}
        y_ids = [label2id.get(lbl, 0) for lbl in y]
        tok = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        inputs = tok(X, padding=True, truncation=True, return_tensors='pt')
        labels = torch.tensor(y_ids)
        weights = torch.tensor(rew) if rew else torch.ones_like(labels, dtype=torch.float)
        model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased', num_labels=len(LABELS))
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
        steps = min(args.steps, len(labels))
        for i in range(steps):
            opt.zero_grad(set_to_none=True)
            out = model(**{k: v[:8] for k, v in inputs.items()}, labels=labels[:8])  # tiny batch 8
            loss = out.loss * (weights[:8].mean())
            loss.backward()
            opt.step()
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out))
        tok.save_pretrained(str(out))
        (out / "labels.json").write_text(json.dumps({"labels": LABELS}), encoding="utf-8")
        print(f"Saved RL-tuned model -> {out}")
    except Exception as e:
        # Fallback: write dataset for external Unsloth/TRL runner
        out.parent.mkdir(parents=True, exist_ok=True)
        (out.with_suffix('.dataset.json')).write_text(json.dumps({"X": X, "y": y, "r": rew}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved dataset for RL external trainer -> {out.with_suffix('.dataset.json')} ({e})")


if __name__ == "__main__":  # pragma: no cover
    main()
