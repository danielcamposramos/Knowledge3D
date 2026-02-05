#!/usr/bin/env python3
"""
Run the Sleep Keeper specialist over Log Galaxy traces and populate SleepGalaxy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch

from knowledge3d.training.math_benchmarks.router_embedder import embed_text
from knowledge3d.training.math_benchmarks.sleep_galaxy import SleepGalaxy


LABELS = {0: "discard", 1: "keep", 2: "uncertain"}


def _iter_log_entries(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _load_model(checkpoint: str) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu")
    embedding_dim = int(payload.get("embedding_dim", 384))
    hidden_dim = int(payload.get("hidden_dim", 128))

    model = torch.nn.Sequential(
        torch.nn.Linear(embedding_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 3),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def _is_noble_failure(
    entry: Dict[str, Any],
    *,
    min_steps: int,
    max_mismatches: int,
) -> bool:
    if bool(entry.get("success", False)):
        return False
    meta = entry.get("metadata") or {}
    policy_mode = str(meta.get("policy_mode", entry.get("policy_mode", "heuristic")))
    if policy_mode not in {"neural", "mixed"}:
        return False
    mismatches = int(meta.get("policy_mismatches", entry.get("policy_mismatches", 0)))
    steps = entry.get("step_sequence") or []
    if len(steps) < int(min_steps):
        return False
    if mismatches > int(max_mismatches):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sleep Keeper specialist on Log Galaxy.")
    parser.add_argument("--input", required=True, help="Log Galaxy JSONL path.")
    parser.add_argument("--model", required=True, help="Sleep specialist checkpoint.")
    parser.add_argument("--output", required=True, help="Output SleepGalaxy JSONL path.")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="Confidence threshold for keep/discard.")
    parser.add_argument("--dust-out", default=None, help="Optional JSONL path for discarded entries.")
    parser.add_argument("--noble-min-steps", type=int, default=3, help="Minimum steps for noble failures.")
    parser.add_argument("--noble-max-mismatches", type=int, default=1, help="Max mismatches for noble failures.")
    parser.add_argument("--no-noble-override", action="store_true", help="Disable noble failure override.")
    args = parser.parse_args()

    model, payload = _load_model(args.model)
    embedding_dim = int(payload.get("embedding_dim", 384))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    sleep_galaxy = SleepGalaxy(embedding_dim=embedding_dim)
    dust_handle = None
    if args.dust_out:
        dust_path = Path(args.dust_out)
        dust_path.parent.mkdir(parents=True, exist_ok=True)
        dust_handle = dust_path.open("w", encoding="utf-8")

    total = 0
    counts = {"keep": 0, "discard": 0, "uncertain": 0}

    for entry in _iter_log_entries(args.input):
        text = str(entry.get("problem_text", "")).strip()
        if not text:
            continue
        embedding = embed_text(text, dim=embedding_dim)
        emb = torch.tensor(embedding, dtype=torch.float32, device=device).unsqueeze(0)
        logits = model(emb)
        probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
        pred_idx = int(torch.argmax(logits, dim=-1).item())
        pred_label = LABELS.get(pred_idx, "uncertain")
        confidence = float(max(probs)) if probs else 0.0
        if confidence < float(args.min_confidence):
            pred_label = "uncertain"

        noble_failure = _is_noble_failure(
            entry,
            min_steps=int(args.noble_min_steps),
            max_mismatches=int(args.noble_max_mismatches),
        )

        metadata = dict(entry.get("metadata") or {})
        metadata["policy_mode"] = metadata.get("policy_mode", entry.get("policy_mode"))
        metadata["policy_mismatches"] = metadata.get("policy_mismatches", entry.get("policy_mismatches", 0))
        metadata["confidence"] = confidence
        metadata["probs"] = probs
        if noble_failure:
            metadata["negative_wisdom"] = True
            metadata["noble_failure"] = True

        if noble_failure and not args.no_noble_override and pred_label != "keep":
            metadata["override"] = "noble_failure"
            pred_label = "keep"

        sleep_galaxy.add_entry(
            trace_id=entry.get("trace_id", ""),
            problem_text=text,
            decision=pred_label,
            decision_score=confidence,
            metadata=metadata,
        )

        total += 1
        counts[pred_label] += 1
        if dust_handle and pred_label == "discard":
            dust_handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    sleep_galaxy.to_jsonl(args.output)
    if dust_handle:
        dust_handle.close()

    print(f"[SleepKeeper] Entries: {total}")
    for label in ("keep", "discard", "uncertain"):
        print(f"  {label}: {counts[label]}")
    print(f"[SleepKeeper] Saved: {args.output}")


if __name__ == "__main__":
    main()
