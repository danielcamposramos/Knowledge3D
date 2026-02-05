#!/usr/bin/env python3
"""
Evaluate navigation model with confidence head on a held-out slice.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader

from knowledge3d.training.math_benchmarks.calibration_loss import expected_calibration_error
from knowledge3d.training.math_benchmarks.navigation_model import PAD_ID, BOS_ID, RULE_OFFSET
from knowledge3d.training.math_benchmarks.navigation_model_with_confidence import (
    NavigationModelWithConfidence,
)


def _iter_jsonl(path: str) -> Iterable[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _tokenize_problem(text: str, vocab_size: int, max_problem_tokens: int) -> List[int]:
    if not text:
        return [BOS_ID]
    tokens = [BOS_ID]
    for byte in text.encode("utf-8", errors="ignore")[:max_problem_tokens]:
        token = int(byte)
        if token >= vocab_size:
            token %= vocab_size
        tokens.append(token)
    return tokens


def _extract_rules(entry: Dict[str, object]) -> Tuple[List[str], List[int]]:
    rules = entry.get("predicted_rules") or []
    labels = entry.get("correctness_labels") or []
    if rules and labels and len(rules) == len(labels):
        return list(rules), [int(l) for l in labels]

    step_sequence = entry.get("step_sequence") or []
    seq_rules: List[str] = []
    for step in step_sequence:
        if isinstance(step, dict):
            rule = step.get("rule")
            if rule:
                seq_rules.append(str(rule))
    if not seq_rules:
        return [], []
    labels = [1] * len(seq_rules)
    return seq_rules, labels


def _load_samples(
    path: str,
    *,
    registry_map: Dict[str, int],
    vocab_size: int,
    max_problem_tokens: int,
    limit: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for entry in _iter_jsonl(path):
        if limit and len(samples) >= limit:
            break
        problem_text = str(entry.get("problem_text") or "").strip()
        if not problem_text:
            continue
        rules, labels = _extract_rules(entry)
        if not rules or not labels:
            continue
        if len(rules) != len(labels):
            continue
        rule_ids: List[int] = []
        conf_labels: List[int] = []
        for rule, label in zip(rules, labels):
            if rule not in registry_map:
                continue
            rule_ids.append(int(registry_map[rule]))
            conf_labels.append(int(label))
        if not rule_ids:
            continue
        problem_tokens = _tokenize_problem(problem_text, vocab_size, max_problem_tokens)
        rule_tokens = [RULE_OFFSET + rid for rid in rule_ids]
        decode_inputs = [BOS_ID] + rule_tokens[:-1]
        input_tokens = problem_tokens + decode_inputs
        target_tokens = [PAD_ID] * len(problem_tokens) + rule_tokens
        confidence_labels = [-1] * len(problem_tokens) + conf_labels
        samples.append(
            (
                torch.tensor(input_tokens, dtype=torch.long),
                torch.tensor(target_tokens, dtype=torch.long),
                torch.tensor(confidence_labels, dtype=torch.long),
            )
        )
    return samples


def _collate(batch, device: torch.device):
    input_seqs = [item[0] for item in batch]
    target_seqs = [item[1] for item in batch]
    conf_seqs = [item[2] for item in batch]
    max_len = max(len(seq) for seq in input_seqs)

    input_tokens = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long, device=device)
    target_tokens = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long, device=device)
    confidence_labels = torch.full((len(batch), max_len), -1, dtype=torch.float32, device=device)

    for i, seq in enumerate(input_seqs):
        input_tokens[i, : len(seq)] = seq.to(device)
        target = target_seqs[i].to(device)
        target_tokens[i, : len(target)] = target
        conf = conf_seqs[i].to(device).float()
        confidence_labels[i, : len(conf)] = conf

    return input_tokens, target_tokens, confidence_labels


def _split_holdout(samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], frac: float, seed: int):
    if not samples:
        return []
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    if frac <= 0:
        return shuffled
    holdout_count = max(1, int(len(shuffled) * frac))
    return shuffled[:holdout_count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate navigation confidence model on held-out data.")
    parser.add_argument(
        "--checkpoint",
        default="/K3D/Knowledge3D.local/checkpoints/navigation_specialist_v7_lstm_confidence_final.pt",
        help="Navigation checkpoint with confidence head.",
    )
    parser.add_argument(
        "--dataset",
        default="data/verification_train_v1.jsonl",
        help="Verification dataset JSONL.",
    )
    parser.add_argument("--holdout-frac", type=float, default=0.2, help="Held-out fraction (0-1).")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for holdout split.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--limit", type=int, default=0, help="Limit samples (0 = all).")
    parser.add_argument("--ece-bins", type=int, default=10, help="Bins for ECE computation.")
    parser.add_argument("--max-problem-tokens", type=int, default=256, help="Max tokens from problem text.")
    args = parser.parse_args()

    base_ckpt = torch.load(args.checkpoint, map_location="cpu")
    hidden_dim = int(base_ckpt.get("hidden_dim", 0))
    vocab_size = int(base_ckpt.get("vocab_size", 0))
    base_vocab_size = int(base_ckpt.get("base_vocab_size", vocab_size))
    registry = base_ckpt.get("rule_registry") or []
    registry_map = {rule: idx for idx, rule in enumerate(registry)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = _load_samples(
        args.dataset,
        registry_map=registry_map,
        vocab_size=vocab_size,
        max_problem_tokens=int(args.max_problem_tokens),
        limit=int(args.limit),
    )
    if not samples:
        raise SystemExit("No samples loaded from verification dataset.")

    holdout = _split_holdout(samples, float(args.holdout_frac), int(args.seed))
    if not holdout:
        raise SystemExit("Holdout split is empty.")

    model = NavigationModelWithConfidence(
        embedding_dim=int(base_ckpt.get("embedding_dim", 0)) or 256,
        hidden_dim=hidden_dim,
        base_vocab_size=base_vocab_size,
        vocab_size=vocab_size,
    ).to(device)
    state = base_ckpt.get("model_state") or base_ckpt
    model.load_state_dict(state)
    model.eval()

    loader = DataLoader(
        holdout,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=lambda batch: _collate(batch, device),
    )

    total_tokens = 0
    correct_tokens = 0
    conf_values: List[torch.Tensor] = []
    conf_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for input_tokens, target_tokens, confidence_labels in loader:
            logits, confidence = model(input_tokens)
            preds = logits.argmax(dim=-1)
            mask = target_tokens != PAD_ID
            correct = (preds == target_tokens) & mask
            correct_tokens += int(correct.sum().item())
            total_tokens += int(mask.sum().item())

            conf_mask = (confidence_labels >= 0) & mask
            if conf_mask.any():
                conf_values.append(confidence.squeeze(-1)[conf_mask].detach().cpu())
                conf_labels.append(confidence_labels[conf_mask].detach().cpu())

    accuracy = (correct_tokens / total_tokens) if total_tokens else 0.0
    if conf_values:
        all_conf = torch.cat(conf_values, dim=0)
        all_labels = torch.cat(conf_labels, dim=0)
        ece = expected_calibration_error(all_conf, all_labels, num_bins=int(args.ece_bins))
        ece_value = float(ece.item())
    else:
        ece_value = 0.0

    print("[Eval] Samples:", len(holdout))
    print("[Eval] Tokens:", total_tokens)
    print(f"[Eval] Rule Accuracy: {accuracy:.4f}")
    print(f"[Eval] ECE: {ece_value:.4f}")


if __name__ == "__main__":
    main()
