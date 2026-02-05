#!/usr/bin/env python3
"""
Train NavigationModelWithConfidence using verification dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader

from knowledge3d.training.math_benchmarks.calibration_loss import compute_training_loss
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train navigation model with confidence head.")
    parser.add_argument(
        "--checkpoint",
        default="/K3D/Knowledge3D.local/checkpoints/navigation_specialist_v5_wake.pt",
        help="Base navigation checkpoint (for registry/dims).",
    )
    parser.add_argument(
        "--dataset",
        default="data/verification_train_v1.jsonl",
        help="Verification dataset JSONL.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--confidence-weight", type=float, default=0.3, help="Confidence loss weight.")
    parser.add_argument("--vocab-size", type=int, default=256, help="Input token vocabulary size.")
    parser.add_argument("--embedding-dim", type=int, default=256, help="Input embedding dimension.")
    parser.add_argument("--hidden-dim", type=int, default=512, help="LSTM hidden dimension.")
    parser.add_argument("--max-problem-tokens", type=int, default=256, help="Max tokens from problem text.")
    parser.add_argument("--use-ece", action="store_true", help="Use ECE instead of MSE.")
    parser.add_argument("--limit", type=int, default=0, help="Limit samples (0 = all).")
    parser.add_argument(
        "--checkpoint-dir",
        default="/K3D/Knowledge3D.local/checkpoints",
        help="Directory to save checkpoint outputs.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output checkpoint path.",
    )
    args = parser.parse_args()

    base_ckpt = torch.load(args.checkpoint, map_location="cpu")
    embedding_dim = int(args.embedding_dim)
    hidden_dim = int(args.hidden_dim)
    vocab_size = int(args.vocab_size)
    registry = base_ckpt.get("rule_registry") or []
    registry_map = {rule: idx for idx, rule in enumerate(registry)}
    if not registry:
        raise SystemExit("Rule registry is empty in base checkpoint.")

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

    model = NavigationModelWithConfidence(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        base_vocab_size=len(registry),
        vocab_size=vocab_size,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    loader = DataLoader(
        samples,
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=lambda batch: _collate(batch, device),
    )

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_rule = 0.0
        total_cal = 0.0
        for input_tokens, target_tokens, conf_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            rule_logits, confidence = model(input_tokens)
            loss, rule_loss, cal_loss = compute_training_loss(
                rule_logits,
                target_tokens,
                confidence,
                conf_labels,
                confidence_weight=float(args.confidence_weight),
                pad_id=PAD_ID,
                use_ece=bool(args.use_ece),
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_rule += float(rule_loss.item())
            total_cal += float(cal_loss.item())

        if epoch == 1 or epoch % 10 == 0 or epoch == int(args.epochs):
            steps = max(1, len(loader))
            print(
                f"[Epoch {epoch:03d}] total={total_loss/steps:.4f} "
                f"rule={total_rule/steps:.4f} cal={total_cal/steps:.4f}"
            )

    output = Path(args.output) if args.output else Path(args.checkpoint_dir) / "navigation_specialist_v7_lstm_confidence_final.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "vocab_size": vocab_size,
            "base_vocab_size": len(registry),
            "rule_registry": registry,
            "pad_id": PAD_ID,
            "bos_id": BOS_ID,
            "rule_offset": RULE_OFFSET,
            "control_tokens": False,
        },
        output,
    )
    print(f"[Saved] {output}")


if __name__ == "__main__":
    main()
