#!/usr/bin/env python3
"""Analyse atomic classifier logits to suggest threshold values."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from scripts import train_atomic_character as atomic


def load_classifier(char: str) -> tuple[np.ndarray, float]:
    weights_path = Path(
        f"/K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars/char_{ord(char)}_{char}_weights.npz"
    )
    data = np.load(weights_path)
    fc_weight = np.asarray(data.get("fc_weight"), dtype=np.float32)
    fc_bias = np.asarray(data.get("fc_bias"), dtype=np.float32)

    if fc_weight.ndim != 2 or fc_weight.shape[0] != 2:
        raise ValueError(f"Unexpected FC weight shape for '{char}': {fc_weight.shape}")

    weight_vec = fc_weight[1] - fc_weight[0]
    if fc_bias.size >= 2:
        bias_val = float(fc_bias[1] - fc_bias[0])
    elif fc_bias.size == 1:
        bias_val = float(fc_bias[0])
    else:
        bias_val = 0.0

    return weight_vec.astype(np.float32), bias_val


def build_dataset(char: str, fonts_per_script: int = 10) -> dict:
    script = atomic.get_character_script(char)
    fonts = atomic.load_fonts_for_script(script, fonts_per_script)
    if not fonts:
        raise RuntimeError(f"No fonts available for script {script}")

    negative_groups = atomic._prepare_negative_groups(char, script)
    font_buckets = atomic._prepare_font_buckets(negative_groups, script, fonts, fonts_per_script)
    dataset = atomic._build_dataset(
        target_char=char,
        target_script=script,
        positive_fonts=fonts,
        negative_groups=negative_groups,
        font_buckets=font_buckets,
    )
    return dataset


def collect_logits(char: str, model, fonts_per_script: int) -> dict:
    dataset = build_dataset(char, fonts_per_script)
    images = dataset["images"]
    labels = dataset["labels"]

    weight_vec, bias_val = load_classifier(char)

    logits_pos: List[float] = []
    logits_neg: List[float] = []

    for image, label in zip(images, labels):
        result = model.forward(image, cache_for_backward=True)
        feature_map = result.get("feature_map")
        if feature_map is None:
            continue
        pooled = feature_map.mean(axis=(0, 1)).astype(np.float32)

        logit = float(np.dot(pooled, weight_vec) + bias_val)
        if label == 1:
            logits_pos.append(logit)
        else:
            logits_neg.append(logit)

    return {
        "char": char,
        "logits_pos": np.array(logits_pos, dtype=np.float32),
        "logits_neg": np.array(logits_neg, dtype=np.float32),
    }


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def summarize(logit_info: dict, percentile: float) -> dict:
    pos = sigmoid(logit_info["logits_pos"])
    neg = sigmoid(logit_info["logits_neg"])
    pos_threshold = np.percentile(pos, percentile)
    neg_threshold = np.percentile(neg, 100 - percentile)
    return {
        "char": logit_info["char"],
        "pos_mean": float(pos.mean()),
        "neg_mean": float(neg.mean()),
        "pos_pct": float(pos_threshold),
        "neg_pct": float(neg_threshold),
        "count_pos": pos.size,
        "count_neg": neg.size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse atomic classifier logits to suggest thresholds")
    parser.add_argument("--fonts", type=int, default=10, help="Fonts per script for sampling")
    parser.add_argument("--percentile", type=float, default=5.0, help="Percentile for positive tail")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of characters to analyse")
    args = parser.parse_args()

    characters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
    if args.limit > 0:
        characters = characters[: args.limit]

    model = atomic.DeepSeekOCRModel()

    summaries: List[dict] = []
    for char in characters:
        info = collect_logits(char, model, args.fonts)
        summary = summarize(info, args.percentile)
        summaries.append(summary)
        print(
            f"{char}: pos_mean={summary['pos_mean']:.4f}, neg_mean={summary['neg_mean']:.4f}, "
            f"pos_{args.percentile:.0f}%={summary['pos_pct']:.4f}, "
            f"neg_{100-args.percentile:.0f}%={summary['neg_pct']:.4f}"
        )

    pos_thresholds = [s["pos_pct"] for s in summaries if s["count_pos"] > 0]
    neg_thresholds = [s["neg_pct"] for s in summaries if s["count_neg"] > 0]
    if pos_thresholds and neg_thresholds:
        suggested = float((np.mean(pos_thresholds) + np.mean(neg_thresholds)) / 2.0)
        print(
            f"\nSuggested global probability threshold ≈ {suggested:.4f} "
            f"(pos_{args.percentile}% mean {np.mean(pos_thresholds):.4f}, neg tail {np.mean(neg_thresholds):.4f})"
        )


if __name__ == "__main__":
    main()
