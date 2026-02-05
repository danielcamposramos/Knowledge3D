#!/usr/bin/env python3
"""
Convert PyTorch V7 checkpoint to Sovereign TRM format.

This script converts a PyTorch checkpoint (.pt file) to NumPy arrays (.npy)
that can be loaded by SovereignTRM, plus a metadata.json file for inference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional


def _resolve_key(state_dict: Dict[str, Any], name: str) -> Optional[str]:
    if name in state_dict:
        return name
    candidates = [key for key in state_dict.keys() if key.endswith(name)]
    if not candidates:
        return None
    candidates.sort(key=len)
    return candidates[0]


def convert_checkpoint(input_path: str, output_dir: str, verbose: bool = True) -> None:
    """Convert PyTorch checkpoint to NumPy arrays.

    Args:
        input_path: Path to PyTorch .pt checkpoint
        output_dir: Directory to save .npy weight files
        verbose: Print conversion progress
    """
    try:
        import torch  # type: ignore
    except ImportError:
        print("ERROR: PyTorch not found. Install with: pip install torch", file=sys.stderr)
        sys.exit(1)

    try:
        import numpy as np  # type: ignore
    except ImportError:
        print("ERROR: NumPy not found. Install with: pip install numpy", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"Loading PyTorch checkpoint: {input_path}")

    try:
        checkpoint = torch.load(input_path, map_location="cpu")
    except FileNotFoundError:
        print(f"ERROR: Checkpoint not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: Failed to load checkpoint: {exc}", file=sys.stderr)
        sys.exit(1)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        print("ERROR: Unrecognized checkpoint format", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        print(f"Output directory: {output_dir}")

    weight_map = {
        "embedding.weight": "embedding.npy",
        "lstm.weight_ih_l0": "lstm_weight_ih.npy",
        "lstm.weight_hh_l0": "lstm_weight_hh.npy",
        "lstm.bias_ih_l0": "lstm_bias_ih.npy",
        "lstm.bias_hh_l0": "lstm_bias_hh.npy",
        "rule_head.weight": "rule_head_weight.npy",
        "rule_head.bias": "rule_head_bias.npy",
        "confidence_head.0.weight": "confidence_head_0_weight.npy",
        "confidence_head.0.bias": "confidence_head_0_bias.npy",
        "confidence_head.2.weight": "confidence_head_2_weight.npy",
        "confidence_head.2.bias": "confidence_head_2_bias.npy",
    }

    converted_count = 0
    converted_arrays: Dict[str, Any] = {}
    for pt_name, npy_name in weight_map.items():
        key = _resolve_key(state_dict, pt_name)
        if key is None:
            print(f"WARNING: {pt_name} not found in checkpoint (skipping)", file=sys.stderr)
            continue

        tensor = state_dict[key]
        array = tensor.detach().cpu().numpy().astype(np.float32)
        output_path = os.path.join(output_dir, npy_name)
        np.save(output_path, array)
        converted_arrays[npy_name] = array

        if verbose:
            print(f"  {key:30s} -> {npy_name:30s} (shape: {array.shape})")
        converted_count += 1

    if converted_count == 0:
        print("ERROR: No weights converted! Check checkpoint format.", file=sys.stderr)
        sys.exit(1)

    embedding_dim = int(checkpoint.get("embedding_dim", 0)) if isinstance(checkpoint, dict) else 0
    hidden_dim = int(checkpoint.get("hidden_dim", 0)) if isinstance(checkpoint, dict) else 0
    vocab_size = int(checkpoint.get("vocab_size", 0)) if isinstance(checkpoint, dict) else 0
    base_vocab_size = int(checkpoint.get("base_vocab_size", 0)) if isinstance(checkpoint, dict) else 0
    if embedding_dim == 0 and "embedding.npy" in converted_arrays:
        embedding_dim = int(converted_arrays["embedding.npy"].shape[1])
    if vocab_size == 0 and "embedding.npy" in converted_arrays:
        vocab_size = int(converted_arrays["embedding.npy"].shape[0])
    if hidden_dim == 0 and "rule_head_weight.npy" in converted_arrays:
        hidden_dim = int(converted_arrays["rule_head_weight.npy"].shape[1])
    if base_vocab_size == 0 and isinstance(checkpoint, dict):
        base_vocab_size = int(checkpoint.get("base_vocab_size", vocab_size))

    metadata = {
        "format": "sovereign_trm_v1",
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "base_vocab_size": base_vocab_size,
        "rule_registry": list(checkpoint.get("rule_registry") or []) if isinstance(checkpoint, dict) else [],
        "control_tokens": bool(checkpoint.get("control_tokens", False)) if isinstance(checkpoint, dict) else False,
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)

    if verbose:
        print(f"\nOK: Converted {converted_count}/{len(weight_map)} weights.")
        print(f"Metadata: {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PyTorch V7 checkpoint to Sovereign TRM format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/convert_v7_to_sovereign.py \\
      --input checkpoints/v7.pt \\
      --output checkpoints/v7_sovereign/
""",
    )
    parser.add_argument("--input", required=True, help="Input PyTorch checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Output directory for .npy files")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    convert_checkpoint(args.input, args.output, verbose=not args.quiet)


if __name__ == "__main__":
    main()
