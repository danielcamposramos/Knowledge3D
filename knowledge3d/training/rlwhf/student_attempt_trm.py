#!/usr/bin/env python3
"""
TRM student attempts to answer generated questions (baseline inference pass).

Reads questions generated in Phase 1, runs the current TRM weights to
produce answer embeddings, and records convergence metrics that will be
used by the teacher evaluation and reward pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine
from knowledge3d.cranium.sovereign.trm_launcher import TRMLauncher
from knowledge3d.cranium.utils.trm import expand_embedding_to_trm


def sigmoid(x: float) -> float:
    """Smooth mapping from norm to confidence range [0, 1]."""
    return 1.0 / (1.0 + np.exp(-x))


def trm_attempt(
    question: str,
    rpn_engine: RPNEmbeddingEngine,
    trm: TRMLauncher,
    weights: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """
    Execute a single TRM reasoning pass for the given question.
    """
    q_emb_128 = rpn_engine.embed_sentence(question)
    q_emb_512 = expand_embedding_to_trm(q_emb_128)

    y = np.zeros(512, dtype=np.float32)
    z = np.zeros(512, dtype=np.float32)

    y_out, z_out = trm.refine(
        q_emb_512,
        y,
        z,
        weights["W1"],
        weights["W2"],
        weights["W3"],
        weights["W4"],
        n_steps=6,
    )

    output_norm = float(np.linalg.norm(y_out))
    confidence = float(sigmoid((output_norm - 50.0) / 50.0))
    converged = output_norm > 1.0

    return {
        "answer_embedding": y_out.tolist(),
        "latent_embedding": z_out.tolist(),
        "output_norm": output_norm,
        "confidence": confidence,
        "converged": converged,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True, help="Generated questions JSONL")
    parser.add_argument(
        "--rpn-embeddings",
        type=Path,
        default=Path("/K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl"),
    )
    parser.add_argument(
        "--trm-weights",
        type=Path,
        default=Path("/K3D/Knowledge3D.local/models/trm_weights_rpn_init.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/K3D/Knowledge3D.local/datasets/rlwhf/student_attempts.jsonl"),
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for smoke testing")
    args = parser.parse_args()

    print("=" * 70)
    print("K3D RLWHF Phase 2 — TRM Student Attempts")
    print("=" * 70)

    # Load resources
    print("\n📥 Loading RPN embeddings…")
    rpn_engine = RPNEmbeddingEngine()
    rpn_engine.load_embeddings(args.rpn_embeddings)
    print(f"   Trigrams loaded: {len(rpn_engine.embeddings):,}")

    print("\n🧠 Initialising TRM…")
    trm = TRMLauncher(use_fused=True)

    print("   Loading weights:", args.trm_weights)
    with np.load(args.trm_weights) as payload:
        weights = {key: payload[key].astype(np.float32) for key in ("W1", "W2", "W3", "W4")}

    # Iterate over questions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    converged = 0

    with args.questions.open("r", encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            if args.max_samples is not None and total >= args.max_samples:
                break

            record = json.loads(line)
            attempt = trm_attempt(record["question"], rpn_engine, trm, weights)

            output_record = {**record, "student_attempt": attempt}
            fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")

            total += 1
            if attempt["converged"]:
                converged += 1

            if total % 100 == 0:
                rate = (converged / total) * 100.0 if total else 0.0
                print(f"   Processed {total} samples (convergence={rate:.1f}%)")

    print("\n✅ Student inference complete")
    rate = (converged / total) * 100.0 if total else 0.0
    print(f"   Total samples: {total}")
    print(f"   Converged: {converged} ({rate:.1f}%)")
    print(f"   Output: {args.output}")


if __name__ == "__main__":
    main()
