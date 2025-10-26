#!/usr/bin/env python3
"""
Construct RLWHF training dataset from teacher evaluations.

Converts teacher evaluations (questions + student attempts + teacher ratings)
into a training dataset with proper reward weighting.

Input:  teacher_evaluations.jsonl
Output: rlwhf_training_dataset.npz

Usage:
    PYTHONPATH=. python knowledge3d/training/rlwhf/construct_dataset.py
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter


def load_evaluations(filepath: str) -> List[Dict[str, Any]]:
    """Load all evaluations from JSONL file."""
    evaluations = []
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                evaluations.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Failed to parse line {line_num}: {e}")
                continue
    return evaluations


def parse_rating_from_response(teacher_response: str) -> Tuple[int, str]:
    """
    Parse rating from teacher response, fixing the text → numeric conversion bug.

    Returns:
        (rating_score, rating_text)
        rating_score: -2 (terrible/bad) to +2 (excellent/perfect)
        rating_text: The original text rating
    """
    # Try to extract rating from markdown format: **Rating:** <text>
    rating_match = re.search(r'\*\*Rating:\*\*\s*(\w+)', teacher_response, re.IGNORECASE)

    if not rating_match:
        # Fallback: try plain format: Rating: <text>
        rating_match = re.search(r'Rating:\s*(\w+)', teacher_response, re.IGNORECASE)

    if not rating_match:
        return (0, "unknown")

    rating_text = rating_match.group(1).lower()

    # Map text ratings to numeric scale
    rating_map = {
        # Very negative (-2)
        'terrible': -2,
        'bad': -2,
        'wrong': -2,
        'incorrect': -2,

        # Negative (-1)
        'poor': -1,
        'partial': -1,
        'incomplete': -1,
        'weak': -1,

        # Neutral (0)
        'neutral': 0,
        'okay': 0,
        'acceptable': 0,
        'fair': 0,

        # Positive (+1)
        'good': +1,
        'correct': +1,
        'right': +1,
        'solid': +1,

        # Very positive (+2)
        'excellent': +2,
        'perfect': +2,
        'outstanding': +2,
        'exceptional': +2,
    }

    rating_score = rating_map.get(rating_text, 0)
    return (rating_score, rating_text)


def extract_thinking_tags(teacher_response: str) -> List[str]:
    """Extract <think>...</think> segments from teacher response."""
    thinking_matches = re.findall(r'<think>(.*?)</think>', teacher_response, re.DOTALL)
    return [match.strip() for match in thinking_matches]


def construct_dataset(
    evaluations_path: str,
    output_path: str,
    min_rating: int = -2,
) -> Dict[str, Any]:
    """
    Construct training dataset from teacher evaluations.

    Args:
        evaluations_path: Path to teacher_evaluations.jsonl
        output_path: Path to save rlwhf_training_dataset.npz
        min_rating: Minimum rating to include (default: -2, include all)

    Returns:
        Dataset statistics
    """
    print("=" * 80)
    print("RLWHF Dataset Construction")
    print("=" * 80)
    print()

    # Load evaluations
    print(f"[1/5] Loading evaluations from {evaluations_path}...")
    evaluations = load_evaluations(evaluations_path)
    print(f"✓ Loaded {len(evaluations)} evaluations")
    print()

    # Filter and parse
    print("[2/5] Parsing ratings and filtering...")
    dataset_samples = []
    rating_counts = Counter()
    failed_count = 0

    for eval_data in evaluations:
        # Get teacher evaluation (nested structure)
        teacher_eval = eval_data.get('teacher_evaluation', {})
        teacher_response = teacher_eval.get('teacher_response', '')

        # Skip failed evaluations
        if not teacher_response or teacher_response.strip() == '':
            failed_count += 1
            continue

        # Parse rating
        rating_score, rating_text = parse_rating_from_response(teacher_response)

        # Skip if below minimum rating
        if rating_score < min_rating:
            continue

        rating_counts[rating_score] += 1

        # Extract thinking tags
        thinking_segments = extract_thinking_tags(teacher_response)

        # Get embeddings from student attempt
        student_attempt = eval_data.get('student_attempt', {})

        # Use pre-computed embeddings if available
        answer_embedding = student_attempt.get('answer_embedding')
        latent_embedding = student_attempt.get('latent_embedding')

        if answer_embedding is None or latent_embedding is None:
            # Skip if embeddings not available
            continue

        dataset_samples.append({
            'question': eval_data.get('question', ''),
            'answer': eval_data.get('answer', ''),
            'difficulty': eval_data.get('difficulty', 'unknown'),
            'source': eval_data.get('source', ''),
            'pdf_name': eval_data.get('pdf_name', ''),
            'rating_score': rating_score,
            'rating_text': rating_text,
            'answer_embedding': np.array(answer_embedding, dtype=np.float32),
            'latent_embedding': np.array(latent_embedding, dtype=np.float32),
            'thinking_segments': thinking_segments,
            'teacher_response': teacher_response,
        })

    print(f"✓ Parsed {len(dataset_samples)} valid samples")
    print(f"  Failed evaluations: {failed_count}")
    print()

    # Rating distribution
    print("Rating distribution:")
    total = len(dataset_samples)
    for rating in sorted(rating_counts.keys()):
        count = rating_counts[rating]
        pct = (count / total) * 100 if total > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"  {rating:+2d}: {count:5d} ({pct:5.1f}%) {bar}")
    print()

    if len(dataset_samples) == 0:
        print("❌ No valid samples found. Cannot construct dataset.")
        return {}

    # Convert to numpy arrays
    print("[3/5] Converting to numpy arrays...")
    questions = [s['question'] for s in dataset_samples]
    answers = [s['answer'] for s in dataset_samples]
    difficulties = [s['difficulty'] for s in dataset_samples]
    sources = [s['source'] for s in dataset_samples]
    ratings = np.array([s['rating_score'] for s in dataset_samples], dtype=np.int32)

    # Stack embeddings
    answer_embeddings = np.stack([s['answer_embedding'] for s in dataset_samples], axis=0)
    latent_embeddings = np.stack([s['latent_embedding'] for s in dataset_samples], axis=0)

    print(f"✓ Answer embeddings: {answer_embeddings.shape}")
    print(f"✓ Latent embeddings: {latent_embeddings.shape}")
    print(f"✓ Ratings: {ratings.shape}")
    print()

    # Calculate reward weights
    print("[4/5] Calculating reward weights...")
    # Normalize ratings to [0, 1] for weighting
    # -2 → 0.0, -1 → 0.25, 0 → 0.5, +1 → 0.75, +2 → 1.0
    reward_weights = (ratings + 2) / 4.0

    print(f"✓ Reward weights range: [{reward_weights.min():.2f}, {reward_weights.max():.2f}]")
    print(f"  Mean: {reward_weights.mean():.3f}")
    print(f"  Std:  {reward_weights.std():.3f}")
    print()

    # Save dataset
    print(f"[5/5] Saving dataset to {output_path}...")
    np.savez_compressed(
        output_path,
        # Embeddings
        answer_embeddings=answer_embeddings,
        latent_embeddings=latent_embeddings,

        # Ratings and rewards
        ratings=ratings,
        reward_weights=reward_weights,

        # Metadata (saved as object arrays)
        questions=np.array(questions, dtype=object),
        answers=np.array(answers, dtype=object),
        difficulties=np.array(difficulties, dtype=object),
        sources=np.array(sources, dtype=object),
    )
    print(f"✓ Dataset saved: {output_path}")
    print()

    # Statistics
    stats = {
        'total_evaluations': len(evaluations),
        'failed_evaluations': failed_count,
        'valid_samples': len(dataset_samples),
        'rating_distribution': dict(rating_counts),
        'answer_embedding_dim': answer_embeddings.shape[1],
        'latent_embedding_dim': latent_embeddings.shape[1],
        'reward_mean': float(reward_weights.mean()),
        'reward_std': float(reward_weights.std()),
    }

    print("=" * 80)
    print("Dataset Construction Complete")
    print("=" * 80)
    print()
    print(f"Total samples: {stats['valid_samples']}")
    print(f"Answer embedding dim: {stats['answer_embedding_dim']}")
    print(f"Latent embedding dim: {stats['latent_embedding_dim']}")
    print(f"Reward mean: {stats['reward_mean']:.3f} ± {stats['reward_std']:.3f}")
    print()

    # Quality assessment
    positive_ratio = sum(1 for r in ratings if r > 0) / len(ratings) if len(ratings) > 0 else 0
    negative_ratio = sum(1 for r in ratings if r < 0) / len(ratings) if len(ratings) > 0 else 0

    print("Quality assessment:")
    print(f"  Positive samples: {positive_ratio*100:.1f}%")
    print(f"  Negative samples: {negative_ratio*100:.1f}%")

    if negative_ratio < 0.1:
        print("  ⚠️  Low negative sample ratio - training may lack error correction")
    elif negative_ratio > 0.5:
        print("  ⚠️  High negative sample ratio - model may need more diverse data")
    else:
        print("  ✓ Good balance of positive and negative examples")

    print()

    return stats


def main():
    evaluations_path = "/K3D/Knowledge3D.local/datasets/rlwhf/teacher_evaluations.jsonl"
    output_path = "/K3D/Knowledge3D.local/datasets/rlwhf/rlwhf_training_dataset.npz"

    stats = construct_dataset(evaluations_path, output_path)

    if stats:
        print("✓ RLWHF training dataset ready!")
        print(f"  Load with: np.load('{output_path}', allow_pickle=True)")
    else:
        print("❌ Dataset construction failed")


if __name__ == "__main__":
    main()
