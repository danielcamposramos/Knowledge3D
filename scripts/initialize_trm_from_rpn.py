#!/usr/bin/env python3
"""Initialize TRM weights from consolidated RPN embeddings.

Strategy:
- Extract top 1024 most frequent trigrams from RPN
- Use trigram embeddings to seed TRM weight matrices
- Xavier initialization for remaining dimensions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine


def initialize_trm_weights_from_rpn():
    """Initialize TRM weights using RPN trigram embeddings."""

    print("=" * 70)
    print("TRM WEIGHT INITIALIZATION FROM RPN EMBEDDINGS")
    print("=" * 70)

    # Load consolidated RPN embeddings
    rpn_engine = RPNEmbeddingEngine()
    embeddings_path = Path('/K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl')

    print(f"\n📥 Loading RPN embeddings from: {embeddings_path}")
    rpn_engine.load_embeddings(embeddings_path)
    print(f"✅ Loaded {len(rpn_engine.embeddings):,} trigrams")

    # Get trigrams sorted by embedding norm (proxy for importance)
    # Higher norm = more frequent/important trigrams
    trigram_items = list(rpn_engine.embeddings.items())
    trigram_items.sort(key=lambda x: np.linalg.norm(x[1]), reverse=True)

    # Take top 1024 trigrams
    top_trigrams = trigram_items[:1024]
    trigram_matrix = np.vstack([emb for _, emb in top_trigrams])  # (1024, 128)

    print(f"\n📊 Selected top {len(top_trigrams)} trigrams by importance")
    print(f"   Trigram embedding dim: {trigram_matrix.shape[1]}")
    print(f"   Average norm: {np.mean([np.linalg.norm(emb) for _, emb in top_trigrams]):.3f}")

    # TRM architecture: 512 → 1024 (W1, SwiGLU) → 512 (W2)
    # Repeated for z-update and y-update
    print("\n🔧 Initializing TRM weight matrices...")

    # W1: (1024, 512) - Input to hidden
    # Strategy: Seed first 128 dims with trigram embeddings, Xavier for rest
    W1 = np.random.randn(1024, 512).astype(np.float32) * np.sqrt(2.0 / 512)
    # Embed trigram patterns in first 128 input dimensions
    W1[:1024, :128] = trigram_matrix
    print(f"✅ W1: {W1.shape} (seeded first 128 dims with trigrams)")

    # W2: (512, 1024) - Hidden to output
    W2 = np.random.randn(512, 1024).astype(np.float32) * np.sqrt(2.0 / 1024)
    print(f"✅ W2: {W2.shape} (Xavier init)")

    # W3: (1024, 512) - Answer update input to hidden
    W3 = np.random.randn(1024, 512).astype(np.float32) * np.sqrt(2.0 / 512)
    W3[:1024, :128] = trigram_matrix  # Same seeding
    print(f"✅ W3: {W3.shape} (seeded first 128 dims with trigrams)")

    # W4: (512, 1024) - Answer update hidden to output
    W4 = np.random.randn(512, 1024).astype(np.float32) * np.sqrt(2.0 / 1024)
    print(f"✅ W4: {W4.shape} (Xavier init)")

    total_params = W1.size + W2.size + W3.size + W4.size
    print(f"\n📈 Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    # Save weights
    output_dir = Path('/K3D/Knowledge3D.local/models')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'trm_weights_rpn_init.npz'

    np.savez(
        output_path,
        W1=W1, W2=W2, W3=W3, W4=W4,
        trigram_indices=[hash_val for hash_val, _ in top_trigrams],
        init_method='rpn_trigram_seeded',
        timestamp=np.datetime64('now')
    )

    print(f"\n💾 Saved initialized weights to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    print("\n" + "=" * 70)
    print("✅ TRM WEIGHT INITIALIZATION COMPLETE")
    print("=" * 70)
    print("\nWeights are ready for training on K3D knowledge!")

    return output_path


if __name__ == '__main__':
    initialize_trm_weights_from_rpn()
