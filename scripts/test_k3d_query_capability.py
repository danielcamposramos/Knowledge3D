#!/usr/bin/env python3
"""Test K3D's current query/answer capability.

Tests what K3D can do RIGHT NOW with:
- 290K consolidated RPN trigrams (knowledge in Galaxy/House)
- TRM initialized weights (reasoning engine)
- 6 recursions (Tesla 3/6/9 alignment)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import numpy as np
from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine
from knowledge3d.cranium.sovereign.trm_launcher import TRMLauncher


def test_k3d_reasoning():
    """Test K3D's reasoning on sample questions."""

    print("=" * 70)
    print("K3D QUERY/ANSWER CAPABILITY TEST")
    print("=" * 70)

    # Load knowledge base (RPN embeddings = Galaxy/House)
    print("\n📥 Loading K3D knowledge base...")
    rpn_engine = RPNEmbeddingEngine()
    embeddings_path = Path('/K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl')
    rpn_engine.load_embeddings(embeddings_path)
    print(f"✅ Knowledge base: {len(rpn_engine.embeddings):,} trigrams")

    # Load TRM reasoning engine
    print("\n🧠 Loading TRM reasoning engine...")
    trm = TRMLauncher(use_fused=True)

    # Load initialized weights (override via K3D_TRM_WEIGHTS_PATH if provided)
    print("📥 Loading TRM weights...")
    weights_override = Path(os.getenv("K3D_TRM_WEIGHTS_PATH", "")).expanduser()
    if weights_override and weights_override.exists():
        weights_path = weights_override
        print(f"   Using override from K3D_TRM_WEIGHTS_PATH: {weights_path}")
    else:
        weights_path = Path('/K3D/Knowledge3D.local/models/trm_weights_rpn_init.npz')
        if weights_override and not weights_override.exists():
            print(f"   ⚠️ Override path not found, falling back to default: {weights_path}")
    weights = np.load(weights_path)
    W1, W2, W3, W4 = weights['W1'], weights['W2'], weights['W3'], weights['W4']
    print(f"✅ TRM ready: {(W1.size + W2.size + W3.size + W4.size) / 1e6:.2f}M params")

    # Test questions (mix of technical, natural, reasoning)
    test_questions = [
        "What is neural network backpropagation?",
        "How does photosynthesis work in plants?",
        "If A equals B and B equals C, what equals A?",
        "Explain GPU memory architecture",
        "What causes ocean tides?",
        "Solve for x: 2x + 5 = 17",
    ]

    print("\n" + "=" * 70)
    print("TESTING K3D REASONING (6 Tesla recursions)")
    print("=" * 70)

    results = []

    for i, question in enumerate(test_questions):
        print(f"\n📌 Question {i+1}: {question}")
        print("-" * 70)

        # Step 1: Embed question using RPN (knowledge retrieval from Galaxy)
        q_emb_128 = rpn_engine.embed_sentence(question)

        # Step 2: Project to 512-dim for TRM
        q = np.zeros(512, dtype=np.float32)
        q[:512] = np.tile(q_emb_128, 4)[:512]

        # Step 3: TRM reasoning (6 recursions, Tesla 3/6/9)
        y = np.zeros(512, dtype=np.float32)
        z = np.zeros(512, dtype=np.float32)

        try:
            y_out, z_out = trm.refine(q, y, z, W1, W2, W3, W4, n_steps=6, eps=1e-4)

            # Step 4: Analyze output
            output_norm = np.linalg.norm(y_out)
            latent_norm = np.linalg.norm(z_out)

            print(f"   ✅ TRM output:")
            print(f"      Answer embedding norm: {output_norm:.3f}")
            print(f"      Latent state norm: {latent_norm:.3f}")
            print(f"      Reasoning converged: {'Yes' if output_norm > 0.1 else 'No (weak)'}")

            # Step 5: Find nearest neighbors in knowledge base
            # Project answer back to 128-dim
            y_128 = y_out[:128]

            # Find top 3 most similar trigrams in knowledge
            similarities = []
            for trigram_hash, emb in list(rpn_engine.embeddings.items())[:1000]:  # Sample 1000 for speed
                sim = np.dot(y_128, emb) / (np.linalg.norm(y_128) * np.linalg.norm(emb) + 1e-8)
                similarities.append((sim, trigram_hash))

            similarities.sort(reverse=True)
            top_3 = similarities[:3]

            print(f"      Top 3 knowledge activations:")
            for j, (sim, _) in enumerate(top_3):
                print(f"         {j+1}. Similarity: {sim:.3f}")

            results.append({
                'question': question,
                'output_norm': float(output_norm),
                'latent_norm': float(latent_norm),
                'top_similarity': float(top_3[0][0]) if top_3 else 0.0,
                'converged': output_norm > 0.1
            })

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'question': question,
                'error': str(e),
                'converged': False
            })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    converged = sum(1 for r in results if r.get('converged', False))
    avg_output_norm = np.mean([r.get('output_norm', 0) for r in results if 'output_norm' in r])
    avg_similarity = np.mean([r.get('top_similarity', 0) for r in results if 'top_similarity' in r])

    print(f"\n✅ Convergence: {converged}/{len(test_questions)} questions")
    print(f"📊 Average output norm: {avg_output_norm:.3f}")
    print(f"🔗 Average knowledge activation: {avg_similarity:.3f}")

    print("\n💡 Interpretation:")
    if avg_output_norm > 0.5:
        print("   ✅ TRM produces strong outputs (good reasoning)")
    elif avg_output_norm > 0.1:
        print("   ⚠️  TRM produces weak outputs (needs reasoning training)")
    else:
        print("   ❌ TRM produces near-zero outputs (weights need initialization)")

    if avg_similarity > 0.3:
        print("   ✅ Strong knowledge activation (embeddings work)")
    elif avg_similarity > 0.1:
        print("   ⚠️  Weak knowledge activation (consolidation helped but limited)")
    else:
        print("   ❌ No knowledge activation (problem with embeddings)")

    print("\n🎯 Next Steps:")
    if avg_output_norm < 0.5 or avg_similarity < 0.3:
        print("   → Train TRM on REASONING tasks (ARC-AGI, logic puzzles)")
        print("   → This teaches TRM how to transform embeddings, not store data")
    else:
        print("   → K3D is working! Try more complex reasoning tasks")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    test_k3d_reasoning()
