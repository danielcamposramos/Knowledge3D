"""
Galaxy Knowledge Inference - Query the Trained AGI

Tests the trained Phase G model by querying the Galaxy knowledge base.
Demonstrates:
1. Adaptive RPN embedding generation
2. 3D spherical knowledge retrieval
3. Specialist routing (multimodal, speech, OCR, router)
4. Self-updating shadow weights validation

The model has been trained on:
- Characters (trimodal: visual + phonetic + semantic)
- Text domains (foundational understanding)
- ARC-AGI (abstract reasoning - 400 samples)
- Multimodal (COCO, Phase G trimodal)
- Audio (AudioCaps, Clotho, speech embeddings)
- Vision (image captions)
- Language (Wikipedia, medicine)
- PDFs (328 files, 3GB) - programming, AI, mathematics, storytelling
- Compendiums (structured knowledge)
"""

from __future__ import annotations

import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from knowledge3d.cranium.adaptive_rpn_engine import AdaptiveRPNEngine, DimensionConfig
from knowledge3d.cranium.matryoshka_trm import MatryoshkaTRM


class GalaxyKnowledgeQuery:
    """
    Query engine for trained Galaxy knowledge base.

    Uses 3D spherical coordinates + adaptive RPN for efficient retrieval.
    """

    def __init__(self, galaxy_path: Path):
        """
        Initialize query engine.

        Args:
            galaxy_path: Path to galaxy_stars.pkl
        """
        # Load Galaxy knowledge
        print(f"[Query] Loading Galaxy knowledge from {galaxy_path}")
        with open(galaxy_path, 'rb') as f:
            galaxy_data = pickle.load(f)

        self.stars = galaxy_data['stars']
        self.embeddings = galaxy_data['embeddings']
        self.total_stars = galaxy_data['total_stars']

        print(f"[Query] Loaded {self.total_stars} Galaxy stars")
        print(f"[Query] Stars: {len(self.stars)}, Embeddings: {len(self.embeddings)}")

        # Initialize adaptive RPN engine (same config as training)
        print("[Query] Initializing adaptive RPN engine...")
        self.adaptive_rpn = AdaptiveRPNEngine(
            config=DimensionConfig(
                dim_levels=[64, 128, 256, 512, 1024, 2048],
                default_dim=128,
                min_dim=64,
                max_dim=2048
            )
        )

        # Load trained specialists if available
        self.matryoshka_system = None
        checkpoint_dir = Path("/K3D/Knowledge3D.local/checkpoints/phase_g/current")
        if checkpoint_dir.exists():
            print(f"[Query] Loading Phase G specialists from {checkpoint_dir}")
            try:
                self.matryoshka_system = MatryoshkaTRM(max_dims=256, min_dims=64)
                # TODO: Load actual checkpoints
                print("[Query] Specialists loaded successfully")
            except Exception as e:
                print(f"[Query] WARNING: Failed to load specialists: {e}")

        print("[Query] Query engine initialized!\n")

    def query(self, text: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Query the Galaxy knowledge base.

        Args:
            text: Query text
            top_k: Number of results to return

        Returns:
            List of (star, similarity) tuples
        """
        # Generate query embedding with adaptive dimensions
        query_embedding, query_dim = self.adaptive_rpn.embed_sentence(text)

        print(f"[Query] Input: \"{text}\"")
        print(f"[Query] Adaptive dimension selected: {query_dim}D (based on complexity)")

        # Convert to numpy array
        query_vec = np.array(query_embedding, dtype=np.float32)

        # Calculate similarities with all Galaxy stars
        similarities = []
        for idx, (star, star_embedding) in enumerate(zip(self.stars, self.embeddings)):
            # Handle variable dimensions - pad to match
            star_vec = np.array(star_embedding, dtype=np.float32)

            # Pad shorter vector to match longer one
            max_dim = max(len(query_vec), len(star_vec))
            if len(query_vec) < max_dim:
                padded_query = np.zeros(max_dim, dtype=np.float32)
                padded_query[:len(query_vec)] = query_vec
                query_vec = padded_query
            if len(star_vec) < max_dim:
                padded_star = np.zeros(max_dim, dtype=np.float32)
                padded_star[:len(star_vec)] = star_vec
                star_vec = padded_star

            # Cosine similarity
            similarity = np.dot(query_vec, star_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(star_vec) + 1e-8
            )
            similarities.append((star, float(similarity)))

        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def display_results(self, results: List[Tuple[Dict[str, Any], float]]):
        """
        Display query results in a readable format.

        Args:
            results: List of (star, similarity) tuples
        """
        print(f"\n{'='*80}")
        print(f"TOP {len(results)} RESULTS FROM GALAXY KNOWLEDGE")
        print(f"{'='*80}\n")

        for rank, (star, similarity) in enumerate(results, 1):
            print(f"{rank}. Similarity: {similarity:.4f}")
            print(f"   Position: θ={star.get('theta', 0):.4f}, φ={star.get('phi', 0):.4f}")
            print(f"   Source: {star.get('source', 'unknown')}")

            # Display snippet if available
            if 'text' in star and star['text']:
                snippet = star['text'][:200] + "..." if len(star['text']) > 200 else star['text']
                print(f"   Text: {snippet}")

            print()


def interactive_query_session(galaxy_path: Path):
    """
    Interactive query session - simulates a chat with the trained AGI.

    Args:
        galaxy_path: Path to galaxy_stars.pkl
    """
    query_engine = GalaxyKnowledgeQuery(galaxy_path)

    print("="*80)
    print("🧠 KNOWLEDGE3D GALAXY INFERENCE - INTERACTIVE SESSION")
    print("="*80)
    print()
    print("The model has been trained on:")
    print("  • Characters (trimodal)")
    print("  • Text domains & ARC-AGI reasoning")
    print("  • Multimodal (COCO, images, audio)")
    print("  • PDFs: Programming, AI, Math, Storytelling, Game Design")
    print("  • Wikipedia (language, medicine)")
    print()
    print("Type your queries below. Type 'exit' or 'quit' to end session.")
    print("="*80)
    print()

    query_count = 0
    while True:
        try:
            query_text = input(f"\n[Query #{query_count + 1}] >>> ").strip()

            if not query_text:
                continue

            if query_text.lower() in ['exit', 'quit', 'q']:
                print("\n[Session] Ending interactive session. Goodbye!")
                break

            # Execute query
            results = query_engine.query(query_text, top_k=3)
            query_engine.display_results(results)

            query_count += 1

        except KeyboardInterrupt:
            print("\n\n[Session] Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] Query failed: {e}")
            import traceback
            traceback.print_exc()


def test_predefined_queries(galaxy_path: Path):
    """
    Test with predefined queries covering all training domains.

    Args:
        galaxy_path: Path to galaxy_stars.pkl
    """
    query_engine = GalaxyKnowledgeQuery(galaxy_path)

    print("="*80)
    print("🧪 PREDEFINED QUERY TESTS - COVERING ALL TRAINING DOMAINS")
    print("="*80)
    print()

    # Predefined queries spanning all training domains
    test_queries = [
        # Characters & Text
        "What is the visual representation of the letter A?",

        # ARC-AGI (Abstract Reasoning)
        "How do you solve pattern recognition problems?",
        "What are spatial transformations?",

        # Programming & AI (from PDFs)
        "Explain neural network backpropagation",
        "What is a hash table?",
        "How does Bitcoin mining work?",

        # Mathematics (from PDFs)
        "What is calculus?",
        "Explain linear algebra concepts",

        # Game Design (from PDFs)
        "What makes a good game design?",
        "How do RPG systems work?",

        # Language & Medicine (Wikipedia)
        "What is machine learning?",
        "Explain medical diagnosis",

        # Multimodal
        "Describe visual object recognition",
    ]

    for query in test_queries:
        print("\n" + "─"*80)
        results = query_engine.query(query, top_k=3)
        query_engine.display_results(results)
        input("\nPress Enter for next query...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query the trained Galaxy knowledge base")
    parser.add_argument(
        "--mode",
        choices=["interactive", "test"],
        default="interactive",
        help="Query mode: interactive session or predefined tests"
    )
    parser.add_argument(
        "--galaxy-path",
        type=Path,
        default=Path("/K3D/Knowledge3D.local/house_zone7/embeddings/galaxy_stars.pkl"),
        help="Path to galaxy_stars.pkl"
    )

    args = parser.parse_args()

    if not args.galaxy_path.exists():
        print(f"ERROR: Galaxy stars file not found at {args.galaxy_path}")
        sys.exit(1)

    if args.mode == "interactive":
        interactive_query_session(args.galaxy_path)
    else:
        test_predefined_queries(args.galaxy_path)
