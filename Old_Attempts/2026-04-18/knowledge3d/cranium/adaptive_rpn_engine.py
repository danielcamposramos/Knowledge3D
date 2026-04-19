"""
Adaptive RPN Embedding Engine with Variable Dimensionality

Extends the sovereign RPN engine with Matryoshka-inspired dimension selection.

Key Innovation: Automatically select embedding dimension based on content complexity:
- Single phrase (5-20 chars) → 64D
- Short text (20-100 chars) → 128D
- Medium text (100-500 chars) → 256D
- Long text (500-2000 chars) → 512D
- Very long text (2000+ chars) → 1024D or 2048D

Benefits:
- Memory efficiency: Why use 512D for "Hello"? 64D is enough!
- Speed: 64× faster embedding for simple content
- Scalability: Can handle both tweets and full documents
- Compatibility: Works with existing RPN trigram system

Integration Points:
- PDF ingestion (variable page complexity)
- Specialist training (task-specific dimensions)
- Galaxy star creation (efficient knowledge storage)
- Sleep consolidation (adaptive materialization)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from dataclasses import dataclass

from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine


@dataclass
class DimensionConfig:
    """Configuration for adaptive dimension selection."""

    # Dimension levels (geometric progression)
    dim_levels: List[int] = None

    # Content length thresholds (characters)
    # Maps: max_chars → dimension
    length_thresholds: Dict[int, int] = None

    # Complexity thresholds (0.0-1.0)
    # Maps: max_complexity → dimension
    complexity_thresholds: Dict[float, int] = None

    # Default dimension if no threshold matches
    default_dim: int = 128

    # Minimum dimension (even for empty content)
    min_dim: int = 64

    # Maximum dimension
    max_dim: int = 2048

    def __post_init__(self):
        """Initialize default thresholds."""
        if self.dim_levels is None:
            self.dim_levels = [64, 128, 256, 512, 1024, 2048]

        if self.length_thresholds is None:
            # Character count → dimension mapping
            self.length_thresholds = {
                20: 64,      # Single phrase: "Hello world"
                100: 128,    # Short sentence: "The quick brown fox..."
                500: 256,    # Paragraph: 2-3 sentences
                2000: 512,   # Multiple paragraphs
                8000: 1024,  # Full page
                32000: 2048  # Multiple pages
            }

        if self.complexity_thresholds is None:
            # Complexity score → dimension mapping
            self.complexity_thresholds = {
                0.1: 64,     # Trivial
                0.3: 128,    # Simple
                0.5: 256,    # Medium
                0.7: 512,    # Complex
                0.85: 1024,  # Very complex
                1.0: 2048    # Maximum
            }


class AdaptiveRPNEngine:
    """
    RPN Embedding Engine with adaptive dimension selection.

    Wraps the base RPNEmbeddingEngine and adds intelligent dimension
    selection based on content complexity.
    """

    def __init__(self, config: Optional[DimensionConfig] = None):
        """
        Initialize adaptive RPN engine.

        Args:
            config: Dimension configuration
        """
        self.config = config or DimensionConfig()

        # Create base engines for each dimension level
        # This allows us to maintain separate vocabularies per dimension
        self.engines: Dict[int, RPNEmbeddingEngine] = {}

        for dim in self.config.dim_levels:
            self.engines[dim] = RPNEmbeddingEngine(embedding_dim=dim)

        # Statistics
        self.dimension_usage_stats: Dict[int, int] = {dim: 0 for dim in self.config.dim_levels}
        self.total_embeddings = 0

        print(f"[AdaptiveRPNEngine] Initialized")
        print(f"  Dimension levels: {self.config.dim_levels}")
        print(f"  Length thresholds: {sorted(self.config.length_thresholds.items())}")

    # ------------------------------------------------------------------ #
    # Dimension Selection
    # ------------------------------------------------------------------ #
    def select_dimension_by_length(self, text: str) -> int:
        """
        Select optimal dimension based on text length.

        Args:
            text: Input text

        Returns:
            Recommended dimension
        """
        text_len = len(text.strip())

        # Empty text → minimum dimension
        if text_len == 0:
            return self.config.min_dim

        # Find smallest dimension that can handle this length
        for max_len, dim in sorted(self.config.length_thresholds.items()):
            if text_len <= max_len:
                return dim

        # Exceeded all thresholds → use maximum
        return self.config.max_dim

    def select_dimension_by_complexity(self, complexity: float) -> int:
        """
        Select optimal dimension based on complexity score.

        Args:
            complexity: Task complexity [0.0, 1.0]

        Returns:
            Recommended dimension
        """
        complexity = max(0.0, min(1.0, complexity))

        for max_complexity, dim in sorted(self.config.complexity_thresholds.items()):
            if complexity <= max_complexity:
                return dim

        return self.config.max_dim

    def estimate_complexity(self, text: str) -> float:
        """
        Estimate text complexity using heuristics.

        Factors:
        - Length (longer = more complex)
        - Vocabulary diversity (unique words / total words)
        - Punctuation density
        - Average word length

        Args:
            text: Input text

        Returns:
            Complexity score [0.0, 1.0]
        """
        text = text.strip()

        if not text:
            return 0.0

        # Length factor (logarithmic scaling)
        length_score = min(1.0, np.log10(len(text) + 1) / 4.0)  # log10(10000) ≈ 4

        # Vocabulary diversity
        tokens = text.split()
        if len(tokens) > 0:
            unique_ratio = len(set(tokens)) / len(tokens)
        else:
            unique_ratio = 0.0

        # Punctuation density (academic text has more punctuation)
        punct_count = sum(1 for ch in text if ch in '.,;:!?-()[]{}')
        punct_density = min(1.0, punct_count / (len(text) + 1) * 50)

        # Average word length (technical text has longer words)
        if tokens:
            avg_word_len = sum(len(tok) for tok in tokens) / len(tokens)
            word_len_score = min(1.0, avg_word_len / 10.0)
        else:
            word_len_score = 0.0

        # Weighted combination
        complexity = (
            0.4 * length_score +
            0.3 * unique_ratio +
            0.2 * punct_density +
            0.1 * word_len_score
        )

        return complexity

    def select_dimension_auto(self, text: str) -> int:
        """
        Automatically select optimal dimension.

        Combines length-based and complexity-based selection.

        Args:
            text: Input text

        Returns:
            Recommended dimension
        """
        # Get dimension from both methods
        dim_length = self.select_dimension_by_length(text)
        complexity = self.estimate_complexity(text)
        dim_complexity = self.select_dimension_by_complexity(complexity)

        # Use the higher dimension (more conservative)
        # This prevents under-representation of complex content
        selected_dim = max(dim_length, dim_complexity)

        # Ensure it's a valid level
        if selected_dim not in self.config.dim_levels:
            # Snap to nearest level
            selected_dim = min(self.config.dim_levels, key=lambda x: abs(x - selected_dim))

        return selected_dim

    # ------------------------------------------------------------------ #
    # Embedding Generation (Adaptive)
    # ------------------------------------------------------------------ #
    def embed_sentence(self, sentence: str, target_dim: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Embed sentence with adaptive dimension selection.

        Args:
            sentence: Input sentence
            target_dim: Target dimension (None = auto-select)

        Returns:
            (embedding, actual_dimension)
        """
        # Select dimension
        if target_dim is None:
            target_dim = self.select_dimension_auto(sentence)
        elif target_dim not in self.engines:
            raise ValueError(f"Unsupported dimension: {target_dim}")

        # Get engine for this dimension
        engine = self.engines[target_dim]

        # Generate embedding
        embedding = engine.embed_sentence(sentence)

        # Update stats
        self.dimension_usage_stats[target_dim] += 1
        self.total_embeddings += 1

        return embedding, target_dim

    def embed_word(self, word: str, target_dim: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Embed word with adaptive dimension selection.

        Args:
            word: Input word
            target_dim: Target dimension (None = auto-select)

        Returns:
            (embedding, actual_dimension)
        """
        if target_dim is None:
            target_dim = self.select_dimension_auto(word)
        elif target_dim not in self.engines:
            raise ValueError(f"Unsupported dimension: {target_dim}")

        engine = self.engines[target_dim]
        embedding = engine.embed_word(word)

        self.dimension_usage_stats[target_dim] += 1
        self.total_embeddings += 1

        return embedding, target_dim

    def embed_batch(self, texts: List[str], target_dim: Optional[int] = None) -> Tuple[np.ndarray, List[int]]:
        """
        Embed batch of texts with adaptive dimensions.

        Each text can have different dimension based on complexity.
        Results are padded to maximum dimension in batch.

        Args:
            texts: List of input texts
            target_dim: Fixed dimension for all (None = auto per text)

        Returns:
            (embeddings_matrix [N × max_dim], dimensions_per_text)
        """
        embeddings = []
        dimensions = []

        for text in texts:
            emb, dim = self.embed_sentence(text, target_dim)
            embeddings.append(emb)
            dimensions.append(dim)

        if not embeddings:
            return np.zeros((0, self.config.min_dim), dtype=np.float32), []

        # Find maximum dimension in batch
        max_dim = max(dimensions)

        # Pad all embeddings to max_dim
        padded_embeddings = []
        for emb, dim in zip(embeddings, dimensions):
            if dim < max_dim:
                # Pad with zeros
                padded = np.zeros(max_dim, dtype=np.float32)
                padded[:dim] = emb
                padded_embeddings.append(padded)
            else:
                padded_embeddings.append(emb)

        return np.vstack(padded_embeddings), dimensions

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_all(self, base_path: Path):
        """
        Save all dimension-specific engines.

        Args:
            base_path: Base directory for checkpoints
        """
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)

        for dim, engine in self.engines.items():
            engine_path = base_path / f"rpn_embeddings_{dim}d.pkl"
            engine.save_embeddings(engine_path)
            print(f"[AdaptiveRPNEngine] Saved {dim}D engine: {engine.vocab_size} trigrams")

        # Save configuration and stats
        import json
        metadata = {
            'dim_levels': self.config.dim_levels,
            'length_thresholds': {str(k): v for k, v in self.config.length_thresholds.items()},
            'complexity_thresholds': {str(k): v for k, v in self.config.complexity_thresholds.items()},
            'usage_stats': self.dimension_usage_stats,
            'total_embeddings': self.total_embeddings,
            'vocab_sizes': {dim: eng.vocab_size for dim, eng in self.engines.items()}
        }

        with open(base_path / 'adaptive_engine_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[AdaptiveRPNEngine] Saved metadata to {base_path}")

    def load_all(self, base_path: Path):
        """
        Load all dimension-specific engines.

        Args:
            base_path: Base directory with checkpoints
        """
        base_path = Path(base_path)

        for dim in self.config.dim_levels:
            engine_path = base_path / f"rpn_embeddings_{dim}d.pkl"
            if engine_path.exists():
                self.engines[dim].load_embeddings(engine_path)
                print(f"[AdaptiveRPNEngine] Loaded {dim}D engine: {self.engines[dim].vocab_size} trigrams")
            else:
                print(f"[AdaptiveRPNEngine] No checkpoint for {dim}D, using fresh engine")

        # Load metadata if available
        metadata_path = base_path / 'adaptive_engine_metadata.json'
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            self.dimension_usage_stats = metadata.get('usage_stats', self.dimension_usage_stats)
            self.total_embeddings = metadata.get('total_embeddings', 0)

            print(f"[AdaptiveRPNEngine] Loaded metadata from {base_path}")

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #
    def get_stats(self) -> Dict:
        """Get usage statistics."""
        stats = {
            'total_embeddings': self.total_embeddings,
            'dimension_usage': {},
            'vocab_sizes': {},
            'efficiency_gain': 0.0
        }

        # Dimension usage percentages
        for dim, count in self.dimension_usage_stats.items():
            if self.total_embeddings > 0:
                percentage = (count / self.total_embeddings) * 100
            else:
                percentage = 0.0
            stats['dimension_usage'][dim] = {
                'count': count,
                'percentage': percentage
            }

        # Vocabulary sizes
        for dim, engine in self.engines.items():
            stats['vocab_sizes'][dim] = engine.vocab_size

        # Efficiency gain (compared to always using max dimension)
        if self.total_embeddings > 0:
            max_dim = self.config.max_dim
            avg_dim_used = sum(
                dim * count for dim, count in self.dimension_usage_stats.items()
            ) / self.total_embeddings

            # Compute ops: O(d²) for embedding generation
            # Efficiency = (max_d² * N) / (avg_d² * N) = (max_d / avg_d)²
            efficiency_gain = (max_dim / avg_dim_used) ** 2 if avg_dim_used > 0 else 1.0
            stats['efficiency_gain'] = efficiency_gain
            stats['avg_dimension_used'] = avg_dim_used

        return stats

    def print_stats(self):
        """Print usage statistics."""
        stats = self.get_stats()

        print("\n" + "="*60)
        print("ADAPTIVE RPN ENGINE STATISTICS")
        print("="*60)
        print(f"Total embeddings generated: {stats['total_embeddings']:,}")
        print(f"Average dimension used: {stats.get('avg_dimension_used', 0):.1f}")
        print(f"Efficiency gain: {stats['efficiency_gain']:.1f}× faster than max dimension")
        print("\nDimension Usage:")

        for dim in sorted(self.config.dim_levels):
            usage = stats['dimension_usage'].get(dim, {'count': 0, 'percentage': 0.0})
            vocab_size = stats['vocab_sizes'].get(dim, 0)
            print(f"  {dim:>4}D: {usage['count']:>6} embeddings ({usage['percentage']:>5.1f}%) | Vocab: {vocab_size:>6} trigrams")

        print("="*60 + "\n")


__all__ = ['AdaptiveRPNEngine', 'DimensionConfig']
