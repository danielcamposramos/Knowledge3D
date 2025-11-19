"""
Procedural Drawing Specialist for Adaptive Swarm.

Handles training and inference for procedural glyph generation and recognition,
enabling atomic cognition through visual-text alignment.

Architecture:
    - Text modality: RPNEmbeddingEngine generates embeddings for characters
    - Visual modality: GPU RPN executor → FractalEmitter generates embeddings
    - Cross-modal training: Align text ≈ visual for same character
    - Generative capability: text → RPN program → visual rendering

Usage:
    specialist = ProceduralDrawingSpecialist(swarm)
    specialist.train_on_rpn_dataset(
        dataset_path="data/font_rpn_168k.jsonl",
        epochs=10
    )
    rpn_program = specialist.generate_glyph('A')
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from knowledge3d.cranium.adaptive_swarm import AdaptiveSwarmTRM
from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge
from knowledge3d.cranium.ptx_runtime.rpn_embedding_engine import RPNEmbeddingEngine
from knowledge3d.cranium.bridges.sovereign_bridges import FractalEmitter


@dataclass
class TrainingMetrics:
    """Metrics for procedural drawing training."""
    epoch: int
    text_visual_alignment: float  # Cosine similarity
    reconstruction_fidelity: float  # SSIM score
    generation_quality: float      # Human eval proxy
    latency_us: float              # Average inference time


class ProceduralDrawingSpecialist:
    """
    Specialist for procedural glyph generation and recognition.

    Implements atomic cognition learning path:
    1. Learn drawing primitives (curves, lines, arcs)
    2. Align visual glyphs with text characters
    3. Enable generative drawing (text → RPN → visual)
    4. Enable OCR (visual → text via learned embeddings)
    """

    def __init__(
        self,
        swarm: AdaptiveSwarmTRM,
        matryoshka_dim: int = 512,
        gpu_id: int = 0
    ):
        """
        Initialize procedural drawing specialist.

        Args:
            swarm: Adaptive swarm instance
            matryoshka_dim: Embedding dimension (64-2048 adaptive)
            gpu_id: CUDA device ID
        """
        self.swarm = swarm
        self.matryoshka_dim = matryoshka_dim
        self.gpu_id = gpu_id

        # Register specialist with swarm
        rank = self._select_rank_from_dim(matryoshka_dim)
        self.swarm.register_specialist(
            'procedural_drawing',
            required_dims=matryoshka_dim,
            rank=rank
        )

        # Initialize bridges
        self.drawing_bridge = ProceduralDrawingBridge(matryoshka_dim=matryoshka_dim)
        self.text_embedder = RPNEmbeddingEngine(gpu_id=gpu_id)
        self.visual_embedder = FractalEmitter(gpu_id=gpu_id)

        # Training state
        self.training_metrics: List[TrainingMetrics] = []
        self.char_to_rpn_cache: Dict[str, str] = {}  # Learned RPN programs

    def _select_rank_from_dim(self, dim: int) -> int:
        """Select LoRA rank based on Matryoshka dimension."""
        # 18× memory reduction principle from Phase H
        return max(8, dim // 16)

    def load_rpn_dataset(self, dataset_path: Path) -> List[Dict]:
        """
        Load RPN dataset (JSONL format).

        Args:
            dataset_path: Path to JSONL file

        Returns:
            List of dataset entries
        """
        entries = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries

    def _compute_text_embedding(self, char: str) -> np.ndarray:
        """Generate text embedding for character using RPN engine."""
        # RPNEmbeddingEngine uses trigram hashing (language-agnostic)
        return self.text_embedder.embed(char).astype(np.float32)

    def _compute_visual_embedding(self, rpn_bytecode: bytes) -> np.ndarray:
        """Generate visual embedding from RPN program execution."""
        # Execute RPN on GPU → segments
        result = self.drawing_bridge.execute_rpn_bytecode_gpu(rpn_bytecode)

        if result.segments is None or len(result.segments) == 0:
            # Empty glyph - return zero embedding
            return np.zeros(self.matryoshka_dim, dtype=np.float32)

        # Convert segments to point cloud for FractalEmitter
        # Segments are (x0,y0,x1,y1,r,g,b,a,w) - extract points
        points = np.vstack([
            result.segments[:, :2],   # Start points
            result.segments[:, 2:4]   # End points
        ]).astype(np.float32)

        # FractalEmitter generates spatial features
        return self.visual_embedder.emit(points)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def train_on_batch(
        self,
        batch: List[Tuple[str, bytes]],
        validation: bool = False
    ) -> TrainingMetrics:
        """
        Train on batch of (character, rpn_bytecode) pairs.

        Args:
            batch: List of (char, rpn_bytecode) tuples
            validation: If True, compute metrics without updating weights

        Returns:
            Training metrics for this batch
        """
        text_embeddings = []
        visual_embeddings = []

        # Compute embeddings
        for char, rpn_bytecode in batch:
            text_emb = self._compute_text_embedding(char)
            visual_emb = self._compute_visual_embedding(rpn_bytecode)

            text_embeddings.append(text_emb)
            visual_embeddings.append(visual_emb)

        text_batch = np.stack(text_embeddings)
        visual_batch = np.stack(visual_embeddings)

        # Compute cross-modal alignment loss
        alignment_scores = []
        for text_emb, visual_emb in zip(text_embeddings, visual_embeddings):
            alignment_scores.append(self._cosine_similarity(text_emb, visual_emb))

        avg_alignment = float(np.mean(alignment_scores))

        # Train swarm specialist (if not validation)
        if not validation:
            # AdaptiveSwarmTRM expects (input, target, loss) tuples
            # We use contrastive learning: pull text/visual together
            training_pairs = [
                (text_emb, visual_emb, 1.0 - sim)  # Loss = 1 - similarity
                for text_emb, visual_emb, sim in zip(
                    text_embeddings, visual_embeddings, alignment_scores
                )
            ]

            # Update specialist via swarm
            self.swarm.train_specialist_epoch(
                'procedural_drawing',
                training_pairs,
                validation_samples=[]  # Validation handled separately
            )

        # Compute metrics
        metrics = TrainingMetrics(
            epoch=len(self.training_metrics),
            text_visual_alignment=avg_alignment,
            reconstruction_fidelity=0.0,  # TODO: Add SSIM computation
            generation_quality=0.0,       # TODO: Add generation eval
            latency_us=0.0                # TODO: Add timing
        )

        return metrics

    def train_on_rpn_dataset(
        self,
        dataset_path: Path,
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.1
    ):
        """
        Train specialist on RPN dataset.

        Args:
            dataset_path: Path to JSONL dataset
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
        """
        # Load dataset
        entries = self.load_rpn_dataset(dataset_path)

        # Compile RPN to bytecode
        training_data = []
        for entry in entries:
            char = entry['char']
            rpn = entry['rpn']

            # Compile RPN to bytecode
            bytecode = self.drawing_bridge.compile_rpn_to_bytecode(rpn)
            training_data.append((char, bytecode))

        # Split train/validation
        n_val = int(len(training_data) * validation_split)
        validation_data = training_data[:n_val]
        train_data = training_data[n_val:]

        print(f"Training on {len(train_data)} samples, validating on {len(validation_data)}")

        # Training loop
        for epoch in range(epochs):
            # Shuffle training data
            np.random.shuffle(train_data)

            # Train on batches
            epoch_metrics = []
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                metrics = self.train_on_batch(batch, validation=False)
                epoch_metrics.append(metrics)

            # Validation
            val_metrics = []
            for i in range(0, len(validation_data), batch_size):
                batch = validation_data[i:i+batch_size]
                metrics = self.train_on_batch(batch, validation=True)
                val_metrics.append(metrics)

            # Aggregate metrics
            avg_train_alignment = np.mean([m.text_visual_alignment for m in epoch_metrics])
            avg_val_alignment = np.mean([m.text_visual_alignment for m in val_metrics])

            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train alignment={avg_train_alignment:.3f}, "
                  f"Val alignment={avg_val_alignment:.3f}")

            # Store metrics
            self.training_metrics.append(TrainingMetrics(
                epoch=epoch,
                text_visual_alignment=avg_val_alignment,
                reconstruction_fidelity=0.0,
                generation_quality=0.0,
                latency_us=0.0
            ))

    def generate_glyph(self, char: str) -> str:
        """
        Generate RPN program for character (generative capability).

        Args:
            char: Character to generate

        Returns:
            RPN program string
        """
        # Check cache first
        if char in self.char_to_rpn_cache:
            return self.char_to_rpn_cache[char]

        # Generate via learned embeddings
        text_emb = self._compute_text_embedding(char)

        # Use swarm to predict visual embedding
        predicted_visual = self.swarm.forward(
            text_emb,
            specialist='procedural_drawing'
        )

        # Decode visual embedding to RPN
        # TODO: Implement decoder (inverse of execute_rpn → fractal_emit)
        # For now, return placeholder
        rpn_program = f"# Generated RPN for '{char}' (decoder pending)"

        self.char_to_rpn_cache[char] = rpn_program
        return rpn_program

    def save_checkpoint(self, path: Path):
        """Save specialist state."""
        checkpoint = {
            'matryoshka_dim': self.matryoshka_dim,
            'training_metrics': [
                {
                    'epoch': m.epoch,
                    'text_visual_alignment': m.text_visual_alignment,
                    'reconstruction_fidelity': m.reconstruction_fidelity,
                    'generation_quality': m.generation_quality,
                    'latency_us': m.latency_us
                }
                for m in self.training_metrics
            ],
            'char_to_rpn_cache': self.char_to_rpn_cache
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)


__all__ = ['ProceduralDrawingSpecialist', 'TrainingMetrics']
