"""
Multi-Modal TRM Trainer: Phase G.1

Parallel training streams:
1. OCR Stream: Visual features → Character embeddings (Phase F.2)
2. Text Stream: Semantic reasoning → Answer quality (RLWHF)

Shared latent space enables cross-modal learning:
- Character 'A' visual pattern shares embedding with semantic concept 'A'
- Grounded language understanding
- Transfer learning between modalities

Architecture:
    Visual Context (PDF) ──┐
                           ├──> TRM Shared Latent [256-dim] ──> Updates
    Text Context (Q&A)  ──┘

Timeline: Samples 8042 → 10K (multi-modal foundation)
          Samples 10K+ (self-updating mode)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import time
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Multi-modal training configuration."""
    ocr_weight: float = 1.0           # Weight for OCR loss
    text_weight: float = 1.0          # Weight for RLWHF text loss
    alignment_weight: float = 0.1     # Weight for cross-modal alignment
    learning_rate: float = 0.001      # Base learning rate
    validation_split: float = 0.1     # 10% holdout for validation
    batch_size: int = 1               # Process one sample at a time (streaming)
    gradient_clip: float = 1.0        # Gradient clipping threshold


class OCRTrainingStream:
    """
    OCR training stream: Extract visual features from PDF sources.

    Uses DeepSeekOCRModel feature extraction + CharacterDetector templates.
    """

    def __init__(self):
        self.feature_extractor = None  # Will initialize on first use
        self.character_templates = None

    def initialize(self):
        """Lazy initialization of OCR components."""
        try:
            from knowledge3d.cranium.ocr.deepseek_ocr_model import DeepSeekOCRModel
            from knowledge3d.cranium.ocr.character_detector import GalacticTemplateBank

            self.feature_extractor = DeepSeekOCRModel(
                num_glyphs=256,
                input_channels=3,
                use_micro_trm=False
            )

            self.character_templates = GalacticTemplateBank(
                num_glyphs=256,
                feature_dim=128
            )

            print("[OCR Stream] ✓ Initialized")
            return True

        except Exception as e:
            print(f"[OCR Stream] ⚠ Could not initialize: {e}")
            return False

    def extract_visual_features(self, pdf_path: str, page_num: int) -> Optional[np.ndarray]:
        """
        Extract visual features from PDF page.

        Returns: Feature map [H/4, W/4, 128] or None if extraction fails
        """
        if self.feature_extractor is None:
            if not self.initialize():
                return None

        try:
            # This would use the PDF rendering + feature extraction
            # For now, placeholder - will integrate with actual PDF pipeline
            # TODO: Connect to pdf2image + DeepSeekOCRModel
            return None

        except Exception as e:
            print(f"[OCR Stream] Failed to extract features: {e}")
            return None

    def compute_ocr_loss(self, visual_features: np.ndarray,
                        ground_truth_chars: List[str]) -> float:
        """
        Compute OCR training loss.

        Args:
            visual_features: Extracted features [H, W, C]
            ground_truth_chars: Expected characters in image

        Returns:
            OCR loss value
        """
        # Template matching loss
        # Compare extracted features to character templates
        # This will be the training signal for GalacticTemplateBank Layer 3

        # For now, placeholder
        return 0.0


class TextTrainingStream:
    """
    Text training stream: RLWHF semantic reasoning.

    Uses existing RLWHF pipeline with TRM student + teacher evaluation.
    """

    def __init__(self):
        self.trm_engine = None
        self.teacher_evaluator = None

    def initialize(self):
        """Initialize TRM and teacher components."""
        try:
            from knowledge3d.cranium.trm_engine import TRMEngine

            self.trm_engine = TRMEngine()
            print("[Text Stream] ✓ Initialized")
            return True

        except Exception as e:
            print(f"[Text Stream] ⚠ Could not initialize: {e}")
            return False

    def compute_text_loss(self, question: str, context: str,
                         ground_truth: str, teacher_eval: Dict[str, Any]) -> float:
        """
        Compute text reasoning loss from teacher evaluation.

        Args:
            question: Question text
            context: Context from PDF
            ground_truth: Expected answer
            teacher_eval: Teacher evaluation results

        Returns:
            Text loss value (lower = better)
        """
        # Extract teacher rating
        rating_score = teacher_eval.get('rating_score', -1)

        if rating_score < 0:
            # Teacher evaluation failed
            return 1.0  # High loss

        # Convert rating to loss (10/10 = 0.0 loss, 1/10 = 0.9 loss)
        loss = 1.0 - (rating_score / 10.0)

        # Weight by honesty score (higher honesty = more reliable signal)
        honesty = teacher_eval.get('honesty_score', 0.0)
        weighted_loss = loss * (1.0 + honesty)  # Higher honesty amplifies signal

        return weighted_loss


class CrossModalAligner:
    """
    Cross-modal alignment: Connect visual and semantic representations.

    When OCR sees visual 'A' and text reasoning uses concept 'A',
    their latent embeddings should be similar.
    """

    def __init__(self, latent_dim: int = 256):
        self.latent_dim = latent_dim

        # Character vocabulary (ASCII printable)
        self.vocab = [chr(i) for i in range(32, 127)]

        # Track visual and semantic embeddings for each character
        self.visual_embeddings: Dict[str, List[np.ndarray]] = {}
        self.semantic_embeddings: Dict[str, List[np.ndarray]] = {}

    def register_visual_embedding(self, char: str, embedding: np.ndarray):
        """Record visual embedding for character."""
        if char not in self.visual_embeddings:
            self.visual_embeddings[char] = []
        self.visual_embeddings[char].append(embedding)

    def register_semantic_embedding(self, char: str, embedding: np.ndarray):
        """Record semantic embedding for character."""
        if char not in self.semantic_embeddings:
            self.semantic_embeddings[char] = []
        self.semantic_embeddings[char].append(embedding)

    def compute_alignment_loss(self) -> float:
        """
        Compute cross-modal alignment loss.

        For each character that appears in both modalities,
        minimize distance between visual and semantic embeddings.
        """
        total_loss = 0.0
        alignment_count = 0

        # For each character with both visual and semantic representations
        for char in self.vocab:
            if (char in self.visual_embeddings and
                char in self.semantic_embeddings and
                len(self.visual_embeddings[char]) > 0 and
                len(self.semantic_embeddings[char]) > 0):

                # Average embeddings for this character
                visual_avg = np.mean(self.visual_embeddings[char], axis=0)
                semantic_avg = np.mean(self.semantic_embeddings[char], axis=0)

                # Cosine distance (1 - cosine similarity)
                visual_norm = np.linalg.norm(visual_avg)
                semantic_norm = np.linalg.norm(semantic_avg)

                if visual_norm > 1e-6 and semantic_norm > 1e-6:
                    cosine_sim = np.dot(visual_avg, semantic_avg) / (visual_norm * semantic_norm)
                    cosine_dist = 1.0 - cosine_sim

                    total_loss += cosine_dist
                    alignment_count += 1

        # Average alignment loss
        return total_loss / max(alignment_count, 1)

    def reset_batch(self):
        """Clear embeddings for next batch."""
        self.visual_embeddings.clear()
        self.semantic_embeddings.clear()


class MultiModalTRMTrainer:
    """
    Multi-modal TRM trainer: OCR + Text in parallel.

    Trains on RLWHF dataset with both:
    1. Visual features from PDF sources (OCR task)
    2. Semantic reasoning from Q&A (RLWHF task)

    Shared TRM latent space enables cross-modal learning.
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()

        # Training streams
        self.ocr_stream = OCRTrainingStream()
        self.text_stream = TextTrainingStream()
        self.aligner = CrossModalAligner()

        # Metrics tracking
        self.step = 0
        self.total_loss_history = []
        self.ocr_loss_history = []
        self.text_loss_history = []
        self.alignment_loss_history = []

        # Validation set
        self.validation_samples = []

        print("[MultiModalTrainer] Initialized")

    def load_rlwhf_dataset(self, jsonl_path: Path,
                          start_idx: int = 0,
                          end_idx: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load RLWHF dataset samples."""
        samples = []

        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f, 1):
                if i < start_idx:
                    continue
                if end_idx and i > end_idx:
                    break

                samples.append(json.loads(line))

        return samples

    def split_train_validation(self, samples: List[Dict[str, Any]]) -> Tuple[List, List]:
        """Split into training and validation sets."""
        n_total = len(samples)
        n_val = int(n_total * self.config.validation_split)

        # Shuffle and split
        indices = np.random.permutation(n_total)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        train_samples = [samples[i] for i in train_indices]
        val_samples = [samples[i] for i in val_indices]

        print(f"[Dataset] Train: {len(train_samples)}, Validation: {len(val_samples)}")

        return train_samples, val_samples

    def training_step(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """
        Single training step on one sample.

        Returns: Dictionary of losses
        """
        losses = {
            'ocr_loss': 0.0,
            'text_loss': 0.0,
            'alignment_loss': 0.0,
            'total_loss': 0.0
        }

        # Reset aligner for this sample
        self.aligner.reset_batch()

        # Stream 1: Text reasoning (always available)
        teacher_eval = sample.get('teacher_evaluation', {})
        text_loss = self.text_stream.compute_text_loss(
            question=sample.get('question', ''),
            context=sample.get('context', ''),
            ground_truth=sample.get('answer', ''),
            teacher_eval=teacher_eval
        )
        losses['text_loss'] = text_loss

        # Register semantic embeddings from student attempt
        student_attempt = sample.get('student_attempt', {})
        if 'latent_embedding' in student_attempt:
            latent_emb = np.array(student_attempt['latent_embedding'])

            # Extract characters from answer
            answer = sample.get('answer', '')
            for char in answer[:10]:  # First 10 chars
                if char in self.aligner.vocab:
                    self.aligner.register_semantic_embedding(char, latent_emb)

        # Stream 2: OCR (only if PDF source available)
        source = sample.get('source', '')
        if 'pdf' in source.lower():
            # Extract PDF info
            pdf_name = sample.get('pdf_name', '')
            page_num = sample.get('page_num', 0)

            # Extract visual features (placeholder for now)
            visual_features = self.ocr_stream.extract_visual_features(pdf_name, page_num)

            if visual_features is not None:
                # Compute OCR loss
                ground_truth_chars = list(sample.get('answer', ''))
                ocr_loss = self.ocr_stream.compute_ocr_loss(visual_features, ground_truth_chars)
                losses['ocr_loss'] = ocr_loss

                # Register visual embeddings (placeholder)
                # TODO: Extract character-level visual embeddings

        # Stream 3: Cross-modal alignment
        alignment_loss = self.aligner.compute_alignment_loss()
        losses['alignment_loss'] = alignment_loss

        # Total loss (weighted combination)
        losses['total_loss'] = (
            self.config.ocr_weight * losses['ocr_loss'] +
            self.config.text_weight * losses['text_loss'] +
            self.config.alignment_weight * losses['alignment_loss']
        )

        # Track metrics
        self.total_loss_history.append(losses['total_loss'])
        self.ocr_loss_history.append(losses['ocr_loss'])
        self.text_loss_history.append(losses['text_loss'])
        self.alignment_loss_history.append(losses['alignment_loss'])

        self.step += 1

        return losses

    def train_epoch(self, train_samples: List[Dict[str, Any]]):
        """Train for one epoch over all samples."""
        print(f"\n[Training] Starting epoch over {len(train_samples)} samples")

        epoch_losses = []

        for i, sample in enumerate(train_samples):
            losses = self.training_step(sample)
            epoch_losses.append(losses['total_loss'])

            # Log progress every 100 steps
            if (i + 1) % 100 == 0:
                avg_loss = np.mean(epoch_losses[-100:])
                print(f"  Step {self.step} ({i+1}/{len(train_samples)}): "
                      f"Loss {avg_loss:.4f} "
                      f"(OCR: {losses['ocr_loss']:.4f}, "
                      f"Text: {losses['text_loss']:.4f}, "
                      f"Align: {losses['alignment_loss']:.4f})")

        avg_epoch_loss = np.mean(epoch_losses)
        print(f"[Training] Epoch complete. Average loss: {avg_epoch_loss:.4f}")

        return avg_epoch_loss

    def evaluate(self, val_samples: List[Dict[str, Any]]) -> float:
        """Evaluate on validation set."""
        print(f"\n[Validation] Evaluating {len(val_samples)} samples")

        val_losses = []

        for sample in val_samples:
            # Compute losses without updating weights
            losses = self.training_step(sample)
            val_losses.append(losses['total_loss'])

        avg_val_loss = np.mean(val_losses)
        print(f"[Validation] Average loss: {avg_val_loss:.4f}")

        return avg_val_loss

    def save_checkpoint(self, checkpoint_path: Path, metrics: Dict[str, Any]):
        """Save training checkpoint."""
        checkpoint_data = {
            'step': self.step,
            'config': {
                'ocr_weight': self.config.ocr_weight,
                'text_weight': self.config.text_weight,
                'alignment_weight': self.config.alignment_weight,
                'learning_rate': self.config.learning_rate
            },
            'metrics': metrics,
            'loss_history': {
                'total': self.total_loss_history[-1000:],  # Last 1000 steps
                'ocr': self.ocr_loss_history[-1000:],
                'text': self.text_loss_history[-1000:],
                'alignment': self.alignment_loss_history[-1000:]
            }
        }

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        print(f"[Checkpoint] Saved to {checkpoint_path}")
