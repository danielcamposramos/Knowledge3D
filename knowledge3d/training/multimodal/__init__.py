"""
Multi-Modal TRM Training: Phase G.1

Parallel training streams enabling cross-modal learning:
- OCR: Visual features → Character embeddings
- Text: Semantic reasoning → Answer quality
- Alignment: Visual ↔ Semantic connections

Safe self-updating mechanism prevents catastrophic forgetting.
"""

from .multimodal_trainer import (
    MultiModalTRMTrainer,
    TrainingConfig,
    OCRTrainingStream,
    TextTrainingStream,
    CrossModalAligner
)

from .self_updating_trm import (
    SelfUpdatingTRM,
    UpdateConfig,
    UpdateStrategy,
    TRMWeightManager
)

__all__ = [
    'MultiModalTRMTrainer',
    'TrainingConfig',
    'OCRTrainingStream',
    'TextTrainingStream',
    'CrossModalAligner',
    'SelfUpdatingTRM',
    'UpdateConfig',
    'UpdateStrategy',
    'TRMWeightManager'
]
