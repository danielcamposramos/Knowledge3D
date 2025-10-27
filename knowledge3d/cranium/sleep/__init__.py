"""
Sleep-time consolidation utilities.

This subpackage houses Phase D components that refine embeddings and glyph
tables during idle windows so ingestion work can remain one-shot.

Phase G extensions:
- ModelSleepCycle: Shadow weights validation (MODELS = LOGIC)
- KnowledgeSleepCycle: Galaxy → House materialization (KNOWLEDGE = 3D SPACE)
"""

from knowledge3d.cranium.sleep.model_sleep import ModelSleepCycle
from knowledge3d.cranium.sleep.knowledge_sleep import KnowledgeSleepCycle

__all__ = [
    "glyph_consolidator",
    "ModelSleepCycle",
    "KnowledgeSleepCycle"
]
