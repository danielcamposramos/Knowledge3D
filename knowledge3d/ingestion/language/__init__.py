"""
Language ingestion pipelines.

These modules convert raw linguistic datasets (text, audio, visual) into
GPU-resident embeddings that can be routed through the specialised nine-chain
swarm and stored within the Knowledge3D memory architecture.
"""

from .text_pipeline import TextLanguageIngestor
from .audio_pipeline import AudioLanguageIngestor
from .visual_pipeline import VisualLanguageIngestor
from .sovereign_text_pipeline import SovereignTextIngestor
from .sovereign_audio_pipeline import SovereignAudioIngestor
from .sovereign_visual_pipeline import SovereignVisualIngestor
from .resource_controller import ResourceSafeIngestionController
from .swarm_integration import LanguageSwarmProcessor

__all__ = [
    "TextLanguageIngestor",
    "SovereignTextIngestor",
    "SovereignAudioIngestor",
    "SovereignVisualIngestor",
    "ResourceSafeIngestionController",
    "AudioLanguageIngestor",
    "VisualLanguageIngestor",
    "LanguageSwarmProcessor",
]
