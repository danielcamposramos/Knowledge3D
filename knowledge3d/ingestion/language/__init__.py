"""
Language ingestion pipelines.

This package exposes multiple ingestion front-ends (text, audio, visual) that
ultimately feed the sovereign memory stack. Imports are resolved lazily so
optional dependencies (e.g. librosa for audio) are only required when their
corresponding pipeline is accessed.
"""

from importlib import import_module
from typing import Any, Dict

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

_MODULE_MAP: Dict[str, str] = {
    "TextLanguageIngestor": "text_pipeline",
    "AudioLanguageIngestor": "audio_pipeline",
    "VisualLanguageIngestor": "visual_pipeline",
    "SovereignTextIngestor": "sovereign_text_pipeline",
    "SovereignAudioIngestor": "sovereign_audio_pipeline",
    "SovereignVisualIngestor": "sovereign_visual_pipeline",
    "ResourceSafeIngestionController": "resource_controller",
    "LanguageSwarmProcessor": "swarm_integration",
}


def __getattr__(name: str) -> Any:
    module_name = _MODULE_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f"{__name__}.{module_name}")
    attr = getattr(module, name)
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:
    return sorted(list(__all__))
