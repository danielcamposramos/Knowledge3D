"""
Link audio representations to Galaxy stars via ternary codec and RPN stack.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from knowledge3d.cranium.codecs.ternary_audio_codec import TernaryAudioCodec
from knowledge3d.bridge.memory_tablet import MemoryTablet


class GalaxyAudioLinker:
    """Link audio signals to Galaxy meaning-stars using the ternary audio codec."""

    def __init__(self, sample_rate: int = 44100, use_gpu: bool = False):
        self.codec = TernaryAudioCodec(sample_rate=sample_rate, use_gpu=use_gpu)
        self.tablet = MemoryTablet()

    def link_audio_to_star(self, star_id: str, audio: np.ndarray, audio_type: str = "characteristic_sound") -> Dict:
        """
        Encode audio and attach it to a Galaxy star.

        Args:
            star_id: Identifier of the star to update.
            audio: Mono float32 samples.
            audio_type: Label for the audio (e.g., characteristic_sound, pronunciation).

        Returns:
            Stored metadata dictionary.
        """
        metadata = self.codec.encode(audio)
        star = self.tablet.get_star(star_id)
        if "audio_representations" not in star:
            star["audio_representations"] = {}
        star["audio_representations"][audio_type] = {
            "codec": "ternary_procedural",
            "metadata": metadata,
            "compression_ratio": self.codec.compute_compression_ratio(len(audio) * 4, metadata),
            "duration_sec": len(audio) / self.codec.sample_rate,
        }
        self.tablet.update_star(star_id, star)
        return star["audio_representations"][audio_type]

    def retrieve_audio_from_star(self, star_id: str, audio_type: str = "characteristic_sound") -> np.ndarray:
        """Decode audio from a star representation."""
        star = self.tablet.get_star(star_id)
        meta = star["audio_representations"][audio_type]["metadata"]
        return self.codec.decode(meta)
