"""
Link video clips to Galaxy stars via ternary video codec and RPN seeds.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from knowledge3d.cranium.codecs.ternary_video_codec import TernaryVideoCodec
from knowledge3d.bridge.memory_tablet import MemoryTablet


class GalaxyVideoLinker:
    """Link video frames to Galaxy stars using ternary procedural codec."""

    def __init__(self, width: int = 1920, height: int = 1080, use_gpu: bool = False):
        self.codec = TernaryVideoCodec(width=width, height=height, use_gpu=use_gpu)
        self.tablet = MemoryTablet()

    def link_video_to_star(self, star_id: str, frames: np.ndarray, label: str = "sample_video", fps: int = 30) -> Dict:
        """
        Encode a video clip and associate it with a Galaxy star.

        Args:
            star_id: Galaxy star identifier.
            frames: Array (T, H, W, 3) uint8/float32 matching codec resolution.
            label: Representation label.
            fps: Frames per second metadata.
        """
        metadata = self.codec.encode_to_rpn(frames, fps=fps)
        star = self.tablet.get_star(star_id)
        if "video_representations" not in star:
            star["video_representations"] = {}
        star["video_representations"][label] = {
            "codec": "ternary_procedural_video",
            "metadata": metadata,
            "fps": fps,
        }
        self.tablet.update_star(star_id, star)
        return star["video_representations"][label]
