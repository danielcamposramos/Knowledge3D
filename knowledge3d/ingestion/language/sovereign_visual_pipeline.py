"""
Sovereign visual ingestion pipeline.

Renders glyphs with PIL, extracts edge maps with OpenCV, and uses the sovereign
FractalEmitter to generate spatial coordinates which are transformed into
128-dimensional embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from knowledge3d.cranium.ptx_runtime.fractal_emitter import FractalEmitter


@dataclass
class SovereignVisualIngestor:
    """
    Sovereign visual/glyph ingestion helper.
    """

    canvas_size: int = 64

    def __post_init__(self) -> None:
        self.fractal_emitter = FractalEmitter()

    def ingest_glyph(self, char: str, font_path: str | Path, lang: str) -> Dict:
        font_path = Path(font_path)
        if not font_path.exists():
            raise FileNotFoundError(font_path)
        if not char:
            raise ValueError("char must be non-empty")

        image = self._render_character(char, font_path, self.canvas_size)
        image_arr = np.array(image, dtype=np.uint8)

        edges = cv2.Canny(image_arr, 50, 150)
        atom_values = edges.flatten().astype(np.float32) / 255.0
        if not np.any(atom_values):
            atom_values = np.array([0.0], dtype=np.float32)

        coords = self.fractal_emitter.emit(atom_values, base_scale=1.0).astype(np.float32)
        embedding = self._coords_to_embedding(coords)
        position_3d = self._visual_features(image_arr, edges)

        return {
            "character": char,
            "font_family": font_path.stem,
            "position_3d": position_3d,
            "embedding_128": embedding,
            "language": lang,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _render_character(self, char: str, font_path: Path, size: int) -> Image.Image:
        font = ImageFont.truetype(str(font_path), size=size, encoding="utf-8")
        canvas = Image.new("L", (size, size), color=255)
        draw = ImageDraw.Draw(canvas)
        try:
            bbox = draw.textbbox((0, 0), char, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = font.getsize(char)

        pos = ((size - text_w) // 2, (size - text_h) // 2)
        draw.text(pos, char, font=font, fill=0)
        return canvas

    def _coords_to_embedding(self, coords: np.ndarray) -> np.ndarray:
        """
        Flatten the emitted coordinates and compress to 128 dimensions.
        """
        flat = coords.flatten()
        if flat.size == 0:
            return np.zeros(128, dtype=np.float32)

        mean = flat.mean()
        std = flat.std() if flat.std() > 1e-6 else 1.0
        normalised = (flat - mean) / std

        if normalised.size >= 128:
            embedding = normalised[:128]
        else:
            embedding = np.zeros(128, dtype=np.float32)
            embedding[: normalised.size] = normalised

        return embedding.astype(np.float32)

    def _visual_features(self, img: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Compute simple 3D feature vector describing the glyph.
        """
        if edges.size == 0:
            return np.zeros(3, dtype=np.float32)

        complexity = np.clip(edges.mean(), 0.0, 1.0)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)
            circularity = 4 * np.pi * area / (perimeter**2 + 1e-6) if perimeter > 0 else 0.0
        else:
            circularity = 0.0

        h, w = img.shape
        aspect = w / (h + 1e-6)

        circularity = float(np.clip(circularity, 0.0, 1.0))
        aspect_norm = float(np.clip(aspect / 2.0, 0.0, 1.0))

        return np.array([float(complexity), circularity, aspect_norm], dtype=np.float32)


__all__ = ["SovereignVisualIngestor"]
