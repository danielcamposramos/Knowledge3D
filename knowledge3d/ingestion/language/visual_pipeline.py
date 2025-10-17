"""
Visual-language ingestion pipeline.

Handles glyph rendering, font analysis, and sign-language video embeddings to
generate multi-modal signals compatible with the nine-chain swarm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from PIL import Image, ImageDraw, ImageFont

from .text_pipeline import _require_optional_dependency


@dataclass
class VisualLanguageIngestor:
    """
    Visual-language ingestion helper.

    Parameters
    ----------
    clip_model_name:
        HuggingFace CLIP identifier.
    device:
        Torch device string (e.g. 'cpu', 'cuda').
    """

    clip_model_name: str = "openai/clip-vit-base-patch32"
    device: str = "cpu"

    _clip_model: "CLIPModel | None" = field(init=False, default=None)
    _clip_processor: "CLIPProcessor | None" = field(init=False, default=None)

    def _load_clip(self):
        if self._clip_model is None or self._clip_processor is None:
            _require_optional_dependency("transformers", "pip install transformers")
            from transformers import CLIPModel, CLIPProcessor  # type: ignore

            self._clip_model = CLIPModel.from_pretrained(self.clip_model_name)
            self._clip_model.to(self.device)
            self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        return self._clip_model, self._clip_processor

    # ------------------------------------------------------------------ #
    # Glyph ingestion
    # ------------------------------------------------------------------ #
    def ingest_glyph(self, character: str, font_path: str | Path, lang: str) -> Dict:
        font_path = Path(font_path)
        if not font_path.exists():
            raise FileNotFoundError(font_path)
        if not character:
            raise ValueError("Character must be a non-empty string")

        image = self._render_character(character, font_path)
        embedding = self._clip_image_embedding(image)
        embedding_128 = self._resize_embedding(embedding, 128)
        position = self._glyph_to_3d(image)

        return {
            "character": character,
            "font_family": font_path.stem,
            "position_3d": position,
            "embedding_128": embedding_128,
            "language": lang,
        }

    # ------------------------------------------------------------------ #
    # Sign-language ingestion
    # ------------------------------------------------------------------ #
    def ingest_sign_language_video(
        self, video_path: str | Path, sign_label: str, lang: str, max_frames: int = 48
    ) -> Dict:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        frames = self._load_video_frames(video_path, max_frames=max_frames)
        if not frames:
            raise ValueError(f"No frames extracted from {video_path}")

        embeddings = [self._clip_image_embedding(frame) for frame in frames]
        embedding_128 = self._resize_embedding(np.mean(embeddings, axis=0), 128)
        trajectory = self._extract_hand_trajectory(frames)

        return {
            "sign": sign_label,
            "trajectory_3d": trajectory,
            "embedding_128": embedding_128,
            "language": lang,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _clip_image_embedding(self, image: Image.Image) -> np.ndarray:
        model, processor = self._load_clip()
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        with np.errstate(all="ignore"):
            outputs = model.get_image_features(**inputs)
        return outputs.detach().cpu().numpy().astype(np.float32).flatten()

    @staticmethod
    def _render_character(
        character: str, font_path: Path, size: int = 96
    ) -> Image.Image:
        font = ImageFont.truetype(str(font_path), size=size, encoding="utf-8")
        canvas = Image.new("L", (size * 2, size * 2), color=255)
        draw = ImageDraw.Draw(canvas)
        width, height = draw.textsize(character, font=font)
        position = ((canvas.width - width) // 2, (canvas.height - height) // 2)
        draw.text(position, character, font=font, fill=0)
        return canvas

    def _glyph_to_3d(self, image: Image.Image) -> np.ndarray:
        _require_optional_dependency("cv2", "pip install opencv-python")
        import cv2  # type: ignore

        array = np.array(image)
        edges = cv2.Canny(array, 50, 150)

        # Complexity: edge density
        complexity = float(edges.mean())

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)
            circularity = (
                4 * np.pi * area / (perimeter**2 + 1e-6) if perimeter > 0 else 0.0
            )
        else:
            circularity = 0.0

        h, w = array.shape
        aspect_ratio = float(w / (h + 1e-6))

        vector = np.array([complexity, circularity, aspect_ratio], dtype=np.float32)
        vector -= vector.min()
        denom = vector.max() or 1.0
        return vector / denom

    def _load_video_frames(
        self, video_path: Path, max_frames: int = 48, every_n: int = 1
    ) -> List[Image.Image]:
        _require_optional_dependency("cv2", "pip install opencv-python")
        import cv2  # type: ignore

        capture = cv2.VideoCapture(str(video_path))
        frames: List[Image.Image] = []
        frame_index = 0
        try:
            while len(frames) < max_frames:
                success, frame = capture.read()
                if not success:
                    break
                if frame_index % every_n != 0:
                    frame_index += 1
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
                frame_index += 1
        finally:
            capture.release()
        return frames

    def _extract_hand_trajectory(self, frames: Sequence[Image.Image]) -> List[np.ndarray]:
        """
        Extract 3D hand landmark trajectories using MediaPipe Hands.
        Returns a list of normalised (x, y, z) coordinates per frame.
        """
        _require_optional_dependency(
            "mediapipe",
            "pip install mediapipe",
        )
        import mediapipe as mp  # type: ignore

        hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.4,
        )

        trajectory: List[np.ndarray] = []
        for frame in frames:
            array = np.array(frame)
            results = hands.process(array)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                coords = [
                    np.array([lm.x, lm.y, lm.z], dtype=np.float32)
                    for lm in hand_landmarks.landmark
                ]
                centroid = np.mean(coords, axis=0)
                trajectory.append(centroid)
            else:
                trajectory.append(np.zeros(3, dtype=np.float32))

        hands.close()
        if trajectory:
            stacked = np.vstack(trajectory)
            stacked -= stacked.min(axis=0, keepdims=True)
            denom = stacked.max(axis=0, keepdims=True)
            denom[denom == 0.0] = 1.0
            trajectory = [row.astype(np.float32) for row in stacked / denom]
        return trajectory

    @staticmethod
    def _resize_embedding(embedding: np.ndarray, target_dim: int) -> np.ndarray:
        if embedding.ndim != 1:
            raise ValueError("Embedding must be 1D")
        if embedding.size == target_dim:
            return embedding.astype(np.float32, copy=False)
        if embedding.size < target_dim:
            return np.pad(
                embedding, (0, target_dim - embedding.size)
            ).astype(np.float32, copy=False)

        segments = np.array_split(embedding, target_dim)
        collapsed = np.array([segment.mean() for segment in segments], dtype=np.float32)
        if collapsed.size != target_dim:
            collapsed = np.resize(collapsed, target_dim)
        return collapsed.astype(np.float32, copy=False)


__all__ = ["VisualLanguageIngestor"]
