"""
Harvest system fonts to create glyph database for GPU-native OCR.

This script regenerates the Phase B font glyph embeddings so that the OCR
fallback in the PDF ingestion pipeline has access to the learned glyphs.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2  # type: ignore
except ImportError as exc:
    raise RuntimeError("opencv-python-headless is required for font harvesting.") from exc

sys.path.insert(0, ".")

from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine


class FontGlyphHarvester:
    """Simplified font harvester for Phase C3 OCR."""

    def __init__(self) -> None:
        self.rpn_engine = RPNEmbeddingEngine()
        self.glyph_database: Dict[str, Dict[str, object]] = {}

    def harvest_fonts(self, font_dirs: List[str], output_path: str, max_fonts: int = 2000) -> None:
        font_files: List[str] = []
        for font_dir in font_dirs:
            directory = Path(font_dir).expanduser()
            if not directory.exists():
                continue
            for path in directory.rglob("*"):
                if path.suffix.lower() in {".ttf", ".otf"}:
                    font_files.append(str(path))

        print(f"[INFO] Found {len(font_files)} font files")
        font_files = font_files[:max_fonts]

        chars = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789"
        )

        total_glyphs = 0

        for idx, font_path in enumerate(font_files, 1):
            font_name = Path(font_path).stem
            print(f"[{idx}/{len(font_files)}] {font_name}")

            font_data = self._harvest_font(font_path, chars)
            if font_data is None:
                print("  ✗ skipped")
                continue

            glyphs = font_data["glyphs"]
            if not glyphs:
                print("  ✗ no glyphs")
                continue

            total_glyphs += len(glyphs)
            self.glyph_database[font_name] = font_data
            print(f"  ✓ {len(glyphs)} glyphs")

        print("\n[RESULT] harvested glyphs:", total_glyphs)
        self._save_database(output_path)

    def _is_symbol_font(self, font_path: str) -> bool:
        name = Path(font_path).stem.lower()
        indicators = [
            "webding",
            "wingding",
            "symbol",
            "dingbat",
            "awesome",
            "icon",
            "emoji",
            "symbola",
        ]
        return any(token in name for token in indicators)

    def _harvest_font(self, font_path: str, chars: str, font_size: int = 32) -> Dict[str, object] | None:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            return None

        glyphs: Dict[str, Dict[str, object]] = {}
        is_symbol = self._is_symbol_font(font_path)
        for char in chars:
            if is_symbol and not char.isalnum():
                continue
            try:
                image = self._render_char(char, font, font_size)
                visual = self._extract_visual_features(image)
                embedding = self._generate_glyph_embedding(char, visual, is_symbol=is_symbol)
                glyphs[char] = {
                    "embedding": embedding,
                    "visual_features": visual,
                    "confidence": 0.8 if is_symbol else 1.0,
                    "is_symbol": is_symbol,
                }
            except Exception:
                continue

        return {
            "font_path": font_path,
            "is_symbol_font": is_symbol,
            "glyphs": glyphs,
        }

    @staticmethod
    def _render_char(char: str, font: ImageFont.FreeTypeFont, font_size: int) -> np.ndarray:
        canvas = Image.new("L", (16, 16), color=255)
        draw = ImageDraw.Draw(canvas)

        try:
            bbox = draw.textbbox((0, 0), char, font=font)
            char_w = bbox[2] - bbox[0]
            char_h = bbox[3] - bbox[1]
        except Exception:
            char_w = char_h = font_size

        x = max(0, (16 - char_w) // 2)
        y = max(0, (16 - char_h) // 2)
        draw.text((x, y), char, font=font, fill=0)
        return np.array(canvas)

    @staticmethod
    def _extract_visual_features(img: np.ndarray) -> np.ndarray:
        """
        Extract HOG (Histogram of Oriented Gradients) features for a glyph image.

        HOG preserves local edge orientation structure, providing much stronger
        discrimination between visually similar characters than frequency-domain
        features.
        """
        import cv2  # local import to match bridge behaviour

        if img.shape != (16, 16):
            img = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)

        img_float = img.astype(np.float32)
        if float(img_float.max() - img_float.min()) > 1e-6:
            img_norm = cv2.normalize(img_float, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        else:
            img_norm = np.zeros_like(img_float)
        img_uint8 = img_norm.astype(np.uint8)

        _, img_bin = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        hog_input = cv2.bitwise_not(img_bin)

        win_size = (16, 16)
        block_size = (8, 8)
        block_stride = (8, 8)
        cell_size = (4, 4)
        nbins = 9

        hog = cv2.HOGDescriptor(
            _winSize=win_size,
            _blockSize=block_size,
            _blockStride=block_stride,
            _cellSize=cell_size,
            _nbins=nbins,
        )

        hog_features = hog.compute(hog_input).flatten()
        feature_128 = np.zeros(128, dtype=np.float32)

        hog_len = hog_features.size
        if hog_len > 0:
            if hog_len >= 126:
                feature_128[:126] = hog_features[:126]
            else:
                repeats = 126 // hog_len
                remainder = 126 % hog_len
                offset = 0

                for _ in range(repeats):
                    end = offset + hog_len
                    feature_128[offset:end] = hog_features
                    offset = end

                if remainder:
                    feature_128[offset:offset + remainder] = hog_features[:remainder]

        moments = cv2.moments(hog_input)
        if moments["m00"] != 0.0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            feature_128[126] = float(cx / 16.0)
            feature_128[127] = float(cy / 16.0)

        norm = np.linalg.norm(feature_128)
        if norm > 1e-8:
            feature_128 = feature_128 / norm

        return feature_128

    def _generate_glyph_embedding(
        self,
        char: str,
        visual_features: np.ndarray,
        *,
        is_symbol: bool = False,
    ) -> np.ndarray:
        embedding = visual_features.copy()
        if not is_symbol:
            try:
                rpn_output = self.rpn_engine.embed_sentence(char)
                char_vec = rpn_output["embedding_128"]
                embedding = 0.7 * embedding + 0.3 * char_vec
            except Exception:
                pass

        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm
        return embedding

    def _save_database(self, output_path: str) -> None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self.glyph_database, handle)

        total_fonts = len(self.glyph_database)
        total_glyphs = sum(len(font["glyphs"]) for font in self.glyph_database.values())
        print(f"[SAVED] {target} -> {total_fonts} fonts, {total_glyphs} glyphs")


def main() -> None:
    harvester = FontGlyphHarvester()
    font_dirs = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.fonts"),
    ]

    output_path = "/K3D/Knowledge3D.local/font_db.pkl"

    print("=" * 60)
    print("K3D Font Glyph Harvesting")
    print("=" * 60)
    harvester.harvest_fonts(font_dirs, output_path, max_fonts=2000)
    print("=" * 60)
    print("Harvest complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
