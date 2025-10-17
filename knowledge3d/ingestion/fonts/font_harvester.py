"""
Harvest glyph embeddings from system fonts for multi-modal grounding.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - hints only
    from knowledge3d.ingestion.language.sovereign_text_pipeline import SovereignTextIngestor
    from knowledge3d.ingestion.language.sovereign_visual_pipeline import SovereignVisualIngestor
    from knowledge3d.ingestion.language.swarm_integration import SovereignLanguageSwarmProcessor


DEFAULT_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
DEFAULT_OUTPUT_PATH = Path("/K3D/Knowledge3D.local/house_zone7/fonts/font_glyphs.json")


def _to_list(array: np.ndarray) -> List[float]:
    return np.asarray(array, dtype=np.float32).tolist()


@dataclass
class FontGlyphHarvester:
    """
    Bridge between visual glyph renderings and text embeddings.

    Parameters
    ----------
    visual_ingestor:
        Optional pre-instantiated :class:`SovereignVisualIngestor`.
    text_ingestor:
        Optional :class:`SovereignTextIngestor`.
    swarm_processor:
        Optional :class:`SovereignLanguageSwarmProcessor` to fuse embeddings.
    output_path:
        Default location for harvested dataset.
    """

    visual_ingestor: "SovereignVisualIngestor | None" = None
    text_ingestor: "SovereignTextIngestor | None" = None
    swarm_processor: "SovereignLanguageSwarmProcessor | None" = None
    output_path: Path = DEFAULT_OUTPUT_PATH

    def __post_init__(self) -> None:
        if self.visual_ingestor is None:
            try:
                from knowledge3d.ingestion.language.sovereign_visual_pipeline import (
                    SovereignVisualIngestor,
                )
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "OpenCV/PIL dependencies required for sovereign visual ingestion."
                ) from exc
            self.visual_ingestor = SovereignVisualIngestor()

        if self.text_ingestor is None:
            from knowledge3d.ingestion.language.sovereign_text_pipeline import (
                SovereignTextIngestor,
            )

            self.text_ingestor = SovereignTextIngestor()

        if self.swarm_processor is None:
            from knowledge3d.ingestion.language.swarm_integration import (
                SovereignLanguageSwarmProcessor,
            )

            self.swarm_processor = SovereignLanguageSwarmProcessor()
        self.output_path = Path(self.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def harvest_font_glyphs(
        self,
        font_path: str | Path,
        *,
        characters: Sequence[str] | str | None = None,
        language: str = "en",
    ) -> dict:
        """
        Harvest glyph data for a single font file.
        """
        font_path = Path(font_path)
        if not font_path.exists():
            raise FileNotFoundError(font_path)

        if characters is None:
            glyphs = list(DEFAULT_CHARSET)
        elif isinstance(characters, str):
            glyphs = list(characters)
        else:
            glyphs = list(characters)

        harvested = []
        for char in glyphs:
            if not char:
                continue
            try:
                visual_result = self.visual_ingestor.ingest_glyph(char, font_path, language)
                text_result = self.text_ingestor.ingest_sentence(language, char)
                fused = self.swarm_processor.fuse_multimodal_embedding(
                    text_emb=text_result["embedding_128"],
                    visual_emb=visual_result["embedding_128"],
                    language=language,
                    include_diagnostics=False,
                )
            except Exception as exc:  # pragma: no cover - robustness for missing glyphs
                print(f"[FontGlyphHarvester] Skipping '{char}' from {font_path.name}: {exc}")
                continue

            harvested.append(
                {
                    "char": char,
                    "visual_embedding": _to_list(visual_result["embedding_128"]),
                    "text_embedding": _to_list(text_result["embedding_128"]),
                    "fused_embedding": _to_list(fused["refined_embedding"]),
                    "visual_position": _to_list(visual_result["position_3d"]),
                    "text_position": text_result["nodes"][0][1].tolist() if text_result["nodes"] else [0.0, 0.0, 0.0],
                    "fused_position": _to_list(fused["position_3d"]),
                }
            )

        return {
            "font_path": str(font_path),
            "font_name": font_path.stem,
            "glyph_count": len(harvested),
            "glyphs": harvested,
            "language": language,
        }

    def harvest_font_directory(
        self,
        font_dir: str | Path,
        *,
        max_fonts: int | None = None,
        characters: Sequence[str] | str | None = None,
        language: str = "en",
        output_path: str | Path | None = None,
    ) -> dict:
        """
        Harvest multiple fonts within a directory and persist the dataset.
        """
        font_dir = Path(font_dir)
        font_files = sorted(font_dir.glob("*.ttf")) + sorted(font_dir.glob("*.otf"))
        if max_fonts is not None:
            font_files = font_files[:max_fonts]

        all_fonts = []
        for font_file in font_files:
            result = self.harvest_font_glyphs(
                font_file,
                characters=characters,
                language=language,
            )
            if result["glyph_count"] > 0:
                all_fonts.append(result)

        payload = {
            "font_count": len(all_fonts),
            "fonts": all_fonts,
            "language": language,
        }

        target = Path(output_path) if output_path else self.output_path
        self._write_json(target, payload)
        return {"output_path": str(target), "font_count": len(all_fonts)}

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


__all__ = ["FontGlyphHarvester"]
