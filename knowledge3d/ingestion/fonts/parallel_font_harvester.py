"""
Parallel font harvesting utilities.

This module mirrors :mod:`knowledge3d.ingestion.fonts.font_harvester` but
pushes glyph rendering into a multiprocessing pool while batching the visual /
text fusion work on the GPU. The design keeps the CUDA path single-threaded to
avoid context contention yet keeps the device saturated with work thanks to the
CPU producer queue.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
import json
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

RenderTask = Tuple[str, str, str]
RenderedGlyph = Dict[str, object]
GlyphRecord = Dict[str, object]


# --------------------------------------------------------------------------- #
# Worker helper                                                               #
# --------------------------------------------------------------------------- #
def render_glyph_worker(task: RenderTask) -> RenderedGlyph | None:
    """
    Render a single glyph to a 64×64 grayscale numpy array.

    The function is top-level so it can be pickled by ``multiprocessing``. Any
    rendering issues (missing glyph or PIL errors) result in ``None`` so the
    caller can simply skip the glyph.
    """
    char, font_path, language = task

    try:
        from PIL import Image, ImageDraw, ImageFont

        canvas = Image.new("L", (64, 64), color=0)
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype(font_path, size=48)

        bbox = draw.textbbox((0, 0), char, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        if text_w <= 0 or text_h <= 0:
            return None

        x = (64 - text_w) // 2 - bbox[0]
        y = (64 - text_h) // 2 - bbox[1]
        draw.text((x, y), char, font=font, fill=255)

        glyph_array = np.array(canvas, dtype=np.uint8)

        return {
            "char": char,
            "font_path": font_path,
            "language": language,
            "glyph_array": glyph_array,
        }
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Parallel harvester                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class ParallelFontHarvester:
    """
    High-throughput font glyph harvester.

    Args:
        num_workers:
            CPU worker count for glyph rendering. Set to ``0``/``1`` to run
            sequentially.
        batch_size:
            Number of glyphs to fuse per GPU batch.
        characters:
            Character set rendered for each font.
        render_worker:
            Optional alternative rendering callable (useful for tests). Must be
            picklable when ``num_workers > 1``.
        gpu_batch_processor:
            Optional override for the GPU batch routine (primarily for unit
            tests).
        visual_ingestor, text_ingestor, swarm_processor:
            Optional pre-instantiated sovereign helpers (or stubs in tests).
    """

    num_workers: int = 8
    batch_size: int = 32
    characters: Sequence[str] | str = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    )
    render_worker: Callable[[RenderTask], RenderedGlyph | None] = render_glyph_worker
    gpu_batch_processor: Optional[Callable[[List[RenderedGlyph]], List[GlyphRecord]]] = None
    visual_ingestor: object | None = None
    text_ingestor: object | None = None
    swarm_processor: object | None = None

    # ------------------------------------------------------------------ #
    def harvest_fonts_parallel(
        self,
        font_dir: str | Path,
        *,
        output_path: str | Path,
        max_fonts: Optional[int] = None,
        language: str = "en",
    ) -> Dict[str, float]:
        """
        Harvest glyph embeddings for ``font_dir`` using the parallel pipeline.
        """
        font_paths = sorted(Path(font_dir).rglob("*.ttf"))
        if max_fonts is not None:
            font_paths = font_paths[:max_fonts]
        font_count = len(font_paths)
        if font_count == 0:
            raise ValueError(f"No .ttf fonts found under {font_dir!s}")

        task_chars = list(self.characters) if isinstance(self.characters, str) else list(self.characters)
        render_tasks: Iterable[RenderTask] = (
            (char, str(font_path), language)
            for font_path in font_paths
            for char in task_chars
        )

        self._output_path = Path(output_path)
        self._stream_handle = self._output_path.open("w", encoding="utf-8")
        self._stream_handle.write("{\n  \"glyphs\": [\n")
        self._first_record_written = False

        start_time = time.perf_counter()
        processed_glyphs = self._orchestrate_pipeline(render_tasks)
        total_time = time.perf_counter() - start_time
        throughput = processed_glyphs / max(total_time, 1e-9)

        # Finalise JSON stream
        self._stream_handle.write("\n  ],\n")
        self._stream_handle.write(f'  "font_count": {font_count},\n')
        self._stream_handle.write(f'  "total_time_s": {total_time:.6f},\n')
        self._stream_handle.write(f'  "throughput_glyphs_per_sec": {throughput:.6f}\n')
        self._stream_handle.write("}\n")
        self._stream_handle.close()
        self._stream_handle = None

        self._cleanup_gpu_components()

        return {
            "font_count": float(font_count),
            "glyph_count": float(processed_glyphs),
            "total_time_s": total_time,
            "throughput_glyphs_per_sec": throughput,
        }

    # ------------------------------------------------------------------ #
    # Pipeline coordination
    # ------------------------------------------------------------------ #
    def _orchestrate_pipeline(self, render_tasks: Iterable[RenderTask]) -> int:
        processed = 0
        batch: List[RenderedGlyph] = []

        def handle_rendered(item: Optional[RenderedGlyph]) -> None:
            nonlocal batch, processed
            if item is None:
                return
            batch.append(item)
            if len(batch) >= self.batch_size:
                processed += self._process_gpu_batch(batch)
                batch = []

        if self.num_workers and self.num_workers > 1:
            with Pool(processes=self.num_workers) as pool:
                for rendered in pool.imap(self.render_worker, render_tasks, chunksize=self.batch_size):
                    handle_rendered(rendered)
        else:
            for task in render_tasks:
                rendered = self.render_worker(task)
                handle_rendered(rendered)

        if batch:
            processed += self._process_gpu_batch(batch)

        return processed

    def _process_gpu_batch(self, batch: List[RenderedGlyph]) -> int:
        if not batch:
            return 0

        if self.gpu_batch_processor is not None:
            records = self.gpu_batch_processor(batch)
        else:
            records = self._default_gpu_batch_processor(batch)

        if not records:
            return 0

        assert self._stream_handle is not None
        for record in records:
            if not self._first_record_written:
                self._stream_handle.write("    ")
                self._first_record_written = True
            else:
                self._stream_handle.write(",\n    ")
            self._stream_handle.write(json.dumps(record))
        self._stream_handle.flush()
        return len(records)

    # ------------------------------------------------------------------ #
    # GPU batch implementation
    # ------------------------------------------------------------------ #
    def _default_gpu_batch_processor(self, batch: List[RenderedGlyph]) -> List[GlyphRecord]:
        self._ensure_gpu_components()

        import cv2  # local import to avoid forcing dependency for tests

        records: List[GlyphRecord] = []
        for item in batch:
            glyph_array = np.asarray(item["glyph_array"], dtype=np.uint8)
            edges = cv2.Canny(glyph_array, 50, 150)
            if not np.any(edges):
                continue

            # Fractal emitter expects flattened float32 inputs
            atom_values = edges.flatten().astype(np.float32) / 255.0
            coords = self.visual_ingestor.fractal_emitter.emit(atom_values, base_scale=1.0)  # type: ignore[attr-defined]
            visual_embedding = self.visual_ingestor._coords_to_embedding(coords)  # type: ignore[attr-defined]
            position_3d = self.visual_ingestor._visual_features(glyph_array, edges)  # type: ignore[attr-defined]

            text_result = self.text_ingestor.ingest_sentence(item["language"], item["char"])  # type: ignore[attr-defined]
            fused = self.swarm_processor.fuse_multimodal_embedding(  # type: ignore[attr-defined]
                text_emb=text_result["embedding_128"],
                visual_emb=visual_embedding,
                language=item["language"],
                include_diagnostics=False,
            )

            records.append(
                {
                    "char": item["char"],
                    "font_path": item["font_path"],
                    "visual_embedding": np.asarray(visual_embedding, dtype=np.float32).tolist(),
                    "text_embedding": np.asarray(text_result["embedding_128"], dtype=np.float32).tolist(),
                    "fused_embedding": np.asarray(fused["refined_embedding"], dtype=np.float32).tolist(),
                    "position_3d": np.asarray(fused["position_3d"], dtype=np.float32).tolist(),
                }
            )
        return records

    def _ensure_gpu_components(self) -> None:
        if self.visual_ingestor is None:
            from knowledge3d.ingestion.language.sovereign_visual_pipeline import SovereignVisualIngestor

            self.visual_ingestor = SovereignVisualIngestor()
        if self.text_ingestor is None:
            from knowledge3d.ingestion.language.sovereign_text_pipeline import SovereignTextIngestor

            self.text_ingestor = SovereignTextIngestor()
        if self.swarm_processor is None:
            from knowledge3d.ingestion.language.swarm_integration import SovereignLanguageSwarmProcessor

            self.swarm_processor = SovereignLanguageSwarmProcessor()

    def _cleanup_gpu_components(self) -> None:
        for component in (self.visual_ingestor, self.text_ingestor, self.swarm_processor):
            if component is not None and hasattr(component, "cleanup"):
                try:
                    component.cleanup()
                except Exception:  # pragma: no cover - defensive
                    pass


__all__ = ["ParallelFontHarvester", "render_glyph_worker"]
