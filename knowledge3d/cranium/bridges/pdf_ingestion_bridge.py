"""
Sovereign-native PDF ingestion bridge for the Phase C prototype.

Implements the hybrid ingestion approach discussed in the chain consensus:
structure-first parsing with an OCR fallback. Kernels are currently stubs but
the bridge keeps the interface sovereign so the PTX implementations can land
without touching higher layers.
"""

from __future__ import annotations

import ctypes
import hashlib
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine
from knowledge3d.cranium.ptx_runtime.atomic_fission_fusion import AtomicFissionFusion
from knowledge3d.cranium.sovereign.loader import (
    gpu_free,
    gpu_malloc,
    launch,
    load_ptx_file,
    memcpy_dtoh,
    memcpy_htod,
    synchronize,
)


class PDFIngestionBridge:
    """
    Sovereign PDF ingestion bridge for Phase C1.

    Pipeline (prototype):
        PDF bytes → (stub) primitive parse → embeddings → layout graph →
        fused embedding → Galaxy position estimate.

    GPU execution is optional; when PTX kernels are unavailable the bridge falls
    back to deterministic CPU stubs so unit tests can exercise the orchestration.
    """

    _MAX_OBJECTS = 1024

    def __init__(self) -> None:
        self.kernel_dir = Path(__file__).parent.parent / "kernels"
        self.rpn_engine = RPNEmbeddingEngine()
        self.fusion_engine = AtomicFissionFusion()
        self.allocated_buffers: List[Tuple[object, int]] = []
        self._temp_text_storage: List[str] = []
        self._temp_image_storage: List[bytes] = []
        self._current_pdf_path: Optional[str] = None
        self._enable_gpu_parser: bool = False

        self.pdf_parser_kernel = None
        self.layout_optimizer_kernel = None
        self.glyph_resonator_kernel = None
        self.gpu_enabled = False

        self._compile_pdf_kernels()
        self._load_kernels()

    # ------------------------------------------------------------------ #
    # Kernel management
    # ------------------------------------------------------------------ #
    def _compile_pdf_kernels(self) -> None:
        """Compile CU sources to PTX if nvcc is available and PTX is missing."""
        sources = [
            "pdf_primitive_parser.cu",
            "layout_graph_optimizer.cu",
            "glyph_resonator.cu",
        ]

        for source in sources:
            cu_path = self.kernel_dir / source
            ptx_path = cu_path.with_suffix(".ptx")
            if not cu_path.exists() or ptx_path.exists():
                continue

            try:
                subprocess.run(
                    [
                        "nvcc",
                        "-ptx",
                        "-arch=sm_86",
                        str(cu_path),
                        "-o",
                        str(ptx_path),
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except (FileNotFoundError, subprocess.CalledProcessError):
                # nvcc not present or compilation failed; fall back to CPU stubs
                return

    def _load_kernels(self) -> None:
        """Load PTX kernels if available."""
        self.pdf_parser_kernel = self._load_kernel(
            "pdf_primitive_parser.ptx", "pdf_primitive_parser"
        )
        self.layout_optimizer_kernel = self._load_kernel(
            "layout_graph_optimizer.ptx", "layout_graph_optimizer"
        )
        self.glyph_resonator_kernel = self._load_kernel(
            "glyph_resonator.ptx", "glyph_resonator"
        )

        self.gpu_enabled = self.pdf_parser_kernel is not None

    def enable_gpu_parser(self, enabled: bool = True) -> None:
        """
        Toggle the GPU parser once the PTX implementation is ready.

        For Phase C1.5 this remains disabled so PyMuPDF provides the parsed
        primitives. Phase C2 can call this method to activate the GPU path.
        """
        if enabled and self.pdf_parser_kernel is None:
            raise RuntimeError("PDF parser kernel not available to enable GPU parsing.")
        self._enable_gpu_parser = bool(enabled) and self.gpu_enabled

    def _load_kernel(self, filename: str, func_name: str):
        ptx_path = self.kernel_dir / filename
        if not ptx_path.exists():
            return None

        try:
            return load_ptx_file(str(ptx_path), func_name)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def ingest_pdf_page(self, pdf_path: str | Path, page_num: int = 0) -> Dict[str, object]:
        """
        Execute the ingestion pipeline for a single PDF page.
        """
        start_time = time.perf_counter()
        pdf_path = Path(pdf_path)
        self._current_pdf_path = str(pdf_path)
        self._temp_text_storage.clear()
        self._temp_image_storage.clear()

        pdf_bytes = self._load_pdf_bytes(pdf_path, page_num)
        use_gpu_parser = self._enable_gpu_parser and self.gpu_enabled
        pdf_buffer_gpu = self._upload_to_gpu(pdf_bytes) if use_gpu_parser else None

        parsed_objects = self._parse_pdf_structure(pdf_buffer_gpu, len(pdf_bytes), page_num)
        if parsed_objects.get("is_scanned"):
            parsed_objects = self._ocr_fallback(pdf_buffer_gpu, len(pdf_bytes), page_num)

        text_embeddings = self._generate_text_embeddings(parsed_objects)
        visual_embeddings = self._generate_visual_embeddings(parsed_objects)
        layout_graph = self._build_layout_graph(parsed_objects, text_embeddings, visual_embeddings)
        optimized_graph = self._optimize_layout_graph(layout_graph)
        fused_embeddings = self._fuse_modalities(text_embeddings, visual_embeddings)
        galaxy_position = self._crystallize_to_galaxy(optimized_graph, fused_embeddings)

        self._cleanup_gpu_buffers()

        processing_time_ms = (time.perf_counter() - start_time) * 1_000.0
        return {
            "galaxy_position": galaxy_position,
            "layout_graph": optimized_graph,
            "embeddings": fused_embeddings,
            "object_count": int(parsed_objects.get("object_count", 0)),
            "processing_time_ms": float(processing_time_ms),
        }

    # ------------------------------------------------------------------ #
    # Low-level helpers
    # ------------------------------------------------------------------ #
    def _load_pdf_bytes(self, pdf_path: str | Path, page_num: int) -> bytes:
        pdf_path = Path(pdf_path)
        try:
            import fitz  # type: ignore

            with fitz.open(pdf_path) as doc:
                if page_num < 0 or page_num >= len(doc):
                    raise IndexError(f"Page {page_num} out of range for {pdf_path}")
        except Exception:
            pass

        return pdf_path.read_bytes()

    def _upload_to_gpu(self, data: bytes):
        if not self.gpu_enabled:
            return None

        host_array = np.frombuffer(data, dtype=np.uint8).copy()
        try:
            buffer = gpu_malloc(host_array.nbytes)
        except RuntimeError:
            self.gpu_enabled = False
            return None
        try:
            memcpy_htod(
                buffer,
                host_array.ctypes.data_as(ctypes.c_void_p),
                host_array.nbytes,
            )
        except RuntimeError:
            gpu_free(buffer)
            self.gpu_enabled = False
            return None

        self.allocated_buffers.append((buffer, host_array.nbytes))
        return buffer

    def _parse_pdf_structure(
        self,
        pdf_buffer_gpu,
        buffer_size: int,
        page_num: int,
    ) -> Dict[str, object]:
        if (
            self._enable_gpu_parser
            and self.gpu_enabled
            and self.pdf_parser_kernel is not None
            and pdf_buffer_gpu is not None
        ):
            try:
                return self._parse_pdf_structure_gpu(pdf_buffer_gpu, buffer_size, page_num)
            except RuntimeError:
                self._enable_gpu_parser = False

        return self._parse_pdf_structure_pymupdf(page_num)

    def _parse_pdf_structure_gpu(
        self,
        pdf_buffer_gpu,
        buffer_size: int,
        page_num: int,
    ) -> Dict[str, object]:
        objects_size = self._MAX_OBJECTS * 8 * 4
        metadata_size = 4 * 4

        objects_gpu = gpu_malloc(objects_size)
        metadata_gpu = gpu_malloc(metadata_size)
        self.allocated_buffers.extend(
            [
                (objects_gpu, objects_size),
                (metadata_gpu, metadata_size),
            ]
        )

        try:
            launch(
                self.pdf_parser_kernel,
                grid=(1, 1, 1),
                block=(256, 1, 1),
                params=[
                    ctypes.c_uint64(int(objects_gpu.value)),
                    ctypes.c_uint64(int(pdf_buffer_gpu.value)),
                    ctypes.c_int(buffer_size),
                    ctypes.c_int(page_num),
                    ctypes.c_int(self._MAX_OBJECTS),
                    ctypes.c_uint64(int(metadata_gpu.value)),
                ],
            )
            synchronize()
        except RuntimeError:
            return self._simulate_parsed_objects()

        metadata = np.zeros(4, dtype=np.int32)
        memcpy_dtoh(
            metadata.ctypes.data_as(ctypes.c_void_p),
            metadata_gpu,
            metadata.nbytes,
        )

        object_count = int(metadata[0])
        object_count = max(0, min(object_count, self._MAX_OBJECTS))

        if object_count == 0:
            objects = np.zeros((0, 8), dtype=np.float32)
        else:
            host_objects = np.zeros(object_count * 8, dtype=np.float32)
            memcpy_dtoh(
                host_objects.ctypes.data_as(ctypes.c_void_p),
                objects_gpu,
                host_objects.nbytes,
            )
            objects = host_objects.reshape(-1, 8)

        return {
            "objects_gpu": objects_gpu,
            "objects": objects,
            "object_count": object_count,
            "processing_time_us": int(metadata[1]),
            "is_scanned": bool(metadata[2]),
        }

    def _parse_pdf_structure_pymupdf(self, page_num: int) -> Dict[str, object]:
        try:
            import fitz  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "PyMuPDF (fitz) is required for PDF parsing in Phase C1.5. "
                "Install via `pip install pymupdf`."
            ) from exc

        if self._current_pdf_path is None:
            raise ValueError("Current PDF path not set before parsing.")

        parse_start = time.perf_counter()
        pdf_path = Path(self._current_pdf_path)
        objects: List[List[float]] = []

        with fitz.open(pdf_path) as doc:
            if page_num < 0 or page_num >= len(doc):
                raise IndexError(f"Page {page_num} out of range for {pdf_path}")

            page = doc[page_num]
            blocks = page.get_text("dict").get("blocks", [])

            seen_images: set = set()

            for block in blocks:
                block_type = block.get("type")
                bbox = block.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue

                x0, y0, x1, y1 = [float(val) for val in bbox]
                w = x1 - x0
                h = y1 - y0
                if w <= 0.0 or h <= 0.0:
                    continue

                if block_type == 0:  # Text
                    text_fragments: List[str] = []
                    max_font = 0.0

                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text_fragment = span.get("text", "")
                            if text_fragment:
                                text_fragments.append(text_fragment)
                            try:
                                max_font = max(max_font, float(span.get("size", 0.0) or 0.0))
                            except (TypeError, ValueError):
                                pass

                    text_content = "".join(text_fragments).strip()
                    if not text_content:
                        continue

                    text_index = len(self._temp_text_storage)
                    self._temp_text_storage.append(text_content)

                    data_len = len(text_content)
                    importance = max_font / 32.0 if max_font else 0.5
                    importance = max(0.1, min(1.0, importance))

                    objects.append(
                        [
                            x0,
                            y0,
                            w,
                            h,
                            1.0,
                            float(text_index),
                            float(data_len),
                            float(importance),
                        ]
                    )

                elif block_type == 1:  # Image
                    image_info = block.get("image", {})
                    image_bytes = b""
                    xref = None
                    if isinstance(image_info, dict):
                        xref = image_info.get("xref")
                        if xref is not None:
                            try:
                                base_image = doc.extract_image(xref)
                                image_bytes = base_image.get("image", b"")
                            except Exception:
                                image_bytes = b""
                        elif "data" in image_info:
                            data_val = image_info.get("data", b"")
                            if isinstance(data_val, (bytes, bytearray)):
                                image_bytes = bytes(data_val)
                    elif isinstance(image_info, (bytes, bytearray)):
                        image_bytes = bytes(image_info)

                    if not image_bytes:
                        continue

                    image_key = (
                        ("xref", int(xref))
                        if xref is not None
                        else ("bytes", hashlib.md5(image_bytes).hexdigest())
                    )
                    if image_key in seen_images:
                        continue
                    seen_images.add(image_key)

                    image_index = len(self._temp_image_storage)
                    self._temp_image_storage.append(image_bytes)

                    objects.append(
                        [
                            x0,
                            y0,
                            w,
                            h,
                            2.0,
                            float(image_index),
                            float(len(image_bytes)),
                            0.8,
                        ]
                    )

            # Additional image sweep (handles some vector-backed images)
            image_list = page.get_images(full=True)
            for img in image_list:
                xref = img[0]
                name = img[7] if len(img) > 7 else None
                try:
                    bbox = page.get_image_bbox(name) if name else None
                except Exception:
                    bbox = None

                if not bbox:
                    continue

                x0, y0, x1, y1 = [float(val) for val in bbox]
                w = x1 - x0
                h = y1 - y0
                if w <= 0.0 or h <= 0.0:
                    continue

                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image.get("image", b"")
                except Exception:
                    image_bytes = b""

                if not image_bytes:
                    continue

                image_key = (
                    ("xref", int(xref))
                    if xref is not None
                    else ("bytes", hashlib.md5(image_bytes).hexdigest())
                )
                if image_key in seen_images:
                    continue
                seen_images.add(image_key)

                # avoid duplicates
                image_index = len(self._temp_image_storage)
                self._temp_image_storage.append(image_bytes)

                objects.append(
                    [
                        x0,
                        y0,
                        w,
                        h,
                        2.0,
                        float(image_index),
                        float(len(image_bytes)),
                        0.75,
                    ]
                )

        parse_time_us = int((time.perf_counter() - parse_start) * 1_000_000)
        text_count = sum(1 for obj in objects if obj[4] == 1.0)
        is_scanned = text_count < 2

        objects_array = (
            np.array(objects, dtype=np.float32)
            if objects
            else np.zeros((0, 8), dtype=np.float32)
        )

        return {
            "objects_gpu": None,
            "objects": objects_array,
            "object_count": len(objects),
            "processing_time_us": parse_time_us,
            "is_scanned": is_scanned,
        }

    def _simulate_parsed_objects(self) -> Dict[str, object]:
        objects = np.array(
            [
                [100.0, 700.0, 400.0, 20.0, 1.0, 0.0, 11.0, 0.9],
                [150.0, 500.0, 200.0, 150.0, 2.0, 0.0, 12345.0, 0.8],
            ],
            dtype=np.float32,
        )
        return {
            "objects_gpu": None,
            "objects": objects,
            "object_count": len(objects),
            "processing_time_us": 0,
            "is_scanned": False,
        }

    def _ocr_fallback(self, pdf_buffer_gpu, buffer_size: int, page_num: int) -> Dict[str, object]:
        return {
            "objects_gpu": None,
            "objects": np.zeros((0, 8), dtype=np.float32),
            "object_count": 0,
            "processing_time_us": 0,
            "is_scanned": True,
        }

    # ------------------------------------------------------------------ #
    # Embedding + graph helpers
    # ------------------------------------------------------------------ #
    def _generate_text_embeddings(self, parsed_objects: Dict[str, object]) -> np.ndarray:
        objects = parsed_objects.get("objects")
        if objects is None or len(objects) == 0:
            return np.zeros((0, self.rpn_engine.embedding_dim), dtype=np.float32)

        text_rows = objects[objects[:, 4] == 1.0]
        embeddings: List[np.ndarray] = []
        for row in text_rows:
            storage_idx = int(row[5])
            if 0 <= storage_idx < len(self._temp_text_storage):
                text_content = self._temp_text_storage[storage_idx]
            else:
                text_content = ""

            text_sample = text_content.strip()
            if not text_sample:
                embeddings.append(np.zeros(self.rpn_engine.embedding_dim, dtype=np.float32))
                continue

            text_snippet = text_sample if len(text_sample) <= 256 else text_sample[:256]
            try:
                embedding = self.rpn_engine.embed_sentence(text_snippet)
            except Exception:
                embedding = np.zeros(self.rpn_engine.embedding_dim, dtype=np.float32)
            embeddings.append(embedding.astype(np.float32))

        if not embeddings:
            return np.zeros((0, self.rpn_engine.embedding_dim), dtype=np.float32)
        return np.vstack(embeddings).astype(np.float32)

    def _generate_visual_embeddings(self, parsed_objects: Dict[str, object]) -> np.ndarray:
        objects = parsed_objects.get("objects")
        if objects is None or len(objects) == 0:
            return np.zeros((0, 128), dtype=np.float32)

        image_rows = objects[objects[:, 4] == 2.0]
        if len(image_rows) == 0:
            return np.zeros((0, 128), dtype=np.float32)

        embeddings = []
        for row in image_rows:
            storage_idx = int(row[5])
            if 0 <= storage_idx < len(self._temp_image_storage):
                image_bytes = self._temp_image_storage[storage_idx]
            else:
                image_bytes = b""
            embeddings.append(self._image_bytes_to_embedding(image_bytes))

        if not embeddings:
            return np.zeros((0, 128), dtype=np.float32)
        return np.vstack(embeddings).astype(np.float32)

    def _image_bytes_to_embedding(self, image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            return np.zeros(128, dtype=np.float32)

        digest = hashlib.md5(image_bytes[:4096]).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        embedding = rng.normal(loc=0.0, scale=1.0, size=128).astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm > 0.0:
            embedding /= norm
        return embedding

    def _build_layout_graph(
        self,
        parsed_objects: Dict[str, object],
        text_embeddings: np.ndarray,
        visual_embeddings: np.ndarray,
    ) -> Dict[str, object]:
        objects = parsed_objects.get("objects", np.zeros((0, 8), dtype=np.float32))

        nodes = []
        edges: List[Tuple[int, int, str]] = []

        text_idx = 0
        image_idx = 0
        zero_vec = np.zeros(128, dtype=np.float32)

        for i, obj in enumerate(objects):
            x, y, w, h, obj_type, data_index, data_len, importance = obj
            data_index_int = int(data_index)

            if obj_type == 1.0:
                embedding = (
                    text_embeddings[text_idx]
                    if text_idx < len(text_embeddings)
                    else zero_vec
                )
                text_idx += 1
                text_sample = ""
                if 0 <= data_index_int < len(self._temp_text_storage):
                    text_sample = self._temp_text_storage[data_index_int][:256]
            elif obj_type == 2.0:
                embedding = (
                    visual_embeddings[image_idx]
                    if image_idx < len(visual_embeddings)
                    else zero_vec
                )
                image_idx += 1
                text_sample = None
            else:
                embedding = zero_vec
                text_sample = None

            node_payload = {
                "id": i,
                "bbox": (float(x), float(y), float(w), float(h)),
                "type": float(obj_type),
                "embedding": embedding.astype(np.float32),
                "importance": float(importance),
                "data_index": data_index_int,
                "data_length": float(data_len),
            }
            if obj_type == 1.0:
                node_payload["text_sample"] = text_sample or ""
            elif obj_type == 2.0:
                node_payload["image_bytes"] = int(data_len)

            nodes.append(node_payload)

        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i == j:
                    continue
                relation = self._infer_spatial_relation(node_i["bbox"], node_j["bbox"])
                if relation is not None:
                    edges.append((i, j, relation))

        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "is_scanned": bool(parsed_objects.get("is_scanned", False)),
        }

    def _infer_spatial_relation(
        self, bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]
    ) -> Optional[str]:
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        if y1 + h1 <= y2:
            return "below"
        if y2 + h2 <= y1:
            return "above"
        if x1 + w1 <= x2:
            return "left"
        if x2 + w2 <= x1:
            return "right"
        return None

    def _optimize_layout_graph(self, layout_graph: Dict[str, object]) -> Dict[str, object]:
        if not self.gpu_enabled or self.layout_optimizer_kernel is None:
            return layout_graph
        # Kernel is a stub today, so the pass-through result is adequate.
        return layout_graph

    def _fuse_modalities(
        self, text_embeddings: np.ndarray, visual_embeddings: np.ndarray
    ) -> np.ndarray:
        if text_embeddings.size == 0 and visual_embeddings.size == 0:
            return np.zeros((1, 128), dtype=np.float32)
        if visual_embeddings.size == 0:
            return text_embeddings.mean(axis=0, keepdims=True)
        if text_embeddings.size == 0:
            return visual_embeddings.mean(axis=0, keepdims=True)

        stacked = np.vstack([text_embeddings, visual_embeddings]).astype(np.float32)
        return stacked.mean(axis=0, keepdims=True)

    def _crystallize_to_galaxy(
        self, layout_graph: Dict[str, object], fused_embeddings: np.ndarray
    ) -> np.ndarray:
        if fused_embeddings.size == 0:
            return np.zeros(3, dtype=np.float32)

        vec = fused_embeddings[0]
        if np.allclose(vec, 0.0):
            return np.zeros(3, dtype=np.float32)

        norm = np.linalg.norm(vec[:3]) or 1.0
        return (vec[:3] / norm).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    def _cleanup_gpu_buffers(self) -> None:
        if not (self._enable_gpu_parser and self.gpu_enabled):
            self.allocated_buffers.clear()
            return

        for ptr, _ in self.allocated_buffers:
            try:
                gpu_free(ptr)
            except RuntimeError:
                pass
        self.allocated_buffers.clear()
