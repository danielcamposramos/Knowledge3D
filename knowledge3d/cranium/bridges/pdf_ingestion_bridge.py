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

# Phase E: DeepSeek-OCR integration (optional)
try:
    from knowledge3d.cranium.ocr.deepseek_bridge import DeepSeekOCRBridge
    DEEPSEEK_OCR_AVAILABLE = True
except ImportError:
    DEEPSEEK_OCR_AVAILABLE = False


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
    _EMBEDDINGS_PATH = Path("/K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl")

    def __init__(self) -> None:
        self.kernel_dir = Path(__file__).parent.parent / "kernels"
        self.rpn_engine = RPNEmbeddingEngine()
        self.embeddings_path = self._EMBEDDINGS_PATH
        self._load_rpn_embeddings()
        self.fusion_engine = AtomicFissionFusion()
        self.allocated_buffers: List[Tuple[object, int]] = []
        self._temp_text_storage: List[str] = []
        self._temp_image_storage: List[bytes] = []
        self._current_pdf_path: Optional[str] = None
        self._enable_gpu_parser: bool = False
        self._enable_deepseek_ocr: bool = False  # Phase E: DeepSeek-OCR toggle

        self.pdf_parser_kernel = None
        self.layout_optimizer_kernel = None
        self.glyph_resonator_kernel = None
        self.gpu_enabled = False

        self.glyph_max_dim: int = 2048
        self.glyph_embeddings: Optional[np.ndarray] = None
        self.glyph_dims: Optional[np.ndarray] = None
        self.glyph_metadata: List[Dict[str, object]] = []
        self.glyph_embeddings_gpu = None
        self.glyph_dims_gpu = None
        self._glyph_embeddings_bytes = 0
        self._glyph_dims_bytes = 0
        self.glyph_count: int = 0
        self._ocr_warned_missing_glyph = False
        self._ocr_warned_missing_cv = False

        self._min_proto_confidence: float = 0.4

        self._compile_pdf_kernels()
        self._load_kernels()
        self._load_glyph_embeddings()

        # Phase E: Initialize DeepSeek OCR bridge (optional)
        self.deepseek_bridge = None
        if DEEPSEEK_OCR_AVAILABLE:
            try:
                self.deepseek_bridge = DeepSeekOCRBridge(mode='small')
                print("[PHASE_E] DeepSeek OCR bridge initialized (mode: small)")
            except Exception as exc:
                print(f"[PHASE_E] WARNING: Could not initialize DeepSeek OCR - {exc}")

        # Initialize sleep scheduler (last step)
        try:
            from knowledge3d.cranium.sleep.scheduler import SleepScheduler

            self.sleep_scheduler = SleepScheduler(
                rpn_engine=self.rpn_engine,
                idle_threshold=300.0,
                log_path="/K3D/Knowledge3D.local/logs/sleep_scheduler.jsonl",
            )
            self.sleep_scheduler.start()
            print("[SLEEP] Sleep scheduler initialized and started")
        except Exception as exc:
            print(f"[SLEEP] WARNING: Could not start sleep scheduler - {exc}")
            self.sleep_scheduler = None

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
            if not cu_path.exists():
                continue

            needs_rebuild = (
                not ptx_path.exists()
                or cu_path.stat().st_mtime > ptx_path.stat().st_mtime
            )

            if not needs_rebuild:
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
            "glyph_resonator.ptx", "glyph_resonator_matryoshka"
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

    def enable_deepseek_ocr(self, enabled: bool = True) -> None:
        """
        Toggle DeepSeek-OCR for Phase E enhanced text extraction.

        Phase E: Uses DeepSeek pipeline (SAM-base + 16× Conv + CLIP-large)
        for improved OCR accuracy and dual-texture generation.

        Args:
            enabled: True to enable DeepSeek OCR, False to use Tesseract fallback
        """
        if enabled and self.deepseek_bridge is None:
            raise RuntimeError("DeepSeek OCR bridge not available. Install Phase E components.")
        self._enable_deepseek_ocr = bool(enabled) and self.deepseek_bridge is not None
        if self._enable_deepseek_ocr:
            print("[PHASE_E] DeepSeek OCR enabled")

    def _load_kernel(self, filename: str, func_name: str):
        ptx_path = self.kernel_dir / filename
        if not ptx_path.exists():
            return None

        try:
            return load_ptx_file(str(ptx_path), func_name)
        except Exception:
            return None

    def _load_glyph_embeddings(self) -> None:
        """
        Load Phase B glyph embeddings into GPU memory for OCR fallback.
        """
        try:
            import pickle
        except ImportError:  # pragma: no cover - should never happen
            return

        font_db_path = Path("/K3D/Knowledge3D.local/font_db.pkl")
        if not font_db_path.exists():
            print(f"[WARN] Glyph font database not found at {font_db_path}. OCR disabled.")
            return

        try:
            with font_db_path.open("rb") as handle:
                font_db = pickle.load(handle)
        except Exception as exc:
            print(f"[WARN] Failed loading glyph database ({exc}). OCR disabled.")
            return

        glyph_variants: List[Dict[str, object]] = []

        for font_name, font_payload in font_db.items():
            glyphs = font_payload.get("glyphs", {}) if isinstance(font_payload, dict) else {}
            is_symbol_font = bool(font_payload.get("is_symbol_font", False))
            font_path = font_payload.get("font_path")

            for char, glyph_data in glyphs.items():
                features = glyph_data.get("visual_features")
                if features is None:
                    features = glyph_data.get("embedding")
                if features is None:
                    continue

                base_embedding = np.asarray(features, dtype=np.float32).reshape(-1)
                if base_embedding.size == 0:
                    continue

                matryoshka_emb, effective_dim, available_dims = self._generate_matryoshka_glyph(
                    base_embedding,
                    char
                )

                glyph_variants.append(
                    {
                        "char": char,
                        "font": font_name,
                        "embedding": matryoshka_emb,
                        "effective_dim": effective_dim,
                        "available_dims": available_dims,
                        "native_dim": int(min(base_embedding.size, self.glyph_max_dim)),
                        "confidence": float(glyph_data.get("confidence", 1.0)),
                        "is_symbol": bool(glyph_data.get("is_symbol", False)),
                        "is_symbol_font": is_symbol_font,
                        "font_path": font_path,
                    }
                )

        if not glyph_variants:
            print("[WARN] No glyph embeddings found in database. OCR disabled.")
            return

        embeddings = np.vstack([variant["embedding"] for variant in glyph_variants]).astype(np.float32)
        dims = np.array([variant["effective_dim"] for variant in glyph_variants], dtype=np.int32)

        for variant in glyph_variants:
            variant.pop("embedding", None)

        self.glyph_embeddings = embeddings
        self.glyph_dims = dims
        self.glyph_metadata = glyph_variants
        self.glyph_count = len(glyph_variants)
        self.glyph_embeddings_gpu = None
        self.glyph_dims_gpu = None

        unique_chars = len({variant["char"] for variant in glyph_variants})
        max_dim_observed = int(dims.max())
        print(
            f"[INFO] Loaded {self.glyph_count} glyph variants across {unique_chars} characters "
            f"(max_dim={max_dim_observed})"
        )

    def _generate_matryoshka_glyph(
        self,
        base_embedding: np.ndarray,
        char: str,
    ) -> Tuple[np.ndarray, int, List[int]]:
        """
        Generate Matryoshka multi-scale embedding for a single glyph variant.
        """
        target_dim = self.glyph_max_dim
        base = np.asarray(base_embedding, dtype=np.float32).flatten()
        if base.size == 0:
            return np.zeros(target_dim, dtype=np.float32), 0, []

        base_dim = min(base.size, target_dim)
        matryoshka = np.zeros(target_dim, dtype=np.float32)
        matryoshka[:base_dim] = base[:base_dim]

        norm = np.linalg.norm(matryoshka[:base_dim])
        if norm > 1e-8:
            matryoshka[:base_dim] /= norm

        complexity = self._estimate_glyph_complexity(char)
        if base_dim >= target_dim:
            effective_dim = target_dim
        elif complexity >= 0.8:
            effective_dim = target_dim
        elif complexity >= 0.6:
            effective_dim = 1024
        elif complexity >= 0.4:
            effective_dim = 512
        else:
            effective_dim = 256

        effective_dim = max(base_dim, effective_dim)
        effective_dim = min(effective_dim, target_dim)

        import hashlib

        seed_bytes = hashlib.sha256(base.tobytes() + char.encode("utf-8")).digest()
        seed = int.from_bytes(seed_bytes[:4], "little", signed=False)
        rng = np.random.RandomState(seed)

        step = 64
        cursor = base_dim
        while cursor < effective_dim:
            chunk = min(step, effective_dim - cursor)
            scale = 0.12 / np.sqrt((cursor + chunk) / max(base_dim, 1))
            matryoshka[cursor:cursor + chunk] = rng.randn(chunk).astype(np.float32) * scale
            cursor += chunk

        refined_norm = np.linalg.norm(matryoshka[:effective_dim])
        if refined_norm > 1e-8:
            matryoshka[:effective_dim] /= refined_norm

        levels = [64, 128, 256, 512, 1024, 2048]
        available = [dim for dim in levels if dim <= effective_dim]
        if not available:
            available = [effective_dim]

        return matryoshka, effective_dim, available

    @staticmethod
    def _estimate_glyph_complexity(char: str) -> float:
        """
        Estimate visual complexity of a glyph using heuristic rules.
        """
        if not char or char.isspace():
            return 0.1
        simple_low = set("il1!|")
        simple_letters = set("acemnorsuvwxz")
        medium_letters = set("bdfghkpqty")
        medium_upper = set("ACEMNORSUVWXZ")
        high_symbols = set("@&%$#")

        if char in simple_low:
            return 0.2
        if char in simple_letters:
            return 0.4
        if char in medium_letters:
            return 0.6
        if char in medium_upper:
            return 0.7
        if char in high_symbols:
            return 0.9
        if char.isdigit():
            return 0.55
        if char.isalpha():
            return 0.5
        return 0.65

    def _load_rpn_embeddings(self) -> None:
        """Load persisted RPN embeddings when available."""
        path = self.embeddings_path
        try:
            if path.exists():
                self.rpn_engine.load_embeddings(path)
                print(
                    f"[LOAD] RPN embeddings loaded: {len(self.rpn_engine.embeddings)} trigrams"
                )
        except Exception as exc:
            print(f"[LOAD] WARNING: Failed to load RPN embeddings ({exc})")

    def save_rpn_embeddings(self) -> None:
        """Persist RPN embeddings to disk."""
        path = self.embeddings_path
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.rpn_engine.save_embeddings(path)
            print(
                f"[SAVE] RPN embeddings saved: {len(self.rpn_engine.embeddings)} trigrams"
            )
        except Exception as exc:
            print(f"[SAVE] WARNING: Failed to save RPN embeddings ({exc})")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def ingest_pdf_page(self, pdf_path: str | Path, page_num: int = 0) -> Dict[str, object]:
        """
        Execute the ingestion pipeline for a single PDF page.
        """
        if hasattr(self, "sleep_scheduler") and self.sleep_scheduler:
            self.sleep_scheduler.mark_activity()

        start_time = time.perf_counter()
        pdf_path = Path(pdf_path)
        self._current_pdf_path = str(pdf_path)
        self._temp_text_storage.clear()
        self._temp_image_storage.clear()

        pdf_bytes = self._load_pdf_bytes(pdf_path, page_num)
        use_gpu_parser = self._enable_gpu_parser and self.gpu_enabled and self._enable_deepseek_ocr
        pdf_buffer_gpu = self._upload_to_gpu(pdf_bytes) if use_gpu_parser else None

        parsed_objects = self._parse_pdf_structure(pdf_bytes, pdf_buffer_gpu, len(pdf_bytes), page_num)
        parsed_objects.setdefault("method", "structured")
        needs_fallback = (
            bool(parsed_objects.get("is_scanned"))
            or int(parsed_objects.get("object_count", 0)) == 0
        )
        if needs_fallback and self._enable_deepseek_ocr:
            parsed_objects = self._ocr_fallback(str(pdf_path), page_num)

        text_embeddings = self._generate_text_embeddings(parsed_objects)
        visual_embeddings = self._generate_visual_embeddings(parsed_objects)
        layout_graph = self._build_layout_graph(parsed_objects, text_embeddings, visual_embeddings)
        optimized_graph = self._optimize_layout_graph(layout_graph)
        fused_embeddings = self._fuse_modalities(text_embeddings, visual_embeddings)
        galaxy_position = self._crystallize_to_galaxy(optimized_graph, fused_embeddings)

        self._cleanup_gpu_buffers()

        processing_time_ms = (time.perf_counter() - start_time) * 1_000.0
        result = {
            "galaxy_position": galaxy_position,
            "layout_graph": optimized_graph,
            "embeddings": fused_embeddings,
            "object_count": int(parsed_objects.get("object_count", 0)),
            "processing_time_ms": float(processing_time_ms),
            "method": parsed_objects.get("method", "structured"),
            "text": parsed_objects.get("text", ""),
        }
        if page_num > 0 and page_num % 100 == 0:
            self.save_rpn_embeddings()
        return result

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
        pdf_bytes: bytes,
        pdf_buffer_gpu,
        buffer_size: int,
        page_num: int,
    ) -> Dict[str, object]:
        if not self._enable_gpu_parser or not self._enable_deepseek_ocr:
            return self._parse_pdf_structure_pymupdf(page_num)

        if (
            self.gpu_enabled
            and self.pdf_parser_kernel is not None
            and pdf_buffer_gpu is not None
        ):
            try:
                return self._parse_pdf_with_ptx_kernel(pdf_bytes, pdf_buffer_gpu, buffer_size, page_num)
            except RuntimeError:
                self._enable_gpu_parser = False

        return self._parse_pdf_structure_pymupdf(page_num)

    def _parse_pdf_with_ptx_kernel(
        self,
        pdf_bytes: bytes,
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

        for idx, obj in enumerate(objects):
            if obj[4] == 1.0:
                text_ptr = int(obj[5])
                text_len = int(obj[6])
                if 0 <= text_ptr < len(pdf_bytes) and text_len > 0:
                    end = min(len(pdf_bytes), text_ptr + text_len)
                    raw = pdf_bytes[text_ptr:end]
                    decoded = self._decode_pdf_string(raw)
                else:
                    decoded = ""

                text_index = len(self._temp_text_storage)
                self._temp_text_storage.append(decoded)
                obj[5] = float(text_index)
                obj[6] = float(len(decoded))

                if decoded:
                    obj[0] = 72.0
                    obj[1] = max(0.0, 720.0 - 18.0 * idx)
                    obj[2] = max(60.0, float(len(decoded) * 5.0))
                    obj[3] = 14.0

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
            consecutive_errors = 0
            max_consecutive_errors = 50  # Bail out if too many bad images

            for img in image_list:
                # Safety: break if too many consecutive errors (malformed PDF)
                if consecutive_errors >= max_consecutive_errors:
                    break

                xref = img[0]
                name = img[7] if len(img) > 7 else None
                try:
                    bbox = page.get_image_bbox(name) if name else None
                except Exception:
                    bbox = None
                    consecutive_errors += 1

                if not bbox:
                    consecutive_errors += 1
                    continue

                # Reset error counter on success
                consecutive_errors = 0

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
            "method": "structured-pymupdf",
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

    def _ocr_fallback(self, pdf_path: str, page_num: int) -> Dict[str, object]:
        # Phase E: Try DeepSeek OCR first if enabled
        if self._enable_deepseek_ocr and self.deepseek_bridge is not None:
            return self._ocr_fallback_deepseek(pdf_path, page_num)

        # Fallback to Tesseract
        return self._ocr_fallback_tesseract(pdf_path, page_num)

    def _ocr_fallback_deepseek(self, pdf_path: str, page_num: int) -> Dict[str, object]:
        """
        Phase E: Enhanced OCR using DeepSeek pipeline.

        Uses DeepSeek architecture:
        - Stage 1: SAM-base (local perception with window attention)
        - Stage 2: 16× Convolutional compressor
        - Stage 3: Text extraction (PyMuPDF + Tesseract fallback)
        - Stage 4: CLIP-large (global context encoding)

        Returns structured objects compatible with existing pipeline.
        """
        import time as _time

        ocr_start = _time.perf_counter()

        try:
            import fitz  # type: ignore
            from PIL import Image  # type: ignore
        except ImportError:
            print("[WARN] PyMuPDF/Pillow not available; falling back to Tesseract")
            return self._ocr_fallback_tesseract(pdf_path, page_num)

        try:
            # Render PDF page to image
            with fitz.open(pdf_path) as doc:
                if page_num < 0 or page_num >= len(doc):
                    raise ValueError(f"Page {page_num} out of range")

                page = doc[page_num]
                matrix = fitz.Matrix(2.0, 2.0)  # 2× resolution
                pix = page.get_pixmap(matrix=matrix, alpha=False)

                # Capture page rect before doc closes
                page_rect = page.rect

                # Convert to PIL then numpy
                import io
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                page_image = np.array(img, dtype=np.uint8)

                # Ensure RGB
                if page_image.ndim == 2:
                    page_image = np.stack([page_image, page_image, page_image], axis=-1)
                elif page_image.shape[2] == 4:
                    page_image = page_image[:, :, :3]

            # Run DeepSeek pipeline
            extraction = self.deepseek_bridge.extract(
                page_image,
                Path(pdf_path),
                page_num
            )

            # Convert extracted text to structured objects (compatible with existing pipeline)
            width_px, height_px = page_image.shape[1], page_image.shape[0]
            scale_x = 2.0  # We rendered at 2× resolution
            scale_y = 2.0

            feature_map = extraction.get("feature_map")
            gpu_text, gpu_blocks = self._decode_feature_map_to_text(
                feature_map,
                page_image.shape,
                page_rect,
                scale_x,
                scale_y,
                feature_dim_hint=extraction.get("feature_dim"),
            )

            # Split text into lines for object creation
            objects: List[List[float]] = []
            text_outputs: List[str] = []

            gpu_text_clean = gpu_text.strip() if gpu_text else ""
            gpu_text_viable = bool(gpu_blocks) and len(gpu_text_clean) >= 64

            if gpu_text_viable:
                for block in gpu_blocks:
                    text_content = (block.get("text") or "").strip()
                    if not text_content:
                        continue
                    bbox_pdf = block.get("bbox") or (72.0, 72.0, 400.0, 16.0)
                    importance = float(block.get("confidence", 0.85))
                    text_index = len(self._temp_text_storage)
                    self._temp_text_storage.append(text_content)
                    objects.append([
                        float(bbox_pdf[0]),
                        float(bbox_pdf[1]),
                        float(bbox_pdf[2]),
                        float(bbox_pdf[3]),
                        1.0,
                        float(text_index),
                        float(len(text_content)),
                        max(0.1, min(1.0, importance)),
                    ])
                    text_outputs.append(text_content)

                full_text = gpu_text or "\n".join(text_outputs)
            else:
                if gpu_blocks and not gpu_text_viable:
                    print("[PhaseG][GPU OCR] GPU text below threshold; using DeepSeek simple extractor fallback")

                lines = extraction.get('full_text', '').split('\n')
                y_offset = 50.0
                for line_text in lines:
                    if not line_text.strip():
                        continue

                    text_width = len(line_text) * 6.0
                    text_height = 12.0
                    x_pdf = 72.0
                    y_pdf = y_offset
                    width_pdf = min(text_width, page_rect.width - 144.0)
                    height_pdf = text_height

                    text_index = len(self._temp_text_storage)
                    self._temp_text_storage.append(line_text)

                    objects.append([
                        float(x_pdf),
                        float(y_pdf),
                        float(width_pdf),
                        float(height_pdf),
                        1.0,
                        float(text_index),
                        float(len(line_text)),
                        0.9,
                    ])

                    text_outputs.append(line_text)
                    y_offset += text_height + 4.0

                full_text = "\n".join(text_outputs)

            objects_array = (
                np.array(objects, dtype=np.float32)
                if objects
                else np.zeros((0, 8), dtype=np.float32)
            )

            return {
                "objects_gpu": None,
                "objects": objects_array,
                "object_count": int(objects_array.shape[0]),
                "processing_time_us": int((_time.perf_counter() - ocr_start) * 1_000_000),
                "is_scanned": True,
                "method": "deepseek",
                "text": full_text,
                "compression_ratio": extraction['compression_ratio'],
                "fidelity": extraction['fidelity'],
            }

        except Exception as exc:
            print(f"[WARN] DeepSeek OCR failed: {exc}")
            # Fallback to Tesseract
            return self._ocr_fallback_tesseract(pdf_path, page_num)

    def _ocr_fallback_tesseract(self, pdf_path: str, page_num: int) -> Dict[str, object]:
        """
        Pragmatic OCR fallback using Tesseract. This replaces the GPU-native OCR
        path temporarily until Phase E revisits glyph recognition with learned priors.
        """
        import io
        import time as _time
        from collections import defaultdict

        ocr_start = _time.perf_counter()

        try:
            import fitz  # type: ignore
        except ImportError:
            print("[WARN] PyMuPDF not available; OCR fallback skipped.")
            return {
                "objects_gpu": None,
                "objects": np.zeros((0, 8), dtype=np.float32),
                "object_count": 0,
                "processing_time_us": 0,
                "is_scanned": True,
                "method": "tesseract-missing",
                "text": "",
            }

        # Sovereign OCR only - pytesseract fallback disabled
        # DeepSeek OCR bridge handles all OCR needs
        return {
            "objects_gpu": None,
            "objects": np.zeros((0, 8), dtype=np.float32),
            "object_count": 0,
            "processing_time_us": 0,
            "is_scanned": True,
            "method": "sovereign-ocr-only",
            "text": "",
        }

        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            matrix = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
        except Exception as exc:
            print(f"[WARN] Tesseract fallback failed to render page: {exc}")
            if "doc" in locals():
                doc.close()
            return {
                "objects_gpu": None,
                "objects": np.zeros((0, 8), dtype=np.float32),
                "object_count": 0,
                "processing_time_us": int((_time.perf_counter() - ocr_start) * 1_000_000),
                "is_scanned": True,
                "method": "tesseract-failed",
                "text": "",
            }

        page_rect = page.rect
        width_px, height_px = pix.width, pix.height
        scale_x = page_rect.width / float(width_px) if width_px > 0 else 1.0
        scale_y = page_rect.height / float(height_px) if height_px > 0 else 1.0

        try:
            data = pytesseract.image_to_data(img, lang="eng", output_type=TessOutput.DICT)
        except Exception as exc:
            print(f"[WARN] Tesseract OCR failed: {exc}")
            data = None

        lines: Dict[Tuple[int, int, int], Dict[str, object]] = {}

        if data and "text" in data:
            for idx, raw_text in enumerate(data["text"]):
                text = raw_text.strip()
                if not text:
                    continue
                try:
                    conf_val = float(data["conf"][idx])
                except (KeyError, ValueError, TypeError):
                    conf_val = 0.0
                if conf_val < 0:
                    continue
                x = int(data.get("left", [0])[idx])
                y = int(data.get("top", [0])[idx])
                w = int(data.get("width", [0])[idx])
                h = int(data.get("height", [0])[idx])
                key = (
                    int(data.get("block_num", [0])[idx]),
                    int(data.get("par_num", [0])[idx]),
                    int(data.get("line_num", [0])[idx]),
                )
                entry = lines.setdefault(
                    key,
                    {
                        "words": [],
                        "conf": [],
                        "x1": float("inf"),
                        "y1": float("inf"),
                        "x2": float("-inf"),
                        "y2": float("-inf"),
                    },
                )
                entry["words"].append(text)
                entry["conf"].append(conf_val)
                entry["x1"] = min(entry["x1"], float(x))
                entry["y1"] = min(entry["y1"], float(y))
                entry["x2"] = max(entry["x2"], float(x + w))
                entry["y2"] = max(entry["y2"], float(y + h))

        doc.close()

        line_entries = list(lines.values())
        if not line_entries:
            try:
                simple_text = pytesseract.image_to_string(img, lang="eng")
            except Exception:
                simple_text = ""
            simple_text = simple_text.strip()
            if simple_text:
                line_entries = [
                    {
                        "words": [simple_text],
                        "conf": [70.0],
                        "x1": 0.0,
                        "y1": 0.0,
                        "x2": float(width_px),
                        "y2": float(height_px),
                    }
                ]

        objects: List[List[float]] = []
        collected_lines: List[str] = []

        for entry in line_entries:
            if not entry["words"]:
                continue

            x1 = max(0.0, entry["x1"])
            y1 = max(0.0, entry["y1"])
            x2 = min(float(width_px), entry["x2"])
            y2 = min(float(height_px), entry["y2"])

            width_pdf = max(1e-3, (x2 - x1) * scale_x)
            height_pdf = max(1e-3, (y2 - y1) * scale_y)
            x_pdf = page_rect.x0 + x1 * scale_x
            top_pdf = page_rect.y1 - y1 * scale_y
            y_pdf = top_pdf - height_pdf

            text_content = " ".join(entry["words"]).strip()
            if not text_content:
                continue

            avg_conf = sum(entry["conf"]) / float(len(entry["conf"])) if entry["conf"] else 0.0
            confidence = max(0.0, min(1.0, avg_conf / 100.0))

            text_index = len(self._temp_text_storage)
            self._temp_text_storage.append(text_content)
            collected_lines.append(text_content)

            objects.append(
                [
                    float(x_pdf),
                    float(y_pdf),
                    float(width_pdf),
                    float(height_pdf),
                    1.0,
                    float(text_index),
                    float(len(text_content)),
                    confidence,
                ]
            )

        objects_array = (
            np.array(objects, dtype=np.float32)
            if objects
            else np.zeros((0, 8), dtype=np.float32)
        )

        return {
            "objects_gpu": None,
            "objects": objects_array,
            "object_count": int(objects_array.shape[0]),
            "processing_time_us": int((_time.perf_counter() - ocr_start) * 1_000_000),
            "is_scanned": True,
            "method": "tesseract",
            "text": "\n".join(collected_lines),
        }

    # ------------------------------------------------------------------ #
    # Embedding + graph helpers
    # ------------------------------------------------------------------ #
    def _extract_character_bboxes(self, gray_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        try:
            import cv2  # type: ignore
        except ImportError:  # pragma: no cover - guarded earlier
            return []

        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            23,
            6,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.erode(binary, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        char_bboxes: List[Tuple[int, int, int, int]] = []

        for idx in range(1, num_labels):
            x, y, w, h, area = stats[idx]
            if area < 25 or w < 6 or h < 6:
                continue
            if w > 260 or h > 260:
                continue
            aspect = w / h if h > 0 else 0.0
            if aspect < 0.08 or aspect > 8.0:
                continue

            splits = self._split_wide_bbox(binary, x, y, w, h)
            char_bboxes.extend(splits)

        char_bboxes.sort(key=lambda bbox: (bbox[1] // 20, bbox[0]))
        return char_bboxes

    @staticmethod
    def _split_wide_bbox(binary_img: np.ndarray, x: int, y: int, w: int, h: int) -> List[Tuple[int, int, int, int]]:
        """
        Split overly wide bounding boxes into individual character boxes using column projections.
        """
        if w <= int(h * 1.3):
            return [(int(x), int(y), int(w), int(h))]

        region = binary_img[y : y + h, x : x + w]
        col_sum = (region > 0).sum(axis=0)

        threshold = max(1, int(h * 0.25))
        segments: List[Tuple[int, int, int, int]] = []
        start = None

        for idx, val in enumerate(col_sum):
            if val > threshold:
                if start is None:
                    start = idx
            else:
                if start is not None and idx - start >= 3:
                    segments.append((start, idx))
                start = None
        if start is not None and w - start >= 3:
            segments.append((start, w))

        if not segments:
            return [(int(x), int(y), int(w), int(h))]

        boxes: List[Tuple[int, int, int, int]] = []
        for seg_start, seg_end in segments:
            seg_w = seg_end - seg_start
            if seg_w < 3:
                continue
            boxes.append((int(x + seg_start), int(y), int(seg_w), int(h)))

        return boxes or [(int(x), int(y), int(w), int(h))]

    def _extract_character_features(
        self,
        gray_img: np.ndarray,
        char_bboxes: List[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        if not char_bboxes:
            return np.zeros((0, 128), dtype=np.float32)

        try:
            import cv2  # type: ignore
        except ImportError:  # pragma: no cover
            return np.zeros((0, 128), dtype=np.float32)

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

        features: List[np.ndarray] = []
        for x, y, w, h in char_bboxes:
            char_crop = gray_img[y : y + h, x : x + w]
            if char_crop.size == 0:
                continue
            resized = cv2.resize(char_crop, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
            if float(resized.max() - resized.min()) > 1e-6:
                normalized = cv2.normalize(resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            else:
                normalized = np.zeros_like(resized)
            resized_uint8 = normalized.astype(np.uint8)

            _, img_bin = cv2.threshold(resized_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            hog_input = cv2.bitwise_not(img_bin)

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

            features.append(feature_128)

        if not features:
            return np.zeros((0, 128), dtype=np.float32)

        return np.ascontiguousarray(np.vstack(features), dtype=np.float32)

    _SLENDER_CHARS = {"1", "I", "i", "l", "t", "f", "j", "J", "7"}
    _DIGIT_TO_LETTER_MAP = {
        "0": [("O", 0.05), ("Q", 0.05)],
        "1": [("I", 0.05), ("L", 0.06)],
        "2": [("Z", 0.06)],
        "3": [("E", 0.05)],
        "4": [("A", 0.05)],
        "5": [("S", 0.05)],
        "6": [("G", 0.05)],
        "7": [("T", 0.05)],
        "8": [("B", 0.05)],
        "9": [("G", 0.05), ("Q", 0.07)],
    }
    _NON_LATIN_TOKENS = (
        "arabic",
        "khmer",
        "devanagari",
        "bengali",
        "malayalam",
        "tamil",
        "telugu",
        "sinhala",
        "thai",
        "lao",
        "myanmar",
        "georgian",
        "ethiopic",
        "armenian",
        "cyrillic",
        "cherokee",
        "syriac",
        "naskh",
        "hebrew",
        "gurmukhi",
        "oriya",
        "kannada",
        "tifinagh",
        "vai",
        "mongolian",
        "khudawadi",
        "buhid",
        "tibetan",
        "balinese",
    )

    # PHASE_E_TODO: The methods below support the shelved GPU-native OCR flow.
    # They remain in-place so Phase E can iterate on sovereign glyph matching
    # without reintroducing the full feature extractor from scratch.
    def _ensure_glyphs_on_gpu(self) -> bool:
        if self.glyph_embeddings is None or self.glyph_dims is None:
            return False

        if self.glyph_resonator_kernel is None:
            return False

        if self.glyph_embeddings_gpu is None:
            embeddings = np.ascontiguousarray(self.glyph_embeddings, dtype=np.float32)
            ptr = gpu_malloc(embeddings.nbytes)
            memcpy_htod(ptr, embeddings.ctypes.data_as(ctypes.c_void_p), embeddings.nbytes)
            self.glyph_embeddings_gpu = ptr
            self._glyph_embeddings_bytes = embeddings.nbytes

        if self.glyph_dims_gpu is None:
            dims = np.ascontiguousarray(self.glyph_dims, dtype=np.int32)
            ptr = gpu_malloc(dims.nbytes)
            memcpy_htod(ptr, dims.ctypes.data_as(ctypes.c_void_p), dims.nbytes)
            self.glyph_dims_gpu = ptr
            self._glyph_dims_bytes = dims.nbytes

        return True

    def _match_glyphs_gpu(
        self,
        char_features: np.ndarray,
        char_bboxes: List[Tuple[int, int, int, int]],
        query_dim: int,
    ) -> List[Dict[str, object]]:
        if (
            char_features.size == 0
            or self.glyph_embeddings is None
            or self.glyph_dims is None
            or not self._ensure_glyphs_on_gpu()
        ):
            return []

        num_chars = char_features.shape[0]
        if num_chars == 0:
            return []

        query_dim = int(max(1, min(query_dim, char_features.shape[1], self.glyph_max_dim)))
        features = np.ascontiguousarray(char_features[:, :query_dim], dtype=np.float32)

        features_gpu = gpu_malloc(features.nbytes)
        memcpy_htod(features_gpu, features.ctypes.data_as(ctypes.c_void_p), features.nbytes)

        output = np.zeros((num_chars, 3), dtype=np.float32)
        output_gpu = gpu_malloc(output.nbytes)

        glyph_count = int(self.glyph_count)

        launch(
            self.glyph_resonator_kernel,
            grid=((num_chars + 255) // 256, 1, 1),
            block=(256, 1, 1),
            params=[
                ctypes.c_uint64(int(output_gpu.value)),
                ctypes.c_uint64(int(features_gpu.value)),
                ctypes.c_int(num_chars),
                ctypes.c_uint64(int(self.glyph_embeddings_gpu.value)),
                ctypes.c_int(glyph_count),
                ctypes.c_uint64(int(self.glyph_dims_gpu.value)),
                ctypes.c_int(query_dim),
                ctypes.c_int(self.glyph_max_dim),
            ],
        )
        synchronize()

        memcpy_dtoh(
            output.ctypes.data_as(ctypes.c_void_p),
            output_gpu,
            output.nbytes,
        )

        gpu_free(features_gpu)
        gpu_free(output_gpu)

        recognized: List[Dict[str, object]] = []

        for idx in range(num_chars):
            glyph_idx = int(output[idx, 1])
            score = float(output[idx, 2])
            if glyph_idx < 0 or glyph_idx >= glyph_count:
                continue
            if not np.isfinite(score):
                continue

            meta = self.glyph_metadata[glyph_idx]
            confidence = score * float(meta.get("confidence", 1.0))

            descriptor = f"{meta.get('font', '')}".lower()
            font_path = meta.get("font_path")
            if font_path:
                descriptor += f" {font_path}".lower()

            if meta.get("is_symbol") or meta.get("is_symbol_font"):
                confidence -= 0.05
            if any(token in descriptor for token in self._NON_LATIN_TOKENS):
                confidence -= 0.05

            if confidence < self._min_proto_confidence:
                continue

            bbox = char_bboxes[idx]
            recognized.append(
                {
                    "char": meta.get("char", ""),
                    "bbox": bbox,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "font": meta.get("font"),
                    "glyph_index": glyph_idx,
                    "available_dims": meta.get("available_dims", []),
                }
            )

        return recognized

    def _decode_feature_map_to_text(
        self,
        feature_map: Optional[np.ndarray],
        page_shape: Tuple[int, int, int],
        page_rect,
        scale_x: float,
        scale_y: float,
        max_candidates: int = 1200,
        feature_dim_hint: Optional[int] = None,
    ) -> Tuple[str, List[Dict[str, object]]]:
        """
        Decode CNN feature map into text using glyph resonance.
        """
        if feature_map is None or not isinstance(feature_map, np.ndarray) or feature_map.size == 0:
            return "", []

        H_feat, W_feat, C_feat = feature_map.shape

        activation = np.linalg.norm(feature_map, axis=2)
        max_activation = float(activation.max())
        if max_activation < 1e-5:
            return "", []

        activation_flat = activation.reshape(-1)
        sorted_indices = np.argsort(activation_flat)[::-1]
        max_candidates = min(max_candidates, sorted_indices.size)
        if max_candidates == 0:
            return "", []

        # Adaptive threshold: keep top activations or those above mean+std
        top_values = activation_flat[sorted_indices[:max_candidates]]
        adaptive_threshold = max(
            max_activation * 0.15,
            float(top_values.mean() * 0.5)
        )

        char_features: List[np.ndarray] = []
        char_bboxes: List[Tuple[int, int, int, int]] = []

        page_h, page_w = page_shape[0], page_shape[1]
        cell_w = max(page_w / float(W_feat), 1.0)
        cell_h = max(page_h / float(H_feat), 1.0)
        bbox_w = max(int(round(cell_w * 1.6)), 4)
        bbox_h = max(int(round(cell_h * 1.6)), 4)
        hint_dim = feature_dim_hint if feature_dim_hint is not None and feature_dim_hint > 0 else C_feat
        query_dim = int(min(max(1, hint_dim), C_feat, self.glyph_max_dim))

        for flat_idx in sorted_indices[:max_candidates]:
            activation_val = float(activation_flat[flat_idx])
            if activation_val < adaptive_threshold:
                continue

            row = flat_idx // W_feat
            col = flat_idx % W_feat
            row_start = max(row - 1, 0)
            row_end = min(row + 2, H_feat)
            col_start = max(col - 1, 0)
            col_end = min(col + 2, W_feat)

            patch = feature_map[row_start:row_end, col_start:col_end, :].astype(np.float32, copy=False)
            vec = patch.mean(axis=(0, 1))

            if vec.size > query_dim:
                vec = vec[:query_dim]
            elif vec.size < query_dim:
                padded = np.zeros(query_dim, dtype=np.float32)
                padded[:vec.size] = vec
                vec = padded

            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                continue
            vec = vec / norm

            x_px = int(round(col * cell_w))
            y_px = int(round(row * cell_h))

            char_features.append(vec)
            char_bboxes.append((x_px, y_px, bbox_w, bbox_h))

        if not char_features:
            print("[PhaseG][GPU OCR] No high-activation patches detected in feature map")
            return "", []

        feature_matrix = np.vstack(char_features).astype(np.float32, copy=False)
        recognized = self._match_glyphs_gpu(feature_matrix, char_bboxes, query_dim)
        if not recognized:
            print(f"[PhaseG][GPU OCR] Glyph matcher produced 0 recognitions (candidates={feature_matrix.shape[0]})")
            return "", []

        blocks = self._group_characters_to_blocks(
            recognized,
            scale_x=scale_x,
            scale_y=scale_y,
            page_rect=page_rect,
        )

        if not blocks:
            print("[PhaseG][GPU OCR] Character grouping yielded no blocks")
            return "", []

        text_output = "\n".join(block["text"] for block in blocks if block.get("text"))
        return text_output.strip(), blocks

    def _select_candidate_by_geometry(
        self,
        candidates: List[Dict[str, object]],
        aspect_ratio: float,
    ) -> Optional[Dict[str, object]]:
        """
        Apply simple geometric heuristics to disambiguate slim vs wide glyphs.
        """
        if not candidates:
            return None

        best = candidates[0]

        if aspect_ratio < 0.45:
            for item in candidates:
                if item["char"] in self._SLENDER_CHARS and item["combined"] >= best["combined"] - 0.08:
                    best = item
                    break
        elif aspect_ratio > 0.78:
            for item in candidates:
                if item["char"] not in self._SLENDER_CHARS and item["combined"] >= best["combined"] - 0.10:
                    best = item
                    break

        digit_map = self._DIGIT_TO_LETTER_MAP.get(best["char"])
        if digit_map:
            for letter, margin in digit_map:
                for item in candidates:
                    if item["char"] == letter and item["combined"] >= best["combined"] - margin:
                        best = item
                        break
                else:
                    continue
                break

        if best["combined"] < 0.50:
            return None

        return best

    def _group_characters_to_blocks(
        self,
        recognized_chars: List[Dict[str, object]],
        *,
        scale_x: float,
        scale_y: float,
        page_rect,
    ) -> List[Dict[str, object]]:
        if not recognized_chars:
            return []

        char_entries: List[Dict[str, object]] = []
        for item in recognized_chars:
            bbox_px = item.get("bbox")
            char_symbol = str(item.get("char", ""))
            if not bbox_px or not char_symbol.strip():
                continue
            bbox_pdf = self._convert_bbox_px_to_pdf(bbox_px, scale_x, scale_y, page_rect)
            char_entries.append(
                {
                    "char": char_symbol,
                    "bbox_px": bbox_px,
                    "bbox_pdf": bbox_pdf,
                    "confidence": float(item.get("confidence", 0.0)),
                }
            )

        if not char_entries:
            return []

        char_entries.sort(key=lambda entry: (-entry["bbox_pdf"][1], entry["bbox_pdf"][0]))

        blocks: List[Dict[str, object]] = []
        current_line: List[Dict[str, object]] = [char_entries[0]]
        baseline = char_entries[0]["bbox_pdf"][1]
        line_threshold = max(char_entries[0]["bbox_pdf"][3] * 0.8, 4.0)

        for entry in char_entries[1:]:
            if abs(entry["bbox_pdf"][1] - baseline) <= line_threshold:
                current_line.append(entry)
            else:
                blocks.extend(self._assemble_line_block(current_line))
                current_line = [entry]
                baseline = entry["bbox_pdf"][1]
                line_threshold = max(entry["bbox_pdf"][3] * 0.8, 4.0)

        if current_line:
            blocks.extend(self._assemble_line_block(current_line))

        return blocks

    def _assemble_line_block(self, line_chars: List[Dict[str, object]]) -> List[Dict[str, object]]:
        if not line_chars:
            return []

        line_chars.sort(key=lambda entry: entry["bbox_pdf"][0])

        text_parts: List[str] = []
        confidences: List[float] = []
        prev_char = None

        for entry in line_chars:
            char_symbol = entry["char"]
            bbox = entry["bbox_pdf"]
            if prev_char is not None:
                prev_bbox = prev_char["bbox_pdf"]
                gap = bbox[0] - (prev_bbox[0] + prev_bbox[2])
                avg_width = (bbox[2] + prev_bbox[2]) / 2.0
                if gap > max(avg_width * 0.25, 0.8):
                    text_parts.append(" ")
            text_parts.append(char_symbol)
            confidences.append(float(entry["confidence"]))
            prev_char = entry

        text_str = "".join(text_parts).strip()
        if not text_str:
            return []

        x0 = min(entry["bbox_pdf"][0] for entry in line_chars)
        y0 = min(entry["bbox_pdf"][1] for entry in line_chars)
        x1 = max(entry["bbox_pdf"][0] + entry["bbox_pdf"][2] for entry in line_chars)
        y1 = max(entry["bbox_pdf"][1] + entry["bbox_pdf"][3] for entry in line_chars)

        confidence = float(np.mean(confidences) if confidences else 0.0)
        confidence = max(0.0, min(1.0, confidence))

        return [
            {
                "text": text_str,
                "bbox": (x0, y0, x1 - x0, y1 - y0),
                "confidence": confidence,
            }
        ]

    @staticmethod
    def _convert_bbox_px_to_pdf(
        bbox_px: Tuple[int, int, int, int],
        scale_x: float,
        scale_y: float,
        page_rect,
    ) -> Tuple[float, float, float, float]:
        x_px, y_px, w_px, h_px = bbox_px
        x_pdf = float(page_rect.x0) + (float(x_px) / float(scale_x))
        width_pdf = float(w_px) / float(scale_x)
        top_pdf = float(page_rect.y1) - (float(y_px) / float(scale_y))
        height_pdf = float(h_px) / float(scale_y)
        y_pdf = top_pdf - height_pdf
        return (x_pdf, y_pdf, width_pdf, height_pdf)

    @staticmethod
    def _decode_pdf_string(raw: bytes) -> str:
        if not raw:
            return ""

        text = raw.decode("latin-1", errors="ignore")
        text = text.replace("\\n", " ")
        text = text.replace("\\r", " ")
        text = text.replace("\\t", " ")
        text = text.replace("\r", " ")
        text = text.replace("\n", " ")
        text = text.replace("\\(", "(")
        text = text.replace("\\)", ")")
        text = "".join(ch if 32 <= ord(ch) <= 126 else " " for ch in text)
        while "  " in text:
            text = text.replace("  ", " ")
        return text.strip()

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

        reservoirs: List[np.ndarray] = []
        if text_embeddings.size > 0:
            reservoirs.append(text_embeddings.astype(np.float32, copy=False))
        if visual_embeddings.size > 0:
            reservoirs.append(visual_embeddings.astype(np.float32, copy=False))

        combined = np.vstack(reservoirs)

        fused_flat = self.fusion_engine.transform(
            combined.flatten().astype(np.float32, copy=False), mode=0, ratio=0.5
        )
        fused_matrix = fused_flat.reshape(combined.shape)
        return fused_matrix.mean(axis=0, keepdims=True)

    def _crystallize_to_galaxy(
        self, layout_graph: Dict[str, object], fused_embeddings: np.ndarray
    ) -> np.ndarray:
        from knowledge3d.cranium.ptx_runtime.graph_crystallizer import GraphCrystallizer

        if not layout_graph.get("nodes"):
            vec = fused_embeddings[0] if fused_embeddings.size else np.zeros(128, dtype=np.float32)
            head = vec[:3]
            norm = np.linalg.norm(head) or 1.0
            return (head / norm).astype(np.float32)

        node_embeddings = np.stack(
            [np.asarray(node.get("embedding", np.zeros(128)), dtype=np.float32) for node in layout_graph["nodes"]],
            axis=0,
        )

        neighbor_accum = np.zeros_like(node_embeddings)
        counts = np.zeros(node_embeddings.shape[0], dtype=np.int32)
        for src, dst, _ in layout_graph.get("edges", []):
            if 0 <= src < node_embeddings.shape[0] and 0 <= dst < node_embeddings.shape[0]:
                neighbor_accum[src] += node_embeddings[dst]
                counts[src] += 1

        counts = np.maximum(counts, 1)[:, None].astype(np.float32)
        neighbor_embeddings = np.divide(neighbor_accum, counts, where=counts > 0)

        crystallizer = GraphCrystallizer()
        crystallized = crystallizer.crystallize(
            node_embeddings.astype(np.float32, copy=False),
            neighbor_embeddings.astype(np.float32, copy=False),
        )

        fused_vector = fused_embeddings[0] if fused_embeddings.size else np.zeros(128, dtype=np.float32)
        avg_vector = 0.5 * crystallized.mean(axis=0) + 0.5 * fused_vector
        head = avg_vector[:3]
        norm = np.linalg.norm(head) or 1.0
        return (head / norm).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    def _cleanup_gpu_buffers(self) -> None:
        if not self.gpu_enabled:
            self.allocated_buffers.clear()
            return

        for ptr, _ in self.allocated_buffers:
            try:
                gpu_free(ptr)
            except RuntimeError:
                pass
        self.allocated_buffers.clear()
