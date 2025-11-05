"""
Phase G PDF Ingestion Bridge - Full AGI Integration

Integrates:
1. Adaptive RPN embeddings (variable dimensions 64-2048)
2. Trained Phase G specialists (multimodal, speech, OCR, router)
3. Shadow weights mechanism (safe self-updating)
4. Galaxy star creation (knowledge storage in 3D space)
5. Two sleep cycles (model logic vs knowledge consolidation)

Architecture:
- MODELS = LOGIC (LoRA adapters with shadow weights)
- KNOWLEDGE = 3D SPACE (Galaxy stars with AI textures)

Key Improvements:
- Adaptive dimensions: "Hello" → 64D, full page → 512D (64× speedup!)
- Specialist routing: OCR for scanned pages, multimodal for rich documents
- Safe updates: Shadow weights prevent catastrophic forgetting
- 3D knowledge: Stars materialize into House objects during sleep
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np

# Base infrastructure
from knowledge3d.cranium.adaptive_rpn_engine import AdaptiveRPNEngine, DimensionConfig
from knowledge3d.cranium.bridges.pdf_ingestion_bridge import PDFIngestionBridge

# Phase G specialists
from knowledge3d.cranium.matryoshka_trm import MatryoshkaTRM


class PhaseGPDFIngestionBridge(PDFIngestionBridge):
    """
    Enhanced PDF ingestion bridge with Phase G integration.

    Extends base bridge with:
    - Adaptive variable-dimension embeddings
    - Trained specialist loading
    - Galaxy star creation for knowledge storage
    """

    def __init__(self, phase_g_checkpoint_dir: Optional[Path] = None):
        """
        Initialize Phase G ingestion bridge.

        Args:
            phase_g_checkpoint_dir: Directory with trained specialist checkpoints
        """
        # Initialize base bridge (sets up kernels, OCR, etc.)
        super().__init__()

        self.ocr_stats: Dict[str, int] = {
            "attempts": 0,
            "gpu_success": 0,
            "gpu_fail": 0,
            "fallback": 0,
            "skipped": 0,
        }
        self._last_ocr_log_ts = time.time()
        self.gpu_ocr_enabled: bool = False
        self._foundational_status: Dict[str, Any] = {}

        if self.deepseek_bridge is None:
            print("[PhaseG] DeepSeek OCR bridge unavailable; GPU OCR disabled")
        else:
            foundations_ready = self._verify_foundational_embeddings()
            glyph_status = "OK" if self._foundational_status.get("glyph_ready") else "MISSING"
            galaxy_ready = self._foundational_status.get("galaxy_ready")
            galaxy_count = self._foundational_status.get("galaxy_total", 0)
            non_zero_samples = self._foundational_status.get("non_zero_samples", 0)
            galaxy_status = "OK" if galaxy_ready else "MISSING"
            print(
                f"[PhaseG] OCR foundation check → glyphs: {glyph_status}, "
                f"galaxy: {galaxy_status} (stars={galaxy_count}, "
                f"non_zero_sample={non_zero_samples})"
            )

            if foundations_ready:
                try:
                    self.enable_deepseek_ocr(True)
                    self.gpu_ocr_enabled = True
                    print("[PhaseG] GPU OCR enabled (DeepSeek bridge active)")
                except Exception as exc:
                    print(f"[PhaseG] WARNING: Failed to enable GPU OCR: {exc}")
                    self.gpu_ocr_enabled = False
            else:
                print("[PhaseG] GPU OCR remains disabled until foundational embeddings are ready")

        # Replace fixed-dimension RPN engine with adaptive engine
        print("[PhaseG] Initializing adaptive RPN engine...")
        self.adaptive_rpn = AdaptiveRPNEngine(
            config=DimensionConfig(
                dim_levels=[64, 128, 256, 512, 1024, 2048],
                default_dim=128,
                min_dim=64,
                max_dim=2048
            )
        )

        # Load Phase G specialists if checkpoint provided
        self.specialists_loaded = False
        self.matryoshka_system: Optional[MatryoshkaTRM] = None

        if phase_g_checkpoint_dir is None:
            phase_g_checkpoint_dir = Path("/K3D/Knowledge3D.local/checkpoints/phase_g/current")

        if phase_g_checkpoint_dir and phase_g_checkpoint_dir.exists():
            self._load_phase_g_specialists(phase_g_checkpoint_dir)
        else:
            print(f"[PhaseG] No specialist checkpoints found at {phase_g_checkpoint_dir}")
            print("[PhaseG] Operating in base mode (adaptive RPN only)")

        # Galaxy knowledge storage (separate from model weights!)
        self.galaxy_stars: List[Dict[str, Any]] = []
        self.galaxy_star_embeddings: List[np.ndarray] = []

        print("[PhaseG] Phase G PDF Ingestion Bridge initialized")

    def _load_phase_g_specialists(self, checkpoint_dir: Path):
        """
        Load trained Phase G specialists from checkpoints.

        Args:
            checkpoint_dir: Directory with specialist checkpoints
        """
        print(f"[PhaseG] Loading specialists from {checkpoint_dir}")

        try:
            # Initialize Matryoshka system
            self.matryoshka_system = MatryoshkaTRM(max_dims=2048, min_dims=64)

            # Check for available specialists
            specialist_dirs = {
                'multimodal': checkpoint_dir.parent / 'multimodal_gpu_epoch_100',
                'speech': checkpoint_dir.parent / 'speech_gpu_epoch_100',
                'ocr': checkpoint_dir.parent / 'ocr_gpu_epoch_100',
                'router': checkpoint_dir.parent / 'router_gpu_epoch_200'
            }

            loaded_count = 0

            for name, spec_dir in specialist_dirs.items():
                if spec_dir.exists():
                    adapter_file = spec_dir / f'{name}_adapter.npz'
                    if adapter_file.exists():
                        # Load adapter metadata to get dims and rank
                        data = np.load(adapter_file)
                        dims = int(data['shape'][0])
                        rank = int(data['rank'])

                        # Register and load specialist
                        self.matryoshka_system.register_specialist(name, dims, rank)
                        self.matryoshka_system.specialists[name]['adapter'].load_checkpoint(spec_dir)

                        loaded_count += 1
                        print(f"[PhaseG]   ✓ Loaded {name}: {dims}D, rank {rank}")
                    else:
                        print(f"[PhaseG]   ✗ Missing adapter for {name}")
                else:
                    print(f"[PhaseG]   ✗ Directory not found: {spec_dir}")

            if loaded_count > 0:
                self.specialists_loaded = True
                print(f"[PhaseG] Successfully loaded {loaded_count}/4 specialists")
            else:
                print("[PhaseG] No specialists loaded")
                self.matryoshka_system = None

        except Exception as exc:
            print(f"[PhaseG] ERROR loading specialists: {exc}")
            self.matryoshka_system = None
            self.specialists_loaded = False

    def _verify_foundational_embeddings(self) -> bool:
        """
        Ensure glyph prototypes and Galaxy embeddings are available before GPU OCR.
        """
        glyph_ready = bool(
            isinstance(self.glyph_embeddings, np.ndarray) and self.glyph_embeddings.size > 0
        )

        atomic_path = Path("/K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars/galaxy_character_embeddings.npz")
        galaxy_ready = False
        total_stars = 0
        non_zero_samples = 0

        atomic_template_bank: Dict[str, np.ndarray] = {}
        atomic_mean_templates: Optional[np.ndarray] = None
        atomic_low_embeddings: Optional[np.ndarray] = None

        if atomic_path.exists():
            try:
                data = np.load(atomic_path)
                embeddings = np.asarray(data.get("embeddings"), dtype=np.float32)
                low_embeddings = (
                    np.asarray(data.get("embeddings_low"), dtype=np.float32)
                    if "embeddings_low" in data
                    else None
                )
                char_ids = np.asarray(data.get("char_ids"), dtype=np.int32)

                if embeddings.ndim != 2 or char_ids.shape[0] != embeddings.shape[0]:
                    raise ValueError("Invalid shape for atomic character embeddings")

                total_stars = int(embeddings.shape[0])

                unique_ids = np.unique(char_ids)
                feature_dim = embeddings.shape[1]
                mean_templates = np.zeros((256, feature_dim), dtype=np.float32)

                for char_id in unique_ids:
                    mask = char_ids == char_id
                    char_vectors = embeddings[mask].astype(np.float32, copy=True)
                    norms = np.linalg.norm(char_vectors, axis=1, keepdims=True)
                    valid_mask = (norms.squeeze(axis=1) > 1e-6)
                    if not np.any(valid_mask):
                        continue
                    char_vectors = char_vectors[valid_mask]
                    norms = np.maximum(np.linalg.norm(char_vectors, axis=1, keepdims=True), 1e-6)
                    char_vectors = char_vectors / norms
                    non_zero_samples += char_vectors.shape[0]

                    try:
                        char_symbol = chr(int(char_id))
                    except ValueError:
                        continue

                    atomic_template_bank[char_symbol] = char_vectors

                    cp = int(char_id)
                    if 0 <= cp < mean_templates.shape[0]:
                        mean_vec = char_vectors.mean(axis=0)
                        mean_norm = np.linalg.norm(mean_vec)
                        if mean_norm > 1e-8:
                            mean_templates[cp] = (mean_vec / mean_norm).astype(np.float32)

                atomic_mean_templates = mean_templates
                atomic_low_embeddings = low_embeddings
                galaxy_ready = total_stars > 0 and non_zero_samples > 0
            except Exception as exc:
                print(f"[PhaseG] WARNING: Unable to load atomic Galaxy embeddings ({exc})")
        else:
            print(f"[PhaseG] WARNING: Atomic Galaxy embeddings missing at {atomic_path}")

        if galaxy_ready and atomic_template_bank:
            self._apply_atomic_template_bank(atomic_template_bank, atomic_mean_templates, atomic_low_embeddings)

        self._foundational_status = {
            "glyph_ready": glyph_ready,
            "galaxy_ready": galaxy_ready,
            "galaxy_total": total_stars,
            "non_zero_samples": non_zero_samples,
            "atomic_path": str(atomic_path),
            "atomic_chars": len(atomic_template_bank),
        }

        return glyph_ready and galaxy_ready

    def _apply_atomic_template_bank(
        self,
        template_bank: Dict[str, np.ndarray],
        mean_templates: Optional[np.ndarray],
        low_embeddings: Optional[np.ndarray],
    ) -> None:
        """Apply atomic character embeddings to the detector and template bank."""
        if not template_bank:
            return

        self._pending_template_bank = template_bank

        if self.character_detector is None:
            print("[PhaseG] CharacterDetector unavailable; atomic templates pending")
            return

        try:
            if mean_templates is not None:
                current_templates = self.character_detector.template_bank.get_templates().copy()
                if mean_templates.shape == current_templates.shape:
                    for idx in range(mean_templates.shape[0]):
                        vec = mean_templates[idx]
                        if np.linalg.norm(vec) > 1e-6:
                            current_templates[idx] = vec
                    self.character_detector.template_bank.set_external_templates(current_templates)
                self.character_detector.set_atomic_mean_templates(mean_templates.copy())
        except Exception as exc:
            print(f"[PhaseG] WARNING: Failed to update Galactic template matrix ({exc})")

        try:
            self.character_detector.set_template_bank(template_bank)
        except Exception as exc:
            print(f"[PhaseG] WARNING: Failed to load atomic template bank ({exc})")
        else:
            total_vectors = sum(arr.shape[0] for arr in template_bank.values())
            print(
                f"[PhaseG] Applied atomic Galaxy character embeddings "
                f"({len(template_bank)} characters, {total_vectors} variants)"
            )

        classifier_map = self._load_atomic_classifiers(template_bank.keys())
        if classifier_map:
            try:
                self.character_detector.set_atomic_classifiers(classifier_map)
                print(f"[PhaseG] Registered {len(classifier_map)} atomic classifiers")
            except Exception as exc:
                print(f"[PhaseG] WARNING: Failed to register atomic classifiers ({exc})")

        if low_embeddings is not None:
            try:
                self.character_detector.set_atomic_low_embeddings(low_embeddings.copy())
            except Exception as exc:
                print(f"[PhaseG] WARNING: Failed to register low-d embeddings ({exc})")

    @staticmethod
    def _load_atomic_classifiers(chars: Iterable[str]) -> Dict[int, Tuple[np.ndarray, float]]:
        """Load binary classifiers (FC heads) for each character."""
        classifier_dir = Path("/K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars")
        classifiers: Dict[int, Tuple[np.ndarray, float]] = {}

        if not classifier_dir.exists():
            return classifiers

        for char in chars:
            if not char:
                continue
            char_code = ord(char)
            weight_path = classifier_dir / f"char_{char_code}_{char}_weights.npz"
            if not weight_path.exists():
                continue

            try:
                data = np.load(weight_path)
                fc_weight = np.asarray(data.get("fc_weight"), dtype=np.float32)
                fc_bias = np.asarray(data.get("fc_bias"), dtype=np.float32)
                if fc_weight is None or fc_bias is None:
                    continue
                if fc_weight.ndim != 2 or fc_weight.shape[0] != 2:
                    continue

                weight_vec = fc_weight[1] - fc_weight[0]
                if fc_bias.size >= 2:
                    bias_val = float(fc_bias[1] - fc_bias[0])
                elif fc_bias.size == 1:
                    bias_val = float(fc_bias[0])
                else:
                    bias_val = 0.0

                classifiers[char_code] = (weight_vec.astype(np.float32), bias_val)
            except Exception:
                continue

        return classifiers

    def _generate_text_embeddings(self, parsed_objects: Dict[str, object]) -> np.ndarray:
        """
        Generate text embeddings using adaptive dimensions.

        Overrides base method to use variable dimensions based on text length.

        Args:
            parsed_objects: Parsed PDF objects

        Returns:
            Text embeddings (variable dimensions per row!)
        """
        objects = parsed_objects.get("objects")
        if objects is None or len(objects) == 0:
            return np.zeros((0, self.adaptive_rpn.config.default_dim), dtype=np.float32)

        text_rows = objects[objects[:, 4] == 1.0]

        if len(text_rows) == 0:
            return np.zeros((0, self.adaptive_rpn.config.default_dim), dtype=np.float32)

        # Collect all text content first to determine batch max dimension
        texts = []
        for row in text_rows:
            storage_idx = int(row[5])
            if 0 <= storage_idx < len(self._temp_text_storage):
                text_content = self._temp_text_storage[storage_idx]
            else:
                text_content = ""
            texts.append(text_content.strip() if text_content else "")

        # Generate embeddings with adaptive dimensions
        embeddings_matrix, dimensions = self.adaptive_rpn.embed_batch(texts)

        # Log dimension selection for monitoring
        if len(dimensions) > 0:
            avg_dim = sum(dimensions) / len(dimensions)
            min_dim = min(dimensions)
            max_dim = max(dimensions)
            print(f"[PhaseG] Text embeddings: {len(dimensions)} items, "
                  f"dims {min_dim}-{max_dim} (avg {avg_dim:.0f})")

        return embeddings_matrix.astype(np.float32)

    @staticmethod
    def _count_objects_of_type(parsed_objects: Dict[str, object], obj_type: float) -> int:
        """Count objects of a given type in parsed PDF objects."""
        objects = parsed_objects.get("objects")
        if isinstance(objects, np.ndarray) and objects.size > 0:
            return int(np.sum(objects[:, 4] == obj_type))
        return 0

    def _estimate_text_length(self, parsed_objects: Dict[str, object]) -> int:
        """
        Estimate total text length using cached storage indices.
        """
        objects = parsed_objects.get("objects")
        if not isinstance(objects, np.ndarray) or objects.size == 0:
            return 0

        total_chars = 0
        text_rows = objects[objects[:, 4] == 1.0]
        for row in text_rows:
            storage_idx = int(row[5])
            if 0 <= storage_idx < len(self._temp_text_storage):
                total_chars += len(self._temp_text_storage[storage_idx])
            else:
                total_chars += int(max(row[6], 0))
        return total_chars

    @staticmethod
    def _project_embeddings_for_layout(embeddings: np.ndarray, target_dim: int = 128) -> np.ndarray:
        """
        Project embeddings to a fixed dimension for layout graph compatibility.
        """
        if embeddings.size == 0:
            return embeddings

        current_dim = embeddings.shape[1]
        if current_dim == target_dim:
            return embeddings

        if current_dim > target_dim:
            return embeddings[:, :target_dim].astype(np.float32, copy=False)

        projected = np.zeros((embeddings.shape[0], target_dim), dtype=np.float32)
        projected[:, :current_dim] = embeddings.astype(np.float32, copy=False)
        return projected

    def _attempt_gpu_ocr(self, pdf_path: str, page_num: int) -> Optional[Dict[str, object]]:
        """
        Run GPU OCR and capture success/failure statistics.
        """
        if not self.gpu_ocr_enabled:
            self.ocr_stats["skipped"] += 1
            return None

        self.ocr_stats["attempts"] += 1

        try:
            result = self._ocr_fallback_deepseek(pdf_path, page_num)
        except RuntimeError as exc:
            self.ocr_stats["gpu_fail"] += 1
            print(f"[PhaseG][GPU OCR] Runtime failure on {Path(pdf_path).name} p{page_num+1}: {exc}")
            self._maybe_log_ocr_stats()
            return None
        except Exception as exc:
            self.ocr_stats["gpu_fail"] += 1
            print(f"[PhaseG][GPU OCR] Unexpected failure on {Path(pdf_path).name} p{page_num+1}: {exc}")
            self._maybe_log_ocr_stats()
            return None

        method = result.get("method", "")
        object_count = int(result.get("object_count", 0))

        if method == "deepseek" and object_count > 0:
            self.ocr_stats["gpu_success"] += 1
            self._maybe_log_ocr_stats()
            return result

        self.ocr_stats["gpu_fail"] += 1
        self._maybe_log_ocr_stats()
        return None

    def _maybe_log_ocr_stats(self, force: bool = False) -> None:
        """Periodically log GPU OCR statistics."""
        attempts = self.ocr_stats.get("attempts", 0)
        if attempts == 0:
            return

        now = time.time()
        should_log = force or attempts % 25 == 0 or (now - self._last_ocr_log_ts) > 300.0
        if not should_log:
            return

        success = self.ocr_stats.get("gpu_success", 0)
        fail = self.ocr_stats.get("gpu_fail", 0)
        fallback = self.ocr_stats.get("fallback", 0)
        skipped = self.ocr_stats.get("skipped", 0)
        success_rate = (success / attempts * 100.0) if attempts else 0.0

        print(
            "[PhaseG][GPU OCR] stats → "
            f"attempts={attempts}, success={success}, fail={fail}, "
            f"fallback={fallback}, skipped={skipped}, "
            f"success_rate={success_rate:.1f}%"
        )
        self._last_ocr_log_ts = now

    def _fuse_modalities(
        self, text_embeddings: np.ndarray, visual_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Override to handle variable-dimension embeddings.

        Text embeddings can be 64-2048D (adaptive), visual embeddings are 128D.
        Pad to max dimension before fusing.
        """
        if text_embeddings.size == 0 and visual_embeddings.size == 0:
            return np.zeros((1, 128), dtype=np.float32)

        reservoirs: List[np.ndarray] = []
        dimensions: List[int] = []

        if text_embeddings.size > 0:
            reservoirs.append(text_embeddings.astype(np.float32, copy=False))
            dimensions.append(text_embeddings.shape[1])
        if visual_embeddings.size > 0:
            reservoirs.append(visual_embeddings.astype(np.float32, copy=False))
            dimensions.append(visual_embeddings.shape[1])

        # Handle variable dimensions: pad to max
        if len(reservoirs) > 1:
            max_dim = max(dimensions)
            padded_reservoirs = []

            for reservoir, dim in zip(reservoirs, dimensions):
                if dim < max_dim:
                    # Pad columns to max_dim
                    num_rows = reservoir.shape[0]
                    padded = np.zeros((num_rows, max_dim), dtype=np.float32)
                    padded[:, :dim] = reservoir
                    padded_reservoirs.append(padded)
                else:
                    padded_reservoirs.append(reservoir)

            reservoirs = padded_reservoirs

        try:
            combined = np.vstack(reservoirs)
        except ValueError as exc:
            print(f"[PhaseG] WARNING: Modal fusion mismatch ({exc}); using primary reservoir only.")
            primary = reservoirs[0]
            if primary.ndim == 1:
                primary = primary.reshape(1, -1)
            return primary[:1]

        # For variable dimensions, use simple averaging instead of complex fusion
        # fusion_engine.transform expects fixed shapes
        return combined.mean(axis=0, keepdims=True)

    def _select_specialist_for_page(self, parsed_objects: Dict[str, object]) -> str:
        """
        Select appropriate specialist for page type.

        Args:
            parsed_objects: Parsed PDF objects

        Returns:
            Specialist name ('ocr', 'multimodal', etc.)
        """
        if not self.specialists_loaded:
            return 'base'

        # Phase G: Prefer OCR specialist for all OCR-enhanced pages
        is_ocr_enhanced = parsed_objects.get('ocr_enhanced', False)
        is_scanned = parsed_objects.get('is_scanned', False)

        if is_ocr_enhanced or is_scanned:
            return 'ocr'  # Use OCR specialist (trained on GPU-extracted text)

        # Check if page has images (use multimodal specialist)
        objects = parsed_objects.get('objects', np.zeros((0, 8), dtype=np.float32))
        has_images = np.any(objects[:, 4] == 2.0) if len(objects) > 0 else False

        if has_images:
            return 'multimodal'

        # Default: use multimodal for general text
        return 'multimodal'

    def _process_with_specialist(self, embedding: np.ndarray, specialist_name: str) -> np.ndarray:
        """
        Process embedding through specialist.

        Args:
            embedding: Input embedding
            specialist_name: Specialist to use

        Returns:
            Processed embedding
        """
        if not self.specialists_loaded or specialist_name == 'base':
            return embedding

        if specialist_name not in self.matryoshka_system.specialists:
            print(f"[PhaseG] WARNING: Specialist '{specialist_name}' not available")
            return embedding

        # Process through specialist
        try:
            output = self.matryoshka_system.compute_with_specialist(embedding, specialist_name)
            return output
        except Exception as exc:
            print(f"[PhaseG] ERROR processing with specialist: {exc}")
            return embedding

    def _create_galaxy_star(self, fused_embedding: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Galaxy star for knowledge storage.

        This is KEY: Knowledge is stored in 3D space, NOT in model weights!

        Args:
            fused_embedding: Final page embedding
            metadata: Page metadata (file, page number, etc.)

        Returns:
            Galaxy star descriptor
        """
        # Extract 3D position from embedding (first 3 dimensions)
        position_vector = fused_embedding[:3] if len(fused_embedding) >= 3 else np.zeros(3, dtype=np.float32)

        # Normalize to unit sphere
        norm = np.linalg.norm(position_vector)
        if norm > 1e-8:
            position_vector = position_vector / norm
        else:
            position_vector = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Create star descriptor
        star = {
            'position': position_vector.tolist(),  # 3D coordinates on galaxy sphere
            'embedding': fused_embedding.tolist(),  # Full embedding (for clustering)
            'embedding_dim': len(fused_embedding),  # Actual dimension used
            'metadata': metadata,
            'created_at': time.time(),
            'source_type': 'pdf_page',
            'pending_consolidation': True  # Mark for sleep-time consolidation
        }

        return star

    def ingest_pdf_page(self, pdf_path: str | Path, page_num: int = 0) -> Dict[str, object]:
        """
        Ingest PDF page with Phase G enhancements.

        Overrides base method to integrate:
        - Adaptive dimension embeddings
        - Specialist processing
        - Galaxy star creation

        Args:
            pdf_path: Path to PDF file
            page_num: Page number

        Returns:
            Ingestion result with Phase G metadata
        """
        # Call base ingestion pipeline (parsing, OCR fallback, etc.)
        start_time = time.perf_counter()

        # Parse PDF structure
        pdf_path = Path(pdf_path)
        self._current_pdf_path = str(pdf_path)
        self._temp_text_storage.clear()
        self._temp_image_storage.clear()

        pdf_bytes = self._load_pdf_bytes(pdf_path, page_num)
        use_gpu_parser = self._enable_gpu_parser and self.gpu_enabled
        pdf_buffer_gpu = self._upload_to_gpu(pdf_bytes) if use_gpu_parser else None

        parsed_objects = self._parse_pdf_structure(pdf_bytes, pdf_buffer_gpu, len(pdf_bytes), page_num)
        parsed_objects.setdefault("method", "structured")

        text_objects = self._count_objects_of_type(parsed_objects, 1.0)
        image_objects = self._count_objects_of_type(parsed_objects, 2.0)
        estimated_text_len = self._estimate_text_length(parsed_objects)
        is_scanned = bool(parsed_objects.get("is_scanned"))

        should_use_gpu = (
            self.gpu_ocr_enabled
            and (
                text_objects == 0
                or (estimated_text_len < 48 and (image_objects > 0 or is_scanned))
            )
        )

        if should_use_gpu:
            gpu_result = self._attempt_gpu_ocr(str(pdf_path), page_num)
            if gpu_result:
                parsed_objects = gpu_result
                parsed_objects["ocr_enhanced"] = True
            else:
                parsed_objects["ocr_enhanced"] = False
                self.ocr_stats["fallback"] += 1
                print(
                    f"[PhaseG][GPU OCR] Fallback to PyMuPDF ({pdf_path.name}, page {page_num + 1})"
                )
                self._maybe_log_ocr_stats()
        else:
            parsed_objects["ocr_enhanced"] = False
            if not self.gpu_ocr_enabled:
                self.ocr_stats["skipped"] += 1
            else:
                self._maybe_log_ocr_stats()

        # Generate embeddings (with adaptive dimensions!)
        text_embeddings = self._generate_text_embeddings(parsed_objects)
        visual_embeddings = self._generate_visual_embeddings(parsed_objects)

        # Build layout graph
        layout_graph = self._build_layout_graph(
            parsed_objects,
            self._project_embeddings_for_layout(text_embeddings),
            self._project_embeddings_for_layout(visual_embeddings),
        )
        optimized_graph = self._optimize_layout_graph(layout_graph)

        # Fuse modalities
        fused_embeddings = self._fuse_modalities(text_embeddings, visual_embeddings)
        fused_embeddings_layout = self._project_embeddings_for_layout(fused_embeddings)

        # Select and apply specialist
        specialist_name = self._select_specialist_for_page(parsed_objects)

        if self.specialists_loaded and len(fused_embeddings) > 0:
            fused_embeddings_processed = self._process_with_specialist(
                fused_embeddings[0], specialist_name
            )
            # Reshape to 2D
            fused_embeddings = fused_embeddings_processed.reshape(1, -1)

        # Crystallize to galaxy position
        galaxy_position = self._crystallize_to_galaxy(optimized_graph, fused_embeddings_layout)

        # Create Galaxy star for knowledge storage
        star_metadata = {
            'pdf_path': str(pdf_path),
            'page_number': page_num,
            'object_count': int(parsed_objects.get('object_count', 0)),
            'method': parsed_objects.get('method', 'structured'),
            'specialist_used': specialist_name,
            'embedding_dim': fused_embeddings.shape[1] if len(fused_embeddings) > 0 else 0
        }

        galaxy_star = self._create_galaxy_star(
            fused_embeddings[0] if len(fused_embeddings) > 0 else np.zeros(128, dtype=np.float32),
            star_metadata
        )

        # Store star in Galaxy (knowledge storage, not model weights!)
        self.galaxy_stars.append(galaxy_star)
        if len(fused_embeddings) > 0:
            self.galaxy_star_embeddings.append(fused_embeddings[0])

        self._cleanup_gpu_buffers()

        processing_time_ms = (time.perf_counter() - start_time) * 1_000.0

        # Return result with Phase G metadata
        result = {
            "galaxy_position": galaxy_position,
            "galaxy_star": galaxy_star,
            "layout_graph": optimized_graph,
            "embeddings": fused_embeddings,
            "embedding_dimension": fused_embeddings.shape[1] if len(fused_embeddings) > 0 else 0,
            "object_count": int(parsed_objects.get("object_count", 0)),
            "processing_time_ms": float(processing_time_ms),
            "method": parsed_objects.get("method", "structured"),
            "specialist_used": specialist_name,
            "text": parsed_objects.get("text", ""),
        }

        # Periodic save
        if page_num > 0 and page_num % 100 == 0:
            self.save_galaxy_stars()
            self.adaptive_rpn.save_all(self.embeddings_path.parent / 'adaptive_rpn')

        return result

    def save_galaxy_stars(self):
        """
        Save Galaxy stars (knowledge storage).

        Stars will be materialized into House objects during sleep consolidation.
        """
        import pickle

        galaxy_path = self.embeddings_path.parent / 'galaxy_stars.pkl'
        galaxy_path.parent.mkdir(parents=True, exist_ok=True)

        with open(galaxy_path, 'wb') as f:
            pickle.dump({
                'stars': self.galaxy_stars,
                'embeddings': self.galaxy_star_embeddings,
                'total_stars': len(self.galaxy_stars)
            }, f)

        print(f"[PhaseG] Saved {len(self.galaxy_stars)} Galaxy stars to {galaxy_path}")

    def load_galaxy_stars(self):
        """Load existing Galaxy stars."""
        import pickle

        galaxy_path = self.embeddings_path.parent / 'galaxy_stars.pkl'

        if not galaxy_path.exists():
            print(f"[PhaseG] No existing Galaxy stars at {galaxy_path}")
            return

        with open(galaxy_path, 'rb') as f:
            data = pickle.load(f)

        self.galaxy_stars = data.get('stars', [])
        self.galaxy_star_embeddings = data.get('embeddings', [])

        print(f"[PhaseG] Loaded {len(self.galaxy_stars)} Galaxy stars from {galaxy_path}")

    def get_phase_g_stats(self) -> Dict[str, Any]:
        """Get Phase G statistics."""
        stats = {
            'specialists_loaded': self.specialists_loaded,
            'galaxy_stars': len(self.galaxy_stars),
            'adaptive_rpn_stats': self.adaptive_rpn.get_stats(),
            'gpu_ocr_enabled': self.gpu_ocr_enabled,
            'ocr_stats': dict(self.ocr_stats),
            'ocr_foundations': dict(self._foundational_status),
        }

        if self.matryoshka_system:
            stats['matryoshka_stats'] = self.matryoshka_system.get_system_stats()

        return stats

    def print_phase_g_stats(self):
        """Print Phase G statistics."""
        stats = self.get_phase_g_stats()

        print("\n" + "="*60)
        print("PHASE G PDF INGESTION STATISTICS")
        print("="*60)
        print(f"Specialists loaded: {stats['specialists_loaded']}")
        print(f"Galaxy stars created: {stats['galaxy_stars']:,}")

        # Adaptive RPN stats
        self.adaptive_rpn.print_stats()

        # GPU OCR stats
        ocr_stats = stats['ocr_stats']
        attempts = ocr_stats.get('attempts', 0)
        success = ocr_stats.get('gpu_success', 0)
        fail = ocr_stats.get('gpu_fail', 0)
        fallback = ocr_stats.get('fallback', 0)
        skipped = ocr_stats.get('skipped', 0)
        success_rate = (success / attempts * 100.0) if attempts else 0.0
        print("\nGPU OCR:")
        print(f"  Enabled: {stats['gpu_ocr_enabled']}")
        print(f"  Attempts: {attempts}, Success: {success}, Fail: {fail}, Fallback: {fallback}, Skipped: {skipped}")
        print(f"  Success rate: {success_rate:.1f}%")

        # Matryoshka stats
        if 'matryoshka_stats' in stats:
            mat_stats = stats['matryoshka_stats']
            print("\nMatryoshka System:")
            print(f"  Base model: {mat_stats['base_model']['max_dims']}D")
            print(f"  Specialists: {mat_stats['num_specialists']}")
            print(f"  Total params: {mat_stats['total_params']/1e6:.2f}M")

        print("="*60 + "\n")


__all__ = ['PhaseGPDFIngestionBridge']
