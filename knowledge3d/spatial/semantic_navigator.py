"""Semantic Navigator – unified Morton octree + LED-A* integration.

This module packages the GPU-native spatial (Morton octree) and semantic
(LED-A*) pipeline so runtime components can answer “what is nearby?” queries and
plan routes between labelled objects in the House.

The implementation intentionally mirrors the demo in
``examples/semantic_navigator_demo.py`` but adds caching helpers, metadata
lookups and persistence hooks required by fused_head and the live server.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:  # CuPy is mandatory for GPU execution
    import cupy as cp  # type: ignore
    CUPY_AVAILABLE = True
except Exception:  # pragma: no cover - runtime guard
    cp = None  # type: ignore
    CUPY_AVAILABLE = False

try:
    from pygltflib import GLTF2  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "SemanticNavigator requires pygltflib; install via `pip install pygltflib`."
    ) from exc

if CUPY_AVAILABLE:
    from knowledge3d.spatial.morton_octree import MortonOctree
    from knowledge3d.spatial.led_pathfinder import LEDPathfinder
    from knowledge3d.spatial.domain_splitter import SemanticDomainSplitter
    from knowledge3d.spatial.multi_domain_navigator import MultiDomainNavigator
else:  # pragma: no cover - fall back to placeholders so imports succeed
    MortonOctree = None  # type: ignore
    LEDPathfinder = None  # type: ignore
    SemanticDomainSplitter = None  # type: ignore
    MultiDomainNavigator = None  # type: ignore


_logger = logging.getLogger(__name__)


class SemanticNavigator:
    """High-level coordinator for GPU spatial + semantic navigation."""

    def __init__(
        self,
        *,
        query_radius: float = 2.0,
        k_neighbors: int = 8,
        similarity_threshold: float = 0.7,
        enable_semantic_highways: bool = True,
        nav_mode: Optional[str] = None,
    ) -> None:
        if not CUPY_AVAILABLE:
            raise RuntimeError(
                "SemanticNavigator requires CuPy/CUDA; install cupy-cudaXX in the k3d env."
            )

        self.query_radius = float(query_radius)
        self.k_neighbors = int(max(1, k_neighbors))
        self.similarity_threshold = float(similarity_threshold)
        self.enable_semantic_highways = bool(enable_semantic_highways)

        # Phase 3: Strategy pattern for multi-domain navigation
        self.nav_mode = nav_mode or os.getenv("K3D_NAV_MODE", "auto")

        self._glb_path: Optional[Path] = None
        self._glb_mtime: Optional[float] = None

        self.positions_gpu: Optional[cp.ndarray] = None
        self.positions_cpu: Optional[np.ndarray] = None
        self.embeddings_gpu: Optional[cp.ndarray] = None
        self.embeddings_cpu: Optional[np.ndarray] = None

        self.labels: List[str] = []
        self.label_to_idx: Dict[str, int] = {}
        self.metadata: List[Dict[str, object]] = []

        self.octree: Optional[MortonOctree] = None
        self.pathfinder: Optional[LEDPathfinder] = None
        self._kernel_built: bool = False

        # Phase 3: Multi-domain navigator (strategy pattern)
        self.multi_domain_navigator: Optional[MultiDomainNavigator] = None
        self._use_multi_domain: bool = False
        self.domain_splitter: Optional[Any] = None  # Qwen's integration: for export

    # ------------------------------------------------------------------
    # Loading utilities
    # ------------------------------------------------------------------
    def load_house(self, glb_path: str | Path) -> None:
        """Load positions, embeddings and metadata directly from a House GLB."""
        path = Path(glb_path)
        if not path.exists():
            raise FileNotFoundError(f"House GLB not found: {path}")

        gltf = GLTF2().load(str(path))
        blob = gltf.binary_blob() or b""

        primitive = self._select_primitive(gltf)
        if primitive is None:
            raise RuntimeError("Unable to locate mesh primitive with POSITION attribute")

        pos_accessor_idx = self._get_accessor_index(primitive, "POSITION")
        positions = self._read_accessor(gltf, pos_accessor_idx, blob)
        if positions.shape[1] != 3:
            raise RuntimeError("POSITION accessor must be VEC3")

        self.positions_cpu = positions.astype(np.float32, copy=True)
        self.positions_gpu = cp.asarray(self.positions_cpu)

        k3d_payload = {}
        if primitive.extras and isinstance(primitive.extras, dict):
            k3d_payload = primitive.extras.get("k3d", {}) or {}

        self.metadata = []
        if isinstance(k3d_payload, dict):
            meta = k3d_payload.get("metadata")
            if isinstance(meta, list):
                self.metadata = meta

        emb_dims = None
        emb_view = None
        if isinstance(k3d_payload, dict):
            emb_dims = k3d_payload.get("embeddingDims")
            emb_view = k3d_payload.get("embeddingsView")

        if isinstance(emb_dims, int) and isinstance(emb_view, int):
            embeddings = self._read_buffer_view(gltf, emb_view, blob, (len(positions), emb_dims))
            self.embeddings_cpu = embeddings.astype(np.float32, copy=False)
            self.embeddings_gpu = cp.asarray(self.embeddings_cpu)
        else:
            # Fallback to positional embeddings
            self.embeddings_cpu = self.positions_cpu.copy()
            self.embeddings_gpu = self.positions_gpu.copy()
            _logger.warning("No embeddings found in GLB; using positions as proxy")

        raw_labels = []
        if isinstance(k3d_payload, dict):
            lbls = k3d_payload.get("labels")
            if isinstance(lbls, list):
                raw_labels = [str(x) for x in lbls]

        self.labels = self._derive_labels(raw_labels)
        self.label_to_idx = {lbl.lower(): idx for idx, lbl in enumerate(self.labels)}

        self._glb_path = path
        self._glb_mtime = path.stat().st_mtime

        # Reset caches – octree/kernel will be lazily rebuilt on demand
        self.octree = None
        self.pathfinder = None
        self._kernel_built = False

        _logger.info(
            "SemanticNavigator loaded %d nodes (%s), embeddings=%s",
            len(self.labels),
            path,
            self.embeddings_cpu.shape if self.embeddings_cpu is not None else None,
        )

    # ------------------------------------------------------------------
    # Public query helpers
    # ------------------------------------------------------------------
    def query_near(
        self,
        anchor_label: str,
        radius: Optional[float] = None,
        *,
        max_results: int = 10,
    ) -> List[Tuple[str, float, Optional[Dict[str, object]]]]:
        """Return labels within ``radius`` of ``anchor_label`` (sorted by distance)."""
        idx = self._resolve_label(anchor_label)
        if idx is None:
            raise ValueError(f"Unknown label: {anchor_label}")

        self._ensure_octree()
        assert self.positions_gpu is not None and self.positions_cpu is not None

        radius_value = float(radius if radius is not None else self.query_radius)
        center = self.positions_gpu[idx]
        candidates = self.octree.query_radius_gpu(
            center,
            radius_value,
            refine_euclidean=True,
            max_results=max(self.k_neighbors * 6, max_results * 3),
        )
        candidates_cpu = cp.asnumpy(candidates) if candidates is not None else np.array([], dtype=np.uint32)

        results: List[Tuple[str, float, Optional[Dict[str, object]]]] = []
        seen = set()
        for cand in candidates_cpu:
            if int(cand) == idx:
                continue
            if int(cand) in seen:
                continue
            seen.add(int(cand))
            label = self.labels[int(cand)]
            dist = float(
                math.sqrt(
                    ((self.positions_cpu[int(cand)] - self.positions_cpu[idx]) ** 2).sum()
                )
            )
            meta = None
            if int(cand) < len(self.metadata):
                entry = self.metadata[int(cand)]
                if isinstance(entry, dict):
                    meta = entry
            results.append((label, dist, meta))

        results.sort(key=lambda item: item[1])
        return results[:max_results]

    def find_path(
        self,
        start_label: str,
        goal_label: str,
        *,
        alpha: float = 0.7,
        beta: float = 0.3,
        max_path_length: int = 512,
    ) -> Tuple[List[str], float]:
        """Compute semantic path between two labels using LED-A* (strategy pattern)."""
        start_idx = self._resolve_label(start_label)
        if start_idx is None:
            raise ValueError(f"Unknown start label: {start_label}")
        goal_idx = self._resolve_label(goal_label)
        if goal_idx is None:
            raise ValueError(f"Unknown goal label: {goal_label}")

        self._ensure_kernel()

        # Phase 3: Delegate to appropriate backend
        if self._use_multi_domain and self.multi_domain_navigator is not None:
            # Multi-domain navigation (>1000 nodes)
            return self.multi_domain_navigator.navigate(
                start_label,
                goal_label,
                alpha=alpha,
                beta=beta
            )
        else:
            # Monolithic navigation (<1000 nodes)
            assert self.pathfinder is not None
            path_indices, cost = self.pathfinder.find_path(
                start_idx,
                goal_idx,
                alpha=alpha,
                beta=beta,
                max_path_length=max_path_length,
            )
            labels = [self.labels[i] for i in path_indices]
            return labels, cost

    def ensure_octree(self) -> None:
        """Public wrapper so callers can prime the octree without triggering kernel builds."""
        self._ensure_octree()

    def ensure_kernel(self) -> None:
        """Public wrapper for explicit kernel builds (sleep-time integration)."""
        self._ensure_kernel()

    def resolve_label(self, label: str) -> Optional[int]:
        """Expose label resolution for external callers."""
        return self._resolve_label(label)

    def serialize(self, output_dir: str | Path) -> None:
        """Persist the current kernel to disk for reuse across sessions."""
        if not self._kernel_built or self.pathfinder is None:
            raise RuntimeError("Kernel not built – call find_path or ensure_kernel() first")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        kernel_path = out_dir / "semantic_kernel.npz"
        metadata_path = out_dir / "semantic_navigator.json"

        self.pathfinder.serialize_kernel(kernel_path)

        payload = {
            "glb_path": str(self._glb_path) if self._glb_path else None,
            "glb_mtime": self._glb_mtime,
            "labels": self.labels,
            "query_radius": self.query_radius,
            "k_neighbors": self.k_neighbors,
            "similarity_threshold": self.similarity_threshold,
            "enable_semantic_highways": self.enable_semantic_highways,
        }
        metadata_path.write_text(json_dumps(payload), encoding="utf-8")
        _logger.info("Serialized semantic kernel → %s", kernel_path)

    def load_serialized(self, output_dir: str | Path) -> bool:
        """Load previously serialized kernel if metadata matches current GLB."""
        out_dir = Path(output_dir)
        kernel_path = out_dir / "semantic_kernel.npz"
        metadata_path = out_dir / "semantic_navigator.json"
        if not kernel_path.exists() or not metadata_path.exists():
            return False

        payload = json_load(metadata_path)
        if not payload:
            return False

        if self._glb_path is None:
            _logger.warning("Navigator not initialised with GLB; cannot load cached kernel")
            return False

        if payload.get("glb_path") != str(self._glb_path):
            _logger.info("Cached kernel belongs to a different GLB; rebuilding")
            return False
        cached_mtime = payload.get("glb_mtime")
        if cached_mtime is None or abs(float(cached_mtime) - float(self._glb_mtime or 0.0)) > 1e-6:
            _logger.info("Cached kernel is stale (mtime mismatch); rebuilding")
            return False

        if self.pathfinder is None:
            self.pathfinder = LEDPathfinder()
        self.pathfinder.load_serialized_kernel(kernel_path)
        self._kernel_built = True
        _logger.info("Loaded cached semantic kernel (%s)", kernel_path)
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_octree(self) -> None:
        if self.octree is not None:
            return
        if self.positions_gpu is None:
            raise RuntimeError("Navigator not initialised with positions")
        self.octree = MortonOctree()
        self.octree.build_from_gpu_positions(self.positions_gpu)

    def _ensure_kernel(self) -> None:
        if self._kernel_built:
            return
        if self.positions_gpu is None or self.embeddings_cpu is None:
            raise RuntimeError("Navigator missing positions/embeddings")
        self._ensure_octree()

        # Phase 3: Determine navigation strategy
        num_nodes = len(self.labels)
        use_multi_domain = (
            self.nav_mode == "multi" or
            (self.nav_mode == "auto" and num_nodes > 1000)
        )

        if use_multi_domain:
            _logger.info(f"Using multi-domain navigation for {num_nodes} nodes")
            self._build_multi_domain_kernel()
        else:
            _logger.info(f"Using monolithic navigation for {num_nodes} nodes")
            self._build_monolithic_kernel()

        self._kernel_built = True

    def _build_monolithic_kernel(self) -> None:
        """Build single LED-A* kernel (for <1000 nodes)."""
        if self.pathfinder is None:
            self.pathfinder = LEDPathfinder()

        edges = self._build_edges_from_octree()
        self.pathfinder.build_kernel_from_octree(
            edges,
            self.embeddings_cpu,
            self.positions_cpu,
            similarity_threshold=self.similarity_threshold,
            enable_semantic_highways=self.enable_semantic_highways,
        )

    def _build_multi_domain_kernel(self) -> None:
        """Build multi-domain navigator with semantic clustering (Phase 3)."""
        edges = self._build_edges_from_octree()
        edges_gpu = cp.asarray(edges, dtype=cp.uint32)

        # Run semantic domain splitter (Qwen's integration: store for export)
        self.domain_splitter = SemanticDomainSplitter(
            sim_threshold=self.similarity_threshold,
            damping=0.9,
            adaptive_threshold=True,  # GLM's enhancement
            render_bridges=True  # GLM's visualization
        )

        domain_ids, bridges, domains = self.domain_splitter.split_domains(
            self.embeddings_gpu,
            self.positions_gpu,
            edges_gpu,
            kb_limit=128  # Increased from 48KB to match LED pathfinder limit
        )

        # Create multi-domain navigator
        self.multi_domain_navigator = MultiDomainNavigator(
            domains=domains,
            bridges=bridges,
            embeddings_gpu=self.embeddings_gpu,
            labels=self.labels,
            domain_ids=domain_ids
        )

        self._use_multi_domain = True
        _logger.info(f"✓ Multi-domain kernel: {len(domains)} domains, {len(bridges)} bridges")

    def _build_edges_from_octree(self) -> np.ndarray:
        assert self.positions_gpu is not None and self.positions_cpu is not None
        num_nodes = len(self.labels)
        edges_set: set[Tuple[int, int]] = set()

        for idx in range(num_nodes):
            center = self.positions_gpu[idx]
            candidates = self.octree.query_radius_gpu(
                center,
                self.query_radius,
                refine_euclidean=True,
                max_results=self.k_neighbors * 6,
            )
            if candidates.size == 0:  # type: ignore[attr-defined]
                continue
            cand_cpu = cp.asnumpy(candidates)
            if cand_cpu.size == 0:
                continue
            # sort by actual distance using CPU positions
            diffs = self.positions_cpu[cand_cpu] - self.positions_cpu[idx]
            dists = np.linalg.norm(diffs, axis=1)
            order = np.argsort(dists)
            for ridx in order[: self.k_neighbors]:
                neighbor = int(cand_cpu[ridx])
                if neighbor == idx:
                    continue
                edges_set.add((idx, neighbor))

        if not edges_set:
            raise RuntimeError("No edges generated for dependency kernel; adjust radius or neighbors")

        edges = np.array(sorted(edges_set), dtype=np.uint32)
        _logger.info("Generated %d edges for semantic kernel", len(edges))
        return edges

    def _resolve_label(self, label: str) -> Optional[int]:
        if not label:
            return None
        idx = self.label_to_idx.get(label.lower())
        if idx is not None:
            return idx
        # fallback to substring search
        matches = [
            (name, pos)
            for name, pos in self.label_to_idx.items()
            if label.lower() in name
        ]
        if len(matches) == 1:
            return matches[0][1]
        return None

    # ------------------ glTF helpers ------------------
    def _select_primitive(self, gltf: GLTF2):
        for mesh in gltf.meshes or []:
            for primitive in mesh.primitives or []:
                attrs = primitive.attributes or {}
                if isinstance(attrs, dict) and "POSITION" in attrs:
                    return primitive
        if gltf.meshes and gltf.meshes[0].primitives:
            return gltf.meshes[0].primitives[0]
        return None

    def _get_accessor_index(self, primitive, attribute: str) -> int:
        attrs = primitive.attributes
        if isinstance(attrs, dict):
            value = attrs.get(attribute)
        else:
            value = getattr(attrs, attribute, None)
        if not isinstance(value, int):
            raise RuntimeError(f"Primitive lacks {attribute} accessor")
        return value

    def _read_accessor(self, gltf: GLTF2, accessor_idx: int, blob: bytes) -> np.ndarray:
        accessor = gltf.accessors[accessor_idx]
        buffer_view = gltf.bufferViews[accessor.bufferView]

        component_type = accessor.componentType
        dtype_map = {
            5126: np.float32,
            5125: np.uint32,
            5123: np.uint16,
        }
        if component_type not in dtype_map:
            raise RuntimeError(f"Unsupported component type: {component_type}")
        dtype = dtype_map[component_type]

        type_map = {
            "SCALAR": 1,
            "VEC2": 2,
            "VEC3": 3,
            "VEC4": 4,
            "MAT4": 16,
        }
        comp_count = type_map.get(accessor.type)
        if comp_count is None:
            raise RuntimeError(f"Unsupported accessor type: {accessor.type}")

        stride = buffer_view.byteStride or comp_count * np.dtype(dtype).itemsize
        offset = (buffer_view.byteOffset or 0) + accessor.byteOffset
        count = accessor.count

        if stride == comp_count * np.dtype(dtype).itemsize:
            data = np.frombuffer(
                blob,
                dtype=dtype,
                count=count * comp_count,
                offset=offset,
            )
            return data.reshape((count, comp_count)).copy()

        # Handle strided/interleaved data
        rows = []
        for i in range(count):
            start = offset + i * stride
            chunk = np.frombuffer(
                blob[start : start + comp_count * np.dtype(dtype).itemsize],
                dtype=dtype,
                count=comp_count,
            )
            rows.append(chunk)
        return np.vstack(rows)

    def _read_buffer_view(
        self,
        gltf: GLTF2,
        view_idx: int,
        blob: bytes,
        shape: Tuple[int, int],
        dtype: np.dtype = np.float32,
    ) -> np.ndarray:
        view = gltf.bufferViews[view_idx]
        offset = view.byteOffset or 0
        stride = view.byteStride or 0
        count, width = shape
        itemsize = np.dtype(dtype).itemsize
        if stride in (0, width * itemsize):
            length = count * width * itemsize
            data = np.frombuffer(blob, dtype=dtype, count=count * width, offset=offset)
            return data.reshape(shape).copy()
        rows = []
        for i in range(count):
            start = offset + i * stride
            chunk = np.frombuffer(blob[start:start + width * itemsize], dtype=dtype, count=width)
            rows.append(chunk)
        return np.vstack(rows)

    def _derive_labels(self, raw_labels: List[str]) -> List[str]:
        num_positions = len(self.positions_cpu) if self.positions_cpu is not None else 0
        if raw_labels and len(raw_labels) == num_positions:
            labels = [str(lbl) for lbl in raw_labels]
        else:
            labels = []
            for idx in range(num_positions):
                label = None
                if idx < len(self.metadata):
                    meta = self.metadata[idx]
                    if isinstance(meta, dict):
                        label = meta.get("label") or meta.get("name") or meta.get("id")
                labels.append(str(label or f"node_{idx}"))
        # Deduplicate while preserving order
        seen = {}
        deduped = []
        for lbl in labels:
            key = lbl.lower()
            if key in seen:
                seen[key] += 1
                deduped.append(f"{lbl}_{seen[key]}")
            else:
                seen[key] = 0
                deduped.append(lbl)
        return deduped


# ---------------------------------------------------------------------------
# Lightweight JSON helpers (avoid importing json at module load if unused)
# ---------------------------------------------------------------------------

def json_dumps(payload: Dict[str, object]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)


def json_load(path: Path) -> Optional[Dict[str, object]]:
    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
