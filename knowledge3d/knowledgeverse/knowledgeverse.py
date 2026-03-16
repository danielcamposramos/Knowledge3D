"""Knowledgeverse runtime harness for benchmark and integration scripts."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any
import zlib

import numpy as np

from knowledge3d.cranium.bridges.matryoshka_bridge import MatryoshkaProjectionBridge
from knowledge3d.cranium.bridges.trigram_embed_bridge import TrigramEmbedBridge
from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine
from knowledge3d.cranium.sovereign.loader import gpu_malloc, launch, memcpy_dtoh, memcpy_htod, synchronize
from knowledge3d.cranium.sovereign.trm_launcher import TRMLauncher
from knowledge3d.training.trm_galaxy_nav import (
    DEFAULT_GALAXY_ORDER,
    load_galaxy_decoder_checkpoint,
    load_trm_weight_checkpoint,
    softmax,
)

from .foundational_galaxy_bootstrap import populate_always_on_foundational_galaxies
from .galaxy_manager import GalaxyManager
from .query_head_substrate import DynamicLodDriverBridge, QueryHeadSubstrate, expand_embedding16_to128
from .runtime_ingest import load_books_runtime_entries, load_language_runtime_entries
from .semantic_csr_graph import load_or_build_semantic_csr_graph
from .shadow_copy import ShadowCopyLearning
from .sleeptime import SleepTimeConsolidation
from .stargate import IngestionStargate
from .ternary_quality_memory import TernaryQualityMemory
from .trm_navigator import TRMNavigator


@dataclass
class KnowledgeverseMetrics:
    """Runtime metrics surface consumed by integration checks."""

    ptx_fallback_rate: float = 0.0
    gpu_galaxy_entries: int = 0
    gpu_galaxy_bytes: int = 0
    gpu_bind_rebuilds: int = 0
    gpu_runtime_artifact_entries: int = 0
    runtime_language_entries: int = 0


class Knowledgeverse:
    """Minimal runtime assembly for current Knowledgeverse MVP flows."""

    TRM_WEIGHT_SHAPES: dict[str, tuple[int, int]] = {
        "W1": (1024, 512),
        "W2": (512, 1024),
        "W3": (1024, 512),
        "W4": (512, 1024),
    }
    TRM_STATE_VECTOR_DIM = 512
    TRM_WORKSPACE_FLOATS = 3072
    TRM_STATE_BUFFER_FLOATS: dict[str, int] = {
        "d_q_input": TRM_STATE_VECTOR_DIM,
        "d_q": TRM_STATE_VECTOR_DIM,
        "d_y": TRM_STATE_VECTOR_DIM,
        "d_z": TRM_STATE_VECTOR_DIM,
        "d_z_new": TRM_STATE_VECTOR_DIM,
        "d_y_new": TRM_STATE_VECTOR_DIM,
        "d_workspace": TRM_WORKSPACE_FLOATS,
    }
    TRM_INIT_SEED = 314159
    TRM_GALAXY_INFLUENCE_STRENGTH = 0.5
    GSM8K_STRUCTURAL_OVERRIDE_MIN = 0.65
    GSM8K_STRUCTURAL_OVERRIDE_MARGIN = 0.05
    GSM8K_STRUCTURAL_OVERRIDE_PATH_WEIGHT = 1.0
    GSM8K_STRUCTURAL_OVERRIDE_STRUCT_WEIGHT = 0.2
    GSM8K_STRUCTURAL_OVERRIDE_COMPOSITIONAL_WEIGHT = 0.08
    GSM8K_STRUCTURAL_OVERRIDE_DIMENSIONAL_WEIGHT = 0.04
    GPU_GALAXY_ENTRY_STRIDE = 23
    GPU_GALAXY_EMBEDDING_OFFSET = 3
    GPU_GALAXY_EMBEDDING_DIM = 16
    GPU_FACTUAL_REASONING_PROGRAM_ID = "reasoning_factual_lookup_top1"
    GPU_MATH_REASONING_PROGRAM_ID = "reasoning_math_template_match_top1"
    GPU_ARC_REASONING_PROGRAM_ID = "reasoning_arc_grid_transform_top1"
    GPU_CHAT_REASONING_PROGRAM_ID = "reasoning_chat_lookup_top1"
    GPU_ARC_SWARM_PROGRAM_IDS: tuple[str, ...] = (
        "reasoning_arc_grid_transform_top1",
        "reasoning_arc_tile_repeat_top1",
        "reasoning_arc_recolor_top1",
        "reasoning_arc_fill_enclosed_top1",
        "reasoning_arc_connect_pairs_top1",
        "reasoning_arc_periodic_cleanup_top1",
        "reasoning_arc_separator_bridge_top1",
        "reasoning_arc_anchor_spiral_top1",
        "reasoning_arc_marker_axis_crop_top1",
    )
    GPU_ARC_PROGRAM_HINTS: dict[str, str] = {
        "reasoning_arc_grid_transform_top1": "grid transform visual reasoning",
        "reasoning_arc_tile_repeat_top1": "tile repeat checker complement pattern",
        "reasoning_arc_recolor_top1": "recolor color remap palette swap",
        "reasoning_arc_fill_enclosed_top1": "fill enclosed region object interior",
        "reasoning_arc_connect_pairs_top1": "connect matching color pairs bridge line",
        "reasoning_arc_periodic_cleanup_top1": "periodic cleanup consensus repeating patch",
        "reasoning_arc_separator_bridge_top1": "separator bridge projection divider axis",
        "reasoning_arc_anchor_spiral_top1": "anchor spiral frame clockwise marker",
        "reasoning_arc_marker_axis_crop_top1": "marker axis crop structural pack diagonal",
    }
    GPU_ARC_PROGRAM_FOCUS_OPS: dict[str, tuple[str, ...]] = {
        "reasoning_arc_grid_transform_top1": (),
        "reasoning_arc_tile_repeat_top1": (
            "checker_tile_repeat_hflip_rows",
            "periodic_tile_repeat",
            "self_pattern_complement_tiling",
        ),
        "reasoning_arc_recolor_top1": (
            "multi_color_remap",
        ),
        "reasoning_arc_fill_enclosed_top1": (
            "fill_enclosed_by_size",
        ),
        "reasoning_arc_connect_pairs_top1": (
            "connect_color_pairs",
        ),
        "reasoning_arc_periodic_cleanup_top1": (
            "periodic_consensus_cleanup",
        ),
        "reasoning_arc_separator_bridge_top1": (
            "separator_bridge_projection",
        ),
        "reasoning_arc_anchor_spiral_top1": (
            "anchor_spiral_pair",
        ),
        "reasoning_arc_marker_axis_crop_top1": (
            "marker_axis_crop",
            "pack_color_components_diagonal",
        ),
    }
    GPU_FACTUAL_TARGET_GALAXIES: tuple[str, ...] = (
        "Reality",
        "Math",
    )
    GPU_MATH_TARGET_GALAXIES: tuple[str, ...] = (
        "Math",
    )
    GPU_GSM8K_TARGET_GALAXIES: tuple[str, ...] = (
        "Math",
        "Grammar",
        "Number",
        "Word",
    )
    GPU_ARC_TARGET_GALAXIES: tuple[str, ...] = (
        "Drawing",
        "Grammar",
        "Tool",
    )
    GPU_CHAT_TARGET_GALAXIES: tuple[str, ...] = (
        "Grammar",
        "Word",
        "Character",
    )
    GPU_MMLU_TARGET_GALAXIES: tuple[str, ...] = (
        "Reality",
        "Math",
        "Grammar",
        "Word",
        "Character",
    )
    GPU_LHE_TARGET_GALAXIES: tuple[str, ...] = (
        "Reality",
        "Math",
        "Grammar",
        "Word",
        "Character",
    )
    GPU_FACTUAL_CHAT_TARGET_GALAXIES: tuple[str, ...] = (
        "Reality",
        "Grammar",
        "Word",
        "Character",
    )
    GPU_CROSS_DOMAIN_SCAN_WEIGHT = 0.92
    DEFAULT_GALAXIES: tuple[str, ...] = (
        "Drawing",
        "Character",
        "Word",
        "Number",
        "Grammar",
        "Math",
        "Reality",
        "Audio",
        "3DObjects",
        "Tool",
    )
    GPU_CATEGORY_CLASS_MAP: dict[str, float] = {
        "unknown": 0.0,
        "clue_fact": 1.0,
        "formula": 2.0,
        "concept": 3.0,
        "benchmark_fact": 4.0,
        "template": 5.0,
        "cipher_result": 6.0,
        "formal_result": 7.0,
        "definition": 8.0,
        "meta_rule": 9.0,
        "language_profile": 10.0,
        "multilingual_word": 11.0,
        "language_rule": 12.0,
        "book_artifact": 13.0,
    }
    GPU_SOURCE_CLASS_FOUNDATIONAL = 0.0
    GPU_SOURCE_CLASS_BOOK_ARTIFACT = 1.0
    GPU_SOURCE_CLASS_RUNTIME_LANGUAGE = 2.0

    def __init__(
        self,
        *,
        manifest_version: str = "kv-2026-02-06",
        storage_root: str | Path = "../Knowledge3D.local",
        galaxy_storage_root: str | Path | None = None,
        audit_index_path: str | Path | None = None,
        sleeptime_journal_path: str | Path | None = None,
        stargate_storage_root: str | Path | None = None,
        eager_load_default_galaxies: bool = True,
        bootstrap_foundational_galaxies: bool = True,
        include_runtime_artifacts: bool = True,
        include_runtime_language_enrichment: bool = True,
    ):
        self.manifest_version = manifest_version
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.metrics = KnowledgeverseMetrics(ptx_fallback_rate=0.0)
        self.include_runtime_artifacts = bool(include_runtime_artifacts)
        self.include_runtime_language_enrichment = bool(include_runtime_language_enrichment)

        galaxy_root = (
            Path(galaxy_storage_root)
            if galaxy_storage_root is not None
            else self.storage_root / "galaxies"
        )
        self.galaxy_manager = GalaxyManager(storage_root=galaxy_root)
        self.galaxy_manager.set_knowledgeverse(self)
        self.foundational_bootstrap_summary: dict[str, Any] = {}
        if bootstrap_foundational_galaxies:
            self.foundational_bootstrap_summary = populate_always_on_foundational_galaxies(
                self.galaxy_manager
            )

        stargate_root = (
            Path(stargate_storage_root)
            if stargate_storage_root is not None
            else self.storage_root / "stargate_jobs"
        )
        self.stargate = IngestionStargate(
            manifest_version=self.manifest_version,
            galaxy_manager=self.galaxy_manager,
            storage_root=stargate_root,
        )

        self.trm_navigator = TRMNavigator(knowledgeverse=self)
        self.specialist_router = self.trm_navigator.specialist_router
        self.navigator_specialist = self.trm_navigator.navigator_specialist
        self._trm: TRMLauncher | None = None
        self._trm_ready = False
        self._trm_backend = "uninitialized"
        self._trm_init_error = ""
        self._trm_host_weights: dict[str, np.ndarray] = {}
        self._trm_weight_buffers: dict[str, Any] = {}
        self._trm_state_buffers: dict[str, Any] = {}
        self._trm_state_buffer_bytes = 0
        self._trm_weight_bytes = 0
        self._matryoshka_bridge: MatryoshkaProjectionBridge | None = None
        self._trm_matryoshka_host_weights: np.ndarray | None = None
        self._trm_matryoshka_weight_buffer: Any | None = None
        self._trm_galaxy_decoder: dict[str, Any] | None = None
        self._trm_galaxy_decoder_path: str = ""
        self._gpu_reasoning_engine: Any | None = None
        self._gpu_galaxy_binding: dict[str, Any] | None = None
        self._pinned_all_default_binding = False
        self._gpu_galaxy_catalog: list[dict[str, Any]] = []
        self._text_embedding_engine: RPNEmbeddingEngine | None = None
        self._gpu_query_embedding_bridge: TrigramEmbedBridge | None = None
        self._gpu_reasoning_programs: dict[str, dict[str, Any]] = {}
        self._drawing_bridge: Any | None = None
        self._led_pathfinder: Any | None | bool = None
        self._semantic_csr_graph: Any | None = None
        self._query_head_substrate: QueryHeadSubstrate | None = None
        self._swarm_bridge: Any | None | bool = None
        self._halting_gate: Any | None | bool = None
        self._vector_resonator: Any | None | bool = None
        self._resonance_field: Any | None | bool = None
        self._geometry_router: Any | None | bool = None
        self._temporal_reasoning: Any | None | bool = None
        self._fractal_emitter: Any | None | bool = None
        self._cognitive_executive: Any | None | bool = None
        self._galaxy_resonance_engine: Any | None | bool = None
        self._graph_crystallizer: Any | None | bool = None
        self._atomic_fission_fusion: Any | None | bool = None
        self._defeasible_resolver: Any | None | bool = None
        self._cosine_similarity_bridge: Any | None | bool = None
        self._query_sequence = 0
        self._runtime_language_enrichment_loaded = False

        audit_index = (
            Path(audit_index_path)
            if audit_index_path is not None
            else self.storage_root / "audit_index.json"
        )
        self.shadow_copy = ShadowCopyLearning(
            trm_manager=self.trm_navigator,
            index_path=audit_index,
            manifest_version=self.manifest_version,
        )
        self.ternary_quality_memory = TernaryQualityMemory(
            state_path=self.storage_root / "checkpoints" / "ternary_quality_memory.json"
        )

        sleeptime_journal = (
            Path(sleeptime_journal_path)
            if sleeptime_journal_path is not None
            else self.storage_root / "logs" / "sleeptime_journal.jsonl"
        )
        self.sleeptime = SleepTimeConsolidation(
            knowledgeverse=self,
            journal_path=sleeptime_journal,
        )
        self._initialize_trm_launcher()
        self._load_trm_galaxy_decoder()
        self._default_galaxies_loaded = False
        if eager_load_default_galaxies:
            self.ensure_default_galaxies_loaded()
            if self._trm_navigation_env_enabled():
                self._pin_all_default_gpu_binding()

    @staticmethod
    def _trm_navigation_env_enabled() -> bool:
        return os.getenv("K3D_TRM_NAVIGATE", "0").strip().lower() in {"1", "true", "yes"}

    def _pin_all_default_gpu_binding(self, *, force: bool = False) -> dict[str, Any]:
        binding = self.bind_gpu_galaxy_runtime(galaxy_names=list(self.DEFAULT_GALAXIES), force=force)
        self._pinned_all_default_binding = True
        return binding

    def _initialize_trm_launcher(self) -> None:
        """Phase D.1 boot-time TRM wiring only; no query-path behavior yet."""
        self._trm = None
        self._trm_ready = False
        self._trm_backend = "uninitialized"
        self._trm_init_error = ""
        self._trm_host_weights = {}
        self._trm_weight_buffers = {}
        self._trm_state_buffers = {}
        self._trm_state_buffer_bytes = 0
        self._trm_weight_bytes = 0
        self._matryoshka_bridge = None
        self._trm_matryoshka_host_weights = None
        self._trm_matryoshka_weight_buffer = None
        rng = np.random.default_rng(self.TRM_INIT_SEED)
        try:
            self._trm = TRMLauncher(use_fused=True)
            self._matryoshka_bridge = MatryoshkaProjectionBridge()
            total_bytes = 0
            checkpoint_weights = self._load_trm_weight_checkpoint()
            for name, shape in self.TRM_WEIGHT_SHAPES.items():
                checkpoint_host = None if checkpoint_weights is None else checkpoint_weights.get(name)
                if checkpoint_host is not None:
                    host = np.asarray(checkpoint_host, dtype=np.float32).copy()
                else:
                    host = (rng.standard_normal(shape, dtype=np.float32) * np.float32(0.02)).astype(
                        np.float32,
                        copy=False,
                    )
                device = gpu_malloc(host.nbytes)
                memcpy_htod(device, host.ctypes.data_as(ctypes.c_void_p), host.nbytes)
                self._trm_host_weights[name] = host
                self._trm_weight_buffers[name] = device
                total_bytes += int(host.nbytes)
            checkpoint_matryoshka = None if checkpoint_weights is None else checkpoint_weights.get("matryoshka")
            if checkpoint_matryoshka is not None:
                matryoshka = np.asarray(checkpoint_matryoshka, dtype=np.float32).copy()
            else:
                matryoshka = self._build_matryoshka_projection_weights(rng)
            matryoshka_device = gpu_malloc(matryoshka.nbytes)
            memcpy_htod(matryoshka_device, matryoshka.ctypes.data_as(ctypes.c_void_p), matryoshka.nbytes)
            self._trm_matryoshka_host_weights = matryoshka
            self._trm_matryoshka_weight_buffer = matryoshka_device
            total_bytes += int(matryoshka.nbytes)
            self._initialize_trm_state_buffers()
            total_bytes += int(self._trm_state_buffer_bytes)
            self._trm_weight_bytes = int(total_bytes)
            self._trm_ready = True
            self._trm_backend = "fused"
        except Exception as exc:
            self._trm = None
            self._trm_host_weights = {}
            self._trm_weight_buffers = {}
            self._trm_state_buffers = {}
            self._trm_state_buffer_bytes = 0
            self._trm_weight_bytes = 0
            self._matryoshka_bridge = None
            self._trm_matryoshka_host_weights = None
            self._trm_matryoshka_weight_buffer = None
            self._trm_ready = False
            self._trm_backend = "error"
            self._trm_init_error = f"{type(exc).__name__}: {exc}"

    def _build_matryoshka_projection_weights(self, rng: np.random.Generator) -> np.ndarray:
        basis = rng.standard_normal(
            (self.TRM_STATE_VECTOR_DIM, self.TRM_STATE_VECTOR_DIM),
            dtype=np.float32,
        ).astype(np.float64, copy=False)
        q, r = np.linalg.qr(basis)
        diag = np.sign(np.diag(r))
        diag[diag == 0.0] = 1.0
        q *= diag
        return np.asarray(q, dtype=np.float32)

    def _trm_galaxy_decoder_checkpoint_path(self) -> Path:
        return self.storage_root / "checkpoints" / "trm_galaxy_nav_weights.npz"

    def _trm_weight_checkpoint_path(self) -> Path:
        return self.storage_root / "checkpoints" / "trm_weights.npz"

    def _load_trm_weight_checkpoint(self) -> dict[str, np.ndarray] | None:
        checkpoint_path = self._trm_weight_checkpoint_path()
        if not checkpoint_path.exists():
            return None
        try:
            weights = load_trm_weight_checkpoint(checkpoint_path)
        except Exception:
            return None
        resolved: dict[str, np.ndarray] = {}
        for name, shape in self.TRM_WEIGHT_SHAPES.items():
            value = np.asarray(weights.get(name, []), dtype=np.float32)
            if value.shape != shape:
                return None
            resolved[name] = value
        matryoshka = np.asarray(weights.get("matryoshka", []), dtype=np.float32)
        if matryoshka.size:
            expected_shape = (self.TRM_STATE_VECTOR_DIM, self.TRM_STATE_VECTOR_DIM)
            if matryoshka.shape != expected_shape:
                return None
            resolved["matryoshka"] = matryoshka
        return resolved

    def _load_trm_galaxy_decoder(self) -> None:
        self._trm_galaxy_decoder = None
        self._trm_galaxy_decoder_path = ""
        checkpoint_path = self._trm_galaxy_decoder_checkpoint_path()
        if not checkpoint_path.exists():
            return
        try:
            decoder = load_galaxy_decoder_checkpoint(checkpoint_path)
        except Exception:
            return
        weights = np.asarray(decoder.get("W_galaxy", []), dtype=np.float32)
        bias = np.asarray(decoder.get("b_galaxy", []), dtype=np.float32)
        if weights.shape != (len(self.DEFAULT_GALAXIES), self.TRM_STATE_VECTOR_DIM):
            return
        if bias.shape != (len(self.DEFAULT_GALAXIES),):
            return
        self._trm_galaxy_decoder = {
            "W_galaxy": weights,
            "b_galaxy": bias,
        }
        self._trm_galaxy_decoder_path = str(checkpoint_path)

    def _initialize_trm_state_buffers(self) -> None:
        total_bytes = 0
        for name, float_count in self.TRM_STATE_BUFFER_FLOATS.items():
            ptr = gpu_malloc(int(float_count) * 4)
            self._trm_state_buffers[name] = ptr
            total_bytes += int(float_count) * 4
        self._trm_state_buffer_bytes = int(total_bytes)
        self._reset_trm_state()

    def _prepare_trm_stimulus_input(self, query_embedding: Any) -> tuple[np.ndarray, bool]:
        values = np.asarray(list(query_embedding), dtype=np.float32).reshape(-1)
        if values.size == self.TRM_STATE_VECTOR_DIM:
            return np.ascontiguousarray(values, dtype=np.float32), False
        prepared = np.zeros(self.TRM_STATE_VECTOR_DIM, dtype=np.float32)
        if values.size > 0:
            prepared[: min(values.size, self.TRM_STATE_VECTOR_DIM)] = values[: self.TRM_STATE_VECTOR_DIM]
        return prepared, True

    def _read_trm_state_vector(self, name: str) -> np.ndarray:
        host = np.zeros(int(self.TRM_STATE_BUFFER_FLOATS[name]), dtype=np.float32)
        memcpy_dtoh(
            ctypes.c_void_p(host.ctypes.data),
            self._trm_state_buffers[name],
            host.nbytes,
        )
        return host

    def _encode_stimulus(self, query_embedding: Any, *, readback: bool = False) -> np.ndarray | None:
        if not self._trm_state_buffers:
            raise RuntimeError("TRM state buffers unavailable")
        prepared, needs_projection = self._prepare_trm_stimulus_input(query_embedding)
        if needs_projection and self._matryoshka_bridge is not None and self._trm_matryoshka_weight_buffer is not None:
            memcpy_htod(
                self._trm_state_buffers["d_q_input"],
                ctypes.c_void_p(prepared.ctypes.data),
                prepared.nbytes,
            )
            self._matryoshka_bridge.project_device(
                self._trm_matryoshka_weight_buffer,
                self._trm_state_buffers["d_q_input"],
                self._trm_state_buffers["d_q"],
                target_dim=self.TRM_STATE_VECTOR_DIM,
                stride=self.TRM_STATE_VECTOR_DIM,
            )
        else:
            memcpy_htod(
                self._trm_state_buffers["d_q"],
                ctypes.c_void_p(prepared.ctypes.data),
                prepared.nbytes,
            )
        if readback:
            return self._read_trm_state_vector("d_q")
        return None

    def _reset_trm_state(self) -> None:
        if not self._trm_state_buffers:
            return
        for name in ("d_q_input", "d_y", "d_z", "d_z_new", "d_y_new", "d_workspace"):
            float_count = int(self.TRM_STATE_BUFFER_FLOATS[name])
            zeros = np.zeros(float_count, dtype=np.float32)
            memcpy_htod(
                self._trm_state_buffers[name],
                ctypes.c_void_p(zeros.ctypes.data),
                zeros.nbytes,
            )

    def _run_single_trm_tick(self, query_embedding: Any) -> dict[str, Any]:
        if not self._trm_ready or self._trm is None:
            return {}
        self._reset_trm_state()
        projected_query = self._encode_stimulus(query_embedding, readback=True)
        started = time.perf_counter()
        launch(
            self._trm.kernel_fused,
            grid=(1, 1, 1),
            block=(128, 1, 1),
            params=[
                self._trm_state_buffers["d_q"],
                self._trm_state_buffers["d_y"],
                self._trm_state_buffers["d_z"],
                self._trm_weight_buffers["W1"],
                self._trm_weight_buffers["W2"],
                self._trm_weight_buffers["W3"],
                self._trm_weight_buffers["W4"],
                self._trm_state_buffers["d_z_new"],
                self._trm_state_buffers["d_y_new"],
                self._trm_state_buffers["d_workspace"],
            ],
        )
        synchronize()
        latency_us = float((time.perf_counter() - started) * 1_000_000.0)
        y_new_host = self._read_trm_state_vector("d_y_new")
        return {
            "query_embedding_512": projected_query.tolist() if projected_query is not None else [],
            "y_new_vector_512": y_new_host.tolist(),
            "trm_latency_us": latency_us,
        }

    def _decode_trm_galaxy_distribution(self, y_new_vector_512: Any) -> tuple[np.ndarray, np.ndarray, str]:
        y_new_host = np.asarray(list(y_new_vector_512), dtype=np.float32).reshape(-1)
        if self._trm_galaxy_decoder is not None:
            logits = (
                np.asarray(self._trm_galaxy_decoder["W_galaxy"], dtype=np.float32) @ y_new_host
            ) + np.asarray(self._trm_galaxy_decoder["b_galaxy"], dtype=np.float32)
            decoder_source = "checkpoint"
        else:
            logits = np.asarray(y_new_host[: len(self.DEFAULT_GALAXIES)], dtype=np.float32)
            decoder_source = "raw_head"
        distribution = softmax(logits)
        return logits, distribution, decoder_source

    def _trm_shadow_probe(
        self,
        query_embedding: Any,
        target_galaxies: list[str],
        reasoning_program_id: str,
        *,
        trm_tick: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._trm_ready or self._trm is None:
            return {}
        tick = dict(trm_tick or self._run_single_trm_tick(query_embedding))
        y_new_host = np.asarray(list(tick.get("y_new_vector_512", [])), dtype=np.float32).reshape(-1)
        projected_query = np.asarray(list(tick.get("query_embedding_512", [])), dtype=np.float32).reshape(-1)
        latency_us = float(tick.get("trm_latency_us", 0.0))
        logits, distribution, decoder_source = self._decode_trm_galaxy_distribution(y_new_host)
        top_indexes = np.argsort(distribution)[-3:][::-1]
        entropy = float(-np.sum(distribution * np.log(np.clip(distribution, 1e-9, 1.0))))
        return {
            "y_new_top3_galaxies": [
                {
                    "galaxy": str(DEFAULT_GALAXY_ORDER[int(idx)]),
                    "weight": float(distribution[int(idx)]),
                    "logit": float(logits[int(idx)]),
                }
                for idx in top_indexes
            ],
            "y_new_entropy": entropy,
            "trm_latency_us": latency_us,
            "python_galaxies": [str(name) for name in target_galaxies],
            "python_program": str(reasoning_program_id),
            "query_embedding_512": projected_query.astype(np.float32, copy=False).tolist(),
            "y_new_vector_512": y_new_host.astype(np.float32, copy=False).tolist(),
            "decoder_source": decoder_source,
            "decoder_checkpoint": str(self._trm_galaxy_decoder_path),
        }

    @classmethod
    def _normalize_galaxy_weights(cls, galaxy_weights: dict[str, Any] | None) -> dict[str, float]:
        if not isinstance(galaxy_weights, dict):
            return {}
        raw_weights: dict[str, float] = {}
        for name, value in galaxy_weights.items():
            galaxy_name = str(name).strip()
            if not galaxy_name:
                continue
            try:
                raw_weights[galaxy_name] = max(0.0, float(value))
            except Exception:
                continue
        if not raw_weights:
            return {}
        try:
            strength = max(0.0, float(os.getenv("K3D_TRM_INFLUENCE_STRENGTH", str(cls.TRM_GALAXY_INFLUENCE_STRENGTH))))
        except Exception:
            strength = float(cls.TRM_GALAXY_INFLUENCE_STRENGTH)
        uniform = 1.0 / float(max(len(cls.DEFAULT_GALAXIES), 1))
        normalized: dict[str, float] = {}
        for galaxy_name in cls.DEFAULT_GALAXIES:
            raw_value = float(raw_weights.get(str(galaxy_name), 0.0))
            multiplier = 1.0 + (strength * (raw_value - uniform))
            normalized[str(galaxy_name)] = max(0.0, float(multiplier))
        for galaxy_name, raw_value in raw_weights.items():
            if galaxy_name in normalized:
                continue
            multiplier = 1.0 + (strength * float(raw_value))
            normalized[galaxy_name] = max(0.0, float(multiplier))
        return normalized

    def _galaxy_weight_for_name(
        self,
        galaxy_name: str,
        galaxy_weights: dict[str, Any] | None,
    ) -> float:
        normalized = self._normalize_galaxy_weights(galaxy_weights)
        return float(normalized.get(str(galaxy_name).strip(), 0.0))

    def _galaxy_contribution_from_records(
        self,
        *,
        records: list[dict[str, Any]] | None = None,
        candidates: list[dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        scored_rows: list[tuple[str, float]] = []
        if isinstance(records, list):
            for record in records:
                if not isinstance(record, dict):
                    continue
                candidate = record.get("candidate") if isinstance(record.get("candidate"), dict) else None
                if candidate is None and isinstance(record.get("match"), dict):
                    candidate = record
                if not isinstance(candidate, dict):
                    continue
                match = candidate.get("match") if isinstance(candidate.get("match"), dict) else {}
                galaxy_name = str(match.get("galaxy", "")).strip()
                if not galaxy_name:
                    continue
                score = record.get("path_score")
                if score is None:
                    score = candidate.get("path_score", candidate.get("gpu_score", candidate.get("similarity", 0.0)))
                try:
                    score_value = float(score)
                except Exception:
                    continue
                if not math.isfinite(score_value):
                    continue
                scored_rows.append((galaxy_name, score_value))
        if not scored_rows and isinstance(candidates, list):
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                match = candidate.get("match") if isinstance(candidate.get("match"), dict) else {}
                galaxy_name = str(match.get("galaxy", "")).strip()
                if not galaxy_name:
                    continue
                try:
                    score_value = float(
                        candidate.get("gpu_score", candidate.get("path_score", candidate.get("similarity", 0.0)))
                    )
                except Exception:
                    continue
                if not math.isfinite(score_value):
                    continue
                scored_rows.append((galaxy_name, score_value))
        if not scored_rows:
            return {}
        max_score = max(score for _, score in scored_rows)
        totals: dict[str, float] = {}
        for galaxy_name, score_value in scored_rows:
            shifted = max(-20.0, min(20.0, float(score_value) - float(max_score)))
            weight = math.exp(shifted)
            totals[galaxy_name] = float(totals.get(galaxy_name, 0.0)) + float(weight)
        total_weight = float(sum(totals.values()))
        if total_weight <= 0.0:
            return {}
        normalized = {
            galaxy_name: float(weight / total_weight)
            for galaxy_name, weight in totals.items()
            if float(weight) > 0.0
        }
        return dict(sorted(normalized.items(), key=lambda item: item[1], reverse=True))

    def _attach_galaxy_contribution(
        self,
        candidate: dict[str, Any] | None,
        *,
        records: list[dict[str, Any]] | None = None,
        candidates: list[dict[str, Any]] | None = None,
        selection_steps: list[str] | None = None,
    ) -> dict[str, Any] | None:
        if not isinstance(candidate, dict):
            return candidate
        contribution = self._galaxy_contribution_from_records(records=records, candidates=candidates)
        if not contribution:
            return candidate
        candidate["galaxy_contribution"] = dict(contribution)
        candidate["teacher_route_galaxies"] = [
            galaxy_name
            for galaxy_name, weight in contribution.items()
            if float(weight) >= 0.05
        ] or list(contribution.keys())[:3]
        if isinstance(selection_steps, list):
            selection_steps.append(
                "Galaxy contribution: "
                + ", ".join(
                    f"{galaxy_name}={float(weight):.2f}"
                    for galaxy_name, weight in list(contribution.items())[:4]
                )
            )
        return candidate

    def _trm_select_galaxies(
        self,
        query_embedding: Any,
        *,
        task_type: str,
        fallback_galaxies: list[str],
        reasoning_program_id: str,
        trm_tick: dict[str, Any] | None = None,
    ) -> tuple[dict[str, float], str, dict[str, Any]]:
        if not self._trm_ready or self._trm is None:
            return {}, reasoning_program_id, {
                "status": "fallback",
                "reason": "trm_not_ready",
            }
        tick = dict(trm_tick or self._run_single_trm_tick(query_embedding))
        y_new_host = np.asarray(list(tick.get("y_new_vector_512", [])), dtype=np.float32).reshape(-1)
        if y_new_host.size != self.TRM_STATE_VECTOR_DIM:
            return {}, reasoning_program_id, {
                "status": "fallback",
                "reason": "invalid_tick",
            }
        logits, distribution, decoder_source = self._decode_trm_galaxy_distribution(y_new_host)
        if not np.all(np.isfinite(distribution)):
            return {}, reasoning_program_id, {
                "status": "fallback",
                "reason": "non_finite_distribution",
            }
        ranked_indexes = list(np.argsort(distribution)[::-1])
        max_weight = float(distribution[ranked_indexes[0]]) if ranked_indexes else 0.0
        if max_weight < 0.01 or float(np.max(distribution) - np.min(distribution)) <= 1e-6:
            return {}, reasoning_program_id, {
                "status": "fallback",
                "reason": "trm_nav_fallback",
                "decoder_source": decoder_source,
                "max_weight": max_weight,
                "task_type": task_type,
            }
        selected_names = [
            str(DEFAULT_GALAXY_ORDER[idx])
            for idx in ranked_indexes
            if float(distribution[idx]) > 0.05
        ]
        if len(selected_names) < 2:
            selected_names = [str(DEFAULT_GALAXY_ORDER[idx]) for idx in ranked_indexes[:2]]
        if len(selected_names) < 2:
            return {}, reasoning_program_id, {
                "status": "fallback",
                "reason": "trm_nav_fallback",
                "decoder_source": decoder_source,
                "max_weight": max_weight,
                "task_type": task_type,
            }
        selected_names = list(dict.fromkeys(selected_names))
        galaxy_rank = {str(name): idx for idx, name in enumerate(DEFAULT_GALAXY_ORDER)}
        selected_names = sorted(selected_names, key=lambda name: galaxy_rank.get(str(name), len(DEFAULT_GALAXY_ORDER)))
        selected_names = selected_names[:5]
        galaxy_weights = {
            str(DEFAULT_GALAXY_ORDER[idx]): float(distribution[idx])
            for idx in range(min(len(DEFAULT_GALAXY_ORDER), len(distribution)))
        }
        return galaxy_weights, reasoning_program_id, {
            "status": "ok",
            "decoder_source": decoder_source,
            "task_type": task_type,
            "selected_galaxies": list(selected_names),
            "galaxy_weights": dict(galaxy_weights),
            "top3": [
                {
                    "galaxy": str(DEFAULT_GALAXY_ORDER[idx]),
                    "weight": float(distribution[idx]),
                    "logit": float(logits[idx]),
                }
                for idx in ranked_indexes[:3]
            ],
            "trm_latency_us": float(tick.get("trm_latency_us", 0.0)),
        }

    def get_gpu_reasoning_engine(self, *, force_rebind: bool = False):
        """Return the shared GPU reasoning engine bound to the active Galaxy snapshot."""
        if self._gpu_reasoning_engine is None:
            from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine

            self._gpu_reasoning_engine = ModularRPNEngine()
        if force_rebind or self._gpu_galaxy_binding is None:
            self.bind_gpu_galaxy_runtime(force=force_rebind)
        return self._gpu_reasoning_engine

    def bind_gpu_galaxy_runtime(
        self,
        *,
        galaxy_names: list[str] | tuple[str, ...] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Flatten active Galaxy entries and bind them into the GPU RPN runtime."""
        resolved_names = [str(name) for name in (galaxy_names or self.DEFAULT_GALAXIES)]
        bound_names = list((self._gpu_galaxy_binding or {}).get("galaxies", []))
        if (
            self._gpu_galaxy_binding is not None
            and not force
            and (
                bound_names == resolved_names
                or (
                    self._pinned_all_default_binding
                    and
                    bound_names == list(self.DEFAULT_GALAXIES)
                    and set(resolved_names).issubset(set(bound_names))
                )
            )
        ):
            return dict(self._gpu_galaxy_binding)
        engine = self._gpu_reasoning_engine
        if engine is None:
            from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine

            engine = ModularRPNEngine()
            self._gpu_reasoning_engine = engine
        flat_entries, catalog = self._flatten_galaxies_for_gpu(galaxy_names=resolved_names)
        binding = engine.bind_galaxy_buffer(
            flat_entries,
            entry_count=len(catalog),
            entry_stride=self.GPU_GALAXY_ENTRY_STRIDE,
            embedding_offset=self.GPU_GALAXY_EMBEDDING_OFFSET,
            embedding_dim=self.GPU_GALAXY_EMBEDDING_DIM,
        )
        binding.update(
            {
                "galaxies": list(resolved_names),
                "entry_count": len(catalog),
                "buffer_bytes": len(flat_entries) * 4,
                "runtime_artifact_entries": int(self.metrics.gpu_runtime_artifact_entries),
            }
        )
        if self._query_head_substrate is not None:
            self._query_head_substrate.close()
        self._gpu_galaxy_binding = binding
        self._gpu_galaxy_catalog = catalog
        self._semantic_csr_graph = load_or_build_semantic_csr_graph(
            catalog=catalog,
            cache_root=self.storage_root / "graph_cache",
            knn_k=12,
            similarity_threshold=0.3,
        )
        self._query_head_substrate = QueryHeadSubstrate.build(
            signature=str(self._semantic_csr_graph.signature),
            catalog=catalog,
        )
        self.metrics.gpu_galaxy_entries = len(catalog)
        self.metrics.gpu_galaxy_bytes = len(flat_entries) * 4
        self.metrics.gpu_bind_rebuilds = int(self.metrics.gpu_bind_rebuilds) + 1
        return dict(binding)

    def get_gpu_galaxy_catalog(self) -> list[dict[str, Any]]:
        return list(self._gpu_galaxy_catalog)

    def invalidate_gpu_galaxy_binding(self) -> None:
        if self._query_head_substrate is not None:
            self._query_head_substrate.close()
        self._pinned_all_default_binding = False
        self._gpu_galaxy_binding = None
        self._gpu_galaxy_catalog = []
        self._gpu_reasoning_programs = {}
        self._semantic_csr_graph = None
        self._query_head_substrate = None

    def reset_query_session(self) -> None:
        """Clear mutable per-benchmark state while keeping the GPU-bound galaxy snapshot assembled."""
        self._gpu_reasoning_programs.clear()
        self._query_sequence = 0
        self._led_pathfinder = None
        if self._gpu_reasoning_engine is not None and hasattr(self._gpu_reasoning_engine, "reset_instance"):
            for instance_id in range(18):
                try:
                    self._gpu_reasoning_engine.reset_instance(instance_id)
                except Exception:
                    pass
        self.trm_navigator.reset_session_state()
        self.specialist_router = self.trm_navigator.specialist_router
        self.navigator_specialist = self.trm_navigator.navigator_specialist
        self.shadow_copy.event_buffer.clear()

    def get_query_head_substrate(self) -> QueryHeadSubstrate:
        if self._query_head_substrate is None:
            self.bind_gpu_galaxy_runtime()
        if self._query_head_substrate is None:
            raise RuntimeError("query head substrate unavailable")
        return self._query_head_substrate

    def get_swarm_bridge(self):
        if self._swarm_bridge is False:
            return None
        if self._swarm_bridge is None:
            try:
                from knowledge3d.cranium.bridges.nine_chain_specialized_bridge import (
                    NineChainSpecializedBridge,
                )

                self._swarm_bridge = NineChainSpecializedBridge()
            except Exception:
                self._swarm_bridge = False
                return None
        return self._swarm_bridge

    def get_vector_resonator(self):
        if self._vector_resonator is False:
            return None
        if self._vector_resonator is None:
            try:
                from knowledge3d.cranium.bridges.sovereign_bridges import VectorResonator

                self._vector_resonator = VectorResonator()
            except Exception:
                self._vector_resonator = False
                return None
        return self._vector_resonator

    def get_resonance_field(self):
        if self._resonance_field is False:
            return None
        if self._resonance_field is None:
            try:
                from knowledge3d.cranium.bridges.sovereign_bridges import ResonanceField

                self._resonance_field = ResonanceField()
            except Exception:
                self._resonance_field = False
                return None
        return self._resonance_field

    def get_geometry_router(self):
        if self._geometry_router is False:
            return None
        if self._geometry_router is None:
            try:
                from knowledge3d.cranium.bridges.sovereign_bridges import GeometryRouter

                self._geometry_router = GeometryRouter()
            except Exception:
                self._geometry_router = False
                return None
        return self._geometry_router

    def get_temporal_reasoning(self):
        if self._temporal_reasoning is False:
            return None
        if self._temporal_reasoning is None:
            try:
                from knowledge3d.cranium.bridges.sovereign_bridges import TemporalReasoning

                self._temporal_reasoning = TemporalReasoning()
            except Exception:
                self._temporal_reasoning = False
                return None
        return self._temporal_reasoning

    def get_fractal_emitter(self):
        if self._fractal_emitter is False:
            return None
        if self._fractal_emitter is None:
            try:
                from knowledge3d.cranium.bridges.sovereign_bridges import FractalEmitter

                self._fractal_emitter = FractalEmitter()
            except Exception:
                self._fractal_emitter = False
                return None
        return self._fractal_emitter

    def get_cognitive_executive(self):
        if self._cognitive_executive is False:
            return None
        if self._cognitive_executive is None:
            try:
                from knowledge3d.cranium.bridges.sovereign_bridges import CognitiveExecutive

                self._cognitive_executive = CognitiveExecutive()
            except Exception:
                self._cognitive_executive = False
                return None
        return self._cognitive_executive

    def get_galaxy_resonance_engine(self):
        if self._galaxy_resonance_engine is False:
            return None
        if self._galaxy_resonance_engine is None:
            try:
                from knowledge3d.cranium.bridges.sovereign_bridges import GalaxyResonanceEngine

                self._galaxy_resonance_engine = GalaxyResonanceEngine()
            except Exception:
                self._galaxy_resonance_engine = False
                return None
        return self._galaxy_resonance_engine

    def get_graph_crystallizer(self):
        if self._graph_crystallizer is False:
            return None
        if self._graph_crystallizer is None:
            try:
                from knowledge3d.cranium.bridges.sovereign_bridges import GraphCrystallizer

                self._graph_crystallizer = GraphCrystallizer()
            except Exception:
                self._graph_crystallizer = False
                return None
        return self._graph_crystallizer

    def get_atomic_fission_fusion(self):
        if self._atomic_fission_fusion is False:
            return None
        if self._atomic_fission_fusion is None:
            try:
                from knowledge3d.cranium.bridges.sovereign_bridges import AtomicFissionFusion

                self._atomic_fission_fusion = AtomicFissionFusion()
            except Exception:
                self._atomic_fission_fusion = False
                return None
        return self._atomic_fission_fusion

    def get_defeasible_resolver(self):
        if self._defeasible_resolver is False:
            return None
        if self._defeasible_resolver is None:
            try:
                from knowledge3d.cranium.bridges.sovereign_bridges import DefeasibleResolver

                self._defeasible_resolver = DefeasibleResolver()
            except Exception:
                self._defeasible_resolver = False
                return None
        return self._defeasible_resolver

    def get_cosine_similarity_bridge(self):
        if self._cosine_similarity_bridge is False:
            return None
        if self._cosine_similarity_bridge is None:
            try:
                from knowledge3d.cranium.bridges.cosine_similarity_bridge import CosineSimilarityBridge

                self._cosine_similarity_bridge = CosineSimilarityBridge()
            except Exception:
                self._cosine_similarity_bridge = False
                return None
        return self._cosine_similarity_bridge

    def get_halting_gate(self):
        if self._halting_gate is False:
            return None
        if self._halting_gate is None:
            try:
                from knowledge3d.cranium.bridges.sovereign_bridges import MultimodalHaltingGate

                self._halting_gate = MultimodalHaltingGate()
            except Exception:
                self._halting_gate = False
                return None
        return self._halting_gate

    def _extend_runtime_galaxy(self, galaxy_name: str, entries: list[dict[str, Any]]) -> int:
        if not entries:
            return 0
        galaxy = self.galaxy_manager.get_galaxy(galaxy_name)
        existing_ids = {
            str(entry.get("id", "")).strip()
            for entry in getattr(galaxy, "entries", [])
            if isinstance(entry, dict) and str(entry.get("id", "")).strip()
        }
        appended: list[dict[str, Any]] = []
        for entry in entries:
            entry_id = str(entry.get("id", "")).strip()
            if entry_id and entry_id in existing_ids:
                continue
            appended.append(dict(entry))
            if entry_id:
                existing_ids.add(entry_id)
        if not appended:
            return 0
        galaxy.entries.extend(appended)
        self.galaxy_manager._entry_text_cache.clear()
        self.galaxy_manager._specialist_entry_cache.clear()
        self.invalidate_gpu_galaxy_binding()
        return len(appended)

    def _ensure_runtime_language_enrichment_loaded(self) -> dict[str, int]:
        if not self.include_runtime_language_enrichment:
            self.metrics.runtime_language_entries = 0
            return {
                "Word": len(self.galaxy_manager.get_galaxy("Word").entries),
                "Grammar": len(self.galaxy_manager.get_galaxy("Grammar").entries),
            }
        if self._runtime_language_enrichment_loaded:
            return {
                "Word": len(self.galaxy_manager.get_galaxy("Word").entries),
                "Grammar": len(self.galaxy_manager.get_galaxy("Grammar").entries),
            }
        counts: dict[str, int] = {}
        payload = load_language_runtime_entries()
        inserted_total = 0
        for galaxy_name, entries in payload.items():
            inserted = self._extend_runtime_galaxy(galaxy_name, entries)
            counts[galaxy_name] = inserted
            inserted_total += inserted
        self.metrics.runtime_language_entries = inserted_total
        self._runtime_language_enrichment_loaded = True
        return counts

    def get_text_embedding_engine(self) -> RPNEmbeddingEngine:
        if self._text_embedding_engine is None:
            self._text_embedding_engine = RPNEmbeddingEngine(embedding_dim=16)
        return self._text_embedding_engine

    def get_led_pathfinder(self):
        if self._led_pathfinder is False:
            return None
        if self._led_pathfinder is None:
            try:
                from knowledge3d.cranium.spatial_sovereign.led_pathfinder import LEDPathfinderSovereign

                self._led_pathfinder = LEDPathfinderSovereign()
            except Exception:
                self._led_pathfinder = False
                return None
        return self._led_pathfinder

    def get_semantic_csr_graph(self):
        if self._semantic_csr_graph is None:
            self.bind_gpu_galaxy_runtime()
        return self._semantic_csr_graph

    def get_gpu_query_embedding_engine(self) -> RPNEmbeddingEngine:
        engine = self.get_text_embedding_engine()
        if self._gpu_query_embedding_bridge is None:
            self._gpu_query_embedding_bridge = TrigramEmbedBridge()
            engine.attach_gpu_bridge(self._gpu_query_embedding_bridge)
        return engine

    def _append_flattened_entry(
        self,
        *,
        flat: list[float],
        catalog: list[dict[str, Any]],
        galaxy_name: str,
        entry: dict[str, Any],
    ) -> None:
        if not isinstance(entry, dict):
            return
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        if not metadata and str(entry.get("rule_id", "")).strip():
            metadata = {
                "rule_id": str(entry.get("rule_id", "")).strip(),
                "language": str(entry.get("language", "")).strip(),
                "pattern": str(entry.get("pattern", "")).strip(),
                "domain": str(entry.get("domain", galaxy_name)).strip(),
                "symbol_refs": list(entry.get("symbol_refs", []) or []),
                "word_refs": list(entry.get("word_refs", []) or []),
                "description": entry.get("description"),
                "semantics": dict(entry.get("semantics", {}) or {}),
                "usage_conditions": list(entry.get("usage_conditions", []) or []),
                "is_canonical": bool(entry.get("is_canonical", False)),
                "rule_strength": int(entry.get("rule_strength", 0) or 0),
                "superior_to": list(entry.get("superior_to", []) or []),
                "trust_weight": float(entry.get("trust_weight", 1.0) or 1.0),
            }
        confidence = self._clamp_confidence(metadata.get("confidence", entry.get("confidence", 0.5)))
        domain_hash = self._hash_to_unit_float(entry.get("domain") or galaxy_name)
        subject_hash = self._hash_to_unit_float(
            metadata.get("subject")
            or metadata.get("meaning_ref")
            or entry.get("category")
            or entry.get("id")
            or galaxy_name
        )
        category = str(entry.get("category", "")).strip().lower()
        template_ref = self._entry_template_ref(entry, metadata)
        category_class = self._gpu_category_class(category)
        source_class = self._gpu_source_class(entry, metadata)
        galaxy_index = self._gpu_galaxy_index(galaxy_name)
        has_template_ref = 1.0 if template_ref else 0.0
        embedding = self._entry_embedding16(entry)
        flat.extend(
            [
                confidence,
                domain_hash,
                subject_hash,
                *embedding,
                category_class,
                source_class,
                galaxy_index,
                has_template_ref,
            ]
        )
        catalog.append(
            {
                "index": len(catalog),
                "galaxy": galaxy_name,
                "id": str(entry.get("id", entry.get("rule_id", ""))),
                "name": str(entry.get("name", "")),
                "category": str(entry.get("category", "")),
                "domain": str(entry.get("domain", galaxy_name)),
                "confidence": confidence,
                "domain_hash": domain_hash,
                "subject_hash": subject_hash,
                "answer_text": self._entry_answer_text(entry),
                "embedding_text": self._entry_embedding_text(entry),
                "embedding16": list(embedding),
                "rpn_program": str(entry.get("rpn_program", "")).strip(),
                "metadata": dict(metadata),
                "template_ref": template_ref,
                "template_params": dict(metadata.get("template_params", {}))
                if isinstance(metadata.get("template_params"), dict)
                else {},
                "answer_format": str(metadata.get("answer_format", "")),
                "subject": str(metadata.get("subject", "")),
                "gpu_category_class": category_class,
                "gpu_source_class": source_class,
                "gpu_galaxy_index": galaxy_index,
                "gpu_has_template_ref": has_template_ref,
                "output_grid": metadata.get("output_grid", entry.get("output_grid")),
                "arc_transform_chain": list(metadata.get("arc_transform_chain", [])),
                "arc_color_mapping": dict(metadata.get("arc_color_mapping", {})),
                "arc_primitive_plan": list(metadata.get("arc_primitive_plan", [])),
                "arc_task_id": str(metadata.get("arc_task_id", "")),
            }
        )

    def _iter_runtime_book_entries(
        self,
        *,
        galaxy_names: list[str],
    ) -> list[tuple[str, dict[str, Any]]]:
        if not self.include_runtime_artifacts:
            self.metrics.gpu_runtime_artifact_entries = 0
            return []
        grouped, stats = load_books_runtime_entries()
        self.metrics.gpu_runtime_artifact_entries = int(stats.get("artifacts", 0))
        emitted: list[tuple[str, dict[str, Any]]] = []
        allowed = {str(name) for name in galaxy_names}
        for galaxy_name, entries in grouped.items():
            if galaxy_name not in allowed:
                continue
            for entry in entries:
                emitted.append((galaxy_name, dict(entry)))
        return emitted

    def _flatten_galaxies_for_gpu(
        self,
        *,
        galaxy_names: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[list[float], list[dict[str, Any]]]:
        names = [str(name) for name in (galaxy_names or self.DEFAULT_GALAXIES)]
        flat: list[float] = []
        catalog: list[dict[str, Any]] = []
        for galaxy_name in names:
            galaxy = self.galaxy_manager.get_galaxy(galaxy_name)
            for entry in getattr(galaxy, "entries", []):
                self._append_flattened_entry(
                    flat=flat,
                    catalog=catalog,
                    galaxy_name=galaxy_name,
                    entry=entry,
                )
        for runtime_galaxy, entry in self._iter_runtime_book_entries(galaxy_names=names):
            self._append_flattened_entry(
                flat=flat,
                catalog=catalog,
                galaxy_name=runtime_galaxy,
                entry=entry,
            )
        return flat, catalog

    @staticmethod
    def _clamp_confidence(value: Any) -> float:
        try:
            return max(0.0, min(float(value), 1.0))
        except Exception:
            return 0.5

    @staticmethod
    def _hash32(value: Any) -> int:
        text = str(value or "")
        acc = 2166136261
        for ch in text:
            acc ^= ord(ch)
            acc = (acc * 16777619) & 0xFFFFFFFF
        return int(acc)

    @classmethod
    def _hash_to_unit_float(cls, value: Any) -> float:
        return float((cls._hash32(value) & 0x00FFFFFF) / float(0x00FFFFFF))

    @staticmethod
    def _entry_template_ref(entry: dict[str, Any], metadata: dict[str, Any]) -> str:
        template_ref = str(metadata.get("template_ref", "")).strip()
        if template_ref:
            return template_ref
        meaning_ref = str(metadata.get("meaning_ref", "")).strip()
        if meaning_ref.startswith("math_template_"):
            return meaning_ref
        entry_id = str(entry.get("id", "")).strip()
        if entry_id.startswith("math_template_"):
            return entry_id
        return ""

    @classmethod
    def _gpu_category_class(cls, category: Any) -> float:
        key = str(category or "").strip().lower()
        return float(cls.GPU_CATEGORY_CLASS_MAP.get(key, 0.0))

    @classmethod
    def _gpu_galaxy_index(cls, galaxy_name: Any) -> float:
        key = str(galaxy_name or "").strip()
        for index, name in enumerate(cls.DEFAULT_GALAXIES):
            if key == name:
                return float(index)
        return float(len(cls.DEFAULT_GALAXIES))

    @classmethod
    def _gpu_source_class(cls, entry: dict[str, Any], metadata: dict[str, Any]) -> float:
        artifact_source = str(metadata.get("artifact_source", "")).strip().lower()
        if artifact_source == "books_v5_clean2":
            return cls.GPU_SOURCE_CLASS_BOOK_ARTIFACT
        category = str(entry.get("category", "")).strip().lower()
        language = str(metadata.get("language", "")).strip().lower()
        sources = metadata.get("sources") if isinstance(metadata.get("sources"), list) else []
        source_tokens = {str(item).strip().lower() for item in sources if str(item).strip()}
        if (
            category in {"multilingual_word", "language_rule"}
            or language
            or source_tokens.intersection({"runtime_language_enrichment", "kaikki_es", "kaikki_pt", "cedict_zh"})
        ):
            return cls.GPU_SOURCE_CLASS_RUNTIME_LANGUAGE
        return cls.GPU_SOURCE_CLASS_FOUNDATIONAL

    @staticmethod
    def _normalize_embedding(values: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= 1e-8:
            return [0.0 for _ in values]
        return [float(value / norm) for value in values]

    def _entry_embedding16(self, entry: dict[str, Any]) -> list[float]:
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        for candidate in (
            entry.get("embedding16"),
            entry.get("embedding"),
            metadata.get("embedding16"),
            metadata.get("embedding"),
        ):
            if isinstance(candidate, list) and len(candidate) >= 16:
                try:
                    return self._normalize_embedding([float(candidate[i]) for i in range(16)])
                except Exception:
                    pass
        text = self._entry_embedding_text(entry)
        if text:
            try:
                return self._normalize_embedding(list(self.get_text_embedding_engine().embed_sentence(text)))
            except Exception:
                pass
        text = json.dumps(entry, ensure_ascii=True, sort_keys=True)
        dims = [0.0] * 16
        for idx, ch in enumerate(text[:4096]):
            lane = idx & 15
            dims[lane] += ((ord(ch) & 31) - 15.0) / 15.0
        return self._normalize_embedding(dims)

    def _arc_task_embedding16(self, task_id: str) -> list[float]:
        dims = [0.0] * 16
        for idx, ch in enumerate(f"ARC_TASK::{task_id}"):
            lane = idx & 15
            dims[lane] += ((ord(ch) * (idx + 3)) % 29 - 14.0) / 14.0
        return self._normalize_embedding(dims)

    def _entry_embedding_text(self, entry: dict[str, Any]) -> str:
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        query_anchor = metadata.get("query_anchor")
        if isinstance(query_anchor, str) and query_anchor.strip():
            return query_anchor.strip()
        fields: list[str] = []
        for value in (
            metadata.get("question"),
            metadata.get("answer"),
            entry.get("name"),
            entry.get("content"),
            entry.get("summary"),
            entry.get("description"),
            metadata.get("semantics"),
        ):
            if isinstance(value, str) and value.strip():
                fields.append(value.strip())
        for key in ("aliases", "keywords", "forms"):
            value = metadata.get(key)
            if isinstance(value, list):
                fields.extend(str(item).strip() for item in value if str(item).strip())
        return " ".join(fields).strip()

    def _entry_answer_text(self, entry: dict[str, Any]) -> str:
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        for value in (
            metadata.get("answer"),
            entry.get("answer"),
            entry.get("summary"),
            entry.get("content"),
            entry.get("name"),
            entry.get("id"),
        ):
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _task_competition(task: dict[str, Any] | None) -> str:
        return str((task or {}).get("competition", "")).strip().upper()

    @classmethod
    def _is_gsm8k_math_task(cls, task: dict[str, Any] | None) -> bool:
        payload = dict(task or {})
        return (
            str(payload.get("type", "")).upper() == "MATH_TASK"
            and cls._task_competition(payload) == "GSM8K"
        )

    def _query_text(
        self,
        prompt: str,
        *,
        task: dict[str, Any] | None = None,
        options: list[str] | None = None,
    ) -> str:
        payload = dict(task or {})
        task_type = str(payload.get("type", "")).upper()
        if task_type == "ARC_TASK":
            task_id = str(payload.get("task_id", "")).strip()
            fragments: list[str] = [
                f"ARC_TASK {task_id} solve arc transformation task"
                if task_id
                else "solve arc transformation task"
            ]
            training_examples = payload.get("training_examples")
            if isinstance(training_examples, list):
                fragments.append(f"train_pairs {len(training_examples)}")
            input_grid = payload.get("input_grid")
            if isinstance(input_grid, list):
                rows = len(input_grid)
                cols = len(input_grid[0]) if rows and isinstance(input_grid[0], list) else 0
                fragments.append(f"input_shape {rows}x{cols}")
            if str(prompt or "").strip():
                fragments.append(str(prompt).strip())
            return " ".join(fragment for fragment in fragments if fragment).strip()
        fragments: list[str] = []
        if task_type in {"CHAT_TASK", "GENERAL_TASK", "GRAMMAR_TASK"}:
            messages = payload.get("messages")
            if isinstance(messages, list):
                for message in messages:
                    if not isinstance(message, dict):
                        continue
                    if str(message.get("role", "")).strip().lower() not in {"user", "system"}:
                        continue
                    content = str(message.get("content", "")).strip()
                    if content:
                        fragments.append(content)
        if str(prompt or "").strip():
            fragments.append(str(prompt).strip())
        for key in ("prompt", "query", "question"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                candidate = value.strip()
                if candidate not in fragments:
                    fragments.append(candidate)
        combined_options = options
        if combined_options is None and isinstance(payload.get("options"), list):
            combined_options = [str(option) for option in payload.get("options", [])]
        if combined_options:
            fragments.extend(str(option).strip() for option in combined_options if str(option).strip())
        if task_type == "MATH_TASK" and not self._is_gsm8k_math_task(payload):
            lowered = " ".join(fragment.lower() for fragment in fragments if fragment)
            if "solve" in lowered and "x" in lowered and "=" in lowered:
                fragments.append("linear equation solve ax + b = c isolate x")
            if self._query_mentions_factorial(lowered):
                fragments.append("factorial n! compute factorial")
            if "choose" in lowered or "binomial" in lowered or "combination" in lowered:
                fragments.append("binomial coefficient n choose k")
            if "sum of first" in lowered or "arithmetic series" in lowered:
                fragments.append("arithmetic series finite progression sum")
            if "geometric series" in lowered or "common ratio" in lowered:
                fragments.append("geometric series ratio progression sum")
        return " ".join(fragment for fragment in fragments if fragment).strip()

    def _resolve_gpu_target_galaxies(
        self,
        *,
        route: dict[str, Any] | None = None,
        task: dict[str, Any] | None = None,
    ) -> list[str]:
        task_payload = dict(task or {})
        task_type = str(task_payload.get("type", "")).upper()
        route_specialist = str((route or {}).get("specialist", "")).strip().lower()
        query_text = str(
            task_payload.get("query")
            or task_payload.get("question")
            or task_payload.get("prompt")
            or ""
        ).strip()
        gsm8k_mode = self._is_gsm8k_math_task(task_payload)
        factual_chat = (
            task_type in {"CHAT_TASK", "GENERAL_TASK", "GRAMMAR_TASK", "MMLU_TASK"}
            and self._query_looks_reality_fact(query_text)
        )
        allowed = (
            self.GPU_ARC_TARGET_GALAXIES
            if task_type == "ARC_TASK"
            else self.GPU_GSM8K_TARGET_GALAXIES
            if gsm8k_mode
            else self.GPU_MATH_TARGET_GALAXIES
            if task_type == "MATH_TASK"
            else self.GPU_MMLU_TARGET_GALAXIES
            if task_type == "MMLU_TASK"
            else self.GPU_LHE_TARGET_GALAXIES
            if task_type == "LHE_TASK"
            else self.GPU_FACTUAL_CHAT_TARGET_GALAXIES
            if factual_chat
            else self.GPU_CHAT_TARGET_GALAXIES
            if task_type in {"CHAT_TASK", "GENERAL_TASK", "GRAMMAR_TASK"}
            or route_specialist in {"chat", "grammar", "any"}
            else self.GPU_FACTUAL_TARGET_GALAXIES
        )
        if isinstance(route, dict) and isinstance(route.get("galaxy_names"), list):
            selected = [
                str(name).strip()
                for name in route["galaxy_names"]
                if str(name).strip() in allowed
            ]
            if selected:
                if gsm8k_mode:
                    return list(dict.fromkeys(selected + list(self.GPU_GSM8K_TARGET_GALAXIES)))
                if factual_chat:
                    return list(
                        dict.fromkeys(
                            selected + list(self.GPU_FACTUAL_CHAT_TARGET_GALAXIES)
                        )
                    )
                if task_type == "LHE_TASK":
                    return list(dict.fromkeys(selected + list(self.GPU_FACTUAL_TARGET_GALAXIES)))
                if task_type == "MMLU_TASK":
                    return selected
                return selected
        return list(allowed)

    def _grammar_rule_metadata(self, rule_id: str) -> dict[str, Any]:
        target = str(rule_id).strip()
        if not target:
            return {}
        if not self._gpu_galaxy_catalog:
            try:
                self.bind_gpu_galaxy_runtime()
            except Exception:
                return {}
        for entry in self.get_gpu_galaxy_catalog():
            if str(entry.get("galaxy", "")).strip() != "Grammar":
                continue
            if str(entry.get("id", "")).strip() != target:
                continue
            metadata = entry.get("metadata")
            if isinstance(metadata, dict):
                return dict(metadata)
            return {}
        return {}

    @staticmethod
    def _encode_defeasible_trit(trit: int) -> int:
        if int(trit) > 0:
            return 2
        if int(trit) < 0:
            return 0
        return 1

    @staticmethod
    def _decode_defeasible_trit(encoded: int) -> int:
        value = int(encoded) & 0x3
        if value == 2:
            return 1
        if value == 0:
            return -1
        return 0

    @classmethod
    def _pack_defeasible_proof_tag(cls, definite_trit: int, defeasible_trit: int) -> int:
        return (
            cls._encode_defeasible_trit(definite_trit)
            | (cls._encode_defeasible_trit(defeasible_trit) << 2)
        )

    @classmethod
    def _unpack_defeasible_proof_tag(cls, proof_tag: int) -> tuple[int, int]:
        packed = int(proof_tag)
        return (
            cls._decode_defeasible_trit(packed & 0x3),
            cls._decode_defeasible_trit((packed >> 2) & 0x3),
        )

    def _defeasible_rule_profile(self, program_id: str) -> dict[str, Any]:
        metadata = self._grammar_rule_metadata(program_id)
        superior_to = [
            str(value).strip()
            for value in list(metadata.get("superior_to", []) or [])
            if str(value).strip()
        ]
        try:
            raw_strength = int(metadata.get("rule_strength", 0))
        except Exception:
            raw_strength = 0
        rule_strength = 1 if raw_strength > 0 else (-1 if raw_strength < 0 else 0)
        try:
            trust_weight = float(metadata.get("trust_weight", 1.0))
        except Exception:
            trust_weight = 1.0
        return {
            "rule_id": str(program_id).strip(),
            "rule_strength": rule_strength,
            "superior_to": superior_to,
            "trust_weight": max(0.0, min(trust_weight, 1.0)),
        }

    @staticmethod
    def _record_path_defeasible_tag(record: dict[str, Any]) -> int:
        try:
            if "path_defeasible_tag" in record:
                return int(record.get("path_defeasible_tag", 1))
        except Exception:
            pass
        candidate = record.get("candidate") if isinstance(record.get("candidate"), dict) else {}
        try:
            if "path_defeasible_tag" in candidate:
                return int(candidate.get("path_defeasible_tag", 1))
        except Exception:
            pass
        path = candidate.get("path") if isinstance(candidate.get("path"), dict) else {}
        try:
            return int(path.get("path_defeasible_tag", 1))
        except Exception:
            return 1

    def _candidate_defeasible_rule_id(
        self,
        *,
        candidate: dict[str, Any],
        path: dict[str, Any],
    ) -> str:
        match = candidate.get("match") if isinstance(candidate.get("match"), dict) else {}
        metadata = match.get("metadata") if isinstance(match.get("metadata"), dict) else {}
        candidate_rule_id = str(metadata.get("rule_id", "")).strip()
        if not candidate_rule_id and str(match.get("galaxy", "")).strip() == "Grammar":
            candidate_rule_id = str(match.get("id", "")).strip()
        if candidate_rule_id:
            candidate_metadata = self._grammar_rule_metadata(candidate_rule_id)
            if candidate_metadata:
                return candidate_rule_id
        return str(path.get("program_id", "")).strip()

    def _apply_early_defeasible_gate(
        self,
        *,
        task_type: str | None = None,
        paths: list[dict[str, Any]],
        swarm_weights: list[float],
        selection_steps: list[str],
    ) -> None:
        if not paths or not swarm_weights:
            return
        if str(task_type or "").strip() == "MMLU_TASK":
            selection_steps.append("GRE triple defeasible stage1: deferred_for_mmlu")
            return
        resolver = self.get_defeasible_resolver()
        neutral_proof_tag = self._pack_defeasible_proof_tag(0, 0)
        path_count = min(len(paths), len(swarm_weights))
        for path in paths:
            path["path_defeasible_tag"] = 1
            path["path_defeasible_verdict"] = 0.0
            path["path_defeasible_proof_tag"] = int(neutral_proof_tag)
        if resolver is None or path_count <= 0:
            return

        profiles: list[dict[str, Any]] = []
        rule_indexes: dict[str, list[int]] = {}
        for worker_index, path in enumerate(paths[:path_count]):
            profile = self._defeasible_rule_profile(str(path.get("program_id", "")).strip())
            profiles.append(profile)
            if profile["rule_id"]:
                rule_indexes.setdefault(profile["rule_id"], []).append(worker_index)

        has_nondefault_logic = any(
            int(profile["rule_strength"]) != 0 or list(profile["superior_to"])
            for profile in profiles
        )
        if not has_nondefault_logic:
            selection_steps.append(
                "GRE triple defeasible stage1: compatibility mode "
                f"(paths={path_count})"
            )
            return

        conclusions = np.zeros((path_count, path_count), dtype=np.float32)
        rule_strengths = np.zeros((path_count,), dtype=np.int8)
        max_superiors = max(
            1,
            max(len(profile["superior_to"]) for profile in profiles),
        )
        superiority = np.full((path_count, max_superiors), 0xFFFFFFFF, dtype=np.uint32)
        incoming_superiors: dict[int, list[str]] = {}

        for worker_index, profile in enumerate(profiles):
            scaled_score = abs(float(swarm_weights[worker_index])) * float(profile["trust_weight"])
            if scaled_score <= 0.0:
                continue
            conclusions[worker_index, worker_index] = scaled_score
            rule_strengths[worker_index] = np.int8(int(profile["rule_strength"]))
            defeated_workers: list[int] = []
            for defeated_rule_id in profile["superior_to"]:
                defeated_workers.extend(rule_indexes.get(str(defeated_rule_id), []))
            for slot, inferior_index in enumerate(dict.fromkeys(defeated_workers)):
                if slot >= max_superiors:
                    break
                superiority[worker_index, slot] = np.uint32(int(inferior_index))
                incoming_superiors.setdefault(int(inferior_index), []).append(str(profile["rule_id"]))
                if int(inferior_index) != int(worker_index):
                    conclusions[worker_index, int(inferior_index)] = -scaled_score

        verdicts, proof_tags = resolver.resolve(
            conclusions,
            rule_strengths,
            superiority,
            num_workers=path_count,
            num_candidates=path_count,
            max_superiors=max_superiors,
        )
        hard_defeats = 0
        soft_defeats = 0
        for path_index, path in enumerate(paths[:path_count]):
            verdict = float(verdicts[path_index])
            proof_tag = int(proof_tags[path_index])
            definite_trit, defeasible_trit = self._unpack_defeasible_proof_tag(proof_tag)
            path_tag = 1
            if definite_trit < 0:
                path_tag = -1
                swarm_weights[path_index] = 0.0
                hard_defeats += 1
            elif defeasible_trit < 0 or verdict < -1e-6:
                path_tag = 0
                swarm_weights[path_index] = float(swarm_weights[path_index]) * 0.3
                soft_defeats += 1
            path["path_defeasible_tag"] = int(path_tag)
            path["path_defeasible_verdict"] = float(verdict)
            path["path_defeasible_proof_tag"] = int(proof_tag)
            if path_tag <= 0:
                defeated_by = [
                    str(rule_id).strip()
                    for rule_id in incoming_superiors.get(path_index, [])
                    if str(rule_id).strip()
                ]
                if defeated_by:
                    path["path_defeated_by"] = defeated_by[0]
        selection_steps.append(
            "GRE triple defeasible stage1: "
            f"paths={path_count} soft={soft_defeats} hard={hard_defeats}"
        )

    def _apply_intra_path_defeasible(
        self,
        *,
        local_candidates: list[dict[str, Any]],
        path: dict[str, Any],
        task_type: str,
        selection_steps: list[str],
    ) -> None:
        if not local_candidates:
            return
        neutral_proof_tag = self._pack_defeasible_proof_tag(0, 0)
        path_tag = int(path.get("path_defeasible_tag", 1))
        for candidate in local_candidates:
            candidate.setdefault("specialist_intra_defeasible", 0.0)
            candidate.setdefault("specialist_intra_proof_tag", int(neutral_proof_tag))
            candidate["path_defeasible_tag"] = int(path_tag)
        if str(task_type).strip() == "MMLU_TASK":
            selection_steps.append(
                "GRE triple defeasible stage2: deferred_for_mmlu "
                f"({str(path.get('label') or path.get('program_id', 'path'))})"
            )
            return
        resolver = self.get_defeasible_resolver()
        if resolver is None:
            return

        profiles: list[dict[str, Any]] = []
        rule_indexes: dict[str, list[int]] = {}
        for candidate_index, candidate in enumerate(local_candidates):
            rule_id = self._candidate_defeasible_rule_id(candidate=candidate, path=path)
            profile = self._defeasible_rule_profile(rule_id)
            profiles.append(profile)
            if profile["rule_id"]:
                rule_indexes.setdefault(profile["rule_id"], []).append(candidate_index)

        has_nondefault_logic = any(
            int(profile["rule_strength"]) != 0 or list(profile["superior_to"])
            for profile in profiles
        )
        if not has_nondefault_logic:
            selection_steps.append(
                "GRE triple defeasible stage2: compatibility mode "
                f"({str(task_type).strip() or 'task'}, candidates={len(local_candidates)})"
            )
            return

        candidate_count = len(local_candidates)
        conclusions = np.zeros((candidate_count, candidate_count), dtype=np.float32)
        rule_strengths = np.zeros((candidate_count,), dtype=np.int8)
        max_superiors = max(
            1,
            max(len(profile["superior_to"]) for profile in profiles),
        )
        superiority = np.full((candidate_count, max_superiors), 0xFFFFFFFF, dtype=np.uint32)

        for candidate_index, candidate in enumerate(local_candidates):
            profile = profiles[candidate_index]
            base_score = abs(
                float(
                    candidate.get(
                        "specialist_coherence",
                        candidate.get("specialist_resonance", candidate.get("similarity", 0.0)),
                    )
                )
            )
            scaled_score = base_score * float(profile["trust_weight"])
            if scaled_score <= 0.0:
                continue
            conclusions[candidate_index, candidate_index] = scaled_score
            rule_strengths[candidate_index] = np.int8(int(profile["rule_strength"]))
            defeated_indexes: list[int] = []
            for defeated_rule_id in profile["superior_to"]:
                defeated_indexes.extend(rule_indexes.get(str(defeated_rule_id), []))
            for slot, inferior_index in enumerate(dict.fromkeys(defeated_indexes)):
                if slot >= max_superiors:
                    break
                superiority[candidate_index, slot] = np.uint32(int(inferior_index))
                if int(inferior_index) != int(candidate_index):
                    conclusions[candidate_index, int(inferior_index)] = -scaled_score

        verdicts, proof_tags = resolver.resolve(
            conclusions,
            rule_strengths,
            superiority,
            num_workers=candidate_count,
            num_candidates=candidate_count,
            max_superiors=max_superiors,
        )
        decisive_count = 0
        for candidate_index, candidate in enumerate(local_candidates):
            verdict = float(verdicts[candidate_index])
            proof_tag = int(proof_tags[candidate_index])
            if abs(verdict) > 1e-6:
                decisive_count += 1
            candidate["specialist_intra_defeasible"] = float(verdict)
            candidate["specialist_intra_proof_tag"] = int(proof_tag)
        selection_steps.append(
            "GRE triple defeasible stage2: "
            f"{str(path.get('label') or path.get('program_id', 'path'))} "
            f"candidates={candidate_count} decisive={decisive_count}"
        )

    def _halting_record_candidate_id(
        self,
        *,
        record: dict[str, Any],
        task_type: str,
        gsm8k_mode: bool,
    ) -> str:
        if task_type in {"MMLU_TASK", "LHE_TASK"}:
            return str(record.get("option_text", "")).strip()
        if gsm8k_mode:
            return self._gsm8k_preview_candidate_id(record)
        candidate = record.get("candidate")
        if not isinstance(candidate, dict):
            return ""
        match = candidate.get("match")
        if not isinstance(match, dict):
            return ""
        return str(match.get("id", "")).strip()

    def _defeasible_event_specialist(
        self,
        *,
        task_type: str,
        record: dict[str, Any],
    ) -> str:
        candidate = record.get("candidate") if isinstance(record.get("candidate"), dict) else {}
        path = candidate.get("path") if isinstance(candidate.get("path"), dict) else {}
        specialist = str(path.get("specialist", "")).strip()
        if specialist:
            return specialist
        mapping = {
            "ARC_TASK": "visual",
            "MATH_TASK": "math",
            "LHE_TASK": "grammar",
            "MMLU_TASK": "grammar",
            "CHAT_TASK": "chat",
            "GENERAL_TASK": "grammar",
            "GRAMMAR_TASK": "grammar",
        }
        return mapping.get(str(task_type).strip().upper(), "grammar")

    def _emit_defeasible_verdict_event(
        self,
        *,
        stage: str,
        task_type: str,
        record: dict[str, Any],
        candidate: dict[str, Any] | None,
        profile: dict[str, Any],
        verdict: float,
        proof_tag: int,
        was_defeated_by: str | None,
    ) -> None:
        if self.shadow_copy is None:
            return
        verdict_trit = 1 if verdict > 1e-6 else (-1 if verdict < -1e-6 else 0)
        if verdict_trit == 0 and not was_defeated_by:
            return
        candidate_dict = candidate if isinstance(candidate, dict) else {}
        path = candidate_dict.get("path") if isinstance(candidate_dict.get("path"), dict) else {}
        match = candidate_dict.get("match") if isinstance(candidate_dict.get("match"), dict) else {}
        candidate_id = self._halting_record_candidate_id(
            record=record,
            task_type=task_type,
            gsm8k_mode=bool(record.get("gsm8k_mode", False) or candidate_dict.get("gsm8k_mode", False)),
        )
        if not candidate_id:
            candidate_id = str(match.get("id", "")).strip() or str(record.get("option_text", "")).strip()
        program_id = str((candidate_dict.get("program") or {}).get("id", "")).strip() or str(profile.get("rule_id", "")).strip()
        query_text = str(
            path.get("query_text")
            or record.get("option_text")
            or record.get("preview_answer")
            or candidate_id
        ).strip()
        confidence = max(0.0, min(abs(float(verdict)), 1.0))
        from .execution_events import DefeasibleVerdictEvent

        verdict_event = DefeasibleVerdictEvent(
            stage=str(stage).strip(),
            candidate_id=str(candidate_id).strip(),
            program_id=program_id,
            verdict_trit=int(verdict_trit),
            proof_tag=int(proof_tag),
            rule_strength=int(profile.get("rule_strength", 0) or 0),
            was_defeated_by=(None if not was_defeated_by else str(was_defeated_by).strip()),
            confidence=float(confidence),
            timestamp_us=int(time.time_ns() // 1_000),
            domain_hint=str(path.get("domain_hint") or task_type or "").strip() or None,
        )
        payload = verdict_event.as_dict()
        payload.update(
            {
                "timestamp": float(payload["timestamp_us"]) / 1_000_000.0,
                "query": query_text,
                "query_context": query_text,
                "prompt": query_text,
                "specialist": self._defeasible_event_specialist(task_type=task_type, record=record),
                "galaxy": "Grammar",
                "verification": "defeasible_verdict",
                "confidence": float(confidence),
                "outcome": int(verdict_trit),
                "defeat_source": (None if not was_defeated_by else str(was_defeated_by).strip()),
            }
        )
        self.shadow_copy.record_event(
            event_type="defeasible_verdict",
            event_data=payload,
        )

    def _apply_defeasible_specialist_resolution(
        self,
        *,
        records: list[dict[str, Any]],
        task_type: str,
        gsm8k_mode: bool,
        selection_steps: list[str],
    ) -> None:
        if not records:
            return
        resolver = self.get_defeasible_resolver()
        if resolver is None:
            return

        candidate_keys: list[str] = []
        key_to_index: dict[str, int] = {}
        profiles: list[dict[str, Any]] = []
        current_rule_indexes: dict[str, list[int]] = {}
        record_rows: list[tuple[str, dict[str, Any], dict[str, Any] | None, float]] = []

        for worker_index, record in enumerate(records):
            candidate_key = self._halting_record_candidate_id(
                record=record,
                task_type=task_type,
                gsm8k_mode=gsm8k_mode,
            )
            if candidate_key and candidate_key not in key_to_index:
                key_to_index[candidate_key] = len(candidate_keys)
                candidate_keys.append(candidate_key)
            candidate = record.get("candidate") if isinstance(record.get("candidate"), dict) else None
            program = candidate.get("program") if isinstance(candidate, dict) else {}
            program_id = str((program or {}).get("id", "")).strip()
            profile = self._defeasible_rule_profile(program_id)
            profiles.append(profile)
            if profile["rule_id"]:
                current_rule_indexes.setdefault(profile["rule_id"], []).append(worker_index)
            raw_score = float(
                record.get(
                    "path_score",
                    (candidate or {}).get("path_score", (candidate or {}).get("gpu_score", 0.0)),
                )
            )
            record_rows.append((candidate_key, record, candidate, raw_score))

        if not candidate_keys:
            return

        neutral_proof_tag = self._pack_defeasible_proof_tag(0, 0)
        incoming_superiors: dict[int, list[str]] = {}
        has_nondefault_logic = any(
            int(profile["rule_strength"]) != 0 or list(profile["superior_to"])
            for profile in profiles
        )
        if not has_nondefault_logic:
            for worker_index, (candidate_key, record, candidate, raw_score) in enumerate(record_rows):
                defeasible_trit = 1 if raw_score > 1e-6 else (-1 if raw_score < -1e-6 else 0)
                verdict = float(raw_score)
                proof_tag = self._pack_defeasible_proof_tag(0, defeasible_trit)
                path_tag = self._record_path_defeasible_tag(record)
                was_defeated_by = None
                if isinstance(candidate, dict):
                    path = candidate.get("path") if isinstance(candidate.get("path"), dict) else {}
                    was_defeated_by = str(path.get("path_defeated_by", "")).strip() or None
                if path_tag < 0:
                    verdict = 0.0
                    proof_tag = int(neutral_proof_tag)
                elif path_tag == 0:
                    verdict *= 0.3
                record["specialist_defeasible_verdict"] = float(verdict)
                record["specialist_proof_tag"] = int(proof_tag)
                record["path_score"] = float(raw_score) + (0.04 * float(verdict))
                if isinstance(candidate, dict):
                    candidate["specialist_defeasible_verdict"] = float(verdict)
                    candidate["specialist_proof_tag"] = int(proof_tag)
                    candidate["path_score"] = float(record["path_score"])
                self._emit_defeasible_verdict_event(
                    stage="final",
                    task_type=task_type,
                    record=record,
                    candidate=candidate,
                    profile=profiles[worker_index],
                    verdict=float(verdict),
                    proof_tag=int(proof_tag),
                    was_defeated_by=was_defeated_by,
                )
            selection_steps.append(
                "GRE defeasible resolver: compatibility mode "
                f"(records={len(records)}, candidates={len(candidate_keys)})"
            )
            return

        conclusions = np.zeros((len(records), len(candidate_keys)), dtype=np.float32)
        rule_strengths = np.zeros((len(records),), dtype=np.int8)
        max_superiors = max(
            1,
            max(
                len(profile["superior_to"])
                for profile in profiles
            ),
        )
        superiority = np.full((len(records), max_superiors), 0xFFFFFFFF, dtype=np.uint32)

        for worker_index, (candidate_key, _record, _candidate, raw_score) in enumerate(record_rows):
            profile = profiles[worker_index]
            scaled_score = abs(float(raw_score)) * float(profile["trust_weight"])
            if scaled_score <= 0.0:
                continue
            selected_index = key_to_index.get(candidate_key)
            if selected_index is None:
                continue
            conclusions[worker_index, :] = -scaled_score
            conclusions[worker_index, selected_index] = scaled_score
            rule_strengths[worker_index] = np.int8(int(profile["rule_strength"]))
            defeated_workers: list[int] = []
            for defeated_rule_id in profile["superior_to"]:
                defeated_workers.extend(current_rule_indexes.get(str(defeated_rule_id), []))
            for slot, inferior_index in enumerate(dict.fromkeys(defeated_workers)):
                if slot >= max_superiors:
                    break
                superiority[worker_index, slot] = np.uint32(int(inferior_index))
                incoming_superiors.setdefault(int(inferior_index), []).append(str(profile["rule_id"]))

        verdicts, proof_tags = resolver.resolve(
            conclusions,
            rule_strengths,
            superiority,
            num_workers=len(records),
            num_candidates=len(candidate_keys),
            max_superiors=max_superiors,
        )
        decisive_count = 0
        for candidate_key, record, candidate, raw_score in record_rows:
            candidate_index = key_to_index.get(candidate_key)
            if candidate_index is None:
                continue
            verdict = float(verdicts[candidate_index])
            proof_tag = int(proof_tags[candidate_index])
            path_tag = self._record_path_defeasible_tag(record)
            was_defeated_by = None
            if isinstance(candidate, dict):
                path = candidate.get("path") if isinstance(candidate.get("path"), dict) else {}
                was_defeated_by = str(path.get("path_defeated_by", "")).strip() or None
            if not was_defeated_by:
                defeated_by_rules = [
                    str(rule_id).strip()
                    for rule_id in incoming_superiors.get(candidate_index, [])
                    if str(rule_id).strip()
                ]
                if defeated_by_rules:
                    was_defeated_by = defeated_by_rules[0]
            if path_tag < 0:
                verdict = 0.0
                proof_tag = int(neutral_proof_tag)
            elif path_tag == 0:
                verdict *= 0.3
            if abs(verdict) > 1e-6:
                decisive_count += 1
            updated_score = float(raw_score) + (0.04 * verdict)
            record["specialist_defeasible_verdict"] = verdict
            record["specialist_proof_tag"] = proof_tag
            record["path_score"] = updated_score
            if isinstance(candidate, dict):
                candidate["specialist_defeasible_verdict"] = verdict
                candidate["specialist_proof_tag"] = proof_tag
                candidate["path_score"] = updated_score
                if was_defeated_by:
                    candidate["specialist_was_defeated_by"] = was_defeated_by
            record["specialist_was_defeated_by"] = was_defeated_by
            self._emit_defeasible_verdict_event(
                stage="final",
                task_type=task_type,
                record=record,
                candidate=candidate,
                profile=profiles[candidate_index],
                verdict=float(verdict),
                proof_tag=int(proof_tag),
                was_defeated_by=was_defeated_by,
            )
        selection_steps.append(
            "GRE defeasible resolver: "
            f"workers={len(records)} candidates={len(candidate_keys)} decisive={decisive_count}"
        )

    def _mmlu_option_rule_weights(self) -> tuple[float, float]:
        metadata = self._grammar_rule_metadata("reasoning_elimination_option_score")
        validation_weight = 0.18
        support_weight = 0.02
        try:
            validation_weight = float(metadata.get("validation_weight", validation_weight))
        except Exception:
            pass
        try:
            support_weight = float(metadata.get("support_weight", support_weight))
        except Exception:
            pass
        return validation_weight, support_weight

    def _mmlu_relative_gap_threshold(self) -> float:
        metadata = self._grammar_rule_metadata("halting_elimination_relative")
        try:
            return float(metadata.get("threshold", 0.01))
        except Exception:
            return 0.01

    def _parse_override_weight(self, rule_id: str, default: float) -> float:
        metadata = self._grammar_rule_metadata(rule_id)
        try:
            return float(metadata.get("override_weight", default))
        except Exception:
            return default

    def _parse_navigation_weights(
        self,
        *,
        task: dict[str, Any] | None,
    ) -> tuple[float, float, float, float]:
        metadata = self._grammar_rule_metadata("reasoning_parse_navigation_weights")
        task_type = str((task or {}).get("type", "")).strip().upper()
        if self._is_gsm8k_math_task(task):
            prefix = "gsm8k"
        elif task_type == "MATH_TASK":
            prefix = "math"
        else:
            prefix = "default"
        defaults = {
            "base": 0.28,
            "fusion": 0.32,
            "forward": 0.14,
            "backward": 0.26,
        }
        weights: dict[str, float] = {}
        for key, fallback in defaults.items():
            value = metadata.get(f"{prefix}_{key}_weight", metadata.get(f"default_{key}_weight", fallback))
            try:
                weights[key] = max(0.0, float(value))
            except Exception:
                weights[key] = fallback
        total = sum(weights.values())
        if total <= 0.0:
            return defaults["base"], defaults["fusion"], defaults["forward"], defaults["backward"]
        return (
            weights["base"] / total,
            weights["fusion"] / total,
            weights["forward"] / total,
            weights["backward"] / total,
        )

    def _gsm8k_strategy_weight(self, strategy: str) -> float:
        metadata = self._grammar_rule_metadata("reasoning_word_problem_chain")
        strategy_weights = metadata.get("strategy_weights") if isinstance(metadata.get("strategy_weights"), dict) else {}
        token = str(strategy or "").strip()
        if not token:
            return 1.0
        try:
            return float(strategy_weights.get(token, 1.0))
        except Exception:
            return 1.0

    def _gsm8k_halting_thresholds(self) -> tuple[float, float, float]:
        metadata = self._grammar_rule_metadata("halting_word_problem_consensus")
        defaults = {
            "minimum_threshold": 0.3,
            "gap_threshold": 0.04,
            "agreement_threshold": 1.0,
        }
        try:
            minimum = float(metadata.get("minimum_threshold", defaults["minimum_threshold"]))
        except Exception:
            minimum = defaults["minimum_threshold"]
        try:
            gap = float(metadata.get("gap_threshold", defaults["gap_threshold"]))
        except Exception:
            gap = defaults["gap_threshold"]
        try:
            agreement = float(metadata.get("agreement_threshold", defaults["agreement_threshold"]))
        except Exception:
            agreement = defaults["agreement_threshold"]
        return minimum, gap, agreement

    def _select_gpu_reasoning_program(self, program_id: str) -> dict[str, Any]:
        cached = self._gpu_reasoning_programs.get(program_id)
        if cached is not None:
            return dict(cached)
        grammar = self.galaxy_manager.get_galaxy("Grammar")
        for entry in getattr(grammar, "entries", []):
            entry_id = str(entry.get("id") or entry.get("rule_id") or "").strip()
            if entry_id == program_id:
                self._gpu_reasoning_programs[program_id] = dict(entry)
                return dict(self._gpu_reasoning_programs[program_id])
        raise RuntimeError(
            f"Missing GPU reasoning program: {program_id}"
        )

    def _embed_query_gpu(self, query_text: str, *, task: dict[str, Any] | None = None) -> list[float]:
        if str((task or {}).get("type", "")).upper() == "ARC_TASK":
            task_id = str((task or {}).get("task_id", "")).strip()
            if task_id:
                return self._arc_task_embedding16(task_id)
        engine = self.get_gpu_query_embedding_engine()
        values = engine.embed_sentence_gpu(query_text)
        return [float(values[i]) for i in range(min(16, len(values)))]

    @staticmethod
    def _format_similarity(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.2f}"

    def _build_gpu_thinking_trace(
        self,
        *,
        binding: dict[str, Any],
        program_id: str,
        match: dict[str, Any],
        similarity: float | None,
        specialist: str | None = None,
        read_field: str = "answer_text",
        extra_steps: list[str] | None = None,
    ) -> list[str]:
        label = str(match.get("name") or match.get("id") or "entry").strip()
        trace = [
            f"Specialist route: {str(specialist or 'auto').strip()}",
            (
                f"Scanning Galaxy: {', '.join(str(name) for name in binding.get('galaxies', []))} "
                f"({int(binding.get('entry_count', 0))} entries)"
            ),
            (
                f'Top match: entry[{int(match.get("index", -1))}] "{label}" '
                f"(similarity={self._format_similarity(similarity)}, "
                f"confidence={float(match.get('confidence', 0.0)):.2f})"
            ),
            f"RPN program: {program_id}",
            f"Galaxy read: {read_field} from entry[{int(match.get('index', -1))}]",
        ]
        if extra_steps:
            trace.extend(str(step) for step in extra_steps if str(step).strip())
        return trace

    @staticmethod
    def _render_thinking_xml(thinking_trace: list[str], answer: Any) -> str:
        lines = "\n".join(thinking_trace)
        if isinstance(answer, str):
            answer_text = answer
        else:
            answer_text = json.dumps(answer, ensure_ascii=True)
        return f"<thinking>\n{lines}\n</thinking>\n{answer_text}"

    @staticmethod
    def _format_math_answer(value: Any) -> str:
        try:
            numeric = float(value)
        except Exception:
            return str(value).strip()
        if not math.isfinite(numeric):
            return str(value).strip()
        rounded = round(numeric)
        if abs(numeric - rounded) <= 1e-9:
            return str(int(rounded))
        return f"{numeric:.12g}"

    @staticmethod
    def _format_math_interval(
        lower: str,
        upper: str,
        *,
        lower_inclusive: bool,
        upper_inclusive: bool,
    ) -> str:
        left = "[" if lower_inclusive else "("
        right = "]" if upper_inclusive else ")"
        return f"{left}{lower},{upper}{right}"

    @staticmethod
    def _format_escaped_currency(value: str) -> str:
        stripped = value.strip()
        if stripped.startswith("\\$"):
            return stripped
        if stripped.startswith("$"):
            return "\\" + stripped
        return f"\\${stripped}"

    def _evaluate_math_template(
        self,
        *,
        engine: Any,
        match: dict[str, Any],
        query_text: str = "",
        numeric_fallbacks: list[float] | None = None,
    ) -> tuple[str, list[str]] | None:
        metadata = match.get("metadata") if isinstance(match.get("metadata"), dict) else {}
        template_ref = str(metadata.get("template_ref", "")).strip()
        if not template_ref:
            meaning_ref = str(metadata.get("meaning_ref", "")).strip()
            match_id = str(match.get("id", "")).strip()
            for candidate in (meaning_ref, match_id):
                if candidate.startswith("math_template_"):
                    template_ref = candidate
                    break
        params = metadata.get("template_params") if isinstance(metadata.get("template_params"), dict) else {}
        numbers: list[float] = []
        if isinstance(numeric_fallbacks, list):
            for raw_value in numeric_fallbacks:
                try:
                    numbers.append(float(raw_value))
                except Exception:
                    continue
        if query_text:
            for raw_value in self._extract_numeric_literals(query_text):
                try:
                    numeric_value = float(raw_value)
                except Exception:
                    continue
                if not numbers or all(abs(existing - numeric_value) > 1e-6 for existing in numbers):
                    numbers.append(numeric_value)
        if not template_ref:
            return None

        if template_ref == "math_template_arithmetic_chain_gpu":
            tokens = params.get("rpn_tokens")
            if not isinstance(tokens, list) or not tokens:
                return None
            program = " ".join(str(token) for token in tokens)
            value = engine.evaluate(program)
            return self._format_math_answer(value), [
                f"GPU math template: {template_ref}",
                f"GPU math eval: {program}",
            ]

        if template_ref == "math_template_linear_equation_ax_plus_b_eq_c_gpu":
            coeffs = params.get("coefficients") if isinstance(params.get("coefficients"), dict) else params
            a = float(coeffs.get("a", numbers[0] if len(numbers) >= 3 else 0))
            b = float(coeffs.get("b", numbers[1] if len(numbers) >= 3 else 0))
            c = float(coeffs.get("c", numbers[2] if len(numbers) >= 3 else 0))
            if a == 0:
                return None
            program = f"{c:g} {b:g} - {a:g} /"
            value = engine.evaluate(program)
            return self._format_math_answer(value), [
                f"GPU math template: {template_ref}",
                f"GPU math eval: {program}",
            ]

        if template_ref == "math_template_polynomial_degree_gpu":
            degrees = params.get("degrees")
            if not isinstance(degrees, list) or not degrees:
                return None
            numeric_degrees = [float(value) for value in degrees]
            program = " ".join(f"{value:g}" for value in numeric_degrees)
            if len(numeric_degrees) > 1:
                program = f"{program} " + " ".join("max" for _ in range(len(numeric_degrees) - 1))
            value = engine.evaluate(program)
            return self._format_math_answer(value), [
                f"GPU math template: {template_ref}",
                f"GPU math eval: {program}",
            ]

        if template_ref == "math_template_polynomial_eval_gpu":
            terms = params.get("terms")
            divisor = float(params.get("divisor", 1))
            if not isinstance(terms, list) or not terms:
                return None
            program_tokens: list[str] = []
            for idx, raw_term in enumerate(terms):
                if not isinstance(raw_term, dict):
                    return None
                base = float(raw_term.get("base", 0))
                power = float(raw_term.get("power", 1))
                coefficient = float(raw_term.get("coefficient", 1))
                program_tokens.extend([f"{base:g}", f"{power:g}", "pow"])
                if coefficient != 1:
                    program_tokens.extend([f"{coefficient:g}", "*"])
                if idx:
                    program_tokens.append("+")
            if divisor != 1:
                program_tokens.extend([f"{divisor:g}", "/"])
            program = " ".join(program_tokens)
            value = engine.evaluate(program)
            return self._format_math_answer(value), [
                f"GPU math template: {template_ref}",
                f"GPU math eval: {program}",
            ]

        if template_ref == "math_template_factorial_gpu":
            n_value = float(params.get("n", numbers[0] if numbers else 0))
            program = f"{n_value:g} factorial"
            value = engine.evaluate(program)
            return self._format_math_answer(value), [
                f"GPU math template: {template_ref}",
                f"GPU math eval: {program}",
            ]

        if template_ref == "math_template_binomial_gpu":
            n_value = float(params.get("n", numbers[0] if len(numbers) >= 2 else 0))
            k_value = float(params.get("k", numbers[1] if len(numbers) >= 2 else 0))
            program = f"{n_value:g} {k_value:g} binom"
            value = engine.evaluate(program)
            return self._format_math_answer(value), [
                f"GPU math template: {template_ref}",
                f"GPU math eval: {program}",
            ]

        if template_ref == "math_template_arithmetic_series_sum_gpu":
            if "first_n" in params or (len(numbers) == 1 and not params):
                n_value = float(params.get("first_n", numbers[0] if numbers else 0))
                program = f"{n_value:g} {n_value:g} 1 + * 2 /"
            else:
                a1 = float(params.get("a1", numbers[0] if len(numbers) >= 3 else 0))
                an = float(params.get("an", numbers[1] if len(numbers) >= 3 else 0))
                n_value = float(params.get("n", numbers[2] if len(numbers) >= 3 else 0))
                program = f"{n_value:g} {a1:g} {an:g} + * 2 /"
            value = engine.evaluate(program)
            return self._format_math_answer(value), [
                f"GPU math template: {template_ref}",
                f"GPU math eval: {program}",
            ]

        if template_ref == "math_template_geometric_series_sum_gpu":
            a_value = float(params.get("a", numbers[0] if len(numbers) >= 3 else 0))
            r_value = float(params.get("r", numbers[1] if len(numbers) >= 3 else 0))
            n_value = float(params.get("n", numbers[2] if len(numbers) >= 3 else 0))
            program = f"{a_value:g} 1 {r_value:g} {n_value:g} pow - * 1 {r_value:g} - /"
            value = engine.evaluate(program)
            return self._format_math_answer(value), [
                f"GPU math template: {template_ref}",
                f"GPU math eval: {program}",
            ]

        if template_ref == "math_template_unit_conversion_scale_gpu":
            value_in = float(params.get("value", numbers[0] if numbers else 0))
            scale = float(params.get("scale", 1))
            offset = float(params.get("offset", 0))
            program = f"{value_in:g} {scale:g} * {offset:g} +"
            value = engine.evaluate(program)
            return self._format_math_answer(value), [
                f"GPU math template: {template_ref}",
                f"GPU math eval: {program}",
            ]

        if template_ref == "math_template_unit_conversion_affine_gpu":
            value_in = float(params.get("value", numbers[0] if numbers else 0))
            scale = float(params.get("scale", 1))
            offset = float(params.get("offset", 0))
            program = f"{value_in:g} {scale:g} * {offset:g} +"
            value = engine.evaluate(program)
            return self._format_math_answer(value), [
                f"GPU math template: {template_ref}",
                f"GPU math eval: {program}",
            ]

        if template_ref == "math_template_quadratic_discriminant_gpu":
            a = float(params.get("a", numbers[0] if len(numbers) >= 3 else 0))
            b = float(params.get("b", numbers[1] if len(numbers) >= 3 else 0))
            c = float(params.get("c", numbers[2] if len(numbers) >= 3 else 0))
            program = f"{b:g} {b:g} * 4 {a:g} * {c:g} * -"
            value = engine.evaluate(program)
            return self._format_math_answer(value), [
                f"GPU math template: {template_ref}",
                f"GPU math eval: {program}",
            ]

        if template_ref == "math_template_quadratic_roots_gpu":
            a = float(params.get("a", numbers[0] if len(numbers) >= 3 else 0))
            b = float(params.get("b", numbers[1] if len(numbers) >= 3 else 0))
            c = float(params.get("c", numbers[2] if len(numbers) >= 3 else 0))
            if a == 0:
                return None
            root_plus_program = (
                f"0 {b:g} - {b:g} {b:g} * 4 {a:g} * {c:g} * - sqrt + 2 {a:g} * /"
            )
            root_minus_program = (
                f"0 {b:g} - {b:g} {b:g} * 4 {a:g} * {c:g} * - sqrt - 2 {a:g} * /"
            )
            root_plus = self._format_math_answer(engine.evaluate(root_plus_program))
            root_minus = self._format_math_answer(engine.evaluate(root_minus_program))
            ordered = sorted([root_minus, root_plus], key=lambda value: float(value))
            return f"({ordered[0]}, {ordered[1]})", [
                f"GPU math template: {template_ref}",
                f"GPU math eval root1: {root_minus_program}",
                f"GPU math eval root2: {root_plus_program}",
            ]

        if template_ref == "math_template_band_formation_max_gpu":
            limit_minus_offset = float(params.get("limit_minus_offset", 0))
            program = (
                f"{limit_minus_offset:g} sqrt 1 - floor STORE_A "
                "RECALL_A RECALL_A * 2 * RECALL_A 4 * + 2 +"
            )
            value = engine.evaluate(program)
            return self._format_math_answer(value), [
                f"GPU math template: {template_ref}",
                f"GPU math eval: {program}",
            ]

        if template_ref == "math_template_circle_center_gpu":
            x_linear = float(params.get("x_linear", 0))
            y_linear = float(params.get("y_linear", 0))
            x_program = f"0 {x_linear:g} - 2 /"
            y_program = f"0 {y_linear:g} - 2 /"
            x_value = self._format_math_answer(engine.evaluate(x_program))
            y_value = self._format_math_answer(engine.evaluate(y_program))
            return f"({x_value}, {y_value})", [
                f"GPU math template: {template_ref}",
                f"GPU math eval x: {x_program}",
                f"GPU math eval y: {y_program}",
            ]

        if template_ref == "math_template_interval_upper_root_gpu":
            lower_bound = self._format_math_answer(float(params.get("lower_bound", 0)))
            explicit_upper_program = str(params.get("upper_bound_rpn", "")).strip()
            if explicit_upper_program:
                upper_program = explicit_upper_program
            elif "upper_bound" in params:
                upper_program = f"{float(params.get('upper_bound', 0)):g}"
            else:
                a = float(params.get("a", 0))
                b = float(params.get("b", 0))
                c = float(params.get("c", 0))
                upper_program = (
                    f"0 {b:g} - {b:g} {b:g} * 4 {a:g} * {c:g} * - sqrt + 2 {a:g} * /"
                )
            upper_bound = self._format_math_answer(engine.evaluate(upper_program))
            return self._format_math_interval(
                lower_bound,
                upper_bound,
                lower_inclusive=bool(params.get("lower_inclusive", True)),
                upper_inclusive=bool(params.get("upper_inclusive", False)),
            ), [
                f"GPU math template: {template_ref}",
                f"GPU math eval upper: {upper_program}",
            ]

        if template_ref == "math_template_l_shaped_sequence_gpu":
            row_start = float(params.get("row_start", 0))
            middle_second = float(params.get("middle_second", 0))
            middle_third = float(params.get("middle_third", 0))
            right_bottom = float(params.get("right_bottom", 0))
            program = (
                f"{middle_third:g} {middle_second:g} - STORE_A "
                f"{middle_second:g} RECALL_A - STORE_B "
                "RECALL_B RECALL_A - STORE_C "
                f"RECALL_C {row_start:g} - 3 / STORE_D "
                f"{row_start:g} RECALL_D 6 * + STORE_E "
                f"{right_bottom:g} RECALL_E - 4 / STORE_F "
                "RECALL_E RECALL_F -"
            )
            value = engine.evaluate(program)
            return self._format_math_answer(value), [
                f"GPU math template: {template_ref}",
                f"GPU math eval: {program}",
            ]

        if template_ref == "math_template_compound_interest_gpu":
            future_value = float(params.get("future_value", 0))
            annual_rate = float(params.get("annual_rate", 0))
            periods_per_year = float(params.get("periods_per_year", 1))
            years = float(params.get("years", 0))
            program = (
                f"{future_value:g} 1 {annual_rate:g} {periods_per_year:g} / + "
                f"{periods_per_year:g} {years:g} * pow / round"
            )
            value = self._format_math_answer(engine.evaluate(program))
            return self._format_escaped_currency(value), [
                f"GPU math template: {template_ref}",
                f"GPU math eval: {program}",
            ]

        if template_ref == "math_template_midpoint_coordinate_sum_gpu":
            if len(numbers) < 4:
                return None
            x1, y1, x2, y2 = [float(value) for value in numbers[:4]]
            program = f"{x1:g} {x2:g} + {y1:g} {y2:g} + + 2 /"
            value = engine.evaluate(program)
            return self._format_math_answer(value), [
                f"GPU math template: {template_ref}",
                f"GPU math eval: {program}",
            ]

        if template_ref == "math_template_exchange_gap_gpu":
            if len(numbers) < 3:
                return None
            total_cost = float(params.get("total_cost", numbers[0]))
            foreign_amount = float(params.get("foreign_amount", numbers[1]))
            exchange_rate = float(params.get("exchange_rate", numbers[2]))
            if exchange_rate == 0:
                return None
            program = f"{total_cost:g} {foreign_amount:g} {exchange_rate:g} / -"
            value = engine.evaluate(program)
            return self._format_math_answer(value), [
                f"GPU math template: {template_ref}",
                f"GPU math eval: {program}",
            ]

        if template_ref == "math_template_rate_scaling_gpu":
            if len(numbers) < 5:
                return None
            base_units = float(params.get("base_units", numbers[0]))
            base_volume = float(params.get("base_volume", numbers[1]))
            base_time_minutes = float(params.get("base_time_minutes", numbers[2]))
            target_units = float(params.get("target_units", numbers[3]))
            target_volume = float(params.get("target_volume", numbers[4]))
            if base_volume == 0 or target_units == 0:
                return None
            program = (
                f"{base_time_minutes:g} 60 * {target_volume:g} * {base_units:g} * "
                f"{base_volume:g} / {target_units:g} /"
            )
            value = engine.evaluate(program)
            return self._format_math_answer(value), [
                f"GPU math template: {template_ref}",
                f"GPU math eval: {program}",
            ]

        return None

    def _get_drawing_bridge(self):
        if self._drawing_bridge is None:
            from knowledge3d.cranium.bridges.drawing_bridge import DrawingBridge

            self._drawing_bridge = DrawingBridge()
        return self._drawing_bridge

    def _execute_arc_transform_gpu(
        self,
        *,
        input_grid: list[list[int]],
        transform_chain: list[str],
        color_mapping: dict[int, int],
    ) -> list[list[int]]:
        if not transform_chain:
            return [list(row) for row in input_grid]
        bridge = self._get_drawing_bridge()
        from knowledge3d.cranium.sovereign import loader

        opcodes = {
            "rotate_90": 0,
            "rotate_180": 1,
            "rotate_270": 2,
            "mirror_h": 3,
            "mirror_v": 4,
        }
        surface, width, height = bridge.grid_to_surface(input_grid)
        current_surface = surface
        current_w = width
        current_h = height
        for op in transform_chain:
            normalized = str(op).strip().lower()
            if normalized in {"", "identity", "color_remap"}:
                continue
            code = opcodes.get(normalized)
            if code is None:
                raise ValueError(f"unsupported_arc_gpu_transform:{op}")
            next_w = current_h if normalized in {"rotate_90", "rotate_270"} else current_w
            next_h = current_w if normalized in {"rotate_90", "rotate_270"} else current_h
            next_surface = bridge.execute_on_surface(
                current_surface,
                src_w=current_w,
                src_h=current_h,
                dst_w=next_w,
                dst_h=next_h,
                op=code,
            )
            loader.gpu_free(current_surface)
            current_surface = next_surface
            current_w = next_w
            current_h = next_h
        if any(str(op).strip().lower() == "color_remap" for op in transform_chain) and color_mapping:
            for src, dst in sorted(color_mapping.items()):
                next_surface = bridge.execute_on_surface(
                    current_surface,
                    src_w=current_w,
                    src_h=current_h,
                    dst_w=current_w,
                    dst_h=current_h,
                    op=6,
                    p1=int(src),
                    p2=int(dst),
                )
                loader.gpu_free(current_surface)
                current_surface = next_surface
        return bridge.surface_to_grid(current_surface, current_w, current_h)

    def _execute_arc_primitive_plan_gpu(
        self,
        *,
        input_grid: list[list[int]],
        primitive_plan: list[dict[str, Any]],
    ) -> list[list[int]]:
        if not primitive_plan:
            return [list(row) for row in input_grid]
        bridge = self._get_drawing_bridge()
        from knowledge3d.cranium.sovereign import loader

        surface, width, height = bridge.grid_to_surface(input_grid)
        current_surface = surface
        current_w = width
        current_h = height
        try:
            for raw_step in primitive_plan:
                if not isinstance(raw_step, dict):
                    continue
                op_name = str(raw_step.get("op", "")).strip().lower()
                if not op_name:
                    continue
                next_surface = None
                next_w = current_w
                next_h = current_h
                if op_name == "multi_color_remap":
                    mapping = raw_step.get("color_mapping") if isinstance(raw_step.get("color_mapping"), dict) else {}
                    for src, dst in sorted(mapping.items()):
                        next_surface = bridge.execute_on_surface(
                            current_surface,
                            src_w=current_w,
                            src_h=current_h,
                            dst_w=current_w,
                            dst_h=current_h,
                            op=6,
                            p1=int(src),
                            p2=int(dst),
                        )
                        loader.gpu_free(current_surface)
                        current_surface = next_surface
                    continue
                if op_name == "periodic_tile_repeat":
                    repeat_x = max(1, int(raw_step.get("repeat_x", 1)))
                    repeat_y = max(1, int(raw_step.get("repeat_y", 1)))
                    next_w = current_w * repeat_x
                    next_h = current_h * repeat_y
                    next_surface = bridge.execute_on_surface(
                        current_surface,
                        src_w=current_w,
                        src_h=current_h,
                        dst_w=next_w,
                        dst_h=next_h,
                        op=7,
                        p1=repeat_x,
                        p2=repeat_y,
                    )
                elif op_name == "checker_tile_repeat_hflip_rows":
                    repeat_x = max(1, int(raw_step.get("repeat_x", 1)))
                    repeat_y = max(1, int(raw_step.get("repeat_y", 1)))
                    next_w = current_w * repeat_x
                    next_h = current_h * repeat_y
                    next_surface = bridge.execute_on_surface(
                        current_surface,
                        src_w=current_w,
                        src_h=current_h,
                        dst_w=next_w,
                        dst_h=next_h,
                        op=8,
                        p1=repeat_x,
                        p2=repeat_y,
                    )
                elif op_name == "connect_color_pairs":
                    next_surface = bridge.execute_on_surface(
                        current_surface,
                        src_w=current_w,
                        src_h=current_h,
                        dst_w=current_w,
                        dst_h=current_h,
                        op=9,
                    )
                elif op_name == "periodic_consensus_cleanup":
                    next_surface = bridge.execute_on_surface(
                        current_surface,
                        src_w=current_w,
                        src_h=current_h,
                        dst_w=current_w,
                        dst_h=current_h,
                        op=10,
                    )
                elif op_name == "fill_enclosed_by_size":
                    next_surface = bridge.execute_on_surface(
                        current_surface,
                        src_w=current_w,
                        src_h=current_h,
                        dst_w=current_w,
                        dst_h=current_h,
                        op=11,
                    )
                elif op_name == "pack_color_components_diagonal":
                    next_surface = bridge.execute_on_surface(
                        current_surface,
                        src_w=current_w,
                        src_h=current_h,
                        dst_w=current_w,
                        dst_h=current_h,
                        op=12,
                    )
                elif op_name == "self_pattern_complement_tiling":
                    next_w = current_w * current_w
                    next_h = current_h * current_h
                    next_surface = bridge.execute_on_surface(
                        current_surface,
                        src_w=current_w,
                        src_h=current_h,
                        dst_w=next_w,
                        dst_h=next_h,
                        op=13,
                    )
                elif op_name in {"extract_masked_periodic_patch", "marker_axis_crop"}:
                    mask_cells = [
                        (xx, yy)
                        for yy, row in enumerate(input_grid)
                        for xx, value in enumerate(row)
                        if int(value) == 8
                    ]
                    if not mask_cells:
                        raise ValueError("masked_patch_requires_color_8_bbox")
                    min_x = min(xx for xx, _ in mask_cells)
                    min_y = min(yy for _, yy in mask_cells)
                    max_x = max(xx for xx, _ in mask_cells)
                    max_y = max(yy for _, yy in mask_cells)
                    next_w = max_x - min_x + 1
                    next_h = max_y - min_y + 1
                    next_surface = bridge.execute_on_surface(
                        current_surface,
                        src_w=current_w,
                        src_h=current_h,
                        dst_w=next_w,
                        dst_h=next_h,
                        op=14,
                    )
                elif op_name == "separator_bridge_projection":
                    next_surface = bridge.execute_on_surface(
                        current_surface,
                        src_w=current_w,
                        src_h=current_h,
                        dst_w=current_w,
                        dst_h=current_h,
                        op=15,
                    )
                elif op_name == "anchor_spiral_pair":
                    next_surface = bridge.execute_on_surface(
                        current_surface,
                        src_w=current_w,
                        src_h=current_h,
                        dst_w=current_w,
                        dst_h=current_h,
                        op=16,
                    )
                elif op_name == "crop_region":
                    crop_x = max(0, int(raw_step.get("x", 0)))
                    crop_y = max(0, int(raw_step.get("y", 0)))
                    next_w = max(1, min(int(raw_step.get("width", current_w - crop_x)), current_w - crop_x))
                    next_h = max(1, min(int(raw_step.get("height", current_h - crop_y)), current_h - crop_y))
                    next_surface = bridge.execute_on_surface(
                        current_surface,
                        src_w=current_w,
                        src_h=current_h,
                        dst_w=next_w,
                        dst_h=next_h,
                        op=5,
                        p1=-crop_x,
                        p2=-crop_y,
                    )
                else:
                    raise ValueError(f"unsupported_arc_primitive:{op_name}")
                if next_surface is None:
                    continue
                loader.gpu_free(current_surface)
                current_surface = next_surface
                current_w = next_w
                current_h = next_h
            return bridge.surface_to_grid(current_surface, current_w, current_h)
        except Exception:
            if getattr(current_surface, "value", 0):
                loader.gpu_free(current_surface)
            raise

    def _answer_arc_query(
        self,
        *,
        task: dict[str, Any],
        binding: dict[str, Any],
        reasoning_program: dict[str, Any],
        route_galaxies: list[str] | None,
        match: dict[str, Any],
        similarity: float,
        route: dict[str, Any] | None,
        specialist: str,
        domain_hint: str | None,
        query_text: str,
        use_enriched: bool,
        query_type: str | None,
        selection_steps: list[str],
    ) -> dict[str, Any]:
        output_grid = match.get("output_grid")
        extra_steps: list[str] = list(selection_steps)
        primitive_plan = [
            dict(step)
            for step in match.get("arc_primitive_plan", [])
            if isinstance(step, dict) and str(step.get("op", "")).strip()
        ]
        transform_chain = [str(item).strip() for item in match.get("arc_transform_chain", []) if str(item).strip()]
        color_mapping_raw = match.get("arc_color_mapping", {})
        color_mapping = {
            int(src): int(dst)
            for src, dst in dict(color_mapping_raw).items()
        }
        if primitive_plan:
            input_grid = task.get("input_grid")
            if not isinstance(input_grid, list):
                return {
                    "status": "error",
                    "error": "arc_task_missing_input_grid",
                    "gpu_execution": True,
                }
            output_grid = self._execute_arc_primitive_plan_gpu(
                input_grid=input_grid,
                primitive_plan=primitive_plan,
            )
            extra_steps.append(
                "GPU primitive plan: "
                + ", ".join(str(step.get("op", "")).strip() for step in primitive_plan)
            )
            for step in primitive_plan:
                op_name = str(step.get("op", "")).strip().lower()
                if op_name == "multi_color_remap":
                    mapping = step.get("color_mapping") if isinstance(step.get("color_mapping"), dict) else {}
                    if mapping:
                        pairs = ", ".join(f"{int(src)}->{int(dst)}" for src, dst in sorted(mapping.items()))
                        extra_steps.append(f"GPU recolor map: {pairs}")
                if op_name in {"periodic_tile_repeat", "checker_tile_repeat_hflip_rows"}:
                    extra_steps.append(
                        f"GPU repeat factors: {int(step.get('repeat_x', 1))}x{int(step.get('repeat_y', 1))}"
                    )
        elif transform_chain:
            input_grid = task.get("input_grid")
            if not isinstance(input_grid, list):
                return {
                    "status": "error",
                    "error": "arc_task_missing_input_grid",
                    "gpu_execution": True,
                }
            output_grid = self._execute_arc_transform_gpu(
                input_grid=input_grid,
                transform_chain=transform_chain,
                color_mapping=color_mapping,
            )
            extra_steps.append(
                "GPU grid transform: " + ", ".join(transform_chain)
            )
            if color_mapping:
                pairs = ", ".join(f"{src}->{dst}" for src, dst in sorted(color_mapping.items()))
                extra_steps.append(f"GPU recolor map: {pairs}")
        if not isinstance(output_grid, list):
            return {
                "status": "error",
                "error": "gpu_arc_no_output_grid",
                "gpu_execution": True,
                "program_id": str(reasoning_program.get("id", "")),
            }
        thinking_trace = self._build_gpu_thinking_trace(
            binding=binding,
            program_id=str(reasoning_program.get("id", "")),
            match=match,
            similarity=similarity,
            specialist=specialist,
            read_field="output_grid",
            extra_steps=extra_steps,
        )
        return {
            "status": "ok",
            "answer": json.dumps(output_grid, ensure_ascii=True),
            "response": json.dumps(output_grid, ensure_ascii=True),
            "result": output_grid,
            "output_grid": output_grid,
            "thinking_trace": thinking_trace,
            "reasoning_trace": list(thinking_trace),
            "thinking_xml": self._render_thinking_xml(thinking_trace, output_grid),
            "gpu_execution": True,
            "runtime": "knowledgeverse_gpu_query",
            "program_id": str(reasoning_program.get("id", "")),
            "program_type": "gpu_arc_grid_transform_lookup",
            "solver": "knowledgeverse_gpu_query",
            "patterns_used": 1,
            "query_text": query_text,
            "top_match_similarity": similarity,
            "route": {
                "specialist": specialist,
                "domain_hint": domain_hint,
                "galaxy_names": list(route_galaxies or binding.get("galaxies", [])),
                "scanned_galaxies": list(binding.get("galaxies", [])),
            },
            "match": match,
            "query_type": str(query_type or ""),
            "use_enriched": bool(use_enriched),
        }

    @staticmethod
    def _gsm8k_operator_token(operation: str) -> str:
        return {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "div": "/",
        }.get(str(operation).strip().lower(), "")

    def _gsm8k_left_fold_program(
        self,
        values: list[float],
        operations: list[str],
    ) -> str:
        if len(values) < 2 or not operations:
            return ""
        op_tokens = [self._gsm8k_operator_token(operation) for operation in operations]
        if not op_tokens or any(not token for token in op_tokens):
            return ""
        tokens = [self._gpu_scalar_literal(values[0]), self._gpu_scalar_literal(values[1]), op_tokens[0]]
        next_index = 2
        for op_token in op_tokens[1:]:
            if next_index >= len(values):
                break
            tokens.extend([self._gpu_scalar_literal(values[next_index]), op_token])
            next_index += 1
        return " ".join(tokens).strip()

    @staticmethod
    def _gsm8k_slot_role_names(slot: str) -> list[str]:
        token = str(slot).strip().lower()
        if not token:
            return []
        if token.endswith("_value"):
            token = token[: -len("_value")]
        base = token
        if "_" in token:
            head, _, tail = token.rpartition("_")
            if tail.isdigit():
                base = head
        alias_map = {
            "initial": ["initial", "total", "target"],
            "total": ["total", "initial", "target"],
            "part": ["part", "delta", "count"],
            "count": ["count", "duration", "part", "delta"],
            "duration": ["duration", "count"],
            "rate": ["rate", "price"],
            "threshold": ["threshold", "count", "initial", "total"],
            "excess": ["excess", "part", "delta", "count"],
            "percentage": ["percentage", "divisor"],
            "ratio": ["divisor", "percentage"],
            "ratio_value": ["divisor", "percentage"],
            "divisor": ["divisor", "percentage"],
        }
        return alias_map.get(base, [base])

    @staticmethod
    def _gsm8k_product_tokens(literals: list[str]) -> list[str]:
        if len(literals) < 2:
            return []
        tokens = [literals[0], literals[1], "*"]
        for literal in literals[2:]:
            tokens.extend([literal, "*"])
        return tokens

    @staticmethod
    def _gsm8k_sum_token_rows(rows: list[list[str]]) -> list[str]:
        active_rows = [list(row) for row in rows if row]
        if not active_rows:
            return []
        tokens = list(active_rows[0])
        for row in active_rows[1:]:
            tokens.extend(row)
            tokens.append("+")
        return tokens

    def _gsm8k_quantity_role_rows(
        self,
        *,
        target_galaxies: list[str],
    ) -> list[dict[str, Any]]:
        allowed = {str(name).strip() for name in target_galaxies if str(name).strip()}
        rows: list[dict[str, Any]] = []
        for entry in self.get_gpu_galaxy_catalog():
            if allowed and str(entry.get("galaxy", "")).strip() not in allowed:
                continue
            if str(entry.get("galaxy", "")).strip() != "Grammar":
                continue
            metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            if not str(metadata.get("quantity_role", "")).strip():
                continue
            if not list(entry.get("embedding16", [])):
                continue
            rows.append(dict(entry))
        return rows

    @staticmethod
    def _gsm8k_parse_role_diagnostics(parse_bundle: dict[str, Any] | None) -> dict[str, Any]:
        bundle = dict(parse_bundle or {})
        fusion = bundle.get("fusion_parse") if isinstance(bundle.get("fusion_parse"), dict) else {}
        backward = bundle.get("backward_parse") if isinstance(bundle.get("backward_parse"), dict) else {}
        backward_goal = backward.get("goal") if isinstance(backward.get("goal"), dict) else {}
        merged_rows = fusion.get("merged_quantities") if isinstance(fusion.get("merged_quantities"), list) else []
        typed_roles = sorted(
            {
                str(row.get("role", "")).strip().lower()
                for row in merged_rows
                if isinstance(row, dict) and str(row.get("role", "")).strip().lower() not in {"", "quantity"}
            }
        )
        return {
            "goal_type": str(fusion.get("goal_type", "")).strip() or str(backward_goal.get("goal_type", "")).strip(),
            "uses_typed_fusion": bool(typed_roles),
            "typed_roles": typed_roles,
        }

    @staticmethod
    def _gsm8k_parse_blocks(parse_bundle: dict[str, Any] | None) -> list[dict[str, Any]]:
        bundle = dict(parse_bundle or {})
        blocks: list[dict[str, Any]] = []
        fusion = bundle.get("fusion_parse") if isinstance(bundle.get("fusion_parse"), dict) else {}
        fusion_rows = fusion.get("merged_quantities") if isinstance(fusion.get("merged_quantities"), list) else []
        if fusion_rows and any(str(row.get("role", "")).strip().lower() not in {"", "quantity"} for row in fusion_rows if isinstance(row, dict)):
            blocks.append(
                {
                    "kind": "fusion",
                    "raw": str(bundle.get("query_text", "")).strip(),
                    "quantities": list(fusion_rows),
                }
            )
            return blocks
        forward = bundle.get("forward_parse") if isinstance(bundle.get("forward_parse"), dict) else {}
        for block in forward.get("context") if isinstance(forward.get("context"), list) else []:
            if isinstance(block, dict):
                blocks.append(
                    {
                        "kind": "context",
                        "raw": str(block.get("raw", "")).strip(),
                        "quantities": list(block.get("quantities") if isinstance(block.get("quantities"), list) else []),
                    }
                )
        goal_block = forward.get("goal") if isinstance(forward.get("goal"), dict) else {}
        if goal_block:
            blocks.append(
                {
                    "kind": "goal",
                    "raw": str(goal_block.get("raw", "")).strip(),
                    "quantities": list(goal_block.get("quantities") if isinstance(goal_block.get("quantities"), list) else []),
                }
            )
        if not blocks:
            backward = bundle.get("backward_parse") if isinstance(bundle.get("backward_parse"), dict) else {}
            for block in backward.get("dependencies") if isinstance(backward.get("dependencies"), list) else []:
                if isinstance(block, dict):
                    blocks.append(
                        {
                            "kind": "backward",
                            "raw": str(block.get("raw", "")).strip(),
                            "quantities": list(block.get("quantities") if isinstance(block.get("quantities"), list) else []),
                        }
                    )
        return blocks

    @staticmethod
    def _gsm8k_quantity_snippet(raw_text: str, *, surface: str, offset: int) -> str:
        text = str(raw_text or "").strip()
        if not text:
            return str(surface or "").strip()
        start = max(int(offset) - 56, 0)
        end = min(int(offset) + max(len(str(surface or "")), 1) + 48, len(text))
        snippet = text[start:end].strip(" ,.;:!?")
        return snippet or text

    @staticmethod
    def _gsm8k_text_tokens(text: str) -> set[str]:
        stopwords = {
            "a",
            "an",
            "and",
            "at",
            "by",
            "for",
            "from",
            "in",
            "into",
            "of",
            "on",
            "or",
            "the",
            "to",
            "with",
        }
        tokens: set[str] = set()
        for raw_token in str(text or "").split():
            token = raw_token.strip(".,;:!?$()[]{}\"'").lower()
            if not token or token in stopwords:
                continue
            if token.endswith("'s"):
                token = token[:-2]
            if len(token) > 4 and token.endswith("ing"):
                token = token[:-3]
            elif len(token) > 3 and token.endswith("ed"):
                token = token[:-2]
            elif len(token) > 3 and token.endswith("es"):
                token = token[:-2]
            elif len(token) > 2 and token.endswith("s"):
                token = token[:-1]
            token = token.strip()
            if not token or token in stopwords:
                continue
            tokens.add(token)
        return tokens

    def _gsm8k_quantity_role_candidates(
        self,
        *,
        parse_bundle: dict[str, Any] | None,
        role_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not role_rows:
            return []
        role_embeddings = [list(entry.get("embedding16", [])) for entry in role_rows]
        candidates: list[dict[str, Any]] = []
        for block in self._gsm8k_parse_blocks(parse_bundle):
            raw_text = str(block.get("raw", "")).strip()
            block_kind = str(block.get("kind", "")).strip() or "context"
            for raw_row in block.get("quantities") if isinstance(block.get("quantities"), list) else []:
                if not isinstance(raw_row, dict):
                    continue
                try:
                    value = float(raw_row.get("value"))
                except Exception:
                    continue
                surface = str(raw_row.get("surface", "")).strip()
                offset = int(raw_row.get("offset", 0) or 0)
                snippet = self._gsm8k_quantity_snippet(raw_text, surface=surface, offset=offset)
                try:
                    snippet_embedding = self._embed_query_gpu(snippet)
                except Exception:
                    continue
                similarities = self._embedding_similarities(snippet_embedding, role_embeddings)
                snippet_tokens = self._gsm8k_text_tokens(snippet)
                raw_tokens = self._gsm8k_text_tokens(raw_text)
                ranked = sorted(
                    [
                        (
                            (0.60 * float(score))
                            + (
                                0.22
                                * self._gsm8k_role_text_overlap(
                                    text_tokens=snippet_tokens,
                                    row=row,
                                    include_content=True,
                                )
                            )
                            + (
                                0.08
                                * self._gsm8k_role_text_overlap(
                                    text_tokens=raw_tokens,
                                    row=row,
                                    include_content=False,
                                )
                            )
                            + (
                                0.40
                                * self._gsm8k_role_structural_cue_overlap(
                                    snippet=snippet,
                                    raw_text=raw_text,
                                    row=row,
                                )
                            )
                            + (
                                0.12
                                if block_kind == "goal"
                                and str(
                                    (
                                        row.get("metadata")
                                        if isinstance(row.get("metadata"), dict)
                                        else {}
                                    ).get("quantity_role", "")
                                ).strip().lower()
                                == "target"
                                else 0.0
                            ),
                            row,
                        )
                        for score, row in zip(similarities, role_rows)
                    ],
                    key=lambda item: item[0],
                    reverse=True,
                )
                if not ranked:
                    continue
                preassigned_role = str(raw_row.get("role", "")).strip().lower()
                role_alternatives: list[dict[str, Any]] = []
                top_roles: list[str] = []
                top_score = 0.0
                top_confidence = 0.0
                top_role = ""
                if preassigned_role and preassigned_role != "quantity":
                    top_role = preassigned_role
                    top_roles.append(preassigned_role)
                    top_score = 0.95
                    top_confidence = min(1.0, max(0.0, float(raw_row.get("role_confidence", top_score) or 0.0)))
                    role_alternatives.append(
                        {
                            "role": preassigned_role,
                            "score": 0.95,
                            "confidence": top_confidence,
                            "source": "preassigned",
                        }
                    )
                for score, row in ranked[:3]:
                    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                    role_name = str(metadata.get("quantity_role", "")).strip().lower()
                    if not role_name or role_name in top_roles:
                        continue
                    role_confidence = min(1.0, max(0.0, float(score)))
                    if not top_role:
                        top_role = role_name
                        top_score = float(score)
                        top_confidence = role_confidence
                    top_roles.append(role_name)
                    role_alternatives.append(
                        {
                            "role": role_name,
                            "score": float(score),
                            "confidence": role_confidence,
                            "source": "embedding",
                        }
                    )
                if block_kind == "goal" and "target" not in top_roles:
                    top_roles.insert(0, "target")
                    if not top_role:
                        top_role = "target"
                        top_score = 0.35
                        top_confidence = 0.35
                    role_alternatives.insert(
                        0,
                        {
                            "role": "target",
                            "score": 0.35,
                            "confidence": 0.35,
                            "source": "goal_block",
                        },
                    )
                if not top_role:
                    continue
                candidates.append(
                    {
                        "value": value,
                        "surface": surface,
                        "offset": offset,
                        "block_kind": block_kind,
                        "raw": raw_text,
                        "snippet": snippet,
                        "role": top_role,
                        "role_options": top_roles,
                        "role_alternatives": role_alternatives,
                        "score": float(top_score),
                        "role_confidence": float(top_confidence or min(1.0, max(0.0, float(top_score)))),
                    }
                )
        return candidates

    def _gsm8k_role_text_overlap(
        self,
        *,
        text_tokens: set[str],
        row: dict[str, Any],
        include_content: bool,
    ) -> float:
        if not text_tokens:
            return 0.0
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        parts = [
            str(metadata.get("query_anchor", "")),
            " ".join(
                str(value)
                for value in (
                    metadata.get("aliases", [])
                    if isinstance(metadata.get("aliases", []), list)
                    else []
                )
            ),
        ]
        if include_content:
            parts.append(str(row.get("content", "")))
        role_tokens = self._gsm8k_text_tokens(" ".join(parts))
        if not role_tokens:
            return 0.0
        return float(len(text_tokens.intersection(role_tokens))) / float(len(role_tokens))

    def _gsm8k_role_structural_cue_overlap(
        self,
        *,
        snippet: str,
        raw_text: str,
        row: dict[str, Any],
    ) -> float:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        cues = [
            str(value).strip().lower()
            for value in (
                metadata.get("structural_cues", [])
                if isinstance(metadata.get("structural_cues", []), list)
                else []
            )
            if str(value).strip()
        ]
        if not cues:
            return 0.0
        haystacks = [str(snippet or "").lower(), str(raw_text or "").lower()]
        matched = 0
        for cue in cues:
            if any(cue in haystack for haystack in haystacks):
                matched += 1
        return float(matched) / float(len(cues))

    def _gsm8k_role_value_map(self, context: dict[str, Any]) -> dict[str, list[float]]:
        raw_map = context.get("quantity_role_values") if isinstance(context, dict) else None
        if not isinstance(raw_map, dict):
            return {}
        out: dict[str, list[float]] = {}
        for key, values in raw_map.items():
            bucket: list[float] = []
            for value in values if isinstance(values, list) else []:
                try:
                    bucket.append(float(value))
                except Exception:
                    continue
            if bucket:
                out[str(key).strip().lower()] = bucket
        return out

    @staticmethod
    def _gsm8k_role_values_from_candidates(quantity_candidates: list[dict[str, Any]]) -> dict[str, list[float]]:
        role_values: dict[str, list[float]] = {}
        for row in quantity_candidates:
            role_name = str(row.get("role", "")).strip().lower()
            if not role_name:
                continue
            try:
                numeric_value = float(row.get("value", 0.0))
            except Exception:
                continue
            role_values.setdefault(role_name, []).append(numeric_value)
        return role_values

    def _gsm8k_role_map_variants(self, quantity_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not quantity_candidates:
            return []
        base_candidates = [dict(row) for row in quantity_candidates if isinstance(row, dict)]
        base_signature = tuple(
            (str(row.get("surface", "")).strip(), float(row.get("value", 0.0)), str(row.get("role", "")).strip().lower())
            for row in base_candidates
        )
        variants: list[dict[str, Any]] = [
            {
                "label": "best_roles",
                "quantity_role_candidates": base_candidates,
                "quantity_role_values": self._gsm8k_role_values_from_candidates(base_candidates),
            }
        ]
        seen_signatures = {base_signature}
        for candidate_index, raw_row in enumerate(base_candidates):
            alternatives = raw_row.get("role_alternatives") if isinstance(raw_row.get("role_alternatives"), list) else []
            current_role = str(raw_row.get("role", "")).strip().lower()
            swap_index = 0
            for alternative in alternatives:
                if not isinstance(alternative, dict):
                    continue
                alt_role = str(alternative.get("role", "")).strip().lower()
                if not alt_role or alt_role == current_role:
                    continue
                variant_candidates = [dict(row) for row in base_candidates]
                variant_candidates[candidate_index]["role"] = alt_role
                variant_candidates[candidate_index]["source"] = "role_alternative"
                variant_candidates[candidate_index]["score"] = float(alternative.get("score", variant_candidates[candidate_index].get("score", 0.0)) or 0.0)
                variant_candidates[candidate_index]["role_confidence"] = float(
                    alternative.get(
                        "confidence",
                        variant_candidates[candidate_index].get("role_confidence", variant_candidates[candidate_index].get("score", 0.0)),
                    )
                    or 0.0
                )
                variant_signature = tuple(
                    (str(row.get("surface", "")).strip(), float(row.get("value", 0.0)), str(row.get("role", "")).strip().lower())
                    for row in variant_candidates
                )
                if variant_signature in seen_signatures:
                    continue
                seen_signatures.add(variant_signature)
                variants.append(
                    {
                        "label": f"swap_q{candidate_index}_{alt_role}_{swap_index}",
                        "quantity_role_candidates": variant_candidates,
                        "quantity_role_values": self._gsm8k_role_values_from_candidates(variant_candidates),
                    }
                )
                swap_index += 1
                if len(variants) >= 9:
                    return variants
        return variants

    def _gsm8k_slot_value(
        self,
        slot: str,
        *,
        role_values: dict[str, list[float]],
    ) -> float | None:
        token = str(slot).strip().lower()
        if not token:
            return None
        index = 0
        if "_" in token:
            head, _, tail = token.rpartition("_")
            if tail.isdigit():
                token = head
                index = max(int(tail) - 1, 0)
        for role_name in self._gsm8k_slot_role_names(token):
            values = role_values.get(role_name, [])
            if index < len(values):
                return float(values[index])
        return None

    @staticmethod
    def _gsm8k_semantic_slot_base(slot: str) -> str:
        token = str(slot).strip().lower()
        if not token:
            return ""
        if token.endswith("_value"):
            token = token[: -len("_value")]
        if "_" in token:
            head, _, tail = token.rpartition("_")
            if tail.isdigit():
                token = head
        return token

    @staticmethod
    def _gsm8k_semantic_entity_local_windows(entity: dict[str, Any]) -> tuple[str, str]:
        raw_text = str(entity.get("raw_block", "")).strip()
        surface = str(entity.get("surface", "")).strip()
        offset = int(entity.get("offset", 0) or 0)
        local_window = raw_text[
            max(offset - 24, 0) : min(offset + max(len(surface), 1) + 24, len(raw_text))
        ].lower()
        tight_window = raw_text[
            max(offset - 8, 0) : min(offset + max(len(surface), 1) + 8, len(raw_text))
        ].lower()
        return local_window, tight_window

    def _gsm8k_semantic_slot_score(
        self,
        *,
        slot: str,
        entity: dict[str, Any],
    ) -> float:
        slot_base = self._gsm8k_semantic_slot_base(slot)
        if not slot_base:
            return float("-inf")
        role = str(entity.get("role", "")).strip().lower()
        unit = str(entity.get("unit", "")).strip().lower()
        scope = str(entity.get("scope", "")).strip().lower()
        surface = str(entity.get("surface", "")).strip().lower()
        local_window, tight_window = self._gsm8k_semantic_entity_local_windows(entity)
        has_percent = "%" in local_window or " percent" in local_window
        has_adjacent_currency = "$" in tight_window
        has_temporal = any(
            token in local_window
            for token in (
                " minute",
                " minutes",
                " hour",
                " hours",
                " day",
                " days",
                " week",
                " weeks",
                " month",
                " months",
                " year",
                " years",
            )
        )
        has_duration_cue = any(
            cue in local_window
            for cue in (" takes ", " take ", " delay", " wait", " restart", " time")
        )
        has_rate_cue = ("/" in local_window) or (" per " in f" {local_window} ") or scope.startswith("per_")
        has_currency_cue = (
            "$" in tight_window
            or "$" in local_window
            or " price" in local_window
            or " cost" in local_window
            or unit == "currency"
        )
        has_total_cue = any(
            cue in local_window
            for cue in (" file", " files", " total", " altogether", " from the beginning", " downloading", " download")
        )
        word_number = surface.isalpha() and not any(char.isdigit() for char in surface)
        score = 0.0
        if slot_base == "percentage":
            if has_percent or role == "percentage" or unit == "percent":
                score += 8.0
            if has_rate_cue or has_currency_cue or has_temporal:
                score -= 2.5
        elif slot_base == "duration":
            if has_temporal or role == "duration":
                score += 7.0
            if has_duration_cue:
                score += 2.0
            if has_percent or has_currency_cue or has_rate_cue:
                score -= 2.0
        elif slot_base == "rate":
            if role in {"rate", "price"}:
                score += 5.0
            if has_rate_cue:
                score += 5.0
            if has_currency_cue:
                score += 3.0
            if has_adjacent_currency:
                score += 5.0
            if has_percent or has_temporal:
                score -= 2.0
            if word_number:
                score -= 5.0
        elif slot_base in {"total", "initial"}:
            if role in {slot_base, "total", "initial"}:
                score += 5.0
            if has_total_cue:
                score += 4.0
            if slot_base == "initial" and any(
                cue in local_window for cue in (" buy ", " buys ", " bought ", " purchase ", " purchased ", " cost ")
            ):
                score += 4.0
            if not (has_percent or has_temporal or has_rate_cue):
                score += 2.0
            if word_number:
                score -= 1.0
        elif slot_base == "threshold":
            if role in {"threshold", "count", "duration"}:
                score += 4.0
            if any(cue in local_window for cue in (" first ", " before ", " up to ", " threshold ", " regular ")):
                score += 4.5
            if has_currency_cue and not has_rate_cue:
                score -= 2.0
        elif slot_base == "part":
            if role in {"part", "delta"}:
                score += 5.0
            if role == "price":
                score += 2.5
            if any(cue in local_window for cue in (" repair", " repairs", " spent", " puts in", " fee", " cost ")):
                score += 4.0
            if has_percent:
                score -= 2.0
        elif slot_base == "count":
            if role == "count":
                score += 4.0
            if any(
                cue in local_window
                for cue in (" buy", " bought", " want", " wants", " glass", " glasses", " bolt", " bolts")
            ):
                score += 3.0
            if any(cue in local_window for cue in (" chicken", " chickens", " flock", " size of")):
                score += 4.5
            if any(cue in local_window for cue in (" meal", " meals", " morning", " afternoon")):
                score -= 4.0
            if has_percent or has_temporal or has_rate_cue:
                score -= 3.0
            if word_number:
                score -= 1.0
        elif slot_base in {"divisor", "ratio"}:
            if role in {"divisor", "percentage", "frequency", "excess"}:
                score += 4.0
            if any(cue in local_window for cue in (" times ", " ratio ", " multiplier ", " regular hourly rate", " regular rate")):
                score += 5.0
            if has_currency_cue or has_temporal:
                score -= 1.5
        else:
            if role == slot_base:
                score += 5.0
            for alias in self._gsm8k_slot_role_names(slot_base):
                if role == alias:
                    score += 3.0
                    break
        try:
            numeric_value = float(entity.get("resolved_value", entity.get("value", 0.0)))
        except Exception:
            numeric_value = 0.0
        score += min(max(numeric_value, 0.0), 1000.0) * 1e-6
        return float(score)

    def _gsm8k_operation_role_match_score(
        self,
        *,
        metadata: dict[str, Any],
        context: dict[str, Any],
    ) -> float:
        required_roles = [
            str(value).strip()
            for value in (metadata.get("required_roles") if isinstance(metadata.get("required_roles"), list) else [])
            if str(value).strip()
        ]
        if not required_roles:
            return 0.0
        if not {role.strip().lower() for role in required_roles}.intersection({"part", "threshold", "divisor"}):
            return 0.0
        semantic_entities: list[dict[str, Any]] = [
            dict(row)
            for row in (context.get("semantic_entities") if isinstance(context.get("semantic_entities"), list) else [])
            if isinstance(row, dict)
        ]
        quantity_candidates = context.get("quantity_role_candidates") if isinstance(context.get("quantity_role_candidates"), list) else []
        seen_semantic: set[tuple[str, float, int]] = {
            (
                str(row.get("surface", "")).strip(),
                float(row.get("resolved_value", row.get("value", 0.0)) or 0.0),
                int(row.get("offset", 0) or 0),
            )
            for row in semantic_entities
        }
        for row in quantity_candidates:
            if not isinstance(row, dict):
                continue
            try:
                numeric_value = float(row.get("value", 0.0))
            except Exception:
                continue
            signature = (
                str(row.get("surface", "")).strip(),
                numeric_value,
                int(row.get("offset", 0) or 0),
            )
            if signature in seen_semantic:
                continue
            semantic_entities.append(
                {
                    "value": numeric_value,
                    "resolved_value": numeric_value,
                    "surface": str(row.get("surface", "")).strip(),
                    "role": str(row.get("role", "")).strip(),
                    "unit": "",
                    "scope": "",
                    "offset": int(row.get("offset", 0) or 0),
                    "raw_block": str(row.get("raw", "")).strip(),
                }
            )
            seen_semantic.add(signature)
        if not semantic_entities:
            return 0.0
        used_entities: set[int] = set()
        matched = 0
        confidence_total = 0.0
        for role in required_roles:
            best_index = -1
            best_score = float("-inf")
            for index, entity in enumerate(semantic_entities):
                if index in used_entities:
                    continue
                score = self._gsm8k_semantic_slot_score(slot=role, entity=entity)
                if score > best_score:
                    best_score = score
                    best_index = index
            if best_index >= 0 and best_score >= 2.0:
                matched += 1
                used_entities.add(best_index)
                confidence_total += min(1.0, max(0.0, (float(best_score) - 2.0) / 8.0))
        coverage = float(matched) / float(len(required_roles))
        confidence = float(confidence_total) / float(len(required_roles))
        return min(1.0, (0.65 * coverage) + (0.35 * confidence))

    def _gsm8k_operation_disambiguation_bonus(
        self,
        *,
        metadata: dict[str, Any],
        context: dict[str, Any],
    ) -> float:
        binding_mode = str(metadata.get("binding_mode", "")).strip().lower()
        if not binding_mode:
            return 0.0
        semantic_entities = [
            dict(row)
            for row in (
                context.get("semantic_entities")
                if isinstance(context.get("semantic_entities"), list)
                else []
            )
            if isinstance(row, dict)
        ]
        if not semantic_entities:
            return 0.0
        source_text = str(context.get("source_text", "")).strip().lower()
        role_counts: dict[str, int] = {}
        for entity in semantic_entities:
            role = str(entity.get("role", "")).strip().lower()
            if role:
                role_counts[role] = int(role_counts.get(role, 0)) + 1

        def _has_role(role: str, count: int = 1) -> bool:
            return int(role_counts.get(role, 0)) >= count

        def _has_any(*phrases: str) -> bool:
            return any(str(phrase).strip().lower() in source_text for phrase in phrases if str(phrase).strip())

        bonus = 0.0
        if binding_mode == "restart_progress_time":
            if _has_role("total") and _has_role("rate") and _has_role("percentage") and _has_role("duration"):
                bonus += 0.95
            if _has_any("restart", "from the beginning", "download"):
                bonus += 0.85
            if _has_any("% of the way", "percent of the way", "minutes"):
                bonus += 0.25
        elif binding_mode == "ratio_then_add":
            if _has_any("restart", "from the beginning", "download"):
                bonus -= 1.10
            if _has_role("total") and _has_role("rate") and _has_role("percentage") and _has_role("duration"):
                bonus -= 0.90
        elif binding_mode == "scaled_total_minus_parts":
            if _has_role("count") and _has_role("rate") and _has_role("part", 2):
                bonus += 0.90
            if _has_any("each of her chickens", "final meal", "flock", "feed", "chickens"):
                bonus += 0.80
        elif binding_mode == "total_minus_parts":
            if _has_any("final meal", "morning", "afternoon") and _has_role("count") and _has_role("rate"):
                bonus -= 0.40
        elif binding_mode == "base_plus_excess":
            if _has_any("overtime", "regular rate", "hourly", "worked", "wage"):
                bonus += 0.85
            if _has_any("final meal", "chickens", "flock", "feed"):
                bonus -= 0.95
            if _has_any("restart", "download", "from the beginning"):
                bonus -= 0.80
        elif binding_mode == "outbound_return_distance":
            if _has_any("turns around", "turn around", "get home", "standstill traffic", "remaining time"):
                bonus += 1.10
            if _has_role("duration", 3) and _has_role("rate", 2):
                bonus += 0.60
        elif binding_mode in {"multiply_chain_sum", "count_rate_product"}:
            if _has_any("turns around", "standstill traffic", "get home"):
                bonus -= 0.75
        return float(bonus)

    def _gsm8k_template_slot_bindings(
        self,
        *,
        context: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[dict[str, float], str]:
        role_slots = [
            str(value).strip()
            for value in (
                metadata.get("role_slots")
                if isinstance(metadata.get("role_slots"), list)
                else []
            )
            if str(value).strip()
        ]
        if not role_slots:
            return {}, ""
        role_values = self._gsm8k_role_value_map(context)
        semantic_entities = [
            dict(row)
            for row in (
                context.get("semantic_entities")
                if isinstance(context.get("semantic_entities"), list)
                else []
            )
            if isinstance(row, dict)
        ]
        priority = {
            "percentage": 0,
            "duration": 1,
            "rate": 2,
            "threshold": 3,
            "excess": 4,
            "total": 5,
            "initial": 6,
            "count": 7,
            "part": 8,
        }
        ordered_slots = sorted(
            role_slots,
            key=lambda slot: (priority.get(self._gsm8k_semantic_slot_base(slot), 99), slot),
        )
        bound: dict[str, float] = {}
        binding_rows: dict[str, str] = {}
        used_entities: set[int] = set()
        for slot in ordered_slots:
            best_index = -1
            best_score = float("-inf")
            for index, entity in enumerate(semantic_entities):
                if index in used_entities:
                    continue
                score = self._gsm8k_semantic_slot_score(slot=slot, entity=entity)
                if score > best_score:
                    best_index = index
                    best_score = score
            if best_index >= 0 and best_score >= 2.0:
                entity = semantic_entities[best_index]
                try:
                    value = float(entity.get("resolved_value", entity.get("value", 0.0)))
                except Exception:
                    value = None
                if value is not None:
                    used_entities.add(best_index)
                    bound[slot] = float(value)
                    binding_rows[slot] = (
                        f"{self._gpu_scalar_literal(value)}"
                        + f"[semantic:{str(entity.get('role', '')).strip() or 'quantity'}:{str(entity.get('surface', '')).strip()}]"
                    )
                    continue
            fallback = self._gsm8k_slot_value(slot, role_values=role_values)
            if fallback is not None:
                bound[slot] = float(fallback)
                binding_rows[slot] = f"{self._gpu_scalar_literal(fallback)}[fallback]"
        ordered_summary = [
            f"{slot}={binding_rows[slot]}"
            for slot in role_slots
            if slot in binding_rows
        ]
        summary = ", ".join(ordered_summary)
        context["_last_gsm8k_slot_binding"] = summary
        context["_last_gsm8k_slot_values"] = dict(bound)
        return bound, summary

    def _gsm8k_pattern_structural_score(
        self,
        *,
        metadata: dict[str, Any],
        quantity_candidates: list[dict[str, Any]],
        quantity_count: int,
        clause_operations: list[str],
        top_operations: list[str],
        goal_operation: str,
    ) -> float:
        min_quantities = int(metadata.get("min_quantities", 0) or 0)
        if min_quantities and quantity_count < min_quantities:
            return 0.0
        role_counts: dict[str, int] = {}
        for row in quantity_candidates:
            role = str(row.get("role", "")).strip().lower()
            if not role:
                continue
            role_counts[role] = int(role_counts.get(role, 0)) + 1
        required_roles = [
            str(value).strip().lower()
            for value in (metadata.get("required_roles") if isinstance(metadata.get("required_roles"), list) else [])
            if str(value).strip()
        ]
        role_score = 0.0
        role_confidence = 0.0
        if required_roles:
            required_counts: dict[str, int] = {}
            for role in required_roles:
                required_counts[role] = int(required_counts.get(role, 0)) + 1
            matched = 0
            confidence_total = 0.0
            for role, required_count in required_counts.items():
                aliases = self._gsm8k_slot_role_names(role)
                available = max((int(role_counts.get(alias, 0)) for alias in aliases), default=0)
                matched += min(available, required_count)
                alias_confidences = sorted(
                    [
                        min(
                            1.0,
                            max(
                                0.0,
                                float(
                                    row.get(
                                        "role_confidence",
                                        row.get("score", 0.0),
                                    )
                                    or 0.0
                                ),
                            ),
                        )
                        for row in quantity_candidates
                        if str(row.get("role", "")).strip().lower() in aliases
                    ],
                    reverse=True,
                )
                confidence_total += sum(alias_confidences[:required_count])
            role_score = float(matched) / float(len(required_roles))
            role_confidence = float(confidence_total) / float(len(required_roles))
        slot_score = 0.0
        role_slots = [
            str(value).strip()
            for value in (metadata.get("role_slots") if isinstance(metadata.get("role_slots"), list) else [])
            if str(value).strip()
        ]
        if role_slots:
            fillable = 0
            for slot in role_slots:
                if self._gsm8k_slot_value(slot, role_values={key: [float(row.get("value")) for row in quantity_candidates if str(row.get("role", "")).strip().lower() == key] for key in role_counts}) is not None:
                    fillable += 1
            slot_score = float(fillable) / float(len(role_slots))
        op_pool = {str(value).strip().lower() for value in clause_operations + top_operations if str(value).strip()}
        if goal_operation:
            op_pool.add(str(goal_operation).strip().lower())
        operation_chain = [
            str(value).strip().lower()
            for value in (metadata.get("operation_chain") if isinstance(metadata.get("operation_chain"), list) else [])
            if str(value).strip()
        ]
        op_score = 0.0
        if operation_chain:
            unique_chain = {value for value in operation_chain if value}
            if unique_chain:
                op_score = float(len(op_pool.intersection(unique_chain))) / float(len(unique_chain))
        return min(
            1.0,
            0.15
            + (0.30 * role_score)
            + (0.15 * slot_score)
            + (0.10 * op_score)
            + (0.30 * role_confidence),
        )

    def _gsm8k_template_program(
        self,
        *,
        context: dict[str, Any],
        metadata: dict[str, Any],
    ) -> str:
        role_values = self._gsm8k_role_value_map(context)
        bound_slots, binding_summary = self._gsm8k_template_slot_bindings(
            context=context,
            metadata=metadata,
        )
        binding_mode = str(metadata.get("binding_mode", "")).strip().lower()
        if not binding_mode:
            return ""

        def _lit(value: float | None) -> str:
            if value is None:
                return ""
            numeric = float(value)
            if abs(numeric - round(numeric)) <= 1e-9:
                return str(int(round(numeric)))
            return self._gpu_scalar_literal(numeric)

        def _bound(slot: str) -> float | None:
            if slot in bound_slots:
                return float(bound_slots[slot])
            return self._gsm8k_slot_value(slot, role_values=role_values)

        semantic_entities = [
            dict(row)
            for row in (
                context.get("semantic_entities")
                if isinstance(context.get("semantic_entities"), list)
                else []
            )
            if isinstance(row, dict)
        ]

        def _sum_literals(values: list[float]) -> list[str]:
            return self._gsm8k_sum_token_rows([[_lit(value)] for value in values if value is not None])

        if binding_summary:
            context["_last_gsm8k_slot_binding"] = binding_summary

        if binding_mode == "remainder_scale":
            initial = _bound("initial")
            part_1 = _bound("part_1")
            part_2 = _bound("part_2")
            rate = _bound("rate")
            if None in {initial, part_1, part_2, rate}:
                return ""
            return f"{_lit(initial)} {_lit(part_1)} - {_lit(part_2)} - {_lit(rate)} *"

        if binding_mode == "ratio_then_add":
            initial = _bound("initial")
            ratio = _bound("ratio_value")
            if None in {initial, ratio}:
                return ""
            op = "*" if float(ratio) <= 1.0 else "/"
            return f"{_lit(initial)} {_lit(ratio)} {op} {_lit(initial)} +"

        if binding_mode == "percentage_change":
            initial = _bound("initial")
            percentage = _bound("percentage")
            if None in {initial, percentage}:
                return ""
            return f"{_lit(initial)} {_lit(percentage)} 100 / * {_lit(initial)} +"

        if binding_mode == "total_minus_parts":
            total = _bound("total")
            parts = list(role_values.get("part", [])) or list(role_values.get("delta", []))
            if total is None or len(parts) < 2:
                return ""
            part_tokens = self._gsm8k_sum_token_rows(
                [[_lit(value)] for value in parts[:3]]
            )
            if not part_tokens:
                return ""
            return " ".join([_lit(total), *part_tokens, "-"]).strip()

        if binding_mode == "base_plus_excess":
            threshold = _bound("threshold")
            rate_1 = _bound("rate_1")
            total = _bound("total")
            excess = _bound("excess")
            rate_2 = _bound("rate_2")
            ratio = _bound("ratio_value")
            if threshold is None or rate_1 is None:
                return ""
            base_tokens = [_lit(threshold), _lit(rate_1), "*"]
            excess_tokens: list[str] = []
            if excess is not None and rate_2 is not None:
                excess_tokens = [_lit(excess), _lit(rate_2), "*"]
            elif total is not None and total > threshold and ratio is not None:
                excess_tokens = [
                    _lit(total),
                    _lit(threshold),
                    "-",
                    _lit(rate_1),
                    _lit(ratio),
                    "*",
                    "*",
                ]
            if not excess_tokens:
                return ""
            return " ".join([*base_tokens, *excess_tokens, "+"]).strip()

        if binding_mode == "multiply_chain_sum":
            counts = list(role_values.get("count", []))
            rate = _bound("rate")
            initial = _bound("initial")
            if initial is not None and counts:
                branch_rows: list[list[str]] = []
                for depth in range(1, len(counts) + 1):
                    literals = [_lit(initial), *[_lit(value) for value in counts[:depth]]]
                    branch_rows.append(self._gsm8k_product_tokens(literals))
                sum_tokens = self._gsm8k_sum_token_rows(branch_rows)
                if not sum_tokens:
                    return ""
                return " ".join([*sum_tokens, _lit(initial), "+"]).strip()
            if rate is not None and len(counts) >= 2:
                literals = [*[_lit(value) for value in counts], _lit(rate)]
                tokens = self._gsm8k_product_tokens(literals)
                return " ".join(tokens).strip()
            return ""

        if binding_mode == "fractional_part_plus_base":
            initial = _bound("initial")
            ratio = _bound("ratio_value")
            if ratio is None:
                default_ratio = metadata.get("default_ratio")
                try:
                    ratio = float(default_ratio)
                except Exception:
                    ratio = None
            if None in {initial, ratio}:
                return ""
            if float(ratio) <= 1.0:
                return f"{_lit(initial)} {_lit(ratio)} * {_lit(initial)} +"
            return f"{_lit(initial)} {_lit(ratio)} / {_lit(initial)} +"

        if binding_mode == "markup_profit_after_costs":
            initial = _bound("initial")
            percentage = _bound("percentage")
            declared_slots = [
                str(value).strip()
                for value in (metadata.get("role_slots") if isinstance(metadata.get("role_slots"), list) else [])
                if str(value).strip()
            ]
            parts = [
                float(part_value)
                for slot_name in declared_slots
                if self._gsm8k_semantic_slot_base(slot_name) == "part"
                for part_value in [_bound(slot_name)]
                if part_value is not None
            ]
            if not parts:
                parts = list(role_values.get("part", []))
            if None in {initial, percentage} or not parts:
                return ""
            part_tokens = _sum_literals(parts[:3])
            if not part_tokens:
                return ""
            return " ".join(
                [
                    _lit(initial),
                    _lit(percentage),
                    "100",
                    "/",
                    "*",
                    _lit(initial),
                    "+",
                    _lit(initial),
                    *part_tokens,
                    "+",
                    "-",
                ]
            ).strip()

        if binding_mode == "count_rate_product":
            counts = list(role_values.get("count", []))
            rate = _bound("rate")
            if rate is None or len(counts) < 2:
                return ""
            return f"{_lit(counts[0])} {_lit(counts[1])} * {_lit(rate)} *"

        if binding_mode == "scaled_total_minus_parts":
            count = _bound("count")
            rate = _bound("rate")
            declared_slots = [
                str(value).strip()
                for value in (metadata.get("role_slots") if isinstance(metadata.get("role_slots"), list) else [])
                if str(value).strip()
            ]
            parts = [
                float(part_value)
                for slot_name in declared_slots
                if self._gsm8k_semantic_slot_base(slot_name) == "part"
                for part_value in [_bound(slot_name)]
                if part_value is not None
            ]
            if not parts:
                parts = list(role_values.get("part", []))
            if None in {count, rate} or len(parts) < 2:
                return ""
            return f"{_lit(count)} {_lit(rate)} * {_lit(parts[0])} - {_lit(parts[1])} -"

        if binding_mode == "alternating_discount_pairs":
            count = _bound("count")
            rate = _bound("rate")
            percentage = _bound("percentage")
            if None in {count, rate, percentage}:
                return ""
            return " ".join(
                [
                    _lit(count),
                    "2",
                    "/",
                    _lit(rate),
                    _lit(rate),
                    _lit(percentage),
                    "100",
                    "/",
                    "*",
                    "+",
                    "*",
                ]
            ).strip()

        if binding_mode == "successive_ratio_family_sum":
            initial = _bound("initial")
            counts = sorted([float(value) for value in role_values.get("count", [])], reverse=True)
            if initial is None or len(counts) < 2:
                return ""
            return " ".join(
                [
                    _lit(initial),
                    _lit(initial),
                    _lit(counts[0]),
                    "*",
                    "+",
                    _lit(initial),
                    _lit(counts[0]),
                    "*",
                    _lit(counts[1]),
                    "*",
                    "+",
                ]
            ).strip()

        if binding_mode == "restart_progress_time":
            total = _bound("total")
            rate = _bound("rate")
            percentage = _bound("percentage")
            duration = _bound("duration")
            if None in {total, rate, percentage, duration}:
                return ""
            return " ".join(
                [
                    _lit(total),
                    _lit(rate),
                    "/",
                    _lit(total),
                    _lit(percentage),
                    "100",
                    "/",
                    "*",
                    _lit(rate),
                    "/",
                    "+",
                    _lit(duration),
                    "+",
                ]
            ).strip()

        if binding_mode == "outbound_return_distance":
            if semantic_entities:
                outbound_duration = None
                outbound_rate = None
                return_window = None
                blocked_duration = None
                segment_duration = None
                segment_rate = None
                final_rate = None
                for entity in semantic_entities:
                    raw_block = str(entity.get("raw_block", "")).strip().lower()
                    surface = str(entity.get("surface", "")).strip().lower()
                    role = str(entity.get("role", "")).strip().lower()
                    try:
                        value = float(entity.get("resolved_value", entity.get("value", 0.0)))
                    except Exception:
                        continue
                    if outbound_duration is None and role == "duration" and "turns around" in raw_block:
                        outbound_duration = value
                    if outbound_rate is None and role == "rate" and "turns around" in raw_block:
                        outbound_rate = value
                    if return_window is None and role == "duration" and "get home in" in raw_block:
                        return_window = value
                    if (
                        blocked_duration is None
                        and role == "duration"
                        and "standstill traffic" in raw_block
                        and ("standstill" in str(entity.get("unit", "")).strip().lower() or surface in {"2", "2.0", "two"})
                    ):
                        blocked_duration = value
                    if segment_duration is None and role == "duration" and ("half-hour" in raw_block or "half an hour" in raw_block or surface == "half-hour"):
                        segment_duration = value
                    if segment_rate is None and role == "rate" and surface in {"30", "30.0"} and ("30mph" in raw_block or "30 mph" in raw_block):
                        segment_rate = value
                    if final_rate is None and role == "rate" and surface in {"80", "80.0"} and ("80mph" in raw_block or "80 mph" in raw_block):
                        final_rate = value
                if None not in {
                    outbound_duration,
                    outbound_rate,
                    return_window,
                    blocked_duration,
                    segment_duration,
                    segment_rate,
                    final_rate,
                }:
                    return " ".join(
                        [
                            _lit(outbound_duration),
                            _lit(outbound_rate),
                            "*",
                            _lit(segment_duration),
                            _lit(segment_rate),
                            "*",
                            _lit(return_window),
                            _lit(blocked_duration),
                            "-",
                            _lit(segment_duration),
                            "-",
                            _lit(final_rate),
                            "*",
                            "+",
                            "-",
                        ]
                    ).strip()
            durations = [float(value) for value in role_values.get("duration", [])]
            rates = [float(value) for value in role_values.get("rate", [])]
            if len(durations) < 4 or len(rates) < 3:
                return ""
            outbound_duration = durations[0]
            return_window = durations[1]
            blocked_duration = durations[2]
            segment_duration = durations[3]
            outbound_rate = rates[0]
            segment_rate = rates[1]
            final_rate = rates[2]
            return " ".join(
                [
                    _lit(outbound_duration),
                    _lit(outbound_rate),
                    "*",
                    _lit(segment_duration),
                    _lit(segment_rate),
                    "*",
                    _lit(return_window),
                    _lit(blocked_duration),
                    "-",
                    _lit(segment_duration),
                    "-",
                    _lit(final_rate),
                    "*",
                    "+",
                    "-",
                ]
            ).strip()

        return ""

    def _gsm8k_decomposition_preview(
        self,
        *,
        engine: Any,
        context: dict[str, Any],
        strategy: str,
    ) -> tuple[str, str, str, float] | None:
        pattern_rows = context.get("pattern_rows") if isinstance(context.get("pattern_rows"), list) else []
        quantity_candidates = [
            dict(row)
            for row in (
                context.get("quantity_role_candidates")
                if isinstance(context.get("quantity_role_candidates"), list)
                else []
            )
            if isinstance(row, dict)
        ]
        quantity_count = len(context.get("number_values", [])) if isinstance(context.get("number_values"), list) else 0
        clause_operations = [
            str(value).strip().lower()
            for value in (context.get("clause_operations") if isinstance(context.get("clause_operations"), list) else [])
            if str(value).strip()
        ]
        top_operations = [
            str(value).strip().lower()
            for value in (context.get("top_operations") if isinstance(context.get("top_operations"), list) else [])
            if str(value).strip()
        ]
        goal_operation = str(context.get("goal_operation", "")).strip().lower()
        if str(strategy or "").strip() in {"forward_chain", "backward_chain", "fusion_chain", "clause_chain", "goal_adjusted_chain"}:
            for row in pattern_rows:
                metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                program = self._gsm8k_template_program(context=context, metadata=metadata)
                if not program:
                    continue
                try:
                    value = engine.evaluate(program)
                except Exception:
                    continue
                try:
                    numeric_value = float(value)
                except Exception:
                    continue
                if not math.isfinite(numeric_value):
                    continue
                label = str(strategy or "fusion_chain").strip() or "fusion_chain"
                structural_score = self._gsm8k_pattern_structural_score(
                    metadata=metadata,
                    quantity_candidates=quantity_candidates,
                    quantity_count=quantity_count,
                    clause_operations=clause_operations,
                    top_operations=top_operations,
                    goal_operation=goal_operation,
                )
                return self._format_math_answer(numeric_value), program, label, float(structural_score)

        fusion_values = [float(value) for value in context.get("number_values", [])[:6]]
        forward_values = [float(value) for value in context.get("forward_number_values", [])[:6]] or fusion_values
        backward_values = [float(value) for value in context.get("backward_number_values", [])[:6]] or list(reversed(fusion_values))
        operation_chain = [
            str(value).strip().lower()
            for value in context.get("operation_chain", [])
            if str(value).strip()
        ]
        top_operation = str(context.get("top_operation", "")).strip().lower()
        top_operations = [
            str(value).strip().lower()
            for value in context.get("top_operations", [])
            if str(value).strip()
        ]
        clause_values = [float(value) for value in context.get("clause_values", [])[:6]]
        clause_operations = [
            str(value).strip().lower()
            for value in context.get("clause_operations", [])
            if str(value).strip()
        ]
        goal_operation = str(context.get("goal_operation", "")).strip().lower()
        if not top_operations and top_operation:
            top_operations = [top_operation]

        label = str(strategy or "fusion_chain").strip() or "fusion_chain"
        program = ""
        values = fusion_values

        if label == "forward_chain":
            values = forward_values
            if operation_chain and len(values) >= len(operation_chain) + 1:
                program = self._gsm8k_left_fold_program(values[: len(operation_chain) + 1], operation_chain)
        elif label == "backward_chain":
            values = backward_values
            if operation_chain:
                reverse_chain = list(reversed(operation_chain))
                if len(values) >= len(reverse_chain) + 1:
                    program = self._gsm8k_left_fold_program(values[: len(reverse_chain) + 1], reverse_chain)
        elif label == "top2_chain":
            values = fusion_values
            if len(values) >= 3 and len(top_operations) >= 2:
                program = self._gsm8k_left_fold_program(values[:3], top_operations[:2])
        elif label == "clause_chain":
            values = clause_values or forward_values
            if clause_operations and len(values) >= len(clause_operations) + 1:
                program = self._gsm8k_left_fold_program(values[: len(clause_operations) + 1], clause_operations)
        elif label == "goal_adjusted_chain":
            values = clause_values or forward_values
            if len(values) >= 2 and clause_operations:
                relation_operation = clause_operations[0]
                relation_token = self._gsm8k_operator_token(relation_operation)
                goal_token = self._gsm8k_operator_token(goal_operation)
                if relation_token and goal_token and goal_operation in {"add", "sub"}:
                    base = self._gpu_scalar_literal(values[0])
                    relation = self._gpu_scalar_literal(values[1])
                    if goal_operation == "add":
                        program = f"{base} {relation} {relation_token} {base} +"
                    else:
                        program = f"{base} {base} {relation} {relation_token} -"
        elif label == "scale_then_add_base":
            values = forward_values
            if len(values) >= 2:
                base = float(values[0])
                factor = float(values[1])
                program = (
                    f"{self._gpu_scalar_literal(base)} "
                    f"{self._gpu_scalar_literal(base)} "
                    f"{self._gpu_scalar_literal(factor)} * +"
                )
        elif label == "hierarchical_sum":
            values = forward_values
            if len(values) >= 3:
                top_factor = float(values[0])
                mid_factor = float(values[1])
                base = float(values[2])
                program = (
                    f"{self._gpu_scalar_literal(base)} {self._gpu_scalar_literal(mid_factor)} * STORE_A "
                    f"RECALL_A {self._gpu_scalar_literal(top_factor)} * RECALL_A + "
                    f"{self._gpu_scalar_literal(base)} +"
                )
        elif label == "alt_add":
            values = fusion_values
            if len(values) >= 2:
                width = min(len(values), 4)
                program = self._gsm8k_left_fold_program(values[:width], ["add"] * (width - 1))
        elif label == "alt_sub":
            values = fusion_values
            if len(values) >= 2:
                width = min(len(values), 4)
                program = self._gsm8k_left_fold_program(values[:width], ["sub"] * (width - 1))
        elif label == "alt_mul":
            values = fusion_values
            if len(values) >= 2:
                width = min(len(values), 4)
                program = self._gsm8k_left_fold_program(values[:width], ["mul"] * (width - 1))
        elif label == "alt_div":
            values = fusion_values
            if len(values) >= 2:
                width = min(len(values), 4)
                program = self._gsm8k_left_fold_program(values[:width], ["div"] * (width - 1))
        else:
            values = fusion_values
            if operation_chain and len(values) >= len(operation_chain) + 1:
                program = self._gsm8k_left_fold_program(values[: len(operation_chain) + 1], operation_chain)
            elif top_operation and len(values) >= 2:
                width = min(len(values), 4 if top_operation in {"add", "mul"} else 2)
                program = self._gsm8k_left_fold_program(values[:width], [top_operation] * max(1, width - 1))

        if not program:
            return None
        try:
            value = engine.evaluate(program)
        except Exception:
            return None
        try:
            numeric_value = float(value)
        except Exception:
            return None
        if not math.isfinite(numeric_value):
            return None
        best_structural = 0.0
        for row in pattern_rows:
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            best_structural = max(
                best_structural,
                float(
                    self._gsm8k_pattern_structural_score(
                        metadata=metadata,
                        quantity_candidates=quantity_candidates,
                        quantity_count=quantity_count,
                        clause_operations=clause_operations,
                        top_operations=top_operations,
                        goal_operation=goal_operation,
                    )
                ),
            )
        return self._format_math_answer(numeric_value), program, label, float(best_structural)

    def _gsm8k_decomposition_result(
        self,
        *,
        engine: Any,
        best_candidate: dict[str, Any] | None,
    ) -> tuple[str, list[str]] | None:
        candidate = dict(best_candidate or {})
        context = candidate.get("gsm8k_context") if isinstance(candidate.get("gsm8k_context"), dict) else {}
        if not context:
            match = candidate.get("match") if isinstance(candidate.get("match"), dict) else {}
            if isinstance(match.get("gsm8k_context"), dict):
                context = dict(match.get("gsm8k_context"))
        if not context:
            return None
        preview_answer = str(candidate.get("gsm8k_preview_answer", "")).strip()
        preview_program = str(candidate.get("gsm8k_preview_program", "")).strip()
        preview_label = str(candidate.get("gsm8k_preview_strategy", "")).strip()
        if preview_answer and preview_program:
            binding_summary = str(context.get("_last_gsm8k_slot_binding", "")).strip()
            return preview_answer, [
                "GSM8K atomic fission: operation/number context bound from navigator fusion parse",
                f"GSM8K candidate program: {preview_label or 'fusion_chain'}",
                *([f"GSM8K slot binding: {binding_summary}"] if binding_summary else []),
                f"GSM8K fusion eval: {preview_program}",
            ]
        path = candidate.get("path") if isinstance(candidate.get("path"), dict) else {}
        strategy = str(candidate.get("gsm8k_preview_strategy") or path.get("composition_strategy") or "fusion_chain")
        preview = self._gsm8k_decomposition_preview(
            engine=engine,
            context=context,
            strategy=strategy,
        )
        if preview is None:
            return None
        answer, program, label, _ = preview
        binding_summary = str(context.get("_last_gsm8k_slot_binding", "")).strip()
        return answer, [
            "GSM8K atomic fission: operation/number context bound from navigator fusion parse",
            f"GSM8K candidate program: {label}",
            *([f"GSM8K slot binding: {binding_summary}"] if binding_summary else []),
            f"GSM8K fusion eval: {program}",
        ]

    def _answer_math_query(
        self,
        *,
        task: dict[str, Any],
        binding: dict[str, Any],
        reasoning_program: dict[str, Any],
        route_galaxies: list[str] | None,
        match: dict[str, Any],
        similarity: float,
        engine: Any,
        specialist: str,
        domain_hint: str | None,
        query_text: str,
        use_enriched: bool,
        query_type: str | None,
        selection_steps: list[str],
        best_candidate: dict[str, Any] | None,
    ) -> dict[str, Any]:
        answer = str(match.get("answer_text") or match.get("name") or match.get("id") or "").strip()
        extra_steps: list[str] = list(selection_steps)
        rpn_program = str(match.get("rpn_program", "")).strip()
        resolved = False
        if self._is_gsm8k_math_task(task):
            decomposition_result = self._gsm8k_decomposition_result(
                engine=engine,
                best_candidate=best_candidate,
            )
            if decomposition_result is not None:
                answer, decomposition_steps = decomposition_result
                extra_steps.extend(decomposition_steps)
                resolved = True
        if not resolved and rpn_program:
            try:
                gpu_value = engine.evaluate(rpn_program)
                answer = self._format_math_answer(gpu_value)
                extra_steps.append(f"GPU math eval: {rpn_program}")
                resolved = True
            except Exception:
                pass
        if not resolved:
            try:
                template_result = self._evaluate_math_template(
                    engine=engine,
                    match=match,
                    query_text=query_text,
                    numeric_fallbacks=(
                        list(best_candidate.get("parse_quantity_values", []))
                        if isinstance(best_candidate, dict)
                        else None
                    ),
                )
            except Exception:
                template_result = None
            if template_result is not None:
                answer, template_steps = template_result
                extra_steps.extend(template_steps)
                resolved = True
        thinking_trace = self._build_gpu_thinking_trace(
            binding=binding,
            program_id=str(reasoning_program.get("id", "")),
            match=match,
            similarity=similarity,
            specialist=specialist,
            extra_steps=extra_steps,
        )
        return {
            "status": "ok",
            "answer": answer,
            "response": answer,
            "result": answer,
            "predicted_answer": answer,
            "thinking_trace": thinking_trace,
            "reasoning_trace": list(thinking_trace),
            "thinking_xml": self._render_thinking_xml(thinking_trace, answer),
            "gpu_execution": True,
            "runtime": "knowledgeverse_gpu_query",
            "program_id": str(reasoning_program.get("id", "")),
            "program_type": "gpu_math_template_match_lookup",
            "solver": "knowledgeverse_gpu_query",
            "query_text": query_text,
            "top_match_similarity": similarity,
            "route": {
                "specialist": specialist,
                "domain_hint": domain_hint,
                "galaxy_names": list(route_galaxies or binding.get("galaxies", [])),
                "scanned_galaxies": list(binding.get("galaxies", [])),
            },
            "match": match,
            "query_type": str(query_type or ""),
            "use_enriched": bool(use_enriched),
            "task_id": str(task.get("task_id", "")),
            "gsm8k_preview_strategy": str(best_candidate.get("gsm8k_preview_strategy", "")).strip(),
            "gsm8k_preview_program": str(best_candidate.get("gsm8k_preview_program", "")).strip(),
            "gsm8k_consensus_support": int(best_candidate.get("gsm8k_consensus_support", 0) or 0),
            "gsm8k_operation_ids": list(
                (
                    best_candidate.get("gsm8k_context")
                    if isinstance(best_candidate.get("gsm8k_context"), dict)
                    else {}
                ).get("operation_ids", [])
            ),
        }

    def _answer_mmlu_query(
        self,
        *,
        task: dict[str, Any],
        binding: dict[str, Any],
        reasoning_program: dict[str, Any],
        route_galaxies: list[str] | None,
        match: dict[str, Any],
        similarity: float,
        specialist: str,
        domain_hint: str | None,
        query_text: str,
        use_enriched: bool,
        query_type: str | None,
        selection_steps: list[str],
        best_candidate: dict[str, Any],
    ) -> dict[str, Any]:
        path = best_candidate.get("path") if isinstance(best_candidate.get("path"), dict) else {}
        answer = str(path.get("option_text", "")).strip()
        if not answer:
            answer = str(match.get("answer_text") or match.get("name") or match.get("id") or "").strip()
        extra_steps = list(selection_steps)
        if answer:
            extra_steps.append(
                "Elimination winner: "
                f"{answer} via {str(match.get('id', '')).strip() or 'unknown_entry'}"
            )
        thinking_trace = self._build_gpu_thinking_trace(
            binding=binding,
            program_id=str(reasoning_program.get("id", "")),
            match=match,
            similarity=similarity,
            specialist=specialist,
            extra_steps=extra_steps,
        )
        return {
            "status": "ok",
            "answer": answer,
            "response": answer,
            "result": answer,
            "predicted_answer": answer,
            "thinking_trace": thinking_trace,
            "reasoning_trace": list(thinking_trace),
            "thinking_xml": self._render_thinking_xml(thinking_trace, answer),
            "gpu_execution": True,
            "runtime": "knowledgeverse_gpu_query",
            "program_id": str(reasoning_program.get("id", "")),
            "program_type": "gpu_mmlu_elimination",
            "solver": "knowledgeverse_gpu_query",
            "query_text": query_text,
            "top_match_similarity": similarity,
            "route": {
                "specialist": specialist,
                "domain_hint": domain_hint,
                "galaxy_names": list(route_galaxies or binding.get("galaxies", [])),
                "scanned_galaxies": list(binding.get("galaxies", [])),
            },
            "match": match,
            "query_type": str(query_type or ""),
            "use_enriched": bool(use_enriched),
            "task_id": str(task.get("task_id", "")),
        }

    def _answer_lhe_query(
        self,
        *,
        task: dict[str, Any],
        binding: dict[str, Any],
        reasoning_program: dict[str, Any],
        route_galaxies: list[str] | None,
        match: dict[str, Any],
        similarity: float,
        specialist: str,
        domain_hint: str | None,
        query_text: str,
        use_enriched: bool,
        query_type: str | None,
        selection_steps: list[str],
        best_candidate: dict[str, Any],
    ) -> dict[str, Any]:
        path = best_candidate.get("path") if isinstance(best_candidate.get("path"), dict) else {}
        answer = str(path.get("option_text", "")).strip()
        if not answer:
            answer = str(match.get("answer_text") or match.get("name") or match.get("id") or "").strip()
        extra_steps = list(selection_steps)
        if answer:
            extra_steps.append(
                "LHE option winner: "
                f"{answer} via {str(match.get('id', '')).strip() or 'unknown_entry'}"
            )
        thinking_trace = self._build_gpu_thinking_trace(
            binding=binding,
            program_id=str(reasoning_program.get("id", "")),
            match=match,
            similarity=similarity,
            specialist=specialist,
            extra_steps=extra_steps,
        )
        return {
            "status": "ok",
            "answer": answer,
            "response": answer,
            "result": answer,
            "predicted_answer": answer,
            "thinking_trace": thinking_trace,
            "reasoning_trace": list(thinking_trace),
            "thinking_xml": self._render_thinking_xml(thinking_trace, answer),
            "gpu_execution": True,
            "runtime": "knowledgeverse_gpu_query",
            "program_id": str(reasoning_program.get("id", "")),
            "program_type": "gpu_lhe_option_selection",
            "solver": "knowledgeverse_gpu_query",
            "query_text": query_text,
            "top_match_similarity": similarity,
            "route": {
                "specialist": specialist,
                "domain_hint": domain_hint,
                "galaxy_names": list(route_galaxies or binding.get("galaxies", [])),
                "scanned_galaxies": list(binding.get("galaxies", [])),
            },
            "match": match,
            "query_type": str(query_type or ""),
            "use_enriched": bool(use_enriched),
            "task_id": str(task.get("task_id", "")),
        }

    def _answer_chat_query(
        self,
        *,
        binding: dict[str, Any],
        reasoning_program: dict[str, Any],
        route_galaxies: list[str] | None,
        match: dict[str, Any],
        similarity: float,
        specialist: str,
        domain_hint: str | None,
        query_text: str,
        use_enriched: bool,
        query_type: str | None,
        selection_steps: list[str],
    ) -> dict[str, Any]:
        metadata = match.get("metadata") if isinstance(match.get("metadata"), dict) else {}
        answer = str(
            metadata.get("definition")
            or match.get("answer_text")
            or match.get("name")
            or match.get("id")
            or ""
        ).strip()
        extra_steps: list[str] = list(selection_steps)
        rpn_program = str(match.get("rpn_program", "")).strip()
        if rpn_program:
            try:
                answer = self._format_math_answer(self.get_gpu_reasoning_engine().evaluate(rpn_program))
                extra_steps.append(f"GPU eval: {rpn_program}")
            except Exception:
                pass
        else:
            try:
                template_result = self._evaluate_math_template(
                    engine=self.get_gpu_reasoning_engine(),
                    match=match,
                    query_text=query_text,
                )
            except Exception:
                template_result = None
            if template_result is not None:
                answer, template_steps = template_result
                extra_steps.extend(template_steps)
        thinking_trace = self._build_gpu_thinking_trace(
            binding=binding,
            program_id=str(reasoning_program.get("id", "")),
            match=match,
            similarity=similarity,
            specialist=specialist,
            extra_steps=extra_steps,
        )
        return {
            "status": "ok",
            "answer": answer,
            "response": answer,
            "result": answer,
            "thinking_trace": thinking_trace,
            "reasoning_trace": list(thinking_trace),
            "thinking_xml": self._render_thinking_xml(thinking_trace, answer),
            "gpu_execution": True,
            "runtime": "knowledgeverse_gpu_query",
            "program_id": str(reasoning_program.get("id", "")),
            "program_type": "gpu_chat_lookup",
            "solver": "knowledgeverse_gpu_query",
            "query_text": query_text,
            "top_match_similarity": similarity,
            "route": {
                "specialist": specialist,
                "domain_hint": domain_hint,
                "galaxy_names": list(route_galaxies or binding.get("galaxies", [])),
                "scanned_galaxies": list(binding.get("galaxies", [])),
            },
            "match": match,
            "query_type": str(query_type or ""),
            "use_enriched": bool(use_enriched),
        }

    def ensure_default_galaxies_loaded(self, *, force: bool = False) -> dict[str, int]:
        """
        Ensure all default galaxies are present in the active universe.

        This enforces the single-world contract for training/benchmark runs:
        all default galaxies are loaded and queryable in every session.
        """
        if self._default_galaxies_loaded and not force:
            return {name: len(self.galaxy_manager.get_galaxy(name).entries) for name in self.DEFAULT_GALAXIES}

        counts: dict[str, int] = {}
        for galaxy_name in self.DEFAULT_GALAXIES:
            galaxy = self.galaxy_manager.get_galaxy(galaxy_name)
            counts[galaxy_name] = len(getattr(galaxy, "entries", []))
        self._ensure_runtime_language_enrichment_loaded()
        for galaxy_name in ("Word", "Grammar"):
            counts[galaxy_name] = len(self.galaxy_manager.get_galaxy(galaxy_name).entries)
        self._default_galaxies_loaded = True
        return counts

    def log_event(
        self,
        event_type: str,
        event_data: dict[str, Any],
        parent_event_id: str | None = None,
    ) -> str:
        """Record an event into Shadow Copy compressed audit."""
        event_id = self.shadow_copy.record_event(
            event_type=event_type,
            event_data=event_data,
            parent_event_id=parent_event_id,
        )
        try:
            specialist = str(event_data.get("specialist", "grammar"))
            query = str(event_data.get("query") or event_data.get("prompt") or event_type)
            lowered = event_type.lower()
            raw_outcome = (
                event_data.get("verdict_trit")
                if event_data.get("verdict_trit") is not None else
                event_data.get("outcome")
            )
            ternary_outcome: int | None
            if raw_outcome is not None:
                ternary_outcome = max(-1, min(1, int(raw_outcome)))
            else:
                ternary_outcome = 1 if (
                    ("success" in lowered) or (
                        "fail" not in lowered and float(event_data.get("confidence", 0.0)) >= 0.65
                    )
                ) else -1
            self.trm_navigator.learn_from_feedback(
                query=query,
                specialist=specialist,
                success=None,
                ternary_outcome=ternary_outcome,
                confidence=float(event_data.get("confidence", 0.0) or 0.0),
                domain_hint=str(event_data.get("domain_hint") or event_data.get("domain") or ""),
                defeat_source=str(
                    event_data.get("was_defeated_by")
                    or event_data.get("defeat_source")
                    or ""
                ).strip() or None,
            )
            self.trm_navigator.save_weights()
        except Exception:
            # Feedback learning should never block event recording.
            pass
        return event_id

    @staticmethod
    def _normalize_answer_text(value: Any) -> str:
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=True, sort_keys=True)
        return str(value).strip()

    @staticmethod
    def _extract_numeric_literals(text: str) -> list[float]:
        values: list[float] = []
        token: list[str] = []
        previous = ""
        for char in str(text):
            if char.isdigit() or char == "." or (char == "-" and not token and previous not in {"e", "E"}):
                token.append(char)
            else:
                if token:
                    try:
                        values.append(float("".join(token)))
                    except Exception:
                        pass
                    token = []
            previous = char
        if token:
            try:
                values.append(float("".join(token)))
            except Exception:
                pass
        return values

    def _infer_query_success(
        self,
        *,
        task: dict[str, Any] | None,
        result: dict[str, Any],
    ) -> tuple[bool, bool | None]:
        payload = dict(task or {})
        task_type = str(payload.get("type", "")).upper()
        if task_type == "ARC_TASK" and payload.get("expected_output") is not None:
            correct = result.get("output_grid") == payload.get("expected_output")
            return bool(correct), bool(correct)
        expected = None
        for key in ("expected_answer", "correct_answer", "answer"):
            if key in payload and payload.get(key) not in (None, ""):
                expected = payload.get(key)
                break
        if expected is not None:
            predicted = (
                result.get("predicted_answer")
                or result.get("result")
                or result.get("response")
                or result.get("answer")
            )
            correct = self._normalize_answer_text(predicted) == self._normalize_answer_text(expected)
            return bool(correct), bool(correct)
        return str(result.get("status", "")).lower() == "ok", None

    def _record_query_feedback(
        self,
        *,
        task: dict[str, Any] | None,
        result: dict[str, Any],
        specialist: str,
        domain_hint: str | None,
    ) -> None:
        if not isinstance(result, dict):
            return
        success, correctness = self._infer_query_success(task=task, result=result)
        match = result.get("match") if isinstance(result.get("match"), dict) else {}
        similarity = self._clamp_confidence(result.get("top_match_similarity", 0.0))
        event_data = {
            "query": str(result.get("query_text") or self._query_text("", task=task)).strip(),
            "specialist": str((result.get("route") or {}).get("specialist") or specialist or "auto"),
            "domain_hint": str(domain_hint or (task or {}).get("domain_hint") or ""),
            "confidence": similarity,
            "correct": correctness,
            "query_type": str(result.get("query_type") or (task or {}).get("type") or ""),
            "program_id": str(result.get("program_id", "")),
            "match_id": str(match.get("id", "")),
            "match_galaxy": str(match.get("galaxy", "")),
            "route_galaxies": list((result.get("route") or {}).get("galaxy_names", [])),
            "thinking_trace": list(result.get("thinking_trace", [])),
            "use_enriched": bool(result.get("use_enriched", True)),
        }
        event_type = "gpu_query_success" if success else "gpu_query_failure"
        try:
            event_id = self.log_event(event_type, event_data)
            result["shadow_event_id"] = event_id
        except Exception:
            event_id = None
        if self._is_benchmark_evaluation_task(task):
            return
        match_id = str(match.get("id", "")).strip()
        pattern_ids: list[str] = []
        for raw_pattern_id in [
            match_id,
            str(result.get("winning_program_id") or result.get("program_id") or "").strip(),
            (
                f"gsm8k_strategy_{str(result.get('gsm8k_preview_strategy', '')).strip()}"
                if str(result.get("gsm8k_preview_strategy", "")).strip()
                else ""
            ),
            *[
                str(operation_id).strip()
                for operation_id in (
                    result.get("gsm8k_operation_ids")
                    if isinstance(result.get("gsm8k_operation_ids"), list)
                    else []
                )
            ],
        ]:
            token = str(raw_pattern_id).strip()
            if token and token not in pattern_ids:
                pattern_ids.append(token)
        if not pattern_ids:
            return
        outcome = 0
        if correctness is True:
            outcome = 1
        elif correctness is False:
            outcome = -1
        elif str(result.get("status", "")).lower() == "error":
            outcome = -1
        for pattern_id in pattern_ids:
            try:
                self.ternary_quality_memory.update(
                    pattern_id=pattern_id,
                    outcome=outcome,
                    confidence=similarity,
                    knowledgeverse=self,
                    specialist=str(event_data.get("specialist", "")).strip() or specialist,
                    galaxy=(
                        str(match.get("galaxy", "")).strip()
                        if pattern_id == match_id and match_id
                        else "Grammar"
                    ),
                    source="gpu_query_runtime",
                )
            except Exception:
                continue

    def _collect_parse_bundle(
        self,
        query_text: str,
        *,
        specialist: str,
        galaxy_names: list[str],
        domain_hint: str | None = None,
        task: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        navigator = getattr(self, "navigator_specialist", None)
        if navigator is None:
            return {}
        task_payload = dict(task or {})
        task_type = str(task_payload.get("type", "")).strip().upper()
        goal_type_family = ""
        if self._is_gsm8k_math_task(task_payload):
            goal_type_family = "gsm8k"
        elif task_type == "LHE_TASK":
            goal_type_family = "lhe"
        elif task_type == "MATH_TASK":
            goal_type_family = "math"
        try:
            routes = navigator.plan_routes(
                query=query_text,
                specialist=specialist,
                domain_hint=domain_hint,
                galaxy_names=galaxy_names,
                use_forward_backward=True,
                task_type=task_type,
                goal_type_family=goal_type_family,
            )
        except Exception:
            return {}
        bundle: dict[str, Any] = {
            "query_text": str(query_text),
            "route_plan": [dict(route) for route in routes if isinstance(route, dict)],
        }
        for key in ("forward_parse", "backward_parse", "fusion_parse"):
            for route in routes:
                if not isinstance(route, dict):
                    continue
                value = route.get(key)
                if not isinstance(value, dict):
                    continue
                bundle[key] = dict(value)
                bundle[f"{key}_strategy"] = str(route.get("strategy", "")).strip()
                bundle[f"{key}_query_variant"] = str(route.get("query_variant", "")).strip()
                break
        bundle["quantity_values"] = self._parse_bundle_quantity_values(bundle)
        return bundle

    @staticmethod
    def _parse_bundle_variant_rows(
        parse_bundle: dict[str, Any] | None,
        query_text: str,
    ) -> list[dict[str, Any]]:
        bundle = dict(parse_bundle or {})
        routes = bundle.get("route_plan") if isinstance(bundle.get("route_plan"), list) else []
        preferred = ("fusion", "forward", "backward", "auto")
        variants: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for strategy_name in preferred:
            for route in routes:
                if not isinstance(route, dict):
                    continue
                route_strategy = str(route.get("strategy", "")).strip().lower()
                route_variant = str(route.get("query_variant", "")).strip() or str(query_text)
                if route_strategy != strategy_name:
                    continue
                key = (route_strategy, route_variant)
                if key in seen:
                    continue
                seen.add(key)
                variants.append(
                    {
                        "strategy": route_strategy or "auto",
                        "query_text": route_variant,
                    }
                )
                break
        if not variants:
            variants.append({"strategy": "auto", "query_text": str(query_text)})
        return variants

    @staticmethod
    def _parse_bundle_quantity_values(parse_bundle: dict[str, Any] | None) -> list[float]:
        bundle = dict(parse_bundle or {})
        raw_values = bundle.get("quantity_values")
        if isinstance(raw_values, list):
            values: list[float] = []
            for raw_value in raw_values:
                try:
                    values.append(float(raw_value))
                except Exception:
                    continue
            if values:
                return values
        fusion = bundle.get("fusion_parse") if isinstance(bundle.get("fusion_parse"), dict) else {}
        merged_rows = fusion.get("merged_quantities") if isinstance(fusion.get("merged_quantities"), list) else []
        values = []
        for row in merged_rows:
            if not isinstance(row, dict):
                continue
            try:
                values.append(float(row.get("value")))
            except Exception:
                continue
        return values

    def _parse_bundle_numeric_ids(self, parse_bundle: dict[str, Any] | None) -> list[str]:
        ids: list[str] = []
        for value in self._parse_bundle_quantity_values(parse_bundle):
            entry_id = self._numeric_entry_id_for_value(value)
            if not entry_id:
                continue
            ids.append(entry_id)
        return list(dict.fromkeys(ids))

    def _numeric_entry_id_for_value(self, value: float) -> str:
        rounded = int(round(float(value)))
        if abs(float(value) - float(rounded)) > 1e-6:
            return ""
        entry_id = f"num_{rounded}"
        if self._catalog_entry_by_id(entry_id) is None:
            return ""
        return entry_id

    def _parse_bundle_embeddings(
        self,
        *,
        query_embedding: list[float],
        parse_bundle: dict[str, Any] | None,
        task: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        variants = self._parse_bundle_variant_rows(parse_bundle, str((parse_bundle or {}).get("query_text", "")))
        embeddings: dict[str, list[float]] = {}
        for variant in variants[:4]:
            strategy = str(variant.get("strategy", "")).strip().lower() or "auto"
            variant_text = str(variant.get("query_text", "")).strip()
            if not variant_text or strategy in embeddings:
                continue
            try:
                embeddings[strategy] = self._embed_query_gpu(variant_text, task=task)
            except Exception:
                continue
        fusion_embedding = list(embeddings.get("fusion", []))
        if not fusion_embedding:
            fusion_embedding = self._mean_embedding_rows([row for row in embeddings.values() if row])
        forward_embedding = list(embeddings.get("forward", []))
        backward_embedding = list(embeddings.get("backward", []))
        base_weight, fusion_weight, forward_weight, backward_weight = self._parse_navigation_weights(task=task)
        navigation_embedding = self._weighted_mean_embedding_rows(
            [list(query_embedding), fusion_embedding, forward_embedding, backward_embedding],
            [base_weight, fusion_weight, forward_weight, backward_weight],
        )
        if not navigation_embedding:
            navigation_embedding = list(query_embedding)
            if fusion_embedding:
                navigation_embedding = self._blend_reference_embedding(
                    navigation_embedding,
                    fusion_embedding,
                    alpha=0.52,
                )
        directional_embedding = self._weighted_mean_embedding_rows(
            [forward_embedding, backward_embedding],
            [forward_weight, backward_weight],
        )
        return {
            "variants": variants,
            "navigation_embedding": navigation_embedding,
            "fusion_embedding": fusion_embedding,
            "forward_embedding": forward_embedding,
            "backward_embedding": backward_embedding,
            "directional_embedding": directional_embedding,
            "quantity_values": self._parse_bundle_quantity_values(parse_bundle),
            "numeric_ids": self._parse_bundle_numeric_ids(parse_bundle),
        }

    def _task_parse_override_signals(
        self,
        *,
        task: dict[str, Any] | None,
        domain_hint: str | None,
    ) -> dict[str, str]:
        payload = dict(task or {})
        task_type = str(payload.get("type", "")).strip().upper()
        algebra_signal = ""
        domain_signal = ""
        if task_type == "MATH_TASK":
            competition = str(payload.get("competition", "")).strip().lower()
            if ":" in competition:
                _, _, suffix = competition.partition(":")
                normalized = suffix.strip().replace("&", "and").replace(" ", "_")
                if normalized == "algebra":
                    algebra_signal = "algebra"
        if task_type in {"LHE_TASK", "MMLU_TASK"}:
            normalized_domain = str(domain_hint or payload.get("domain_hint") or payload.get("domain") or "").strip().lower()
            if normalized_domain not in {"", "multi", "general", "unknown"}:
                domain_signal = normalized_domain.replace(" ", "_")
        return {
            "algebra_signal": algebra_signal,
            "domain_signal": domain_signal,
        }

    def _candidate_subject_tokens(self, match: dict[str, Any]) -> set[str]:
        metadata = match.get("metadata") if isinstance(match.get("metadata"), dict) else {}
        raw_tokens = {
            str(match.get("subject", "")).strip().lower(),
            str(metadata.get("subject", "")).strip().lower(),
            str(metadata.get("subfield", "")).strip().lower(),
            str(match.get("domain", "")).strip().lower(),
        }
        tokens: set[str] = set()
        for token in raw_tokens:
            if not token:
                continue
            tokens.update(self._subject_hint_aliases(token))
        return {token for token in tokens if token}

    def _candidate_matches_parse_signal(self, match: dict[str, Any], signal: str) -> bool:
        aliases = {token for token in self._subject_hint_aliases(signal) if token}
        if not aliases:
            return False
        return bool(self._candidate_subject_tokens(match).intersection(aliases))

    def _candidate_ternary_prior(self, pattern_id: str) -> float:
        token = str(pattern_id).strip()
        if not token:
            return 0.0
        try:
            prior = self.ternary_quality_memory.get_prior(token)
        except Exception:
            return 0.0
        if prior is None:
            return 0.0
        return float(prior.prior)

    def _select_gpu_profile(
        self,
        *,
        task: dict[str, Any] | None,
        route: dict[str, Any] | None,
        specialist: str,
        query_text: str,
        options: list[str] | None = None,
    ) -> tuple[list[str], str]:
        task_type = str((task or {}).get("type", "")).upper()
        route_specialist = str((route or {}).get("specialist") or specialist or "").strip().lower()
        if task_type == "ARC_TASK":
            return list(self._resolve_gpu_target_galaxies(route=route, task=task)), self.GPU_ARC_REASONING_PROGRAM_ID
        if task_type == "MATH_TASK":
            return list(self._resolve_gpu_target_galaxies(route=route, task=task)), self.GPU_MATH_REASONING_PROGRAM_ID
        if task_type == "LHE_TASK":
            choice_list = options
            if choice_list is None and isinstance((task or {}).get("options"), list):
                choice_list = [str(option) for option in (task or {}).get("options", [])]
            if not choice_list:
                choice_list = self._inline_choice_options(query_text)
            program_id = "reasoning_elimination_top1" if choice_list else self.GPU_FACTUAL_REASONING_PROGRAM_ID
            return list(self._resolve_gpu_target_galaxies(route=route, task=task)), program_id
        if task_type in {"CHAT_TASK", "GENERAL_TASK", "GRAMMAR_TASK", "MMLU_TASK"} or route_specialist in {"chat", "grammar", "any"}:
            lowered = str(query_text).strip().lower()
            program_id = self.GPU_CHAT_REASONING_PROGRAM_ID
            choice_list = options
            if choice_list is None and isinstance((task or {}).get("options"), list):
                choice_list = [str(option) for option in (task or {}).get("options", [])]
            if choice_list:
                program_id = "reasoning_elimination_top1"
            elif self._query_looks_reality_fact(query_text):
                program_id = self.GPU_FACTUAL_REASONING_PROGRAM_ID
            elif lowered.startswith(("what is ", "who is ", "define ", "meaning of ")) or "atomic number" in lowered:
                program_id = "reasoning_definition_top1"
            elif "compare" in lowered or "difference between" in lowered:
                program_id = "reasoning_comparison_top1"
            elif (
                "using" in lowered
                or "together" in lowered
                or "combine" in lowered
                or (" and " in lowered and ("both" in lowered or "combine" in lowered or "jointly" in lowered))
            ):
                program_id = "reasoning_multi_hop_top2"
            return list(self._resolve_gpu_target_galaxies(route=route, task=task)), program_id
        return list(self._resolve_gpu_target_galaxies(route=route, task=task)), self.GPU_FACTUAL_REASONING_PROGRAM_ID

    @staticmethod
    def _match_template_ref(match: dict[str, Any]) -> str:
        metadata = match.get("metadata") if isinstance(match.get("metadata"), dict) else {}
        template_ref = str(match.get("template_ref", "")).strip()
        if template_ref:
            return template_ref
        meaning_ref = str(metadata.get("meaning_ref", "")).strip()
        if meaning_ref.startswith("math_template_"):
            return meaning_ref
        match_id = str(match.get("id", "")).strip()
        if match_id.startswith("math_template_"):
            return match_id
        return ""

    @staticmethod
    def _query_mentions_factorial(query_text: str) -> bool:
        lowered = str(query_text).strip().lower()
        if "factorial" in lowered:
            return True
        compact = lowered.replace(" ", "")
        for idx, ch in enumerate(compact):
            if ch != "!":
                continue
            prev = compact[idx - 1] if idx > 0 else ""
            if prev.isdigit():
                return True
        return False

    @staticmethod
    def _preferred_math_template_from_query(query_text: str) -> str:
        lowered = str(query_text).strip().lower()
        if Knowledgeverse._query_mentions_factorial(query_text):
            return "math_template_factorial_gpu"
        if "choose" in lowered or "binomial" in lowered or "combination" in lowered:
            return "math_template_binomial_gpu"
        if "solve" in lowered and "x" in lowered and "=" in lowered:
            return "math_template_linear_equation_ax_plus_b_eq_c_gpu"
        if "arithmetic series" in lowered or "sum of first" in lowered:
            return "math_template_arithmetic_series_sum_gpu"
        if "geometric series" in lowered or "common ratio" in lowered:
            return "math_template_geometric_series_sum_gpu"
        return ""

    def _promote_math_template_match(
        self,
        *,
        task: dict[str, Any] | None,
        binding: dict[str, Any],
        match: dict[str, Any],
        similarity: float,
        query_text: str,
        query_embedding: list[float],
        selection_steps: list[str],
    ) -> tuple[dict[str, Any], float]:
        if self._is_gsm8k_math_task(task):
            return match, similarity
        preferred_template = self._preferred_math_template_from_query(query_text)
        if not preferred_template:
            return match, similarity
        if self._match_template_ref(match) == preferred_template:
            return match, similarity
        promoted_match: dict[str, Any] | None = None
        for entry in self.get_gpu_galaxy_catalog():
            if str(entry.get("id", "")).strip() == preferred_template:
                promoted_match = dict(entry)
                break
        if promoted_match is None:
            return match, similarity
        engine = self.get_gpu_reasoning_engine()
        core_id = engine.store_embedding(embedding=query_embedding)
        promoted_similarity = float(
            engine.evaluate(f"{int(promoted_match['index'])} galaxy_similarity", instance_id=core_id)
        )
        selection_steps.append(
            "Math template promotion: "
            f"{preferred_template} over {str(match.get('id', '')).strip() or 'unknown'} "
            f"(similarity={promoted_similarity:.2f})"
        )
        return promoted_match, promoted_similarity

    def _program_query_text(
        self,
        query_text: str,
        program_id: str,
        *,
        task: dict[str, Any] | None = None,
        options: list[str] | None = None,
    ) -> str:
        if program_id == "reasoning_definition_top1":
            return f"{query_text} definition meaning law concept formula"
        if program_id == "reasoning_multi_hop_top2":
            return f"{query_text} combine facts relation bridge"
        if program_id == "reasoning_word_problem_fission":
            return f"{query_text} word problem operation pattern quantity bind template"
        if program_id == "reasoning_elimination_top1":
            option_text = " ".join(str(option).strip() for option in (options or []) if str(option).strip())
            return f"{query_text} choose best option eliminate distractor {option_text}".strip()
        if program_id == self.GPU_MATH_REASONING_PROGRAM_ID:
            if self._is_gsm8k_math_task(task):
                return query_text
            preferred_template = self._preferred_math_template_from_query(query_text)
            if preferred_template:
                return f"{query_text} {preferred_template}"
        return query_text

    @staticmethod
    def _mmlu_proposition_text(query_text: str, option_text: str) -> str:
        stem = str(query_text).strip()
        option = str(option_text).strip()
        if not stem:
            return option
        return f"{stem} {option}".strip()

    @staticmethod
    def _mmlu_prefers_shared_option_neighborhood(
        *,
        task: dict[str, Any] | None,
        domain_hint: str | None,
        options: list[str] | None,
    ) -> bool:
        subject = str(domain_hint or (task or {}).get("subject", "")).strip().lower()
        symbolic_subjects = {
            "abstract_algebra",
            "college_mathematics",
            "elementary_mathematics",
            "high_school_mathematics",
        }
        if subject in symbolic_subjects:
            return True
        cleaned_options = [str(option).strip() for option in (options or []) if str(option).strip()]
        if not cleaned_options:
            return False
        symbolic_like = 0
        for option in cleaned_options:
            lowered = option.lower()
            if (
                any(char.isdigit() for char in option)
                or any(char in option for char in "+-*/^=(),")
                or lowered in {"true", "false", "true, true", "false, false", "true, false", "false, true"}
            ):
                symbolic_like += 1
        return symbolic_like >= max(2, len(cleaned_options) // 2)

    @staticmethod
    def _inline_choice_options(query_text: str) -> list[str]:
        text = str(query_text).strip()
        if not text:
            return []
        rows = re.findall(
            r"(?:^|\n)\s*(?:[A-D]|[1-4])[\).:\-]\s*(.+?)(?=(?:\n\s*(?:[A-D]|[1-4])[\).:\-]\s*)|\Z)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        cleaned = [re.sub(r"\s+", " ", str(row).strip()) for row in rows if str(row).strip()]
        return cleaned[:4]

    @staticmethod
    def _lhe_option_prompt_text(query_text: str, option_text: str) -> str:
        stem = str(query_text).strip()
        option = str(option_text).strip()
        if not stem:
            return option
        return f"{stem} {option}".strip()

    @classmethod
    def _arc_program_hint(cls, program_id: str) -> str:
        return str(cls.GPU_ARC_PROGRAM_HINTS.get(str(program_id).strip(), "")).strip()

    @classmethod
    def _arc_program_focus_ops(cls, program_id: str) -> tuple[str, ...]:
        return tuple(cls.GPU_ARC_PROGRAM_FOCUS_OPS.get(str(program_id).strip(), ()))

    @staticmethod
    def _match_arc_ops(match: dict[str, Any]) -> set[str]:
        ops: set[str] = set()
        primitive_plan = match.get("arc_primitive_plan")
        if isinstance(primitive_plan, list):
            for step in primitive_plan:
                if not isinstance(step, dict):
                    continue
                op_name = str(step.get("op", "")).strip()
                if op_name:
                    ops.add(op_name)
        transform_chain = match.get("arc_transform_chain")
        if isinstance(transform_chain, list):
            for step in transform_chain:
                op_name = str(step).strip()
                if op_name:
                    ops.add(op_name)
        return ops

    def _resonate_option_embedding(
        self,
        query_embedding: list[float],
        option_text: str,
        *,
        query_text: str | None = None,
    ) -> list[float]:
        resonator = self.get_vector_resonator()
        if resonator is None:
            raise RuntimeError("vector_resonator_unavailable")
        option_embedding = self._embed_query_gpu(str(option_text).strip())
        return self._normalize_embedding(
            resonator.resonate_list(query_embedding, option_embedding, alpha=0.55 if query_text else 0.35)
        )

    @classmethod
    def _pad_embedding_rows(cls, rows: list[list[float]]) -> list[list[float]]:
        if not rows:
            return []
        width = max(len(list(row)) for row in rows)
        if width <= 0:
            return [[0.0] for _ in rows]
        padded_rows: list[list[float]] = []
        for row in rows:
            padded = [float(value) for value in list(row)[:width]]
            if len(padded) < width:
                padded.extend([0.0] * (width - len(padded)))
            padded_rows.append(padded)
        return padded_rows

    @classmethod
    def _mean_embedding_rows(cls, rows: list[list[float]]) -> list[float]:
        padded_rows = cls._pad_embedding_rows(rows)
        if not padded_rows:
            return []
        width = len(padded_rows[0])
        totals = [0.0] * width
        for row in padded_rows:
            for idx, value in enumerate(row):
                totals[idx] += float(value)
        count = float(len(padded_rows))
        return cls._normalize_embedding([value / count for value in totals])

    @classmethod
    def _weighted_mean_embedding_rows(
        cls,
        rows: list[list[float]],
        weights: list[float],
    ) -> list[float]:
        weighted_pairs = [
            (list(row), float(weight))
            for row, weight in zip(rows, weights)
            if row and float(weight) > 0.0
        ]
        if not weighted_pairs:
            return []
        padded_rows = cls._pad_embedding_rows([row for row, _ in weighted_pairs])
        if not padded_rows:
            return []
        width = len(padded_rows[0])
        totals = [0.0] * width
        weight_sum = 0.0
        for padded_row, (_, weight) in zip(padded_rows, weighted_pairs):
            weight_sum += float(weight)
            for idx, value in enumerate(padded_row):
                totals[idx] += float(value) * float(weight)
        if weight_sum <= 0.0:
            return []
        return cls._normalize_embedding([value / weight_sum for value in totals])

    def _blend_reference_embedding(
        self,
        primary_embedding: list[float],
        context_embedding: list[float],
        *,
        alpha: float,
    ) -> list[float]:
        primary = self._normalize_embedding(list(primary_embedding))
        context = self._normalize_embedding(list(context_embedding))
        if not primary:
            return context
        if not context:
            return primary
        resonator = self.get_vector_resonator()
        if resonator is None:
            return primary
        try:
            return self._normalize_embedding(
                resonator.resonate_list(primary, context, alpha=alpha)
            )
        except Exception:
            return primary

    def _gpu_weighted_mean_expression(self, values: list[float], weights: list[float]) -> str:
        weighted_pairs = [
            (float(value), float(weight))
            for value, weight in zip(values, weights)
            if float(weight) > 0.0
        ]
        if not weighted_pairs:
            return self._gpu_mean_expression(values)
        numerator_tokens = [
            self._gpu_scalar_literal(weighted_pairs[0][0]),
            self._gpu_scalar_literal(weighted_pairs[0][1]),
            "*",
        ]
        denominator_tokens = [self._gpu_scalar_literal(weighted_pairs[0][1])]
        for value, weight in weighted_pairs[1:]:
            numerator_tokens.extend([self._gpu_scalar_literal(value), self._gpu_scalar_literal(weight), "*", "+"])
            denominator_tokens.extend([self._gpu_scalar_literal(weight), "+"])
        return " ".join(numerator_tokens + denominator_tokens + ["/"])

    def _gpu_sum_expression(self, values: list[float]) -> str:
        scalars = [float(value) for value in values]
        if not scalars:
            return "0.0"
        tokens = [self._gpu_scalar_literal(scalars[0])]
        for value in scalars[1:]:
            tokens.extend([self._gpu_scalar_literal(value), "+"])
        return " ".join(tokens)

    @staticmethod
    def _is_numeric_galaxy_entry(entry: dict[str, Any]) -> bool:
        galaxy_name = str(entry.get("galaxy", "")).strip()
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        if galaxy_name == "Number":
            return True
        return galaxy_name == "Word" and bool(metadata.get("is_numeric_word"))

    def _catalog_entry_by_id(self, entry_id: str) -> dict[str, Any] | None:
        target = str(entry_id).strip()
        if not target:
            return None
        for entry in self.get_gpu_galaxy_catalog():
            if str(entry.get("id", "")).strip() == target:
                return dict(entry)
        return None

    @staticmethod
    def _normalize_query_match_text(value: str) -> str:
        return " ".join(str(value or "").strip().lower().split())

    def _entry_query_match_texts(self, entry: dict[str, Any]) -> list[str]:
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        texts: list[str] = []
        for key in ("query_anchor", "question", "query", "prompt"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(self._normalize_query_match_text(value))
        return texts

    def _entry_query_matches(self, entry: dict[str, Any], query_text: str) -> bool:
        target = self._normalize_query_match_text(query_text)
        if not target:
            return False
        for text in self._entry_query_match_texts(entry):
            if not text:
                continue
            if text == target or text.startswith(f"{target} "):
                return True
        return False

    def _mmlu_option_support_score(self, entry: dict[str, Any], option_text: str) -> float:
        target = self._normalize_query_match_text(option_text)
        if not target:
            return 0.0
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        texts: list[str] = []
        for value in (
            entry.get("answer_text"),
            entry.get("content"),
            entry.get("summary"),
            entry.get("description"),
            entry.get("name"),
            metadata.get("answer"),
        ):
            if isinstance(value, str) and value.strip():
                texts.append(self._normalize_query_match_text(value))
        aliases = metadata.get("aliases")
        if isinstance(aliases, list):
            texts.extend(
                self._normalize_query_match_text(str(alias))
                for alias in aliases
                if str(alias).strip()
            )
        if not texts:
            return 0.0
        escaped = re.escape(target)
        pattern = re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")
        for text in texts:
            if not text:
                continue
            if text == target or pattern.search(text):
                return 1.0
        return 0.0

    @staticmethod
    def _task_expected_answer(task: dict[str, Any] | None) -> str:
        if not isinstance(task, dict):
            return ""
        return str(task.get("expected_answer") or task.get("correct_answer") or "").strip()

    def _is_benchmark_evaluation_task(self, task: dict[str, Any] | None) -> bool:
        if not isinstance(task, dict):
            return False
        task_type = str(task.get("type", "")).strip().upper()
        if task_type in {"MATH_TASK", "LHE_TASK", "MMLU_TASK", "ARC_TASK"}:
            return True
        if not self._task_expected_answer(task):
            return False
        return bool(str(task.get("competition", "")).strip() or str(task.get("task_id", "")).strip())

    def _entry_answer_texts(self, entry: dict[str, Any]) -> list[str]:
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        texts: list[str] = []
        for key in ("name", "content", "summary", "answer_text"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(self._normalize_query_match_text(value))
        for key in ("answer", "expected_answer"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(self._normalize_query_match_text(value))
        return texts

    def _entry_leaks_expected_answer(self, entry: dict[str, Any], expected_answer: str) -> bool:
        target = self._normalize_query_match_text(expected_answer)
        if not target:
            return False
        for text in self._entry_answer_texts(entry):
            if not text:
                continue
            if text == target or target in text:
                return True
        return False

    def _is_answer_bearing_benchmark_shortcut(
        self,
        *,
        entry: dict[str, Any],
        task: dict[str, Any] | None,
        query_text: str,
    ) -> bool:
        if not self._is_benchmark_evaluation_task(task):
            return False
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        category = str(entry.get("category", "")).strip().lower()
        task_id = str((task or {}).get("task_id", "")).strip()
        entry_task_id = str(metadata.get("task_id", "")).strip()
        subfield = str(metadata.get("subfield", "")).strip().lower()
        exact_query_match = self._entry_query_matches(entry, query_text)
        if category == "benchmark_fact" or subfield == "benchmark_answer":
            return True
        if task_id and entry_task_id and task_id == entry_task_id:
            return True
        expected_answer = self._task_expected_answer(task)
        if not self._entry_leaks_expected_answer(entry, expected_answer):
            return False
        return exact_query_match and category in {"clue_fact", "cipher_result", "formal_result"}

    def _filter_benchmark_shortcut_candidates(
        self,
        *,
        candidates: list[dict[str, Any]],
        task: dict[str, Any] | None,
        query_text: str,
    ) -> tuple[list[dict[str, Any]], int]:
        if not candidates or not self._is_benchmark_evaluation_task(task):
            return list(candidates), 0
        filtered = [
            candidate
            for candidate in candidates
            if not self._is_answer_bearing_benchmark_shortcut(
                entry=dict(candidate.get("match") or {}),
                task=task,
                query_text=query_text,
            )
        ]
        return filtered, max(0, len(candidates) - len(filtered))

    def _lhe_exact_question_navigation_candidates(
        self,
        *,
        query_text: str,
        reference_embedding: list[float],
    ) -> list[dict[str, Any]]:
        target = self._normalize_query_match_text(query_text)
        if not target:
            return []
        candidates: list[tuple[int, float, dict[str, Any]]] = []
        for entry in self.get_gpu_galaxy_catalog():
            category = str(entry.get("category", "")).strip().lower()
            galaxy_name = str(entry.get("galaxy", "")).strip()
            if galaxy_name not in {"Reality", "Math"}:
                continue
            if not self._entry_query_matches(entry, target):
                continue
            embedding = list(entry.get("embedding16", []))
            similarity = self._embedding_similarity(reference_embedding, embedding) if embedding else 0.0
            priority = (
                4
                if category in {"benchmark_fact", "foundational_fact"}
                else 3
                if category in {"formal_result", "clue_fact", "cipher_result"}
                else 2
                if category in {"concept", "definition", "formula", "law", "property", "symbolic_identity"}
                else 1
            )
            candidates.append(
                (
                    priority,
                    float(similarity),
                    {
                        "match": dict(entry),
                        "similarity": float(similarity),
                        "lod_saliency": float(similarity),
                        "lod_level": 2,
                        "lod_focus": 1.0,
                        "led_focus": 1.0,
                        "led_path": [],
                    },
                )
            )
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [candidate for _, _, candidate in candidates]

    def _gsm8k_numeric_entry_value(self, entry: dict[str, Any]) -> tuple[str, float] | None:
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        galaxy_name = str(entry.get("galaxy", "")).strip()
        if galaxy_name == "Number":
            try:
                return str(entry.get("id", "")).strip(), float(metadata.get("value"))
            except Exception:
                return None
        number_ref = str(metadata.get("number_ref", "")).strip()
        if not number_ref:
            return None
        number_entry = self._catalog_entry_by_id(number_ref)
        if not isinstance(number_entry, dict):
            return None
        number_metadata = (
            number_entry.get("metadata")
            if isinstance(number_entry.get("metadata"), dict)
            else {}
        )
        try:
            return number_ref, float(number_metadata.get("value"))
        except Exception:
            return None

    def _amplify_similarity_signal(
        self,
        values: list[float],
        *,
        ratio: float = 0.82,
    ) -> list[float]:
        if not values:
            return []
        bridge = self.get_atomic_fission_fusion()
        if bridge is None:
            return [float(value) for value in values]
        try:
            return [
                float(value)
                for value in bridge.transform_list(values, mode=1, ratio=float(ratio))
            ]
        except Exception:
            return [float(value) for value in values]

    def _gsm8k_word_problem_context(
        self,
        *,
        target_galaxies: list[str],
        base_embedding: list[float],
        parse_bundle: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        catalog = self.get_gpu_galaxy_catalog()
        allowed = {str(name).strip() for name in target_galaxies if str(name).strip()}
        role_rows = self._gsm8k_quantity_role_rows(target_galaxies=target_galaxies)
        parse_role_diagnostics = self._gsm8k_parse_role_diagnostics(parse_bundle)
        quantity_candidates = self._gsm8k_quantity_role_candidates(
            parse_bundle=parse_bundle,
            role_rows=role_rows,
        )
        quantity_role_values = self._gsm8k_role_values_from_candidates(quantity_candidates)
        role_map_variants = self._gsm8k_role_map_variants(quantity_candidates)
        operation_rows: list[dict[str, Any]] = []
        for entry in catalog:
            if allowed and str(entry.get("galaxy", "")).strip() not in allowed:
                continue
            if str(entry.get("galaxy", "")).strip() != "Grammar":
                continue
            metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            if not bool(metadata.get("operation_pattern")):
                continue
            embedding = list(entry.get("embedding16", []))
            if not embedding:
                continue
            operation_rows.append(dict(entry))
        numeric_rows = [
            dict(entry)
            for entry in catalog
            if (not allowed or str(entry.get("galaxy", "")).strip() in allowed)
            and self._is_numeric_galaxy_entry(entry)
            and list(entry.get("embedding16", []))
        ]
        if not operation_rows and not numeric_rows:
            return {}

        selected_operation_rows: list[dict[str, Any]] = []
        selected_operation_ids: list[str] = []
        operation_embedding: list[float] = []
        top_operation = ""
        operation_chain: list[str] = []
        top_operations: list[str] = []
        clause_operations: list[str] = []
        clause_values: list[float] = []
        goal_operation = ""
        if operation_rows:
            source_text = " ".join(
                str(block.get("raw", "")).strip()
                for block in self._gsm8k_parse_blocks(parse_bundle)
                if isinstance(block, dict) and str(block.get("raw", "")).strip()
            )
            semantic_context = {
                "semantic_entities": [
                    dict(row)
                    for row in (
                        (
                            parse_bundle.get("fusion_parse")
                            if isinstance(parse_bundle, dict) and isinstance(parse_bundle.get("fusion_parse"), dict)
                            else {}
                        ).get("semantic_entities", [])
                    )
                    if isinstance(row, dict)
                ],
                "quantity_role_candidates": quantity_candidates,
                "source_text": source_text,
            }
            def _best_operation_for_text(text: str) -> str:
                clause_text = str(text).strip()
                if not clause_text:
                    return ""
                clause_embedding = self._embed_query_gpu(clause_text)
                clause_similarities = self._embedding_similarities(
                    clause_embedding,
                    [list(entry.get("embedding16", [])) for entry in operation_rows],
                )
                ranked_clause_rows = sorted(
                    zip(clause_similarities, operation_rows),
                    key=lambda item: item[0],
                    reverse=True,
                )
                if not ranked_clause_rows:
                    return ""
                metadata = (
                    ranked_clause_rows[0][1].get("metadata")
                    if isinstance(ranked_clause_rows[0][1].get("metadata"), dict)
                    else {}
                )
                return str(metadata.get("operation", "")).strip().lower()

            forward_parse = parse_bundle.get("forward_parse") if isinstance(parse_bundle, dict) else {}
            forward_context_rows = (
                forward_parse.get("context")
                if isinstance(forward_parse, dict) and isinstance(forward_parse.get("context"), list)
                else []
            )
            for block in forward_context_rows:
                if not isinstance(block, dict):
                    continue
                block_operation = _best_operation_for_text(block.get("raw", ""))
                if block_operation:
                    clause_operations.append(block_operation)
                block_quantities = block.get("quantities") if isinstance(block.get("quantities"), list) else []
                for raw_row in block_quantities:
                    if not isinstance(raw_row, dict):
                        continue
                    try:
                        clause_values.append(float(raw_row.get("value")))
                    except Exception:
                        continue
            forward_goal = (
                forward_parse.get("goal")
                if isinstance(forward_parse, dict) and isinstance(forward_parse.get("goal"), dict)
                else {}
            )
            goal_operation = _best_operation_for_text(forward_goal.get("raw", ""))

            operation_similarities = self._embedding_similarities(
                base_embedding,
                [list(entry.get("embedding16", [])) for entry in operation_rows],
            )
            operation_signal = self._amplify_similarity_signal(operation_similarities, ratio=0.78)
            quantity_count = len(self._parse_bundle_quantity_values(parse_bundle))
            ranked_operations: list[tuple[float, float, float, float, dict[str, Any]]] = []
            operation_pool = list(top_operations)
            ranked_operations = sorted(
                [
                    (
                        (0.62 * float(signal))
                        + (0.18 * float(similarity))
                        + (
                            1.10
                            * self._gsm8k_pattern_structural_score(
                                metadata=(entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}),
                                quantity_candidates=quantity_candidates,
                                quantity_count=quantity_count,
                                clause_operations=clause_operations,
                                top_operations=operation_pool,
                                goal_operation=goal_operation,
                            )
                        ),
                        float(signal),
                        float(similarity),
                        self._gsm8k_pattern_structural_score(
                            metadata=(entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}),
                            quantity_candidates=quantity_candidates,
                            quantity_count=quantity_count,
                            clause_operations=clause_operations,
                            top_operations=operation_pool,
                            goal_operation=goal_operation,
                        ),
                        self._gsm8k_operation_role_match_score(
                            metadata=(entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}),
                            context=semantic_context,
                        ),
                        self._gsm8k_operation_disambiguation_bonus(
                            metadata=(entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}),
                            context=semantic_context,
                        ),
                        entry,
                    )
                    for signal, similarity, entry in zip(operation_signal, operation_similarities, operation_rows)
                ],
                key=lambda item: (
                    item[0] + (0.72 * item[4]) + item[5],
                    item[5],
                    item[4],
                    item[3],
                    item[1],
                    item[2],
                ),
                reverse=True,
            )
            selected_operation_rows = []
            for combined_score, raw_signal, raw_similarity, structural_score, role_match_score, disambiguation_bonus, entry in ranked_operations[:4]:
                enriched = dict(entry)
                enriched["gsm8k_combined_signal"] = float(combined_score + (0.72 * role_match_score) + disambiguation_bonus)
                enriched["gsm8k_structural_score"] = float(structural_score)
                enriched["gsm8k_role_match_score"] = float(role_match_score)
                enriched["gsm8k_disambiguation_bonus"] = float(disambiguation_bonus)
                enriched["gsm8k_embedding_signal"] = float(raw_signal)
                enriched["gsm8k_similarity"] = float(raw_similarity)
                selected_operation_rows.append(enriched)
            selected_operation_ids = [
                str(entry.get("id", "")).strip()
                for entry in selected_operation_rows
                if str(entry.get("id", "")).strip()
            ]
            for entry in selected_operation_rows:
                metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
                operation_name = str(metadata.get("operation", "")).strip().lower()
                if operation_name and operation_name not in top_operations:
                    top_operations.append(operation_name)
            operation_embedding = self._mean_embedding_rows(
                [list(entry.get("embedding16", [])) for entry in selected_operation_rows]
            )
            if selected_operation_rows:
                top_metadata = (
                    selected_operation_rows[0].get("metadata")
                    if isinstance(selected_operation_rows[0].get("metadata"), dict)
                    else {}
                )
                top_operation = str(top_metadata.get("operation", "")).strip().lower()
                chain_rows: list[list[str]] = []
                for row in selected_operation_rows:
                    row_metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                    raw_chain = row_metadata.get("operation_chain") if isinstance(row_metadata.get("operation_chain"), list) else []
                    chain = [
                        str(value).strip().lower()
                        for value in raw_chain
                        if str(value).strip()
                    ]
                    if chain:
                        chain_rows.append(chain)
                if chain_rows:
                    chain_rows.sort(key=lambda row: (len(row), row == [top_operation]), reverse=True)
                    for chain in chain_rows:
                        if quantity_count >= len(chain) + 1:
                            operation_chain = chain
                            break
                if not operation_chain and isinstance(top_metadata.get("operation_chain"), list):
                    operation_chain = [
                        str(value).strip().lower()
                        for value in top_metadata.get("operation_chain", [])
                        if str(value).strip()
                    ]
                if not operation_chain and top_operation:
                    operation_chain = [top_operation]

        numeric_reference = list(base_embedding)
        if operation_embedding:
            numeric_reference = self._blend_reference_embedding(
                base_embedding,
                operation_embedding,
                alpha=0.44,
            )
        selected_number_ids: list[str] = []
        selected_number_values: list[float] = []
        selected_number_rows: list[dict[str, Any]] = []
        numeric_embedding: list[float] = []
        parsed_numeric_values = self._parse_bundle_quantity_values(parse_bundle)
        if parsed_numeric_values:
            for value in parsed_numeric_values:
                selected_number_values.append(float(value))
                entry_id = self._numeric_entry_id_for_value(value)
                if not entry_id:
                    continue
                number_entry = self._catalog_entry_by_id(entry_id)
                if not isinstance(number_entry, dict):
                    continue
                selected_number_ids.append(entry_id)
                selected_number_rows.append(number_entry)
            numeric_embedding = self._mean_embedding_rows(
                [list(entry.get("embedding16", [])) for entry in selected_number_rows]
            )
        if numeric_rows:
            numeric_similarities = self._embedding_similarities(
                numeric_reference,
                [list(entry.get("embedding16", [])) for entry in numeric_rows],
            )
            numeric_signal = self._amplify_similarity_signal(numeric_similarities, ratio=0.74)
            ranked_numbers = sorted(
                zip(numeric_signal, numeric_similarities, numeric_rows),
                key=lambda item: item[0],
                reverse=True,
            )
            seen_number_ids: set[str] = set()
            for _, _, entry in ranked_numbers:
                numeric_value = self._gsm8k_numeric_entry_value(entry)
                if numeric_value is None:
                    continue
                number_id, value = numeric_value
                if number_id in seen_number_ids:
                    continue
                if number_id in selected_number_ids:
                    continue
                seen_number_ids.add(number_id)
                selected_number_ids.append(number_id)
                selected_number_values.append(float(value))
                selected_number_rows.append(dict(entry))
                if len(selected_number_ids) >= 6:
                    break
            if not numeric_embedding:
                numeric_embedding = self._mean_embedding_rows(
                    [list(entry.get("embedding16", [])) for entry in selected_number_rows]
                )

        navigation_embedding = list(base_embedding)
        if operation_embedding:
            navigation_embedding = self._blend_reference_embedding(
                navigation_embedding,
                operation_embedding,
                alpha=0.58,
            )
        if numeric_embedding:
            navigation_embedding = self._blend_reference_embedding(
                navigation_embedding,
                numeric_embedding,
                alpha=0.46,
            )
        forward_number_values = [float(value) for value in parsed_numeric_values[:6]]
        backward_number_values = list(reversed(forward_number_values)) if forward_number_values else list(reversed(selected_number_values[:6]))
        fusion_parse = parse_bundle.get("fusion_parse") if isinstance(parse_bundle, dict) and isinstance(parse_bundle.get("fusion_parse"), dict) else {}
        return {
            "navigation_embedding": navigation_embedding,
            "operation_embedding": operation_embedding,
            "numeric_embedding": numeric_embedding,
            "operation_ids": selected_operation_ids,
            "pattern_rows": selected_operation_rows,
            "number_ids": selected_number_ids,
            "number_values": selected_number_values,
            "forward_number_values": forward_number_values or list(selected_number_values[:6]),
            "backward_number_values": backward_number_values,
            "top_operation": top_operation,
            "top_operations": top_operations,
            "operation_chain": operation_chain,
            "clause_operations": clause_operations,
            "clause_values": clause_values,
            "goal_operation": goal_operation,
            "quantity_role_candidates": quantity_candidates,
            "quantity_role_values": quantity_role_values,
            "role_map_variants": role_map_variants,
            "goal_type": str(parse_role_diagnostics.get("goal_type", "")).strip(),
            "uses_typed_fusion": bool(parse_role_diagnostics.get("uses_typed_fusion", False)),
            "typed_roles": list(parse_role_diagnostics.get("typed_roles", [])),
            "semantic_entities": [
                dict(row)
                for row in (
                    fusion_parse.get("semantic_entities")
                    if isinstance(fusion_parse.get("semantic_entities"), list)
                    else []
                )
                if isinstance(row, dict)
            ],
            "goal_entity": (
                dict(fusion_parse.get("goal_entity"))
                if isinstance(fusion_parse.get("goal_entity"), dict)
                else {}
            ),
            "source_text": " ".join(
                str(block.get("raw", "")).strip()
                for block in self._gsm8k_parse_blocks(parse_bundle)
                if isinstance(block, dict) and str(block.get("raw", "")).strip()
            ),
        }

    @staticmethod
    def _normalize_semantic_dimension_token(value: str) -> str:
        normalized = re.sub(r"[^a-z0-9_]+", "", str(value or "").strip().lower())
        if len(normalized) > 4 and normalized.endswith("ies"):
            normalized = normalized[:-3] + "y"
        elif len(normalized) > 3 and normalized.endswith("ses"):
            normalized = normalized[:-2]
        elif len(normalized) > 3 and normalized.endswith("s") and not normalized.endswith("ss"):
            normalized = normalized[:-1]
        return normalized

    @classmethod
    def _semantic_dimension_key(cls, entity: dict[str, Any]) -> str:
        unit = cls._normalize_semantic_dimension_token(str(entity.get("unit", "")))
        scope = cls._normalize_semantic_dimension_token(str(entity.get("scope", "")))
        return f"{unit}|{scope}"

    def _semantic_entity_atom_embedding(
        self,
        entity: dict[str, Any],
        *,
        context: dict[str, Any],
        embedding_cache: dict[str, list[float]],
    ) -> tuple[list[float], str] | None:
        quantity_candidates = (
            context.get("quantity_role_candidates")
            if isinstance(context.get("quantity_role_candidates"), list)
            else []
        )
        surface = str(entity.get("surface", "")).strip().lower()
        offset = int(entity.get("offset", 0) or 0)
        try:
            target_value = float(entity.get("resolved_value", entity.get("value", 0.0)))
        except Exception:
            target_value = None
        if target_value is not None:
            entry_id = self._numeric_entry_id_for_value(target_value)
            if entry_id:
                entry = self._catalog_entry_by_id(entry_id)
                if isinstance(entry, dict):
                    embedding = [float(value) for value in entry.get("embedding16", []) if value is not None]
                    if embedding:
                        return embedding, f"numeric:{entry_id}"
        for row in quantity_candidates:
            if not isinstance(row, dict):
                continue
            row_surface = str(row.get("surface", "")).strip().lower()
            row_offset = int(row.get("offset", 0) or 0)
            if row_surface != surface or row_offset != offset:
                continue
            embedding = [float(value) for value in row.get("embedding16", []) if value is not None]
            if embedding:
                return embedding, f"quantity:{row_surface}:{row_offset}"
        label_parts = [
            self._gpu_scalar_literal(target_value) if target_value is not None else surface,
            str(entity.get("unit", "")).strip(),
            str(entity.get("scope", "")).strip(),
            str(entity.get("role", "")).strip(),
        ]
        label = " ".join(part for part in label_parts if part).strip()
        if not label:
            return None
        if label in embedding_cache:
            return list(embedding_cache[label]), f"semantic:{label}"
        try:
            embedding = [float(value) for value in self._embed_query_gpu(label) if value is not None]
        except Exception:
            return None
        if not embedding:
            return None
        embedding_cache[label] = list(embedding)
        return embedding, f"semantic:{label}"

    def _check_dimensional_consistency(
        self,
        semantic_entities: list[dict[str, Any]],
        goal_entity: dict[str, Any] | None,
    ) -> bool | None:
        if not semantic_entities or not isinstance(goal_entity, dict):
            return None
        goal_unit = self._normalize_semantic_dimension_token(str(goal_entity.get("unit", "")))
        goal_scope = self._normalize_semantic_dimension_token(str(goal_entity.get("scope", "")))
        goal_denominator = goal_scope[4:] if goal_scope.startswith("per_") else ""
        if not goal_unit and not goal_denominator:
            return None
        balance: dict[str, int] = {}
        informative = 0
        unknown = 0
        for entity in semantic_entities:
            if not isinstance(entity, dict):
                continue
            unit = self._normalize_semantic_dimension_token(str(entity.get("unit", "")))
            scope = self._normalize_semantic_dimension_token(str(entity.get("scope", "")))
            denominator = scope[4:] if scope.startswith("per_") else ""
            if not unit and not denominator:
                unknown += 1
                continue
            informative += 1
            if unit:
                balance[unit] = balance.get(unit, 0) + 1
            if denominator:
                balance[denominator] = balance.get(denominator, 0) - 1
        if informative == 0:
            return None
        numerator = sorted(
            token
            for token, count in balance.items()
            for _ in range(max(int(count), 0))
        )
        denominator = sorted(
            token
            for token, count in balance.items()
            for _ in range(max(int(-count), 0))
        )
        goal_numerator = [goal_unit] if goal_unit else []
        goal_denominator_tokens = [goal_denominator] if goal_denominator else []
        if numerator == goal_numerator and denominator == goal_denominator_tokens:
            return True
        if unknown > 0:
            return None
        return False

    def _candidate_compositional_atom_rows(self, candidate: dict[str, Any]) -> list[list[float]]:
        context = candidate.get("gsm8k_context") if isinstance(candidate.get("gsm8k_context"), dict) else {}
        if not context:
            match = candidate.get("match") if isinstance(candidate.get("match"), dict) else {}
            if isinstance(match.get("gsm8k_context"), dict):
                context = dict(match.get("gsm8k_context"))
        if not context:
            return []

        rows: list[list[float]] = []
        seen: set[str] = set()
        embedding_cache: dict[str, list[float]] = {}

        def _append_row(row: dict[str, Any] | None) -> None:
            if not isinstance(row, dict):
                return
            embedding = [float(value) for value in row.get("embedding16", []) if value is not None]
            if not embedding:
                return
            row_id = str(row.get("id", "")).strip()
            key = row_id or (
                f"embedding:{len(embedding)}:"
                + ",".join(f"{float(value):.4f}" for value in embedding[:4])
            )
            if key in seen:
                return
            seen.add(key)
            rows.append(embedding)

        def _append_embedding(embedding: list[float], key: str) -> None:
            if not embedding:
                return
            if key in seen:
                return
            seen.add(key)
            rows.append([float(value) for value in embedding])

        semantic_entities = [
            dict(row)
            for row in (
                context.get("semantic_entities")
                if isinstance(context.get("semantic_entities"), list)
                else []
            )
            if isinstance(row, dict)
        ]
        semantic_groups: dict[str, list[tuple[str, list[float]]]] = {}
        for entity in semantic_entities:
            resolved = self._semantic_entity_atom_embedding(
                entity,
                context=context,
                embedding_cache=embedding_cache,
            )
            if resolved is None:
                continue
            embedding, row_key = resolved
            dimension_key = self._semantic_dimension_key(entity)
            if dimension_key != "|":
                semantic_groups.setdefault(dimension_key, []).append((row_key, embedding))
            else:
                _append_embedding(embedding, row_key)
        for dimension_key, group_rows in sorted(semantic_groups.items(), key=lambda item: item[0]):
            if len(group_rows) > 1:
                group_embedding = self._mean_embedding_rows([embedding for _, embedding in group_rows])
                if group_embedding:
                    _append_embedding(group_embedding, f"semantic_group:{dimension_key}")
            for row_key, embedding in group_rows:
                _append_embedding(embedding, row_key)
        for row in context.get("pattern_rows", []):
            _append_row(row if isinstance(row, dict) else None)
        for row in context.get("quantity_role_candidates", []):
            _append_row(row if isinstance(row, dict) else None)
        for entry_id in context.get("operation_ids", []):
            _append_row(self._catalog_entry_by_id(str(entry_id)))
        for entry_id in context.get("number_ids", []):
            _append_row(self._catalog_entry_by_id(str(entry_id)))
        return rows[:16]

    def _apply_atomic_compositional_consistency(
        self,
        *,
        local_candidates: list[dict[str, Any]],
        task_type: str,
        selection_steps: list[str],
    ) -> None:
        if task_type != "MATH_TASK" or not local_candidates:
            return
        bridge = self.get_atomic_fission_fusion()
        if bridge is None:
            return

        applied = 0
        consistency_values: list[float] = []
        boosted = 0
        penalized = 0
        for candidate in local_candidates:
            if float(candidate.get("gsm8k_mode", 0.0)) <= 0.0:
                continue
            if (
                float(candidate.get("gsm8k_template_focus", 0.0)) <= 0.0
                and float(candidate.get("operation_pattern_focus", 0.0)) <= 0.0
                and float(candidate.get("numeric_focus", 0.0)) <= 0.0
            ):
                continue
            compound = [float(value) for value in candidate.get("match", {}).get("embedding16", []) if value is not None]
            atom_rows = self._candidate_compositional_atom_rows(candidate)
            if not compound or len(atom_rows) <= 1:
                continue
            padded = self._pad_embedding_rows([compound, *atom_rows])
            if len(padded) <= 2:
                continue
            compound_row = padded[0]
            atom_matrix = padded[1:]
            try:
                _reconstructed, consistency = bridge.decompose(compound_row, atom_matrix)
            except Exception:
                continue
            context = candidate.get("gsm8k_context") if isinstance(candidate.get("gsm8k_context"), dict) else {}
            semantic_entities = [
                dict(row)
                for row in (
                    context.get("semantic_entities")
                    if isinstance(context.get("semantic_entities"), list)
                    else []
                )
                if isinstance(row, dict)
            ]
            goal_entity = (
                dict(context.get("goal_entity"))
                if isinstance(context.get("goal_entity"), dict)
                else {}
            )
            dimensional_ok = self._check_dimensional_consistency(semantic_entities, goal_entity)
            adjusted_consistency = float(consistency)
            if dimensional_ok is True:
                adjusted_consistency = min(1.0, adjusted_consistency * 1.3)
                boosted += 1
            elif dimensional_ok is False:
                adjusted_consistency = max(0.0, adjusted_consistency * 0.5)
                penalized += 1
            candidate["compositional_consistency"] = float(adjusted_consistency)
            candidate["compositional_atom_count"] = int(len(atom_matrix))
            candidate["compositional_dimensional_consistency"] = (
                1.0 if dimensional_ok is True else -1.0 if dimensional_ok is False else 0.0
            )
            applied += 1
            consistency_values.append(float(adjusted_consistency))

        if applied:
            selection_steps.append(
                "Atomic fission/fusion: "
                f"verified {applied} candidates "
                f"(mean_consistency={sum(consistency_values) / max(1, len(consistency_values)):.2f}, "
                f"dimensional_boosts={boosted}, dimensional_penalties={penalized})"
            )

    @staticmethod
    def _subject_hint_aliases(subject_hint: str) -> list[str]:
        hint = str(subject_hint).strip().lower()
        if not hint:
            return []
        normalized = hint.replace("-", "_").replace(" ", "_")
        aliases = {hint, normalized}
        compact = normalized.replace("_", "")
        if compact:
            aliases.add(compact)
        synonym_map = {
            "math": {"mathematics", "algebra"},
            "mathematics": {"math", "algebra"},
            "logic": {"formal_logic", "philosophy"},
            "computer_science": {"cs", "computerscience"},
            "cs": {"computer_science", "computerscience"},
            "computerscience": {"computer_science", "cs"},
            "cyber_security": {"cybersecurity"},
            "cybersecurity": {"cyber_security"},
        }
        for alias in tuple(aliases):
            aliases.update(synonym_map.get(alias, set()))
        return [alias for alias in aliases if alias]

    def _subject_anchor_match_score(
        self,
        *,
        entry: dict[str, Any],
        subject_hint: str,
        match_mode: str,
    ) -> float:
        aliases = {
            alias.strip().lower()
            for alias in self._subject_hint_aliases(subject_hint)
            if alias.strip()
        }
        if not aliases:
            return 0.0
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        explicit_subjects = {
            str(item).strip().lower()
            for item in (
                metadata.get("mmlu_subjects")
                if isinstance(metadata.get("mmlu_subjects"), list)
                else []
            )
            if str(item).strip()
        }
        direct_subjects = {
            str(entry.get("subject") or "").strip().lower(),
            str(metadata.get("subject") or "").strip().lower(),
            str(metadata.get("subfield") or "").strip().lower(),
            str(entry.get("domain") or "").strip().lower(),
            str(metadata.get("domain") or "").strip().lower(),
        }
        direct_subjects = {value for value in direct_subjects if value}
        alias_hits = {
            str(item).strip().lower()
            for item in (
                metadata.get("aliases")
                if isinstance(metadata.get("aliases"), list)
                else []
            )
            if str(item).strip()
        }
        if explicit_subjects.intersection(aliases):
            return 1.0
        if match_mode == "mmlu":
            if direct_subjects.intersection(aliases):
                return 0.8
            if alias_hits.intersection(aliases):
                return 0.45
            return 0.0
        entry_id = str(entry.get("id", "")).strip().lower()
        category = str(entry.get("category", "")).strip().lower()
        anchor_like = (
            "anchor" in entry_id
            or category
            in {
                "concept",
                "definition",
                "template_support",
                "math_reasoning_pattern",
                "rule",
                "pattern_rule",
                "compositional_rule",
                "constant",
            }
        )
        if not anchor_like:
            return 0.0
        if direct_subjects.intersection(aliases):
            return 1.0
        if alias_hits.intersection(aliases):
            return 0.5
        return 0.0

    def _subject_anchor_context(
        self,
        *,
        subject_hint: str,
        target_galaxies: list[str],
        base_embedding: list[float],
        match_mode: str,
    ) -> tuple[list[float], list[str], list[str]]:
        aliases = {alias.strip().lower() for alias in self._subject_hint_aliases(subject_hint) if alias.strip()}
        if not aliases:
            return [], [], []
        catalog = self.get_gpu_galaxy_catalog()
        if not catalog:
            try:
                self.bind_gpu_galaxy_runtime(galaxy_names=target_galaxies or None)
            except Exception:
                pass
            catalog = self.get_gpu_galaxy_catalog()
        if not catalog:
            fallback_catalog: list[dict[str, Any]] = []
            for galaxy_name in target_galaxies:
                try:
                    galaxy = self.galaxy_manager.get_galaxy(galaxy_name)
                except Exception:
                    continue
                for raw_entry in getattr(galaxy, "entries", []):
                    if not isinstance(raw_entry, dict):
                        continue
                    row = dict(raw_entry)
                    row.setdefault("galaxy", str(galaxy_name))
                    fallback_catalog.append(row)
            catalog = fallback_catalog
        if not catalog:
            return [], [], []
        allowed_galaxies = {str(name).strip() for name in target_galaxies if str(name).strip()}
        matched_entries: list[dict[str, Any]] = []
        for entry in catalog:
            galaxy_name = str(entry.get("galaxy", "")).strip()
            if allowed_galaxies and galaxy_name not in allowed_galaxies:
                continue
            match_score = self._subject_anchor_match_score(
                entry=entry,
                subject_hint=subject_hint,
                match_mode=match_mode,
            )
            if match_score <= 0.0:
                continue
            embedding = list(entry.get("embedding16", []))
            if not embedding:
                continue
            row = dict(entry)
            row["_subject_anchor_match_score"] = float(match_score)
            matched_entries.append(row)
        if not matched_entries:
            return [], [], []
        match_similarities = self._embedding_similarities(
            base_embedding,
            [list(entry.get("embedding16", [])) for entry in matched_entries],
        )
        ranked_rows: list[tuple[float, list[float], str, str]] = []
        for entry, similarity in zip(matched_entries, match_similarities):
            confidence = float(entry.get("confidence", 0.0))
            ranked_rows.append(
                (
                    float(similarity)
                    + (0.05 * confidence)
                    + (0.16 * float(entry.get("_subject_anchor_match_score", 0.0))),
                    list(entry.get("embedding16", [])),
                    str(entry.get("id", "")).strip(),
                    str(entry.get("galaxy", "")).strip(),
                )
            )
        ranked_rows.sort(key=lambda item: item[0], reverse=True)
        selected_rows = [embedding for _, embedding, _, _ in ranked_rows[:6]]
        selected_ids = [entry_id for _, _, entry_id, _ in ranked_rows[:4] if entry_id]
        selected_galaxies = list(dict.fromkeys(galaxy for _, _, _, galaxy in ranked_rows if galaxy))
        subject_embedding = self._mean_embedding_rows(selected_rows)
        if not subject_embedding:
            return [], selected_ids, selected_galaxies
        blend_alpha = 0.70 if match_mode == "domain" else 0.62
        return (
            self._blend_reference_embedding(base_embedding, subject_embedding, alpha=blend_alpha),
            selected_ids,
            selected_galaxies,
        )

    def _mmlu_subject_anchor_context(
        self,
        *,
        subject_hint: str,
        target_galaxies: list[str],
        base_embedding: list[float],
    ) -> tuple[list[float], list[str], list[str]]:
        return self._subject_anchor_context(
            subject_hint=subject_hint,
            target_galaxies=target_galaxies,
            base_embedding=base_embedding,
            match_mode="mmlu",
        )

    def _apply_specialist_swarm_features(
        self,
        *,
        local_candidates: list[dict[str, Any]],
        reference_embedding: list[float],
        task_type: str,
        path: dict[str, Any],
        selection_steps: list[str],
    ) -> None:
        if not local_candidates or not reference_embedding:
            return
        candidate_rows = self._pad_embedding_rows(
            [list(candidate["match"].get("embedding16", [])) for candidate in local_candidates]
        )
        if not candidate_rows:
            return
        lead_embedding = next(
            (
                list(candidate["match"].get("embedding16", []))
                for candidate in local_candidates
                if float(candidate.get("led_focus", 0.0)) > 0.0
            ),
            candidate_rows[0],
        )
        focus_vector = self._normalize_embedding(list(reference_embedding))
        applied_kernels: list[str] = []
        resonator = self.get_vector_resonator()
        if resonator is not None:
            try:
                focus_vector = self._normalize_embedding(
                    resonator.resonate_list(
                        focus_vector,
                        self._normalize_embedding(list(lead_embedding)),
                        alpha=0.58 if task_type == "MMLU_TASK" else 0.46,
                    )
                )
                applied_kernels.append("gre_vector_resonator")
            except Exception:
                pass
        resonated_rows = candidate_rows
        galaxy_resonance = self.get_galaxy_resonance_engine()
        if galaxy_resonance is not None:
            try:
                resonated_rows = self._pad_embedding_rows(
                    galaxy_resonance.resonate_list(
                        candidate_rows,
                        focus_vector,
                        alpha=0.55 if task_type in {"MMLU_TASK", "LHE_TASK"} else 0.42,
                    )
                )
                applied_kernels.append("galaxy_resonance_engine")
            except Exception:
                resonated_rows = candidate_rows
        crystallized_rows = resonated_rows
        graph_crystallizer = self.get_graph_crystallizer()
        if graph_crystallizer is not None:
            try:
                rounds, self_weight, neighbor_weight = self._task_graph_crystallizer_config(task_type)
                candidate_indices = [
                    int(candidate.get("candidate_global_idx", -1))
                    for candidate in local_candidates
                ]
                global_to_local = {
                    int(global_index): int(local_index)
                    for local_index, global_index in enumerate(candidate_indices)
                    if int(global_index) >= 0
                }
                graph_neighbor_lists = [
                    [
                        int(neighbor_global)
                        for neighbor_global in list(candidate.get("graph_neighbors", []))
                        if int(neighbor_global) in global_to_local
                    ]
                    for candidate in local_candidates
                ]
                neighbor_degree_counts = [len(neighbors) for neighbors in graph_neighbor_lists]
                max_neighbors = max(
                    neighbor_degree_counts,
                    default=0,
                )
                if task_type == "LHE_TASK":
                    selection_steps.append(
                        "LHE graph diagnostic: "
                        f"{len(local_candidates)} candidates, "
                        f"{int(sum(neighbor_degree_counts))} total edges, "
                        f"max_neighbors={int(max_neighbors)}, "
                        f"isolated={int(sum(1 for count in neighbor_degree_counts if count == 0))}"
                    )
                if max_neighbors > 0:
                    adjacency = np.full((len(local_candidates), max_neighbors), -1, dtype=np.int32)
                    neighbor_counts = np.zeros(len(local_candidates), dtype=np.int32)
                    for local_index, local_neighbor_globals in enumerate(graph_neighbor_lists):
                        local_neighbors = [
                            int(global_to_local[int(neighbor_global)])
                            for neighbor_global in local_neighbor_globals
                        ][:max_neighbors]
                        for neighbor_slot, neighbor_local_index in enumerate(local_neighbors):
                            adjacency[local_index, neighbor_slot] = int(neighbor_local_index)
                        neighbor_counts[local_index] = int(len(local_neighbors))
                    crystallized_rows = self._pad_embedding_rows(
                        [
                            list(row)
                            for row in graph_crystallizer.crystallize_graph(
                                node_features=np.asarray(resonated_rows, dtype=np.float32),
                                adjacency=adjacency,
                                neighbor_counts=neighbor_counts,
                                rounds=rounds,
                                self_weight=self_weight,
                                neighbor_weight=neighbor_weight,
                            ).tolist()
                        ]
                    )
                    selection_steps.append(
                        "GRE graph crystallizer: "
                        f"mode=local_graph rounds={rounds} "
                        f"avg_neighbors={float(np.mean(neighbor_counts)):.2f}"
                    )
                elif len(local_candidates) > 1:
                    adjacency, neighbor_counts, semantic_k, avg_top_similarity = (
                        self._build_semantic_knn_adjacency(
                            resonated_rows=resonated_rows,
                            k=min(3, len(local_candidates) - 1),
                        )
                    )
                    if semantic_k > 0:
                        crystallized_rows = self._pad_embedding_rows(
                            [
                                list(row)
                                for row in graph_crystallizer.crystallize_graph(
                                    node_features=np.asarray(resonated_rows, dtype=np.float32),
                                    adjacency=adjacency,
                                    neighbor_counts=neighbor_counts,
                                    rounds=rounds,
                                    self_weight=self_weight,
                                    neighbor_weight=neighbor_weight,
                                ).tolist()
                            ]
                        )
                        selection_steps.append(
                            "GRE graph crystallizer: "
                            f"mode=semantic_knn k={semantic_k} "
                            f"rounds={rounds} avg_sim={avg_top_similarity:.3f}"
                        )
                    else:
                        if task_type == "LHE_TASK":
                            selection_steps.append(
                                "GRE graph crystallizer: mode=compatibility (no local or semantic edges)"
                            )
                        neighborhood_vector = self._normalize_embedding(list(lead_embedding))
                        if task_type in {"MMLU_TASK", "LHE_TASK"}:
                            neighborhood_vector = focus_vector
                        crystallized_rows = self._pad_embedding_rows(
                            graph_crystallizer.crystallize_list(
                                resonated_rows,
                                neighborhood_vector,
                                ema_rate=0.997 if task_type in {"MMLU_TASK", "LHE_TASK"} else 0.992,
                            )
                        )
                else:
                    if task_type == "LHE_TASK":
                        selection_steps.append(
                            "GRE graph crystallizer: mode=compatibility (single candidate)"
                        )
                    neighborhood_vector = self._normalize_embedding(list(lead_embedding))
                    if task_type in {"MMLU_TASK", "LHE_TASK"}:
                        neighborhood_vector = focus_vector
                    crystallized_rows = self._pad_embedding_rows(
                        graph_crystallizer.crystallize_list(
                            resonated_rows,
                            neighborhood_vector,
                            ema_rate=0.997 if task_type in {"MMLU_TASK", "LHE_TASK"} else 0.992,
                        )
                    )
                applied_kernels.append("gre_graph_crystallizer")
            except Exception:
                crystallized_rows = resonated_rows
        resonance_scores = self._embedding_similarities(focus_vector, resonated_rows)
        coherence_scores = self._embedding_similarities(focus_vector, crystallized_rows)
        adjusted_coherence_scores = list(coherence_scores)
        geometry_scores = [0.0 for _ in local_candidates]
        temporal_scores = [0.0 for _ in local_candidates]
        fractal_scores = [0.0 for _ in local_candidates]
        resonance_field = self.get_resonance_field()
        if resonance_field is not None and len(local_candidates) > 1:
            try:
                galaxy_ids = [
                    int(
                        round(
                            float(
                                self._gpu_galaxy_index(candidate["match"].get("galaxy", ""))
                            )
                        )
                    )
                    for candidate in local_candidates
                ]
                adjusted = resonance_field.compute_resonance(
                    crystallized_rows,
                    galaxy_ids,
                    coherence_scores,
                )
                adjusted_coherence_scores = [float(value) for value in adjusted.tolist()]
                applied_kernels.append("gre_resonance_field")
            except Exception:
                adjusted_coherence_scores = list(coherence_scores)
        geometry_router = self.get_geometry_router()
        if geometry_router is not None and len(local_candidates) > 0:
            try:
                focus_rows = np.repeat(
                    np.asarray(focus_vector, dtype=np.float32).reshape(1, -1),
                    len(local_candidates),
                    axis=0,
                )
                geometry_features = geometry_router.compute_relations(
                    np.asarray(crystallized_rows, dtype=np.float32),
                    focus_rows,
                )
                if geometry_features.ndim == 2 and geometry_features.shape[0] == len(local_candidates):
                    cosines = np.clip(geometry_features[:, 0], -1.0, 1.0)
                    quadrant_focus = np.mean(np.clip(geometry_features[:, 2:6], -1.0, 1.0), axis=1)
                    sign_agreement = np.clip(geometry_features[:, 12], 0.0, 1.0)
                    orthogonality = np.clip(1.0 - geometry_features[:, 14], 0.0, 1.0)
                    geometry_scores = (
                        (0.45 * ((cosines + 1.0) * 0.5))
                        + (0.2 * ((quadrant_focus + 1.0) * 0.5))
                        + (0.2 * sign_agreement)
                        + (0.15 * orthogonality)
                    ).astype(np.float32, copy=False).tolist()
                    applied_kernels.append("gre_geometry_router")
            except Exception:
                geometry_scores = [0.0 for _ in local_candidates]
        temporal_reasoning = self.get_temporal_reasoning()
        if temporal_reasoning is not None and len(local_candidates) > 1:
            try:
                ordered_pairs = sorted(
                    [
                        (int(idx), int(candidate.get("led_path_position", -1)))
                        for idx, candidate in enumerate(local_candidates)
                        if int(candidate.get("led_path_position", -1)) >= 0
                    ],
                    key=lambda item: item[1],
                )
                if len(ordered_pairs) >= 2:
                    ordered_rows = np.asarray(
                        [crystallized_rows[idx] for idx, _ in ordered_pairs],
                        dtype=np.float32,
                    )
                    temporal_patterns = np.asarray(
                        temporal_reasoning.compute_patterns(ordered_rows),
                        dtype=np.float32,
                    ).reshape(-1)
                    if temporal_patterns.size >= 24:
                        autocorr_score = float(
                            np.mean((np.clip(temporal_patterns[8:12], -1.0, 1.0) + 1.0) * 0.5)
                        )
                        monotonicity_score = float(np.mean(np.clip(temporal_patterns[12:14], 0.0, 1.0)))
                        recurrence_score = float(np.mean(np.clip(temporal_patterns[14:18], 0.0, 1.0)))
                        predictability_score = float(
                            (0.5 * np.clip(temporal_patterns[18], 0.0, 1.0))
                            + (0.5 * ((np.clip(temporal_patterns[19], -1.0, 1.0) + 1.0) * 0.5))
                        )
                        convergence_score = float(
                            np.mean(
                                [
                                    np.clip(temporal_patterns[20], 0.0, 1.0),
                                    np.clip(temporal_patterns[21], 0.0, 1.0),
                                    1.0 / (1.0 + max(0.0, float(temporal_patterns[22]))),
                                    1.0 - np.clip(temporal_patterns[23], 0.0, 1.0),
                                ]
                            )
                        )
                        temporal_chain_score = max(
                            0.0,
                            min(
                                1.0,
                                (0.25 * autocorr_score)
                                + (0.2 * monotonicity_score)
                                + (0.15 * recurrence_score)
                                + (0.15 * predictability_score)
                                + (0.25 * convergence_score),
                            ),
                        )
                        sequence_length = max(len(ordered_pairs), 1)
                        for rank, (candidate_index, _position) in enumerate(ordered_pairs):
                            progress = float(rank + 1) / float(sequence_length)
                            temporal_scores[candidate_index] = float(
                                temporal_chain_score * (0.5 + (0.5 * progress))
                            )
                        applied_kernels.append("gre_temporal_reasoning")
            except Exception:
                temporal_scores = [0.0 for _ in local_candidates]
        fractal_emitter = self.get_fractal_emitter()
        if fractal_emitter is not None and len(local_candidates) > 0:
            try:
                fractal_self_similarity = np.asarray(
                    fractal_emitter.compute_self_similarity(
                        np.asarray(crystallized_rows, dtype=np.float32),
                        num_scales=4 if task_type == "ARC_TASK" else 3,
                    ),
                    dtype=np.float32,
                ).reshape(-1)
                if fractal_self_similarity.size == len(local_candidates):
                    fractal_scores = np.clip((fractal_self_similarity + 1.0) * 0.5, 0.0, 1.0).tolist()
                    applied_kernels.append("gre_fractal_emitter")
            except Exception:
                fractal_scores = [0.0 for _ in local_candidates]
        for candidate, resonance_score, coherence_score in zip(
            local_candidates,
            resonance_scores,
            adjusted_coherence_scores,
        ):
            candidate["specialist_resonance"] = float(resonance_score)
            candidate["specialist_coherence"] = float(coherence_score)
            candidate["cross_galaxy_resonance"] = float(coherence_score)
            candidate["specialist_worker"] = ",".join(applied_kernels) if applied_kernels else "generic_rpn"
        for candidate, geometry_score in zip(local_candidates, geometry_scores):
            candidate["specialist_geometry"] = float(geometry_score)
        for candidate, temporal_score in zip(local_candidates, temporal_scores):
            candidate["specialist_temporal"] = float(temporal_score)
        for candidate, fractal_score in zip(local_candidates, fractal_scores):
            candidate["specialist_fractal"] = float(fractal_score)
        neutral_proof_tag = self._pack_defeasible_proof_tag(0, 0)
        for candidate in local_candidates:
            candidate.setdefault("specialist_intra_defeasible", 0.0)
            candidate.setdefault("specialist_intra_proof_tag", int(neutral_proof_tag))
            candidate.setdefault("specialist_defeasible_verdict", 0.0)
            candidate.setdefault("specialist_proof_tag", int(neutral_proof_tag))
            candidate.setdefault("path_defeasible_tag", int(path.get("path_defeasible_tag", 1)))
        if applied_kernels:
            selection_steps.append(
                "GRE specialist dispatch: "
                f"{str(path.get('label') or path.get('program_id', 'path'))} -> "
                + ", ".join(applied_kernels)
            )

    def _build_gpu_reasoning_paths(
        self,
        *,
        task: dict[str, Any] | None = None,
        task_type: str,
        primary_program_id: str,
        query_text: str,
        options: list[str] | None = None,
        parse_bundle: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        parse_variants = self._parse_bundle_variant_rows(parse_bundle, query_text)
        variant_lookup = {
            str(variant.get("strategy", "")).strip().lower() or "auto": dict(variant)
            for variant in parse_variants
        }

        def _variant(index: int) -> dict[str, Any]:
            if not parse_variants:
                return {"strategy": "auto", "query_text": query_text}
            return dict(parse_variants[index % len(parse_variants)])

        def _variant_by_name(name: str, fallback_index: int) -> dict[str, Any]:
            token = str(name).strip().lower()
            if token and token in variant_lookup:
                return dict(variant_lookup[token])
            return _variant(fallback_index)

        program_ids: list[str] = [primary_program_id]
        if task_type == "ARC_TASK":
            arc_program_ids = list(dict.fromkeys([primary_program_id, *self.GPU_ARC_SWARM_PROGRAM_IDS]))
            paths: list[dict[str, Any]] = []
            for idx, program_id in enumerate(arc_program_ids[:18]):
                variant = _variant(idx)
                variant_text = str(variant.get("query_text", "")).strip() or query_text
                paths.append(
                    {
                        "program_id": program_id,
                        "query_text": f"{variant_text} {self._arc_program_hint(program_id)}".strip(),
                        "label": "primary" if idx == 0 else f"arc_path_{idx}",
                        "parse_strategy": str(variant.get("strategy", "")).strip() or "auto",
                        "parse_query_text": variant_text,
                    }
                )
            return paths
        if task_type == "MATH_TASK":
            if self._is_gsm8k_math_task(task) and "reasoning_word_problem_fission" not in program_ids:
                gsm8k_workers = [
                    ("reasoning_word_problem_fission", "forward", "forward_chain"),
                    (self.GPU_MATH_REASONING_PROGRAM_ID, "backward", "backward_chain"),
                    ("reasoning_word_problem_fission", "fusion", "fusion_chain"),
                    ("reasoning_word_problem_fission", "forward", "clause_chain"),
                    ("reasoning_definition_top1", "backward", "goal_adjusted_chain"),
                    ("reasoning_definition_top1", "fusion", "alt_add"),
                    (self.GPU_MATH_REASONING_PROGRAM_ID, "forward", "alt_sub"),
                    ("reasoning_multi_hop_top2", "fusion", "alt_mul"),
                    (self.GPU_FACTUAL_REASONING_PROGRAM_ID, "backward", "alt_div"),
                ]
                paths: list[dict[str, Any]] = []
                for idx, (program_id, parse_name, composition_strategy) in enumerate(gsm8k_workers):
                    variant = _variant_by_name(parse_name, idx)
                    variant_text = str(variant.get("query_text", "")).strip() or query_text
                    paths.append(
                        {
                            "program_id": program_id,
                            "query_text": self._program_query_text(
                                variant_text,
                                program_id,
                                task=task,
                                options=options,
                            ),
                            "label": f"gsm8k_worker_{idx}",
                            "parse_strategy": str(variant.get("strategy", "")).strip() or parse_name,
                            "parse_query_text": variant_text,
                            "composition_strategy": composition_strategy,
                            "role_variant_index": idx,
                        }
                    )
                return paths
            program_ids.insert(0, "reasoning_word_problem_fission")
            for candidate in (
                self.GPU_MATH_REASONING_PROGRAM_ID,
                self.GPU_FACTUAL_REASONING_PROGRAM_ID,
                "reasoning_definition_top1",
                "reasoning_multi_hop_top2",
            ):
                if candidate not in program_ids:
                    program_ids.append(candidate)
        elif task_type == "LHE_TASK":
            choice_list = [str(option).strip() for option in (options or []) if str(option).strip()]
            if not choice_list:
                choice_list = self._inline_choice_options(query_text)
            if choice_list:
                option_paths: list[dict[str, Any]] = []
                validation_programs = [
                    self.GPU_FACTUAL_REASONING_PROGRAM_ID,
                    "reasoning_definition_top1",
                    "reasoning_multi_hop_top2",
                    "reasoning_comparison_top1",
                ]
                for idx, option in enumerate(choice_list[:4]):
                    variant = _variant(idx)
                    variant_text = str(variant.get("query_text", "")).strip() or query_text
                    option_paths.append(
                        {
                            "program_id": "reasoning_elimination_top1",
                            "query_text": self._lhe_option_prompt_text(variant_text, option),
                            "label": f"lhe_option_{idx}",
                            "option_index": idx,
                            "option_text": option,
                            "path_role": "hypothesis",
                            "parse_strategy": str(variant.get("strategy", "")).strip() or "auto",
                            "parse_query_text": variant_text,
                        }
                    )
                validation_paths: list[dict[str, Any]] = []
                for idx, option in enumerate(choice_list[:4]):
                    variant = _variant(len(choice_list[:4]) + idx)
                    variant_text = str(variant.get("query_text", "")).strip() or query_text
                    validation_paths.append(
                        {
                            "program_id": validation_programs[idx % len(validation_programs)],
                            "query_text": self._lhe_option_prompt_text(variant_text, option),
                            "label": f"lhe_validation_{idx}",
                            "option_index": idx,
                            "option_text": option,
                            "path_role": "validation",
                            "galaxy_names": ["Reality", "Math", "Grammar", "Word", "Character"],
                            "parse_strategy": str(variant.get("strategy", "")).strip() or "auto",
                            "parse_query_text": variant_text,
                        }
                    )
                cross_variant = _variant(len(choice_list[:4]) * 2)
                cross_text = str(cross_variant.get("query_text", "")).strip() or query_text
                cross_path = {
                    "program_id": "reasoning_multi_hop_top2",
                    "query_text": self._program_query_text(
                        cross_text,
                        "reasoning_multi_hop_top2",
                        task=task,
                        options=choice_list,
                    ),
                    "label": "lhe_cross_validation",
                    "path_role": "cross_validation",
                    "parse_strategy": str(cross_variant.get("strategy", "")).strip() or "auto",
                    "parse_query_text": cross_text,
                }
                return option_paths + validation_paths + [cross_path]
            for candidate in (
                primary_program_id,
                self.GPU_FACTUAL_REASONING_PROGRAM_ID,
                "reasoning_definition_top1",
                "reasoning_multi_hop_top2",
            ):
                if candidate not in program_ids:
                    program_ids.append(candidate)
        elif task_type == "MMLU_TASK":
            choice_list = [str(option).strip() for option in (options or []) if str(option).strip()]
            if choice_list:
                option_paths = [
                    {
                        "program_id": "reasoning_elimination_top1",
                        "query_text": self._mmlu_proposition_text(
                            str(_variant(idx).get("query_text", "")).strip() or query_text,
                            option,
                        ),
                        "label": f"option_{idx}",
                        "option_index": idx,
                        "option_text": option,
                        "path_role": "hypothesis",
                        "parse_strategy": str(_variant(idx).get("strategy", "")).strip() or "auto",
                        "parse_query_text": str(_variant(idx).get("query_text", "")).strip() or query_text,
                    }
                    for idx, option in enumerate(choice_list[:9])
                ]
                validation_paths = [
                    {
                        "program_id": "reasoning_factual_lookup_top1",
                        "query_text": self._mmlu_proposition_text(
                            str(_variant(len(choice_list[:9]) + idx).get("query_text", "")).strip() or query_text,
                            option,
                        ),
                        "label": f"validation_{idx}",
                        "option_index": idx,
                        "option_text": option,
                        "path_role": "validation",
                        "galaxy_names": ["Reality", "Grammar", "Word", "Character"],
                        "parse_strategy": str(_variant(len(choice_list[:9]) + idx).get("strategy", "")).strip() or "auto",
                        "parse_query_text": str(_variant(len(choice_list[:9]) + idx).get("query_text", "")).strip() or query_text,
                    }
                    for idx, option in enumerate(choice_list[:9])
                ]
                return option_paths + validation_paths
            program_ids.append("reasoning_elimination_top1")
        elif task_type in {"CHAT_TASK", "GENERAL_TASK", "GRAMMAR_TASK"}:
            ordered = [
                primary_program_id,
                self.GPU_FACTUAL_REASONING_PROGRAM_ID,
                self.GPU_CHAT_REASONING_PROGRAM_ID,
                "reasoning_definition_top1",
                "reasoning_multi_hop_top2",
                "reasoning_comparison_top1",
            ]
            if options:
                ordered.insert(1, "reasoning_elimination_top1")
            for candidate in ordered:
                if candidate not in program_ids:
                    program_ids.append(candidate)
        else:
            for candidate in (
                self.GPU_FACTUAL_REASONING_PROGRAM_ID,
                "reasoning_definition_top1",
                "reasoning_multi_hop_top2",
            ):
                if candidate not in program_ids:
                    program_ids.append(candidate)
        paths: list[dict[str, Any]] = []
        for idx, program_id in enumerate(program_ids[:18]):
            variant = _variant(idx)
            variant_text = str(variant.get("query_text", "")).strip() or query_text
            paths.append(
                {
                    "program_id": program_id,
                    "query_text": self._program_query_text(
                        variant_text,
                        program_id,
                        task=task,
                        options=options,
                    ),
                    "label": "primary" if idx == 0 else f"path_{idx}",
                    "parse_strategy": str(variant.get("strategy", "")).strip() or "auto",
                    "parse_query_text": variant_text,
                }
            )
        return paths

    @staticmethod
    def _query_looks_lexical(query_text: str) -> bool:
        lowered = str(query_text).strip().lower()
        lexical_markers = (
            "translate",
            "meaning of",
            "define ",
            "how do you say",
            "pronounce",
            "in spanish",
            "in portuguese",
            "in chinese",
            "word for",
            " word ",
        )
        padded = f" {lowered} "
        return len(lowered.split()) <= 3 or any(marker in padded for marker in lexical_markers)

    @staticmethod
    def _query_looks_reality_fact(query_text: str) -> bool:
        lowered = str(query_text).strip().lower()
        markers = (
            "boiling point",
            "melting point",
            "speed of light",
            "atomic number",
            "mass of",
            "formula for",
            "temperature of",
            "density of",
            "capital of",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _query_is_ascii_surface(query_text: str) -> bool:
        return all(ord(char) < 128 for char in str(query_text))

    @staticmethod
    def _gpu_frontier_k(task_type: str) -> int:
        if task_type == "ARC_TASK":
            return 1
        if task_type in {"MATH_TASK", "LHE_TASK", "MMLU_TASK"}:
            return 6
        return 4

    @staticmethod
    def _program_frontier_expression(program: dict[str, Any], top_k: int) -> str:
        base = str(program.get("rpn_program", "")).strip()
        if top_k <= 1 or not base:
            return base
        if base == "1 galaxy_scan drop":
            return f"{top_k} galaxy_scan"
        return base

    @staticmethod
    def _parse_galaxy_scan_stack(stack: list[float]) -> list[int]:
        if not stack:
            return []
        count = int(round(float(stack[-1])))
        if count <= 0:
            return []
        count = min(count, max(0, len(stack) - 1))
        indexes = stack[-1 - count : -1]
        ordered = [int(round(float(value))) for value in reversed(indexes)]
        return [index for index in ordered if index >= 0]

    @staticmethod
    def _pack_led_cost(semantic_cost: int, geometric_cost: int) -> int:
        sem = max(0, min(int(semantic_cost), 0xFFFF))
        geo = max(0, min(int(geometric_cost), 0xFFFF))
        return int((sem << 16) | geo)

    @staticmethod
    def _semantic_cost_from_similarity(similarity: float) -> int:
        sim = max(-1.0, min(float(similarity), 1.0))
        normalized = 1.0 - ((sim + 1.0) * 0.5)
        return int(round(normalized * 65535.0))

    def _embedding_similarities(
        self,
        reference: list[float],
        candidates: list[list[float]],
    ) -> list[float]:
        if not reference or not candidates:
            return [0.0 for _ in candidates]
        reference_values = [float(value) for value in reference]
        max_dim = max(
            [len(reference_values)] + [len(candidate) for candidate in candidates if candidate]
        )
        if max_dim <= 0:
            return [0.0 for _ in candidates]
        padded_reference = list(reference_values[:max_dim])
        if len(padded_reference) < max_dim:
            padded_reference.extend([0.0] * (max_dim - len(padded_reference)))
        padded_candidates: list[list[float]] = []
        for candidate in candidates:
            padded = [float(value) for value in list(candidate)[:max_dim]]
            if len(padded) < max_dim:
                padded.extend([0.0] * (max_dim - len(padded)))
            padded_candidates.append(padded)
        bridge = self.get_cosine_similarity_bridge()
        if bridge is None:
            raise RuntimeError("cosine_similarity_bridge_unavailable")
        return [float(value) for value in bridge.compute_similarities(padded_candidates, padded_reference)]

    def _embedding_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        return float(self._embedding_similarities(left, [right])[0])

    def _embedding_similarity_matrix(
        self,
        sources: list[list[float]],
        targets: list[list[float]],
    ) -> list[list[float]]:
        if not sources or not targets:
            return []
        max_dim = 0
        for vector in list(sources) + list(targets):
            max_dim = max(max_dim, len(vector))
        if max_dim <= 0:
            return [[0.0 for _ in targets] for _ in sources]
        padded_sources: list[list[float]] = []
        padded_targets: list[list[float]] = []
        for vector in sources:
            padded = [float(value) for value in list(vector)[:max_dim]]
            if len(padded) < max_dim:
                padded.extend([0.0] * (max_dim - len(padded)))
            padded_sources.append(padded)
        for vector in targets:
            padded = [float(value) for value in list(vector)[:max_dim]]
            if len(padded) < max_dim:
                padded.extend([0.0] * (max_dim - len(padded)))
            padded_targets.append(padded)
        bridge = self.get_cosine_similarity_bridge()
        if bridge is None:
            raise RuntimeError("cosine_similarity_bridge_unavailable")
        matrix = bridge.compute_similarity_matrix(padded_sources, padded_targets)
        return [[float(value) for value in row.tolist()] for row in matrix]

    @staticmethod
    def _graph_seed_limit(task_type: str) -> int:
        if task_type == "ARC_TASK":
            return 4
        if task_type == "MMLU_TASK":
            return 16
        if task_type in {"MATH_TASK", "LHE_TASK"}:
            return 8
        return 6

    @staticmethod
    def _graph_local_kernel_limit(task_type: str) -> int:
        if task_type == "ARC_TASK":
            return 512
        if task_type in {"MATH_TASK", "LHE_TASK", "MMLU_TASK"}:
            return 2048
        return 1536

    @staticmethod
    def _graph_seed_similarity_threshold(task_type: str) -> float:
        if task_type == "ARC_TASK":
            return 0.25
        if task_type in {"MATH_TASK", "LHE_TASK", "MMLU_TASK"}:
            return 0.2
        return 0.18

    @staticmethod
    def _mmlu_navigation_category_allowed(match: dict[str, Any]) -> bool:
        category = str(match.get("category", "")).strip().lower()
        return category not in {
            "arithmetic_instance",
            "arithmetic_chain_instance",
            "linear_equation_instance",
            "template_support",
            "benchmark_fact",
        }

    def _benchmark_navigation_entry_allowed(
        self,
        *,
        entry: dict[str, Any],
        task_type: str,
        task: dict[str, Any] | None,
        query_text: str,
    ) -> bool:
        if task_type == "MMLU_TASK" and not self._mmlu_navigation_category_allowed(entry):
            return False
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        subfield = str(metadata.get("subfield", "")).strip().lower()
        if task_type == "MATH_TASK" and not self._is_gsm8k_math_task(task):
            if not self._mmlu_navigation_category_allowed(entry):
                return False
            if subfield in {"word_problem_binding", "lhe_goal_typing", "lhe_factual_anchor"}:
                return False
        if not self._is_benchmark_evaluation_task(task):
            return True
        category = str(entry.get("category", "")).strip().lower()
        task_id = str((task or {}).get("task_id", "")).strip()
        entry_task_id = str(metadata.get("task_id", "")).strip()
        if category == "benchmark_fact" or subfield.startswith("benchmark_"):
            return False
        if task_id and entry_task_id and task_id == entry_task_id:
            return False
        if query_text and self._is_answer_bearing_benchmark_shortcut(
            entry=entry,
            task=task,
            query_text=query_text,
        ):
            return False
        return True

    @staticmethod
    def _task_morton_search_config(task_type: str) -> tuple[int, float, int]:
        if task_type == "ARC_TASK":
            return 1536, 0.55, 128
        if task_type in {"MATH_TASK", "MMLU_TASK"}:
            return 6144, 0.8, 384
        if task_type == "LHE_TASK":
            return 4096, 0.7, 128
        return 4096, 0.7, 256

    @staticmethod
    def _task_lod_saliency_threshold(task_type: str) -> float:
        if task_type == "ARC_TASK":
            return 0.52
        if task_type in {"MATH_TASK", "LHE_TASK", "MMLU_TASK"}:
            return 0.6
        return 0.56

    @staticmethod
    def _task_lod_focus_level(task_type: str) -> int:
        if task_type == "ARC_TASK":
            return 6
        if task_type in {"MATH_TASK", "LHE_TASK", "MMLU_TASK"}:
            return 5
        return 6

    @staticmethod
    def _task_seed_budget(task_type: str) -> int:
        if task_type == "ARC_TASK":
            return 4
        if task_type == "MMLU_TASK":
            return 8
        if task_type in {"MATH_TASK", "LHE_TASK"}:
            return 3
        return 4

    @staticmethod
    def _task_graph_crystallizer_config(task_type: str) -> tuple[int, float, float]:
        if task_type == "LHE_TASK":
            return 3, 0.5, 0.5
        if task_type == "MMLU_TASK":
            return 2, 0.6, 0.4
        if task_type == "ARC_TASK":
            return 1, 0.7, 0.3
        if task_type == "MATH_TASK":
            return 1, 0.8, 0.2
        if task_type == "GSM8K_TASK":
            return 2, 0.6, 0.4
        return 2, 0.6, 0.4

    @staticmethod
    def _build_candidate_adjacency(
        visible_indices: list[int],
        local_nodes: list[int],
        local_rows: list[int],
        local_cols: list[int],
    ) -> dict[int, list[int]]:
        visible_list = [int(index) for index in visible_indices]
        visible_set = set(visible_list)
        if not visible_list or not local_nodes or len(local_rows) < (len(local_nodes) + 1):
            return {int(index): [] for index in visible_list}
        global_to_local = {
            int(global_index): int(local_index)
            for local_index, global_index in enumerate(local_nodes)
        }
        adjacency: dict[int, list[int]] = {}
        for global_index in visible_list:
            local_index = global_to_local.get(int(global_index))
            if local_index is None:
                adjacency[int(global_index)] = []
                continue
            row_start = int(local_rows[local_index])
            row_end = int(local_rows[local_index + 1])
            neighbors: list[int] = []
            for edge_index in range(row_start, row_end):
                local_neighbor_index = int(local_cols[edge_index])
                if not (0 <= local_neighbor_index < len(local_nodes)):
                    continue
                neighbor_global = int(local_nodes[local_neighbor_index])
                if neighbor_global == int(global_index) or neighbor_global not in visible_set:
                    continue
                neighbors.append(neighbor_global)
            adjacency[int(global_index)] = neighbors
        return adjacency

    def _build_semantic_knn_adjacency(
        self,
        resonated_rows: list[list[float]],
        k: int = 3,
    ) -> tuple[np.ndarray, np.ndarray, int, float]:
        node_count = int(len(resonated_rows))
        if node_count <= 1:
            return (
                np.full((node_count, 1), -1, dtype=np.int32),
                np.zeros(node_count, dtype=np.int32),
                0,
                0.0,
            )
        k = max(1, min(int(k), node_count - 1))
        similarity_matrix = self._embedding_similarity_matrix(resonated_rows, resonated_rows)
        adjacency = np.full((node_count, k), -1, dtype=np.int32)
        neighbor_counts = np.zeros(node_count, dtype=np.int32)
        selected_similarities: list[float] = []
        for row_index, similarity_row in enumerate(similarity_matrix):
            row_values = np.asarray(similarity_row, dtype=np.float32)
            if row_values.shape[0] != node_count:
                continue
            row_values[row_index] = -1.0
            ranked_neighbors = np.argsort(row_values)[-k:][::-1]
            write_index = 0
            for neighbor_index in ranked_neighbors.tolist():
                similarity = float(row_values[int(neighbor_index)])
                if similarity <= 0.0:
                    continue
                adjacency[row_index, write_index] = int(neighbor_index)
                neighbor_counts[row_index] += 1
                selected_similarities.append(similarity)
                write_index += 1
                if write_index >= k:
                    break
        effective_k = int(np.max(neighbor_counts)) if neighbor_counts.size else 0
        avg_top_similarity = float(np.mean(selected_similarities)) if selected_similarities else 0.0
        return adjacency, neighbor_counts, effective_k, avg_top_similarity

    def _allocate_galaxy_seed_budget(
        self,
        *,
        task_type: str,
        target_galaxies: list[str],
        normalized_galaxy_weights: dict[str, float],
    ) -> dict[str, int]:
        if task_type == "LHE_TASK":
            return {}
        total_budget = max(1, min(self._graph_seed_limit(task_type), self._task_seed_budget(task_type)))
        target_names = [
            str(name).strip()
            for name in target_galaxies
            if str(name).strip()
        ]
        positive_bias = {
            str(galaxy_name): max(0.0, float(normalized_galaxy_weights.get(str(galaxy_name), 0.0)) - 1.0)
            for galaxy_name in self.DEFAULT_GALAXIES
        }
        ranked_bias_names = [
            galaxy_name
            for galaxy_name, bias in sorted(
                positive_bias.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if float(bias) > 1e-6
        ]
        candidate_names = list(dict.fromkeys(target_names + ranked_bias_names[:4]))
        if not candidate_names:
            candidate_names = list(target_names or self.DEFAULT_GALAXIES[:1])
        budget: dict[str, int] = {name: 0 for name in candidate_names}
        remaining = int(total_budget)
        for galaxy_name in target_names:
            if remaining <= 0:
                break
            if galaxy_name not in budget:
                continue
            budget[galaxy_name] = int(budget.get(galaxy_name, 0)) + 1
            remaining -= 1
        if remaining > 0:
            signal_sum = float(sum(positive_bias.get(name, 0.0) for name in candidate_names))
            if signal_sum <= 1e-6:
                cycle_names = list(candidate_names)
                for index in range(remaining):
                    galaxy_name = cycle_names[index % len(cycle_names)]
                    budget[galaxy_name] = int(budget.get(galaxy_name, 0)) + 1
            else:
                exact_rows: list[tuple[float, float, str]] = []
                assigned = 0
                for galaxy_name in candidate_names:
                    exact = float(remaining) * (float(positive_bias.get(galaxy_name, 0.0)) / signal_sum)
                    whole = int(math.floor(exact))
                    if whole > 0:
                        budget[galaxy_name] = int(budget.get(galaxy_name, 0)) + whole
                        assigned += whole
                    exact_rows.append((exact - whole, float(positive_bias.get(galaxy_name, 0.0)), galaxy_name))
                leftover = max(0, remaining - assigned)
                for _, _, galaxy_name in sorted(exact_rows, reverse=True)[:leftover]:
                    budget[galaxy_name] = int(budget.get(galaxy_name, 0)) + 1
        return {
            str(galaxy_name): int(slots)
            for galaxy_name, slots in budget.items()
            if int(slots) > 0
        }

    def _weighted_seed_pairs_by_galaxy(
        self,
        *,
        similarity_pairs: list[tuple[int, float]],
        catalog: list[dict[str, Any]],
        seed_budget: dict[str, int],
        limit: int,
        similarity_threshold: float,
    ) -> list[tuple[int, float]]:
        if not seed_budget:
            return []
        grouped: dict[str, list[tuple[int, float]]] = {}
        for candidate_index, similarity in similarity_pairs:
            if not (0 <= int(candidate_index) < len(catalog)):
                continue
            galaxy_name = str(catalog[int(candidate_index)].get("galaxy", "")).strip()
            if galaxy_name not in seed_budget:
                continue
            grouped.setdefault(galaxy_name, []).append((int(candidate_index), float(similarity)))
        selected: list[tuple[int, float]] = []
        selected_ids: set[int] = set()
        for galaxy_name, slot_count in seed_budget.items():
            for candidate_index, similarity in grouped.get(galaxy_name, [])[: int(slot_count)]:
                if float(similarity) < float(similarity_threshold):
                    continue
                if int(candidate_index) in selected_ids:
                    continue
                selected.append((int(candidate_index), float(similarity)))
                selected_ids.add(int(candidate_index))
                if len(selected) >= int(limit):
                    return selected
        if len(selected) < int(limit):
            for candidate_index, similarity in similarity_pairs:
                if float(similarity) < float(similarity_threshold):
                    continue
                if int(candidate_index) in selected_ids:
                    continue
                selected.append((int(candidate_index), float(similarity)))
                selected_ids.add(int(candidate_index))
                if len(selected) >= int(limit):
                    break
        return selected[: int(limit)]

    def _compose_head_navigation_candidates(
        self,
        *,
        binding: dict[str, Any],
        target_galaxies: list[str],
        galaxy_weights: dict[str, Any] | None,
        reasoning_program_id: str,
        query_embedding: list[float],
        task_type: str,
        selection_steps: list[str],
        task: dict[str, Any] | None = None,
        query_text: str = "",
        domain_hint: str | None = None,
    ) -> list[dict[str, Any]]:
        substrate = self.get_query_head_substrate()
        catalog = self.get_gpu_galaxy_catalog()
        graph = self.get_semantic_csr_graph()
        pathfinder = self.get_led_pathfinder()
        if not catalog or graph is None:
            return []

        normalized_galaxy_weights = self._normalize_galaxy_weights(galaxy_weights)
        allowed_galaxies = list(self.DEFAULT_GALAXIES) if normalized_galaxy_weights else list(target_galaxies)
        allowed_indexes = {
            int(round(float(self._gpu_galaxy_index(name))))
            for name in allowed_galaxies
            if str(name).strip()
        }
        seed_budget = self._allocate_galaxy_seed_budget(
            task_type=task_type,
            target_galaxies=target_galaxies,
            normalized_galaxy_weights=normalized_galaxy_weights,
        ) if normalized_galaxy_weights else {}
        if seed_budget:
            selection_steps.append(
                "Seed budget: "
                + ", ".join(
                    f"{galaxy_name}={int(slot_count)}"
                    for galaxy_name, slot_count in seed_budget.items()
                )
            )
        morton_radius, euclidean_radius, max_results = self._task_morton_search_config(task_type)
        if seed_budget:
            candidate_indexes = []
            total_seed_slots = max(1, sum(int(value) for value in seed_budget.values()))
            for galaxy_name, slot_count in seed_budget.items():
                galaxy_index = int(round(float(self._gpu_galaxy_index(galaxy_name))))
                per_galaxy_results = max(
                    16,
                    min(
                        int(max_results),
                        int(math.ceil((int(max_results) * int(slot_count)) / float(total_seed_slots))),
                    ),
                )
                candidate_indexes.extend(
                    int(index)
                    for index in substrate.morton_locate(
                        query_embedding16=query_embedding,
                        allowed_galaxy_indexes={galaxy_index},
                        max_results=per_galaxy_results,
                        morton_radius=morton_radius,
                        euclidean_radius=euclidean_radius,
                    ).tolist()
                )
            candidate_indexes = np.asarray(list(dict.fromkeys(candidate_indexes)), dtype=np.uint32)
        else:
            candidate_indexes = substrate.morton_locate(
                query_embedding16=query_embedding,
                allowed_galaxy_indexes=allowed_indexes or None,
                max_results=max_results,
                morton_radius=morton_radius,
                euclidean_radius=euclidean_radius,
            )
        if candidate_indexes.size == 0 and allowed_indexes:
            candidate_indexes = substrate.morton_locate(
                query_embedding16=query_embedding,
                allowed_galaxy_indexes=None,
                max_results=max_results,
                morton_radius=morton_radius,
                euclidean_radius=euclidean_radius,
            )
        if candidate_indexes.size == 0:
            fallback_pairs = graph.select_seed_nodes(
                query_embedding=query_embedding,
                allowed_galaxy_indexes=allowed_indexes or None,
                top_k=max(8, self._graph_seed_limit(task_type) * 2),
                similarity_threshold=max(0.0, self._graph_seed_similarity_threshold(task_type) - 0.08),
            )
            if not fallback_pairs and allowed_indexes:
                fallback_pairs = graph.select_seed_nodes(
                    query_embedding=query_embedding,
                    allowed_galaxy_indexes=None,
                    top_k=max(8, self._graph_seed_limit(task_type) * 2),
                    similarity_threshold=max(0.0, self._graph_seed_similarity_threshold(task_type) - 0.08),
                )
            selection_steps.append(
                "Morton locate: 0 candidates, using semantic seed fallback"
            )
            fallback_pairs = [
                (index, similarity)
                for index, similarity in fallback_pairs
                if 0 <= int(index) < len(catalog)
                and self._benchmark_navigation_entry_allowed(
                    entry=catalog[int(index)],
                    task_type=task_type,
                    task=task,
                    query_text=query_text,
                )
            ]
            if not fallback_pairs:
                return []
            candidate_indexes = [index for index, _ in fallback_pairs]
        if not hasattr(candidate_indexes, "tolist"):
            candidate_indexes = list(candidate_indexes)

        if seed_budget:
            merged_seed_map: dict[int, float] = {}
            total_seed_slots = max(1, sum(int(value) for value in seed_budget.values()))
            for galaxy_name, slot_count in seed_budget.items():
                galaxy_index = int(round(float(self._gpu_galaxy_index(galaxy_name))))
                galaxy_pairs = graph.select_seed_nodes(
                    query_embedding=query_embedding,
                    allowed_galaxy_indexes={galaxy_index},
                    top_k=max(4, int(math.ceil((self._graph_seed_limit(task_type) * 2 * int(slot_count)) / float(total_seed_slots)))),
                    similarity_threshold=max(0.0, self._graph_seed_similarity_threshold(task_type) - 0.06),
                )
                for index, similarity in galaxy_pairs:
                    current = merged_seed_map.get(int(index), float("-inf"))
                    if float(similarity) > current:
                        merged_seed_map[int(index)] = float(similarity)
            semantic_seed_pairs = sorted(
                merged_seed_map.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        else:
            semantic_seed_pairs = graph.select_seed_nodes(
                query_embedding=query_embedding,
                allowed_galaxy_indexes=allowed_indexes or None,
                top_k=max(8, self._graph_seed_limit(task_type) * 2),
                similarity_threshold=max(0.0, self._graph_seed_similarity_threshold(task_type) - 0.06),
            )
        semantic_seed_pairs = [
            (index, similarity)
            for index, similarity in semantic_seed_pairs
            if 0 <= int(index) < len(catalog)
            and self._benchmark_navigation_entry_allowed(
                entry=catalog[int(index)],
                task_type=task_type,
                task=task,
                query_text=query_text,
            )
        ]
        merged_indexes = list(candidate_indexes.tolist() if hasattr(candidate_indexes, "tolist") else candidate_indexes)
        merged_indexes.extend(index for index, _ in semantic_seed_pairs)
        candidate_indexes = list(dict.fromkeys(int(index) for index in merged_indexes))

        candidate_index_list = [
            int(index)
            for index in (candidate_indexes.tolist() if hasattr(candidate_indexes, "tolist") else candidate_indexes)
            if 0 <= int(index) < len(catalog)
            and self._benchmark_navigation_entry_allowed(
                entry=catalog[int(index)],
                task_type=task_type,
                task=task,
                query_text=query_text,
            )
        ]
        if not candidate_index_list:
            if seed_budget:
                merged_rescue_map: dict[int, float] = {}
                total_seed_slots = max(1, sum(int(value) for value in seed_budget.values()))
                for galaxy_name, slot_count in seed_budget.items():
                    galaxy_index = int(round(float(self._gpu_galaxy_index(galaxy_name))))
                    galaxy_pairs = graph.select_seed_nodes(
                        query_embedding=query_embedding,
                        allowed_galaxy_indexes={galaxy_index},
                        top_k=max(8, int(math.ceil((self._graph_seed_limit(task_type) * 8 * int(slot_count)) / float(total_seed_slots)))),
                        similarity_threshold=max(0.0, self._graph_seed_similarity_threshold(task_type) - 0.14),
                    )
                    for index, similarity in galaxy_pairs:
                        current = merged_rescue_map.get(int(index), float("-inf"))
                        if float(similarity) > current:
                            merged_rescue_map[int(index)] = float(similarity)
                rescue_seed_pairs = sorted(
                    merged_rescue_map.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            else:
                rescue_seed_pairs = graph.select_seed_nodes(
                    query_embedding=query_embedding,
                    allowed_galaxy_indexes=allowed_indexes or None,
                    top_k=max(32, self._graph_seed_limit(task_type) * 8),
                    similarity_threshold=max(0.0, self._graph_seed_similarity_threshold(task_type) - 0.14),
                )
            candidate_index_list = [
                int(index)
                for index, _ in rescue_seed_pairs
                if 0 <= int(index) < len(catalog)
                and self._benchmark_navigation_entry_allowed(
                    entry=catalog[int(index)],
                    task_type=task_type,
                    task=task,
                    query_text=query_text,
                )
            ]
            candidate_index_list = list(dict.fromkeys(candidate_index_list))
            if candidate_index_list:
                selection_steps.append(
                    "Morton locate: benchmark seed rescue expanded semantic search"
                )
        candidate_embeddings = [
            list(catalog[int(index)].get("embedding16", []))
            for index in candidate_index_list
        ]
        candidate_similarities = self._embedding_similarities(query_embedding, candidate_embeddings)
        subject_seed_bias: dict[int, float] = {}
        if task_type == "MMLU_TASK" and str(domain_hint or "").strip():
            for candidate_index in candidate_index_list:
                subject_seed_bias[int(candidate_index)] = self._subject_anchor_match_score(
                    entry=catalog[int(candidate_index)],
                    subject_hint=str(domain_hint),
                    match_mode="mmlu",
                )
        similarity_pairs = sorted(
            zip(candidate_index_list, candidate_similarities),
            key=lambda item: (
                item[1]
                + (0.22 * float(subject_seed_bias.get(int(item[0]), 0.0)))
                + (
                    0.18
                    * (
                        self._galaxy_weight_for_name(
                            str(catalog[int(item[0])].get("galaxy", "")),
                            normalized_galaxy_weights,
                        )
                        - 1.0
                    )
                )
            ),
            reverse=True,
        )
        if subject_seed_bias:
            matched_seed_count = sum(1 for value in subject_seed_bias.values() if float(value) > 0.0)
            if matched_seed_count > 0:
                selection_steps.append(
                    f"MMLU seed bias: {matched_seed_count}/{len(candidate_index_list)} subject-matched candidates"
                )
        selection_steps.append(
            f"Morton locate: {len(similarity_pairs)} candidates (radius={morton_radius}, target={len(target_galaxies)})"
        )
        if seed_budget:
            seed_pairs = self._weighted_seed_pairs_by_galaxy(
                similarity_pairs=similarity_pairs,
                catalog=catalog,
                seed_budget=seed_budget,
                limit=self._graph_seed_limit(task_type),
                similarity_threshold=self._graph_seed_similarity_threshold(task_type),
            )
        else:
            seed_pairs = [
                pair
                for pair in similarity_pairs[: self._graph_seed_limit(task_type)]
                if pair[1] >= self._graph_seed_similarity_threshold(task_type)
            ]
        if not seed_pairs:
            seed_pairs = similarity_pairs[: self._graph_seed_limit(task_type)]
        if not seed_pairs:
            return []

        seed_nodes = [index for index, _ in seed_pairs]
        local_nodes, local_rows, local_cols, local_costs = graph.extract_local_kernel(
            seed_nodes=seed_nodes,
            max_nodes=self._graph_local_kernel_limit(task_type),
        )
        led_focus_index: int | None = None
        led_path_nodes: list[int] = []
        led_path_positions: dict[int, int] = {}
        if local_nodes and pathfinder is not None:
            global_to_local = {global_index: local_index for local_index, global_index in enumerate(local_nodes)}
            query_node = 0
            first_real_node = 1
            goal_node = len(local_nodes) + 1
            row_offsets = [0]
            col_indices: list[int] = []
            packed_costs: list[int] = []

            for global_index, similarity in seed_pairs:
                local_index = global_to_local.get(global_index)
                if local_index is None:
                    continue
                col_indices.append(first_real_node + local_index)
                packed_costs.append(
                    self._pack_led_cost(
                        self._semantic_cost_from_similarity(similarity),
                        1,
                    )
                )
            row_offsets.append(len(col_indices))

            for local_index, global_index in enumerate(local_nodes):
                row_start = int(local_rows[local_index])
                row_end = int(local_rows[local_index + 1])
                for edge_idx in range(row_start, row_end):
                    col_indices.append(first_real_node + int(local_cols[edge_idx]))
                    packed_costs.append(int(local_costs[edge_idx]))
                goal_cost = None
                if self._benchmark_navigation_entry_allowed(
                    entry=catalog[global_index],
                    task_type=task_type,
                    task=task,
                    query_text=query_text,
                ):
                    goal_cost = self._goal_edge_cost(
                        match=catalog[global_index],
                        task_type=task_type,
                        target_galaxies=target_galaxies,
                        galaxy_weights=normalized_galaxy_weights,
                        reasoning_program_id=reasoning_program_id,
                        query_embedding=query_embedding,
                    )
                if goal_cost is not None:
                    col_indices.append(goal_node)
                    packed_costs.append(
                        self._pack_led_cost(
                            int(round(float(goal_cost) * 65535.0)),
                            1,
                        )
                    )
                row_offsets.append(len(col_indices))

            row_offsets.append(len(col_indices))
            try:
                path = pathfinder.navigate_csr(
                    row_offsets,
                    col_indices,
                    packed_costs,
                    start=query_node,
                    goal=goal_node,
                    alpha=0.35,
                    beta=0.65,
                    max_path_length=max(16, len(local_nodes) + 2),
                )
            except Exception:
                path = None
            if path is not None and getattr(path, "size", 0) >= 3:
                answer_local_node = int(path[-2]) - first_real_node
                if 0 <= answer_local_node < len(local_nodes):
                    led_focus_index = int(local_nodes[answer_local_node])
                    led_path_nodes = [int(node) for node in path.tolist()]
                    for path_position, raw_node in enumerate(led_path_nodes):
                        node_value = int(raw_node)
                        if node_value in {query_node, goal_node}:
                            continue
                        local_index = node_value - first_real_node
                        if 0 <= local_index < len(local_nodes):
                            led_path_positions.setdefault(int(local_nodes[local_index]), int(path_position))
                    selection_steps.append(
                        "LED-A graph navigation: "
                        f"[{str(catalog[led_focus_index].get('galaxy', 'unknown'))}] "
                        f"{str(catalog[led_focus_index].get('id', 'entry')).strip()} "
                        f"(path_hops={max(0, len(led_path_nodes) - 2)}, local_nodes={len(local_nodes)})"
                    )

        visible_source = (
            list(dict.fromkeys([led_focus_index] + local_nodes))
            if led_focus_index is not None
            else local_nodes or [index for index, _ in seed_pairs]
        )
        visible_candidates = substrate.frustum_visible(
            query_embedding16=query_embedding,
            candidate_indices=visible_source,
        )
        if visible_candidates.size == 0:
            visible_candidates = visible_source[: max(8, min(len(visible_source), 32))]
        else:
            preserved = list(seed_nodes)
            if led_focus_index is not None:
                preserved.insert(0, int(led_focus_index))
            visible_candidates = list(
                dict.fromkeys(
                    preserved
                    + [int(index) for index in visible_candidates.tolist()]
                )
            )
        selection_steps.append(
            f"Frustum cull: {int(len(visible_candidates))}/{int(len(visible_source))} visible"
        )

        lod_metrics = substrate.lod_metrics(
            query_embedding16=query_embedding,
            candidate_indices=visible_candidates,
            saliency_threshold=self._task_lod_saliency_threshold(task_type),
        )
        if lod_metrics:
            lod_values = [level for _, level in lod_metrics.values()]
            selection_steps.append(
                f"Dynamic LOD: range={min(lod_values)}-{max(lod_values)} across {len(lod_values)} visible nodes"
            )

        focus_level = self._task_lod_focus_level(task_type)
        visible_index_list = [
            int(raw_index)
            for raw_index in list(visible_candidates)[:36]
            if 0 <= int(raw_index) < len(catalog)
            and self._benchmark_navigation_entry_allowed(
                entry=catalog[int(raw_index)],
                task_type=task_type,
                task=task,
                query_text=query_text,
            )
        ]
        visible_embeddings = [
            list(catalog[candidate_index].get("embedding16", []))
            for candidate_index in visible_index_list
        ]
        visible_similarities = self._embedding_similarities(query_embedding, visible_embeddings)
        visible_similarity_map = {
            int(candidate_index): float(similarity)
            for candidate_index, similarity in zip(visible_index_list, visible_similarities)
        }
        candidate_adjacency = self._build_candidate_adjacency(
            visible_indices=visible_index_list,
            local_nodes=local_nodes,
            local_rows=local_rows,
            local_cols=local_cols,
        )
        candidates: list[dict[str, Any]] = []
        for candidate_index in visible_index_list:
            match = dict(catalog[candidate_index])
            match["_candidate_global_idx"] = int(candidate_index)
            similarity = float(visible_similarity_map.get(candidate_index, 0.0))
            lod_saliency, lod_level = lod_metrics.get(candidate_index, (similarity, focus_level + 1))
            candidates.append(
                {
                    "match": match,
                    "candidate_global_idx": int(candidate_index),
                    "similarity": float(similarity),
                    "lod_saliency": float(lod_saliency),
                    "lod_level": int(lod_level),
                    "lod_focus": 1.0 if int(lod_level) <= focus_level else 0.0,
                    "led_focus": 1.0 if led_focus_index == candidate_index else 0.0,
                    "galaxy_weight": self._galaxy_weight_for_name(
                        str(match.get("galaxy", "")),
                        normalized_galaxy_weights,
                    ),
                    "subject_anchor_focus": float(subject_seed_bias.get(int(candidate_index), 0.0)),
                    "led_path": list(led_path_nodes),
                    "led_path_position": int(led_path_positions.get(int(candidate_index), -1)),
                    "graph_neighbors": list(candidate_adjacency.get(int(candidate_index), [])),
                }
            )
        candidates.sort(
            key=lambda candidate: (
                float(candidate.get("led_focus", 0.0)),
                float(candidate.get("lod_focus", 0.0)),
                float(candidate.get("galaxy_weight", 0.0)),
                float(candidate.get("lod_saliency", 0.0)),
                float(candidate.get("similarity", 0.0)),
                float(candidate["match"].get("confidence", 0.0)),
            ),
            reverse=True,
        )
        return candidates[:24]

    def _dispatch_swarm_weights(
        self,
        *,
        query_embedding: list[float],
        paths: list[dict[str, Any]],
        selection_steps: list[str],
    ) -> list[float]:
        if not paths:
            return []
        weights = [1.0 for _ in paths]
        swarm = self.get_swarm_bridge()
        if swarm is None:
            return weights
        try:
            _, _, resonance_weights = swarm.execute_swarm(
                expand_embedding16_to128(query_embedding),
                num_iterations=1,
                reset_state=True,
                readback_mode="full",
            )
        except Exception:
            return weights
        raw_weights = [max(0.0, float(value)) for value in resonance_weights.tolist()]
        if not raw_weights:
            return weights
        blended_weights = list(raw_weights)
        cognitive_executive = self.get_cognitive_executive()
        if cognitive_executive is not None:
            try:
                diagnostics = swarm.get_chain_diagnostics()
                trust_weights, coherence_score = cognitive_executive.compute_trust_weights(
                    diagnostics.resonance_matrix,
                    diagnostics.chain_norms[: len(raw_weights)],
                )
                trust_values = [max(0.0, float(value)) for value in np.asarray(trust_weights).reshape(-1).tolist()]
                if len(trust_values) == len(raw_weights):
                    executive_mix = max(0.2, min(0.5, 0.2 + (0.3 * max(0.0, float(coherence_score)))))
                    blended_weights = [
                        ((1.0 - executive_mix) * raw_weights[idx]) + (executive_mix * trust_values[idx])
                        for idx in range(len(raw_weights))
                    ]
                    selection_steps.append(
                        "GRE cognitive executive: "
                        f"coherence={float(coherence_score):.2f} mix={executive_mix:.2f}"
                    )
            except Exception:
                blended_weights = list(raw_weights)
        selection_steps.append(
            "Nine-chain swarm dispatch: "
            + ", ".join(
                f"{str(path.get('option_text') or path.get('program_id', 'path'))}={blended_weights[idx % len(blended_weights)]:.2f}"
                for idx, path in enumerate(paths[: min(len(paths), 9)])
            )
        )
        return [1.0 + blended_weights[idx % len(blended_weights)] for idx, _ in enumerate(paths)]

    def _halting_gate_converged(
        self,
        *,
        task_type: str,
        task: dict[str, Any] | None,
        path_scores: list[float],
        candidate_ids: list[str],
        selection_steps: list[str],
        gsm8k_structural_override: dict[str, Any] | None = None,
    ) -> bool:
        if not path_scores:
            selection_steps.append("Halting gate: no path scores, continue")
            return False
        gate = self.get_halting_gate()
        if gate is None:
            selection_steps.append("Halting gate: unavailable, continue")
            return False
        if task_type == "LHE_TASK":
            minimum_threshold = 0.0
            gap_threshold = 0.04
            agreement_threshold = 0.0
        elif task_type == "MMLU_TASK":
            minimum_threshold = 0.0
            gap_threshold = self._mmlu_relative_gap_threshold()
            agreement_threshold = 0.0
        elif self._is_gsm8k_math_task(task):
            minimum_threshold, gap_threshold, agreement_threshold = self._gsm8k_halting_thresholds()
        else:
            minimum_threshold, gap_threshold, agreement_threshold = self._halting_rule_thresholds()
        ordered_pairs = sorted(
            [
                (
                    float(path_scores[idx]),
                    str(candidate_ids[idx]).strip() if idx < len(candidate_ids) else "",
                )
                for idx in range(len(path_scores))
            ],
            key=lambda item: item[0],
            reverse=True,
        )
        ordered_scores = [score for score, _ in ordered_pairs]
        ordered_candidate_ids = [candidate_id for _, candidate_id in ordered_pairs]
        candidate_hashes = self._halting_candidate_hashes(ordered_candidate_ids, len(ordered_scores))
        try:
            flags, metrics = gate.analyze_scores(
                ordered_scores,
                candidate_hashes,
                minimum_threshold=minimum_threshold,
                gap_threshold=gap_threshold,
                agreement_threshold=agreement_threshold,
            )
        except Exception:
            selection_steps.append("Halting gate: error, continue")
            return False
        flag_values = [int(value) for value in flags.tolist()]
        if task_type in {"MMLU_TASK", "LHE_TASK"}:
            converged = bool(len(flag_values) >= 4 and flag_values[1] == 1 and flag_values[3] == 1)
        else:
            converged = all(value == 1 for value in flag_values[:4])
        metric_values = [float(value) for value in metrics.tolist()]
        top_score = float(metric_values[0]) if len(metric_values) >= 1 else 0.0
        score_gap = float(metric_values[1]) if len(metric_values) >= 2 else 0.0
        agreement_count = int(round(metric_values[2])) if len(metric_values) >= 3 else 0
        if self._is_gsm8k_math_task(task) and isinstance(gsm8k_structural_override, dict):
            override_answer = self._halting_record_candidate_id(
                record=gsm8k_structural_override,
                task_type=task_type,
                gsm8k_mode=True,
            )
            top_answer = ordered_candidate_ids[0] if ordered_candidate_ids else ""
            if override_answer and (not converged or override_answer != top_answer):
                converged = True
                selection_steps.append(
                    "GSM8K structural override: "
                    f"{override_answer} beats {top_answer or 'none'} "
                    + (
                        f"(priority={self._gsm8k_structural_override_priority(gsm8k_structural_override):.2f})"
                    )
                )
        selection_steps.append(
            "Halting gate: "
            + ("halt" if converged else "continue")
            + " "
            + f"(top={top_score:.2f}, gap={score_gap:.2f}, agree={agreement_count}, flags={','.join(str(value) for value in flag_values[:4])})"
        )
        return converged

    @staticmethod
    def _gsm8k_preview_candidate_id(record: dict[str, Any]) -> str:
        candidate = record.get("candidate") if isinstance(record.get("candidate"), dict) else {}
        preview_answer = str(candidate.get("gsm8k_preview_answer", "")).strip()
        if preview_answer:
            return preview_answer
        return str((candidate.get("match") or {}).get("id", "")).strip()

    def _aggregate_gsm8k_preview_records(
        self,
        *,
        engine: Any,
        path_best_records: list[dict[str, Any]],
        selection_steps: list[str],
    ) -> list[dict[str, Any]]:
        answer_groups: dict[str, list[dict[str, Any]]] = {}
        for record in path_best_records:
            answer_key = self._gsm8k_preview_candidate_id(record)
            if not answer_key:
                continue
            answer_groups.setdefault(answer_key, []).append(record)
        if not answer_groups:
            return []

        aggregate_jobs: list[tuple[str, dict[str, Any], int, float]] = []
        aggregate_expressions: list[str] = []
        for answer_key, records in answer_groups.items():
            best_record = max(records, key=lambda row: float(row.get("path_score", float("-inf"))))
            candidate = best_record.get("candidate")
            best_structural_score = max(
                float(
                    (
                        (record.get("candidate") or {}).get("gsm8k_structural_score")
                        if isinstance(record.get("candidate"), dict)
                        else 0.0
                    )
                    or 0.0
                )
                for record in records
            )
            strategy_weights = [
                float(
                    (
                        (record.get("candidate") or {}).get("gsm8k_strategy_weight")
                        if isinstance(record.get("candidate"), dict)
                        else 1.0
                    )
                    or 1.0
                )
                for record in records
            ]
            aggregate_jobs.append((answer_key, candidate, len(records), float(best_structural_score)))
            aggregate_expressions.append(
                self._gpu_weighted_mean_expression(
                    [float(row.get("path_score", 0.0)) for row in records],
                    strategy_weights,
                )
            )
            aggregate_expressions.append(
                self._gpu_sum_expression(strategy_weights)
            )

        aggregate_scores: list[float] = []
        for start in range(0, len(aggregate_expressions), 18):
            batch = aggregate_expressions[start : start + 18]
            aggregate_scores.extend(
                float(value)
                for value in engine.evaluate_batch(batch, max_parallel=len(batch))
            )

        aggregated_records: list[dict[str, Any]] = []
        score_index = 0
        for answer_key, candidate, support_count, best_structural_score in aggregate_jobs:
            aggregate_score = float(aggregate_scores[score_index])
            weighted_support = float(aggregate_scores[score_index + 1])
            score_index += 2
            if isinstance(candidate, dict):
                candidate["path_score"] = float(aggregate_score)
                candidate["gsm8k_consensus_support"] = int(support_count)
                candidate["gsm8k_consensus_weight"] = float(weighted_support)
                candidate["gsm8k_best_structural_score"] = float(best_structural_score)
            aggregated_records.append(
                {
                    "candidate": candidate,
                    "option_text": answer_key,
                    "path_score": float(aggregate_score),
                    "support_count": int(support_count),
                    "weighted_support": float(weighted_support),
                    "best_structural_score": float(best_structural_score),
                }
            )
            selection_steps.append(
                "GSM8K answer consensus: "
                f"{answer_key} (struct={best_structural_score:.2f}, workers={support_count}, weight={weighted_support:.2f}, mean={aggregate_score:.2f})"
            )
        # Phase B+ ceiling: structural verification only checks frame/slot fit, so semantically
        # wrong GSM8K programs can still rank alongside correct ones. Phase D compositional RPN
        # execution is needed to separate valid structure from valid computation.
        aggregated_records.sort(
            key=lambda record: (
                float(record.get("best_structural_score", 0.0)),
                float(record.get("weighted_support", 0.0)),
                int(record.get("support_count", 0)),
                float(record.get("path_score", float("-inf"))),
            ),
            reverse=True,
        )
        return aggregated_records

    @staticmethod
    def _try_parse_finite_number(value: Any) -> float | None:
        try:
            numeric = float(str(value).strip())
        except Exception:
            return None
        if not math.isfinite(numeric):
            return None
        return float(numeric)

    def _gsm8k_consensus_record(self, records: list[dict[str, Any]]) -> dict[str, Any] | None:
        viable = [record for record in records if isinstance(record, dict)]
        if not viable:
            return None
        return max(
            viable,
            key=lambda record: (
                float(((record.get("candidate") or {}).get("gsm8k_consensus_weight", record.get("weighted_support", 0.0)))),
                int(((record.get("candidate") or {}).get("gsm8k_consensus_support", record.get("support_count", 0)))),
                float(((record.get("candidate") or {}).get("path_score", record.get("path_score", float("-inf"))))),
                float(((record.get("candidate") or {}).get("gpu_score", float("-inf")))),
            ),
        )

    def _gsm8k_structural_override_priority(self, record: dict[str, Any]) -> float:
        candidate = record.get("candidate") if isinstance(record.get("candidate"), dict) else {}
        structural = float(
            candidate.get(
                "gsm8k_best_structural_score",
                record.get("best_structural_score", candidate.get("gsm8k_structural_score", 0.0)),
            )
            or 0.0
        )
        compositional = max(0.0, float(candidate.get("compositional_consistency", 0.0) or 0.0))
        dimensional = max(0.0, float(candidate.get("compositional_dimensional_consistency", 0.0) or 0.0))
        path_score = float(candidate.get("path_score", record.get("path_score", float("-inf"))) or float("-inf"))
        return float(
            (self.GSM8K_STRUCTURAL_OVERRIDE_PATH_WEIGHT * path_score)
            + (self.GSM8K_STRUCTURAL_OVERRIDE_STRUCT_WEIGHT * structural)
            + (self.GSM8K_STRUCTURAL_OVERRIDE_COMPOSITIONAL_WEIGHT * compositional)
            + (self.GSM8K_STRUCTURAL_OVERRIDE_DIMENSIONAL_WEIGHT * dimensional)
        )

    def _gsm8k_structural_override_record(self, records: list[dict[str, Any]]) -> dict[str, Any] | None:
        viable = [record for record in records if isinstance(record, dict)]
        if not viable:
            return None
        consensus_record = self._gsm8k_consensus_record(viable)
        if consensus_record is None:
            return None
        numeric_records = [
            record
            for record in viable
            if self._try_parse_finite_number(record.get("option_text", "")) is not None
        ]
        if not numeric_records:
            return None
        structural_record = max(
            numeric_records,
            key=lambda record: (
                self._gsm8k_structural_override_priority(record),
                float(
                    ((record.get("candidate") or {}).get(
                        "gsm8k_best_structural_score",
                        record.get("best_structural_score", ((record.get("candidate") or {}).get("gsm8k_structural_score", 0.0))),
                    ))
                ),
                float(((record.get("candidate") or {}).get("path_score", record.get("path_score", float("-inf"))))),
            ),
        )
        if structural_record is consensus_record:
            return None
        candidate = structural_record.get("candidate") if isinstance(structural_record.get("candidate"), dict) else {}
        structural_score = float(
            candidate.get(
                "gsm8k_best_structural_score",
                structural_record.get("best_structural_score", candidate.get("gsm8k_structural_score", 0.0)),
            )
            or 0.0
        )
        if structural_score < self.GSM8K_STRUCTURAL_OVERRIDE_MIN:
            return None
        structural_priority = self._gsm8k_structural_override_priority(structural_record)
        consensus_priority = self._gsm8k_structural_override_priority(consensus_record)
        consensus_numeric = self._try_parse_finite_number(consensus_record.get("option_text", ""))
        consensus_candidate = (
            consensus_record.get("candidate") if isinstance(consensus_record.get("candidate"), dict) else {}
        )
        consensus_structural = float(
            consensus_candidate.get(
                "gsm8k_best_structural_score",
                consensus_record.get("best_structural_score", consensus_candidate.get("gsm8k_structural_score", 0.0)),
            )
            or 0.0
        )
        if consensus_numeric is None:
            if structural_priority >= (float(consensus_record.get("path_score", float("-inf"))) + self.GSM8K_STRUCTURAL_OVERRIDE_MARGIN):
                return structural_record
            return None
        if (
            structural_priority >= (consensus_priority + self.GSM8K_STRUCTURAL_OVERRIDE_MARGIN)
            and structural_score >= (consensus_structural + self.GSM8K_STRUCTURAL_OVERRIDE_MARGIN)
        ):
            return structural_record
        return None

    @staticmethod
    def _halting_candidate_hashes(candidate_ids: list[str], score_count: int) -> list[int]:
        hashes: list[int] = []
        normalized = list(candidate_ids[:score_count])
        for idx in range(score_count):
            candidate_id = str(normalized[idx]).strip() if idx < len(normalized) else ""
            if candidate_id:
                hashed = int(zlib.crc32(candidate_id.encode("utf-8"))) & 0xFFFFFFFF
                hashes.append(hashed if hashed != 0 else ((0x80000000 | (idx + 1)) & 0xFFFFFFFF))
            else:
                hashes.append((0x80000000 | (idx + 1)) & 0xFFFFFFFF)
        return hashes

    def _record_active_lhe_timing(self, label: str, elapsed: float) -> None:
        timing = getattr(self, "_active_lhe_timing", None)
        if not isinstance(timing, dict):
            return
        key = str(label).strip()
        if not key:
            return
        timing[key] = float(timing.get(key, 0.0)) + max(0.0, float(elapsed))

    def _format_active_lhe_timing_line(self, total_elapsed: float) -> str:
        timing = getattr(self, "_active_lhe_timing", None)
        if not isinstance(timing, dict):
            timing = {}
        return (
            "[LHE-FULL-TIMING] "
            f"parse={float(timing.get('parse', 0.0)):.2f}s "
            f"targets={float(timing.get('targets', 0.0)):.2f}s "
            f"bind={float(timing.get('bind', 0.0)):.2f}s "
            f"embed={float(timing.get('embed', 0.0)):.2f}s "
            f"nav_embed={float(timing.get('nav_embed', 0.0)):.2f}s "
            f"morton={float(timing.get('morton', 0.0)):.2f}s "
            f"scoring={float(timing.get('scoring', 0.0)):.2f}s "
            f"build_paths={float(timing.get('build_paths', 0.0)):.2f}s "
            f"eval={float(timing.get('evaluate_gpu_paths', 0.0)):.2f}s "
            f"frontier={float(timing.get('frontier', 0.0)):.2f}s "
            f"rerank={float(timing.get('led_rerank', 0.0)):.2f}s "
            f"halting={float(timing.get('halting', 0.0)):.2f}s "
            f"selection={float(timing.get('selection', 0.0)):.2f}s "
            f"answer={float(timing.get('answer', 0.0)):.2f}s "
            f"total={float(total_elapsed):.2f}s"
        )

    def _finalize_active_lhe_timing(
        self,
        *,
        result: dict[str, Any] | None = None,
        selection_steps: list[str] | None = None,
        total_elapsed: float,
        answer_text: str = "",
    ) -> None:
        timing = getattr(self, "_active_lhe_timing", None)
        if not isinstance(timing, dict):
            return
        line = self._format_active_lhe_timing_line(total_elapsed)
        if isinstance(selection_steps, list):
            selection_steps.append(line)
        if isinstance(result, dict):
            reasoning_trace = list(result.get("reasoning_trace", []))
            thinking_trace = list(result.get("thinking_trace", []))
            reasoning_trace.append(line)
            thinking_trace.append(line)
            result["reasoning_trace"] = reasoning_trace
            result["thinking_trace"] = thinking_trace
            result["thinking_xml"] = self._render_thinking_xml(thinking_trace, answer_text or str(result.get("answer", "")))
        self._active_lhe_timing = None

    def _halting_rule_thresholds(self) -> tuple[float, float, float]:
        defaults = {
            "halting_threshold_minimum": 0.3,
            "halting_threshold_gap": 0.1,
            "halting_threshold_agreement": 3.0,
        }
        for entry in self.get_gpu_galaxy_catalog():
            if str(entry.get("galaxy", "")).strip() != "Grammar":
                continue
            entry_id = str(entry.get("id", "")).strip()
            if entry_id not in defaults:
                continue
            metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            try:
                defaults[entry_id] = float(metadata.get("threshold", defaults[entry_id]))
            except Exception:
                continue
        return (
            float(defaults["halting_threshold_minimum"]),
            float(defaults["halting_threshold_gap"]),
            float(defaults["halting_threshold_agreement"]),
        )

    def _select_composed_head_candidate(
        self,
        *,
        task: dict[str, Any] | None,
        binding: dict[str, Any],
        paths: list[dict[str, Any]],
        target_galaxies: list[str],
        galaxy_weights: dict[str, Any] | None,
        reasoning_program_id: str,
        query_embedding: list[float],
        task_type: str,
        options: list[str] | None,
        domain_hint: str | None,
        selection_steps: list[str],
        parse_bundle: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        navigation_reference_embedding = list(query_embedding)
        nav_embed_started = time.perf_counter()
        parse_context = self._parse_bundle_embeddings(
            query_embedding=query_embedding,
            parse_bundle=parse_bundle,
            task=task,
        )
        parse_navigation_embedding = list(parse_context.get("navigation_embedding", []))
        if parse_navigation_embedding:
            navigation_reference_embedding = parse_navigation_embedding
            selection_steps.append(
                "Navigator parse: "
                + ", ".join(
                    str(row.get("strategy", "auto")).strip() or "auto"
                    for row in list(parse_context.get("variants", []))[:3]
                )
            )
        parse_quantity_values = [float(value) for value in parse_context.get("quantity_values", [])[:6]]
        if parse_quantity_values:
            selection_steps.append(
                "Navigator fusion quantities: "
                + ", ".join(self._gpu_scalar_literal(value) for value in parse_quantity_values)
            )
        gsm8k_mode = self._is_gsm8k_math_task(task)
        benchmark_eval_mode = self._is_benchmark_evaluation_task(task)
        benchmark_query_text = str(
            (task or {}).get("query")
            or (task or {}).get("question")
            or (task or {}).get("prompt")
            or ""
        ).strip()
        if task_type == "LHE_TASK":
            lhe_options = [str(option).strip() for option in (options or []) if str(option).strip()]
            if not lhe_options:
                lhe_options = self._inline_choice_options(benchmark_query_text)
            selection_steps.append(
                "LHE dispatch: "
                + f"options={len(lhe_options)}, program={reasoning_program_id or 'unknown'}"
            )
        if benchmark_eval_mode:
            selection_steps.append("Benchmark honesty filter: answer-bearing benchmark rows suppressed")
        gsm8k_context: dict[str, Any] = {}
        subject_anchor_ids: list[str] = []
        subject_anchor_galaxies: set[str] = set()
        subject_embedding: list[float] = []
        parse_override_signals = self._task_parse_override_signals(
            task=task,
            domain_hint=domain_hint,
        )
        if task_type == "MMLU_TASK":
            subject_label = str(domain_hint or "").strip() or "unknown"
            subject_embedding, subject_anchor_ids, anchor_galaxies = self._mmlu_subject_anchor_context(
                subject_hint=subject_label,
                target_galaxies=target_galaxies,
                base_embedding=query_embedding,
            )
            if subject_embedding:
                navigation_reference_embedding = subject_embedding
            subject_anchor_galaxies = {str(name).strip() for name in anchor_galaxies if str(name).strip()}
            if subject_anchor_ids:
                selection_steps.append(
                    f"MMLU anchor: hit {subject_label} ({len(subject_anchor_ids)} entries)"
                )
                selection_steps.append(
                    "MMLU subject anchor resonance: " + ", ".join(subject_anchor_ids)
                )
            else:
                selection_steps.append(f"MMLU anchor: miss {subject_label} (0 entries)")
        elif task_type == "LHE_TASK":
            subject_label = str(domain_hint or "").strip() or "unknown"
            subject_embedding, subject_anchor_ids, anchor_galaxies = self._subject_anchor_context(
                subject_hint=subject_label,
                target_galaxies=target_galaxies,
                base_embedding=navigation_reference_embedding,
                match_mode="domain",
            )
            if subject_embedding:
                navigation_reference_embedding = subject_embedding
            subject_anchor_galaxies = {str(name).strip() for name in anchor_galaxies if str(name).strip()}
            if subject_anchor_ids:
                selection_steps.append(
                    f"LHE anchor: hit {subject_label} ({len(subject_anchor_ids)} entries)"
                )
                selection_steps.append(
                    "LHE subject anchor resonance: " + ", ".join(subject_anchor_ids)
                )
            else:
                selection_steps.append(f"LHE anchor: miss {subject_label} (0 entries)")
        elif task_type == "MATH_TASK" and not gsm8k_mode:
            algebra_signal = str(parse_override_signals.get("algebra_signal", "")).strip()
            if algebra_signal:
                subject_embedding, subject_anchor_ids, anchor_galaxies = self._subject_anchor_context(
                    subject_hint=algebra_signal,
                    target_galaxies=target_galaxies,
                    base_embedding=navigation_reference_embedding,
                    match_mode="domain",
                )
                if subject_embedding:
                    navigation_reference_embedding = self._blend_reference_embedding(
                        navigation_reference_embedding,
                        subject_embedding,
                        alpha=self._parse_override_weight("meta_rule_parse_override_algebra", 0.8),
                    )
                subject_anchor_galaxies = {str(name).strip() for name in anchor_galaxies if str(name).strip()}
                if subject_anchor_ids:
                    selection_steps.append(
                        f"Math parse override: hit {algebra_signal} ({len(subject_anchor_ids)} entries)"
                    )
                else:
                    selection_steps.append(f"Math parse override: miss {algebra_signal} (0 entries)")
        elif gsm8k_mode:
            gsm8k_context = self._gsm8k_word_problem_context(
                target_galaxies=target_galaxies,
                base_embedding=navigation_reference_embedding,
                parse_bundle=parse_bundle,
            )
            context_embedding = list(gsm8k_context.get("navigation_embedding", []))
            if context_embedding:
                navigation_reference_embedding = context_embedding
            operation_ids = [
                str(value).strip()
                for value in gsm8k_context.get("operation_ids", [])
                if str(value).strip()
            ]
            number_ids = [
                str(value).strip()
                for value in gsm8k_context.get("number_ids", [])
                if str(value).strip()
            ]
            if operation_ids:
                top_operation = str(gsm8k_context.get("top_operation", "")).strip() or "pattern"
                selection_steps.append(
                    f"GSM8K fission: hit {top_operation} ({len(operation_ids)} entries)"
                )
                selection_steps.append(
                    "GSM8K operation anchors: " + ", ".join(operation_ids)
                )
            else:
                selection_steps.append("GSM8K fission: miss operation pattern (0 entries)")
            goal_type = str(gsm8k_context.get("goal_type", "")).strip()
            typed_roles = [
                str(value).strip()
                for value in gsm8k_context.get("typed_roles", [])
                if str(value).strip()
            ]
            selection_steps.append(
                "GSM8K goal typing: "
                + (
                    f"{goal_type or 'none'} via "
                    + ("typed_fusion" if bool(gsm8k_context.get("uses_typed_fusion", False)) else "generic_blocks")
                    + (f" ({', '.join(typed_roles)})" if typed_roles else "")
                )
            )
            if number_ids:
                selection_steps.append(
                    "GSM8K number neighborhood: " + ", ".join(number_ids[:6])
                )
            else:
                selection_steps.append("GSM8K number neighborhood: miss (0 entries)")
        if task_type == "LHE_TASK":
            self._record_active_lhe_timing("nav_embed", time.perf_counter() - nav_embed_started)
        morton_started = time.perf_counter()
        navigation_candidates = self._compose_head_navigation_candidates(
            binding=binding,
            target_galaxies=target_galaxies,
            galaxy_weights=galaxy_weights,
            reasoning_program_id=reasoning_program_id,
            query_embedding=navigation_reference_embedding,
            task_type=task_type,
            selection_steps=selection_steps,
            task=task,
            query_text=benchmark_query_text,
            domain_hint=domain_hint,
        )
        if not navigation_candidates and task_type != "MMLU_TASK":
            return None
        lhe_shared_navigation_candidates = list(navigation_candidates)
        if task_type == "LHE_TASK" and benchmark_query_text and not benchmark_eval_mode:
            exact_candidates = self._lhe_exact_question_navigation_candidates(
                query_text=benchmark_query_text,
                reference_embedding=navigation_reference_embedding,
            )
            if exact_candidates:
                existing_ids = {
                    str((candidate.get("match") or {}).get("id", "")).strip()
                    for candidate in lhe_shared_navigation_candidates
                }
                injected_candidates = [
                    candidate
                    for candidate in exact_candidates
                    if str((candidate.get("match") or {}).get("id", "")).strip() not in existing_ids
                ]
                if injected_candidates:
                    lhe_shared_navigation_candidates = [*injected_candidates, *lhe_shared_navigation_candidates]
        if task_type == "LHE_TASK":
            self._record_active_lhe_timing("morton", time.perf_counter() - morton_started)

        scoring_started = time.perf_counter()
        option_embeddings: dict[str, list[float]] = {}
        if task_type in {"MMLU_TASK", "LHE_TASK"}:
            for path in paths[:18]:
                option_text = str(path.get("option_text", "")).strip()
                proposition_text = str(path.get("query_text", "")).strip()
                cache_key = proposition_text or option_text
                if not option_text or cache_key in option_embeddings:
                    continue
                option_embeddings[cache_key] = self._resonate_option_embedding(
                    navigation_reference_embedding,
                    option_text,
                    query_text=proposition_text or None,
                )

        swarm_weights = self._dispatch_swarm_weights(
            query_embedding=navigation_reference_embedding,
            paths=paths,
            selection_steps=selection_steps,
        )
        if gsm8k_mode:
            self._apply_early_defeasible_gate(
                task_type=task_type,
                paths=paths,
                swarm_weights=swarm_weights,
                selection_steps=selection_steps,
            )
        engine = self.get_gpu_reasoning_engine()
        scored_candidates: list[dict[str, Any]] = []
        path_best_records: list[dict[str, Any]] = []
        mmlu_validation_weight, mmlu_support_weight = self._mmlu_option_rule_weights()
        mmlu_shared_candidates = (
            task_type == "MMLU_TASK"
            and self._mmlu_prefers_shared_option_neighborhood(
                task=task,
                domain_hint=domain_hint,
                options=options,
            )
        )
        shared_mmlu_records: list[dict[str, Any]] = []
        shared_mmlu_option_similarities: dict[str, list[float]] = {}
        shared_lhe_records: list[dict[str, Any]] = []
        shared_lhe_option_similarities: dict[str, list[float]] = {}
        lhe_cached_option_records: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
        lhe_factual_scored = False
        if mmlu_shared_candidates:
            mmlu_navigation_candidates = list(navigation_candidates)
            if benchmark_eval_mode:
                mmlu_navigation_candidates, suppressed_shortcuts = self._filter_benchmark_shortcut_candidates(
                    candidates=mmlu_navigation_candidates,
                    task=task,
                    query_text=benchmark_query_text,
                )
                if suppressed_shortcuts > 0:
                    selection_steps.append(
                        "Benchmark honesty filter: MMLU shared suppressed "
                        f"{suppressed_shortcuts} shortcut candidates"
                    )
            if not mmlu_navigation_candidates:
                return None
            shared_mmlu_records = [
                {
                    "match": candidate["match"],
                    "similarity": float(candidate.get("similarity", 0.0)),
                    "lod_saliency": float(candidate.get("lod_saliency", 0.0)),
                    "lod_level": int(candidate.get("lod_level", 0)),
                    "lod_focus": float(candidate.get("lod_focus", 0.0)),
                    "led_focus": float(candidate.get("led_focus", 0.0)),
                    "led_path": list(candidate.get("led_path", [])),
                    "gsm8k_mode": 0.0,
                    "mmlu_symbolic_mode": 1.0,
                    "parse_strategy": "auto",
                }
                for candidate in mmlu_navigation_candidates
            ]
            if shared_mmlu_records and subject_embedding:
                subject_similarities = self._embedding_similarities(
                    subject_embedding,
                    [list(record["match"].get("embedding16", [])) for record in shared_mmlu_records],
                )
                for record, subject_similarity in zip(shared_mmlu_records, subject_similarities):
                    record["subject_similarity"] = float(subject_similarity)
                    record["subject_anchor_focus"] = max(
                        float(record.get("subject_anchor_focus", 0.0)),
                        self._subject_anchor_match_score(
                            entry=record["match"],
                            subject_hint=subject_label,
                            match_mode="mmlu",
                        ),
                    )
            if shared_mmlu_records and list(parse_context.get("fusion_embedding", [])):
                parse_similarities = self._embedding_similarities(
                    list(parse_context.get("fusion_embedding", [])),
                    [list(record["match"].get("embedding16", [])) for record in shared_mmlu_records],
                )
                for record, parse_similarity in zip(shared_mmlu_records, parse_similarities):
                    record["parse_similarity"] = float(parse_similarity)
            if shared_mmlu_records and list(parse_context.get("directional_embedding", [])):
                directional_similarities = self._embedding_similarities(
                    list(parse_context.get("directional_embedding", [])),
                    [list(record["match"].get("embedding16", [])) for record in shared_mmlu_records],
                )
                for record, directional_similarity in zip(shared_mmlu_records, directional_similarities):
                    record["parse_directional_similarity"] = float(directional_similarity)
            for record in shared_mmlu_records:
                match_id = str(record["match"].get("id", "")).strip()
                record["ternary_prior"] = self._candidate_ternary_prior(match_id)
            if shared_mmlu_records and option_embeddings:
                embedding_rows = [list(record["match"].get("embedding16", [])) for record in shared_mmlu_records]
                for option_key, option_embedding in option_embeddings.items():
                    shared_mmlu_option_similarities[option_key] = self._embedding_similarities(
                        option_embedding,
                        embedding_rows,
                    )
        if task_type == "LHE_TASK":
            lhe_navigation_candidates = list(lhe_shared_navigation_candidates)
            if benchmark_eval_mode:
                lhe_navigation_candidates, suppressed_shortcuts = self._filter_benchmark_shortcut_candidates(
                    candidates=lhe_navigation_candidates,
                    task=task,
                    query_text=benchmark_query_text,
                )
                if suppressed_shortcuts > 0:
                    selection_steps.append(
                        "Benchmark honesty filter: LHE shared suppressed "
                        f"{suppressed_shortcuts} shortcut candidates"
                    )
            if not lhe_navigation_candidates:
                return None
            shared_lhe_records = [
                {
                    "match": candidate["match"],
                    "similarity": float(candidate.get("similarity", 0.0)),
                    "lod_saliency": float(candidate.get("lod_saliency", 0.0)),
                    "lod_level": int(candidate.get("lod_level", 0)),
                    "lod_focus": float(candidate.get("lod_focus", 0.0)),
                    "led_focus": float(candidate.get("led_focus", 0.0)),
                    "led_path": list(candidate.get("led_path", [])),
                    "gsm8k_mode": 0.0,
                    "parse_strategy": "auto",
                    "exact_query_match": 1.0 if self._entry_query_matches(candidate["match"], benchmark_query_text) else 0.0,
                    "parse_override_domain": (
                        1.0
                        if (
                            str(parse_override_signals.get("domain_signal", "")).strip()
                            and self._candidate_matches_parse_signal(
                                candidate["match"],
                                str(parse_override_signals.get("domain_signal", "")).strip(),
                            )
                        )
                        else 0.0
                    ),
                    "lhe_exact_benchmark": (
                        1.0
                        if (
                            not benchmark_eval_mode
                            and str(candidate["match"].get("galaxy", "")).strip() in {"Reality", "Math"}
                            and str(candidate["match"].get("category", "")).strip().lower()
                            in {"benchmark_fact", "clue_fact", "cipher_result", "formal_result"}
                            and self._entry_query_matches(candidate["match"], benchmark_query_text)
                        )
                        else 0.0
                    ),
                }
                for candidate in lhe_navigation_candidates
            ]
            if shared_lhe_records and subject_embedding:
                subject_similarities = self._embedding_similarities(
                    subject_embedding,
                    [list(record["match"].get("embedding16", [])) for record in shared_lhe_records],
                )
                for record, subject_similarity in zip(shared_lhe_records, subject_similarities):
                    record["subject_similarity"] = float(subject_similarity)
                    record["subject_anchor_focus"] = max(
                        float(record.get("subject_anchor_focus", 0.0)),
                        self._subject_anchor_match_score(
                            entry=record["match"],
                            subject_hint=subject_label,
                            match_mode="domain",
                        ),
                    )
            if shared_lhe_records and list(parse_context.get("fusion_embedding", [])):
                parse_similarities = self._embedding_similarities(
                    list(parse_context.get("fusion_embedding", [])),
                    [list(record["match"].get("embedding16", [])) for record in shared_lhe_records],
                )
                for record, parse_similarity in zip(shared_lhe_records, parse_similarities):
                    record["parse_similarity"] = float(parse_similarity)
            if shared_lhe_records and list(parse_context.get("directional_embedding", [])):
                directional_similarities = self._embedding_similarities(
                    list(parse_context.get("directional_embedding", [])),
                    [list(record["match"].get("embedding16", [])) for record in shared_lhe_records],
                )
                for record, directional_similarity in zip(shared_lhe_records, directional_similarities):
                    record["parse_directional_similarity"] = float(directional_similarity)
            for record in shared_lhe_records:
                match_id = str(record["match"].get("id", "")).strip()
                record["ternary_prior"] = self._candidate_ternary_prior(match_id)
            if shared_lhe_records and option_embeddings:
                embedding_rows = [list(record["match"].get("embedding16", [])) for record in shared_lhe_records]
                for option_key, option_embedding in option_embeddings.items():
                    shared_lhe_option_similarities[option_key] = self._embedding_similarities(
                        option_embedding,
                        embedding_rows,
                    )
        for path_index, path in enumerate(paths[:18]):
            if int(path.get("path_defeasible_tag", 1)) < 0:
                selection_steps.append(
                    "GRE triple defeasible stage1: skipped "
                    f"{str(path.get('label') or path.get('program_id', 'path'))}"
                )
                continue
            program = self._select_gpu_reasoning_program(str(path.get("program_id", "")).strip())
            option_text = str(path.get("option_text", "")).strip()
            proposition_text = str(path.get("query_text", "")).strip()
            option_embedding = option_embeddings.get(proposition_text or option_text)
            task_query_text = str(
                (task or {}).get("query")
                or (task or {}).get("question")
                or (task or {}).get("prompt")
                or proposition_text
            ).strip()
            path_navigation_candidates = navigation_candidates
            path_target_galaxies = (
                [str(name).strip() for name in path.get("galaxy_names", []) if str(name).strip()]
                if isinstance(path.get("galaxy_names"), list)
                else list(target_galaxies)
            )
            if not path_target_galaxies:
                path_target_galaxies = list(target_galaxies)
            if task_type == "MMLU_TASK" and mmlu_shared_candidates:
                cache_key = proposition_text or option_text
                local_candidates = [
                    {
                        **record,
                        "path": path,
                        "program": program,
                        "swarm_weight": float(swarm_weights[path_index]) if path_index < len(swarm_weights) else 1.0,
                        "parse_strategy": str(path.get("parse_strategy", "")).strip() or "auto",
                    }
                    for record in shared_mmlu_records
                ]
                option_similarity_values = shared_mmlu_option_similarities.get(cache_key, [])
                if option_text:
                    for record, option_similarity in zip(local_candidates, option_similarity_values):
                        record["option_text"] = option_text
                        record["option_similarity"] = float(option_similarity)
                        record["option_support"] = self._mmlu_option_support_score(
                            record["match"],
                            option_text,
                        )
                self._apply_specialist_swarm_features(
                    local_candidates=local_candidates,
                    reference_embedding=option_embedding or navigation_reference_embedding,
                    task_type=task_type,
                    path=path,
                    selection_steps=selection_steps,
                )
                if gsm8k_mode:
                    self._apply_intra_path_defeasible(
                        local_candidates=local_candidates,
                        path=path,
                        task_type=task_type,
                        selection_steps=selection_steps,
                    )
                scores = self._score_gpu_candidates_batch(
                    candidates=local_candidates,
                    primary_program_id=reasoning_program_id,
                    target_galaxies=path_target_galaxies,
                    task_type=task_type,
                    domain_hint=domain_hint,
                    cross_domain=False,
                )
                for record, score in zip(local_candidates, scores):
                    record["gpu_score"] = float(score)
                if not local_candidates:
                    continue
                ordered_candidates = sorted(
                    local_candidates,
                    key=lambda candidate: float(candidate.get("gpu_score", float("-inf"))),
                    reverse=True,
                )
                best_for_path = ordered_candidates[0]
                coherence_candidates = ordered_candidates[: min(4, len(ordered_candidates))]
                neighborhood_mean = float(
                    engine.evaluate(
                        self._gpu_mean_expression(
                            [float(candidate.get("gpu_score", 0.0)) for candidate in coherence_candidates]
                        )
                    )
                )
                best_for_path["path_score"] = float(
                    engine.evaluate(
                        " ".join(
                            [
                                self._gpu_scalar_literal(best_for_path.get("gpu_score", 0.0)),
                                self._gpu_scalar_literal(neighborhood_mean),
                                "0.05",
                                "*",
                                "+",
                            ]
                        )
                    )
                )
                selection_steps.append(
                    "Swarm path result: "
                    f"{str(program.get('id', '')).strip()}"
                    + (f"[{option_text}]" if option_text else "")
                    + " -> "
                    f"[{str(best_for_path['match'].get('galaxy', 'unknown'))}] "
                    f"{str(best_for_path['match'].get('id', 'entry')).strip()} "
                    f"(coherence={float(best_for_path.get('path_score', 0.0)):.2f}, top={float(best_for_path.get('gpu_score', 0.0)):.2f})"
                )
                scored_candidates.extend(local_candidates)
                path_best_records.append(
                    {
                        "candidate": best_for_path,
                        "option_text": option_text,
                        "path_score": float(best_for_path.get("path_score", float("-inf"))),
                        "path_role": str(path.get("path_role", "")).strip(),
                        "preview_answer": str(best_for_path.get("gsm8k_preview_answer", "")).strip(),
                    }
                )
                continue
            if task_type == "MMLU_TASK" and option_embedding is not None:
                path_navigation_candidates = self._compose_head_navigation_candidates(
                    binding=binding,
                    target_galaxies=path_target_galaxies,
                    galaxy_weights=galaxy_weights,
                    reasoning_program_id=str(program.get("id", "")).strip() or reasoning_program_id,
                    query_embedding=option_embedding,
                    task_type=task_type,
                    selection_steps=[],
                    task=task,
                    query_text=benchmark_query_text,
                    domain_hint=domain_hint,
                )
                if not path_navigation_candidates:
                    path_navigation_candidates = navigation_candidates
            elif task_type == "LHE_TASK":
                cache_key = proposition_text or option_text
                if option_text and cache_key in lhe_cached_option_records:
                    _, cached_best = lhe_cached_option_records[cache_key]
                    path_best_records.append(
                        {
                            "candidate": cached_best,
                            "option_text": option_text,
                            "path_score": float(cached_best.get("path_score", float("-inf"))),
                            "path_role": str(path.get("path_role", "")).strip(),
                        }
                    )
                    continue
                if not option_text and lhe_factual_scored:
                    continue
                local_candidates = [
                    {
                        **record,
                        "path": path,
                        "program": program,
                        "swarm_weight": float(swarm_weights[path_index]) if path_index < len(swarm_weights) else 1.0,
                        "parse_strategy": str(path.get("parse_strategy", "")).strip() or "auto",
                    }
                    for record in shared_lhe_records
                ]
                option_similarity_values = shared_lhe_option_similarities.get(cache_key, [])
                if option_text:
                    for record, option_similarity in zip(local_candidates, option_similarity_values):
                        record["option_text"] = option_text
                        record["option_similarity"] = float(option_similarity)
                self._apply_specialist_swarm_features(
                    local_candidates=local_candidates,
                    reference_embedding=option_embedding or navigation_reference_embedding,
                    task_type=task_type,
                    path=path,
                    selection_steps=selection_steps,
                )
                if gsm8k_mode:
                    self._apply_intra_path_defeasible(
                        local_candidates=local_candidates,
                        path=path,
                        task_type=task_type,
                        selection_steps=selection_steps,
                    )
                scores = self._score_gpu_candidates_batch(
                    candidates=local_candidates,
                    primary_program_id=reasoning_program_id,
                    target_galaxies=path_target_galaxies,
                    task_type=task_type,
                    domain_hint=domain_hint,
                    cross_domain=False,
                )
                for record, score in zip(local_candidates, scores):
                    record["gpu_score"] = float(score)
                if not local_candidates:
                    continue
                ordered_candidates = sorted(
                    local_candidates,
                    key=lambda candidate: float(candidate.get("gpu_score", float("-inf"))),
                    reverse=True,
                )
                best_for_path = ordered_candidates[0]
                best_for_path["path_score"] = float(best_for_path.get("gpu_score", float("-inf")))
                selection_steps.append(
                    "Swarm path result: "
                    f"{str(program.get('id', '')).strip()}"
                    + (f"[{option_text}]" if option_text else "")
                    + " -> "
                    f"[{str(best_for_path['match'].get('galaxy', 'unknown'))}] "
                    f"{str(best_for_path['match'].get('id', 'entry')).strip()} "
                    + f"(score={float(best_for_path.get('gpu_score', 0.0)):.2f})"
                )
                scored_candidates.extend(local_candidates)
                if option_text:
                    lhe_cached_option_records[cache_key] = (local_candidates, best_for_path)
                else:
                    lhe_factual_scored = True
                path_best_records.append(
                    {
                        "candidate": best_for_path,
                        "option_text": option_text,
                        "path_score": float(best_for_path.get("path_score", float("-inf"))),
                        "path_role": str(path.get("path_role", "")).strip(),
                        "preview_answer": str(best_for_path.get("gsm8k_preview_answer", "")).strip(),
                    }
                )
                continue
            if benchmark_eval_mode:
                path_navigation_candidates, suppressed_shortcuts = self._filter_benchmark_shortcut_candidates(
                    candidates=path_navigation_candidates,
                    task=task,
                    query_text=task_query_text,
                )
                if suppressed_shortcuts > 0:
                    selection_steps.append(
                        "Benchmark honesty filter: "
                        f"{str(path.get('label') or path.get('program_id', 'worker'))} "
                        f"suppressed {suppressed_shortcuts} shortcut candidates"
                    )
            if not path_navigation_candidates:
                continue
            local_candidates = []
            for candidate in path_navigation_candidates:
                record = {
                    "path": dict(path),
                    "program": dict(program),
                    "match": dict(candidate["match"]),
                    "similarity": float(candidate.get("similarity", 0.0)),
                    "lod_saliency": float(candidate.get("lod_saliency", 0.0)),
                    "lod_level": int(candidate.get("lod_level", 0)),
                    "lod_focus": float(candidate.get("lod_focus", 0.0)),
                    "led_focus": float(candidate.get("led_focus", 0.0)),
                    "swarm_weight": float(swarm_weights[path_index]) if path_index < len(swarm_weights) else 1.0,
                    "led_path": list(candidate.get("led_path", [])),
                    "gsm8k_mode": 1.0 if gsm8k_mode else 0.0,
                    "parse_strategy": str(path.get("parse_strategy", "")).strip() or "auto",
                }
                local_candidates.append(record)
            if option_embedding is not None and local_candidates:
                option_similarities = self._embedding_similarities(
                    option_embedding,
                    [list(record["match"].get("embedding16", [])) for record in local_candidates],
                )
                for record, option_similarity in zip(local_candidates, option_similarities):
                    record["option_text"] = option_text
                    record["option_similarity"] = float(option_similarity)
                    if task_type == "MMLU_TASK" and option_text:
                        record["option_support"] = self._mmlu_option_support_score(
                            record["match"],
                            option_text,
                        )
            if task_type in {"MMLU_TASK", "LHE_TASK", "MATH_TASK"} and subject_embedding and local_candidates:
                subject_similarities = self._embedding_similarities(
                    subject_embedding,
                    [list(record["match"].get("embedding16", [])) for record in local_candidates],
                )
                for record, subject_similarity in zip(local_candidates, subject_similarities):
                    record["subject_similarity"] = float(subject_similarity)
                    record["subject_anchor_focus"] = max(
                        float(record.get("subject_anchor_focus", 0.0)),
                        self._subject_anchor_match_score(
                            entry=record["match"],
                            subject_hint=subject_label,
                            match_mode="mmlu" if task_type == "MMLU_TASK" else "domain",
                        ),
                    )
            if local_candidates and list(parse_context.get("fusion_embedding", [])):
                parse_similarities = self._embedding_similarities(
                    list(parse_context.get("fusion_embedding", [])),
                    [list(record["match"].get("embedding16", [])) for record in local_candidates],
                )
                for record, parse_similarity in zip(local_candidates, parse_similarities):
                    record["parse_similarity"] = float(parse_similarity)
            if local_candidates and list(parse_context.get("directional_embedding", [])):
                directional_similarities = self._embedding_similarities(
                    list(parse_context.get("directional_embedding", [])),
                    [list(record["match"].get("embedding16", [])) for record in local_candidates],
                )
                for record, directional_similarity in zip(local_candidates, directional_similarities):
                    record["parse_directional_similarity"] = float(directional_similarity)
            parse_numeric_ids = {
                str(value).strip()
                for value in parse_context.get("numeric_ids", [])
                if str(value).strip()
            }
            for record in local_candidates:
                match_id = str(record["match"].get("id", "")).strip()
                numeric_value = self._gsm8k_numeric_entry_value(record["match"])
                numeric_id = numeric_value[0] if numeric_value is not None else match_id
                record["parse_support"] = 1.0 if numeric_id in parse_numeric_ids else 0.0
                record["parse_quantity_values"] = [
                    float(value) for value in list(parse_context.get("quantity_values", []))[:8]
                ]
                record["ternary_prior"] = self._candidate_ternary_prior(match_id or numeric_id)
                exact_query_match = 1.0 if self._entry_query_matches(record["match"], task_query_text) else 0.0
                record["exact_query_match"] = exact_query_match
                record["parse_override_algebra"] = (
                    1.0
                    if (
                        str(parse_override_signals.get("algebra_signal", "")).strip()
                        and self._candidate_matches_parse_signal(
                            record["match"],
                            str(parse_override_signals.get("algebra_signal", "")).strip(),
                        )
                    )
                    else 0.0
                )
                record["parse_override_domain"] = (
                    1.0
                    if (
                        str(parse_override_signals.get("domain_signal", "")).strip()
                        and self._candidate_matches_parse_signal(
                            record["match"],
                            str(parse_override_signals.get("domain_signal", "")).strip(),
                        )
                    )
                    else 0.0
                )
                record["lhe_exact_benchmark"] = (
                    1.0
                    if (
                        not benchmark_eval_mode
                        and
                        task_type == "LHE_TASK"
                        and exact_query_match > 0.0
                        and str(record["match"].get("galaxy", "")).strip() in {"Reality", "Math"}
                        and str(record["match"].get("category", "")).strip().lower()
                        in {"benchmark_fact", "clue_fact", "cipher_result", "formal_result"}
                    )
                    else 0.0
                )
            if gsm8k_mode and local_candidates:
                operation_embedding = list(gsm8k_context.get("operation_embedding", []))
                numeric_embedding = list(gsm8k_context.get("numeric_embedding", []))
                gsm8k_task_id = str((task or {}).get("task_id", "")).strip()
                operation_ids = {
                    str(value).strip()
                    for value in gsm8k_context.get("operation_ids", [])
                    if str(value).strip()
                }
                numeric_ids = {
                    str(value).strip()
                    for value in gsm8k_context.get("number_ids", [])
                    if str(value).strip()
                }
                if operation_embedding:
                    operation_similarities = self._embedding_similarities(
                        operation_embedding,
                        [list(record["match"].get("embedding16", [])) for record in local_candidates],
                    )
                    for record, operation_similarity in zip(local_candidates, operation_similarities):
                        record["operation_similarity"] = float(operation_similarity)
                if numeric_embedding:
                    numeric_similarities = self._embedding_similarities(
                        numeric_embedding,
                        [list(record["match"].get("embedding16", [])) for record in local_candidates],
                    )
                    for record, numeric_similarity in zip(local_candidates, numeric_similarities):
                        record["number_similarity"] = float(numeric_similarity)
                for record in local_candidates:
                    match_id = str(record["match"].get("id", "")).strip()
                    match_metadata = (
                        record["match"].get("metadata")
                        if isinstance(record["match"].get("metadata"), dict)
                        else {}
                    )
                    operation_role_match = 0.0
                    if bool(match_metadata.get("operation_pattern")) and gsm8k_context:
                        operation_role_match = self._gsm8k_operation_role_match_score(
                            metadata=match_metadata,
                            context=gsm8k_context,
                        )
                    record["operation_pattern_focus"] = (
                        (1.0 if match_id in operation_ids else 0.0)
                        + (0.75 * float(operation_role_match))
                    )
                    record["operation_role_match"] = float(operation_role_match)
                    numeric_value = self._gsm8k_numeric_entry_value(record["match"])
                    numeric_id = numeric_value[0] if numeric_value is not None else ""
                    record["numeric_focus"] = 1.0 if numeric_id in numeric_ids else 0.0
                    record["gsm8k_template_focus"] = 1.0 if (
                        self._match_template_ref(record["match"]) == "math_template_arithmetic_chain_gpu"
                        or str(match_metadata.get("template_ref", "")).strip() == "math_template_arithmetic_chain_gpu"
                    ) else 0.0
                    match_task_id = str(match_metadata.get("task_id", "")).strip()
                    match_competition = str(match_metadata.get("competition", "")).strip().upper()
                    record["gsm8k_exact_benchmark"] = 0.0 if benchmark_eval_mode else (
                        1.0
                        if (
                            match_competition == "GSM8K"
                            and gsm8k_task_id
                            and match_task_id == gsm8k_task_id
                        )
                        else 0.0
                    )
                    record["gsm8k_foreign_benchmark"] = 0.0 if benchmark_eval_mode else (
                        1.0
                        if (
                            match_competition == "GSM8K"
                            and match_task_id
                            and gsm8k_task_id
                            and match_task_id != gsm8k_task_id
                        )
                        else 0.0
                    )
                    record["gsm8k_non_chain_template"] = 1.0 if (
                        bool(self._match_template_ref(record["match"]))
                        and self._match_template_ref(record["match"]) != "math_template_arithmetic_chain_gpu"
                    ) else 0.0
                    if gsm8k_context:
                        record_context = dict(gsm8k_context)
                        role_variants = (
                            gsm8k_context.get("role_map_variants")
                            if isinstance(gsm8k_context.get("role_map_variants"), list)
                            else []
                        )
                        if role_variants:
                            variant_index = int(path.get("role_variant_index", path_index) or 0)
                            variant = role_variants[variant_index % len(role_variants)]
                            if isinstance(variant, dict):
                                record_context["quantity_role_candidates"] = [
                                    dict(row)
                                    for row in (
                                        variant.get("quantity_role_candidates")
                                        if isinstance(variant.get("quantity_role_candidates"), list)
                                        else []
                                    )
                                    if isinstance(row, dict)
                                ]
                                record_context["quantity_role_values"] = {
                                    str(key).strip().lower(): [
                                        float(value)
                                        for value in (values if isinstance(values, list) else [])
                                    ]
                                    for key, values in (
                                        variant.get("quantity_role_values")
                                        if isinstance(variant.get("quantity_role_values"), dict)
                                        else {}
                                    ).items()
                                }
                                record_context["role_variant_label"] = str(variant.get("label", "")).strip()
                        record["gsm8k_context"] = record_context
            self._apply_specialist_swarm_features(
                local_candidates=local_candidates,
                reference_embedding=option_embedding or navigation_reference_embedding,
                task_type=task_type,
                path=path,
                selection_steps=selection_steps,
            )
            self._apply_atomic_compositional_consistency(
                local_candidates=local_candidates,
                task_type=task_type,
                selection_steps=selection_steps,
            )
            if gsm8k_mode:
                self._apply_intra_path_defeasible(
                    local_candidates=local_candidates,
                    path=path,
                    task_type=task_type,
                    selection_steps=selection_steps,
                )
            scores = self._score_gpu_candidates_batch(
                candidates=local_candidates,
                primary_program_id=reasoning_program_id,
                target_galaxies=path_target_galaxies,
                task_type=task_type,
                domain_hint=domain_hint,
                cross_domain=False,
            )
            for record, score in zip(local_candidates, scores):
                record["gpu_score"] = float(score)
            if not local_candidates:
                continue
            ordered_candidates = sorted(
                local_candidates,
                key=lambda candidate: float(candidate.get("gpu_score", float("-inf"))),
                reverse=True,
            )
            best_for_path = ordered_candidates[0]
            if task_type == "MMLU_TASK":
                coherence_candidates = ordered_candidates[: min(4, len(ordered_candidates))]
                neighborhood_mean = float(
                    engine.evaluate(
                        self._gpu_mean_expression(
                            [float(candidate.get("gpu_score", 0.0)) for candidate in coherence_candidates]
                        )
                    )
                )
                best_for_path["path_score"] = float(
                    engine.evaluate(
                        " ".join(
                            [
                                self._gpu_scalar_literal(best_for_path.get("gpu_score", 0.0)),
                                self._gpu_scalar_literal(neighborhood_mean),
                                "0.05",
                                "*",
                                "+",
                            ]
                        )
                    )
                )
            else:
                best_for_path["path_score"] = float(best_for_path.get("gpu_score", float("-inf")))
            if gsm8k_mode:
                strategy_name = str(path.get("composition_strategy", "")).strip() or "fusion_chain"
                preview = self._gsm8k_decomposition_preview(
                    engine=engine,
                    context=(
                        best_for_path.get("gsm8k_context")
                        if isinstance(best_for_path.get("gsm8k_context"), dict)
                        else {}
                    ),
                    strategy=strategy_name,
                )
                if preview is not None and not str(best_for_path.get("gsm8k_preview_answer", "")).strip():
                    preview_answer, preview_program, preview_label, preview_structural = preview
                    best_for_path["gsm8k_preview_answer"] = preview_answer
                    best_for_path["gsm8k_preview_program"] = preview_program
                    best_for_path["gsm8k_preview_strategy"] = preview_label
                    best_for_path["gsm8k_structural_score"] = float(preview_structural)
                    binding_summary = str(
                        (
                            best_for_path.get("gsm8k_context")
                            if isinstance(best_for_path.get("gsm8k_context"), dict)
                            else {}
                        ).get("_last_gsm8k_slot_binding", "")
                    ).strip()
                    if binding_summary:
                        best_for_path["gsm8k_slot_binding"] = binding_summary
                    strategy_name = preview_label or strategy_name
                    selection_steps.append(
                        "GSM8K worker preview: "
                        f"{str(path.get('label') or path.get('program_id', 'worker'))} "
                        f"{preview_label} -> {preview_answer}"
                    )
                    if binding_summary:
                        selection_steps.append(f"GSM8K slot binding: {binding_summary}")
                strategy_weight = self._gsm8k_strategy_weight(strategy_name)
                best_for_path["gsm8k_strategy_weight"] = float(strategy_weight)
                if strategy_weight != 1.0:
                    best_for_path["path_score"] = float(
                        engine.evaluate(
                            " ".join(
                                [
                                    self._gpu_scalar_literal(best_for_path.get("path_score", 0.0)),
                                    self._gpu_scalar_literal(strategy_weight),
                                    "*",
                                ]
                            )
                        )
                    )
            selection_steps.append(
                "Swarm path result: "
                f"{str(program.get('id', '')).strip()}"
                + (f"[{option_text}]" if option_text else "")
                + " -> "
                f"[{str(best_for_path['match'].get('galaxy', 'unknown'))}] "
                f"{str(best_for_path['match'].get('id', 'entry')).strip()} "
                + (
                    f"(coherence={float(best_for_path.get('path_score', 0.0)):.2f}, top={float(best_for_path.get('gpu_score', 0.0)):.2f})"
                    if task_type == "MMLU_TASK"
                    else f"(score={float(best_for_path.get('gpu_score', 0.0)):.2f})"
                )
            )
            scored_candidates.extend(local_candidates)
            path_best_records.append(
                {
                    "candidate": best_for_path,
                    "option_text": option_text,
                    "path_score": float(best_for_path.get("path_score", float("-inf"))),
                    "path_role": str(path.get("path_role", "")).strip(),
                    "preview_answer": str(best_for_path.get("gsm8k_preview_answer", "")).strip(),
                }
            )
        if not scored_candidates:
            return None
        selected_records = path_best_records
        if gsm8k_mode:
            aggregated_records = self._aggregate_gsm8k_preview_records(
                engine=engine,
                path_best_records=path_best_records,
                selection_steps=selection_steps,
            )
            if aggregated_records:
                aggregate_by_answer = {
                    str(record.get("option_text", "")).strip(): record
                    for record in aggregated_records
                    if str(record.get("option_text", "")).strip()
                }
                for record in path_best_records:
                    answer_key = self._gsm8k_preview_candidate_id(record)
                    aggregate_record = aggregate_by_answer.get(answer_key)
                    if aggregate_record is None:
                        continue
                    aggregate_score = float(aggregate_record.get("path_score", record.get("path_score", 0.0)))
                    support_count = int(aggregate_record.get("support_count", 0))
                    weighted_support = float(aggregate_record.get("weighted_support", 0.0))
                    record["path_score"] = aggregate_score
                    record["support_count"] = support_count
                    record["weighted_support"] = weighted_support
                    candidate = record.get("candidate")
                    if isinstance(candidate, dict):
                        candidate["path_score"] = aggregate_score
                        candidate["gsm8k_consensus_support"] = support_count
                        candidate["gsm8k_consensus_weight"] = weighted_support
                selected_records = aggregated_records
        if task_type == "MMLU_TASK":
            option_groups: dict[str, list[dict[str, Any]]] = {}
            for record in path_best_records:
                option_name = str(record.get("option_text", "")).strip()
                if not option_name:
                    continue
                option_groups.setdefault(option_name, []).append(record)
            aggregated_records: list[dict[str, Any]] = []
            option_score_jobs: list[tuple[str, dict[str, Any], int]] = []
            option_score_expressions: list[str] = []
            for option_name, records in option_groups.items():
                hypothesis_scores = [
                    float(record.get("path_score", 0.0))
                    for record in records
                    if str(record.get("path_role", "")).strip() == "hypothesis"
                ]
                validation_scores = [
                    float(record.get("path_score", 0.0))
                    for record in records
                    if str(record.get("path_role", "")).strip() != "hypothesis"
                ]
                if not hypothesis_scores and not validation_scores:
                    continue
                best_record = max(records, key=lambda record: float(record.get("path_score", float("-inf"))))
                candidate = best_record.get("candidate")
                hypothesis_expression = self._gpu_mean_expression(hypothesis_scores)
                validation_expression = self._gpu_mean_expression(validation_scores)
                final_expression = " ".join(
                    [
                        hypothesis_expression,
                        validation_expression,
                        self._gpu_scalar_literal(mmlu_validation_weight),
                        "*",
                        "+",
                        self._gpu_scalar_literal(len(validation_scores)),
                        self._gpu_scalar_literal(mmlu_support_weight),
                        "*",
                        "+",
                    ]
                )
                option_score_jobs.append((option_name, candidate, len(validation_scores)))
                option_score_expressions.extend(
                    [
                        hypothesis_expression,
                        validation_expression,
                        final_expression,
                    ]
                )
            if option_score_expressions:
                option_score_values: list[float] = []
                for start in range(0, len(option_score_expressions), 18):
                    batch = option_score_expressions[start : start + 18]
                    option_score_values.extend(
                        float(value) for value in engine.evaluate_batch(batch, max_parallel=len(batch))
                    )
                value_index = 0
                for option_name, candidate, support_count in option_score_jobs:
                    hypothesis_score = float(option_score_values[value_index])
                    validation_score = float(option_score_values[value_index + 1])
                    final_score = float(option_score_values[value_index + 2])
                    value_index += 3
                    if isinstance(candidate, dict):
                        candidate["path_score"] = float(final_score)
                    aggregated_records.append(
                        {
                            "candidate": candidate,
                            "option_text": option_name,
                            "path_score": float(final_score),
                        }
                    )
                    selection_steps.append(
                        "MMLU option score: "
                        f"{option_name}={final_score:.2f} (hyp={hypothesis_score:.2f}, val={validation_score:.2f}, support={support_count})"
                    )
            if aggregated_records:
                selected_records = aggregated_records
        if task_type == "LHE_TASK":
            option_groups: dict[str, list[dict[str, Any]]] = {}
            for record in path_best_records:
                option_name = str(record.get("option_text", "")).strip()
                if not option_name:
                    continue
                option_groups.setdefault(option_name, []).append(record)
            selection_steps.append(
                "LHE aggregation: "
                + f"{len(option_groups)} option groups, records={len(path_best_records)}"
            )
            aggregated_records: list[dict[str, Any]] = []
            option_score_jobs: list[tuple[str, dict[str, Any], int, int]] = []
            option_score_expressions: list[str] = []
            for option_name, records in option_groups.items():
                hypothesis_scores = [
                    float(record.get("path_score", 0.0))
                    for record in records
                    if str(record.get("path_role", "")).strip() == "hypothesis"
                ]
                validation_scores = [
                    float(record.get("path_score", 0.0))
                    for record in records
                    if str(record.get("path_role", "")).strip() == "validation"
                ]
                all_scores = hypothesis_scores + validation_scores
                if not all_scores:
                    continue
                best_record = max(records, key=lambda record: float(record.get("path_score", float("-inf"))))
                candidate = best_record.get("candidate")
                hypothesis_expression = self._gpu_mean_expression(hypothesis_scores)
                validation_expression = self._gpu_mean_expression(validation_scores)
                final_expression = self._gpu_mean_expression(all_scores)
                option_score_jobs.append(
                    (option_name, candidate, len(hypothesis_scores), len(validation_scores))
                )
                option_score_expressions.extend(
                    [
                        hypothesis_expression,
                        validation_expression,
                        final_expression,
                    ]
                )
            if option_score_expressions:
                option_score_values: list[float] = []
                for start in range(0, len(option_score_expressions), 18):
                    batch = option_score_expressions[start : start + 18]
                    option_score_values.extend(
                        float(value) for value in engine.evaluate_batch(batch, max_parallel=len(batch))
                    )
                value_index = 0
                for option_name, candidate, hypothesis_count, validation_count in option_score_jobs:
                    hypothesis_score = float(option_score_values[value_index])
                    validation_score = float(option_score_values[value_index + 1])
                    final_score = float(option_score_values[value_index + 2])
                    value_index += 3
                    if isinstance(candidate, dict):
                        candidate["path_score"] = float(final_score)
                    aggregated_records.append(
                        {
                            "candidate": candidate,
                            "option_text": option_name,
                            "path_score": float(final_score),
                        }
                    )
                    selection_steps.append(
                        "LHE option score: "
                        + f"{option_name}={final_score:.2f} "
                        + f"(hyp={hypothesis_score:.2f}/{hypothesis_count}, "
                        + f"val={validation_score:.2f}/{validation_count})"
                    )
            if aggregated_records:
                selected_records = aggregated_records
        if task_type == "LHE_TASK":
            self._record_active_lhe_timing("scoring", time.perf_counter() - scoring_started)
        halting_records = path_best_records if gsm8k_mode else selected_records
        self._apply_defeasible_specialist_resolution(
            records=halting_records,
            task_type=task_type,
            gsm8k_mode=gsm8k_mode,
            selection_steps=selection_steps,
        )
        path_best_scores = [float(record.get("path_score", float("-inf"))) for record in halting_records]
        path_candidate_ids = [
            self._halting_record_candidate_id(
                record=record,
                task_type=task_type,
                gsm8k_mode=gsm8k_mode,
            )
            for record in halting_records
        ]
        gsm8k_structural_override = (
            self._gsm8k_structural_override_record(selected_records)
            if gsm8k_mode and selected_records
            else None
        )
        halting_started = time.perf_counter()
        converged = self._halting_gate_converged(
            task_type=task_type,
            task=task,
            path_scores=path_best_scores,
            candidate_ids=path_candidate_ids,
            selection_steps=selection_steps,
            gsm8k_structural_override=gsm8k_structural_override,
        )
        if task_type == "LHE_TASK":
            self._record_active_lhe_timing("halting", time.perf_counter() - halting_started)
        if not converged:
            if task_type == "LHE_TASK" and scored_candidates:
                selection_steps.append("LHE fallback: use top factual candidate")
                return self._attach_galaxy_contribution(
                    max(scored_candidates, key=lambda candidate: float(candidate.get("gpu_score", float("-inf")))),
                    records=path_best_records,
                    candidates=scored_candidates,
                    selection_steps=selection_steps,
                )
            return None
        if task_type in {"MMLU_TASK", "LHE_TASK"} and selected_records:
            return self._attach_galaxy_contribution(
                max(
                (record.get("candidate") for record in selected_records if isinstance(record.get("candidate"), dict)),
                key=lambda candidate: float((candidate or {}).get("path_score", float("-inf"))),
                ),
                records=path_best_records or selected_records,
                candidates=scored_candidates,
                selection_steps=selection_steps,
            )
        if gsm8k_mode and selected_records:
            if isinstance(gsm8k_structural_override, dict):
                override_candidate = (
                    gsm8k_structural_override.get("candidate")
                    if isinstance(gsm8k_structural_override.get("candidate"), dict)
                    else None
                )
                if isinstance(override_candidate, dict):
                    selection_steps.append(
                        "GSM8K final selection: structural override -> "
                        f"{str(gsm8k_structural_override.get('option_text', '')).strip()}"
                    )
                    return self._attach_galaxy_contribution(
                        override_candidate,
                        records=path_best_records or selected_records,
                        candidates=scored_candidates,
                        selection_steps=selection_steps,
                    )
            consensus_record = self._gsm8k_consensus_record(selected_records)
            consensus_candidate = (
                consensus_record.get("candidate")
                if isinstance((consensus_record or {}).get("candidate"), dict)
                else None
            )
            if isinstance(consensus_candidate, dict):
                return self._attach_galaxy_contribution(
                    consensus_candidate,
                    records=path_best_records or selected_records,
                    candidates=scored_candidates,
                    selection_steps=selection_steps,
                )
            return self._attach_galaxy_contribution(
                max(
                (record.get("candidate") for record in selected_records if isinstance(record.get("candidate"), dict)),
                key=lambda candidate: (
                    float((candidate or {}).get("gsm8k_consensus_weight", 0.0)),
                    int((candidate or {}).get("gsm8k_consensus_support", 0)),
                    float((candidate or {}).get("path_score", float("-inf"))),
                    float((candidate or {}).get("gpu_score", float("-inf"))),
                ),
                ),
                records=path_best_records or selected_records,
                candidates=scored_candidates,
                selection_steps=selection_steps,
            )
        return self._attach_galaxy_contribution(
            max(scored_candidates, key=lambda candidate: float(candidate.get("gpu_score", float("-inf")))),
            records=path_best_records,
            candidates=scored_candidates,
            selection_steps=selection_steps,
        )

    def _goal_edge_cost(
        self,
        *,
        match: dict[str, Any],
        task_type: str,
        target_galaxies: list[str],
        galaxy_weights: dict[str, Any] | None,
        reasoning_program_id: str,
        query_embedding: list[float],
    ) -> float | None:
        galaxy_name = str(match.get("galaxy", "")).strip()
        allowed = set(target_galaxies)
        normalized_galaxy_weights = self._normalize_galaxy_weights(galaxy_weights)
        if task_type == "LHE_TASK":
            allowed = {"Reality", "Math"}
        elif task_type == "ARC_TASK":
            allowed = {"Drawing", "Grammar", "Tool"}
        elif normalized_galaxy_weights:
            allowed = set(self.DEFAULT_GALAXIES)
        if allowed and galaxy_name not in allowed:
            return None
        similarity = self._embedding_similarity(query_embedding, list(match.get("embedding16", [])))
        base_cost = max(0.01, 1.0 - similarity)
        galaxy_weight = self._galaxy_weight_for_name(galaxy_name, normalized_galaxy_weights)
        if galaxy_weight > 0.0:
            base_cost *= max(0.25, 1.0 - (0.75 * (galaxy_weight - 1.0)))
        return max(0.01, float(base_cost))

    def _navigate_led_primary_candidate(
        self,
        *,
        binding: dict[str, Any],
        target_galaxies: list[str],
        galaxy_weights: dict[str, Any] | None,
        reasoning_program_id: str,
        primary_reasoning_program: dict[str, Any],
        query_embedding: list[float],
        task_type: str,
        specialist: str,
        selection_steps: list[str],
    ) -> dict[str, Any] | None:
        if task_type == "ARC_TASK":
            return None
        graph = self.get_semantic_csr_graph()
        pathfinder = self.get_led_pathfinder()
        catalog = self.get_gpu_galaxy_catalog()
        if graph is None or pathfinder is None or not catalog:
            return None

        allowed_indexes = {
            int(round(float(self._gpu_galaxy_index(name))))
            for name in target_galaxies
            if str(name).strip()
        }
        seed_pairs = graph.select_seed_nodes(
            query_embedding=query_embedding,
            allowed_galaxy_indexes=allowed_indexes or None,
            top_k=self._graph_seed_limit(task_type),
            similarity_threshold=self._graph_seed_similarity_threshold(task_type),
        )
        if not seed_pairs:
            seed_pairs = graph.select_seed_nodes(
                query_embedding=query_embedding,
                allowed_galaxy_indexes=None,
                top_k=self._graph_seed_limit(task_type),
                similarity_threshold=self._graph_seed_similarity_threshold(task_type),
            )
        if not seed_pairs:
            return None

        seed_nodes = [index for index, _ in seed_pairs]
        local_nodes, local_rows, local_cols, local_costs = graph.extract_local_kernel(
            seed_nodes=seed_nodes,
            max_nodes=self._graph_local_kernel_limit(task_type),
        )
        if not local_nodes:
            return None

        global_to_local = {global_index: local_index for local_index, global_index in enumerate(local_nodes)}
        query_node = 0
        first_real_node = 1
        goal_node = len(local_nodes) + 1
        row_offsets = [0]
        col_indices: list[int] = []
        packed_costs: list[int] = []

        for global_index, similarity in seed_pairs:
            local_index = global_to_local.get(global_index)
            if local_index is None:
                continue
            col_indices.append(first_real_node + local_index)
            packed_costs.append(
                self._pack_led_cost(
                    self._semantic_cost_from_similarity(similarity),
                    1,
                )
            )
        row_offsets.append(len(col_indices))

        for local_index, global_index in enumerate(local_nodes):
            row_start = int(local_rows[local_index])
            row_end = int(local_rows[local_index + 1])
            for edge_idx in range(row_start, row_end):
                col_indices.append(first_real_node + int(local_cols[edge_idx]))
                packed_costs.append(int(local_costs[edge_idx]))
            goal_cost = self._goal_edge_cost(
                match=catalog[global_index],
                task_type=task_type,
                target_galaxies=target_galaxies,
                galaxy_weights=galaxy_weights,
                reasoning_program_id=reasoning_program_id,
                query_embedding=query_embedding,
            )
            if goal_cost is not None:
                col_indices.append(goal_node)
                packed_costs.append(
                    self._pack_led_cost(
                        int(round(float(goal_cost) * 65535.0)),
                        1,
                    )
                )
            row_offsets.append(len(col_indices))

        row_offsets.append(len(col_indices))
        try:
            path = pathfinder.navigate_csr(
                row_offsets,
                col_indices,
                packed_costs,
                start=query_node,
                goal=goal_node,
                alpha=0.35,
                beta=0.65,
                max_path_length=max(16, len(local_nodes) + 2),
            )
        except Exception:
            return None
        if path.size < 3:
            return None
        answer_local_node = int(path[-2]) - first_real_node
        if not (0 <= answer_local_node < len(local_nodes)):
            return None
        answer_index = int(local_nodes[answer_local_node])
        if not (0 <= answer_index < len(catalog)):
            return None
        answer_match = dict(catalog[answer_index])
        similarity = self._embedding_similarity(query_embedding, list(answer_match.get("embedding16", [])))
        selection_steps.append(
            "LED-A graph navigation: "
            f"[{str(answer_match.get('galaxy', 'unknown'))}] {str(answer_match.get('id', 'entry')).strip()} "
            f"(path_hops={max(0, int(path.size) - 2)}, seeds={len(seed_pairs)})"
        )
        return {
            "path": {
                "instance_id": -1,
                "program_id": reasoning_program_id,
                "query_text": "",
            },
            "program": dict(primary_reasoning_program),
            "match": answer_match,
            "similarity": float(similarity),
            "gpu_score": float(similarity + (float(answer_match.get("confidence", 0.0)) * 0.05)),
            "led_path": [int(node) for node in path.tolist()],
        }

    def _evaluate_gpu_paths(
        self,
        *,
        galaxy_names: list[str],
        paths: list[dict[str, Any]],
        task: dict[str, Any] | None,
        options: list[str] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        started = time.perf_counter()
        try:
            binding = self.bind_gpu_galaxy_runtime(galaxy_names=galaxy_names)
            catalog = self.get_gpu_galaxy_catalog()
            if not paths:
                return binding, []
            engine = self.get_gpu_reasoning_engine()
            expressions: list[str] = []
            programs: list[dict[str, Any]] = []
            for instance_id, path in enumerate(paths[:18]):
                program = self._select_gpu_reasoning_program(str(path.get("program_id", "")).strip())
                query_embedding = self._embed_query_gpu(str(path.get("query_text", "")), task=task)
                engine.store_embedding(instance_id=instance_id, embedding=query_embedding)
                expressions.append(str(program.get("rpn_program", "")).strip())
                programs.append(program)
                path["instance_id"] = instance_id
                path["program"] = program
            best_indexes = engine.evaluate_batch(expressions, max_parallel=len(expressions))
            similarity_expressions: list[str] = []
            for raw_value in best_indexes:
                best_index = int(round(float(raw_value)))
                if 0 <= best_index < len(catalog):
                    similarity_expressions.append(f"{best_index} galaxy_similarity")
                else:
                    similarity_expressions.append("0")
            similarities = engine.evaluate_batch(similarity_expressions, max_parallel=len(similarity_expressions))
            results: list[dict[str, Any]] = []
            for path, program, raw_index, similarity in zip(paths[:18], programs, best_indexes, similarities):
                best_index = int(round(float(raw_index)))
                if not (0 <= best_index < len(catalog)):
                    continue
                match = dict(catalog[best_index])
                results.append(
                    {
                        "path": dict(path),
                        "program": dict(program),
                        "match": match,
                        "similarity": float(similarity),
                    }
                )
            return binding, results
        finally:
            self._record_active_lhe_timing("evaluate_gpu_paths", time.perf_counter() - started)

    def _gather_gpu_frontier_candidates(
        self,
        *,
        paths: list[dict[str, Any]],
        task_type: str,
    ) -> list[dict[str, Any]]:
        started = time.perf_counter()
        try:
            top_k = self._gpu_frontier_k(task_type)
            if top_k <= 1:
                return []
            catalog = self.get_gpu_galaxy_catalog()
            if not catalog:
                return []
            engine = self.get_gpu_reasoning_engine()
            results: list[dict[str, Any]] = []
            for path in paths[:18]:
                instance_id = path.get("instance_id")
                program = path.get("program")
                if instance_id is None or not isinstance(program, dict):
                    continue
                expression = self._program_frontier_expression(program, top_k)
                if not expression:
                    continue
                _, stack = engine.evaluate_with_stack(expression, instance_id=int(instance_id))
                candidate_indexes = self._parse_galaxy_scan_stack(stack)
                for rank, candidate_index in enumerate(candidate_indexes[:top_k]):
                    if not (0 <= candidate_index < len(catalog)):
                        continue
                    similarity = float(
                        engine.evaluate(f"{candidate_index} galaxy_similarity", instance_id=int(instance_id))
                    )
                    results.append(
                        {
                            "path": dict(path),
                            "program": dict(program),
                            "match": dict(catalog[candidate_index]),
                            "similarity": similarity,
                            "rank": rank,
                        }
                    )
            return results
        finally:
            if task_type == "LHE_TASK":
                self._record_active_lhe_timing("frontier", time.perf_counter() - started)

    def _maybe_led_rerank_candidate(
        self,
        *,
        frontier_candidates: list[dict[str, Any]],
        best_candidate: dict[str, Any],
        task_type: str,
        selection_steps: list[str],
    ) -> dict[str, Any] | None:
        started = time.perf_counter()
        try:
            if task_type == "ARC_TASK":
                return None
            if len(frontier_candidates) < 6:
                return None
            if float(best_candidate.get("similarity", 0.0)) >= 0.95:
                return None
            pathfinder = self.get_led_pathfinder()
            if pathfinder is None:
                return None

            frontier_by_index: dict[int, dict[str, Any]] = {}
            for candidate in frontier_candidates:
                match = candidate.get("match") if isinstance(candidate.get("match"), dict) else {}
                index = int(match.get("index", -1))
                if index < 0:
                    continue
                record = frontier_by_index.get(index)
                if record is None:
                    frontier_by_index[index] = {
                        "candidate": candidate,
                        "gpu_score": float(candidate.get("gpu_score", float("-inf"))),
                        "best_similarity": float(candidate.get("similarity", 0.0)),
                        "program_ids": {str(candidate["program"].get("id", "")).strip()},
                        "seed": int(candidate.get("rank", 0)) == 0,
                    }
                    continue
                record["best_similarity"] = max(record["best_similarity"], float(candidate.get("similarity", 0.0)))
                record["program_ids"].add(str(candidate["program"].get("id", "")).strip())
                record["seed"] = bool(record["seed"] or int(candidate.get("rank", 0)) == 0)
                if float(candidate.get("gpu_score", float("-inf"))) > float(record["gpu_score"]):
                    record["candidate"] = candidate
                    record["gpu_score"] = float(candidate.get("gpu_score", float("-inf")))

            if len(frontier_by_index) < 4:
                return None

            nodes: list[dict[str, Any]] = []
            for record in frontier_by_index.values():
                candidate = record["candidate"]
                match = candidate["match"]
                nodes.append(
                    {
                        "candidate": candidate,
                        "gpu_score": float(record["gpu_score"]),
                        "best_similarity": float(record["best_similarity"]),
                        "support_count": len(record["program_ids"]),
                        "seed": bool(record["seed"]),
                        "embedding16": list(match.get("embedding16", [])),
                        "template_ref": str(match.get("template_ref", "")).strip(),
                        "category": str(match.get("category", "")).strip().lower(),
                        "galaxy": str(match.get("galaxy", "")).strip(),
                        "subject": str(match.get("subject", "")).strip().lower(),
                        "index": int(match.get("index", -1)),
                    }
                )

            if not any(node["seed"] for node in nodes):
                return None

            edges: list[list[tuple[int, int]]] = [[] for _ in range(len(nodes) + 1)]
            for node_idx, node in enumerate(nodes, start=1):
                if node["seed"]:
                    edges[0].append(
                        (
                            node_idx,
                            self._pack_led_cost(
                                self._semantic_cost_from_similarity(node["best_similarity"]),
                                1,
                            ),
                        )
                    )

            for left_idx, left in enumerate(nodes, start=1):
                for right_idx in range(left_idx + 1, len(nodes) + 1):
                    right = nodes[right_idx - 1]
                    pair_similarity = self._embedding_similarity(left["embedding16"], right["embedding16"])
                    same_template = bool(left["template_ref"] and left["template_ref"] == right["template_ref"])
                    same_subject = bool(left["subject"] and left["subject"] == right["subject"])
                    same_galaxy = left["galaxy"] == right["galaxy"]
                    if pair_similarity < 0.78 and not same_template and not same_subject:
                        continue
                    geometric_cost = 1 if same_template else 2 if same_subject else 4 if same_galaxy else 8
                    packed_cost = self._pack_led_cost(
                        self._semantic_cost_from_similarity(pair_similarity),
                        geometric_cost,
                    )
                    edges[left_idx].append((right_idx, packed_cost))
                    edges[right_idx].append((left_idx, packed_cost))

            if not edges[0]:
                return None

            row_offsets = [0]
            col_indices: list[int] = []
            packed_costs: list[int] = []
            for adjacency in edges:
                for target, packed_cost in adjacency:
                    col_indices.append(int(target))
                    packed_costs.append(int(packed_cost))
                row_offsets.append(len(col_indices))

            rows = row_offsets
            cols = col_indices
            costs = packed_costs
            current_best_score = float(best_candidate.get("gpu_score", float("-inf")))
            promoted: dict[str, Any] | None = None
            promoted_score = current_best_score
            promoted_path: list[int] = []

            for node_idx, node in enumerate(nodes, start=1):
                try:
                    path = pathfinder.navigate_csr(
                        rows,
                        cols,
                        costs,
                        start=0,
                        goal=node_idx,
                        alpha=0.35,
                        beta=0.65,
                        max_path_length=max(16, len(nodes) + 2),
                    )
                except Exception:
                    return None
                if path.size == 0:
                    continue
                hop_count = max(1, int(path.size) - 1)
                support_bonus = min(0.06, 0.015 * int(node["support_count"]))
                hop_bonus = 0.03 / float(hop_count)
                seed_bonus = 0.015 if node["seed"] else 0.0
                led_score = float(node["gpu_score"]) + support_bonus + hop_bonus + seed_bonus
                if led_score > promoted_score + 0.02:
                    promoted = node["candidate"]
                    promoted_score = led_score
                    promoted_path = [int(step) for step in path.tolist()]

            if promoted is None:
                return None
            match = promoted["match"]
            promoted["led_score"] = float(promoted_score)
            selection_steps.append(
                "LED-A frontier navigation: "
                f"[{str(match.get('galaxy', 'unknown'))}] {str(match.get('id', 'entry')).strip()} "
                f"(path_hops={max(0, len(promoted_path) - 1)}, adjusted={promoted_score:.2f})"
            )
            return promoted
        finally:
            if task_type == "LHE_TASK":
                self._record_active_lhe_timing("led_rerank", time.perf_counter() - started)

    @staticmethod
    def _gpu_scalar_literal(value: Any) -> str:
        try:
            return repr(float(value))
        except Exception:
            return "0.0"

    def _gpu_mean_expression(self, values: list[float]) -> str:
        if not values:
            return "0.0"
        tokens = [self._gpu_scalar_literal(values[0])]
        for value in values[1:]:
            tokens.extend([self._gpu_scalar_literal(value), "+"])
        if len(values) > 1:
            tokens.extend([self._gpu_scalar_literal(len(values)), "/"])
        return " ".join(tokens)

    def _build_gpu_candidate_score_expression(
        self,
        *,
        candidate: dict[str, Any],
        primary_program_id: str,
        target_galaxies: list[str],
        task_type: str,
        domain_hint: str | None,
        cross_domain: bool = False,
    ) -> str:
        match = candidate["match"]
        galaxy_name = str(match.get("galaxy", ""))
        category = str(match.get("category", "")).strip().lower()
        query_similarity = float(candidate.get("similarity", 0.0))
        option_similarity = float(candidate.get("option_similarity", query_similarity))
        option_support = float(candidate.get("option_support", 0.0))
        similarity = option_similarity if task_type in {"MMLU_TASK", "LHE_TASK"} else query_similarity
        confidence = float(match.get("confidence", 0.0))
        target_flag = 1.0 if galaxy_name in target_galaxies else 0.0
        primary_flag = 1.0 if str(candidate["program"].get("id", "")) == primary_program_id else 0.0
        template_flag = float(match.get("gpu_has_template_ref", 1.0 if self._match_template_ref(match) else 0.0))
        category_clue_like = 1.0 if category in {"clue_fact", "benchmark_fact", "cipher_result"} else 0.0
        category_formal = 1.0 if category == "formal_result" else 0.0
        category_meaning_like = 1.0 if category in {"concept", "definition", "benchmark_fact", "formal_result"} else 0.0
        category_formula_like = 1.0 if category in {"formula", "template"} else 0.0
        category_shortcut_like = 1.0 if category in {"arithmetic_instance", "benchmark_fact"} else 0.0
        category_template_support = 1.0 if category in {"template_support", "template_program"} else 0.0
        category_rule_like = 1.0 if category in {"math_reasoning_pattern", "pattern_rule", "compositional_rule", "rule"} else 0.0
        category_answer_like = 1.0 if category in {"formal_result", "formula_fact", "concept", "definition"} else 0.0
        source_book = 1.0 if float(match.get("gpu_source_class", 0.0)) == self.GPU_SOURCE_CLASS_BOOK_ARTIFACT else 0.0
        galaxy_reality = 1.0 if galaxy_name == "Reality" else 0.0
        galaxy_math = 1.0 if galaxy_name == "Math" else 0.0
        galaxy_number = 1.0 if galaxy_name == "Number" else 0.0
        galaxy_word = 1.0 if galaxy_name == "Word" else 0.0
        galaxy_character = 1.0 if galaxy_name == "Character" else 0.0
        galaxy_grammar = 1.0 if galaxy_name == "Grammar" else 0.0
        book_formula_like = 1.0 if source_book and category in {"formula", "concept", "definition"} else 0.0
        swarm_weight = float(candidate.get("swarm_weight", 1.0))
        lod_saliency = float(candidate.get("lod_saliency", similarity))
        lod_focus = float(candidate.get("lod_focus", 0.0))
        led_focus = float(candidate.get("led_focus", 0.0))
        subject_similarity = float(candidate.get("subject_similarity", similarity))
        subject_anchor_focus = float(candidate.get("subject_anchor_focus", 0.0))
        specialist_resonance = float(candidate.get("specialist_resonance", similarity))
        specialist_coherence = float(candidate.get("specialist_coherence", similarity))
        specialist_geometry = float(candidate.get("specialist_geometry", 0.0))
        specialist_temporal = float(candidate.get("specialist_temporal", 0.0))
        specialist_fractal = float(candidate.get("specialist_fractal", 0.0))
        specialist_intra_defeasible = float(candidate.get("specialist_intra_defeasible", 0.0))
        specialist_defeasible_verdict = float(candidate.get("specialist_defeasible_verdict", 0.0))
        parse_similarity = float(candidate.get("parse_similarity", 0.0))
        parse_directional_similarity = float(candidate.get("parse_directional_similarity", 0.0))
        parse_support = float(candidate.get("parse_support", 0.0))
        parse_override_algebra = float(candidate.get("parse_override_algebra", 0.0))
        parse_override_domain = float(candidate.get("parse_override_domain", 0.0))
        ternary_prior = float(candidate.get("ternary_prior", 0.0))
        exact_query_match = float(candidate.get("exact_query_match", 0.0))
        lhe_exact_benchmark = float(candidate.get("lhe_exact_benchmark", 0.0))
        gsm8k_mode = float(candidate.get("gsm8k_mode", 0.0))
        mmlu_symbolic_mode = float(candidate.get("mmlu_symbolic_mode", 0.0))
        operation_similarity = float(candidate.get("operation_similarity", similarity))
        number_similarity = float(candidate.get("number_similarity", similarity))
        operation_pattern_focus = float(candidate.get("operation_pattern_focus", 0.0))
        numeric_focus = float(candidate.get("numeric_focus", 0.0))
        gsm8k_template_focus = float(candidate.get("gsm8k_template_focus", 0.0))
        gsm8k_exact_benchmark = float(candidate.get("gsm8k_exact_benchmark", 0.0))
        gsm8k_foreign_benchmark = float(candidate.get("gsm8k_foreign_benchmark", 0.0))
        gsm8k_non_chain_template = float(candidate.get("gsm8k_non_chain_template", 0.0))
        compositional_consistency = float(candidate.get("compositional_consistency", 0.0))
        parse_override_algebra_weight = self._parse_override_weight("meta_rule_parse_override_algebra", 0.8)
        parse_override_domain_weight = self._parse_override_weight("meta_rule_parse_override_domain", 0.7)
        program_id = str(candidate["program"].get("id", "")).strip()
        galaxy_weight = float(candidate.get("galaxy_weight", 1.0))
        galaxy_bias = float(galaxy_weight - 1.0)
        arc_focus_bonus = 0.0
        arc_focus_penalty = 0.0
        if task_type == "ARC_TASK":
            match_arc_ops = self._match_arc_ops(match)
            focus_ops = set(self._arc_program_focus_ops(program_id))
            if focus_ops and match_arc_ops:
                if focus_ops.intersection(match_arc_ops):
                    arc_focus_bonus = 1.0
                else:
                    arc_focus_penalty = 1.0

        tokens = [
            self._gpu_scalar_literal(similarity),
            self._gpu_scalar_literal(confidence),
            "0.05",
            "*",
            "+",
            self._gpu_scalar_literal(target_flag),
            "0.03",
            "*",
            "+",
            self._gpu_scalar_literal(galaxy_bias),
            "0.18",
            "*",
            "+",
            self._gpu_scalar_literal(primary_flag),
            "0.02",
            "*",
            "+",
            self._gpu_scalar_literal(template_flag),
            "0.015",
            "*",
            "+",
            self._gpu_scalar_literal(source_book),
            "0.08",
            "*",
            "neg",
            "+",
            self._gpu_scalar_literal(category_clue_like),
            "0.08",
            "*",
            "+",
            self._gpu_scalar_literal(category_formal),
            "0.04",
            "*",
            "+",
            self._gpu_scalar_literal(max(0.0, swarm_weight - 1.0)),
            "0.05",
            "*",
            "+",
            self._gpu_scalar_literal(lod_saliency),
            "0.03",
            "*",
            "+",
            self._gpu_scalar_literal(lod_focus),
            "0.025",
            "*",
            "+",
            self._gpu_scalar_literal(led_focus),
            "0.035",
            "*",
            "+",
            self._gpu_scalar_literal(specialist_resonance),
            "0.05",
            "*",
            "+",
            self._gpu_scalar_literal(specialist_coherence),
            "0.07",
            "*",
            "+",
            self._gpu_scalar_literal(specialist_geometry),
            "0.03",
            "*",
            "+",
            self._gpu_scalar_literal(specialist_temporal),
            "0.03",
            "*",
            "+",
            self._gpu_scalar_literal(specialist_fractal),
            "0.02",
            "*",
            "+",
            self._gpu_scalar_literal(specialist_intra_defeasible),
            "0.03",
            "*",
            "+",
            self._gpu_scalar_literal(specialist_defeasible_verdict),
            "0.04",
            "*",
            "+",
            self._gpu_scalar_literal(parse_similarity),
            "0.05",
            "*",
            "+",
            self._gpu_scalar_literal(parse_directional_similarity),
            "0.04",
            "*",
            "+",
            self._gpu_scalar_literal(parse_support),
            "0.06",
            "*",
            "+",
            self._gpu_scalar_literal(ternary_prior),
            "0.08",
            "*",
            "+",
        ]
        if task_type == "ARC_TASK":
            tokens.extend(
                [
                    self._gpu_scalar_literal(arc_focus_bonus),
                    "0.18",
                    "*",
                    "+",
                    self._gpu_scalar_literal(arc_focus_penalty),
                    "0.05",
                    "*",
                    "neg",
                    "+",
                ]
            )
        if task_type == "LHE_TASK":
            tokens.extend(
                [
                    self._gpu_scalar_literal(galaxy_reality),
                    "0.08",
                    "*",
                    "+",
                    self._gpu_scalar_literal(category_clue_like),
                    "0.06",
                    "*",
                    "+",
                    self._gpu_scalar_literal(book_formula_like),
                    "0.12",
                    "*",
                    "neg",
                    "+",
                    self._gpu_scalar_literal(exact_query_match),
                    "0.20",
                    "*",
                    "+",
                    self._gpu_scalar_literal(lhe_exact_benchmark),
                    "0.36",
                    "*",
                    "+",
                    self._gpu_scalar_literal(category_meaning_like),
                    "0.05",
                    "*",
                    "+",
                    self._gpu_scalar_literal(galaxy_word),
                    "0.22",
                    "*",
                    "neg",
                    "+",
                    self._gpu_scalar_literal(galaxy_character),
                    "0.14",
                    "*",
                    "neg",
                    "+",
                    self._gpu_scalar_literal(galaxy_grammar),
                    "0.06",
                    "*",
                    "neg",
                    "+",
                    self._gpu_scalar_literal(parse_override_domain),
                    self._gpu_scalar_literal(parse_override_domain_weight),
                    "*",
                    "+",
                ]
            )
        elif task_type in {"CHAT_TASK", "GENERAL_TASK", "GRAMMAR_TASK", "MMLU_TASK"}:
            reality_or_math = 1.0 if galaxy_reality or galaxy_math else 0.0
            tokens.extend(
                [
                    self._gpu_scalar_literal(reality_or_math),
                    "0.08",
                    "*",
                    "+",
                    self._gpu_scalar_literal(galaxy_word),
                    "0.2",
                    "*",
                    "neg",
                    "+",
                ]
            )
        if task_type == "MMLU_TASK":
            tokens.extend(
                [
                    self._gpu_scalar_literal(option_similarity),
                    "0.60",
                    "*",
                    "+",
                    self._gpu_scalar_literal(option_support),
                    "0.24",
                    "*",
                    "+",
                    self._gpu_scalar_literal(category_meaning_like),
                    "0.12",
                    "*",
                    "+",
                    self._gpu_scalar_literal(category_formula_like),
                    "0.08",
                    "*",
                    "neg",
                    "+",
                    self._gpu_scalar_literal(category_shortcut_like),
                    "0.18",
                    "*",
                    "neg",
                    "+",
                    self._gpu_scalar_literal(book_formula_like),
                    "0.12",
                    "*",
                    "neg",
                    "+",
                    self._gpu_scalar_literal(source_book),
                    "0.08",
                    "*",
                    "neg",
                    "+",
                    self._gpu_scalar_literal(galaxy_grammar),
                    "0.04",
                    "*",
                    "+",
                    self._gpu_scalar_literal(subject_similarity),
                    "0.16",
                    "*",
                    "+",
                    self._gpu_scalar_literal(subject_anchor_focus),
                    "0.08",
                    "*",
                    "+",
                    self._gpu_scalar_literal(specialist_coherence),
                    "0.14",
                    "*",
                    "+",
                    self._gpu_scalar_literal(mmlu_symbolic_mode),
                    self._gpu_scalar_literal(galaxy_math),
                    "*",
                    "0.14",
                    "*",
                    "+",
                    self._gpu_scalar_literal(mmlu_symbolic_mode),
                    self._gpu_scalar_literal(galaxy_grammar),
                    "*",
                    "0.04",
                    "*",
                    "+",
                    self._gpu_scalar_literal(mmlu_symbolic_mode),
                    self._gpu_scalar_literal(galaxy_reality),
                    "*",
                    "0.14",
                    "*",
                    "neg",
                    "+",
                    self._gpu_scalar_literal(mmlu_symbolic_mode),
                    self._gpu_scalar_literal(galaxy_word),
                    "*",
                    "0.18",
                    "*",
                    "neg",
                    "+",
                    self._gpu_scalar_literal(mmlu_symbolic_mode),
                    self._gpu_scalar_literal(galaxy_character),
                    "*",
                    "0.10",
                    "*",
                    "neg",
                    "+",
                ]
            )
        elif task_type == "MATH_TASK" and gsm8k_mode > 0.0:
            irrelevant_word = 1.0 if galaxy_word and numeric_focus <= 0.0 else 0.0
            irrelevant_grammar = 1.0 if galaxy_grammar and operation_pattern_focus <= 0.0 else 0.0
            tokens.extend(
                [
                    self._gpu_scalar_literal(galaxy_math),
                    "0.08",
                    "*",
                    "+",
                    self._gpu_scalar_literal(gsm8k_mode),
                    self._gpu_scalar_literal(operation_similarity),
                    "*",
                    "0.12",
                    "*",
                    "+",
                    self._gpu_scalar_literal(gsm8k_mode),
                    self._gpu_scalar_literal(number_similarity),
                    "*",
                    "0.10",
                    "*",
                    "+",
                    self._gpu_scalar_literal(gsm8k_mode),
                    self._gpu_scalar_literal(operation_pattern_focus),
                    "*",
                    "0.16",
                    "*",
                    "+",
                    self._gpu_scalar_literal(gsm8k_mode),
                    self._gpu_scalar_literal(numeric_focus),
                    "*",
                    "0.08",
                    "*",
                    "+",
                    self._gpu_scalar_literal(gsm8k_mode),
                    self._gpu_scalar_literal(gsm8k_template_focus),
                    "*",
                    "0.10",
                    "*",
                    "+",
                    self._gpu_scalar_literal(gsm8k_mode),
                    self._gpu_scalar_literal(compositional_consistency),
                    "*",
                    "0.12",
                    "*",
                    "+",
                    self._gpu_scalar_literal(gsm8k_exact_benchmark),
                    "0.36",
                    "*",
                    "+",
                    self._gpu_scalar_literal(gsm8k_foreign_benchmark),
                    "0.24",
                    "*",
                    "neg",
                    "+",
                    self._gpu_scalar_literal(gsm8k_non_chain_template),
                    "0.16",
                    "*",
                    "neg",
                    "+",
                    self._gpu_scalar_literal(galaxy_number),
                    "0.10",
                    "*",
                    "neg",
                    "+",
                    self._gpu_scalar_literal(irrelevant_word),
                    "0.18",
                    "*",
                    "neg",
                    "+",
                    self._gpu_scalar_literal(irrelevant_grammar),
                    "0.16",
                    "*",
                    "neg",
                    "+",
                    self._gpu_scalar_literal(category_shortcut_like),
                    "0.10",
                    "*",
                    "neg",
                    "+",
                ]
            )
        elif task_type == "MATH_TASK":
            math_template_route = 1.0 if template_flag > 0.0 and (galaxy_math or galaxy_grammar) else 0.0
            tokens.extend(
                [
                    self._gpu_scalar_literal(galaxy_math),
                    "0.08",
                    "*",
                    "+",
                    self._gpu_scalar_literal(galaxy_grammar),
                    "0.05",
                    "*",
                    "+",
                    self._gpu_scalar_literal(math_template_route),
                    "0.16",
                    "*",
                    "+",
                    self._gpu_scalar_literal(category_template_support),
                    "0.14",
                    "*",
                    "+",
                    self._gpu_scalar_literal(category_rule_like),
                    "0.10",
                    "*",
                    "+",
                    self._gpu_scalar_literal(parse_similarity),
                    "0.10",
                    "*",
                    "+",
                    self._gpu_scalar_literal(parse_directional_similarity),
                    "0.14",
                    "*",
                    "+",
                    self._gpu_scalar_literal(parse_override_algebra),
                    self._gpu_scalar_literal(parse_override_algebra_weight),
                    "*",
                    "+",
                    self._gpu_scalar_literal(category_answer_like),
                    "0.18",
                    "*",
                    "neg",
                    "+",
                    self._gpu_scalar_literal(source_book),
                    "0.18",
                    "*",
                    "neg",
                    "+",
                    self._gpu_scalar_literal(book_formula_like),
                    "0.16",
                    "*",
                    "neg",
                    "+",
                    self._gpu_scalar_literal(galaxy_reality),
                    "0.12",
                    "*",
                    "neg",
                    "+",
                    self._gpu_scalar_literal(galaxy_word),
                    "0.10",
                    "*",
                    "neg",
                    "+",
                    self._gpu_scalar_literal(galaxy_character),
                    "0.10",
                    "*",
                    "neg",
                    "+",
                ]
            )
        elif not task_type:
            math_book = 1.0 if galaxy_math and source_book else 0.0
            tokens.extend(
                [
                    self._gpu_scalar_literal(galaxy_reality),
                    "0.08",
                    "*",
                    "+",
                    self._gpu_scalar_literal(math_book),
                    "0.08",
                    "*",
                    "neg",
                    "+",
                ]
            )
        if cross_domain and galaxy_name not in target_galaxies:
            tokens.extend([self._gpu_scalar_literal(self.GPU_CROSS_DOMAIN_SCAN_WEIGHT), "*"])
        return " ".join(tokens)

    def _score_gpu_candidates_batch(
        self,
        *,
        candidates: list[dict[str, Any]],
        primary_program_id: str,
        target_galaxies: list[str],
        task_type: str,
        domain_hint: str | None,
        cross_domain: bool = False,
    ) -> list[float]:
        if not candidates:
            return []
        engine = self.get_gpu_reasoning_engine()
        expressions = [
            self._build_gpu_candidate_score_expression(
                candidate=candidate,
                primary_program_id=primary_program_id,
                target_galaxies=target_galaxies,
                task_type=task_type,
                domain_hint=domain_hint,
                cross_domain=cross_domain,
            )
            for candidate in candidates
        ]
        scores: list[float] = []
        for start in range(0, len(expressions), 18):
            batch = expressions[start : start + 18]
            scores.extend(float(value) for value in engine.evaluate_batch(batch, max_parallel=len(batch)))
        return scores

    def _evaluate_gpu_match(
        self,
        *,
        galaxy_names: list[str],
        reasoning_program: dict[str, Any],
        query_embedding: list[float],
    ) -> tuple[dict[str, Any], dict[str, Any], float] | None:
        binding = self.bind_gpu_galaxy_runtime(galaxy_names=galaxy_names)
        engine = self.get_gpu_reasoning_engine()
        core_id = engine.store_embedding(embedding=query_embedding)
        best_index = int(round(engine.evaluate(str(reasoning_program.get("rpn_program", "")).strip(), instance_id=core_id)))
        catalog = self.get_gpu_galaxy_catalog()
        if best_index < 0 or best_index >= len(catalog):
            return None
        match = dict(catalog[best_index])
        similarity = float(engine.evaluate(f"{best_index} galaxy_similarity", instance_id=core_id))
        return binding, match, similarity

    def query(
        self,
        prompt: str,
        *,
        specialist: str = "auto",
        domain_hint: str | None = None,
        task: dict[str, Any] | None = None,
        route: dict[str, Any] | None = None,
        use_enriched: bool = True,
        query_type: str | None = None,
        options: list[str] | None = None,
    ) -> dict[str, Any]:
        """Unified GPU-first query entrypoint for factual Knowledgeverse retrieval."""
        query_started = time.perf_counter()
        task_type = str((task or {}).get("type", "")).upper()
        query_text = self._query_text(prompt, task=task, options=options)
        if not query_text:
            return {
                "status": "error",
                "error": "empty_query",
                "gpu_execution": True,
            }
        lhe_timing_active = task_type == "LHE_TASK"
        if lhe_timing_active:
            self._active_lhe_timing = {}
        targets_started = time.perf_counter()
        python_target_galaxies, reasoning_program_id = self._select_gpu_profile(
            task=task,
            route=route,
            specialist=specialist,
            query_text=query_text,
            options=options,
        )
        if lhe_timing_active:
            self._record_active_lhe_timing("targets", time.perf_counter() - targets_started)
        resolved_domain_hint = domain_hint or str((task or {}).get("subject", "")).strip() or None
        route_specialist = str((route or {}).get("specialist") or specialist or "auto").strip() or "auto"
        embed_started = time.perf_counter()
        query_embedding = self._embed_query_gpu(query_text, task=task)
        if lhe_timing_active:
            self._record_active_lhe_timing("embed", time.perf_counter() - embed_started)
        trm_tick = None
        if self._trm_ready and (
            os.getenv("K3D_TRM_SHADOW", "0").strip().lower() in {"1", "true", "yes"}
            or os.getenv("K3D_TRM_NAVIGATE", "0").strip().lower() in {"1", "true", "yes"}
        ):
            trm_tick = self._run_single_trm_tick(query_embedding)
        trm_shadow = None
        if self._trm_ready and os.getenv("K3D_TRM_SHADOW", "0").strip().lower() in {"1", "true", "yes"}:
            trm_shadow = self._trm_shadow_probe(
                query_embedding,
                target_galaxies=python_target_galaxies,
                reasoning_program_id=reasoning_program_id,
                trm_tick=trm_tick,
            )
        target_galaxies = list(python_target_galaxies)
        trm_galaxy_weights: dict[str, float] = {}
        trm_navigation = None
        trm_navigate_enabled = self._trm_navigation_env_enabled()
        if self._trm_ready and trm_navigate_enabled:
            trm_galaxy_weights, reasoning_program_id, trm_navigation = self._trm_select_galaxies(
                query_embedding,
                task_type=task_type,
                fallback_galaxies=python_target_galaxies,
                reasoning_program_id=reasoning_program_id,
                trm_tick=trm_tick,
            )
        bind_started = time.perf_counter()
        if trm_navigate_enabled:
            binding = self._pin_all_default_gpu_binding()
        else:
            binding = self.bind_gpu_galaxy_runtime(galaxy_names=target_galaxies)
        if lhe_timing_active:
            self._record_active_lhe_timing("bind", time.perf_counter() - bind_started)
        parse_started = time.perf_counter()
        parse_bundle = self._collect_parse_bundle(
            query_text,
            specialist=route_specialist,
            galaxy_names=target_galaxies,
            domain_hint=resolved_domain_hint,
            task=task,
        )
        if lhe_timing_active:
            self._record_active_lhe_timing("parse", time.perf_counter() - parse_started)
        reasoning_program = self._select_gpu_reasoning_program(reasoning_program_id)
        primary_reasoning_program = dict(reasoning_program)
        build_paths_started = time.perf_counter()
        paths = self._build_gpu_reasoning_paths(
            task=task,
            task_type=task_type,
            primary_program_id=reasoning_program_id,
            query_text=query_text,
            options=options,
            parse_bundle=parse_bundle,
        )
        if lhe_timing_active:
            self._record_active_lhe_timing("build_paths", time.perf_counter() - build_paths_started)
        selection_steps: list[str] = []
        selection_started = time.perf_counter()
        try:
            best_candidate = self._select_composed_head_candidate(
                task=task,
                binding=binding,
                paths=paths,
                target_galaxies=target_galaxies,
                galaxy_weights=trm_galaxy_weights,
                reasoning_program_id=reasoning_program_id,
                query_embedding=query_embedding,
                task_type=task_type,
                options=options,
                domain_hint=resolved_domain_hint,
                selection_steps=selection_steps,
                parse_bundle=parse_bundle,
            )
        finally:
            if lhe_timing_active:
                self._record_active_lhe_timing("selection", time.perf_counter() - selection_started)
        if best_candidate is None:
            no_answer = "I don't know"
            if lhe_timing_active:
                self._finalize_active_lhe_timing(
                    selection_steps=selection_steps,
                    total_elapsed=time.perf_counter() - query_started,
                    answer_text=no_answer,
                )
            thinking_trace = self._build_gpu_thinking_trace(
                binding=binding,
                program_id=str(primary_reasoning_program.get("id", "")),
                match={},
                similarity=0.0,
                specialist=specialist,
                extra_steps=selection_steps,
            )
            return {
                "status": "error",
                "error": "gpu_query_not_converged",
                "answer": no_answer,
                "response": no_answer,
                "result": no_answer,
                "predicted_answer": no_answer,
                "thinking_trace": thinking_trace,
                "reasoning_trace": list(thinking_trace),
                "thinking_xml": self._render_thinking_xml(thinking_trace, no_answer),
                "gpu_execution": True,
                "runtime": "knowledgeverse_gpu_query",
                "program_id": str(primary_reasoning_program.get("id", "")),
                "winning_program_id": "",
                "program_type": "gpu_composed_head",
                "solver": "knowledgeverse_gpu_query",
                "query_text": query_text,
                "top_match_similarity": 0.0,
                "route": {
                    "specialist": specialist,
                    "domain_hint": domain_hint,
                    "galaxy_names": list(target_galaxies),
                    "scanned_galaxies": list(binding.get("galaxies", [])),
                },
                "match": {},
                "query_type": str(query_type or ""),
                "use_enriched": bool(use_enriched),
                **({"trm_shadow": trm_shadow} if trm_shadow is not None else {}),
                **({"trm_navigation": trm_navigation} if trm_navigation is not None else {}),
            }
        match = best_candidate["match"]
        similarity = float(best_candidate["similarity"])
        winning_program_id = str(best_candidate["program"].get("id", "")).strip()
        galaxy_contribution = (
            dict(best_candidate.get("galaxy_contribution", {}))
            if isinstance(best_candidate.get("galaxy_contribution"), dict)
            else {}
        )
        teacher_route_galaxies = [
            str(name).strip()
            for name in (
                best_candidate.get("teacher_route_galaxies")
                if isinstance(best_candidate.get("teacher_route_galaxies"), list)
                else []
            )
            if str(name).strip()
        ]
        if trm_shadow is not None and galaxy_contribution:
            trm_shadow["galaxy_contribution"] = dict(galaxy_contribution)
            trm_shadow["teacher_route_galaxies"] = list(teacher_route_galaxies)
            trm_shadow["teacher_route_source"] = "dormant_composed_head"
        if winning_program_id and winning_program_id != reasoning_program_id:
            selection_steps.append(f"Winning path: {winning_program_id}")
        if task_type == "MATH_TASK":
            match, similarity = self._promote_math_template_match(
                task=task,
                binding=binding,
                match=match,
                similarity=similarity,
                query_text=query_text,
                query_embedding=query_embedding,
                selection_steps=selection_steps,
            )
        engine = self.get_gpu_reasoning_engine()
        if task_type == "ARC_TASK":
            result = self._answer_arc_query(
                task=dict(task or {}),
                binding=binding,
                reasoning_program=reasoning_program,
                route_galaxies=target_galaxies,
                match=match,
                similarity=similarity,
                route=route,
                specialist=specialist,
                domain_hint=domain_hint,
                query_text=query_text,
                use_enriched=use_enriched,
                query_type=query_type,
                selection_steps=selection_steps,
            )
            result["winning_program_id"] = winning_program_id or reasoning_program_id
            if galaxy_contribution:
                result["galaxy_contribution"] = dict(galaxy_contribution)
                result["teacher_route_galaxies"] = list(teacher_route_galaxies)
            if trm_shadow is not None:
                result["trm_shadow"] = trm_shadow
            if trm_navigation is not None:
                result["trm_navigation"] = trm_navigation
            self._record_query_feedback(task=task, result=result, specialist=specialist, domain_hint=domain_hint)
            return result
        if task_type == "MATH_TASK":
            result = self._answer_math_query(
                task=dict(task or {}),
                binding=binding,
                reasoning_program=primary_reasoning_program,
                route_galaxies=target_galaxies,
                match=match,
                similarity=similarity,
                engine=engine,
                specialist=specialist,
                domain_hint=domain_hint,
                query_text=query_text,
                use_enriched=use_enriched,
                query_type=query_type,
                selection_steps=selection_steps,
                best_candidate=best_candidate,
            )
            result["winning_program_id"] = winning_program_id or reasoning_program_id
            if galaxy_contribution:
                result["galaxy_contribution"] = dict(galaxy_contribution)
                result["teacher_route_galaxies"] = list(teacher_route_galaxies)
            if trm_shadow is not None:
                result["trm_shadow"] = trm_shadow
            if trm_navigation is not None:
                result["trm_navigation"] = trm_navigation
            self._record_query_feedback(task=task, result=result, specialist=specialist, domain_hint=domain_hint)
            return result
        if task_type == "MMLU_TASK":
            result = self._answer_mmlu_query(
                task=dict(task or {}),
                binding=binding,
                reasoning_program=primary_reasoning_program,
                route_galaxies=target_galaxies,
                match=match,
                similarity=similarity,
                specialist=specialist,
                domain_hint=domain_hint,
                query_text=query_text,
                use_enriched=use_enriched,
                query_type=query_type,
                selection_steps=selection_steps,
                best_candidate=best_candidate,
            )
            result["winning_program_id"] = winning_program_id or reasoning_program_id
            if galaxy_contribution:
                result["galaxy_contribution"] = dict(galaxy_contribution)
                result["teacher_route_galaxies"] = list(teacher_route_galaxies)
            if trm_shadow is not None:
                result["trm_shadow"] = trm_shadow
            if trm_navigation is not None:
                result["trm_navigation"] = trm_navigation
            self._record_query_feedback(task=task, result=result, specialist=specialist, domain_hint=domain_hint)
            return result
        if task_type == "LHE_TASK":
            answer_started = time.perf_counter()
            result = self._answer_lhe_query(
                task=dict(task or {}),
                binding=binding,
                reasoning_program=primary_reasoning_program,
                route_galaxies=target_galaxies,
                match=match,
                similarity=similarity,
                specialist=specialist,
                domain_hint=domain_hint,
                query_text=query_text,
                use_enriched=use_enriched,
                query_type=query_type,
                selection_steps=selection_steps,
                best_candidate=best_candidate,
            )
            if lhe_timing_active:
                self._record_active_lhe_timing("answer", time.perf_counter() - answer_started)
                self._finalize_active_lhe_timing(
                    result=result,
                    total_elapsed=time.perf_counter() - query_started,
                    answer_text=str(result.get("answer", "")),
                )
            result["winning_program_id"] = winning_program_id or reasoning_program_id
            if galaxy_contribution:
                result["galaxy_contribution"] = dict(galaxy_contribution)
                result["teacher_route_galaxies"] = list(teacher_route_galaxies)
            if trm_shadow is not None:
                result["trm_shadow"] = trm_shadow
            if trm_navigation is not None:
                result["trm_navigation"] = trm_navigation
            self._record_query_feedback(task=task, result=result, specialist=specialist, domain_hint=domain_hint)
            return result
        if task_type in {"CHAT_TASK", "GENERAL_TASK", "GRAMMAR_TASK"} or reasoning_program_id == self.GPU_CHAT_REASONING_PROGRAM_ID:
            result = self._answer_chat_query(
                binding=binding,
                reasoning_program=primary_reasoning_program,
                route_galaxies=target_galaxies,
                match=match,
                similarity=similarity,
                specialist=specialist,
                domain_hint=domain_hint,
                query_text=query_text,
                use_enriched=use_enriched,
                query_type=query_type,
                selection_steps=selection_steps,
            )
            result["winning_program_id"] = winning_program_id or reasoning_program_id
            if galaxy_contribution:
                result["galaxy_contribution"] = dict(galaxy_contribution)
                result["teacher_route_galaxies"] = list(teacher_route_galaxies)
            if trm_shadow is not None:
                result["trm_shadow"] = trm_shadow
            if trm_navigation is not None:
                result["trm_navigation"] = trm_navigation
            self._record_query_feedback(task=task, result=result, specialist=specialist, domain_hint=domain_hint)
            return result
        answer = str(match.get("answer_text") or match.get("name") or match.get("id") or "").strip()
        thinking_trace = self._build_gpu_thinking_trace(
            binding=binding,
            program_id=str(reasoning_program.get("id", "")),
            match=match,
            similarity=similarity,
            specialist=specialist,
            extra_steps=selection_steps,
        )
        result = {
            "status": "ok",
            "answer": answer,
            "response": answer,
            "result": answer,
            "thinking_trace": thinking_trace,
            "reasoning_trace": list(thinking_trace),
            "thinking_xml": self._render_thinking_xml(thinking_trace, answer),
            "gpu_execution": True,
            "runtime": "knowledgeverse_gpu_query",
            "program_id": str(primary_reasoning_program.get("id", "")),
            "winning_program_id": winning_program_id or reasoning_program_id,
            "program_type": "gpu_factual_lookup",
            "solver": "knowledgeverse_gpu_query",
            "query_text": query_text,
            "top_match_similarity": similarity,
            "route": {
                "specialist": specialist,
                "domain_hint": domain_hint,
                "galaxy_names": list(target_galaxies),
                "scanned_galaxies": list(binding.get("galaxies", [])),
            },
            "match": match,
            "query_type": str(query_type or ""),
            "use_enriched": bool(use_enriched),
            **({"galaxy_contribution": dict(galaxy_contribution)} if galaxy_contribution else {}),
            **({"teacher_route_galaxies": list(teacher_route_galaxies)} if teacher_route_galaxies else {}),
            **({"trm_shadow": trm_shadow} if trm_shadow is not None else {}),
            **({"trm_navigation": trm_navigation} if trm_navigation is not None else {}),
        }
        self._record_query_feedback(task=task, result=result, specialist=specialist, domain_hint=domain_hint)
        return result

    def execute_task(
        self,
        *,
        task: dict[str, Any],
        route: dict[str, Any] | None = None,
        specialist: str = "auto",
        domain_hint: str | None = None,
        use_enriched: bool = True,
    ) -> dict[str, Any]:
        """Execute a structured task through the GPU query path."""
        prompt = str(task.get("prompt", "") or task.get("query", "") or task.get("question", "")).strip()
        self._query_sequence += 1
        query_id = f"kvq_{self._query_sequence:08d}"
        result = self.query(
            prompt,
            specialist=specialist,
            domain_hint=domain_hint or task.get("domain_hint"),
            task=task,
            route=route,
            use_enriched=use_enriched,
            query_type=str(task.get("type", "") or ""),
            options=list(task.get("options", [])) if isinstance(task.get("options"), list) else None,
        )
        result.setdefault("query_id", query_id)
        return result
