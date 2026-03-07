"""Persistent K3D daemon entrypoint (game-style runtime).

The daemon keeps one Knowledgeverse + TRM instance alive and serves JSON
commands over stdio or TCP line protocol. This avoids one-shot script
orchestration and enforces a single-world process lifecycle.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socketserver
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.specialists.math_specialist import MathSpecialist
from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge
from knowledge3d.cranium.bridges.procedural_geometry_bridge import ProceduralGeometryBridge
from knowledge3d.cranium.bridges.procedural_material_bridge import ProceduralMaterialBridge
from benchmarks.arc_agi_2_adapter import ArcAgi2Adapter

try:
    from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
except Exception:  # pragma: no cover
    ModularRPNEngine = None  # type: ignore[assignment]

try:
    from knowledge3d.cranium.sovereign.loader import get_vram_usage
except Exception:  # pragma: no cover
    get_vram_usage = None  # type: ignore[assignment]

try:
    from knowledge3d.gpu.perf_counters import gpu_utilisation
except Exception:  # pragma: no cover
    gpu_utilisation = None  # type: ignore[assignment]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _append_env_path(var_name: str, path_value: str) -> None:
    current = os.environ.get(var_name, "").strip()
    if not current:
        os.environ[var_name] = path_value
        return
    items = [item for item in current.split(":") if item]
    if path_value in items:
        return
    os.environ[var_name] = f"{current}:{path_value}"


def _configure_cuda_include_paths() -> dict[str, Any]:
    """
    Ensure NVRTC can resolve core CUDA headers (e.g., cuda_fp16.h).

    This is a daemon-level safeguard for sovereign PTX query/runtime paths:
    we do not enable fallbacks; we only make required CUDA include paths explicit.
    """
    include_candidates: list[Path] = [
        Path("/usr/local/cuda/include"),
        Path("/usr/include"),
    ]
    include_candidates.extend(sorted(Path("/usr/local").glob("cuda*/include")))

    selected: Path | None = None
    for inc in include_candidates:
        if not inc.exists():
            continue
        if (inc / "cuda_fp16.h").exists():
            selected = inc
            break

    configured = {"applied": False, "include_path": None, "cuda_path": None}
    if selected is None:
        return configured

    include_str = str(selected)
    _append_env_path("CPATH", include_str)
    _append_env_path("CPLUS_INCLUDE_PATH", include_str)

    # Derive CUDA_PATH from include parent when possible.
    cuda_root = selected.parent if selected.name == "include" else selected
    if cuda_root.exists() and not os.environ.get("CUDA_PATH"):
        os.environ["CUDA_PATH"] = str(cuda_root)

    configured["applied"] = True
    configured["include_path"] = include_str
    configured["cuda_path"] = os.environ.get("CUDA_PATH")
    return configured


@dataclass
class DaemonConfig:
    storage_root: Path
    require_ptx_query: bool = True
    eager_load_default_galaxies: bool = True
    host: str = "127.0.0.1"
    port: int = 7777


class K3DDaemon:
    """Long-lived command server for K3D runtime orchestration."""

    _LHE_STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "use",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }

    def __init__(
        self,
        config: DaemonConfig,
        *,
        knowledgeverse: Knowledgeverse | None = None,
        math_specialist: MathSpecialist | None = None,
    ):
        self.config = config
        self.started_at = _now_iso()
        self._shutdown_requested = False
        self._command_count = 0
        self._gpu_calls_total = 0
        self._cuda_env = _configure_cuda_include_paths()
        self._repo_root = Path(__file__).resolve().parents[2]
        self._boot_status_paths = [
            config.storage_root / "runtime" / "runtime_boot.json",
            self._repo_root / "viewer" / "public" / "runtime_boot.json",
        ]
        self._drawing_bridge: ProceduralDrawingBridge | None = None
        self._geometry_bridge: ProceduralGeometryBridge | None = None
        self._material_bridge: ProceduralMaterialBridge | None = None
        self._arc_adapter_cache: dict[bool, ArcAgi2Adapter] = {}
        self._drawing_warmup: dict[str, Any] = {}
        self._geometry_warmup: dict[str, Any] = {}
        self._material_warmup: dict[str, Any] = {}
        self._write_boot_status(stage="daemon_boot", progress=0.05, state="starting")

        os.environ["K3D_REQUIRE_PTX_QUERY"] = "true" if config.require_ptx_query else "false"

        self._write_boot_status(stage="knowledgeverse_load", progress=0.2, state="loading")
        self.kv = knowledgeverse or Knowledgeverse(
            storage_root=config.storage_root,
            eager_load_default_galaxies=config.eager_load_default_galaxies,
        )
        self.trm = self.kv.trm_navigator
        self.math_specialist = math_specialist or MathSpecialist(knowledgeverse=self.kv, parent=self.trm)
        self._default_counts = self.kv.ensure_default_galaxies_loaded()
        self._write_boot_status(
            stage="knowledgeverse_ready",
            progress=0.55,
            state="loading",
            extra={"default_galaxy_counts": dict(self._default_counts)},
        )
        self._warmup_boot_runtime()
        self._write_boot_status(
            stage="ready",
            progress=1.0,
            state="ready",
            extra={
                "drawing_warmup": dict(self._drawing_warmup),
                "geometry_warmup": dict(self._geometry_warmup),
                "material_warmup": dict(self._material_warmup),
            },
        )

    def _write_boot_status(
        self,
        *,
        stage: str,
        progress: float,
        state: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "status": "ok",
            "state": state,
            "stage": stage,
            "progress": max(0.0, min(1.0, float(progress))),
            "timestamp": _now_iso(),
            "pid": int(os.getpid()),
        }
        if extra:
            payload.update(extra)
        for path in self._boot_status_paths:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
            except Exception:
                continue

    def _warmup_boot_runtime(self) -> None:
        if os.environ.get("K3D_WARMUP_DRAWING", "1") != "1":
            self._drawing_warmup = {"status": "skipped", "reason": "K3D_WARMUP_DRAWING=0"}
        else:
            self._write_boot_status(stage="drawing_runtime_warmup", progress=0.72, state="warming")
            try:
                self._drawing_bridge = ProceduralDrawingBridge(matryoshka_dim=64)
                self._drawing_warmup = self._drawing_bridge.warmup_runtime()
                self._write_boot_status(
                    stage="drawing_runtime_warm",
                    progress=0.84,
                    state="warming",
                    extra={"drawing_warmup": dict(self._drawing_warmup)},
                )
            except Exception as exc:
                self._drawing_warmup = {
                    "status": "error",
                    "exception_type": type(exc).__name__,
                    "detail": str(exc),
                }
                self._write_boot_status(
                    stage="drawing_runtime_warmup_failed",
                    progress=0.84,
                    state="warning",
                    extra={"drawing_warmup": dict(self._drawing_warmup)},
                )

        if os.environ.get("K3D_WARMUP_GEOMETRY", "1") != "1":
            self._geometry_warmup = {"status": "skipped", "reason": "K3D_WARMUP_GEOMETRY=0"}
        else:
            self._write_boot_status(
                stage="geometry_runtime_warmup",
                progress=0.9,
                state="warming",
                extra={"drawing_warmup": dict(self._drawing_warmup)},
            )
            try:
                self._geometry_bridge = ProceduralGeometryBridge()
                self._geometry_warmup = self._geometry_bridge.warmup_runtime()
                self._write_boot_status(
                    stage="geometry_runtime_warm",
                    progress=0.96,
                    state="warming",
                    extra={
                        "drawing_warmup": dict(self._drawing_warmup),
                        "geometry_warmup": dict(self._geometry_warmup),
                    },
                )
            except Exception as exc:
                self._geometry_warmup = {
                    "status": "error",
                    "exception_type": type(exc).__name__,
                    "detail": str(exc),
                }
                self._write_boot_status(
                    stage="geometry_runtime_warmup_failed",
                    progress=0.96,
                    state="warning",
                    extra={
                        "drawing_warmup": dict(self._drawing_warmup),
                        "geometry_warmup": dict(self._geometry_warmup),
                    },
                )

        if os.environ.get("K3D_WARMUP_MATERIAL", "1") != "1":
            self._material_warmup = {"status": "skipped", "reason": "K3D_WARMUP_MATERIAL=0"}
            return

        self._write_boot_status(
            stage="material_runtime_warmup",
            progress=0.985,
            state="warming",
            extra={
                "drawing_warmup": dict(self._drawing_warmup),
                "geometry_warmup": dict(self._geometry_warmup),
            },
        )
        try:
            self._material_bridge = ProceduralMaterialBridge()
            self._material_warmup = self._material_bridge.warmup_runtime()
            self._write_boot_status(
                stage="material_runtime_warm",
                progress=0.995,
                state="warming",
                extra={
                    "drawing_warmup": dict(self._drawing_warmup),
                    "geometry_warmup": dict(self._geometry_warmup),
                    "material_warmup": dict(self._material_warmup),
                },
            )
        except Exception as exc:
            self._material_warmup = {
                "status": "error",
                "exception_type": type(exc).__name__,
                "detail": str(exc),
            }
            self._write_boot_status(
                stage="material_runtime_warmup_failed",
                progress=0.995,
                state="warning",
                extra={
                    "drawing_warmup": dict(self._drawing_warmup),
                    "geometry_warmup": dict(self._geometry_warmup),
                    "material_warmup": dict(self._material_warmup),
                },
            )

    def _gpu_snapshot(self) -> dict[str, Any]:
        used = 0
        total = 0
        util = 0.0
        if get_vram_usage is not None:
            try:
                used, total = get_vram_usage()
            except Exception:
                used, total = 0, 0
        if gpu_utilisation is not None:
            try:
                util = float(gpu_utilisation(default=0.0))
            except Exception:
                util = 0.0
        return {
            "vram_used_bytes": int(used),
            "vram_total_bytes": int(total),
            "gpu_utilization": float(util),
        }

    @property
    def should_shutdown(self) -> bool:
        return self._shutdown_requested

    def status_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "timestamp": _now_iso(),
            "daemon_started_at": self.started_at,
            "pid": int(os.getpid()),
            "require_ptx_query": bool(self.config.require_ptx_query),
            "manifest_version": str(self.kv.manifest_version),
            "default_galaxy_counts": dict(self._default_counts),
            "command_count": int(self._command_count),
            "gpu_calls_total": int(self._gpu_calls_total),
            "cuda_env": dict(self._cuda_env),
            "drawing_warmup": dict(self._drawing_warmup),
            "geometry_warmup": dict(self._geometry_warmup),
            "material_warmup": dict(self._material_warmup),
            "boot_status_paths": [str(path) for path in self._boot_status_paths],
        }

    def _gpu_call_snapshot(self) -> int:
        if ModularRPNEngine is None:
            return 0
        try:
            return int(ModularRPNEngine.get_global_gpu_call_count())
        except Exception:
            return 0

    def _collect_parse_bundle(
        self,
        query: str,
        *,
        specialist: str,
        galaxy_names: list[str],
        domain_hint: str | None = None,
    ) -> dict[str, Any]:
        navigator = getattr(self.kv, "navigator_specialist", None)
        if navigator is None:
            navigator = getattr(self.trm, "navigator_specialist", None)
        if navigator is None:
            return {}
        try:
            routes = navigator.plan_routes(
                query=query,
                specialist=specialist,
                galaxy_names=galaxy_names,
                domain_hint=domain_hint,
                use_forward_backward=True,
            )
        except Exception:
            return {}
        bundle: dict[str, Any] = {"route_plan": routes}
        for key in ("forward_parse", "backward_parse", "fusion_parse"):
            for route in routes:
                if not isinstance(route, dict):
                    continue
                value = route.get(key)
                if isinstance(value, dict):
                        bundle[key] = value
                        break
        return bundle

    def _collect_math_parse_bundle(self, question: str) -> dict[str, Any]:
        return self._collect_parse_bundle(
            question,
            specialist="math",
            galaxy_names=["Math", "Grammar", "Tool"],
            domain_hint="math",
        )

    def _normalize_lhe_domain_hint(self, domain_hint: str | None) -> str:
        domain = str(domain_hint or "multi").strip().lower()
        aliases = {
            "mathematics": "math",
            "math": "math",
            "physics": "physics",
            "chemistry": "physics",
            "biology": "physics",
            "philosophy": "grammar",
            "trivia": "grammar",
            "cybersecurity": "grammar",
            "history": "grammar",
            "chess": "grammar",
            "multi": "multi",
        }
        return aliases.get(domain, domain or "multi")

    def _split_lhe_clauses(self, text: str) -> list[str]:
        clauses = [part.strip() for part in re.split(r"(?:[.?!]\s+|;\s+|\n+)", str(text or "").strip()) if part.strip()]
        return clauses or ([str(text).strip()] if str(text).strip() else [])

    def _tokenize_lhe_text(self, text: str, *, preserve_single: bool = False) -> list[str]:
        raw_tokens = re.findall(r"[A-Za-z0-9_+#$=:/.-]+", str(text or ""))
        tokens: list[str] = []
        for raw in raw_tokens:
            token = raw.strip().strip(".,:;!?()[]{}").lower()
            if not token:
                continue
            if not preserve_single and len(token) == 1 and not token.isdigit():
                continue
            if token in self._LHE_STOPWORDS:
                continue
            tokens.append(token)
        return tokens

    def _normalize_answer_text(self, value: str) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-z0-9+#=,./:;() \\-]+", "", text)
        return text.strip()

    def _append_lhe_text_entities(
        self,
        entities: list[dict[str, Any]],
        *,
        text: str,
        role: str,
        source_pass: str,
        confidence: float,
        preserve_single: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        raw = str(text or "").strip()
        if not raw:
            return
        payload = dict(extra or {})
        entities.append(
            {
                "kind": "phrase",
                "value": raw.lower(),
                "raw": raw,
                "role": role,
                "source_pass": source_pass,
                "confidence": confidence,
                **payload,
            }
        )
        for token_index, token in enumerate(self._tokenize_lhe_text(raw, preserve_single=preserve_single)[:16]):
            entities.append(
                {
                    "kind": "token" if role != "option" else "option_token",
                    "value": token,
                    "raw": raw,
                    "role": role,
                    "source_pass": source_pass,
                    "token_index": token_index,
                    "confidence": min(confidence + 0.1, 0.99),
                    **payload,
                }
            )

    def _build_lhe_parse_entities(
        self,
        *,
        parse_bundle: dict[str, Any],
        options: list[str] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        forward_parse = parse_bundle.get("forward_parse", {}) if isinstance(parse_bundle.get("forward_parse"), dict) else {}
        backward_parse = parse_bundle.get("backward_parse", {}) if isinstance(parse_bundle.get("backward_parse"), dict) else {}
        fusion_parse = parse_bundle.get("fusion_parse", {}) if isinstance(parse_bundle.get("fusion_parse"), dict) else {}

        forward_entities: list[dict[str, Any]] = []
        backward_entities: list[dict[str, Any]] = []
        fused_entities: list[dict[str, Any]] = []

        for idx, block in enumerate(forward_parse.get("context", []) or []):
            if not isinstance(block, dict):
                continue
            raw = str(block.get("raw", "")).strip()
            self._append_lhe_text_entities(
                forward_entities,
                text=raw,
                role="context",
                source_pass="forward",
                confidence=0.55,
                extra={"clause_index": idx},
            )
            data = block.get("data", {})
            if block.get("type") == "variables" and isinstance(data, dict):
                for key, value in data.items():
                    forward_entities.append(
                        {
                            "kind": "variable",
                            "value": str(key),
                            "raw": f"{key}={value}",
                            "role": "context",
                            "source_pass": "forward",
                            "confidence": 0.72,
                            "clause_index": idx,
                            "data_value": value,
                        }
                    )
        if isinstance(forward_parse.get("goal"), dict):
            goal = forward_parse["goal"]
            self._append_lhe_text_entities(
                forward_entities,
                text=str(goal.get("raw") or goal.get("expression") or ""),
                role="goal",
                source_pass="forward",
                confidence=0.7,
            )

        for idx, block in enumerate(backward_parse.get("dependencies", []) or []):
            if not isinstance(block, dict):
                continue
            raw = str(block.get("raw", "")).strip()
            self._append_lhe_text_entities(
                backward_entities,
                text=raw,
                role="context",
                source_pass="backward",
                confidence=0.55,
                extra={"clause_index": idx},
            )
            data = block.get("data", {})
            if block.get("type") == "variables" and isinstance(data, dict):
                for key, value in data.items():
                    backward_entities.append(
                        {
                            "kind": "variable",
                            "value": str(key),
                            "raw": f"{key}={value}",
                            "role": "context",
                            "source_pass": "backward",
                            "confidence": 0.72,
                            "clause_index": idx,
                            "data_value": value,
                        }
                    )
        if isinstance(backward_parse.get("goal"), dict):
            goal = backward_parse["goal"]
            self._append_lhe_text_entities(
                backward_entities,
                text=str(goal.get("raw") or goal.get("expression") or ""),
                role="goal",
                source_pass="backward",
                confidence=0.74,
            )

        merged_variables = fusion_parse.get("merged_variables", {})
        if isinstance(merged_variables, dict):
            for key, value in merged_variables.items():
                fused_entities.append(
                    {
                        "kind": "variable",
                        "value": str(key),
                        "raw": f"{key}={value}",
                        "role": "context",
                        "source_pass": "fusion",
                        "confidence": 0.9,
                        "data_value": value,
                    }
                )
        unified_goal = fusion_parse.get("unified_goal", {})
        if isinstance(unified_goal, dict):
            self._append_lhe_text_entities(
                fused_entities,
                text=str(unified_goal.get("raw") or unified_goal.get("expression") or ""),
                role="goal",
                source_pass="fusion",
                confidence=0.9,
            )

        for option_index, option in enumerate(options or []):
            self._append_lhe_text_entities(
                forward_entities,
                text=str(option),
                role="option",
                source_pass="forward",
                confidence=0.65,
                preserve_single=True,
                extra={"option_index": option_index},
            )
            self._append_lhe_text_entities(
                backward_entities,
                text=str(option),
                role="option",
                source_pass="backward",
                confidence=0.65,
                preserve_single=True,
                extra={"option_index": option_index},
            )
            self._append_lhe_text_entities(
                fused_entities,
                text=str(option),
                role="option",
                source_pass="fusion",
                confidence=0.68,
                preserve_single=True,
                extra={"option_index": option_index},
            )

        return {
            "forward_entities": forward_entities,
            "backward_entities": backward_entities,
            "fused_entities": fused_entities,
        }

    def _build_lhe_goal(
        self,
        *,
        prompt: str,
        options: list[str],
        parse_bundle: dict[str, Any],
        domain_hint: str,
    ) -> dict[str, Any]:
        backward = parse_bundle.get("backward_parse", {})
        goal = backward.get("goal") if isinstance(backward, dict) else None
        raw_goal = ""
        if isinstance(goal, dict):
            raw_goal = str(goal.get("expression") or goal.get("raw") or "")
        if not raw_goal:
            raw_goal = str(prompt)
        return {
            "kind": "multiple_choice" if options else "open_ended",
            "domain": domain_hint,
            "raw": raw_goal,
            "tokens": self._tokenize_lhe_text(raw_goal, preserve_single=bool(options)),
            "requires_short_answer": not options,
        }

    def _augment_lhe_route(self, route: dict[str, Any], *, domain_hint: str) -> dict[str, Any]:
        effective = dict(route)
        names = [str(name) for name in route.get("galaxy_names") or [] if str(name).strip()]
        for required in ("Reality", "Grammar", "Tool"):
            if required not in names:
                names.append(required)
        if domain_hint == "math" and "Math" not in names:
            names.append("Math")
        effective["galaxy_names"] = names
        return effective

    def _extract_lhe_evidence_fields(self, row: dict[str, Any]) -> dict[str, str]:
        entry = row.get("entry") if isinstance(row.get("entry"), dict) else {}
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        fields = {
            "content": str(entry.get("content", "")).strip(),
            "description": str(entry.get("description", "")).strip(),
            "rpn_program": str(entry.get("rpn_program", "")).strip(),
            "pattern_form": str(entry.get("pattern_form", "")).strip(),
            "pattern_type": str(entry.get("pattern_type", "")).strip(),
            "semantics": str(metadata.get("semantics", "")).strip(),
            "usage_conditions": str(metadata.get("usage_conditions", "")).strip(),
            "domain": str(metadata.get("domain", "")).strip(),
            "category": str(metadata.get("category", "")).strip(),
        }
        return {key: value for key, value in fields.items() if value}

    def _extract_lhe_row_text(self, row: dict[str, Any]) -> str:
        entry = row.get("entry") if isinstance(row.get("entry"), dict) else {}
        parts: list[str] = [
            str(entry.get("name", "")),
            str(entry.get("title", "")),
        ]
        evidence_fields = self._extract_lhe_evidence_fields(row)
        parts.extend(evidence_fields.values())
        return " ".join(part.strip() for part in parts if str(part).strip())

    def _query_lhe_evidence(
        self,
        *,
        prompt: str,
        route: dict[str, Any],
        parse_bundle: dict[str, Any],
        use_enriched: bool,
        options: list[str],
    ) -> list[dict[str, Any]]:
        route_plan = parse_bundle.get("route_plan")
        plan_rows = route_plan if isinstance(route_plan, list) else []
        query_specs: list[tuple[str, dict[str, Any]]] = []
        for planned in plan_rows[:4]:
            if not isinstance(planned, dict):
                continue
            variant = str(planned.get("query_variant", "")).strip()
            if variant:
                query_specs.append((variant, planned))
        if not query_specs:
            query_specs.append((prompt, route))

        seen: set[tuple[str, str, str]] = set()
        evidence: list[dict[str, Any]] = []
        for query_text, planned in query_specs:
            rows = self.trm.query(
                query=query_text,
                galaxy_names=planned.get("galaxy_names") or route.get("galaxy_names"),
                top_k=20 if use_enriched else 8,
                specialist=str(planned.get("specialist", route.get("specialist", "auto"))),
                domain_hint=str(planned.get("domain", route.get("domain", "multi"))),
            )
            for index, row in enumerate(rows):
                row_text = self._extract_lhe_row_text(row)
                if not row_text:
                    continue
                entry = row.get("entry") if isinstance(row.get("entry"), dict) else {}
                dedupe_key = (
                    str(entry.get("id", "")),
                    str(entry.get("name", "")),
                    row_text[:160],
                )
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                evidence.append(
                    {
                        "row": row,
                        "text": row_text,
                        "tokens": set(self._tokenize_lhe_text(row_text, preserve_single=bool(options))),
                        "fields": self._extract_lhe_evidence_fields(row),
                        "query": query_text,
                        "rank_weight": max(0.05, 1.0 - (index * 0.08)),
                    }
                )
        return evidence

    def _query_lhe_option_evidence(
        self,
        *,
        prompt: str,
        option: str,
        route: dict[str, Any],
        use_enriched: bool,
    ) -> list[dict[str, Any]]:
        rows = self.trm.query(
            query=f"{prompt}\nCandidate answer: {option}",
            galaxy_names=route.get("galaxy_names"),
            top_k=10 if use_enriched else 4,
            specialist=str(route.get("specialist", "auto")),
            domain_hint=str(route.get("domain", "multi")),
        )
        out: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            text = self._extract_lhe_row_text(row)
            if not text:
                continue
            out.append(
                {
                    "row": row,
                    "text": text,
                    "tokens": set(self._tokenize_lhe_text(text, preserve_single=True)),
                    "fields": self._extract_lhe_evidence_fields(row),
                    "rank_weight": max(0.05, 1.0 - (index * 0.1)),
                }
            )
        return out

    def _score_lhe_option(
        self,
        *,
        prompt: str,
        options: list[str],
        option: str,
        goal: dict[str, Any],
        fused_entities: list[dict[str, Any]],
        evidence_rows: list[dict[str, Any]],
        option_rows: list[dict[str, Any]],
    ) -> float:
        prompt_lower = str(prompt).lower()
        normalized_option = str(option).strip()
        option_tokens = set(self._tokenize_lhe_text(normalized_option, preserve_single=True))
        goal_tokens = set(goal.get("tokens", []))
        fused_tokens = {
            str(entity.get("value", ""))
            for entity in fused_entities
            if str(entity.get("kind", "")) in {"token", "option_token"} and entity.get("value")
        }
        normalized_phrase = self._normalize_answer_text(normalized_option)
        other_phrases = {
            self._normalize_answer_text(candidate)
            for candidate in options
            if self._normalize_answer_text(candidate) and self._normalize_answer_text(candidate) != normalized_phrase
        }
        score = 0.0
        if normalized_option and re.search(rf"\b(?:pick|choose|select)\s+{re.escape(normalized_option.lower())}\b", prompt_lower):
            score += 10.0
        score += float(len(option_tokens & goal_tokens)) * 0.10
        score += float(len(option_tokens & fused_tokens)) * 0.05
        support = 0.0
        contradiction = 0.0
        for evidence in [*evidence_rows, *option_rows]:
            fields = evidence.get("fields", {}) if isinstance(evidence.get("fields"), dict) else {}
            normalized_fields = [self._normalize_answer_text(value) for value in fields.values()]
            exact_hits = sum(1 for value in normalized_fields if normalized_phrase and normalized_phrase in value)
            if exact_hits:
                support += exact_hits * (1.5 + float(evidence.get("rank_weight", 0.0)))
            for other in other_phrases:
                if other and any(other in value for value in normalized_fields):
                    contradiction += 0.9 + (0.2 * float(evidence.get("rank_weight", 0.0)))
        score += support
        score -= contradiction
        return score

    def _extract_lhe_open_candidates(self, text: str, *, field_name: str = "") -> list[str]:
        candidates: list[str] = []
        normalized = " ".join(str(text).split()).strip()
        if not normalized:
            return candidates
        field = str(field_name or "").strip().lower()
        patterns = [
            r"\$[^$]{1,160}\$",
            r"\\\([^)]{1,160}\\\)",
            r"\b-?\d+(?:\.\d+)?\b",
        ]
        if field not in {"rpn_program", "pattern_form"}:
            patterns.extend(
                [
                    r"[^.!?;\n]{6,180}",
                    r"\b[A-Z][a-z][A-Za-z0-9,'-]{1,40}(?: [A-Za-z][A-Za-z0-9,'-]{1,40}){0,10}\b",
                ]
            )
        for pattern in patterns:
            for match in re.findall(pattern, text):
                item = " ".join(str(match).split()).strip(" ,;:.")
                if item and item not in candidates:
                    candidates.append(item)
        return candidates

    def _is_lhe_meta_candidate(self, candidate: str) -> bool:
        lowered = str(candidate).strip().lower()
        if not lowered:
            return True
        generic = {
            "tool",
            "grammar",
            "reality",
            "math",
            "drawing",
            "audio",
            "3dobjects",
            "english svo",
            "syntax",
            "procedural systems",
            "procedural reality specialist",
            "basic math specialist",
        }
        return lowered in generic

    def _is_lhe_code_like_candidate(self, candidate: str) -> bool:
        text = str(candidate).strip()
        if not text:
            return False
        compact = text.replace(" ", "")
        if re.fullmatch(r"[A-Z0-9_#+-]{2,40}", compact):
            return True
        if text.count("_") >= 1:
            return True
        if re.fullmatch(r"[A-Za-z]+ [A-Z]{2,8}", text):
            return True
        return False

    def _canonicalize_lhe_short_numeric_candidate(self, candidate: str) -> str:
        normalized = self._normalize_answer_text(candidate)
        if not normalized:
            return ""
        if re.fullmatch(r"-?\d+(?:\.\d+)?", normalized):
            return normalized
        digit_match = re.search(r"\b-?\d+(?:\.\d+)?\b", normalized)
        if digit_match:
            return digit_match.group(0)
        word_to_num = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "eleven": "11",
            "twelve": "12",
            "thirteen": "13",
            "fourteen": "14",
            "fifteen": "15",
            "sixteen": "16",
            "seventeen": "17",
            "eighteen": "18",
            "nineteen": "19",
            "twenty": "20",
        }
        for token in self._tokenize_lhe_text(normalized, preserve_single=True):
            mapped = word_to_num.get(token)
            if mapped:
                return mapped
        return ""

    def _synthesize_lhe_open_answer(
        self,
        *,
        fused_entities: list[dict[str, Any]],
        goal: dict[str, Any],
        evidence_rows: list[dict[str, Any]],
    ) -> str:
        goal_tokens = set(goal.get("tokens", []))
        fused_tokens = {
            str(entity.get("value", ""))
            for entity in fused_entities
            if str(entity.get("kind", "")) in {"token", "phrase"} and entity.get("value")
        }
        goal_raw = str(goal.get("raw", "")).strip()
        goal_lower = goal_raw.lower()
        wants_formula = any(token in goal_raw for token in ("$", "\\(", "\\)", "^", "{", "}")) or any(
            token in goal_lower for token in ("equation", "expression", "formula", "polynomial")
        )
        wants_short_numeric = any(token in goal_lower for token in ("how many", "how much", "what is", "value", "number"))
        field_weights = {
            "content": 1.0,
            "description": 0.92,
            "semantics": 0.86,
            "usage_conditions": 0.78,
            "domain": 0.2,
            "category": 0.15,
            "rpn_program": 0.02,
            "pattern_form": 0.02,
        }
        best_candidate = ""
        best_score = float("-inf")
        for evidence in evidence_rows:
            fields = evidence.get("fields", {}) if isinstance(evidence.get("fields"), dict) else {}
            candidates: list[tuple[str, str, float]] = []
            for key in ("content", "description", "semantics", "usage_conditions", "domain", "category", "rpn_program", "pattern_form"):
                value = str(fields.get(key, "")).strip()
                if not value:
                    continue
                field_weight = float(field_weights.get(key, 0.1))
                for candidate in self._extract_lhe_open_candidates(value, field_name=key):
                    if candidate and all(existing != candidate for existing, _, _ in candidates):
                        candidates.append((candidate, key, field_weight))
            for candidate, field_name, field_weight in candidates:
                lowered_candidate = candidate.strip().lower()
                if self._is_lhe_meta_candidate(candidate):
                    continue
                if self._is_lhe_code_like_candidate(candidate) and not wants_formula:
                    continue
                numeric_variant = self._canonicalize_lhe_short_numeric_candidate(candidate) if wants_short_numeric else ""
                candidate_variants = [candidate.strip()]
                if numeric_variant and numeric_variant not in candidate_variants:
                    candidate_variants.append(numeric_variant)
                for candidate_variant in candidate_variants:
                    candidate_tokens = set(self._tokenize_lhe_text(candidate_variant, preserve_single=True))
                    normalized_candidate = self._normalize_answer_text(candidate_variant)
                    score = float(evidence.get("rank_weight", 0.0)) * 0.9
                    score += field_weight
                    score += float(len(candidate_tokens & goal_tokens)) * 0.7
                    score += float(len(candidate_tokens & fused_tokens)) * 0.25
                    if wants_short_numeric and re.fullmatch(r"-?\d+(?:\.\d+)?", normalized_candidate):
                        score += 1.8
                    if wants_short_numeric and candidate_variant == candidate.strip() and numeric_variant and candidate_variant != numeric_variant:
                        score -= 1.2
                    if not wants_formula and not wants_short_numeric and len(candidate_variant.split()) >= 5:
                        score += 0.2
                    if field_name in {"rpn_program", "pattern_form"}:
                        score -= 0.35
                    score -= min(len(candidate_variant), 120) * 0.002
                    if score > best_score:
                        best_score = score
                        best_candidate = candidate_variant.strip()
        return best_candidate

    def _solve_lhe_structured(
        self,
        *,
        prompt: str,
        options: list[str],
        route: dict[str, Any],
        parse_bundle: dict[str, Any],
        use_enriched: bool,
        domain_hint: str,
    ) -> dict[str, Any]:
        parse_entities = self._build_lhe_parse_entities(parse_bundle=parse_bundle, options=options)
        forward_entities = list(parse_entities.get("forward_entities", []))
        backward_entities = list(parse_entities.get("backward_entities", []))
        fused_entities = list(parse_entities.get("fused_entities", []))
        goal = self._build_lhe_goal(
            prompt=prompt,
            options=options,
            parse_bundle=parse_bundle,
            domain_hint=domain_hint,
        )
        evidence_rows = self._query_lhe_evidence(
            prompt=prompt,
            route=route,
            parse_bundle=parse_bundle,
            use_enriched=use_enriched,
            options=options,
        )
        reasoning_trace = [
            f"lhe_four_pass forward={len(forward_entities)} backward={len(backward_entities)} fused={len(fused_entities)} evidence={len(evidence_rows)}",
            f"lhe_goal {goal.get('kind')} domain={domain_hint}",
        ]
        if options:
            scored: list[tuple[float, str]] = []
            for option in options:
                option_rows = self._query_lhe_option_evidence(
                    prompt=prompt,
                    option=option,
                    route=route,
                    use_enriched=use_enriched,
                )
                score = self._score_lhe_option(
                    prompt=prompt,
                    options=options,
                    option=option,
                    goal=goal,
                    fused_entities=fused_entities,
                    evidence_rows=evidence_rows,
                    option_rows=option_rows,
                )
                scored.append((score, option))
            scored.sort(key=lambda item: item[0], reverse=True)
            predicted = str(scored[0][1]) if scored else ""
            reasoning_trace.append(
                "lhe_option_scores "
                + ", ".join(f"{opt}={score:.3f}" for score, opt in scored[:4])
            )
        else:
            predicted = self._synthesize_lhe_open_answer(
                fused_entities=fused_entities,
                goal=goal,
                evidence_rows=evidence_rows,
            )
            reasoning_trace.append(f"lhe_open_answer {'present' if predicted else 'empty'}")
        return {
            "predicted_answer": predicted,
            "reasoning_trace": reasoning_trace,
            "four_pass": {
                "forward_entities": forward_entities,
                "backward_entities": backward_entities,
                "fused_entities": fused_entities,
                "goal": goal,
                "evidence_count": len(evidence_rows),
                "composition_depth": 4 if fused_entities else 1,
            },
        }

    def _dispatch_lhe_task(self, *, route: dict[str, Any], task: dict[str, Any], use_enriched: bool) -> dict[str, Any]:
        prompt = str(task.get("prompt", "") or task.get("query", "")).strip()
        if not prompt:
            return {"status": "error", "error": "lhe_task_missing_prompt"}
        domain_hint = self._normalize_lhe_domain_hint(task.get("domain_hint"))
        options = task.get("options")
        option_list = [str(item) for item in options] if isinstance(options, list) else []
        effective_route = self._augment_lhe_route(route, domain_hint=domain_hint)

        parse_bundle = self._collect_parse_bundle(
            prompt,
            specialist=str(effective_route.get("specialist", "auto") or "auto"),
            galaxy_names=[str(name) for name in effective_route.get("galaxy_names") or ["Grammar", "Reality", "Tool"]],
            domain_hint=domain_hint,
        )

        if domain_hint == "math":
            enriched_task = {
                "question": prompt,
                "query": prompt,
                "options": option_list,
                "domain_hint": domain_hint,
                **parse_bundle,
            }
            solved = self.math_specialist.process(enriched_task, use_enriched=use_enriched)
            predicted = solved.get("result")
            return {
                "status": "ok" if solved.get("status") == "success" else "error",
                "task_type": "LHE_TASK",
                "task_id": task.get("task_id"),
                "response": predicted,
                "answer": predicted,
                "result": predicted,
                "reasoning_trace": list(solved.get("reasoning_trace", [])),
                "parse_bundle": parse_bundle,
                "route": route,
            }
        structured = self._solve_lhe_structured(
            prompt=prompt,
            options=option_list,
            route=effective_route,
            parse_bundle=parse_bundle,
            use_enriched=use_enriched,
            domain_hint=domain_hint,
        )
        response = structured.get("predicted_answer", "")
        return {
            "status": "ok",
            "task_type": "LHE_TASK",
            "task_id": task.get("task_id"),
            "response": response,
            "answer": response,
            "result": response,
            "reasoning_trace": list(structured.get("reasoning_trace", [])),
            "four_pass": structured.get("four_pass", {}),
            "parse_bundle": parse_bundle,
            "route": effective_route,
        }

    def _get_arc_adapter(self, *, use_enriched: bool) -> ArcAgi2Adapter:
        key = bool(use_enriched)
        adapter = self._arc_adapter_cache.get(key)
        if adapter is None:
            adapter = ArcAgi2Adapter(
                use_enriched=key,
                strict_legacy=False,
                knowledgeverse=self.kv,
                enable_contrastive_learning=True,
                enable_validity_gates=True,
                enable_fuzzy_oracle=True,
                enable_figure_ground_reversal=True,
                enable_object_aware_generation=True,
                enable_ptx_ranking=False,
                enable_full_ptx=False,
            )
            self._arc_adapter_cache[key] = adapter
        return adapter

    def _dispatch_task(self, *, route: dict[str, Any], task: dict[str, Any], use_enriched: bool) -> dict[str, Any]:
        specialist = str(route.get("specialist", "grammar")).lower()
        task_type = str(task.get("type", "")).upper()

        if task_type == "LHE_TASK":
            return self._dispatch_lhe_task(route=route, task=task, use_enriched=use_enriched)

        if specialist == "visual":
            if task_type != "ARC_TASK":
                return {"status": "not_implemented", "reason": "visual_specialist_expected_arc_task"}
            training_examples = task.get("training_examples")
            input_grid = task.get("input_grid")
            if not isinstance(training_examples, list) or input_grid is None:
                return {"status": "error", "error": "arc_task_missing_training_or_input"}
            benchmark_task = {
                "id": str(task.get("task_id") or "arc_task"),
                "train": list(training_examples),
                "test": [
                    {
                        "input": input_grid,
                        "output": task.get("expected_output"),
                    }
                ],
            }
            solved = self._get_arc_adapter(use_enriched=use_enriched).solve_task(benchmark_task)
            output_grid = solved.get("predicted")
            response = {
                "status": "ok" if solved.get("predicted") is not None else "error",
                "task_type": "ARC_TASK",
                "task_id": task.get("task_id"),
                "program_type": "arc_benchmark_adapter",
                "output_grid": output_grid,
                "reasoning_trace": list(solved.get("reasoning_trace", [])),
                "solver": solved.get("solver"),
                "patterns_used": int(solved.get("patterns_used", 0)),
                "generated_pattern_count": int(solved.get("generated_pattern_count", 0)),
                "score": float(solved.get("score", 0.0)),
                "fuzzy_score": float(solved.get("fuzzy_score", 0.0)),
                "exact_match": bool(solved.get("exact_match", False)),
            }
            passthrough_keys = (
                "ranking_top_components",
                "pattern_source",
                "selected_source",
                "selected_rank",
                "selected_oracle_track",
                "selected_exact_match",
                "selected_fuzzy_score",
                "correct_rank",
                "oracle_at_3",
                "oracle_at_10",
                "oracle_at_all",
                "fuzzy_oracle_at_all",
                "oracle_fuzzy_0_80",
                "oracle_fuzzy_0_85",
                "oracle_fuzzy_0_90",
                "oracle_fuzzy_0_95",
                "oracle_failure_modes",
                "queried_galaxies",
                "ptx_ranking_used",
                "ptx_full_used",
                "ptx_oracle_used",
                "generation_filter_generated_total",
                "generation_filter_accept_rate",
                "generation_filter_reject_rate",
                "generation_object_count_distribution",
                "generation_object_count_distribution_accepted",
                "generation_object_count_distribution_rejected",
            )
            for key in passthrough_keys:
                if key in solved:
                    response[key] = solved[key]
            return response

        if specialist == "math":
            question = str(task.get("question", "") or task.get("query", "")).strip()
            if not question:
                return {"status": "error", "error": "math_task_missing_question"}
            enriched_task = dict(task)
            enriched_task.update(self._collect_math_parse_bundle(question))
            solved = self.math_specialist.process(enriched_task, use_enriched=use_enriched)
            return {
                "status": "ok" if solved.get("status") == "success" else "error",
                "task_type": task_type or "MATH_TASK",
                "task_id": task.get("task_id"),
                **solved,
            }

        if specialist in {"chat", "grammar", "any"}:
            messages = task.get("messages")
            if not isinstance(messages, list):
                prompt = str(task.get("prompt", "") or task.get("query", "")).strip()
                if not prompt:
                    return {"status": "error", "error": "chat_task_missing_prompt"}
                messages = [{"role": "user", "content": prompt}]
            response = self.trm.process_chat(messages, use_enriched=use_enriched)
            return {
                "status": "ok",
                "task_type": task_type or "CHAT_TASK",
                "task_id": task.get("task_id"),
                "response": response,
            }

        return {
            "status": "not_implemented",
            "reason": f"specialist_dispatch_not_implemented:{specialist}",
            "task_type": task_type,
        }

    def handle_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._command_count += 1
        cmd = str(payload.get("command", "")).strip().upper()
        if not cmd:
            return {"status": "error", "error": "missing_command"}

        if cmd in {"PING", "STATUS"}:
            return self.status_payload()

        if cmd == "SHUTDOWN":
            self._shutdown_requested = True
            return {"status": "ok", "message": "shutdown_requested", "timestamp": _now_iso()}

        if cmd == "ROUTE":
            task = payload.get("task")
            if task is not None and not isinstance(task, dict):
                return {"status": "error", "error": "task_must_be_object"}
            task_obj = task if isinstance(task, dict) else None
            query = str(
                payload.get("query", "")
                or (task_obj or {}).get("query", "")
                or (task_obj or {}).get("question", "")
                or (task_obj or {}).get("prompt", "")
                or (task_obj or {}).get("type", "")
            ).strip()
            if not query:
                return {"status": "error", "error": "missing_query_or_task"}
            use_enriched = bool(payload.get("use_enriched", True))
            route = self.trm.route(
                query=query,
                specialist=str(payload.get("specialist", "auto")),
                domain_hint=payload.get("domain_hint") or (task_obj or {}).get("domain_hint"),
                galaxy_names=payload.get("galaxies") or (task_obj or {}).get("galaxies"),
            )
            response: dict[str, Any] = {"status": "ok", "route": route}
            if task_obj is not None:
                response["task_result"] = self._dispatch_task(
                    route=route,
                    task=task_obj,
                    use_enriched=use_enriched,
                )
            return response

        if cmd == "QUERY":
            query = str(payload.get("query", "")).strip()
            if not query:
                return {"status": "error", "error": "missing_query"}
            top_k = int(payload.get("top_k", 10))
            rows = self.trm.query(
                query=query,
                galaxy_names=payload.get("galaxies"),
                top_k=max(1, top_k),
                specialist=str(payload.get("specialist", "auto")),
                domain_hint=payload.get("domain_hint"),
            )
            return {
                "status": "ok",
                "count": len(rows),
                "results": rows,
            }

        if cmd == "SOLVE_MATH":
            question = str(payload.get("question", "") or payload.get("query", "")).strip()
            if not question:
                return {"status": "error", "error": "missing_question"}
            use_enriched = bool(payload.get("use_enriched", True))
            math_task = {"question": question}
            math_task.update(self._collect_math_parse_bundle(question))
            solved = self.math_specialist.process(math_task, use_enriched=use_enriched)
            if solved.get("status") != "success":
                return {
                    "status": "error",
                    "error": "math_specialist_failed",
                    "detail": solved,
                }
            return {
                "status": "ok",
                "result": solved.get("result"),
                "rpn_program": solved.get("rpn_program"),
                "coefficients": solved.get("coefficients"),
                "pattern_id": solved.get("pattern_id"),
                "template_id": solved.get("template_id"),
            }

        if cmd == "CHAT":
            messages = payload.get("messages")
            if not isinstance(messages, list):
                prompt = str(payload.get("prompt", "")).strip()
                if not prompt:
                    return {"status": "error", "error": "missing_messages_or_prompt"}
                messages = [{"role": "user", "content": prompt}]
            response = self.trm.process_chat(messages, use_enriched=bool(payload.get("use_enriched", True)))
            return {"status": "ok", "response": response}

        return {"status": "error", "error": "unknown_command", "command": cmd}

    def _handle_line(self, raw_line: str) -> str:
        cmd_started = time.perf_counter()
        gpu_before = self._gpu_snapshot()
        gpu_calls_before = self._gpu_call_snapshot()
        line = raw_line.strip()
        if not line:
            response = {"status": "error", "error": "empty_command"}
            return json.dumps(response, separators=(",", ":"))
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            response = {
                "status": "error",
                "error": "invalid_json",
                "detail": str(exc),
            }
            return json.dumps(
                {"status": "error", "error": "invalid_json", "detail": str(exc)},
                separators=(",", ":"),
            )
        if not isinstance(payload, dict):
            return json.dumps({"status": "error", "error": "command_must_be_object"}, separators=(",", ":"))
        try:
            result = self.handle_command(payload)
        except Exception as exc:
            result = {
                "status": "error",
                "error": "command_execution_failed",
                "exception_type": type(exc).__name__,
                "detail": str(exc),
            }
        gpu_after = self._gpu_snapshot()
        gpu_calls_after = self._gpu_call_snapshot()
        gpu_calls_this_command = max(0, int(gpu_calls_after - gpu_calls_before))
        self._gpu_calls_total += gpu_calls_this_command
        elapsed_ms = (time.perf_counter() - cmd_started) * 1000.0
        result["telemetry"] = {
            "elapsed_ms": float(elapsed_ms),
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "daemon_command_count": int(self._command_count),
            "gpu_call_counter_before": int(gpu_calls_before),
            "gpu_call_counter_after": int(gpu_calls_after),
            "gpu_calls_this_command": int(gpu_calls_this_command),
            "gpu_calls_total": int(self._gpu_calls_total),
            "fallback_triggered": False,
        }
        return json.dumps(result, separators=(",", ":"), sort_keys=True)

    def serve_stdio(self) -> int:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "message": "k3d_daemon_started",
                    "mode": "stdio",
                    "timestamp": _now_iso(),
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            flush=True,
        )
        for line in sys.stdin:
            response = self._handle_line(line)
            print(response, flush=True)
            if self._shutdown_requested:
                break
        return 0

    def serve_tcp(self) -> int:
        daemon = self

        class ReusableTCPServer(socketserver.TCPServer):
            allow_reuse_address = True

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:  # type: ignore[override]
                raw = self.rfile.readline().decode("utf-8", errors="replace")
                if not raw:
                    return
                out = daemon._handle_line(raw) + "\n"
                self.wfile.write(out.encode("utf-8"))

        with ReusableTCPServer((self.config.host, self.config.port), Handler) as server:
            server.timeout = 0.2
            while not self._shutdown_requested:
                server.handle_request()
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run persistent K3D daemon command loop.")
    parser.add_argument("--storage-root", default="../Knowledge3D.local", help="Knowledgeverse storage root.")
    parser.add_argument(
        "--mode",
        choices=("stdio", "tcp"),
        default="stdio",
        help="Command transport mode.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="TCP host when --mode=tcp.")
    parser.add_argument("--port", type=int, default=7777, help="TCP port when --mode=tcp.")
    parser.add_argument(
        "--allow-nonsovereign-query",
        action="store_true",
        help="Allow CPU query path for diagnostics (default is strict PTX query required).",
    )
    parser.add_argument(
        "--no-eager-load-default-galaxies",
        action="store_true",
        help="Disable eager default galaxy load at startup.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = DaemonConfig(
        storage_root=Path(args.storage_root),
        require_ptx_query=not bool(args.allow_nonsovereign_query),
        eager_load_default_galaxies=not bool(args.no_eager_load_default_galaxies),
        host=str(args.host),
        port=int(args.port),
    )
    daemon = K3DDaemon(config=config)
    if args.mode == "tcp":
        return daemon.serve_tcp()
    return daemon.serve_stdio()


if __name__ == "__main__":
    raise SystemExit(main())
