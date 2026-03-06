"""Knowledgeverse TRM navigator with resilient composition helpers."""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .execution_grammar_detector import ExecutionGrammarDetector
from .galaxy_manager import GalaxyManager
from .execution_quality_tracker import ExecutionQualityTracker
from .navigator_specialist import NavigatorSpecialist
from .resilience import SelfHealingWrapper
from .specialist_base import SpecialistBase
from .specialist_router import SpecialistRouter
from .specialist_spawner import SpecialistSpawner
from .tool_execution import ToolExecutionResolver


class TRMNavigator(SpecialistBase):
    """Deterministic navigator surface used by benchmark/integration flows."""

    def __init__(self, knowledgeverse: Any | None = None, galaxy_manager: GalaxyManager | None = None):
        storage_dir: Path | None = None
        if knowledgeverse is not None and hasattr(knowledgeverse, "storage_root"):
            storage_dir = Path(getattr(knowledgeverse, "storage_root")) / "checkpoints"
        super().__init__(
            name="NavigatorSpecialist",
            domain="navigator",
            level=0,
            parent=None,
            storage_dir=storage_dir,
        )
        self.knowledgeverse = knowledgeverse
        self.galaxy_manager = galaxy_manager or getattr(knowledgeverse, "galaxy_manager", None) or GalaxyManager()
        self.specialist_router = SpecialistRouter()
        self.navigator_specialist = NavigatorSpecialist(
            knowledgeverse=knowledgeverse,
            router=self.specialist_router,
        )
        self._specialist_tree_path = (
            (storage_dir / "trm_specialist_tree.json")
            if storage_dir is not None
            else Path("../Knowledge3D.local/checkpoints/trm_specialist_tree.json")
        )
        self._specialist_tree_path.parent.mkdir(parents=True, exist_ok=True)
        self._bootstrap_matryoshka_specialists()
        self._load_specialist_tree()
        self.specialist_spawner = SpecialistSpawner(
            root=self,
            storage_path=self._specialist_tree_path.parent / "trm_specialist_spawner.json",
            frequency_threshold=100,
            low_confidence_threshold=0.6,
            low_confidence_min_samples=20,
            max_children_per_parent=16,
        )
        self._trace: list[str] = []
        self._math_debug = str(os.getenv("K3D_MATH_DEBUG", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._last_math_missing_signal: dict[str, Any] | None = None
        self._last_math_execution_error: str | None = None
        self.execution_quality_tracker: ExecutionQualityTracker | None = None
        self.execution_grammar_detector: ExecutionGrammarDetector | None = None
        if knowledgeverse is not None and hasattr(knowledgeverse, "storage_root"):
            root = Path(getattr(knowledgeverse, "storage_root"))
            self.execution_quality_tracker = ExecutionQualityTracker(
                state_path=root / "checkpoints" / "execution_quality_tracker.json",
                gap_log_path=root / "logs" / "specialist_gaps.jsonl",
            )
            self.execution_grammar_detector = ExecutionGrammarDetector(
                storage_root=root,
                galaxy_manager=self.galaxy_manager,
            )

    def _bootstrap_matryoshka_specialists(self) -> None:
        """
        Bootstrap default master/worker hierarchy.

        Layout:
        - InputPrimerSpecialist (input normalization pre-routing)
        - ChatSpecialist (conversational I/O compatibility layer)
        - MathSpecialist -> BasicMathSpecialist, PhDMathSpecialist
        - VisualSpecialist -> ArcVisualSpecialist, SpatialVisualSpecialist
        - PhysicsSpecialist -> MechanicsSpecialist, ProceduralRealitySpecialist
        - GrammarSpecialist -> SyntaxSpecialist, SemanticsSpecialist
        """
        self.children.clear()
        self.routing_bias.clear()

        # Input primer + chat specialist for standard LLM I/O compatibility.
        self.spawn_child(name="InputPrimerSpecialist", domain="input_normalization")
        chat_master = self.spawn_child(name="ChatSpecialist", domain="conversational")

        math_master = self.spawn_child(name="MathSpecialist", domain="math")
        visual_master = self.spawn_child(name="VisualSpecialist", domain="visual")
        physics_master = self.spawn_child(name="PhysicsSpecialist", domain="physics")
        audio_master = self.spawn_child(name="AudioSpecialist", domain="audio")
        grammar_master = self.spawn_child(name="GrammarSpecialist", domain="language")

        math_master.spawn_child(name="BasicMathSpecialist", domain="basic_math")
        math_master.spawn_child(name="PhDMathSpecialist", domain="phd_math")

        visual_master.spawn_child(name="ArcVisualSpecialist", domain="arc_visual")
        visual_master.spawn_child(name="SpatialVisualSpecialist", domain="spatial_reasoning")

        physics_master.spawn_child(name="MechanicsSpecialist", domain="mechanics")
        physics_master.spawn_child(name="ProceduralRealitySpecialist", domain="procedural_systems")
        audio_master.spawn_child(name="SignalAudioSpecialist", domain="signal_audio")
        audio_master.spawn_child(name="SpectralAudioSpecialist", domain="spectral_audio")

        grammar_master.spawn_child(name="SyntaxSpecialist", domain="syntax")
        grammar_master.spawn_child(name="SemanticsSpecialist", domain="semantics")

    def _load_specialist_tree(self) -> None:
        if not self._specialist_tree_path.exists():
            return
        try:
            payload = json.loads(self._specialist_tree_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        try:
            loaded = SpecialistBase.from_dict(payload, storage_dir=self.storage_dir)
        except Exception:
            return
        self.children = loaded.children
        for child in self.children.values():
            child.parent = self
        self.routing_bias = dict(loaded.routing_bias)
        self.query_count = int(loaded.query_count)
        self.success_count = int(loaded.success_count)
        self.failure_count = int(loaded.failure_count)

    def _save_specialist_tree(self) -> None:
        self._specialist_tree_path.parent.mkdir(parents=True, exist_ok=True)
        self._specialist_tree_path.write_text(
            json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )

    def list_specialists(self) -> list[str]:
        return [node.name for node in self.iter_tree()]

    def count_specialists(self, *, include_root: bool = False) -> int:
        count = len(self.iter_tree())
        return count if include_root else max(0, count - 1)

    def find_specialist(self, name: str) -> SpecialistBase | None:
        return self.find(name)

    def _master_for_specialist(self, specialist: str) -> SpecialistBase:
        mapping = {
            "math": "MathSpecialist",
            "visual": "VisualSpecialist",
            "physics": "PhysicsSpecialist",
            "audio": "AudioSpecialist",
            "grammar": "GrammarSpecialist",
            "logic": "GrammarSpecialist",
            "language": "GrammarSpecialist",
            "cartographer": "VisualSpecialist",
        }
        target = mapping.get(str(specialist).lower(), "GrammarSpecialist")
        return self.find_specialist(target) or self

    def _resolve_specialist_node(
        self,
        *,
        specialist: str,
        query: str,
        domain_hint: str | None = None,
    ) -> SpecialistBase:
        master = self._master_for_specialist(specialist)
        if master is self:
            return self
        routed = master.route(query=query, domain_hint=domain_hint)
        if routed is master:
            return master
        return routed

    @SelfHealingWrapper.circuit_breaker(failure_threshold=5, timeout=60.0)
    def navigate_and_compose(
        self,
        query: str,
        specialist: str = "auto",
        domain_hint: str | None = None,
        use_enriched: bool = True,
        use_forward_backward: bool = True,
    ) -> dict[str, Any]:
        if specialist == "auto":
            composed = self.navigator_specialist.navigate_and_compose(
                trm_navigator=self,
                query=query,
                use_enriched=use_enriched,
                use_forward_backward=use_forward_backward,
                specialist=specialist,
                domain_hint=domain_hint,
            )
            return composed

        route = self.route(query=query, specialist=specialist, domain_hint=domain_hint)
        results = self.query(
            query=query,
            galaxy_names=route["galaxy_names"],
            top_k=20,
            specialist=route["specialist"],
            domain_hint=route["domain"],
        )
        composed = self.compose(
            query=query,
            patterns=results,
            specialist=route["specialist"],
            use_enriched=use_enriched,
        )
        if isinstance(composed, dict):
            composed["route"] = route
        return composed

    def generate_from_procedural(
        self,
        query: str,
        source_galaxy: str,
        target_galaxy: str | None = None,
        store_result: bool = True,
    ) -> dict[str, Any]:
        """
        Generate new knowledge from procedural primitives and optionally persist it.

        The method intentionally stays in Knowledgeverse orchestration space
        (ingestion/reasoning path), while preserving sovereign hot-path constraints.
        """
        source_name = str(source_galaxy).strip()
        if not source_name:
            return {"error": "source_galaxy is required", "query": query}

        specialist = self._specialist_for_galaxy(source_name)
        target_name = str(target_galaxy or self._default_generation_target(source_name))
        candidates = self.query(
            query=f"{query} procedural generation",
            galaxy_names=[source_name],
            top_k=20,
            specialist=specialist,
            domain_hint=source_name.lower(),
        )
        if not candidates:
            return {"error": "No procedural primitives found", "query": query}

        composed = self._compose_procedural_program(query=query, candidates=candidates)
        rpn_program = str(composed.get("rpn_program", "")).strip()
        if not rpn_program:
            return {"error": "Failed to compose procedural program", "query": query}

        now = datetime.datetime.now(datetime.timezone.utc)
        generated_id = f"generated_{source_name.lower()}_{now.strftime('%Y%m%d_%H%M%S_%f')}"
        tool_context = composed.get("tool_context")
        execution_plan = composed.get("execution_plan")
        promotion_targets = list(composed.get("promotion_targets", []))
        if tool_context:
            self._record_tool_promotion_pressure(
                query=query,
                source_galaxy=source_name,
                target_galaxy=target_name,
                specialist=specialist,
                tool_context=tool_context,
                promotion_targets=promotion_targets,
            )
        generated_entry = {
            "id": generated_id,
            "name": f"{query} (Generated)",
            "domain": target_name.lower(),
            "category": "generated_pattern",
            "rpn_program": rpn_program,
            "metadata": {
                "generated": True,
                "source_primitives": composed.get("source_primitives", []),
                "source_galaxy": source_name,
                "composition_depth": int(composed.get("depth", 1)),
                "lineage": (
                    f"{source_name}.{composed.get('category', 'unknown')} -> "
                    f"{target_name}.generated_pattern"
                ),
                "confidence": float(composed.get("confidence", 0.7)),
                "tool_context": tool_context,
                "execution_plan": execution_plan,
                "promotion_targets": promotion_targets,
                "timestamp": now.isoformat(),
                "query": query,
            },
        }

        if store_result and self.knowledgeverse is not None:
            self.knowledgeverse.galaxy_manager.add_entry(target_name, generated_entry)
            self.knowledgeverse.log_event(
                event_type="autonomous_generation",
                event_data={
                    "query": query,
                    "source_galaxy": source_name,
                    "target_galaxy": target_name,
                    "generated_id": generated_id,
                    "composition_depth": int(composed.get("depth", 1)),
                    "confidence": float(composed.get("confidence", 0.7)),
                    "specialist": specialist,
                    "galaxy": target_name,
                    "verification": "procedural_generation",
                    "tool_ids": list((tool_context or {}).get("tool_ids", [])),
                    "promotion_targets": promotion_targets,
                },
            )

        return generated_entry

    def route(
        self,
        query: str,
        *,
        specialist: str = "auto",
        domain_hint: str | None = None,
        galaxy_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        resolved = self.specialist_router.route(
            query=query,
            specialist=specialist,
            domain_hint=domain_hint,
            galaxy_names=galaxy_names,
        )
        selected_node = self._resolve_specialist_node(
            specialist=str(resolved.get("specialist", specialist)),
            query=query,
            domain_hint=domain_hint or str(resolved.get("domain", "")),
        )
        resolved["matryoshka_specialist"] = selected_node.name
        resolved["matryoshka_domain"] = selected_node.domain
        resolved["matryoshka_level"] = selected_node.level
        self._trace.append(
            "route specialist="
            f"{resolved['specialist']} domain={resolved['domain']} reason={resolved['reason']}"
        )
        return resolved

    def query(
        self,
        query: str,
        galaxy_names: Sequence[str] | None = None,
        top_k: int = 10,
        specialist: str = "any",
        domain_hint: str | None = None,
    ) -> list[dict[str, Any]]:
        route = self.route(
            query=query,
            specialist=specialist,
            domain_hint=domain_hint,
            galaxy_names=galaxy_names,
        )
        resolved_specialist = str(route["specialist"])
        names = [str(name) for name in route["galaxy_names"]] if route["galaxy_names"] else None
        self._trace.append(f"query specialist={resolved_specialist} top_k={top_k}")
        return self.galaxy_manager.query(
            query_text=query,
            specialist=resolved_specialist,
            top_k=top_k,
            galaxies=names,
        )

    def compose(
        self,
        query: str | None = None,
        patterns: Sequence[dict[str, Any]] | None = None,
        specialist: str = "math",
        task_examples: Sequence[dict[str, Any]] | None = None,
        use_enriched: bool = True,
    ) -> dict[str, Any]:
        self._trace.append(f"compose specialist={specialist} enriched={use_enriched}")
        if task_examples:
            tool_context = self._extract_tool_context(patterns or [], query_text=query or "")
            transform = self._infer_arc_transform(task_examples, prefer_enriched=use_enriched)
            return {
                "program_type": "arc_transform",
                "transform": transform,
                "specialist": specialist,
                "patterns_used": len(patterns or []),
                "tool_context": tool_context,
                "execution_plan": self._build_tool_execution_plan(tool_context),
            }

        tool_context = self._extract_tool_context(patterns or [], query_text=query or "")
        return {
            "program_type": "math_expression",
            "expression": query or "",
            "specialist": specialist,
            "patterns_used": len(patterns or []),
            "use_enriched": bool(use_enriched),
            "tool_context": tool_context,
            "execution_plan": self._build_tool_execution_plan(tool_context),
        }

    def execute(self, program: dict[str, Any], input_data: Any | None = None) -> Any:
        program_type = str(program.get("program_type", "unknown"))
        self._trace.append(f"execute type={program_type}")
        execution_plan = program.get("execution_plan") if isinstance(program, dict) else None

        if execution_plan and isinstance(input_data, dict):
            self._trace.append("execute via tool_entrypoint_chain payload")
            query_context = str(
                program.get("query")
                or program.get("expression")
                or program.get("prompt")
                or ""
            )
            specialist_id = str(program.get("specialist", "") or "") or None
            domain_hint = str(program.get("domain_hint", "") or "") or None
            return self.invoke_execution_plan_from_payload(
                execution_plan,
                input_data,
                query_context=query_context,
                specialist_id=specialist_id,
                domain_hint=domain_hint,
            )

        if program_type == "arc_transform":
            if input_data is None:
                raise ValueError("ARC execution requires input_data grid")
            transform = program.get("transform", {"op": "identity"})
            return self._apply_arc_transform(input_data, transform)

        if program_type == "math_expression":
            expression = str(program.get("expression", ""))
            use_enriched = bool(program.get("use_enriched", True))
            return self._solve_math(expression, use_enriched=use_enriched)

        raise ValueError(f"Unsupported program type: {program_type}")

    def resolve_execution_plan(self, execution_plan: dict[str, Any] | None) -> dict[str, Any] | None:
        return ToolExecutionResolver.resolve_plan_blueprint(execution_plan)

    def invoke_execution_plan(
        self,
        execution_plan: dict[str, Any] | None,
        *args: Any,
        query_context: str | None = None,
        specialist_id: str | None = None,
        domain_hint: str | None = None,
        **kwargs: Any,
    ) -> Any:
        return ToolExecutionResolver.invoke_primary_observed(
            execution_plan,
            *args,
            knowledgeverse=self.knowledgeverse,
            query_context=query_context,
            specialist_id=specialist_id,
            domain_hint=domain_hint,
            **kwargs,
        )

    def invoke_execution_plan_from_payload(
        self,
        execution_plan: dict[str, Any] | None,
        payload: Mapping[str, Any],
        *,
        query_context: str | None = None,
        specialist_id: str | None = None,
        domain_hint: str | None = None,
    ) -> Any:
        return ToolExecutionResolver.invoke_primary_from_payload_observed(
            execution_plan,
            payload,
            quality_tracker=self.execution_quality_tracker,
            knowledgeverse=self.knowledgeverse,
            query_context=query_context,
            specialist_id=specialist_id,
            domain_hint=domain_hint,
        )

    def observe_execution_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        outcome = int(max(-1, min(1, int(event.get("outcome", 0) or 0))))
        specialist = str(event.get("specialist_id", "") or "").strip()
        query = str(event.get("query_context", "") or "").strip()
        domain_hint = str(event.get("domain_hint", "") or "").strip()
        if specialist and outcome != 0 and query:
            self.learn_from_feedback(
                query=query,
                specialist=specialist,
                success=(outcome > 0),
                confidence=float(event.get("quality_signal", 0.0) or 0.0),
                domain_hint=(domain_hint or None),
            )
            self.save_weights()
        summary: dict[str, Any] = {}
        if self.execution_quality_tracker is not None:
            summary["quality"] = self.execution_quality_tracker.observe_event(
                event,
                specialist_catalog=self.list_specialists(),
            )
        if self.execution_grammar_detector is not None:
            summary["grammar"] = self.execution_grammar_detector.observe_event(event)
        return summary

    def _compose_procedural_program(
        self,
        *,
        query: str,
        candidates: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compose top procedural candidates into a new RPN program."""
        tool_selected: list[dict[str, Any]] = []
        base_selected: list[dict[str, Any]] = []
        for candidate in candidates[:12]:
            entry = candidate.get("entry", {})
            if not isinstance(entry, dict):
                continue
            rpn = str(entry.get("rpn_program", "")).strip()
            if not rpn:
                continue
            item = {
                "id": str(entry.get("id", "")),
                "category": str(entry.get("category", "unknown")),
                "rpn_program": rpn,
                "score": float(candidate.get("score", 1.0)),
                "confidence": self._candidate_confidence(entry, candidate),
                "entry": entry,
                "candidate": candidate,
            }
            if str(entry.get("type", "")).strip().lower() == "tool_node":
                if len(tool_selected) < 2:
                    tool_selected.append(item)
            elif len(base_selected) < 3:
                base_selected.append(item)
            if len(base_selected) >= 3 and len(tool_selected) >= 2:
                break

        selected = tool_selected + base_selected
        if not selected:
            # Fallback to original permissive behavior if the split produced nothing.
            for candidate in candidates[:3]:
                entry = candidate.get("entry", {})
                if not isinstance(entry, dict):
                    continue
                rpn = str(entry.get("rpn_program", "")).strip()
                if not rpn:
                    continue
                selected.append(
                    {
                        "id": str(entry.get("id", "")),
                        "category": str(entry.get("category", "unknown")),
                        "rpn_program": rpn,
                        "score": float(candidate.get("score", 1.0)),
                        "confidence": self._candidate_confidence(entry, candidate),
                        "entry": entry,
                        "candidate": candidate,
                    }
                )
        if not selected:
            return {}

        composed_rpn = "  ".join(item["rpn_program"] for item in selected)
        categories = [item["category"] for item in base_selected if item["category"]]
        if not categories:
            categories = [item["category"] for item in selected if item["category"]]
        category = max(set(categories), key=categories.count) if categories else "unknown"
        avg_conf = sum(item["confidence"] for item in selected) / len(selected)
        depth = len(selected)
        source_primitives = [item["id"] for item in base_selected if item["id"]]
        if not source_primitives:
            source_primitives = [item["id"] for item in selected if item["id"]]
        tool_context = self._extract_tool_context(
            [item["candidate"] for item in selected if isinstance(item.get("candidate"), dict)],
            query_text=query,
        )
        promotion_targets = self._extract_promotion_targets(
            [item["entry"] for item in selected if isinstance(item.get("entry"), dict)]
        )

        self._trace.append(
            "autogen compose "
            f"query='{query[:48]}' depth={depth} category={category} confidence={avg_conf:.3f}"
        )

        return {
            "rpn_program": composed_rpn,
            "depth": depth,
            "category": category,
            "confidence": max(0.05, min(avg_conf, 0.99)),
            "source_primitives": source_primitives,
            "tool_context": tool_context,
            "execution_plan": self._build_tool_execution_plan(tool_context),
            "promotion_targets": promotion_targets,
        }

    def _candidate_confidence(self, entry: dict[str, Any], candidate: dict[str, Any]) -> float:
        meta = entry.get("metadata", {}) if isinstance(entry.get("metadata", {}), dict) else {}
        for container in (entry, meta, candidate):
            raw = container.get("confidence")
            if raw is None:
                continue
            try:
                return max(0.0, min(float(raw), 1.0))
            except Exception:
                continue
        try:
            score = float(candidate.get("score", 1.0))
        except Exception:
            score = 1.0
        return max(0.05, min(score / 10.0, 0.95))

    def _extract_tool_context(
        self,
        patterns: Sequence[dict[str, Any]],
        query_text: str = "",
    ) -> dict[str, Any] | None:
        tool_entries: list[dict[str, Any]] = []
        for candidate in patterns:
            entry = candidate.get("entry")
            if not isinstance(entry, dict):
                continue
            if str(entry.get("type", "")).strip().lower() != "tool_node":
                continue
            tool_entries.append(entry)

        if not tool_entries:
            return None

        tool_ids: list[str] = []
        tool_kinds: list[str] = []
        runtime_statuses: list[str] = []
        codec_ops: list[str] = []
        component_refs: list[str] = []
        modalities: list[str] = []
        promotion_targets: list[str] = []
        math_core_tiers: list[str] = []
        math_core_roles: list[str] = []
        math_core_spawn_policies: list[str] = []
        math_core_cascades: list[str] = []
        memory_residencies: list[str] = []
        execution_residencies: list[str] = []
        execution_rows: list[dict[str, Any]] = []
        for entry in tool_entries[:6]:
            tool_id = str(entry.get("id", "")).strip()
            if tool_id:
                tool_ids.append(tool_id)
            component_refs.extend(str(ref) for ref in entry.get("component_refs", []) if str(ref).strip())
            metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            tool_kind = str(metadata.get("tool_kind", "")).strip()
            if tool_kind:
                tool_kinds.append(tool_kind)
            runtime_status = str(metadata.get("runtime_status", "")).strip()
            if runtime_status:
                runtime_statuses.append(runtime_status)
            memory_residency = str(metadata.get("memory_residency", "")).strip()
            if memory_residency:
                memory_residencies.append(memory_residency)
            execution_residency = str(metadata.get("execution_residency", "")).strip()
            if execution_residency:
                execution_residencies.append(execution_residency)
            codec_ops.extend(str(op) for op in metadata.get("codec_ops", []) if str(op).strip())
            modalities.extend(str(mod) for mod in metadata.get("modalities", []) if str(mod).strip())
            promotion_targets.extend(
                str(item) for item in metadata.get("promotion_targets", []) if str(item).strip()
            )
            math_core = metadata.get("math_core") if isinstance(metadata.get("math_core"), dict) else {}
            tier = math_core.get("preferred_tier")
            if tier is not None:
                math_core_tiers.append(str(tier).strip())
            role = str(math_core.get("tier_role", "")).strip()
            if role:
                math_core_roles.append(role)
            spawn_policy = str(math_core.get("spawn_policy", "")).strip()
            if spawn_policy:
                math_core_spawn_policies.append(spawn_policy)
            math_core_cascades.extend(
                str(item) for item in math_core.get("cascade", []) if str(item).strip()
            )
            entrypoints = [
                str(item).strip()
                for item in metadata.get("entrypoints", [])
                if str(item).strip()
            ]
            inputs = [str(item).strip() for item in metadata.get("inputs", []) if str(item).strip()]
            outputs = [str(item).strip() for item in metadata.get("outputs", []) if str(item).strip()]
            raw_argument_schemas = (
                metadata.get("entrypoint_argument_schemas")
                if isinstance(metadata.get("entrypoint_argument_schemas"), dict)
                else {}
            )
            raw_chain_presets = (
                metadata.get("execution_chain_presets")
                if isinstance(metadata.get("execution_chain_presets"), dict)
                else {}
            )
            argument_schemas = {
                str(key).strip(): dict(value)
                for key, value in raw_argument_schemas.items()
                if str(key).strip() and isinstance(value, dict)
            }
            chain_presets = {
                str(key).strip(): dict(value)
                for key, value in raw_chain_presets.items()
                if str(key).strip() and isinstance(value, dict)
            }
            if tool_id and runtime_status and entrypoints:
                execution_rows.append(
                    {
                        "tool_id": tool_id,
                        "tool_kind": tool_kind,
                        "runtime_status": runtime_status,
                        "entrypoints": entrypoints,
                        "inputs": inputs,
                        "outputs": outputs,
                        "argument_schemas": argument_schemas,
                        "chain_presets": chain_presets,
                        "math_core": math_core,
                    }
                )

        def _dedupe(values: list[str]) -> list[str]:
            out: list[str] = []
            seen: set[str] = set()
            for value in values:
                if not value or value in seen:
                    continue
                seen.add(value)
                out.append(value)
            return out

        deduped_tiers = _dedupe(math_core_tiers)
        deduped_roles = _dedupe(math_core_roles)
        deduped_spawn = _dedupe(math_core_spawn_policies)
        deduped_cascades = _dedupe(math_core_cascades)
        deduped_memory = _dedupe(memory_residencies)
        deduped_execution = _dedupe(execution_residencies)
        sorted_execution = self._prioritize_executable_tools(execution_rows, query_text=query_text)
        executable_tool_ids = _dedupe([str(row.get("tool_id", "")).strip() for row in sorted_execution])
        entrypoints = _dedupe(
            [
                str(entrypoint).strip()
                for row in sorted_execution
                for entrypoint in row.get("entrypoints", [])
                if str(entrypoint).strip()
            ]
        )
        primary_tool_id = executable_tool_ids[0] if executable_tool_ids else ""
        primary_entrypoint = entrypoints[0] if entrypoints else ""
        all_inputs = _dedupe(
            [
                str(item).strip()
                for row in sorted_execution
                for item in row.get("inputs", [])
                if str(item).strip()
            ]
        )
        all_outputs = _dedupe(
            [
                str(item).strip()
                for row in sorted_execution
                for item in row.get("outputs", [])
                if str(item).strip()
            ]
        )
        primary_argument_schema: dict[str, Any] | None = None
        primary_inputs: list[str] = []
        primary_outputs: list[str] = []
        primary_chain_presets: dict[str, Any] = {}
        for row in sorted_execution:
            row_entrypoints = [str(item).strip() for item in row.get("entrypoints", []) if str(item).strip()]
            if primary_tool_id and str(row.get("tool_id", "")).strip() != primary_tool_id and primary_entrypoint not in row_entrypoints:
                continue
            primary_argument_schema = dict(row.get("argument_schemas", {}).get(primary_entrypoint, {})) or None
            primary_inputs = [str(item).strip() for item in row.get("inputs", []) if str(item).strip()]
            primary_outputs = [str(item).strip() for item in row.get("outputs", []) if str(item).strip()]
            primary_chain_presets = {
                str(key).strip(): dict(value)
                for key, value in row.get("chain_presets", {}).items()
                if str(key).strip() and isinstance(value, dict)
            }
            break

        return {
            "tool_ids": _dedupe(tool_ids),
            "tool_kinds": _dedupe(tool_kinds),
            "runtime_statuses": _dedupe(runtime_statuses),
            "component_refs": _dedupe(component_refs),
            "codec_ops": _dedupe(codec_ops),
            "modalities": _dedupe(modalities),
            "inputs": all_inputs,
            "outputs": all_outputs,
            "promotion_targets": _dedupe(promotion_targets),
            "math_core_tiers": deduped_tiers,
            "math_core_roles": deduped_roles,
            "math_core_spawn_policies": deduped_spawn,
            "math_core_cascades": deduped_cascades,
            "memory_residencies": deduped_memory,
            "execution_residencies": deduped_execution,
            "executable_tool_ids": executable_tool_ids,
            "entrypoints": entrypoints,
            "primary_tool_id": primary_tool_id,
            "primary_entrypoint": primary_entrypoint,
            "primary_argument_schema": primary_argument_schema,
            "primary_inputs": primary_inputs,
            "primary_outputs": primary_outputs,
            "chain_presets": primary_chain_presets,
            "execution_chain": [
                {
                    "tool_id": str(row.get("tool_id", "")).strip(),
                    "tool_kind": str(row.get("tool_kind", "")).strip(),
                    "runtime_status": str(row.get("runtime_status", "")).strip(),
                    "inputs": [str(item).strip() for item in row.get("inputs", []) if str(item).strip()],
                    "outputs": [str(item).strip() for item in row.get("outputs", []) if str(item).strip()],
                    "entrypoints": [
                        str(item).strip() for item in row.get("entrypoints", []) if str(item).strip()
                    ],
                    "argument_schemas": {
                        str(key).strip(): dict(value)
                        for key, value in row.get("argument_schemas", {}).items()
                        if str(key).strip() and isinstance(value, dict)
                    },
                    "chain_presets": {
                        str(key).strip(): dict(value)
                        for key, value in row.get("chain_presets", {}).items()
                        if str(key).strip() and isinstance(value, dict)
                    },
                }
                for row in sorted_execution
            ],
            "math_core_plan": self._synthesize_math_core_plan(
                tiers=deduped_tiers,
                roles=deduped_roles,
                spawn_policies=deduped_spawn,
                cascades=deduped_cascades,
                memory_residencies=deduped_memory,
                execution_residencies=deduped_execution,
            ),
        }

    def _synthesize_math_core_plan(
        self,
        *,
        tiers: Sequence[str],
        roles: Sequence[str],
        spawn_policies: Sequence[str],
        cascades: Sequence[str],
        memory_residencies: Sequence[str],
        execution_residencies: Sequence[str],
    ) -> dict[str, Any] | None:
        if not tiers and not roles and not spawn_policies and not cascades:
            return None

        preferred_tier = 1
        if "3" in tiers or "master" in roles:
            preferred_tier = 3
        elif "2" in tiers or "worker" in roles:
            preferred_tier = 2

        tier_role = {1: "worker_worker", 2: "worker", 3: "master"}[preferred_tier]
        if preferred_tier == 3:
            cascade = ["parallel_fanout", "worker_reduce", "master_commit"]
        elif preferred_tier == 2:
            cascade = ["parallel_fanout", "worker_reduce"]
        else:
            cascade = ["parallel_fanout"]
        for step in cascades:
            if step and step not in cascade:
                cascade.append(step)

        return {
            "preferred_tier": preferred_tier,
            "tier_role": tier_role,
            "spawn_policy": str(spawn_policies[0]) if spawn_policies else "adaptive_reuse",
            "cascade": cascade,
            "memory_residency": (
                str(memory_residencies[0]) if memory_residencies else "knowledgeverse_galaxy"
            ),
            "execution_residency": str(execution_residencies[0]) if execution_residencies else "gpu_ptx",
        }

    def _prioritize_executable_tools(
        self,
        execution_rows: Sequence[dict[str, Any]],
        *,
        query_text: str = "",
    ) -> list[dict[str, Any]]:
        def _runtime_priority(status: str) -> int:
            normalized = str(status).strip().lower()
            if normalized == "ptx_rpn_available":
                return 4
            if normalized == "ptx_bridge_available":
                return 3
            if normalized == "ptx_runtime_available":
                return 2
            if normalized.startswith("ptx_"):
                return 1
            return 0

        def _kind_priority(kind: str) -> int:
            normalized = str(kind).strip().lower()
            if "fusion" in normalized:
                return 5
            if "projection" in normalized or "material" in normalized:
                return 4
            if "surface" in normalized or "displacement" in normalized:
                return 3
            if "signal" in normalized or "surface" in normalized:
                return 2
            if "codec" in normalized:
                return 1
            if "math_core" in normalized:
                return 0
            return 0

        query_tokens = str(query_text or "").strip().lower()

        def _query_bonus(row: Mapping[str, Any]) -> int:
            normalized_kind = str(row.get("tool_kind", "")).strip().lower()
            temporal_query = any(
                token in query_tokens
                for token in ("temporal", "timeline", "video", "animate", "animation", "sequence", "frame")
            )
            scene_query = any(
                token in query_tokens
                for token in ("scene", "layer", "layered", "composite", "composition", "playback")
            )
            replay_query = any(
                token in query_tokens
                for token in ("replay", "journal", "audit", "history")
            )
            library_query = any(
                token in query_tokens
                for token in ("knowledge", "library", "settled", "stable", "what i know")
            )
            garden_query = any(
                token in query_tokens
                for token in ("learning", "growing", "garden", "exploring", "exploration", "what i'm learning", "what i am learning")
            )
            museum_query = any(
                token in query_tokens
                for token in ("museum", "archive", "failures", "failure", "lessons", "history", "my history")
            )
            tour_query = any(
                token in query_tokens
                for token in ("tour", "overview", "all", "everything", "whole house")
            )
            ui_query = any(
                token in query_tokens
                for token in ("ui", "hud", "overlay", "widget", "panel", "cursor", "icon", "focus")
            )
            world_query = any(
                token in query_tokens
                for token in ("world", "ambient", "environment", "scene", "room", "house", "orbit", "breathe")
            )
            bonus = 0
            if temporal_query:
                if "temporal" in normalized_kind or "timeline" in normalized_kind or "video" in normalized_kind:
                    bonus += 6
            elif "temporal" in normalized_kind or "timeline" in normalized_kind or "video" in normalized_kind:
                bonus -= 6
            if scene_query:
                if "scene" in normalized_kind or "layering" in normalized_kind or "composition" in normalized_kind:
                    bonus += 7
            elif "scene" in normalized_kind:
                bonus -= 5
            if replay_query:
                if "replay_scene" in normalized_kind:
                    bonus += 10
            elif "replay_scene" in normalized_kind:
                bonus -= 8
            if library_query:
                if "library_scene" in normalized_kind:
                    bonus += 12
                elif "garden_scene" in normalized_kind or "museum_scene" in normalized_kind:
                    bonus -= 4
            elif "library_scene" in normalized_kind:
                bonus -= 8
            if garden_query:
                if "garden_scene" in normalized_kind:
                    bonus += 12
                elif "library_scene" in normalized_kind or "museum_scene" in normalized_kind:
                    bonus -= 4
            elif "garden_scene" in normalized_kind:
                bonus -= 8
            if museum_query:
                if "museum_scene" in normalized_kind:
                    bonus += 12
                elif "library_scene" in normalized_kind or "garden_scene" in normalized_kind:
                    bonus -= 4
            elif "museum_scene" in normalized_kind:
                bonus -= 8
            if tour_query:
                if "tour_scene" in normalized_kind:
                    bonus += 14
                elif "library_scene" in normalized_kind or "garden_scene" in normalized_kind or "museum_scene" in normalized_kind:
                    bonus -= 2
            elif "tour_scene" in normalized_kind:
                bonus -= 10
            if ui_query:
                if "ui_animation" in normalized_kind:
                    bonus += 8
                elif "ui_scene" in normalized_kind:
                    bonus += 9
                elif "world_animation" in normalized_kind:
                    bonus -= 3
                elif "world_scene" in normalized_kind:
                    bonus -= 4
            elif "ui_animation" in normalized_kind:
                bonus -= 8
            elif "ui_scene" in normalized_kind:
                bonus -= 9
            if world_query:
                if "world_animation" in normalized_kind:
                    bonus += 8
                elif "world_scene" in normalized_kind:
                    bonus += 9
                elif "ui_animation" in normalized_kind:
                    bonus -= 3
                elif "ui_scene" in normalized_kind:
                    bonus -= 4
            elif "world_animation" in normalized_kind:
                bonus -= 8
            elif "world_scene" in normalized_kind:
                bonus -= 9
            return bonus

        def _quality_bonus(row: Mapping[str, Any]) -> float:
            if self.execution_quality_tracker is None:
                return 0.0
            tool_id = str(row.get("tool_id", "")).strip()
            runtime_status = str(row.get("runtime_status", "")).strip()
            tool_kind = str(row.get("tool_kind", "")).strip()
            return float(
                self.execution_quality_tracker.tool_quality_bonus(tool_id)
                + self.execution_quality_tracker.source_quality_bonus(
                    tool_id,
                    runtime_status=runtime_status,
                    tool_kind=tool_kind,
                )
            )

        def _routing_bonus(row: Mapping[str, Any]) -> float:
            if self.execution_quality_tracker is None:
                return 0.0
            gate = self.execution_quality_tracker.routing_gate(
                str(row.get("tool_id", "")).strip(),
                runtime_status=str(row.get("runtime_status", "")).strip(),
                tool_kind=str(row.get("tool_kind", "")).strip(),
            )
            return float(self.execution_quality_tracker.routing_alignment_bonus(gate))

        ranked = list(execution_rows)
        ranked.sort(
            key=lambda row: (
                _query_bonus(row) + _quality_bonus(row) + _routing_bonus(row),
                _kind_priority(str(row.get("tool_kind", ""))),
                _runtime_priority(str(row.get("runtime_status", ""))),
            ),
            reverse=True,
        )
        return ranked

    def _extract_promotion_targets(self, entries: Sequence[dict[str, Any]]) -> list[str]:
        targets: list[str] = []
        for entry in entries:
            metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            for value in metadata.get("promotion_targets", []):
                token = str(value).strip()
                if token:
                    targets.append(token)
        out: list[str] = []
        seen: set[str] = set()
        for token in targets:
            if token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out

    def _build_tool_execution_plan(self, tool_context: dict[str, Any] | None) -> dict[str, Any] | None:
        if not tool_context:
            return None
        primary_tool_id = str(tool_context.get("primary_tool_id", "")).strip()
        primary_entrypoint = str(tool_context.get("primary_entrypoint", "")).strip()
        execution_chain = tool_context.get("execution_chain", [])
        if not primary_tool_id and not primary_entrypoint and not execution_chain:
            return None
        return {
            "mode": "tool_entrypoint_chain",
            "primary_tool_id": primary_tool_id,
            "primary_entrypoint": primary_entrypoint,
            "primary_argument_schema": tool_context.get("primary_argument_schema"),
            "primary_inputs": list(tool_context.get("primary_inputs", [])),
            "primary_outputs": list(tool_context.get("primary_outputs", [])),
            "chain_presets": dict(tool_context.get("chain_presets", {}) or {}),
            "executable_tool_ids": list(tool_context.get("executable_tool_ids", [])),
            "inputs": list(tool_context.get("inputs", [])),
            "outputs": list(tool_context.get("outputs", [])),
            "entrypoints": list(tool_context.get("entrypoints", [])),
            "execution_chain": list(execution_chain),
            "math_core_plan": tool_context.get("math_core_plan"),
            "promotion_targets": list(tool_context.get("promotion_targets", [])),
        }

    def _record_tool_promotion_pressure(
        self,
        *,
        query: str,
        source_galaxy: str,
        target_galaxy: str,
        specialist: str,
        tool_context: dict[str, Any],
        promotion_targets: Sequence[str],
    ) -> None:
        if self.knowledgeverse is None or not hasattr(self.knowledgeverse, "storage_root"):
            return
        payload = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "query": query,
            "source_galaxy": source_galaxy,
            "target_galaxy": target_galaxy,
            "specialist": specialist,
            "tool_ids": list(tool_context.get("tool_ids", [])),
            "tool_kinds": list(tool_context.get("tool_kinds", [])),
            "runtime_statuses": list(tool_context.get("runtime_statuses", [])),
            "codec_ops": list(tool_context.get("codec_ops", [])),
            "math_core_tiers": list(tool_context.get("math_core_tiers", [])),
            "math_core_roles": list(tool_context.get("math_core_roles", [])),
            "math_core_spawn_policies": list(tool_context.get("math_core_spawn_policies", [])),
            "math_core_cascades": list(tool_context.get("math_core_cascades", [])),
            "memory_residencies": list(tool_context.get("memory_residencies", [])),
            "execution_residencies": list(tool_context.get("execution_residencies", [])),
            "executable_tool_ids": list(tool_context.get("executable_tool_ids", [])),
            "entrypoints": list(tool_context.get("entrypoints", [])),
            "primary_tool_id": str(tool_context.get("primary_tool_id", "")),
            "primary_entrypoint": str(tool_context.get("primary_entrypoint", "")),
            "math_core_plan": tool_context.get("math_core_plan"),
            "promotion_targets": list(promotion_targets),
            "component_refs_count": len(tool_context.get("component_refs", [])),
        }
        logs_dir = Path(getattr(self.knowledgeverse, "storage_root")) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        out_path = logs_dir / "tool_promotion_pressure.jsonl"
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")

    def _specialist_for_galaxy(self, galaxy_name: str) -> str:
        lowered = galaxy_name.strip().lower()
        mapping = {
            "reality": "physics",
            "3dobjects": "cartographer",
            "3d_objects": "cartographer",
            "drawing": "visual",
            "math": "math",
            "grammar": "grammar",
            "audio": "audio",
            "tool": "cartographer",
        }
        return mapping.get(lowered, "any")

    def _default_generation_target(self, source_galaxy: str) -> str:
        lowered = source_galaxy.strip().lower()
        if lowered in {"reality", "3dobjects", "3d_objects"}:
            return "Grammar"
        return source_galaxy

    def select_answer(self, reasoning: Any, options: Sequence[str]) -> str:
        self._trace.append("select_answer")
        if not options:
            return ""
        numeric_reasoning = self._to_float(reasoning)
        if numeric_reasoning is not None:
            for option in options:
                val = self._to_float(option)
                if val is None:
                    continue
                if abs(val - numeric_reasoning) <= 1e-6:
                    return str(option)
        reason = str(reasoning).strip().lower()
        for option in options:
            normalized = str(option).strip().lower()
            if normalized == reason or reason in normalized:
                return str(option)
        return str(options[0])

    def get_reasoning_trace(self) -> list[str]:
        return list(self._trace)

    def clear_trace(self) -> None:
        self._trace.clear()

    def learn_from_feedback(
        self,
        *,
        query: str,
        specialist: str,
        success: bool,
        confidence: float | None = None,
        domain_hint: str | None = None,
    ) -> None:
        """Update persistent routing weights from observed outcomes."""
        self.navigator_specialist.learn_routing_topology(
            query=query,
            specialist=specialist,
            success=success,
        )
        node = self._resolve_specialist_node(
            specialist=specialist,
            query=query,
            domain_hint=domain_hint,
        )
        node.mark_query(success=success)
        if node.parent is not None:
            node.parent.update_routing_bias(node.name, success)
        observed_confidence = 0.5 if confidence is None else float(confidence)
        spawn_parent = node if node.children else (node.parent or self)
        decision = self.specialist_spawner.observe(
            parent=spawn_parent,
            query=query,
            confidence=observed_confidence,
            success=success,
            domain_hint=domain_hint,
        )
        if decision is not None:
            self._trace.append(
                "matryoshka_spawn "
                f"parent={decision.parent} child={decision.child} reason={decision.reason}"
            )

    def consolidate_weights_from_events(self, events: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """
        Consolidate routing weights from buffered Shadow Copy events.

        Returns a lightweight summary for SleepTime reporting.
        """
        updated = 0
        specialists: set[str] = set()
        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type", "")).lower()
            data = event.get("data", {})
            if not isinstance(data, dict):
                data = {}
            specialist = str(event.get("specialist") or data.get("specialist") or "grammar")
            query = str(data.get("query") or data.get("prompt") or event_type or specialist)
            confidence = float(event.get("confidence", data.get("confidence", 0.0)) or 0.0)
            success = ("success" in event_type) or (
                "fail" not in event_type and confidence >= 0.65
            )
            self.learn_from_feedback(
                query=query,
                specialist=specialist,
                success=success,
                confidence=confidence,
                domain_hint=str(data.get("domain_hint") or data.get("domain") or ""),
            )
            updated += 1
            specialists.add(specialist)
        self.navigator_specialist.save_state()
        return {
            "updated_count": updated,
            "updated_specialists": sorted(specialists),
            "weights_path": str(self.navigator_specialist.weight_store.path),
        }

    def save_weights(self) -> None:
        self.navigator_specialist.save_state()
        self.specialist_spawner.persist()
        self._save_specialist_tree()

    def _infer_arc_transform(
        self,
        task_examples: Sequence[dict[str, Any]],
        *,
        prefer_enriched: bool,
    ) -> dict[str, Any]:
        # Empty-mind baseline intentionally limits adaptation.
        if not prefer_enriched:
            return {"op": "identity"}

        # Galaxy-first ARC transform proposal from Grammar Galaxy.
        grammar_best_transform: dict[str, Any] | None = None
        grammar_best_score = -1.0
        grammar_rule_id: str | None = None
        try:
            grammar = self.galaxy_manager.get_galaxy("Grammar")
            proposer = getattr(grammar, "propose_arc_transform", None)
            if callable(proposer):
                proposal = proposer(list(task_examples))
                proposed = proposal.get("transform")
                confidence = float(proposal.get("confidence", 0.0))
                if isinstance(proposed, dict):
                    grammar_best_transform = proposed
                    grammar_best_score = confidence
                    grammar_rule_id = str(proposal.get("rule_id", ""))
        except Exception:
            grammar_best_transform = None
            grammar_best_score = -1.0
            grammar_rule_id = None

        op_candidates = (
            "identity",
            "flip_h",
            "flip_v",
            "rot90",
            "rot180",
            "rot270",
            "transpose",
        )
        best_transform: dict[str, Any] = {"op": "identity"}
        best_score = -1.0
        for op in op_candidates:
            score = 0.0
            for example in task_examples:
                predicted = self._apply_arc_transform(example["input"], {"op": op})
                score += self._grid_match_score(predicted, example["output"])
            avg_score = score / max(1, len(task_examples))
            if avg_score > best_score:
                best_score = avg_score
                best_transform = {"op": op}

        mapping = self._infer_color_mapping(task_examples)
        if mapping:
            score = 0.0
            for example in task_examples:
                predicted = self._apply_arc_transform(
                    example["input"],
                    {"op": "color_map", "mapping": mapping},
                )
                score += self._grid_match_score(predicted, example["output"])
            avg_score = score / max(1, len(task_examples))
            if avg_score > best_score:
                best_score = avg_score
                best_transform = {"op": "color_map", "mapping": mapping}

        if best_score >= 0.45:
            if grammar_best_transform is not None and grammar_best_score >= best_score:
                self._trace.append(
                    f"arc_transform source=grammar rule={grammar_rule_id or 'unknown'} confidence={grammar_best_score:.3f}"
                )
                return grammar_best_transform
            return best_transform

        if grammar_best_transform is not None and grammar_best_score >= 0.35:
            self._trace.append(
                f"arc_transform source=grammar rule={grammar_rule_id or 'unknown'} confidence={grammar_best_score:.3f}"
            )
            return grammar_best_transform

        return {"op": "identity"}

    def _infer_color_mapping(self, task_examples: Sequence[dict[str, Any]]) -> dict[int, int]:
        mapping: dict[int, int] = {}
        for example in task_examples:
            inp = example["input"]
            out = example["output"]
            if len(inp) != len(out):
                return {}
            for in_row, out_row in zip(inp, out):
                if len(in_row) != len(out_row):
                    return {}
                for in_val, out_val in zip(in_row, out_row):
                    in_int = int(in_val)
                    out_int = int(out_val)
                    prev = mapping.get(in_int)
                    if prev is None:
                        mapping[in_int] = out_int
                    elif prev != out_int:
                        return {}
        return mapping

    def _apply_arc_transform(self, grid: Sequence[Sequence[int]], transform: dict[str, Any]) -> list[list[int]]:
        op = str(transform.get("op", "identity"))
        rows = [list(map(int, row)) for row in grid]
        if op == "identity":
            return rows
        if op == "flip_h":
            return [list(reversed(row)) for row in rows]
        if op == "flip_v":
            return list(reversed(rows))
        if op == "rot90":
            return [list(col) for col in zip(*rows[::-1])]
        if op == "rot180":
            return [list(reversed(row)) for row in reversed(rows)]
        if op == "rot270":
            return [list(col) for col in zip(*rows)][::-1]
        if op == "transpose":
            return [list(col) for col in zip(*rows)]
        if op == "color_map":
            mapping = {int(k): int(v) for k, v in dict(transform.get("mapping", {})).items()}
            return [[mapping.get(val, val) for val in row] for row in rows]
        return rows

    def _grid_match_score(
        self,
        predicted: Sequence[Sequence[int]],
        expected: Sequence[Sequence[int]],
    ) -> float:
        if not predicted or not expected:
            return 0.0
        if len(predicted) != len(expected):
            return 0.0
        total = 0
        matched = 0
        for pred_row, exp_row in zip(predicted, expected):
            if len(pred_row) != len(exp_row):
                return 0.0
            total += len(pred_row)
            matched += sum(1 for a, b in zip(pred_row, exp_row) if int(a) == int(b))
        return (matched / total) if total else 0.0

    def _solve_math(self, text: str, *, use_enriched: bool) -> float | None:
        if self._math_debug:
            print(
                f"[K3D_MATH_DEBUG] solve_start use_enriched={bool(use_enriched)} "
                f"query={text[:200]!r}"
            )
        specialist = self.get_math_specialist()
        solved = specialist.process({"question": text}, use_enriched=use_enriched)
        if solved.get("status") == "success":
            result = self._to_float(solved.get("result"))
            if result is None:
                self._record_math_missing_signal(
                    reason="math_specialist_non_numeric_result",
                    query=text,
                    use_enriched=use_enriched,
                    extra={"raw_result": solved.get("result")},
                )
                return None
            rpn_program = str(solved.get("rpn_program", "")).strip()
            self._trace.append(f"math_solve_success rpn={rpn_program}")
            self._last_math_missing_signal = None
            if self._math_debug:
                print(
                    f"[K3D_MATH_DEBUG] solve_success result={result} rpn={rpn_program!r}"
                )
            return result

        reason = str(solved.get("reason", "math_specialist_failed")).strip() or "math_specialist_failed"
        extra: dict[str, Any] = {}
        for key in ("detail", "rpn_program", "pattern_type", "template_id", "pattern_id"):
            if key in solved:
                extra[key] = solved.get(key)
        self._record_math_missing_signal(
            reason=reason,
            query=text,
            use_enriched=use_enriched,
            extra=extra or None,
        )
        return None

    def _record_math_missing_signal(
        self,
        *,
        reason: str,
        query: str,
        use_enriched: bool,
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "reason": reason,
            "query": query[:240],
            "use_enriched": bool(use_enriched),
        }
        if extra:
            payload.update(extra)
        self._last_math_missing_signal = payload
        self._trace.append(f"math_solve_missing reason={reason}")
        if self._math_debug:
            print(
                f"[K3D_MATH_DEBUG] solve_missing {json.dumps(payload, ensure_ascii=True, sort_keys=True)}"
            )
        kv = getattr(self, "knowledgeverse", None)
        if kv is not None and hasattr(kv, "log_event"):
            try:
                kv.log_event("math_solve_missing_signal", payload)
            except Exception:
                pass

    def _select_math_rpn_template(
        self,
        grammar_candidates: Sequence[dict[str, Any]],
        math_candidates: Sequence[dict[str, Any]],
    ) -> str | None:
        for pool in (grammar_candidates, math_candidates):
            for candidate in pool:
                entry = candidate.get("entry", {})
                if not isinstance(entry, dict):
                    continue
                template = str(entry.get("rpn_program", "")).strip()
                if not template:
                    continue
                lowered = template.lower()
                if lowered in {"noop", "noop exec", "exec", "noop_exec"}:
                    continue
                return template
        return None

    def _extract_numeric_literals(self, text: str) -> list[float]:
        values: list[float] = []
        token: list[str] = []
        for char in text:
            if char.isdigit() or char in {".", "-"}:
                token.append(char)
                continue
            if token:
                val = self._to_float("".join(token))
                if val is not None:
                    values.append(val)
                token = []
        if token:
            val = self._to_float("".join(token))
            if val is not None:
                values.append(val)
        return values

    def _render_math_rpn_template(self, template: str, numbers: Sequence[float]) -> str | None:
        rendered = template
        if not rendered:
            return None
        for idx, number in enumerate(numbers):
            num = f"{number:.12g}"
            rendered = rendered.replace(f"{{g{idx}}}", num)
            rendered = rendered.replace(f"{{{idx}}}", num)
        if "{" in rendered and "}" in rendered:
            return None
        return rendered

    def _execute_math_rpn(self, rpn_program: str) -> float | None:
        try:
            from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine

            engine = ModularRPNEngine()
            try:
                value = engine.evaluate(rpn_program)
            finally:
                engine.close()
            self._last_math_execution_error = None
            return self._to_float(value)
        except Exception as exc:
            self._last_math_execution_error = f"{type(exc).__name__}: {exc}"
            return None

    def _to_float(self, value: Any) -> float | None:
        try:
            return float(value)
        except Exception:
            return None

    def get_last_math_missing_signal(self) -> dict[str, Any] | None:
        if self._last_math_missing_signal is None:
            return None
        return dict(self._last_math_missing_signal)

    # ========== Chat Specialist Integration (Sovereign LLM-compatible I/O) ==========

    def get_math_specialist(self):
        """
        Get or create Math Specialist instance.

        This keeps math solving on the specialist composition path even when
        benchmark runners use TRMNavigator directly.
        """
        if "MathSpecialist" not in self.children:
            self._bootstrap_matryoshka_specialists()

        math_child = self.children.get("MathSpecialist")
        if math_child is None:
            from knowledge3d.knowledgeverse.specialists.math_specialist import MathSpecialist

            math_specialist = MathSpecialist(
                knowledgeverse=self.knowledgeverse,
                parent=self,
            )
            self.children["MathSpecialist"] = math_specialist
            return math_specialist

        from knowledge3d.knowledgeverse.specialists.math_specialist import MathSpecialist

        if not isinstance(math_child, MathSpecialist):
            math_specialist = MathSpecialist(
                knowledgeverse=self.knowledgeverse,
                parent=self,
            )
            math_specialist.query_count = math_child.query_count
            math_specialist.success_count = math_child.success_count
            math_specialist.failure_count = math_child.failure_count
            self.children["MathSpecialist"] = math_specialist
            return math_specialist

        return math_child

    def get_chat_specialist(self):
        """
        Get or create Chat Specialist instance.

        Lazy initialization to avoid circular imports.
        """
        # Check if already initialized
        if "ChatSpecialist" not in self.children:
            # Re-bootstrap to ensure Chat Specialist exists
            self._bootstrap_matryoshka_specialists()

        chat_child = self.children.get("ChatSpecialist")
        if chat_child is None:
            # Fallback: create directly
            from knowledge3d.knowledgeverse.chat_specialist import ChatSpecialist
            chat_specialist = ChatSpecialist(
                knowledgeverse=self.knowledgeverse,
                parent=self
            )
            self.children["ChatSpecialist"] = chat_specialist
            return chat_specialist

        # Wrap base specialist with Chat Specialist functionality
        from knowledge3d.knowledgeverse.chat_specialist import ChatSpecialist
        if not isinstance(chat_child, ChatSpecialist):
            # Convert base specialist to Chat Specialist
            chat_specialist = ChatSpecialist(
                knowledgeverse=self.knowledgeverse,
                parent=self
            )
            chat_specialist.query_count = chat_child.query_count
            chat_specialist.success_count = chat_child.success_count
            chat_specialist.failure_count = chat_child.failure_count
            self.children["ChatSpecialist"] = chat_specialist
            return chat_specialist

        return chat_child

    def get_input_primer_specialist(self):
        """
        Get or create Input Primer specialist.

        The primer is pre-routing only. It normalizes inbound text/MCQ payloads
        before they are routed/solved by downstream specialists.
        """
        if "InputPrimerSpecialist" not in self.children:
            self._bootstrap_matryoshka_specialists()

        primer_child = self.children.get("InputPrimerSpecialist")
        if primer_child is None:
            from knowledge3d.knowledgeverse.input_primer_specialist import InputPrimerSpecialist

            primer = InputPrimerSpecialist(parent=self)
            self.children["InputPrimerSpecialist"] = primer
            return primer

        from knowledge3d.knowledgeverse.input_primer_specialist import InputPrimerSpecialist

        if not isinstance(primer_child, InputPrimerSpecialist):
            primer = InputPrimerSpecialist(parent=self)
            primer.query_count = primer_child.query_count
            primer.success_count = primer_child.success_count
            primer.failure_count = primer_child.failure_count
            self.children["InputPrimerSpecialist"] = primer
            return primer

        return primer_child

    def process_chat(
        self,
        messages: list[dict[str, str]],
        use_enriched: bool = True
    ) -> str:
        """
        Process chat messages using Chat Specialist (sovereign).

        Args:
            messages: Standard LLM chat format [{"role": "user", "content": "..."}]
            use_enriched: Whether to use enriched Galaxy content

        Returns:
            Standard LLM response string

        Example:
            >>> navigator = TRMNavigator(knowledgeverse)
            >>> messages = [
            ...     {"role": "user", "content": "What is the derivative of x^2?"}
            ... ]
            >>> response = navigator.process_chat(messages)
            >>> print(response)
            "Based on my mathematical knowledge: ..."
        """
        primer = self.get_input_primer_specialist()
        prepared_messages = primer.normalize_chat_messages(messages)
        chat_specialist = self.get_chat_specialist()
        return chat_specialist.process_chat_message(prepared_messages, use_enriched)

    def answer_multiple_choice(
        self,
        question_text: str,
        options: list[str],
        use_enriched: bool = True,
        galaxy_scope: list[str] | None = None,
    ) -> str:
        """
        Answer multiple-choice question using Chat Specialist.

        Used by MMLU and other benchmarks. Routes to best matching option
        using sovereign Galaxy navigation.

        Args:
            question_text: The question to answer
            options: List of possible answers (e.g., ["A", "B", "C", "D"])
            use_enriched: Whether to use enriched Galaxy content

        Returns:
            Best matching option from the list
        """
        primer = self.get_input_primer_specialist()
        prepared = primer.prepare_multiple_choice(question_text, options)

        chat_specialist = self.get_chat_specialist()
        predicted = chat_specialist.answer_multiple_choice(
            prepared["question_text"],
            prepared["options"],
            use_enriched,
            galaxy_scope=galaxy_scope,
        )
        return self._map_mcq_prediction_to_original(
            predicted=predicted,
            normalized_options=list(prepared["options"]),
            original_options=list(prepared["original_options"]),
        )

    def _map_mcq_prediction_to_original(
        self,
        *,
        predicted: str,
        normalized_options: list[str],
        original_options: list[str],
    ) -> str:
        pred = str(predicted).strip()
        if not pred:
            return original_options[0] if original_options else ""

        if pred in normalized_options:
            idx = normalized_options.index(pred)
            if 0 <= idx < len(original_options):
                return original_options[idx]

        if pred in original_options:
            return pred

        # Handle model returning option labels (A/B/C/D).
        label = pred.upper()
        if len(label) == 1 and "A" <= label <= "Z":
            idx = ord(label) - ord("A")
            if 0 <= idx < len(original_options):
                return original_options[idx]

        return original_options[0] if original_options else pred
