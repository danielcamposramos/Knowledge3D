"""Knowledgeverse TRM navigator with resilient composition helpers."""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .galaxy_manager import GalaxyManager
from .navigator_specialist import NavigatorSpecialist
from .resilience import SelfHealingWrapper
from .specialist_base import SpecialistBase
from .specialist_router import SpecialistRouter
from .specialist_spawner import SpecialistSpawner


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
        grammar_master = self.spawn_child(name="GrammarSpecialist", domain="language")

        math_master.spawn_child(name="BasicMathSpecialist", domain="basic_math")
        math_master.spawn_child(name="PhDMathSpecialist", domain="phd_math")

        visual_master.spawn_child(name="ArcVisualSpecialist", domain="arc_visual")
        visual_master.spawn_child(name="SpatialVisualSpecialist", domain="spatial_reasoning")

        physics_master.spawn_child(name="MechanicsSpecialist", domain="mechanics")
        physics_master.spawn_child(name="ProceduralRealitySpecialist", domain="procedural_systems")

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
        candidates = self.query(
            query=f"{query} procedural generation",
            galaxy_names=[source_name],
            top_k=20,
            specialist="any",
            domain_hint=source_name.lower(),
        )
        if not candidates:
            return {"error": "No procedural primitives found", "query": query}

        composed = self._compose_procedural_program(query=query, candidates=candidates)
        rpn_program = str(composed.get("rpn_program", "")).strip()
        if not rpn_program:
            return {"error": "Failed to compose procedural program", "query": query}

        target_name = str(target_galaxy or self._default_generation_target(source_name))
        now = datetime.datetime.now(datetime.timezone.utc)
        generated_id = f"generated_{source_name.lower()}_{now.strftime('%Y%m%d_%H%M%S_%f')}"
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
            transform = self._infer_arc_transform(task_examples, prefer_enriched=use_enriched)
            return {
                "program_type": "arc_transform",
                "transform": transform,
                "specialist": specialist,
                "patterns_used": len(patterns or []),
            }

        return {
            "program_type": "math_expression",
            "expression": query or "",
            "specialist": specialist,
            "patterns_used": len(patterns or []),
            "use_enriched": bool(use_enriched),
        }

    def execute(self, program: dict[str, Any], input_data: Any | None = None) -> Any:
        program_type = str(program.get("program_type", "unknown"))
        self._trace.append(f"execute type={program_type}")

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

    def _compose_procedural_program(
        self,
        *,
        query: str,
        candidates: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compose top procedural candidates into a new RPN program."""
        selected: list[dict[str, Any]] = []
        for candidate in candidates[:8]:
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
                }
            )
            if len(selected) >= 3:
                break

        if not selected:
            return {}

        composed_rpn = "  ".join(item["rpn_program"] for item in selected)
        categories = [item["category"] for item in selected if item["category"]]
        category = max(set(categories), key=categories.count) if categories else "unknown"
        avg_conf = sum(item["confidence"] for item in selected) / len(selected)
        depth = len(selected)
        source_primitives = [item["id"] for item in selected if item["id"]]

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

    def _specialist_for_galaxy(self, galaxy_name: str) -> str:
        lowered = galaxy_name.strip().lower()
        mapping = {
            "reality": "physics",
            "3dobjects": "cartographer",
            "3d_objects": "cartographer",
            "drawing": "visual",
            "math": "math",
            "grammar": "grammar",
            "audio": "any",
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
