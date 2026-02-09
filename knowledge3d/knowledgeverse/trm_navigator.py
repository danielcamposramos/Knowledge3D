"""Knowledgeverse TRM navigator with resilient composition helpers."""

from __future__ import annotations

import ast
import datetime
import json
import math
from pathlib import Path
import re
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

    def _bootstrap_matryoshka_specialists(self) -> None:
        """
        Bootstrap default master/worker hierarchy.

        Layout:
        - MathSpecialist -> BasicMathSpecialist, PhDMathSpecialist
        - VisualSpecialist -> ArcVisualSpecialist, SpatialVisualSpecialist
        - PhysicsSpecialist -> MechanicsSpecialist, ProceduralRealitySpecialist
        - GrammarSpecialist -> SyntaxSpecialist, SemanticsSpecialist
        """
        self.children.clear()
        self.routing_bias.clear()

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

        if not names:
            return self.galaxy_manager.query(
                query_text=query,
                specialist=resolved_specialist,
                top_k=top_k,
            )

        tokens = {tok for tok in re.split(r"[^A-Za-z0-9_]+", query.lower()) if tok}
        scored: list[tuple[int, dict[str, Any], str]] = []
        for name in names:
            galaxy = self.galaxy_manager.get_galaxy(name)
            for entry in galaxy.entries:
                haystack = str(entry).lower()
                score = sum(1 for tok in tokens if tok in haystack)
                if score <= 0 and tokens:
                    continue
                scored.append((max(score, 1), entry, name))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {"galaxy": name, "score": score, "entry": entry}
            for score, entry, name in scored[: max(1, int(top_k))]
        ]

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
        # Empty-mind baseline supports only direct arithmetic.
        if not use_enriched:
            expr = self._extract_arithmetic_expr(text)
            return self._safe_eval(expr) if expr else None

        derivative = self._solve_derivative_prompt(text)
        if derivative is not None:
            return derivative

        expr = self._extract_arithmetic_expr(text)
        if expr:
            val = self._safe_eval(expr)
            if val is not None:
                return val
        return None

    def _solve_derivative_prompt(self, text: str) -> float | None:
        lowered = text.lower()
        x_value = self._extract_eval_x(text)
        if "sin(x)" in lowered and x_value is not None:
            return math.cos(x_value)
        if "cos(x)" in lowered and x_value is not None:
            return -math.sin(x_value)
        if "e^x" in lowered and x_value is not None:
            return math.exp(x_value)

        quotient_match = re.search(
            r"f\(x\)\s*=\s*\(([-+]?\d+)x([+-]\d+)\)\s*/\s*\(([-+]?\d+)x([+-]\d+)\)",
            text.replace(" ", ""),
        )
        if quotient_match and x_value is not None:
            a, b, c, d = [float(part) for part in quotient_match.groups()]
            numerator = (a * c * x_value + a * d) - (a * c * x_value + b * c)
            denominator = (c * x_value + d) ** 2
            if denominator == 0:
                return None
            return numerator / denominator

        poly_match = re.search(r"derivative of ([^@]+?) at x\s*=\s*([-+]?\d*\.?\d+)", lowered)
        if poly_match:
            expr_raw = poly_match.group(1)
            eval_x = float(poly_match.group(2))
            return self._differentiate_polynomial(expr_raw, eval_x)

        generic = re.search(r"f\(x\)\s*=\s*([^,]+?)\s+at x\s*=\s*([-+]?\d*\.?\d+)", lowered)
        if generic:
            expr_raw = generic.group(1)
            eval_x = float(generic.group(2))
            return self._differentiate_polynomial(expr_raw, eval_x)

        return None

    def _differentiate_polynomial(self, expr_raw: str, x_value: float) -> float | None:
        expr = expr_raw.replace(" ", "")
        if "/" in expr and "x" in expr:
            return None
        normalized = expr.replace("-", "+-")
        terms = [term for term in normalized.split("+") if term]
        result = 0.0
        matched_any = False
        for term in terms:
            if "x^" in term:
                coef_part, pow_part = term.split("x^", 1)
                coef = self._parse_coef(coef_part)
                power = self._to_float(pow_part)
                if power is None:
                    continue
                matched_any = True
                result += coef * power * (x_value ** (power - 1))
                continue
            if term.endswith("x"):
                coef = self._parse_coef(term[:-1])
                matched_any = True
                result += coef
                continue
        if not matched_any:
            return None
        return result

    def _parse_coef(self, raw: str) -> float:
        if raw in ("", "+"):
            return 1.0
        if raw == "-":
            return -1.0
        val = self._to_float(raw)
        return float(val) if val is not None else 0.0

    def _extract_eval_x(self, text: str) -> float | None:
        match = re.search(r"at x\s*=\s*([-+]?\d*\.?\d+)", text.lower())
        if not match:
            return None
        return float(match.group(1))

    def _extract_arithmetic_expr(self, text: str) -> str | None:
        # First quoted/extracted expressions.
        match = re.search(r"([\-+*/()0-9\.\s]{3,})", text)
        if match:
            expr = match.group(1).strip()
            if any(ch.isdigit() for ch in expr) and any(op in expr for op in "+-*/"):
                return expr
        # Last fallback: collect tokens.
        tokens = re.findall(r"[-+]?\d*\.?\d+|[()+\-*/]", text)
        if len(tokens) >= 3 and any(tok in "+-*/" for tok in tokens):
            return " ".join(tokens)
        return None

    def _safe_eval(self, expr: str) -> float | None:
        try:
            node = ast.parse(expr, mode="eval")
        except SyntaxError:
            return None
        if not self._is_safe_math_ast(node):
            return None
        try:
            value = eval(compile(node, "<math>", "eval"), {"__builtins__": {}}, {})
        except Exception:
            return None
        return self._to_float(value)

    def _is_safe_math_ast(self, node: ast.AST) -> bool:
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Constant,
            ast.Load,
            ast.Mod,
            ast.FloorDiv,
        )
        for child in ast.walk(node):
            if not isinstance(child, allowed_nodes):
                return False
            if isinstance(child, ast.Constant) and not isinstance(child.value, (int, float)):
                return False
        return True

    def _to_float(self, value: Any) -> float | None:
        try:
            return float(value)
        except Exception:
            return None
