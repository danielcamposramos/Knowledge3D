"""Execution-plan resolution for Tool galaxy entrypoints."""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
import importlib
import time
from typing import Any, Mapping

from .execution_events import (
    ExecutionEventRecorder,
    attach_execution_event,
    build_execution_event,
)
from .execution_quality_tracker import ExecutionQualityTracker


@dataclass(frozen=True)
class EntrypointBlueprint:
    import_path: str
    module_path: str
    owner_name: str | None
    callable_name: str


class ToolExecutionResolver:
    """Resolve Tool execution plans into importable blueprints or callables."""

    @staticmethod
    def _normalize_param_specs(rows: Any) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        if not isinstance(rows, list):
            return specs
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "")).strip()
            if name:
                specs.append(
                    {
                        "name": name,
                        "aliases": [
                            str(value).strip()
                            for value in row.get("aliases", [])
                            if str(value).strip()
                        ],
                        "has_default": "default" in row,
                        "default": row.get("default"),
                    }
                )
        return specs

    @classmethod
    def _normalize_param_rows(cls, rows: Any) -> list[str]:
        return [spec["name"] for spec in cls._normalize_param_specs(rows)]

    @staticmethod
    def _resolve_payload_value(payload: Mapping[str, Any], spec: Mapping[str, Any]) -> tuple[str | None, Any]:
        name = str(spec.get("name", "")).strip()
        if name and name in payload:
            return name, payload[name]
        for alias in spec.get("aliases", []) or []:
            token = str(alias).strip()
            if token and token in payload:
                return token, payload[token]
        if bool(spec.get("has_default", False)):
            return "__default__", spec.get("default")
        return None, None

    @classmethod
    def validate_argument_schema(
        cls,
        argument_schema: dict[str, Any] | None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if not argument_schema:
            return

        positional_names = cls._normalize_param_rows(argument_schema.get("positional"))
        required_kwargs = cls._normalize_param_rows(argument_schema.get("required_kwargs"))
        optional_kwargs = cls._normalize_param_rows(argument_schema.get("optional_kwargs"))
        strict_kwargs = bool(argument_schema.get("strict_kwargs", True))
        allow_additional_positionals = bool(argument_schema.get("allow_additional_positionals", False))

        if len(args) < len(positional_names):
            missing = positional_names[len(args):]
            raise ValueError(f"missing required positional arguments: {', '.join(missing)}")
        if not allow_additional_positionals and len(args) > len(positional_names):
            raise ValueError(
                "too many positional arguments: "
                f"expected {len(positional_names)}, received {len(args)}"
            )

        missing_kwargs = [name for name in required_kwargs if name not in kwargs]
        if missing_kwargs:
            raise ValueError(f"missing required keyword arguments: {', '.join(missing_kwargs)}")

        if strict_kwargs:
            allowed = set(required_kwargs) | set(optional_kwargs)
            extra = sorted(name for name in kwargs if name not in allowed)
            if extra:
                raise ValueError(f"unexpected keyword arguments: {', '.join(extra)}")

    @staticmethod
    def resolve_entrypoint_blueprint(import_path: str) -> EntrypointBlueprint:
        token = str(import_path).strip()
        if not token:
            raise ValueError("entrypoint import path is required")

        parts = token.split(".")
        if len(parts) < 2:
            raise ValueError(f"invalid entrypoint import path: {token}")

        if len(parts) >= 3:
            module_path = ".".join(parts[:-2])
            owner_name = parts[-2]
            callable_name = parts[-1]
            try:
                importlib.import_module(module_path)
                return EntrypointBlueprint(
                    import_path=token,
                    module_path=module_path,
                    owner_name=owner_name,
                    callable_name=callable_name,
                )
            except Exception:
                pass

        module_path = ".".join(parts[:-1])
        callable_name = parts[-1]
        importlib.import_module(module_path)
        return EntrypointBlueprint(
            import_path=token,
            module_path=module_path,
            owner_name=None,
            callable_name=callable_name,
        )

    @staticmethod
    def instantiate_entrypoint(blueprint: EntrypointBlueprint) -> Any:
        module = importlib.import_module(blueprint.module_path)
        if blueprint.owner_name:
            owner = getattr(module, blueprint.owner_name)
            bound = getattr(owner(), blueprint.callable_name)
            return bound
        return getattr(module, blueprint.callable_name)

    @classmethod
    def instantiate_primary(cls, execution_plan: dict[str, Any] | None) -> Any:
        resolved = cls.resolve_plan_blueprint(execution_plan)
        if not resolved or resolved.get("primary_entrypoint") is None:
            raise ValueError("execution plan does not define a primary entrypoint")
        return cls.instantiate_entrypoint(resolved["primary_entrypoint"])

    @classmethod
    def invoke_primary(
        cls,
        execution_plan: dict[str, Any] | None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        resolved = cls.resolve_plan_blueprint(execution_plan)
        if not resolved:
            raise ValueError("execution plan is required")
        cls.validate_argument_schema(resolved.get("primary_argument_schema"), *args, **kwargs)
        bound = cls.instantiate_primary(execution_plan)
        return bound(*args, **kwargs)

    @staticmethod
    def _resolve_recorder(knowledgeverse: Any | None) -> ExecutionEventRecorder | None:
        if knowledgeverse is None or not hasattr(knowledgeverse, "storage_root"):
            return None
        return ExecutionEventRecorder(storage_root=getattr(knowledgeverse, "storage_root"))

    @staticmethod
    def _record_execution_event(
        *,
        knowledgeverse: Any | None,
        event: Any,
    ) -> None:
        recorder = ToolExecutionResolver._resolve_recorder(knowledgeverse)
        if recorder is not None:
            recorder.append(event, knowledgeverse=knowledgeverse)
        navigator = getattr(knowledgeverse, "trm_navigator", None) if knowledgeverse is not None else None
        if navigator is not None and hasattr(navigator, "observe_execution_event"):
            try:
                navigator.observe_execution_event(event.as_dict())
            except Exception:
                pass

    @staticmethod
    def _chain_tool_index(
        resolved: Mapping[str, Any] | None,
    ) -> tuple[dict[str, tuple[str, str]], list[str], list[str]]:
        mapping: dict[str, tuple[str, str]] = {}
        tool_ids: list[str] = []
        runtime_statuses: list[str] = []
        if not isinstance(resolved, Mapping):
            return mapping, tool_ids, runtime_statuses
        for row in resolved.get("chain", []) or []:
            if not isinstance(row, Mapping):
                continue
            tool_id = str(row.get("tool_id", "")).strip()
            runtime_status = str(row.get("runtime_status", "")).strip()
            if tool_id:
                tool_ids.append(tool_id)
                runtime_statuses.append(runtime_status)
            for blueprint in row.get("entrypoints", []) or []:
                if isinstance(blueprint, EntrypointBlueprint):
                    mapping[blueprint.import_path] = (tool_id, runtime_status)
        return mapping, tool_ids, runtime_statuses

    @classmethod
    def invoke_primary_observed(
        cls,
        execution_plan: dict[str, Any] | None,
        *args: Any,
        knowledgeverse: Any | None = None,
        query_context: str | None = None,
        specialist_id: str | None = None,
        domain_hint: str | None = None,
        **kwargs: Any,
    ) -> Any:
        resolved = cls.resolve_plan_blueprint(execution_plan)
        if not resolved:
            raise ValueError("execution plan is required")
        chain_depth = 1
        primary_tool_id = str(resolved.get("primary_tool_id", "")).strip()
        runtime_status = ""
        _chain_tool_index, plan_chain_tool_ids, plan_chain_statuses = cls._chain_tool_index(resolved)
        for row in resolved.get("chain", []):
            if str(row.get("tool_id", "")).strip() == primary_tool_id:
                runtime_status = str(row.get("runtime_status", "")).strip()
                break
        started_ns = time.perf_counter_ns()
        try:
            result = cls.invoke_primary(execution_plan, *args, **kwargs)
        except Exception as exc:
            event = build_execution_event(
                execution_plan=execution_plan,
                tool_id=primary_tool_id or "unknown_tool",
                query_context=str(query_context or ""),
                specialist_id=specialist_id,
                domain_hint=domain_hint,
                chain_depth=chain_depth,
                runtime_status=runtime_status,
                execution_us=(time.perf_counter_ns() - started_ns) // 1_000,
                error=exc,
                chain_tool_ids=plan_chain_tool_ids,
                chain_runtime_statuses=plan_chain_statuses,
            )
            cls._record_execution_event(knowledgeverse=knowledgeverse, event=event)
            raise
        event = build_execution_event(
            execution_plan=execution_plan,
            tool_id=primary_tool_id or "unknown_tool",
            query_context=str(query_context or ""),
            specialist_id=specialist_id,
            domain_hint=domain_hint,
            chain_depth=chain_depth,
            runtime_status=runtime_status,
            execution_us=(time.perf_counter_ns() - started_ns) // 1_000,
            result=result,
            chain_tool_ids=plan_chain_tool_ids,
            chain_runtime_statuses=plan_chain_statuses,
        )
        cls._record_execution_event(knowledgeverse=knowledgeverse, event=event)
        return attach_execution_event(result, event)

    @classmethod
    def bind_primary_arguments(
        cls,
        execution_plan: dict[str, Any] | None,
        payload: Mapping[str, Any],
    ) -> tuple[list[Any], dict[str, Any]]:
        if not isinstance(payload, Mapping):
            raise ValueError("payload mapping is required")
        resolved = cls.resolve_plan_blueprint(execution_plan)
        if not resolved:
            raise ValueError("execution plan is required")

        argument_schema = resolved.get("primary_argument_schema") or {}
        positional_specs = cls._normalize_param_specs(argument_schema.get("positional"))
        required_kwargs_specs = cls._normalize_param_specs(argument_schema.get("required_kwargs"))
        optional_kwargs_specs = cls._normalize_param_specs(argument_schema.get("optional_kwargs"))

        args: list[Any] = []
        for spec in positional_specs:
            matched_key, value = cls._resolve_payload_value(payload, spec)
            if matched_key is None:
                raise ValueError(f"missing payload field for positional argument: {spec['name']}")
            args.append(value)

        kwargs: dict[str, Any] = {}
        for spec in required_kwargs_specs:
            matched_key, value = cls._resolve_payload_value(payload, spec)
            if matched_key is None:
                raise ValueError(f"missing payload field for required keyword argument: {spec['name']}")
            kwargs[str(spec["name"])] = value
        for spec in optional_kwargs_specs:
            matched_key, value = cls._resolve_payload_value(payload, spec)
            if matched_key is not None:
                kwargs[str(spec["name"])] = value

        cls.validate_argument_schema(argument_schema, *args, **kwargs)
        return args, kwargs

    @classmethod
    def _payload_satisfies_schema(
        cls,
        payload: Mapping[str, Any],
        argument_schema: Mapping[str, Any] | None,
    ) -> tuple[bool, int]:
        if not argument_schema:
            return True, 0
        positional_specs = cls._normalize_param_specs(argument_schema.get("positional"))
        required_specs = cls._normalize_param_specs(argument_schema.get("required_kwargs"))
        optional_specs = cls._normalize_param_specs(argument_schema.get("optional_kwargs"))
        score = 0

        for spec in positional_specs:
            matched_key, _ = cls._resolve_payload_value(payload, spec)
            if matched_key is None:
                return False, 0
            if matched_key != "__default__":
                score += 1
        for spec in required_specs:
            matched_key, _ = cls._resolve_payload_value(payload, spec)
            if matched_key is None:
                return False, 0
            if matched_key != "__default__":
                score += 1
        for spec in optional_specs:
            matched_key, _ = cls._resolve_payload_value(payload, spec)
            if matched_key is not None and matched_key != "__default__":
                score += 1
        return True, score

    @staticmethod
    def _payload_matches_selectors(
        payload: Mapping[str, Any],
        selectors: Mapping[str, Any] | None,
    ) -> tuple[bool, int]:
        if not selectors:
            return True, 0
        score = 0
        for raw_key, expected in selectors.items():
            key = str(raw_key).strip()
            if not key or key not in payload:
                return False, 0
            value = payload[key]
            if isinstance(expected, (list, tuple, set, frozenset)):
                try:
                    matched = value in expected
                except Exception:
                    matched = False
                if not matched:
                    return False, 0
            elif value != expected:
                return False, 0
            score += 1
        return True, score

    @staticmethod
    def _resolve_execution_row(
        resolved: Mapping[str, Any] | None,
        *,
        tool_id: str = "",
        import_path: str = "",
    ) -> dict[str, Any]:
        if not isinstance(resolved, Mapping):
            return {}
        tool_token = str(tool_id).strip()
        import_token = str(import_path).strip()
        for row in resolved.get("chain", []) or []:
            if not isinstance(row, Mapping):
                continue
            row_tool_id = str(row.get("tool_id", "")).strip()
            row_runtime = str(row.get("runtime_status", "")).strip()
            row_kind = str(row.get("tool_kind", "")).strip()
            row_entrypoints = row.get("entrypoints", []) or []
            if tool_token and row_tool_id == tool_token:
                return {
                    "tool_id": row_tool_id,
                    "tool_kind": row_kind,
                    "runtime_status": row_runtime,
                }
            if import_token and any(
                isinstance(blueprint, EntrypointBlueprint) and blueprint.import_path == import_token
                for blueprint in row_entrypoints
            ):
                return {
                    "tool_id": row_tool_id,
                    "tool_kind": row_kind,
                    "runtime_status": row_runtime,
                }
        return {
            "tool_id": tool_token,
            "tool_kind": "",
            "runtime_status": "",
        }

    @classmethod
    def _candidate_rank(
        cls,
        *,
        schema_score: int,
        tool_id: str,
        runtime_status: str,
        tool_kind: str,
        plan_priority_bonus: float = 0.0,
        quality_tracker: ExecutionQualityTracker | None = None,
    ) -> dict[str, Any]:
        route_source = ExecutionQualityTracker.classify_route_source(runtime_status, tool_kind)
        quality_bonus = 0.0
        routing_gate: dict[str, Any] | None = None
        routing_bonus = 0.0
        if quality_tracker is not None:
            quality_bonus = float(
                quality_tracker.tool_quality_bonus(tool_id)
                + quality_tracker.source_quality_bonus(
                    tool_id,
                    runtime_status=runtime_status,
                    tool_kind=tool_kind,
                )
            )
            routing_gate = quality_tracker.routing_gate(
                tool_id,
                runtime_status=runtime_status,
                tool_kind=tool_kind,
            )
            routing_bonus = float(quality_tracker.routing_alignment_bonus(routing_gate))
        total_score = (2.0 * float(schema_score)) + float(plan_priority_bonus) + quality_bonus + routing_bonus
        return {
            "tool_id": tool_id,
            "tool_kind": tool_kind,
            "runtime_status": runtime_status,
            "route_source": route_source,
            "routing_gate": routing_gate,
            "schema_score": int(schema_score),
            "plan_priority_bonus": float(plan_priority_bonus),
            "quality_bonus": float(quality_bonus),
            "routing_bonus": float(routing_bonus),
            "score": float(total_score),
        }

    @classmethod
    def select_entrypoint_for_payload(
        cls,
        execution_plan: dict[str, Any] | None,
        payload: Mapping[str, Any],
        *,
        quality_tracker: ExecutionQualityTracker | None = None,
    ) -> dict[str, Any]:
        resolved = cls.resolve_plan_blueprint(execution_plan)
        if not resolved:
            raise ValueError("execution plan is required")
        if not isinstance(payload, Mapping):
            raise ValueError("payload mapping is required")

        candidates: list[dict[str, Any]] = []
        primary = resolved.get("primary_entrypoint")
        primary_schema = resolved.get("primary_argument_schema")
        primary_ok, primary_score = cls._payload_satisfies_schema(payload, primary_schema)
        primary_candidate: dict[str, Any] | None = None
        if primary is not None and primary_ok:
            primary_meta = cls._resolve_execution_row(
                resolved,
                tool_id=str(resolved.get("primary_tool_id", "")).strip(),
                import_path=primary.import_path,
            )
            rank = cls._candidate_rank(
                schema_score=primary_score,
                tool_id=str(primary_meta.get("tool_id", "")).strip(),
                runtime_status=str(primary_meta.get("runtime_status", "")).strip(),
                tool_kind=str(primary_meta.get("tool_kind", "")).strip(),
                plan_priority_bonus=0.5,
                quality_tracker=quality_tracker,
            )
            candidates.append(
                {
                    "tool_id": str(primary_meta.get("tool_id", "")).strip(),
                    "tool_kind": str(primary_meta.get("tool_kind", "")).strip(),
                    "runtime_status": str(primary_meta.get("runtime_status", "")).strip(),
                    "route_source": str(rank.get("route_source", "")).strip(),
                    "routing_gate": rank.get("routing_gate"),
                    "blueprint": primary,
                    "argument_schema": primary_schema,
                    "schema_score": int(rank["schema_score"]),
                    "plan_priority_bonus": float(rank["plan_priority_bonus"]),
                    "quality_bonus": float(rank["quality_bonus"]),
                    "routing_bonus": float(rank["routing_bonus"]),
                    "score": float(rank["score"]),
                }
            )
            primary_candidate = candidates[-1]

        for row in resolved.get("chain", []):
            tool_id = str(row.get("tool_id", "")).strip()
            tool_kind = str(row.get("tool_kind", "")).strip()
            runtime_status = str(row.get("runtime_status", "")).strip()
            row_schemas = row.get("argument_schemas", {}) if isinstance(row.get("argument_schemas"), dict) else {}
            for blueprint in row.get("entrypoints", []) or []:
                schema = row_schemas.get(blueprint.import_path)
                ok, score = cls._payload_satisfies_schema(payload, schema)
                if not ok:
                    continue
                rank = cls._candidate_rank(
                    schema_score=score,
                    tool_id=tool_id,
                    runtime_status=runtime_status,
                    tool_kind=tool_kind,
                    quality_tracker=quality_tracker,
                )
                candidates.append(
                    {
                        "tool_id": tool_id,
                        "tool_kind": tool_kind,
                        "runtime_status": runtime_status,
                        "route_source": str(rank.get("route_source", "")).strip(),
                        "routing_gate": rank.get("routing_gate"),
                        "blueprint": blueprint,
                        "argument_schema": schema,
                        "schema_score": int(rank["schema_score"]),
                        "plan_priority_bonus": float(rank["plan_priority_bonus"]),
                        "quality_bonus": float(rank["quality_bonus"]),
                        "routing_bonus": float(rank["routing_bonus"]),
                        "score": float(rank["score"]),
                    }
                )

        if not candidates:
            raise ValueError("no executable Tool entrypoint matches the provided payload")
        candidates.sort(
            key=lambda item: (
                float(item["score"]),
                int(item["schema_score"]),
                float(item.get("plan_priority_bonus", 0.0)),
                float(item["routing_bonus"]),
                float(item["quality_bonus"]),
            ),
            reverse=True,
        )
        if (
            primary_candidate is not None
            and candidates
            and str(candidates[0].get("tool_id", "")).strip() != str(primary_candidate.get("tool_id", "")).strip()
            and int(candidates[0].get("schema_score", 0)) < int(primary_candidate.get("schema_score", 0))
        ):
            return primary_candidate
        return candidates[0]

    @classmethod
    def bind_payload_to_entrypoint(
        cls,
        blueprint: EntrypointBlueprint,
        argument_schema: Mapping[str, Any] | None,
        payload: Mapping[str, Any],
    ) -> tuple[list[Any], dict[str, Any]]:
        if not isinstance(payload, Mapping):
            raise ValueError("payload mapping is required")
        positional_specs = cls._normalize_param_specs((argument_schema or {}).get("positional"))
        required_kwargs_specs = cls._normalize_param_specs((argument_schema or {}).get("required_kwargs"))
        optional_kwargs_specs = cls._normalize_param_specs((argument_schema or {}).get("optional_kwargs"))

        args: list[Any] = []
        for spec in positional_specs:
            matched_key, value = cls._resolve_payload_value(payload, spec)
            if matched_key is None:
                raise ValueError(f"missing payload field for positional argument: {spec['name']}")
            args.append(value)

        kwargs: dict[str, Any] = {}
        for spec in required_kwargs_specs:
            matched_key, value = cls._resolve_payload_value(payload, spec)
            if matched_key is None:
                raise ValueError(f"missing payload field for required keyword argument: {spec['name']}")
            kwargs[str(spec["name"])] = value
        for spec in optional_kwargs_specs:
            matched_key, value = cls._resolve_payload_value(payload, spec)
            if matched_key is not None:
                kwargs[str(spec["name"])] = value

        cls.validate_argument_schema(dict(argument_schema or {}), *args, **kwargs)
        return args, kwargs

    @staticmethod
    def _extract_result_field(result: Any, field_name: str) -> Any:
        token = str(field_name).strip()
        if not token:
            return None
        if token == "__self__":
            return result
        if isinstance(result, Mapping) and token in result:
            return result[token]
        if hasattr(result, token):
            return getattr(result, token)
        raise ValueError(f"result field not found: {token}")

    @classmethod
    def _store_chain_result(
        cls,
        context: dict[str, Any],
        step: Mapping[str, Any],
        result: Any,
    ) -> None:
        for alias in step.get("store_as", []) or []:
            token = str(alias).strip()
            if token:
                context[token] = result
        for alias, field_name in (step.get("store_fields", {}) or {}).items():
            token = str(alias).strip()
            if not token:
                continue
            context[token] = cls._extract_result_field(result, str(field_name))

    @classmethod
    def _enrich_chain_result(cls, result: Any, context: Mapping[str, Any]) -> Any:
        if not hasattr(result, "metadata"):
            return result
        metadata = getattr(result, "metadata")
        if not isinstance(metadata, dict):
            return result
        merged = dict(metadata)

        selection = context.get("material_selection")
        if selection is not None:
            if hasattr(selection, "target_stops"):
                merged.setdefault("target_material_stops", [list(row) for row in getattr(selection, "target_stops")])
            if hasattr(selection, "selected_stops"):
                merged.setdefault("selected_material_stops", [list(row) for row in getattr(selection, "selected_stops")])
            if hasattr(selection, "score_table"):
                merged.setdefault("material_score_table", list(getattr(selection, "score_table")))
            if hasattr(selection, "math_core_plan"):
                merged.setdefault("selection_math_core_plan", getattr(selection, "math_core_plan"))

        projection = context.get("spectrogram_projection")
        if projection is not None and hasattr(projection, "metadata"):
            projection_meta = getattr(projection, "metadata")
            if isinstance(projection_meta, dict):
                merged.setdefault(
                    "signal_projection_summary",
                    {
                        "frame_count": int(projection_meta.get("frame_count", 0)),
                        "frequency_bins": int(projection_meta.get("frequency_bins", 0)),
                        "positive_ratio": float(projection_meta.get("positive_ratio", 0.0)),
                        "negative_ratio": float(projection_meta.get("negative_ratio", 0.0)),
                        "neutral_ratio": float(projection_meta.get("neutral_ratio", 0.0)),
                    },
                )
                if "clip_id" in projection_meta:
                    merged.setdefault("clip_id", str(projection_meta.get("clip_id", "")))
                if "frame_size" in projection_meta:
                    merged.setdefault("signal_frame_size", int(projection_meta.get("frame_size", 0)))
                if "threshold" in projection_meta:
                    merged.setdefault("signal_threshold", float(projection_meta.get("threshold", 0.0)))

        if is_dataclass(result):
            return replace(result, metadata=merged)
        return result

    @classmethod
    def select_chain_preset_for_payload(
        cls,
        execution_plan: dict[str, Any] | None,
        payload: Mapping[str, Any],
        *,
        quality_tracker: ExecutionQualityTracker | None = None,
    ) -> dict[str, Any] | None:
        resolved = cls.resolve_plan_blueprint(execution_plan)
        if not resolved:
            raise ValueError("execution plan is required")
        if not isinstance(payload, Mapping):
            raise ValueError("payload mapping is required")

        candidates: list[dict[str, Any]] = []
        primary_tool_id = str(resolved.get("primary_tool_id", "")).strip()
        primary_candidate: dict[str, Any] | None = None
        for preset_name, preset in (resolved.get("chain_presets", {}) or {}).items():
            required_inputs = [str(item).strip() for item in preset.get("required_inputs", []) if str(item).strip()]
            if any(name not in payload for name in required_inputs):
                continue
            selectors_ok, selector_score = cls._payload_matches_selectors(
                payload,
                preset.get("selectors") if isinstance(preset, dict) else None,
            )
            if not selectors_ok:
                continue
            primary_meta = cls._resolve_execution_row(resolved, tool_id=primary_tool_id)
            rank = cls._candidate_rank(
                schema_score=len(required_inputs) + int(selector_score),
                tool_id=str(primary_meta.get("tool_id", "")).strip(),
                runtime_status=str(primary_meta.get("runtime_status", "")).strip(),
                tool_kind=str(primary_meta.get("tool_kind", "")).strip(),
                plan_priority_bonus=0.5,
                quality_tracker=quality_tracker,
            )
            candidates.append(
                {
                    "tool_id": str(primary_meta.get("tool_id", "")).strip(),
                    "tool_kind": str(primary_meta.get("tool_kind", "")).strip(),
                    "runtime_status": str(primary_meta.get("runtime_status", "")).strip(),
                    "route_source": str(rank.get("route_source", "")).strip(),
                    "routing_gate": rank.get("routing_gate"),
                    "preset_name": str(preset_name).strip(),
                    "preset": preset,
                    "schema_score": len(required_inputs) + int(selector_score),
                    "plan_priority_bonus": float(rank["plan_priority_bonus"]),
                    "quality_bonus": float(rank["quality_bonus"]),
                    "routing_bonus": float(rank["routing_bonus"]),
                    "score": float(rank["score"]),
                }
            )
            primary_candidate = candidates[-1]

        for row in resolved.get("chain", []):
            tool_id = str(row.get("tool_id", "")).strip()
            tool_kind = str(row.get("tool_kind", "")).strip()
            runtime_status = str(row.get("runtime_status", "")).strip()
            for preset_name, preset in (row.get("chain_presets", {}) or {}).items():
                required_inputs = [str(item).strip() for item in preset.get("required_inputs", []) if str(item).strip()]
                if any(name not in payload for name in required_inputs):
                    continue
                selectors_ok, selector_score = cls._payload_matches_selectors(
                    payload,
                    preset.get("selectors") if isinstance(preset, dict) else None,
                )
                if not selectors_ok:
                    continue
                rank = cls._candidate_rank(
                    schema_score=len(required_inputs) + int(selector_score),
                    tool_id=tool_id,
                    runtime_status=runtime_status,
                    tool_kind=tool_kind,
                    quality_tracker=quality_tracker,
                )
                candidates.append(
                    {
                        "tool_id": tool_id,
                        "tool_kind": tool_kind,
                        "runtime_status": runtime_status,
                        "route_source": str(rank.get("route_source", "")).strip(),
                        "routing_gate": rank.get("routing_gate"),
                        "preset_name": str(preset_name).strip(),
                        "preset": preset,
                        "schema_score": len(required_inputs) + int(selector_score),
                        "plan_priority_bonus": float(rank["plan_priority_bonus"]),
                        "quality_bonus": float(rank["quality_bonus"]),
                        "routing_bonus": float(rank["routing_bonus"]),
                        "score": float(rank["score"]),
                    }
                )

        if not candidates:
            return None
        candidates.sort(
            key=lambda item: (
                float(item["score"]),
                int(item["schema_score"]),
                float(item.get("plan_priority_bonus", 0.0)),
                len((item["preset"] or {}).get("steps", [])),
                float(item["routing_bonus"]),
                float(item["quality_bonus"]),
            ),
            reverse=True,
        )
        if (
            primary_candidate is not None
            and candidates
            and str(candidates[0].get("tool_id", "")).strip() != str(primary_candidate.get("tool_id", "")).strip()
            and int(candidates[0].get("schema_score", 0)) < int(primary_candidate.get("schema_score", 0))
        ):
            return primary_candidate
        return candidates[0]

    @classmethod
    def invoke_chain_from_payload(
        cls,
        execution_plan: dict[str, Any] | None,
        payload: Mapping[str, Any],
        *,
        quality_tracker: ExecutionQualityTracker | None = None,
    ) -> Any:
        selected = cls.select_chain_preset_for_payload(
            execution_plan,
            payload,
            quality_tracker=quality_tracker,
        )
        if not selected:
            raise ValueError("no execution chain preset matches the provided payload")
        preset = selected["preset"]
        context: dict[str, Any] = dict(payload)
        last_result: Any = None
        for step in preset.get("steps", []) or []:
            blueprint = step.get("entrypoint")
            if not isinstance(blueprint, EntrypointBlueprint):
                raise ValueError("chain step entrypoint blueprint is required")
            args, kwargs = cls.bind_payload_to_entrypoint(
                blueprint,
                step.get("argument_schema"),
                context,
            )
            bound = cls.instantiate_entrypoint(blueprint)
            last_result = bound(*args, **kwargs)
            cls._store_chain_result(context, step, last_result)

        return_alias = str(preset.get("return_alias", "")).strip()
        if return_alias and return_alias in context:
            last_result = context[return_alias]
        return cls._enrich_chain_result(last_result, context)

    @classmethod
    def _invoke_chain_from_payload_observed(
        cls,
        execution_plan: dict[str, Any] | None,
        payload: Mapping[str, Any],
        *,
        quality_tracker: ExecutionQualityTracker | None = None,
        knowledgeverse: Any | None = None,
        query_context: str | None = None,
        specialist_id: str | None = None,
        domain_hint: str | None = None,
    ) -> Any:
        resolved = cls.resolve_plan_blueprint(execution_plan)
        if not resolved:
            raise ValueError("execution plan is required")
        selected = cls.select_chain_preset_for_payload(
            execution_plan,
            payload,
            quality_tracker=quality_tracker,
        )
        if not selected:
            raise ValueError("no execution chain preset matches the provided payload")

        preset = selected["preset"]
        context: dict[str, Any] = dict(payload)
        last_result: Any = None
        chain_steps = list(preset.get("steps", []) or [])
        chain_depth = max(1, len(chain_steps))
        chain_tool_index, plan_chain_tool_ids, plan_chain_statuses = cls._chain_tool_index(resolved)
        executed_tool_ids: list[str] = []
        executed_statuses: list[str] = []
        started_chain_ns = time.perf_counter_ns()

        for step in chain_steps:
            blueprint = step.get("entrypoint")
            if not isinstance(blueprint, EntrypointBlueprint):
                raise ValueError("chain step entrypoint blueprint is required")
            step_tool_id, step_runtime_status = chain_tool_index.get(
                blueprint.import_path,
                (str(selected.get("tool_id", "")).strip(), ""),
            )
            args, kwargs = cls.bind_payload_to_entrypoint(
                blueprint,
                step.get("argument_schema"),
                context,
            )
            bound = cls.instantiate_entrypoint(blueprint)
            started_step_ns = time.perf_counter_ns()
            try:
                step_result = bound(*args, **kwargs)
            except Exception as exc:
                step_event = build_execution_event(
                    execution_plan=execution_plan,
                    tool_id=step_tool_id or "unknown_tool",
                    query_context=str(query_context or ""),
                    specialist_id=specialist_id,
                    domain_hint=domain_hint,
                    chain_depth=chain_depth,
                    runtime_status=step_runtime_status,
                    execution_us=(time.perf_counter_ns() - started_step_ns) // 1_000,
                    error=exc,
                    chain_tool_ids=plan_chain_tool_ids or tuple(executed_tool_ids + ([step_tool_id] if step_tool_id else [])),
                    chain_runtime_statuses=plan_chain_statuses or tuple(executed_statuses + ([step_runtime_status] if step_runtime_status else [])),
                    execution_mode="tool_chain_step",
                )
                cls._record_execution_event(knowledgeverse=knowledgeverse, event=step_event)
                top_event = build_execution_event(
                    execution_plan=execution_plan,
                    tool_id=str(selected.get("tool_id", "")).strip() or "unknown_tool",
                    query_context=str(query_context or ""),
                    specialist_id=specialist_id,
                    domain_hint=domain_hint,
                    chain_depth=chain_depth,
                    runtime_status="chain_failed",
                    execution_us=(time.perf_counter_ns() - started_chain_ns) // 1_000,
                    error=exc,
                    chain_tool_ids=plan_chain_tool_ids or tuple(executed_tool_ids + ([step_tool_id] if step_tool_id else [])),
                    chain_runtime_statuses=plan_chain_statuses or tuple(executed_statuses + ([step_runtime_status] if step_runtime_status else [])),
                )
                cls._record_execution_event(knowledgeverse=knowledgeverse, event=top_event)
                raise

            executed_tool_ids.append(step_tool_id)
            executed_statuses.append(step_runtime_status)
            step_event = build_execution_event(
                execution_plan=execution_plan,
                tool_id=step_tool_id or "unknown_tool",
                query_context=str(query_context or ""),
                specialist_id=specialist_id,
                domain_hint=domain_hint,
                chain_depth=chain_depth,
                runtime_status=step_runtime_status,
                execution_us=(time.perf_counter_ns() - started_step_ns) // 1_000,
                result=step_result,
                chain_tool_ids=tuple(executed_tool_ids),
                chain_runtime_statuses=tuple(executed_statuses),
                execution_mode="tool_chain_step",
            )
            cls._record_execution_event(knowledgeverse=knowledgeverse, event=step_event)
            cls._store_chain_result(context, step, step_result)
            last_result = step_result

        return_alias = str(preset.get("return_alias", "")).strip()
        if return_alias and return_alias in context:
            last_result = context[return_alias]
        final_result = cls._enrich_chain_result(last_result, context)
        top_event = build_execution_event(
            execution_plan=execution_plan,
            tool_id=str(selected.get("tool_id", "")).strip() or "unknown_tool",
            query_context=str(query_context or ""),
            specialist_id=specialist_id,
            domain_hint=domain_hint,
            chain_depth=chain_depth,
            runtime_status="chain_complete",
            execution_us=(time.perf_counter_ns() - started_chain_ns) // 1_000,
            result=final_result,
            chain_tool_ids=plan_chain_tool_ids or tuple(executed_tool_ids),
            chain_runtime_statuses=plan_chain_statuses or tuple(executed_statuses),
        )
        cls._record_execution_event(knowledgeverse=knowledgeverse, event=top_event)
        return attach_execution_event(final_result, top_event)

    @classmethod
    def invoke_primary_from_payload(
        cls,
        execution_plan: dict[str, Any] | None,
        payload: Mapping[str, Any],
        *,
        quality_tracker: ExecutionQualityTracker | None = None,
    ) -> Any:
        chain_selected = cls.select_chain_preset_for_payload(
            execution_plan,
            payload,
            quality_tracker=quality_tracker,
        )
        if chain_selected is not None:
            return cls.invoke_chain_from_payload(
                execution_plan,
                payload,
                quality_tracker=quality_tracker,
            )
        selected = cls.select_entrypoint_for_payload(
            execution_plan,
            payload,
            quality_tracker=quality_tracker,
        )
        args, kwargs = cls.bind_payload_to_entrypoint(
            selected["blueprint"],
            selected.get("argument_schema"),
            payload,
        )
        bound = cls.instantiate_entrypoint(selected["blueprint"])
        return bound(*args, **kwargs)

    @classmethod
    def invoke_primary_from_payload_observed(
        cls,
        execution_plan: dict[str, Any] | None,
        payload: Mapping[str, Any],
        *,
        quality_tracker: ExecutionQualityTracker | None = None,
        knowledgeverse: Any | None = None,
        query_context: str | None = None,
        specialist_id: str | None = None,
        domain_hint: str | None = None,
    ) -> Any:
        resolved = cls.resolve_plan_blueprint(execution_plan)
        if not resolved:
            raise ValueError("execution plan is required")
        selected_chain = cls.select_chain_preset_for_payload(
            execution_plan,
            payload,
            quality_tracker=quality_tracker,
        )
        if selected_chain is not None:
            return cls._invoke_chain_from_payload_observed(
                execution_plan,
                payload,
                quality_tracker=quality_tracker,
                knowledgeverse=knowledgeverse,
                query_context=query_context,
                specialist_id=specialist_id,
                domain_hint=domain_hint,
            )
        selected_entry = None if selected_chain is not None else cls.select_entrypoint_for_payload(
            execution_plan,
            payload,
            quality_tracker=quality_tracker,
        )
        tool_id = ""
        chain_depth = 1
        runtime_status = ""
        _chain_tool_index, plan_chain_tool_ids, plan_chain_statuses = cls._chain_tool_index(resolved)
        if selected_chain is not None:
            tool_id = str(selected_chain.get("tool_id", "")).strip()
            chain_depth = len((selected_chain.get("preset") or {}).get("steps", []))
            runtime_status = str(selected_chain.get("runtime_status", "")).strip()
        elif selected_entry is not None:
            tool_id = str(selected_entry.get("tool_id", "")).strip()
            runtime_status = str(selected_entry.get("runtime_status", "")).strip()
        if resolved and not runtime_status:
            for row in resolved.get("chain", []):
                if str(row.get("tool_id", "")).strip() == tool_id:
                    runtime_status = str(row.get("runtime_status", "")).strip()
                    break

        started_ns = time.perf_counter_ns()
        try:
            result = cls.invoke_primary_from_payload(
                execution_plan,
                payload,
                quality_tracker=quality_tracker,
            )
        except Exception as exc:
            event = build_execution_event(
                execution_plan=execution_plan,
                tool_id=tool_id or "unknown_tool",
                query_context=str(query_context or ""),
                specialist_id=specialist_id,
                domain_hint=domain_hint,
                chain_depth=chain_depth,
                runtime_status=runtime_status,
                execution_us=(time.perf_counter_ns() - started_ns) // 1_000,
                error=exc,
                chain_tool_ids=plan_chain_tool_ids,
                chain_runtime_statuses=plan_chain_statuses,
            )
            cls._record_execution_event(knowledgeverse=knowledgeverse, event=event)
            raise
        event = build_execution_event(
            execution_plan=execution_plan,
            tool_id=tool_id or "unknown_tool",
            query_context=str(query_context or ""),
            specialist_id=specialist_id,
            domain_hint=domain_hint,
            chain_depth=chain_depth,
            runtime_status=runtime_status,
            execution_us=(time.perf_counter_ns() - started_ns) // 1_000,
            result=result,
            chain_tool_ids=plan_chain_tool_ids,
            chain_runtime_statuses=plan_chain_statuses,
        )
        cls._record_execution_event(knowledgeverse=knowledgeverse, event=event)
        return attach_execution_event(result, event)

    @classmethod
    def resolve_plan_blueprint(cls, execution_plan: dict[str, Any] | None) -> dict[str, Any] | None:
        if not execution_plan:
            return None
        chain = []
        for row in execution_plan.get("execution_chain", []) or []:
            entrypoints = [
                cls.resolve_entrypoint_blueprint(str(item).strip())
                for item in row.get("entrypoints", [])
                if str(item).strip()
            ]
            argument_schemas = (
                row.get("argument_schemas") if isinstance(row.get("argument_schemas"), dict) else {}
            )
            raw_chain_presets = row.get("chain_presets") if isinstance(row.get("chain_presets"), dict) else {}
            chain_presets: dict[str, Any] = {}
            for preset_name, preset in raw_chain_presets.items():
                if not isinstance(preset, dict):
                    continue
                steps: list[dict[str, Any]] = []
                for step in preset.get("steps", []) or []:
                    if not isinstance(step, dict):
                        continue
                    entrypoint_token = str(step.get("entrypoint", "")).strip()
                    if not entrypoint_token:
                        continue
                    steps.append(
                        {
                            "entrypoint": cls.resolve_entrypoint_blueprint(entrypoint_token),
                            "argument_schema": dict(step.get("argument_schema", {}) or {}),
                            "store_as": list(step.get("store_as", []) or []),
                            "store_fields": dict(step.get("store_fields", {}) or {}),
                        }
                    )
                chain_presets[str(preset_name).strip()] = {
                    "required_inputs": list(preset.get("required_inputs", []) or []),
                    "return_alias": str(preset.get("return_alias", "")).strip(),
                    "selectors": dict(preset.get("selectors", {}) or {}),
                    "steps": steps,
                }
            chain.append(
                {
                    "tool_id": str(row.get("tool_id", "")).strip(),
                    "tool_kind": str(row.get("tool_kind", "")).strip(),
                    "runtime_status": str(row.get("runtime_status", "")).strip(),
                    "inputs": list(row.get("inputs", []) or []),
                    "outputs": list(row.get("outputs", []) or []),
                    "entrypoints": entrypoints,
                    "argument_schemas": {
                        str(key).strip(): dict(value)
                        for key, value in argument_schemas.items()
                        if str(key).strip() and isinstance(value, dict)
                    },
                    "chain_presets": chain_presets,
                }
            )
        primary = None
        primary_token = str(execution_plan.get("primary_entrypoint", "")).strip()
        if primary_token:
            primary = cls.resolve_entrypoint_blueprint(primary_token)
        primary_argument_schema = execution_plan.get("primary_argument_schema")
        if not isinstance(primary_argument_schema, dict):
            primary_argument_schema = None
            for row in chain:
                row_schemas = row.get("argument_schemas", {})
                if primary_token and primary_token in row_schemas:
                    primary_argument_schema = dict(row_schemas[primary_token])
                    break
        raw_top_chain_presets = execution_plan.get("chain_presets") if isinstance(execution_plan.get("chain_presets"), dict) else {}
        top_chain_presets: dict[str, Any] = {}
        for preset_name, preset in raw_top_chain_presets.items():
            if not isinstance(preset, dict):
                continue
            steps: list[dict[str, Any]] = []
            for step in preset.get("steps", []) or []:
                if not isinstance(step, dict):
                    continue
                entrypoint_token = str(step.get("entrypoint", "")).strip()
                if not entrypoint_token:
                    continue
                steps.append(
                    {
                        "entrypoint": cls.resolve_entrypoint_blueprint(entrypoint_token),
                        "argument_schema": dict(step.get("argument_schema", {}) or {}),
                        "store_as": list(step.get("store_as", []) or []),
                        "store_fields": dict(step.get("store_fields", {}) or {}),
                    }
                )
            top_chain_presets[str(preset_name).strip()] = {
                "required_inputs": list(preset.get("required_inputs", []) or []),
                "return_alias": str(preset.get("return_alias", "")).strip(),
                "selectors": dict(preset.get("selectors", {}) or {}),
                "steps": steps,
            }
        return {
            "mode": str(execution_plan.get("mode", "")).strip(),
            "primary_tool_id": str(execution_plan.get("primary_tool_id", "")).strip(),
            "primary_entrypoint": primary,
            "primary_argument_schema": primary_argument_schema,
            "primary_inputs": list(execution_plan.get("primary_inputs", []) or []),
            "primary_outputs": list(execution_plan.get("primary_outputs", []) or []),
            "executable_tool_ids": list(execution_plan.get("executable_tool_ids", []) or []),
            "inputs": list(execution_plan.get("inputs", []) or []),
            "outputs": list(execution_plan.get("outputs", []) or []),
            "chain_presets": top_chain_presets,
            "chain": chain,
        }


__all__ = ["EntrypointBlueprint", "ToolExecutionResolver"]
