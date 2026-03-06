"""Auto-detect execution grammars from recurring Tool chains.

This is the first self-observation path that creates Grammar Galaxy entries from
runtime behavior instead of human-authored rules or offline ingestion.
It now supports both:
- positive recurring chains (successful execution grammars)
- contrastive recurring chains (failing anti-pattern grammars)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .execution_quality_tracker import ExecutionQualityTracker
from .execution_events import ternary_quantize_quality

_QUERY_STOPWORDS = {
    "a",
    "all",
    "am",
    "an",
    "and",
    "as",
    "at",
    "for",
    "from",
    "i",
    "in",
    "into",
    "is",
    "learning",
    "material",
    "my",
    "of",
    "on",
    "projection",
    "scene",
    "surface",
    "the",
    "to",
    "what",
    "with",
}
_QUERY_ALLOW_SHORT = {"3d", "ai", "ui", "ptx", "rpn"}
_FAMILY_STOPWORDS = {
    "available",
    "bridge",
    "entrypoint",
    "fusion",
    "knowledge3d",
    "material",
    "projection",
    "ptx",
    "runtime",
    "surface",
    "tool",
    "v1",
}


class ExecutionGrammarDetector:
    def __init__(
        self,
        *,
        storage_root: str | Path,
        galaxy_manager: Any,
        min_occurrences: int = 3,
    ):
        self.storage_root = Path(storage_root)
        self.galaxy_manager = galaxy_manager
        self.min_occurrences = max(2, int(min_occurrences))
        self.state_path = self.storage_root / "checkpoints" / "execution_grammar_detector.json"
        self.log_path = self.storage_root / "logs" / "execution_grammar_patterns.jsonl"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._state: dict[str, Any] = {
            "patterns": {},
            "promoted_rules": {},
        }
        self._tool_entry_index: dict[str, dict[str, Any]] | None = None
        self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        self._state["patterns"] = dict(payload.get("patterns", {}) or {})
        self._state["promoted_rules"] = dict(payload.get("promoted_rules", {}) or {})

    def _save(self) -> None:
        self.state_path.write_text(
            json.dumps(self._state, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _append_log(self, payload: Mapping[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), separators=(",", ":"), sort_keys=True) + "\n")

    @staticmethod
    def _dedupe(values: list[str] | tuple[str, ...]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            token = str(value).strip()
            if not token or token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text or ""))
        return [
            token
            for token in re.split(r"[^a-z0-9]+", expanded.lower())
            if token
        ]

    @classmethod
    def _normalize_query_tokens(cls, query: str) -> tuple[str, ...]:
        tokens: list[str] = []
        seen: set[str] = set()
        for token in cls._tokenize(query):
            if token in seen:
                continue
            if token in _QUERY_STOPWORDS:
                continue
            if len(token) < 3 and token not in _QUERY_ALLOW_SHORT:
                continue
            seen.add(token)
            tokens.append(token)
        return tuple(tokens[:6])

    @classmethod
    def _tool_family_tokens(cls, tool_id: str, tool_kind: str) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for token in cls._tokenize(tool_kind) + cls._tokenize(tool_id):
            if token in _FAMILY_STOPWORDS or token.isdigit():
                continue
            if len(token) < 3 and token not in _QUERY_ALLOW_SHORT:
                continue
            if token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out

    @staticmethod
    def _counted_tokens(record: Mapping[str, Any], field: str, *, limit: int = 4) -> list[str]:
        counts = record.get(field, {})
        if not isinstance(counts, Mapping):
            return []
        ranked = sorted(
            (
                (str(key).strip(), int(value or 0))
                for key, value in counts.items()
                if str(key).strip()
            ),
            key=lambda item: (-item[1], item[0]),
        )
        return [token for token, _ in ranked[:limit]]

    @staticmethod
    def _increment_counts(record: dict[str, Any], field: str, values: tuple[str, ...] | list[str]) -> None:
        counts = dict(record.get(field, {}) or {})
        for value in values:
            token = str(value).strip()
            if not token:
                continue
            counts[token] = int(counts.get(token, 0) or 0) + 1
        record[field] = counts

    def _tool_index(self) -> dict[str, dict[str, Any]]:
        if self._tool_entry_index is not None:
            return self._tool_entry_index
        index: dict[str, dict[str, Any]] = {}
        try:
            galaxy = self.galaxy_manager.get_galaxy("Tool")
        except Exception:
            self._tool_entry_index = index
            return index
        for entry in getattr(galaxy, "entries", []) or []:
            if not isinstance(entry, dict):
                continue
            tool_id = str(entry.get("id", "")).strip()
            if tool_id:
                index[tool_id] = entry
        self._tool_entry_index = index
        return index

    @staticmethod
    def _normalize_chain(event: Mapping[str, Any]) -> tuple[str, ...]:
        chain = [
            str(value).strip()
            for value in event.get("chain_tool_ids", [])
            if str(value).strip()
        ]
        if chain:
            return tuple(chain)
        tool_id = str(event.get("tool_id", "")).strip()
        if tool_id:
            return (tool_id,)
        return ()

    @staticmethod
    def _subsequences(chain: tuple[str, ...]) -> list[tuple[str, ...]]:
        if len(chain) < 2:
            return []
        windows: list[tuple[str, ...]] = []
        max_size = min(len(chain), 4)
        for size in range(2, max_size + 1):
            for idx in range(0, len(chain) - size + 1):
                windows.append(tuple(chain[idx:idx + size]))
        # Keep the longest sequence first when later sorted by occurrence.
        windows.sort(key=lambda row: (-len(row), row))
        deduped: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        for row in windows:
            if row in seen:
                continue
            deduped.append(row)
            seen.add(row)
        return deduped

    @staticmethod
    def _pattern_key(sequence: tuple[str, ...], *, polarity: str) -> str:
        payload = f"{polarity}::" + "::".join(sequence)
        return f"exec_chain_{polarity}_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _rpn_program(sequence: tuple[str, ...], *, polarity: str) -> str:
        suffix = "TOOL_CHAIN" if polarity == "positive" else "TOOL_CHAIN_ANTI"
        return " ".join(sequence) + f" {suffix}"

    @staticmethod
    def _multimodal_pattern_key(signature: Mapping[str, Any], *, polarity: str) -> str:
        payload = json.dumps(
            {
                "polarity": str(polarity),
                "families": list(signature.get("tool_family_signature", [])),
                "modalities": list(signature.get("modality_signature", [])),
                "routes": list(signature.get("route_signature", [])),
                "tokens": list(signature.get("query_signature", [])),
                "domain": str(signature.get("domain_hint", "multimodal")),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        return "exec_multimodal_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _multimodal_rpn_program(signature: Mapping[str, Any], *, polarity: str) -> str:
        families = [str(token).strip() for token in signature.get("tool_family_signature", []) if str(token).strip()]
        modalities = [str(token).strip().upper() for token in signature.get("modality_signature", []) if str(token).strip()]
        routes = [str(token).strip().upper() for token in signature.get("route_signature", []) if str(token).strip()]
        sequence = families + modalities + routes
        suffix = "MULTIMODAL_GRAMMAR" if polarity == "positive" else "MULTIMODAL_GRAMMAR_ANTI"
        return " ".join(sequence or ["MULTIMODAL"]) + f" {suffix}"

    def _build_rule_entry(
        self,
        pattern_key: str,
        sequence: tuple[str, ...],
        record: Mapping[str, Any],
        *,
        polarity: str,
    ) -> dict[str, Any]:
        count = int(record.get("count", 0) or 0)
        quality_sum = float(record.get("quality_sum", 0.0) or 0.0)
        avg_quality = quality_sum / float(max(1, count))
        bayesian_quality = float(count + 1) / float(count + 2)
        dominant_domain = str(record.get("dominant_domain_hint", "multimodal") or "multimodal")
        query_examples = list(record.get("query_examples", []) or [])[:3]
        is_positive = str(polarity).strip().lower() == "positive"
        pattern = "tool_chain_positive" if is_positive else "tool_chain_negative"
        source = "auto_detected" if is_positive else "auto_detected_contrastive"
        description = (
            f"Auto-detected execution grammar from {count} successful recurrences of "
            f"{' -> '.join(sequence)}"
            if is_positive else
            f"Auto-detected contrastive anti-pattern from {count} failing recurrences of "
            f"{' -> '.join(sequence)}"
        )
        return {
            "id": pattern_key,
            "rule_id": pattern_key,
            "language": "execution",
            "pattern": pattern,
            "rpn_program": self._rpn_program(sequence, polarity=polarity),
            "domain": dominant_domain,
            "description": description,
            "semantics": {
                "source": source,
                "pattern_type": (
                    "execution_tool_chain"
                    if is_positive else
                    "execution_tool_chain_antipattern"
                ),
                "chain_tool_ids": list(sequence),
                "occurrence_count": count,
                "success_count": count if is_positive else 0,
                "failure_count": 0 if is_positive else count,
                "avg_quality_signal": float(avg_quality),
                "bayesian_quality": float(bayesian_quality),
                "ternary_confidence": int(ternary_quantize_quality(bayesian_quality)),
                "query_examples": query_examples,
                "contrastive_recommendation": (
                    "reuse_and_promote"
                    if is_positive else
                    "avoid_or_invert"
                ),
            },
            "usage_conditions": [
                f"domain_hint:{dominant_domain}",
                ("outcome:+1" if is_positive else "outcome:-1"),
                f"min_occurrences:{self.min_occurrences}",
            ],
            "is_canonical": False,
        }

    def _accumulate_pattern(
        self,
        sequence: tuple[str, ...],
        event: Mapping[str, Any],
        *,
        polarity: str,
    ) -> tuple[str, dict[str, Any]]:
        key = self._pattern_key(sequence, polarity=polarity)
        patterns = self._state.setdefault("patterns", {})
        record = dict(patterns.get(key, {}) or {})
        count = int(record.get("count", 0) or 0) + 1
        quality = float(event.get("quality_signal", 0.0) or 0.0)
        query = str(event.get("query_context", "") or "").strip()
        domain_hint = str(event.get("domain_hint", "") or "multimodal").strip() or "multimodal"
        query_examples = [str(value).strip() for value in record.get("query_examples", []) if str(value).strip()]
        if query and query not in query_examples:
            query_examples.append(query)
        record.update(
            {
                "pattern_key": key,
                "sequence": list(sequence),
                "count": count,
                "quality_sum": float(record.get("quality_sum", 0.0) or 0.0) + quality,
                "last_quality_signal": quality,
                "last_timestamp_us": int(event.get("timestamp_us", 0) or 0),
                "dominant_domain_hint": domain_hint,
                "polarity": str(polarity),
                "query_examples": query_examples[-5:],
            }
        )
        patterns[key] = record
        return key, record

    def _infer_multimodal_signature(
        self,
        event: Mapping[str, Any],
        chain: tuple[str, ...],
    ) -> dict[str, Any] | None:
        tool_index = self._tool_index()
        chain_statuses = [
            str(value).strip()
            for value in event.get("chain_runtime_statuses", [])
            if str(value).strip()
        ]
        query_tokens = self._normalize_query_tokens(str(event.get("query_context", "") or ""))
        family_tokens: list[str] = []
        modalities: list[str] = []
        route_sources: list[str] = []
        tool_kinds: list[str] = []
        for idx, tool_id in enumerate(chain):
            entry = tool_index.get(str(tool_id).strip(), {})
            metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            tool_kind = str(metadata.get("tool_kind", "")).strip()
            if tool_kind:
                tool_kinds.append(tool_kind)
            family_tokens.extend(self._tool_family_tokens(str(tool_id), tool_kind))
            modalities.extend(str(mod).strip() for mod in metadata.get("modalities", []) if str(mod).strip())
            runtime_status = ""
            if idx < len(chain_statuses):
                runtime_status = chain_statuses[idx]
            if not runtime_status:
                runtime_status = str(metadata.get("runtime_status", "")).strip() or str(event.get("runtime_status", "")).strip()
            route_sources.append(
                ExecutionQualityTracker.classify_route_source(runtime_status, tool_kind)
            )

        deduped_modalities = tuple(sorted(self._dedupe(modalities)))
        deduped_routes = tuple(sorted(self._dedupe(route_sources)))
        deduped_families = tuple(sorted(self._dedupe(family_tokens))[:6])
        if len(deduped_modalities) < 2 and len(deduped_families) < 2:
            return None
        if not query_tokens:
            return None
        return {
            "tool_family_signature": deduped_families,
            "modality_signature": deduped_modalities,
            "route_signature": deduped_routes,
            "query_signature": tuple(sorted(self._dedupe(list(query_tokens))[:4])),
            "tool_kind_signature": tuple(sorted(self._dedupe(tool_kinds))),
            "domain_hint": str(event.get("domain_hint", "") or "multimodal").strip() or "multimodal",
            "chain_example": list(chain),
        }

    def _accumulate_multimodal_pattern(
        self,
        signature: Mapping[str, Any],
        event: Mapping[str, Any],
        *,
        polarity: str,
    ) -> tuple[str, dict[str, Any]]:
        key = self._multimodal_pattern_key(signature, polarity=polarity)
        patterns = self._state.setdefault("patterns", {})
        record = dict(patterns.get(key, {}) or {})
        count = int(record.get("count", 0) or 0) + 1
        quality = float(event.get("quality_signal", 0.0) or 0.0)
        query = str(event.get("query_context", "") or "").strip()
        query_examples = [
            str(value).strip()
            for value in record.get("query_examples", [])
            if str(value).strip()
        ]
        if query and query not in query_examples:
            query_examples.append(query)
        chain_examples = [
            list(value)
            for value in record.get("chain_examples", [])
            if isinstance(value, list) and value
        ]
        chain_example = [
            str(value).strip()
            for value in signature.get("chain_example", [])
            if str(value).strip()
        ]
        if chain_example and chain_example not in chain_examples:
            chain_examples.append(chain_example)
        record.update(
            {
                "pattern_key": key,
                "pattern_class": "multimodal",
                "count": count,
                "quality_sum": float(record.get("quality_sum", 0.0) or 0.0) + quality,
                "last_quality_signal": quality,
                "last_timestamp_us": int(event.get("timestamp_us", 0) or 0),
                "dominant_domain_hint": str(signature.get("domain_hint", "multimodal")),
                "polarity": str(polarity),
                "query_examples": query_examples[-5:],
                "chain_examples": chain_examples[-4:],
                "tool_kind_signature": list(signature.get("tool_kind_signature", [])),
                "tool_family_signature": list(signature.get("tool_family_signature", [])),
                "modality_signature": list(signature.get("modality_signature", [])),
                "route_signature": list(signature.get("route_signature", [])),
                "query_signature": list(signature.get("query_signature", [])),
            }
        )
        self._increment_counts(record, "query_token_counts", list(signature.get("query_signature", [])))
        self._increment_counts(record, "tool_family_counts", list(signature.get("tool_family_signature", [])))
        self._increment_counts(record, "modality_counts", list(signature.get("modality_signature", [])))
        self._increment_counts(record, "route_source_counts", list(signature.get("route_signature", [])))
        patterns[key] = record
        return key, record

    def _build_multimodal_rule_entry(
        self,
        pattern_key: str,
        record: Mapping[str, Any],
        *,
        polarity: str,
    ) -> dict[str, Any]:
        count = int(record.get("count", 0) or 0)
        quality_sum = float(record.get("quality_sum", 0.0) or 0.0)
        avg_quality = quality_sum / float(max(1, count))
        bayesian_quality = float(count + 1) / float(count + 2)
        dominant_domain = str(record.get("dominant_domain_hint", "multimodal") or "multimodal")
        query_examples = list(record.get("query_examples", []) or [])[:3]
        chain_examples = list(record.get("chain_examples", []) or [])[:3]
        family_tokens = self._counted_tokens(record, "tool_family_counts")
        query_tokens = self._counted_tokens(record, "query_token_counts")
        modalities = self._counted_tokens(record, "modality_counts", limit=6)
        route_sources = self._counted_tokens(record, "route_source_counts", limit=3)
        is_positive = str(polarity).strip().lower() == "positive"
        pattern = "multimodal_execution_positive" if is_positive else "multimodal_execution_negative"
        source = "auto_detected_multimodal" if is_positive else "auto_detected_multimodal_contrastive"
        description = (
            f"Auto-detected multimodal execution grammar from {count} recurring successful observations "
            f"covering {', '.join(modalities or ['multimodal'])}"
            if is_positive else
            f"Auto-detected multimodal contrastive anti-pattern from {count} recurring failed observations "
            f"covering {', '.join(modalities or ['multimodal'])}"
        )
        usage_conditions = [
            f"domain_hint:{dominant_domain}",
            ("outcome:+1" if is_positive else "outcome:-1"),
            f"min_occurrences:{self.min_occurrences}",
        ]
        usage_conditions.extend(f"modality:{token}" for token in modalities[:4])
        usage_conditions.extend(f"route_source:{token}" for token in route_sources[:2])
        usage_conditions.extend(f"query_token:{token}" for token in query_tokens[:3])
        return {
            "id": pattern_key,
            "rule_id": pattern_key,
            "language": "execution",
            "pattern": pattern,
            "rpn_program": self._multimodal_rpn_program(record, polarity=polarity),
            "domain": dominant_domain,
            "description": description,
            "semantics": {
                "source": source,
                "pattern_type": (
                    "execution_multimodal_pattern"
                    if is_positive else
                    "execution_multimodal_antipattern"
                ),
                "occurrence_count": count,
                "success_count": count if is_positive else 0,
                "failure_count": 0 if is_positive else count,
                "avg_quality_signal": float(avg_quality),
                "bayesian_quality": float(bayesian_quality),
                "ternary_confidence": int(ternary_quantize_quality(bayesian_quality)),
                "tool_kind_signature": list(record.get("tool_kind_signature", []) or []),
                "tool_family_tokens": family_tokens,
                "modalities": modalities,
                "route_sources": route_sources,
                "stable_query_tokens": query_tokens,
                "query_examples": query_examples,
                "chain_examples": chain_examples,
                "contrastive_recommendation": (
                    "reuse_and_generalize"
                    if is_positive else
                    "avoid_or_invert"
                ),
            },
            "usage_conditions": usage_conditions,
            "is_canonical": False,
        }

    def observe_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        outcome = int(event.get("outcome", 0) or 0)
        if outcome == 0:
            return {}
        chain = self._normalize_chain(event)
        if len(chain) < 2:
            return {}
        polarity = "positive" if outcome > 0 else "negative"

        promoted_rules: list[str] = []
        updated_patterns: list[str] = []
        promoted_multimodal_rules: list[str] = []
        updated_multimodal_patterns: list[str] = []
        for sequence in self._subsequences(chain):
            key, record = self._accumulate_pattern(sequence, event, polarity=polarity)
            updated_patterns.append(key)
            if int(record.get("count", 0) or 0) < self.min_occurrences:
                continue
            if key in self._state.setdefault("promoted_rules", {}):
                continue
            entry = self._build_rule_entry(key, sequence, record, polarity=polarity)
            self.galaxy_manager.add_entry("Grammar", entry)
            self._state["promoted_rules"][key] = {
                "rule_id": key,
                "sequence": list(sequence),
                "polarity": polarity,
                "promoted_at": int(event.get("timestamp_us", 0) or 0),
            }
            self._append_log(
                {
                    "event": (
                        "execution_grammar_promoted"
                        if polarity == "positive" else
                        "execution_antipattern_promoted"
                    ),
                    "rule_id": key,
                    "polarity": polarity,
                    "sequence": list(sequence),
                    "count": int(record.get("count", 0) or 0),
                    "avg_quality_signal": float(record.get("quality_sum", 0.0) or 0.0) / float(max(1, int(record.get("count", 0) or 0))),
                }
            )
            promoted_rules.append(key)

        multimodal_signature = self._infer_multimodal_signature(event, chain)
        if multimodal_signature is not None:
            key, record = self._accumulate_multimodal_pattern(
                multimodal_signature,
                event,
                polarity=polarity,
            )
            updated_multimodal_patterns.append(key)
            if (
                int(record.get("count", 0) or 0) >= self.min_occurrences
                and key not in self._state.setdefault("promoted_rules", {})
            ):
                entry = self._build_multimodal_rule_entry(key, record, polarity=polarity)
                self.galaxy_manager.add_entry("Grammar", entry)
                self._state["promoted_rules"][key] = {
                    "rule_id": key,
                    "pattern_class": "multimodal",
                    "polarity": polarity,
                    "promoted_at": int(event.get("timestamp_us", 0) or 0),
                }
                self._append_log(
                    {
                        "event": (
                            "execution_multimodal_grammar_promoted"
                            if polarity == "positive" else
                            "execution_multimodal_antipattern_promoted"
                        ),
                        "rule_id": key,
                        "polarity": polarity,
                        "modalities": list(record.get("modality_signature", []) or []),
                        "route_sources": list(record.get("route_signature", []) or []),
                        "tool_family_tokens": self._counted_tokens(record, "tool_family_counts"),
                        "query_tokens": self._counted_tokens(record, "query_token_counts"),
                        "count": int(record.get("count", 0) or 0),
                        "avg_quality_signal": float(record.get("quality_sum", 0.0) or 0.0) / float(max(1, int(record.get("count", 0) or 0))),
                    }
                )
                promoted_multimodal_rules.append(key)

        self._save()
        return {
            "updated_patterns": updated_patterns,
            "promoted_rules": promoted_rules,
            "updated_multimodal_patterns": updated_multimodal_patterns,
            "promoted_multimodal_rules": promoted_multimodal_rules,
        }

    def harvest_from_event_log(self, event_log_path: str | Path | None = None) -> dict[str, Any]:
        path = Path(event_log_path) if event_log_path is not None else (self.storage_root / "logs" / "execution_events.jsonl")
        if not path.exists():
            return {"harvested": 0, "promoted_rules": []}

        harvested = 0
        promoted: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                summary = self.observe_event(payload)
                harvested += 1
                for rule_id in summary.get("promoted_rules", []) or []:
                    token = str(rule_id).strip()
                    if token and token not in promoted:
                        promoted.append(token)
        return {
            "harvested": harvested,
            "promoted_rules": promoted,
        }


__all__ = ["ExecutionGrammarDetector"]
