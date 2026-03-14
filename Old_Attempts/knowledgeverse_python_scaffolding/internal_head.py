"""Internal Knowledgeverse runtime head for structured benchmark execution.

This module centralizes benchmark-facing reasoning flows inside the
Knowledgeverse runtime instead of leaving them inside the daemon transport
layer. The current implementation focuses on the LHE structured path while
preserving the existing swarm/meaning-first reasoning stack.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .foundational_operations_bootstrap import search_foundational_reasoning_entries
from .lhe_reasoning_swarm import LHEReasoningSwarm, LHEWorkerHelpers
from .meaning_first_reasoning import (
    fuse_meaning_atoms,
    meaning_atoms_from_evidence_rows,
    meaning_atoms_from_parse_entities,
)


class KnowledgeverseInternalHead:
    """Runtime head that executes structured benchmark flows inside Knowledgeverse."""

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

    _LHE_SEMANTIC_EXPANSIONS = {
        "irony": {"incongruity", "opposite", "contrast", "unexpected"},
        "ironic": {"irony", "contrast", "incongruity"},
        "sarcasm": {"irony", "mockery", "ridicule"},
        "metaphor": {"figurative", "analogy", "comparison"},
        "simile": {"figurative", "comparison", "like", "as"},
        "hyperbole": {"exaggeration", "overstatement"},
        "understatement": {"minimization", "downplay"},
        "pun": {"wordplay", "double", "meaning"},
        "paradox": {"contradiction", "tension"},
        "oxymoron": {"contradiction", "opposites"},
        "allusion": {"reference", "indirect"},
        "euphemism": {"indirect", "softened"},
        "personification": {"human_trait", "figurative"},
        "sadism": {"harm", "cruelty", "suffering"},
        "nonsadism": {"avoid_harm", "anti_cruelty", "harm", "cruelty"},
        "cruelty": {"harm", "suffering"},
        "egalitarian": {"equality", "equal"},
        "elitism": {"elite", "priority_best"},
        "priority": {"precedence", "weight"},
        "quality": {"welfare", "value"},
        "addition": {"increase", "add"},
        "cipher": {"substitution", "decode", "decrypt"},
        "notation": {"symbolic", "formal"},
        "mate": {"checkmate", "winning_sequence"},
        "chess": {"board", "move", "checkmate"},
        "trivia": {"fact", "reference"},
    }

    def __init__(
        self,
        *,
        knowledgeverse: Any,
        trm: Any | None = None,
        storage_root: str | Path | None = None,
    ) -> None:
        self.knowledgeverse = knowledgeverse
        self.trm = trm or getattr(knowledgeverse, "trm_navigator", None)
        if self.trm is None:
            raise RuntimeError("KnowledgeverseInternalHead requires a TRM navigator")
        root = storage_root
        if root is None:
            root = getattr(knowledgeverse, "storage_root", "../Knowledge3D.local")
        self.storage_root = Path(root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self._repo_root = Path(__file__).resolve().parents[2]
        self._lhe_snapshot_path: Path | None = None
        self.lhe_reasoning_swarm: LHEReasoningSwarm | None = None
        helpers = LHEWorkerHelpers(
            tokenize=self._tokenize_lhe_text,
            semanticize=self._semanticize_lhe_tokens,
            normalize_answer=self._normalize_answer_text,
            extract_candidates=self._extract_lhe_open_candidates,
            is_meta_candidate=self._is_lhe_meta_candidate,
            is_code_like_candidate=self._is_lhe_code_like_candidate,
            canonicalize_short_numeric=self._canonicalize_lhe_short_numeric_candidate,
            resolve_snapshot_path=self._resolve_lhe_snapshot_path,
            query_evidence=self._query_lhe_evidence,
            rpn_batch_eval=self._rpn_batch_eval,
        )
        self.lhe_reasoning_swarm = LHEReasoningSwarm(
            storage_dir=self.storage_root / "checkpoints" / "lhe_reasoning_swarm",
            helpers=helpers,
        )

    def _rpn_batch_eval(self, expressions: list[str]) -> tuple[list[float], int]:
        if self.lhe_reasoning_swarm is None:
            return [0.0 for _ in expressions], 0
        return self.lhe_reasoning_swarm._evaluate_condition_batch(expressions)

    def execute_packet(self, packet: dict[str, Any]) -> dict[str, Any]:
        prompt = str(
            packet.get("prompt", "")
            or packet.get("query", "")
            or ((packet.get("task") or {}).get("prompt") if isinstance(packet.get("task"), dict) else "")
            or ((packet.get("task") or {}).get("query") if isinstance(packet.get("task"), dict) else "")
        ).strip()
        task = packet.get("task") if isinstance(packet.get("task"), dict) else {}
        task_type = str(task.get("type") or packet.get("query_type") or "").upper()
        specialist = str(packet.get("specialist", "auto"))
        domain_hint = packet.get("domain_hint") or task.get("domain_hint")
        route = packet.get("route") if isinstance(packet.get("route"), dict) else None
        use_enriched = bool(packet.get("use_enriched", True))

        if task_type in {"LHE_TASK", "LHE_OPEN"} or task.get("options") or str(task.get("question_type", "")).lower() == "open_ended":
            task_payload = dict(task)
            if prompt and not str(task_payload.get("prompt", "") or task_payload.get("query", "")).strip():
                task_payload["prompt"] = prompt
            task_payload.setdefault("type", "LHE_TASK")
            if route is None:
                route = self.trm.route(
                    query=prompt,
                    specialist=specialist,
                    domain_hint=domain_hint,
                    galaxy_names=packet.get("galaxies") or task_payload.get("galaxies"),
                )
            result = self.dispatch_lhe_task(route=route, task=task_payload, use_enriched=use_enriched)
            result.setdefault("internal_head", "knowledgeverse")
            result.setdefault("query_id", packet.get("query_id"))
            return result

        if route is None:
            route = self.trm.route(
                query=prompt,
                specialist=specialist,
                domain_hint=domain_hint,
                galaxy_names=packet.get("galaxies"),
            )
        composed = self.trm.navigate_and_compose(
            query=prompt,
            specialist=str(route.get("specialist", specialist)),
            domain_hint=str(route.get("domain", domain_hint or "")),
            use_enriched=use_enriched,
        )
        if isinstance(composed, dict):
            composed.setdefault("route", route)
            composed.setdefault("internal_head", "knowledgeverse")
            composed.setdefault("query_id", packet.get("query_id"))
        return composed

    def _collect_parse_bundle(
        self,
        query: str,
        *,
        specialist: str,
        galaxy_names: list[str],
        domain_hint: str | None = None,
    ) -> dict[str, Any]:
        navigator = getattr(self.knowledgeverse, "navigator_specialist", None)
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

    def _normalize_lhe_domain_hint(self, domain_hint: str | None) -> str:
        domain = str(domain_hint or "multi").strip().lower()
        aliases = {
            "mathematics": "math",
            "math": "math",
            "physics": "physics",
            "chemistry": "chemistry",
            "biology": "biology",
            "philosophy": "philosophy",
            "trivia": "trivia",
            "cybersecurity": "cybersecurity",
            "history": "history",
            "chess": "chess",
            "multi": "multi",
        }
        return aliases.get(domain, domain or "multi")

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

    def _semanticize_lhe_tokens(self, text: str, *, preserve_single: bool = False) -> set[str]:
        out: set[str] = set()
        for token in self._tokenize_lhe_text(text, preserve_single=preserve_single):
            cleaned = token.replace("-", "").replace("_", "")
            if not cleaned:
                continue
            out.add(cleaned)
            if cleaned.startswith("non") and len(cleaned) > 5:
                root = cleaned[3:]
                out.add(root)
                out.add(f"anti_{root}")
            for extra in self._LHE_SEMANTIC_EXPANSIONS.get(cleaned, set()):
                out.add(str(extra))
        return out

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
        prompt: str,
        parse_bundle: dict[str, Any],
        options: list[str] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        prompt_tokens = self._semanticize_lhe_tokens(prompt, preserve_single=bool(options))

        def _is_prompt_aligned(text: str) -> bool:
            candidate = str(text or "").strip()
            if not candidate:
                return False
            if not prompt_tokens:
                return True
            candidate_tokens = self._semanticize_lhe_tokens(candidate, preserve_single=bool(options))
            return bool(prompt_tokens & candidate_tokens)

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
            if not _is_prompt_aligned(raw):
                continue
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
            goal_text = str(goal.get("raw") or goal.get("expression") or "")
            if _is_prompt_aligned(goal_text):
                self._append_lhe_text_entities(
                    forward_entities,
                    text=goal_text,
                    role="goal",
                    source_pass="forward",
                    confidence=0.7,
                )

        for idx, block in enumerate(backward_parse.get("dependencies", []) or []):
            if not isinstance(block, dict):
                continue
            raw = str(block.get("raw", "")).strip()
            if not _is_prompt_aligned(raw):
                continue
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
            goal_text = str(goal.get("raw") or goal.get("expression") or "")
            if _is_prompt_aligned(goal_text):
                self._append_lhe_text_entities(
                    backward_entities,
                    text=goal_text,
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
            goal_text = str(unified_goal.get("raw") or unified_goal.get("expression") or "")
            if _is_prompt_aligned(goal_text):
                self._append_lhe_text_entities(
                    fused_entities,
                    text=goal_text,
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
        if raw_goal:
            prompt_tokens = self._semanticize_lhe_tokens(prompt, preserve_single=bool(options))
            goal_tokens = self._semanticize_lhe_tokens(raw_goal, preserve_single=bool(options))
            overlap = len(prompt_tokens & goal_tokens)
            if overlap <= 0:
                raw_goal = ""
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
        names: list[str] = []
        seen: set[str] = set()
        for required in ("Reality", "Word", "Grammar", "Tool"):
            key = required.lower()
            if key in seen:
                continue
            seen.add(key)
            names.append(required)
        for name in [str(item) for item in route.get("galaxy_names") or [] if str(item).strip()]:
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            names.append(name)
        if domain_hint in {"cybersecurity", "cryptography"} and "Character" not in names:
            names.append("Character")
        if domain_hint in {"math", "physics", "chemistry", "biology", "engineering"} and "Math" not in names:
            names.append("Math")
        effective["galaxy_names"] = names
        if domain_hint in {
            "multi",
            "math",
            "physics",
            "chemistry",
            "biology",
            "philosophy",
            "trivia",
            "cybersecurity",
            "history",
            "chess",
        }:
            effective["specialist"] = "auto"
        effective["domain"] = domain_hint
        return effective

    def _extract_lhe_entry(self, row: dict[str, Any]) -> dict[str, Any]:
        direct = row.get("entry")
        if isinstance(direct, dict):
            return direct
        nested = row.get("row")
        if isinstance(nested, dict):
            nested_entry = nested.get("entry")
            if isinstance(nested_entry, dict):
                return nested_entry
        return {}

    def _normalize_lhe_query_row(self, row: dict[str, Any]) -> dict[str, Any]:
        nested = row.get("row")
        if isinstance(nested, dict) and isinstance(nested.get("entry"), dict):
            return nested
        return row

    def _extract_lhe_evidence_fields(self, row: dict[str, Any]) -> dict[str, str]:
        entry = self._extract_lhe_entry(row)
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        entities = metadata.get("entities") if isinstance(metadata.get("entities"), list) else []
        relationships = metadata.get("relationships") if isinstance(metadata.get("relationships"), list) else []

        def _coerce(value: Any) -> str:
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, (list, tuple, set)):
                return " ".join(str(item).strip() for item in value if str(item).strip()).strip()
            if value is None:
                return ""
            return str(value).strip()

        fields = {
            "content": _coerce(entry.get("content", "")),
            "description": _coerce(entry.get("description", "")),
            "summary": _coerce(entry.get("summary", "")),
            "embedding_text": _coerce(metadata.get("embedding_text", "")),
            "notes": _coerce(metadata.get("notes", "")),
            "keywords": _coerce(metadata.get("keywords", "")),
            "tags": _coerce(metadata.get("tags", "")),
            "aliases": _coerce(metadata.get("aliases", "")),
            "rpn_program": _coerce(entry.get("rpn_program", "")),
            "pattern_form": _coerce(entry.get("pattern_form", "")),
            "pattern_type": _coerce(entry.get("pattern_type", "")),
            "semantics": _coerce(metadata.get("semantics", "")),
            "usage_conditions": _coerce(metadata.get("usage_conditions", "")),
            "domain": _coerce(metadata.get("domain", "")),
            "category": _coerce(metadata.get("category", "")),
            "entities": _coerce(
                [
                    f"{item.get('name', '')} {item.get('content', '')}".strip()
                    for item in entities
                    if isinstance(item, dict)
                ]
            ),
            "relationships": _coerce(
                [
                    f"{item.get('from', '')} {item.get('relation', '')} {item.get('to', '')}".strip()
                    for item in relationships
                    if isinstance(item, dict)
                ]
            ),
        }
        return {key: value for key, value in fields.items() if value}

    def _extract_lhe_row_text(self, row: dict[str, Any]) -> str:
        entry = self._extract_lhe_entry(row)
        evidence_fields = self._extract_lhe_evidence_fields(row)
        parts: list[str] = list(evidence_fields.values())
        if not parts:
            parts = [
                str(entry.get("name", "")),
                str(entry.get("title", "")),
            ]
        return " ".join(part.strip() for part in parts if str(part).strip())

    def _has_lhe_semantic_fields(self, fields: dict[str, str]) -> bool:
        semantic_keys = {
            "content",
            "description",
            "summary",
            "embedding_text",
            "notes",
            "keywords",
            "tags",
            "aliases",
            "semantics",
            "usage_conditions",
            "entities",
            "relationships",
        }
        return any(key in fields for key in semantic_keys)

    def _resolve_lhe_snapshot_path(self) -> Path | None:
        if self._lhe_snapshot_path is not None:
            return self._lhe_snapshot_path
        checkpoints_dir = self._repo_root.parent / "Knowledge3D.local" / "fundamental_augmentation" / "checkpoints"
        candidates: list[Path] = []
        if checkpoints_dir.is_dir():
            patterns = (
                "full_pdf_payloads_paused_*.jsonl",
                "full_pdf_payloads_stopped_*.jsonl",
                "full_pdf_payloads_snapshot_*.jsonl",
            )
            for pattern in patterns:
                candidates.extend(checkpoints_dir.glob(pattern))
        if not candidates:
            self._lhe_snapshot_path = None
            return None
        self._lhe_snapshot_path = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]
        return self._lhe_snapshot_path

    def _query_lhe_snapshot_evidence(
        self,
        *,
        prompt: str,
        options: list[str],
    ) -> list[dict[str, Any]]:
        snapshot_path = self._resolve_lhe_snapshot_path()
        if snapshot_path is None or not snapshot_path.is_file():
            return []
        query_tokens = self._semanticize_lhe_tokens(prompt, preserve_single=bool(options))
        if not query_tokens:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        try:
            with snapshot_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        row = json.loads(raw)
                    except Exception:
                        continue
                    if not isinstance(row, dict):
                        continue
                    entry = row.get("entry") if isinstance(row.get("entry"), dict) else {}
                    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
                    category = str(entry.get("category") or metadata.get("category") or "").strip().lower()
                    if category == "pdf_reasoning_bridge":
                        continue
                    fields = self._extract_lhe_evidence_fields(row)
                    if not fields or not self._has_lhe_semantic_fields(fields):
                        continue
                    row_text = self._extract_lhe_row_text(row)
                    if not row_text:
                        continue
                    evidence_tokens = self._semanticize_lhe_tokens(row_text, preserve_single=bool(options))
                    overlap = len(query_tokens & evidence_tokens)
                    if overlap <= 0:
                        continue
                    score = float(overlap)
                    if category and "definition" in category:
                        score += 0.2
                    scored.append(
                        (
                            score,
                            {
                                "row": row,
                                "text": row_text,
                                "tokens": evidence_tokens,
                                "fields": fields,
                                "query": prompt,
                                "rank_weight": min(1.0, 0.25 + (0.08 * score)),
                            },
                        )
                    )
        except OSError:
            return []
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:12]]

    def _query_lhe_evidence(
        self,
        *,
        prompt: str,
        route: dict[str, Any],
        parse_bundle: dict[str, Any],
        use_enriched: bool,
        options: list[str],
    ) -> list[dict[str, Any]]:
        query_tokens = self._semanticize_lhe_tokens(prompt, preserve_single=bool(options))
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
            galaxy_names = planned.get("galaxy_names") or route.get("galaxy_names")
            foundational_rows = search_foundational_reasoning_entries(
                query_text,
                galaxy_names=[str(name) for name in galaxy_names or []],
                limit=12 if use_enriched else 6,
            )
            rows = [
                *foundational_rows,
                *self.trm.query(
                    query=query_text,
                    galaxy_names=galaxy_names,
                    top_k=20 if use_enriched else 8,
                    specialist=str(planned.get("specialist", route.get("specialist", "auto"))),
                    domain_hint=str(planned.get("domain", route.get("domain", "multi"))),
                ),
            ]
            for index, row in enumerate(rows):
                row_text = self._extract_lhe_row_text(row)
                if not row_text:
                    continue
                entry = self._extract_lhe_entry(row)
                fields = self._extract_lhe_evidence_fields(row)
                if not self._has_lhe_semantic_fields(fields):
                    continue
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
                        "row": self._normalize_lhe_query_row(row),
                        "text": row_text,
                        "tokens": self._semanticize_lhe_tokens(row_text, preserve_single=bool(options)),
                        "fields": fields,
                        "query": query_text,
                        "rank_weight": max(0.05, 1.0 - (index * 0.08)),
                    }
                )
        best_overlap = 0
        strong_local_rows = 0
        for evidence_row in evidence:
            evidence_tokens = set(evidence_row.get("tokens", set()) or set())
            overlap = len(query_tokens & evidence_tokens)
            evidence_row["token_overlap"] = overlap
            best_overlap = max(best_overlap, overlap)
            if overlap >= 3:
                strong_local_rows += 1
        needs_snapshot = (not options) and (len(evidence) < 4 or best_overlap < 3 or strong_local_rows < 2)
        if needs_snapshot:
            for evidence_row in self._query_lhe_snapshot_evidence(prompt=prompt, options=options):
                row = evidence_row.get("row", {}) if isinstance(evidence_row.get("row"), dict) else {}
                entry = row.get("entry") if isinstance(row.get("entry"), dict) else {}
                row_text = str(evidence_row.get("text", "")).strip()
                dedupe_key = (
                    str(entry.get("id", "")),
                    str(entry.get("name", "")),
                    row_text[:160],
                )
                if not row_text or dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                evidence_row["token_overlap"] = len(query_tokens & set(evidence_row.get("tokens", set()) or set()))
                evidence.append(evidence_row)
        evidence.sort(
            key=lambda item: (
                int(item.get("token_overlap", 0)),
                float(item.get("rank_weight", 0.0)),
            ),
            reverse=True,
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
        query_text = f"{prompt}\nCandidate answer: {option}"
        rows = [
            *search_foundational_reasoning_entries(
                query_text,
                galaxy_names=[str(name) for name in route.get("galaxy_names") or []],
                limit=8 if use_enriched else 4,
            ),
            *self.trm.query(
                query=query_text,
                galaxy_names=route.get("galaxy_names"),
                top_k=10 if use_enriched else 4,
                specialist=str(route.get("specialist", "auto")),
                domain_hint=str(route.get("domain", "multi")),
            ),
        ]
        out: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            text = self._extract_lhe_row_text(row)
            if not text:
                continue
            fields = self._extract_lhe_evidence_fields(row)
            if not self._has_lhe_semantic_fields(fields):
                continue
            out.append(
                {
                    "row": self._normalize_lhe_query_row(row),
                    "text": text,
                    "tokens": self._semanticize_lhe_tokens(text, preserve_single=True),
                    "fields": fields,
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
        option_tokens = self._semanticize_lhe_tokens(normalized_option, preserve_single=True)
        goal_tokens = self._semanticize_lhe_tokens(" ".join(str(tok) for tok in goal.get("tokens", [])), preserve_single=True)
        fused_tokens = {
            str(entity.get("value", ""))
            for entity in fused_entities
            if str(entity.get("kind", "")) in {"token", "option_token"} and entity.get("value")
        }
        fused_semantic_tokens = self._semanticize_lhe_tokens(" ".join(fused_tokens), preserve_single=True)
        normalized_phrase = self._normalize_answer_text(normalized_option)
        other_options = [
            str(candidate)
            for candidate in options
            if str(candidate).strip() and str(candidate).strip() != normalized_option
        ]
        other_phrases = {
            self._normalize_answer_text(candidate)
            for candidate in other_options
            if self._normalize_answer_text(candidate)
        }
        option_unique = set(option_tokens)
        for other in other_options:
            option_unique -= self._semanticize_lhe_tokens(other, preserve_single=True)
        score = 0.0
        if normalized_option and re.search(rf"\b(?:pick|choose|select)\s+{re.escape(normalized_option.lower())}\b", prompt_lower):
            score += 100.0
        score += float(len(option_tokens & goal_tokens)) * 0.20
        score += float(len(option_tokens & fused_semantic_tokens)) * 0.12
        support = 0.0
        contradiction = 0.0
        contrastive_support = 0.0
        for evidence in [*evidence_rows, *option_rows]:
            fields = evidence.get("fields", {}) if isinstance(evidence.get("fields"), dict) else {}
            normalized_fields = [self._normalize_answer_text(value) for value in fields.values()]
            evidence_tokens = set(evidence.get("tokens", set()) or set())
            own_semantic_hits = len(option_unique & evidence_tokens)
            other_semantic_hits = 0
            for other in other_options:
                other_unique = self._semanticize_lhe_tokens(other, preserve_single=True) - option_tokens
                other_semantic_hits = max(other_semantic_hits, len(other_unique & evidence_tokens))
            exact_hits = sum(1 for value in normalized_fields if normalized_phrase and normalized_phrase in value)
            if exact_hits:
                support += exact_hits * (1.5 + float(evidence.get("rank_weight", 0.0)))
            support += own_semantic_hits * (0.55 + (0.35 * float(evidence.get("rank_weight", 0.0))))
            if normalized_phrase:
                own_mentions = sum(1 for value in normalized_fields if normalized_phrase and normalized_phrase in value)
                other_mentions = sum(
                    1
                    for value in normalized_fields
                    for other in other_phrases
                    if other and other in value
                )
                if own_mentions and not other_mentions:
                    contrastive_support += own_mentions * (1.1 + (0.2 * float(evidence.get("rank_weight", 0.0))))
            if own_semantic_hits and not other_semantic_hits:
                contrastive_support += own_semantic_hits * (0.8 + (0.15 * float(evidence.get("rank_weight", 0.0))))
            for other in other_phrases:
                if other and any(other in value for value in normalized_fields):
                    contradiction += 0.9 + (0.2 * float(evidence.get("rank_weight", 0.0)))
            contradiction += other_semantic_hits * (0.45 + (0.2 * float(evidence.get("rank_weight", 0.0))))
        score += support
        score += contrastive_support
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
        if lowered in generic:
            return True
        if lowered.startswith(("apply ", "chain ", "emit ", "use ", "compute ", "consume ", "rate ", "stack ")):
            return True
        if any(
            phrase in lowered
            for phrase in (
                "benchmark answer",
                "final stack value",
                "multiple benchmark steps",
                "per-unit multiplier",
                "carrying the intermediate stack state",
                "notation examples",
                "eigenvalue equation",
            )
        ):
            return True
        return False

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
        goal_tokens = self._semanticize_lhe_tokens(" ".join(str(tok) for tok in goal.get("tokens", [])), preserve_single=True)
        fused_tokens = {
            str(entity.get("value", ""))
            for entity in fused_entities
            if str(entity.get("kind", "")) in {"token", "phrase"} and entity.get("value")
        }
        fused_semantic_tokens = self._semanticize_lhe_tokens(" ".join(fused_tokens), preserve_single=True)
        goal_raw = str(goal.get("raw", "")).strip()
        goal_lower = goal_raw.lower()
        domain = str(goal.get("domain", "")).strip().lower()
        wants_formula = any(token in goal_raw for token in ("$", "\\(", "\\)", "^", "{", "}")) or any(
            token in goal_lower for token in ("equation", "expression", "formula", "polynomial")
        )
        wants_short_numeric = any(token in goal_lower for token in ("how many", "how much", "what is", "value", "number"))
        wants_chess_notation = any(
            token in goal_lower
            for token in ("chess", "mate", "checkmate", "black to move", "white does", "notation")
        )
        wants_letter_code = (
            any(token in goal_lower for token in ("concatenation", "concatenate", "lowercase", "character", "third letter", "second letter"))
            and "sentence" not in goal_lower
        )
        wants_sentence = any(
            token in goal_lower
            for token in ("sentence", "phrase", "tautogram", "pangram", "write a sentence")
        )
        wants_plaintext_sentence = any(
            token in goal_lower
            for token in ("decipher", "cipher", "decrypt", "plaintext", "decoded")
        )
        wants_symbolic_formula = domain in {"math", "physics"} and (
            wants_formula
            or any(token in goal_lower for token in ("proportionality factor", "bordism", "resolvent", "poincaré polynomial"))
        )
        code_refs = {int(match) for match in re.findall(r"\bc(\d+)\b", goal_lower)}
        expected_code_length = len(code_refs) if code_refs else 0
        field_weights = {
            "content": 1.0,
            "description": 0.92,
            "summary": 0.90,
            "embedding_text": 0.88,
            "semantics": 0.86,
            "usage_conditions": 0.78,
            "keywords": 0.55,
            "aliases": 0.52,
            "tags": 0.48,
            "notes": 0.50,
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
            for key in (
                "content",
                "description",
                "summary",
                "embedding_text",
                "semantics",
                "usage_conditions",
                "keywords",
                "aliases",
                "tags",
                "notes",
                "domain",
                "category",
                "rpn_program",
                "pattern_form",
            ):
                value = str(fields.get(key, "")).strip()
                if not value:
                    continue
                field_weight = float(field_weights.get(key, 0.1))
                if wants_sentence or wants_plaintext_sentence:
                    normalized_full = " ".join(value.split()).strip()
                    if normalized_full and all(existing != normalized_full for existing, _, _ in candidates):
                        candidates.append((normalized_full, key, field_weight + 0.08))
                for candidate in self._extract_lhe_open_candidates(value, field_name=key):
                    if candidate and all(existing != candidate for existing, _, _ in candidates):
                        candidates.append((candidate, key, field_weight))
                if wants_letter_code:
                    for match in re.findall(r"(?:answer is|output(?: is)?|concatenation(?: is)?)\s+([a-z]{2,12})\b", value.lower()):
                        candidate = str(match).strip()
                        if candidate and all(existing != candidate for existing, _, _ in candidates):
                            candidates.append((candidate, key, field_weight + 0.2))
            for candidate, field_name, field_weight in candidates:
                if self._is_lhe_meta_candidate(candidate):
                    continue
                if self._is_lhe_code_like_candidate(candidate) and not wants_formula:
                    continue
                numeric_variant = self._canonicalize_lhe_short_numeric_candidate(candidate) if wants_short_numeric else ""
                candidate_variants = [candidate.strip()]
                if numeric_variant and numeric_variant not in candidate_variants:
                    candidate_variants.append(numeric_variant)
                for candidate_variant in candidate_variants:
                    candidate_tokens = self._semanticize_lhe_tokens(candidate_variant, preserve_single=True)
                    normalized_candidate = self._normalize_answer_text(candidate_variant)
                    if wants_chess_notation and not re.search(
                        r"\b(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)(?:,\s*[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)*\b",
                        candidate_variant,
                    ):
                        continue
                    if wants_letter_code and expected_code_length and not re.fullmatch(rf"[a-z]{{{expected_code_length}}}", normalized_candidate):
                        continue
                    if wants_plaintext_sentence:
                        lowered_plain = normalized_candidate.lower()
                        if any(term in lowered_plain for term in ("cipher", "cryptographic", "aes", "blowfish", "cast5", "notation")):
                            continue
                        if len(candidate_variant.split()) < 6:
                            continue
                    if wants_symbolic_formula and not wants_short_numeric:
                        if not any(token in candidate_variant for token in ("$", "\\(", "\\)", "^", "{", "}", "+", "-", "=")):
                            continue
                    if wants_short_numeric and len(candidate_variant.split()) >= 4 and not numeric_variant:
                        continue
                    score = float(evidence.get("rank_weight", 0.0)) * 0.9
                    score += field_weight
                    score += float(len(candidate_tokens & goal_tokens)) * 0.7
                    score += float(len(candidate_tokens & fused_semantic_tokens)) * 0.25
                    if wants_short_numeric and re.fullmatch(r"-?\d+(?:\.\d+)?", normalized_candidate):
                        score += 1.8
                    if not wants_short_numeric and re.fullmatch(r"-?\d+(?:\.\d+)?", normalized_candidate):
                        score -= 1.6
                    if wants_short_numeric and candidate_variant == candidate.strip() and numeric_variant and candidate_variant != numeric_variant:
                        score -= 1.2
                    if wants_formula:
                        if any(token in candidate_variant for token in ("$", "\\(", "\\)", "^", "{", "}", "+", "-", "=")):
                            score += 1.4
                        else:
                            score -= 1.0
                    if wants_symbolic_formula:
                        if any(token in candidate_variant for token in ("$", "\\(", "\\)", "^", "{", "}", "+", "-", "=", "mathcal", "cap_")):
                            score += 1.6
                        if len(candidate_variant.split()) >= 4 and not any(token in candidate_variant for token in ("$", "\\(", "\\)", "^", "{", "}", "+", "-", "=")):
                            score -= 1.5
                    if wants_chess_notation:
                        if re.search(r"\b(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)(?:,\s*[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)*\b", candidate_variant):
                            score += 2.0
                        else:
                            score -= 1.4
                    if wants_letter_code:
                        if re.fullmatch(r"[a-z]{2,12}", normalized_candidate):
                            score += 1.6
                        else:
                            score -= 1.6
                        if expected_code_length:
                            if len(normalized_candidate) == expected_code_length:
                                score += 1.6
                            else:
                                score -= 1.8
                        if len(candidate_variant.split()) > 1:
                            score -= 0.8
                    if wants_plaintext_sentence:
                        if len(candidate_variant.split()) >= 6 and candidate_variant[:1].isupper():
                            score += 1.5
                        else:
                            score -= 1.2
                        if self._is_lhe_code_like_candidate(candidate_variant):
                            score -= 1.4
                    if wants_sentence:
                        if len(candidate_variant.split()) >= 5 and not self._is_lhe_code_like_candidate(candidate_variant):
                            score += 1.0
                        else:
                            score -= 0.9
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
        parse_entities = self._build_lhe_parse_entities(prompt=prompt, parse_bundle=parse_bundle, options=options)
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
        meaning_atoms = fuse_meaning_atoms(
            [
                *meaning_atoms_from_parse_entities(fused_entities),
                *meaning_atoms_from_evidence_rows(evidence_rows),
            ]
        )
        reasoning_trace = [
            f"lhe_four_pass forward={len(forward_entities)} backward={len(backward_entities)} fused={len(fused_entities)} evidence={len(evidence_rows)}",
            f"lhe_meaning_atoms count={len(meaning_atoms)}",
            f"lhe_goal {goal.get('kind')} domain={domain_hint}",
        ]
        swarm_result: dict[str, Any] = {}
        if options:
            option_adjustments = self.lhe_reasoning_swarm.adjust_option_scores(
                prompt=prompt,
                options=options,
                goal=goal,
                fused_entities=fused_entities,
                evidence_rows=evidence_rows,
                parse_bundle=parse_bundle,
                route=route,
            )
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
                score += float(option_adjustments.get(option, 0.0))
                scored.append((score, option))
            scored.sort(key=lambda item: item[0], reverse=True)
            predicted = str(scored[0][1]) if scored else ""
            reasoning_trace.append(
                "lhe_option_scores "
                + ", ".join(f"{opt}={score:.3f}" for score, opt in scored[:4])
            )
        else:
            baseline_answer = ""
            if str(goal.get("kind", "")).strip().lower() == "semantic":
                baseline_answer = self._synthesize_lhe_open_answer(
                    fused_entities=fused_entities,
                    goal=goal,
                    evidence_rows=evidence_rows,
                )
            swarm_result = self.lhe_reasoning_swarm.reason_open_answer(
                prompt=prompt,
                goal=goal,
                fused_entities=fused_entities,
                meaning_atoms=meaning_atoms,
                evidence_rows=evidence_rows,
                parse_bundle=parse_bundle,
                route=route,
                baseline_answer=baseline_answer,
            )
            predicted = str(swarm_result.get("answer", "") or baseline_answer)
            reasoning_trace.append(f"lhe_open_answer {'present' if baseline_answer else 'empty'}")
            reasoning_trace.extend(list(swarm_result.get("reasoning_trace", [])))
        return {
            "predicted_answer": predicted,
            "reasoning_trace": reasoning_trace,
            "four_pass": {
                "forward_entities": forward_entities,
                "backward_entities": backward_entities,
                "fused_entities": fused_entities,
                "goal": goal,
                "meaning_atom_count": len(meaning_atoms),
                "evidence_count": len(evidence_rows),
                "composition_depth": 4 if fused_entities else 1,
            },
            "swarm": swarm_result,
        }

    def dispatch_lhe_task(
        self,
        *,
        route: dict[str, Any],
        task: dict[str, Any],
        use_enriched: bool,
    ) -> dict[str, Any]:
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
            "runtime": "knowledgeverse_internal_head",
        }
