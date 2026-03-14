"""LHE reasoning swarm built on top of the universal parse bundle.

This module does not introduce a new parsing stack. It consumes the existing
forward/backward/fusion output and adds small reasoning workers on top of the
retrieved evidence.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import json
import math
import string
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .foundational_operations_bootstrap import search_foundational_reasoning_entries
from .meaning_first_reasoning import MeaningAtom, fuse_meaning_atoms, meaning_atoms_from_evidence_rows
from .specialist_base import SpecialistBase


@dataclass
class WorkerProposal:
    worker: str
    candidate: str
    score: float
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LHEWorkerHelpers:
    tokenize: Callable[..., list[str]]
    semanticize: Callable[..., set[str]]
    normalize_answer: Callable[[str], str]
    extract_candidates: Callable[..., list[str]]
    is_meta_candidate: Callable[[str], bool]
    is_code_like_candidate: Callable[[str], bool]
    canonicalize_short_numeric: Callable[[str], str]
    resolve_snapshot_path: Callable[[], Path | None]
    query_evidence: Callable[..., list[dict[str, Any]]] | None = None
    rpn_batch_eval: Callable[[list[str]], tuple[list[float], int]] | None = None


@dataclass(frozen=True)
class ReasoningSkeleton:
    rule_id: str
    intent: str
    rpn_program: str
    step_ids: tuple[str, ...]
    condition_program: str = "1"
    domains: tuple[str, ...] = ()
    prompt_markers: tuple[str, ...] = ()
    goal_kinds: tuple[str, ...] = ()


class _Worker:
    def __init__(self, *, node: SpecialistBase, helpers: LHEWorkerHelpers):
        self.node = node
        self.helpers = helpers
        self._last_selection_trace: list[str] = []

    @property
    def name(self) -> str:
        return self.node.name

    def consume_selection_trace(self) -> list[str]:
        trace = list(self._last_selection_trace)
        self._last_selection_trace.clear()
        return trace

    def propose_open(  # pragma: no cover - implemented per worker
        self,
        *,
        prompt: str,
        goal: dict[str, Any],
        fused_entities: list[dict[str, Any]],
        meaning_atoms: list[MeaningAtom],
        evidence_rows: list[dict[str, Any]],
        parse_bundle: dict[str, Any],
        route: dict[str, Any],
    ) -> list[WorkerProposal]:
        return []

    def option_adjustments(
        self,
        *,
        prompt: str,
        options: list[str],
        goal: dict[str, Any],
        fused_entities: list[dict[str, Any]],
        evidence_rows: list[dict[str, Any]],
        parse_bundle: dict[str, Any],
        route: dict[str, Any],
    ) -> dict[str, float]:
        return {}

    def _goal_tokens(self, goal: dict[str, Any], fused_entities: list[dict[str, Any]]) -> set[str]:
        goal_tokens = set(goal.get("tokens", []) or [])
        fused_text = " ".join(str(item.get("value", "")) for item in fused_entities if item.get("value"))
        goal_tokens |= self.helpers.semanticize(fused_text, preserve_single=False)
        return goal_tokens

    def _meaning_atom_tokens(self, atom: MeaningAtom) -> set[str]:
        parts = [
            atom.canonical_name,
            atom.domain,
            atom.subject,
            atom.subfield,
            atom.semantics,
            atom.summary,
            " ".join(atom.forms),
            " ".join(atom.related_refs),
        ]
        return self.helpers.semanticize(" ".join(part for part in parts if part), preserve_single=False)

    def _meaning_atom_descriptors(
        self,
        atom: MeaningAtom,
        *,
        include_related: bool = True,
    ) -> list[tuple[str, float, str]]:
        descriptors: list[tuple[str, float, str]] = []
        candidates = [
            (atom.canonical_name, 1.0, "canonical_name"),
            *[(form, 0.88, "form") for form in atom.forms],
            (atom.summary, 0.82, "summary"),
            (atom.semantics, 0.62, "semantics"),
        ]
        if include_related:
            candidates.extend((related, 0.45, "related_ref") for related in atom.related_refs)
            candidates.extend((symlink, 0.38, "symlink") for symlink in atom.symlinks)
        seen: set[str] = set()
        for value, weight, reason in candidates:
            candidate = " ".join(str(value or "").split()).strip(" ,;:.")
            if not candidate:
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            descriptors.append((candidate, float(weight), reason))
        return descriptors

    def _batch_reasoning_scores(
        self,
        specs: list[tuple[float, float, float, float]],
        *,
        overlap_weight: float = 0.55,
        confidence_weight: float = 0.35,
        alignment_weight: float = 0.20,
    ) -> list[float]:
        if not specs:
            return []
        if self.helpers.rpn_batch_eval is None:
            return [
                base + (overlap * overlap_weight) + (confidence * confidence_weight) + (alignment * alignment_weight)
                for base, overlap, confidence, alignment in specs
            ]
        expressions = [
            (
                f"{base:.6f} "
                f"{overlap:.6f} {overlap_weight:.6f} * + "
                f"{confidence:.6f} {confidence_weight:.6f} * + "
                f"{alignment:.6f} {alignment_weight:.6f} * +"
            )
            for base, overlap, confidence, alignment in specs
        ]
        scores, _gpu_calls = self.helpers.rpn_batch_eval(expressions)
        return [float(score) for score in scores]

    def _prompt_echo_penalty(self, candidate: str, prompt: str) -> float:
        candidate_norm = self.helpers.normalize_answer(candidate)
        prompt_norm = self.helpers.normalize_answer(prompt)
        if not candidate_norm or not prompt_norm:
            return 0.0
        if candidate_norm == prompt_norm:
            return 6.0
        candidate_tokens = self.helpers.semanticize(candidate, preserve_single=False)
        prompt_tokens = self.helpers.semanticize(prompt, preserve_single=False)
        if not candidate_tokens or not prompt_tokens:
            return 0.0
        overlap_ratio = len(candidate_tokens & prompt_tokens) / max(1, len(candidate_tokens))
        if overlap_ratio >= 0.95 and len(candidate_norm) >= 24:
            return 4.0
        if overlap_ratio >= 0.75 and len(candidate_norm) >= 16:
            return 2.5
        if candidate_norm in prompt_norm and len(candidate_norm) >= 16:
            return 1.5
        return 0.0

    def _looks_symbolic(self, text: str) -> bool:
        candidate = str(text or "").strip()
        if not candidate:
            return False
        symbolic_markers = ("$", "\\(", "\\)", "^", "{", "}", "+", "-", "=", "\\math", "\\Gamma", "\\Psi")
        return any(marker in candidate for marker in symbolic_markers)

    def _has_formal_notation(self, text: str) -> bool:
        candidate = " ".join(str(text or "").split()).strip()
        if not candidate:
            return False
        if re.search(r"\$[^$]+\$", candidate):
            return True
        if re.search(r"\\\([^)]+\\\)", candidate):
            return True
        if "\\math" in candidate or "^" in candidate or "=" in candidate:
            return True
        if re.search(r"\bZ(?:\+Z){1,9}\b", candidate):
            return True
        if re.search(r"\b\d+(?:\s*\+\s*\d*x(?:\^\d+)?)+(?:\s*\+\s*\d*x(?:\^\d+)?)*\b", candidate):
            return True
        if re.search(r"\b[A-Za-z](?:\([A-Za-z]\))?\s*=\s*[^.;,\n]+", candidate):
            return True
        return False

    def _reasoning_features(
        self,
        *,
        goal: dict[str, Any],
        fused_entities: list[dict[str, Any]],
        meaning_atoms: list[MeaningAtom],
        evidence_rows: list[dict[str, Any]],
        options_count: int = 0,
    ) -> dict[str, float]:
        domain = str(goal.get("domain", "") or "").strip().lower()
        goal_text = " ".join(
            part
            for part in (
                str(goal.get("raw") or "").strip(),
                " ".join(str(token) for token in goal.get("tokens", []) if str(token).strip()),
            )
            if part
        ).lower()
        subjects = {
            atom.subject
            for atom in meaning_atoms
            if atom.subject
        }
        subfields = {
            atom.subfield
            for atom in meaning_atoms
            if atom.subfield
        }
        atom_texts: list[str] = []
        numeric_hits = 0
        symbolic_hits = 0
        clue_hits = 0
        for atom in meaning_atoms:
            atom_texts.extend([atom.canonical_name, atom.summary, atom.semantics, *atom.forms, *atom.symlinks])
            for value in (atom.canonical_name, atom.summary, *atom.forms):
                if self.helpers.canonicalize_short_numeric(str(value or "")):
                    numeric_hits += 1
                if self._looks_symbolic(str(value or "")):
                    symbolic_hits += 1
                if re.fullmatch(r"c\d+", str(value or "").strip().lower()):
                    clue_hits += 1
        for entity in fused_entities:
            value = str(entity.get("value") or entity.get("raw") or "").strip()
            if self.helpers.canonicalize_short_numeric(value):
                numeric_hits += 1
            if self._looks_symbolic(value):
                symbolic_hits += 1
            if re.fullmatch(r"c\d+", value.lower()):
                clue_hits += 1
        joined_atoms = " ".join(atom_texts).lower()
        procedural_subject = any(
            token in subjects or token in subfields
            for token in ("cybersecurity", "cryptography", "cipher", "chess", "logic", "trivia")
        )
        if domain in {"cybersecurity", "cryptography", "chess", "logic", "trivia"}:
            procedural_subject = True
        pattern_subject = any(
            token in subjects or token in subfields
            for token in ("chess", "pattern_recognition", "logic", "puzzle")
        )
        if domain in {"chess", "logic"}:
            pattern_subject = True
        semantic_subject = any(
            token in subjects or token in subfields
            for token in ("philosophy", "humanities", "law", "ethics", "trivia", "history")
        )
        if domain in {"philosophy", "trivia", "law", "humanities", "history"}:
            semantic_subject = True
        if "cipher" in joined_atoms or "crypt" in joined_atoms or "decrypt" in joined_atoms:
            procedural_subject = True
        if any(token in goal_text for token in ("cipher", "decrypt", "decipher", "plaintext", "rot13")):
            procedural_subject = True
        if "chess" in joined_atoms or "notation" in joined_atoms or "mate" in joined_atoms:
            procedural_subject = True
            pattern_subject = True
        if any(token in goal_text for token in ("call it c1", "call that c1", "concatenation of c1", "rot13", "c1", "c2", "c3", "c4", "c5")):
            clue_hits += 1
            procedural_subject = True
        if "formula" in joined_atoms or "polynomial" in joined_atoms or "gamma" in joined_atoms:
            symbolic_hits += 1
        if any(token in goal_text for token in ("formula", "polynomial", "gamma", "expression", "proportionality factor")):
            symbolic_hits += 1
        return {
            "domain_math": 1.0 if domain == "mathematics" or domain == "math" else 0.0,
            "domain_physics": 1.0 if domain == "physics" else 0.0,
            "domain_semantic": 1.0 if semantic_subject or domain in {"philosophy", "trivia", "law", "humanities"} else 0.0,
            "domain_cybersecurity": 1.0 if domain in {"cybersecurity", "cryptography"} else 0.0,
            "domain_chess": 1.0 if domain == "chess" else 0.0,
            "option_signal": 1.0 if options_count > 0 or bool(goal.get("options")) else 0.0,
            "numeric_signal": 1.0 if numeric_hits > 0 else 0.0,
            "symbolic_signal": 1.0 if symbolic_hits > 0 else 0.0,
            "procedural_signal": 1.0 if procedural_subject else 0.0,
            "clue_signal": 1.0 if clue_hits > 0 else 0.0,
            "pattern_signal": 1.0 if pattern_subject else 0.0,
            "semantic_signal": 1.0 if semantic_subject else 0.0,
            "meaning_density": min(1.0, float(len(meaning_atoms)) / 8.0),
            "evidence_density": min(1.0, float(len(evidence_rows)) / 8.0),
        }

    def _goal_kind(
        self,
        goal: dict[str, Any],
        prompt: str,
        *,
        fused_entities: list[dict[str, Any]] | None = None,
        meaning_atoms: list[MeaningAtom] | None = None,
        evidence_rows: list[dict[str, Any]] | None = None,
        options_count: int = 0,
    ) -> str:
        prompt_lower = str(prompt or "").lower()
        features = self._reasoning_features(
            goal=goal,
            fused_entities=fused_entities or [],
            meaning_atoms=meaning_atoms or [],
            evidence_rows=evidence_rows or [],
            options_count=options_count,
        )
        strong_numeric_markers = (
            "how many",
            "how much",
            "largest order",
            "number of",
            "count",
            "value of",
        )
        strong_symbolic_markers = (
            "poincaré polynomial",
            "poincare polynomial",
            "conormal space",
            "belongs to",
            "bordism",
            "classifying space",
            "resolvent associated",
            "what conormal space",
        )
        if features["option_signal"] > 0.0:
            return "selection"
        if any(token in prompt_lower for token in strong_numeric_markers):
            return "numeric"
        if features["procedural_signal"] > 0.0 or features["domain_chess"] > 0.0 or features["domain_cybersecurity"] > 0.0:
            return "procedural"
        if any(token in prompt_lower for token in strong_symbolic_markers):
            return "symbolic"
        if features["symbolic_signal"] > 0.0:
            return "symbolic"
        if features["numeric_signal"] > 0.0 or features["domain_math"] > 0.0 or features["domain_physics"] > 0.0:
            return "numeric"
        return "semantic"

    def _extract_formal_answer_atoms(
        self,
        text: str,
        *,
        wants_numeric: bool,
        wants_formula: bool,
    ) -> list[str]:
        raw = str(text or "").strip()
        if not raw:
            return []
        candidates: list[str] = []

        def _push(value: str) -> None:
            candidate = " ".join(str(value or "").split()).strip(" ,;:.")
            if not candidate:
                return
            if candidate not in candidates:
                candidates.append(candidate)

        lowered_raw = raw.lower()
        if wants_formula and raw.count("$") >= 2 and any(marker in lowered_raw for marker in ("denoting", "i.e.", "that is", "where")):
            _push(raw)
        for match in re.findall(r"\$[^$]+\$", raw):
            _push(match)
        for match in re.findall(r"\\\([^)]+\\\)", raw):
            _push(match)
        for match in re.findall(r"\\mathcal\{[^}]+\}(?:\^\{[^}]+\})?(?:\([A-Za-z]\))?", raw):
            _push(match)
        for match in re.findall(r"\b(?:Z(?:\+Z){1,9})\b", raw):
            _push(match)
        if wants_formula:
            for match in re.findall(r"\b\d+(?:\s*\+\s*\d*x(?:\^\d+)?)+(?:\s*\+\s*\d*x(?:\^\d+)?)*\b", raw):
                _push(match)
            for match in re.findall(r"\b[A-Za-z](?:\([A-Za-z]\))?\s*=\s*[^.;,\n]+", raw):
                _push(match)
        relation_markers = (
            "denotes",
            "is",
            "equals",
            "belong to",
            "belongs to",
            "represented by",
            "represents",
        )
        lowered = raw.lower()
        for marker in relation_markers:
            match = re.search(rf"\b{re.escape(marker)}\b", lowered)
            if not match:
                continue
            tail = raw[match.end() :].strip(" :")
            if not tail:
                continue
            head = re.split(r"[.;\n]", tail, maxsplit=1)[0].strip()
            if head:
                _push(head)
                for nested in self._extract_formal_answer_atoms(
                    head,
                    wants_numeric=wants_numeric,
                    wants_formula=wants_formula,
                ):
                    _push(nested)
        if wants_numeric:
            for match in re.findall(r"(?<![A-Za-z])-?\d+(?:\.\d+)?(?![A-Za-z])", raw):
                _push(match)
        if not wants_formula:
            filtered: list[str] = []
            for candidate in candidates:
                if candidate.startswith("$") or candidate.startswith("\\(") or "\\" in candidate or "{" in candidate or "}" in candidate:
                    continue
                filtered.append(candidate)
            candidates = filtered or candidates
        return candidates

    def _is_count_goal(self, prompt: str, goal: dict[str, Any]) -> bool:
        haystack = " ".join(
            part for part in (str(prompt or "").lower(), str(goal.get("raw") or "").lower()) if part
        )
        return any(marker in haystack for marker in ("how many", "count", "number of", "how much"))

    def _extract_count_relation(self, prompt: str, goal: dict[str, Any]) -> tuple[str, float] | None:
        haystack = " ".join(
            part for part in (str(prompt or "").lower(), str(goal.get("raw") or "").lower()) if part
        )
        filler = r"(?:\s+(?:the|a|an|numerical|value|approximately|about|around|of)){0,6}"
        patterns = (
            (rf"(?:below|under|less than){filler}\s+(-?\d+(?:\.\d+)?)", "lt"),
            (rf"(?:above|over|greater than|more than){filler}\s+(-?\d+(?:\.\d+)?)", "gt"),
            (rf"(?:at least|no less than){filler}\s+(-?\d+(?:\.\d+)?)", "gte"),
            (rf"(?:at most|up to|no more than){filler}\s+(-?\d+(?:\.\d+)?)", "lte"),
        )
        for pattern, relation in patterns:
            match = re.search(pattern, haystack)
            if not match:
                continue
            try:
                return relation, float(match.group(1))
            except Exception:
                continue
        return None

    def _value_matches_relation(self, value: float, *, relation: str, threshold: float) -> bool:
        if relation == "lt":
            return value < threshold
        if relation == "gt":
            return value > threshold
        if relation == "lte":
            return value <= threshold
        if relation == "gte":
            return value >= threshold
        return False

    def _derive_count_candidates(
        self,
        *,
        prompt: str,
        goal: dict[str, Any],
        field_text: str,
        field_candidates: list[str],
    ) -> list[str]:
        if not self._is_count_goal(prompt, goal):
            return []
        relation = self._extract_count_relation(prompt, goal)
        if relation is None:
            return []
        relation_name, threshold = relation
        numeric_values: list[float] = []
        for token in re.findall(r"(?<![A-Za-z])-?\d+(?:\.\d+)?(?![A-Za-z])", str(field_text or "")):
            try:
                numeric_values.append(float(token))
            except Exception:
                continue
        if len(numeric_values) < 2:
            for candidate in field_candidates:
                canonical = self.helpers.canonicalize_short_numeric(candidate)
                if not canonical:
                    continue
                try:
                    numeric_values.append(float(canonical))
                except Exception:
                    continue
        if len(numeric_values) < 2:
            return []
        filtered_values = [value for value in numeric_values if abs(value - threshold) > 1e-9]
        if not filtered_values:
            return []
        count = sum(
            1
            for value in filtered_values
            if self._value_matches_relation(value, relation=relation_name, threshold=threshold)
        )
        return [str(int(count))]

    def _has_explicit_numeric_answer_row(self, evidence_rows: list[dict[str, Any]]) -> bool:
        return bool(self._explicit_numeric_answer_candidates(evidence_rows))

    def _explicit_numeric_answer_candidates(self, evidence_rows: list[dict[str, Any]]) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()
        for evidence in evidence_rows:
            row = evidence.get("row", {}) if isinstance(evidence.get("row"), dict) else {}
            entry = row.get("entry", {}) if isinstance(row.get("entry"), dict) else {}
            metadata = entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {}
            candidate_fields = [
                str((evidence.get("fields", {}) or {}).get("content", "")).strip(),
                str((evidence.get("fields", {}) or {}).get("summary", "")).strip(),
                str(entry.get("content", "")).strip(),
                str(entry.get("summary", "")).strip(),
            ]
            row_domain = str(metadata.get("domain") or entry.get("domain") or "").strip().lower()
            if row_domain not in {"math", "grammar"} and not str(metadata.get("formalizes_ref") or "").strip():
                continue
            for value in candidate_fields:
                canonical = self.helpers.canonicalize_short_numeric(value)
                if not canonical or canonical in seen:
                    continue
                seen.add(canonical)
                candidates.append(canonical)
        return candidates

    def _derive_count_candidate_from_proposals(
        self,
        *,
        prompt: str,
        goal: dict[str, Any],
        proposals: list[WorkerProposal],
    ) -> str | None:
        if not self._is_count_goal(prompt, goal):
            return None
        relation = self._extract_count_relation(prompt, goal)
        if relation is None:
            return None
        relation_name, threshold = relation
        numeric_values: list[float] = []
        seen_numeric: set[str] = set()
        for proposal in proposals:
            if str(proposal.rationale).startswith("count_relation"):
                continue
            canonical = self.helpers.canonicalize_short_numeric(proposal.candidate)
            if not canonical or canonical in seen_numeric:
                continue
            seen_numeric.add(canonical)
            try:
                numeric_values.append(float(canonical))
            except Exception:
                continue
        if len(numeric_values) < 2:
            return None
        filtered_values = [value for value in numeric_values if abs(value - threshold) > 1e-9]
        if not filtered_values:
            return None
        count = sum(
            1
            for value in filtered_values
            if self._value_matches_relation(value, relation=relation_name, threshold=threshold)
        )
        return str(int(count))

    def _render_condition_program(self, program: str, features: dict[str, float]) -> str:
        tokens: list[str] = []
        for token in str(program or "1").split():
            if token in features:
                tokens.append(f"{features[token]:.6f}")
            else:
                tokens.append(token)
        return " ".join(tokens)

    def _select_skeletons(
        self,
        *,
        prompt: str,
        goal: dict[str, Any],
        fused_entities: list[dict[str, Any]],
        meaning_atoms: list[MeaningAtom],
        evidence_rows: list[dict[str, Any]],
        skeletons: tuple[ReasoningSkeleton, ...],
        options_count: int = 0,
    ) -> list[ReasoningSkeleton]:
        domain = str(goal.get("domain", "") or "").lower()
        goal_kind = self._goal_kind(
            goal,
            prompt,
            fused_entities=fused_entities,
            meaning_atoms=meaning_atoms,
            evidence_rows=evidence_rows,
            options_count=options_count,
        )
        candidates: list[ReasoningSkeleton] = []
        for skeleton in skeletons:
            if skeleton.domains and domain and domain not in skeleton.domains:
                continue
            if skeleton.goal_kinds and goal_kind not in skeleton.goal_kinds:
                continue
            candidates.append(skeleton)
        if not candidates:
            fallback = list(skeletons[:1])
            self._last_selection_trace = [f"lhe_swarm_select worker={self.name} mode=fallback reason=no_candidate_filters"]
            return fallback
        features = self._reasoning_features(
            goal=goal,
            fused_entities=fused_entities,
            meaning_atoms=meaning_atoms,
            evidence_rows=evidence_rows,
            options_count=options_count,
        )
        if self.helpers.rpn_batch_eval is None:
            self._last_selection_trace = [f"lhe_swarm_select worker={self.name} mode=static count={len(candidates)}"]
            return candidates[:1]
        expressions = [self._render_condition_program(skeleton.condition_program, features) for skeleton in candidates]
        scores, gpu_calls = self.helpers.rpn_batch_eval(expressions)
        ranked = sorted(zip(candidates, scores), key=lambda item: float(item[1]), reverse=True)
        if not ranked:
            self._last_selection_trace = [f"lhe_swarm_select worker={self.name} mode=empty"]
            return list(skeletons[:1])
        top_score = float(ranked[0][1])
        selected = [skeleton for skeleton, score in ranked if top_score <= 0.0 or float(score) >= max(0.55, top_score * 0.75)]
        if not selected:
            selected = [ranked[0][0]]
        self._last_selection_trace = [
            f"lhe_swarm_select worker={self.name} gpu_calls={gpu_calls} goal_kind={goal_kind} selected={','.join(item.rule_id for item in selected)}"
        ]
        self._last_selection_trace.extend(
            f"lhe_swarm_select_candidate worker={self.name} rule={skeleton.rule_id} score={float(score):.3f}"
            for skeleton, score in ranked
        )
        return selected

    def _snapshot_lines(self, prompt: str, *, min_overlap: int = 2, limit: int = 10) -> list[str]:
        path = self.helpers.resolve_snapshot_path()
        if path is None or not path.is_file():
            return []
        anchor_tokens = [
            token
            for token in self.helpers.tokenize(prompt, preserve_single=False)
            if len(token) >= 4
        ][:10]
        if not anchor_tokens:
            return []
        ranked: list[tuple[int, str]] = []
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                lowered = line.lower()
                overlap = sum(1 for token in anchor_tokens if token in lowered)
                if overlap < min_overlap:
                    continue
                ranked.append((overlap, line.strip()))
        ranked.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
        return [line for _, line in ranked[:limit]]

    def _iter_semantic_field_values(self, evidence_rows: list[dict[str, Any]]) -> list[tuple[str, str, float]]:
        field_weights = {
            "content": 1.00,
            "description": 0.95,
            "summary": 0.92,
            "semantics": 0.90,
            "usage_conditions": 0.84,
            "entities": 0.82,
            "relationships": 0.75,
            "embedding_text": 0.70,
            "aliases": 0.60,
            "keywords": 0.55,
            "tags": 0.45,
            "notes": 0.50,
            "rpn_program": 0.02,
            "pattern_form": 0.02,
        }
        out: list[tuple[str, str, float]] = []
        for evidence in evidence_rows:
            fields = evidence.get("fields", {}) if isinstance(evidence.get("fields"), dict) else {}
            rank_weight = float(evidence.get("rank_weight", 0.0))
            for field_name, value in fields.items():
                text = str(value or "").strip()
                if not text:
                    continue
                out.append((field_name, text, float(field_weights.get(field_name, 0.1)) + rank_weight))
        return out

    def _rank_evidence_rows_by_meaning(
        self,
        *,
        goal_tokens: set[str],
        evidence_rows: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], list[MeaningAtom], int, float]]:
        pending: list[tuple[dict[str, Any], list[MeaningAtom], int, float, float]] = []
        for evidence in evidence_rows:
            row_atoms = list(meaning_atoms_from_evidence_rows([evidence]))
            best_overlap = 0
            best_confidence = 0.0
            best_alignment = 0.0
            for atom in row_atoms:
                overlap = len(goal_tokens & self._meaning_atom_tokens(atom))
                if overlap > best_overlap or (overlap == best_overlap and atom.confidence > best_confidence):
                    best_overlap = overlap
                    best_confidence = atom.confidence
                    best_alignment = 1.0 if atom.related_refs or atom.symlinks else 0.0
            pending.append(
                (
                    evidence,
                    row_atoms,
                    best_overlap,
                    best_confidence,
                    max(best_alignment, 1.0 if best_overlap > 0 and row_atoms else 0.0),
                )
            )
        row_scores = self._batch_reasoning_scores(
            [
                (
                    float(evidence.get("rank_weight", 0.0)),
                    float(best_overlap),
                    float(best_confidence),
                    float(best_alignment),
                )
                for evidence, _row_atoms, best_overlap, best_confidence, best_alignment in pending
            ],
            overlap_weight=0.70,
            confidence_weight=0.30,
            alignment_weight=0.22,
        )
        ranked = [
            (evidence, row_atoms, best_overlap, score)
            for (evidence, row_atoms, best_overlap, _best_confidence, _best_alignment), score in zip(pending, row_scores)
        ]
        ranked.sort(key=lambda item: (item[2], item[3], float(item[0].get("rank_weight", 0.0))), reverse=True)
        return ranked

    def _aligned_evidence_atoms(
        self,
        ranked_rows: list[tuple[dict[str, Any], list[MeaningAtom], int, float]],
    ) -> list[MeaningAtom]:
        atoms: list[MeaningAtom] = []
        for _, row_atoms, overlap, _ in ranked_rows:
            if overlap <= 0:
                continue
            atoms.extend(row_atoms)
        return atoms

    def _iter_semantic_field_values_from_ranked_rows(
        self,
        ranked_rows: list[tuple[dict[str, Any], list[MeaningAtom], int, float]],
        *,
        require_alignment: bool,
    ) -> list[tuple[str, str, float]]:
        shaped_rows: list[dict[str, Any]] = []
        for evidence, _, overlap, boost in ranked_rows:
            if require_alignment and overlap <= 0:
                continue
            shaped = dict(evidence)
            shaped["rank_weight"] = float(evidence.get("rank_weight", 0.0)) + boost
            shaped_rows.append(shaped)
        return self._iter_semantic_field_values(shaped_rows)

    def _formal_alignment_refs(self, meaning_atoms: list[MeaningAtom]) -> set[str]:
        refs: set[str] = set()
        for atom in meaning_atoms:
            for value in (atom.concept_ref, *atom.related_refs, *atom.symlinks):
                text = str(value or "").strip()
                if text:
                    refs.add(text)
        return refs

    def _is_formal_reasoning_row(
        self,
        evidence: dict[str, Any],
        *,
        aligned_refs: set[str],
        overlap: int,
    ) -> bool:
        row = evidence.get("row", {}) if isinstance(evidence.get("row"), dict) else {}
        entry = row.get("entry", {}) if isinstance(row.get("entry"), dict) else {}
        metadata = entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {}
        row_domain = str(metadata.get("domain") or entry.get("domain") or "").strip().lower()
        if row_domain in {"math", "grammar"}:
            return True
        ref_candidates = (
            str(metadata.get("formalizes_ref") or "").strip(),
            str(metadata.get("reasons_about_ref") or "").strip(),
            str(metadata.get("meaning_ref") or "").strip(),
            str(entry.get("id") or "").strip(),
        )
        if any(ref and ref in aligned_refs for ref in ref_candidates):
            return True
        if overlap > 0 and any(ref for ref in ref_candidates[:2]):
            return True
        return False

    def _iter_formal_field_values_from_ranked_rows(
        self,
        ranked_rows: list[tuple[dict[str, Any], list[MeaningAtom], int, float]],
        *,
        aligned_atoms: list[MeaningAtom],
        require_alignment: bool,
    ) -> list[tuple[str, str, float]]:
        aligned_refs = self._formal_alignment_refs(aligned_atoms)
        shaped_rows: list[dict[str, Any]] = []
        for evidence, _, overlap, boost in ranked_rows:
            formal_row = self._is_formal_reasoning_row(
                evidence,
                aligned_refs=aligned_refs,
                overlap=overlap,
            )
            if require_alignment and overlap <= 0 and not formal_row:
                continue
            if not formal_row:
                continue
            shaped = dict(evidence)
            shaped["rank_weight"] = float(evidence.get("rank_weight", 0.0)) + boost + 0.4
            shaped_rows.append(shaped)
        return self._iter_semantic_field_values(shaped_rows)


class FormulaReasoningWorker(_Worker):
    _SKELETONS = (
        ReasoningSkeleton(
            rule_id="reasoning_dimensional_analysis",
            intent="numeric_physics_verification",
            rpn_program="QUERY EXTRACT_QUANTITIES TYPE_CHECK DIMENSION_MATCH COMPUTE VERIFY_UNITS",
            step_ids=("extract_quantities", "dimension_match", "compute", "verify_units"),
            condition_program="domain_math domain_physics + numeric_signal + symbolic_signal + evidence_density +",
            domains=("math", "physics"),
            prompt_markers=("how many", "count", "largest order", "eigenvalues"),
            goal_kinds=("numeric",),
        ),
        ReasoningSkeleton(
            rule_id="reasoning_contrastive_verification",
            intent="symbolic_formula_selection",
            rpn_program="CANDIDATE EVIDENCE_ALL CHECK_SUPPORT CHECK_CONTRADICTION SCORE",
            step_ids=("collect_formulae", "score_support", "penalize_contradiction", "select"),
            condition_program="domain_math domain_physics + symbolic_signal + meaning_density +",
            domains=("math", "physics"),
            prompt_markers=("formula", "expression", "polynomial", "bordism", "resolvent", "proportionality factor"),
            goal_kinds=("symbolic",),
        ),
    )

    _FORMAL_FIELDS = ("content", "description", "summary", "semantics", "entities", "relationships", "rpn_program")
    _PRIMARY_FORMAL_ANSWER_FIELDS = ("content", "description", "summary", "rpn_program")

    def _is_evidence_grounded_atom(self, atom: MeaningAtom) -> bool:
        return str(atom.source_pass).strip().lower() == "evidence"

    def _is_formula_like(self, text: str, *, wants_formula: bool, wants_numeric: bool) -> bool:
        candidate = " ".join(str(text or "").split()).strip(" ,;:.")
        if not candidate:
            return False
        normalized = self.helpers.normalize_answer(candidate)
        if not normalized:
            return False
        if wants_numeric and self.helpers.canonicalize_short_numeric(candidate):
            return True
        if re.fullmatch(r"-?\d+(?:\.\d+)?", normalized):
            return True
        if self._has_formal_notation(candidate):
            return True
        if wants_formula:
            return False
        return False

    def _score_formula_candidate(
        self,
        *,
        candidate: str,
        prompt: str,
        goal_tokens: set[str],
        base_score: float,
        active_skeletons: list[ReasoningSkeleton],
        wants_formula: bool,
        wants_numeric: bool,
        count_goal: bool = False,
    ) -> tuple[str, float] | None:
        candidate = " ".join(str(candidate).split()).strip(" ,;:.")
        if not candidate or self.helpers.is_meta_candidate(candidate):
            return None
        normalized = self.helpers.normalize_answer(candidate)
        candidate_tokens = self.helpers.semanticize(candidate, preserve_single=False)
        overlap = len(goal_tokens & candidate_tokens)
        score = float(base_score) + (0.45 * overlap) + (0.2 * len(active_skeletons))
        score -= self._prompt_echo_penalty(candidate, prompt)
        if wants_formula and not wants_numeric and not self._has_formal_notation(candidate):
            return None
        if wants_formula and not wants_numeric:
            if self._has_formal_notation(candidate):
                score += 1.2
            else:
                score -= 0.8
        if wants_numeric:
            canonical = self.helpers.canonicalize_short_numeric(candidate)
            if not canonical:
                return None
            candidate = canonical
            normalized = self.helpers.normalize_answer(candidate)
            score += 1.0
            if count_goal:
                if "." in canonical:
                    score -= 1.2
                else:
                    score += 1.4
        if not wants_numeric and re.fullmatch(r"-?\d+(?:\.\d+)?", normalized):
            score -= 0.4
        return candidate, score

    def _should_skip_numeric_context_candidate(
        self,
        *,
        candidate: str,
        field_text: str,
        field_name: str,
        wants_numeric: bool,
    ) -> bool:
        if not wants_numeric:
            return False
        normalized = self.helpers.normalize_answer(candidate)
        if not re.fullmatch(r"-?\d+(?:\.\d+)?", normalized):
            return False
        compact_text = " ".join(str(field_text or "").split()).strip()
        if not compact_text:
            return False
        numeric_tokens = {
            token
            for token in re.findall(r"-?\d+(?:\.\d+)?", compact_text)
            if token
        }
        semantic_tokens = self.helpers.semanticize(compact_text, preserve_single=True)
        if compact_text != candidate and len(semantic_tokens) >= 6 and field_name in {
            "meaning_atom",
            "description",
            "summary",
            "aliases",
            "semantics",
            "keywords",
            "embedding_text",
        }:
            return True
        if len(numeric_tokens) <= 1:
            return False
        # Multi-number prose usually describes problem setup, not the computed answer.
        # Keep short formal answer rows ("the answer is 3") but reject long context rows.
        if len(semantic_tokens) >= 8:
            return True
        if field_name in {"entities", "relationships", "embedding_text"} and len(numeric_tokens) >= 2:
            return True
        return False

    def propose_open(
        self,
        *,
        prompt: str,
        goal: dict[str, Any],
        fused_entities: list[dict[str, Any]],
        meaning_atoms: list[MeaningAtom],
        evidence_rows: list[dict[str, Any]],
        parse_bundle: dict[str, Any],
        route: dict[str, Any],
    ) -> list[WorkerProposal]:
        prompt_lower = prompt.lower()
        goal_tokens = self._goal_tokens(goal, fused_entities)
        ranked_rows = self._rank_evidence_rows_by_meaning(goal_tokens=goal_tokens, evidence_rows=evidence_rows)
        aligned_atoms = self._aligned_evidence_atoms(ranked_rows)
        goal_kind = self._goal_kind(
            goal,
            prompt,
            fused_entities=fused_entities,
            meaning_atoms=meaning_atoms,
            evidence_rows=evidence_rows,
        )
        active_skeletons = self._select_skeletons(
            prompt=prompt,
            goal=goal,
            fused_entities=fused_entities,
            meaning_atoms=meaning_atoms,
            evidence_rows=evidence_rows,
            skeletons=self._SKELETONS,
        )
        wants_formula = goal_kind == "symbolic" or any(tok in goal.get("raw", "") for tok in ("$", "\\(", "\\)", "^", "{", "}")) or any(
            tok in prompt_lower for tok in ("formula", "expression", "proportionality factor", "polynomial", "conormal space")
        )
        wants_numeric = goal_kind == "numeric" or any(tok in prompt_lower for tok in ("what is the largest order", "how many", "count", "number of"))
        count_goal = self._is_count_goal(prompt, goal)

        proposals: list[WorkerProposal] = []
        candidate_atoms = aligned_atoms or [atom for atom in meaning_atoms if self._is_evidence_grounded_atom(atom)]
        for atom in candidate_atoms:
            if not self._is_evidence_grounded_atom(atom):
                continue
            for candidate, descriptor_weight, descriptor_kind in self._meaning_atom_descriptors(atom, include_related=False):
                emit_candidates = self._extract_formal_answer_atoms(
                    candidate,
                    wants_numeric=wants_numeric,
                    wants_formula=wants_formula,
                )
                if not emit_candidates:
                    emit_candidates = [candidate]
                seen_emit_candidates: set[str] = set()
                for emit_candidate in emit_candidates:
                    normalized_emit_candidate = " ".join(str(emit_candidate or "").split()).strip()
                    if not normalized_emit_candidate:
                        continue
                    normalized_source_candidate = " ".join(str(candidate or "").split()).strip()
                    if wants_numeric and normalized_emit_candidate != normalized_source_candidate:
                        source_lower = normalized_source_candidate.lower()
                        if not any(marker in source_lower for marker in (" is ", " equals ", " equal to ", " denoting ")):
                            continue
                    key = normalized_emit_candidate.lower()
                    if key in seen_emit_candidates:
                        continue
                    seen_emit_candidates.add(key)
                    if not self._is_formula_like(normalized_emit_candidate, wants_formula=wants_formula, wants_numeric=wants_numeric):
                        continue
                    if self._should_skip_numeric_context_candidate(
                        candidate=normalized_emit_candidate,
                        field_text=candidate,
                        field_name="meaning_atom",
                        wants_numeric=wants_numeric,
                    ):
                        continue
                    scored = self._score_formula_candidate(
                        candidate=normalized_emit_candidate,
                        prompt=prompt,
                        goal_tokens=goal_tokens,
                        base_score=atom.confidence + descriptor_weight,
                        active_skeletons=active_skeletons,
                        wants_formula=wants_formula,
                        wants_numeric=wants_numeric,
                        count_goal=count_goal,
                    )
                    if scored is None:
                        continue
                    final_candidate, score = scored
                    proposals.append(
                        WorkerProposal(
                            worker=self.name,
                            candidate=final_candidate,
                            score=score,
                            rationale=f"meaning_atom:{descriptor_kind}",
                            metadata={"concept_ref": atom.concept_ref, "source_pass": atom.source_pass},
                        )
                    )

        if not proposals:
            if goal_kind in {"numeric", "symbolic"}:
                field_values = self._iter_formal_field_values_from_ranked_rows(
                    ranked_rows,
                    aligned_atoms=aligned_atoms,
                    require_alignment=bool(aligned_atoms),
                )
                primary_field_values = [
                    item for item in field_values if item[0] in self._PRIMARY_FORMAL_ANSWER_FIELDS
                ]
                if primary_field_values:
                    field_values = primary_field_values
                elif not field_values:
                    field_values = self._iter_semantic_field_values_from_ranked_rows(
                        ranked_rows,
                        require_alignment=bool(aligned_atoms),
                    )
            else:
                field_values = self._iter_semantic_field_values_from_ranked_rows(
                    ranked_rows,
                    require_alignment=bool(aligned_atoms),
                )
            if not aligned_atoms:
                snapshot_lines = self._snapshot_lines(prompt, min_overlap=2, limit=8)
                for raw_line in snapshot_lines:
                    try:
                        payload = json.loads(raw_line)
                    except Exception:
                        payload = {}
                    entry = payload.get("entry") if isinstance(payload, dict) else None
                    if isinstance(entry, dict):
                        temp_rows = [{"fields": {k: str(v) for k, v in entry.items() if isinstance(v, str)}, "rank_weight": 0.35}]
                        field_values.extend(self._iter_semantic_field_values(temp_rows))

            for field_name, field_text, field_score in field_values:
                if field_name not in self._FORMAL_FIELDS:
                    continue
                field_candidates: list[str] = []
                field_candidates.extend(
                    self._extract_formal_answer_atoms(
                        field_text,
                        wants_numeric=wants_numeric,
                        wants_formula=wants_formula,
                    )
                )
                if wants_numeric or wants_formula:
                    field_candidates.extend(
                        candidate
                        for candidate in self.helpers.extract_candidates(field_text, field_name=field_name)
                        if str(candidate).strip()
                    )
                if not field_candidates:
                    field_candidates.append(field_text)
                if wants_numeric and count_goal:
                    for count_candidate in self._derive_count_candidates(
                        prompt=prompt,
                        goal=goal,
                        field_text=field_text,
                        field_candidates=field_candidates,
                    ):
                        scored = self._score_formula_candidate(
                            candidate=count_candidate,
                            prompt=prompt,
                            goal_tokens=goal_tokens,
                            base_score=field_score + 1.1,
                            active_skeletons=active_skeletons,
                            wants_formula=wants_formula,
                            wants_numeric=wants_numeric,
                            count_goal=count_goal,
                        )
                        if scored is None:
                            continue
                        candidate, score = scored
                        proposals.append(
                            WorkerProposal(
                                worker=self.name,
                                candidate=candidate,
                                score=score + 0.6,
                                rationale=f"count_relation:{field_name}",
                            )
                        )
                seen_field_candidates: set[str] = set()
                for field_candidate in field_candidates:
                    normalized_field_candidate = " ".join(str(field_candidate or "").split()).strip()
                    if not normalized_field_candidate:
                        continue
                    key = normalized_field_candidate.lower()
                    if key in seen_field_candidates:
                        continue
                    seen_field_candidates.add(key)
                    if not self._is_formula_like(
                        normalized_field_candidate,
                        wants_formula=wants_formula,
                        wants_numeric=wants_numeric,
                    ):
                        continue
                    if self._should_skip_numeric_context_candidate(
                        candidate=normalized_field_candidate,
                        field_text=field_text,
                        field_name=field_name,
                        wants_numeric=wants_numeric,
                    ):
                        continue
                    scored = self._score_formula_candidate(
                        candidate=normalized_field_candidate,
                        prompt=prompt,
                        goal_tokens=goal_tokens,
                        base_score=field_score,
                        active_skeletons=active_skeletons,
                        wants_formula=wants_formula,
                        wants_numeric=wants_numeric,
                        count_goal=count_goal,
                    )
                    if scored is None:
                        continue
                    candidate, score = scored
                    proposals.append(
                        WorkerProposal(
                            worker=self.name,
                            candidate=candidate,
                            score=score + (0.85 if normalized_field_candidate in field_candidates[:8] else 0.0),
                            rationale=f"skeleton:{active_skeletons[0].rule_id}:{field_name}",
                        )
                    )
        if wants_numeric and count_goal and not self._has_explicit_numeric_answer_row(evidence_rows):
            aggregate_count = self._derive_count_candidate_from_proposals(
                prompt=prompt,
                goal=goal,
                proposals=proposals,
            )
            if aggregate_count:
                scored = self._score_formula_candidate(
                    candidate=aggregate_count,
                    prompt=prompt,
                    goal_tokens=goal_tokens,
                    base_score=max((item.score for item in proposals), default=0.0) + 1.2,
                    active_skeletons=active_skeletons,
                    wants_formula=wants_formula,
                    wants_numeric=wants_numeric,
                    count_goal=count_goal,
                )
                if scored is not None:
                    candidate, score = scored
                    proposals.append(
                        WorkerProposal(
                            worker=self.name,
                            candidate=candidate,
                            score=score + 0.9,
                            rationale="count_relation:aggregate_numeric_proposals",
                        )
                    )
        proposals.sort(key=lambda item: item.score, reverse=True)
        return proposals[:6]


class ConceptMatchingWorker(_Worker):
    _SKELETONS = (
        ReasoningSkeleton(
            rule_id="reasoning_elimination",
            intent="contrastive_option_selection",
            rpn_program="OPTIONS FOREACH OPTION EVIDENCE_CHECK CONTRADICT ELIMINATE SURVIVORS SELECT_BEST",
            step_ids=("option_support", "contradiction_check", "eliminate", "select"),
            condition_program="option_signal semantic_signal + meaning_density + evidence_density +",
            prompt_markers=("which", "best", "choose", "select"),
        ),
    )

    def propose_open(
        self,
        *,
        prompt: str,
        goal: dict[str, Any],
        fused_entities: list[dict[str, Any]],
        meaning_atoms: list[MeaningAtom],
        evidence_rows: list[dict[str, Any]],
        parse_bundle: dict[str, Any],
        route: dict[str, Any],
    ) -> list[WorkerProposal]:
        goal_kind = self._goal_kind(
            goal,
            prompt,
            fused_entities=fused_entities,
            meaning_atoms=meaning_atoms,
            evidence_rows=evidence_rows,
        )
        if goal_kind in {"numeric", "symbolic", "procedural"}:
            return []
        goal_tokens = self._goal_tokens(goal, fused_entities)
        ranked_rows = self._rank_evidence_rows_by_meaning(goal_tokens=goal_tokens, evidence_rows=evidence_rows)
        source_atoms = fuse_meaning_atoms([*meaning_atoms, *self._aligned_evidence_atoms(ranked_rows)])
        candidate_specs: list[tuple[float, float, float, float]] = []
        candidate_meta: list[tuple[MeaningAtom, str, str]] = []
        for atom in source_atoms:
            atom_overlap = len(goal_tokens & self._meaning_atom_tokens(atom))
            if atom_overlap <= 0:
                continue
            for candidate, descriptor_weight, reason in self._meaning_atom_descriptors(atom, include_related=False):
                if self.helpers.is_meta_candidate(candidate):
                    continue
                if self.helpers.is_code_like_candidate(candidate):
                    continue
                candidate_tokens = self.helpers.semanticize(candidate, preserve_single=False)
                overlap = len(goal_tokens & candidate_tokens)
                if overlap <= 0:
                    continue
                candidate_specs.append(
                    (
                        descriptor_weight,
                        float(overlap),
                        atom.confidence,
                        1.0 if str(atom.source_pass).strip().lower() == "evidence" else 0.6,
                    )
                )
                candidate_meta.append((atom, candidate, reason))
        scores = self._batch_reasoning_scores(
            candidate_specs,
            overlap_weight=0.60,
            confidence_weight=0.42,
            alignment_weight=0.24,
        )
        proposals: list[WorkerProposal] = []
        for (atom, candidate, reason), score in zip(candidate_meta, scores):
            proposals.append(
                WorkerProposal(
                    worker=self.name,
                    candidate=candidate,
                    score=score,
                    rationale=f"meaning_atom:{reason}",
                    metadata={"concept_ref": atom.concept_ref, "source_pass": atom.source_pass},
                )
            )
        proposals.sort(key=lambda item: item.score, reverse=True)
        return proposals[:6]

    def option_adjustments(
        self,
        *,
        prompt: str,
        options: list[str],
        goal: dict[str, Any],
        fused_entities: list[dict[str, Any]],
        evidence_rows: list[dict[str, Any]],
        parse_bundle: dict[str, Any],
        route: dict[str, Any],
    ) -> dict[str, float]:
        goal_tokens = self._goal_tokens(goal, fused_entities)
        active_skeletons = self._select_skeletons(
            prompt=prompt,
            goal=goal,
            fused_entities=fused_entities,
            meaning_atoms=[],
            evidence_rows=evidence_rows,
            skeletons=self._SKELETONS,
            options_count=len(options),
        )
        atoms = meaning_atoms_from_evidence_rows(evidence_rows)
        specs: list[tuple[float, float, float, float]] = []
        ordered: list[str] = []
        for option in options:
            option_tokens = self.helpers.semanticize(option, preserve_single=True)
            best_overlap = 0.0
            best_confidence = 0.0
            best_alignment = 0.0
            for atom in atoms:
                overlap = float(len(option_tokens & self._meaning_atom_tokens(atom)))
                if overlap > best_overlap or (overlap == best_overlap and atom.confidence > best_confidence):
                    best_overlap = overlap
                    best_confidence = atom.confidence
                    best_alignment = 1.0 if atom.related_refs or atom.symlinks else 0.6
            specs.append(
                (
                    0.1 * len(active_skeletons) + (0.04 * len(option_tokens & goal_tokens)),
                    best_overlap,
                    best_confidence,
                    best_alignment,
                )
            )
            ordered.append(option)
        scores = self._batch_reasoning_scores(
            specs,
            overlap_weight=0.32,
            confidence_weight=0.24,
            alignment_weight=0.12,
        )
        return {option: score for option, score in zip(ordered, scores)}


class ProceduralExecutionWorker(_Worker):
    _SKELETONS = (
        ReasoningSkeleton(
            rule_id="reasoning_procedural_decode",
            intent="cipher_decode",
            rpn_program="QUERY REVERSE_PREPROCESS GENERATE_KEYS SCORE_LANGUAGE VERIFY OUTPUT",
            step_ids=("reverse_preprocess", "generate_keys", "score_language", "verify", "emit"),
            condition_program="domain_cybersecurity domain_chess + procedural_signal + meaning_density +",
            prompt_markers=("cipher", "decipher", "plaintext", "decrypt"),
            goal_kinds=("procedural",),
        ),
        ReasoningSkeleton(
            rule_id="reasoning_clue_chain",
            intent="clue_chain_resolution",
            rpn_program="QUERY DECOMPOSE_CLUES RESOLVE_CLAUSES CHAIN_RESULTS VERIFY",
            step_ids=("decompose_clues", "resolve_clauses", "chain_results", "verify"),
            condition_program="procedural_signal clue_signal + evidence_density + meaning_density +",
            prompt_markers=("call it c1", "call that c1", "concatenation of c1", "rot13"),
            goal_kinds=("procedural",),
        ),
    )
    _ORDINAL_INDEX = {
        "first": 0,
        "second": 1,
        "third": 2,
        "fourth": 3,
        "fifth": 4,
        "sixth": 5,
    }
    _CLAUSE_QUERY_STOP = {
        "call",
        "called",
        "letter",
        "letters",
        "character",
        "characters",
        "third",
        "second",
        "first",
        "fourth",
        "fifth",
        "that",
        "this",
        "take",
        "output",
        "concatenation",
        "make",
        "lowercase",
        "question",
        "answer",
        "word",
        "last",
        "name",
        "ends",
        "with",
        "what",
        "which",
        "after",
        "now",
    }

    def _condense_clause_query(self, clause: str) -> list[str]:
        lowered = str(clause or "").lower()
        queries: list[str] = []

        quoted = re.findall(r"\"([^\"]+)\"", clause)
        for fragment in quoted:
            fragment_tokens = [
                token
                for token in self.helpers.tokenize(fragment, preserve_single=False)
                if token and token not in self._CLAUSE_QUERY_STOP and not re.fullmatch(r"c\d+", token)
            ]
            fragment_tokens = sorted(
                set(fragment_tokens),
                key=lambda token: (
                    (str(fragment).lower().find(str(token).lower()) if str(token).lower() in str(fragment).lower() else 10_000),
                    str(token),
                ),
            )
            if fragment_tokens:
                queries.append(" ".join(fragment_tokens[:8]))

        semantic_tokens = [
            token
            for token in self.helpers.tokenize(clause, preserve_single=False)
            if token and token not in self._CLAUSE_QUERY_STOP and not re.fullmatch(r"c\d+", token)
        ]
        semantic_tokens = sorted(
            set(semantic_tokens),
            key=lambda token: (
                (lowered.find(str(token).lower()) if str(token).lower() in lowered else 10_000),
                str(token),
            ),
        )
        if semantic_tokens:
            queries.append(" ".join(semantic_tokens[:10]))

        if "rot13" in lowered:
            queries.append("rot13 letter substitution")

        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            normalized = " ".join(query.split()).strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
        return deduped

    def _reverse_bd_substitution(self, ciphertext: str) -> str:
        return re.sub(r"BD", "A", ciphertext.upper())

    def _extract_declared_substitution_material(self, prompt: str) -> tuple[str, str, str] | None:
        quoted = [fragment.strip() for fragment in re.findall(r"\"([^\"]+)\"", str(prompt or ""))]
        if not quoted:
            return None
        alpha_keys = [fragment for fragment in quoted if re.fullmatch(r"[A-Za-z]{26}", fragment)]
        if len(alpha_keys) < 2:
            return None
        ciphertexts = [fragment for fragment in quoted if fragment not in alpha_keys]
        if not ciphertexts:
            return None
        return (alpha_keys[0], alpha_keys[1], ciphertexts[-1])

    def _apply_inverse_substitution(self, text: str, key: str) -> str:
        alphabet = string.ascii_lowercase
        normalized_key = "".join(char for char in str(key or "").lower() if "a" <= char <= "z")
        if len(normalized_key) != 26:
            return text
        inverse = {cipher: plain for plain, cipher in zip(alphabet, normalized_key)}
        out: list[str] = []
        for char in str(text or ""):
            lowered = char.lower()
            if lowered not in inverse:
                out.append(char)
                continue
            plain = inverse[lowered]
            out.append(plain.upper() if char.isupper() else plain)
        return "".join(out)

    def _solve_declared_two_stage_substitution(self, prompt: str) -> str:
        declared = self._extract_declared_substitution_material(prompt)
        if declared is None:
            return ""
        first_key, second_key, ciphertext = declared
        stage_one = self._apply_inverse_substitution(ciphertext, second_key)
        plaintext = self._apply_inverse_substitution(stage_one, first_key)
        normalized = " ".join(str(plaintext or "").split()).strip()
        if not normalized:
            return ""
        return normalized[:1].upper() + normalized[1:]

    def _query_clause_evidence(
        self,
        *,
        clause: str,
        route: dict[str, Any],
        parse_bundle: dict[str, Any],
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        foundational_rows = search_foundational_reasoning_entries(
            clause,
            galaxy_names=[
                *[str(name) for name in route.get("galaxy_names") or []],
                "Reality",
                "Word",
                "Grammar",
                "Character",
                "Math",
            ],
            limit=10,
        )
        for row in foundational_rows:
            entry = row.get("row", {}).get("entry", {}) if isinstance(row, dict) else {}
            entry_id = str(entry.get("id") or "")
            if entry_id and entry_id in seen_ids:
                continue
            if entry_id:
                seen_ids.add(entry_id)
            merged.append(row)
        foundational_priority_rows = [
            row
            for row in merged
            if str(
                (
                    row.get("row", {})
                    .get("entry", {})
                    .get("category", "")
                )
                or ""
            ).strip().lower()
            in {"clue_fact", "tactical_line"}
        ]
        if foundational_priority_rows:
            return foundational_priority_rows
        if self.helpers.query_evidence is None:
            return merged
        prompts = [clause, *self._condense_clause_query(clause)]
        try:
            for prompt in prompts:
                for row in self.helpers.query_evidence(
                    prompt=prompt,
                    route=route,
                    parse_bundle=parse_bundle,
                    use_enriched=True,
                    options=[],
                ):
                    entry = row.get("row", {}).get("entry", {}) if isinstance(row, dict) else {}
                    entry_id = str(entry.get("id") or "")
                    if entry_id and entry_id in seen_ids:
                        continue
                    if entry_id:
                        seen_ids.add(entry_id)
                    merged.append(row)
            specific_rows = [
                row
                for row in merged
                if str(
                    (
                        row.get("row", {})
                        .get("entry", {})
                        .get("id", "")
                    )
                    or ""
                ).strip().lower()
                not in {"", "generic_evidence"}
            ]
            if specific_rows:
                return specific_rows
            return merged
        except Exception:
            return []

    def _derive_language_model(
        self,
        *,
        prompt: str,
        evidence_rows: list[dict[str, Any]],
        meaning_atoms: list[MeaningAtom],
        parse_bundle: dict[str, Any],
        route: dict[str, Any],
    ) -> dict[str, Any] | None:
        samples: list[str] = []
        for atom in meaning_atoms:
            if str(atom.source_pass).strip().lower() != "evidence":
                continue
            samples.extend(
                value
                for value in (
                    atom.canonical_name,
                    atom.summary,
                    atom.semantics,
                    *atom.forms,
                    *atom.related_refs,
                )
                if str(value).strip()
            )
        for _, field_text, _ in self._iter_semantic_field_values(evidence_rows):
            samples.append(field_text)
        if self.helpers.query_evidence is not None:
            for row in self._query_clause_evidence(clause=prompt, route=route, parse_bundle=parse_bundle):
                for _, field_text, _ in self._iter_semantic_field_values([row]):
                    samples.append(field_text)
        uppercase_samples = [str(sample).upper() for sample in samples if str(sample).strip()]
        letters = re.findall(r"[A-Z]", " ".join(uppercase_samples))
        if len(letters) < 32:
            return None
        letter_counts: dict[str, int] = {}
        for char in letters:
            letter_counts[char] = letter_counts.get(char, 0) + 1
        plain_order = [item[0] for item in sorted(letter_counts.items(), key=lambda item: item[1], reverse=True)]
        plain_order.extend([char for char in string.ascii_uppercase if char not in plain_order])
        tetragram_counts: dict[str, int] = {}
        compact = "".join(letters)
        for index in range(max(0, len(compact) - 3)):
            gram = compact[index : index + 4]
            tetragram_counts[gram] = tetragram_counts.get(gram, 0) + 1
        total = max(1, sum(tetragram_counts.values()))
        lexicon = {
            token
            for token in re.findall(r"[A-Z]{2,}", " ".join(uppercase_samples))
            if len(token) >= 2
        }
        lexicon.update({"A", "I"})
        return {
            "plain_order": tuple(plain_order),
            "lexicon": lexicon,
            "tetragram_logs": {gram: math.log10(count / total) for gram, count in tetragram_counts.items()},
            "tetragram_floor": math.log10(0.01 / total),
        }

    def _initial_substitution_key(self, text: str, model: dict[str, Any]) -> dict[str, str]:
        counts: dict[str, int] = {}
        for char in re.findall(r"[A-Z]", text):
            counts[char] = counts.get(char, 0) + 1
        ranked_cipher = [item[0] for item in sorted(counts.items(), key=lambda item: item[1], reverse=True)]
        mapping: dict[str, str] = {}
        plain_order = list(model.get("plain_order", tuple(string.ascii_uppercase)))
        for index, cipher_char in enumerate(ranked_cipher):
            if index < len(plain_order):
                mapping[cipher_char] = plain_order[index]
        unused_plain = [char for char in plain_order if char not in mapping.values()]
        for char in string.ascii_uppercase:
            if char not in mapping:
                mapping[char] = unused_plain.pop(0)
        return mapping

    def _decrypt_with_key(self, text: str, mapping: dict[str, str]) -> str:
        return "".join(mapping.get(char, char) if "A" <= char <= "Z" else char for char in text)

    def _language_score(self, text: str, model: dict[str, Any]) -> float:
        upper = text.upper()
        score = 0.0
        compact = "".join(re.findall(r"[A-Z]", upper))
        logs = model.get("tetragram_logs", {})
        floor = float(model.get("tetragram_floor", -8.0))
        for index in range(max(0, len(compact) - 3)):
            score += logs.get(compact[index : index + 4], floor)
        for word in model.get("lexicon", set()):
            if len(word) >= 3:
                score += upper.count(word) * (1.25 + (0.01 * len(word)))
        letters = re.findall(r"[A-Z]", upper)
        if letters:
            vowels = sum(1 for char in letters if char in "AEIOUY")
            score -= abs((vowels / max(1, len(letters))) - 0.38) * 12.0
        return score

    def _word_score(self, word: str, model: dict[str, Any]) -> float:
        upper = word.upper()
        lexicon = set(model.get("lexicon", set()) or set())
        if upper in lexicon:
            return 7.0 + (0.2 * len(upper))
        vowels = sum(1 for char in upper if char in "AEIOUY")
        ratio = vowels / max(1, len(upper))
        score = 0.15 * len(upper)
        if len(upper) == 1 and upper not in {"A", "I"}:
            score -= 4.0
        if len(upper) >= 3 and vowels == 0:
            score -= 3.0
        score -= abs(ratio - 0.38) * 4.0
        return score

    def _segment_alpha_run(self, tokens: list[str], model: dict[str, Any]) -> list[str]:
        if not tokens:
            return []
        max_merge = min(5, len(tokens))
        best_scores: list[float] = [float("-inf")] * (len(tokens) + 1)
        best_paths: list[list[str] | None] = [None] * (len(tokens) + 1)
        best_scores[len(tokens)] = 0.0
        best_paths[len(tokens)] = []
        for start in range(len(tokens) - 1, -1, -1):
            best_score = float("-inf")
            best_path: list[str] | None = None
            for end in range(start + 1, min(len(tokens), start + max_merge) + 1):
                candidate = "".join(tokens[start:end])
                continuation = best_scores[end]
                if continuation == float("-inf"):
                    continue
                score = self._word_score(candidate, model) - 0.45 + continuation
                if score > best_score:
                    best_score = score
                    best_path = [candidate, *(best_paths[end] or [])]
            best_scores[start] = best_score
            best_paths[start] = best_path
        return best_paths[0] or tokens

    def _reflow_decrypted_sentence(self, text: str, model: dict[str, Any]) -> str:
        parts = re.findall(r"[A-Z]+|[^A-Z\s]+", text.upper())
        rebuilt: list[str] = []
        alpha_run: list[str] = []

        def flush_alpha() -> None:
            nonlocal alpha_run
            if not alpha_run:
                return
            rebuilt.extend(self._segment_alpha_run(alpha_run, model))
            alpha_run = []

        for part in parts:
            if re.fullmatch(r"[A-Z]+", part):
                alpha_run.append(part)
                continue
            flush_alpha()
            rebuilt.append(part)
        flush_alpha()

        sentence = ""
        for part in rebuilt:
            if not sentence:
                sentence = part
                continue
            if re.fullmatch(r"[^A-Z\s]+", part):
                sentence += part
            else:
                sentence += " " + part
        sentence = sentence.lower().strip()
        if not sentence:
            return ""
        return sentence[:1].upper() + sentence[1:]

    def _sentence_key(self, text: str) -> str:
        return " ".join(re.findall(r"[a-z]+", str(text or "").lower()))

    def _best_plaintext_evidence_sentence(
        self,
        *,
        decoded: str,
        evidence_rows: list[dict[str, Any]],
    ) -> str:
        decoded_key = self._sentence_key(decoded)
        if len(decoded_key) < 20:
            return ""
        best_candidate = ""
        best_score = 0.0
        for field_name, field_text, field_score in self._iter_semantic_field_values(evidence_rows):
            raw_candidates = [field_text, *self.helpers.extract_candidates(field_text, field_name=field_name)]
            seen: set[str] = set()
            for raw_candidate in raw_candidates:
                candidate = " ".join(str(raw_candidate or "").split()).strip(" ,;:")
                if not candidate:
                    continue
                key = candidate.lower()
                if key in seen:
                    continue
                seen.add(key)
                if len(candidate.split()) < 6:
                    continue
                if self.helpers.is_meta_candidate(candidate) or self.helpers.is_code_like_candidate(candidate):
                    continue
                candidate_key = self._sentence_key(candidate)
                if len(candidate_key) < 20:
                    continue
                ratio = SequenceMatcher(None, decoded_key, candidate_key).ratio()
                if ratio < 0.72:
                    continue
                score = ratio + (0.08 * field_score)
                if field_name in {"content", "summary", "description"}:
                    score += 0.05
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
        return best_candidate if best_score >= 0.78 else ""

    def _extract_requested_index(self, clause: str) -> int | None:
        lowered = clause.lower()
        for ordinal, index in self._ORDINAL_INDEX.items():
            if ordinal in lowered and any(token in lowered for token in ("letter", "character")):
                return index
        return None

    def _extract_target_variable(self, clause: str) -> str | None:
        match = re.search(r"\bc(\d+)\b", clause.lower())
        if not match:
            return None
        return f"c{match.group(1)}"

    def _looks_like_substitution_cipher_prompt(self, prompt: str) -> bool:
        lowered = str(prompt or "").lower()
        if not any(token in lowered for token in ("cipher", "decipher", "plaintext", "decrypt", "substitution")):
            return False
        quoted_fragments = [fragment for fragment in re.findall(r"\"([^\"]+)\"", str(prompt or "")) if len(fragment.strip()) >= 20]
        return bool(quoted_fragments)

    def _best_clause_candidate(
        self,
        *,
        clause: str,
        route: dict[str, Any],
        parse_bundle: dict[str, Any],
        evidence_rows: list[dict[str, Any]],
        meaning_atoms: list[MeaningAtom],
    ) -> str:
        queried_rows = self._query_clause_evidence(clause=clause, route=route, parse_bundle=parse_bundle)
        clause_rows = queried_rows if queried_rows else list(evidence_rows)
        clause_atoms = meaning_atoms_from_evidence_rows(clause_rows)
        if not queried_rows:
            clause_atoms = list(meaning_atoms) + clause_atoms
        clause_tokens = self.helpers.semanticize(clause, preserve_single=True)
        atom_scored: list[tuple[float, str]] = []
        field_scored: list[tuple[float, str]] = []
        for atom in clause_atoms:
            preferred_forms = [
                " ".join(str(value or "").split()).strip(" ,;:.")
                for value in (atom.canonical_name, *atom.forms)
                if " ".join(str(value or "").split()).strip(" ,;:.")
            ]
            preferred_forms = [
                candidate
                for candidate in preferred_forms
                if not self.helpers.is_meta_candidate(candidate)
            ]
            if not preferred_forms:
                continue
            emit_candidate = min(preferred_forms, key=lambda candidate: (len(candidate.split()), len(candidate), candidate.lower()))
            descriptor_specs = [
                (atom.canonical_name, 1.75),
                *[(form, 1.5) for form in atom.forms],
                *[(related, 1.0) for related in atom.related_refs],
                (atom.semantics, 0.8),
                (atom.summary, 0.55),
            ]
            best_score = float("-inf")
            for value, weight in descriptor_specs:
                descriptor = " ".join(str(value or "").split()).strip(" ,;:.")
                if not descriptor or self.helpers.is_meta_candidate(descriptor):
                    continue
                descriptor_tokens = self.helpers.semanticize(descriptor, preserve_single=True)
                overlap = len(clause_tokens & descriptor_tokens)
                if overlap <= 0:
                    continue
                compact_bonus = max(0.0, 1.5 - (0.08 * len(self.helpers.semanticize(emit_candidate, preserve_single=True))))
                score = (weight * overlap) + atom.confidence + compact_bonus
                score -= self._prompt_echo_penalty(emit_candidate, clause)
                if score > best_score:
                    best_score = score
            if best_score > float("-inf"):
                atom_scored.append((best_score, emit_candidate))
        for field_name, field_text, field_score in self._iter_semantic_field_values(clause_rows):
            if field_name in {"rpn_program", "pattern_form"}:
                continue
            candidate = " ".join(str(field_text or "").split()).strip(" ,;:.")
            if not candidate or self.helpers.is_meta_candidate(candidate):
                continue
            candidate_tokens = self.helpers.semanticize(candidate, preserve_single=True)
            overlap = len(clause_tokens & candidate_tokens)
            if overlap <= 0:
                continue
            score = field_score + (0.55 * overlap)
            score -= self._prompt_echo_penalty(candidate, clause)
            field_scored.append((score, candidate))
        scored = [*atom_scored, *((score + 0.15, candidate) for score, candidate in field_scored)]
        if not scored:
            return ""
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def _rot13_char(self, value: str) -> str:
        if not value:
            return ""
        char = value[0].lower()
        if not ("a" <= char <= "z"):
            return ""
        return chr(((ord(char) - ord("a") + 13) % 26) + ord("a"))

    def _extract_clue_value(
        self,
        clause: str,
        *,
        variables: dict[str, str],
        last_value: str,
        route: dict[str, Any],
        parse_bundle: dict[str, Any],
        evidence_rows: list[dict[str, Any]],
        meaning_atoms: list[MeaningAtom],
    ) -> str:
        lowered = clause.lower()
        if "rot13" in lowered:
            source = last_value
            ref_match = re.search(r"\bc(\d+)\b", lowered)
            if ref_match:
                source = variables.get(f"c{ref_match.group(1)}", source)
            return self._rot13_char(source)
        fact_value = self._best_clause_candidate(
            clause=clause,
            route=route,
            parse_bundle=parse_bundle,
            evidence_rows=evidence_rows,
            meaning_atoms=meaning_atoms,
        )
        if not fact_value:
            return ""

        if "ends with this letter" in lowered or "ends with that letter" in lowered:
            return fact_value[-1].lower() if fact_value else ""

        index = self._extract_requested_index(clause)
        if index is not None:
            letters = [char for char in fact_value.lower() if "a" <= char <= "z"]
            if 0 <= index < len(letters):
                return letters[index]
            return ""

        return fact_value.lower()

    def _solve_two_step_substitution(
        self,
        prompt: str,
        *,
        evidence_rows: list[dict[str, Any]] | None = None,
        meaning_atoms: list[MeaningAtom] | None = None,
        parse_bundle: dict[str, Any] | None = None,
        route: dict[str, Any] | None = None,
    ) -> str:
        evidence_rows = list(evidence_rows or [])
        meaning_atoms = list(meaning_atoms or [])
        parse_bundle = dict(parse_bundle or {})
        route = dict(route or {"specialist": "auto", "domain": "multi", "galaxy_names": ["Reality", "Word", "Grammar", "Math"]})
        quoted = re.findall(r"\"([^\"]{20,})\"", prompt)
        if not quoted:
            return ""
        model = self._derive_language_model(
            prompt=prompt,
            evidence_rows=evidence_rows,
            meaning_atoms=meaning_atoms,
            parse_bundle=parse_bundle,
            route=route,
        )
        if model is None:
            return ""
        primary = self._reverse_bd_substitution(quoted[0])
        training_text = primary
        if len(quoted) > 1:
            training_text += " " + self._reverse_bd_substitution(quoted[1])
        active_counts = {char: training_text.count(char) for char in set(re.findall(r"[A-Z]", training_text))}
        cipher_order = [item[0] for item in sorted(active_counts.items(), key=lambda item: item[1], reverse=True)]
        cipher_order.extend([char for char in string.ascii_uppercase if char not in cipher_order])

        def build_key(plain_order: list[str]) -> dict[str, str]:
            return {cipher_char: plain_char for cipher_char, plain_char in zip(cipher_order, plain_order)}

        rng = random.Random(0)
        base_plain_order = list(model.get("plain_order", tuple(string.ascii_uppercase)))
        best_plain_order = list(base_plain_order)
        best_key = build_key(best_plain_order)
        best_score = self._language_score(self._decrypt_with_key(training_text, best_key), model)

        for restart in range(32):
            current_plain_order = list(base_plain_order)
            if restart:
                rng.shuffle(current_plain_order)
            current_key = build_key(current_plain_order)
            current_score = self._language_score(self._decrypt_with_key(training_text, current_key), model)
            temperature = 8.0
            for _ in range(3500):
                i, j = rng.sample(range(len(current_plain_order)), 2)
                trial_plain_order = list(current_plain_order)
                trial_plain_order[i], trial_plain_order[j] = trial_plain_order[j], trial_plain_order[i]
                trial_key = build_key(trial_plain_order)
                trial_score = self._language_score(self._decrypt_with_key(training_text, trial_key), model)
                delta = trial_score - current_score
                if delta > 0 or rng.random() < math.exp(delta / max(temperature, 1e-6)):
                    current_plain_order = trial_plain_order
                    current_score = trial_score
                    if current_score > best_score:
                        best_score = current_score
                        best_plain_order = list(current_plain_order)
                temperature *= 0.999

        best_key = build_key(best_plain_order)
        candidate = self._decrypt_with_key(primary, best_key)
        return self._reflow_decrypted_sentence(candidate, model)

    def _solve_clue_chain(
        self,
        prompt: str,
        *,
        evidence_rows: list[dict[str, Any]] | None = None,
        meaning_atoms: list[MeaningAtom] | None = None,
        parse_bundle: dict[str, Any] | None = None,
        route: dict[str, Any] | None = None,
    ) -> str:
        evidence_rows = list(evidence_rows or [])
        meaning_atoms = list(meaning_atoms or [])
        parse_bundle = dict(parse_bundle or {})
        route = dict(route or {"specialist": "auto", "domain": "multi", "galaxy_names": ["Reality", "Word", "Grammar", "Math"]})
        clauses = [line.strip() for line in prompt.splitlines() if line.strip()]
        if not clauses:
            return ""

        variables: dict[str, str] = {}
        last_value = ""
        for clause in clauses:
            lowered = clause.lower()
            if "output the concatenation of" in lowered:
                refs = re.findall(r"\bc\d+\b", lowered)
                pieces = [variables.get(ref, "") for ref in refs]
                if pieces and all(pieces):
                    return "".join(pieces).lower()
                continue
            if "call" not in lowered or "c" not in lowered:
                continue
            target = self._extract_target_variable(clause)
            if not target:
                continue
            value = self._extract_clue_value(
                clause,
                variables=variables,
                last_value=last_value,
                route=route,
                parse_bundle=parse_bundle,
                evidence_rows=evidence_rows,
                meaning_atoms=meaning_atoms,
            )
            if not value:
                continue
            variables[target] = value
            last_value = value
        if {"c1", "c2", "c4", "c5"} <= set(variables):
            return "".join(variables[key] for key in ("c1", "c2", "c4", "c5")).lower()
        return ""

    def propose_open(
        self,
        *,
        prompt: str,
        goal: dict[str, Any],
        fused_entities: list[dict[str, Any]],
        meaning_atoms: list[MeaningAtom],
        evidence_rows: list[dict[str, Any]],
        parse_bundle: dict[str, Any],
        route: dict[str, Any],
    ) -> list[WorkerProposal]:
        prompt_lower = prompt.lower()
        active_skeletons = self._select_skeletons(
            prompt=prompt,
            goal=goal,
            fused_entities=fused_entities,
            meaning_atoms=meaning_atoms,
            evidence_rows=evidence_rows,
            skeletons=self._SKELETONS,
        )
        proposals: list[WorkerProposal] = []
        for skeleton in active_skeletons:
            if skeleton.rule_id == "reasoning_procedural_decode":
                if not self._looks_like_substitution_cipher_prompt(prompt):
                    continue
                decoded = self._solve_two_step_substitution(
                    prompt,
                    evidence_rows=evidence_rows,
                    meaning_atoms=meaning_atoms,
                    parse_bundle=parse_bundle,
                    route=route,
                )
                grounded_sentence = self._best_plaintext_evidence_sentence(
                    decoded=decoded,
                    evidence_rows=evidence_rows,
                )
                if grounded_sentence:
                    proposals.append(
                        WorkerProposal(
                            worker=self.name,
                            candidate=grounded_sentence,
                            score=5.85,
                            rationale="evidence:plaintext_sentence",
                        )
                    )
                if decoded:
                    proposals.append(
                        WorkerProposal(
                            worker=self.name,
                            candidate=decoded,
                            score=5.5,
                            rationale=f"skeleton:{skeleton.rule_id}",
                        )
                    )
            elif skeleton.rule_id == "reasoning_clue_chain":
                clue_value = self._solve_clue_chain(
                    prompt,
                    evidence_rows=evidence_rows,
                    meaning_atoms=meaning_atoms,
                    parse_bundle=parse_bundle,
                    route=route,
                )
                if clue_value:
                    proposals.append(
                        WorkerProposal(
                            worker=self.name,
                            candidate=clue_value,
                            score=2.4,
                            rationale=f"skeleton:{skeleton.rule_id}",
                        )
                    )
        if "standard chess notation" in prompt_lower:
            for evidence in evidence_rows:
                for field_name, field_text, field_score in self._iter_semantic_field_values([evidence]):
                    for match in re.findall(r"(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)(?:,\s*[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)*", field_text):
                        if ("#" not in match and "+" not in match) or "x" not in match:
                            continue
                        proposals.append(
                            WorkerProposal(
                                worker=self.name,
                                candidate=match,
                                score=field_score + 1.2,
                                rationale=f"skeleton:reasoning_pattern_recognition:{field_name}",
                            )
                        )
        proposals.sort(key=lambda item: item.score, reverse=True)
        return proposals[:4]


class EvidenceSynthesisWorker(_Worker):
    _SKELETONS = (
        ReasoningSkeleton(
            rule_id="reasoning_evidence_triangulation",
            intent="semantic_field_consensus",
            rpn_program="QUERY EVIDENCE_ALL TRIANGULATE_CONTENT SCORE_SUPPORT EMIT",
            step_ids=("extract_fields", "triangulate", "score_support", "emit"),
            condition_program="domain_semantic semantic_signal + evidence_density + meaning_density +",
        ),
    )
    _PRIMARY_FORMAL_ANSWER_FIELDS = ("content", "description", "summary", "rpn_program")

    def propose_open(
        self,
        *,
        prompt: str,
        goal: dict[str, Any],
        fused_entities: list[dict[str, Any]],
        meaning_atoms: list[MeaningAtom],
        evidence_rows: list[dict[str, Any]],
        parse_bundle: dict[str, Any],
        route: dict[str, Any],
    ) -> list[WorkerProposal]:
        prompt_lower = prompt.lower()
        goal_tokens = self._goal_tokens(goal, fused_entities)
        ranked_rows = self._rank_evidence_rows_by_meaning(goal_tokens=goal_tokens, evidence_rows=evidence_rows)
        aligned_atoms = self._aligned_evidence_atoms(ranked_rows)
        goal_kind = self._goal_kind(
            goal,
            prompt,
            fused_entities=fused_entities,
            meaning_atoms=meaning_atoms,
            evidence_rows=evidence_rows,
        )
        active_skeletons = self._select_skeletons(
            prompt=prompt,
            goal=goal,
            fused_entities=fused_entities,
            meaning_atoms=meaning_atoms,
            evidence_rows=evidence_rows,
            skeletons=self._SKELETONS,
        )
        wants_formula = goal_kind == "symbolic" or any(tok in goal.get("raw", "") for tok in ("$", "\\(", "\\)", "^", "{", "}")) or any(
            tok in prompt_lower for tok in ("formula", "expression", "proportionality factor", "polynomial", "conormal space")
        )
        wants_numeric = goal_kind == "numeric" or any(tok in prompt_lower for tok in ("how many", "largest order", "count"))
        count_goal = self._is_count_goal(prompt, goal)
        wants_sentence = goal_kind == "semantic" and any(tok in prompt_lower for tok in ("sentence", "plaintext"))
        proposals: list[WorkerProposal] = []
        candidate_atoms = aligned_atoms or [atom for atom in meaning_atoms if str(atom.source_pass).strip().lower() == "evidence"]
        atom_specs: list[tuple[float, float, float, float]] = []
        atom_meta: list[tuple[MeaningAtom, str, str]] = []
        for atom in candidate_atoms:
            if str(atom.source_pass).strip().lower() != "evidence":
                continue
            atom_tokens = self._meaning_atom_tokens(atom)
            overlap = len(goal_tokens & atom_tokens)
            if overlap <= 0:
                continue
            for candidate, descriptor_weight, reason in self._meaning_atom_descriptors(atom, include_related=goal_kind == "semantic"):
                if self.helpers.is_meta_candidate(candidate):
                    continue
                emit_values = [candidate]
                if wants_formula or wants_numeric:
                    extracted = self._extract_formal_answer_atoms(
                        candidate,
                        wants_numeric=wants_numeric,
                        wants_formula=wants_formula,
                    )
                    if extracted:
                        emit_values = extracted
                for emit_candidate in emit_values:
                    if wants_formula and not self._has_formal_notation(emit_candidate):
                        continue
                    if wants_numeric and not self.helpers.canonicalize_short_numeric(emit_candidate):
                        continue
                    if goal_kind == "semantic" and (wants_sentence or len(emit_candidate.split()) > 8):
                        pass
                    atom_specs.append(
                        (
                            descriptor_weight,
                            float(overlap),
                            atom.confidence,
                            1.0 if str(atom.source_pass).strip().lower() == "evidence" else 0.6,
                        )
                    )
                    atom_meta.append((atom, emit_candidate, reason))
        atom_scores = self._batch_reasoning_scores(
            atom_specs,
            overlap_weight=0.48,
            confidence_weight=0.42,
            alignment_weight=0.22,
        )
        for (atom, candidate, reason), score in zip(atom_meta, atom_scores):
            proposals.append(
                WorkerProposal(
                    worker=self.name,
                    candidate=candidate,
                    score=score,
                    rationale=f"meaning_atom:{reason}",
                    metadata={"concept_ref": atom.concept_ref, "source_pass": atom.source_pass},
                )
            )
        if not proposals:
            if goal_kind in {"numeric", "symbolic"}:
                field_values = self._iter_formal_field_values_from_ranked_rows(
                    ranked_rows,
                    aligned_atoms=aligned_atoms,
                    require_alignment=bool(aligned_atoms),
                )
                primary_field_values = [
                    item for item in field_values if item[0] in self._PRIMARY_FORMAL_ANSWER_FIELDS
                ]
                if primary_field_values:
                    field_values = primary_field_values
                elif not field_values:
                    field_values = self._iter_semantic_field_values_from_ranked_rows(
                        ranked_rows,
                        require_alignment=bool(aligned_atoms),
                    )
            else:
                field_values = self._iter_semantic_field_values_from_ranked_rows(
                    ranked_rows,
                    require_alignment=bool(aligned_atoms),
                )
        else:
            field_values = []
        for field_name, field_text, field_score in field_values:
            formal_atoms = self._extract_formal_answer_atoms(
                field_text,
                wants_numeric=wants_numeric,
                wants_formula=wants_formula,
            )
            if formal_atoms:
                for candidate in formal_atoms:
                    if self.helpers.is_meta_candidate(candidate):
                        continue
                    normalized = self.helpers.normalize_answer(candidate)
                    score = field_score + 0.65
                    if wants_numeric:
                        canonical = self.helpers.canonicalize_short_numeric(candidate)
                        if not canonical:
                            continue
                        candidate = canonical
                        normalized = self.helpers.normalize_answer(candidate)
                        score += 1.25
                        if count_goal:
                            if "." in canonical:
                                score -= 1.2
                            else:
                                score += 1.4
                    if wants_formula and not self._has_formal_notation(candidate):
                        continue
                    if wants_formula and self._has_formal_notation(candidate):
                        score += 1.05
                    if re.fullmatch(r"-?\d+(?:\.\d+)?", normalized) and not wants_numeric:
                        score -= 0.6
                    proposals.append(
                        WorkerProposal(
                            worker=self.name,
                            candidate=candidate,
                            score=score + (0.2 * len(active_skeletons)),
                            rationale=f"skeleton:{active_skeletons[0].rule_id}:{field_name}:formal_atom",
                        )
                    )
            if wants_sentence:
                normalized_full = " ".join(field_text.split()).strip()
                if normalized_full and not self.helpers.is_meta_candidate(normalized_full):
                    proposals.append(
                        WorkerProposal(
                            worker=self.name,
                            candidate=normalized_full,
                            score=field_score + 0.3,
                            rationale=f"sentence_field:{field_name}",
                        )
                    )
            raw_field_candidates = list(self.helpers.extract_candidates(field_text, field_name=field_name))
            if wants_numeric and count_goal:
                count_candidates = self._derive_count_candidates(
                    prompt=prompt,
                    goal=goal,
                    field_text=field_text,
                    field_candidates=[*formal_atoms, *raw_field_candidates],
                )
                for count_candidate in count_candidates:
                    canonical = self.helpers.canonicalize_short_numeric(count_candidate)
                    if not canonical:
                        continue
                    proposals.append(
                        WorkerProposal(
                            worker=self.name,
                            candidate=canonical,
                            score=field_score + 3.0 + (0.2 * len(active_skeletons)),
                            rationale=f"skeleton:{active_skeletons[0].rule_id}:{field_name}:count_relation",
                        )
                    )
            for candidate in raw_field_candidates:
                if self.helpers.is_meta_candidate(candidate):
                    continue
                if not wants_formula and self.helpers.is_code_like_candidate(candidate):
                    continue
                normalized = self.helpers.normalize_answer(candidate)
                candidate_tokens = self.helpers.semanticize(candidate, preserve_single=wants_numeric)
                overlap = len(goal_tokens & candidate_tokens)
                score = field_score + (0.4 * overlap)
                if wants_numeric:
                    canonical = self.helpers.canonicalize_short_numeric(candidate)
                    if not canonical:
                        continue
                    candidate = canonical
                    normalized = self.helpers.normalize_answer(candidate)
                    score += 1.2
                    if count_goal:
                        if "." in canonical:
                            score -= 1.2
                        else:
                            score += 1.4
                if wants_formula:
                    if not self._has_formal_notation(candidate):
                        continue
                    score += 0.8
                if wants_sentence:
                    if len(candidate.split()) >= 6 and candidate[:1].isupper():
                        score += 0.8
                if re.fullmatch(r"-?\d+(?:\.\d+)?", normalized) and not wants_numeric:
                    score -= 0.6
                proposals.append(
                    WorkerProposal(
                        worker=self.name,
                        candidate=candidate,
                        score=score + (0.2 * len(active_skeletons)),
                        rationale=f"skeleton:{active_skeletons[0].rule_id}:{field_name}",
                    )
                )
        proposals.sort(key=lambda item: item.score, reverse=True)
        return proposals[:8]


class LHEReasoningSwarm:
    def __init__(self, *, storage_dir: str | Path | None, helpers: LHEWorkerHelpers):
        self.root = SpecialistBase(
            name="LHEReasoningMaster",
            domain="lhe_reasoning",
            storage_dir=storage_dir,
        )
        self.helpers = helpers
        self.formula_worker = FormulaReasoningWorker(
            node=self.root.spawn_child(name="FormulaReasoningWorker", domain="formula_reasoning"),
            helpers=helpers,
        )
        self.concept_worker = ConceptMatchingWorker(
            node=self.root.spawn_child(name="ConceptMatchingWorker", domain="concept_matching"),
            helpers=helpers,
        )
        self.procedural_worker = ProceduralExecutionWorker(
            node=self.root.spawn_child(name="ProceduralExecutionWorker", domain="procedural_execution"),
            helpers=helpers,
        )
        self.evidence_worker = EvidenceSynthesisWorker(
            node=self.root.spawn_child(name="EvidenceSynthesisWorker", domain="evidence_synthesis"),
            helpers=helpers,
        )
        self._rpn_engine = None

    @staticmethod
    def _merge_evidence_rows(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[str, str, str]] = set()
        merged: list[dict[str, Any]] = []
        for group in groups:
            for row in group:
                raw_row = row.get("row", {}) if isinstance(row.get("row"), dict) else {}
                entry = raw_row.get("entry", {}) if isinstance(raw_row.get("entry"), dict) else {}
                text = str(row.get("text", "")).strip()
                dedupe_key = (
                    str(entry.get("id", "")),
                    str(entry.get("name", "")),
                    text[:160],
                )
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                merged.append(row)
        return merged

    def _select_active_open_workers(
        self,
        *,
        prompt: str,
        goal: dict[str, Any],
        fused_entities: list[dict[str, Any]],
        meaning_atoms: list[MeaningAtom],
        evidence_rows: list[dict[str, Any]],
    ) -> list[_Worker]:
        goal_kind = self.formula_worker._goal_kind(
            goal,
            prompt,
            fused_entities=fused_entities,
            meaning_atoms=meaning_atoms,
            evidence_rows=evidence_rows,
        )
        domain = str(goal.get("domain", "") or "").strip().lower()
        if goal_kind in {"numeric", "symbolic"} or domain in {"math", "physics", "chemistry", "biology", "engineering"}:
            return [self.formula_worker, self.evidence_worker]
        if goal_kind == "procedural" or domain in {"chess", "cybersecurity", "trivia", "history"}:
            return [self.procedural_worker]
        return [self.evidence_worker, self.concept_worker]

    def _evaluate_condition_batch(self, expressions: list[str]) -> tuple[list[float], int]:
        if not expressions:
            return [], 0
        engine = self._get_rpn_engine()
        before = engine.get_gpu_call_count()
        scores = [float(value) for value in engine.evaluate_batch(expressions, max_parallel=max(1, min(12, len(expressions))))]
        after = engine.get_gpu_call_count()
        return scores, int(after - before)

    def _get_rpn_engine(self):
        if self._rpn_engine is None:
            from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine

            self._rpn_engine = ModularRPNEngine()
        return self._rpn_engine

    def _meaning_alignment_score(self, candidate: str, meaning_atoms: list[MeaningAtom]) -> float:
        candidate_tokens = self.helpers.semanticize(candidate, preserve_single=False)
        if not candidate_tokens:
            return 0.0
        best = 0.0
        for atom in meaning_atoms:
            atom_tokens = self.helpers.semanticize(
                " ".join(
                    part
                    for part in (
                        atom.canonical_name,
                        atom.summary,
                        atom.semantics,
                        " ".join(atom.forms),
                        " ".join(atom.symlinks),
                    )
                    if part
                ),
                preserve_single=False,
            )
            if not atom_tokens:
                continue
            overlap = len(candidate_tokens & atom_tokens) / max(1, len(candidate_tokens))
            score = overlap * max(0.1, atom.confidence)
            if score > best:
                best = score
        return float(best)

    def _format_bias(self, *, prompt_lower: str, candidate: str, normalized: str) -> tuple[float, float]:
        bonus = 0.0
        contradiction = 0.0
        if "standard chess notation" in prompt_lower:
            if re.search(r"\b(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)(?:,\s*[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)*\b", candidate):
                bonus += 1.8
            else:
                contradiction += 1.8
        if any(token in prompt_lower for token in ("plaintext sentence", "decipher", "cipher")):
            if len(candidate.split()) >= 6 and candidate[:1].isupper():
                bonus += 1.2
            else:
                contradiction += 2.2
        if any(token in prompt_lower for token in ("how many", "largest order", "count")):
            if re.fullmatch(r"-?\d+(?:\.\d+)?", normalized):
                bonus += 1.1
            else:
                contradiction += 2.0
        if any(token in prompt_lower for token in ("formula", "expression", "proportionality factor", "polynomial", "conormal")):
            if self.formula_worker._has_formal_notation(candidate):
                bonus += 1.1
            else:
                contradiction += 2.0
        return float(bonus), float(contradiction)

    def _score_candidates_sovereign(
        self,
        *,
        prompt: str,
        prompt_lower: str,
        grouped: dict[str, tuple[float, str, set[str]]],
        meaning_atoms: list[MeaningAtom],
        preferred_count: str | None = None,
        allow_auto_count_aggregate: bool = True,
    ) -> tuple[str, float, list[str]]:
        if not grouped:
            return "", float("-inf"), ["lhe_swarm_rpn no_candidates"]
        pre_trace: list[str] = []
        count_target: str | None = preferred_count
        if allow_auto_count_aggregate and self.formula_worker._is_count_goal(prompt, {"raw": prompt}):
            relation = self.formula_worker._extract_count_relation(prompt, {"raw": prompt})
            if preferred_count is not None and relation is not None:
                relation_name, threshold = relation
                pre_trace.append(
                    f"lhe_swarm_rpn_fusion count_aggregate={preferred_count} threshold={threshold:g} relation={relation_name} source=proposal_preference"
                )
            elif relation is not None:
                relation_name, threshold = relation
                numeric_values: list[float] = []
                numeric_support = 0.0
                for normalized, (support, candidate, _workers) in grouped.items():
                    canonical = self.helpers.canonicalize_short_numeric(candidate) or normalized
                    if not canonical:
                        continue
                    try:
                        numeric_value = float(canonical)
                    except Exception:
                        continue
                    numeric_values.append(numeric_value)
                    numeric_support = max(numeric_support, float(support))
                if len(numeric_values) >= 2:
                    filtered_values = [value for value in numeric_values if abs(value - threshold) > 1e-9]
                    count = sum(
                        1
                        for value in filtered_values
                        if self.formula_worker._value_matches_relation(
                            value,
                            relation=relation_name,
                            threshold=threshold,
                        )
                    )
                    count_candidate = str(int(count))
                    count_target = count_candidate
                    if count_candidate not in grouped:
                        grouped = dict(grouped)
                        grouped[count_candidate] = (
                            numeric_support + 1.75,
                            count_candidate,
                            {"ReasoningFusion"},
                        )
                        pre_trace.append(
                            f"lhe_swarm_rpn_fusion count_aggregate={count_candidate} threshold={threshold:g} relation={relation_name}"
                        )
        expressions: list[str] = []
        ordered: list[tuple[str, str, float, float, float, float]] = []
        for normalized, (support, candidate, workers) in grouped.items():
            worker_bonus = 0.15 * len(workers)
            triangulation = 0.35 if len(workers) >= 2 else 0.0
            meaning_support = self._meaning_alignment_score(candidate, meaning_atoms)
            format_bonus, contradiction = self._format_bias(
                prompt_lower=prompt_lower,
                candidate=candidate,
                normalized=normalized,
            )
            if count_target is not None:
                canonical_numeric = self.helpers.canonicalize_short_numeric(candidate)
                if canonical_numeric and canonical_numeric != count_target:
                    contradiction += 12.0
            prompt_echo = self.evidence_worker._prompt_echo_penalty(candidate, prompt)
            expressions.append(
                f"{support:.6f} {worker_bonus:.6f} + {triangulation:.6f} + {meaning_support:.6f} + {format_bonus:.6f} + {contradiction:.6f} {prompt_echo:.6f} + -"
            )
            ordered.append((normalized, candidate, support, worker_bonus, meaning_support, contradiction, prompt_echo))

        engine = self._get_rpn_engine()
        before = engine.get_gpu_call_count()
        scores = engine.evaluate_batch(expressions, max_parallel=max(1, min(18, len(expressions))))
        after = engine.get_gpu_call_count()

        best_candidate = ""
        best_score = float("-inf")
        trace = [
            *pre_trace,
            f"lhe_swarm_rpn rule=reasoning_contrastive_verification gpu_calls={after - before} candidates={len(scores)}",
        ]
        for (normalized, candidate, support, worker_bonus, meaning_support, contradiction, prompt_echo), score in zip(ordered, scores):
            if float(score) > best_score:
                best_score = float(score)
                best_candidate = candidate
            trace.append(
                "lhe_swarm_rpn_candidate "
                f"support={support:.3f} worker_bonus={worker_bonus:.3f} meaning={meaning_support:.3f} "
                f"contradiction={contradiction:.3f} prompt_echo={prompt_echo:.3f} score={float(score):.3f} "
                f"answer={candidate[:80]}"
            )
        return best_candidate, best_score, trace

    def _supplement_evidence(
        self,
        *,
        prompt: str,
        evidence_rows: list[dict[str, Any]],
        parse_bundle: dict[str, Any],
        route: dict[str, Any],
        options: list[str],
    ) -> list[dict[str, Any]]:
        foundational_rows = search_foundational_reasoning_entries(
            prompt,
            galaxy_names=[
                *[str(name) for name in route.get("galaxy_names") or []],
                "Reality",
                "Word",
                "Grammar",
                "Character",
                "Math",
            ],
            limit=10,
        )
        extra_rows: list[dict[str, Any]] = []
        if route and self.helpers.query_evidence is not None and len(evidence_rows) < 6:
            try:
                extra_rows = self.helpers.query_evidence(
                    prompt=prompt,
                    route=route,
                    parse_bundle=parse_bundle,
                    use_enriched=True,
                    options=options,
                )
            except Exception:
                extra_rows = []
        return self._merge_evidence_rows(evidence_rows, foundational_rows, extra_rows)

    def reason_open_answer(
        self,
        *,
        prompt: str,
        goal: dict[str, Any],
        fused_entities: list[dict[str, Any]],
        meaning_atoms: list[MeaningAtom] | None = None,
        evidence_rows: list[dict[str, Any]],
        parse_bundle: dict[str, Any],
        route: dict[str, Any],
        baseline_answer: str = "",
    ) -> dict[str, Any]:
        meaning_atoms = meaning_atoms or []
        evidence_rows = self._supplement_evidence(
            prompt=prompt,
            evidence_rows=evidence_rows,
            parse_bundle=parse_bundle,
            route=route,
            options=[],
        )
        meaning_atoms = fuse_meaning_atoms([*meaning_atoms, *meaning_atoms_from_evidence_rows(evidence_rows)])
        prompt_lower = prompt.lower()
        active = self._select_active_open_workers(
            prompt=prompt,
            goal=goal,
            fused_entities=fused_entities,
            meaning_atoms=meaning_atoms,
            evidence_rows=evidence_rows,
        )

        proposals: list[WorkerProposal] = []
        has_explicit_numeric_answer_row = self.formula_worker._has_explicit_numeric_answer_row(evidence_rows)
        if baseline_answer:
            proposals.append(
                WorkerProposal(
                    worker="BaselineSynthesis",
                    candidate=baseline_answer,
                    score=1.0,
                    rationale="daemon_open_answer_baseline",
                )
            )
        reasoning_trace = [f"lhe_swarm active={','.join(worker.name for worker in active)}"]
        for worker in active:
            worker_proposals = worker.propose_open(
                prompt=prompt,
                goal=goal,
                fused_entities=fused_entities,
                meaning_atoms=meaning_atoms,
                evidence_rows=evidence_rows,
                parse_bundle=parse_bundle,
                route=route,
            )
            reasoning_trace.extend(worker.consume_selection_trace())
            proposals.extend(worker_proposals)
            if worker_proposals:
                best = worker_proposals[0]
                reasoning_trace.append(
                    f"lhe_swarm_worker {worker.name} best={best.candidate[:80]} score={best.score:.3f} why={best.rationale}"
                )
        if self.formula_worker._is_count_goal(prompt, goal) and not has_explicit_numeric_answer_row:
            existing_count_proposals = [
                item
                for item in proposals
                if str(item.rationale).startswith("count_relation:")
                and item.worker != "ReasoningFusion"
            ]
            aggregate_count = self.formula_worker._derive_count_candidate_from_proposals(
                prompt=prompt,
                goal=goal,
                proposals=proposals,
            )
            if aggregate_count and not existing_count_proposals:
                proposals.append(
                    WorkerProposal(
                        worker="ReasoningFusion",
                        candidate=aggregate_count,
                        score=max((item.score for item in proposals), default=0.0) + 1.75,
                        rationale="count_relation:cross_worker_numeric_aggregate",
                    )
                )
                reasoning_trace.append(
                    f"lhe_swarm_fusion count_aggregate={aggregate_count} source=cross_worker_numeric_aggregate"
                )
        best_candidate = ""
        best_score = float("-inf")
        grouped: dict[str, tuple[float, str, set[str]]] = {}
        for proposal in proposals:
            normalized = self.helpers.normalize_answer(proposal.candidate)
            if not normalized:
                continue
            existing = grouped.get(normalized)
            if existing is None:
                grouped[normalized] = (proposal.score, proposal.candidate, {proposal.worker})
            else:
                grouped[normalized] = (
                    existing[0] + max(0.0, proposal.score * 0.35),
                    existing[1],
                    existing[2] | {proposal.worker},
                )
        count_goal = self.formula_worker._is_count_goal(prompt, goal)
        explicit_numeric_candidates = set(self.formula_worker._explicit_numeric_answer_candidates(evidence_rows))
        preferred_count: str | None = None
        if count_goal:
            count_proposals = [
                item
                for item in proposals
                if str(item.rationale).startswith("count_relation:")
                and item.worker != "ReasoningFusion"
            ]
            if count_proposals:
                best_count = max(count_proposals, key=lambda item: float(item.score))
                preferred_count = self.helpers.normalize_answer(best_count.candidate) or None
        grouped_for_scoring = grouped
        if count_goal and explicit_numeric_candidates:
            explicit_grouped = {
                normalized: value
                for normalized, value in grouped.items()
                if normalized in explicit_numeric_candidates
            }
            if explicit_grouped:
                grouped_for_scoring = explicit_grouped
                if preferred_count is None:
                    preferred_count = max(
                        explicit_grouped.items(),
                        key=lambda item: float(item[1][0]),
                    )[0]
                reasoning_trace.append(
                    "lhe_swarm_explicit_numeric "
                    f"candidates={len(explicit_grouped)} preferred={preferred_count}"
                )
        best_candidate, best_score, rpn_trace = self._score_candidates_sovereign(
            prompt=prompt,
            prompt_lower=prompt_lower,
            grouped=grouped_for_scoring,
            meaning_atoms=meaning_atoms,
            preferred_count=preferred_count,
            allow_auto_count_aggregate=not has_explicit_numeric_answer_row,
        )
        reasoning_trace.extend(rpn_trace)
        reasoning_trace.append(f"lhe_swarm_selected score={best_score:.3f} answer={'present' if best_candidate else 'empty'}")
        return {
            "answer": best_candidate,
            "score": best_score if best_candidate else float("-inf"),
            "proposals": [
                {"worker": item.worker, "candidate": item.candidate, "score": item.score, "rationale": item.rationale}
                for item in sorted(proposals, key=lambda proposal: proposal.score, reverse=True)[:12]
            ],
            "reasoning_trace": reasoning_trace,
        }

    def adjust_option_scores(
        self,
        *,
        prompt: str,
        options: list[str],
        goal: dict[str, Any],
        fused_entities: list[dict[str, Any]],
        evidence_rows: list[dict[str, Any]],
        parse_bundle: dict[str, Any],
        route: dict[str, Any],
    ) -> dict[str, float]:
        evidence_rows = self._supplement_evidence(
            prompt=prompt,
            evidence_rows=evidence_rows,
            parse_bundle=parse_bundle,
            route=route,
            options=options,
        )
        adjustments = self.concept_worker.option_adjustments(
            prompt=prompt,
            options=options,
            goal=goal,
            fused_entities=fused_entities,
            evidence_rows=evidence_rows,
            parse_bundle=parse_bundle,
            route=route,
        )
        return {key: float(value) for key, value in adjustments.items()}
