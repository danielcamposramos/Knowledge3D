"""Math competitions benchmark integration for Knowledgeverse Week 14."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from knowledge3d.bridge.headless_tablet import HeadlessTabletMPC, TabletIngest
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.trm_navigator import TRMNavigator


class MathCompetitionBenchmark:
    """Benchmark AMC/AIME/IMO style prompts with empty/enriched comparison."""

    def __init__(
        self,
        knowledgeverse: Knowledgeverse | None = None,
        dataset_path: str | Path | None = None,
        max_problems: int | None = None,
        query_scope_galaxies: str | list[str] | None = None,
        runtime_seed_knowledge: bool = False,
        tablet_boundary: HeadlessTabletMPC | None = None,
    ):
        self.kv = knowledgeverse or Knowledgeverse()
        self.dataset_path = self._resolve_dataset_path(dataset_path)
        self.max_problems = max_problems
        self.query_scope_galaxies = self._normalize_query_scope(query_scope_galaxies)
        self.runtime_seed_knowledge = bool(runtime_seed_knowledge)
        self.tablet_boundary = tablet_boundary
        self.dataset_sources: list[str] = []
        self.problems = self._load_problems()
        self.results: list[dict[str, Any]] = []

    def _resolve_dataset_path(self, dataset_path: str | Path | None) -> Path:
        if dataset_path is not None:
            return Path(dataset_path)
        candidates = [
            Path("/K3D/K3D_llama_cpp/datasets"),
            Path("/K3D/K3D_llama_cpp/datasets/GSM8K"),
            Path("/K3D/K3D_llama_cpp/datasets/math"),
            Path("/K3D/Knowledge3D.local/datasets/math_competitions"),
            Path("../Knowledge3D.local/datasets/math_competitions"),
            Path("data"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return Path("")

    def _load_problems(self) -> list[dict[str, Any]]:
        if self.dataset_path and self.dataset_path.exists():
            staged = self._load_from_present_datasets(self.dataset_path, limit=self.max_problems)
            if staged:
                return staged
        fallback = self._load_from_calculus_microbench()
        if fallback:
            return fallback[: self.max_problems] if self.max_problems is not None else fallback
        synthetic = self._synthetic_problems()
        return synthetic[: self.max_problems] if self.max_problems is not None else synthetic

    def _load_from_present_datasets(self, root: Path, limit: int | None = None) -> list[dict[str, Any]]:
        batches: list[list[dict[str, Any]]] = []
        for loader in (
            self._load_from_competition_files,
            self._load_from_gsm8k,
            self._load_from_math_dataset,
        ):
            batch = loader(root, limit=limit)
            if batch:
                batches.append(batch)
        if limit is None:
            return [row for batch in batches for row in batch]
        out: list[dict[str, Any]] = []
        offset = 0
        while len(out) < int(limit):
            progressed = False
            for batch in batches:
                if offset < len(batch):
                    out.append(batch[offset])
                    progressed = True
                    if len(out) >= int(limit):
                        break
            if not progressed:
                break
            offset += 1
        return out

    def _load_from_competition_files(self, root: Path, limit: int | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        sources = [
            ("AMC", root / "amc_problems.json"),
            ("AIME", root / "aime_problems.json"),
            ("IMO", root / "imo_problems.json"),
        ]
        for competition, source in sources:
            if not source.exists():
                continue
            try:
                payload = json.loads(source.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict):
                records = payload.get("problems", [])
            else:
                records = payload
            if not isinstance(records, list):
                continue
            for idx, record in enumerate(records):
                if limit is not None and len(out) >= int(limit):
                    return out
                if not isinstance(record, dict):
                    continue
                text = str(record.get("problem_text") or record.get("question") or "").strip()
                answer = record.get("answer")
                if not text or answer is None:
                    continue
                out.append(
                    {
                        "id": str(record.get("id") or f"{competition.lower()}_{idx}"),
                        "competition": competition,
                        "problem_text": text,
                        "answer": answer,
                    }
                )
        if out:
            self.dataset_sources.append("competition_json")
        return out

    def _load_from_gsm8k(self, root: Path, limit: int | None = None) -> list[dict[str, Any]]:
        candidates = [
            root / "grade_school_math" / "data" / "test.jsonl",
            root / "GSM8K" / "grade_school_math" / "data" / "test.jsonl",
            root / "grade_school_math" / "data" / "train.jsonl",
            root / "GSM8K" / "grade_school_math" / "data" / "train.jsonl",
        ]
        for path in candidates:
            if not path.exists():
                continue
            out: list[dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as handle:
                for idx, line in enumerate(handle):
                    if limit is not None and len(out) >= int(limit):
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = str(payload.get("question") or "").strip()
                    answer = self._extract_gsm8k_answer(payload.get("answer"))
                    if not text or answer is None:
                        continue
                    out.append(
                        {
                            "id": f"gsm8k_{idx}",
                            "competition": "GSM8K",
                            "problem_text": text,
                            "answer": answer,
                        }
                    )
            if out:
                self.dataset_sources.append(str(path))
                return out
        return []

    def _load_from_math_dataset(self, root: Path, limit: int | None = None) -> list[dict[str, Any]]:
        candidates = [
            root / "data" / "train.jsonl",
            root / "math" / "data" / "train.jsonl",
            root / "data_train.jsonl",
            root / "math" / "data_train.jsonl",
        ]
        for path in candidates:
            if not path.exists():
                continue
            out: list[dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as handle:
                for idx, line in enumerate(handle):
                    if limit is not None and len(out) >= int(limit):
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = str(payload.get("problem") or "").strip()
                    answer = self._extract_math_answer(payload.get("solution"))
                    if not text or answer is None:
                        continue
                    math_type = str(payload.get("type") or "MATH").strip() or "MATH"
                    out.append(
                        {
                            "id": f"math_{idx}",
                            "competition": f"MATH:{math_type}",
                            "problem_text": text,
                            "answer": answer,
                        }
                    )
            if out:
                self.dataset_sources.append(str(path))
                return out
        return []

    def _load_from_calculus_microbench(self) -> list[dict[str, Any]]:
        paths = [
            Path("data/calculus_microbench.jsonl"),
            Path("../Knowledge3D.local/datasets/calculus_microbench.jsonl"),
        ]
        records: list[dict[str, Any]] = []
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for idx, line in enumerate(handle):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = str(payload.get("problem", "")).strip()
                    answer = payload.get("answer")
                    if not text or answer is None:
                        continue
                    records.append(
                        {
                            "id": str(payload.get("id") or f"calculus_{idx}"),
                            "competition": "AMC",
                            "problem_text": text,
                            "answer": answer,
                        }
                    )
        return records

    def _synthetic_problems(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "amc_poly_1",
                "competition": "AMC",
                "problem_text": "Find derivative of x^2 + 4x at x=3",
                "answer": "10",
            },
            {
                "id": "amc_quotient_1",
                "competition": "AMC",
                "problem_text": "Find f'(1) where f(x) = (3x-4)/(2x+3)",
                "answer": "0.68",
            },
            {
                "id": "aime_basic_1",
                "competition": "AIME",
                "problem_text": "Compute 12 * (3 + 2)",
                "answer": "60",
            },
            {
                "id": "imo_basic_1",
                "competition": "IMO",
                "problem_text": "Evaluate derivative of cos(x) at x=0",
                "answer": "0",
            },
        ]

    def run_benchmark(self, use_enriched: bool = True) -> dict[str, Any]:
        self.results = []
        navigator = TRMNavigator(knowledgeverse=self.kv)
        by_competition: dict[str, dict[str, Any]] = {}
        for problem in self.problems:
            result = self._solve_problem(navigator=navigator, problem=problem, use_enriched=use_enriched)
            self.results.append(result)
            comp = result["competition"]
            if comp not in by_competition:
                by_competition[comp] = {"total": 0, "correct": 0, "results": []}
            by_competition[comp]["total"] += 1
            if result["correct"]:
                by_competition[comp]["correct"] += 1
            by_competition[comp]["results"].append(result)
        for comp_data in by_competition.values():
            total = comp_data["total"]
            comp_data["accuracy"] = (comp_data["correct"] / total) if total else 0.0
        total_correct = sum(row["correct"] for row in self.results)
        total_count = len(self.results)
        pred_none_count = sum(1 for row in self.results if row.get("predicted_answer") is None)
        pred_numeric_count = sum(1 for row in self.results if self._to_float(row.get("predicted_answer")) is not None)
        exp_numeric_count = sum(1 for row in self.results if self._to_float(row.get("expected_answer")) is not None)
        route_specialists: dict[str, int] = {}
        failure_reason_counts: dict[str, int] = {}
        for row in self.results:
            specialist = str(row.get("route", {}).get("specialist", "unknown"))
            route_specialists[specialist] = int(route_specialists.get(specialist, 0)) + 1
            if row.get("predicted_answer") is None:
                reason = str(row.get("failure_reason", "") or "unknown")
                failure_reason_counts[reason] = int(failure_reason_counts.get(reason, 0)) + 1
        return {
            "benchmark": "Math Competitions",
            "dataset_path": str(self.dataset_path) if self.dataset_path else "synthetic",
            "dataset_sources": list(self.dataset_sources),
            "use_enriched": use_enriched,
            "results_by_competition": by_competition,
            "overall_accuracy": (total_correct / total_count) if total_count else 0.0,
            "total": total_count,
            "correct": total_correct,
            "diagnostics": {
                "predicted_none_count": int(pred_none_count),
                "predicted_none_rate": (pred_none_count / total_count) if total_count else 0.0,
                "predicted_numeric_count": int(pred_numeric_count),
                "expected_numeric_count": int(exp_numeric_count),
                "route_specialist_counts": route_specialists,
                "failure_reason_counts": failure_reason_counts,
            },
        }

    def _solve_problem(
        self,
        *,
        navigator: TRMNavigator,
        problem: dict[str, Any],
        use_enriched: bool,
    ) -> dict[str, Any]:
        navigator.clear_trace()
        generated_entry: dict[str, Any] | None = None
        if self.tablet_boundary is not None:
            return self._solve_problem_via_tablet(problem=problem, use_enriched=use_enriched)
        if use_enriched and self.runtime_seed_knowledge:
            self._seed_math_knowledge(problem)
            if self._should_attempt_autonomous_generation(str(problem["problem_text"])):
                generated_entry = navigator.generate_from_procedural(
                    query=str(problem["problem_text"]),
                    source_galaxy="Reality",
                    target_galaxy="Math",
                    store_result=True,
                )
        composed = navigator.navigate_and_compose(
            query=str(problem["problem_text"]),
            specialist="auto",
            domain_hint="math",
            use_enriched=use_enriched,
        )
        route = dict(
            composed.get(
                "route",
                navigator.route(
                    query=str(problem["problem_text"]),
                    specialist="auto",
                    domain_hint="math",
                    galaxy_names=self.query_scope_galaxies,
                ),
            )
        )
        route = self._apply_query_scope(route)
        patterns = navigator.query(
            query=str(problem["problem_text"]),
            galaxy_names=route["galaxy_names"],
            top_k=30 if use_enriched else 5,
            specialist=route["specialist"],
            domain_hint=route["domain"],
        )
        predicted = navigator.execute(composed)
        reasoning_trace = navigator.get_reasoning_trace()
        missing_signal = None
        if hasattr(navigator, "get_last_math_missing_signal"):
            try:
                missing_signal = navigator.get_last_math_missing_signal()
            except Exception:
                missing_signal = None
        expected = problem["answer"]
        correct = self._answers_match(predicted, expected)
        self.kv.log_event(
            "math_problem_solved" if correct else "math_problem_failed",
            {
                "specialist": route["specialist"],
                "confidence": 0.9 if correct else 0.35,
                "competition": problem["competition"],
            },
        )
        return {
            "problem_id": problem["id"],
            "competition": problem["competition"],
            "correct": int(correct),
            "predicted_answer": predicted,
            "expected_answer": expected,
            "failure_reason": (
                str(missing_signal.get("reason", ""))
                if isinstance(missing_signal, dict)
                else self._extract_failure_reason(reasoning_trace)
            ),
            "failure_signal": (dict(missing_signal) if isinstance(missing_signal, dict) else None),
            "symbols_used": int(composed.get("patterns_used", len(patterns))),
            "reasoning_trace": reasoning_trace,
            "route": route,
            "meta_specialist": composed.get("meta_specialist"),
            "method": (
                "autonomous_generation+navigation"
                if generated_entry and "error" not in generated_entry
                else "navigation"
            ),
            "generated_id": (
                str(generated_entry.get("id", "")) if generated_entry and "error" not in generated_entry else None
            ),
        }

    def _solve_problem_via_tablet(
        self,
        *,
        problem: dict[str, Any],
        use_enriched: bool,
    ) -> dict[str, Any]:
        envelope = TabletIngest.math_problem(
            task_id=str(problem["id"]),
            question=str(problem["problem_text"]),
            competition=str(problem.get("competition") or ""),
            expected_answer=problem.get("answer"),
        )
        tablet_result = self.tablet_boundary.submit(envelope, use_enriched=use_enriched)
        emitted = dict(tablet_result["emitted"])
        route = emitted.get("route", {})
        predicted = emitted.get("predicted_answer")
        expected = problem["answer"]
        correct = bool(emitted.get("correct", False))
        self.kv.log_event(
            "math_problem_solved" if correct else "math_problem_failed",
            {
                "specialist": route.get("specialist", "math"),
                "confidence": 0.9 if correct else 0.35,
                "competition": problem["competition"],
            },
        )
        return {
            "problem_id": problem["id"],
            "competition": problem["competition"],
            "correct": int(correct),
            "predicted_answer": predicted,
            "expected_answer": expected,
            "failure_reason": "" if predicted is not None else "tablet_boundary_no_result",
            "failure_signal": None,
            "symbols_used": 0,
            "reasoning_trace": emitted.get("task_result", {}).get("reasoning_trace", []),
            "route": route,
            "meta_specialist": route.get("specialist"),
            "method": "tablet_boundary",
            "generated_id": None,
            "tablet_contract": tablet_result["tablet_contract"],
        }

    def _extract_failure_reason(self, reasoning_trace: list[str]) -> str:
        for item in reversed(reasoning_trace):
            if not isinstance(item, str):
                continue
            marker = "math_solve_missing reason="
            if marker in item:
                return item.split(marker, 1)[1].strip() or "unknown"
        return "unknown"

    def _normalize_query_scope(self, value: str | list[str] | None) -> list[str] | None:
        if isinstance(value, list):
            raw = [str(item).strip() for item in value]
        elif isinstance(value, str):
            raw = [segment.strip() for segment in value.split(",")]
        else:
            raw = []
        out: list[str] = []
        seen: set[str] = set()
        for item in raw:
            if not item:
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out or None

    def _apply_query_scope(self, route: dict[str, Any]) -> dict[str, Any]:
        if not self.query_scope_galaxies:
            return route
        route_names = [str(name) for name in route.get("galaxy_names") or [] if str(name).strip()]
        if not route_names:
            route["galaxy_names"] = list(self.query_scope_galaxies)
            return route
        scope_keys = {name.lower() for name in self.query_scope_galaxies}
        filtered = [name for name in route_names if name.lower() in scope_keys]
        route["galaxy_names"] = filtered if filtered else list(self.query_scope_galaxies)
        return route

    def _seed_math_knowledge(self, problem: dict[str, Any]) -> None:
        self.kv.galaxy_manager.add_entry(
            "Math",
            {
                "domain": "math",
                "competition": problem["competition"],
                "problem_id": problem["id"],
                "kind": "symbolic_pattern",
            },
        )
        self.kv.galaxy_manager.add_entry(
            "Grammar",
            {
                "domain": "math",
                "problem_id": problem["id"],
                "kind": "derivation_rule",
            },
        )

    def _answers_match(self, predicted: Any, expected: Any) -> bool:
        pred = self._to_float(predicted)
        exp = self._to_float(expected)
        if pred is not None and exp is not None:
            return abs(pred - exp) <= 1e-3
        return self._normalize_text_answer(predicted) == self._normalize_text_answer(expected)

    def _to_float(self, value: Any) -> float | None:
        try:
            return float(value)
        except Exception:
            try:
                cleaned = str(value).strip().replace(",", "").replace("\\", "")
                if cleaned.endswith("%"):
                    cleaned = cleaned[:-1]
                if cleaned.startswith("$"):
                    cleaned = cleaned[1:]
                return float(cleaned)
            except Exception:
                return None

    def _normalize_text_answer(self, value: Any) -> str:
        return (
            str(value)
            .strip()
            .lower()
            .replace("\\", "")
            .replace(" ", "")
        )

    def _extract_gsm8k_answer(self, raw_answer: Any) -> str | None:
        text = str(raw_answer or "").strip()
        if not text:
            return None
        match = re.search(r"####\s*([^\n]+)", text)
        if match:
            return match.group(1).strip().replace(",", "")
        return text.splitlines()[-1].strip() or None

    def _extract_math_answer(self, solution: Any) -> str | None:
        text = str(solution or "").strip()
        if not text:
            return None
        boxed = self._extract_last_boxed(text)
        if boxed:
            return boxed.strip()
        match = re.search(r"####\s*([^\n]+)", text)
        if match:
            return match.group(1).strip()
        tail = text.splitlines()[-1].strip()
        tail = tail.rstrip(".")
        eq_match = re.search(r"=\s*([^=]+)$", tail)
        if eq_match:
            return eq_match.group(1).strip()
        return tail or None

    def _extract_last_boxed(self, text: str) -> str | None:
        markers = (r"\boxed{", r"\fbox{")
        last_pos = -1
        marker_used = ""
        for marker in markers:
            pos = text.rfind(marker)
            if pos > last_pos:
                last_pos = pos
                marker_used = marker
        if last_pos < 0:
            return None
        start = last_pos + len(marker_used)
        depth = 1
        chars: list[str] = []
        for ch in text[start:]:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return "".join(chars)
            chars.append(ch)
        return None

    def _should_attempt_autonomous_generation(self, text: str) -> bool:
        lowered = text.lower()
        triggers = (
            "differential",
            "rate of change",
            "decay",
            "growth",
            "velocity",
            "acceleration",
            "projectile",
            "pendulum",
            "field",
            "thermo",
        )
        return any(token in lowered for token in triggers)

    def save_results(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark": "Math Competitions",
            "total": len(self.results),
            "correct": sum(int(row.get("correct", 0)) for row in self.results),
            "accuracy": (
                sum(int(row.get("correct", 0)) for row in self.results) / len(self.results)
                if self.results
                else 0.0
            ),
            "results": self.results,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
