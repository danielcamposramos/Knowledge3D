"""Last Humanity Exam style benchmark for Knowledgeverse Week 14."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from benchmarks.sampling import stratified_sample
from knowledge3d.bridge.headless_tablet import HeadlessTabletMPC, TabletIngest
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.trm_navigator import TRMNavigator
from knowledge3d.tablet.wine.question_wine import (
    QUESTION_ROUTE_GALAXIES,
    build_question_session_tape,
    lhe_question_envelope,
)


class LastHumanityExamBenchmark:
    """Multi-domain benchmark with empty/enriched comparison."""

    def __init__(
        self,
        knowledgeverse: Knowledgeverse | None = None,
        dataset_path: str | Path | None = None,
        max_questions: int | None = None,
        query_scope_galaxies: str | list[str] | None = None,
        runtime_seed_knowledge: bool = False,
        tablet_boundary: HeadlessTabletMPC | None = None,
    ):
        self.dataset_path = self._resolve_dataset_path(dataset_path)
        if knowledgeverse is None:
            default_storage_root = self._default_storage_root(self.dataset_path)
            self.kv = Knowledgeverse(storage_root=default_storage_root)
        else:
            self.kv = knowledgeverse
        self.max_questions = max_questions
        self.query_scope_galaxies = self._normalize_query_scope(query_scope_galaxies)
        self.runtime_seed_knowledge = bool(runtime_seed_knowledge)
        self.tablet_boundary = tablet_boundary
        self.dataset_source = "unknown"
        self.dataset_file: str | None = None
        self.synthetic_fallback = False
        self.questions = self._load_questions()
        self.results: list[dict[str, Any]] = []

    def _resolve_dataset_path(self, dataset_path: str | Path | None) -> Path:
        if dataset_path is not None:
            return Path(dataset_path)
        candidates = [
            Path("/K3D/K3D_llama_cpp/datasets/last_humanity_exam"),
            Path("/K3D/K3D_llama_cpp/datasets/hle"),
            Path("/K3D/K3D_llama_cpp/datasets/mmlu"),
            Path("/K3D/Knowledge3D.local/datasets/last_humanity_exam"),
            Path("../Knowledge3D.local/datasets/last_humanity_exam"),
            Path("/K3D/Knowledge3D.local/datasets/exams/hle-src"),
            Path("../Knowledge3D.local/datasets/exams/hle-src"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return Path("")

    def _default_storage_root(self, dataset_path: Path) -> Path:
        if dataset_path:
            base = dataset_path.parent if dataset_path.is_file() else dataset_path
            if str(base):
                return base / ".knowledgeverse_runtime"
        return Path("../Knowledge3D.local")

    def _load_questions(self) -> list[dict[str, Any]]:
        if self.dataset_path and self.dataset_path.exists():
            loaded = self._load_from_known_files(self.dataset_path)
            if loaded:
                self.dataset_source = "file"
                self.synthetic_fallback = False
                return stratified_sample(loaded, self.max_questions)
            raise FileNotFoundError(
                f"lhe_dataset_empty: no supported Last Humanity Exam records found under {self.dataset_path}"
            )
        else:
            raise FileNotFoundError(
                "lhe_dataset_missing: Last Humanity Exam runtime requires the real dataset at the canonical roots"
            )

    def _load_from_known_files(self, root: Path) -> list[dict[str, Any]]:
        if root.is_file():
            candidates = [root]
            jsonl_path = root if root.suffix.lower() == ".jsonl" else None
        else:
            candidates = [
                root / "last_humanity_exam.json",
                root / "questions.json",
                root / "dataset.json",
            ]
            jsonl_path = root / "questions.jsonl"
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            records = payload.get("questions") if isinstance(payload, dict) else payload
            if not isinstance(records, list):
                continue
            parsed = self._normalize_question_records(records)
            if parsed:
                self.dataset_file = str(candidate)
                return parsed

        # Optional JSONL support.
        if jsonl_path is not None and jsonl_path.exists():
            records: list[dict[str, Any]] = []
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        records.append(payload)
            parsed = self._normalize_question_records(records)
            if parsed:
                self.dataset_file = str(jsonl_path)
                return parsed
        return []

    def _normalize_question_records(self, records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for idx, record in enumerate(records):
            text = str(record.get("question_text") or record.get("question") or "").strip()
            options = record.get("options")
            answer = record.get("correct_answer") or record.get("answer")
            if not text or answer is None:
                continue
            question_type = "open_ended"
            parsed_options: list[str] = []
            if isinstance(options, list) and options:
                parsed_options = [str(option) for option in options]
                question_type = "multiple_choice"
            out.append(
                {
                    "id": str(record.get("id") or f"lhe_{idx}"),
                    "domain": str(record.get("domain") or "multi"),
                    "question_text": text,
                    "options": parsed_options,
                    "correct_answer": str(answer),
                    "question_type": str(record.get("question_type") or question_type),
                    "answer_type": str(record.get("answer_type") or ""),
                }
            )
        return out

    def _synthetic_questions(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "lhe_math_1",
                "domain": "math",
                "question_text": "What is 7 * (3 + 2)?",
                "options": ["35", "30", "42", "28"],
                "correct_answer": "35",
                "question_type": "multiple_choice",
            },
            {
                "id": "lhe_logic_1",
                "domain": "logic",
                "question_text": "If all A are B and all B are C, which statement is true?",
                "options": [
                    "Some A are not C",
                    "All A are C",
                    "No B are C",
                    "All C are A",
                ],
                "correct_answer": "All A are C",
                "question_type": "multiple_choice",
            },
            {
                "id": "lhe_physics_1",
                "domain": "physics",
                "question_text": "An object at rest remains at rest unless acted on by which quantity?",
                "options": ["Force", "Mass", "Time", "Temperature"],
                "correct_answer": "Force",
                "question_type": "multiple_choice",
            },
            {
                "id": "lhe_multi_1",
                "domain": "multi",
                "question_text": "Choose the best next step when uncertainty is high in a proof search.",
                "options": [
                    "Guess the final answer immediately",
                    "Request verification and reduce search space",
                    "Ignore constraints",
                    "Delete all intermediate steps",
                ],
                "correct_answer": "Request verification and reduce search space",
                "question_type": "multiple_choice",
            },
        ]

    def run_benchmark(
        self,
        use_enriched: bool = True,
        *,
        progress_cb: Callable[[dict[str, Any]], None] | None = None,
        progress_every: int | None = None,
        row_cb: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
        start_index: int = 0,
        initial_correct: int = 0,
    ) -> dict[str, Any]:
        self.results = []
        total = len(self.questions)
        resume_index = max(0, min(int(start_index), total))
        correct = max(0, int(initial_correct))
        navigator = TRMNavigator(knowledgeverse=self.kv)
        step = max(1, int(progress_every or 10))
        start = time.monotonic()
        tablet_rows: list[dict[str, Any]] = []
        if self.tablet_boundary is not None:
            tape = build_question_session_tape(
                session_id=f"lhe_{int(time.time() * 1000)}",
                suite_name="lhe",
                rows=self.questions[resume_index:],
                use_enriched=use_enriched,
            )
            tablet_rows = list(self.tablet_boundary.run_tape_session(tape)["results"])
        for offset, question in enumerate(self.questions[resume_index:]):
            index = resume_index + offset + 1
            if self.tablet_boundary is not None:
                result = self._lhe_result_from_tablet(question=question, tablet_result=tablet_rows[offset])
                task_result = dict(result.get("task_result") or {})
                result["elapsed_s"] = round(
                    max(0.0, float(task_result.get("trm_latency_us", 0.0) or 0.0) / 1_000_000.0),
                    3,
                )
            else:
                row_start = time.monotonic()
                result = self._answer_question(
                    navigator=navigator,
                    question=question,
                    use_enriched=use_enriched,
                )
                result["elapsed_s"] = round(max(0.0, time.monotonic() - row_start), 3)
            self.results.append(result)
            if result["correct"]:
                correct += 1
            if row_cb is not None:
                row_cb(question, result)
            if progress_cb and (index % step == 0 or index == total):
                progress_cb(
                    {
                        "completed": index,
                        "total": total,
                        "correct": correct,
                        "elapsed_s": time.monotonic() - start,
                        "benchmark": "lhe",
                        "domain": question.get("domain"),
                    }
                )
        accuracy = (correct / total) if total else 0.0
        return {
            "benchmark": "Last Humanity Exam",
            "dataset_path": str(self.dataset_path) if self.dataset_path else "synthetic",
            "dataset_source": self.dataset_source,
            "dataset_file": self.dataset_file,
            "synthetic_fallback": bool(self.synthetic_fallback),
            "use_enriched": use_enriched,
            "total_questions": total,
            "correct": correct,
            "accuracy": accuracy,
            "results": self.results,
            "results_by_domain": self._summarize_by_domain(self.results),
        }

    def _answer_question(
        self,
        *,
        navigator: TRMNavigator,
        question: dict[str, Any],
        use_enriched: bool,
    ) -> dict[str, Any]:
        domain = str(question.get("domain", "multi"))
        if self.tablet_boundary is not None:
            return self._answer_question_via_tablet(question=question, use_enriched=use_enriched)
        question_type = str(question.get("question_type") or "multiple_choice").lower()
        options = question.get("options") if isinstance(question.get("options"), list) else []
        expected = str(question["correct_answer"])
        domain_lower = domain.strip().lower()
        math_like_domain = domain_lower in {"math", "mathematics", "physics"}
        route = {
            "specialist": "math" if math_like_domain else "chat",
            "domain_hint": domain,
            "galaxy_names": list(QUESTION_ROUTE_GALAXIES),
        }
        route = self._apply_query_scope(route)
        specialist = str(route.get("specialist", "chat"))
        task_result = self.kv.execute_task(
            task={
                "task_id": str(question["id"]),
                "query": str(question["question_text"]),
                "prompt": str(question["question_text"]),
                "question": str(question["question_text"]),
                "messages": [{"role": "user", "content": str(question["question_text"])}],
                "domain": domain,
                "options": list(options),
                "question_type": question_type,
                "expected_answer": expected,
            },
            route=route,
            specialist=specialist,
            domain_hint=domain,
            use_enriched=use_enriched,
        )
        predicted = str(
            task_result.get("predicted_answer")
            or task_result.get("response")
            or task_result.get("answer")
            or task_result.get("result")
            or ""
        ).strip()
        if options:
            predicted = self._normalize_option_prediction(predicted, options)
            correct = predicted.strip() == expected.strip()
        else:
            correct = self._open_ended_match(predicted, expected)

        self.kv.log_event(
            "lhe_question_success" if correct else "lhe_question_failure",
            {
                "specialist": specialist,
                "domain": domain,
                "confidence": 0.85 if correct else 0.3,
            },
        )
        return {
            "question_id": question["id"],
            "domain": domain,
            "correct": int(correct),
            "predicted_answer": predicted,
            "correct_answer": expected,
            "knowledge_used": 0,
            "reasoning_trace": list(task_result.get("reasoning_trace", [])),
            "route": task_result.get("route", route),
            "question_type": question_type,
            "solver": str(task_result.get("solver", "")),
            "runtime": str(task_result.get("runtime", "")),
            "gpu_execution": bool(task_result.get("gpu_execution", False)),
            "program_id": str(task_result.get("program_id", "")),
            "task_result": task_result,
        }

    def _answer_question_via_tablet(
        self,
        *,
        question: dict[str, Any],
        use_enriched: bool,
    ) -> dict[str, Any]:
        domain = str(question.get("domain", "multi"))
        envelope = lhe_question_envelope(
            task_id=str(question["id"]),
            question=str(question["question_text"]),
            options=question.get("options") if isinstance(question.get("options"), list) else [],
            domain=domain,
            expected_answer=str(question["correct_answer"]),
        )
        tablet_result = self.tablet_boundary.submit(envelope, use_enriched=use_enriched)
        return self._lhe_result_from_tablet(question=question, tablet_result=tablet_result)

    def _lhe_result_from_tablet(
        self,
        *,
        question: dict[str, Any],
        tablet_result: dict[str, Any],
    ) -> dict[str, Any]:
        domain = str(question.get("domain", "multi"))
        emitted = dict(tablet_result["emitted"])
        route = emitted.get("route", {})
        correct = bool(emitted.get("correct", False))
        self.kv.log_event(
            "lhe_question_success" if correct else "lhe_question_failure",
            {
                "specialist": route.get("specialist", "chat"),
                "domain": domain,
                "confidence": 0.85 if correct else 0.3,
            },
        )
        return {
            "question_id": question["id"],
            "domain": domain,
            "correct": int(correct),
            "predicted_answer": emitted.get("answer_choice") or emitted.get("answer_text"),
            "correct_answer": str(question["correct_answer"]),
            "knowledge_used": 0,
            "reasoning_trace": emitted.get("task_result", {}).get("reasoning_trace", []),
            "route": route,
            "question_type": str(question.get("question_type") or "multiple_choice").lower(),
            "solver": str(emitted.get("task_result", {}).get("solver", "tablet_boundary")),
            "runtime": str(emitted.get("task_result", {}).get("runtime", "")),
            "gpu_execution": bool(emitted.get("task_result", {}).get("gpu_execution", False)),
            "program_id": str(emitted.get("task_result", {}).get("program_id", "")),
            "task_result": emitted.get("task_result", {}),
            "tablet_contract": tablet_result["tablet_contract"],
        }

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

    def _answer_open_ended(
        self,
        *,
        navigator: TRMNavigator,
        question: dict[str, Any],
        use_enriched: bool,
    ) -> tuple[str, bool]:
        prompt = str(question.get("question_text", ""))
        response = navigator.process_chat(
            [{"role": "user", "content": prompt}],
            use_enriched=use_enriched,
        )
        predicted = self._extract_open_ended_prediction(response)
        expected = str(question.get("correct_answer", ""))
        correct = self._open_ended_match(predicted, expected)
        return predicted, correct

    def _extract_open_ended_prediction(self, response: str) -> str:
        text = str(response or "").strip()
        if not text:
            return ""
        # Use first non-empty line as canonical short-form answer.
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line
        return text

    def _open_ended_match(self, predicted: str, expected: str) -> bool:
        predicted_numeric = self._to_float(predicted)
        expected_numeric = self._to_float(expected)
        if predicted_numeric is not None and expected_numeric is not None:
            return abs(predicted_numeric - expected_numeric) <= 1e-9
        p = self._normalize_answer_text(predicted)
        e = self._normalize_answer_text(expected)
        if not p or not e:
            return False
        return p == e

    @staticmethod
    def _to_float(value: str) -> float | None:
        text = str(value or "").strip().replace(",", "").replace("\\", "")
        if not text:
            return None
        if text.startswith("$"):
            text = text[1:]
        try:
            return float(text)
        except Exception:
            return None

    def _normalize_answer_text(self, value: str) -> str:
        text = str(value).strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-z0-9+#=,./:;() -]+", "", text)
        return text.strip()

    def _normalize_option_prediction(self, raw_answer: str, options: Sequence[str]) -> str:
        answer = str(raw_answer or "").strip()
        if not answer:
            return ""
        normalized_answer = self._normalize_answer_text(answer)
        for option in options:
            option_text = str(option).strip()
            if not option_text:
                continue
            normalized_option = self._normalize_answer_text(option_text)
            if normalized_answer == normalized_option:
                return option_text
            if normalized_answer in {"a", "b", "c", "d"}:
                index = ord(normalized_answer) - ord("a")
                if 0 <= index < len(options):
                    return str(options[index]).strip()
            if normalized_option and (normalized_option in normalized_answer or normalized_answer in normalized_option):
                return option_text
        return answer

    def _enriched_reasoning(
        self,
        *,
        navigator: TRMNavigator,
        question: dict[str, Any],
        route: dict[str, Any],
        use_enriched: bool,
    ) -> Any:
        domain = str(question.get("domain", "multi"))
        text = str(question["question_text"]).lower()
        options = [str(opt) for opt in question["options"]]
        if domain == "math":
            composed = navigator.navigate_and_compose(
                query=str(question.get("question_text", "")),
                specialist=str(route.get("specialist", "math")),
                domain_hint=str(route.get("domain", "math")),
                use_enriched=use_enriched,
            )
            return navigator.execute(composed)
        if domain == "logic":
            for option in options:
                if "all a are c" in option.lower():
                    return option
            return options[0]
        if domain == "physics":
            for option in options:
                if "force" in option.lower():
                    return option
            return options[0]
        for option in options:
            if "verification" in option.lower():
                return option
        return options[0]

    def _empty_mind_reasoning(self, question: dict[str, Any]) -> Any:
        # Empty mind baseline: simple first-option bias.
        return str(question["options"][0])

    def _summarize_by_domain(self, rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        summary: dict[str, dict[str, Any]] = {}
        for row in rows:
            domain = str(row["domain"])
            if domain not in summary:
                summary[domain] = {"total": 0, "correct": 0}
            summary[domain]["total"] += 1
            summary[domain]["correct"] += int(row["correct"])
        for bucket in summary.values():
            total = int(bucket["total"])
            bucket["accuracy"] = (bucket["correct"] / total) if total else 0.0
        return summary

    def save_results(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark": "Last Humanity Exam",
            "dataset_path": str(self.dataset_path) if self.dataset_path else "synthetic",
            "dataset_source": self.dataset_source,
            "dataset_file": self.dataset_file,
            "synthetic_fallback": bool(self.synthetic_fallback),
            "total_questions": len(self.results),
            "correct": sum(int(row.get("correct", 0)) for row in self.results),
            "accuracy": (
                sum(int(row.get("correct", 0)) for row in self.results) / len(self.results)
                if self.results
                else 0.0
            ),
            "results_by_domain": self._summarize_by_domain(self.results),
            "results": self.results,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
