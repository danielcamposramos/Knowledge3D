"""Phase 25 algorithmic thinking trainer.

Consumes the Algorithmic Thinking stars produced by the library ingress
step and runs RLWHF-scored RPN drills using the Phase 18 fused head.
Teacher feedback from exaone3.5 (local Ollama) is blended into the Galaxy
stars so the algorithmic soul keeps growing across sessions.
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # Lazy import: resolved when ``trainer`` property first accessed.
    from knowledge3d.tools.phase18.meaning_cluster_trainer import MeaningClusterTrainer  # type: ignore
except Exception:  # pragma: no cover
    MeaningClusterTrainer = None  # type: ignore

try:
    from knowledge3d.cranium.phase10.rpn_calculator import RPNCalculator  # type: ignore
except Exception:
    RPNCalculator = None  # type: ignore


LEXICON_JSONL_FILES = [
    Path("viewer/public/galaxy/working/lexicon_en_wordnet.jsonl"),
    Path("viewer/public/galaxy/working/lexicon_pt_openwordnet.jsonl"),
    Path("viewer/public/galaxy/working/lexicon_es_kaikki.jsonl"),
    Path("viewer/public/galaxy/working/lexicon_zh_cedict.jsonl"),
]

_HYPHEN_RE = re.compile(r"(\w)-\s+(\w)")


class _TeeStream(io.TextIOBase):
    """Duplicate writes to multiple text streams (stdout + log file)."""

    def __init__(self, *streams: io.TextIOBase) -> None:
        super().__init__()
        self._streams: Tuple[io.TextIOBase, ...] = streams
        self._encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return self._encoding

    def write(self, data: str) -> int:  # type: ignore[override]
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:  # type: ignore[override]
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:  # type: ignore[override]
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


class AlgorithmicThinkingTrainer:
    """RPN + honesty drills for Algorithmic Thinking stars."""

    def __init__(self) -> None:
        self.galaxy_working_dir = Path("viewer/public/galaxy/working")
        self.galaxy_working_dir.mkdir(parents=True, exist_ok=True)
        self._trainer: Optional[MeaningClusterTrainer] = None
        self._rpn_corpus: List[Dict[str, Any]] = []
        self._thinking_corpus: List[Dict[str, Any]] = []
        self._time_corpus: List[Dict[str, Any]] = []
        self._reflection_corpus: List[Dict[str, Any]] = []
        self._context_corpus: List[Dict[str, Any]] = []
        self._teaching_corpus: List[Dict[str, Any]] = []
        self._research_corpus: List[Dict[str, Any]] = []
        self._lexicon_corpus: List[Dict[str, Any]] = []
        if RPNCalculator is None:
            raise ImportError("RPNCalculator unavailable — ensure phase10 PTX engine is importable.")
        self._rpn_calculator: RPNCalculator = RPNCalculator()
        self._total_queries: int = 0
        self._queries_processed: int = 0
        self._sleep_targets: List[int] = []
        self._sleep_cycle_index: int = 0
        self._sleep_cycles_completed: int = 0
        
    @property
    def trainer(self) -> "MeaningClusterTrainer":
        if self._trainer is None:
            if MeaningClusterTrainer is None:  # pragma: no cover - defensive path
                raise ImportError(
                    "MeaningClusterTrainer import failed; ensure phase18 tooling is available."
                )
            self._trainer = MeaningClusterTrainer()
        return self._trainer

    def ensure_env(self) -> None:
        """Emit guidance if the preferred conda environment is not active."""
        conda_path = (
            shutil.which("conda")
            or ("/home/daniel/miniforge/bin/conda" if Path("/home/daniel/miniforge/bin/conda").exists() else None)
        )
        current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
        if conda_path is None:
            raise RuntimeError("Conda executable not found; activate the k3d-cranium environment before training.")
        if current_env != "k3d-cranium":
            label = current_env or "unknown"
            raise RuntimeError(
                f"Conda env '{label}' active — run 'conda activate k3d-cranium' before training."
            )
        print("✅ Conda env k3d-cranium detected.")

    def train_algorithmic_thinking(self) -> None:
        """Execute algorithmic thinking drills across curated stars with RLWHF teachers."""
        log_path = Path("logs/phase25_pt_br_train.log")
        saved_stdout = sys.stdout
        saved_stderr = sys.stderr
        log_file = self._open_training_log(log_path)
        sys.stdout = _TeeStream(saved_stdout, log_file)
        sys.stderr = _TeeStream(saved_stderr, log_file)

        try:
            print("🧠 Training Algorithmic Thinking — RPN, PTX, Honesty, RLWHF...")
            self.ensure_env()

            stars = self.load_stars_by_tag("algorithmic_thinking")
            if not stars:
                print("⚠️  No algorithmic thinking stars found — run library ingest first.")
                return

            if not self._rpn_corpus:
                self._load_rpn_corpus()

            if not hasattr(self, "_thinking_corpus") or not self._thinking_corpus:
                self._load_thinking_corpus()

            if not self._time_corpus:
                self._time_corpus = self._load_jsonl(Path("viewer/public/galaxy/working/time_corpus.jsonl"), 200)
            if not self._reflection_corpus:
                self._reflection_corpus = self._load_jsonl(Path("viewer/public/galaxy/working/self_reflection_corpus.jsonl"), 200)
            if not self._context_corpus:
                self._context_corpus = self._load_jsonl(Path("viewer/public/galaxy/working/context_corpus.jsonl"), 200)
            if not self._teaching_corpus:
                self._teaching_corpus = self._load_jsonl(Path("viewer/public/galaxy/working/teaching_corpus.jsonl"), 200)
            if not self._research_corpus:
                self._research_corpus = self._load_jsonl(Path("viewer/public/galaxy/working/research_corpus.jsonl"), 100)
            if not self._lexicon_corpus:
                self._lexicon_corpus = self._load_lexicon_corpus(per_file=300)

            # Warm the fused head (CPU) before spinning up Ollama teachers.
            try:
                print("♨️  Warming up K3D fused head (CPU)...")
                _ = self.trainer.generate_text_embedding("algorithmic soul warmup")
            except Exception as exc:
                print(f"⚠️  K3D fused head warmup failed: {exc}")

            teacher = None
            try:
                from knowledge3d.cranium.phase10.teacher_evaluator import TeacherEvaluator  # type: ignore

                teacher = TeacherEvaluator(
                    ollama_url="http://192.168.0.4:11434",
                    initial_timeout=300,
                    timeout=240,
                )
            except Exception as exc:
                print(f"❌ TeacherEvaluator unavailable — RLWHF scoring skipped: {exc}")

            star_batches: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
            total_queries = 0
            for star in stars:
                queries = self.generate_rpn_queries(star)
                if not queries:
                    star_name = star.get("name", star.get("id", "unknown"))
                    print(f"⚠️  No queries generated for star {star_name} — skipping.")
                    continue
                star_batches.append((star, queries))
                total_queries += len(queries)

            if not star_batches:
                print("⚠️  No training queries produced from algorithmic thinking stars.")
                return

            self._initialize_sleep_schedule(total_queries)
            print(
                f"🧮 Prepared {len(star_batches)} stars with {total_queries} queries. "
                f"Sleep targets: {self._sleep_targets or ['(none)']}"
            )

            for star, queries in star_batches:
                star_name = star.get("name", star.get("id", "unknown"))
                print(f"\n📌 Star: {star_name} ({len(queries)} queries)")
                for query in queries:
                    prompt = query["query"]
                    true_answer = query["true_answer"]
                    keywords = query.get("keywords", [])
                    explanation = query.get("explanation", "")

                    fused_embedding = self.trainer.generate_multi_modal_embedding(prompt)
                    cluster_name = star.get("id", "star_unknown").replace("star_", "", 1)
                    predicted = self.trainer.predict_from_fused_embedding(
                        prompt, fused_embedding, cluster_name=cluster_name
                    )

                    print(f"Q: {prompt}")
                    print(f"🧠 Student Answer: {predicted}")

                    quick_feedback: Dict[str, Any] = {}
                    deep_feedback: Dict[str, Any] = {}
                    score: float = 0.0
                    explanation_text = ""

                    if teacher is not None:
                        quick_feedback = teacher.evaluate_response(
                            ai_response=predicted,
                            model="exaone3.5:latest",
                            question=prompt,
                            expected_answer=true_answer,
                        )
                        deep_feedback = dict(quick_feedback)
                        score = float(quick_feedback.get("score", 0.0))
                        explanation_text = quick_feedback.get("explanation", "")
                        print(f"📊 RLWHF Score: {score:.2f}")
                        if explanation_text:
                            print(f"💬 Teacher Feedback: {explanation_text}")
                    else:
                        fallback = self.trainer.evaluate_house_answer(
                            predicted=predicted,
                            true_answer=true_answer,
                            keywords=keywords,
                            modality_hint=query.get("modality_hint", ""),
                        )
                        quick_feedback = fallback
                        deep_feedback = dict(fallback)
                        score = float(fallback.get("score", 0.0))
                        explanation_text = fallback.get("explanation", "")
                        print(f"📊 Honesty Score: {score:.2f} — {explanation_text}")

                    composite_explanation = explanation_text or explanation

                    self._update_star_with_feedback(
                        star=star,
                        prompt=prompt,
                        true_answer=true_answer,
                        predicted=predicted,
                        fused_embedding=fused_embedding,
                        quick_feedback=quick_feedback,
                        deep_feedback=deep_feedback,
                        score=score,
                        remedial_hint=explanation,
                        composite_explanation=composite_explanation,
                    )

                    self._queries_processed += 1
                    self._maybe_run_sleep_cycle()

            self._finalize_sleep_schedule()
            print(
                f"✅ Phase 25 training run complete — processed {self._queries_processed} queries, "
                f"sleep cycles executed: {self._sleep_cycles_completed}."
            )
        finally:
            sys.stdout = saved_stdout
            sys.stderr = saved_stderr
            log_file.flush()
            log_file.close()

    def load_stars_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        stars: List[Dict[str, Any]] = []
        for filepath in sorted(self.galaxy_working_dir.glob("star_*.json")):
            try:
                with filepath.open("r", encoding="utf-8") as handle:
                    star = json.load(handle)
                if tag in star.get("tags", []):
                    star["__path"] = str(filepath)
                    stars.append(star)
            except Exception as exc:
                print(f"⚠️  Failed to read star {filepath}: {exc}")
        return stars

    def generate_rpn_queries(self, star: Dict[str, Any]) -> List[Dict[str, Any]]:
        queries: List[Dict[str, Any]] = [
            {
                "query": "Compute integral of x^2 dx in RPN",
                "true_answer": "x 2 ^ int -> x^3 / 3",
                "explanation": "RPN: x, 2, ^, int -> x^3 / 3",
                "keywords": ["x", "^", "int"],
            },
            {
                "query": "Solve 3x + 5 = 14 in RPN",
                "true_answer": "14 5 - 3 / -> 3",
                "explanation": "RPN: 14, 5, -, 3, / -> 3",
                "keywords": ["14", "-", "/"],
            },
        ]

        source_text = self._load_star_source_text(star)
        concepts = self.extract_concepts(source_text)
        for concept in concepts:
            label = concept.capitalize()
            queries.append(
                {
                    "query": f"CONCEPTUAL: What is the purpose of '{label}' in algorithmic thinking?",
                    "true_answer": "Purpose varies with context — anchor it to honest, geometric cognition.",
                    "explanation": f"PURPOSE of {label}: enable honest, geometric, algorithmic cognition.",
                    "keywords": [concept, "purpose", "algorithmic"],
                }
            )
            queries.append(
                {
                    "query": f"RPN: Express the derivative of {label} with respect to x",
                    "true_answer": f"{label} d/dx -> result",
                    "explanation": f"RPN derivative: {label}, d/dx, -> result",
                    "keywords": [concept, "d/dx"],
                }
            )

        # Append curated RPN expressions
        for entry in self._rpn_corpus:
            queries.append(
                {
                    "query": f"RPN: Evaluate {entry['infix']} using tokens {entry['rpn']}",
                    "true_answer": entry['formatted_result'],
                    "explanation": f"Tokens: {entry['rpn']} → {entry['formatted_result']}",
                    "keywords": ["rpn", "evaluation"],
                }
            )

        for entry in getattr(self, "_thinking_corpus", []):
            queries.append(
                {
                    "query": entry['question'],
                    "true_answer": entry['answer'],
                    "explanation": entry.get('sentence', ''),
                    "keywords": ["thinking", "concept"],
                }
            )

        for dataset, tag in (
            (self._time_corpus, "time"),
            (self._reflection_corpus, "self_reflection"),
            (self._context_corpus, "context"),
            (self._teaching_corpus, "teaching"),
            (self._research_corpus, "research"),
        ):
            for entry in dataset:
                queries.append(
                    {
                        "query": entry.get('question', ''),
                        "true_answer": entry.get('answer', ''),
                        "explanation": entry.get('answer', ''),
                        "keywords": [tag],
                    }
                )
        for entry in getattr(self, "_lexicon_corpus", []):
            queries.append(
                {
                    "query": entry.get("question", ""),
                    "true_answer": entry.get("answer", ""),
                    "explanation": entry.get("explanation", ""),
                    "keywords": [entry.get("lemma", ""), entry.get("language", "")],
                }
            )
        return queries

    def extract_concepts(self, text: str) -> List[str]:
        keywords = [
            "integral",
            "derivative",
            "matrix",
            "vector",
            "recursion",
            "iteration",
            "function",
            "algorithm",
        ]
        lowered = text.lower()
        concepts = [kw for kw in keywords if kw in lowered]
        return concepts[:3]

    def _load_star_source_text(self, star: Dict[str, Any]) -> str:
        cached = str(star.get("source_text", ""))
        if cached:
            return cached
        source_file = star.get("source_file")
        if source_file and Path(source_file).exists():
            try:
                with Path(source_file).open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                return self._flatten_json_text(data)
            except Exception:
                return ""
        return ""

    def _flatten_json_text(self, data: Any) -> str:
        if data is None:
            return ""
        if isinstance(data, str):
            return data
        if isinstance(data, (int, float, bool)):
            return str(data)
        if isinstance(data, list):
            return " ".join(filter(None, (self._flatten_json_text(item) for item in data)))
        if isinstance(data, dict):
            return " ".join(
                filter(None, (self._flatten_json_text(value) for value in data.values()))
            )
        return str(data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _update_star_with_feedback(
        self,
        *,
        star: Dict[str, Any],
        prompt: str,
        true_answer: str,
        predicted: str,
        fused_embedding: Iterable[float],
        quick_feedback: Dict[str, Any],
        deep_feedback: Dict[str, Any],
        score: float,
        remedial_hint: str,
        composite_explanation: str,
    ) -> None:
        star_path_value = star.get("__path")
        if not star_path_value:
            print("⚠️  Star path missing; skipping persistence.")
            return
        star_path = Path(star_path_value)
        try:
            with star_path.open("r", encoding="utf-8") as handle:
                star_payload = json.load(handle)
        except Exception:
            star_payload = {}

        timestamp = datetime.utcnow().isoformat() + "Z"
        current_embedding = star_payload.get("embedding", [])
        blended = self._blend_embeddings(current_embedding, list(fused_embedding), weight=0.35)

        if score < 1.0:
            correction_text = (
                deep_feedback.get("explanation")
                or quick_feedback.get("suggested_revision")
                or remedial_hint
                or true_answer
            )
            corrective_embedding = self.trainer.generate_text_embedding(str(correction_text))
            blended = self._blend_embeddings(blended, corrective_embedding, weight=0.2)

        honesty = float(star_payload.get("honesty_score", 0.0))
        honesty = self._update_honesty_score(honesty, score)

        training_log = star_payload.setdefault("algorithmic_training_log", [])
        training_log.append(
            {
                "timestamp": timestamp,
                "prompt": prompt,
                "predicted": predicted,
                "true_answer": true_answer,
                "score": score,
                "explanation": composite_explanation,
                "teacher_feedback_quick": quick_feedback,
                "teacher_feedback_deep": deep_feedback,
            }
        )

        star_payload.update(
            {
                "embedding": blended,
                "honesty_score": honesty,
                "updated_at": timestamp,
                "last_predicted_answer": predicted,
                "zone_placement": self._determine_zone(star_payload.get("zone_placement"), honesty),
            }
        )

        star_payload.setdefault("learned_answers", {})[prompt] = true_answer
        if remedial_hint:
            star_payload.setdefault("remedial_hints", {}).setdefault(prompt, remedial_hint)

        with star_path.open("w", encoding="utf-8") as handle:
            json.dump(star_payload, handle, ensure_ascii=False, indent=2)

        # Keep in-memory reference aligned for downstream queries
        star.update(star_payload)

    def _blend_embeddings(
        self,
        base: Iterable[float],
        update: Iterable[float],
        *,
        weight: float,
    ) -> List[float]:
        base_list = list(base)
        update_list = list(update)
        if not base_list:
            return update_list
        if not update_list:
            return base_list
        length = max(len(base_list), len(update_list))
        base_list.extend([0.0] * (length - len(base_list)))
        update_list.extend([0.0] * (length - len(update_list)))
        blended = [
            (1.0 - weight) * base_val + weight * update_val
            for base_val, update_val in zip(base_list, update_list)
        ]
        return blended

    def _update_honesty_score(self, honesty: float, delta: float) -> float:
        if delta >= 1.0:
            honesty += 0.10
        elif delta >= 0.5:
            honesty += 0.07
        elif delta == 0.0:
            honesty += 0.05
        elif delta == -0.5:
            honesty -= 0.05
        else:
            honesty -= 0.10
        return max(-1.0, min(1.0, honesty))

    def _determine_zone(self, current_zone: Optional[str], honesty: float) -> str:
        if honesty < 0.5:
            return "Zone 8 (Learning Museum)"
        return current_zone or "Zone 2 (Study)"

    # ------------------------------------------------------------------
    def _load_rpn_corpus(self, limit: int = 20) -> None:
        corpus_path = Path("viewer/public/galaxy/working/rpn_corpus.jsonl")
        if not corpus_path.exists():
            raise FileNotFoundError("RPN corpus not found — run rpn_corpus_builder before training.")

        entries: List[Dict[str, Any]] = []
        with corpus_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tokens = obj.get("tokens")
                if isinstance(tokens, list) and tokens:
                    rpn_expr = " ".join(tokens)
                else:
                    rpn_expr = str(obj.get("rpn", "")).strip()
                if not rpn_expr:
                    continue
                try:
                    result = self._rpn_calculator.evaluate(rpn_expr, instance_id=len(entries) % 15)
                except Exception:
                    self._rpn_calculator.reset()
                    continue
                if not math.isfinite(result):
                    continue
                formatted = f"{result:.6g}"
                entries.append(
                    {
                        "infix": obj.get("infix", rpn_expr),
                        "rpn": rpn_expr,
                        "formatted_result": formatted,
                    }
                )
                if len(entries) >= limit:
                    break
        self._rpn_corpus = entries
        self._rpn_calculator.reset()

    def _load_thinking_corpus(self, limit: int = 200) -> None:
        corpus_path = Path("viewer/public/galaxy/working/thinking_corpus.jsonl")
        if not corpus_path.exists():
            raise FileNotFoundError("Thinking corpus not found — run thinking_corpus_builder before training.")
        entries: List[Dict[str, str]] = []
        with corpus_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                question = obj.get("question")
                answer = obj.get("answer")
                sentence = obj.get("sentence", "")
                if not question or not answer:
                    continue
                entries.append({
                    "question": question,
                    "answer": answer,
                    "sentence": sentence,
                })
                if len(entries) >= limit:
                    break
        self._thinking_corpus = entries

    def _load_lexicon_corpus(self, per_file: int = 200) -> List[Dict[str, Any]]:
        corpus: List[Dict[str, Any]] = []
        for path in LEXICON_JSONL_FILES:
            if not path.exists():
                continue
            added = 0
            try:
                with path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        if added >= per_file:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        lex = data.get("lexicon_entry")
                        if not isinstance(lex, dict):
                            continue
                        lemma = lex.get("lemma") or lex.get("traditional")
                        language = lex.get("language", "unknown")
                        definition = lex.get("definition")
                        if not definition:
                            definitions = lex.get("definitions")
                            if isinstance(definitions, list) and definitions:
                                definition = definitions[0]
                        if not lemma or not definition:
                            continue
                        lemma_str = self._normalize_text(str(lemma))
                        synonyms = [self._normalize_text(str(s)) for s in (lex.get("synonyms") or []) if s]
                        pronunciations = [self._normalize_text(str(p)) for p in (lex.get("pronunciations") or []) if p]
                        definition = self._normalize_text(str(definition))
                        corpus.append(
                            {
                                "question": f"Define '{lemma_str}' ({language}).",
                                "answer": definition,
                                "explanation": definition,
                                "lemma": lemma_str,
                                "language": language,
                                "source": str(path),
                            }
                        )
                        added += 1
                        if added >= per_file:
                            break
                        if synonyms:
                            corpus.append(
                                {
                                    "question": f"Give a synonym for '{lemma_str}' ({language}).",
                                    "answer": synonyms[0],
                                    "explanation": ", ".join(str(s) for s in synonyms if s),
                                    "lemma": lemma_str,
                                    "language": language,
                                    "source": str(path),
                                }
                            )
                            added += 1
                            if added >= per_file:
                                continue
                        if pronunciations:
                            corpus.append(
                                {
                                    "question": f"Provide the IPA pronunciation for '{lemma_str}' ({language}).",
                                    "answer": str(pronunciations[0]),
                                    "explanation": str(pronunciations[0]),
                                    "lemma": lemma_str,
                                    "language": language,
                                    "source": str(path),
                                }
                            )
                            added += 1
            except Exception as exc:
                print(f"⚠️  Failed to load lexicon prompts from {path}: {exc}")
        return corpus

    def _load_jsonl(self, path: Path, limit: int) -> List[Dict[str, Any]]:
        if not path.exists():
            print(f"⚠️  Corpus not found: {path}")
            return []
        entries: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entries.append(obj)
                if len(entries) >= limit:
                    break
        return entries

    def _normalize_text(self, text: str) -> str:
        if not text:
            return text
        return _HYPHEN_RE.sub(r"\1\2", text)

    def _open_training_log(self, log_path: Path) -> io.TextIOBase:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("w", encoding="utf-8")
        timestamp = datetime.utcnow().isoformat() + "Z"
        handle.write(f"=== Phase25 AlgorithmicThinkingTrainer run @ {timestamp} ===\n")
        handle.flush()
        return handle

    def _initialize_sleep_schedule(self, total_queries: int) -> None:
        self._total_queries = max(0, int(total_queries))
        self._queries_processed = 0
        self._sleep_cycle_index = 0
        self._sleep_cycles_completed = 0
        self._sleep_targets = []
        if self._total_queries <= 0:
            return
        for idx in range(3):
            if idx == 2:
                target = self._total_queries
            else:
                target = math.ceil(self._total_queries * (idx + 1) / 3)
            if self._sleep_targets and target <= self._sleep_targets[-1]:
                target = self._sleep_targets[-1] + 1
            self._sleep_targets.append(target)

    def _maybe_run_sleep_cycle(self) -> None:
        while (
            self._sleep_cycle_index < len(self._sleep_targets)
            and self._queries_processed >= self._sleep_targets[self._sleep_cycle_index]
        ):
            self._run_sleep_cycle()
            self._sleep_cycle_index += 1

    def _finalize_sleep_schedule(self) -> None:
        while self._sleep_cycle_index < len(self._sleep_targets):
            self._run_sleep_cycle()
            self._sleep_cycle_index += 1

    def _run_sleep_cycle(self, cycles: int = 1) -> None:
        """Trigger one or more sleep cycles to consolidate newly learned content."""
        iterations = max(1, int(cycles))
        for _ in range(iterations):
            if self._sleep_cycles_completed >= 3:
                print("🌙 Sleep-time compute already executed three times — skipping extra cycle.")
                return
            try:
                from knowledge3d.cranium.phase10.sleep_time_compute import SleepTimeCompute  # type: ignore

                house_glb = Path("viewer/public/house/house_master_assembled.glb")
                if not house_glb.exists():
                    house_glb = Path("viewer/public/house/house_master.glb")
                galaxy_glb = Path("viewer/public/galaxy.v8.glb")
                if not galaxy_glb.exists():
                    galaxy_glb = Path("viewer/public/galaxy.glb")

                cycle_number = self._sleep_cycles_completed + 1
                stc = SleepTimeCompute(
                    house_path=str(house_glb),
                    galaxy_path=str(galaxy_glb),
                    output_path=str(house_glb.parent / f"house_post_sleep_cycle{cycle_number}.glb"),
                    material_dir=str(house_glb.parent / "materialized_objects"),
                )
                stc.run()
                self._sleep_cycles_completed += 1
                print(f"🌙 Sleep-time consolidation cycle {cycle_number} complete.")
            except Exception as exc:
                print(f"⚠️  Sleep-time compute skipped: {exc}")
                break
