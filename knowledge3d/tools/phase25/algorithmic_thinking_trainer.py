"""Phase 25 algorithmic thinking trainer.

Consumes the Algorithmic Thinking stars produced by the library ingress
step and runs RLWHF-scored RPN drills using the Phase 18 fused head.
Feedback from house honesty evaluators is blended into the Galaxy stars so
the algorithmic soul keeps growing across sessions.
"""
from __future__ import annotations

import io
import hashlib
import json
import locale
import math
import os
import re
import shutil
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np  # type: ignore

try:  # Lazy import: resolved when ``trainer`` property first accessed.
    from knowledge3d.tools.phase18.meaning_cluster_trainer import MeaningClusterTrainer  # type: ignore
except Exception:  # pragma: no cover
    MeaningClusterTrainer = None  # type: ignore

try:
    from knowledge3d.cranium.phase10.rpn_calculator import RPNCalculator  # type: ignore
except Exception:
    RPNCalculator = None  # type: ignore

try:
    from knowledge3d.cranium.ptx import PTX_OPS  # type: ignore
except Exception:  # pragma: no cover
    PTX_OPS = None  # type: ignore


LEXICON_JSONL_FILES = [
    Path("viewer/public/galaxy/working/lexicon_en_wordnet.jsonl"),
    Path("viewer/public/galaxy/working/lexicon_pt_openwordnet.jsonl"),
    Path("viewer/public/galaxy/working/lexicon_es_kaikki.jsonl"),
    Path("viewer/public/galaxy/working/lexicon_zh_cedict.jsonl"),
]

LANGUAGE_GALAXY_DIR = Path("viewer/public/galaxy")

_HYPHEN_RE = re.compile(r"(\w)-\s+(\w)")
_SIGNIFICANT_EVENT_RE = re.compile(r"^what significant event happened in \d{1,3}\??$", re.IGNORECASE)
_BOXED_ANSWER_RE = re.compile(r"\\boxed\{\s*([0-9]{1,3})\s*\}")
_NUMERIC_TOKEN_RE = re.compile(r"\b\d{1,3}\b")


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

    @staticmethod
    def _get_env_int(name: str, default: int, minimum: int = 1) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return default
        return max(minimum, value)

    def __init__(self) -> None:
        self.galaxy_working_dir = Path("viewer/public/galaxy/working")
        self.galaxy_working_dir.mkdir(parents=True, exist_ok=True)
        self.learning_memory_path = self.galaxy_working_dir / "learning_memory.jsonl"
        self._trainer: Optional[MeaningClusterTrainer] = None
        self._rpn_corpus: List[Dict[str, Any]] = []
        self._thinking_corpus: List[Dict[str, Any]] = []
        self._time_corpus: List[Dict[str, Any]] = []
        self._math_corpus: List[Dict[str, Any]] = []
        self._reflection_corpus: List[Dict[str, Any]] = []
        self._context_corpus: List[Dict[str, Any]] = []
        self._teaching_corpus: List[Dict[str, Any]] = []
        self._research_corpus: List[Dict[str, Any]] = []
        self._lexicon_corpus: List[Dict[str, Any]] = []
        self._meta_math_corpus: List[Dict[str, Any]] = []
        self._aime_queue: List[Dict[str, Any]] = []
        self._language_galaxies: List[Dict[str, Any]] = []
        if RPNCalculator is None:
            raise ImportError("RPNCalculator unavailable — ensure phase10 PTX engine is importable.")
        self._rpn_calculator: RPNCalculator = RPNCalculator()
        self._total_queries: int = 0
        self._queries_processed: int = 0
        self._sleep_targets: List[int] = []
        self._sleep_cycle_index: int = 0
        self._sleep_cycles_completed: int = 0
        self.mastery_threshold: int = 2
        self.mastered_prompts_path = self.galaxy_working_dir / "mastered_prompts.jsonl"
        self._mastered_prompts: Dict[str, Dict[str, Any]] = self._load_mastered_prompts()
        self._retired_prompts: set[str] = {
            prompt
            for prompt, meta in self._mastered_prompts.items()
            if meta.get("retired") or int(meta.get("count", 0)) >= self.mastery_threshold
        }
        self.max_aime_prompts: int = self._get_env_int("K3D_AIME_MAX_ITEMS", 90)
        self.max_aime_per_star: int = self._get_env_int("K3D_AIME_PER_STAR", 5)
        self._aime_cache_path: Path = self.galaxy_working_dir / "aime_2024_problems.parquet"
        
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
        allowed_envs = {"k3d-cranium", "k3d-ptx"}
        if current_env not in allowed_envs:
            label = current_env or "unknown"
            raise RuntimeError(
                f"Conda env '{label}' active — run 'conda activate k3d-cranium' before training."
            )
        print(f"✅ Conda env {current_env or 'k3d-cranium'} detected.")

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
            self._verify_ptx_head()

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
            if not self._math_corpus:
                self._math_corpus = self._load_jsonl(Path("viewer/public/galaxy/working/math_corpus.jsonl"), 300)
            if not self._meta_math_corpus:
                self._meta_math_corpus = self._load_meta_math_corpus(limit=self._get_env_int("K3D_META_MATH_LIMIT", 200))
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

            self._refresh_language_galaxies()
            self._warm_language_galaxies()

            self._prepare_aime_prompts(max_items=self.max_aime_prompts)

            # Warm the fused head (CPU) before spinning up Ollama teachers.
            try:
                print("♨️  Warming up K3D fused head (GPU)...")
                _ = self.trainer.generate_text_embedding("algorithmic soul warmup")
            except Exception as exc:
                print(f"⚠️  K3D fused head warmup failed: {exc}")

            teacher = None
            disable_teacher = str(os.environ.get("K3D_DISABLE_TEACHER", "0")).lower() in {"1", "true", "yes"}
            if disable_teacher:
                print("ℹ️ TeacherEvaluator disabled via K3D_DISABLE_TEACHER.")
            else:
                try:
                    from knowledge3d.cranium.phase10.teacher_evaluator import TeacherEvaluator  # type: ignore

                    teacher = TeacherEvaluator(
                        ollama_url="http://192.168.0.4:11434",
                        initial_timeout=420,
                        timeout=300,
                    )
                    print("🧑‍🏫 Teacher ready: exaone-deep:latest")
                except Exception as exc:
                    print(f"⚠️  TeacherEvaluator unavailable — falling back to house honesty evaluator: {exc}")

            def _env_limit(name: str) -> Optional[int]:
                raw = os.environ.get(name)
                if raw is None or str(raw).strip() == "":
                    return None
                try:
                    value = int(str(raw).strip())
                except (TypeError, ValueError):
                    return None
                return max(1, value)

            star_limit = _env_limit("K3D_AT_MAX_STARS")
            total_limit = _env_limit("K3D_AT_MAX_QUERIES")

            star_batches: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
            total_queries = 0
            for index, star in enumerate(stars):
                if star_limit is not None and index >= star_limit:
                    break
                queries = self.generate_rpn_queries(star)
                if not queries:
                    star_name = star.get("name", star.get("id", "unknown"))
                    print(f"⚠️  No queries generated for star {star_name} — skipping.")
                    continue
                if total_limit is not None:
                    remaining = total_limit - total_queries
                    if remaining <= 0:
                        break
                    if len(queries) > remaining:
                        queries = queries[:remaining]
                star_batches.append((star, queries))
                total_queries += len(queries)
                if total_limit is not None and total_queries >= total_limit:
                    break

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

                    true_canonical = self._normalize_numeric_answer(true_answer) or true_answer.strip()
                    predicted_canonical = self._normalize_numeric_answer(predicted) or predicted.strip()
                    exact_match = predicted_canonical.lower() == true_canonical.lower()
                    if exact_match:
                        score = 1.0
                        explanation_text = "Auto-validated exact match with expected answer."
                        quick_feedback = {"score": score, "explanation": explanation_text}
                        deep_feedback = dict(quick_feedback)
                        print(f"✅ Auto Honesty Score: {score:.2f} — {explanation_text}")
                    elif teacher is not None:
                        quick_feedback = teacher.evaluate_response(
                            ai_response=predicted,
                            model="exaone-deep:latest",
                            question=prompt,
                            expected_answer=true_answer,
                        )
                        deep_feedback = dict(quick_feedback)
                        score = float(quick_feedback.get("score", 0.0))
                        explanation_text = quick_feedback.get("explanation", "")
                        print(f"📊 RLWHF Score (exaone-deep): {score:.2f}")
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

                    # Train the fused head's internal math head on numeric answers (0..999)
                    try:
                        if true_canonical.isdigit():
                            self.trainer.fused_head.train_step(fused_embedding, true_canonical)
                    except Exception:
                        pass

                    self._queries_processed += 1
                    self._maybe_run_sleep_cycle()

            self._finalize_sleep_schedule()
            print(
                f"✅ Phase 25 training run complete — processed {self._queries_processed} queries, "
                f"sleep cycles executed: {self._sleep_cycles_completed}."
            )
            print(f"🕒 Session timestamp: {datetime.now(timezone.utc).isoformat()}")
        finally:
            try:
                self._rebuild_learning_memory_glb()
            except Exception as exc:
                print(f"⚠️  Learning memory rebuild skipped: {exc}")
            try:
                self._save_mastered_prompts()
            except Exception as exc:
                print(f"⚠️  Mastered prompt log skipped: {exc}")
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
            rpn_expr = entry['rpn']
            result = entry['formatted_result']
            queries.append(
                {
                    "query": f"Evaluate the RPN expression '{rpn_expr}'.",
                    "true_answer": result,
                    "explanation": f"The RPN stack reduces to {result}.",
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
            (self._math_corpus, "math"),
            (self._reflection_corpus, "self_reflection"),
            (self._context_corpus, "context"),
            (self._teaching_corpus, "teaching"),
            (self._research_corpus, "research"),
        ):
            if not dataset:
                continue
            for entry in dataset:
                question = self._normalize_text(entry.get('question', ''))
                answer = str(entry.get('answer', '')).strip()
                if not question or not answer:
                    continue
                queries.append(
                    {
                        "query": question,
                        "true_answer": answer,
                        "explanation": entry.get('explanation', answer),
                        "keywords": [tag],
                    }
                )
        for entry in self._meta_math_corpus:
            question = entry.get("question", "")
            answer = entry.get("answer", "")
            if not question or not answer:
                continue
            queries.append(
                {
                    "query": question,
                    "true_answer": answer,
                    "explanation": entry.get("explanation", answer),
                    "keywords": ["meta_math"],
                }
            )
        for entry in getattr(self, "_lexicon_corpus", []):
            question = entry.get("question", "")
            answer = entry.get("answer", "")
            if not question or not answer:
                continue
            queries.append(
                {
                    "query": question,
                    "true_answer": answer,
                    "explanation": entry.get("explanation", answer),
                    "keywords": [entry.get("lemma", ""), entry.get("language", "")],
                }
            )
        if self._aime_queue:
            take = min(len(self._aime_queue), self.max_aime_per_star)
            for _ in range(take):
                queries.append(self._aime_queue.pop(0))
        filtered = [q for q in queries if self._should_use_prompt(q.get("query", ""))]
        return filtered

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

    def _log_learning_memory(
        self,
        *,
        timestamp: str,
        prompt: str,
        true_answer: str,
        predicted: str,
        score: float,
        quick_feedback: Dict[str, Any],
        deep_feedback: Dict[str, Any],
        star: Dict[str, Any],
    ) -> None:
        try:
            self.learning_memory_path.parent.mkdir(parents=True, exist_ok=True)
            language_hint = self._parse_language_hint(prompt)
            record_id = f"learning_{hashlib.sha256((timestamp + prompt).encode('utf-8')).hexdigest()[:16]}"
            try:
                fused_head = getattr(self.trainer, "fused_head")
            except AttributeError as exc:
                raise RuntimeError("Phase25 trainer requires fused head instance; missing on Phase18 trainer.") from exc
            concepts = self.extract_concepts(prompt)
            base_tags = list(star.get("tags", [])) if isinstance(star.get("tags"), list) else []
            merged_tags: List[str] = []
            for value in base_tags + concepts:
                if value and value not in merged_tags:
                    merged_tags.append(value)

            metadata = {
                "star_id": star.get("id"),
                "concepts": concepts,
                "trainer_record_id": record_id,
            }
            if not hasattr(fused_head, "append_learning_memory"):
                raise RuntimeError("AdaptedFusedHead missing append_learning_memory during Phase25 logging.")
            result = fused_head.append_learning_memory(
                prompt=prompt,
                true_answer=true_answer,
                predicted=predicted,
                score=float(score),
                quick_feedback=quick_feedback,
                deep_feedback=deep_feedback,
                tags=merged_tags,
                language=language_hint or star.get("language"),
                metadata=metadata,
            )
            if result is None:
                raise RuntimeError("Fused head failed to persist Phase25 learning memory entry.")
        except Exception as exc:
            raise RuntimeError("Phase25 learning memory logging failed; resolve fused head pipeline.") from exc

    def _parse_language_hint(self, query: str) -> Optional[str]:
        if not query:
            return None
        trimmed = query.strip()
        patterns = [
            re.compile(
                r"^(?:define|give a synonym for|provide the ipa pronunciation for)\s+'[^']+'\s*\(([^)]+)\)",
                re.IGNORECASE,
            ),
            re.compile(r"\(([^)]+)\)\s*$"),
        ]
        for pattern in patterns:
            match = pattern.search(trimmed)
            if match:
                return match.group(1).strip().lower()
        return None

    def _rebuild_learning_memory_glb(self) -> None:
        if not self.learning_memory_path.exists():
            return
        try:
            from knowledge3d.tools.learning_memory_builder import build_learning_memory  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"learning_memory_builder unavailable: {exc}")
        args = SimpleNamespace(
            input=[str(self.learning_memory_path)],
            out=str(Path("viewer/public/galaxy/learning_memory.glb")),
            manifest=str(Path("viewer/public/galaxy/learning_memory.json")),
            limit=None,
            label="Learning Memory Galaxy",
            embedding_dim=512,
        )
        build_learning_memory(args)
        print("💾 Learning memory galaxy refreshed (trainer).")

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

    def _verify_ptx_head(self) -> None:
        """Ensure the PTX geometry kernel is available before training."""
        try:
            from knowledge3d.cranium.phase10.nvrtc_ptx_loader import NVRTCPTXLoader  # type: ignore

            loader = NVRTCPTXLoader()
            probe = np.zeros(32, dtype=np.float32)
            probe[:3] = np.array([1.0, 0.5, -0.75], dtype=np.float32)
            vertices = loader.generate_vertices(probe, 4, 0)
            if not isinstance(vertices, np.ndarray) or vertices.size == 0:
                raise RuntimeError("PTX kernel returned empty vertex buffer")
            print("✅ PTX geometry head verified (tetrahedron probe).")
        except Exception as exc:
            raise RuntimeError(
                "PTX head unavailable — resolve CUDA/NVRTC configuration before rerunning training."
            ) from exc

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
        self._log_learning_memory(
            timestamp=timestamp,
            prompt=prompt,
            true_answer=true_answer,
            predicted=predicted,
            score=score,
            quick_feedback=quick_feedback,
            deep_feedback=deep_feedback,
            star=star,
        )
        self._track_mastered_prompt(prompt, score)

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
                tokens_list = rpn_expr.split()
                if len(tokens_list) < 2:
                    continue
                operator_tokens = {"+", "-", "*", "/", "^", "neg", "sqrt", "sin", "cos", "tan", "log", "ln", "exp", "int", "d/dx"}
                if not any(tok in operator_tokens for tok in tokens_list):
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

    def _refresh_language_galaxies(self) -> None:
        manifests = sorted(LANGUAGE_GALAXY_DIR.glob("language_*.json"))
        galaxies: List[Dict[str, Any]] = []
        for manifest in manifests:
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"⚠️  Unable to read language manifest {manifest}: {exc}")
                continue
            glb_path = manifest.with_suffix(".glb")
            if not glb_path.exists():
                print(f"⚠️  Language galaxy GLB missing for manifest {manifest}")
                continue
            galaxies.append(
                {
                    "manifest": manifest,
                    "glb": glb_path,
                    "language": data.get("language", manifest.stem.split("_")[1]),
                    "label": data.get("label", manifest.stem),
                    "count": data.get("count"),
                }
            )
        self._language_galaxies = galaxies
        if galaxies:
            summary = ", ".join(f"{g['label']} ({g.get('count','?')} stars)" for g in galaxies)
            print(f"🗺️  Language galaxies discovered: {summary}")
        else:
            print("⚠️  No PTX language galaxies found under viewer/public/galaxy/")

    def _warm_language_galaxies(self) -> None:
        if not self._language_galaxies:
            return
        if PTX_OPS is None:
            print("⚠️  PTX ops unavailable — skipping language galaxy warmup.")
            return
        for galaxy in self._language_galaxies:
            glb_path = galaxy["glb"]
            label = galaxy.get("label", glb_path.stem)
            try:
                PTX_OPS.geometry_load_scene(glb_path.as_posix())
                PTX_OPS.geometry_release()
                print(f"🌐 PTX-warmed language galaxy: {label}")
            except Exception as exc:
                print(f"⚠️  Failed to load language galaxy {label}: {exc}")

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
                if not self._is_valid_corpus_entry(obj):
                    continue
                entries.append(obj)
                if len(entries) >= limit:
                    break
        return entries

    def _load_meta_math_corpus(self, limit: int) -> List[Dict[str, Any]]:
        try:
            from datasets import load_dataset  # type: ignore
        except Exception as exc:
            print(f"⚠️  datasets package unavailable for meta-math corpus: {exc}")
            return []

        desired_locale = "C.UTF-8"
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        os.environ.setdefault("PYTHONUTF8", "1")
        for var in ("LC_ALL", "LANG", "LC_CTYPE"):
            current = os.environ.get(var, "")
            if not current or current.lower() in {"c", "posix"}:
                os.environ[var] = desired_locale
        try:
            locale.setlocale(locale.LC_ALL, os.environ.get("LC_ALL", desired_locale))
        except locale.Error:
            pass

        sample: List[Dict[str, Any]] = []
        try:
            ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=False)
        except Exception as exc:
            print(f"⚠️  Unable to load meta-math dataset: {exc}")
            return []

        for idx, row in enumerate(ds):
            if idx >= limit:
                break
            prompt = str(row.get("problem") or row.get("question") or row.get("query") or "").strip()
            raw_response = str(row.get("solution") or row.get("answer") or row.get("response") or "").strip()
            answer = raw_response
            if "The answer is:" in raw_response:
                tail = raw_response.split("The answer is:")[-1].strip()
                # Remove trailing punctuation or TeX box wrappers.
                answer = tail.strip().strip(".")
            if not prompt or not answer:
                continue
            sample.append(
                {
                    "question": f"META-MATH: {prompt}",
                    "answer": answer,
                    "explanation": row.get("rationale") or raw_response,
                }
            )
        print(f"📘 Loaded {len(sample)} MetaMathQA prompts for reinforcement.")
        return sample

    def _normalize_text(self, text: str) -> str:
        if not text:
            return text
        return _HYPHEN_RE.sub(r"\1\2", text)

    def _is_valid_corpus_entry(self, entry: Dict[str, Any]) -> bool:
        question = str(entry.get("question") or entry.get("query") or "").strip()
        answer = str(entry.get("answer") or entry.get("true_answer") or "").strip()
        if not question or not answer:
            return False
        normalized = question.lower()
        if _SIGNIFICANT_EVENT_RE.match(normalized):
            return False
        if question in self._retired_prompts:
            return False
        return True

    def _load_mastered_prompts(self) -> Dict[str, Dict[str, Any]]:
        data: Dict[str, Dict[str, Any]] = {}
        path = self.mastered_prompts_path
        if not path.exists():
            return data
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = obj.get("prompt")
                if not prompt:
                    continue
                data[str(prompt)] = obj
        return data

    def _save_mastered_prompts(self) -> None:
        entries = [self._mastered_prompts[prompt] for prompt in sorted(self._mastered_prompts.keys())]
        self.mastered_prompts_path.parent.mkdir(parents=True, exist_ok=True)
        with self.mastered_prompts_path.open("w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _track_mastered_prompt(self, prompt: str, score: float) -> None:
        normalized = prompt.strip()
        if not normalized or score < 1.0:
            return
        meta = self._mastered_prompts.setdefault(
            normalized,
            {
                "prompt": normalized,
                "count": 0,
                "first_mastered": datetime.now(timezone.utc).isoformat(),
            },
        )
        meta["count"] = int(meta.get("count", 0)) + 1
        meta["last_mastered"] = datetime.now(timezone.utc).isoformat()
        if meta["count"] >= self.mastery_threshold and not meta.get("retired"):
            meta["retired"] = True
            meta["retired_at"] = datetime.now(timezone.utc).isoformat()
            self._retired_prompts.add(normalized)

    def _should_use_prompt(self, prompt: str) -> bool:
        if not prompt or not prompt.strip():
            return False
        normalized = prompt.strip().lower()
        if _SIGNIFICANT_EVENT_RE.match(normalized):
            return False
        if prompt.strip() in self._retired_prompts:
            return False
        return True

    def _normalize_numeric_answer(self, text: str | None) -> Optional[str]:
        if not text:
            return None
        raw = str(text).strip()
        if not raw:
            return None
        boxed = _BOXED_ANSWER_RE.findall(raw)
        if boxed:
            try:
                value = int(boxed[-1])
                return f"{value:03d}"
            except ValueError:
                pass
        cleaned = raw.replace("$", " ").replace("−", "-")
        tokens = _NUMERIC_TOKEN_RE.findall(cleaned)
        for candidate in reversed(tokens):
            try:
                value = int(candidate)
            except ValueError:
                continue
            if 0 <= value <= 999:
                return f"{value:03d}"
        return None

    def _format_aime_answer(self, answer: str) -> str:
        normalized = self._normalize_numeric_answer(answer)
        if normalized is not None:
            return normalized
        return answer.strip()

    def _prepare_aime_prompts(self, max_items: int = 30) -> None:
        if self._aime_queue:
            return
        prompts: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()

        def _ensure_utf8_locale() -> None:
            desired_locale = "C.UTF-8"
            for key, default in (
                ("PYTHONIOENCODING", "utf-8"),
                ("PYTHONUTF8", "1"),
            ):
                os.environ.setdefault(key, default)
            for key in ("LC_ALL", "LANG", "LC_CTYPE"):
                current = os.environ.get(key, "")
                if not current or current.lower() in {"c", "posix"}:
                    os.environ[key] = desired_locale
            try:
                locale.setlocale(locale.LC_ALL, os.environ.get("LC_ALL", desired_locale))
            except locale.Error:
                try:
                    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
                    os.environ["LC_ALL"] = "en_US.UTF-8"
                except locale.Error:
                    pass

        _ensure_utf8_locale()

        cache_target = getattr(self, "_aime_cache_path", self.galaxy_working_dir / "aime_2024_problems.parquet")
        cache_target.parent.mkdir(parents=True, exist_ok=True)

        def _persist_parquet_cache() -> Optional[Path]:
            try:
                from huggingface_hub import hf_hub_download  # type: ignore

                parquet_path = Path(
                    hf_hub_download(
                        repo_id="Maxwell-Jia/AIME_2024",
                        filename="aime_2024_problems.parquet",
                        repo_type="dataset",
                    )
                )
                try:
                    if parquet_path.resolve() != cache_target.resolve():
                        shutil.copyfile(parquet_path, cache_target)
                except Exception as copy_exc:
                    print(f"⚠️  Unable to persist AIME parquet cache: {copy_exc}")
                return parquet_path
            except Exception as exc:
                print(f"⚠️  Unable to download AIME parquet cache: {exc}")
                return None

        def _coerce(value: Any) -> str:
            if value is None:
                return ""
            if hasattr(value, "as_py"):
                try:
                    value = value.as_py()
                except Exception:
                    value = str(value)
            if isinstance(value, bytes):
                text: Optional[str] = None
                for codec in ("utf-8", "latin-1"):
                    try:
                        text = value.decode(codec)
                        break
                    except UnicodeDecodeError:
                        continue
                if text is None:
                    text = value.decode("utf-8", errors="replace")
            else:
                text = str(value)

            normalized = unicodedata.normalize("NFC", text)
            normalized = normalized.replace("﻿", "").replace(" ", " ")
            normalized = re.sub(r"\s+", " ", normalized)
            return normalized.strip()

        try:
            from datasets import load_dataset  # type: ignore

            ds = load_dataset(
                "Maxwell-Jia/AIME_2024",
                split="train",
                download_mode="reuse_cache_if_exists",
            )
            _persist_parquet_cache()
            iterator: Iterable[Any]
            if hasattr(ds, "__iter__") and not hasattr(ds, "__getitem__"):
                iterator = ds
            else:
                iterator = iter(ds)

            for row in iterator:
                if len(prompts) >= max_items:
                    break
                container = dict(row)
                question = _coerce(container.get("Problem"))
                answer_str = _coerce(container.get("Answer"))
                problem_id = _coerce(container.get("ID")) or "AIME"
                if not question or not answer_str:
                    continue
                key = f"{problem_id}:{question}"
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                normalized_answer = self._format_aime_answer(answer_str)
                prompts.append(
                    {
                        "query": f"AIME problem {problem_id}: {question}",
                        "true_answer": normalized_answer,
                        "explanation": f"Official AIME answer: {normalized_answer}.",
                        "keywords": ["AIME", problem_id],
                    }
                )
        except Exception as primary_exc:
            print(f"⚠️  Primary AIME dataset load failed: {primary_exc}")
            try:
                parquet_path = _persist_parquet_cache()
                if parquet_path is None:
                    raise RuntimeError("AIME parquet cache unavailable")

                records: List[Dict[str, Any]] = []
                load_error: Optional[Exception] = None
                try:
                    import pandas as pd  # type: ignore

                    df = pd.read_parquet(parquet_path)
                    records = df.to_dict(orient="records")
                except Exception as pandas_exc:  # pragma: no cover - optional dependency
                    load_error = pandas_exc
                    try:
                        import pyarrow.parquet as pq  # type: ignore

                        table = pq.read_table(parquet_path)
                        records = table.to_pylist()
                        load_error = None
                    except Exception as arrow_exc:  # pragma: no cover - optional dependency
                        load_error = arrow_exc

                if load_error is not None:
                    raise load_error

                for row in records:
                    if len(prompts) >= max_items:
                        break
                    question = _coerce(row.get("Problem"))
                    answer_str = _coerce(row.get("Answer"))
                    problem_id = _coerce(row.get("ID")) or "AIME"
                    if not question or not answer_str:
                        continue
                    key = f"{problem_id}:{question}"
                    if key in seen_ids:
                        continue
                    seen_ids.add(key)
                    normalized_answer = self._format_aime_answer(answer_str)
                    prompts.append(
                        {
                            "query": f"AIME problem {problem_id}: {question}",
                            "true_answer": normalized_answer,
                            "explanation": f"Official AIME answer: {normalized_answer}.",
                            "keywords": ["AIME", problem_id],
                        }
                    )
            except Exception as fallback_exc:
                print(f"⚠️  Fallback AIME parquet load failed: {fallback_exc}")

        if not prompts:
            print("⚠️  No AIME prompts loaded — dataset empty or unavailable.")
        self._aime_queue = prompts

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
