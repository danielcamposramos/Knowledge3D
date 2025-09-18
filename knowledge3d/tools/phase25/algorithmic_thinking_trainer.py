"""Phase 25 algorithmic thinking trainer.

Consumes the Algorithmic Thinking stars produced by the library ingress
step and runs RLWHF-scored RPN drills using the Phase 18 fused head.
Teacher feedback from exaone (quick scoring) and exaone-deep (conceptual
analysis) is blended into the Galaxy stars so the algorithmic soul keeps
growing across sessions.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:  # Lazy import: resolved when ``trainer`` property first accessed.
    from knowledge3d.tools.phase18.meaning_cluster_trainer import MeaningClusterTrainer  # type: ignore
except Exception:  # pragma: no cover
    MeaningClusterTrainer = None  # type: ignore


class AlgorithmicThinkingTrainer:
    """RPN + honesty drills for Algorithmic Thinking stars."""

    def __init__(self) -> None:
        self.galaxy_working_dir = Path("viewer/public/galaxy/working")
        self.galaxy_working_dir.mkdir(parents=True, exist_ok=True)
        self._trainer: Optional[MeaningClusterTrainer] = None

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
        conda_path = shutil.which("conda")
        current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
        if conda_path is None:
            print("⚠️  Conda executable not found; ensure dependencies are available before running heavy trainers.")
            return
        if current_env != "k3d-cranium":
            label = current_env or "unknown"
            print(
                "⚠️  Detected conda env '",
                label,
                "' — activate 'k3d-cranium' (conda activate k3d-cranium) for full GPU/RLWHF support.",
                sep="",
            )
        else:
            print("✅ Conda env k3d-cranium detected.")

    def train_algorithmic_thinking(self) -> None:
        """Execute algorithmic thinking drills across curated stars with RLWHF teachers."""
        print("🧠 Training Algorithmic Thinking — RPN, PTX, Honesty, RLWHF...")
        self.ensure_env()

        stars = self.load_stars_by_tag("algorithmic_thinking")
        if not stars:
            print("⚠️  No algorithmic thinking stars found — run library ingest first.")
            return

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
                initial_timeout=240,
                timeout=75,
            )
        except Exception as exc:
            print(f"❌ TeacherEvaluator unavailable — RLWHF scoring skipped: {exc}")

        for star in stars:
            star_name = star.get("name", star.get("id", "unknown"))
            print(f"\n📌 Star: {star_name}")
            for query in self.generate_rpn_queries(star):
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
                    )
                    deep_feedback = teacher.evaluate_response(
                        ai_response=predicted,
                        model="exaone-deep:latest",
                    )
                    score = float(quick_feedback.get("score", 0.0))
                    deep_score = deep_feedback.get("score")
                    if isinstance(deep_score, (int, float)) and deep_score > score:
                        score = float(deep_score)
                    explanation_text = deep_feedback.get("explanation") or quick_feedback.get("explanation", "")
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
