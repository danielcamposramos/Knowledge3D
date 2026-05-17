"""Knowledgeverse SleepTime integration stubs with resilience wrappers."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .navigator_specialist import derive_symlink_histogram, meaning_class_from_symlink_votes
from .resilience import SelfHealingWrapper
from .temporal_metadata import TemporalMetadataManager


class SleepTimeError(RuntimeError):
    """Raised when a SleepTime stage fails."""


class SleepTimeConsolidation:
    """Minimal SleepTime execution surface for MVP resilience integration."""

    def __init__(
        self,
        knowledgeverse: object | None = None,
        manifest_version: str = "unknown",
        journal_path: str | Path = "../Knowledge3D.local/logs/sleeptime_journal.jsonl",
        health_log_path: str | Path | None = "../Knowledge3D.local/logs/health_log.jsonl",
        consume_health_log: bool = False,
    ):
        self.kv = knowledgeverse
        if knowledgeverse is not None and hasattr(knowledgeverse, "manifest_version"):
            manifest_version = str(getattr(knowledgeverse, "manifest_version"))
        self.temporal_manager = TemporalMetadataManager(
            manifest_version=manifest_version,
            region_id="sleeptime",
        )
        self.journal_path = Path(journal_path)
        self.health_log_path = Path(health_log_path) if health_log_path else None
        self.consume_health_log = bool(consume_health_log)

    @SelfHealingWrapper.with_fallback(
        fallback_func=lambda self: self._load_last_good_checkpoint(),
        cache_duration=300.0,
    )
    def execute(self) -> Any:
        transaction_temporal = self.temporal_manager.create_metadata()
        transaction_id = transaction_temporal.event_id

        try:
            stage_a_temporal = self.temporal_manager.create_metadata(
                parent_event_id=transaction_id
            )
            stage_a_result = self._stage_a_knowledge()
            stage_a_result["temporal"] = asdict(stage_a_temporal)
            if not stage_a_result.get("success", False):
                raise SleepTimeError("Stage A failed")

            stage_b_temporal = self.temporal_manager.create_metadata(
                parent_event_id=transaction_id
            )
            stage_b_result = self._stage_b_logic()
            stage_b_result["temporal"] = asdict(stage_b_temporal)
            if not stage_b_result.get("success", False):
                raise SleepTimeError("Stage B failed")

            result = {
                "success": True,
                "transaction_id": transaction_id,
                "temporal": asdict(transaction_temporal),
                "stage_a": stage_a_result,
                "stage_b": stage_b_result,
            }
            self._commit_transaction(result)
            checkpoint_saver = None
            if self.kv is not None:
                checkpoint_saver = getattr(self.kv, "_save_consolidated_state", None)
                if checkpoint_saver is None:
                    checkpoint_saver = getattr(self.kv, "save_consolidated_state", None)
            if callable(checkpoint_saver):
                try:
                    result["checkpoint"] = checkpoint_saver()
                except Exception as exc:
                    result["checkpoint_error"] = str(exc)
            elif self.kv is not None and hasattr(self.kv, "save_house_state"):
                try:
                    result["house_state"] = self.kv.save_house_state()
                except Exception as exc:
                    result["house_state_error"] = str(exc)
            return result
        except SleepTimeError as exc:
            rollback_temporal = self.temporal_manager.create_metadata(
                parent_event_id=transaction_id
            )
            self._rollback_transaction(
                transaction_id=transaction_id,
                error=str(exc),
                temporal=asdict(rollback_temporal),
            )
            raise

    def _execute_two_phase_commit(self) -> Any:
        raise NotImplementedError("Bind SleepTime to concrete consolidation pipeline")

    def _load_last_good_checkpoint(self) -> Any:
        return {
            "success": False,
            "fallback": "last_good_checkpoint",
        }

    def _stage_a_knowledge(self) -> dict[str, Any]:
        result = {"success": True, "stage": "knowledge"}
        health_summary = self._summarize_health_log()
        if health_summary is not None:
            result["health_log"] = health_summary
        return result

    def _stage_b_logic(self) -> dict[str, Any]:
        if self.kv is None:
            return {"success": True, "stage": "logic", "updated_specialists": []}

        trm = getattr(self.kv, "trm_navigator", None)
        shadow_copy = getattr(self.kv, "shadow_copy", None)
        if trm is None or shadow_copy is None:
            return {"success": True, "stage": "logic", "updated_specialists": []}

        events = list(getattr(shadow_copy, "event_buffer", []))
        summary = trm.consolidate_weights_from_events(events)
        contrastive_summary = self._run_contrastive_training()
        jarvis_summary = None
        if hasattr(self.kv, "jarvis_sleep_consolidation"):
            try:
                jarvis_summary = self.kv.jarvis_sleep_consolidation(persist=False)
            except Exception:
                jarvis_summary = None
        return {
            "success": True,
            "stage": "logic",
            "updated_specialists": summary.get("updated_specialists", []),
            "updated_count": int(summary.get("updated_count", 0)),
            "weights_path": summary.get("weights_path", ""),
            "contrastive": contrastive_summary,
            **({"jarvis": jarvis_summary} if isinstance(jarvis_summary, dict) else {}),
        }

    def _load_health_log_rows(self, *, session_id: str | None = None) -> list[dict[str, Any]]:
        path = self.health_log_path
        if path is None or not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                if session_id and str(payload.get("session_id") or "").strip() != str(session_id).strip():
                    continue
                rows.append(payload)
        return rows

    def _current_session_id(self) -> str:
        if self.kv is None:
            return ""
        run_state_path = Path(getattr(self.kv, "storage_root", Path("."))) / "logs" / "health_log.full.run_state.json"
        if not run_state_path.exists():
            return ""
        try:
            payload = json.loads(run_state_path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        return str(payload.get("session_id") or "").strip()

    @staticmethod
    def _suite_specialist_name(suite: str) -> str:
        token = str(suite or "").strip().lower()
        if token in {"math", "gsm8k"}:
            return "math"
        if token == "arc":
            return "visual"
        if token == "lhe":
            return "grammar"
        return "chat"

    def _run_contrastive_training(self) -> dict[str, Any]:
        if self.kv is None:
            return {"skipped": True, "reason": "no_knowledgeverse"}
        sovereign_runtime = getattr(self.kv, "_sovereign_hot_path", None)
        device_learning: dict[str, Any] | None = None
        if sovereign_runtime is not None and hasattr(sovereign_runtime, "current_learning_state"):
            try:
                device_learning = dict(sovereign_runtime.current_learning_state())
            except Exception as exc:
                return {
                    "skipped": True,
                    "reason": f"device_lesson_ring_error:{exc}",
                }
        swarm = getattr(self.kv, "adaptive_swarm", None)
        if swarm is None or not hasattr(swarm, "train_specialist_contrastive"):
            return {"skipped": True, "reason": "no_swarm"}
        session_id = self._current_session_id()
        rows = self._load_health_log_rows(session_id=session_id or None)
        if not rows:
            return {"skipped": True, "reason": "no_health_rows", "session_id": session_id}
        try:
            engine = self.kv.get_gpu_query_embedding_engine()
        except Exception as exc:
            return {"skipped": True, "reason": f"no_query_embedding_engine:{exc}"}

        specialist_positive: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
            "math": [],
            "visual": [],
            "grammar": [],
            "chat": [],
        }
        specialist_negative: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
            "math": [],
            "visual": [],
            "grammar": [],
            "chat": [],
        }
        navigator = getattr(self.kv, "navigator_specialist", None)
        navigator_samples: list[dict[str, Any]] = []
        for row in rows:
            suite = str(row.get("suite", "")).strip().lower()
            question = str(row.get("question", "")).strip()
            if not question:
                continue
            specialist_name = self._suite_specialist_name(suite)
            if bool(row.get("correct", False)):
                expected = row.get("expected")
                if expected is None or (isinstance(expected, str) and not expected.strip()):
                    continue
                try:
                    question_values = engine.embed_sentence_gpu(question)
                    expected_text = expected if isinstance(expected, str) else json.dumps(expected, ensure_ascii=False, sort_keys=True)
                    expected_values = engine.embed_sentence_gpu(str(expected_text))
                except Exception:
                    continue
                q_emb = np.asarray([float(value) for value in list(question_values)[:16]], dtype=np.float32)
                e_emb = np.asarray([float(value) for value in list(expected_values)[:16]], dtype=np.float32)
                if q_emb.size == 0 or e_emb.size == 0:
                    continue
                specialist_positive[specialist_name].append((q_emb, e_emb))
                if navigator is not None:
                    navigator_sample = self._build_navigator_training_sample(
                        question=question,
                        expected=expected,
                        correct=True,
                        query_embedding=q_emb,
                        knowledgeverse=self.kv,
                        navigator=navigator,
                        retrieved_stars=row.get("retrieved_stars"),
                    )
                    if isinstance(navigator_sample, dict):
                        navigator_samples.append(navigator_sample)
                continue

            answer = row.get("answer")
            expected = row.get("expected")
            if answer is None or (isinstance(answer, str) and not answer.strip()):
                if expected is None or (isinstance(expected, str) and not expected.strip()):
                    continue
                try:
                    question_values = engine.embed_sentence_gpu(question)
                    expected_text = expected if isinstance(expected, str) else json.dumps(expected, ensure_ascii=False, sort_keys=True)
                    expected_values = engine.embed_sentence_gpu(str(expected_text))
                except Exception:
                    continue
                q_emb = np.asarray([float(value) for value in list(question_values)[:16]], dtype=np.float32)
                e_emb = np.asarray([float(value) for value in list(expected_values)[:16]], dtype=np.float32)
                if q_emb.size == 0 or e_emb.size == 0:
                    continue
                specialist_positive[specialist_name].append((q_emb, e_emb))
                if navigator is not None:
                    navigator_sample = self._build_navigator_training_sample(
                        question=question,
                        expected=expected,
                        correct=True,
                        query_embedding=q_emb,
                        knowledgeverse=self.kv,
                        navigator=navigator,
                        retrieved_stars=row.get("retrieved_stars"),
                    )
                    if isinstance(navigator_sample, dict):
                        navigator_samples.append(navigator_sample)
                continue
            try:
                question_values = engine.embed_sentence_gpu(question)
                answer_text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False, sort_keys=True)
                answer_values = engine.embed_sentence_gpu(str(answer_text))
            except Exception:
                continue
            q_emb = np.asarray([float(value) for value in list(question_values)[:16]], dtype=np.float32)
            a_emb = np.asarray([float(value) for value in list(answer_values)[:16]], dtype=np.float32)
            if q_emb.size == 0 or a_emb.size == 0:
                continue
            specialist_negative[specialist_name].append((q_emb, a_emb))
            if navigator is not None:
                navigator_sample = self._build_navigator_training_sample(
                    question=question,
                    expected=answer,
                    correct=False,
                    query_embedding=q_emb,
                    knowledgeverse=self.kv,
                    navigator=navigator,
                    retrieved_stars=row.get("retrieved_stars"),
                )
                if isinstance(navigator_sample, dict):
                    navigator_samples.append(navigator_sample)

        results: dict[str, Any] = {}
        trained_any = False
        for specialist_name in specialist_positive:
            positive_pairs = specialist_positive[specialist_name]
            negative_pairs = specialist_negative[specialist_name]
            if not positive_pairs and not negative_pairs:
                results[specialist_name] = {"trained": False, "positives": 0, "negatives": 0, "reason": "no_pairs"}
                continue
            try:
                stats = swarm.train_specialist_contrastive(
                    specialist_name,
                    positive_pairs=positive_pairs,
                    negative_pairs=negative_pairs,
                )
            except Exception as exc:
                results[specialist_name] = {
                    "trained": False,
                    "positives": len(positive_pairs),
                    "negatives": len(negative_pairs),
                    "error": str(exc),
                }
                continue
            trained_any = True
            results[specialist_name] = {
                "trained": True,
                "positives": len(positive_pairs),
                "negatives": len(negative_pairs),
                "contrast_signal": float(stats.get("contrast_signal", stats.get("avg_loss", 0.0))),
                "positive_contrast_signal": float(stats.get("positive_loss", 0.0)),
                "negative_contrast_signal": float(stats.get("negative_loss", 0.0)),
                "steps": int(stats.get("steps", 0)),
            }
        if navigator is not None and navigator_samples:
            for sample in navigator_samples:
                navigator.update_from_trace(sample)
            split_idx = max(1, int(len(navigator_samples) * 0.8))
            navigator_train = navigator_samples[:split_idx]
            navigator_val = navigator_samples[split_idx:] if split_idx < len(navigator_samples) else navigator_samples[-1:]
            try:
                stats = swarm.consolidate_specialist_dream_cycle(
                    "navigator",
                    consolidation_wave=navigator_train,
                    gate_check_samples=navigator_val,
                )
            except Exception as exc:
                results["navigator"] = {
                    "trained": False,
                    "positives": sum(1 for sample in navigator_train if bool(sample.get("correct", True))),
                    "negatives": sum(1 for sample in navigator_train if not bool(sample.get("correct", True))),
                    "error": str(exc),
                }
            else:
                trained_any = True
                navigator._training_state["last_training_stats"] = {
                    "contrast_signal": float(stats.get("contrast_signal", stats.get("avg_loss", 0.0))),
                    "steps": int(stats.get("steps", 0)),
                    "gate_check_samples": int(stats.get("gate_check_samples", stats.get("validation_samples", 0))),
                    "consolidation_wave_size": int(stats.get("consolidation_wave_size", len(navigator_train))),
                }
                navigator.save_state()
                results["navigator"] = {
                    "trained": True,
                    "positives": sum(1 for sample in navigator_train if bool(sample.get("correct", True))),
                    "negatives": sum(1 for sample in navigator_train if not bool(sample.get("correct", True))),
                    "contrast_signal": float(stats.get("contrast_signal", stats.get("avg_loss", 0.0))),
                    "steps": int(stats.get("steps", 0)),
                    "gate_check_samples": int(stats.get("gate_check_samples", stats.get("validation_samples", 0))),
                }
        checkpoint = {}
        if trained_any and hasattr(self.kv, "_save_adaptive_swarm_state"):
            try:
                checkpoint = self.kv._save_adaptive_swarm_state()
            except Exception as exc:
                checkpoint = {"saved": False, "reason": str(exc)}
        return {
            "skipped": False,
            "session_id": session_id,
            "rows": len(rows),
            "specialists_trained": results,
            "checkpoint": checkpoint,
            **({"device_learning": device_learning} if isinstance(device_learning, dict) else {}),
        }

    @staticmethod
    def _build_navigator_training_sample(
        *,
        question: str,
        expected: Any,
        correct: bool,
        query_embedding: np.ndarray,
        knowledgeverse: Any,
        navigator: Any,
        retrieved_stars: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        stars: list[dict[str, Any]] = [
            dict(row)
            for row in list(retrieved_stars or [])
            if isinstance(row, dict)
        ]
        if not stars:
            galaxy_manager = getattr(knowledgeverse, "galaxy_manager", None)
            if galaxy_manager is not None and hasattr(galaxy_manager, "query"):
                try:
                    raw_rows = list(
                        galaxy_manager.query(
                            query_text=str(question),
                            specialist="any",
                            top_k=8,
                        )
                    )
                except Exception:
                    raw_rows = []
                for row in raw_rows:
                    if not isinstance(row, dict):
                        continue
                    entry = row.get("entry") if isinstance(row.get("entry"), dict) else row
                    if isinstance(entry, dict):
                        stars.append(dict(entry))
        meaning_class = meaning_class_from_symlink_votes(stars) or "FACTUAL_RECALL"
        trace = {
            "query_text": str(question),
            "query_embedding": [float(value) for value in list(query_embedding)[:64]],
            "symlink_histogram": derive_symlink_histogram(stars),
            "meaning_class": meaning_class,
            "halting_weight_vec": [1.0] * 9,
            "correct": bool(correct),
            "task_payload": {},
            "expected": expected,
            "retrieved_stars": [dict(row) for row in stars if isinstance(row, dict)],
        }
        return navigator._trace_to_training_sample(trace)

    def _commit_transaction(self, payload: dict[str, Any]) -> None:
        self._append_journal({"event": "commit", **payload})

    def _rollback_transaction(self, transaction_id: str, error: str, temporal: dict[str, Any]) -> None:
        self._append_journal(
            {
                "event": "rollback",
                "transaction_id": transaction_id,
                "error": error,
                "temporal": temporal,
            }
        )

    def _append_journal(self, payload: dict[str, Any]) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")

    def _summarize_health_log(self) -> dict[str, Any] | None:
        path = self.health_log_path
        if path is None or not path.exists():
            return None
        total = 0
        correct = 0
        incorrect = 0
        neutral = 0
        suites: dict[str, int] = {}
        frequent_correct: dict[tuple[str, str], int] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            suite = str(payload.get("suite", "unknown")).strip() or "unknown"
            suites[suite] = int(suites.get(suite, 0)) + 1
            correct_value = payload.get("correct", None)
            if isinstance(correct_value, bool) and correct_value:
                correct += 1
                question_id = str(payload.get("question_id", "")).strip()
                if question_id:
                    key = (suite, question_id)
                    frequent_correct[key] = int(frequent_correct.get(key, 0)) + 1
            elif isinstance(correct_value, bool):
                incorrect += 1
            else:
                neutral += 1
        if self.consume_health_log:
            path.write_text("", encoding="utf-8")
        top_patterns = sorted(
            (
                {"suite": suite, "question_id": question_id, "count": count}
                for (suite, question_id), count in frequent_correct.items()
            ),
            key=lambda row: (-int(row["count"]), str(row["suite"]), str(row["question_id"])),
        )[:10]
        return {
            "path": str(path),
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "neutral": neutral,
            "suites": dict(sorted(suites.items())),
            "frequent_correct_patterns": top_patterns,
            "consumed": bool(self.consume_health_log),
        }
