"""Knowledgeverse SleepTime integration stubs with resilience wrappers."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
        return {
            "success": True,
            "stage": "logic",
            "updated_specialists": summary.get("updated_specialists", []),
            "updated_count": int(summary.get("updated_count", 0)),
            "weights_path": summary.get("weights_path", ""),
        }

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
