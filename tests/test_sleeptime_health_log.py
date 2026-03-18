from __future__ import annotations

import json
from pathlib import Path

from knowledge3d.knowledgeverse.sleeptime import SleepTimeConsolidation


def test_sleeptime_stage_a_summarizes_health_log(tmp_path: Path) -> None:
    health_log = tmp_path / "health_log.jsonl"
    health_log.write_text(
        "\n".join(
            [
                json.dumps({"question_id": "gsm8k_0", "suite": "gsm8k", "correct": True}),
                json.dumps({"question_id": "mmlu_0", "suite": "mmlu", "correct": False}),
                "",
            ]
        ),
        encoding="utf-8",
    )

    sleep = SleepTimeConsolidation(health_log_path=health_log, consume_health_log=False)
    stage_a = sleep._stage_a_knowledge()

    assert stage_a["success"] is True
    assert stage_a["health_log"]["total"] == 2
    assert stage_a["health_log"]["correct"] == 1
    assert stage_a["health_log"]["incorrect"] == 1
    assert stage_a["health_log"]["suites"]["gsm8k"] == 1


def test_sleeptime_can_consume_health_log(tmp_path: Path) -> None:
    health_log = tmp_path / "health_log.jsonl"
    health_log.write_text(
        json.dumps({"question_id": "gsm8k_0", "suite": "gsm8k", "correct": True}) + "\n",
        encoding="utf-8",
    )

    sleep = SleepTimeConsolidation(health_log_path=health_log, consume_health_log=True)
    stage_a = sleep._stage_a_knowledge()

    assert stage_a["health_log"]["consumed"] is True
    assert health_log.read_text(encoding="utf-8") == ""


def test_sleeptime_treats_voice_entries_as_neutral(tmp_path: Path) -> None:
    health_log = tmp_path / "health_log.jsonl"
    health_log.write_text(
        json.dumps({"question_id": "voice_0", "suite": "voice", "source": "voice", "correct": None}) + "\n",
        encoding="utf-8",
    )

    sleep = SleepTimeConsolidation(health_log_path=health_log, consume_health_log=False)
    stage_a = sleep._stage_a_knowledge()

    assert stage_a["health_log"]["total"] == 1
    assert stage_a["health_log"]["correct"] == 0
    assert stage_a["health_log"]["incorrect"] == 0
    assert stage_a["health_log"]["neutral"] == 1
