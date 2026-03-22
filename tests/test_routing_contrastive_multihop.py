from __future__ import annotations

import json

import numpy as np
import pytest

from knowledge3d.bridge.headless_tablet import TabletIngest
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.sleeptime import SleepTimeConsolidation


class _FakeEmbedEngine:
    def embed_sentence_gpu(self, text: str):
        token = str(text or "")
        base = max(1, sum((idx + 1) * ord(ch) for idx, ch in enumerate(token)))
        return [float(((idx + 1) * base) % 997) / 997.0 for idx in range(16)]


def test_arc_embedding_uses_visual_semantics_not_task_id(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_arc_embed", eager_load_default_galaxies=False)
    monkeypatch.setattr(kv, "get_gpu_query_embedding_engine", lambda: _FakeEmbedEngine())

    shared_training = [
        {
            "input": [
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            "output": [
                [0, 2, 0],
                [0, 2, 0],
                [0, 0, 0],
            ],
        }
    ]
    task_a = {
        "type": "ARC_TASK",
        "task_id": "arc_alpha",
        "training_examples": shared_training,
        "input_grid": shared_training[0]["input"],
    }
    task_b = {
        "type": "ARC_TASK",
        "task_id": "arc_beta",
        "training_examples": shared_training,
        "input_grid": shared_training[0]["input"],
    }
    task_c = {
        "type": "ARC_TASK",
        "task_id": "arc_gamma",
        "training_examples": [
            {
                "input": [
                    [1, 0, 1],
                    [0, 0, 0],
                    [2, 0, 2],
                ],
                "output": [
                    [1, 3, 1],
                    [0, 3, 0],
                    [2, 3, 2],
                ],
            }
        ],
        "input_grid": [
            [1, 0, 1],
            [0, 0, 0],
            [2, 0, 2],
        ],
    }

    emb_a = kv._embed_query_gpu("visual transformation task", task=task_a)
    emb_b = kv._embed_query_gpu("visual transformation task", task=task_b)
    assert emb_a == pytest.approx(emb_b)
    visual_text = kv._arc_visual_feature_text(task_c)
    assert "find objects connected regions discrete shapes separate groups" in visual_text
    assert "color remap substitution pattern change" in visual_text
    assert "rotate mirror transform reflect symmetry" in visual_text
    query_text = kv._query_text("visual transformation task", task=task_a)
    assert query_text.startswith("visual transformation task")
    assert "ARC_TASK" not in query_text
    assert "arc_alpha" not in query_text


def test_arc_tablet_query_uses_visual_wording():
    envelope = TabletIngest.arc_task(
        task_id="arc_demo",
        training_examples=[],
        input_grid=[[0]],
    )

    assert envelope.query == "visual transformation task"
    assert envelope.task["query"] == "visual transformation task"


def test_build_candidate_graph_edges_populates_neighbors(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_lhe_edges", eager_load_default_galaxies=False)
    monkeypatch.setattr(
        kv,
        "_embedding_similarity_matrix",
        lambda _sources, _targets: [
            [1.0, 0.92, 0.12],
            [0.92, 1.0, 0.41],
            [0.12, 0.41, 1.0],
        ],
    )

    candidates = [
        {
            "candidate_global_idx": 10,
            "graph_neighbors": [],
            "match": {"embedding16": [1.0] + [0.0] * 15},
        },
        {
            "candidate_global_idx": 11,
            "graph_neighbors": [],
            "match": {"embedding16": [0.0, 1.0] + [0.0] * 14},
        },
        {
            "candidate_global_idx": 12,
            "graph_neighbors": [],
            "match": {"embedding16": [0.0, 0.0, 1.0] + [0.0] * 13},
        },
    ]

    kv._build_candidate_graph_edges(candidates, similarity_threshold=0.2, max_neighbors=4)

    assert candidates[0]["graph_neighbors"] == [11]
    assert candidates[1]["graph_neighbors"] == [10, 12]
    assert candidates[2]["graph_neighbors"] == [11]


def test_sleeptime_contrastive_trains_all_specialists(tmp_path, monkeypatch):
    storage_root = tmp_path / "kv_contrastive"
    kv = Knowledgeverse(storage_root=storage_root, eager_load_default_galaxies=False)
    monkeypatch.setattr(kv, "get_gpu_query_embedding_engine", lambda: _FakeEmbedEngine())

    logs_dir = storage_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    session_id = "session-contrastive"
    health_log_path = logs_dir / "health_log.jsonl"
    rows = [
        {"session_id": session_id, "suite": "math", "question": "2+2=?", "expected": "4", "correct": True},
        {"session_id": session_id, "suite": "math", "question": "3+5=?", "answer": "11", "correct": False},
        {"session_id": session_id, "suite": "arc", "question": "ARC task demo", "answer": [[0, 0], [0, 0]], "correct": False},
        {"session_id": session_id, "suite": "lhe", "question": "All A are B; all B are C", "expected": "All A are C", "correct": True},
        {"session_id": session_id, "suite": "lhe", "question": "No A are B", "answer": "All A are B", "correct": False},
        {"session_id": session_id, "suite": "mmlu", "question": "What is 1+1?", "expected": "A", "correct": True},
        {"session_id": session_id, "suite": "mmlu", "question": "Capital of France?", "answer": "B", "correct": False},
    ]
    with health_log_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.write('{"broken": ')

    run_state_path = logs_dir / "health_log.full.run_state.json"
    run_state_path.write_text(json.dumps({"session_id": session_id}), encoding="utf-8")

    sleeptime = SleepTimeConsolidation(
        knowledgeverse=kv,
        journal_path=logs_dir / "sleeptime_journal.jsonl",
        health_log_path=health_log_path,
    )
    summary = sleeptime._run_contrastive_training()

    assert summary["skipped"] is False
    assert summary["rows"] == 7
    trained = summary["specialists_trained"]
    assert trained["math"]["trained"] is True
    assert trained["math"]["positives"] == 1
    assert trained["math"]["negatives"] == 1
    assert trained["visual"]["trained"] is True
    assert trained["visual"]["positives"] == 0
    assert trained["visual"]["negatives"] == 1
    assert trained["grammar"]["trained"] is True
    assert trained["grammar"]["positives"] == 1
    assert trained["grammar"]["negatives"] == 1
    assert trained["chat"]["trained"] is True
    assert trained["chat"]["positives"] == 1
    assert trained["chat"]["negatives"] == 1
    assert summary["checkpoint"]["saved"] is True
    assert (storage_root / "checkpoints" / "adaptive_swarm" / "swarm_state.json").exists()
