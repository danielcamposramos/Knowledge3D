from __future__ import annotations

import inspect
from pathlib import Path

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


_BANNED = ("MMLU", "LHE", "GSM8K", "ARC_TASK", "competition", "benchmark")


def test_hot_path_functions_do_not_reference_benchmark_labels() -> None:
    functions = (
        Knowledgeverse._legacy_surface_kind,
        Knowledgeverse._infer_query_mode,
        Knowledgeverse._embed_query_batch_gpu,
        Knowledgeverse._build_n_chain_swarm_packet,
        Knowledgeverse._apply_specialist_swarm_features,
        Knowledgeverse._resolve_halting_weights,
    )
    for fn in functions:
        source = inspect.getsource(fn)
        for token in _BANNED:
            assert token not in source, f"{fn.__name__} leaked benchmark token {token}"


def test_sovereign_hot_path_file_is_free_of_benchmark_labels() -> None:
    source = Path("knowledge3d/knowledgeverse/sovereign_hot_path.py").read_text(encoding="utf-8")
    for token in _BANNED:
        assert token not in source
