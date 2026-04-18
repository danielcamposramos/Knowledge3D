from __future__ import annotations

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.navigator_specialist import MEANING_CLASS_INDEX


MULTI_HOP_PROMPTS = [
    "Janet buys 3 notebooks at $4 each and then uses a $2 coupon. How much does she pay?",
    "A train travels 2 hours at 50 mph and then 1 hour at 30 mph. How far did it go altogether?",
    "If all blue boxes contain 4 marbles and there are 3 blue boxes, how many marbles are there?",
    "Mia reads 12 pages on Monday and twice that on Tuesday. How many pages did she read in total?",
    "A baker makes 18 rolls, sells 7, and packs the rest equally into 11 bags. How many go in each bag?",
]

DIRECT_COMPUTE_PROMPTS = [
    "2 + 3 = ?",
    "sqrt(16)",
    "14 * 6",
    "100 / 4",
    "9 squared",
]


def _meaning_scores(kv: Knowledgeverse, prompt: str) -> list[float]:
    task = {"query": prompt, "question": prompt, "prompt": prompt}
    _, meaning_dist, _, _ = kv._navigator_emission(
        query_embedding=kv._embed_query_gpu(prompt, task=task),
        task=task,
        query_text=prompt,
        options=None,
        stars=None,
    )
    return list(meaning_dist)


def test_multi_hop_prompts_separate_from_numeric_compute(tmp_path) -> None:
    kv = Knowledgeverse(
        storage_root=tmp_path / "kv_multi_hop_separability",
        eager_load_default_galaxies=False,
        bootstrap_foundational_galaxies=False,
        include_runtime_artifacts=False,
        include_runtime_language_enrichment=False,
    )

    multi_hop_scores = [_meaning_scores(kv, prompt) for prompt in MULTI_HOP_PROMPTS]
    _direct_scores = [_meaning_scores(kv, prompt) for prompt in DIRECT_COMPUTE_PROMPTS]
    multi_hop_index = MEANING_CLASS_INDEX["MULTI_HOP_INFERENCE"]
    numeric_index = MEANING_CLASS_INDEX["NUMERIC_COMPUTE"]
    wins = sum(
        1
        for scores in multi_hop_scores
        if float(scores[multi_hop_index]) > float(scores[numeric_index])
    )

    assert wins >= 4
