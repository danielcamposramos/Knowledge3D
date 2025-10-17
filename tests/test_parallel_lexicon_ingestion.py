import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from knowledge3d.ingestion.lexicons.parallel_lexicon_ingestor import ParallelLexiconIngestor


class FakeSynset:
    def __init__(self, name: str, definition: str, examples=None):
        self._name = name
        self._definition = definition
        self._examples = examples or []

    def name(self) -> str:
        return self._name

    def definition(self) -> str:
        return self._definition

    def examples(self):
        return list(self._examples)


class FakeWordNet:
    def __init__(self, synsets):
        self._synsets = synsets

    def all_synsets(self):
        return self._synsets


def fake_gpu_batch(batch):
    results = []
    for idx, item in enumerate(batch):
        embedding = np.full(3, idx, dtype=np.float32).tolist()
        results.append(
            {
                "synset": item["synset_name"],
                "lemma": item["lemma"],
                "definition": item["definition"],
                "examples": item["examples"],
                "position_3d": embedding,
                "embedding": embedding,
            }
        )
    return results


def test_parallel_lexicon_ingestor_with_stub(tmp_path):
    fake_synsets = [
        FakeSynset("alpha.n.01", "first definition", ["example a"]),
        FakeSynset("beta.n.01", "second definition"),
        FakeSynset("gamma.n.01", "third definition"),
    ]
    fake_wordnet = FakeWordNet(fake_synsets)

    output_file = tmp_path / "wordnet_stub.json"

    ingestor = ParallelLexiconIngestor(
        num_workers=0,
        batch_size=2,
        gpu_batch_processor=fake_gpu_batch,
    )

    metrics = ingestor.ingest_wordnet_parallel(
        output_file,
        wordnet_module=fake_wordnet,
    )

    assert metrics["synset_count"] == len(fake_synsets)
    assert Path(output_file).exists()

    payload = json.loads(output_file.read_text())
    assert payload["synset_count"] == len(fake_synsets)
    assert payload["synsets"][0]["lemma"] == "alpha"
