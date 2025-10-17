import json
from types import SimpleNamespace

import numpy as np

from knowledge3d.ingestion.lexicons.lexicon_ingestor import LexiconIngestor


class StubTextIngestor:
    def __init__(self):
        self.calls = []

    def ingest_sentence(self, lang: str, sentence: str):
        self.calls.append((lang, sentence))
        seed = len(self.calls)
        embedding = np.full(128, seed, dtype=np.float32)
        return {"embedding_128": embedding}


class StubSwarmProcessor:
    def __init__(self):
        self.calls = []

    def process_language_embedding(self, embedding_128, *, modality, language, include_diagnostics=False):
        self.calls.append((modality, language))
        embedding = np.asarray(embedding_128, dtype=np.float32)
        position = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        return SimpleNamespace(
            refined_embedding=embedding,
            position_3d=position,
            diagnostics=None,
            modality=modality,
            language=language,
        )


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
        return self._examples


class FakeWordNet:
    def __init__(self, synsets):
        self._synsets = synsets

    def all_synsets(self):
        return self._synsets


def test_ingest_simple_vocabulary(tmp_path):
    text_ingestor = StubTextIngestor()
    swarm = StubSwarmProcessor()
    ingestor = LexiconIngestor(
        text_ingestor=text_ingestor,
        swarm_processor=swarm,
        output_root=tmp_path,
    )

    summary = ingestor.ingest_simple_vocabulary("en", ["alpha", "beta"], output_path=tmp_path / "sample.json")

    assert summary["token_count"] == 2
    data = json.loads((tmp_path / "sample.json").read_text())
    assert data["language"] == "en"
    assert len(data["tokens"]) == 2
    assert text_ingestor.calls[0][1] == "alpha"
    assert swarm.calls == [("text", "en"), ("text", "en")]


def test_ingest_wordnet_en_with_stub(tmp_path):
    text_ingestor = StubTextIngestor()
    swarm = StubSwarmProcessor()
    fake_wordnet = FakeWordNet(
        [
            FakeSynset("think.v.01", "to use the mind", ["He thinks quickly"]),
            FakeSynset("reflect.v.02", "to consider carefully"),
        ]
    )

    ingestor = LexiconIngestor(
        text_ingestor=text_ingestor,
        swarm_processor=swarm,
        output_root=tmp_path,
    )

    summary = ingestor.ingest_wordnet_en(output_path=tmp_path / "wordnet.json", wordnet_module=fake_wordnet)
    assert summary["synset_count"] == 2

    payload = json.loads((tmp_path / "wordnet.json").read_text())
    assert payload["source"] == "wordnet"
    assert len(payload["synsets"]) == 2
    assert payload["synsets"][0]["lemma"] == "think"
