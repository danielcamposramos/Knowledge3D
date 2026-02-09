from __future__ import annotations

from knowledge3d.knowledgeverse.ternary_quality_memory import TernaryQualityMemory


class _FakeGalaxyManager:
    def __init__(self):
        self.rows = []

    def add_entry(self, galaxy_name, entry):
        self.rows.append((galaxy_name, entry))


class _FakeKV:
    def __init__(self):
        self.events = []
        self.galaxy_manager = _FakeGalaxyManager()

    def log_event(self, event_type, event_data):
        self.events.append((event_type, event_data))


def test_quality_memory_update_and_reload(tmp_path):
    state_path = tmp_path / "quality.json"
    mem = TernaryQualityMemory(state_path=state_path, alpha=0.2, emit_galaxy_entries=False)

    rec = mem.update(
        pattern_id="p1",
        outcome=1,
        confidence=0.9,
        transfer_signal=1.0,
    )
    assert rec is not None
    assert rec.pattern_id == "p1"
    assert rec.count == 1
    assert rec.prior > 0.0
    assert rec.pool_id.startswith("pool_")

    mem2 = TernaryQualityMemory(state_path=state_path, emit_galaxy_entries=False)
    rec2 = mem2.get_prior("p1")
    assert rec2 is not None
    assert rec2.count == 1
    assert rec2.prior == rec.prior


def test_quality_memory_emits_event_and_grammar_entry(tmp_path):
    state_path = tmp_path / "quality_emit.json"
    mem = TernaryQualityMemory(state_path=state_path, emit_galaxy_entries=True)
    kv = _FakeKV()

    rec = mem.update(
        pattern_id="arc_rule_1",
        outcome=-1,
        confidence=0.7,
        transfer_signal=-1.0,
        knowledgeverse=kv,
        specialist="visual",
        galaxy="Grammar",
        source="test",
    )
    assert rec is not None
    assert kv.events
    assert kv.events[-1][0] == "ternary_quality_update"
    assert kv.galaxy_manager.rows
    galaxy_name, entry = kv.galaxy_manager.rows[-1]
    assert galaxy_name == "Grammar"
    assert entry["category"] == "quality_prior"
    assert entry["metadata"]["pattern_id"] == "arc_rule_1"


def test_quality_memory_ternary_pooling_not_binary(tmp_path):
    state_path = tmp_path / "quality_pooling.json"
    mem = TernaryQualityMemory(state_path=state_path, emit_galaxy_entries=False)

    mem.update(pattern_id="p_good", outcome=1, confidence=0.95, transfer_signal=1.0)
    mem.update(pattern_id="p_mid", outcome=0, confidence=0.50, transfer_signal=0.0)
    mem.update(pattern_id="p_bad", outcome=-1, confidence=0.20, transfer_signal=-1.0)

    good = mem.get_prior("p_good")
    mid = mem.get_prior("p_mid")
    bad = mem.get_prior("p_bad")
    assert good is not None and mid is not None and bad is not None
    assert good.correctness_t == 1
    assert mid.correctness_t == 0
    assert bad.correctness_t == -1
    # multi-pool behavior: at least two distinct pools should appear.
    assert len({good.pool_id, mid.pool_id, bad.pool_id}) >= 2
