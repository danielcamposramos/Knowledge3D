from __future__ import annotations

from knowledge3d.knowledgeverse.specialist_base import SpecialistBase
from knowledge3d.knowledgeverse.specialist_spawner import SpecialistSpawner


def test_spawner_triggers_on_frequency_threshold(tmp_path):
    root = SpecialistBase(name="MathSpecialist", domain="math", storage_dir=tmp_path)
    spawner = SpecialistSpawner(
        root=root,
        storage_path=tmp_path / "spawner.json",
        frequency_threshold=3,
        low_confidence_min_samples=20,
    )

    decision = None
    for _ in range(3):
        decision = spawner.observe(
            parent=root,
            query="prove topology manifold theorem",
            confidence=0.8,
            success=True,
            domain_hint=None,
        )

    assert decision is not None
    assert decision.reason == "frequency_threshold"
    assert root.find(decision.child) is not None


def test_spawner_triggers_on_low_confidence_gap(tmp_path):
    root = SpecialistBase(name="VisualSpecialist", domain="visual", storage_dir=tmp_path)
    spawner = SpecialistSpawner(
        root=root,
        storage_path=tmp_path / "spawner.json",
        frequency_threshold=9999,
        low_confidence_threshold=0.7,
        low_confidence_min_samples=4,
    )

    decision = None
    for _ in range(4):
        decision = spawner.observe(
            parent=root,
            query="arc grid transform pattern",
            confidence=0.4,
            success=False,
            domain_hint=None,
        )

    assert decision is not None
    assert decision.reason == "performance_gap"
    assert root.find(decision.child) is not None


def test_spawner_persistence_roundtrip(tmp_path):
    root = SpecialistBase(name="PhysicsSpecialist", domain="reality", storage_dir=tmp_path)
    storage = tmp_path / "spawner.json"
    spawner = SpecialistSpawner(
        root=root,
        storage_path=storage,
        frequency_threshold=2,
        low_confidence_min_samples=10,
    )
    for _ in range(2):
        spawner.observe(
            parent=root,
            query="energy momentum conservation",
            confidence=0.9,
            success=True,
            domain_hint=None,
        )
    spawner.persist()

    reloaded = SpecialistSpawner(
        root=root,
        storage_path=storage,
        frequency_threshold=2,
        low_confidence_min_samples=10,
    )
    assert "PhysicsSpecialist" in reloaded.stats
    assert reloaded.decisions
