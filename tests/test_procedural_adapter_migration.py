from __future__ import annotations

from pathlib import Path

from knowledge3d.cranium.procedural_adapter_weights import ProceduralAdapterWeights
from knowledge3d.cranium.trm_adapters import SelfUpdatingAdapter


def test_procedural_adapter_migration_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "navigator_adapter.zip"
    adapter = SelfUpdatingAdapter(shape=(4, 4), rank=2, specialist_name="navigator")
    adapter.A.set_flat([0.1 * idx for idx in range(adapter.A.size)])
    adapter.B.set_flat([0.2 * idx for idx in range(adapter.B.size)])
    adapter.A_shadow.set_flat([0.3 * idx for idx in range(adapter.A_shadow.size)])
    adapter.B_shadow.set_flat([0.4 * idx for idx in range(adapter.B_shadow.size)])
    adapter.procedural_weights.sync_from_legacy_adapter(adapter)

    adapter.save(path)
    metadata = SelfUpdatingAdapter.peek_saved_metadata(path)
    assert metadata["storage"] == ProceduralAdapterWeights.STORAGE_KIND

    restored = SelfUpdatingAdapter(shape=(4, 4), rank=2, specialist_name="navigator")
    restored.load(path)

    assert restored.A.to_flat_list() == adapter.A.to_flat_list()
    assert restored.B.to_flat_list() == adapter.B.to_flat_list()
    assert restored.A_shadow.to_flat_list() == adapter.A_shadow.to_flat_list()
    assert restored.B_shadow.to_flat_list() == adapter.B_shadow.to_flat_list()
    assert restored.procedural_weights.to_payload()["storage_kind"] == ProceduralAdapterWeights.STORAGE_KIND


def test_procedural_adapter_shadow_absorb_updates_shadow_only() -> None:
    adapter = SelfUpdatingAdapter(shape=(4, 4), rank=2, specialist_name="navigator")
    procedural = adapter.procedural_weights
    primary_before = procedural.primary_a.to_flat_list()
    shadow_before = procedural.shadow_a.to_flat_list()
    gradient = [[0.1] * 4 for _ in range(4)]

    norm = procedural.absorb_contrast(gradient, lr=0.01, shadow=True)

    assert norm > 0.0
    assert procedural.primary_a.to_flat_list() == primary_before
    assert procedural.shadow_a.to_flat_list() != shadow_before
