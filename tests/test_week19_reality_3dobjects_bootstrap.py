from __future__ import annotations

from knowledge3d.knowledgeverse.objects_3d_galaxy import bootstrap_3d_objects_galaxy
from knowledge3d.knowledgeverse.reality_galaxy import bootstrap_reality_galaxy
from knowledge3d.knowledgeverse.specialist_router import SpecialistRouter


def test_week19_reality_bootstrap_is_additive_and_idempotent(tmp_path):
    storage_root = tmp_path / "kv_week19"
    first = bootstrap_reality_galaxy(storage_root=storage_root)
    second = bootstrap_reality_galaxy(storage_root=storage_root)

    assert first["appended"] > 0
    assert first["after"] >= 800
    assert second["appended"] == 0
    assert second["after"] == first["after"]


def test_week19_3dobjects_bootstrap_is_additive_and_idempotent(tmp_path):
    storage_root = tmp_path / "kv_week19"
    first = bootstrap_3d_objects_galaxy(storage_root=storage_root)
    second = bootstrap_3d_objects_galaxy(storage_root=storage_root)

    assert first["appended"] > 0
    assert first["after"] >= 200
    assert second["appended"] == 0
    assert second["after"] == first["after"]


def test_week19_router_includes_3dobjects_for_physics_and_multi():
    router = SpecialistRouter()
    physics_route = router.route("simulate collision in 3d mesh field", specialist="auto")
    assert physics_route["specialist"] in {"physics", "cartographer"}
    if physics_route["specialist"] == "physics":
        assert "3DObjects" in physics_route["galaxy_names"]

    multi_route = router.route("rotate mesh then compute force and energy", specialist="auto")
    assert multi_route["specialist"] == "cartographer"
    assert "3DObjects" in multi_route["galaxy_names"]

