from __future__ import annotations

from knowledge3d.knowledgeverse.specialist_base import SpecialistBase


def test_specialist_base_spawn_and_route_prefers_domain_overlap():
    root = SpecialistBase(name="RootSpecialist", domain="navigator")
    math_child = root.spawn_child(name="MathSpecialist", domain="math")
    visual_child = root.spawn_child(name="VisualSpecialist", domain="visual")
    math_child.spawn_child(name="TopologySpecialist", domain="topology")
    math_child.spawn_child(name="NumberTheorySpecialist", domain="number_theory")

    routed_root = root.route("compute derivative and integral")
    assert routed_root.name == "MathSpecialist"

    routed_math = math_child.route("homology manifold topology proof")
    assert routed_math.name == "TopologySpecialist"
    assert visual_child.name in root.children


def test_specialist_base_serialization_roundtrip_preserves_tree():
    root = SpecialistBase(name="NavigatorSpecialist", domain="navigator")
    math_child = root.spawn_child(name="MathSpecialist", domain="math")
    math_child.spawn_child(name="PhDMathSpecialist", domain="phd_math")
    root.update_routing_bias("MathSpecialist", True)
    root.mark_query(success=True)

    payload = root.to_dict()
    restored = SpecialistBase.from_dict(payload)

    assert restored.name == "NavigatorSpecialist"
    assert restored.find("MathSpecialist") is not None
    assert restored.find("PhDMathSpecialist") is not None
    assert restored.routing_bias["MathSpecialist"] > 0.5
    assert restored.success_count == 1
