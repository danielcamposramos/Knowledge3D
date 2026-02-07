from __future__ import annotations

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.navigator_specialist import NavigatorSpecialist


def test_navigator_specialist_plans_multiple_strategies():
    nav = NavigatorSpecialist()
    routes = nav.plan_routes(
        "Rotate grid and compute derivative of x^2",
        specialist="auto",
    )
    assert routes
    assert any(route["specialist"] == "cartographer" for route in routes)
    assert any(route["strategy"] == "legacy_keywords_2025" for route in routes)


def test_navigator_specialist_learns_topology_bias():
    nav = NavigatorSpecialist(max_paths=3)
    query = "Find derivative of x^2 + 4x at x=3"

    for _ in range(3):
        nav.learn_routing_topology(query, specialist="math", success=True)
    nav.learn_routing_topology(query, specialist="visual", success=False)

    routes = nav.plan_routes(query, specialist="auto")
    assert routes
    assert routes[0]["specialist"] == "math"


def test_trm_navigator_auto_uses_meta_specialist(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_nav_meta")
    kv.galaxy_manager.add_entry("Math", {"domain": "math", "text": "derivative power rule"})
    kv.galaxy_manager.add_entry("Grammar", {"domain": "math", "text": "compose steps"})

    result = kv.trm_navigator.navigate_and_compose(
        query="Find derivative of x^2 + 4x at x=3",
        specialist="auto",
        domain_hint="math",
    )
    assert "meta_specialist" in result
    assert result["meta_specialist"]["paths_considered"] >= 1
    assert result["route"]["specialist"] in {"math", "cartographer"}
