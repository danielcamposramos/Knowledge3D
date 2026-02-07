from __future__ import annotations

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.specialist_router import SpecialistRouter


def test_router_infers_math_domain_from_query():
    router = SpecialistRouter()
    route = router.route("Find derivative of x^2 + 4x at x=3", specialist="auto")
    assert route["domain"] == "math"
    assert route["specialist"] == "math"
    assert route["galaxy_names"] == ["Math", "Grammar"]


def test_router_prefers_domain_hint_over_query():
    router = SpecialistRouter()
    route = router.route(
        "This sentence has no obvious math tokens",
        specialist="auto",
        domain_hint="visual",
    )
    assert route["domain"] == "visual"
    assert route["specialist"] == "visual"
    assert route["galaxy_names"] == ["Drawing", "Grammar"]


def test_router_maps_multi_to_cartographer():
    router = SpecialistRouter()
    route = router.route(
        "Rotate this grid and compute area 3*4 after transform",
        specialist="auto",
    )
    assert route["domain"] == "multi"
    assert route["specialist"] == "cartographer"
    assert "Drawing" in route["galaxy_names"]
    assert "Math" in route["galaxy_names"]


def test_router_keeps_explicit_specialist_and_galaxies():
    router = SpecialistRouter()
    route = router.route(
        "Any query",
        specialist="physics",
        galaxy_names=["Reality", "Math"],
    )
    assert route["specialist"] == "physics"
    assert route["galaxy_names"] == ["Reality", "Math"]
    assert route["reason"] == "explicit_specialist"


def test_trm_navigator_auto_routing_uses_router_contract(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_auto_router")
    kv.galaxy_manager.add_entry("Math", {"domain": "math", "text": "derivative rule"})
    kv.galaxy_manager.add_entry("Grammar", {"domain": "math", "text": "rule chain"})

    rows = kv.trm_navigator.query(
        query="Find derivative of x^2",
        specialist="auto",
    )
    trace = kv.trm_navigator.get_reasoning_trace()
    assert rows
    assert any("route specialist=math" in item for item in trace)
