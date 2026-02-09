from __future__ import annotations

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.navigator_specialist import NavigatorSpecialist, PathCandidate


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


def test_navigator_forward_backward_route_variants():
    nav = NavigatorSpecialist(max_paths=6)
    routes = nav.plan_routes(
        "What is 2z, given x=5 and y=10. Let z = x + y.",
        specialist="math",
        use_forward_backward=True,
    )
    strategies = {route["strategy"] for route in routes}
    assert "forward" in strategies
    assert "backward" in strategies
    assert "fusion" in strategies
    assert "auto" in strategies
    fwd = next(route for route in routes if route["strategy"] == "forward")
    bwd = next(route for route in routes if route["strategy"] == "backward")
    fus = next(route for route in routes if route["strategy"] == "fusion")
    assert "context:" in fwd["query_variant"]
    assert "goal:" in bwd["query_variant"]
    assert "Given" in fus["query_variant"] or "given" in fus["query_variant"]


def test_navigator_cross_path_agreement_prefers_consensus():
    nav = NavigatorSpecialist(max_paths=3)
    path_consensus_a = PathCandidate(
        specialist="math",
        domain="math",
        route={"strategy": "forward"},
        patterns=[{"id": "a"}] * 5,
        composed={"program_type": "math_expression", "expression": "x+y"},
        confidence=0.60,
    )
    path_consensus_b = PathCandidate(
        specialist="math",
        domain="math",
        route={"strategy": "backward"},
        patterns=[{"id": "b"}] * 4,
        composed={"program_type": "math_expression", "expression": "x+y"},
        confidence=0.59,
    )
    path_solo = PathCandidate(
        specialist="math",
        domain="math",
        route={"strategy": "heuristic_auto"},
        patterns=[{"id": "c"}] * 8,
        composed={"program_type": "math_expression", "expression": "x*y"},
        confidence=0.68,
    )

    composed = nav.compose_paths("Calculate x+y", [path_consensus_a, path_consensus_b, path_solo])
    assert composed["meta_specialist"]["cross_path_agreement"] == 2
    assert composed["expression"] == "x+y"


def test_navigator_grammar_composition_boost_prefers_richer_path():
    nav = NavigatorSpecialist(max_paths=3)
    path_rich = PathCandidate(
        specialist="visual",
        domain="visual",
        route={"strategy": "forward"},
        patterns=[
            {
                "entry": {
                    "id": "rich_path",
                    "rpn_program": "A B ADD C MUL D SUB",
                    "metadata": {"confidence": 0.9, "symlink": "math_galaxy"},
                }
            }
        ],
        composed={"program_type": "arc_transform", "transform": {"op": "composed", "steps": [{"op": "rot90"}, {"op": "color_map"}]}},
        confidence=0.55,
    )
    path_plain = PathCandidate(
        specialist="visual",
        domain="visual",
        route={"strategy": "heuristic_auto"},
        patterns=[{"entry": {"id": "plain_path", "rpn_program": "NOP", "metadata": {"confidence": 0.5}}}],
        composed={"program_type": "arc_transform", "transform": {"op": "identity"}},
        confidence=0.62,
    )

    # Keep deterministic and isolated from galaxy-query side effects.
    nav._query_grammar_galaxy_confidence = lambda entry: float(entry.get("metadata", {}).get("confidence", 0.5))
    nav._query_cross_modal_confidence = lambda entry, galaxy: 0.9 if entry.get("id") == "rich_path" else 0.5

    composed = nav.compose_paths("visual transform", [path_rich, path_plain])
    assert composed["route"]["strategy"] == "forward"
    assert composed["meta_specialist"]["grammar_boosted"] >= 1


def test_fusion_reading_path_deduplicates_variables():
    nav = NavigatorSpecialist(max_paths=6)
    base = nav.router.route(
        query="Let x=5 and y=10. Calculate 2z, where z=x+y.",
        specialist="math",
    )
    fusion = nav._fusion_reading_path(
        "Let x=5 and y=10. Calculate 2z, where z=x+y.",
        base,
    )
    assert "fusion_parse" in fusion
    fparse = fusion["fusion_parse"]
    assert isinstance(fparse.get("merged_variables"), dict)
    assert "x" in fparse["merged_variables"]
    assert "y" in fparse["merged_variables"]
    assert fparse.get("deduplication_savings", 0) >= 0


def test_trm_navigator_auto_includes_fusion_strategy(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_nav_fusion")
    kv.galaxy_manager.add_entry("Math", {"id": "m_a", "domain": "math", "text": "a=3 b=4 c=5"})
    kv.galaxy_manager.add_entry("Grammar", {"id": "g_a", "domain": "math", "text": "what is a^2+b^2+c^2"})
    result = kv.trm_navigator.navigate_and_compose(
        query="Given a=3, b=4, c=5. What is a^2 + b^2 + c^2?",
        specialist="auto",
        domain_hint="math",
        use_forward_backward=True,
    )
    meta = result.get("meta_specialist", {})
    strategies = set(meta.get("strategies", []))
    assert meta.get("paths_considered", 0) >= 4
    assert {"forward", "backward", "fusion", "auto"}.issubset(strategies)


def test_navigator_logs_path_contribution_events(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_nav_events")
    kv.galaxy_manager.add_entry("Math", {"id": "m_rule", "domain": "math", "rpn_program": "X Y ADD", "metadata": {"confidence": 0.8}})
    kv.galaxy_manager.add_entry("Grammar", {"id": "g_rule", "domain": "math", "rpn_program": "goal parse", "metadata": {"confidence": 0.9}})
    baseline = len(kv.shadow_copy.event_buffer)

    _ = kv.trm_navigator.navigate_and_compose(
        query="What is x+y, given x=5 and y=10?",
        specialist="auto",
        domain_hint="math",
        use_enriched=True,
        use_forward_backward=True,
    )
    events = kv.shadow_copy.event_buffer[baseline:]
    event_types = {event["type"] for event in events}
    assert "navigator_compose" in event_types
    assert "navigator_path_contribution" in event_types
