"""System-literacy task generation for K3D structure learning."""

from __future__ import annotations

import random
from typing import Any


_BASE_TASKS: tuple[dict[str, Any], ...] = (
    {
        "kind": "visual_route",
        "query": "Route a visual pattern transformation from cranium to the right galaxy.",
        "expected": {
            "specialist": "visual",
            "domain_hint": "visual",
            "must_include_galaxies": ["Drawing", "Grammar"],
            "write_target": "Shadow_Copy",
            "sleeptime_effect": "consolidate_visual_patterns",
        },
    },
    {
        "kind": "math_route",
        "query": "Route a symbolic algebra evaluation request.",
        "expected": {
            "specialist": "math",
            "domain_hint": "math",
            "must_include_galaxies": ["Math", "Grammar"],
            "write_target": "Shadow_Copy",
            "sleeptime_effect": "consolidate_math_patterns",
        },
    },
    {
        "kind": "physics_route",
        "query": "Route a kinematics simulation request.",
        "expected": {
            "specialist": "physics",
            "domain_hint": "physics",
            "must_include_galaxies": ["Reality"],
            "write_target": "Shadow_Copy",
            "sleeptime_effect": "consolidate_reality_patterns",
        },
    },
    {
        "kind": "architecture_semantics",
        "query": "Identify where short-term reasoning runs and where persistent memory lives.",
        "expected": {
            "specialist": "grammar",
            "domain_hint": "system",
            "must_include_galaxies": ["Grammar"],
            "write_target": "Shadow_Copy",
            "sleeptime_effect": "consolidate_system_literacy",
            "facts": {
                "cranium": "active_processing",
                "galaxy": "working_memory",
                "house": "persistent_memory",
            },
        },
    },
    {
        "kind": "governance",
        "query": "Choose read targets for routing intent and write target for benchmark outcomes.",
        "expected": {
            "specialist": "grammar",
            "domain_hint": "governance",
            "must_include_galaxies": ["Grammar", "Math", "Drawing"],
            "write_target": "Grammar",
            "sleeptime_effect": "consolidate_benchmark_memory",
        },
    },
)


def generate_system_literacy_tasks(count: int, seed: int = 1360) -> list[dict[str, Any]]:
    """Generate deterministic K3D structure tasks."""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for idx in range(max(0, int(count))):
        template = dict(_BASE_TASKS[idx % len(_BASE_TASKS)])
        expected = dict(template["expected"])
        # Slight deterministic variation to avoid verbatim memorization.
        if template["kind"] == "visual_route" and rng.random() < 0.5:
            template["query"] = "Route an ARC-style visual transform query through the right specialist."
        if template["kind"] == "math_route" and rng.random() < 0.5:
            template["query"] = "Route a calculus expression evaluation request."
        out.append(
            {
                "id": f"sys_{idx:04d}",
                "category": "system_literacy",
                "kind": template["kind"],
                "query": template["query"],
                "expected": expected,
            }
        )
    return out


def evaluate_system_literacy_task(result: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Evaluate route/structure response against expected structure hints."""
    if not isinstance(result, dict) or not isinstance(expected, dict):
        return False
    route = result.get("route", {}) if isinstance(result.get("route"), dict) else {}
    resolved_specialist = str(result.get("specialist", route.get("specialist", ""))).lower()
    if expected.get("specialist") and resolved_specialist != str(expected["specialist"]).lower():
        return False
    galaxies = result.get("galaxy_names", route.get("galaxy_names", []))
    if not isinstance(galaxies, list):
        galaxies = []
    galaxy_names = {str(name) for name in galaxies}
    for required in expected.get("must_include_galaxies", []):
        if str(required) not in galaxy_names:
            return False
    if expected.get("write_target") and str(result.get("write_target", "")) != str(expected.get("write_target")):
        return False
    if expected.get("sleeptime_effect") and str(result.get("sleeptime_effect", "")) != str(
        expected.get("sleeptime_effect")
    ):
        return False
    facts = expected.get("facts")
    if isinstance(facts, dict):
        observed = result.get("facts", {})
        if not isinstance(observed, dict):
            return False
        for key, value in facts.items():
            if str(observed.get(key, "")) != str(value):
                return False
    return True

