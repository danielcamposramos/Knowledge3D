"""
Advanced profiling for the text-to-3D pipeline.

Implements GLM's Step 13‑B benchmark specifications with graceful fallbacks for
optional dependencies such as matplotlib and memory_profiler.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import pytest

try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore

from tests.utils.μbench import μBench
from tests.utils.bridge_import import get_thinking_tag_bridge
from tests.benchmarks.test_text_to_3d_pipeline import _ensure_pipeline_bridge

ThinkingTagBridge = get_thinking_tag_bridge()


@pytest.fixture
def test_prompts(sample_prompts) -> List[str]:
    """Alias fixture to match the specification."""
    return [
        sample_prompts["simple"],
        sample_prompts["moderate"],
        sample_prompts["complex"],
        sample_prompts["very_complex"],
    ]


@pytest.fixture
def profiler_bridge():
    return _ensure_pipeline_bridge(ThinkingTagBridge())


@pytest.fixture
def profiler_timer() -> μBench:
    return μBench("advanced_text_to_3d")


class TestAdvancedTextTo3DProfiler:
    """GLM Phase‑3 profiling suite for the text-to-3D generator."""

    @pytest.fixture(autouse=True)
    def _setup(self, profiler_bridge, profiler_timer, test_prompts):
        self.bridge = profiler_bridge
        self.μ = profiler_timer
        self.test_prompts = test_prompts

    @pytest.mark.benchmark
    def test_detailed_pipeline_breakdown(self):
        """Detailed breakdown of each pipeline stage with optional visualization."""
        stages: Dict[str, Dict[str, Dict[str, float]]] = {}

        for prompt in self.test_prompts:
            prompt_stages: Dict[str, Dict[str, float]] = {}

            prompt_stages["parsing"] = self.μ(self.bridge._parse_prompt, prompt)

            parsed = self.bridge._parse_prompt(prompt)
            prompt_stages["synthesis"] = self.μ(self.bridge._synthesize_shape, parsed)

            synthesized = self.bridge._synthesize_shape(parsed)
            prompt_stages["geometry"] = self.μ(self.bridge._generate_geometry, synthesized)

            geometry = self.bridge._generate_geometry(synthesized)
            prompt_stages["materials"] = self.μ(self.bridge._apply_materials, geometry)

            materials = self.bridge._apply_materials(geometry)
            prompt_stages["assembly"] = self.μ(self.bridge._assemble_3d_object, materials)

            stages[prompt] = prompt_stages

            assert prompt_stages["parsing"]["p50"] < 5000, f"Prompt parsing too slow: {prompt}"
            assert prompt_stages["synthesis"]["p50"] < 20000, f"Shape synthesis too slow: {prompt}"
            assert prompt_stages["geometry"]["p50"] < 10000, f"Geometry generation too slow: {prompt}"
            assert prompt_stages["materials"]["p50"] < 5000, f"Material application too slow: {prompt}"
            assert prompt_stages["assembly"]["p50"] < 2000, f"Final assembly too slow: {prompt}"

        self._generate_pipeline_breakdown_chart(stages)
        return stages

    def _generate_pipeline_breakdown_chart(self, results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        """Create a visualization if matplotlib is available."""
        if plt is None:  # pragma: no cover - optional dependency
            return

        os.makedirs("reports", exist_ok=True)
        figure, axis = plt.subplots(figsize=(12, 8))

        prompts = list(results.keys())
        stages = ["parsing", "synthesis", "geometry", "materials", "assembly"]
        indices = range(len(prompts))
        width = 0.15

        for offset, stage in enumerate(stages):
            values = [results[prompt][stage]["p50"] for prompt in prompts]
            axis.bar(
                [idx + offset * width for idx in indices],
                values,
                width,
                label=stage,
            )

        axis.set_xlabel("Prompt Complexity")
        axis.set_ylabel("Latency (µs)")
        axis.set_title("Text-to-3D Pipeline Stage Breakdown")
        axis.set_xticks([idx + width * 2 for idx in indices])
        axis.set_xticklabels([f"Prompt {i + 1}" for i in indices])
        axis.legend()

        figure.tight_layout()
        figure.savefig("reports/pipeline_breakdown.png")
        plt.close(figure)

    def test_memory_usage_profiling(self):
        """Profile memory usage throughout the pipeline."""
        try:
            from memory_profiler import memory_usage  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            pytest.skip("memory_profiler not available")

        os.makedirs("reports", exist_ok=True)
        results: Dict[str, Dict[str, float]] = {}

        for prompt in self.test_prompts:
            mem_usage = memory_usage((self.bridge.generate_3d_from_text, (prompt,)))
            peak = max(mem_usage)
            average = sum(mem_usage) / len(mem_usage)
            growth = peak - min(mem_usage)

            results[prompt] = {
                "peak_mb": peak,
                "avg_mb": average,
                "growth_mb": growth,
            }

            assert peak < 500, f"Memory usage too high for: {prompt}"

        with open("reports/memory_usage_profile.json", "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

        return results
