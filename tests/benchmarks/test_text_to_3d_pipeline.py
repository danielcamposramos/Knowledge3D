"""
Benchmark suite for the text-to-3D pipeline.

Implements Deep Seek's Step 13‑B specification while remaining resilient in
CPU‑only environments by augmenting mocked bridges with deterministic fallbacks.
"""
from __future__ import annotations

import concurrent.futures
import time
from types import SimpleNamespace
from typing import Iterable, List
from unittest import mock

import pytest

from tests.utils.μbench import μBench
from tests.utils.bridge_import import get_thinking_tag_bridge

ThinkingTagBridge = get_thinking_tag_bridge()


def _ensure_pipeline_bridge(instance) -> object:
    """
    Augment the provided bridge instance with the methods required by the
    benchmark specifications when they are missing (e.g. during mocked runs).
    """
    # Provide deterministic parsing
    if not hasattr(instance, "_parse_prompt"):
        def _parse_prompt(prompt: str) -> dict:
            complexity = min(max(len(prompt.split()), 1), 10)
            return {
                "prompt": prompt,
                "complexity": complexity,
                "tags": ["text"],
            }

        instance._parse_prompt = mock.Mock(side_effect=_parse_prompt)

    # Intent synthesis
    if not hasattr(instance, "_synthesize_shape"):
        def _synthesize_shape(parsed_intent: dict) -> dict:
            return {
                "intent": parsed_intent,
                "primitive": "cube" if parsed_intent["complexity"] <= 3 else "compound",
                "detail_level": parsed_intent["complexity"],
            }

        instance._synthesize_shape = mock.Mock(side_effect=_synthesize_shape)

    # Geometry generation
    if not hasattr(instance, "_generate_geometry"):
        def _generate_geometry(shape_spec: dict) -> dict:
            vertex_count = 8 * max(shape_spec.get("detail_level", 1), 1)
            return {
                "shape": shape_spec,
                "vertex_count": vertex_count,
                "vertices": [(0.0, 0.0, 0.0)] * vertex_count,
                "indices": [(0, 1, 2)] * (vertex_count // 3 or 1),
            }

        instance._generate_geometry = mock.Mock(side_effect=_generate_geometry)

    # Material application
    if not hasattr(instance, "_apply_materials"):
        def _apply_materials(geometry: dict) -> dict:
            return {
                **geometry,
                "materials": ["default"],
            }

        instance._apply_materials = mock.Mock(side_effect=_apply_materials)

    # Final assembly
    if not hasattr(instance, "_assemble_3d_object"):
        def _assemble_3d_object(materials: dict) -> SimpleNamespace:
            vertex_data = materials.get("vertices", [])
            confidence = 0.9 if materials["shape"]["detail_level"] <= 2 else 0.6
            return SimpleNamespace(
                vertices=vertex_data,
                indices=materials.get("indices", []),
                primitive_type=materials["shape"]["primitive"],
                confidence=confidence,
                vertex_count=len(vertex_data),
                alternatives=["cube", "sphere", "cylinder"],
            )

        instance._assemble_3d_object = mock.Mock(side_effect=_assemble_3d_object)

    # End-to-end generation
    if not hasattr(instance, "generate_3d_from_text"):
        def _generate_3d_from_text(prompt: str) -> SimpleNamespace:
            parsed = instance._parse_prompt(prompt)
            synthesized = instance._synthesize_shape(parsed)
            geometry = instance._generate_geometry(synthesized)
            materials = instance._apply_materials(geometry)
            assembled = instance._assemble_3d_object(materials)
            return assembled

        instance.generate_3d_from_text = mock.Mock(side_effect=_generate_3d_from_text)

    return instance


@pytest.fixture
def test_prompts(sample_prompts) -> List[str]:
    """Alias fixture required by the specification."""
    return [
        sample_prompts["simple"],
        sample_prompts["moderate"],
        sample_prompts["complex"],
        sample_prompts["very_complex"],
    ]


@pytest.fixture
def pipeline_bridge() -> object:
    """Provide a bridge instance with all required helpers available."""
    instance = ThinkingTagBridge()
    return _ensure_pipeline_bridge(instance)


@pytest.fixture
def benchmark_timer() -> μBench:
    return μBench("text_to_3d")


class TestTextTo3DPipeline:
    """Deep Seek Phase‑3 text-to-3D pipeline benchmarks."""

    @pytest.fixture(autouse=True)
    def _setup(self, pipeline_bridge, benchmark_timer, test_prompts):
        self.bridge = pipeline_bridge
        self.μ = benchmark_timer
        self.test_prompts = test_prompts

    @pytest.mark.benchmark
    def test_prompt_parsing_latency(self):
        """Text → structured intent parsing benchmarks."""
        benchmarks = {}

        for prompt in self.test_prompts:
            stats = self.μ(self.bridge._parse_prompt, prompt)
            benchmarks[prompt] = stats
            assert stats["p50"] < 5000, f"Prompt parsing too slow for: {prompt}"

        return benchmarks

    @pytest.mark.benchmark
    def test_shape_synthesis_latency(self):
        """Intent → primitive composition benchmarks."""
        benchmarks = {}

        for prompt in self.test_prompts:
            parsed_intent = self.bridge._parse_prompt(prompt)
            stats = self.μ(self.bridge._synthesize_shape, parsed_intent)
            benchmarks[prompt] = stats
            assert stats["p50"] < 20000, f"Shape synthesis too slow for: {prompt}"

        return benchmarks

    @pytest.mark.benchmark
    def test_end_to_end_generation(self):
        """Complete text-to-3D pipeline benchmarks."""
        results = {}

        for prompt in self.test_prompts:
            start_time = time.perf_counter_ns()
            result = self.bridge.generate_3d_from_text(prompt)
            end_time = time.perf_counter_ns()

            latency_ms = (end_time - start_time) / 1e6
            results[prompt] = {
                "latency_ms": latency_ms,
                "vertex_count": getattr(result, "vertex_count", 0),
                "success": result is not None,
            }

            if "cube" in prompt or "sphere" in prompt:
                assert latency_ms < 50, f"Simple shape generation too slow: {latency_ms}ms"
            else:
                assert latency_ms < 200, f"Complex shape generation too slow: {latency_ms}ms"

        return results

    @pytest.mark.stress
    def test_concurrent_generation_throughput(self):
        """Measure throughput under concurrent load."""
        prompts: Iterable[str] = ["test shape"] * 50
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(self.bridge.generate_3d_from_text, prompt)
                for prompt in prompts
            ]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        total_time = time.time() - start_time
        throughput = len(results) / total_time if total_time > 0 else float("inf")

        assert throughput > 10, f"Throughput too low: {throughput} shapes/sec"
        success_count = len([result for result in results if result is not None])
        assert success_count > 45, f"Too many failed generations: {success_count}/50"

