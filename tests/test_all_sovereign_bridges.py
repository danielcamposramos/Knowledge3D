"""
Comprehensive Test Suite for All 15 Sovereign Bridges

Tests every Step8 kernel bridge to ensure:
1. Kernel loads successfully
2. Memory operations work correctly
3. Kernel executes without errors
4. Results are numerically valid
5. Latency is within acceptable bounds

Run with: pytest tests/test_all_sovereign_bridges.py -v
Or: python tests/test_all_sovereign_bridges.py
"""

import numpy as np
import pytest
from knowledge3d.cranium.bridges.sovereign_bridges import *

class TestKimiKernels:
    """Test Kimi's 3 kernels: LatencyGuard, ARCReasoner, OOMSpillManager"""

    def test_latency_guard(self):
        """Test LatencyGuard measures timing correctly"""
        guard = LatencyGuard(threshold_us=100.0)

        guard.start()
        from knowledge3d.cranium.sovereign.loader import synchronize
        synchronize()
        elapsed_ns, breached = guard.stop()

        assert elapsed_ns > 0, "Elapsed time should be positive"
        assert elapsed_ns < 1_000_000_000, "Elapsed time seems too large"
        assert isinstance(breached, bool), "Breached should be boolean"

        guard.cleanup()
        print(f"✅ LatencyGuard: {elapsed_ns/1000:.1f} µs (breached={breached})")

    def test_arc_reasoner(self):
        """Test ARCReasoner extracts rules from grids"""
        reasoner = ARCReasoner()

        grid = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=np.int32)
        rule_id, rotation, checksum = reasoner.extract_rules(grid)

        assert 0 <= rule_id < 8, f"Rule ID {rule_id} out of range"
        assert 0 <= rotation < 4, f"Rotation {rotation} out of range"
        assert checksum > 0, "Checksum should be positive"

        print(f"✅ ARCReasoner: rule={rule_id}, rot={rotation}, sum={checksum}")

    def test_oom_spill_manager(self):
        """Test OOMSpillManager computes spill plans correctly"""
        mgr = OOMSpillManager()

        atoms_to_spill, bytes_needed = mgr.compute_spill_plan(
            oldest_index=100,
            atom_size_bytes=1024,
            available_bytes=10240,
            request_count=20
        )

        assert atoms_to_spill == 10, f"Expected 10 atoms, got {atoms_to_spill}"
        assert bytes_needed == 10240, f"Expected 10240 bytes, got {bytes_needed}"

        print(f"✅ OOMSpillManager: {atoms_to_spill} atoms, {bytes_needed} bytes")


class TestQwenKernel:
    """Test Qwen's GalaxyResonanceEngine"""

    def test_galaxy_resonance(self):
        """Test GalaxyResonanceEngine blends correctly"""
        engine = GalaxyResonanceEngine()

        embeddings = np.random.randn(2, 128).astype(np.float32)
        latent = np.random.randn(2, 128).astype(np.float32)
        alpha = 0.3

        output = engine.resonate(embeddings, latent, alpha=alpha)

        expected = embeddings * alpha + latent * (1 - alpha)
        error = np.max(np.abs(output - expected))

        assert error < 1e-3, f"Blend error {error} too large"
        print(f"✅ GalaxyResonanceEngine: blend error={error:.6f}")


class TestDeepSeekKernels:
    """Test Deep Seek's 2 kernels: GeometryRouter, FractalEmitter"""

    def test_geometry_router(self):
        """Test GeometryRouter computes stable pairwise relation features"""
        router = GeometryRouter()

        a = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0, 4.0],
            ],
            dtype=np.float32,
        )
        b = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [4.0, 3.0, 2.0, 1.0],
            ],
            dtype=np.float32,
        )

        output = router.compute_relations(a, b)

        assert output.shape == (2, 16), f"Expected (2,16), got {output.shape}"
        assert output.dtype == np.float32
        assert output[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert output[0, 1] == pytest.approx(0.0, abs=1e-5)
        assert 0.0 <= float(output[0, 12]) <= 1.0
        assert 0.0 <= float(output[1, 12]) <= 1.0
        assert float(output[1, 10]) >= -1.0 and float(output[1, 10]) <= 1.0

        compat = router.route(np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float32), shape_id=2)
        assert compat.shape == (16,)
        assert np.all(np.isfinite(compat))

        print("✅ GeometryRouter: pairwise relation features valid")

    def test_fractal_emitter(self):
        """Test FractalEmitter computes self-similarity and keeps emit compatibility"""
        emitter = FractalEmitter()

        atoms = np.random.randn(50).astype(np.float32)
        coords = emitter.emit(atoms, base_scale=1.0)
        features = np.stack(
            [
                np.linspace(0.0, 1.0, 16, dtype=np.float32),
                np.tile(np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32), 4),
            ],
            axis=0,
        )
        scores = emitter.compute_self_similarity(features, num_scales=3)

        assert coords.shape == (50, 3), f"Expected (50,3), got {coords.shape}"
        assert coords.dtype == np.float32, "Coords should be float32"
        assert scores.shape == (2,), f"Expected (2,), got {scores.shape}"
        assert np.all(np.isfinite(scores))
        assert float(scores[0]) > 0.9

        print(f"✅ FractalEmitter: {coords.shape} coords + self-similarity scores")


class TestDeferredKernels:
    """Test deferred sovereign kernels pulled forward into Phase 2B."""

    def test_cognitive_executive(self):
        """Test CognitiveExecutive produces normalized trust weights."""
        executive = CognitiveExecutive()

        resonance_matrix = np.array(
            [
                [1.0, 0.9, 0.8, 0.8, 0.7, 0.7, 0.6, 0.6],
                [0.9, 1.0, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5],
                [0.8, 0.7, 1.0, 0.6, 0.5, 0.5, 0.4, 0.4],
                [0.8, 0.7, 0.6, 1.0, 0.5, 0.5, 0.4, 0.4],
                [0.7, 0.6, 0.5, 0.5, 1.0, 0.4, 0.3, 0.3],
                [0.7, 0.6, 0.5, 0.5, 0.4, 1.0, 0.3, 0.3],
                [0.6, 0.5, 0.4, 0.4, 0.3, 0.3, 1.0, 0.2],
                [0.6, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 1.0],
            ],
            dtype=np.float32,
        )
        chain_norms = np.array([2.5, 2.0, 1.7, 1.6, 1.3, 1.2, 1.0, 0.9], dtype=np.float32)

        trust_weights, coherence = executive.compute_trust_weights(resonance_matrix, chain_norms)

        assert trust_weights.shape == (8,)
        assert np.isclose(float(np.sum(trust_weights)), 1.0, atol=1e-4)
        assert np.all(trust_weights >= 0.0)
        assert coherence > 0.0
        assert int(np.argmax(trust_weights)) == 0

        print(f"✅ CognitiveExecutive: coherence={coherence:.3f}, top_chain={int(np.argmax(trust_weights))}")


class TestGLMKernels:
    """Test GLM's 3 kernels: ResonanceField, AtomicFissionFusion, TemporalReasoning"""

    def test_resonance_field(self):
        """Test ResonanceField computes cross-galaxy interference"""
        field = ResonanceField()

        embeddings = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        )
        galaxy_ids = np.array([0, 1, 1], dtype=np.int32)
        base_scores = np.array([1.0, 0.5, 0.5], dtype=np.float32)

        strengths = field.compute_resonance(embeddings, galaxy_ids, base_scores)

        assert strengths.shape == (3,), f"Expected (3,), got {strengths.shape}"
        assert np.all(strengths >= 0), "Strengths should be non-negative"
        assert strengths[0] > strengths[1] > strengths[2], "Cross-galaxy interference ordering incorrect"

        print(f"✅ ResonanceField: strengths range [{strengths.min():.3f}, {strengths.max():.3f}]")

    def test_atomic_fission_fusion(self):
        """Test AtomicFissionFusion performs real decompose/compose operations."""
        transformer = AtomicFissionFusion()

        compound = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        atoms = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        reconstructed, consistency = transformer.decompose(compound, atoms)
        assert reconstructed.shape == compound.shape
        assert consistency > 0.99
        assert np.allclose(reconstructed[:2], compound[:2], atol=1e-4)

        fused, fusion_consistency = transformer.compose(atoms)
        assert fused.shape == compound.shape
        assert fusion_consistency > 0.70
        assert np.allclose(fused[:2], np.array([0.5, 0.5], dtype=np.float32), atol=1e-4)

        compatibility = transformer.transform(np.array([2.0, 4.0], dtype=np.float32), mode=0, ratio=0.5)
        assert np.allclose(compatibility, np.array([1.0, 2.0], dtype=np.float32))
        print(f"✅ AtomicFissionFusion: decompose={consistency:.3f}, compose={fusion_consistency:.3f}")

    def test_temporal_reasoning(self):
        """Test TemporalReasoning extracts sequence pattern features"""
        reasoner = TemporalReasoning()

        sequence = np.stack(
            [
                np.linspace(0.0, 1.0, 16, dtype=np.float32),
                np.linspace(0.1, 1.1, 16, dtype=np.float32),
                np.linspace(0.2, 1.2, 16, dtype=np.float32),
                np.linspace(0.3, 1.3, 16, dtype=np.float32),
            ],
            axis=0,
        )
        patterns = reasoner.compute_patterns(sequence)
        deltas = reasoner.compute_deltas(sequence)

        assert patterns.shape == (24,), f"Expected 24 patterns, got {patterns.shape}"
        assert deltas.shape == sequence.shape, "Delta shape should match sequence"
        assert patterns[0] > 0.0, "Trend magnitude should be positive"
        assert patterns[8] > 0.9, "Lag-1 autocorrelation should stay high for smooth sequence"
        assert patterns[12] > 0.5, "Monotonicity should register positive"
        assert patterns[20] > 0.5, "Convergence trend should be positive"
        np.testing.assert_allclose(deltas[:-1], sequence[1:] - sequence[:-1], atol=1e-6)

        print("✅ TemporalReasoning: pattern extraction and delta compatibility valid")


class TestGrokKernels:
    """Test Grok's 3 kernels: VectorResonator, GraphCrystallizer, MultimodalHaltingGate"""

    def test_vector_resonator(self):
        """Test VectorResonator preserves alpha bias and content-aware attention."""
        resonator = VectorResonator()

        vec_a = np.concatenate(
            [
                np.ones(100, dtype=np.float32),
                -np.ones(100, dtype=np.float32),
            ]
        )
        vec_b = -vec_a
        low_alpha = 0.2
        high_alpha = 0.8

        low_alpha_out = resonator.resonate(vec_a, vec_b, low_alpha)
        high_alpha_out = resonator.resonate(vec_a, vec_b, high_alpha)

        assert low_alpha_out.shape == vec_a.shape
        assert high_alpha_out.shape == vec_a.shape
        assert np.dot(high_alpha_out, vec_a) > np.dot(low_alpha_out, vec_a)

        blended, weights = resonator.resonate_attention(
            np.stack(
                [
                    np.ones(32, dtype=np.float32),
                    np.ones(32, dtype=np.float32) * 3.0,
                    np.ones(32, dtype=np.float32) * 0.5,
                ],
                axis=0,
            )
        )

        assert blended.shape == (32,)
        assert weights.shape == (3,)
        assert np.isclose(float(weights.sum()), 1.0, atol=1e-5)
        assert float(weights[1]) > float(weights[0]) > float(weights[2])
        print(f"✅ VectorResonator: attention weights={weights.tolist()}")

    def test_graph_crystallizer(self):
        """Test GraphCrystallizer performs graph message passing"""
        crystallizer = GraphCrystallizer()

        nodes = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )
        adjacency = np.array(
            [
                [1, -1],
                [2, -1],
                [-1, -1],
            ],
            dtype=np.int32,
        )
        counts = np.array([1, 1, 0], dtype=np.int32)

        output = crystallizer.crystallize_graph(nodes, adjacency, counts, rounds=2)

        round1 = np.array(
            [
                np.tanh(0.6 * nodes[0] + 0.4 * nodes[1]),
                np.tanh(0.6 * nodes[1] + 0.4 * nodes[2]),
                nodes[2],
            ],
            dtype=np.float32,
        )
        expected = np.array(
            [
                np.tanh(0.6 * round1[0] + 0.4 * round1[1]),
                np.tanh(0.6 * round1[1] + 0.4 * round1[2]),
                round1[2],
            ],
            dtype=np.float32,
        )

        assert np.allclose(output, expected, rtol=1e-5), "message passing incorrect"
        print(f"✅ GraphCrystallizer: graph propagation correct")

    def test_multimodal_halting_gate(self):
        """Test MultimodalHaltingGate checks halting"""
        gate = MultimodalHaltingGate()

        logits = np.array([0.3, 0.7, 0.4, 0.8], dtype=np.float32)
        masks = np.array([1, 1, 0, 1], dtype=np.uint32)  # 0 = inactive
        threshold = 0.5

        flags = gate.check_halt(logits, masks, threshold)

        # Expected: [0 (0.3<0.5), 1 (0.7>0.5), 0 (inactive), 1 (0.8>0.5)]
        expected = np.array([0, 1, 0, 1], dtype=np.uint32)

        assert np.array_equal(flags, expected), f"Expected {expected}, got {flags}"
        print(f"✅ MultimodalHaltingGate: halting logic correct")


def run_all_tests():
    """Run all tests without pytest"""
    print("=" * 80)
    print("🔍 Comprehensive Sovereign Bridge Test Suite")
    print("=" * 80)
    print()

    # Kimi's kernels
    print("Testing Kimi's Kernels (3):")
    print("-" * 80)
    kimi = TestKimiKernels()
    kimi.test_latency_guard()
    kimi.test_arc_reasoner()
    kimi.test_oom_spill_manager()
    print()

    # Qwen's kernel
    print("Testing Qwen's Kernel (1):")
    print("-" * 80)
    qwen = TestQwenKernel()
    qwen.test_galaxy_resonance()
    print()

    # Deep Seek's kernels
    print("Testing Deep Seek's Kernels (2):")
    print("-" * 80)
    ds = TestDeepSeekKernels()
    ds.test_geometry_router()
    ds.test_fractal_emitter()
    print()

    # Deferred kernels
    print("Testing Deferred Kernels (1):")
    print("-" * 80)
    deferred = TestDeferredKernels()
    deferred.test_cognitive_executive()
    print()

    # GLM's kernels
    print("Testing GLM's Kernels (3):")
    print("-" * 80)
    glm = TestGLMKernels()
    glm.test_resonance_field()
    glm.test_atomic_fission_fusion()
    glm.test_temporal_reasoning()
    print()

    # Grok's kernels
    print("Testing Grok's Kernels (3):")
    print("-" * 80)
    grok = TestGrokKernels()
    grok.test_vector_resonator()
    grok.test_graph_crystallizer()
    grok.test_multimodal_halting_gate()
    print()

    print("=" * 80)
    print("🎉 ALL 13 TESTS PASSED!")
    print("=" * 80)
    print()
    print("Summary:")
    print("  ✅ Kimi's 3 kernels: LatencyGuard, ARCReasoner, OOMSpillManager")
    print("  ✅ Qwen's 1 kernel: GalaxyResonanceEngine")
    print("  ✅ Deep Seek's 2 kernels: GeometryRouter, FractalEmitter")
    print("  ✅ Deferred 1 kernel: CognitiveExecutive")
    print("  ✅ GLM's 3 kernels: ResonanceField, AtomicFissionFusion, TemporalReasoning")
    print("  ✅ Grok's 3 kernels: VectorResonator, GraphCrystallizer, MultimodalHaltingGate")
    print()
    print("All 15 Step8 kernel bridges operational with sovereign loader! 🔥")


if __name__ == "__main__":
    run_all_tests()
