from knowledge3d.cranium.utils.cupy_env import ensure_nvrtc_include_path

ensure_nvrtc_include_path()

import cupy as cp
import numpy as np

from knowledge3d.cranium.bridges.arc import ArcReasoner
from knowledge3d.cranium.bridges.spill import SpillPlanner
from knowledge3d.cranium.bridges.router import GeometryRouter
from knowledge3d.cranium.bridges.fractal import FractalEmitter
from knowledge3d.cranium.bridges.resonance import (
    GalaxyResonanceEngine,
    ResonanceField,
    VectorResonator,
)
from knowledge3d.cranium.bridges.atomic_evolution import AtomicEvolution
from knowledge3d.cranium.bridges.temporal_reasoning import TemporalReasoner
from knowledge3d.cranium.bridges.graph_crystallizer import GraphCrystallizer
from knowledge3d.cranium.bridges.halting import MultimodalHaltingGate
from knowledge3d.cranium.bridges.cognitive_executive import CognitiveExecutive


def test_arc_reasoner_basic():
    reasoner = ArcReasoner()
    grid = cp.arange(9, dtype=cp.int32).reshape(3, 3)
    result = reasoner.infer(grid)
    assert result.shape == (3,)
    assert result.dtype == cp.int32


def test_spill_planner():
    planner = SpillPlanner()
    atoms, bytes_required = planner.plan(0, 16, available_bytes=128, request_count=20)
    assert atoms == 8
    assert bytes_required == 128


def test_geometry_router_shapes():
    router = GeometryRouter()
    tensor = cp.ones(16, dtype=cp.float32)
    out = router.route(tensor, shape_id=2)
    assert out.shape == tensor.shape
    assert float(out.mean()) != 0.0


def test_fractal_emitter_output():
    emitter = FractalEmitter()
    atoms = cp.linspace(0, 1, 5, dtype=cp.float32)
    coords = emitter.emit(atoms, scale=1.5)
    assert coords.shape == (5, 3)


def test_resonance_components():
    engine = GalaxyResonanceEngine()
    emb = cp.ones((2, 4), dtype=cp.float32)
    lat = cp.zeros_like(emb)
    blended = engine.run(emb, lat, alpha=0.7)
    assert blended.shape == emb.shape

    field = ResonanceField()
    pos = cp.zeros((4, 3), dtype=cp.float32)
    den = cp.ones(4, dtype=cp.float32)
    strengths = field.compute(pos, den)
    assert strengths.shape == (4,)

    resonator = VectorResonator()
    vec = cp.arange(4, dtype=cp.float32)
    blend = resonator.blend(vec, vec[::-1])
    assert blend.shape == vec.shape


def test_atomic_evolution_modes():
    evo = AtomicEvolution()
    atoms = cp.ones(8, dtype=cp.float32)
    fused = evo.apply(atoms, mode=0, ratio=0.5)
    fission = evo.apply(atoms, mode=1, ratio=0.5)
    assert float(fused.mean()) == 0.5
    assert float(fission.mean()) == 0.5


def test_temporal_reasoning():
    reasoner = TemporalReasoner()
    seq = cp.asarray([[0, 1], [1, 3], [4, 9]], dtype=cp.float32)
    deltas = reasoner.deltas(seq)
    np.testing.assert_allclose(cp.asnumpy(deltas[0]), np.array([1, 2], dtype=np.float32))


def test_graph_crystallizer():
    crystallizer = GraphCrystallizer()
    nodes = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    adjacency = np.asarray(
        [
            [1, -1],
            [2, -1],
            [-1, -1],
        ],
        dtype=np.int32,
    )
    counts = np.asarray([1, 1, 0], dtype=np.int32)
    out = crystallizer.crystallize_graph(nodes, adjacency, counts, rounds=2)
    assert out.shape == nodes.shape
    assert out[0, 1] > 0.13
    assert out[1, 1] > 0.53


def test_halting_gate():
    gate = MultimodalHaltingGate()
    logits = cp.asarray([0.1, 0.8, 0.3], dtype=cp.float32)
    mask = cp.asarray([1, 1, 0], dtype=cp.uint32)
    result = gate.evaluate(logits, mask, threshold=0.5)
    assert tuple(cp.asnumpy(result)) == (0, 1, 0)


def test_cognitive_executive_pipeline():
    executive = CognitiveExecutive()
    sensory = cp.linspace(0, 1, 8, dtype=cp.float32)
    weights = cp.ones_like(sensory)
    scores, coords = executive.process(sensory, media_shape_id=1, weights=weights, bias=0.1)
    assert scores.shape == sensory.shape
    assert coords.shape[1] == 3
