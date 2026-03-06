"""
Pytest Configuration for Knowledge3D Test Suite

Provides centralized test fixtures, environment setup, and configuration
for all Knowledge3D tests.

Developed by: GLM, enhanced by Claude
"""
import pytest
import random
import os
import sys
from unittest import mock
from importlib.util import find_spec

def _module_available(module_name: str) -> bool:
    return find_spec(module_name) is not None


_HAS_CUPY = _module_available("cupy")

# Avoid importing CuPy / touching CUDA at import-time, which can trigger GPU
# allocations, driver init, or segfaults in constrained environments.
# Enable explicitly when needed:
#   K3D_PYTEST_PROBE_CUDA=1 pytest -q
if os.environ.get("K3D_PYTEST_PROBE_CUDA") == "1" and _HAS_CUPY:
    try:
        import cupy as _cp

        try:
            _cp.cuda.Device(0).compute_capability
            _HAS_CUDA = True
        except Exception:
            _HAS_CUDA = False
    except Exception:
        _HAS_CUDA = False
else:
    _HAS_CUDA = False

_HAS_CV2 = _module_available("cv2")
_HAS_SOUNDFILE = _module_available("soundfile")
_HAS_WIKIPEDIAAPI = _module_available("wikipediaapi")


def _cuda_context_available() -> bool:
    """
    Return True if a CUDA context can be created in this pytest process.

    Defaults to False unless the test harness explicitly enables probing via
    `K3D_PYTEST_PROBE_CUDA=1`, to avoid import-time driver initialization.
    """
    if not (_HAS_CUPY and _HAS_CUDA):
        return False
    try:
        import cupy as _cp

        _cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


def pytest_ignore_collect(collection_path, config):  # type: ignore[override]
    """
    Skip optional-dependency test modules when the dependency isn't available.

    This keeps `pytest -q` runnable in CPU-only / minimal environments while
    still allowing the full suite inside the canonical k3d GPU env.
    """
    path_str = str(collection_path)

    # GPU / CuPy-only tests.
    if not (_HAS_CUPY and _HAS_CUDA):
        cupy_required = (
            "knowledge3d/cranium/tests/test_gpu_kernels.py",
            "knowledge3d/cranium/tests/test_latency_guard.py",
            "knowledge3d/cranium/tests/test_trm_core.py",
            "knowledge3d/cranium/tests/test_trm_engine.py",
            "tests/test_adaptive_convergence.py",
            "tests/test_alpha_optimizers.py",
            "tests/test_bridge_threshold.py",
            "tests/test_led_warp_regression.py",
            "tests/test_multi_modal_confidence.py",
            "tests/test_phase3_complete.py",
            "tests/test_phase3_domain_splitting.py",
            "tests/test_ptx_modality_ops.py",
            "tests/test_sleep_time_compute_compat.py",
        )
        if any(p in path_str for p in cupy_required):
            return True

    # Optional deps: audio / vision / wikipedia ingestion.
    if not _HAS_SOUNDFILE and "tests/test_step15_audio_sovereign.py" in path_str:
        return True
    if not _HAS_CV2 and "tests/test_step15_visual_sovereign.py" in path_str:
        return True
    if not _HAS_WIKIPEDIAAPI and "tests/test_step15_wikipedia_benchmark.py" in path_str:
        return True

    return False

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.utils import get_thinking_tag_bridge, ensure_step12_surface


@pytest.fixture(autouse=True)
def set_test_environment():
    """
    Ensure consistent test environment across all test files.

    Sets:
    - Deterministic random seed
    - Test mode environment variables
    - PTX strict mode (disabled for CPU-only tests)
    """
    # Set deterministic random seed
    random.seed(42)

    # Set environment variables for testing
    os.environ['K3D_TEST_MODE'] = '1'
    os.environ['K3D_PTX_STRICT'] = '0'  # Disable for CPU-only tests
    if not (_HAS_CUPY and _HAS_CUDA):
        os.environ['K3D_REQUIRE_PTX_QUERY'] = 'false'

    yield

    # Cleanup after tests
    if 'K3D_TEST_MODE' in os.environ:
        del os.environ['K3D_TEST_MODE']
    if 'K3D_PTX_STRICT' in os.environ:
        del os.environ['K3D_PTX_STRICT']
    if 'K3D_REQUIRE_PTX_QUERY' in os.environ:
        del os.environ['K3D_REQUIRE_PTX_QUERY']


@pytest.fixture
def step12_bridge_factory():
    """
    Factory that creates a ThinkingTagBridge instance with the Step 12 test surface applied.
    """
    ThinkingTagBridge = get_thinking_tag_bridge()

    def _factory():
        try:
            instance = ThinkingTagBridge()
        except RuntimeError:
            instance = mock.Mock()
        ensure_step12_surface(instance)
        return instance

    return _factory


@pytest.fixture(autouse=True)
def _inject_step12_factory(request, step12_bridge_factory):
    """
    Automatically expose `make_step12_bridge` on test classes so unittest-style
    suites can request Step 12 ready bridges without duplicating boilerplate.
    """
    cls = getattr(request, "cls", None)
    if cls is not None and not hasattr(cls, "make_step12_bridge"):
        cls.make_step12_bridge = staticmethod(step12_bridge_factory)


@pytest.fixture
def bridge(step12_bridge_factory):
    """
    Provide a test-ready ThinkingTagBridge instance.

    Returns mocked bridge with GPU operations stubbed for CPU-only testing.
    """
    bridge_instance = step12_bridge_factory()

    # Mock GPU operations for CPU-only testing
    if not hasattr(bridge_instance, 'inference') or callable(bridge_instance.inference):
        bridge_instance.inference = mock.Mock(return_value=mock.Mock(
            action_buffer=mock.Mock(
                confidence=0.85,
                action_type=1,
                curiosity=0.6,
                modal_signature=0b00011
            )
        ))

    # Mock other GPU-dependent methods
    if not hasattr(bridge_instance, 'get_state_trace_report'):
        bridge_instance.get_state_trace_report = mock.Mock(return_value={
            'stages': [
                {'name': 'INGEST', 'duration_us': 5},
                {'name': 'FUSE', 'duration_us': 10},
                {'name': 'SPATIAL', 'duration_us': 15},
                {'name': 'REASON', 'duration_us': 8},
                {'name': 'OUTPUT', 'duration_us': 7}
            ],
            'transitions': [
                {'from': 'INGEST', 'to': 'FUSE'},
                {'from': 'FUSE', 'to': 'SPATIAL'},
                {'from': 'SPATIAL', 'to': 'REASON'},
                {'from': 'REASON', 'to': 'OUTPUT'}
            ],
            'statistics': {
                'p50': 45.0,
                'p95': 55.0,
                'p99': 60.0
            }
        })

    if not hasattr(bridge_instance, 'export_state_trace'):
        bridge_instance.export_state_trace = mock.Mock()

    if not hasattr(bridge_instance, 'clear_state_trace'):
        bridge_instance.clear_state_trace = mock.Mock()

    # Mock LOD functionality
    if not hasattr(bridge_instance, 'lod_enabled'):
        bridge_instance.lod_enabled = True

    if not hasattr(bridge_instance, 'dynamic_lod_kernel'):
        bridge_instance.dynamic_lod_kernel = mock.Mock()

    return bridge_instance


@pytest.fixture
def mock_embedding():
    """Provide a mock input embedding for testing."""
    return random.randbytes(512)


@pytest.fixture
def sample_prompts():
    """Provide sample text prompts of varying complexity."""
    return {
        'simple': 'red cube',
        'moderate': 'blue sphere with metallic texture',
        'complex': 'wooden table with intricate carved legs and glass top',
        'very_complex': 'fantasy castle with multiple towers, bridges, and surrounding landscape'
    }


@pytest.fixture
def sample_modalities():
    """Provide sample modality combinations."""
    return [
        ['text'],
        ['image'],
        ['text', 'image'],
        ['text', 'image', 'audio'],
        ['text', 'image', 'audio', 'video', '3d']
    ]


@pytest.fixture
def temp_test_dir(tmp_path):
    """
    Provide a temporary directory for test file operations.

    Uses pytest's tmp_path fixture.
    """
    test_dir = tmp_path / "k3d_tests"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture
def mock_shape():
    """Provide a mock shape object for composition tests."""
    import numpy as np

    shape = mock.Mock()
    shape.vertices = np.random.randn(8, 3).astype(np.float32)
    shape.indices = np.array([[0, 1, 2], [2, 3, 0], [4, 5, 6], [6, 7, 4]], dtype=np.int32)
    shape.normals = np.random.randn(8, 3).astype(np.float32)
    shape.aabb = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]  # Bounding box
    shape.vertex_count = 8
    shape.face_count = 4
    shape.primitive_type = 'cube'
    shape.confidence = 0.9

    return shape


# Pytest configuration options
def pytest_configure(config):
    """
    Pytest configuration hook.

    Registers custom markers and sets up test environment.
    """
    # Register custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (skip in CPU-only CI)"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks performance benchmark tests"
    )
    config.addinivalue_line(
        "markers", "stress: marks stress/load tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to handle markers and skip conditions.

    Automatically skips GPU tests in CPU-only environments.
    """
    # Check if GPU/context is available
    context_available = _cuda_context_available()
    gpu_present = os.path.exists('/dev/nvidia0') or os.environ.get('CUDA_VISIBLE_DEVICES')
    gpu_available = bool(gpu_present and context_available)

    skip_gpu = pytest.mark.skip(reason="GPU not available")
    skip_cuda_ctx = pytest.mark.skip(reason="CUDA context unavailable")

    for item in items:
        # Skip GPU tests if no GPU
        if "gpu" in item.keywords and not gpu_available:
            item.add_marker(skip_gpu)
        if "test_step11" in item.nodeid and not context_available:
            item.add_marker(skip_cuda_ctx)


# Custom assertions for Knowledge3D
class K3DAssertions:
    """Custom assertion helpers for Knowledge3D tests."""

    @staticmethod
    def assert_valid_embedding(embedding):
        """Assert embedding is valid format."""
        assert isinstance(embedding, (bytes, bytearray))
        assert len(embedding) > 0

    @staticmethod
    def assert_valid_action_buffer(buffer):
        """Assert ActionBuffer has required fields."""
        required_fields = ['confidence', 'action_type', 'curiosity', 'modal_signature']
        for field in required_fields:
            assert hasattr(buffer, field), f"ActionBuffer missing field: {field}"

        # Value ranges
        assert 0 <= buffer.confidence <= 1.0, f"Invalid confidence: {buffer.confidence}"
        assert 0 <= buffer.action_type <= 255, f"Invalid action_type: {buffer.action_type}"
        assert 0 <= buffer.curiosity <= 1.0, f"Invalid curiosity: {buffer.curiosity}"

    @staticmethod
    def assert_valid_state_trace(report):
        """Assert state trace report is valid."""
        assert 'stages' in report
        assert 'transitions' in report
        assert 'statistics' in report

        expected_stages = ['INGEST', 'FUSE', 'SPATIAL', 'REASON', 'OUTPUT']
        actual_stages = [s['name'] for s in report['stages']]
        assert actual_stages == expected_stages, f"Invalid stage order: {actual_stages}"

    @staticmethod
    def assert_latency_budget(latency_us, budget_us=35):
        """Assert latency is within budget."""
        assert latency_us < budget_us, f"Latency {latency_us}µs exceeds budget {budget_us}µs"


@pytest.fixture
def k3d_assert():
    """Provide K3D-specific assertions."""
    return K3DAssertions()


# Pytest reporting hooks
def pytest_report_header(config):
    """Add custom header to pytest output."""
    return [
        "Knowledge3D Test Suite",
        "GPU-Sovereign Multi-Modal AI Testing Framework",
        f"Test Mode: {'CPU-Only (Mocked)' if os.environ.get('K3D_PTX_STRICT') == '0' else 'GPU-Enabled'}"
    ]


def pytest_runtest_makereport(item, call):
    """
    Hook to add custom reporting for test results.

    Captures latency metrics for benchmark tests.
    """
    if call.when == "call":
        # Check if this is a benchmark test
        if "benchmark" in item.keywords:
            # Could add custom metrics here
            pass
