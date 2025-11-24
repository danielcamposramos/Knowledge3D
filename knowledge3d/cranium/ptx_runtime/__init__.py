"""PTX runtime helpers migrated from legacy phase directories."""

# Conditional imports to avoid cuda.bindings dependency issues
# ModularRPNEngine requires cuda.bindings which may not be available
# Other modules use CuPy which works in k3d-cranium environment

try:
    from .modular_rpn_engine import ModularRPNEngine
    _HAS_MODULAR_RPN = True
except (ImportError, RuntimeError):
    ModularRPNEngine = None  # type: ignore
    _HAS_MODULAR_RPN = False

from .math_core_pool import MathCorePool, get_global_math_core_pool
from .rpn_calculator import RPNCalculator

# Optional imports (may have external dependencies)
try:
    from .thinking_tag_embedder import ThinkingTagEmbedder
    _HAS_THINKING_EMBEDDER = True
except (ImportError, AttributeError, RuntimeError):
    ThinkingTagEmbedder = None  # type: ignore
    _HAS_THINKING_EMBEDDER = False

try:
    from .text_to_3d_generator import TextTo3DGenerator
    _HAS_TEXT_TO_3D = True
except (ImportError, RuntimeError):
    TextTo3DGenerator = None  # type: ignore
    _HAS_TEXT_TO_3D = False

try:
    from .sleep_time_compute import SleepTimeCompute
    _HAS_SLEEP_TIME = True
except (ImportError, RuntimeError):
    SleepTimeCompute = None  # type: ignore
    _HAS_SLEEP_TIME = False

try:
    from .galaxy_state_serializer import GalaxyStateSerializer
    _HAS_GALAXY_SERIALIZER = True
except (ImportError, RuntimeError):
    GalaxyStateSerializer = None  # type: ignore
    _HAS_GALAXY_SERIALIZER = False

try:
    from .galaxy_memory_updater import GalaxyMemoryUpdater
    _HAS_GALAXY_MEMORY = True
except (ImportError, RuntimeError):
    GalaxyMemoryUpdater = None  # type: ignore
    _HAS_GALAXY_MEMORY = False

try:
    from .nvrtc_ptx_loader import NVRTCPTXLoader
    _HAS_NVRTC_LOADER = True
except (ImportError, RuntimeError):
    NVRTCPTXLoader = None  # type: ignore
    _HAS_NVRTC_LOADER = False

# TRMEngine uses CuPy (no cuda.bindings dependency)
try:
    from .trm_engine import TRMEngine, TRMConfig
    _HAS_TRM_ENGINE = True
except (ImportError, RuntimeError):
    TRMEngine = None  # type: ignore
    TRMConfig = None  # type: ignore
    _HAS_TRM_ENGINE = False

from .trm_rpn_program import build_trm_refine_program, expected_trm_opcode_sequence

__all__ = [
    "ModularRPNEngine",
    "MathCorePool",
    "get_global_math_core_pool",
    "RPNCalculator",
    "TextTo3DGenerator",
    "SleepTimeCompute",
    "ThinkingTagEmbedder",
    "GalaxyStateSerializer",
    "GalaxyMemoryUpdater",
    "NVRTCPTXLoader",
    "TRMEngine",
    "TRMConfig",
    "build_trm_refine_program",
    "expected_trm_opcode_sequence",
]
