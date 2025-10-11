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

from .rpn_calculator import RPNCalculator
from .text_to_3d_generator import TextTo3DGenerator
from .sleep_time_compute import SleepTimeCompute
from .thinking_tag_embedder import ThinkingTagEmbedder
from .galaxy_state_serializer import GalaxyStateSerializer
from .galaxy_memory_updater import GalaxyMemoryUpdater
from .nvrtc_ptx_loader import NVRTCPTXLoader

# TRMEngine uses CuPy (no cuda.bindings dependency)
from .trm_engine import TRMEngine, TRMConfig

__all__ = [
    "ModularRPNEngine",
    "RPNCalculator",
    "TextTo3DGenerator",
    "SleepTimeCompute",
    "ThinkingTagEmbedder",
    "GalaxyStateSerializer",
    "GalaxyMemoryUpdater",
    "NVRTCPTXLoader",
    "TRMEngine",
    "TRMConfig",
]
