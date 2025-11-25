"""PTX runtime helpers migrated from legacy phase directories.

This package exposes PTX runtime helpers but avoids importing heavy
dependencies (NumPy/CuPy) at import time. Symbols are resolved lazily
on attribute access.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple


_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    "ModularRPNEngine": ("modular_rpn_engine", "ModularRPNEngine"),
    "MathCorePool": ("math_core_pool", "MathCorePool"),
    "get_global_math_core_pool": ("math_core_pool", "get_global_math_core_pool"),
    "RPNCalculator": ("rpn_calculator", "RPNCalculator"),
    "ThinkingTagEmbedder": ("thinking_tag_embedder", "ThinkingTagEmbedder"),
    "TextTo3DGenerator": ("text_to_3d_generator", "TextTo3DGenerator"),
    "SleepTimeCompute": ("sleep_time_compute", "SleepTimeCompute"),
    "GalaxyStateSerializer": ("galaxy_state_serializer", "GalaxyStateSerializer"),
    "GalaxyMemoryUpdater": ("galaxy_memory_updater", "GalaxyMemoryUpdater"),
    "NVRTCPTXLoader": ("nvrtc_ptx_loader", "NVRTCPTXLoader"),
    "TRMEngine": ("trm_engine", "TRMEngine"),
    "TRMConfig": ("trm_engine", "TRMConfig"),
    "build_trm_refine_program": ("trm_rpn_program", "build_trm_refine_program"),
    "expected_trm_opcode_sequence": ("trm_rpn_program", "expected_trm_opcode_sequence"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = __import__(f"{__name__}.{module_name}", fromlist=[attr_name])
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_ATTRS.keys()))


__all__ = list(_LAZY_ATTRS.keys())
