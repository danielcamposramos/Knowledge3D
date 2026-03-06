"""Knowledgeverse runtime hardening utilities.

This package hosts MVP stabilization components for the Knowledgeverse
architecture, starting with sovereignty boundary enforcement.
"""

from .compressed_audit import BTreeIndex, CompressedAuditJournal
from .drawing_galaxy import DrawingGalaxy, DrawingItem
from .execution_grammar_detector import ExecutionGrammarDetector
from .foundational_galaxy_bootstrap import populate_always_on_foundational_galaxies
from .foundational_operations_bootstrap import bootstrap_default, populate_foundational_operations
from .galaxy_manager import GalaxyManager
from .grammar_galaxy import GrammarGalaxy, GrammarRule
from .knowledgeverse import Knowledgeverse, KnowledgeverseMetrics
from .navigator_specialist import NavigatorSpecialist, PathCandidate
from .resilience import CircuitBreakerOpen, SelfHealingWrapper
from .ring_buffer import RingBuffer
from .shadow_copy import ShadowCopyLearning
from .sleeptime import SleepTimeConsolidation, SleepTimeError
from .specialist_base import SpecialistBase, SpecialistDelta
from .specialist_spawner import SpecialistSpawner, SpawnDecision
from .specialist_router import SpecialistRoute, SpecialistRouter
from .sovereignty_firewall import SovereigntyFirewall
from .temporal_metadata import TemporalMetadata, TemporalMetadataManager
from .ternary_quality_memory import QualityPrior, TernaryQualityMemory
from .tool_galaxy import (
    ToolNode,
    bootstrap_tool_galaxy,
    build_multimodal_tool_payload,
    build_tool_payload,
    default_tool_entries,
)
from .trm_navigator import TRMNavigator
from .trm_weight_store import TRMWeightStore

__all__ = [
    "BTreeIndex",
    "CircuitBreakerOpen",
    "CompressedAuditJournal",
    "DrawingGalaxy",
    "DrawingItem",
    "ExecutionGrammarDetector",
    "populate_always_on_foundational_galaxies",
    "bootstrap_default",
    "GalaxyManager",
    "GrammarGalaxy",
    "GrammarRule",
    "Knowledgeverse",
    "KnowledgeverseMetrics",
    "NavigatorSpecialist",
    "PathCandidate",
    "populate_foundational_operations",
    "RingBuffer",
    "SelfHealingWrapper",
    "SpecialistRoute",
    "SpecialistRouter",
    "ShadowCopyLearning",
    "SleepTimeConsolidation",
    "SleepTimeError",
    "SpecialistBase",
    "SpecialistDelta",
    "SpecialistSpawner",
    "SpawnDecision",
    "SovereigntyFirewall",
    "TemporalMetadata",
    "TemporalMetadataManager",
    "ToolNode",
    "QualityPrior",
    "TernaryQualityMemory",
    "TRMNavigator",
    "TRMWeightStore",
    "bootstrap_tool_galaxy",
    "build_tool_payload",
    "build_multimodal_tool_payload",
    "default_tool_entries",
]
