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
from .meaning_star import MeaningCentricStar, SurfaceForm, compute_star_id, wrap_galaxy_entry_with_meaning_star
from .navigator_specialist import NavigatorSpecialist, PathCandidate
from .resilience import CircuitBreakerOpen, SelfHealingWrapper
from .ring_buffer import RingBuffer
from .seed_stars import SEED_STARS, build_seed_stars, seed_star_entries, seed_word_entries
from .semantic_gravity import gravity_tick, meaning_mass, semantic_gravity_force, ternary_semantic_force
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
    "MeaningCentricStar",
    "NavigatorSpecialist",
    "PathCandidate",
    "populate_foundational_operations",
    "RingBuffer",
    "SEED_STARS",
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
    "SurfaceForm",
    "SovereigntyFirewall",
    "TemporalMetadata",
    "TemporalMetadataManager",
    "ToolNode",
    "QualityPrior",
    "TernaryQualityMemory",
    "TRMNavigator",
    "TRMWeightStore",
    "bootstrap_tool_galaxy",
    "build_seed_stars",
    "compute_star_id",
    "build_tool_payload",
    "build_multimodal_tool_payload",
    "default_tool_entries",
    "gravity_tick",
    "meaning_mass",
    "seed_star_entries",
    "seed_word_entries",
    "semantic_gravity_force",
    "ternary_semantic_force",
    "wrap_galaxy_entry_with_meaning_star",
]
