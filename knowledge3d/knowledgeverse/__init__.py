"""Knowledgeverse runtime hardening utilities.

This package hosts MVP stabilization components for the Knowledgeverse
architecture, starting with sovereignty boundary enforcement.
"""

from .compressed_audit import BTreeIndex, CompressedAuditJournal
from .book_content_biology import BIOLOGY_ATLAS_ENTRIES
from .book_content_language import LANGUAGE_FOUNDATIONS_ENTRIES
from .book_content_mathematics import MATHEMATICS_PRIMER_ENTRIES
from .book_content_physics import PHYSICS_HANDBOOK_ENTRIES
from .book_content_tools import TOOL_MANUAL_ENTRIES
from .drawing_galaxy import DrawingGalaxy, DrawingItem
from .execution_grammar_detector import ExecutionGrammarDetector
from .foundational_galaxy_bootstrap import populate_always_on_foundational_galaxies, populate_book_galaxies
from .foundational_operations_bootstrap import bootstrap_default, populate_foundational_operations
from .galaxy_manager import GalaxyManager
from .grammar_galaxy import GrammarGalaxy, GrammarRule
from .house_builder import build_house
from .house_knowledge_tree import KNOWLEDGE_TREE_BRANCHES
from .house_books import HOUSE_BOOKS
from .house_doors import HOUSE_DOORS
from .house_furniture import HOUSE_FURNITURE
from .house_gallery_displays import GALLERY_DISPLAYS
from .house_memory_tablet import MEMORY_TABLET
from .house_nav_graph import HouseNavEdge, HouseNavGraph, HouseNavNode, build_house_nav_graph
from .house_observatory import OBSERVATORY_INSTRUMENTS
from .house_rooms import HOUSE_ROOMS
from .house_workshop_tools import WORKSHOP_TOOLS
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
from .trm_game_loop import TRMGameLoop
from .trm_navigator import TRMNavigator
from .trm_weight_store import TRMWeightStore

__all__ = [
    "BTreeIndex",
    "BIOLOGY_ATLAS_ENTRIES",
    "CircuitBreakerOpen",
    "CompressedAuditJournal",
    "DrawingGalaxy",
    "DrawingItem",
    "ExecutionGrammarDetector",
    "LANGUAGE_FOUNDATIONS_ENTRIES",
    "MATHEMATICS_PRIMER_ENTRIES",
    "PHYSICS_HANDBOOK_ENTRIES",
    "TOOL_MANUAL_ENTRIES",
    "populate_always_on_foundational_galaxies",
    "populate_book_galaxies",
    "bootstrap_default",
    "build_house",
    "GalaxyManager",
    "GrammarGalaxy",
    "GrammarRule",
    "HOUSE_BOOKS",
    "HOUSE_DOORS",
    "HOUSE_FURNITURE",
    "GALLERY_DISPLAYS",
    "HouseNavEdge",
    "HouseNavGraph",
    "HouseNavNode",
    "KNOWLEDGE_TREE_BRANCHES",
    "MEMORY_TABLET",
    "OBSERVATORY_INSTRUMENTS",
    "HOUSE_ROOMS",
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
    "TRMGameLoop",
    "QualityPrior",
    "TernaryQualityMemory",
    "TRMNavigator",
    "TRMWeightStore",
    "WORKSHOP_TOOLS",
    "build_house_nav_graph",
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
