"""Corpus manifest for knowledge preparation and staged ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class CorpusTier(Enum):
    """Tiered priority for ingestion ordering."""

    TIER_1_FOUNDATIONAL = 1
    TIER_2_DOMAIN = 2
    TIER_3_INTEGRATION = 3


class CorpusType(Enum):
    """Supported source formats for ingestion entries."""

    PDF = "pdf"
    MARKDOWN = "markdown"
    JSON = "json"
    CODE = "code"
    DATASET = "dataset"


@dataclass
class CorpusEntry:
    """Single source entry tracked by the corpus manifest."""

    id: str
    name: str
    path: str
    tier: CorpusTier
    corpus_type: CorpusType
    target_galaxies: list[str]
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    ingested: bool = False
    embedding_count: int | None = None


class CorpusManifest:
    """Registry and dependency resolver for ingestion corpus entries."""

    def __init__(self, load_defaults: bool = True):
        self.entries: dict[str, CorpusEntry] = {}
        if load_defaults:
            self._load_default_corpus()

    def _load_default_corpus(self) -> None:
        """Populate default Tier 1/2/3 entries from preparation spec."""
        self.add_entry(
            CorpusEntry(
                id="algorithmic_thinking",
                name="Algorithmic Thinking",
                path="docs/pdfs/algorithmic_thinking.pdf",
                tier=CorpusTier.TIER_1_FOUNDATIONAL,
                corpus_type=CorpusType.PDF,
                target_galaxies=["Grammar", "Math", "Reality"],
                metadata={
                    "priority": 1,
                    "description": "Foundational problem-solving strategies",
                    "domains": ["algorithms", "logic", "reasoning"],
                },
            )
        )
        self.add_entry(
            CorpusEntry(
                id="math_foundations",
                name="Mathematics Foundations",
                path="docs/pdfs/math_foundations/",
                tier=CorpusTier.TIER_1_FOUNDATIONAL,
                corpus_type=CorpusType.PDF,
                target_galaxies=["Math", "Grammar"],
                dependencies=["algorithmic_thinking"],
                metadata={
                    "priority": 2,
                    "description": "Algebra, calculus, geometry, number theory",
                    "domains": ["mathematics"],
                },
            )
        )
        self.add_entry(
            CorpusEntry(
                id="logic_reasoning",
                name="Logic & Reasoning",
                path="docs/pdfs/logic_reasoning/",
                tier=CorpusTier.TIER_1_FOUNDATIONAL,
                corpus_type=CorpusType.PDF,
                target_galaxies=["Grammar", "Math"],
                dependencies=["algorithmic_thinking"],
                metadata={
                    "priority": 3,
                    "description": "Propositional and predicate logic",
                    "domains": ["logic", "mathematics"],
                },
            )
        )
        self.add_entry(
            CorpusEntry(
                id="cs_fundamentals",
                name="Computer Science Fundamentals",
                path="docs/pdfs/cs_fundamentals/",
                tier=CorpusTier.TIER_1_FOUNDATIONAL,
                corpus_type=CorpusType.PDF,
                target_galaxies=["Grammar", "Reality"],
                dependencies=["algorithmic_thinking"],
                metadata={
                    "priority": 4,
                    "description": "Data structures, algorithms, complexity",
                    "domains": ["computer_science", "algorithms"],
                },
            )
        )
        self.add_entry(
            CorpusEntry(
                id="competition_math",
                name="Competition Math Problems",
                path="docs/datasets/competition_math/",
                tier=CorpusTier.TIER_2_DOMAIN,
                corpus_type=CorpusType.DATASET,
                target_galaxies=["Math", "Grammar"],
                dependencies=["math_foundations", "logic_reasoning"],
                metadata={
                    "priority": 5,
                    "description": "AMC/AIME/IMO style datasets",
                    "domains": ["mathematics", "problem_solving"],
                },
            )
        )
        self.add_entry(
            CorpusEntry(
                id="undergraduate_math",
                name="Undergraduate Mathematics",
                path="docs/pdfs/undergraduate_math/",
                tier=CorpusTier.TIER_2_DOMAIN,
                corpus_type=CorpusType.PDF,
                target_galaxies=["Math", "Grammar"],
                dependencies=["math_foundations"],
                metadata={
                    "priority": 6,
                    "description": "Proof and analysis oriented material",
                    "domains": ["mathematics"],
                },
            )
        )
        self.add_entry(
            CorpusEntry(
                id="geometry_theorems",
                name="Geometry Theorems",
                path="docs/pdfs/geometry/",
                tier=CorpusTier.TIER_2_DOMAIN,
                corpus_type=CorpusType.PDF,
                target_galaxies=["Drawing", "Math"],
                dependencies=["math_foundations"],
                metadata={
                    "priority": 7,
                    "description": "Geometric constructions and transformations",
                    "domains": ["mathematics", "visual"],
                },
            )
        )
        self.add_entry(
            CorpusEntry(
                id="arc_agi_training",
                name="ARC-AGI Training Set",
                path="docs/datasets/arc_agi/",
                tier=CorpusTier.TIER_2_DOMAIN,
                corpus_type=CorpusType.DATASET,
                target_galaxies=["Drawing", "Grammar"],
                dependencies=["geometry_theorems"],
                metadata={
                    "priority": 8,
                    "description": "Visual reasoning transformations",
                    "domains": ["visual", "reasoning"],
                },
            )
        )
        self.add_entry(
            CorpusEntry(
                id="classical_mechanics",
                name="Classical Mechanics",
                path="docs/pdfs/physics/classical_mechanics/",
                tier=CorpusTier.TIER_2_DOMAIN,
                corpus_type=CorpusType.PDF,
                target_galaxies=["Reality", "Math"],
                dependencies=["math_foundations", "cs_fundamentals"],
                metadata={
                    "priority": 9,
                    "description": "Mechanics formulations and worked problems",
                    "domains": ["physics", "mathematics"],
                },
            )
        )
        self.add_entry(
            CorpusEntry(
                id="grammar_rules",
                name="Grammar Transformation Rules",
                path="docs/datasets/grammar/",
                tier=CorpusTier.TIER_2_DOMAIN,
                corpus_type=CorpusType.DATASET,
                target_galaxies=["Grammar", "Word"],
                dependencies=["logic_reasoning"],
                metadata={
                    "priority": 10,
                    "description": "Transformation rules and parsing structures",
                    "domains": ["language", "logic"],
                },
            )
        )
        self.add_entry(
            CorpusEntry(
                id="problem_solving_strategies",
                name="Problem-Solving Strategies",
                path="docs/pdfs/problem_solving/",
                tier=CorpusTier.TIER_3_INTEGRATION,
                corpus_type=CorpusType.PDF,
                target_galaxies=["Grammar", "Math", "Reality", "Drawing"],
                dependencies=["algorithmic_thinking", "competition_math"],
                metadata={
                    "priority": 11,
                    "description": "Cross-domain solving heuristics",
                    "domains": ["reasoning", "meta_cognition"],
                },
            )
        )

    def add_entry(self, entry: CorpusEntry) -> None:
        if entry.id in self.entries:
            raise ValueError(f"Duplicate corpus entry id: {entry.id}")
        self.entries[entry.id] = entry

    def get_by_tier(self, tier: CorpusTier) -> list[CorpusEntry]:
        return [entry for entry in self.entries.values() if entry.tier == tier]

    def get_ingestion_order(self) -> list[CorpusEntry]:
        """Topological order by dependency graph with cycle detection."""
        result: list[CorpusEntry] = []
        state: dict[str, str] = {}  # unvisited/marks: visiting/visited

        def visit(entry_id: str) -> None:
            if entry_id not in self.entries:
                raise KeyError(f"Unknown dependency referenced: {entry_id}")
            node_state = state.get(entry_id)
            if node_state == "visited":
                return
            if node_state == "visiting":
                raise ValueError(f"Circular dependency detected at: {entry_id}")
            state[entry_id] = "visiting"
            entry = self.entries[entry_id]
            for dep_id in entry.dependencies:
                visit(dep_id)
            state[entry_id] = "visited"
            result.append(entry)

        for entry_id in self.entries:
            if state.get(entry_id) != "visited":
                visit(entry_id)

        return result

    def validate_paths(self, root: str | Path = ".") -> tuple[bool, list[str]]:
        """Validate that all configured paths exist under `root`."""
        root_path = Path(root)
        missing: list[str] = []
        for entry in self.entries.values():
            candidate = root_path / entry.path
            if not candidate.exists():
                missing.append(entry.path)
        return len(missing) == 0, missing

    def mark_ingested(self, entry_id: str, embedding_count: int | None = None) -> None:
        entry = self.entries[entry_id]
        entry.ingested = True
        entry.embedding_count = embedding_count

    def get_stats(self) -> dict[str, Any]:
        tier_1 = len(self.get_by_tier(CorpusTier.TIER_1_FOUNDATIONAL))
        tier_2 = len(self.get_by_tier(CorpusTier.TIER_2_DOMAIN))
        tier_3 = len(self.get_by_tier(CorpusTier.TIER_3_INTEGRATION))
        ingested = sum(1 for entry in self.entries.values() if entry.ingested)
        total = len(self.entries)
        return {
            "total_entries": total,
            "tier_1_count": tier_1,
            "tier_2_count": tier_2,
            "tier_3_count": tier_3,
            "ingested_count": ingested,
            "pending_count": total - ingested,
        }
