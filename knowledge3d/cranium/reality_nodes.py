"""Stacked reality node definitions for the Reality Enabler.

Mirrors the text galaxy pattern:
atoms → molecules → materials → systems (with optional subatomic extensions).

Each node:
- uses `component_refs` to compose higher tiers without duplication;
- carries procedural programs (`visual_rpn`, `behavior_rpn`, `law_rpn`);
- can hold a Matryoshka+PD04 embedding payload;
- keeps domain metadata for downstream specialists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RealityNode:
    """Base class for all reality_* nodes."""

    node_id: str
    node_type: str = ""
    component_refs: List[str] = field(default_factory=list)
    visual_rpn: str = ""
    behavior_rpn: str = ""
    law_rpn: str = ""
    embedding: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RealityAtom(RealityNode):
    """Floor 0: Atomic reality primitive (physics/chem/bio)."""

    def __post_init__(self) -> None:
        if not self.node_id.startswith("atom:"):
            self.node_id = f"atom:{self.node_id}"
        self.node_type = "reality_atom"


@dataclass
class RealityMolecule(RealityNode):
    """Floor 1: Molecule/compound built from atoms."""

    def __post_init__(self) -> None:
        if not self.node_id.startswith("molecule:"):
            self.node_id = f"molecule:{self.node_id}"
        self.node_type = "reality_molecule"


@dataclass
class RealityMaterial(RealityNode):
    """Floor 2: Material built from molecules or atoms."""

    def __post_init__(self) -> None:
        if not self.node_id.startswith("material:"):
            self.node_id = f"material:{self.node_id}"
        self.node_type = "reality_material"


@dataclass
class RealitySystem(RealityNode):
    """Floor 3+: System/experiment orchestrating components."""

    state: Dict[str, float] = field(default_factory=dict)
    # Tier metadata
    rpn_tier: int = 1  # 1: simple, 2: mid, 3: high
    rpn_instance: int = 0  # specific math core instance id
    matryoshka_dim: int = 128  # preferred embedding dimension

    def __post_init__(self) -> None:
        if not self.node_id.startswith("system:"):
            self.node_id = f"system:{self.node_id}"
        self.node_type = "reality_system"
