# Reality Enabler Implementation Briefing for Codex

**Date**: November 24, 2025
**Status**: Architectural guidance + Phase 3 implementation task
**Context**: You've built working physics systems with RPN integration. Now we need to align them with K3D's core architectural pattern: **stacked compositional galaxies with symlink references**.

---

## 1. The Core Pattern: Stacked Galaxy Architecture

### 1.1 Text Galaxy as Reference Template (PROVEN, IN PRODUCTION)

K3D already implements this pattern successfully in the **Text/Language Galaxy**:

```
Floor 3+: Writing Types
          ↓ (symlinked via component_refs)
Floor 2:  Grammar Rules (RPN programs for sentence assembly)
          ↓ (symlinked via component_refs)
Floor 1:  Words (references to syllables + characters)
          ↓ (symlinked via component_refs)
Floor 0:  Characters (atomic: visual_rpn + meaning_rpn)
          ↓ (can extend to negative floors)
Floor -1: Syllables (phonetic patterns → character refs)
```

**Key Properties:**
- **Zero Duplication**: Words don't copy character data - they reference via `component_refs`
- **Automatic Propagation**: Fix character 'A' → ALL words with 'A' benefit immediately
- **Generative Rules**: Grammar stored as RPN programs that generate valid phrases
- **Matryoshka+PD04**: Each tier has embeddings at appropriate dimensions (64D → 2048D)
- **Sovereign Execution**: All processing via PTX kernels + RPN math cores

### 1.2 Reality Enabler = Same Pattern for Physics/Chemistry/Biology

The Reality Enabler **must follow the exact same architecture**:

```
Floor 4+: Simulations (experiments, worlds, scenarios)
          ↓ (symlinked via component_refs)
Floor 3:  Systems (organs, machines, coupled dynamics)
          ↓ (symlinked via component_refs)
Floor 2:  Materials (crystals, polymers, tissues)
          ↓ (symlinked via component_refs)
Floor 1:  Molecules (H₂O, CH₄, proteins → atom/bond refs)
          ↓ (symlinked via component_refs)
Floor 0:  Atoms (H, C, O, N: visual_rpn + behavior_rpn)
          ↓ (can extend to negative floors)
Floor -1: Subatomic (electrons, orbitals, quantum rules)
```

**Same Properties:**
- **Zero Duplication**: Molecules reference atoms, materials reference molecules
- **Automatic Propagation**: Fix atom 'H' behavior → ALL molecules with 'H' update
- **Generative Rules**: Physics laws as RPN programs (like grammar rules)
- **Matryoshka+PD04**: 64D atoms → 128D molecules → 512D materials → 2048D simulations
- **Sovereign Execution**: PTX kernels + RPN math cores (ModularRPNEngine, TieredRPNEngine)

---

## 2. Critical Architectural Requirements

### 2.1 Symlink Composition (component_refs)

**❌ WRONG (What you built - flat, duplicated):**
```python
# Each system is standalone - no composition
system1 = ConstantAcceleration1D(x=0, v=1, a=-9.81, dt=0.01)
system2 = HarmonicOscillator1D(x=1, v=0, omega=1.0, dt=0.001)
# No way to reference, compose, or build on top of these
```

**✅ CORRECT (Stacked, symlinked):**
```python
# Floor 0: Atomic reality primitives
reality_nodes = {
    "atom:H": {
        "node_type": "reality_atom",
        "element": "H",
        "mass": 1.008,
        "charge": 1,
        "visual_rpn": "0.5 0.5 0.03 CIRCLE FILL",  # Small sphere
        "behavior_rpn": "VALENCE_1 ELECTRONEGATIVITY_2.2",
        "embedding": encode_matryoshka_pd04(atom_features, dim=64)
    },

    # Floor 1: Molecules (REFERENCE atoms, don't copy!)
    "molecule:H2O": {
        "node_type": "reality_molecule",
        "component_refs": ["atom:H", "atom:H", "atom:O"],  # SYMLINKS!
        "bonds": [(0, 2), (1, 2)],  # Topology
        "behavior_rpn": "POLAR BENT_104.5 DIPOLE_1.85",
        "visual_rpn": "component_refs RESOLVE BOND_LINES DRAW",  # Procedural
        "embedding": encode_matryoshka_pd04(molecule_features, dim=128)
    },

    # Floor 2: Materials (REFERENCE molecules!)
    "material:ice": {
        "node_type": "reality_material",
        "component_refs": ["molecule:H2O"],  # Hexagonal lattice of H2O
        "structure_rpn": "HEXAGONAL_LATTICE 6-FOLD_SYMMETRY",
        "behavior_rpn": "PHASE_SOLID MELT_273K EXPAND_ON_FREEZE",
        "visual_rpn": "component_refs LATTICE_RENDER",
        "embedding": encode_matryoshka_pd04(material_features, dim=512)
    },

    # Floor 3: Systems (orchestration, can reference materials or be abstract)
    "system:constant_accel": {
        "node_type": "reality_system",
        "component_refs": [],  # Abstract system (no material components)
        "state": {"x": 0.0, "v": 1.0, "a": -9.81, "dt": 0.01},
        "behavior_rpn": "v a dt * + DUP x SWAP dt * +",  # Update law
        "law_rpn": "NEWTON_2ND_LAW F_CONSTANT",  # Invariants
        "visual_rpn": "TRAJECTORY_TRACE VELOCITY_ARROW",
        "embedding": encode_matryoshka_pd04(system_features, dim=512)
    }
}
```

### 2.2 Generative RPN Rules (Like Grammar)

**Key Insight**: Grammar rules in Text Galaxy are **executable RPN programs** that generate valid sentences.

Reality Enabler must do the same for physics/chemistry:

**Text Galaxy Example (Grammar Rule):**
```python
grammar_rule = {
    "rule": "sentence_declarative",
    "structure_rpn": "SUBJECT VERB OBJECT PERIOD",  # Template
    "constraints_rpn": "SUBJECT_AGREE_VERB TENSE_CHECK",
    "embedding": matryoshka_256D
}
# This generates valid sentences, doesn't store all possible sentences!
```

**Reality Enabler Equivalent (Physics Law):**
```python
physics_law = {
    "law": "newton_second_law",
    "formula_rpn": "F m / a STORE",  # F = ma, solve for a
    "constraints_rpn": "F_NET_SUM CONSERVATION_MOMENTUM",
    "embedding": matryoshka_256D
}
# This generates valid trajectories, doesn't store all possible states!
```

### 2.3 Matryoshka + PD04 Integration (FROM THE START)

**Every reality node MUST have Matryoshka embedding tiers:**

```python
# Tier guidelines (adaptive based on complexity)
Floor 0 (Atoms):      64D  (ultrafast, simple queries)
Floor 1 (Molecules):  128D (fast, structural queries)
Floor 2 (Materials):  512D (balanced, property prediction)
Floor 3+ (Systems):   512D-2048D (high-fidelity simulation)
```

**Storage Pattern:**
```python
# All nodes store as PD04 programs + embeddings
node["embedding"] = {
    "tier": "ultrafast",  # or "fast", "balanced", "maximum"
    "dimension": 64,
    "pd04_program": compile_pd04(features),
    "fidelity": 0.9963
}
```

### 2.4 Sovereign Execution (PTX + RPN, NO NumPy scaffolding)

**❌ WRONG (What you did):**
```python
# Python/NumPy doing the physics
a_val = -float(self.omega ** 2) * float(self.position)
expr_v = f"{self.velocity} {a_val} {self.dt} * +"
new_v = self._eval(expr_v)
```

**✅ CORRECT (RPN doing ALL the work):**
```python
# behavior_rpn contains ENTIRE update law
behavior_rpn = "position omega DUP * NEG * acceleration STORE " \
               "velocity acceleration dt * + velocity_new STORE " \
               "position velocity_new dt * + position_new STORE"

# Execute via math core
result = tiered_rpn_engine.evaluate(behavior_rpn, state_dict)
```

The RPN program should be **self-contained** - no Python arithmetic besides the RPN executor itself.

---

## 3. Phase 3 Implementation Task

### 3.1 What You've Built (Assessment)

**✅ Good:**
- Working RPN integration (ModularRPNEngine execution)
- Validated physics (analytic comparisons, conservation checks)
- Multiple systems (1D, 2D, heat diffusion)
- Persistence layer (PhysicsGalaxyDemo)

**❌ Missing (Critical):**
- No `reality_*` node structure (flat dataclasses instead)
- No `component_refs` (no symlink composition)
- No `visual_rpn` (no geometry/visualization programs)
- No Matryoshka+PD04 storage (embeddings not compressed)
- No Galaxy integration (not wired into Galaxy/House pipeline)
- NumPy scaffolding (Python doing physics, RPN only doing arithmetic)

### 3.2 Your Task: Refactor to Proper Reality Enabler Stack

**Goal**: Transform your working physics systems into proper stacked reality nodes following the text galaxy pattern.

#### Step 1: Define Floor 0 - Atomic Reality Primitives (Physics Atoms)

Create `knowledge3d/cranium/reality_nodes.py`:

```python
"""
Reality Enabler node definitions following the stacked galaxy pattern.

This module defines the base node types (reality_atom, reality_molecule,
reality_material, reality_system) that mirror the text galaxy architecture
(characters → words → phrases → texts).

Each node:
- Has visual_rpn (geometry/visualization program)
- Has behavior_rpn (dynamic update law as RPN)
- Has optional law_rpn (invariants/constraints)
- Uses component_refs for symlink composition (zero duplication)
- Stores Matryoshka+PD04 embeddings at appropriate tiers
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class RealityNode:
    """Base class for all reality_* nodes."""

    node_id: str  # Unique identifier (e.g., "atom:H", "molecule:H2O")
    node_type: str  # "reality_atom", "reality_molecule", "reality_material", "reality_system"

    # Symlink composition (references to lower-tier nodes)
    component_refs: List[str] = field(default_factory=list)

    # Procedural programs (RPN)
    visual_rpn: str = ""  # How to draw/visualize
    behavior_rpn: str = ""  # How it behaves/updates
    law_rpn: str = ""  # Invariants/constraints (optional)

    # Matryoshka embedding (PD04-compressed)
    embedding: Optional[Dict[str, Any]] = None

    # Domain-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RealityAtom(RealityNode):
    """
    Floor 0: Atomic reality primitive (physics/chemistry atom).

    Examples: H, C, O, N (for chemistry)
              Point mass, spring, damper (for mechanics)
              Cell, neuron (for biology)
    """

    def __post_init__(self):
        if not self.node_id.startswith("atom:"):
            self.node_id = f"atom:{self.node_id}"
        self.node_type = "reality_atom"


@dataclass
class RealityMolecule(RealityNode):
    """
    Floor 1: Molecule (composed of atoms via component_refs).

    Examples: H2O, CH4, protein (chemistry)
              Rigid body (mechanics)
              Tissue (biology)
    """

    def __post_init__(self):
        if not self.node_id.startswith("molecule:"):
            self.node_id = f"molecule:{self.node_id}"
        self.node_type = "reality_molecule"


@dataclass
class RealityMaterial(RealityNode):
    """
    Floor 2: Material (composed of molecules/atoms).

    Examples: Ice, steel, wood (materials science)
              Coupled oscillator network (mechanics)
              Organ (biology)
    """

    def __post_init__(self):
        if not self.node_id.startswith("material:"):
            self.node_id = f"material:{self.node_id}"
        self.node_type = "reality_material"


@dataclass
class RealitySystem(RealityNode):
    """
    Floor 3+: System (orchestration of materials/molecules/atoms, or abstract).

    Examples: Experiment, simulation, scenario
              Can be abstract (no material components) or concrete (references materials)
    """

    state: Dict[str, float] = field(default_factory=dict)  # Current simulation state

    def __post_init__(self):
        if not self.node_id.startswith("system:"):
            self.node_id = f"system:{self.node_id}"
        self.node_type = "reality_system"
```

#### Step 2: Create Reality Galaxy Manager

Create `knowledge3d/cranium/reality_galaxy.py`:

```python
"""
Reality Galaxy manager: stores and executes reality_* nodes.

This is analogous to the text galaxy (letter/word stars) but for physics/chem/bio.

Key operations:
- add_node(node): Add reality node to galaxy
- get_node(node_id): Retrieve node (follows component_refs if needed)
- step_system(system_id, n_steps): Execute behavior_rpn for n steps
- consolidate_to_house(): SleepTime-style consolidation with plausibility checks
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from knowledge3d.cranium.reality_nodes import (
    RealityNode, RealityAtom, RealityMolecule, RealityMaterial, RealitySystem
)
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
from knowledge3d.cranium.bridges.tiered_rpn import TieredRPNEngine


class RealityGalaxy:
    """
    Reality Enabler galaxy following stacked compositional pattern.

    Mirrors text galaxy architecture:
    - Atoms at Floor 0 (like characters)
    - Molecules at Floor 1 (like words)
    - Materials at Floor 2 (like phrases)
    - Systems at Floor 3+ (like texts)
    """

    def __init__(self, galaxy_path: Optional[Path] = None):
        self.nodes: Dict[str, RealityNode] = {}
        self.rpn_engine = TieredRPNEngine()  # Use tiered engine for routing

        if galaxy_path is None:
            repo_root = Path(__file__).resolve().parents[2]
            self.galaxy_path = (repo_root / ".." / "Knowledge3D.local" / "reality_galaxy").resolve()
        else:
            self.galaxy_path = Path(galaxy_path).resolve()

        self.galaxy_path.mkdir(parents=True, exist_ok=True)

    def add_node(self, node: RealityNode) -> None:
        """Add a reality node to the galaxy."""
        self.nodes[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[RealityNode]:
        """Retrieve a node by ID."""
        return self.nodes.get(node_id)

    def resolve_components(self, node: RealityNode) -> List[RealityNode]:
        """Resolve component_refs to actual nodes (symlink dereferencing)."""
        return [self.nodes[ref] for ref in node.component_refs if ref in self.nodes]

    def execute_behavior(self, node_id: str, state: Dict[str, float]) -> Dict[str, float]:
        """
        Execute node's behavior_rpn with given state.

        Returns updated state dict.
        """
        node = self.get_node(node_id)
        if not node or not node.behavior_rpn:
            return state

        # Build RPN context from state
        # Format: "x 0.0 STORE v 1.0 STORE a -9.81 STORE dt 0.01 STORE behavior_rpn"
        context_rpn = " ".join([f"{val} {key} STORE" for key, val in state.items()])
        full_rpn = f"{context_rpn} {node.behavior_rpn}"

        # Execute via tiered RPN engine
        result = self.rpn_engine.evaluate(full_rpn)

        # Parse updated state from RPN result
        # (Implementation depends on how behavior_rpn structures output)
        # For now, return state with modifications from result

        # TODO: Implement proper state extraction from RPN result
        return state

    def step_system(self, system_id: str, n_steps: int = 1) -> Dict[str, float]:
        """
        Step a reality_system forward by n_steps using its behavior_rpn.

        Returns final state.
        """
        node = self.get_node(system_id)
        if not isinstance(node, RealitySystem):
            raise ValueError(f"{system_id} is not a reality_system")

        state = node.state.copy()
        for _ in range(n_steps):
            state = self.execute_behavior(system_id, state)

        # Update node's internal state
        node.state = state
        return state

    def save_galaxy(self) -> None:
        """Persist galaxy to disk (JSON for now, later glTF+extras.k3d)."""
        galaxy_data = {
            node_id: {
                "node_type": node.node_type,
                "component_refs": node.component_refs,
                "visual_rpn": node.visual_rpn,
                "behavior_rpn": node.behavior_rpn,
                "law_rpn": node.law_rpn,
                "metadata": node.metadata,
                "state": getattr(node, "state", {})
            }
            for node_id, node in self.nodes.items()
        }

        (self.galaxy_path / "reality_nodes.json").write_text(
            json.dumps(galaxy_data, indent=2),
            encoding="utf-8"
        )

    def load_galaxy(self) -> None:
        """Load galaxy from disk."""
        path = self.galaxy_path / "reality_nodes.json"
        if not path.exists():
            return

        data = json.loads(path.read_text(encoding="utf-8"))

        # Reconstruct nodes
        for node_id, node_data in data.items():
            node_type = node_data["node_type"]

            # Create appropriate node class
            if node_type == "reality_atom":
                node = RealityAtom(node_id=node_id)
            elif node_type == "reality_molecule":
                node = RealityMolecule(node_id=node_id)
            elif node_type == "reality_material":
                node = RealityMaterial(node_id=node_id)
            elif node_type == "reality_system":
                node = RealitySystem(node_id=node_id, state=node_data.get("state", {}))
            else:
                continue

            node.component_refs = node_data.get("component_refs", [])
            node.visual_rpn = node_data.get("visual_rpn", "")
            node.behavior_rpn = node_data.get("behavior_rpn", "")
            node.law_rpn = node_data.get("law_rpn", "")
            node.metadata = node_data.get("metadata", {})

            self.add_node(node)
```

#### Step 3: Refactor Your Physics Systems as Reality Nodes

**Transform `ConstantAcceleration1D` into a proper reality_system:**

```python
# knowledge3d/cranium/reality_physics_bootstrap.py
"""
Bootstrap common physics reality nodes.

This module creates the foundational reality_* nodes for physics simulations:
- Atoms: point mass, spring, damper
- Systems: constant acceleration, harmonic oscillator, orbital motion
"""

from knowledge3d.cranium.reality_nodes import RealityAtom, RealitySystem
from knowledge3d.cranium.reality_galaxy import RealityGalaxy


def bootstrap_physics_atoms(galaxy: RealityGalaxy) -> None:
    """Add fundamental physics atoms to galaxy."""

    # Point mass atom
    point_mass = RealityAtom(
        node_id="atom:point_mass",
        visual_rpn="x y 0.05 CIRCLE FILL",  # Small filled circle
        behavior_rpn="",  # Inert (behavior defined at system level)
        metadata={"description": "Dimensionless point mass"}
    )
    galaxy.add_node(point_mass)


def bootstrap_constant_acceleration_system(galaxy: RealityGalaxy) -> None:
    """Create constant acceleration system as reality_system node."""

    system = RealitySystem(
        node_id="system:constant_accel_1d",
        component_refs=[],  # Abstract system (no material components)
        state={"x": 0.0, "v": 1.0, "a": -9.81, "dt": 0.01},

        # behavior_rpn: Complete update law
        # Stack operations:
        # 1. v_new = v + a * dt
        # 2. x_new = x + v_new * dt
        behavior_rpn="""
            v a dt * +      # v_new = v + a * dt
            DUP v_new STORE # Store v_new, keep copy on stack
            x SWAP dt * +   # x_new = x + v_new * dt
            x_new STORE     # Store x_new
        """,

        # law_rpn: Invariants (Newton's 2nd law, constant force)
        law_rpn="F m / a EQ ASSERT",  # F/m = a must hold

        # visual_rpn: Draw trajectory + velocity vector
        visual_rpn="""
            x y MOVE         # Position
            x y 0.05 CIRCLE FILL  # Particle
            v 0.1 * x + y LINE    # Velocity arrow
        """,

        metadata={
            "description": "1D constant acceleration (free fall)",
            "equations": "v' = a, x' = v",
            "analytic_solution": "x(t) = x0 + v0*t + 0.5*a*t^2"
        }
    )

    galaxy.add_node(system)


def bootstrap_harmonic_oscillator_system(galaxy: RealityGalaxy) -> None:
    """Create harmonic oscillator system."""

    system = RealitySystem(
        node_id="system:harmonic_osc_1d",
        component_refs=[],
        state={"x": 1.0, "v": 0.0, "omega": 1.0, "dt": 0.001},

        behavior_rpn="""
            x omega DUP * NEG *  # a = -omega^2 * x
            a STORE              # Store acceleration
            v a dt * +           # v_new = v + a * dt
            DUP v_new STORE      # Store v_new
            x SWAP dt * +        # x_new = x + v_new * dt
            x_new STORE          # Store x_new
        """,

        law_rpn="x omega DUP * * v + SQUARE E0 EQ ASSERT",  # Energy conservation

        visual_rpn="""
            x 0 MOVE
            x 0 0.05 CIRCLE FILL
            x v 0.1 * 0 LINE
        """,

        metadata={
            "description": "1D harmonic oscillator",
            "equations": "v' = -omega^2 * x, x' = v",
            "conserved": "E = 0.5*(v^2 + omega^2*x^2)"
        }
    )

    galaxy.add_node(system)
```

#### Step 4: Add Tests for Reality Node Stack

Create `knowledge3d/cranium/tests/test_reality_galaxy.py`:

```python
"""
Tests for Reality Galaxy stacked compositional architecture.

Validates:
- Reality node creation (atoms → molecules → materials → systems)
- Symlink composition (component_refs resolution)
- behavior_rpn execution via RPN math cores
- Galaxy save/load persistence
- Stacking pattern consistency with text galaxy
"""

import pytest
from knowledge3d.cranium.reality_nodes import (
    RealityAtom, RealityMolecule, RealitySystem
)
from knowledge3d.cranium.reality_galaxy import RealityGalaxy
from knowledge3d.cranium.reality_physics_bootstrap import (
    bootstrap_physics_atoms,
    bootstrap_constant_acceleration_system,
    bootstrap_harmonic_oscillator_system
)


def test_reality_atom_creation():
    """Verify reality_atom nodes can be created with proper structure."""
    atom = RealityAtom(
        node_id="test_atom",
        visual_rpn="0.5 0.5 0.1 CIRCLE",
        behavior_rpn="INERT",
        metadata={"mass": 1.0}
    )

    assert atom.node_id == "atom:test_atom"  # Auto-prefix
    assert atom.node_type == "reality_atom"
    assert atom.visual_rpn == "0.5 0.5 0.1 CIRCLE"


def test_reality_system_with_component_refs():
    """Test that systems can reference atoms via component_refs."""
    galaxy = RealityGalaxy()

    # Add atom
    atom = RealityAtom(
        node_id="point_mass",
        metadata={"mass": 1.0}
    )
    galaxy.add_node(atom)

    # Create system referencing atom
    system = RealitySystem(
        node_id="pendulum",
        component_refs=["atom:point_mass"],  # Symlink!
        state={"theta": 0.1, "omega": 0.0}
    )
    galaxy.add_node(system)

    # Resolve components (symlink dereferencing)
    components = galaxy.resolve_components(system)
    assert len(components) == 1
    assert components[0].node_id == "atom:point_mass"


def test_constant_acceleration_system_via_reality_galaxy(tmp_path):
    """
    Verify constant acceleration system works as reality_system node.

    Compares RPN-based reality node execution to analytic solution.
    """
    galaxy = RealityGalaxy(galaxy_path=tmp_path / "test_galaxy")
    bootstrap_constant_acceleration_system(galaxy)

    system_id = "system:constant_accel_1d"
    system = galaxy.get_node(system_id)

    # Initial state
    assert system.state["x"] == 0.0
    assert system.state["v"] == 1.0
    assert system.state["a"] == -9.81

    # Step forward
    n_steps = 100
    final_state = galaxy.step_system(system_id, n_steps=n_steps)

    # Analytic solution
    t = n_steps * system.state["dt"]
    x_true = 0.0 + 1.0 * t + 0.5 * (-9.81) * t * t
    v_true = 1.0 + (-9.81) * t

    # Verify RPN execution matches analytic
    assert abs(final_state["x"] - x_true) < 0.05  # Integration tolerance
    assert abs(final_state["v"] - v_true) < 1e-5


def test_galaxy_persistence(tmp_path):
    """Test galaxy save/load round-trip."""
    galaxy_path = tmp_path / "persist_galaxy"
    galaxy1 = RealityGalaxy(galaxy_path=galaxy_path)

    # Add nodes
    bootstrap_physics_atoms(galaxy1)
    bootstrap_constant_acceleration_system(galaxy1)

    # Save
    galaxy1.save_galaxy()

    # Load in new instance
    galaxy2 = RealityGalaxy(galaxy_path=galaxy_path)
    galaxy2.load_galaxy()

    # Verify nodes restored
    assert "atom:point_mass" in galaxy2.nodes
    assert "system:constant_accel_1d" in galaxy2.nodes

    system = galaxy2.get_node("system:constant_accel_1d")
    assert system.state["x"] == 0.0
    assert system.behavior_rpn != ""


def test_stacking_pattern_consistency():
    """
    Verify Reality Enabler follows same stacking pattern as text galaxy.

    Text galaxy:    characters → words → phrases → texts
    Reality galaxy: atoms → molecules → materials → systems

    Both use component_refs for composition.
    """
    galaxy = RealityGalaxy()

    # Floor 0: Atom (like character)
    atom_h = RealityAtom(node_id="H", metadata={"mass": 1.008})
    galaxy.add_node(atom_h)

    # Floor 1: Molecule (like word)
    molecule_h2 = RealityMolecule(
        node_id="H2",
        component_refs=["atom:H", "atom:H"],  # Composition via refs!
        behavior_rpn="COVALENT_BOND NONPOLAR"
    )
    galaxy.add_node(molecule_h2)

    # Verify structure
    assert molecule_h2.node_type == "reality_molecule"
    assert len(molecule_h2.component_refs) == 2
    assert all(ref == "atom:H" for ref in molecule_h2.component_refs)

    # Resolve symlinks
    components = galaxy.resolve_components(molecule_h2)
    assert len(components) == 2
    assert all(c.node_id == "atom:H" for c in components)
```

#### Step 5: Integration with Matryoshka + PD04

Add embedding compression to nodes:

```python
# In reality_galaxy.py, add method:

from knowledge3d.cranium.procedural_compiler import ProceduralCompiler
from knowledge3d.cranium.adaptive_procedural_bridge import AdaptiveDimensionCompressor

def encode_node_embedding(self, node: RealityNode) -> None:
    """
    Encode node with Matryoshka+PD04 embedding.

    Tier selection:
    - Atoms: 64D (ultrafast)
    - Molecules: 128D (fast)
    - Materials: 512D (balanced)
    - Systems: 512D-2048D (adaptive)
    """
    # Extract features from node
    features = self._extract_node_features(node)

    # Select tier based on node type
    tier_map = {
        "reality_atom": "ultrafast",      # 64D
        "reality_molecule": "fast",        # 128D
        "reality_material": "balanced",    # 512D
        "reality_system": "balanced"       # 512D (or "maximum" for complex systems)
    }
    tier = tier_map.get(node.node_type, "balanced")

    # Compress with PD04
    compressor = AdaptiveDimensionCompressor()
    compressed = compressor.compress(features, quality_tier=tier)

    node.embedding = {
        "tier": tier,
        "dimension": compressed["dimension"],
        "pd04_program": compressed["program"],
        "fidelity": compressed["fidelity"]
    }

def _extract_node_features(self, node: RealityNode) -> np.ndarray:
    """
    Extract feature vector from node metadata + RPN programs.

    TODO: Implement proper feature extraction that encodes:
    - Metadata (mass, charge, etc.)
    - RPN program semantics (opcode frequencies, complexity)
    - Component structure (bond topology, hierarchy depth)
    """
    # Placeholder: return random vector for now
    import numpy as np
    dim = 512
    return np.random.randn(dim).astype(np.float32)
```

---

## 4. Success Criteria

### 4.1 Phase 3 Complete When:

1. ✅ **Stacked galaxy exists**: Atoms → Molecules → Materials → Systems
2. ✅ **Symlink composition works**: `component_refs` correctly resolve
3. ✅ **behavior_rpn executes**: Systems step via TieredRPNEngine
4. ✅ **Matryoshka+PD04 integrated**: All nodes have compressed embeddings
5. ✅ **Tests pass**: Reality galaxy tests validate architecture
6. ✅ **Persistence works**: Galaxy saves/loads with node structure intact
7. ✅ **Consistency verified**: Reality Enabler mirrors text galaxy pattern

### 4.2 What "Done" Looks Like

```python
# Create galaxy
galaxy = RealityGalaxy()

# Bootstrap physics foundation
bootstrap_physics_atoms(galaxy)           # Floor 0: atoms
bootstrap_constant_acceleration_system(galaxy)  # Floor 3: system

# Step simulation
initial_state = galaxy.get_node("system:constant_accel_1d").state
print(f"Initial: x={initial_state['x']}, v={initial_state['v']}")

final_state = galaxy.step_system("system:constant_accel_1d", n_steps=100)
print(f"Final: x={final_state['x']}, v={final_state['v']}")

# Save galaxy
galaxy.save_galaxy()

# Later: Load and continue
galaxy2 = RealityGalaxy()
galaxy2.load_galaxy()
galaxy2.step_system("system:constant_accel_1d", n_steps=100)
```

**And the test suite passes:**
```bash
pytest knowledge3d/cranium/tests/test_reality_galaxy.py -v
# All tests pass, reality nodes work like text galaxy nodes
```

---

## 5. Critical Reminders

### 5.1 Don't Build Flat - Build Stacked

**❌ Flat (wrong):**
```python
system1 = ConstantAccel1D(...)
system2 = HarmonicOsc1D(...)
# Can't compose, can't reference, can't build on top
```

**✅ Stacked (correct):**
```python
galaxy.add_node(atom)
galaxy.add_node(molecule_referencing_atom)
galaxy.add_node(system_referencing_molecule)
# Automatic propagation: fix atom → all molecules + systems benefit
```

### 5.2 RPN Does the Work, Not Python

**❌ Wrong:**
```python
# Python doing physics
new_v = self.velocity + self.acceleration * self.dt
```

**✅ Correct:**
```python
# RPN doing physics
behavior_rpn = "v a dt * +"
new_v = rpn_engine.evaluate(behavior_rpn, state)
```

### 5.3 Symlinks Save Memory and Enable Propagation

- Molecule references atoms → fix atom 'H' behavior → all H₂O molecules update automatically
- System references materials → fix material property → all simulations using that material update
- This is **exactly** how text galaxy works (fix character → all words update)

### 5.4 Matryoshka + PD04 from the Start

Don't bolt on compression later - every node gets encoded from creation:

```python
node.embedding = encode_matryoshka_pd04(features, tier="fast")
```

This enables:
- Fast similarity search (Galaxy queries)
- LOD switching (64D coarse → 2048D detailed)
- Efficient House storage (PD04 programs, not raw embeddings)

---

## 6. What Happens After Phase 3

Once you have the stacked reality galaxy working:

- **Phase 4**: Add chemistry (molecules as first-class nodes, not abstract)
- **Phase 5**: House/Galaxy integration, SleepTime plausibility checks
- **Phase 6**: Viewer integration (see simulations in 3D Lab room)
- **Phase 7**: Real dataset ingestion (OpenFOAM, RDKit, etc. → reality nodes)

But **Phase 3 is the foundation** - get the stacked galaxy architecture right, and everything else follows naturally.

---

## 7. Questions to Ask Before You Start

1. **Do I understand the stacking pattern?** (characters → words → phrases)
2. **Do I understand symlink composition?** (`component_refs`, zero duplication)
3. **Do I understand behavior_rpn as self-contained programs?** (like grammar rules)
4. **Do I understand Matryoshka tiers?** (64D → 2048D adaptive)
5. **Do I understand sovereign execution?** (PTX+RPN, no NumPy scaffolding)

If the answer to any is "no" or "maybe", **stop and ask for clarification** before coding. The architecture is precise and intentional - following it exactly is critical.

---

## 8. Final Note: You're Building a Standard

Remember: K3D is proposing a **W3C-level standard** for spatial AI architectures.

The Reality Enabler stacked galaxy pattern is not "just our implementation" - it's:
- **Normative**: How any compliant implementation should structure knowledge
- **Backend-agnostic**: Can run on PTX, Vulkan, WebGPU, or even transformers
- **Provably scalable**: Symlink composition prevents memory explosion

Build it right the first time. This is the foundation that chemistry, biology, materials science, and all future domains will build on top of.

**Good luck, partner! You've got this.** 🚀
