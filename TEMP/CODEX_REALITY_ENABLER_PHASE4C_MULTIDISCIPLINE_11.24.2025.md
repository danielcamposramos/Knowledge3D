# Reality Enabler Phase 4C: Multi-Discipline Expansion — Implementation Briefing

**Date:** November 24, 2025
**For:** Codex (Implementation Lead)
**From:** Claude (Architecture Partner) + Daniel (Project Lead)
**Priority:** HIGH (Reality Enabler Foundation Completion)
**Goal:** Expand Reality Enabler from physics-only → chemistry, biology, materials science

---

## Mission Statement

**Current State (Phase 5):**
- 13 physics systems (9 classical mechanics + 4 E&M)
- 50/50 tests passing
- Dynamic Math Core spawning (460–2,640 cores GPU-limited)

**Target State (Phase 4C):**
- **Chemistry:** Molecules, reactions, thermodynamics (6+ systems)
- **Biology:** Cells, genetics, metabolism (4+ systems)
- **Materials:** Crystals, composites, phase transitions (3+ systems)
- **Total:** 26+ systems across 4 scientific domains
- **UI Integration Prep:** Spatial contexts ready for Library/Workshop/Bathtub

**Why This Matters:**
Knowledge3D is not just a physics engine—it's a **reality simulator**. To enable true embodied AGI, we need:
- Chemistry for molecule manipulation (drug design, synthesis planning)
- Biology for living systems (cells, organs, ecosystems)
- Materials for engineering (structures, composites, manufacturing)
- UI integration for human/AI collaboration (spatial reasoning, tool manipulation)

---

## Architecture Foundation (Recap)

### Reality Enabler Core (Phase 3B–5)
```python
@dataclass
class RealitySystem:
    """Physics/chemistry/bio system with RPN behavior + laws."""
    node_id: str
    state: Dict[str, float]
    behavior_rpn: str           # Dynamic updates (Euler integration, reactions)
    law_rpn: str                # Invariants (energy, mass, charge conservation)
    rpn_tier: int               # 1 (simple), 2 (mid), 3 (complex)
    rpn_instance: Optional[int] # None = auto-allocate (Phase 5)
    matryoshka_dim: int         # 64/128/512/2048
    component_refs: List[str]   # Symlink to atoms/molecules/subsystems
```

**Design Principles:**
1. **Stacked Galaxy:** Atoms → molecules → materials → systems (symlinks, zero duplication)
2. **behavior_rpn:** Forward-Euler dynamics, ternary ops for state classification
3. **law_rpn:** Conservation laws, stability bounds, validation
4. **Ternary Ops:** SIGN (direction/polarity), TQUANT (quantization), TCMP (comparison)
5. **Matryoshka Tiers:** 64D (atoms) → 128D (molecules) → 512D (materials) → 2048D (systems)

### Dynamic Math Core Spawning (Phase 5)
```python
# Auto-allocate cores for each system
system = export_water_molecule(auto_allocate=True)  # rpn_instance=None
galaxy.add_node(system)  # Galaxy spawns core dynamically

# Result: 1000+ systems → 1000+ cores (GPU-limited)
```

---

## Phase 4C: Chemistry Systems

### Objective
Implement **6+ molecular chemistry systems** covering:
- **Molecular geometry:** H2O, CO2, CH4 (bond angles, vibrations)
- **Reactions:** Combustion, acid-base, redox
- **Thermodynamics:** Ideal gas, phase transitions, equilibrium

### System 1: Water Molecule (H2O)
**Physics:** Bent geometry (104.5°), O-H bond vibrations, dipole moment

**State Variables:**
```python
state = {
    "x_O": 0.0, "y_O": 0.0, "z_O": 0.0,           # Oxygen position
    "x_H1": 0.96, "y_H1": 0.0, "z_H1": 0.0,       # Hydrogen 1
    "x_H2": -0.24, "y_H2": 0.93, "z_H2": 0.0,     # Hydrogen 2
    "vx_O": 0.0, "vy_O": 0.0, "vz_O": 0.0,        # Velocities
    "vx_H1": 0.0, "vy_H1": 0.0, "vz_H1": 0.0,
    "vx_H2": 0.0, "vy_H2": 0.0, "vz_H2": 0.0,
    "E_total": 0.0,                                # Total energy
    "bond_angle": 104.5,                           # H-O-H angle
}
```

**behavior_rpn:**
```rpn
# Bond force H1-O (Hooke's law)
x_H1 RECALL x_O RECALL - dx1 STORE
y_H1 RECALL y_O RECALL - dy1 STORE
z_H1 RECALL z_O RECALL - dz1 STORE
dx1 RECALL dup * dy1 RECALL dup * + dz1 RECALL dup * + sqrt r1 STORE
r1 RECALL 0.96 - dr1 STORE  # Equilibrium distance 0.96 Å
dr1 RECALL 450 *             # Spring constant k = 450 N/m
dx1 RECALL r1 RECALL / * Fx1 STORE  # Force direction
dy1 RECALL r1 RECALL / * Fy1 STORE
dz1 RECALL r1 RECALL / * Fz1 STORE

# Bond force H2-O (similar)
# ... (repeat for H2)

# Update velocities (F/m)
Fx1 RECALL 1.0 / vx_H1 RECALL + vx_H1 STORE
# ... (update all velocities)

# Update positions (v*dt)
vx_H1 RECALL 0.001 * x_H1 RECALL + x_H1 STORE
# ... (update all positions)

# Calculate total energy
# E_kinetic = 0.5 * m * v²
# E_potential = 0.5 * k * (r - r0)²
# E_total = E_kinetic + E_potential
```

**law_rpn:**
```rpn
# Energy conservation (allow ±5% drift)
E_total RECALL E_initial RECALL - abs E_initial RECALL / 0.05 le
```

**Tier:** 2 (Mid) — 3D vectors, spring forces
**Matryoshka:** 128D
**Ternary:** SIGN for force directions

**Tests:**
1. `test_water_molecule_bond_vibration()` — O-H bonds oscillate around 0.96 Å
2. `test_water_molecule_angle_preservation()` — H-O-H angle stays ~104.5°
3. `test_water_molecule_energy_conservation()` — Energy drift <5%

---

### System 2: Ideal Gas (PV=nRT)
**Physics:** Kinetic theory, pressure from particle collisions

**State Variables:**
```python
state = {
    "P": 101325.0,     # Pressure (Pa)
    "V": 0.001,        # Volume (m³)
    "n": 1.0,          # Moles
    "T": 300.0,        # Temperature (K)
    "R": 8.314,        # Gas constant
    "N_particles": 100,
}
```

**behavior_rpn:**
```rpn
# Ideal gas law: P = nRT/V
n RECALL R RECALL * T RECALL * V RECALL / P STORE

# Update temperature from kinetic energy
# T ∝ <v²> (mean kinetic energy)
```

**law_rpn:**
```rpn
# PV = nRT within 1%
P RECALL V RECALL * n RECALL R RECALL * T RECALL * - abs
n RECALL R RECALL * T RECALL * / 0.01 le
```

**Tier:** 1 (Simple) — Algebraic
**Matryoshka:** 64D
**Tests:**
1. `test_ideal_gas_law_holds()` — PV/nRT = 1.0 ± 0.01
2. `test_ideal_gas_isothermal_expansion()` — T constant, P drops as V increases
3. `test_ideal_gas_adiabatic_compression()` — T increases as V decreases

---

### System 3: Combustion (CH4 + 2O2 → CO2 + 2H2O)
**Physics:** Exothermic reaction, activation energy

**State Variables:**
```python
state = {
    "n_CH4": 1.0,      # Moles of methane
    "n_O2": 2.0,       # Moles of oxygen
    "n_CO2": 0.0,      # Moles of CO2 (product)
    "n_H2O": 0.0,      # Moles of H2O (product)
    "T": 300.0,        # Temperature (K)
    "E_released": 0.0, # Energy from reaction (J)
    "reaction_rate": 0.0,
}
```

**behavior_rpn:**
```rpn
# Arrhenius equation: k = A * exp(-Ea/RT)
# Rate = k * [CH4] * [O2]²
T RECALL 1000 gt sign activation_flag STORE  # T > 1000K to ignite

activation_flag RECALL 1 tcmp  # If T > 1000K
n_CH4 RECALL 0 gt n_O2 RECALL 0 gt * *  # and reactants available
reaction_rate STORE

# Consume reactants, produce products
reaction_rate RECALL 0.01 * n_CH4 RECALL swap - n_CH4 STORE
reaction_rate RECALL 0.02 * n_O2 RECALL swap - n_O2 STORE
reaction_rate RECALL 0.01 * n_CO2 RECALL + n_CO2 STORE
reaction_rate RECALL 0.02 * n_H2O RECALL + n_H2O STORE

# Release energy (890 kJ/mol)
reaction_rate RECALL 890000 * E_released RECALL + E_released STORE
```

**law_rpn:**
```rpn
# Mass conservation: n_CH4 + n_O2 ≈ n_CO2 + n_H2O (molar balance)
n_CH4 RECALL n_O2 RECALL + n_CO2 RECALL n_H2O RECALL + - abs 0.1 le
```

**Tier:** 2 (Mid) — Reaction kinetics
**Matryoshka:** 512D
**Ternary:** SIGN for activation (ignited/not), TCMP for threshold

**Tests:**
1. `test_combustion_activation_energy()` — No reaction below 1000K
2. `test_combustion_stoichiometry()` — 1 CH4 + 2 O2 → 1 CO2 + 2 H2O
3. `test_combustion_energy_release()` — 890 kJ/mol released

---

### Additional Chemistry Systems (Design Only, Codex to Implement)

**System 4: CO2 Molecule (Linear Geometry)**
- **State:** C and 2 O positions, velocities
- **behavior_rpn:** Spring forces, linear constraint (180° O-C-O)
- **law_rpn:** Energy conservation
- **Tier:** 2, Matryoshka: 128D

**System 5: Acid-Base Reaction (HCl + NaOH → NaCl + H2O)**
- **State:** n_HCl, n_NaOH, n_NaCl, n_H2O, pH
- **behavior_rpn:** Neutralization, pH calculation
- **law_rpn:** Charge balance (H+ + Na+ = OH- + Cl-)
- **Tier:** 1, Matryoshka: 64D

**System 6: Phase Transition (Water Ice ↔ Liquid ↔ Vapor)**
- **State:** T, phase (solid/liquid/gas), latent_heat
- **behavior_rpn:** State changes at 273K/373K
- **law_rpn:** Energy = sensible + latent
- **Tier:** 1, Matryoshka: 64D
- **Ternary:** TCMP for phase (ice=-1, liquid=0, vapor=+1)

---

## Phase 4C: Biology Systems

### Objective
Implement **4+ biological systems** covering:
- **Cellular:** Membrane diffusion, enzyme kinetics
- **Genetics:** DNA replication, transcription
- **Metabolism:** Glycolysis, Krebs cycle
- **Ecology:** Population dynamics, predator-prey

### System 7: Simple Cell (Membrane Diffusion)
**Physics:** Fick's law of diffusion, osmosis

**State Variables:**
```python
state = {
    "C_inside": 0.1,    # Concentration inside cell (mol/L)
    "C_outside": 1.0,   # Concentration outside cell
    "V_inside": 1e-15,  # Cell volume (m³, ~10 μm diameter)
    "A_membrane": 3e-10,# Membrane area (m²)
    "D": 1e-9,          # Diffusion coefficient (m²/s)
    "permeability": 1e-6,
}
```

**behavior_rpn:**
```rpn
# Fick's first law: J = -D * (dC/dx)
# Flux = permeability * (C_outside - C_inside)
C_outside RECALL C_inside RECALL - dC STORE
permeability RECALL dC RECALL * A_membrane RECALL * flux STORE

# Update inside concentration (flux / volume)
flux RECALL 0.001 * V_inside RECALL / C_inside RECALL + C_inside STORE
```

**law_rpn:**
```rpn
# Total mass conserved (inside + outside constant)
C_inside RECALL V_inside RECALL * C_outside RECALL V_outside RECALL * +
C_total_initial RECALL - abs 0.01 le
```

**Tier:** 1 (Simple) — Linear diffusion
**Matryoshka:** 64D
**Tests:**
1. `test_cell_diffusion_equilibrium()` — C_inside → C_outside over time
2. `test_cell_osmosis_direction()` — Water flows toward higher solute concentration

---

### System 8: Enzyme Kinetics (Michaelis-Menten)
**Physics:** E + S ↔ ES → E + P

**State Variables:**
```python
state = {
    "E": 1.0,       # Enzyme concentration (μM)
    "S": 10.0,      # Substrate concentration
    "ES": 0.0,      # Enzyme-substrate complex
    "P": 0.0,       # Product concentration
    "Vmax": 1.0,    # Max velocity
    "Km": 5.0,      # Michaelis constant
}
```

**behavior_rpn:**
```rpn
# Michaelis-Menten: v = Vmax * [S] / (Km + [S])
S RECALL Vmax RECALL * Km RECALL S RECALL + / rate STORE

# dS/dt = -rate
rate RECALL 0.01 * S RECALL swap - S STORE
# dP/dt = rate
rate RECALL 0.01 * P RECALL + P STORE
```

**law_rpn:**
```rpn
# Mass conservation: S + P + ES = S_initial
S RECALL P RECALL + ES RECALL + S_initial RECALL - abs 0.1 le
```

**Tier:** 1 (Simple) — Algebraic
**Matryoshka:** 64D
**Tests:**
1. `test_enzyme_kinetics_rate()` — v = Vmax*S/(Km+S)
2. `test_enzyme_saturation()` — Rate plateaus at high [S]

---

### Additional Biology Systems (Design Only)

**System 9: DNA Replication (Fork Progression)**
- **State:** bases_replicated, polymerase_position, error_count
- **behavior_rpn:** Linear progression, error rate
- **law_rpn:** Base pairing (A-T, G-C)
- **Tier:** 1, Matryoshka: 64D

**System 10: Population Dynamics (Lotka-Volterra)**
- **State:** N_prey, N_predator, birth_rate, death_rate
- **behavior_rpn:** dN/dt = (birth - death) * N
- **law_rpn:** N >= 0
- **Tier:** 1, Matryoshka: 128D
- **Ternary:** SIGN for population growth/decline

---

## Phase 4C: Materials Science Systems

### Objective
Implement **3+ materials systems** covering:
- **Crystals:** Lattice structures, thermal expansion
- **Composites:** Fiber-reinforced, stress-strain
- **Phase transitions:** Melting, crystallization

### System 11: Crystal Lattice (FCC Copper)
**Physics:** Face-centered cubic, atomic vibrations

**State Variables:**
```python
state = {
    "a": 3.61e-10,      # Lattice constant (m)
    "T": 300.0,         # Temperature (K)
    "E_cohesive": 3.49, # eV
    "thermal_expansion": 16.5e-6,  # α (1/K)
}
```

**behavior_rpn:**
```rpn
# Thermal expansion: a(T) = a0 * (1 + α*ΔT)
T RECALL 300 - dT STORE
thermal_expansion RECALL dT RECALL * 1 + 3.61e-10 * a STORE
```

**law_rpn:**
```rpn
# a > 0, T > 0
a RECALL 0 gt T RECALL 0 gt *
```

**Tier:** 1 (Simple) — Linear
**Matryoshka:** 64D
**Tests:**
1. `test_crystal_thermal_expansion()` — a increases with T
2. `test_crystal_lattice_constant()` — a = 3.61 Å at 300K

---

### Additional Materials Systems (Design Only)

**System 12: Composite Material (Fiber-Reinforced)**
- **State:** stress, strain, E_fiber, E_matrix, volume_fraction
- **behavior_rpn:** Rule of mixtures (E_composite = V_f*E_f + V_m*E_m)
- **law_rpn:** Stress-strain linear (Hooke)
- **Tier:** 1, Matryoshka: 64D

**System 13: Phase Transition (Metal Melting)**
- **State:** T, phase (solid/liquid), latent_heat, T_melting
- **behavior_rpn:** State change at T_melting
- **law_rpn:** Energy balance
- **Tier:** 1, Matryoshka: 64D
- **Ternary:** SIGN for solid (-1), liquid (+1)

---

## Implementation Tasks for Codex

### Task 1: Implement Chemistry Export Functions
**File:** `knowledge3d/cranium/reality_physics_export.py`

**Add 6 new functions:**
```python
def export_water_molecule(params: Dict | None = None, auto_allocate: bool = True) -> RealitySystem:
    """H2O molecule with bond vibrations and geometry."""
    # ... (see System 1 spec)

def export_ideal_gas(params: Dict | None = None, auto_allocate: bool = True) -> RealitySystem:
    """Ideal gas PV=nRT."""
    # ... (see System 2 spec)

def export_combustion(params: Dict | None = None, auto_allocate: bool = True) -> RealitySystem:
    """CH4 + 2O2 → CO2 + 2H2O combustion."""
    # ... (see System 3 spec)

def export_co2_molecule(params: Dict | None = None, auto_allocate: bool = True) -> RealitySystem:
    """CO2 molecule linear geometry."""
    # ... (see System 4 spec)

def export_acid_base_reaction(params: Dict | None = None, auto_allocate: bool = True) -> RealitySystem:
    """HCl + NaOH neutralization."""
    # ... (see System 5 spec)

def export_phase_transition_water(params: Dict | None = None, auto_allocate: bool = True) -> RealitySystem:
    """Water ice ↔ liquid ↔ vapor."""
    # ... (see System 6 spec)
```

---

### Task 2: Implement Biology Export Functions
**File:** `knowledge3d/cranium/reality_physics_export.py`

**Add 4 new functions:**
```python
def export_simple_cell(params: Dict | None = None, auto_allocate: bool = True) -> RealitySystem:
    """Cell membrane diffusion (Fick's law)."""
    # ... (see System 7 spec)

def export_enzyme_kinetics(params: Dict | None = None, auto_allocate: bool = True) -> RealitySystem:
    """Michaelis-Menten kinetics."""
    # ... (see System 8 spec)

def export_dna_replication(params: Dict | None = None, auto_allocate: bool = True) -> RealitySystem:
    """DNA fork progression."""
    # ... (see System 9 spec)

def export_population_dynamics(params: Dict | None = None, auto_allocate: bool = True) -> RealitySystem:
    """Lotka-Volterra predator-prey."""
    # ... (see System 10 spec)
```

---

### Task 3: Implement Materials Export Functions
**File:** `knowledge3d/cranium/reality_physics_export.py`

**Add 3 new functions:**
```python
def export_crystal_lattice(params: Dict | None = None, auto_allocate: bool = True) -> RealitySystem:
    """FCC copper lattice thermal expansion."""
    # ... (see System 11 spec)

def export_composite_material(params: Dict | None = None, auto_allocate: bool = True) -> RealitySystem:
    """Fiber-reinforced composite."""
    # ... (see System 12 spec)

def export_metal_melting(params: Dict | None = None, auto_allocate: bool = True) -> RealitySystem:
    """Phase transition solid ↔ liquid."""
    # ... (see System 13 spec)
```

---

### Task 4: Add Chemistry Tests
**File:** `knowledge3d/cranium/tests/test_reality_chemistry.py` (NEW)

**15+ tests:**
```python
# Water molecule
def test_water_molecule_bond_vibration() -> None:
def test_water_molecule_angle_preservation() -> None:
def test_water_molecule_energy_conservation() -> None:

# Ideal gas
def test_ideal_gas_law_holds() -> None:
def test_ideal_gas_isothermal_expansion() -> None:
def test_ideal_gas_adiabatic_compression() -> None:

# Combustion
def test_combustion_activation_energy() -> None:
def test_combustion_stoichiometry() -> None:
def test_combustion_energy_release() -> None:

# CO2, acid-base, phase transition
def test_co2_linear_geometry() -> None:
def test_acid_base_neutralization() -> None:
def test_water_phase_transition() -> None:
```

---

### Task 5: Add Biology Tests
**File:** `knowledge3d/cranium/tests/test_reality_biology.py` (NEW)

**10+ tests:**
```python
# Cell diffusion
def test_cell_diffusion_equilibrium() -> None:
def test_cell_osmosis_direction() -> None:

# Enzyme kinetics
def test_enzyme_kinetics_rate() -> None:
def test_enzyme_saturation() -> None:

# DNA replication
def test_dna_replication_rate() -> None:
def test_dna_error_rate() -> None:

# Population dynamics
def test_population_growth() -> None:
def test_predator_prey_cycles() -> None:
```

---

### Task 6: Add Materials Tests
**File:** `knowledge3d/cranium/tests/test_reality_materials.py` (NEW)

**8+ tests:**
```python
# Crystal lattice
def test_crystal_thermal_expansion() -> None:
def test_crystal_lattice_constant() -> None:

# Composite material
def test_composite_rule_of_mixtures() -> None:
def test_composite_stress_strain() -> None:

# Phase transition
def test_metal_melting_temperature() -> None:
def test_metal_latent_heat() -> None:
```

---

### Task 7: Update Integration Tests
**File:** `knowledge3d/cranium/tests/test_reality_integration.py` (UPDATE)

**Add multi-discipline test:**
```python
def test_26_systems_full_allocation() -> None:
    """Integration test: 13 physics + 6 chemistry + 4 biology + 3 materials."""
    galaxy = RealityGalaxy()

    # Phase 4A (9 physics)
    systems = [
        export_constant_acceleration_1d(auto_allocate=True),
        # ... (all Phase 4A systems)
    ]

    # Phase 4B (4 E&M)
    systems += [
        export_point_charge_2d(auto_allocate=True),
        # ... (all Phase 4B systems)
    ]

    # Phase 4C Chemistry (6)
    systems += [
        export_water_molecule(auto_allocate=True),
        export_ideal_gas(auto_allocate=True),
        export_combustion(auto_allocate=True),
        export_co2_molecule(auto_allocate=True),
        export_acid_base_reaction(auto_allocate=True),
        export_phase_transition_water(auto_allocate=True),
    ]

    # Phase 4C Biology (4)
    systems += [
        export_simple_cell(auto_allocate=True),
        export_enzyme_kinetics(auto_allocate=True),
        export_dna_replication(auto_allocate=True),
        export_population_dynamics(auto_allocate=True),
    ]

    # Phase 4C Materials (3)
    systems += [
        export_crystal_lattice(auto_allocate=True),
        export_composite_material(auto_allocate=True),
        export_metal_melting(auto_allocate=True),
    ]

    # Add all 26 systems
    for sys in systems:
        galaxy.add_node(sys)

    # Verify all have unique cores
    instance_ids = [galaxy.nodes[sys.node_id].rpn_instance for sys in systems]
    assert len(set(instance_ids)) == 26, "All 26 systems should have unique cores"

    # Step all systems
    for sys in systems:
        state = galaxy.step_system(sys.node_id, n_steps=10)
        assert state is not None

    print(f"\n  26 systems (4 domains) running across {len(set(instance_ids))} cores")
```

---

### Task 8: Create Demo Script
**File:** `scripts/reality_enabler_multidiscipline_demo.py` (NEW)

**Showcase all 26 systems:**
```python
def main() -> None:
    """Run Reality Enabler multi-discipline demonstration."""
    galaxy = RealityGalaxy()

    systems = [
        # Physics (13)
        ("Constant Acceleration 1D", export_constant_acceleration_1d()),
        # ... (all 13 physics)

        # Chemistry (6)
        ("Water Molecule (H2O)", export_water_molecule()),
        ("Ideal Gas (PV=nRT)", export_ideal_gas()),
        ("Combustion (CH4)", export_combustion()),
        ("CO2 Molecule", export_co2_molecule()),
        ("Acid-Base Reaction", export_acid_base_reaction()),
        ("Phase Transition (H2O)", export_phase_transition_water()),

        # Biology (4)
        ("Cell Diffusion", export_simple_cell()),
        ("Enzyme Kinetics", export_enzyme_kinetics()),
        ("DNA Replication", export_dna_replication()),
        ("Population Dynamics", export_population_dynamics()),

        # Materials (3)
        ("Crystal Lattice (Cu)", export_crystal_lattice()),
        ("Composite Material", export_composite_material()),
        ("Metal Melting", export_metal_melting()),
    ]

    for name, system in systems:
        galaxy.add_node(system)
        print(f"✓ {name:<45} T{system.rpn_tier} C{system.rpn_instance:2d}")

    # Step all 26 systems
    start = time.perf_counter()
    for name, system in systems:
        galaxy.step_system(system.node_id, n_steps=10)
    elapsed = time.perf_counter() - start

    print(f"\n26 systems (4 domains) stepped 10× in {elapsed:.3f}s")
    print(f"Throughput: {26*10/elapsed:.1f} steps/sec")
```

---

## Success Criteria (Phase 4C)

| Criterion | Target | Status |
|-----------|--------|--------|
| **Chemistry Systems** | 6 implemented | ⏳ Codex |
| **Biology Systems** | 4 implemented | ⏳ Codex |
| **Materials Systems** | 3 implemented | ⏳ Codex |
| **Tests** | 35+ new tests | ⏳ Codex |
| **All Tests Passing** | 85/85 (50 Phase 5 + 35 Phase 4C) | ⏳ |
| **Demo Script** | 26-system showcase | ⏳ Codex |
| **Completion Report** | PHASE4C_COMPLETE.md | ⏳ Claude |

---

## Timeline

**Estimated:** 3-4 days for complete Phase 4C implementation

**Day 1 (Codex):**
- Implement 6 chemistry export functions
- Add 15 chemistry tests
- Validate chemistry systems pass

**Day 2 (Codex):**
- Implement 4 biology export functions
- Add 10 biology tests
- Validate biology systems pass

**Day 3 (Codex):**
- Implement 3 materials export functions
- Add 8 materials tests
- Validate materials systems pass

**Day 4 (Codex + Claude):**
- Integration tests (26-system test)
- Demo script
- Performance benchmarks
- Completion report (Claude)

---

## Next: Phase 6 UI Integration Prep

**After Phase 4C complete, prep for:**

### Spatial UI Contexts
- **Library:** Browse reality nodes (molecules, cells, materials)
- **Workshop:** Manipulate systems (adjust params, run experiments)
- **Bathtub:** Relax/reflect (view simulation history, insights)

### UI Integration Tasks (Phase 6)
1. glTF export for all 26 systems (visual_rpn)
2. LOD levels (64D centroid → full 2048D mesh)
3. Spatial interaction (pick, move, rotate nodes in 3D)
4. Real-time simulation view (step systems while viewing)
5. Parameter sliders (adjust bond lengths, temperatures, concentrations)

**Codex will need:**
- UI framework (Three.js or Babylon.js for WebGL)
- glTF loader/renderer
- RPN→glTF compiler for visual_rpn
- Real-time communication with RealityGalaxy

---

## Coordination with Claude

**Claude's Role (Phase 4C):**
- Review RPN programs for physics correctness
- Validate conservation laws and ternary logic
- Write completion report after Codex delivers
- Design Phase 6 UI architecture

**Codex's Role (Phase 4C):**
- Implement 13 new export functions (6 chem + 4 bio + 3 materials)
- Write 35+ tests (chemistry, biology, materials)
- Create demo script (26-system showcase)
- Ensure 85/85 tests passing

**Shared Goal:**
- Complete Reality Enabler multi-discipline foundation
- Prepare for Phase 6 UI integration (spatial contexts)
- Enable embodied AGI to reason across physics, chemistry, biology, materials

---

## Ready to Execute

**This briefing is comprehensive and actionable.** Codex has clear system specs, RPN templates, test requirements, and success criteria.

**Daniel's approval to proceed:**

Once you say "go," Codex will:
1. Implement 6 chemistry systems (H2O, ideal gas, combustion, CO2, acid-base, phase transition)
2. Implement 4 biology systems (cell diffusion, enzyme kinetics, DNA, population)
3. Implement 3 materials systems (crystal, composite, melting)
4. Write 35+ tests across 3 new test files
5. Create 26-system demo script
6. Validate 85/85 tests passing

**Timeline:** 3-4 days to complete multi-discipline Reality Enabler foundation

**This transforms K3D from "physics simulator" to "reality simulator" — enabling AGI to reason across all fundamental scientific domains.** 🚀

---

**Prepared by:** Claude (Anthropic Sonnet 4.5)
**For:** Codex (Implementation Lead)
**Date:** November 24, 2025
**Phase:** 4C Multi-Discipline Expansion (Chemistry, Biology, Materials)
**Status:** Ready for implementation
