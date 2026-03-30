# Claude -- Phase E.28: Knowledge Population + Kernel Wiring

**Date:** 2026-03-30
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** CRITICAL -- the mansion is built, now furnish it

---

## Diagnosis

After E.27, the architecture is sound: TRM is always-on, all galaxies are discovered
and bound to GPU at boot, shadow copy learns from every query.

But the system is **knowledge-starved and kernel-idle**:

| Component | Spec Requirement | Actual State | Gap |
|-----------|-----------------|--------------|-----|
| Character Galaxy (L1) | ~200 glyph entries | 1 entry | 99% empty |
| Word Galaxy (L2) | ~15,000 definitions | 3,651 entries | 76% empty |
| Grammar Galaxy (L3) | ~1,000 RPN rules | 301 entries | 70% empty |
| Meta-Rules (L4) | Comprehensive strategies | 4 entries | ~99% empty |
| Reality Galaxy | 5+ physics systems as RPN | 0 enabler entries | 100% empty |
| PTX kernels wired | 62 exist | ~10 called | 83% idle |
| GRE specialists | 15 loaded | 0 dispatched by swarm | 100% idle |

The spec is clear (`FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md` section 1):
> "Programs before opcodes. Prefer to build domain semantics as RPN programs
> over the existing math surface."

And (`RPN_DOMAIN_OPCODE_REGISTRY.md` section 1):
> "Shared math substrate: Physics, chemistry, and biology share the same
> rpn_opcodes.py math core; Reality Enabler composes these into dual-program stars."

The fix is NOT more Python orchestration. It is:
1. **More knowledge entries** in the Galaxy JSONL files (with proper symlinks)
2. **Wire existing kernels** into the swarm worker dispatch

---

## Spec References

- `docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md` -- 4-layer architecture
- `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` -- "programs before opcodes" + opcode promotion pipeline
- `docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md` -- Form + Meaning for humans AND AI
- `docs/vocabulary/HYPER_PARALLEL_PROCESSING.md` -- specialist swarm paradigm
- `docs/vocabulary/SOVEREIGN_NSI_SPECIFICATION.md` -- kernel function contracts

---

## Part 1: Reality Galaxy Population (Physics Systems as Galaxy Entries)

### What Exists (Python, Not Galaxy)

`knowledge3d/cranium/physics_demo.py` has 5 complete physics systems implemented
as Python classes. Each has `law_rpn`, `behavior_rpn`, and `visual_rpn` programs.
These are EXACTLY what should be Galaxy entries -- but they exist only as dead Python.

From `RPN_DOMAIN_OPCODE_REGISTRY.md` section 2.1:
> "ConstantAcceleration1D, HarmonicOscillator1D, Orbital2D, Heat1D, Heat2D"
> "All reuse the shared RPN math surface; they are examples of how to encode
> ODEs and simple PDEs without adding new opcodes."

### What to Create

A population script (`scripts/populate_reality_systems.py`) that writes these 5
systems + their component atoms as Galaxy JSONL entries to `Reality.jsonl`.

Each physics system becomes a **reality_system** entry following the Reality Enabler
architecture (from `FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md`):

```jsonl
{"id": "reality_system_constant_acceleration_1d", "galaxy": "Reality", "category": "reality_system", "layer": 3, "content": "1D constant acceleration: v(t+1) = v(t) + a*dt, x(t+1) = x(t) + v(t+1)*dt", "metadata": {"domain": "physics_kinematics", "component_refs": ["reality_atom_position_1d", "reality_atom_velocity_1d", "reality_atom_acceleration_1d"], "behavior_rpn": "v RECALL a RECALL dt RECALL MUL ADD v STORE v RECALL dt RECALL MUL x RECALL ADD x STORE", "law_rpn": "v RECALL a RECALL dt RECALL MUL ADD v_expected RECALL SUB ABS tolerance RECALL LTE", "visual_rpn": "x RECALL 0 DRAW_MOVE x RECALL 1 ADD 0 DRAW_LINE DRAW_STROKE", "surface_forms": {"en": "constant acceleration", "pt": "aceleracao constante"}, "reusable_contexts": ["physics_sim", "kinematics", "projectile_motion", "free_fall"]}}
```

And its component atoms:

```jsonl
{"id": "reality_atom_position_1d", "galaxy": "Reality", "category": "reality_atom", "layer": 1, "content": "1D position coordinate (scalar, meters)", "metadata": {"domain": "physics_kinematics", "symbol": "x", "unit": "m", "behavior_rpn": "x RECALL", "visual_rpn": "x RECALL 0 DRAW_MOVE x RECALL 1 ADD 0 DRAW_LINE DRAW_STROKE", "surface_forms": {"en": "position", "pt": "posicao"}}}
{"id": "reality_atom_velocity_1d", "galaxy": "Reality", "category": "reality_atom", "layer": 1, "content": "1D velocity (scalar, m/s)", "metadata": {"domain": "physics_kinematics", "symbol": "v", "unit": "m/s", "behavior_rpn": "x RECALL x_prev RECALL SUB dt RECALL DIV", "surface_forms": {"en": "velocity", "pt": "velocidade"}}}
{"id": "reality_atom_acceleration_1d", "galaxy": "Reality", "category": "reality_atom", "layer": 1, "content": "1D acceleration (scalar, m/s^2)", "metadata": {"domain": "physics_kinematics", "symbol": "a", "unit": "m/s^2", "behavior_rpn": "v RECALL v_prev RECALL SUB dt RECALL DIV", "surface_forms": {"en": "acceleration", "pt": "aceleracao"}}}
```

### 5 Systems to Populate

| System | RPN Program (behavior_rpn) | Atoms |
|--------|---------------------------|-------|
| ConstantAcceleration1D | `v a dt MUL ADD v STORE v dt MUL x ADD x STORE` | position, velocity, acceleration |
| HarmonicOscillator1D | `omega_sq RECALL x RECALL MUL NEG a STORE a dt MUL v ADD v STORE v dt MUL x ADD x STORE` | position, velocity, angular_frequency |
| Orbital2D | `mu RECALL r_mag RECALL 3 POW DIV NEG (per component)` | position_2d, velocity_2d, mass, gravitational_param |
| Heat1D | `alpha RECALL T_left T_center 2 MUL SUB T_right ADD MUL dx_sq DIV dt MUL T_center ADD` | temperature, thermal_diffusivity, grid_spacing |
| Heat2D | Same pattern with 5-point stencil | temperature_2d, thermal_diffusivity, grid_spacing_x, grid_spacing_y |

**Symlink pattern:** Each system's `component_refs` points to its atoms by ID.
Atoms are shared: `reality_atom_position_1d` is used by ConstantAcceleration1D,
HarmonicOscillator1D, and Heat1D. This is the symlink architecture from the spec.

**Expected: ~25 new entries** (5 systems + ~20 atoms).

---

## Part 2: Grammar Galaxy Population (Transformation Rules)

### Current State

Grammar.jsonl has 301 entries. Most are ARC transform rules. The spec says 1,000+.

Missing categories (from `FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md` section 1.4):
- **Math transformations**: power rule, chain rule, product rule, quotient rule,
  integration by parts, trig identities, algebraic simplification
- **Logical rules**: modus ponens, modus tollens, syllogism, contrapositive,
  De Morgan's laws, distributive law
- **Unit conversions**: SI ↔ Imperial, temperature scales, time units

### What to Create

A population script (`scripts/populate_grammar_rules.py`) that writes foundational
transformation rules as Galaxy JSONL entries with RPN programs.

**Math transformation rules (Layer 3):**

```jsonl
{"id": "grammar_math_power_rule", "galaxy": "Grammar", "category": "math_transform", "layer": 3, "content": "Power rule: d/dx(x^n) = n*x^(n-1)", "metadata": {"domain": "calculus_differentiation", "rpn_program": "n RECALL x RECALL n RECALL 1 SUB POW MUL", "symbol_refs": ["math_symbol_partial_derivative"], "word_refs": ["word_derivative", "word_exponent"], "surface_forms": {"en": "power rule", "pt": "regra da potencia"}, "examples": [{"input": "x^3", "output": "3*x^2"}], "rule_strength": 1}}
{"id": "grammar_math_chain_rule", "galaxy": "Grammar", "category": "math_transform", "layer": 3, "content": "Chain rule: d/dx(f(g(x))) = f'(g(x)) * g'(x)", "metadata": {"domain": "calculus_differentiation", "rpn_program": "g_x RECALL f_prime RECALL g_x RECALL APPLY g_prime RECALL x RECALL APPLY MUL", "symbol_refs": ["math_symbol_partial_derivative"], "word_refs": ["word_derivative", "word_composition"], "surface_forms": {"en": "chain rule", "pt": "regra da cadeia"}, "rule_strength": 1}}
{"id": "grammar_math_product_rule", "galaxy": "Grammar", "category": "math_transform", "layer": 3, "content": "Product rule: d/dx(f*g) = f'*g + f*g'", "metadata": {"domain": "calculus_differentiation", "rpn_program": "f_prime RECALL g RECALL MUL f RECALL g_prime RECALL MUL ADD", "symbol_refs": ["math_symbol_partial_derivative"], "word_refs": ["word_derivative", "word_product"], "surface_forms": {"en": "product rule", "pt": "regra do produto"}, "rule_strength": 1}}
```

**Logical rules (Layer 3):**

```jsonl
{"id": "grammar_logic_modus_ponens", "galaxy": "Grammar", "category": "logic_rule", "layer": 3, "content": "Modus ponens: if P then Q; P; therefore Q", "metadata": {"domain": "formal_logic", "rpn_program": "P RECALL P_implies_Q RECALL AND Q STORE", "surface_forms": {"en": "modus ponens", "pt": "modus ponens"}, "rule_strength": 1}}
{"id": "grammar_logic_modus_tollens", "galaxy": "Grammar", "category": "logic_rule", "layer": 3, "content": "Modus tollens: if P then Q; not Q; therefore not P", "metadata": {"domain": "formal_logic", "rpn_program": "Q RECALL NOT P_implies_Q RECALL AND P RECALL NOT STORE", "surface_forms": {"en": "modus tollens", "pt": "modus tollens"}, "rule_strength": 1}}
{"id": "grammar_logic_demorgan_and", "galaxy": "Grammar", "category": "logic_rule", "layer": 3, "content": "De Morgan: NOT(A AND B) = (NOT A) OR (NOT B)", "metadata": {"domain": "formal_logic", "rpn_program": "A RECALL NOT B RECALL NOT OR", "surface_forms": {"en": "De Morgan's law (AND)", "pt": "lei de De Morgan (E)"}, "rule_strength": 1}}
```

**Unit conversion rules (Layer 3):**

```jsonl
{"id": "grammar_unit_meters_to_feet", "galaxy": "Grammar", "category": "unit_conversion", "layer": 3, "content": "meters to feet: 1 m = 3.28084 ft", "metadata": {"domain": "measurement_conversion", "rpn_program": "value RECALL 3.28084 MUL", "surface_forms": {"en": "meters to feet", "pt": "metros para pes"}, "rule_strength": 1}}
{"id": "grammar_unit_celsius_to_fahrenheit", "galaxy": "Grammar", "category": "unit_conversion", "layer": 3, "content": "Celsius to Fahrenheit: F = C * 9/5 + 32", "metadata": {"domain": "measurement_conversion", "rpn_program": "value RECALL 9 MUL 5 DIV 32 ADD", "surface_forms": {"en": "Celsius to Fahrenheit", "pt": "Celsius para Fahrenheit"}, "rule_strength": 1}}
```

**Target: ~100 new Grammar entries** across math (30), logic (20), unit conversions (20),
algebraic identities (15), trigonometric identities (15).

---

## Part 3: Swarm Specialist Kernel Wiring

### Current State

From `HYPER_PARALLEL_PROCESSING.md`:
> "N RPN cores x specialist weights x cross-referenceable stacks = hyper-parallel processing"

15 GRE specialist kernels are loaded into VRAM. Zero are dispatched by the swarm.
All 9 swarm workers execute identical Python logic.

### What the Spec Says

From `SOVEREIGN_NSI_SPECIFICATION.md` section 9 (kernel function contract map):
each specialist kernel has a defined input/output contract. The swarm should
dispatch domain-appropriate kernels per worker.

### What to Wire

The nine-chain swarm workers should route to GRE specialist kernels based on
the query domain. This is NOT new Python orchestration -- it's wiring existing
GPU infrastructure.

**Mapping (domain_hint -> specialist kernel):**

| Query Domain | GRE Kernel | What It Does |
|-------------|-----------|--------------|
| math, calculus | `gre_symbolic_math` | Symbolic manipulation on GPU |
| physics, kinematics | `gre_world_model` | Physics simulation step |
| visual, arc, drawing | `gre_shape_generator` | Shape analysis/generation |
| logic, reasoning | `gre_defeasible_resolver` | Defeasible logic resolution |
| language, grammar | `gre_grammar_transform` | Grammar rule application |
| spatial, navigation | `gre_spatial_reasoning` | Spatial relationship computation |
| temporal, sequence | `gre_temporal_reasoning` | Time-series pattern matching |
| clustering, similarity | `gre_cluster_analysis` | Similarity-based grouping |
| general, factual | `gre_multimodal_halting_gate` | General convergence check |

**Implementation approach:**

In the nine-chain swarm dispatch, each worker receives a `specialist_hint` from
the TRM galaxy distribution. Instead of all workers running the same Python path,
each worker calls `sovereign_bridges.execute_kernel(kernel_name, input_buffer)`
where `kernel_name` is determined by the specialist mapping.

The bridge infrastructure already exists (`knowledge3d/cranium/sovereign_bridges.py`
has 24 bridge classes). The gap is: nobody calls them during query processing.

**This is a wiring task, not a development task.** The kernels exist. The bridges
exist. The swarm exists. They just aren't connected.

---

## Part 4: Character Galaxy Seeding (Math Symbols as Procedural RPN)

### What the Spec Says

From `FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md` section 1.2:
> "152 Math Symbols stored as procedural RPN in Math Galaxy"
> Uses `scripts/extract_math_symbol_glyphs.py` for Bezier extraction -> RPN conversion

### Minimum Viable Set

Start with the 17 high-priority math symbols + 10 digits + basic operators:

```jsonl
{"id": "char_math_summation", "galaxy": "Character", "category": "math_symbol", "layer": 1, "content": "Summation symbol U+2211", "metadata": {"unicode": "U+2211", "glyph": "\u2211", "visual_rpn": "32 8 MOVE 8 32 LINE 32 56 LINE 56 56 LINE STROKE", "surface_forms": {"en": "summation", "pt": "somatorio"}, "related_words": ["word_summation", "word_series"]}}
{"id": "char_math_integral", "galaxy": "Character", "category": "math_symbol", "layer": 1, "content": "Integral symbol U+222B", "metadata": {"unicode": "U+222B", "glyph": "\u222b", "visual_rpn": "40 8 MOVE 32 16 CURVE 32 48 CURVE 24 56 CURVE STROKE", "surface_forms": {"en": "integral", "pt": "integral"}, "related_words": ["word_integration", "word_antiderivative"]}}
{"id": "char_math_partial", "galaxy": "Character", "category": "math_symbol", "layer": 1, "content": "Partial derivative symbol U+2202", "metadata": {"unicode": "U+2202", "glyph": "\u2202", "visual_rpn": "40 16 MOVE 24 16 CURVE 16 32 CURVE 24 48 CURVE 40 48 LINE 40 16 LINE STROKE", "surface_forms": {"en": "partial derivative", "pt": "derivada parcial"}, "related_words": ["word_derivative", "word_partial"]}}
```

**Target: ~35 entries** (17 high-priority symbols + 10 digits + 8 basic operators).

---

## Execution Sequence

1. **`scripts/populate_reality_systems.py`** -- write 5 physics systems + ~20 atoms
   to `Reality.jsonl` (Part 1)
2. **`scripts/populate_grammar_rules.py`** -- write ~100 transformation rules to
   `Grammar.jsonl` (Part 2)
3. **`scripts/populate_math_symbols.py`** -- write ~35 character entries to
   `Character.jsonl` (Part 4)
4. **Wire GRE kernels into swarm dispatch** -- connect domain_hint -> specialist
   kernel in nine-chain worker (Part 3)
5. **Run local ARC3 + full benchmark** -- validate knowledge is discoverable and
   scoring improves

---

## What NOT to Do

- Do NOT add Python orchestration to knowledgeverse.py
- Do NOT add new opcodes (programs before opcodes -- use existing math surface)
- Do NOT add framework dependencies
- Do NOT change the query path structure
- Population scripts are **ingestion-path** (run once, write JSONL, done)
- Knowledge lives in House (JSONL), loads into Galaxy at boot

---

## Success Criteria

- [ ] Reality.jsonl grows by ~25 entries (5 systems + 20 atoms with behavior_rpn)
- [ ] Grammar.jsonl grows by ~100 entries (math, logic, unit conversion rules)
- [ ] Character.jsonl grows by ~35 entries (math symbols with visual_rpn)
- [ ] All new entries use symlink pattern (component_refs, symbol_refs, word_refs)
- [ ] At least 3 GRE specialist kernels dispatched by swarm workers
- [ ] Local ARC3 benchmark still works (no regression)
- [ ] Population scripts are standalone (run once, write JSONL, no runtime dependency)

---

## Architecture Note: Why Knowledge > Code

The system's benchmark scores are not limited by the reasoning pipeline -- the composed
head, TRM, swarm, and GPU infrastructure are proven. They are limited by the Galaxy
being nearly empty.

When the TRM navigates to the Grammar Galaxy looking for a calculus rule, it finds
nothing. When the swarm dispatches a physics specialist, the specialist has no
reality_system entries to work with. When a query needs unit conversion, there are
no conversion rules to apply.

The fix is not smarter Python. It is more knowledge in the Galaxy -- properly
structured, properly symlinked, with real RPN programs that the existing GPU
infrastructure can execute.

Daniel said it: "architecture is more than proven -- path forward is knowledge curation
and proceduralization."
