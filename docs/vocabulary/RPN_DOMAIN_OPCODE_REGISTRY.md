# RPN Domain Opcode Registry — Physics, Chemistry, Biology (Initial Sketch)

**Version**: 0.1 (Draft, non-normative)  
**Status**: Design sketch (grounded in existing opcodes)  
**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Implementation)  
**Date**: November 2025

---

## Abstract

This document sketches a **minimal, domain-oriented RPN opcode registry** for Reality Enabler domains (physics, chemistry, biology), grounded in the existing `rpn_opcodes.py` surface. It does not introduce new opcodes yet; instead, it:

- Groups existing opcodes into domain-specific roles.
- Identifies where physics/chemistry/biology semantics are constructed via **programs**, not new primitives.
- Provides a roadmap for future opcode extensions once the math core spec and Reality Enabler are stable.

---

## 1. Philosophy

1. **Programs before opcodes**: Prefer to build domain semantics as RPN programs over the existing math surface instead of immediately adding domain-specific opcodes.
2. **Shared math substrate**: Physics, chemistry, and biology share the same `rpn_opcodes.py` math core; Reality Enabler composes these into dual-program stars.
3. **Tiered execution via math cores**: Simple/mid/high math cores (see `MATH_CORE_SPECIFICATION.md`) decide where and how programs run, not which math is available.

---

## 2. Physics-Oriented Opcodes (Using Existing Surface)

Without introducing new opcodes, physics primitives can be described via:

- **Vector & matrix ops**:
  - `OP_VEC_L2_NORM`, `OP_VEC_NORMALIZE`, `OP_VEC_BLEND`
  - `OP_DOT_PRODUCT`, `OP_CROSS_PRODUCT`, `OP_OUTER_PRODUCT`
  - `OP_MATVEC_F32`, `OP_MATMUL_SMALL`

- **Calculus / field operators**:
  - `OP_DIVERGENCE`, `OP_CURL`, `OP_LAPLACIAN`

- **Temporal reasoning**:
  - `OP_TEMPORAL_COHERENCE`, `OP_TEMPORAL_MASK`, `OP_TEMPORAL_AGGREGATE`

- **Programmability**:
  - `OP_BRANCH`, `OP_LOOP`, `OP_STORE`, `OP_RECALL`, `OP_LIMIT`, `OP_SERIES_SUM`, `OP_SERIES_PRODUCT`.

**Example program roles** (to be stored in `behavior_rpn`):

- Kinematics update:
  - Use addition/multiplication, vector ops, and temporal aggregation to implement `x_{t+1} = x_t + v_t·Δt`, `v_{t+1} = v_t + a_t·Δt`.

- Force accumulation:
  - Use dot/cross/outer products and reductions to sum forces from neighbors and compute accelerations.

- Field evaluation:
  - Use `OP_DIVERGENCE`, `OP_CURL`, `OP_LAPLACIAN` over grid‑encoded states to approximate PDE operators (e.g., Navier–Stokes, Poisson).

In Reality Enabler terms, a **physics galaxy** star’s `behavior_rpn` is a composition of these existing ops, scheduled across simple/mid/high math cores depending on complexity.

### 2.1 Current Physics Implementations (Reference)

The following small systems in `knowledge3d/cranium/physics_demo.py` and `physics_galaxy.py` use this surface today:

- **ConstantAcceleration1D**:  
  - Law: `v_{t+1} = v_t + a·dt`, `x_{t+1} = x_t + v_{t+1}·dt`.  
  - RPN usage: scalar `+`/`*` via `ModularRPNEngine` (Tier‑1/2).

- **HarmonicOscillator1D**:  
  - Law: `x'' + ω²x = 0` rewritten as `v' = -ω²x`, `x' = v`.  
  - RPN usage: integration steps in RPN; state‑dependent acceleration computed in host, ready to be inlined into RPN if needed.

- **Orbital2D**:  
  - Law: `a = -μ r / |r|³` for central gravitational force in 2D.  
  - RPN usage: per‑component integration for `(vx, vy, x, y)`; radius/energy checks in tests demonstrate physical plausibility.

- **Heat1D**:  
  - Law: 1D heat diffusion `T_i^{n+1} = T_i^n + α·dt/dx²·(T_{i+1}^n − 2T_i^n + T_{i−1}^n)`.  
  - RPN usage: integration step delegated to RPN; stencil computed in host code, with future work to move more of the stencil math into pure RPN if desired.

- **Heat2D**:  
   - Law: 2D heat diffusion with 5-point stencil over a rectangular grid.  
   - RPN usage: similar pattern to Heat1D; the 2D Laplacian is assembled host-side, and the integration step `T_{i,j}^{n+1} = T_{i,j}^n + dT_{i,j}` is executed via scalar RPN programs per cell.

All of these reuse the shared RPN math surface; they are examples of how to encode ODEs and simple PDEs without adding new opcodes.

---

## 3. Chemistry-Oriented Opcodes (Via Math + Graph Programs)

Chemistry semantics can be expressed through:

- **Graph-like operations using existing math**:
  - Use `OP_SET_UNION`, `OP_SET_INTERSECTION`, `OP_SET_DIFFERENCE`, `OP_SET_CARTESIAN` for combinatorial aspects (bond possibilities, reaction templates).
  - Use `OP_CLUSTER_ASSIGN`, `OP_COSINE_SIM_BATCH` to cluster or compare molecular embeddings.

- **Linear algebra + nonlinearity**:
  - Property prediction and simple reaction scoring can be implemented as small MLP programs using `OP_MATVEC_F32`, `OP_VECTOR_RELU`, `OP_VECTOR_SIGMOID`.

- **Quantum-ish helpers**:
  - `OP_QUANTUM_SUPERPOSE`, `OP_QUANTUM_MEASURE`, `OP_QUANTUM_PHASE` and related ops can be used to approximate orbital-style reasoning when needed (without claiming physical accuracy).

**Role in Reality Enabler**:

- Atoms and functional groups:
  - `behavior_rpn` encodes valence and allowed bond patterns using set and comparison ops.

- Molecules and materials:
  - `behavior_rpn` uses vector/matrix ops and clustering to predict coarse properties (polarity, stability, simple reactivity classes).

Again, no new opcodes are needed for a first pass; chemistry semantics are programs on existing math, executed via math cores.

---

## 4. Biology-Oriented Opcodes (Growth, Networks, Evolution)

Biology relies on:

- **Fractal and geometric kernels** (already implemented via bridges):
  - `FractalEmitter` and drawing opcodes allow L‑system / fractal growth representation at the visual level.

- **Graph and clustering ops**:
  - `OP_CLUSTER_ASSIGN`, `OP_VEC_BLEND`, `OP_SET_*` ops to build and update connectivity graphs (e.g., neural networks, signaling pathways).

- **Temporal reasoning**:
  - `OP_TEMPORAL_COHERENCE`, `OP_TEMPORAL_AGGREGATE` to model stability and temporal integration.

- **Programmability opcodes**:
  - `OP_BRANCH`, `OP_LOOP`, `OP_STORE`, `OP_RECALL` for cell‑automaton rules and multi-step growth programs.

These tools are sufficient to encode:

- Growth rules (L‑systems + loops + branch conditions).
- Simple cellular automata for consolidation/pruning.
- Network formation and strengthening/weakening over time.

---

## 5. Capability Classification (A/B/C)

Every multimodal or domain feature proposal MUST be classified into one of three capability classes before implementation begins:

### Class A: Executable Now
- Can be expressed with current opcodes and bridges.
- Examples: drawing primitives, STORE/RECALL-driven temporal state, vector/matrix transforms, ternary routing, simple procedural animation.
- **Action**: Compose Galaxy recipes immediately. No new opcodes needed.

### Class B: Representable Now, Kernel Later
- Can be encoded immediately as Galaxy recipes or Grammar macros using existing opcodes.
- Deserves a dedicated opcode only after usage frequency and performance justify it.
- Examples: L-system expansion, spectrogram pipelines, triplanar mapping, mesh extrusion from 2D contours, boid update rules.
- **Action**: Encode as recipes/macros first. Monitor usage. Promote to opcode only when justified (see Section 6).

### Class C: Research / Not Yet Admitted
- Requires new data structures, too many host-side assumptions, or unproven performance.
- Examples: full volumetric remeshing pipelines, robust production-grade cloth/fluids, heavy differentiable rendering loops.
- **Action**: Document as research target. Do not encode or plan implementation until prerequisites are resolved.

**Rule**: Every new multimodal proposal must state which class it belongs to.

---

## 6. Opcode Admission and Promotion Pipeline

The principle from Section 1 ("programs before opcodes") is formalized here as a four-stage pipeline:

### Stage 0: Galaxy Recipe
- Store the technique as Grammar/Reality/Drawing/Audio knowledge only.
- No new opcode. Prove semantic usefulness first.

### Stage 1: Macro Surface
- Add a stable macro or token expansion in the high-level compiler (`ModularRPNEngine`).
- Examples: `STORE_X`, `RECALL_DISC`, future `SPECTROGRAM_MACRO`.
- Still no new kernel.

### Stage 2: Opcode Candidate
- Measure repeated usage across Galaxy recipes and tool-nodes (see `KNOWLEDGEVERSE_SPECIFICATION.md`).
- Profile cost of recipe/macro execution.
- Show that a dedicated opcode reduces complexity materially.

### Stage 3: PTX Kernel Admission

**Pre-reservation prerequisite (Ruling 5, 2026-04-18)**: before a Stage 3 admission writes an opcode number into the registry, the lane MUST verify its reservation exists in §11 (Reserved Future Blocks). If no reservation covers the target number, the admission is blocked until the Reservation Table is amended. See `TEMP/CLAUDE_CODEX_OPCODE_RANGE_RESERVATION_DOCTRINE_04.18.2026.md` for the full workflow.

- Add kernel only after ALL of the following are satisfied:

| Criterion | Requirement |
|-----------|-------------|
| **Recipe exists** | Already expressed as composition of existing ops |
| **Frequency is high** | Appears in many tool-nodes or object recipes |
| **Speedup is meaningful** | Not 5%, but enough to matter architecturally |
| **Semantics are stable** | Same meaning across use cases |
| **Graph complexity reduced** | Fewer references, cleaner composition, easier TRM routing |
| **Test coverage** | Sovereignty tests + functional tests |
| **Sovereignty review** | Pure PTX, no CPU fallbacks, no external dependencies |

### Budget Control Recipes (Class B — Stage 0-1)

The **Adaptive Reasoning Budget** (see [ADAPTIVE_REASONING_BUDGET_SPECIFICATION.md](ADAPTIVE_REASONING_BUDGET_SPECIFICATION.md) §A.2) is expressed entirely as Class B recipes using existing opcodes — no new kernel admission required:

- **Budget state** stored in STORE/RECALL registers 60-67 (budget_total, budget_remaining, budget_min, ternary_signal, decomposition_depth, sub_task_count, priority, watermark_level)
- **Budget computation**: `OP_PUSH B_base → OP_PUSH 2 → OP_PUSH 1 → OP_RECALL signal → OP_SUB → OP_POW → OP_MUL` = B(q) = B_base × 2^(1−σ)
- **Budget enforcement**: `OP_RECALL budget_remaining → OP_PUSH 0 → OP_CMP → OP_BRANCH halt_if_zero`
- **Sub-task decomposition**: `OP_RECALL depth → OP_PUSH D_max → OP_CMP → OP_BRANCH decompose_or_serialize`

These recipes demonstrate the "programs before opcodes" principle: budget governance composes from existing RPN primitives. If profiling shows budget computation as a bottleneck (unlikely — budget logic runs once per TRM tick, not per-element), promotion to Stage 2 is warranted.

### Multimodal Target Opcodes (Stage 2 Candidates)

The following are valid design targets currently at Stage 0-1. They MUST pass the promotion pipeline before receiving PTX kernels:

**3D Generation Domain** (Stage 0):
- `OP_MESH_EXTRUDE`, `OP_BOOLEAN_UNION`, `OP_BOOLEAN_SUBTRACT`, `OP_BOOLEAN_INTERSECT`
- `OP_LSYSTEM_STEP`, `OP_MARCHING_CUBES`, `OP_NOISE_3D`

**Signal Domain** (Stage 0):
- `OP_FFT_FORWARD`, `OP_FFT_INVERSE`
- `OP_AUDIO_TO_SPECTROGRAM`, `OP_SPECTROGRAM_TO_AUDIO`
- `OP_FREQUENCY_FILTER`

**Image Generation Domain** (Stage 0):
- `OP_GRADIENT_BLEND`, `OP_TEXTURE_SAMPLE`, `OP_CONVOLUTION_2D`

**Temporal Domain** (Stage 0):
- `OP_TEMPORAL_SEQUENCE`, `OP_TEMPORAL_INTERPOLATE`, `OP_PHYSICS_INTEGRATE`

---

## 7. Extended Tier — Reasoning-Paradigm Block (0xA0–0xF1)

**Authority**: `TEMP/CLAUDE_REASONING_PARADIGMS_AND_N_SWARM_SPEC_2026-04-13.md` §4.
These opcodes are reserved for the first reasoning-paradigm wave. Reservation
does not bypass the admission pipeline: each opcode still lands as a PTX
implementation only when its Batch slice admits it.

**Ternary packing convention**: every value crossing lane/kernel boundaries is
encoded as a ternary-ready triple `(value, confidence, polarity)`. Current binary
hardware packs this as explicit fields; future ternary hardware maps the trit
lane natively. Stack diagrams below use bare symbolic names for readability,
but each operand is ternary-packed at transport boundaries.

| Opcode | Mnemonic | Family | Stack diagram | Semantics | Ternary packing |
|---|---:|---|---|---|---|
| `0xA0` | `ABDUCE` | Abductive | `[obs] -> [hyp_id]` | Retrieve a hypothesis candidate for an observation from the Hypothesis Galaxy. | `obs` and `hyp_id` carry confidence; polarity is observation support. |
| `0xA1` | `EXPLAIN` | Abductive | `[hyp_id, obs] -> [cover_mask]` | Verify `effects(hyp_id) ⊇ obs` using `TCOMP` over a ternary effect mask. | `cover_mask` packs covered `+1`, unknown `0`, contradicted `-1`. |
| `0xA2` | `SUSPECT` | Abductive | `[hyp_id] -> [simplicity_score]` | Rank by simplicity via `TQUANT` term-size and symbol-rarity score. | Score confidence mirrors hypothesis support; polarity marks plausible/implausible. |
| `0xA3` | `ABDUCE_HALT` | Abductive | `[score] -> [halt_bit]` | Push abductive result to the halting gate if score exceeds threshold. | Halt bit is packed as `+1` accept, `0` continue, `-1` reject. |
| `0xA4` | `SCUNION` | Abductive | `[mask_a, mask_b] -> [mask_union]` | Warp-popcount greedy set-cover union. | Masks are ternary coverage masks. |
| `0xA5` | `ICHECK` | Abductive | `[hyp_id, ic_mask] -> [valid]` | Integrity-constraint validation via ternary AND. | `valid` uses `+1` valid, `0` unknown, `-1` violation. |
| `0xA6` | `ABDRES` | Abductive | `[goal, assumptions] -> [resolvent]` | Abductive resolution: unify goal and collect required assumptions. | Assumption support stored as ternary confidence/polarity. |
| `0xA7` | `ABDNEG` | Abductive | `[goal] -> [naf_result]` | Negation-as-failure under finite failure. | `+1` proven finite failure, `0` unknown, `-1` contradicted. |
| `0xB0` | `EBELIEF` | Subjective/frame | `[evidence] -> [belief]` | Evidence-to-belief transform by Bayes inversion over `TQUANT`. | Belief is a ternary opinion triple. |
| `0xB1` | `BIDUCE` | Subjective/frame | `[pre, post] -> [frame]` | Bi-abductive frame inference. | Frame confidence tracks pre/post support. |
| `0xB2` | `FRAME` | Subjective/frame | `[heap_a, heap_b] -> [frame_delta]` | Separation-logic frame extraction. | Delta polarity marks add/remove/unknown. |
| `0xB3` | `EULER_COMPLETE` | Subjective/frame | `[chain_root] -> [closure]` | Transitive property-chain closure for Euler/N3-style reasoning. | Closure entries carry ternary entailment strength. |
| `0xB4` | `DL_SATURATE` | Subjective/frame | `[node] -> [saturated_node]` | Description-logic tableau saturation. | Clash/no-clash encoded as ternary branch state. |
| `0xB5` | `BLOCKING_CHECK` | Subjective/frame | `[node, predecessor] -> [blocked]` | Binary blocking over node × predecessor. | `blocked` is `+1` blocked, `0` unknown, `-1` open. |
| `0xB6` | `CTX_SWITCH` | Subjective/frame | `[ctx_id] -> []` | Set lane-local Galaxy read filter to `star.context_id == ctx_id || star.context_id == 0`. | `ctx_id` is a u32 value with neutral polarity. |
| `0xB7` | `ALPCHAIN` | Subjective/frame | `[goal] -> [subgoals]` | Abductive-logic-programming backward-chain step. | Subgoal masks retain ternary proof status. |
| `0xC0` | `TUNIFY` | Deductive/ATP | `[term_a, term_b] -> [subst_handle]` | Robinson unification with occur-check. | `subst_handle=0` means failure; confidence marks substitution completeness. |
| `0xC1` | `TRESOLVE` | Deductive/ATP | `[clause_a, clause_b] -> [resolvent]` | Binary resolution to resolvent clause. | Resolvent polarity marks derived support or contradiction. |
| `0xC2` | `TORDER` | Deductive/ATP | `[term_a, term_b] -> [order]` | KBO/LPO term ordering. | `order` uses ternary less/equal/greater mapping. |
| `0xC3` | `TSUBSUME` | Deductive/ATP | `[clause_a, clause_b] -> [subsumed]` | Forward/backward clause subsumption. | `subsumed` uses `+1` yes, `0` unknown, `-1` no. |
| `0xC4` | `TSUPERPOS` | Deductive/ATP | `[rule, term] -> [critical_pair]` | Superposition critical-pair generation. | Pair carries equation confidence and polarity. |
| `0xC5` | `TREWRITE` | Deductive/ATP | `[term, rule_set] -> [rewritten_term]` | Ordered rewriting from indexed rule set. | Rewritten term keeps rule confidence. |
| `0xD0` | `TSPLIT` | Tableaux/SAT | `[alpha_formula, beta_formula] -> [branch_id_alpha, branch_id_beta]` | Alpha/beta branch split. | Branch state records ternary open/closed/unknown. |
| `0xD1` | `TCLOSE` | Tableaux/SAT | `[branch_id, lit_a, lit_b] -> [closed]` | Branch closure via complementarity. | `closed` uses `+1` closed, `0` unknown, `-1` still open. |
| `0xD2` | `TEXPAND` | Tableaux/SAT | `[branch_id, formula] -> [expanded_formula]` | Gamma/delta quantifier expansion with on-GPU Skolemisation. | Expansion confidence follows source formula. |
| `0xD3` | `TBCP` | Tableaux/SAT | `[watch_state] -> [propagated]` | Boolean constraint propagation with two-watch representation. | Propagation result is ternary satisfiability state. |
| `0xD4` | `TLEARNT` | Tableaux/SAT | `[conflict] -> [learnt_clause]` | Conflict-clause learning and minimisation. | Learnt clause polarity marks conflict evidence. |
| `0xE0` | `RETE_ALPHA_TEST` | Rete | `[fact, alpha_node] -> [match]` | Rete alpha-memory test. | Match is ternary true/unknown/false. |
| `0xE1` | `RETE_BETA_JOIN` | Rete | `[left_token, right_token] -> [joined_token]` | Rete beta-memory join. | Joined token confidence is the ternary conjunction of parents. |
| `0xE2` | `AGENDA_INSERT` | Rete | `[activation] -> [agenda_handle]` | Insert activation into agenda. | Agenda priority carries ternary support. |
| `0xF0` | `HALT_SET` | System | `[local_halt] -> []` | Write lane-local consensus register view. | Halt trit uses `+1` halt, `0` continue, `-1` reject. |
| `0xF1` | `HALT_SYNC` | System | `[] -> [global_halt]` | Global barrier and cross-lane halt check. | Global halt trit is reduced across active lanes. |

### 7.1 Batch 4 CBR Extension Block (0x100–0x103)

Batch 4 keeps **Cyc/context**, **casuistry**, and **model checking** on the
reuse-existing-surface path:

- `CTX_SWITCH` remains the public context gate and now admits bounded
  heuristic filtering over `context_id`, `ethical_trit`, and rule salience.
- Casuistry is implemented as an ethical projection over case-based reasoning,
  not as a separate opcode family.
- Model checking remains a bounded kernel/bridge surface reusing existing
  graph/state infrastructure, so no new public opcode is reserved for it in
  Batch 4.

The only new public opcode reservations in Batch 4 are the bounded
case-based-reasoning primitives below.

| Opcode | Mnemonic | Family | Stack diagram | Semantics | Ternary packing |
|---|---:|---|---|---|---|
| `0x100` | `CASE_FETCH` | CBR | `[query, case_a, case_b] -> [case_handle]` | Select the nearest valid case from a bounded case window using existing similarity/state surfaces. | Returned handle carries case id, context, ethical code, and bounded confidence. |
| `0x101` | `CASE_REBIND` | CBR | `[case_handle, rebind_spec] -> [bound_case]` | Clone/rebind a stored case program or solution handle to current symbols/facts. | Bound case preserves source support and rewrites value/context fields only. |
| `0x102` | `CASE_REVISE` | CBR/Casuistry | `[bound_case, revise_constraint] -> [revised_case]` | Run bounded revise logic and ethical gating; returns `0` on failure. | Ethical polarity is preserved and may be vetoed by stage-3 style gate semantics. |
| `0x103` | `CASE_RETAIN_HINT` | CBR | `[revised_case] -> [retain_hint]` | Emit a bounded shadow-copy retention hint for sleep-time materialisation. | Hint carries confidence/polarity but does not write permanent state in hot path. |
| `0x104` | `LORA_LOAD_BASE` | Procedural Adapter | `[] -> [base_weights_ref]` | Push the referenced base-weight star for the active specialist adapter. | Residency/support of the base star determines confidence. |
| `0x105` | `LORA_LOW_RANK_ADD` | Procedural Adapter | `[left_factor, right_factor] -> [delta]` | Materialize a low-rank adapter delta from factor pair(s). | Delta inherits factor confidence and sign. |
| `0x106` | `LORA_SCALE` | Procedural Adapter | `[delta, alpha] -> [scaled_delta]` | Scale the adapter delta by the absorption factor. | Magnitude changes, polarity preserved. |
| `0x107` | `LORA_TERNARY_MASK` | Procedural Adapter | `[delta, mask] -> [masked_delta]` | Apply {-1, 0, +1} gating per output row/dimension. | Mask trits map directly to polarity lanes. |
| `0x108` | `LORA_SHADOW_ABSORB` | Procedural Adapter | `[shadow_delta, contrast_signal] -> [updated_shadow_delta]` | Absorb sleep-time contrast signal into the shadow adapter program. | Signal ternary-packs promote/hold/reject pressure. |

### 7.2 Procedural Adapter Note

The `0x104–0x108` block keeps the **programs before opcodes** doctrine intact:
specialist deltas are still small executable programs over shared base weights,
not an external tensor-training stack. These opcodes make the adapter program
first-class, auditable, and sleep-time-updatable inside the sovereign RPN/TRM
substrate.

---

### 7.3 WINE I/O Contract Block (0x180–0x18F)

**Authority**: `TEMP/CLAUDE_CODEX_GPU_GAME_LOOP_CLOSURE_04.18.2026.md` §4.2–4.3
and `docs/vocabulary/CANONICAL_REGISTRY_SPECIFICATION.md` §8.5.

These opcodes implement the Translation Surface: the device-side path from
external bytes (DOM, ARC3, stdin, audio, image) through a Galaxy-resident
`WineContractStar` and back. All execution is on-device. Python on the hot path
never calls these ops directly — it only pushes bytes into the input ring.

Admission class: **Class A** (immediately executable via existing `GALAXY_SCAN`
and RPN dispatch infrastructure). `wine_contract_scan.cu` is the companion kernel.

| Opcode | Mnemonic | Stack diagram | Semantics | Ternary packing |
| --- | --- | --- | --- | --- |
| `0x180` | `WINE_INGRESS_DECODE` | `[input_bytes_ptr, length, contract_ptr] -> [galaxy_form_ptr]` | Execute the `ingress_rpn_addr` program stored in the active `WineContractStar`. Reads raw I/O bytes, produces a Galaxy form pointer consumed by downstream phases. | `galaxy_form_ptr` confidence reflects parse completeness; polarity `+1` clean, `0` partial, `-1` decode error. |
| `0x181` | `WINE_EGRESS_ENCODE` | `[galaxy_form_ptr, contract_ptr] -> [bytes_ptr, length]` | Execute the `egress_rpn_addr` program stored in the active `WineContractStar`. Serialises a Galaxy form to output bytes for the Tablet output ring. | `length` confidence reflects encode completeness; polarity mirrors `WINE_INGRESS_DECODE` convention. |
| `0x182` | `WINE_RESOLVE` | `[paradigm_type] -> [contract_ptr]` | Scan the `WineContractStar` table (via `wine_contract_scan.cu`) for the entry whose `paradigm_type` field matches the input. Returns the contract VRAM pointer, or `0` if no contract is registered for that paradigm. On miss, the kernel writes `tick_status = CONTRACT_MISS`; sleep-time handles the craft ticket. | `contract_ptr` polarity: `+1` resolved, `0` miss (no matching paradigm), `-1` ambiguous (two entries share paradigm — loader conflict, see §8.5.5 invariant 2). |
| `0x183–0x18F` | *(reserved)* | — | Reserved for future contract ops: audio frame windowing, image tile ingress, video keyframe sync, multi-modal interleave. Opcodes are placeholders; no PTX kernel is admitted until the Stage 0–3 promotion pipeline is complete for each. | — |

### 7.4 Physics-to-Visual Bridge (0x190)

**Authority**: `TEMP/CLAUDE_CODEX_GPU_GAME_LOOP_CLOSURE_04.18.2026.md` §12
(Nemotron warp-shuffle patterns, post-chain MVCIC note).

This opcode closes the loop between the PHYSICS phase output and the Tablet
visual surface, enabling live physics state rendering without any Python
involvement.

| Opcode | Mnemonic | Stack diagram | Semantics | Ternary packing |
| --- | --- | --- | --- | --- |
| `0x190` | `PHYSICS_EMIT_VISUAL` | `[phys_state_ptr] -> [draw_op_count]` | Pops a physics state struct pointer (`{float3 position, float3 velocity, float3 acceleration, uint32_t entity_id}`). Emits a sequence of Drawing Galaxy ops (`CIRCLE` for position, `LINE` for velocity vector, optional `LINE` for acceleration) targeting the Tablet surface buffer. Draw ops are pushed directly to the RPN stack for consumption by the act_phase renderer. | `draw_op_count` confidence reflects how many ops were emitted; polarity `+1` full, `0` partial (clipped by LOD), `-1` suppressed (entity culled by frustum). |

`0x191–0x19F` are reserved for future physics visual ops (rigid-body wireframe,
field-line render, heat-map overlay). Same admission gating applies.

---

## 8. Ternary-Ready Register Semantics (March 2026)

The [Hyper-Parallel Processing](HYPER_PARALLEL_PROCESSING.md) paradigm (§6) establishes that all RPN register values MUST be representable with **value + confidence + polarity** — either natively (future ternary hardware: balanced ternary −1/0/+1) or by convention (current binary hardware: explicit encoding).

**Implications for RPN opcodes:**
- `OP_STORE` / `OP_RECALL` registers carry ternary-ready triples, not bare scalars
- Cross-core register sharing (hyper-parallel specialist swarm) uses the same STORE/RECALL mechanism
- "Uncertain / no evidence" (0 in balanced ternary) is a first-class value, not a sentinel or NaN
- When ternary hardware accelerators arrive, RPN programs migrate by changing the hardware mapping, not the opcode semantics

**Current implementation**: Binary encoding with explicit confidence fields. Future ternary hardware maps each trit natively.

---

## 9. Future Extensions

Once the math core tiering and Reality Enabler galaxies are stable, the promotion pipeline (Section 6) governs admission of new opcodes. Candidates include:

- Physics-specific shortcuts (e.g., `OP_FORCE_ACCUM`, `OP_INTEGRATE_EULER`), if profiling shows strong benefit.
- Chemistry-specific combinators (e.g., `OP_APPLY_REACTION_RULE`) built on top of set/graph ops.
- Biology-specific pattern operators (e.g., `OP_BRANCH_LSYSTEM`) for more compact growth programs.

All candidates MUST pass the Stage 0-3 pipeline documented above.

---

## 10. References

- `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py`  
- `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py`  
- `knowledge3d/cranium/ptx_runtime/rpn_math_core.py`  
- `docs/vocabulary/MATH_CORE_SPECIFICATION.md`  

---

## 11. Reserved Future Blocks (Range Reservation Table)

**Authority**: `TEMP/CLAUDE_CODEX_OPCODE_RANGE_RESERVATION_DOCTRINE_04.18.2026.md` (Ruling 5, 2026-04-18).
**Status**: Normative — this table is the single source of truth for opcode range reservations.

Per the Opcode Range Reservation Doctrine, any parallel-lane task that will mint new opcodes MUST pre-reserve its block in this table before dispatching spec or implementation work. Opcodes assigned outside a reserved block fail the Gate R acceptance check. A reservation written only in a design doc is not a reservation.

**Cross-reference**: §6.3 (Stage 3: PTX Kernel Admission) — the pre-reservation prerequisite blocks admission until §11 is amended.

### 11.1 Schema

| Field | Type | Meaning |
|---|---|---|
| `block_start` | hex u16 | Lowest opcode number in the block (inclusive) |
| `block_end` | hex u16 | Highest opcode number in the block (inclusive) |
| `owner_spec` | path | Relative path to the spec file governing opcode assignments inside this block |
| `date_reserved` | YYYY-MM-DD | Date the reservation was appended |
| `status` | enum | `active` (lane working) / `released` (lane done, opcodes permanent) / `superseded` (spec withdrawn, range free) |

### 11.2 Initial Reservations (reconstructed 2026-04-18)

| block_start | block_end | owner_spec | date_reserved | status |
|---|---|---|---|---|
| `0x100` | `0x10F` | `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md §7.1` | 2026-04-13 | released |
| `0x170` | `0x17F` | `TEMP/CLAUDE_CODEX_TRANSFER_YARD_AND_EMBEDDING_SOVEREIGNTY_04.18.2026.md` | 2026-04-18 | active |
| `0x178` | `0x17A` | `TEMP/CLAUDE_CODEX_INSTANTIABLE_CORE_ISOLATION_04.18.2026.md` (queue ops — sub-reservation within 0x170–0x17F) | 2026-04-18 | active |
| `0x180` | `0x18F` | `TEMP/CLAUDE_CODEX_GPU_GAME_LOOP_CLOSURE_04.18.2026.md §4.2–4.3` (WINE I/O) | 2026-04-18 | active |
| `0x190` | `0x19F` | `TEMP/CLAUDE_CODEX_GPU_GAME_LOOP_CLOSURE_04.18.2026.md §12` (physics-to-visual bridge) | 2026-04-18 | active |
| `0x1A0` | `0x1A6` | `TEMP/CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md` (math/utility) | 2026-04-18 | active |
| `0x1AA` | `0x1AF` | **BitNet b1.58 Attention** — 0x1AA (TERNARY_MATMUL_ADDSUB), 0x1AB (TERNARY_PACK5), 0x1AC (TERNARY_UNPACK5), 0x1AD (VEC_NORM_L2_INT8), 0x1AE (ATTENTION_MARGIN_SHIFT), 0x1AF (ATTENTION_MARGIN_SCALED). Corrected 2026-04-18: 0x1AD matches production kernel (`knowledge3d/cranium/kernels/bitnet_attention.cu`). | 2026-04-18 | active |
| `0x1B0` | `0x1B0` | Reserved for future attention expansion (was draft assignment for VEC_NORM_L2_INT8; superseded by 0x1AD actual production). | 2026-04-18 | active |
| `0x1B1` | `0x1B5` | **Attention Future Expansion** — halting gate variants, sparse-K attention, per `TEMP/supersession_patches_04.18.2026_v4.md §4` (Ruling 4, turn-6) | 2026-04-18 | active |
| `0x1B6` | `0x1B9` | `TEMP/CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md` (TENSOR_INTERPOLATE, KMEANS_PLUS_INIT, CTYPES_VIEW_AS_PTX, CUDA_MALLOC_ASYNC — minted; not renumberable per expand-not-replace) | 2026-04-18 | active |
| `0x1BA` | `0x1BF` | future normalization/attention headroom (narrowed from 0x1B1-0x1BF by v4 sub-reservation) | 2026-04-18 | active |
| `0x1C0` | `0x1C5` | `TEMP/CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md` (IMAGE/SPARSE — relocated from 0x1AA–0x1AF per v2 supersession) | 2026-04-18 | active |
| `0x1C6` | `0x1CF` | future physics expansion headroom (tied to §7.4) | 2026-04-18 | active |
| `0x1D0` | `0x1FF` | `TEMP/CLAUDE_INGESTION_SYMLINK_REWIRE_04.18.2026.md §13` — `VIRTUAL_PAGE_*` graph-grammar RPN for ingestion-path virtual pages. Sub-families: `PAGE_*` / `FRAME_*` / `TABLE_*` / `PARAGRAPH_*` / `LINE_*` / `RUN_*` / `*_EMIT*` (WORD/GLYPH/NUMERAL/EQUATION/FIGURE/SYMBOL) / `LAYOUT_*` / `STYLE_*` / `BIDI_*` / `SCRIPT_*` / `HYPHEN_*`. Specific numbers NOT assigned in this reservation — only the range. Extension into 0x200+ permitted on registry-owner review if the ~58 planned opcodes exceed 48 slots. | 2026-04-18 | active |
| `0x217` | `0x21F` | `TEMP/CLAUDE_CODEC_SOVEREIGNTY_AUDIT_04.20.2026.md` — **DotMap procedural color-map codec** (procedural dot placement + procedural color): DOT_PLACE_PROCEDURAL (content-adaptive density), COLOR_RPN_REF (RPN/galaxy-symlink color source), COLOR_PALETTE_REF, DOTMAP_SCAN_EMIT (line-based scan), DOTMAP_RLE_ENCODE/DECODE, DOTMAP_DELTA_ENCODE/DECODE, DOTMAP_HEADER. Premise (Daniel 2026-04-20): the dots already exist as 0x213 DOT_EMIT; the *count, layout, and per-dot color are all procedural* — NOT fixed, NOT tied to ingested resolution. Image → procedural dot placement program (density field / importance sampling) + per-dot color RPN ref; decoder renders at any output resolution. | 2026-04-20 | active |
| `0x240` | `0x24F` | `TEMP/CLAUDE_CODEC_SOVEREIGNTY_AUDIT_04.20.2026.md` — **JPEG-equivalent line-scan image codec** (line-based, dot-grid native): LINE_SCAN_START, LINE_SCAN_ROW, LINE_SCAN_COL, BLOCK_8X8_ZIGZAG, BLOCK_8X8_INV_ZIGZAG, QUANT_APPLY, QUANT_INVERT, DCT_8X8_FORWARD, IDCT_8X8, CHROMA_SUBSAMPLE_422, CHROMA_UPSAMPLE_422, HUFF_ENCODE_RUN, HUFF_DECODE_RUN, IMG_HEADER_EMIT, IMG_HEADER_PARSE, IMG_FINALIZE. Existing `ternary_dct_2d.cu` may be promoted to back DCT_8X8_FORWARD. | 2026-04-20 | active |
| `0x250` | `0x25F` | `TEMP/CLAUDE_CODEC_SOVEREIGNTY_AUDIT_04.20.2026.md` — **Audio FFT / spectrogram family** (promotes §6.5 Stage 0 candidates): FFT_FORWARD_256, FFT_FORWARD_512, FFT_FORWARD_1024, FFT_FORWARD_2048, FFT_INVERSE, FFT_WINDOW_HANN, FFT_WINDOW_HAMM, STFT_FORWARD, STFT_INVERSE, MEL_FILTER_BANK, SPECTROGRAM_LINEAR, SPECTROGRAM_MEL, SPECTROGRAM_LOG, AUDIO_TO_DOTMAP (spectrogram as procedural image), DOTMAP_TO_AUDIO, HRTF_CONVOLVE. Backed by new `audio_fft.cu` + existing `ternary_mdct.cu`. | 2026-04-20 | active |
| `0x260` | `0x26F` | `TEMP/CLAUDE_CODEC_SOVEREIGNTY_AUDIT_04.20.2026.md` — **Frame codec / temporal video RPN**: FRAME_KEYFRAME, FRAME_DELTA, MOTION_VECTOR, FRAME_WARP, FRAME_BLEND, FRAME_SPRITE_LOAD, FRAME_SPRITE_DRAW, FRAME_SPRITE_BATCH, FRAME_PALETTE_SET, FRAME_CELL_FILL, FRAME_MORTON_2D, FRAME_SEQUENCE_RENDER, SCENE_LOAD, SCENE_QUEUE, SCENE_LOOP, FRAME_OBSERVE_64D (wraps existing `arc3_frame_encoder.cu`). | 2026-04-20 | active |
| `0x270` | `0x27F` | `TEMP/CLAUDE_CODEC_SOVEREIGNTY_AUDIT_04.20.2026.md` — **Projection screen / unified A/V playback** (K3D procedural video codec surface): VIDEO_FIELD_LOAD, AUDIO_FIELD_LOAD, SYNC_TIMELINE, TIMELINE_ADVANCE, PLAYBACK_TICK, PLAYBACK_START, PLAYBACK_STOP, SCREEN_PROJECT, SCREEN_RESIZE, SCREEN_COMPOSE, VIEWPORT_SET, LOD_SELECT, DOF_APERTURE, DOF_FOCUS, VIGNETTE, ATMOSPHERE_FOG. | 2026-04-20 | active |
| `0x280` | `0x28F` | `TEMP/CLAUDE_TEXTURE_FORGE_IMAGE_TO_3D_ARC3_SCREEN_04.20.2026.md` §2.1 — **Lane A: Texture Forge** (Werkkzeug-class, sovereign): TEX_SPLAT, TEX_KUWAHARA, TEX_WAVE, TEX_RIPPLE, TEX_VORTEX, TEX_FRACTAL_NOISE, TEX_CELLULAR_F1F2, TEX_GRAPH_EVAL, TEX_GRAPH_BIND, TEX_GRAPH_DIFF, TEX_FIT_STEP, TEX_FIT_CONVERGE, TEX_PALETTE_EXTRACT, TEX_PALETTE_APPLY, TEX_TILE_SYMMETRIZE, TEX_NORMAL_FROM_HEIGHT. Relocated from draft-0x1D0 on discovery that 0x1D0-0x1FF was already reserved by `CLAUDE_INGESTION_SYMLINK_REWIRE_04.18.2026.md §13` for VIRTUAL_PAGE_*. | 2026-04-20 | active |
| `0x290` | `0x29F` | `TEMP/CLAUDE_TEXTURE_FORGE_IMAGE_TO_3D_ARC3_SCREEN_04.20.2026.md` §2.2 — **Lane C: Image→3D**: IMG_TO_HEIGHTMAP, IMG_TO_SILHOUETTE, SILHOUETTE_EXTRUDE, HEIGHTMAP_TO_TERRAIN, HEIGHTMAP_TO_DISPLACEMENT, DEPTH_MONO_ESTIMATE, DEPTH_TO_POINTCLOUD, POINTCLOUD_TO_MESH, SPRITE_BILLBOARD, SPRITE_MULTIPLANE, DOODLE_TO_SYMMETRIC_MESH, LATHE_FROM_PROFILE, MESH_CSG_GPU, MESH_MARCHING_CUBES_GPU, MESH_NURBS_GPU, MESH_WRITE_GALAXY. Sovereign GPU-native replacements for host-fallback 0x170-0x176 (those entries remain per expand-not-replace; fallback path is DELETED in the same PR per `feedback_delete_dead_code_no_fallbacks_no_old_paths.md`). | 2026-04-20 | active |
| `0x2A0` | `0x2AF` | `TEMP/CLAUDE_TEXTURE_FORGE_IMAGE_TO_3D_ARC3_SCREEN_04.20.2026.md` §2.3 — **Lane D: ARC3 live-screen wiring**: ARC3_FRAME_DECODE, ARC3_PALETTE_SET, ARC3_FRAME_TO_DOTMAP, ARC3_PROJECT_TO_SCREEN, ARC3_CLICK_INVERT, ARC3_ACTION_EMIT, ARC3_REPLAY_STEP, ARC3_DIFF_HIGHLIGHT, ARC3_LIVES_HUD, ARC3_GAME_ID_BIND. Closes human-sees-what-AI-sees loop by composing existing `arc3_frame_encoder.cu` → `dotmap_codec.cu` → `projection_screen.cu`. | 2026-04-20 | active |
| `0x2B0` | `0x2BF` | `TEMP/CLAUDE_TEXTURE_FORGE_IMAGE_TO_3D_ARC3_SCREEN_04.20.2026.md` §10 — **Lane E: Memory-as-Image** (DeepSeek OCR extension per Daniel 2026-04-20): MEM_TO_DOTMAP, DOTMAP_TO_MEM, MEM_IMAGE_BIND, MEM_IMAGE_RECALL, MEM_IMAGE_COMPOSE, MEM_IMAGE_DIFF (0x2B0-0x2B5). 0x2B6-0x2BF reserved for expansion. Rationale: images as AI memory surfaces, not character recognition — reasoning traces bake to DotMap stars that are simultaneously raster + RPN program + Galaxy star + Matryoshka embedding (4-way addressable memory cell). | 2026-04-20 | active |
| `0x2C0` | `0x2CF` | `TEMP/CLAUDE_TEXTURE_FORGE_IMAGE_TO_3D_ARC3_SCREEN_04.20.2026.md` §12.4 — **MVCIC-sourced extensions** (6-partner chain + post-grounding): TEX_TERNARY_STREAM (0x2C0, Idea A — frustum-gated resonance streaming), SLEEP_PHYSICS_WEIGHT (0x2C1, Idea B — physics impulse → sleep cluster weight), META_RPN_EDIT (0x2C2, Idea C — self-modifying RPN, GATED to Layer 4 anneal context), MEM_FOVEAL_ENCODE (0x2C3, Kimi post-grounding — variable-resolution memory-image), ARC3_ATTENTION_HEATMAP (0x2C4). 0x2C5-0x2CF reserved for future MVCIC findings. | 2026-04-20 | active |
| `0x2D0` | `0x2DF` | `docs/vocabulary/DOCUMENT_GALAXY_SYMLINK_SPECIFICATION.md` — **Document Galaxy as Symlinks to Words** (Daniel 2026-04-20): DOC_STAR_NEW, DOC_WORD_REF, DOC_CHAR_REF, DOC_MEANING_REF, DOC_STYLE_SPAN, DOC_PARA_BREAK, DOC_STRUCT_EMIT, DOC_RENDER_IN_LANG, DOC_RENDER_DOTMAP, DOC_SYMLINK_RESOLVE, DOC_CONTENT_HASH, DOC_MATRYOSHKA_EMBED (0x2D0-0x2DB). 0x2DC-0x2DF reserved. Rationale: a document is a star whose content is an ordered list of symlinks to Word Galaxy stars; characters, glyphs, fonts, and meanings are NEVER duplicated in the document. "The" appears 10,000 times → stored ONCE. Meanings are language-agnostic (English "walk" / Portuguese "caminhar" / Japanese "歩く" → same Meaning star). | 2026-04-20 | active |
| `0x2E0` | `0x2FF` | future expansion headroom (Minecraft-for-Cognition lane family) | 2026-04-20 | active |
| `0x300` | `0x30F` | `knowledge3d/knowledgeverse/sleeptime_ingest.py` + `knowledge3d/cranium/kernels/sleeptime_lane_a.cu` — **Sleeptime Lane A: temporary-star consolidation**: SLEEPTIME_LANE_A_EVAL (0x300) — per-candidate gravity probe + defeasibility resolution → ternary promote/merge/discard trit; output: `out_trits[N]` int8 + `out_target_galaxy[N]` int32 + `out_best_house_idx[N]` int32. Inputs: candidate embeddings [N×16] float32, house embeddings [H×16] float32, grammar rule strengths [R] int8. Grid-stride, one thread per candidate. Reuses `gre_defeasible_resolver.cu` quantisation logic via device function call; reuses `ternary_depth_field.cu` cosine-similarity ternary encoding. 0x301-0x30F reserved for Lane A variants (merge-only pass, multi-pass anneal, confidence decay). | 2026-04-21 | active |
| `0x310` | `0x312` | `knowledge3d/knowledgeverse/sleeptime_weights.py` + `knowledge3d/cranium/kernels/sleeptime_lane_b.cu` — **Sleeptime Lane B: wake-cycle weight consolidation**: SLEEPTIME_LANE_B_DELTA_AGGREGATE (0x310) — aggregate per-tile float32 deltas across wake-cycle shadow-copy traces weighted by success confidence; SLEEPTIME_LANE_B_TRIT_ACCEPT_REJECT (0x311) — ternary trit decision per tile: +1 accept / 0 pending / -1 reject via `quantize_trit` (symlinked from `gre_defeasible_resolver.cu`); SLEEPTIME_LANE_B_INPLACE_UPDATE (0x312) — fold accepted deltas into BitNet b1.58 1.6-bit-packed TRM + specialist weight tiles using `pack5`/`unpack5`/`bitnet_dot_add_sub_skip` (symlinked from `bitnet_attention.cu`), re-quantise, write back in place. Grid-stride, one thread per tile. Python launches kernel and writes final BitNet-packed weight checkpoint to disk; all tile iteration and trit math is PTX-only. | 2026-04-21 | active |
| `0x313` | `0x313` | `knowledge3d/cranium/kernels/sleeptime_lane_b.cu` — **SLEEPTIME_LANE_B_TILE_QUANTIZE_F32_TO_BITNET** (Gap 2, 2026-04-21): per-tile float32 → BitNet b1.58 1.6-bit-packed quantiser; idempotent — reads per-tile `tile_format[T]` byte (WEIGHT_FORMAT_F32=0 / WEIGHT_FORMAT_BITNET=1), skips if already packed; calls `quantize_trit` (symlinked from `gre_defeasible_resolver.cu`) per trit and `pack5` (symlinked from `bitnet_attention.cu`) per 5-trit group; overwrites weight tile in place and sets `tile_format[tile_id]=WEIGHT_FORMAT_BITNET`. Runs live during sleeptime Lane B ticks — convert-on-first-touch, never at boot. Companion device function `lane_b_tile_quantize_f32_to_bitnet()` called as prologue of Stage B.3 (0x312). | 2026-04-21 | active |
| `0x314` | `0x31F` | reserved for Lane B variants (LoRA-only pass, specialist-region-only pass, full-net pass, confidence-decay variant). | 2026-04-21 | active |
| `0x320` | `0x32F` | `knowledge3d/knowledgeverse/wake_delta_capture.py` + `knowledge3d/cranium/kernels/wake_delta_capture.cu` — **Wake-Cycle Delta Capture** (Gap 1, 2026-04-20): WAKE_CYCLE_DELTA_CAPTURE (0x320) — per-tile activation-magnitude aggregator for successful convergence traces; gated by halting-gate threshold; produces signed per-tile float32 delta (sign = mean-activation direction, magnitude = L1-sum / N, normalised) written to `out_delta_tiles[T]` float32; `out_fired` int32 flag signals convergence (1) or skip (0). One thread per tile, grid-stride. Inputs: `activations[T×N]` float32, `halting_value` float32 scalar, `halting_threshold` float32 scalar. Feeds shadow_copy.event_buffer as `delta_tiles` (expanded to `n_tiles × TILE_TRITS` floats by Python readback) for Lane B consumption. Fires ONLY on successful halting-gate convergence; if `halting_value < halting_threshold` writes zero delta and `out_fired=0`. Python wrapper: ctypes readback I/O only; no numpy; no Python math over tiles. 0x321–0x32F reserved for delta-capture variants (per-specialist pass, contrastive delta, failure-trace anti-signal). | 2026-04-20 | active |

**Note on 0xA0–0xF1**: the reasoning-paradigm block (§7) predates this doctrine; its reservation authority is `TEMP/CLAUDE_REASONING_PARADIGMS_AND_N_SWARM_SPEC_2026-04-13.md §4` and is treated as `released`.

### 11.3 Maintenance Rules

1. **Append-only.** Never delete a row.
2. **Status transitions are one-way**: `active → released → superseded`. A `released` block's opcodes are permanent (expand-not-replace). A `superseded` range may be re-reserved by a new owner in a new row, but previously-assigned opcodes inside it remain permanent.
3. **No overlapping `active` rows.** Gate R rejects them.
4. **Size-1 reservations are valid.** Emergency single-opcode additions still require a row in the same commit as the opcode assignment.
5. **The registry wins when specs disagree.** Patch the specs; never patch the registry to match a spec.

### 11.4 Attention-Family Normative Notes (Turn-6 Rulings, 2026-04-18)

Per `TEMP/supersession_patches_04.18.2026_v4.md`:

- **0x1A9 `CONTRASTIVE_RANK_TOPK`** (Ruling 3 v4): accepts a 1-bit `margin_path` operand. Default `0` = Path A (SHIFT, 1-cycle SHR, no metadata load). Opt-in `1` = Path B (SCALED, smem-prefetch-mandatory, silent d-mismatch rescale). Lane-switchable mid-program.
- **0x1A9 `CONTRASTIVE_RANK_TOPK`** (Ruling 2 v4): when invoked in Path B mode and `d_active != star.d_tier`, kernel applies `margin × d_active / d_tier` inline. No warning, no log, no exit.
- **0x1AF `ATTENTION_MARGIN_SCALED`** (Ruling 1 v4): every kernel that invokes 0x1AF MUST cooperatively prefetch `confidence_margin` into shared memory before the scoring loop and issue `__syncthreads()`. This is part of the opcode's contract, not an optimization. Gate R-prefetch enforces.
- **0x1AF `ATTENTION_MARGIN_SCALED`** (Ruling 2 v4): handles d-mismatch via silent in-kernel rescale (1 IMUL + 1 SHR; tier ratios are powers of 2). No log output.
- **0x1B1-0x1B5 Sub-reservation** (Ruling 4 v4): "Attention Future Expansion — halting gate variants / sparse-K attention". Narrowed from pre-v4 generic "normalization/attention headroom". No opcodes minted yet; lane may pre-reserve per doctrine.

- `docs/vocabulary/REALITY_ENABLER_SPECIFICATION.md`  
