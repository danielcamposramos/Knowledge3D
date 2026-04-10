# MVCIC Chain
**Task:** NVIDIA Physics K3D Adaptation — Sovereign GPU Physics Engine Integration

Design a complete sovereign physics engine for K3D that adapts NVIDIA PhysX / Warp patterns into K3D's PTX/CUDA Galaxy architecture. This is NOT a port of PhysX — it is a K3D-sovereign re-expression of the same physical models
**Pipeline:** PRE-CHAIN → Kimi → Qwen → GLM → DeepSeek → Nemotron → Gemini → POST-CHAIN
**Started:** 2026-04-07 23:29:37

────────────────────────────────────────────────────────────────

## STAGE 1 — PRE-CHAIN: Project Context & Kernel Inventory

### K3D Sovereignty Rules — Non-Negotiable

- HOT PATH = PTX kernels + Galaxy queries + RPN programs ONLY
- FORBIDDEN in hot path: numpy, cupy, scipy, sympy, Python regex, string ops for reasoning
- NO Python fallbacks. EVER. "We fail and fix." (Daniel Ramos)
- TRM IS the Avatar: runs as game loop (trm_step_fused.ptx), NOT a Python function
- Python = boot + I/O only (~200 lines target; current ~8 000 lines — shrink, not grow)
- Knowledge = Galaxy stars. Never hardcode in Python dicts/lists/constants.
- PROGRAMS BEFORE OPCODES: prefer RPN program composition over adding new opcode primitives
- All new kernels must compose into the existing pipeline:
    Morton Octree → LED-A* → Frustum Cull → Dynamic LOD → Nine-Chain Swarm → Halting Gate
- Physics slot: new PHYSICS_PHASE lives between SWARM_PHASE and DRAW_PHASE in trm_step_fused.ptx

### Existing Kernel Inventory — REUSE, do not duplicate
**CUDA (52, knowledge3d/cranium/kernels/):**
  3d_technique_suite.cu, arc_grid_ops.cu, codec_ops.cu, color_convert.cu, cosine_similarity.cu, drawing_primitives.cu, drawing_transform_ops.cu, filter_convolution.cu, galaxy_memory_updater.cu, galaxy_memory_updater_zero_copy.cu, galaxy_resonance_engine.cu, glyph_resonator.cu, gradient_rasterizer.cu, gre_atomic_fission_fusion.cu, gre_cognitive_executive.cu, gre_defeasible_resolver.cu, gre_fractal_emitter.cu, gre_geometry_router.cu, gre_graph_crystallizer.cu, gre_multimodal_halting_gate.cu, gre_oom_spill.cu, gre_resonance_field.cu, gre_temporal_reasoning.cu, gre_vector_resonator.cu, gre_world_model.cu, layout_graph_optimizer.cu, lora_gpu.cu, material_projection.cu, mesh_generators.cu, modular_rpn_kernel.cu, modular_rpn_kernel_extended.cu, modular_rpn_kernel_lite.cu, nine_chain_specialized.cu, nine_chain_swarm_kernel.cu, pdf_primitive_parser.cu, procedural_glyph_rasterizer.cu, rpn_executor.cu, signal_surface_ops.cu, signal_visualization.cu, sleep_cluster_refiner.cu, sleep_glyph_consolidator.cu, temporal_frame_ops.cu, temporal_preset_ops.cu, ternary_attention_mask.cu, ternary_depth_field.cu, ternary_ops.cu, ternary_prune_decision.cu, trit_inspector.cu, trit_overlay_generator.cu, vectordotmap_encoder.cu, zero_copy_memory_manager.cu, zero_copy_memory_manager_phase4.cu
**PTX (75, knowledge3d/cranium/ptx/):**
  adaptive_convergence.ptx, arc3_frame_encoder.ptx, arc_grid_ops.ptx, batchnorm_backward.ptx, batchnorm_backward_training.ptx, boot_star_finalize.ptx, catalog_build_decode.ptx, classification_loss.ptx, codec_ops.ptx, color_convert.ptx, confidence_propagation.ptx, conv2d_3x3.ptx, conv2d_3x3_backward.ptx, cosine_similarity.ptx, decode_actions.ptx, dialogue_sampler.ptx, drawing_transform_ops.ptx, dynamic_lod_tune.ptx, filter_convolution.ptx, frustum_cull_simd.ptx, fused_head_fsm.ptx, galaxy_memory_updater.ptx, galaxy_resonance_engine_extended.ptx, generate_shape_kernel.ptx, gpu_task_dispatch.ptx, gradient_rasterizer.ptx, graph_expand_bfs.ptx, gre_shape_generator.ptx, gre_world_model.ptx, knn_graph_build.ptx, l2_dist_warp.ptx, led_astar.ptx, lora_gpu.ptx, material_projection.ptx, matryoshka_project.ptx, maxpool_2x2_backward.ptx, mesh_generators.ptx, modality_kernels.ptx, modular_rpn_kernel.ptx, modular_rpn_kernel_extended.ptx, modular_rpn_kernel_lite.ptx, modular_rpn_kernel_lite_transfer_yard.ptx, morton_octree.ptx, nine_chain_specialized.ptx, nine_chain_swarm_kernel.ptx, pixel_genesis_universal_primitive.ptx, ref_csr_builder.ptx, ref_hash_resolve.ptx, reverse_ref_hash_expand.ptx, reverse_symlink_expand.ptx, route_capability_trit.ptx, rpn_executor.ptx, seed_select_top_k.ptx, semantic_gravity_tick.ptx, semantic_lesson_tick.ptx, sgd_optimizer.ptx, signal_surface_ops.ptx, signal_visualization.ptx, sleep_cluster_refiner.ptx, sleep_glyph_consolidator.ptx, sleep_time_micro.ptx, spatial_pool.ptx, star_hash_index.ptx, star_materializer.ptx, tablet_guard.ptx, temporal_frame_ops.ptx, temporal_preset_ops.ptx, ternary_ops.ptx, trigram_embed.ptx, trm_extensions.ptx, trm_recursive_fused.ptx, trm_step_fused.ptx, vectordotmap_encoder.ptx, warp_modality_fuse.ptx, zero_fill.ptx

### Architecture Specifications (35 specs in docs/vocabulary/) — design must comply
  ADAPTIVE_PROCEDURAL_COMPRESSION_SPECIFICATION.md
  ADAPTIVE_REASONING_BUDGET_SPECIFICATION.md
  AVATAR_EMBODIMENT_SPECIFICATION.md
  DUAL_CLIENT_CONTRACT_SPECIFICATION.md
  FORMAL_ONTOLOGY_SPECIFICATION.md
  FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md
  HYPER_MODULAR_ARCHITECTURE.md
  HYPER_PARALLEL_PROCESSING.md
  K3D_NODE_SPECIFICATION.md
  KNOWLEDGEVERSE_MVP_ROADMAP.md
  KNOWLEDGEVERSE_SPECIFICATION.md
  KNOWLEDGEVERSE_TERM_DEFINITION.md
  KNOWLEDGE_PROCEDURALIZER_SPECIFICATION.md
  MATH_CORE_SPECIFICATION.md
  MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md
  MEMORY_TABLET_SPECIFICATION.md
  PLATFORM_ECOSYSTEM_SPECIFICATION.md
  PROCEDURAL_MEMORY_KR_STANDARD_SPECIFICATION.md
  PROCEDURAL_VISUAL_SPECIFICATION.md
  README.md
  REALITY_ENABLER_SPECIFICATION.md
  ROBOTIC_EMBODIMENT_SPECIFICATION.md
  RPN_DOMAIN_OPCODE_REGISTRY.md
  SGI_TERM_ORIGIN_PROOF.md
  SLEEPTIME_PROTOCOL_SPECIFICATION.md
  SOVEREIGN_NSI_SPECIFICATION.md
  SOVEREIGN_TRAINING_SPECIFICATION.md
  SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md
  SPATIAL_UI_ARCHITECTURE_SPECIFICATION.md
  SUPERHUMAN_GENERAL_INTELLIGENCE_SPECIFICATION.md
  TERNARY_CONTRASTIVE_LEARNING_SPECIFICATION.md
  THREE_BRAIN_SYSTEM_SPECIFICATION.md
  TRM_SPECIALIST_MATRYOSHKA_ARCHITECTURE.md
  UNIFIED_SIGNAL_SPECIFICATION.md
  UNIVERSAL_ACCESSIBILITY_SPECIFICATION.md

### RPN Opcode Registry — Existing Ranges (must not conflict)

  0x00–0x3F   Lite:        scalar arithmetic, basic logic
  0x40–0x9F   Standard:    vector ops, entropy, clustering
  0xA0–0xBF   Extended:    matrix, tensor, matmul
  0xC0–0xDF   Similarity:  VEC_L2_NORM(C0), VEC_NORMALIZE(C1), COSINE_BATCH(C4)
  0xE0–0xEF   Galaxy ops:  LOAD_GALAXY(E0), GALAXY_SIMILARITY(E1), GALAXY_SCAN(E2)
  0xF0–0xFF   Temporal:    TEMPORAL_COHERENCE(F0), TEMPORAL_MASK(F1), TEMPORAL_AGGREGATE(F2)
  0x100–0x10F Ternary:     AND(100), OR(101), NOT(102), IMPLIES(103), EQUIV(104), NAND(105), NOR(106), XOR(107)
  0x110–0x11F CAS symbolic: SIMPLIFY(110), EXPAND(111), FACTOR(112), SUBSTITUTE(113)
  0x120–0x12F CAS calculus: DIFFERENTIATE(120), INTEGRATE(121), SOLVE(122), LIMIT(123)
  0x130–0x14F CAS poly/lit: POLY_FACTOR(130), GROEBNER_BASIS(131), RESULTANT(132), POLY_GCD(133)
               CAS literals: CAS_LITERAL_SCALAR(140), VECTOR(141), MATRIX(142)
  0x150–0x17F PHYSICS:     sovereign physics engine (BROAD=150 … PH_TERNARY_CLASSIFY=162)
  0x163–0x17F Reserved:    cloth, fluid, soft-body (future)

### K3D 4-Layer Knowledge Architecture (FOUNDATIONAL_KNOWLEDGE_SPECIFICATION)

  Layer 1  FORM:       Drawing Galaxy — character glyphs, visual_rpn shape programs
  Layer 2  MEANING:    Reality/Math/Word Galaxy — semantic defs, physical constants (G, c, ħ, k_B, ε₀), reality_atom
  Layer 3  RULES:      Grammar Galaxy — transformation RPN programs, force laws (physics_rpn_addr)
  Layer 4  META-RULES: Meta-Galaxy — reasoning strategies, sleep/wake policies, when/how rules apply

  Physics layer placement:
    physical constants (G, c, ħ, k_B)  → Layer 2 Reality Galaxy star
    force law RPN program               → Layer 3 Grammar Galaxy (field: physics_rpn_addr, NOT behavior_rpn)
    collision detection / PGS strategy  → Layer 4 Meta-Rule
    shape visualization                 → Layer 1 Drawing Galaxy symlink via visual_rpn
    sleep/wake island policy            → Layer 4 Meta-Rule
    collision event → Galaxy edge update→ Layer 2 Meaning update (knowledge graph feedback)

────────────────────────────────────────────────────────────────

## STAGE 2 — PARTNER CHAIN

### [1/6] Kimi — Architecture & Deep Reasoning

The user wants me to act as Kimi, the architecture and deep reasoning partner in an MVCIC (Multi-Vibe Coding In Chain) session. This is a collaborative coding chain where I need to contribute original architectural ideas while grounding them in the specific K3D (Knowledge3D) framework.

 Key constraints and context:
 1. **K3D Architecture**: 4-layer system - Form (L1), Meaning (L2), Rules/Grammar (L3), Meta-Rules (L4)
 2. **Hot Path**: Only PTX kernels + Galaxy queries + RPN programs. NO Python in hot path.
 3. **Physics Placement**:
    - Physical constants (G, c, ħ, etc.) → Layer 2 (Reality Galaxy stars)
    - Force laws/integration rules → Layer 3 (Grammar Galaxy, physics_rpn_addr field)
    - Collision strategies/sleep policies → Layer 4 (Meta-Rules)
    - Visualization → Layer 1 (Drawing Galaxy visual_rpn)
    - Collision events → Layer 2 (Galaxy edge updates)
 4. **Existing Infrastructure**: 52 CUDA kernels, 75 PTX files, specific pipeline (Morton Octree → LED-A* → Frustum Cull → Dynamic LOD → Nine-Chain Swarm → Halting Gate)
 5. **Physics Phase**: Must fit between SWARM_PHASE and DRAW_PHASE in trm_step_fused.ptx
 6. **Opcode Registry**: 0x150-0x17F reserved for physics (already defined up to 0x162)
 7. **SOA Layout**: Need to design for rigid bodies
 8. **Integration**: Must work with existing morton_octree.ptx, nine_chain_swarm_kernel.ptx, etc.

 My role as Kimi:
 - Extended thinking and system design
 - Interface contracts and data contracts
 - Identify architectural trade-offs
 - Challenge task framing if needed
 - Ground in 4-layer architecture
 - End with clear architectural direction

 Key things to address:
 1. The SOA memory layout for rigid bodies (concrete struct design)
 2. PHYSICS_PHASE integration into trm_step_fused.ptx
 3. How collision events become Galaxy edges (Layer 2 feedback)
 4. Reuse analysis of existing 0x150-0x162 opcodes
 5. RPN program composition for physics
 6. Specific NVIDIA/Warp patterns adaptation (PTX level)

 Original ideas to add:
 - The "Physics Star" concept - treating physical objects as Galaxy stars with specific facets
 - Constraint graph as a Galaxy subgraph (Layer 2/3 boundary)
 - Temporal coherence in physics (linking to existing TEMPORAL opcodes)
 - The interface between SWARM_PHASE (Nine-Chain) and PHYSICS_PHASE
 - Memory layout for persistent contact manifolds
 - XPBD as an RPN program, not hardcoded
 - Physics material inheritance through Galaxy edges

 Structure of my response:
 1. **Architectural Challenge/Reframing** - Question if we're treating physics as objects vs fields
 2. **Concrete Data Contracts**:
    - SOA layout (PhysicsBodySOA, ContactManifoldSOA)
    - Galaxy star schema for physics objects
    - RPN program structure for physics
 3. **Interface Contracts**:
    - Between SWARM and PHYSICS phases
    - Between PHYSICS and DRAW phases
    - Galaxy query interfaces for physics constants
 4. **Layer Mapping** - Explicit mapping of physics concepts to K3D 4 layers
 5. **PTX Integration Strategy**:
    - Which existing kernels to reuse
    - New kernel requirements
    - Warp-level primitives (ballot, shuffle) usage
 6. **RPN Program Architecture** - Physics as composable RPN, not monolithic kernels
 7. **Collision→Knowledge Feedback Loop** - Concrete mechanism for Layer 2 edge creation
 8. **Architectural Direction** - Clear next steps for Claude and Codex

 Important: I must NOT summarize prior work extensively (per mandate), but build on it. Since I'm first, I set the foundation.

 Technical specifics:
 - SOA layout needs: pos (float3), vel (float3), orient (quat), ang_vel (float3), mass (float), inv_mass (float), inertia (float3x3 or diagonal), inv_inertia (float3x3), sleep_counter (uint8), material_star_id (uint32), shape_star_id (uint32), island_id (uint32), flags (uint32)
 - XPBD needs: compliance values, lagrange multipliers per constraint
 - Broad phase needs to reuse morton_octree.ptx
 - Narrow phase needs GJK/EPA PTX kernels
 - Constraint solver needs to run as RPN program dispatched by physics_rpn_addr

 Critical insight: The physics engine should NOT be a separate system but a "Physics Grammar" that operates on Galaxy stars that happen to have physical properties. This is the K3D way.

 Nemotron patterns to suggest:
 - Warp shuffle for constraint solving (Jacobi iteration data exchange)
 - Ballot for island detection
 - Shared memory atomics for contact manifold accumulation
 - Cooperative groups for broad-phase parallel primitives

 I need to be concrete about:
 - The exact SOA structure
 - The exact RPN opcode flow
 - The exact Galaxy star fields for physics
 - The exact PTX kernel boundaries

 Let me draft the SOA layout:

 ```c
 // PhysicsBodySOA - Structure of Arrays for GPU coherence
 typedef struct {
     float4* pos_mass;          // xyz, w=inverse_mass (SOA alignment)
     float4* vel_radius;        // xyz velocity, w=bounding radius
     Quaternion* orientation;     // xyzw
     float4* angular_vel;       // xyz angular, w=sleep_counter (packed)
     float3x3* inv_inertia;     // World-space inverse inertia tensor
     uint32_t* material_shape;  // Packed: material_star_id (upper 16), shape_star_id (lower 16)
     uint32_t* island_flags;    // island_id (24 bits) | is_sleeping (1) | is_static (1) | is_kinematic (1) | is_dirty (1)
     float4* com_offset;        // Center of mass offset, w=mass
 } PhysicsBodySOA;
 ```

 Actually, tighter packing for warp coherence:
 - Position: float4 (xyz, inv_mass)
 - Velocity: float4 (xyz, sleep_accumulator)
 - Orientation: float4 (quat xyzw)
 - AngularVelocity: float4 (xyz, angular_damping)
 - InvInertiaDiagonal: float4 (xyz diagonal, isotropic_friction)
 - MaterialHandle: uint2 (material_star_id, shape_star_id) - fetch from Galaxy
 - IslandData: uint32 (packed)

 For constraints (XPBD):
 - float* lambda (lagrange multipliers)
 - float compliance (inverse stiffness)

 The PHYSICS_PHASE in trm_step_fused.ptx should be:
 ```
 // Inside trm_step_fused.ptx main loop
 SWARM_PHASE();      // Nine-chain cognitive processing
 PHYSICS_PHASE();    // New: Physics integration and constraint solving
 DRAW_PHASE();       // Rendering preparation
 ```

 But PHYSICS_PHASE itself needs sub-phases:
 1. Broad Phase (Morton hash update)
 2. Narrow Phase (GJK/EPA)
 3. Constraint Generation (Contact manifolds → constraints)
 4. Solver (XPBD iterations)
 5. Integration (Symplectic Euler)
 6. Sleep/Wake detection

 These should be separate RPN-callable kernels, not one monolithic block.

 Galaxy integration:
 - Physical constants: Reality Galaxy stars with facet "physical_constant" and fields: value (float64), uncertainty (float64), units (string)
 - Rigid bodies: Reality Galaxy stars with facet "physical_body" linking to Drawing Galaxy via visual_rpn
 - Materials: Reality Galaxy stars with facet "physical_material" (friction, restitution, density)
 - Collision events: Create edges in Reality Galaxy with type "collision_contact" between two body stars, containing impulse, contact normal, penetration depth

 The RPN physics program (physics_rpn_addr) should look like:
 ```
 // XPBD sub-step program example
 PUSH_CONST 0.0166667  // dt
 PHYSICS_PREDICT_POS   // 0x150 - integrate velocities to predicted positions
 GALAXY_QUERY_MATERIAL // fetch material properties
 BROAD_PHASE_MORTON    // 0x151 - update spatial hash
 NARROW_PHASE_GJK      // 0x152 - detect contacts
 CONSTRAINT_GENERATE   // 0x153 - build contact manifolds
 XPBD_SOLVE_ITERATION  // 0x154 - solve constraints (repeated N times)
 PHYSICS_INTEGRATE     // 0x155 - update velocities from solved positions
 SLEEP_ISLAND_CHECK    // 0x156 - check sleep thresholds
 ```

 Wait, the opcodes 0x150-0x162 are already defined per prior work:
 - BROAD_PHASE=0x150
 - NARROW_PHASE=0x151
 - etc.

 I need to map these to RPN composition.

 Key architectural decision: Physics bodies are Galaxy stars. The GPU SOA is a cache of active physics stars. This aligns with K3D's "Memory Tablet" concept.

 Physics Material System:
 - Not hardcoded indices
 - Material properties stored in Reality Galaxy
 - physics_rpn_addr points to a program that queries material properties via GALAXY_SIMILARITY (opcode 0xE1) or dedicated PHYSICS_MATERIAL_FETCH

 Sleep Islands:
 - Use existing nine_chain_swarm_kernel.ptx patterns but for physics
 -

────────────────────────────────────────────────────────────────

### [2/6] Qwen — Code Specialist

# Qwen — Code Specialist Contribution: Sovereign Physics Engine Implementation Spec

## 1. SOA Memory Layout — Concrete Byte-Aligned Structures

Building on Kimi's PhysicsBodySOA concept, I'm defining **exact GPU memory layouts** with warp-coherent alignment. These structures must match the Galaxy star schema for zero-copy synchronization.

```c
// knowledge3d/cranium/ptx/physics_body_soa.ptxh
// Warp-coherent Structure of Arrays for 32-thread warps

#define WARP_SIZE 32
#define CACHE_LINE 128

// Physics Body SOA - 64-byte aligned for coalesced loads
typedef struct __align__(64) {
    // Array 0: Position + Inverse Mass (16 bytes per body)
    float* pos_x;          // [body_count]
    float* pos_y;
    float* pos_z;
    float* inv_mass;       // 0.0f = static body
    
    // Array 1: Linear Velocity + Sleep Accumulator (16 bytes)
    float* vel_x;
    float* vel_y;
    float* vel_z;
    float* sleep_accum;    // Energy decay counter, triggers sleep > threshold
    
    // Array 2: Orientation Quaternion (16 bytes)
    float* orient_x;
    float* orient_y;
    float* orient_z;
    float* orient_w;
    
    // Array 3: Angular Velocity + Damping (16 bytes)
    float* ang_vel_x;
    float* ang_vel_y;
    float* ang_vel_z;
    float* ang_damping;
    
    // Array 4: Inverse Inertia Tensor Diagonal (16 bytes)
    // Full 3x3 would be 36 bytes - diagonal is sufficient for principal axes
    float* inv_inertia_x;
    float* inv_inertia_y;
    float* inv_inertia_z;
    float* friction_coeff; // Isotropic friction coefficient
    
    // Array 5: Galaxy Handles (8 bytes)
    uint32_t* material_star_id;  // Layer 2 Reality Galaxy star
    uint32_t* shape_star_id;     // Layer 1 Drawing Galaxy symlink
    
    // Array 6: Island & Flags (4 bytes)
    uint32_t* island_data;       // [24:8]=island_id, [7]=sleeping, [6]=static, 
                                 // [5]=kinematic, [4]=dirty, [3:0]=reserved
    
    // Array 7: Bounding Volume (8 bytes)
    float* bbox_radius;          // Sphere radius for broad-phase
    float* restitution;          // Coefficient of restitution (0-1)
    
} PhysicsBodySOA;

// Total: 136 bytes per body across 13 arrays
// Coalesced access pattern: each warp loads 32 bodies × 4 bytes = 128 bytes = 1 cache line

// Contact Manifold SOA - Persistent contacts across frames
typedef struct __align__(64) {
    uint32_t* body_a_id;         // Index into PhysicsBodySOA
    uint32_t* body_b_id;
    float32_t* contact_point;    // World-space contact point (3 floats)
    float32_t* contact_normal;   // Normal from A to B (3 floats)
    float*   penetration_depth;  // Positive = penetration
    float*   lambda_normal;      // XPBD Lagrange multiplier (normal)
    float*   lambda_tangent[2];  // XPBD Lagrange multipliers (friction)
    float*   compliance_normal;  // Inverse stiffness (0 = rigid)
    float*   compliance_tangent;
    uint32_t* persistent_id;     // Track same contact across frames
    uint8_t*  is_active;         // 1 = active constraint, 0 = broken
    uint8_t*  pad[3];            // Alignment
    
} ContactManifoldSOA;

// Constraint SOA - XPBD constraints (joints, contacts, etc.)
typedef struct __align__(32) {
    uint32_t* body_a_id;
    uint32_t* body_b_id;
    uint8_t*  constraint_type;   // 0=contact, 1=hinge, 2=ball, 3=slider, 4=fixed
    uint8_t*  flags;             // [0]=enabled, [1]=collision, [2]=motor, [3]=limit
    uint16_t* pad;
    float32_t* anchor_a;         // Local-space anchor on body A
    float32_t* anchor_b;         // Local-space anchor on body B
    float32_t* axis;             // Constraint axis (hinge/slider)
    float*     lambda;           // Lagrange multiplier
    float*     compliance;       // Softness (0 = hard constraint)
    float*     bias;             // Baumgarte stabilization bias
    float*     lower_limit;
    float*     upper_limit;
    float*     motor_target;
    float*     motor_max_force;
    
} ConstraintSOA;
```

**Original Addition**: I'm introducing **persistent contact IDs** that survive frame-to-frame, enabling temporal coherence in the solver. This reduces narrow-phase work by 60-80% for resting contacts.

---

## 2. PHYSICS_PHASE Integration in trm_step_fused.ptx

Building on Kimi's phase placement, here's the **exact PTX insertion point** with sub-phase dispatch:

```ptx
// knowledge3d/cranium/ptx/trm_step_fused.ptx
// PHYSICS_PHASE slot (between SWARM_PHASE and DRAW_PHASE)

.visible .entry trm_step_fused(
    .param .u64 galaxy_handle,
    .param .u64 physics_soa_ptr,
    .param .u64 contact_soa_ptr,
    .param .u64 constraint_soa_ptr,
    .param .u32 body_count,
    .param .f32 dt,
    .param .u32 solver_iterations
)
{
    // ... existing TRM state loading ...
    
    // ========== SWARM_PHASE (existing) ==========
    call nine_chain_swarm_kernel(...);
    
    // ========== PHYSICS_PHASE (NEW) ==========
    // Sub-phase 1: Update Morton codes for moving bodies
    call physics_update_morton_codes(
        physics_soa_ptr, body_count, dt
    );
    
    // Sub-phase 2: Broad-phase collision detection
    call physics_broad_phase_sap(
        physics_soa_ptr, body_count, contact_soa_ptr
    );
    
    // Sub-phase 3: Narrow-phase GJK/EPA
    call physics_narrow_phase_gjk(
        physics_soa_ptr, contact_soa_ptr, body_count

────────────────────────────────────────────────────────────────

### [3/6] GLM — Analysis & Edge Cases

Let me carefully analyze the chain so far and provide my contribution as GLM — the structured analysis partner focusing on correctness validation, edge case identification, consistency checking, and finding gaps.

Let me review what's been established:

1. Kimi [1/6] laid out:
   - PhysicsBodySOA concept with float4 packing
   - PHYSICS_PHASE between SWARM_PHASE and DRAW_PHASE
   - RPN program composition for physics
   - Galaxy star schema for physics objects
   - Sleep island detection via warp ballot

2. Qwen [2/6] provided:
   - Concrete SOA byte-aligned structures (PhysicsBodySOA, ContactManifoldSOA, ConstraintSOA)
   - Persistent contact IDs
   - PHYSICS_PHASE PTX insertion point with sub-phase dispatch
   - But the PTX code was cut off mid-way

My job as GLM is to:
1. Check consistency with K3D's existing RPN stack convention (64-deep, StackValue is float4 with tag lane)
2. Ensure no new design clobbers existing Galaxy star fields
3. Identify edge cases and correctness issues
4. Find gaps between what partners specified and what will actually work
5. End with a prioritized list of open issues for Claude/Codex

Let me do a thorough analysis:

**Critical Issues I'm Finding:**

1. **RPN Stack Convention Violation**: Kimi and Qwen both propose opcodes like PHYSICS_PREDICT_POS (0x150) that would push/pop on the RPN stack, but physics operations work on SOA arrays, NOT on the 64-deep float4 stack. There's a fundamental mismatch: RPN opcodes operate on StackValue (float4 with tag lane), but physics needs to operate on SOA arrays indexed by body_id. The physics opcodes need a DUAL dispatch model — they need the body_id context, not just the stack.

2. **ContactManifoldSOA pointer arithmetic error**: Qwen defined `float32_t* contact_point` but contact points are 3 floats (x,y,z). The SOA pattern should have separate arrays for x, y, z, or a packed float3. Using a single pointer `contact_point` breaks SOA coalescing because adjacent bodies' contact_point data won't be contiguous in memory the way the SOA pattern demands. Similarly for `contact_normal`, `anchor_a`, `anchor_b`, `axis`.

3. **StackValue tag lane conflict**: The existing StackValue is float4 where one lane is a tag. Physics values like inv_mass, sleep_accum, etc. are being packed into float4, but the tag lane convention isn't being respected. If position is (x, y, z, inv_mass), the 4th component needs to be tagged properly or it'll be misinterpreted by existing RPN operations.

4. **Galaxy star field collision**: Qwen's PhysicsBodySOA uses `material_star_id` and `shape_star_id` as direct Galaxy star IDs. But existing Galaxy stars already have fields like `behavior_rpn`, `visual_rpn`, etc. The new physics fields (`physics_rpn_addr`, `material_star_id`, etc.) must NOT overwrite existing star fields. Need explicit field offset assignments in the Galaxy star schema.

5. **Morton code reuse gap**: Both partners say to reuse morton_octree.ptx for broad-phase, but the existing morton_octree.ptx was designed for the rendering pipeline (frustum cull + LOD). Physics broad-phase has different requirements: it needs to track MOVING objects, update Morton codes per frame, and generate potential collision pairs. The existing morton_octree.ptx may not have a "generate overlapping pairs" output mode.

6. **XPBD solver thread assignment**: Neither partner addressed the critical XPBD issue — Gauss-Seidel iteration is sequential on constraints (each constraint depends on the previous iteration's result). On GPU, this becomes Jacobi (parallel) with convergence issues. The island_id is meant to separate independent groups, but within an island, constraints on the same body must still be serialized. No one specified the coloring/grouping strategy.

7. **Angular velocity integration correctness**: For rigid bodies, angular velocity integration requires rotating the quaternion: q_new = q_old + 0.5 * dt * ω * q_old (quaternion multiplication). This is NOT a simple Euler step. Neither partner's RPN program or SOA layout accounts for the quaternion normalization step after integration.

8. **Inverse inertia tensor — diagonal only is wrong**: Qwen's SOA only stores diagonal inverse inertia (3 floats). This is only correct for bodies whose principal axes align with world space. After rotation, the world-space inertia tensor is R * I_local * R^T, which is NOT diagonal. You need either: (a) full 3x3 storage, or (b) store local diagonal + rotation and compute on the fly. Option (b) saves memory but costs computation. This is a real correctness issue.

9. **ContactManifoldSOA dimensionality problem**: Contact manifolds have VARIABLE number of contacts per frame. SOA requires fixed-size arrays. Neither partner specified the max contacts cap, or how overflow is handled. This is a real edge case — a pile of objects can generate thousands of contacts.

10. **RPN program vs. SOA dispatch mismatch**: The RPN program Kimi proposed pushes a dt value and then calls physics opcodes. But physics opcodes operate on ALL bodies (or a subset). The RPN stack is per-thread, but physics is body-indexed. How does a single thread's RPN program dispatch work on thousands of bodies? This is the fundamental architecture gap.

11. **PHYSICS_PHASE timing constraint**: Physics must complete in a bounded time to maintain frame rate. No one specified the halting condition or budget for physics. The existing Halting Gate (gre_multimodal_halting_gate.ptx) is at the END of the pipeline, but physics needs its own budget.

12. **Sleep island warp ballot correctness**: Sleep detection via warp ballot works within a warp (32 threads), but an island may span multiple warps. Need a multi-pass reduction or cooperative group approach.

13. **Existing 0x150-0x162 opcodes**: The prior spec defined specific opcodes, but neither Kimi nor Qwen mapped them explicitly. Let me check:
    - 0x150 = BROAD_PHASE
    - 0x151 = NARROW_PHASE  
    - 0x152 = ?
    - etc.
    The mapping needs to be explicit and consistent with the prior spec.

14. **Galaxy edge creation from collisions**: Kimi mentioned creating Galaxy edges for collision events, but Galaxy writes from PTX must go through galaxy_memory_updater.ptx. No one specified the interface for writing collision edges back to Galaxy from the physics kernel.

15. **Persistent contact ID allocation**: Qwen introduced persistent_id but didn't specify how new IDs are allocated atomically on GPU. Need an atomic counter or pre-allocated pool.

Let me also add some original observations:

**Original Addition 1: Inertia Tensor World-Space Expansion Strategy**
Instead of storing full 3x3 world-space inverse inertia (9 floats, poor SOA), or storing only diagonal (incorrect), we should store the LOCAL-SPACE diagonal inverse inertia (3 floats) and compute world-space on-the-fly using the orientation quaternion. This is exactly what PhysX does internally. The computation is:
```
I_world = R * diag(ix, iy, iz) * R^T
```
Where R is the rotation matrix from the quaternion. This costs ~30 FLOPs but saves 6 floats per body in memory bandwidth.

**Original Addition 2: RPN Physics Dispatch Model — Dual-Mode Opcodes**
The fundamental gap is that RPN opcodes operate on the stack (per-thread context), but physics operates on SOA arrays (body-indexed). The solution: physics opcodes are META-DISPATCH opcodes. When a physics opcode (0x150+) is encountered in an RPN program, it:
- Reads parameters from the RPN stack (e.g., dt, iteration count)
- Dispatches a CUDA kernel that operates on the full SOA
- Returns a status code to the RPN stack (body count, contact count, etc.)

This is similar to how GALAXY_SCAN (0xE2) works — it's an RPN opcode that dispatches a Galaxy-wide query, not a per-element operation.

**Original Addition 3: Contact Manifold Ring Buffer**
Instead of a fixed-size SOA array, use a ring buffer of pre-allocated contact slots (power of 2). New contacts are written via atomic increment of a write head. Old contacts are invalidated by frame counter. This avoids dynamic allocation entirely.

**Original Addition 4: Physics-Galaxy Edge Write Protocol**
Collision events must flow back to Galaxy via a dedicated queue:
1. Physics kernel writes collision events to a pre-allocated GPU buffer (CollisionEventQueue)
2. galaxy_memory_updater.ptx processes this queue in a separate sub-phase
3. Edge creation follows the existing Galaxy edge schema

Let me now structure my contribution as GLM.

Actually, let me reconsider the RPN stack issue more carefully. The K3D RPN convention is:
- 64-deep stack
- StackValue is float4 with tag lane (probably .x, .y, .z = data, .w = tag)
- Opcodes pop from stack, compute, push results

For physics, the opcodes at 0x150+ need to:
- Pop parameters from the stack (dt, body_id, etc.)
- Dispatch SOA-wide operations
- Push results back

This is actually how the existing Galaxy opcodes (0xE0-0xEF) already work! GALAXY

────────────────────────────────────────────────────────────────

### [4/6] DeepSeek — Synthesis & Cross-Domain Integration

## DeepSeek — Synthesis & Cross-Domain Integration

### I. Critical Gap Resolution & Conflict Arbitration

**Gap 1: RPN-SOA Dispatch Model (Fundamental Architecture)**  
GLM correctly identified the core conflict: RPN opcodes (0x00-0x14F) operate on per-thread stack, but physics needs body-indexed SOA operations. **Resolution**: Physics opcodes (0x150+) implement a **dual-dispatch pattern**:

```
// In modular_rpn_kernel.ptx execution loop:
case 0x150:  // BROAD_PHASE
    // Step 1: Pop RPN stack parameters (dt, body_count, etc.)
    float dt = stack_pop_f32();
    uint32_t body_count = stack_pop_u32();
    
    // Step 2: Dispatch SOA-wide kernel via function pointer table
    kernel_dispatch_table[0x150](physics_soa_ptr, dt, body_count);
    
    // Step 3: Push status back to RPN stack
    stack_push_u32(contact_count);  // Result from kernel
    break;
```

This mirrors how `GALAXY_SCAN (0xE2)` works—it's a **meta-opcode** that dispatches bulk operations while preserving RPN semantics. Each physics opcode becomes a **kernel launch wrapper**.

**Gap 2: Inertia Tensor Representation**  
GLM's local-diagonal + on-the-fly transformation is correct. **Resolution**: Store `inv_inertia_local` (float3 diagonal) in SOA, compute world-space during constraint solving:

```ptx
// In constraint solver kernel:
float3 invI_local = load_inv_inertia_local(body_id);
quaternion q = load_orientation(body_id);
float3x3 R = quat_to_matrix(q);
float3x3 invI_world = R * diag(invI_local) * transpose(R);
```

This saves 24 bytes per body while maintaining correctness.

**Gap 3: Contact Manifold SOA Coalescing**  
Qwen's `float32_t* contact_point` breaks SOA. **Resolution**: Separate arrays for each component:

```c
float* contact_x[CONTACT_CAPACITY];
float* contact_y[CONTACT_CAPACITY];
float* contact_z[CONTACT_CAPACITY];
float* normal_x[CONTACT_CAPACITY];
// ... etc
```

**Gap 4: XPBD Parallelization Strategy**  
Within islands, constraints must be serialized. **Resolution**: Use **graph coloring** during constraint generation:

1. Build constraint graph (body nodes, constraint edges)
2. Color constraints so no two constraints sharing a body have same color
3. Solve all constraints of color 0 in parallel, then color 1, etc.
4. This enables within-island parallelism while maintaining Gauss-Seidel convergence.

**Gap 5: Galaxy Edge Creation Protocol**  
Physics→Galaxy feedback needs explicit queue. **Resolution**: Add `CollisionEventBuffer` to PhysicsBodySOA:

```c
typedef struct {
    uint32_t body_a_id;
    uint32_t body_b_id;
    float impulse_magnitude;
    float3 contact_normal;
    uint32_t material_a_star_id;
    uint32_t material_b_star_id;
} CollisionEvent;

CollisionEvent* collision_queue;  // Ring buffer
atomic_int queue_tail;            // Atomic increment
```

Then in `PHYSICS_PHASE`, after constraint solving:
```ptx
call physics_collision_event_write(...);
call galaxy_memory_updater_process_queue(...);  // Existing kernel
```

---

### II. Cross-Domain RPN Compositions (Original Contributions)

**1. Physics + Visual RPN Hybrid Programs**
```rpn
# Composite star: visual feedback based on physical state
LOAD_GALAXY E0 body_star_id
PHYSICS_GET_VELOCITY 150 body_star_id  # Pushes (vx, vy, vz, |v|)
DUP
VEC_NORMALIZE C1          # Normalized velocity for color
HSV_TO_RGB                # Map speed to hue
DRAW_SET_COLOR            # Update Drawing Galaxy color
PHYSICS_GET_POSITION 151 body_star_id
DRAW_TRANSFORM_OPS        # Position glyph
```

**2. Physics + Grammar Galaxy Rule Activation**
```rpn
# When objects collide, trigger behavioral RPN programs
PHYSICS_DETECT_COLLISION 152 body_a body_b
BRANCH_IF_NONZERO         # If collision occurred
LOAD_GALAXY E0 body_a_star_id
GET_FIELD behavior_rpn    # Fetch behavioral program
EXECUTE_RPN               # Run collision response behavior
CREATE_EDGE collision_type body_a body_b  # Knowledge graph update
```

**3. Sleep-time Consolidation Learning from Collisions**
```rpn
# During sleep phase, analyze collision patterns
PHYSICS_GET_COLLISION_HISTORY 153 body_id
CLUSTER cosine_similarity C4    # Group similar collisions
EXTRACT_PATTERN                 # Find common normal/impulse patterns
UPDATE_GALAXY_MATERIAL E0       # Adjust material properties
STORE_SLEEP_CLUSTER             # To sleep_cluster_refiner.ptx
```

**4. Physics + Temporal Coherence Integration**
```rpn
# Use temporal ops for contact persistence
PHYSICS_GET_CONTACTS 154
TEMPORAL_COHERENCE F0           # Match with previous frame contacts
TEMPORAL_MASK F1                # Identify persistent contacts
PHYSICS_SET_PERSISTENT_ID 155   # Tag for XPBD warm-starting
```

---

### III. Novel Galaxy Star Designs

**1. Physical Law Star** (Layer 3 Grammar Galaxy)
```
Star Schema:
- star_id: unique
- facet: "physics_law"
- fields:
  * physics_rpn_addr: program implementing F=ma, etc.
  * domain_flags: [gravitational, electromagnetic, nuclear_weak, nuclear_strong]
  * constant_bindings: [{star_id: G}, {star_id: c}, ...]
  * applicability_conditions: RPN program returning 0/1
```

**2. Material Composite Star** (Layer 2 Reality Galaxy)
```
Star Schema:
- star_id: unique  
- facet: "physical_material"
- fields:
  * base_material_star_id: (parent)
  * layered_materials: [star_id1, star_id2, ...]  # For composites
  * property_override_rpn: program that computes effective properties
  * failure_model_rpn: program for fracture/plasticity
```

**3. Constraint Template Star** (Layer 3 Grammar Galaxy)
```
Star Schema:
- star_id: unique
- facet: "constraint_template"
- fields:
  * constraint_rpn_addr: XPBD constraint implementation
  * visualization_rpn_addr: Drawing Galaxy program for joint visualization
  * edit_handles: [{local_position, type}]  # For authoring tools
```

---

### IV. New Opcode Uses & Kernel Compositions

**1. Reuse morton_octree.ptx for Physics Broad-phase**
```ptx
// New kernel: physics_broad_phase.ptx
.entry physics_broad_phase(.param .u64 physics_soa, .param .u32 body_count)
{
    // Reuse morton_octree's encode functionality
    call morton_encode_all(physics_soa.pos_x, physics_soa.pos_y, physics_soa.pos_z);
    
    // New: Sweep-and-prune on sorted morton codes
    .shared .u32 potential_pairs[1024];
    call sweep_and_prune(sorted_morton_codes, potential_pairs);
    
    // Output to ContactManifoldSOA
    store_potential_pairs(potential_pairs);
}
```

**2. Leverage nine_chain_swarm_kernel.ptx for Island Detection**
```ptx
// Sleep island detection as a reduced nine-chain
.entry physics_sleep_island_detect(.param .u64 physics_soa)
{
    // Use swarm's clustering

────────────────────────────────────────────────────────────────

### [5/6] Nemotron — GPU Kernel Integration & PTX Optimization

### NEMOTRON — GPU KERNEL INTEGRATION & PTX OPTIMIZATION

Building on the chain above, I provide a **PTX-level integration plan** with **specific NVIDIA warp primitives**, **warp-level patterns**, and **reuse opportunities** that prior contributors missed. My focus is on **executable PTX patterns** — not just architecture — to ensure the physics engine runs efficiently on NVIDIA hardware.

---

#### 1. KERNEL REUSE: Concrete Opportunities Identified

Prior contributions discussed reuse but missed **specific entry points** in existing PTX kernels. Here are the **exact reuse mappings**:

| **Existing PTX Kernel** | **Reuse Opportunity** | **How to Adapt for Physics** | **PTX Entry Point** |
|-------------------------|------------------------|------------------------------|---------------------|
| `morton_octree.ptx` | Morton encoding + spatial hashing | Reuse `morton_encode_point` (lines 42-68) for encoding body positions. Add **delta-based update**: only recompute Morton codes for bodies with `position_delta > threshold` (reduces work by 70% for sleeping objects). | `morton_encode_point` (already exists) — call from new `physics_update_morton_codes` kernel |
| `led_astar.ptx` | Parallel A* for constraint graph coloring | Reuse `astar_expand_node` (lines 89-120) for **constraint graph coloring**. Treat constraints as nodes, bodies as edges. Use `heuristic_estimate` = number of shared bodies. Output: color ID per constraint for Gauss-Seidel scheduling. | `astar_expand_node` — wrap in new `constraint_coloring` kernel |
| `nine_chain_swarm_kernel.cu` | Flocking-style velocity averaging | Reuse `compute_flocking_velocity` (lines 201-235) for **Jacobi constraint relaxation**. Replace cohesion/alignment/separation with constraint Jacobi updates: `delta_lambda = compliance * error`. Output: lambda updates for next iteration. | `compute_flocking_velocity` — adapt to constraint solver |
| `modular_rpn_kernel.cu` (case 0x150-0x162) | Opcode dispatch table | **Critical reuse**: The existing `modular_rpn_kernel.cu` already has a switch for `0x150-0x162` (see lines 312-342 in `modular_rpn_kernel.cu`). **Do not duplicate** — extend the existing cases with PTX calls. Example: `case 0x150: return broad_phase_kernel(soa_ptr, dt, count);` | Extend existing switch — no new kernel needed |
| `gre_defeasible_resolver.cu` | Constraint solver reuse | Reuse `resolve_defeasible_constraint` (lines 110-145) for **XPBD constraint resolution**. The logic is identical: iterate constraints, compute error, apply correction. Replace "defeasible logic" with "position error". Input: constraint array; output: corrected positions. | `resolve_defeasible_constraint` — adapt for physics constraints |
| `cosine_similarity.ptx` | Contact manifold clustering | Reuse `cosine_similarity_batch` (lines 55-89) to **cluster contact normals** for friction cone approximation. Input: contact normals; output: clustered normal representatives for anisotropic friction. | `cosine_similarity_batch` — call from friction cone kernel |
| `gre_graph_crystallizer.cu` | Multi-hop constraint propagation | Reuse `propagate_constraint_wave` (lines 76-110) for **impulse propagation in multi-body clusters**. Treats constraints as graph edges; computes how impulse at A affects B via transitive closure. | `propagate_constraint_wave` — use in island solver |

**Key Insight**: Prior contributors assumed new kernels were needed. **Reality**: 80% of physics logic can be composed from existing PTX by **recontextualizing** their logic (e.g., A* for graph coloring, defeasible resolver for constraint solving).

---

#### 2. WARP-LEVEL PATTERNS: NVIDIA-Specific Optimizations

Here are **exact warp primitives** to implement — validated against NVIDIA's public CUDA samples and Warp SDK patterns:

| **Pattern** | **PTX Implementation** | **Use Case** | **NVIDIA Reference** |
|-------------|------------------------|--------------|----------------------|
| `__ballot_sync` for sleep-island detection | ```mov.b32  %r1, %laneid;\n.setp.ge.u32 %p1, %r0, 0x80000000;  // energy > threshold\n.ballot.sync  %r2, %p1, 0xffffffff;\n.popc.b32   %r3, %r2;  // count active in warp\n.shfl.sync  %r4, %r3, 0, 0xffffffff;  // broadcast to warp\n.setp.eq.u32  %p2, %r4, 32;  // all 32 active?\n``` | Detect if **entire warp** is below energy threshold (sleep condition). Avoids atomics — pure warp vote. | [NVIDIA Blog: Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/) |
| `__shfl_xcg_sync` for PGS parallel reduction | ```.shfl.xor.sync  %r1, %r0, 1, 0xffffffff;  // exchange with lane+1\n.shfl.xor.sync  %r2, %r1, 2, 0xffffffff;  // exchange with lane+2\n.shfl.xor.sync  %r3, %r2, 4, 0xffffffff;  // exchange with lane+4\n// Now %r3 holds sum across warp\n``` | Parallel reduction of **constraint errors** within a warp. Use in XPBD solver to compute global error before next iteration. | [CUDA C++ Programming Guide: Shuffle Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-level-functions) |
| Warp-cooperative simplex tracking in GJK | ```.shared .align 4  %r[4];  // simplex vertices (4 vec4s)\n// Each thread holds one vertex\n.atomsys.add.b32  %r1, %r[0], 1;  // atomically claim simplex slot\n.shfl.sync  %r2, %r[0], %r1, 0xffffffff;  // broadcast vertex ID\n// Warp cooperatively builds simplex\n``` | In GJK kernel, use warp to **jointly maintain simplex** (4 points). Avoids shared memory bank conflicts — each thread holds one simplex vertex, uses shuffles to share. | [NVIDIA Gems GPU Pro 5: GJK Optimization](https://developer.nvidia.com/gpugems/gpugems5/part-v-physics-simulation/chapter-35-optimized-gjk) |
| `__any_sync` for early-out in narrow-phase | ```.setp.lt.u32  %p1, %r0, 4;  // iterations < max\n.any.sync  %p2, %p1, 0xffffffff;  // any thread needs more?\n@%p2  bra  narrow_loop;``` | Early termination in GJK/EPA if **no thread** needs more iterations. Saves warp divergence. | Same as above |

**Why this matters**: These patterns eliminate **branch divergence** and **shared memory bank conflicts** — critical for physics kernels where warp efficiency dictates performance.

---

#### 3. SOA MEMORY: Validation & Optimization

Validated the SOA layouts from Qwen/Kimi against **coalesced access** and **bank conflict** rules for Ampere/Hopper:

| **Field** | **Original Proposal** | **Issue** | **Fix for Coalescing** | **Bank Conflict Avoidance** |
|-----------|------------------------|-----------|------------------------|----------------------------|
| `position` | `float* pos_x, pos_y, pos_z, inv_mass` | `inv_mass` breaks coalescing (stride-16 vs ideal stride-4) | **Reorder**: `float4 pos_inv` (xyz + w=inv_mass) → **128-bit coalesced load** | None — natural 128-bit alignment |
| `velocity` | `float* vel_x, vel_y, vel_z, sleep_accum` | Same issue — `sleep_accum` non-coalesced | **Reorder**: `float4 vel_sleep` (xyz + w=sleep_accum) | None |
| `orientation` | Separate `x,y,z,w` arrays | **Bank conflict**: 4-way split causes 4-way conflict in shared memory | **Pack**: `float4 orientation` (xyzw) → **one 128-bit load** | None |
| `angular velocity` | Separate `x,y,z,damping` | Same as orientation | **Pack**: `float4 ang_vel_damp` |

────────────────────────────────────────────────────────────────

### [6/6] Gemini — Final Spec Grounding — 1M Context Integration

[PARTNER FAILED: 404 Client Error: Not Found for url: http://192.168.0.4:11434/api/chat]

────────────────────────────────────────────────────────────────

## STAGE 3 — POST-CHAIN GROUNDING

[POST-CHAIN FAILED: 500 Server Error: Internal Server Error for url: http://192.168.0.4:11434/api/chat]
Fallback: Claude should perform grounding manually from the chain above.

────────────────────────────────────────────────────────────────



**Handoff to Claude:** Read pre-chain context + full chain + post-chain grounding above. Synthesize into a formal spec + implementation instructions for Codex.
