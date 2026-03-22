# Glass Box: 2D Diagnostic Views for K3D Cognitive OS

**Date:** 2026-03-22
**Purpose:** Complete specification of all internal 2D views, their data structures, data volumes, transport format, and visualization approach.
**Audience:** Christoph (viewer implementation), Claude+Codex (daemon-side emission)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│  GPU Engine (Python/PTX)                                 │
│  knowledgeverse.py │ trm_navigator.py │ adaptive_swarm.py│
│  sleeptime.py │ math_core_pool.py │ action_types.py      │
│                                                          │
│  ┌─────────────────────┐                                 │
│  │ Structured Snapshots │──── JSON over WebSocket ────┐  │
│  └─────────────────────┘                              │  │
└──────────────────────────────────────────────────────┘│──┘
                                                        │
          ┌─────────────────────────────────────────────┘
          ▼
┌──────────────────────────────────────────────────────────┐
│  Viewer (TypeScript / Three.js / Vite)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Memory Tablet │  │  HUD Panels  │  │  Galaxy Pod  │   │
│  │  (19 apps)   │  │  (overlays)  │  │ (stellarium) │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└──────────────────────────────────────────────────────────┘
```

**Transport budget:** Each snapshot should be <4 KB JSON to allow 10+ FPS streaming without saturating a local WebSocket. Views that exceed this use pagination or diff-only updates.

---

## View 01: Knowledgeverse Telemetry

**Subsystem:** Global VRAM Allocation & System Health
**Existing tablet app:** `StatsApp` (partial), `SummaryApp` (partial)

### Data Structure (daemon → viewer)

```typescript
type KnowledgeverseTelemetry = {
  type: "telemetry";
  ts: number;                        // epoch ms

  // Galaxy Universe
  total_entries: number;             // 117,497 (current)
  galaxies: {
    name: string;                    // "Drawing", "Character", etc.
    entry_count: number;             // per-galaxy count
    vram_bytes: number;              // estimated VRAM per galaxy
  }[];                               // 10 default galaxies

  // VRAM budget
  vram_used_mb: number;              // current total VRAM used
  vram_budget_mb: number;            // 200 MB target
  vram_total_mb: number;             // 12,288 MB (RTX 3070 8GB actual)

  // GPU streams
  streams: {
    pipeline: string;                // "cranium" | "galaxy" | "house"
    active: boolean;
    utilization_pct: number;         // 0-100
  }[];                               // 3 pipelines

  // Kernel latency (rolling window)
  kernel_latency_us: {
    median: number;                  // target <100μs
    p95: number;
    p99: number;
    sample_count: number;
  };

  // Composed head pipeline stage timings
  pipeline_stages: {
    stage: string;                   // "morton" | "led_astar" | "frustum" | "lod" | "swarm" | "halting"
    last_us: number;                 // last execution time in μs
  }[];                               // 6 stages
};
```

### Data Volume
- **Galaxy list:** 10 entries × ~40 bytes = ~400 bytes
- **Streams:** 3 entries × ~30 bytes = ~90 bytes
- **Pipeline stages:** 6 entries × ~30 bytes = ~180 bytes
- **Total per snapshot:** ~800 bytes
- **Update frequency:** 1 Hz (sufficient for dashboard)

### Visualization
- **Gauge ring cluster:** 5 circular gauges (VRAM %, GPU util %, kernel latency, entry count, stream health)
- **Sparkline:** Rolling kernel latency over last 60 seconds (60 data points)
- **Pipeline waterfall:** Horizontal stacked bar showing Morton → LED-A* → Frustum → LOD → Swarm → Halting timing breakdown
- **Galaxy bar chart:** 10 horizontal bars showing entry count per galaxy, colored by domain

### Source Files
- `knowledgeverse.py:286-297` — DEFAULT_GALAXIES (10 galaxies)
- `knowledgeverse.py:721` — `bind_gpu_galaxy_runtime()` (entry count, VRAM)
- `knowledgeverse.py:2386,2505` — `num_candidates` (pipeline stage)

---

## View 02: Cranium RPN Stack Monitor

**Subsystem:** Sovereign Reasoning Engine
**Existing tablet app:** `RpnApp` (browser-side execution only)

### Data Structure (daemon → viewer)

```typescript
type RpnStackSnapshot = {
  type: "rpn_stack";
  ts: number;

  // Active program
  program_id: string;                // RPN program identifier
  program_text: string;              // e.g., "3 4 ADD PUSH_NODE OP_SIMILAR"
  specialist: string;                // which specialist dispatched this

  // Stack state
  stack_depth: number;               // current depth (0-69)
  stack_limit: 69;                   // Tesla 6-9 constant
  stack_top: string[];               // top 8 values as display strings

  // Execution trace (last N ops)
  trace: {
    op: string;                      // "PUSH" | "ADD" | "OP_SIMILAR" | "EVALUATE" | "POP" etc.
    operand?: string;                // node label or value
    depth_after: number;             // stack depth after op
  }[];                               // last 12 operations

  // Tier info
  tier: number;                      // 0=lite (64D), 1=standard (512D), 2=extended (2048D)
  core_id: number;                   // MathCore instance_id
};
```

### Data Volume
- **Stack top:** 8 × ~20 chars = ~160 bytes
- **Trace:** 12 × ~40 bytes = ~480 bytes
- **Total per snapshot:** ~800 bytes
- **Update frequency:** On every RPN execution (burst during benchmarks, ~10-50/sec)

### Visualization
- **Stack column:** Vertical bar divided into 69 cells, filled cells = current depth, color gradient from green (shallow) to red (near limit)
- **Op trace scroll:** Monospace log of last 12 operations, color-coded by op type (PUSH=blue, arithmetic=green, similarity=cyan, control=yellow)
- **Tier badge:** Colored indicator (green=lite/64D, amber=standard/512D, red=extended/2048D)
- **Depth gauge:** Circular progress showing depth/69 with numeric readout

### Source Files
- `cranium/rpn_executor.py:28` — `STACK_DEPTH = 69`
- `cranium/sovereign_rpn_executor.py:30` — `STACK_DEPTH = 69`
- `cranium/bridges/sovereign_bridges.py:1673` — `STACK_DEPTH = 69`
- RPN opcode registry: `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md`
- 3 tiers: Lite (0x00-0x3F), Standard (0x40-0x9F), Extended (0xA0-0xFF)

---

## View 03: Tiered Math Core Dispatcher

**Subsystem:** Computation Routing & Load Balancing
**Existing tablet app:** None (new)

### Data Structure (daemon → viewer)

```typescript
type MathCoreDispatch = {
  type: "math_cores";
  ts: number;

  // Pool state
  max_cores: number;                 // GPU-queried capacity (fallback: 18)
  active_cores: {
    instance_id: number;
    tier: number;                    // 0=simple (64D/128D), 1=mid (512D), 2=high (2048D TRM)
    gpu_id: number;
    age_sec: number;                 // seconds since spawn
    last_used_sec: number;           // seconds since last use
  }[];
  idle_count: number;                // cores in idle pool
  idle_timeout_sec: number;          // 60s default

  // Routing stats (rolling)
  dispatch_counts: {
    tier_0: number;                  // simple queries routed
    tier_1: number;                  // mid queries routed
    tier_2: number;                  // high/TRM queries routed
  };

  // TRM state (the 7M parameter master)
  trm: {
    state_vector_dim: number;        // 2048
    parameter_count: string;         // "~7M"
    specialist_active: string;       // current specialist name
  };
};
```

### Data Volume
- **Active cores:** up to 18 × ~60 bytes = ~1,080 bytes
- **Total per snapshot:** ~1,400 bytes
- **Update frequency:** 0.5 Hz (pool changes slowly)

### Visualization
- **3-tier flowchart:** Left: "Routing Dispatcher" node. Three lanes: Simple (green, 64D/128D), Mid (amber, 512D), High (red, 2048D TRM). Right: "Output Collector." Animated dots flowing through lanes proportional to dispatch_counts.
- **Core grid:** 18 squares (max_cores), filled = active, gray = idle, empty = available. Color by tier.
- **TRM badge:** Special highlighted box for the tier-2 TRM master with parameter count and active specialist.

### Source Files
- `cranium/ptx_runtime/math_core_pool.py:12-30` — MathCore dataclass (instance_id, tier, stack_depth=69, gpu_id)
- `cranium/ptx_runtime/math_core_pool.py:32-35` — MathCorePool (FALLBACK_MAX_CORES=18, idle_timeout=60s)
- `cranium/ptx_runtime/math_core_pool.py:51` — `spawn_core(tier, reuse)`
- `cranium/ptx_runtime/math_core_pool.py:80` — `release_core(instance_id)`

---

## View 04: Galaxy Semantic Starfield Map

**Subsystem:** Active Working Memory (VRAM)
**Existing tablet app:** `GalaxyApp` (ring layout), `GraphApp` (XY mini-map), `EmbeddingsApp` (cosine peek)

### Data Structure (daemon → viewer)

```typescript
type GalaxyStarfield = {
  type: "galaxy_starfield";
  ts: number;

  // Projection (pre-computed 2D from N-dim embeddings)
  stars: {
    id: string;                      // star ID
    x: number;                       // projected 2D x (normalized 0-1)
    y: number;                       // projected 2D y (normalized 0-1)
    domain: string;                  // "Drawing" | "Math" | "Grammar" | etc.
    label?: string;                  // human-readable label (truncated)
    active: boolean;                 // TRM is currently visiting this star
  }[];

  // Summary stats
  total_stars: number;               // 117,497
  projection_method: string;         // "umap" | "pca" | "tsne"

  // Domain counts
  domain_counts: Record<string, number>;  // { "Drawing": 15230, "Math": 8921, ... }

  // Active query overlay
  query_trail?: {
    star_ids: string[];              // ordered list of visited stars
    specialist: string;
    task_type: string;
  };
};
```

### Data Volume
- **Full dump:** 117,497 × ~30 bytes = ~3.5 MB (TOO LARGE for streaming)
- **Condensed strategy:** Send only domain_counts + active stars (max ~200 highlighted) + query_trail
- **Condensed snapshot:** ~2 KB
- **Full projection:** Sent once on connect, then diff-only updates
- **Update frequency:** Active overlay at 5 Hz, full re-project on demand

### Transport Condensation

```typescript
// Initial load (once): full star positions as binary Float32Array
type GalaxyStarfieldInit = {
  type: "galaxy_starfield_init";
  total: number;                     // 117,497
  projection_method: string;
  domain_counts: Record<string, number>;
  // Binary attachment: Float32Array of [x, y] pairs = 117,497 × 8 bytes = ~940 KB
  // Sent as binary WebSocket frame, not JSON
};

// Live updates (streaming):
type GalaxyStarfieldUpdate = {
  type: "galaxy_starfield_update";
  ts: number;
  active_stars: string[];            // IDs of stars TRM is visiting (max ~20)
  query_trail: string[];             // ordered visit path
  specialist: string;
  domain_activity: Record<string, number>; // query counts per domain in last 10s
};
```

### Visualization
- **2D scatter plot:** All 117K stars as 1-pixel dots, colored by domain (10 colors). Pre-loaded from binary init.
- **Domain legend:** Color key with counts: Drawing (blue-15K), Character (teal-23K), Word (green-18K), etc.
- **Active pulse:** Stars in `active_stars` rendered as larger glowing dots (4px, pulsing)
- **Query trail:** Polyline connecting visited stars, fading from bright (recent) to dim (older)
- **Domain heatmap mode:** Toggle to show density as heat colors instead of domain colors
- **Cross-modal fusion zones:** Regions where stars from different domains overlap = fusion areas, highlighted with dashed circles

### Source Files
- `knowledgeverse.py:286-297` — DEFAULT_GALAXIES (10 galaxy names)
- Galaxy entries: each has `id`, `name`, `domain`, `embedding` (16-dim trigram), `metadata`
- `viewer/src/apps.ts:725-778` — EmbeddingsApp (cosine similarity, 5000 cap)
- `viewer/src/apps.ts:780-817` — GraphApp (XY mini-map, 4096 cap)
- `viewer/src/apps.ts:819-931` — GalaxyApp (ring layout with phi expansion)
- `viewer/src/projection/galaxyPodProjector.ts` — 3D stellarium (fibonacci sphere)

---

## View 05: Ternary (Trit) Depth Inspector

**Subsystem:** AI Navigation & Decision Routing
**Existing tablet app:** None (new)

### Data Structure (daemon → viewer)

```typescript
type TernaryInspector = {
  type: "ternary_trace";
  ts: number;

  // Current decision context
  task_type: string;                 // "ARC_TASK" | "MATH_TASK" | "MMLU_TASK" etc.
  question_id?: string;

  // Ternary routing decisions (per swarm worker)
  decisions: {
    worker_id: number;               // 0-8 (nine-chain swarm)
    route_decision: -1 | 0 | 1;     // reject / uncertain / accept
    confidence: number;              // 0.0-1.0
    specialist: string;              // which specialist this worker used
    candidate_label: string;         // what Galaxy entry it evaluated
  }[];                               // 9 workers

  // Ternary logic path
  logic_path: {
    op: string;                      // "TSWITCH" | "TAND" | "TOR" | "TNOT" | "TCOMP"
    inputs: number[];                // trit values in
    output: number;                  // trit value out
    description: string;             // human-readable
  }[];                               // sequence of ternary ops that led to final decision

  // Final outcome
  final_trit: -1 | 0 | 1;           // reject / uncertain / accept
  halting_converged: boolean;        // did halting gate accept?
};
```

### Data Volume
- **Decisions:** 9 × ~80 bytes = ~720 bytes
- **Logic path:** ~5-15 ops × ~60 bytes = ~600 bytes
- **Total per snapshot:** ~1,500 bytes
- **Update frequency:** Per question (burst during benchmarks)

### Visualization
- **Hex grid:** 3-color hexagonal grid (red=-1/reject, gray=0/uncertain, blue=+1/accept). Each hex = one ternary decision point.
- **Path line:** Bold colored line weaving through the hex grid showing the logic chain (TSWITCH/TAND/TOR)
- **Worker columns:** 9 vertical columns (one per swarm worker), each showing their trit decision as colored circle + confidence bar
- **Convergence indicator:** Green checkmark or red X for halting gate outcome
- **XAI tooltip:** Hover any hex → shows "Why this path?" explanation with the ternary op and its inputs/output

### Source Files
- `cranium/ternary_utils.py:28` — TernaryDecision class (value, confidence)
- `cranium/ternary_utils.py:88` — `ternary_route(confidence, threshold_low, threshold_high)` → -1/0/+1
- `cranium/ternary_utils.py:154` — `ternary_to_matryoshka_dim(complexity_ternary)` → 64/128/512
- Ternary opcodes: TADD(0x70), TMUL(0x71), TNOT(0x72), TCOMP(0x73), TQUANT(0x74), TPACK(0x75), TUNPACK(0x76)
- `knowledgeverse.py:414` — `_halting_gate`
- `knowledgeverse.py:1510-1521` — `get_halting_gate()` → MultimodalHaltingGate

---

## View 06: House Topology & Door Federation

**Subsystem:** Persistent Memory (Disk) & Networking
**Existing tablet app:** `DoorsApp` (door list), `SummaryApp` (partial)

### Data Structure (daemon → viewer)

```typescript
type HouseTopology = {
  type: "house_topology";
  ts: number;

  // Rooms
  rooms: {
    id: string;                      // "foyer" | "library" | "study" | "workshop" | "garden" | "living_room"
    domain: string;                  // primary knowledge domain
    node_count: number;              // objects in this room
    active: boolean;                 // TRM is currently "in" this room
  }[];                               // 6 rooms

  // Doors (internal)
  doors: {
    id: string;
    from_room: string;
    to_room: string;
    traversals: number;              // how many times traversed this session
  }[];

  // Navigation graph
  nav_graph: {
    node_count: number;              // 60+
    edge_count: number;
  };

  // Federation (external houses)
  federation: {
    house_id: string;                // "House: Medical"
    endpoint: string;                // "k3d://med:8080"
    status: "connected" | "disconnected" | "error";
    latency_ms: number;
    lazy_load_mb: number;            // current buffer usage
  }[];                               // 0+ federated connections

  // GLB state
  glb_size_kb: number;               // 305 KB current
  house_state_entries: number;       // 247,974 (persisted)
  lazy_load_budget_mb: number;       // 200 MB target
  lazy_load_used_mb: number;
};
```

### Data Volume
- **Rooms:** 6 × ~50 bytes = ~300 bytes
- **Doors:** ~8 × ~60 bytes = ~480 bytes
- **Federation:** variable (0 for single-house)
- **Total per snapshot:** ~1,200 bytes
- **Update frequency:** 0.2 Hz (topology changes rarely)

### Visualization
- **Node-link diagram:** Central "House: Default" node, connected to room nodes via door edges. Room size proportional to node_count. Active room highlighted.
- **Federation links:** External house nodes connected via dashed lines with latency labels
- **Lazy-load bar:** Progress bar showing lazy_load_used_mb / lazy_load_budget_mb
- **Traversal heatmap:** Door edges colored by traversal count (cold=blue/few → hot=red/many)
- **GLB badge:** File size indicator

### Source Files
- `viewer/src/loadHouseScene.ts:8-21` — HouseNode interface (starId, domain, houseRoom, housePosition, behaviorRpn, etc.)
- `viewer/src/loadHouseScene.ts:23-26` — HouseNavGraph (nodes[], edges[{door, from, to, cost}])
- 6 rooms: Foyer, Library, Study, Workshop, Garden, Living Room
- 60+ nodes, 305 KB GLB
- `viewer/src/apps.ts:960-986` — DoorsApp (door list with labels, addresses, roomA/roomB)

---

## View 07: Knowledge Garden Ontology Tree

**Subsystem:** Structured Ontology & Domain Logic
**Existing tablet app:** `ContentApp` (content sections), `LayersApp` (layer toggle)

### Data Structure (daemon → viewer)

```typescript
type OntologyTree = {
  type: "ontology_tree";
  ts: number;

  // Tree structure (flattened for 2D rendering)
  nodes: {
    id: string;                      // star ID
    label: string;                   // display name
    parent_id: string | null;        // null = root
    depth: number;                   // 0 = root domain
    domain: string;                  // galaxy domain
    children_count: number;          // number of direct children
    entry_count: number;             // Galaxy entries under this branch
    is_leaf: boolean;
  }[];

  // Pruning stats (from sleep-time)
  pruned_this_session: number;
  added_this_session: number;

  // Domain roots
  domains: string[];                 // ["Physics", "Mathematics", "Biology", ...]
};
```

### Data Volume
- **Shallow tree (depth ≤ 3):** ~200-500 nodes × ~80 bytes = ~16-40 KB
- **Condensation:** Send only expanded subtree (user-requested domain), not full ontology
- **Initial load:** Domain roots only (~10 nodes, ~800 bytes)
- **Expand on demand:** Request subtree for specific domain
- **Update frequency:** On demand (user clicks to expand)

### Visualization
- **Collapsible tree view:** Standard indented tree with expand/collapse chevrons
- **Domain color coding:** Each root domain has its galaxy color
- **Leaf count badges:** Number in brackets showing how many Galaxy entries live under each branch
- **Pruning indicators:** Red strikethrough for recently pruned nodes, green highlight for newly added
- **Search:** Text filter to find nodes by label

### Source Files
- Galaxy entries have `domain` and hierarchical `metadata` (category, subcategory)
- Content pages in `viewer/src/behavior/contentRenderer.ts` already render section hierarchies
- `viewer/src/apps.ts:1102-1144` — LayersApp (layer/tag enumeration from records)

---

## View 08: Dual-Client Node Inspector

**Subsystem:** Shared Reality Contract (Human + AI)
**Existing tablet app:** `ContentApp` (content view), `RpnApp` (RPN execution)

### Data Structure (daemon → viewer)

```typescript
type DualClientInspector = {
  type: "node_inspector";
  ts: number;

  node_id: string;                   // star ID being inspected

  // Human client layer
  human: {
    label: string;                   // display name (localized)
    surface_forms: Record<string, {  // language code → surface form
      word_ref: string;
      char_refs: string[];
    }>;
    visual_rpn: string;              // RPN program for visual form
    meaning_class: string;           // "concept" | "formula" | "template" etc.
    room: string;                    // House room placement
    position: [number, number, number]; // 3D position in room
  };

  // AI client layer (procedural metadata)
  ai: {
    domain: string;                  // Galaxy domain
    embedding: number[];             // 16-dim trigram embedding
    category_class: number;          // GPU_CATEGORY_CLASS_MAP value (0-6)
    rpn_program: string;             // procedural program
    taxonomy_refs: string[];         // ontology references
    component_refs: string[];        // sub-component links
    galaxy_ref?: string;             // if this node loads a Galaxy when opened
    behavior_rpn: string;            // interaction behavior
    metadata_size_bytes: number;     // compressed metadata size
    compression_ratio: number;       // 7-20x typical
  };

  // Parity check
  parity: {
    human_renderable: boolean;       // can human see/interact with this?
    ai_decodable: boolean;           // can PTX decode the metadata?
    decode_time_us: number;          // PTX decode latency
  };
};
```

### Data Volume
- **Per inspection:** ~1-2 KB (single node)
- **Update frequency:** On demand (user clicks node to inspect)

### Visualization
- **Split panel:** Left = "Human View" (rendered 3D preview + localized labels), Right = "AI View" (raw metadata as colored data matrix / QR-like grid)
- **Embedding sparkline:** 16 bars showing the trigram embedding values
- **Parity badge:** Green check if both clients can decode, red warning if mismatch
- **Decode latency:** Small gauge showing PTX decode time vs 20μs target
- **Surface forms:** Tab per language showing character references

### Source Files
- `viewer/src/loadHouseScene.ts:8-21` — HouseNode (starId, meaningClass, domain, surfaceForms, behaviorRpn, visualRpn, taxonomyRefs, componentRefs, galaxyRef)
- `knowledgeverse.py:298-305` — GPU_CATEGORY_CLASS_MAP (7 categories: unknown, clue_fact, formula, concept, benchmark_fact, template, cipher_result)
- `viewer/src/apps.ts:447-559` — ContentApp (content page with sections, DOM projection, RPN preview)
- `docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md` — Form + Meaning architecture

---

## View 09: SleepTime Consolidation Tracker

**Subsystem:** Memory Consolidation, Safety & Governance
**Existing tablet app:** `ControlApp` (sleep/wake buttons), `SummaryApp` (consolidation counts)

### Data Structure (daemon → viewer)

```typescript
type SleepTimeTracker = {
  type: "sleeptime";
  ts: number;

  // Current state
  state: "awake" | "stage_a" | "stage_b" | "committing" | "done" | "error";

  // Stage A: Knowledge consolidation
  stage_a: {
    health_log_rows: number;         // rows consumed
    success: boolean;
  } | null;

  // Stage B: Logic consolidation
  stage_b: {
    // Routing weight updates
    updated_specialists: string[];   // ["math", "visual", "grammar", "chat"]
    updated_count: number;           // e.g., 31,565

    // Contrastive learning results
    contrastive: {
      [specialist: string]: {        // "math" | "visual" | "grammar" | "chat"
        trained: boolean;
        positives: number;           // positive pairs trained
        negatives: number;           // negative pairs trained (repulsive)
        reason?: string;             // "no_pairs" if untrained
      };
    };

    // Jarvis coordinator
    jarvis: {
      briefs_consolidated: number;   // 128
      agreements: number;            // 440-512
      contradictions: number;        // 288-339
      task_stats: {
        [task_type: string]: {       // "ARC_TASK" | "MATH_TASK" | "MMLU_TASK" | "LHE_TASK"
          count: number;
          avg_workers: number;
          avg_planned_groups: number;
        };
      };
    } | null;

    // Weights persistence
    weights_path: string;            // "/K3D/.../trm_routing_state.json"
  } | null;

  // House state persistence
  house_saved: boolean;
  house_entries: number;             // 247,974
  house_path: string;                // "/K3D/.../galaxy_state.bin"

  // Safety
  transaction_id: string;
  checksum_valid: boolean;
  atomic: boolean;                   // was transition atomic?
};
```

### Data Volume
- **Per consolidation event:** ~2 KB
- **Update frequency:** Per stage transition during sleep-time (4-6 updates per sleep cycle)

### Visualization
- **Pipeline diagram:** Left: "Galaxy Bucket (Volatile RAM)" with scattered shapes. Arrow through "Policy & Checksum" gate. Right: "House Bucket (Persistent glTF)" with organized grid.
- **Stage progress:** Horizontal stepper: Stage A → Stage B → Commit → Done. Current stage highlighted.
- **Contrastive grid:** 4 rows (one per specialist), columns: trained?, positive count, negative count. Green/red indicators.
- **Jarvis summary:** Agreements vs contradictions ratio bar + brief count
- **Safety badges:** Atomic ✓, Checksum ✓, Transaction ID
- **History sparkline:** Last 10 consolidation cycles showing positive/negative pair counts trending

### Source Files
- `knowledgeverse/sleeptime.py:20` — SleepTimeConsolidation class
- `knowledgeverse/sleeptime.py:46-80` — execute() (Stage A → Stage B → commit → house save)
- `knowledgeverse/sleeptime.py:108-134` — _stage_b_logic() (routing + contrastive + jarvis)
- `knowledgeverse/sleeptime.py:180-260` — _run_contrastive_training() (positive_pairs, negative_pairs per specialist)
- `knowledgeverse/knowledgeverse.py:9961` — jarvis_sleep_consolidation()

---

## View 10: The 288-Byte Action Buffer Log

**Subsystem:** Universal Agency & Accessibility
**Existing tablet app:** None (new)

### Data Structure (daemon → viewer)

```typescript
type ActionBufferLog = {
  type: "action_buffer";
  ts: number;

  // Buffer contents (288 bytes decoded)
  header: {
    action_type: string;             // "NAV_MOVE" | "NAV_LOOK" | "DIALOGUE" | "WRITE_MEM" | "UPDATE_TABLET" | "NO_ACTION"
    action_code: number;             // 0x00-0xFF
    confidence: number;              // 0.0-1.0
    curiosity: number;               // curiosity bias
    flags: number;                   // bitfield
  };

  navigation: {
    position: [number, number, number];
    direction: [number, number, number];
    velocity: number;
    room_id: number;
    confidence: number;
  } | null;                          // populated if action_type is NAV_*

  dialogue: {
    token_ids: number[];             // up to 32 uint16 tokens
    length: number;
    temperature: number;
    thinking_score: number;
  } | null;                          // populated if action_type is DIALOGUE

  memory: {
    summary_hash: string;            // uint64 as hex
    zone_id: number;
    confidence: number;
    embedding: number[];             // 4-float compressed summary
  } | null;                          // populated if action_type is WRITE_MEM

  tablet: {
    mutation_type: number;
    data: number[];                  // 6 uint32 words
  } | null;                          // populated if action_type is UPDATE_TABLET

  // Source attribution
  source: string;                    // "ptx_kernel" | "vr_controller" | "bci" | "synthetic_ai" | "keyboard"
  buffer_size_bytes: 288;            // always 288
};
```

### Data Volume
- **Per action:** ~500 bytes JSON (288 bytes raw + field names + source)
- **Update frequency:** Per action (variable, 1-50/sec during interaction)

### Visualization
- **Action timeline:** Horizontal log showing recent actions as colored blocks. NAV=blue, DIALOGUE=green, WRITE_MEM=purple, TABLET=orange, NO_ACTION=gray.
- **Buffer hex dump:** Raw 288 bytes as hex grid (18 rows × 16 bytes), sections color-coded: Header (white), Navigation (blue), Dialogue (green), Memory (purple), Tablet (orange)
- **Source convergence proof:** Table showing different input sources (keyboard, AI, VR) all producing the same 288-byte format — the key accessibility insight
- **Confidence gauge:** Circular gauge for action confidence
- **Curiosity indicator:** Separate small gauge for curiosity bias

### Source Files
- `cranium/actions/action_types.py:29-37` — ActionType enum (NAV_MOVE=0x00, NAV_LOOK=0x01, DIALOGUE=0x02, WRITE_MEM=0x03, UPDATE_TABLET=0x04, NO_ACTION=0xFF)
- `cranium/actions/action_types.py:43-78` — ACTION_BUFFER_DTYPE (288 bytes, 72 × 4-byte words)
  - Header: action_type(u32), confidence(f32), curiosity(f32), flags(u32) = 16 bytes
  - Navigation: position(3×f32), direction(3×f32), velocity(f32), room_id(u32), confidence(f32), reserved(6×u32) = 60 bytes
  - Dialogue: token_ids(32×u16), length(u32), temperature(f32), thinking_score(f32), reserved(6×u32) = 100 bytes
  - Memory: summary_hash(u64), zone_id(u32), confidence(f32), embedding(4×f32), reserved(8×u32) = 64 bytes
  - Tablet: mutation_type(u32), data(6×u32), reserved(4×u32) = 48 bytes
- `bridge/headless_tablet.py:114` — `to_action_buffer()`

---

## Diagnostic Matrix Summary

| # | View | Subsystem | Primary Metric | Data Size | Freq | Target User | Existing App |
|---|------|-----------|---------------|-----------|------|-------------|--------------|
| 01 | Knowledgeverse Telemetry | VRAM/Health | Entry count, latency | ~800 B | 1 Hz | Developer | StatsApp, SummaryApp |
| 02 | RPN Stack Monitor | Reasoning | Stack depth/69, ops | ~800 B | 10-50 Hz | Developer | RpnApp |
| 03 | Math Core Dispatcher | Routing | Core count, tier dist | ~1.4 KB | 0.5 Hz | Developer | (new) |
| 04 | Galaxy Starfield Map | Working Memory | 117K stars, domains | ~940 KB init + 2 KB/update | 5 Hz | Architect | GalaxyApp, GraphApp |
| 05 | Ternary Depth Inspector | Decision | Trit values, path | ~1.5 KB | Per question | Architect | (new) |
| 06 | House Topology | Persistence | 6 rooms, doors, GLB | ~1.2 KB | 0.2 Hz | Admin | DoorsApp |
| 07 | Ontology Tree | Domain Logic | Tree depth, leaf count | ~800 B base + expand | On demand | Architect | ContentApp, LayersApp |
| 08 | Dual-Client Inspector | Shared Reality | Decode time, parity | ~1.5 KB | On demand | Developer | ContentApp, RpnApp |
| 09 | SleepTime Tracker | Consolidation | Pairs, agreements | ~2 KB | Per stage | Admin | ControlApp, SummaryApp |
| 10 | Action Buffer Log | Agency | 288-byte decode | ~500 B | 1-50 Hz | Developer | (new) |

---

## Transport Protocol

All views share a single WebSocket connection (same one used by chat). Message routing by `type` field:

```typescript
// Daemon emits these message types:
type DiagnosticMessage =
  | KnowledgeverseTelemetry    // type: "telemetry"
  | RpnStackSnapshot           // type: "rpn_stack"
  | MathCoreDispatch           // type: "math_cores"
  | GalaxyStarfieldInit        // type: "galaxy_starfield_init"
  | GalaxyStarfieldUpdate      // type: "galaxy_starfield_update"
  | TernaryInspector           // type: "ternary_trace"
  | HouseTopology              // type: "house_topology"
  | OntologyTree               // type: "ontology_tree"
  | DualClientInspector        // type: "node_inspector"
  | SleepTimeTracker           // type: "sleeptime"
  | ActionBufferLog            // type: "action_buffer"
  ;

// Viewer can request specific views:
type DiagnosticRequest = {
  type: "diagnostic_request";
  view: string;                      // view type to request
  params?: Record<string, any>;      // e.g., { domain: "Physics" } for ontology tree expand
};
```

### Bandwidth Budget

| Stream | Size | Freq | Bandwidth |
|--------|------|------|-----------|
| Telemetry | 800 B | 1 Hz | 800 B/s |
| RPN Stack | 800 B | 10 Hz avg | 8 KB/s |
| Math Cores | 1.4 KB | 0.5 Hz | 700 B/s |
| Galaxy Update | 2 KB | 5 Hz | 10 KB/s |
| Ternary | 1.5 KB | 2 Hz avg | 3 KB/s |
| House | 1.2 KB | 0.2 Hz | 240 B/s |
| SleepTime | 2 KB | 0.01 Hz | 20 B/s |
| Action Buffer | 500 B | 10 Hz avg | 5 KB/s |
| **Total continuous** | | | **~28 KB/s** |

Plus one-time Galaxy init (~940 KB) and on-demand Ontology/Inspector requests. Well within local WebSocket capacity.

---

## Implementation Notes for Christoph

1. **Start with Views 01, 02, 09** — These have the smallest data structures and most direct engine sources. Good for establishing the WS diagnostic protocol.

2. **View 04 (Galaxy Starfield) is the hardest** — 117K stars requires binary transport + WebGL instanced points. The existing `GalaxyPodProjector` and `GraphApp` are good starting points but need to scale from 4096 to 117K.

3. **Mock data first** — Christoph can build all 10 views with mock data while Claude+Codex adds daemon-side emission. The TypeScript interfaces above define the contract.

4. **Existing apps to enhance vs. new apps:**
   - Enhance: StatsApp → View 01, RpnApp → View 02, DoorsApp → View 06, ContentApp → Views 07+08, ControlApp → View 09
   - New: View 03 (MathCoreApp), View 05 (TernaryApp), View 10 (ActionBufferApp)

5. **Memory Tablet canvas is 768×480** — Views must be legible at this resolution. Use the overlay (HTML) for detailed inspection, canvas for at-a-glance monitoring.
