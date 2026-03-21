# Codex Prompt: Four Architectural Corrections — Persistent House, Full Knowledge, Jarvis Coordinator, GPU Saturation

**Date:** March 21, 2026
**Architecture:** Claude (spec) → Codex (implementation)
**Priority:** CRITICAL — these are foundational violations that must be fixed before competition prep

---

## Context

Daniel identified four architectural violations during the validation run. These are not bugs — they're design-level corrections to how the system operates.

**GPU evidence during active benchmark:**
- SM utilization: 0-6% (average ~1%) over 120 seconds
- Memory bandwidth: 0%
- VRAM used: 1,200 MiB of 12,288 MiB (9.7%)
- 88 PTX kernels available, barely tickled

---

## Correction 1: ALL Stars Must Always Be Loaded

### The Problem

`scripts/ingest_meaning_layer.py` selects 35,579 out of 117,497 available stars. This is WRONG.

Daniel: "Why aren't all stars loaded, only 35k from a 117k library? Wrong! All knowledge is always loaded, this is the game — always loaded, always persistent."

### The Fix

Remove ALL quantity filters/caps in the meaning layer ingestion pipeline. The sovereign pipeline handles knowledge volume via LOD + Frustum Culling on GPU — that IS the knowledge management mechanism.

**What to keep:** Quality filters only:
- Stopword deduplication (remove exact duplicates)
- Foundation dedup (content-based deduplication)
- Malformed entry removal (missing required fields)

**What to REMOVE:**
- `min_languages` filter (was filtering to only entries with 5+ languages)
- Any `max_stars` or `limit` caps
- Any `keyword_filter` that removes valid entries
- Any sampling or selection logic

**Files to modify:**
- `scripts/ingest_meaning_layer.py` — remove quantity filters
- `scripts/run_enriched_benchmarks.py` — if it passes `max_stars` or filter args, remove them

**After fix:** All 117,497 stars should load. Log should show `117,497 / 117,497` (minus only quality dedup).

---

## Correction 2: Bootstrap Is a Hack — Transmute to Persistent House

### The Problem

`foundational_operations_bootstrap.py` loads ~170 Drawing entries + ~94 Grammar entries + math anchors EVERY TIME the system starts. This violates the core principle: K3D is a persistent, always-evolving system.

Daniel: "These should be transmuted to knowledge in the House — not loaded every run. At sleep-time compute, so the ingestion happens once and consolidates. On the next run, we only load the House."

### The Architecture

```
COLD START (first ever run):
  bootstrap.py → populate Galaxy → sleep-time → consolidate to House → save state

EVERY SUBSEQUENT RUN:
  load House state from disk → Galaxy is populated → ready to go
  (bootstrap.py NOT called — knowledge already in House)

SLEEP-TIME:
  consolidate new discoveries → update House state → save to disk
  (new knowledge from benchmarks, ingestion, etc. becomes permanent)
```

### The Fix

#### Step 1: Galaxy State Serialization

Add to `Knowledgeverse`:

```python
HOUSE_STATE_PATH = Path("/K3D/Knowledge3D.local/house/galaxy_state.bin")

def save_house_state(self) -> None:
    """Persist current Galaxy state to disk (House = permanent knowledge)."""
    state = {
        "galaxies": self.galaxy_manager.serialize_all(),
        "version": self._state_version(),
        "timestamp": time.time(),
        "entry_count": self.galaxy_manager.total_entries(),
    }
    self.HOUSE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Use pickle or msgpack for speed — this is boot-time, not hot-path
    with open(self.HOUSE_STATE_PATH, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_house_state(self) -> bool:
    """Load persistent House state. Returns True if loaded, False if cold start."""
    if not self.HOUSE_STATE_PATH.exists():
        return False
    with open(self.HOUSE_STATE_PATH, "rb") as f:
        state = pickle.load(f)
    self.galaxy_manager.deserialize_all(state["galaxies"])
    return True
```

#### Step 2: Boot Sequence Change

Current:
```python
def __init__(self):
    self._init_galaxies()
    self._run_bootstrap()        # ALWAYS runs — hack
    self._load_meaning_layer()   # ALWAYS runs — slow
```

New:
```python
def __init__(self):
    self._init_galaxies()
    if not self.load_house_state():
        # Cold start: bootstrap + ingest + save
        self._run_bootstrap()
        self._load_meaning_layer()
        self.save_house_state()
    # else: House loaded from disk — ready immediately
```

#### Step 3: Sleep-Time Saves House

After sleep-time consolidation, call `save_house_state()` to persist the evolved Galaxy:

```python
def sleep_time_consolidation(self):
    # ... existing consolidation logic ...
    self._strengthen_correct_paths()
    self._weaken_incorrect_paths()
    self._materialize_patterns()
    # Persist evolved state
    self.save_house_state()
```

#### Step 4: Incremental Updates

When new knowledge is ingested (math rules, meaning layer updates, etc.), it goes into Galaxy AND persists to House:

```python
def ingest_new_knowledge(self, entries):
    for entry in entries:
        self.galaxy_manager.add_entry(entry["galaxy"], entry)
    # Save updated state — knowledge is now permanent
    self.save_house_state()
```

### Expected Impact

- **Boot time:** From ~30-60s (bootstrap + ingest) to <5s (load binary state)
- **Knowledge persistence:** Discoveries from benchmarks survive across runs
- **Sleep-time consolidation:** Actually permanent — consolidated paths stay consolidated
- **System identity:** K3D remembers everything, always. This IS the persistent mind.

---

## Correction 3: TRM Needs Internal "Jarvis" Coordinator Specialist

### The Problem

Currently, specialists (math, visual, grammar, chat) operate in relative isolation. There's no persistent coordinator tracking what all specialists are doing, delegating efficiently, or maintaining coherent multi-specialist workflows.

Daniel: "TRM needs an internal 'personal Jarvis' specialist acting like a secretary that delegates and keeps track of the other specialists' work. Always on with the main TRM."

### The Architecture

**Critical clarification from Daniel:** Jarvis does NOT decide. TRM main model is the brain. Jarvis is the BRIDGE between TRM and the swarm — a secretary that keeps track, organizes, and presents so TRM can see the FULL picture and make connections that no individual swarm worker can.

```
TRM Main Model (THE BRAIN — decides, connects, reasons)
     │
     │  commands
     ▼
┌─────────────────────────────────────┐
│  JARVIS (Secretary / Swarm Bridge)   │  ← Always on alongside TRM
│                                       │
│  Commanded by: TRM main model         │
│  Bridges to: all swarm workers        │
│  Tracks: what every worker produces   │
│  Reports: organized status to TRM     │
│  Enables: TRM to see full picture     │
│           and make cross-connections   │
│           no single worker can see     │
│                                       │
│  Swarm Group A (math focus):                │
│  ┌──────┐ ┌──────┐ ┌──────┐                │
│  │ w1   │ │ w2   │ │ w3   │ ... w9         │  ← 9 workers
│  └──────┘ └──────┘ └──────┘                │
│                                              │
│  Swarm Group B (visual focus):               │
│  ┌──────┐ ┌──────┐ ┌──────┐                │
│  │ w1   │ │ w2   │ │ w3   │ ... w9         │  ← 9 workers
│  └──────┘ └──────┘ └──────┘                │
│                                              │
│  Swarm Group C..N (as many as GPU allows):   │
│  ┌──────┐ ┌──────┐ ┌──────┐                │
│  │ w1   │ │ w2   │ │ w3   │ ... w9         │  ← 9 workers each
│  └──────┘ └──────┘ └──────┘                │
└──────────────────────────────────────────────┘
     │
     │  organized briefs back to TRM
     ▼
TRM Main Model (sees ALL worker output from ALL swarm groups,
                makes overall connections, enhances the entire chain)
```

**The key insight:** Without Jarvis, TRM only sees the halting-gate winner (1 result from 9 workers in 1 swarm group). With Jarvis, TRM sees ALL intermediate work from ALL workers across ALL swarm groups — enabling cross-pollination, contradiction detection, and meta-reasoning that no individual worker could achieve. Jarvis makes TRM's "superdotados" model actually work: parallel cognitive channels become coherent because the main model can connect them.

**Critical: N swarm groups, not just one.** Jarvis can spawn MULTIPLE nine-chain swarm groups in parallel — as many as the hardware allows and the task demands. Current GPU utilization is ~1% with a single 9-worker swarm. A RTX 3070 (5,888 cores, 12 GB VRAM) can run dozens of swarm groups simultaneously. An L4×4 (29,696 cores, 96 GB VRAM) can run hundreds. Jarvis manages the fleet:

- **Simple query:** 1 swarm group, 9 workers → fast answer
- **Complex math:** 2-3 swarm groups (algebra + geometry + number theory focus) → 18-27 workers
- **ARC-AGI-3 interactive:** 4+ swarm groups (visual + spatial + symbolic + exploration) → 36+ workers
- **Full benchmark sprint:** Scale to GPU capacity, Jarvis tracks all groups

Jarvis monitors live GPU utilization and scales swarm count dynamically:
```python
def _jarvis_determine_swarm_count(self, task_complexity: float) -> int:
    """Scale swarm groups based on task demand and GPU headroom."""
    gpu_utilization = self.cranium.get_sm_utilization()
    vram_available = self.cranium.get_vram_free()
    per_swarm_vram = self._estimate_swarm_vram_cost()

    max_by_vram = int(vram_available / per_swarm_vram)
    max_by_compute = max(1, int((100 - gpu_utilization) / 10))  # ~10% per swarm group

    desired = max(1, int(task_complexity * 5))  # 1-5+ groups based on complexity
    return min(desired, max_by_vram, max_by_compute)
```

This is how we go from 1% to 50%+ GPU utilization — not by making one swarm faster, but by running N swarms in parallel under Jarvis's coordination. The GPU is a parallel machine; we must USE it as one.

### The Fix

#### Step 1: Jarvis Specialist in Galaxy

Jarvis lives as a permanent Galaxy entry (not bootstrap — House resident):

```python
{
    "id": "specialist_jarvis_coordinator",
    "name": "Jarvis Internal Coordinator",
    "domain": "tool",
    "category": "meta_specialist",
    "layer": 4,  # Meta-Rules layer — coordinates other layers
    "always_active": True,
    "description": "TRM's internal secretary — bridges TRM main model to swarm workers, tracks all worker output so TRM can make overall connections.",
    "metadata": {
        "role": "secretary_bridge",
        "commanded_by": "trm_main_model",
        "bridges_to": "nine_chain_swarm",
        "tracks": ["math", "visual", "grammar", "chat", "arc_interactive"],
        "responsibilities": [
            "receive_dispatch_from_trm",
            "distribute_work_to_swarm_workers",
            "track_all_worker_intermediate_output",
            "detect_cross_worker_contradictions",
            "organize_worker_results_for_trm",
            "present_full_picture_to_trm",
            "maintain_swarm_state_registry",
        ],
    },
}
```

#### Step 2: Jarvis in the Game Loop

Jarvis sits in the TRM game tick between dispatch and decision:

```python
def trm_game_tick(self, task):
    # 1. TRM perceives the task
    perception = self._trm_perceive(task)

    # 2. TRM commands Jarvis: "send this to the swarm"
    jarvis_dispatch = self._trm_command_jarvis(perception)

    # 3. Jarvis dispatches to swarm workers (parallel)
    worker_outputs = self._jarvis_dispatch_to_swarm(jarvis_dispatch)

    # 4. Jarvis tracks ALL intermediate output (not just winners)
    jarvis_brief = self._jarvis_compile_brief(worker_outputs)
    # brief contains: all worker results, contradictions, agreements,
    # partial progress, confidence distributions, reasoning traces

    # 5. TRM receives the FULL brief from Jarvis
    # THIS is the key: TRM sees everything, not just the halting winner
    # TRM can now make connections no single worker could:
    #   - "Worker 3's partial result + Worker 7's approach = full answer"
    #   - "Workers 1,4,5 agree but Worker 2 found a contradiction"
    #   - "No worker solved it, but combining their partial traces..."
    decision = self._trm_decide(jarvis_brief)

    # 6. TRM may command Jarvis for another round
    #    (refine based on what workers produced)
    if decision.get("needs_refinement"):
        refined_dispatch = self._trm_refine_command(jarvis_brief, decision)
        worker_outputs_2 = self._jarvis_dispatch_to_swarm(refined_dispatch)
        jarvis_brief_2 = self._jarvis_compile_brief(worker_outputs_2)
        decision = self._trm_decide(jarvis_brief_2)

    # 7. TRM emits final answer
    return decision
```

#### Step 3: What Jarvis Tracks (The Brief)

```python
{
    "workers": {
        "w1_math": {
            "status": "completed",
            "result": ...,
            "confidence": 0.82,
            "reasoning_trace": [...],
            "partial_progress": [...],
        },
        "w2_visual": {
            "status": "completed",
            "result": ...,
            "confidence": 0.45,
            "reasoning_trace": [...],
        },
        # ... all 9 workers
    },
    "agreements": [("w1", "w4", "w7")],      # These workers agree
    "contradictions": [("w1", "w2")],          # These workers disagree
    "highest_confidence": "w1_math",
    "novel_partial": "w5_grammar",             # Didn't finish but found something new
    "cross_connections": [                      # Jarvis detected potential links
        {"w3_partial + w6_partial": "might combine to full answer"},
    ],
}
```

This brief is what TRM sees. TRM can then:
- Accept the majority answer
- Combine partial results from different workers
- Spot a contradiction and investigate
- Command Jarvis to send a refinement round to specific workers
- Recognize that a "failed" worker actually found a useful sub-result

#### Step 4: Jarvis Learns via Sleep-Time

During sleep-time, Jarvis consolidates:
- Which worker combinations produce the best results for which task types
- Which workers are redundant for simple queries (skip them, save GPU time)
- Optimal dispatch patterns (parallel all 9? or sequential 3→6 based on first results?)
- Cross-connection patterns that TRM frequently uses (pre-compute them next time)

This is the "secretary learning what the boss needs before being asked" pattern — but the BOSS (TRM) still makes all decisions. Jarvis just gets better at presenting the right information.

---

## Correction 4: GPU Saturation — Use the Metal

### The Problem

GPU SM utilization during active benchmark: 0-6% (average ~1%). The sovereign engine has 88 PTX kernels and 5,888 CUDA cores sitting idle.

Root causes:
1. **Python orchestration overhead:** Most time is spent in Python between GPU calls
2. **Sequential kernel dispatch:** Kernels fire one at a time, not pipelined
3. **Small work units:** Each kernel processes one query, returns to Python, then next kernel
4. **VRAM underuse:** Only 1.2 GB of 12 GB used — Galaxy data fits but isn't streamed

### The Fix Path (Phase D Acceleration)

This is the Phase D target (`knowledgeverse.py` 4000→200 lines), but we can start now:

#### Step 1: Batch Kernel Dispatch

Instead of calling kernels one-at-a-time per question, batch N questions and dispatch to GPU together:

```python
# Current: serial (1% GPU)
for question in questions:
    result = self.cranium.execute_kernel("math_solve", question)

# Better: batched (target 30%+ GPU)
batch = [q for q in questions[:BATCH_SIZE]]
results = self.cranium.execute_kernel_batch("math_solve", batch)
```

This keeps the GPU busy while Python prepares the next batch.

#### Step 2: Kernel Pipelining

The composed head pipeline (Morton → LED-A* → Frustum → LOD → Swarm → Halting) should be a FUSED pipeline on GPU, not 6 separate Python→GPU→Python round trips:

```
Current:
  Python → Morton GPU → Python → LEDA* GPU → Python → Frustum GPU → Python → ...
  (6 round trips per query = GPU idle 90% of the time)

Target:
  Python → [Morton → LEDA* → Frustum → LOD → Swarm → Halting] GPU → Python
  (1 round trip per query = GPU stays busy)
```

This is what `trm_step_fused.ptx` is SUPPOSED to be — the fused game tick.

#### Step 3: Async Overlap

While GPU processes batch N, Python prepares batch N+1:

```python
# Launch batch N on GPU (non-blocking)
future = self.cranium.execute_kernel_batch_async("pipeline", batch_n)
# While GPU works, prepare batch N+1 in Python
batch_n1 = self._prepare_next_batch(questions, offset)
# Wait for GPU result
results = future.get()
```

#### Step 4: VRAM Utilization

With all 117k stars loaded (Correction 1) and House state in VRAM (Correction 2), VRAM usage should rise from 1.2 GB to 3-4 GB. Combined with batched dispatch, this puts the GPU closer to its design capacity.

### GPU Utilization Targets

| Phase | SM Target | Path |
|-------|-----------|------|
| Current | ~1% | Serial Python→GPU round trips |
| After Correction 1+2 | ~5% | More data in VRAM, same dispatch |
| After batch dispatch | ~20% | Batched kernels, less Python overhead |
| After kernel pipelining | ~40% | Fused composed head on GPU |
| Phase D complete | ~60%+ | TRM game loop on GPU, Python = I/O only |

---

## Execution Order

1. **Correction 1 (immediate):** Remove star loading caps — load all 117k
2. **Correction 2 (this sprint):** Galaxy state serialization + persistent House boot
3. **Correction 3 (this sprint):** Jarvis coordinator specialist in execute_task
4. **Correction 4 (progressive):** Batch dispatch first, then pipeline fusion

Corrections 1+2 directly benefit the validation run and competition prep.
Correction 3 directly benefits ARC-AGI-3 (multi-specialist coordination for interactive tasks).
Correction 4 is the ongoing Phase D target.

---

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Stars loaded | 35,579 / 117,497 | 117,497 / 117,497 |
| Boot time (warm) | ~30-60s | <5s (load House state) |
| Bootstrap re-run | Every startup | Only cold start |
| GPU SM utilization | ~1% | >20% (batch dispatch) |
| VRAM usage | 1.2 GB / 12 GB | 3-4 GB / 12 GB |
| Specialist coordination | Single specialist per query | Jarvis dispatches N specialists |
| Knowledge persistence | Lost between runs | Permanent in House |

---

## Sovereignty Notes

- All corrections STRENGTHEN sovereignty:
  - More knowledge in VRAM = more sovereign reasoning
  - Less Python overhead = more GPU execution
  - Jarvis coordinator = smarter specialist dispatch ON GPU
  - Persistent House = no external re-ingestion dependency
- No new Python in hot path — corrections move work FROM Python TO GPU
- Bootstrap transmutation is ingestion-path (runs once, then House takes over)
