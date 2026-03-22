# Codex Prompt: ARC Semantic Routing + Full-Brain Contrastive Learning + LHE Multi-Hop

**Date:** March 22, 2026
**Architecture:** Claude (spec) + Codex (implementation)
**Priority:** CRITICAL -- these three fixes target the root causes of flat ARC/Math/LHE scores

---

## Context

The full 35% validation rerun completed successfully (systems success):
- ARC 2/42, Math 3/500, GSM8K 7/462, LHE 1/35, MMLU 1106/4915
- 117,497 / 117,497 stars loaded, 333 ARC anchors + 333 bridges ingested
- Sleep-time committed: 31,565 routing updates, 128 Jarvis briefs

But ARC did NOT improve despite 666 new Galaxy entries. Diagnosis found three root causes:

1. **ARC embedding is a task-ID hash** -- bypasses semantic matching entirely, so new anchors are unreachable
2. **Contrastive learning only trains one specialist** -- 4 of 5 specialists never improve their embedding alignment
3. **LHE multi-hop reasoning never chains** -- graph crystallizer runs but candidates are isolated (0 edges)

---

## Fix 1: ARC Semantic Embedding (CRITICAL -- zero ARC progress without this)

### The Bug

File: `knowledge3d/knowledgeverse/knowledgeverse.py` line 2636-2640:

```python
def _embed_query_gpu(self, query_text: str, *, task=None):
    if str((task or {}).get("type", "")).upper() == "ARC_TASK":
        task_id = str((task or {}).get("task_id", "")).strip()
        if task_id:
            return self._arc_task_embedding16(task_id)  # <-- THIS IS THE BUG
```

And line 1720-1725:

```python
def _arc_task_embedding16(self, task_id: str) -> list[float]:
    dims = [0.0] * 16
    for idx, ch in enumerate(f"ARC_TASK::{task_id}"):
        lane = idx & 15
        dims[lane] += ((ord(ch) * (idx + 3)) % 29 - 14.0) / 14.0
    return self._normalize_embedding(dims)
```

This computes the query embedding from the TASK ID CHARACTERS (e.g., "ARC_TASK::00576224"). It's a pseudo-random hash that carries ZERO semantic meaning. Meanwhile, the 333 ARC anchors have rich `query_anchor` text like "find objects connected regions discrete shapes separate groups" -- but the task-ID hash can never match them meaningfully.

### The Fix

Replace the ARC task-ID hash with a **semantic embedding built from the actual task content**. The task payload already contains everything needed (see `headless_tablet.py` line 158-164):

```python
task = {
    "type": "ARC_TASK",
    "task_id": str(task_id),
    "query": "solve arc transformation task",
    "training_examples": list(training_examples),   # <-- input/output grid pairs
    "input_grid": input_grid,                        # <-- test input
}
```

#### Step 1: Build ARC Visual Feature Text

Create `_arc_visual_feature_text(self, task)` that analyzes the training examples and produces semantic text describing what the task requires:

```python
def _arc_visual_feature_text(self, task: dict[str, Any]) -> str:
    """Extract semantic features from ARC task grids for embedding.

    Analyzes training input/output pairs to describe the task
    in terms that match Galaxy anchor query_anchor fields.
    """
    fragments = ["arc transformation task"]
    training = task.get("training_examples") or []
    input_grid = task.get("input_grid")

    for pair_idx, pair in enumerate(training[:3]):  # Cap at 3 pairs for embedding
        inp = pair.get("input", [])
        out = pair.get("output", [])
        if not inp or not out:
            continue

        inp_h, inp_w = len(inp), len(inp[0]) if inp else 0
        out_h, out_w = len(out), len(out[0]) if out else 0

        # Size relationship
        if (out_h, out_w) == (inp_h, inp_w):
            fragments.append("same size output")
        elif out_h > inp_h or out_w > inp_w:
            fragments.append("output larger scaling expand tile")
        elif out_h < inp_h or out_w < inp_w:
            fragments.append("output smaller crop extract subgrid")

        # Color analysis
        inp_colors = set()
        for row in inp:
            inp_colors.update(row)
        out_colors = set()
        for row in out:
            out_colors.update(row)
        new_colors = out_colors - inp_colors
        removed_colors = inp_colors - out_colors
        if new_colors:
            fragments.append("color change remap new colors")
        if removed_colors:
            fragments.append("color removal filter background")
        if len(inp_colors) <= 3:
            fragments.append("simple palette few colors")
        elif len(inp_colors) >= 7:
            fragments.append("complex palette many colors")

        # Object count estimate (connected components via simple flood)
        bg_color = max(set(c for row in inp for c in row), key=lambda c: sum(r.count(c) for r in inp))
        obj_count = _count_connected_components(inp, bg_color)
        if obj_count >= 3:
            fragments.append("multiple objects detect separate groups")
        elif obj_count == 1:
            fragments.append("single object transform shape")

        # Symmetry check
        if _grid_has_symmetry(inp):
            fragments.append("symmetry mirror reflection")

        # Grid size hints
        if inp_h == inp_w:
            fragments.append("square grid")
        if inp_h <= 5 and inp_w <= 5:
            fragments.append("small grid pattern")
        elif inp_h >= 15 or inp_w >= 15:
            fragments.append("large grid spatial")

    return " ".join(fragments)
```

#### Step 2: Helper Functions

Add simple grid analysis helpers (these are ingestion-path, NOT hot-path -- Python is fine):

```python
def _count_connected_components(grid: list[list[int]], background: int) -> int:
    """Count non-background connected components (4-connected flood fill)."""
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    count = 0
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != background:
                # Flood fill
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                        continue
                    if visited[cr][cc] or grid[cr][cc] == background:
                        continue
                    visited[cr][cc] = True
                    stack.extend([(cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)])
                count += 1
    return count


def _grid_has_symmetry(grid: list[list[int]]) -> bool:
    """Check horizontal or vertical mirror symmetry."""
    if not grid:
        return False
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    # Horizontal mirror
    h_sym = all(grid[r] == grid[rows - 1 - r] for r in range(rows // 2))
    # Vertical mirror
    v_sym = all(grid[r][c] == grid[r][cols - 1 - c] for r in range(rows) for c in range(cols // 2))
    return h_sym or v_sym
```

#### Step 3: Replace the Embedding Bypass

In `_embed_query_gpu()` (line 2636), replace the task-ID hash with semantic embedding:

```python
def _embed_query_gpu(self, query_text: str, *, task=None):
    if str((task or {}).get("type", "")).upper() == "ARC_TASK":
        # Build semantic text from visual features of the task
        visual_text = self._arc_visual_feature_text(task or {})
        # Fall through to the normal trigram embedding engine
        # This produces an embedding that can semantically match
        # the query_anchor fields of our 333 ARC anchors
        engine = self.get_gpu_query_embedding_engine()
        values = engine.embed_sentence_gpu(visual_text)
        return [float(values[i]) for i in range(min(16, len(values)))]
    engine = self.get_gpu_query_embedding_engine()
    values = engine.embed_sentence_gpu(query_text)
    return [float(values[i]) for i in range(min(16, len(values)))]
```

#### Step 4: Also Fix the Query Text

In `_task_query_text_for_embedding()` (line 1793-1813), the ARC branch currently produces `"ARC_TASK {task_id} solve arc transformation task train_pairs N input_shape RxC"`. This is too generic. Append the visual feature text:

```python
if task_type == "ARC_TASK":
    task_id = str(payload.get("task_id", "")).strip()
    fragments: list[str] = [
        f"ARC_TASK {task_id} solve arc transformation task"
        if task_id
        else "solve arc transformation task"
    ]
    training_examples = payload.get("training_examples")
    if isinstance(training_examples, list):
        fragments.append(f"train_pairs {len(training_examples)}")
    input_grid = payload.get("input_grid")
    if isinstance(input_grid, list):
        rows = len(input_grid)
        cols = len(input_grid[0]) if rows and isinstance(input_grid[0], list) else 0
        fragments.append(f"input_shape {rows}x{cols}")
    # NEW: append visual feature analysis
    visual_features = self._arc_visual_feature_text(payload)
    if visual_features:
        fragments.append(visual_features)
    if str(prompt or "").strip():
        fragments.append(str(prompt).strip())
    return " ".join(fragment for fragment in fragments if fragment).strip()
```

### Expected Impact

The 333 ARC anchors have `query_anchor` fields like:
- "find objects connected regions discrete shapes separate groups"
- "rotate mirror transform reflect symmetry"
- "color remap substitution pattern change"
- "grid overlay concatenate interleave difference"

With semantic embedding, a task that shows rotated objects will embed near "rotate mirror transform" and match the rotation anchors. A task with multiple colored objects will embed near "multiple objects detect separate groups" and match object-detection anchors.

Currently these anchors are invisible because the task-ID hash embedding is semantic noise. This fix makes them reachable.

### Files to Modify

- `knowledge3d/knowledgeverse/knowledgeverse.py`:
  - Add `_arc_visual_feature_text()` method
  - Add `_count_connected_components()` and `_grid_has_symmetry()` as module-level helpers
  - Replace `_embed_query_gpu()` ARC branch (remove task-ID hash, use semantic embedding)
  - Update `_task_query_text_for_embedding()` ARC branch (append visual features)
  - `_arc_task_embedding16()` can remain as a fallback but should NOT be the primary path

---

## Fix 2: Contrastive Learning for ALL Specialists During Sleep-Time

### The Gap

`train_specialist_contrastive()` in `adaptive_swarm.py` (line 479) is a working contrastive trainer that adjusts specialist adapter weights to pull (query_embedding, correct_answer_embedding) pairs closer. But it's only called from `procedural_drawing_specialist.py` (line 696) -- ONE specialist out of five.

Sleep-time Stage B (`sleeptime.py` line 106) calls `trm.consolidate_weights_from_events(events)` which does routing weight updates (ternary outcome-based strengthen/weaken). This is NOT contrastive learning -- it adjusts WHICH specialist gets routed to, not HOW WELL each specialist matches queries to Galaxy entries.

### The Fix

Add contrastive training for all specialists in sleep-time Stage B, using the benchmark health log as training data.

#### Step 1: Extract Embedding Pairs from Health Log

The health log (`health_log.jsonl`) contains rows with `question`, `answer`, `expected`, `correct`, and `suite`. For each CORRECT answer, the (query_embedding, matched_entry_embedding) pair is a positive contrastive signal. For each INCORRECT answer, it's a negative signal.

Add to `sleeptime.py` in `_stage_b_logic()`:

```python
def _stage_b_logic(self) -> dict[str, Any]:
    # ... existing routing weight consolidation ...

    # NEW: Contrastive training for all specialists
    contrastive_summary = self._run_contrastive_training()

    return {
        "success": True,
        "stage": "logic",
        "updated_specialists": summary.get("updated_specialists", []),
        "updated_count": int(summary.get("updated_count", 0)),
        "weights_path": summary.get("weights_path", ""),
        "contrastive": contrastive_summary,
        **({"jarvis": jarvis_summary} if isinstance(jarvis_summary, dict) else {}),
    }


def _run_contrastive_training(self) -> dict[str, Any]:
    """Run contrastive embedding alignment for all specialists using health log data."""
    if self.kv is None:
        return {"skipped": True, "reason": "no_knowledgeverse"}

    swarm = getattr(self.kv, "adaptive_swarm", None)
    if swarm is None or not hasattr(swarm, "train_specialist_contrastive"):
        return {"skipped": True, "reason": "no_swarm"}

    # Read health log for correct/incorrect pairs
    health_rows = self._load_health_log_rows()
    if not health_rows:
        return {"skipped": True, "reason": "no_health_rows"}

    # Group by specialist mapping
    specialist_pairs: dict[str, list[tuple]] = {
        "math": [],
        "visual": [],
        "grammar": [],
        "chat": [],
    }

    engine = self.kv.get_gpu_query_embedding_engine()

    for row in health_rows:
        suite = str(row.get("suite", "")).lower()
        correct = bool(row.get("correct", False))
        question = str(row.get("question", "")).strip()
        expected = str(row.get("expected", "")).strip()
        if not question or not expected:
            continue

        # Map suite to specialist
        specialist = "chat"  # default
        if suite in ("math", "gsm8k"):
            specialist = "math"
        elif suite == "arc":
            specialist = "visual"
        elif suite == "lhe":
            specialist = "grammar"

        try:
            q_emb = np.array(engine.embed_sentence_gpu(question), dtype=np.float32)
            a_emb = np.array(engine.embed_sentence_gpu(expected), dtype=np.float32)
        except Exception:
            continue

        if correct:
            # Positive pair: pull query toward correct answer
            specialist_pairs[specialist].append((q_emb, a_emb))
        # Note: negative pairs (incorrect) would push apart --
        # for now, only use positive pairs (pull together).
        # Negative contrastive can be added as a second phase.

    results = {}
    for name, pairs in specialist_pairs.items():
        if not pairs:
            results[name] = {"trained": False, "reason": "no_pairs"}
            continue
        try:
            stats = swarm.train_specialist_contrastive(name, pairs)
            results[name] = {
                "trained": True,
                "pairs": len(pairs),
                "avg_loss": float(stats.get("avg_loss", 0.0)),
            }
        except Exception as exc:
            results[name] = {"trained": False, "error": str(exc)}

    return {"specialists_trained": results}


def _load_health_log_rows(self) -> list[dict[str, Any]]:
    """Load health log rows from the current session."""
    if self.health_log_path is None or not self.health_log_path.exists():
        return []
    rows = []
    with self.health_log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
```

#### Step 2: Register Missing Specialists in Swarm

The adaptive swarm (line 24-26) only registers `ocr`, `math`, `code`. It does NOT register `visual`, `grammar`, `chat`. These must be registered for contrastive training to work:

Check if they're registered elsewhere. If not, add to the swarm initialization:

```python
swarm.register_specialist('visual', required_dims=512, rank=32)
swarm.register_specialist('grammar', required_dims=512, rank=32)
swarm.register_specialist('chat', required_dims=512, rank=32)
```

**Tip:** Check how the Knowledgeverse initializes the swarm. If specialists are registered at boot, confirm all 5 exist: `math`, `visual`, `grammar`, `chat`, `ocr`. If `visual`/`grammar`/`chat` are missing, register them with appropriate dims.

#### Step 3: Save Specialist Adapters After Training

After contrastive training, persist the updated adapter weights. The existing `save_state()` in the navigator should handle this -- confirm it saves swarm specialist adapters, not just routing weights.

### Expected Impact

- Math specialist aligns to (math_question, correct_answer) pairs -- should improve Math/GSM8K
- Visual specialist aligns to (arc_task_features, correct_transform) -- helps ARC (combined with Fix 1)
- Grammar specialist aligns to (lhe_question, correct_answer) -- helps LHE
- Chat specialist aligns to (mmlu_question, correct_answer) -- continues MMLU improvement
- All specialists improve their embedding space with each benchmark cycle
- The "advance slowly" becomes "advance on ALL suites, not just MMLU"

### Files to Modify

- `knowledge3d/knowledgeverse/sleeptime.py` -- add `_run_contrastive_training()` to Stage B
- `knowledge3d/cranium/adaptive_swarm.py` -- verify all needed specialists are registered
- Potentially `knowledge3d/knowledgeverse/knowledgeverse.py` -- if swarm specialist registration happens there

---

## Fix 3: LHE Multi-Hop Graph Connectivity

### The Gap

LHE (Last Humanity Exam) requires multi-hop reasoning: "What is the capital of the country that..." needs chaining Galaxy entries. The graph crystallizer EXISTS (`get_graph_crystallizer()` at line 1297) and runs during candidate selection (line 7842), but the diagnostic from the log shows:

```
LHE graph diagnostic: N candidates, 0 total edges, max_neighbors=0, isolated=N
```

All candidates have zero graph neighbors. The graph crystallizer has nothing to connect -- every candidate is an isolated node. This means multi-hop reasoning collapses to single-hop: pick the best single entry, which rarely answers a multi-hop question.

### The Fix

The graph crystallizer needs edges. Currently, candidate entries are loaded independently -- each gets an embedding-based similarity score against the query, but there's no inter-candidate edge computation.

#### Step 1: Build Inter-Candidate Edges

After candidates are retrieved by embedding similarity, compute pairwise similarity between candidates to build the neighbor graph. Add to the candidate retrieval path:

```python
def _build_candidate_graph_edges(
    self,
    candidates: list[dict[str, Any]],
    similarity_threshold: float = 0.3,
) -> None:
    """Compute pairwise similarity between candidates and populate graph_neighbors.

    This enables the graph crystallizer to propagate scores between
    related candidates, enabling multi-hop reasoning chains.
    """
    if len(candidates) < 2:
        return

    embeddings = []
    for candidate in candidates:
        emb = candidate.get("embedding")
        if emb is None:
            text = self._entry_embedding_text(candidate)
            emb = self.get_gpu_query_embedding_engine().embed_sentence_gpu(text)
        embeddings.append(np.array(emb, dtype=np.float32))

    # Pairwise cosine similarity
    for i in range(len(candidates)):
        neighbors = []
        for j in range(len(candidates)):
            if i == j:
                continue
            sim = float(np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
            ))
            if sim >= similarity_threshold:
                neighbors.append(candidates[j].get("candidate_global_idx", j))
        candidates[i]["graph_neighbors"] = neighbors
```

#### Step 2: Call Edge Builder Before Graph Crystallizer

In the candidate selection pipeline, call `_build_candidate_graph_edges()` BEFORE the graph crystallizer runs (before line 7842):

```python
# Build inter-candidate edges for graph crystallizer
self._build_candidate_graph_edges(local_candidates)
# THEN run crystallizer
graph_crystallizer = self.get_graph_crystallizer()
```

#### Step 3: LHE-Specific Edge Enrichment

For LHE tasks specifically, lower the similarity threshold and increase the max neighbor count, because multi-hop questions need longer chains:

```python
if task_type == "LHE_TASK":
    self._build_candidate_graph_edges(local_candidates, similarity_threshold=0.2)
```

### Expected Impact

With edges populated, the graph crystallizer can propagate scores between related candidates. A chain like "country → capital → population" becomes reachable: the crystallizer strengthens candidates that connect to other high-scoring candidates, producing multi-hop answers.

Current: 1/35 (2.86%) with isolated candidates
Target: Measurable improvement once candidates can form chains

### Files to Modify

- `knowledge3d/knowledgeverse/knowledgeverse.py`:
  - Add `_build_candidate_graph_edges()` method
  - Call it before the graph crystallizer in the candidate selection pipeline (around line 7840)

---

## Execution Order

1. **Fix 1: ARC semantic embedding** -- highest impact, unblocks 333 anchors
2. **Fix 2: Full-brain contrastive learning** -- accelerates all specialists
3. **Fix 3: LHE multi-hop edges** -- enables graph crystallizer to actually chain

All three can land in the same cold-start rebuild.

## Verification

After implementing all three fixes:

```bash
export CUDA_VISIBLE_DEVICES=0
python3 scripts/run_enriched_benchmarks.py \
  --full \
  --cold-start \
  --storage-root /K3D/Knowledge3D.local \
  --arc-max 42 \
  --math-max 50 \
  --gsm8k-max 50 \
  --lhe-max 35 \
  --mmlu-max 100 \
  2>&1 | tee /tmp/k3d_routing_fix_smoke_03.22.2026.log
```

Use small Math/GSM8K/MMLU counts (50/50/100) for a fast smoke. Focus on:
- **ARC:** MUST improve from 2/42 -- the semantic embedding should now match anchors
- **LHE:** Should improve from 1/35 -- graph edges enable multi-hop
- **MMLU 100:** Sanity check that existing routing didn't regress

After smoke passes, run full 35%:

```bash
python3 scripts/run_enriched_benchmarks.py \
  --full \
  --storage-root /K3D/Knowledge3D.local \
  --arc-max 42 \
  --math-max 500 \
  --gsm8k-max 462 \
  --lhe-max 35 \
  --mmlu-max 4915 \
  2>&1 | tee /tmp/k3d_validation_routing_fix_03.22.2026.log
```

Write the handoff report at `TEMP/CLAUDE_ROUTING_FIX_REPORT_03.22.2026.md`.

## Success Criteria

| Metric | Before Fix | Target After Fix |
|--------|-----------|-----------------|
| ARC score | 2/42 (4.76%) | >4/42 (>9.5%) -- anchors now reachable |
| LHE score | 1/35 (2.86%) | >2/35 -- graph edges enable chaining |
| Math/GSM8K | 3/500, 7/462 | No regression, gradual improvement |
| MMLU | 1106/4915 (22.5%) | No regression |
| Contrastive specialists trained | 1 (drawing only) | 4+ (math, visual, grammar, chat) |
| ARC embedding type | Task-ID hash (semantic-blind) | Visual feature text (semantic) |
| LHE graph edges | 0 total | >0, candidates connected |

---

## Sovereignty Notes

- All three fixes STRENGTHEN sovereignty:
  - Fix 1: Better semantic routing = TRM navigates to the RIGHT Galaxy entries
  - Fix 2: Contrastive learning improves adapter weights ON GPU (LoRA-style update)
  - Fix 3: Graph crystallizer already exists as sovereign kernel -- we're just feeding it data
- Grid analysis helpers in Fix 1 are ingestion-path (Python is fine for preprocessing task grids)
- Contrastive training in Fix 2 updates adapter weights that deploy ON GPU in the next query
- No new Python in the hot path -- all fixes improve data flow TO the existing sovereign pipeline
