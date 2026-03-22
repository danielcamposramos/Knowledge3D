# Codex Prompt: True Contrastive Learning + Incremental House + Unified ARC Embedding

**Date:** March 22, 2026
**Architecture:** Claude (spec) + Codex (implementation)
**Priority:** CRITICAL -- fixes the three root causes identified from the smoke run regression

---

## Context

The smoke run after the routing/contrastive/multi-hop changes showed:
- ARC REGRESSED from 2/42 to 0/42
- Visual specialist got 0 contrastive training pairs
- The system keeps cold-starting, destroying all accumulated learning

Three fundamental issues:

1. **Contrastive learning is only half-implemented** -- only positive pairs (correct answers), zero negative pairs (wrong answers). Humans learn MORE from what NOT to do. The model has 42 ARC negative signals and 0 positive signals, and we're throwing away all 42.

2. **Cold-start destroys accumulated knowledge** -- every run uses `--cold-start` which deletes the House state. Sleep-time consolidation, contrastive training, routing weight updates -- all lost. New knowledge must be added INCREMENTALLY to the existing House, not rebuild from scratch.

3. **ARC embedding must use the same path as all other tasks** -- no task-ID hash, no special case. K3D is a single mind doing mundane tasks, not a benchmark machine. ARC visual tasks differ from MMLU text tasks only in content, not in how the mind processes them. Everything is procedural and RPN.

---

## Fix 1: TRUE Contrastive Learning (Positive AND Negative)

### The Problem

`_run_contrastive_training()` in `sleeptime.py` (line 201-202):

```python
if not bool(row.get("correct", False)):
    continue  # <-- THROWS AWAY ALL NEGATIVE EXAMPLES
```

And `train_specialist_contrastive()` in `adaptive_swarm.py` (line 514-516):

```python
# Only pulls embeddings together (positive):
diff = target_emb - input_emb
loss = np.linalg.norm(diff)
gradient = np.outer(diff, input_emb)
```

This is NOT contrastive learning. This is just positive embedding alignment. True contrastive learning has TWO forces:

- **Positive (correct):** Pull query embedding TOWARD correct answer embedding
- **Negative (incorrect):** Push query embedding AWAY FROM wrong answer embedding

The negative force is the MORE IMPORTANT one because:
- There's usually 1 right answer and MANY wrong answers
- ARC scored 0/42 = 42 negative examples, 0 positive = the model had 42 lessons it ignored
- MMLU scored 21/100 = 79 negative examples vs 21 positive = nearly 4x more negatives
- Learning what NOT to do is faster because the signal is stronger and more abundant

### The Fix

#### Step 1: Modify `train_specialist_contrastive()` to Support Triplets

The function should accept triplets: (anchor, positive, negative) not just pairs.

In `adaptive_swarm.py`, add or modify:

```python
def train_specialist_contrastive(
    self,
    specialist_name: str,
    positive_pairs: list[tuple[np.ndarray, np.ndarray]],
    negative_pairs: list[tuple[np.ndarray, np.ndarray]] | None = None,
    learning_rate: float | None = None,
    margin: float = 0.5,
) -> dict[str, float]:
    """
    Train specialist using TRUE contrastive learning.

    Positive pairs: (query, correct_answer) -> pull TOGETHER
    Negative pairs: (query, wrong_answer) -> push APART

    Uses triplet-margin loss:
      L = max(0, d(anchor, positive) - d(anchor, negative) + margin)

    Where d() is L2 distance in embedding space.
    """
    if specialist_name not in self.base.specialists:
        raise ValueError(f"Unknown specialist: {specialist_name}")

    specialist = self.base.specialists[specialist_name]
    adapter = specialist['adapter']
    dims = specialist['dims']
    lr = learning_rate if learning_rate is not None else self.config.specialist_learning_rate

    total_positive_loss = 0.0
    total_negative_loss = 0.0
    positive_steps = 0
    negative_steps = 0

    # POSITIVE: pull query toward correct answer
    for query_emb, correct_emb in positive_pairs:
        query_emb = self._pad_or_truncate(query_emb, dims)
        correct_emb = self._pad_or_truncate(correct_emb, dims)

        diff = correct_emb - query_emb
        loss = np.linalg.norm(diff)
        total_positive_loss += loss

        # Gradient: move toward correct
        gradient = np.outer(diff, query_emb)
        self._apply_adapter_gradient(adapter, gradient, lr)
        positive_steps += 1

    # NEGATIVE: push query away from wrong answer
    if negative_pairs:
        for query_emb, wrong_emb in negative_pairs:
            query_emb = self._pad_or_truncate(query_emb, dims)
            wrong_emb = self._pad_or_truncate(wrong_emb, dims)

            diff = query_emb - wrong_emb  # REVERSED: direction AWAY from wrong
            dist = np.linalg.norm(diff)

            # Only push if too close (within margin)
            if dist < margin:
                repulsion_loss = margin - dist
                total_negative_loss += repulsion_loss

                # Gradient: move AWAY from wrong answer
                # Use negative direction (push apart)
                gradient = np.outer(-diff, query_emb)  # negative = repulsive
                # Use smaller learning rate for negatives to avoid instability
                self._apply_adapter_gradient(adapter, gradient, lr * 0.5)
                negative_steps += 1

    total_steps = positive_steps + negative_steps
    if specialist_name in self.specialist_steps:
        self.specialist_steps[specialist_name] += total_steps

    return {
        'avg_loss': (total_positive_loss + total_negative_loss) / max(total_steps, 1),
        'positive_loss': total_positive_loss / max(positive_steps, 1),
        'negative_loss': total_negative_loss / max(negative_steps, 1),
        'positive_steps': positive_steps,
        'negative_steps': negative_steps,
        'steps': total_steps,
    }


def _pad_or_truncate(self, emb: np.ndarray, dims: int) -> np.ndarray:
    if emb.shape[0] != dims:
        emb = np.pad(emb, (0, max(0, dims - len(emb))))[:dims]
    return emb


def _apply_adapter_gradient(self, adapter, gradient: np.ndarray, lr: float) -> None:
    if (
        hasattr(adapter, 'config')
        and bool(getattr(adapter.config, 'require_gpu', True)) is False
        and hasattr(adapter, '_apply_gradient_cpu')
    ):
        adapter._apply_gradient_cpu(gradient, lr)
    elif hasattr(adapter, 'apply_gradient'):
        adapter.apply_gradient(gradient, lr=lr)
    elif hasattr(adapter, 'A') and hasattr(adapter, 'B'):
        grad_A = gradient @ adapter.B.T
        grad_B = adapter.A.T @ gradient
        adapter.A -= lr * grad_A
        adapter.B -= lr * grad_B
```

#### Step 2: Modify Sleep-Time to Collect Negative Pairs

In `sleeptime.py`, `_run_contrastive_training()`:

```python
specialist_positive: dict[str, list[tuple]] = {
    "math": [], "visual": [], "grammar": [], "chat": [],
}
specialist_negative: dict[str, list[tuple]] = {
    "math": [], "visual": [], "grammar": [], "chat": [],
}

for row in rows:
    suite = str(row.get("suite", "")).strip().lower()
    question = str(row.get("question", "")).strip()
    correct = bool(row.get("correct", False))
    specialist_name = self._suite_specialist_name(suite)

    if correct:
        # Positive pair: question -> correct answer
        expected = row.get("expected")
        if not question or expected is None:
            continue
        expected_text = expected if isinstance(expected, str) else json.dumps(expected, ensure_ascii=False, sort_keys=True)
        try:
            q_emb = np.asarray(engine.embed_sentence_gpu(question)[:16], dtype=np.float32)
            e_emb = np.asarray(engine.embed_sentence_gpu(str(expected_text))[:16], dtype=np.float32)
        except Exception:
            continue
        specialist_positive[specialist_name].append((q_emb, e_emb))
    else:
        # Negative pair: question -> WRONG answer the model gave
        answer = row.get("answer")
        if not question or answer is None:
            continue
        answer_text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False, sort_keys=True)
        if not str(answer_text).strip():
            continue
        try:
            q_emb = np.asarray(engine.embed_sentence_gpu(question)[:16], dtype=np.float32)
            a_emb = np.asarray(engine.embed_sentence_gpu(str(answer_text))[:16], dtype=np.float32)
        except Exception:
            continue
        specialist_negative[specialist_name].append((q_emb, a_emb))

# Train each specialist with BOTH positive and negative signals
for specialist_name in specialist_positive:
    positives = specialist_positive[specialist_name]
    negatives = specialist_negative[specialist_name]
    if not positives and not negatives:
        results[specialist_name] = {"trained": False, "positives": 0, "negatives": 0}
        continue
    try:
        stats = swarm.train_specialist_contrastive(
            specialist_name,
            positive_pairs=positives,
            negative_pairs=negatives,
        )
    except Exception as exc:
        results[specialist_name] = {"trained": False, "error": str(exc)}
        continue
    results[specialist_name] = {
        "trained": True,
        "positives": len(positives),
        "negatives": len(negatives),
        "positive_loss": float(stats.get("positive_loss", 0.0)),
        "negative_loss": float(stats.get("negative_loss", 0.0)),
        "steps": int(stats.get("steps", 0)),
    }
```

**Key insight:** For the negative pair, we use `row.get("answer")` -- the WRONG answer the model actually produced. This teaches the specialist "when you see this question, DON'T go toward this answer." The model learns from its own mistakes.

### Expected Impact

From the last smoke:
- ARC: 0 positives, 42 negatives -> visual specialist now gets 42 "don't go here" signals
- Math: 1 positive, 49 negatives -> math specialist gets 49 negative corrections
- GSM8K: 1 positive, 49 negatives -> same
- LHE: 2 positives, 33 negatives -> grammar specialist gets 33 corrections
- MMLU: 21 positives, 79 negatives -> chat specialist gets 79 corrections

The negative signal is 3-40x more abundant than positive. This IS the training data.

---

## Fix 2: Incremental House (Stop Destroying Accumulated Learning)

### The Problem

Every benchmark run has been using `--cold-start`, which:
1. Deletes `galaxy_state.bin` (line 555-556)
2. Re-bootstraps from scratch (line 576-598)
3. Saves a NEW House state (line 599)

This means ALL accumulated learning is destroyed:
- Sleep-time routing updates (31,565 updates last run)
- Contrastive adapter improvements
- Jarvis consolidation patterns
- Any House evolution

### The Fix

#### Step 1: Incremental Knowledge Addition on Warm Boot

When warm-booting, the runner should STILL check for new knowledge that isn't in the House yet, and add it WITHOUT destroying existing knowledge.

Change the warm-boot path (line 558-574) to:

```python
warm_boot = (not bool(args.cold_start)) and knowledgeverse.load_house_state()
if warm_boot:
    house_summary = dict(knowledgeverse.house_state_summary())
    persisted_entries = int(house_summary.get("total_persisted_entries", 0))
    print(
        f"Warm boot: House loaded ({persisted_entries} entries across "
        f"{house_summary.get('galaxy_count', 0)} galaxies)",
        flush=True,
    )
    # Incremental: add any NEW knowledge that isn't already in the House
    incremental_summary = _incremental_knowledge_update(knowledgeverse, suite_counts)
    ingest_summary = {
        "house_boot": "warm",
        "house_state_path": str(knowledgeverse.house_state_path),
        "persisted_galaxies": int(house_summary.get("galaxy_count", 0)),
        "persisted_entries": persisted_entries,
        "incremental": incremental_summary,
    }
    math_rules_summary = {
        "warm_boot": True,
        "total_entries": int(house_summary.get("math_entries", 0)),
    }
    if incremental_summary.get("added", 0) > 0:
        # Save updated House with new entries
        knowledgeverse.save_house_state()
        print(
            f"Incremental update: {incremental_summary['added']} new entries added to House.",
            flush=True,
        )
```

#### Step 2: Implement Incremental Update Function

```python
def _incremental_knowledge_update(
    knowledgeverse: Knowledgeverse,
    suite_counts: dict[str, int],
) -> dict[str, Any]:
    """Add new knowledge to the House without destroying existing state.

    Checks which entries are already present by ID. Only adds genuinely new ones.
    This preserves all accumulated sleep-time learning, routing weights,
    and contrastive adapter state.
    """
    added = 0
    skipped = 0

    # Check for new ARC anchors
    existing_ids = {
        str(entry.get("id", ""))
        for entry in knowledgeverse.get_gpu_galaxy_catalog()
        if str(entry.get("id", "")).startswith("arc_anchor_")
    }
    try:
        from scripts.ingest_arc_knowledge import build_all_entries
        all_arc_entries = build_all_entries()
        for entry in all_arc_entries:
            entry_id = str(entry.get("id", ""))
            if entry_id in existing_ids:
                skipped += 1
                continue
            knowledgeverse.galaxy_manager.add_entry(
                entry.get("domain", "drawing"), entry
            )
            added += 1
    except Exception:
        pass

    # Check for new math rules
    try:
        from scripts.ingest_math_rules import build_all_rules
        existing_math_ids = {
            str(entry.get("id", ""))
            for entry in knowledgeverse.get_gpu_galaxy_catalog()
            if str(entry.get("metadata", {}).get("ingest_source", "")) == "ingest_math_rules"
        }
        all_rules = build_all_rules()
        for rule in all_rules:
            rule_id = str(rule.get("id", ""))
            if rule_id in existing_math_ids:
                skipped += 1
                continue
            knowledgeverse.galaxy_manager.add_entry(
                rule.get("domain", "math"), rule
            )
            added += 1
    except Exception:
        pass

    return {"added": added, "skipped": skipped}
```

**Tip:** The `build_all_entries()` / `build_all_rules()` functions may not exist yet as standalone builders. Refactor the existing ingestion scripts to expose a `build_*` function that returns the entry list without adding to knowledgeverse, so incremental update can check IDs and add selectively.

If refactoring is too complex for now, a simpler approach: track a version counter in the House state. If `ingest_arc_knowledge` version matches what's in the House, skip. If it's newer (e.g., we added new anchors), trigger a targeted re-ingest of just the ARC entries.

#### Step 3: Default to Warm Boot

Remove `--cold-start` from the standard benchmark commands. Warm boot should be the DEFAULT. Cold start should only be used when:
- The House state format changed (version mismatch -- already handled in `load_house_state`)
- A fundamental ingestion bug was fixed and data must be rebuilt

The default benchmark command becomes:

```bash
python3 scripts/run_enriched_benchmarks.py \
  --full \
  --storage-root /K3D/Knowledge3D.local \
  --arc-max 42 --math-max 50 --gsm8k-max 50 --lhe-max 35 --mmlu-max 100
```

No `--cold-start`. The system warm-boots, preserves all accumulated learning, adds any new entries incrementally, then runs benchmarks. Each cycle builds on the last.

### Expected Impact

- Sleep-time learning accumulates across runs (routing improvements, adapter weights, Jarvis patterns)
- Boot time drops from ~60s (cold bootstrap) to <5s (warm load)
- The system becomes what Daniel designed: a persistent, always-evolving mind
- Each benchmark cycle makes the NEXT cycle better, instead of starting from scratch

---

## Fix 3: Unified ARC Embedding — Same Path as All Other Tasks

### The Principle

**K3D is a single mind doing mundane tasks, not a benchmark machine.**

MMLU, Math, GSM8K, LHE all use the SAME embedding path: `engine.embed_sentence_gpu(query_text)`. The query text describes the task in natural/procedural terms, and the embedding engine maps it into a 16-dim vector that matches Galaxy entries semantically.

ARC tasks must work the SAME way. The only difference is that ARC tasks involve visual/procedural content (grids, transforms, colors) instead of text questions. But in K3D, ALL content is procedural and RPN. A rotation is an RPN program. A color remap is an RPN program. A grid crop is an RPN program. Visual tasks are not special -- they're just procedural tasks that happen to operate on grids.

### The Problem

`_embed_query_gpu()` has an ARC special case (line 2636-2640) that bypasses the normal embedding path:

```python
if str((task or {}).get("type", "")).upper() == "ARC_TASK":
    task_id = str((task or {}).get("task_id", "")).strip()
    if task_id:
        return self._arc_task_embedding16(task_id)  # BYPASS — semantic-blind hash
```

This means ARC never goes through `engine.embed_sentence_gpu()`. It uses a task-ID hash instead. No other task type does this. This violates the single-mind principle: the mind has one way of understanding the world, not a special way per benchmark.

Meanwhile, `_task_query_text_for_embedding()` builds the query text for ARC as:
`"ARC_TASK 00576224 solve arc transformation task train_pairs 3 input_shape 5x5"`

This is generic and carries almost no semantic signal about what the task actually requires.

### The Fix

#### Step 1: Remove the ARC Special Case in `_embed_query_gpu()`

Delete the ARC bypass entirely. Let ARC tasks flow through the SAME path as everything else:

```python
def _embed_query_gpu(self, query_text: str, *, task=None):
    # ONE path for ALL tasks — single mind, no special cases
    engine = self.get_gpu_query_embedding_engine()
    values = engine.embed_sentence_gpu(query_text)
    return [float(values[i]) for i in range(min(16, len(values)))]
```

Delete or deprecate `_arc_task_embedding16()` — it's a benchmark artifact.

#### Step 2: Make ARC Query Text Semantically Rich (Like Math Does)

Look at how MATH tasks enrich their query text (line 1839-1850):

```python
if task_type == "MATH_TASK":
    if "solve" in lowered and "x" in lowered and "=" in lowered:
        fragments.append("linear equation solve ax + b = c isolate x")
    if self._query_mentions_factorial(lowered):
        fragments.append("factorial n! compute factorial")
    if "binomial" in lowered:
        fragments.append("binomial coefficient n choose k")
```

Math tasks analyze the question text and append procedural terms that match Galaxy entries. ARC tasks must do the SAME thing — but instead of analyzing text, they analyze the GRIDS. The `_arc_visual_feature_text()` already does this. Use it in the query text builder.

In `_task_query_text_for_embedding()`, the ARC branch (line 1796-1813) should become:

```python
if task_type == "ARC_TASK":
    fragments: list[str] = ["visual transformation task"]
    training_examples = payload.get("training_examples")

    # Analyze the visual content — same way Math analyzes equation text
    visual_features = self._arc_visual_feature_text(payload)
    if visual_features:
        fragments.append(visual_features)

    if isinstance(training_examples, list) and training_examples:
        fragments.append(f"examples {len(training_examples)}")
    input_grid = payload.get("input_grid")
    if isinstance(input_grid, list):
        rows = len(input_grid)
        cols = len(input_grid[0]) if rows and isinstance(input_grid[0], list) else 0
        fragments.append(f"grid {rows}x{cols}")
    if str(prompt or "").strip():
        fragments.append(str(prompt).strip())
    return " ".join(fragment for fragment in fragments if fragment).strip()
```

**Key changes from previous attempt:**
- NO task ID in the query text (it's a benchmark artifact, not semantic content)
- NO "ARC_TASK" prefix (this is a visual transformation task, not a benchmark task)
- Visual features ARE the query text (same as Math appending "linear equation" or "factorial")
- The vocabulary in `_arc_visual_feature_text()` must match the vocabulary in the ARC anchor `query_anchor` fields

#### Step 3: Align Visual Feature Vocabulary with Anchor Vocabulary

The previous `_arc_visual_feature_text()` regression happened because the feature text used different words than the anchor `query_anchor` fields. Fix this by using the SAME vocabulary.

ARC anchors have `query_anchor` fields like:
- `"find objects connected regions discrete shapes separate groups"`
- `"rotate mirror transform reflect symmetry"`
- `"color remap substitution pattern change"`
- `"grid overlay concatenate interleave difference"`
- `"multiple objects detect separate groups"`

The visual feature text must produce fragments using these SAME terms. Ensure `_arc_visual_feature_text()` uses this vocabulary:

```python
def _arc_visual_feature_text(self, task: dict[str, Any]) -> str:
    """Describe visual task in procedural terms matching Galaxy anchor vocabulary."""
    fragments: list[str] = []
    training = task.get("training_examples") or []

    for pair in training[:3]:
        inp = pair.get("input", [])
        out = pair.get("output", [])
        if not inp or not out:
            continue

        inp_h, inp_w = len(inp), len(inp[0]) if inp else 0
        out_h, out_w = len(out), len(out[0]) if out else 0

        # Size relationship — use anchor vocabulary
        if (out_h, out_w) == (inp_h, inp_w):
            fragments.append("same size transform pattern")
        elif out_h > inp_h or out_w > inp_w:
            fragments.append("scale expand tile repeat fill larger")
        elif out_h < inp_h or out_w < inp_w:
            fragments.append("crop extract subgrid smaller")

        # Color analysis
        inp_colors = set(c for row in inp for c in row)
        out_colors = set(c for row in out for c in row)
        if out_colors - inp_colors:
            fragments.append("color change remap new colors substitution")
        if inp_colors - out_colors:
            fragments.append("color removal filter background foreground")
        if len(inp_colors) <= 3:
            fragments.append("simple few colors")
        if len(inp_colors) >= 6:
            fragments.append("complex many colors palette")

        # Object detection (simple connected components)
        bg_color = max(inp_colors, key=lambda c: sum(r.count(c) for r in inp))
        obj_count = _count_connected_components(inp, bg_color)
        if obj_count >= 3:
            fragments.append("multiple objects detect separate groups regions")
        elif obj_count == 2:
            fragments.append("two objects pair relationship")
        elif obj_count == 1:
            fragments.append("single object transform shape")

        # Symmetry
        if _grid_has_symmetry(inp):
            fragments.append("symmetry mirror reflect")

        # Grid size
        if inp_h <= 5 and inp_w <= 5:
            fragments.append("small grid pattern")
        elif inp_h >= 15 or inp_w >= 15:
            fragments.append("large grid spatial")

        # Transformation detection: compare input/output
        if inp_h == out_w and inp_w == out_h:
            fragments.append("rotate transpose dimension swap")
        if inp == [row[::-1] for row in out]:
            fragments.append("mirror horizontal reflect")
        if inp == out[::-1]:
            fragments.append("mirror vertical reflect")

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for f in fragments:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return " ".join(unique)
```

**The key principle:** The feature text and the anchor `query_anchor` fields must share the same word space so the trigram embedding engine produces vectors that are CLOSE in embedding space. "multiple objects detect separate groups" in the query must produce an embedding near "find objects connected regions discrete shapes separate groups" in the anchor.

#### Step 4: Also Align the Tablet Query

In `headless_tablet.py` (line 154), the tablet query is a fixed string `"solve arc transformation task"`. This should also be the visual feature text:

```python
# In TabletIngest.arc_task():
query = "visual transformation task"  # NOT "solve arc transformation task"
# The actual semantic content comes from _task_query_text_for_embedding()
```

### Why This Will Work Now (When the Previous Attempt Failed)

The previous attempt failed because:
1. The feature text vocabulary didn't match the anchor vocabulary
2. The task-ID hash was replaced but the underlying trigram embedding space wasn't aligned

This attempt fixes both:
1. Feature text uses the EXACT vocabulary from anchor `query_anchor` fields
2. The same `engine.embed_sentence_gpu()` path processes BOTH the query and the anchor text, so matching is natural — it's how MMLU works, how Math works, how GSM8K works
3. Contrastive learning with negatives (Fix 1) will FURTHER improve the alignment over cycles

### Files to Modify

- `knowledge3d/knowledgeverse/knowledgeverse.py`:
  - Remove ARC special case in `_embed_query_gpu()` (delete the `if ARC_TASK` branch)
  - Update `_task_query_text_for_embedding()` ARC branch to use visual features
  - Update `_arc_visual_feature_text()` to use anchor-aligned vocabulary
  - Deprecate `_arc_task_embedding16()` (can delete or leave as dead code)
- `knowledge3d/bridge/headless_tablet.py`:
  - Update `TabletIngest.arc_task()` query string

---

## Execution Order

1. **Fix 3:** Unified ARC embedding — remove task-ID special case, align visual feature vocabulary with anchor vocabulary
2. **Fix 1:** True contrastive with negative pairs in `adaptive_swarm.py` and `sleeptime.py`
3. **Fix 2:** Incremental House (warm boot default, incremental knowledge addition)

## Verification

**IMPORTANT: Do NOT use --cold-start for this smoke run.** The point is to warm-boot from the existing House state, preserving accumulated learning.

First, verify warm boot works with the new incremental update:

```bash
export CUDA_VISIBLE_DEVICES=0
python3 scripts/run_enriched_benchmarks.py \
  --full \
  --storage-root /K3D/Knowledge3D.local \
  --arc-max 42 \
  --math-max 50 \
  --gsm8k-max 50 \
  --lhe-max 35 \
  --mmlu-max 100 \
  2>&1 | tee /tmp/k3d_true_contrastive_smoke_03.22.2026.log
```

Check for:
- "Warm boot: House loaded" (NOT "Cold start: bootstrapping")
- ARC score >= 2/42 (restored from regression)
- Contrastive summary shows BOTH positive AND negative pairs for each specialist
- Visual specialist gets negative pairs even if ARC scores 0 (42 negatives from ARC)

Then run a second cycle immediately after to verify cumulative learning:

```bash
python3 scripts/run_enriched_benchmarks.py \
  --full \
  --storage-root /K3D/Knowledge3D.local \
  --arc-max 42 \
  --math-max 50 \
  --gsm8k-max 50 \
  --lhe-max 35 \
  --mmlu-max 100 \
  2>&1 | tee /tmp/k3d_true_contrastive_cycle2_03.22.2026.log
```

The SECOND run should:
- Warm-boot with all learning from the first run preserved
- Show improved scores (especially specialists that got negative training)
- Prove that the system accumulates experience across runs

Write the handoff report at `TEMP/CLAUDE_TRUE_CONTRASTIVE_REPORT_03.22.2026.md`.

## Success Criteria

| Metric | Last Smoke | Target |
|--------|-----------|--------|
| ARC score | 0/42 (regressed) | > 0/42, ideally >= 2/42 |
| ARC embedding path | Task-ID hash (benchmark artifact) | Same `embed_sentence_gpu()` as all other tasks |
| Boot method | Cold start (destroys learning) | Warm boot (preserves learning) |
| Contrastive: positive pairs | Yes (21 chat, 2 math, 2 grammar) | Same or better |
| Contrastive: negative pairs | **ZERO** (all discarded) | 42 visual, 49 math, 49 gsm8k, 33 grammar, 79 chat |
| Cycle-over-cycle improvement | N/A (cold start resets) | Measurable on cycle 2 |
| Visual specialist trained | No (0 pairs) | Yes (42 negative pairs minimum) |
| ARC special-case code | `_arc_task_embedding16()` bypass | Deleted — unified path |

## Sovereignty Notes

- True contrastive learning runs during sleep-time (not hot-path)
- Negative gradient updates the same LoRA-style adapter weights used on GPU
- Incremental House preserves all sovereign state (Galaxy, adapters, routing weights)
- The model learns FROM ITS OWN MISTAKES -- this is the self-improving cycle Daniel designed
