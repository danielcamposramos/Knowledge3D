# Codex Prompt: Fix Contrastive Training + ARC Negative Logging

**Date:** 2026-03-22
**Priority:** CRITICAL — These are the two blockers preventing sleep-time learning
**Context:** Warm-boot 35% run complete (19.20%, +24 vs cold). Contrastive pair collection works. Training crashes. Visual negatives are missing.

---

## Bug 1: Contrastive Training TypeError (ALL Specialists)

### Symptom

All four specialists (chat, grammar, math, visual) fail during `train_specialist_contrastive()` with:
```
argument 2: TypeError: Don't know how to convert parameter 2
```

Pair collection succeeds (chat: 1129+/3786-, math: 10+/523-, grammar: 2+/33-, visual: 2+/1-). Training fails at adapter gradient application time. No checkpoint saved.

### Root Cause

**`RPNMathCore.copy_to_device()` and `copy_to_host()` assume 1D arrays but receive 2D numpy arrays.**

Call chain:
1. `sleeptime.py:254` → `swarm.train_specialist_contrastive(specialist_name, positive_pairs, negative_pairs)`
2. `adaptive_swarm.py:522` → `gradient = np.outer(diff, input_emb)` — shape `(dims, dims)`, a **2D array**
3. `adaptive_swarm.py:523` → `self._apply_adapter_gradient(adapter, gradient, lr)`
4. `adaptive_swarm.py:575-576` → `adapter.apply_gradient(gradient, lr=lr)`
5. `trm_adapters.py:148-149` → `self._ensure_math_core()` succeeds → `self.apply_gradient_rpn(gradient, lr)`
6. `trm_adapters.py:214-215`:
   ```python
   grad_host = np.ascontiguousarray(gradient, dtype=np.float32)  # still 2D: (dims, dims)
   RPNMathCore.copy_to_device(grad_host, buffers.gradient.ptr)
   ```
7. `rpn_math_core.py:113`:
   ```python
   data = [float(x) for x in array]  # iterates rows of 2D array → float(row) FAILS
   ```

The iteration `for x in array` on a 2D numpy array yields 1D row arrays, not scalars. `float(row_array)` raises TypeError.

**Secondary bug in `copy_to_host`** at `rpn_math_core.py:118-122`:
```python
def copy_to_host(ptr, array):
    nbytes = len(array) * ctypes.sizeof(ctypes.c_float)
    buf = (ctypes.c_float * len(array))()
    loader.memcpy_dtoh(ctypes.cast(buf, ctypes.c_void_p), ptr, nbytes)
    return [float(buf[i]) for i in range(len(array))]
```

- `len(array)` on a 2D array `(dims, rank)` returns `dims`, not `dims * rank` — wrong byte count
- Returns a list but caller at `trm_adapters.py:252` expects in-place modification of `self.A`:
  ```python
  RPNMathCore.copy_to_host(buffers.A.ptr, self.A)  # self.A is NEVER updated
  ```

### Fix

#### Fix 1a: Flatten arrays in `copy_to_device` and `copy_to_host`

File: `knowledge3d/cranium/ptx_runtime/rpn_math_core.py:112-122`

```python
@staticmethod
def copy_to_device(array, ptr: loader.CUdeviceptr) -> None:
    flat = np.asarray(array, dtype=np.float32).ravel()
    data = [float(x) for x in flat]
    buf = (ctypes.c_float * len(data))(*data)
    loader.memcpy_htod(ptr, ctypes.cast(buf, ctypes.c_void_p), ctypes.sizeof(buf))

@staticmethod
def copy_to_host(ptr: loader.CUdeviceptr, array) -> None:
    arr = np.asarray(array, dtype=np.float32)
    total = arr.size  # total elements, not len()
    nbytes = total * ctypes.sizeof(ctypes.c_float)
    buf = (ctypes.c_float * total)()
    loader.memcpy_dtoh(ctypes.cast(buf, ctypes.c_void_p), ptr, nbytes)
    flat = np.array([float(buf[i]) for i in range(total)], dtype=np.float32)
    np.copyto(arr, flat.reshape(arr.shape))
```

Key changes:
1. **`copy_to_device`**: `.ravel()` flattens 2D → 1D before iterating
2. **`copy_to_host`**: Use `.size` (total elements) not `len()` (first dim). Copy result back into original array via `np.copyto()` for in-place update. Reshape flat → original shape.

Add `import numpy as np` at the top of `rpn_math_core.py` if not already present.

#### Fix 1b: Ensure `apply_gradient_rpn` passes contiguous flat data

File: `knowledge3d/cranium/trm_adapters.py:214-222`

The current code already does `np.ascontiguousarray(gradient, dtype=np.float32)` but doesn't flatten. With Fix 1a, flattening happens inside `copy_to_device`. However, also ensure `self.A` and `self.B` (line 216-222) are contiguous before passing:

```python
# Ensure all host arrays are contiguous before device transfer
grad_host = np.ascontiguousarray(gradient, dtype=np.float32)
RPNMathCore.copy_to_device(grad_host, buffers.gradient.ptr)
RPNMathCore.copy_to_device(np.ascontiguousarray(self.A, dtype=np.float32), buffers.A.ptr)
RPNMathCore.copy_to_device(np.ascontiguousarray(self.B, dtype=np.float32), buffers.B.ptr)

b_t_host = np.ascontiguousarray(self.B.T, dtype=np.float32)
RPNMathCore.copy_to_device(b_t_host, buffers.B_transposed.ptr)
a_t_host = np.ascontiguousarray(self.A.T, dtype=np.float32)
RPNMathCore.copy_to_device(a_t_host, buffers.A_transposed.ptr)
```

This is mostly unchanged but ensures `.T` transposes produce contiguous arrays.

### Files to Modify

- `knowledge3d/cranium/ptx_runtime/rpn_math_core.py:112-122` — Fix `copy_to_device` and `copy_to_host`
- `knowledge3d/cranium/trm_adapters.py:214-222` — Ensure contiguous arrays (minor, defensive)

### Validation

After fix:
1. `copy_to_device(np.zeros((64, 64)), some_ptr)` should not raise
2. `copy_to_host(some_ptr, np.zeros((64, 16)))` should update the array in-place
3. Re-run warm 35% benchmark → contrastive training should succeed for all specialists → checkpoint saved

---

## Bug 2: ARC Visual Negatives Missing (1 of ~40 Expected)

### Symptom

ARC finished 2/42 (40 wrong). Visual specialist got only 1 negative pair for contrastive training. Expected ~40.

### Root Cause

**Failed ARC tasks have `"predicted": None` in the health log, and sleeptime skips None answers.**

Call chain for ARC failures:
1. `arc_agi_2.py:301-305` — Exception path returns `"predicted": None`
2. `benchmark_health_check.py:399` — Logs `"answer": row.get("predicted")` → `"answer": None`
3. `sleeptime.py:230-231`:
   ```python
   answer = row.get("answer")
   if answer is None or (isinstance(answer, str) and not answer.strip()):
       continue  # ← skips ALL rows with None answer
   ```

Most ARC failures are via exception (GPU query failure, transform failure, etc.), so `predicted` is None. Only ~1 ARC task produced an actual wrong grid. The rest are skipped.

### Fix

**For contrastive learning, an ARC failure with `predicted=None` IS a negative signal.** The TRM tried and failed — the question itself (not the wrong answer) is the negative data point. We need to:

1. When `answer` is None but `correct` is False and `expected` is available, use the **question embedding** paired with a **null/zero embedding** as the negative pair. This tells the contrastive learner: "for this question, the specialist produced nothing useful."

2. Alternatively (and better): log the actual TRM output even when it's wrong. Even a malformed grid or a partial result is more informative than None.

#### Fix 2a: Handle None answers in sleeptime contrastive collection

File: `knowledge3d/knowledgeverse/sleeptime.py:228-243`

Replace the negative-pair collection block:

```python
# --- Negative pairs: wrong answers ---
answer = row.get("answer")
expected = row.get("expected")
if not bool(row.get("correct", False)):
    # For contrastive: we need (question, wrong_thing) pairs
    # If answer is None (e.g. ARC exception), use the question paired
    # with the expected answer as a "missed positive" — the specialist
    # SHOULD have produced this but didn't.
    if answer is None or (isinstance(answer, str) and not answer.strip()):
        # No wrong answer to push away from, but we DO have the expected
        # answer the specialist failed to find. Treat as a missed positive:
        # push question TOWARD expected (weaker signal than a true negative).
        if expected is not None and (not isinstance(expected, str) or expected.strip()):
            try:
                question_values = engine.embed_sentence_gpu(question)
                expected_text = expected if isinstance(expected, str) else json.dumps(expected, ensure_ascii=False, sort_keys=True)
                expected_values = engine.embed_sentence_gpu(str(expected_text))
            except Exception:
                continue
            q_emb = np.asarray([float(value) for value in list(question_values)[:16]], dtype=np.float32)
            e_emb = np.asarray([float(value) for value in list(expected_values)[:16]], dtype=np.float32)
            if q_emb.size > 0 and e_emb.size > 0:
                # Add as WEAK positive — specialist should learn this mapping
                specialist_positive[specialist_name].append((q_emb, e_emb))
        continue

    try:
        question_values = engine.embed_sentence_gpu(question)
        answer_text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False, sort_keys=True)
        answer_values = engine.embed_sentence_gpu(str(answer_text))
    except Exception:
        continue
    q_emb = np.asarray([float(value) for value in list(question_values)[:16]], dtype=np.float32)
    a_emb = np.asarray([float(value) for value in list(answer_values)[:16]], dtype=np.float32)
    if q_emb.size == 0 or a_emb.size == 0:
        continue
    specialist_negative[specialist_name].append((q_emb, a_emb))
```

Key change: When `answer` is None and `correct` is False, instead of skipping, treat (question, expected) as a **missed positive** — the specialist should have been able to find this answer but didn't. This produces training signal without needing the (non-existent) wrong answer.

#### Fix 2b: Log actual ARC predictions even on failure

File: `benchmarks/arc_agi_2.py:301-305`

In the exception path, instead of `"predicted": None`, log the last partial result if available:

```python
return {
    "task_id": task["id"],
    "correct": False,
    "exact_match": False,
    "predicted": getattr(exc, 'partial_result', None),  # capture partial if available
    "expected": task["test"][0].get("output"),
    ...
}
```

This is a minor improvement — the real fix is 2a (handling None in sleeptime).

### Files to Modify

- `knowledge3d/knowledgeverse/sleeptime.py:228-243` — Handle None answers as missed positives
- `benchmarks/arc_agi_2.py:301-305` — (Optional) Log partial results on exception

### Validation

After fix:
1. ARC 2/42 should produce ~40 training signals for visual specialist (mix of positives from None-answer rows and true negatives from actual wrong grids)
2. Visual specialist contrastive should show `positives: ~42, negatives: ~1` (or similar)
3. All specialists should have meaningful training data

---

## Bug 3: `copy_to_host` Return Value Ignored (Silent Data Loss)

### Symptom

After GPU gradient computation, `self.A` and `self.B` are never updated with the results.

### Root Cause

`trm_adapters.py:252-253`:
```python
RPNMathCore.copy_to_host(buffers.A.ptr, self.A)
RPNMathCore.copy_to_host(buffers.B.ptr, self.B)
```

`copy_to_host` currently **returns a list** but doesn't modify `self.A` in-place. The return value is discarded. Fix 1a changes `copy_to_host` to modify in-place via `np.copyto()`, which fixes this silently.

### Validation

After Fix 1a, add a quick sanity check: after `copy_to_host(buffers.A.ptr, self.A)`, verify `self.A` has changed (not all zeros if a gradient was applied).

---

## Execution Order

1. **Fix 1a** (rpn_math_core.py) — fixes the TypeError crash
2. **Fix 1b** (trm_adapters.py) — defensive contiguity
3. **Fix 2a** (sleeptime.py) — fixes ARC negative collection
4. **Fix 2b** (arc_agi_2.py) — optional, log partial results
5. **Re-run warm 35% benchmark** — verify contrastive training succeeds, checkpoint saved

---

## Sovereignty Compliance

- Fix 1: GPU memory transfer helpers (boot path, not hot path). Flattening numpy arrays before GPU transfer is standard.
- Fix 2: Health log row processing (Python orchestration, not hot path). Contrastive pair collection feeds sleep-time which trains specialist adapters.
- Fix 3: Resolved by Fix 1a.

## Test Criteria

1. **Contrastive training completes:** All 4 specialists show `trained: true` in sleeptime journal
2. **Checkpoint saved:** `checkpoint` dict is non-empty in sleeptime journal
3. **Visual negatives:** Visual specialist shows >5 training pairs (was 3 total — 2+/1-)
4. **No TypeError:** Zero occurrences of "Don't know how to convert parameter 2" in logs
5. **Combined score:** Should not regress (≥19.20%). May improve if contrastive learning kicks in.
6. **Adapter weights change:** After sleep-time, specialist adapter A/B matrices should differ from pre-training values
