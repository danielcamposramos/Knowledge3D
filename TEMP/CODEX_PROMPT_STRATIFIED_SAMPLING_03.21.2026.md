# Codex Prompt: Stratified Sampling Across ALL Benchmark Loaders

**Date:** March 21, 2026
**Architecture:** Claude (spec) → Codex (implementation)
**Priority:** HIGH — current sampling takes first N questions = only easy ones, giving misleading scores

---

## The Problem

Every benchmark loader currently reads questions sequentially from the dataset file. When we sample small quantities (10, 50, 200), we ONLY get the first N — which are the easiest questions. This gives inflated scores that don't represent the real exam difficulty.

**Daniel's directive:** Separate each dataset into 3 regions (easy, mid, hard), sample equally from all three. Small samples must be a full representative slice of the exam.

---

## The Fix: Stratified Third Sampling

For EVERY benchmark that loads from a dataset file, apply this pattern:

```python
def _stratified_sample(items: list, limit: int | None) -> list:
    """Sample equally from easy (first third), mid (middle third), hard (last third).

    Datasets are typically ordered by difficulty (MATH by Level, GSM8K by
    complexity, MMLU by subject blocks, etc.). Splitting into thirds and
    sampling equally from each ensures small samples represent the full
    difficulty spectrum.
    """
    if limit is None or limit >= len(items):
        return items

    n = len(items)
    third = n // 3
    easy = items[:third]
    mid = items[third:2 * third]
    hard = items[2 * third:]

    # Distribute limit equally across thirds, remainder to hard
    per_region = limit // 3
    extra = limit - per_region * 3

    # Sample deterministically: evenly spaced within each region
    def _even_pick(region: list, count: int) -> list:
        if count >= len(region):
            return list(region)
        if count <= 0:
            return []
        step = len(region) / count
        return [region[int(i * step)] for i in range(count)]

    result = []
    result.extend(_even_pick(easy, per_region))
    result.extend(_even_pick(mid, per_region))
    result.extend(_even_pick(hard, per_region + extra))
    return result
```

**Why deterministic (evenly spaced) instead of random?**
- Reproducible results — same limit always gives same questions
- No seed management needed
- Still representative — evenly spaced within each third covers the full range

**Why remainder goes to hard?**
- Hard questions are where we need the most signal
- Easy questions inflate scores; we already know those work

---

## Files to Modify

### 1. `benchmarks/math_competitions.py` — MathCompetitionBenchmark / UnifiedMathBenchmark

**Where:** `_load_problems()` and `_load_from_math_dataset()` and `_load_from_present_datasets()`

The MATH dataset (`train.jsonl`) is ordered by type then by level within type. The first third is mostly Level 3-5 Algebra, the middle is Geometry/Number Theory, the last third is Precalculus/Counting. Apply stratified sampling when `max_problems` is set.

```python
def _load_problems(self) -> list[dict]:
    # ... existing loading logic ...
    problems = <all loaded problems>
    return _stratified_sample(problems, self.max_problems)
```

Remove all existing `[:self.max_problems]` slicing — replace with `_stratified_sample()` at the END of loading.

### 2. `benchmarks/gsm8k.py` — GSM8KBenchmark

**Where:** `_load_questions()`

GSM8K `test.jsonl` has 1,319 questions ordered roughly by complexity. Apply stratified sampling when `max_questions` is set.

```python
def _load_questions(self) -> list[dict]:
    # ... existing loading logic ...
    questions = <all loaded questions>
    return _stratified_sample(questions, self.max_questions)
```

Remove the `if self.max_questions is not None and len(questions) >= int(self.max_questions): break` early-exit in the read loop — read ALL questions first, THEN sample.

### 3. `benchmarks/mmlu.py` — MMLUBenchmark

**Where:** `_load_questions()`

MMLU has 14,042 questions across 57 subjects. The file is ordered by subject. The first third is one set of subjects, last third is another. Apply stratified sampling when `max_questions` is set.

```python
def _load_questions(self) -> list[dict]:
    # ... existing loading logic ...
    questions = <all loaded questions>
    return _stratified_sample(questions, self.max_questions)
```

### 4. `benchmarks/last_humanity_exam.py` — LHEBenchmark

**Where:** `_load_questions()`

LHE has 100 questions. Even at this small size, sampling first 10 vs stratified 10 makes a difference. Apply the same pattern.

### 5. `benchmarks/arc_agi_2.py` — ARCAGIBenchmark

**Where:** `_load_tasks()`

ARC-AGI has 120 tasks. Same pattern.

---

## Shared Utility

Put `_stratified_sample()` in a shared location so all 5 benchmark files use the SAME function. Options:

**Option A (preferred):** Add to a small utility module:
```
benchmarks/sampling.py  (new file, ~30 lines)
```
Then each benchmark imports: `from benchmarks.sampling import stratified_sample`

**Option B:** Add as a staticmethod to each class (code duplication, less clean).

Go with Option A.

---

## Important: Read ALL Then Sample

Current loaders have early-exit patterns like:
```python
if self.max_questions is not None and len(questions) >= int(self.max_questions):
    break
```

This MUST change. The new pattern is:
1. **Read ALL questions** from the dataset file (no early exit)
2. **Then apply** `stratified_sample(all_questions, limit)` at the end

For large datasets (MMLU 14K), reading all into memory is fine — these are small JSON records.

---

## Test Plan

Create `tests/test_stratified_sampling.py`:

```python
def test_stratified_covers_all_thirds():
    items = list(range(300))  # 0-99 easy, 100-199 mid, 200-299 hard
    sample = stratified_sample(items, 9)
    assert len(sample) == 9
    # Should have 3 from each region
    easy = [x for x in sample if x < 100]
    mid = [x for x in sample if 100 <= x < 200]
    hard = [x for x in sample if x >= 200]
    assert len(easy) == 3
    assert len(mid) == 3
    assert len(hard) == 3

def test_stratified_none_limit_returns_all():
    items = list(range(100))
    assert stratified_sample(items, None) == items

def test_stratified_limit_exceeds_size():
    items = list(range(10))
    assert stratified_sample(items, 50) == items

def test_stratified_deterministic():
    items = list(range(300))
    a = stratified_sample(items, 15)
    b = stratified_sample(items, 15)
    assert a == b  # same input, same output

def test_stratified_small_limit():
    items = list(range(300))
    sample = stratified_sample(items, 3)
    assert len(sample) == 3
    # One from each third
    assert sample[0] < 100
    assert 100 <= sample[1] < 200
    assert sample[2] >= 200
```

---

## Validation After Implementation

1. `python3 -m pytest tests/test_stratified_sampling.py` — all pass
2. `python3 -m pytest tests/test_math_zero_fix.py` — no regression
3. Quick check: load Math benchmark with `max_problems=9`, verify problems come from 3 different difficulty levels (Level 1-2, Level 3, Level 4-5)
4. `git diff --check` — clean

---

## Success Criteria

- All 5 benchmark loaders use `stratified_sample()`
- Small samples (10, 50) contain questions from easy, mid, AND hard regions
- No regression on existing test suite
- Deterministic: same limit always produces same sample
- Full runs (`limit=None`) return all questions unchanged
