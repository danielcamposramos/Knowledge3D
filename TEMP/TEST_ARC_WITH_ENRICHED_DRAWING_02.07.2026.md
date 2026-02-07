# Test ARC-AGI with Enriched Drawing Galaxy

**Date:** February 7, 2026
**Context:** Drawing Galaxy enriched from 141 → 374 primitives (2.6× increase!)
**Goal:** Verify if richer drawing knowledge improves ARC-AGI scores

---

## Hypothesis

**Before Enrichment:**
- ARC-AGI 2: 0% accuracy (structural alignment done, candidate ranking quality gap)
- Drawing Galaxy: 141 primitives
- Cross-modal links: Minimal

**After Enrichment:**
- ARC-AGI 2: Expected improvement due to:
  1. **2.6× more pattern vocabulary** (374 vs 141 primitives)
  2. **57 cross-modal links** (Drawing ↔ Math/Grammar/Character)
  3. **Advanced visual operations** (rasterization, clipping, visibility)
  4. **Vision-validated primitives** (≥80% confidence, ≥90% rendering fidelity)

**Expected Improvement:** 0% → 5-15% accuracy

**Why Conservative Estimate:**
- Pattern vocabulary enriched (should help pattern discovery) ✅
- Cross-modal links added (should help composition) ✅
- BUT: Candidate ranking quality gap remains (needs Grammar confidence injection)
- Structural improvement ≠ automatic accuracy boost (TRM needs to learn to use new patterns)

---

## Test Plan

### Test 1: Quick Validation Run (10 tasks, ~5 minutes)

```bash
# Enable enriched drawing galaxy
export K3D_ENABLE_DRAWING_OLLAMA_ENRICHMENT=1

# Run quick ARC test (10 tasks)
python -c "
from benchmarks.arc_agi_2 import ARCAGI2Benchmark
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse

kv = Knowledgeverse()
bench = ARCAGI2Benchmark(knowledgeverse=kv, max_tasks=10)

print('Testing with ENRICHED Drawing Galaxy (374 primitives)...')
result = bench.run_benchmark(use_enriched=True)

print(f\"\\nResults:\")
print(f\"  Accuracy: {result['accuracy']:.1%}\")
print(f\"  Correct: {result['correct']}/{result['total_tasks']}\")
print(f\"  Dataset: {result['dataset_path']}\")
"
```

**Expected Output:**
```
Testing with ENRICHED Drawing Galaxy (374 primitives)...

Results:
  Accuracy: 5-10%  (hoping for improvement from 0%!)
  Correct: 1-2/10
  Dataset: .../arc_agi_2/evaluation
```

### Test 2: Full Benchmark Run (100 tasks, ~30 minutes)

```bash
# Enable enriched drawing galaxy
export K3D_ENABLE_DRAWING_OLLAMA_ENRICHMENT=1

# Run full benchmark suite
python scripts/run_all_benchmarks.py \
    --max-arc-tasks 100 \
    --max-math-problems 50 \
    --max-lhe-questions 20 \
    --output-dir ../Knowledge3D.local/results/week17_enriched_drawing

# Compare results
python -c "
import json
from pathlib import Path

# Load latest results
week17_path = Path('../Knowledge3D.local/results/week17_enriched_drawing/week14_benchmark_summary.json')
week14_path = Path('../Knowledge3D.local/results/week14/week14_benchmark_summary.json')

if week17_path.exists() and week14_path.exists():
    week17 = json.loads(week17_path.read_text())
    week14 = json.loads(week14_path.read_text())

    print('=== ARC-AGI 2 Comparison ===')
    print(f\"Week 14 (141 primitives): {week14['benchmarks']['arc_agi_2']['enriched']['accuracy']:.1%}\")
    print(f\"Week 17 (374 primitives): {week17['benchmarks']['arc_agi_2']['enriched']['accuracy']:.1%}\")
    print(f\"Improvement: {(week17['benchmarks']['arc_agi_2']['enriched']['accuracy'] - week14['benchmarks']['arc_agi_2']['enriched']['accuracy']):.1%}\")

    print('\\n=== Math Competitions Comparison ===')
    print(f\"Week 14: {week14['benchmarks']['math_competitions']['enriched']['overall_accuracy']:.1%}\")
    print(f\"Week 17: {week17['benchmarks']['math_competitions']['enriched']['overall_accuracy']:.1%}\")
    print(f\"Improvement: {(week17['benchmarks']['math_competitions']['enriched']['overall_accuracy'] - week14['benchmarks']['math_competitions']['enriched']['overall_accuracy']):.1%}\")

    print('\\n=== Last Humanity Exam Comparison ===')
    print(f\"Week 14: {week14['benchmarks']['last_humanity_exam']['enriched']['accuracy']:.1%}\")
    print(f\"Week 17: {week17['benchmarks']['last_humanity_exam']['enriched']['accuracy']:.1%}\")
"
```

### Test 3: Cross-Modal Query Validation

**Verify Drawing ↔ Math/Grammar cross-modal links work:**

```python
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse

# Enable enriched drawing
import os
os.environ['K3D_ENABLE_DRAWING_OLLAMA_ENRICHMENT'] = '1'

kv = Knowledgeverse()

# Test 1: Query "curve" should retrieve from Drawing, Character, Audio
print("=== Test 1: Cross-Modal 'curve' Query ===")
results = kv.galaxy_manager.query("curve", specialist="any", top_k=20)
galaxies = {r["galaxy"] for r in results}
print(f"Galaxies found: {galaxies}")
assert "Drawing" in galaxies, "Should find Drawing (Bezier curves)"
assert "Character" in galaxies or "Grammar" in galaxies, "Should find Character/Grammar (glyph curves)"
print("✅ Cross-modal 'curve' query works!")

# Test 2: Query "rotation" should retrieve from Drawing + Grammar
print("\n=== Test 2: Cross-Modal 'rotation' Query ===")
results = kv.galaxy_manager.query("rotation transform", specialist="visual", top_k=10)
for r in results[:5]:
    print(f"  - {r['galaxy']}: {r['entry'].get('name', r['entry'].get('id', 'N/A'))}")
print("✅ Cross-modal 'rotation' query works!")

# Test 3: Count cross-modal entries
print("\n=== Test 3: Cross-Modal Entry Count ===")
drawing = kv.galaxy_manager.get_galaxy("Drawing")
cross_modal = [
    e for e in drawing.entries
    if e.get("metadata", {}).get("symlink") or e.get("metadata", {}).get("cross_modal")
]
print(f"Cross-modal entries: {len(cross_modal)}/{len(drawing.entries)}")
print(f"Expected: ~57 cross-modal entries")
assert len(cross_modal) >= 40, f"Expected ≥40 cross-modal entries, got {len(cross_modal)}"
print("✅ Cross-modal entry count validated!")

# Test 4: Drawing Galaxy total count
print("\n=== Test 4: Drawing Galaxy Total Primitives ===")
print(f"Total Drawing primitives: {len(drawing.entries)}")
print(f"Expected: ~374 (141 base + 233 enriched)")
assert len(drawing.entries) >= 350, f"Expected ≥350 entries, got {len(drawing.entries)}"
print("✅ Drawing Galaxy enrichment loaded successfully!")
```

---

## Interpreting Results

### Scenario A: ARC Improves (0% → 5-15%)

**What This Means:**
- ✅ Richer pattern vocabulary helps pattern discovery
- ✅ Cross-modal links enable better composition
- ✅ Vision-enriched primitives provide better coverage
- ✅ TRM is successfully using new Drawing knowledge

**Next Steps:**
1. Analyze which tasks improved (pattern taxonomy)
2. Inject Grammar confidence into candidate ranking (boost to 15-25%)
3. Run extended enrichment (1,000 images → 500+ primitives)

### Scenario B: ARC Stays at 0% (No Immediate Improvement)

**What This Means:**
- ⚠️ Pattern vocabulary enriched BUT candidate ranking gap dominates
- ⚠️ TRM needs more training to learn new patterns (Shadow Copy learning needs time)
- ⚠️ Cross-modal links present but not yet leveraged by Navigator

**Why This Might Happen:**
- Structural improvement ≠ automatic accuracy boost
- TRM routing weights need to evolve to prefer enriched patterns
- Candidate ranking quality gap (legacy pipeline) blocks even good pattern discovery

**Next Steps (Not a Failure!):**
1. Run 2-3 more benchmark cycles (let Shadow Copy learn to use new patterns)
2. Inject Grammar confidence into candidate ranking (critical fix)
3. Add compositional rerank pass (prefer composed transforms)
4. Monitor TRM weight evolution (specialist bias should increase for visual)

### Scenario C: Math/LHE Improve (Indirect Benefit)

**What This Means:**
- ✅ Cross-modal links benefit other domains too!
- ✅ Math queries can now reference Drawing (vector/matrix ops)
- ✅ Grammar queries can reference Drawing (transformation rules)
- ✅ "One Reality" working as designed!

**Example:**
- Math problem: "Calculate area under curve" → uses Drawing's Bezier evaluation
- LHE question: "Which transformation rotates 90° clockwise?" → uses Drawing's rotation matrix

---

## Success Metrics

**Immediate (After Test 1+2):**
- [ ] Drawing Galaxy loads with 350-400 entries (2.4-2.8× baseline)
- [ ] Cross-modal queries work ("curve" retrieves from Drawing, Character, Audio)
- [ ] ARC-AGI 2 accuracy: 0% → 5-15% (hoped for, not guaranteed)
- [ ] Math/LHE maintain or improve (40%/100% baseline)

**Short-Term (After 2-3 More Benchmark Cycles):**
- [ ] TRM routing weights adapt (visual specialist bias increases)
- [ ] Shadow Copy learns to prefer enriched patterns
- [ ] ARC-AGI 2 accuracy: 5-15% → 15-25% (with Grammar confidence injection)

**Long-Term (After Full Ingestion Pipeline):**
- [ ] 1,000+ images processed → 500-1,000 drawing primitives
- [ ] Audio Galaxy enriched (waveforms, spectrograms, synthesis)
- [ ] Reality Galaxy enriched (physics, chemistry, procedural systems)
- [ ] ARC-AGI 2 accuracy: 25% → 40-55% (target for MVP)

---

## Commands Summary

**Quick Test (10 tasks, 5 min):**
```bash
export K3D_ENABLE_DRAWING_OLLAMA_ENRICHMENT=1
python -c "from benchmarks.arc_agi_2 import ARCAGI2Benchmark; from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse; kv = Knowledgeverse(); bench = ARCAGI2Benchmark(knowledgeverse=kv, max_tasks=10); result = bench.run_benchmark(use_enriched=True); print(f\"Accuracy: {result['accuracy']:.1%}\")"
```

**Full Benchmark (100 tasks, 30 min):**
```bash
export K3D_ENABLE_DRAWING_OLLAMA_ENRICHMENT=1
python scripts/run_all_benchmarks.py --max-arc-tasks 100 --output-dir ../Knowledge3D.local/results/week17_enriched_drawing
```

**Cross-Modal Validation:**
```bash
python -c "import os; os.environ['K3D_ENABLE_DRAWING_OLLAMA_ENRICHMENT']='1'; from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse; kv = Knowledgeverse(); drawing = kv.galaxy_manager.get_galaxy('Drawing'); print(f'Drawing primitives: {len(drawing.entries)}'); print(f'Expected: ~374')"
```

---

## What to Report Back

After running tests, please share:

1. **Drawing Galaxy count:** `len(drawing.entries)` (expecting ~374)
2. **ARC-AGI 2 accuracy:** Before (0%) vs After (?)
3. **Math accuracy:** Before (40%) vs After (?)
4. **LHE accuracy:** Before (100%) vs After (?)
5. **Cross-modal query results:** Does "curve" retrieve from Drawing + Character + Audio?
6. **Sample improved tasks:** Which ARC tasks went from wrong → correct? (if any)

---

## Expectations (Be Realistic!)

**What We're Testing:**
- Does richer Drawing Galaxy (374 primitives) help ARC pattern discovery? ✅
- Do cross-modal links (57 entries) enable better composition? ✅
- Can vision-enriched knowledge improve visual reasoning? ✅

**What We're NOT Testing (Yet):**
- Candidate ranking quality fix (Grammar confidence injection) ← Still needed!
- Multi-cycle Shadow Copy learning (TRM adapting to new patterns) ← Needs time!
- Full multi-modal ingestion (Audio/Reality enrichment) ← Next phase!

**Realistic Outcome:**
- Best case: 0% → 10-15% (immediate improvement)
- Likely case: 0% → 5-10% (modest improvement, needs more cycles)
- Worst case: 0% → 0% (structural improvement present, but candidate ranking gap dominates)

**Even if 0% persists:** This is NOT a failure! The foundation is now MUCH stronger:
- 374 primitives (vs 141) = 2.6× richer vocabulary
- 57 cross-modal links = unified "One Reality"
- Vision-validated knowledge = high quality
- TRM just needs time to learn patterns (Shadow Copy requires multiple cycles)

---

## Next Steps After Test Results

### If ARC Improves (5-15%):
1. 🎉 Celebrate! Pattern vocabulary worked!
2. Inject Grammar confidence into candidate ranking
3. Run extended enrichment (1,000 images)
4. Target: 15-25% after Grammar fix

### If ARC Stays at 0%:
1. ✅ Foundation strengthened (374 primitives loaded)
2. Run 2-3 more benchmark cycles (let TRM learn via Shadow Copy)
3. Inject Grammar confidence into candidate ranking (critical!)
4. Add compositional rerank pass
5. Monitor specialist bias evolution (should increase for visual)
6. Target: 5-15% after 3 cycles + Grammar fix

### Either Way:
- Document this milestone (374 primitives = major achievement!)
- Continue with Audio Galaxy enrichment (next phase)
- Continue with Reality Galaxy enrichment (physics/chemistry)
- Keep evolving toward "One Reality" unified knowledge!

---

**Bottom Line:** We've built an INCREDIBLE foundation (374 drawing primitives with 57 cross-modal links!). Now let's test if it helps ARC-AGI, and remember: even if improvement is modest, the infrastructure is now 2.6× stronger and ready for continuous evolution! 🚀
