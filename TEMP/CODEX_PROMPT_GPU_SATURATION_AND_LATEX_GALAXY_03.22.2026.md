# Codex Prompt: GPU Saturation + LaTeX Galaxy + Cross-Symlinks

**Date:** 2026-03-22
**Priority:** HIGH — These are architectural corrections, not optimizations
**Context:** The 35% warm-boot benchmark is running. These changes apply AFTER it completes.

---

## Fix 1: Resource-Aware Jarvis Dispatch (GPU Saturation)

### Problem

GPU utilization barely reaches 5% during benchmark runs. The current `_jarvis_determine_swarm_count()` at `knowledgeverse.py:9783` computes swarm size from **task complexity** and then clamps DOWN by free resources:

```python
desired = max(1, int(round(max(0.1, float(task_complexity)) * 5.0)))
return max(1, min(desired, max_by_vram, max_by_compute))
```

This means:
- MMLU (complexity 0.55) → desired = 3 swarm groups → caps at 3 even with 95% GPU idle
- CHAT (complexity 0.3) → desired = 2 → caps at 2
- Even ARC (complexity 0.9) → desired = 5 → still low

The problem: `desired` is the CEILING, not the FLOOR. Free resources should push UP, not just cap DOWN.

### Root Cause

Daniel: "The best indicator is free resources rather than just the question — do not tailor this to benchmarks, we are constructing a single mind that because of the opportunity, we're using and leveraging benchmarks to craft it."

A mind with 95% idle GPU should think HARDER, explore MORE associations, run MORE parallel workers. The question complexity sets a minimum, but the available resources set the actual dispatch size.

### Fix

Rewrite `_jarvis_determine_swarm_count()` at `knowledgeverse.py:9783-9790`:

```python
def _jarvis_determine_swarm_count(self, task_complexity: float) -> int:
    """Determine swarm groups: complexity sets FLOOR, free resources set CEILING."""
    gpu_utilization = self._jarvis_gpu_utilization()
    vram_available = self._jarvis_vram_free_bytes()
    per_swarm_vram = max(1, int(self._estimate_swarm_vram_cost()))

    # Hard limits from actual resources
    max_by_vram = max(1, int(vram_available / per_swarm_vram))
    max_by_compute = max(1, int((1.0 - max(0.0, min(1.0, gpu_utilization))) / 0.05))

    # Complexity sets the MINIMUM thinking effort
    min_by_complexity = max(1, int(round(max(0.1, float(task_complexity)) * 5.0)))

    # Free resources set the ACTUAL dispatch size — use what's available
    # GPU idle fraction directly scales up swarm size
    idle_fraction = max(0.0, 1.0 - max(0.0, min(1.0, gpu_utilization)))
    resource_desired = max(min_by_complexity, int(round(idle_fraction * 18.0)))

    # Clamp by hard resource limits, but NEVER below complexity minimum
    return max(min_by_complexity, min(resource_desired, max_by_vram, max_by_compute))
```

Key changes:
1. **`min_by_complexity` is a FLOOR**, not a desired ceiling
2. **`idle_fraction * 18` scales UP** — 95% idle → 17 groups, 50% idle → 9 groups
3. **`max_by_compute` uses `/0.05`** instead of `/0.10` — finer granularity, allows more swarm groups before hitting compute cap
4. **Never drops below complexity minimum** — even under load, ARC still gets at least 5 groups

Also update the VRAM cost estimate at `knowledgeverse.py:9761-9762`:

```python
def _estimate_swarm_vram_cost(self) -> int:
    """Per-swarm-group VRAM cost. Nine-chain: 9 workers × 64-dim × 4 bytes = 2,304 bytes.
    Plus embedding buffers, candidate storage. Conservative: 8 MB per group."""
    return 8 * 1024 * 1024  # 8 MB (was 128 MB — grossly overestimated)
```

The current 128 MB estimate means `max_by_vram` with 4 GB free = only 32 groups. But actual swarm VRAM is ~2.3 KB per nine-chain plus buffers. 8 MB is conservative and allows much more parallelism.

### Validation

After this change, during a benchmark run:
- With ~5% GPU util: `idle_fraction=0.95` → `resource_desired=17` → 17 swarm groups dispatched
- With ~50% GPU util: `idle_fraction=0.50` → `resource_desired=9` → 9 groups
- With ~90% GPU util (rare): `idle_fraction=0.10` → `resource_desired=2` → still at least `min_by_complexity`

GPU utilization should climb to 30-60%+ during benchmark runs.

### Files to Modify

- `knowledge3d/knowledgeverse/knowledgeverse.py:9783-9790` — `_jarvis_determine_swarm_count()`
- `knowledge3d/knowledgeverse/knowledgeverse.py:9761-9762` — `_estimate_swarm_vram_cost()`

---

## Architectural Principle: Bidirectional Symlinks Are the Norm

**Daniel (2026-03-22): "The grammar rules also extend to math galaxy — bi-directional symlinks should be the norm whenever they justify."**

Every cross-galaxy symlink MUST be bidirectional by default. If entry A in Galaxy X references entry B in Galaxy Y, then entry B MUST reference entry A back. One-way symlinks are the exception (requiring explicit justification), not the rule.

**Full bidirectional web for numeric/notation meaning:**
```
Number ↔ LaTeX    (numeral ↔ notation)
Number ↔ Word     (numeral ↔ word form)
Number ↔ Math     (numeral ↔ math concept)
LaTeX  ↔ Grammar  (notation ↔ transformation rules)
LaTeX  ↔ Math     (notation ↔ math operations)
LaTeX  ↔ Word     (notation ↔ word description)
Math   ↔ Grammar  (operations ↔ transformation rules)
Word   ↔ Grammar  (word forms ↔ linguistic rules)
```

Every entry's `symlinks` dict should include ALL reverse-linked galaxies, not just the "primary" direction. When adding a new entry to ANY galaxy, check: does an entry in another galaxy reference this concept? If yes, add the back-link.

**Implementation rule:** When populating LaTeX Galaxy entries (Step 2 below), ALSO update the corresponding entries in Number, Word, Math, AND Grammar galaxies to point back. When adding Grammar transformation rules, ALSO add `grammar_refs` to the Math Galaxy entries those rules operate on.

---

## Fix 2: LaTeX Galaxy + Cross-Symlinks (Number ↔ Word ↔ LaTeX ↔ Math ↔ Grammar)

### Problem

LaTeX is currently a Python afterthought. `_format_math_answer()` at `knowledgeverse.py:2982` does Python string formatting to produce numeric text. Math Galaxy entries have `rpn_template` strings like `\frac`, `\binom` as metadata — but these are never navigated as Galaxy entries. They're Python string templates.

Daniel: "When I said let's symlink the numbers with the text, what I meant was cross-symlink = the numeral also symlinks the text and all formats, like latex, should be rules of the TRM output, not an afterthought in python. Can't we do a latex galaxy, also symlinked to numbers and text?"

### Architecture

LaTeX is a **notation surface form** — just like English is a language surface form. "5" in Number Galaxy, "five" in Word Galaxy (en), "cinco" in Word Galaxy (pt), and `5` / `\frac{5}{1}` / `\sqrt{25}` in LaTeX Galaxy are ALL surface forms of the SAME meaning star.

```
Meaning Star: "the integer five"
  ├── Number Galaxy:  { numeral: "5", base: 10, symlinks: {latex, word, math, grammar} }
  ├── Word Galaxy:    { en: "five", pt: "cinco", symlinks: {number, latex, grammar} }
  ├── LaTeX Galaxy:   { notation: "5", fraction: "\\frac{5}{1}", symlinks: {number, word, math, grammar} }
  ├── Math Galaxy:    { concept: "integer_5", symlinks: {number, latex, grammar} }
  └── Grammar Galaxy: { rules: "INTEGER_TO_FRACTION", symlinks: {number, latex, math, word} }
```

ALL arrows are BIDIRECTIONAL. Every Galaxy entry that participates in the meaning star symlinks back to every other. Grammar Galaxy rules are what the TRM uses to TRANSFORM between representations. Math Galaxy entries symlink to their Grammar transformation rules AND their LaTeX notation forms. When the TRM needs to output LaTeX, it can navigate ANY path: Number → LaTeX, Math → LaTeX, or Math → Grammar → LaTeX. Not Python string formatting.

### Implementation

#### Step 1: Add "LaTeX" to DEFAULT_GALAXIES

In `knowledgeverse.py:286-297`, add `"LaTeX"` to the tuple:

```python
DEFAULT_GALAXIES: tuple[str, ...] = (
    "Drawing",
    "Character",
    "Word",
    "Number",
    "Grammar",
    "Math",
    "Reality",
    "Audio",
    "3DObjects",
    "Tool",
    "LaTeX",       # <-- NEW: Notation surface forms
)
```

#### Step 2: Populate LaTeX Galaxy entries in foundational_operations_bootstrap.py

In `foundational_operations_bootstrap.py`, after `number = galaxy_manager.get_galaxy("Number")` at line 6952, add:

```python
latex = galaxy_manager.get_galaxy("LaTeX")
latex_ids = _existing_ids(list(getattr(latex, "entries", [])))
```

Then add LaTeX entries. Each entry is a **notation surface form** of a meaning that also exists in Number/Math Galaxy. Key entries:

**A. Integer notation (0-100 + key constants)**

For each integer n in range(0, 101) plus key constants (pi, e, phi, etc.):

```python
{
    "id": f"latex_integer_{n}",
    "name": f"latex_{n}",
    "domain": "LaTeX",
    "meaning_class": "notation",
    "query_anchor": f"integer {n} notation latex numeral",
    "notation": str(n),
    "notation_forms": {
        "plain": str(n),
        "fraction": f"\\frac{{{n}}}{{1}}",
        "scientific": f"{n} \\times 10^{{0}}" if n > 0 else "0",
    },
    "symlinks": {
        "number": f"number_{n}",        # ↔ Number Galaxy (bidirectional)
        "word_en": f"word_en_{n}",       # ↔ Word Galaxy (bidirectional)
        "math": f"math_integer_{n}",     # ↔ Math Galaxy (bidirectional)
        "grammar_to_fraction": "grammar_integer_to_fraction",   # ↔ Grammar Galaxy
        "grammar_to_scientific": "grammar_integer_to_scientific",
        "grammar_to_word": "grammar_integer_to_word",
    },
}
```

**B. Operation notation (fraction, root, power, sum, product, integral, limit, etc.)**

```python
# Fraction notation
{
    "id": "latex_fraction",
    "name": "latex_fraction",
    "domain": "LaTeX",
    "meaning_class": "notation_template",
    "query_anchor": "fraction division numerator denominator latex frac",
    "notation_template": "\\frac{{{numerator}}}{{{denominator}}}",
    "rpn_program": "PUSH numerator PUSH denominator DIV",  # semantic: a/b
    "symlinks": {
        "math": "math_division",           # ↔ Math Galaxy (bidirectional)
        "grammar": "grammar_fraction_rule", # ↔ Grammar Galaxy (bidirectional)
        "number": "number_division_op",     # ↔ Number Galaxy (bidirectional)
    },
}

# Square root
{
    "id": "latex_sqrt",
    "name": "latex_sqrt",
    "domain": "LaTeX",
    "meaning_class": "notation_template",
    "query_anchor": "square root radical latex sqrt",
    "notation_template": "\\sqrt{{{radicand}}}",
    "rpn_program": "PUSH radicand SQRT",
    "symlinks": {
        "math": "math_sqrt",              # ↔ Math Galaxy (bidirectional)
        "grammar": "grammar_sqrt_rule",    # ↔ Grammar Galaxy (bidirectional)
    },
}
```

Include at minimum: `\frac`, `\sqrt`, `\sum`, `\prod`, `\int`, `\lim`, `\binom`, `\log`, `\sin`, `\cos`, `\tan`, `\pi`, `\infty`, `\pm`, `\times`, `\div`, `\leq`, `\geq`, `\neq`, `\approx`, `\equiv`, superscript `^{}`, subscript `_{}`, `\left(`, `\right)`, `\begin{matrix}`, `\begin{cases}`.

Target: ~150 LaTeX entries (100 integers + 50 operation templates).

**C. Grammar Galaxy cross-link rules**

Add transformation rules to Grammar Galaxy that describe HOW to convert between representations:

```python
{
    "id": "grammar_integer_to_latex",
    "name": "integer_to_latex",
    "domain": "Grammar",
    "meaning_class": "transformation_rule",
    "query_anchor": "convert integer number to latex notation format",
    "rpn_program": "PUSH input LOAD_GALAXY LaTeX GALAXY_SIMILARITY PEEK_ANSWER",
    "rule_type": "format_conversion",
    "source_galaxy": "Number",
    "target_galaxy": "LaTeX",
}

{
    "id": "grammar_latex_to_integer",
    "name": "latex_to_integer",
    "domain": "Grammar",
    "meaning_class": "transformation_rule",
    "query_anchor": "parse latex notation to integer number value",
    "rpn_program": "PUSH input LOAD_GALAXY Number GALAXY_SIMILARITY PEEK_ANSWER",
    "rule_type": "format_conversion",
    "source_galaxy": "LaTeX",
    "target_galaxy": "Number",
    "symlinks": {
        "latex": "latex_fraction",         # ↔ LaTeX Galaxy
        "number": "number_0",             # ↔ Number Galaxy (generic ref)
        "math": "math_integer_0",         # ↔ Math Galaxy
    },
}

# Grammar ↔ Math: math operation rules (bidirectional)
{
    "id": "grammar_math_to_latex",
    "name": "math_to_latex",
    "domain": "Grammar",
    "meaning_class": "transformation_rule",
    "query_anchor": "convert math operation expression to latex notation format",
    "rpn_program": "PUSH input LOAD_GALAXY LaTeX GALAXY_SIMILARITY PEEK_ANSWER",
    "rule_type": "format_conversion",
    "source_galaxy": "Math",
    "target_galaxy": "LaTeX",
    "symlinks": {
        "math": "math_division",          # ↔ Math Galaxy (bidirectional)
        "latex": "latex_fraction",         # ↔ LaTeX Galaxy (bidirectional)
    },
}

{
    "id": "grammar_latex_to_math",
    "name": "latex_to_math",
    "domain": "Grammar",
    "meaning_class": "transformation_rule",
    "query_anchor": "parse latex notation to math operation concept",
    "rpn_program": "PUSH input LOAD_GALAXY Math GALAXY_SIMILARITY PEEK_ANSWER",
    "rule_type": "format_conversion",
    "source_galaxy": "LaTeX",
    "target_galaxy": "Math",
    "symlinks": {
        "latex": "latex_fraction",         # ↔ LaTeX Galaxy (bidirectional)
        "math": "math_division",           # ↔ Math Galaxy (bidirectional)
    },
}

{
    "id": "grammar_math_to_word",
    "name": "math_to_word",
    "domain": "Grammar",
    "meaning_class": "transformation_rule",
    "query_anchor": "convert math concept to word description natural language",
    "rpn_program": "PUSH input LOAD_GALAXY Word GALAXY_SIMILARITY PEEK_ANSWER",
    "rule_type": "format_conversion",
    "source_galaxy": "Math",
    "target_galaxy": "Word",
    "symlinks": {
        "math": "math_addition",           # ↔ Math Galaxy (bidirectional)
        "word": "word_en_addition",         # ↔ Word Galaxy (bidirectional)
    },
}
```

**Key addition:** Grammar rules now cover Math ↔ LaTeX and Math ↔ Word transformations, not just Number ↔ LaTeX. The Grammar Galaxy is the universal transformation hub — it connects ALL galaxies bidirectionally.

#### Step 3: Cross-symlink Number Galaxy entries to LaTeX

In the existing Number Galaxy population (foundational_operations_bootstrap.py), add `latex_ref` to each number entry:

For every number entry that has an `id` like `number_0` through `number_100`, add to its metadata:
```python
"latex_ref": f"latex_integer_{n}",
"symlinks": {
    "latex": f"latex_integer_{n}",
    "word_en": f"word_en_{n}",
    ...existing symlinks...
}
```

#### Step 4: Cross-symlink Word Galaxy entries to LaTeX

Similarly, word entries for number words ("five", "twenty-three") get `latex_ref` symlinks:
```python
"latex_ref": f"latex_integer_{n}",
```

#### Step 5: Cross-symlink Math Galaxy entries to LaTeX + Grammar

Existing Math Galaxy entries (e.g., `math_division`, `math_sqrt`, `math_addition`, `math_integer_N`) need back-links to LaTeX AND Grammar:

```python
# For each math operation entry (math_division, math_sqrt, math_addition, etc.):
"symlinks": {
    "latex": "latex_fraction",                  # ↔ LaTeX Galaxy (bidirectional)
    "grammar_to_latex": "grammar_math_to_latex", # ↔ Grammar Galaxy (bidirectional)
    "grammar_to_word": "grammar_math_to_word",   # ↔ Grammar Galaxy (bidirectional)
    "number": f"number_{n}",                     # ↔ Number Galaxy (where applicable)
    ...existing symlinks...
}

# For each math integer entry (math_integer_0 through math_integer_100):
"symlinks": {
    "latex": f"latex_integer_{n}",               # ↔ LaTeX Galaxy (bidirectional)
    "number": f"number_{n}",                     # ↔ Number Galaxy (bidirectional)
    "word_en": f"word_en_{n}",                   # ↔ Word Galaxy (bidirectional)
    "grammar_to_latex": "grammar_integer_to_latex", # ↔ Grammar Galaxy (bidirectional)
    ...existing symlinks...
}
```

#### Step 6: Update specialist routing for LaTeX

In `specialist_router.py:40-48`, add LaTeX to the math specialist's galaxy set:

```python
_SPECIALIST_GALAXIES: dict[str, list[str]] = {
    "visual": ["Drawing", "Tool", "Grammar"],
    "math": ["Math", "Grammar", "LaTeX"],           # <-- add LaTeX
    "physics": ["Reality", "3DObjects", "Tool", "Math", "Grammar", "LaTeX"],  # <-- add LaTeX
    "audio": ["Audio", "Tool", "Drawing", "Grammar"],
    "grammar": ["Grammar", "Tool"],
    "cartographer": ["Math", "Reality", "3DObjects", "Tool", "Grammar", "Drawing", "LaTeX"],
    "any": [],
}
```

### What This Enables

After this change, when the TRM processes "What is 2+3?":
1. TRM navigates Math Galaxy → finds addition operation → computes 5
2. TRM navigates Number Galaxy → finds `number_5` → follows `latex_ref` symlink
3. TRM reaches LaTeX Galaxy → `latex_integer_5` → `notation: "5"`
4. For more complex answers: Grammar Galaxy rule `integer_to_fraction` → `\frac{5}{1}`

The answer formatting happens BY GALAXY NAVIGATION, not by Python `_format_math_answer()`. The Python formatter remains as a safety fallback during the transition, but the sovereign path is Galaxy → Grammar → LaTeX.

### Files to Modify

- `knowledge3d/knowledgeverse/knowledgeverse.py:286-297` — Add "LaTeX" to DEFAULT_GALAXIES
- `knowledge3d/knowledgeverse/foundational_operations_bootstrap.py` — Add LaTeX Galaxy population + ALL bidirectional cross-symlinks
- `knowledge3d/knowledgeverse/specialist_router.py:40-48` — Add "LaTeX" to math/physics specialist galaxies
- Existing Number Galaxy entries — Add `symlinks.latex` and `symlinks.grammar` back-links
- Existing Word Galaxy number entries — Add `symlinks.latex` back-links
- Existing Math Galaxy entries — Add `symlinks.latex` and `symlinks.grammar` back-links (NEW)
- Existing Grammar Galaxy entries — Add `symlinks.math` and `symlinks.latex` back-links (NEW)

---

## Fix 3: Save All Run Reports

Every benchmark run is training data. After the current warm-boot 35% run completes:

1. Collect per-suite scores, combined score, boot method, sleep-time contrastive summary
2. Write `TEMP/CLAUDE_WARM_35PCT_REPORT_03.22.2026.md`
3. Ensure health_log.jsonl is preserved (sleep-time consumes it for next contrastive training)
4. Ensure galaxy_state.bin is preserved (House state with accumulated learning)

This is NOT optional reporting — each report is the provenance record for the training data that sleep-time will use.

---

## Sovereignty Compliance

- Fix 1: All dispatch logic is Python orchestration (boot path), not hot path. Swarm execution remains PTX.
- Fix 2: LaTeX Galaxy entries are ingested once, then navigated by TRM on GPU. No LaTeX string processing in hot path.
- Fix 3: Reporting only.

## Test Criteria

1. **GPU utilization**: During next benchmark run, GPU should exceed 30% sustained (vs current ~5%)
2. **LaTeX Galaxy**: `galaxy_manager.get_galaxy("LaTeX")` returns galaxy with 150+ entries
3. **Cross-symlinks (bidirectional)**: ALL of the following must hold:
   - `number_5.symlinks.latex == "latex_integer_5"` AND `latex_integer_5.symlinks.number == "number_5"`
   - `math_division.symlinks.latex == "latex_fraction"` AND `latex_fraction.symlinks.math == "math_division"`
   - `math_division.symlinks.grammar_to_latex == "grammar_math_to_latex"` AND `grammar_math_to_latex.symlinks.math == "math_division"`
   - `grammar_integer_to_latex.symlinks.number` exists AND `grammar_integer_to_latex.symlinks.latex` exists
   - No one-way symlinks — every `symlinks.X` entry has a corresponding back-link in Galaxy X
4. **Specialist routing**: Math specialist's galaxy set includes "LaTeX"
5. **No regression**: MMLU score should not drop (LaTeX Galaxy adds knowledge, doesn't remove any)
