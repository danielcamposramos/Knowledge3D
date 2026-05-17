---
date: 2026-04-20
author: Claude (pilot mode, Codex limit-locked)
status: diagnostic — corrects earlier audit claims; redirects ingestion priorities
---

# Data-State Diagnostic — Grammar + Proceduralized Corpora

## TL;DR

Earlier sub-agent audits claimed "Grammar has 103,039 defeasibility-enabled
rules." **False.** A 10k-entry sample of the live on-disk Grammar galaxy
shows:

- `rule_strength` populated: **0**
- `superior_to` populated: **0**
- `trust_weight` populated: **0**
- `answer_eligible=True`: **1**
- Entries with `eval_program` + `arg_keys` (executable RPN templates): **0**

The defeasibility schema exists; the populating pass never ran. The 103k
count is real, but these are structural stubs with `rpn_program: "noop exec"`
or `"QUERY CONTEXT ALIGN COMPOSE"`, not executable rules.

## What IS populated (20k scan)

| Metadata key           | Entries | Notes |
|------------------------|---------|-------|
| `source`               | 19,660  | All entries tagged |
| `cross_modal`          | 19,421  | Cross-modal refs present |
| `symlink`              | 14,921  | **76% have symlink targets** (e.g. `"reality_galaxy"`) |
| `augmented_by_ollama`  | 13,770  | Ollama PDF augmentation ran |
| `source_pdf`           | 13,770  | Traceable to source PDF |
| `entities`             | 13,770  | Named entities extracted |
| `relationships`        | 13,770  | Subject-verb-object triples |
| `target_galaxy`        | 7,060   | Canonical symlink target |
| `supervision_answer`   | 600     | **Gold answers on some entries** |

### By source breakdown (top 9 of 20k scan)

| Source                            | Entries |
|-----------------------------------|---------|
| `pdf_intelligent_augmentation`    | 13,770 |
| `benchmark_augmentation_lhe`      | 2,500 |
| `benchmark_augmentation_mmlu`     | 2,000 |
| `benchmark_augmentation_math`     | 751 |
| `benchmark_augmentation_arc`      | 400 |
| `unknown`                         | 340 |
| `gpu_query_runtime`               | 223 |
| `math_specialist_bootstrap`       | 13 |
| `pattern`                         | 3 |

## Proceduralized corpora — schema vs executable gap

Sample entry `gsm8k_train_0` (file:
`/K3D/Knowledge3D.local/galaxies/proceduralized_gsm8k_train_10.jsonl`):

```json
{
  "id": "gsm8k_train_0",
  "rpn_program": "number divide_by_two add original_number equals_total_sum",
  "answer_eligible": true,
  "metadata": {
    "meaning_star": {
      "meaning_rpn": "number divide_by_two add original_number equals_total_sum",
      "surface_forms": {"en": {"char_refs": ["char_c", "char_a", ...]}},
      "meta_refs": ["grammar_rules:[{\"pattern\": \"IF have_original_number THEN divide_by_two_and_add_to_original\", \"strength\": 1}]"]
    }
  }
}
```

**What's missing:**
- No `arg_keys` — the math template binder (`_materialize_math_template_program`
  at `knowledgeverse.py:5931`) needs these to know which query numbers bind
  to which template slots.
- No `eval_program` with `ARG_X` placeholders — the `rpn_program` is in
  natural-language form, not engine-executable form (engine expects things
  like `"ARG_N 2 / ARG_N +"`).

Result: the math path finds these entries but `_math_match_allows_direct_eval`
short-circuits, and `engine.evaluate("number divide_by_two add ...")` would
throw. The entry is effectively dead weight for answering.

## What this means for priorities

The earlier audit recommendations were:

1. ✅ **Wire proceduralized + Book galaxies into query list** — DONE
   (commit `70c465c3`). Correct move; entries are now reachable.
2. ⚠️ **Use Grammar rule defeasibility to filter/rank** — BLOCKED.
   The fields aren't populated. Implementing defeasibility precedence
   sorts against uniformly-zero rule strength would be a no-op.
3. ⚠️ **Symlink dereferencing** — LOW LEVERAGE right now. The 14,921
   entries with `symlink: reality_galaxy` point at Reality stars that
   themselves don't have executable RPN. Dereferencing expands the
   surface text for keyword matching but won't lift arithmetic accuracy.
4. 🎯 **Real lift: ingestion enrichment** — add `arg_keys` + `eval_program`
   to proceduralized corpora, and `rule_strength` + `superior_to` to
   Grammar entries. This is a data-pipeline change, not a hot-path code
   change.

## Recommended next passes

### Code-side (Claude-pilot capable)

- **Surface `supervision_answer`** — 600 entries have gold answers
  already. The math materialization path could short-circuit when an
  entry with a matching query text AND a `supervision_answer` field is
  found. This is a real, small-but-concrete lift.
- **Symlink follow-hop** — on query, fetch top-k from the primary galaxy,
  then for each result with `metadata.symlink`, fetch top-k from the
  target galaxy and concat. Low-cost, small accuracy upside on
  MMLU/LHE where the answer lives in a different galaxy than the query
  resolves to.

### Ingestion-side (Codex-capable, deferred)

- Add `eval_program` + `arg_keys` to `proceduralized_*.jsonl` during
  ingestion. Template a few dozen GSM8K patterns (sum, difference,
  ratio, percentage, time-conversion).
- Populate `rule_strength` on Grammar entries based on
  `supervision_answer` match frequency (rules that resolve correctly
  get higher strength). Populate `superior_to` from competition
  patterns where one rule subsumes another.

## Stable checkpoint

Everything above is diagnostic. No code changes in this commit. The math
path still has the improvements from `70c465c3` (grammar-ranked fallback,
multi-operand chains, Book/proceduralized galaxies in query list).

Baselines held: GSM8K 10% / MMLU 20% / Math 10%.

## Files inspected

- `/K3D/Knowledge3D.local/galaxies/Grammar.jsonl` (103,039 lines; 20k scanned)
- `/K3D/Knowledge3D.local/galaxies/proceduralized_gsm8k_train_10.jsonl`
- `/K3D/GitHub/Knowledge3D/knowledge3d/knowledgeverse/knowledgeverse.py`
  (template binder at line 5931, direct-eval gate at line 5895)
