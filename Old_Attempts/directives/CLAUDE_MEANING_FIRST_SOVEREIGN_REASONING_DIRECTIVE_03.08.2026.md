# Claude Architecture Directive: Meaning-First Sovereign Reasoning -- Analysis and Next Steps

**Date:** March 8, 2026
**From:** Claude (Architecture) + Daniel (Direction)
**To:** Codex (Implementation)
**Context:** Codex delivered MeaningAtom layer + sovereign RPN scoring. ARC 10/10, Math 20/20 restored, LHE 2/10 with real PTX evidence (`gpu_calls=9`). The scoring/selection core is now sovereign. The upstream candidate generation is not.

---

## What Codex Built (Analysis)

### 1. MeaningAtom Layer (meaning_first_reasoning.py) -- Good Foundation

`MeaningAtom` is a frozen dataclass that extracts structured meaning from Galaxy evidence rows:

```
atom_id, concept_ref, canonical_name, domain, subject, subfield,
source_pass, confidence, forms, related_refs, symlinks, semantics, summary
```

**What works:**
- Extracts `meaning_ref`, `formalizes_ref`, `reasons_about_ref` from Galaxy metadata -- these ARE meaning-layer references (Layer 2 per FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md Section 1.3)
- `fuse_meaning_atoms()` merges atoms by `concept_ref` -- this implements the Save Information Principle (reference, don't duplicate)
- `forms` tuple collects canonical_name + aliases + keywords -- this captures multiple FORMS for the same MEANING
- `symlinks` tuple collects Galaxy symlinks and word_refs -- this is the cross-Galaxy navigation path

**What's still host-side Python:**
- `_tokenize_semantic()` uses `re.split()` on the host CPU -- this should be a Word Galaxy lookup
- `_coerce_list()` and `_dedupe_keep_order()` are pure Python utilities -- acceptable for ingestion/prep but not for hot-path reasoning

**Verdict:** MeaningAtom is a correct INTERMEDIATE step. It extracts the right fields. But the atoms are built on the host and consumed by Python workers. The sovereign path would be: Galaxy query returns entries -> entries already contain meaning-layer fields -> RPN programs navigate meaning via symlinks -> no Python extraction needed.

### 2. Sovereign RPN Scoring (_score_candidates_sovereign) -- Good

The scoring function in `LHEReasoningSwarm` now:
1. Builds RPN expressions per candidate: `support worker_bonus + triangulation + meaning_support + format_bonus + contradiction -`
2. Evaluates batch via `ModularRPNEngine.evaluate_batch()`
3. Logs `gpu_calls=9` -- confirmed PTX execution

This is genuinely sovereign. The RPN engine evaluates the arithmetic on the PTX stack. The trace confirms it ran on GPU outside the sandbox.

### 3. Workers -- Still Language-Surface Pattern Matchers

Despite MeaningAtom being available, the workers still:

**FormulaReasoningWorker (lines 192-303):**
- Hardcoded answer template for gamma matrices (line 244-252) -- still there
- `_FORMULA_PATTERNS` are regex patterns matched against evidence TEXT (lines 214-220)
- Scoring: token overlap between `goal_tokens` and `candidate_tokens` (Python sets)
- Skeleton selection based on `prompt_markers` keyword matching (lines 136-139)

**ConceptMatchingWorker (lines 305-416):**
- Uses `meaning_atoms` but scores by `goal_tokens & atom_tokens` overlap (Python sets, line 333)
- Falls back to iterating evidence rows and extracting candidates from field text (lines 356-389)

**ProceduralExecutionWorker (lines 418-829):**
- `_solve_two_step_substitution()`: Simulated annealing with English frequency tables, English bigrams, English common words (lines 437-730) -- **language-specific, sovereignty violation**
- `_solve_clue_chain()`: Hardcoded `_CLUE_FACT_REGISTRY` with answer strings (lines 460-481) -- **template trap**
- Chess: regex extraction of algebraic notation from evidence text (lines 798-809)

**EvidenceSynthesisWorker (lines 832-939):**
- Good: uses MeaningAtom canonical_name and summary
- Still falls back to `_iter_semantic_field_values()` which iterates evidence fields as Python strings

**Pattern:** Workers RECEIVE MeaningAtoms but PROCESS them with Python string/regex operations. The MeaningAtom is consumed as a Python dataclass, not as a Galaxy-navigable meaning reference.

---

## What's Wrong (Grounded in Specs)

### Problem 1: Candidate Generation is Host-Side Heuristic

Per FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md Section 1.4, Layer 3 Rules are "executable RPN programs." The workers should execute RPN programs that navigate Galaxy to find candidates. Instead, they scan Python strings with regex.

**Current flow:**
```
Evidence rows (Galaxy query result)
  -> Python dict access (entry.get("name"))
  -> Python regex (re.findall pattern, field_text)
  -> Python set intersection (goal_tokens & candidate_tokens)
  -> Python sorting (proposals.sort)
  -> [sovereign] RPN scoring of final candidates
```

**Spec-aligned flow:**
```
Evidence rows (Galaxy query result)
  -> Each row has concept_ref (meaning ID) + symlinks (cross-Galaxy refs)
  -> RPN program navigates: concept_ref LOAD_GALAXY -> follow symlinks -> extract canonical
  -> RPN program composes: candidates from meaning-layer navigation
  -> RPN scoring of candidates
```

The gap is between steps 1 and 4. The sovereign scoring at the end is correct but insufficient -- it scores candidates that were generated by language-surface heuristics.

### Problem 2: ProceduralExecutionWorker is English-Specific

The cipher solver (`_solve_two_step_substitution`) contains:
- `_ENGLISH_FREQ = "ETAOINSHRDLCUMWFGYPBVKJXQZ"` -- English letter frequency
- `_COMMON_WORDS = ("THE", "AND", ...)` -- English words
- `_GOOD_BIGRAMS = ("TH", "HE", "IN", ...)` -- English bigrams
- `_LANGUAGE_SAMPLE` -- English text sample

Per DUAL_CLIENT_CONTRACT_SPECIFICATION.md, K3D stars encode MEANING with FORM as a separate layer. A cipher is a form-level transformation. The correct approach:

1. The cipher's substitution mapping is a **Layer 3 Rule** (Grammar Galaxy transformation)
2. The plaintext is a **Layer 2 Meaning** sequence (Word Galaxy entries)
3. Frequency analysis should query **Character Galaxy** for character frequency distributions (language-agnostic -- stored per language as Galaxy entries)
4. Word validation should navigate **Word Galaxy** symlinks (does this character sequence map to a known meaning?)

This makes cipher decoding work for ANY K3D-registered language, not just English.

### Problem 3: Hardcoded Fact Registry

`_CLUE_FACT_REGISTRY` (lines 460-481) contains:
```python
{"match_all": ("logical", "depth", "reciprocal", "charles", "bennett"), "value": "crypticity"},
{"match_all": ("gell-man", "didn't commute"), "value": "operators"},
```

These are template answers, not reasoning. They should be Galaxy entries queried by meaning reference, not hardcoded Python dicts.

### Problem 4: Skeleton Selection is Keyword Matching

`_select_skeletons()` (lines 120-139) matches skeletons by checking if `prompt_markers` appear in `prompt.lower()`. This is language-surface matching.

Per FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md Section 1.5, Layer 4 Meta-Rules have:
- `condition`: RPN predicate (evaluates to true/false)
- `action`: RPN program (what to execute)

Skeleton selection should be: the Meta-Rule's `condition` RPN program evaluates against the parse_bundle's fused entity graph. If condition evaluates true, the Meta-Rule's `action` RPN program executes. No keyword matching needed.

---

## GPU Environment Note

**CRITICAL for Codex:** When `cuInit` fails in the sandbox, it's because the GPU isn't exposed. Per `docs/ENV_POLICY.md` line 54:

> On the Debian 14 workstation the KDE session runs on the iGPU; export `CUDA_VISIBLE_DEVICES=0` before launching tmux so the RTX 3070 is exposed inside the conda shell.

**Required setup for GPU runs:**
```bash
export CUDA_VISIBLE_DEVICES=0
# Then activate conda env:
source /home/daniel/miniforge/etc/profile.d/conda.sh
conda activate k3d-cranium
# Or use SSD env:
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium env PYTHONPATH=$(pwd) pytest ...
```

The sandbox (Codex sandbox) does NOT have GPU access. GPU tests must run outside the sandbox via tmux with `CUDA_VISIBLE_DEVICES=0` set. This is not a code bug -- it's an environment setup requirement.

---

## Recommended Next Steps (Priority Order)

### Step 1: Move Candidate Generation from Python Regex to Galaxy Navigation

**What:** Replace the regex-based candidate extraction in workers with meaning-layer Galaxy navigation.

**Currently:** Workers iterate `evidence_rows`, extract field text with Python, match regex patterns, score with Python set intersection.

**Target:** Workers receive MeaningAtoms (already built). Instead of tokenizing and intersecting Python sets, the worker should:

1. For each MeaningAtom, follow its `concept_ref` to the Galaxy entry
2. Extract the entry's `rpn_program` field (if it has one -- many Galaxy entries DO have RPN programs)
3. If the entry has `symlinks`, follow them to related entries (Word -> Reality, Math -> Grammar)
4. Build candidate from the MEANING fields (canonical_name, definition) not from raw text scanning

**Concrete change in FormulaReasoningWorker:**
```
Current (line 266-300):
  for field_name, field_text, field_score in field_values:
      for pattern in self._FORMULA_PATTERNS:
          for match in re.findall(pattern, field_text):
              # score by token overlap

Target:
  for atom in meaning_atoms:
      if atom.domain in ("math", "physics"):
          # atom.concept_ref -> Galaxy entry -> entry.rpn_program
          # atom.symlinks -> follow to Math Galaxy entries with formulas
          # Compose RPN program from connected entries
          # Evaluate on PTX stack -> candidate = result
```

**Why this matters:** The candidates would be COMPUTED from Galaxy navigation, not EXTRACTED from text. This is the difference between "find a formula in the text" and "compose a formula from connected knowledge."

### Step 2: Replace English-Specific Cipher Logic with Form+Meaning Navigation

**What:** Remove `_ENGLISH_FREQ`, `_COMMON_WORDS`, `_GOOD_BIGRAMS`, `_LANGUAGE_SAMPLE` and replace with Galaxy-navigable language-agnostic equivalents.

**Target architecture:**
1. Character frequency distributions: stored as **Character Galaxy** entries per language. The TRM queries the appropriate language's frequency distribution.
2. Word validation: after applying a candidate substitution mapping, check if resulting character sequences are valid **Word Galaxy** entries (i.e., sequences that have meaning-layer references).
3. Bigram/n-gram scoring: stored as **Grammar Galaxy** rules per language.

This makes cipher decoding work for Portuguese, Japanese, or any K3D-registered language -- same algorithm, different Galaxy entries.

**Intermediate step (pragmatic):** If full Galaxy population for character frequencies isn't ready, at minimum:
- Move the frequency/bigram data INTO Grammar Galaxy entries (not hardcoded Python)
- Query them via Galaxy lookup at runtime
- This at least makes them sovereign (Galaxy-stored, not code-stored) even if language selection is still manual

### Step 3: Replace Hardcoded Fact Registry with Galaxy Queries

**What:** Remove `_CLUE_FACT_REGISTRY` and `gamma_bivector_sandwich_identity` hardcoded answer.

**Target:** These facts should be Reality Galaxy entries, queried by meaning reference. The clue chain solver should:
1. Parse each clue into entity references (already done by four-pass)
2. Query Galaxy for each entity reference
3. Extract the answer from the Galaxy entry's CONTENT/meaning fields
4. Compose the final answer from sub-results

The gamma matrices identity should be a Math Galaxy entry with an RPN program that COMPUTES `-(d-2k)^2 + d`, not a hardcoded string.

### Step 4: Move Skeleton Selection to Meta-Rule Condition Evaluation

**What:** Replace keyword-matching skeleton selection with Layer 4 Meta-Rule condition evaluation.

**Currently:**
```python
if skeleton.prompt_markers and not any(marker in prompt_lower for marker in skeleton.prompt_markers):
    continue
```

**Target:**
```python
# Each skeleton has a condition RPN program (Layer 4 Meta-Rule)
# Condition evaluates against parse_bundle entities
# Example: "domain 'math' == goal_kind 'symbolic' == AND"
condition_result = rpn_engine.evaluate(skeleton.condition_rpn, context=parse_bundle)
if condition_result:  # RPN predicate returned true
    selected.append(skeleton)
```

This makes skeleton selection sovereign -- RPN predicates evaluated on PTX stack, not Python keyword matching.

### Step 5: End-to-End Sovereign Reasoning Chain

Once Steps 1-4 are done, the full LHE reasoning chain becomes:

```
Question -> Four-Pass (existing, sovereign)
  -> Meta-Rule condition evaluation (Step 4, sovereign)
  -> Selected skeleton's action RPN program executes:
     -> LOAD_GALAXY concept_ref -> follow symlinks (Step 1, sovereign)
     -> Compose sub-results via RPN (Step 1, sovereign)
     -> Score candidates via RPN batch (already sovereign)
  -> Return best candidate
```

Zero Python regex. Zero hardcoded answers. Zero language-specific constants. Pure Galaxy navigation + RPN composition + PTX evaluation.

---

## Constraints (Unchanged)

1. **ARC 10/10 and Math 20/20 must not regress.** Test after each step.
2. **No new external dependencies.** Workers use Galaxy queries + RPN + PTX only.
3. **MeaningAtom can stay as prep-layer** for now -- it's acceptable as an intermediate data structure that bridges Galaxy query results to worker logic. But long-term, workers should query Galaxy directly.
4. **Incremental migration.** Don't rewrite all workers at once. Start with FormulaReasoningWorker (Step 1) since math questions are 4/10 of LHE failures.

---

## The Principle

Daniel: "the skeleton and such works should be based on meaning, not language -- and that's how we make this truly universal."

K3D's form+meaning architecture (FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md) already defines this:
- **Layer 1 (Form):** How things look (Character Galaxy)
- **Layer 2 (Meaning):** What things mean (Word Galaxy, Reality Galaxy)
- **Layer 3 (Rules):** How to transform (Grammar Galaxy RPN programs)
- **Layer 4 (Meta-Rules):** When/why to apply rules (condition + action RPN)

Reasoning operates on Layers 2-4. Form (Layer 1) is only relevant when the question specifically involves form (like a cipher, which transforms character sequences). Even then, the form-level operations reference Galaxy entries, not hardcoded Python constants.

The current workers partially bridge to meaning (MeaningAtom extracts meaning-layer fields) but still PROCESS at the language surface (Python regex on English text). The next step is to move PROCESSING to the meaning layer: Galaxy navigation, symlink following, RPN composition. That's when K3D reasons with knowledge instead of scanning text.
