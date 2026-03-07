# Claude Architecture Directive: LHE Knowledge Density + ARC Primitive Expansion

**Date:** March 7, 2026
**From:** Claude (Architecture) + Daniel (Direction)
**To:** Codex (Implementation)
**Context:** Math 20/20 solid. ARC moved from 0/10 to 1/10 via four-pass composition (first real compositional solve). LHE structurally on four-pass boundary but 0/10 due to knowledge density + synthesis quality.

---

## Assessment of Current State

### Math: 20/20 -- REGRESSION GUARD HOLDS

Do not touch math. It works.

### ARC: 1/10 -- Four-Pass Composition Validated

**What proved itself:**
- Task `00576224` solved correctly via `arc_four_pass` source
- `composition_depth: 2` -- first multi-step transform chain (was locked at 1.0)
- `generation_failure_rate: 0.9` (was 1.0)

**What limits further progress:**
- 7/9 remaining failures are `family` mismatches -- Grammar Galaxy needs more compositional primitives
- `composition_depth` distribution: `{1: 9, 2: 1}` -- only 1 task reached depth 2
- The `tile_pattern`, `phase_shift`, `color_remap`, `object_extract`, `object_place`, `grid_resize`, `conditional_fill`, `symmetry_complete` primitives from the Universal Four-Pass Directive need to be added

**Codex's proposed approach** (expand from audited failures) is correct for ARC. Each failed task reveals which primitives are missing. Add them as Grammar Galaxy entries (compositional, not family-specific), rerun, measure delta.

### LHE: 0/10 -- Three Fixable Problems + One Structural Dependency

The four-pass structure is now in place. The LHE path runs forward entity extraction, backward goal extraction, fusion, and evidence query. But three implementation problems prevent it from producing correct answers:

#### Problem 1: Open-Ended Answer Synthesis Returns Galaxy Names

`_synthesize_lhe_open_answer()` (main.py:740-774) extracts candidates from evidence text using regex patterns:
```python
r"\$[^$]{1,160}\$",          # LaTeX
r"\\\([^)]{1,160}\\\)",      # inline math
r"\b-?\d+(?:\.\d+)?\b",     # numbers
r"\b[A-Z][A-Za-z0-9#+-]{1,40}..."  # capitalized words
```

It then scores by token overlap with goal tokens. Result: Physics questions get `"Tool"` (the Galaxy name appears in entry text and matches as a capitalized word). The Cybersecurity question gets `"3"` (a random number extracted from evidence).

**Fix:** The candidate extraction must be domain-aware. For physics, candidates should be mathematical expressions. For chess, candidates should be move notation. For general knowledge, candidates should be proper nouns or domain terms. The `domain_hint` from Pass 3 (goal building) should filter candidate extraction patterns.

Additionally, when the evidence rows contain actual domain knowledge (a Galaxy entry about a physics formula, a Reality Galaxy entry about a concept), the synthesis should extract the MEANING from the entry, not pattern-match surface text. The entry's `rpn_program`, `description`, `content`, or `metadata.semantics` fields carry the actual knowledge.

#### Problem 2: Multiple-Choice Scoring is Token Overlap, Not Contrastive

`_score_lhe_option()` (main.py:688-724) scores options by counting token overlaps:
- `option_tokens & goal_tokens` weighted 0.35
- `option_tokens & fused_tokens` weighted 0.2
- `option_tokens & evidence_tokens` weighted 0.45 + rank_weight

This is surface-level matching. The Philosophy question scored "Weak Quality Addition" above "Weak Non-Sadism" because more tokens overlapped, not because the concept matched.

**Fix:** Apply the ternary contrastive principle from TERNARY_CONTRASTIVE_LEARNING_SPECIFICATION.md:
- **+1 (confirm):** Option text appears in or closely matches Galaxy entry content/description
- **-1 (eliminate):** Option text CONTRADICTS Galaxy entry (e.g., entry says "X is NOT Y" and option claims Y)
- **0 (explore):** No Galaxy evidence for or against this option

For multiple-choice, elimination is as valuable as confirmation. If the Galaxy contains knowledge that rules out 3 of 4 options, the remaining option wins even without direct confirmation.

The symlink nature of K3D matters here: "Weak Non-Sadism" as a concept should connect to ethics/philosophy entries in Reality Galaxy. If the Word Galaxy entries for "sadism", "non", "weak" have symlinks to Reality Galaxy concepts about population ethics, the scoring can navigate those connections instead of just counting shared tokens.

#### Problem 3: Domain Routing Defaults to Grammar+Tool

Current routing for LHE:
```
Chess       -> Grammar + Tool      (SyntaxSpecialist)
Philosophy  -> Grammar + Tool      (SyntaxSpecialist)
Trivia      -> Grammar + Tool      (SyntaxSpecialist)
Physics     -> Reality + 3DObjects + Tool + Math + Grammar (MechanicsSpecialist)
Cybersecurity -> Grammar + Tool    (SyntaxSpecialist)
```

Physics gets the right galaxies. Everything else gets Grammar+Tool, which contains grammar rules but no domain knowledge about chess moves, philosophical concepts, or trivia facts.

**Fix:** ALL LHE domains should include Reality Galaxy in their query set. Reality Galaxy is where domain knowledge lives (per THREE_BRAIN_SYSTEM_SPECIFICATION.md). Even if Reality Galaxy is sparse for philosophy right now, the routing should ALWAYS include it so that as augmentation populates domain knowledge, LHE accuracy improves automatically.

Recommended routing for LHE:
```
ALL LHE domains -> Reality + Grammar + Word + [domain-specific galaxies]
Physics         -> + Math + 3DObjects
Chemistry       -> + Math
Biology         -> + Math
Math domain     -> redirect to MathSpecialist (already implemented)
```

#### Structural Dependency: Galaxy Knowledge Density

Daniel's core point: LHE accuracy depends on what's IN the galaxies. The four-pass structures the REASONING, but without knowledge in Reality/Grammar/Word galaxies, it has nothing to reason with.

The augmentation process populates galaxies with domain knowledge. LHE 0/10 with empty galaxies is expected. The goal is:
1. Four-pass structure wired (DONE)
2. Routing includes correct galaxies (needs fix)
3. Synthesis quality improved (needs fix)
4. Augmentation populates domain knowledge (paused, needs resumption)
5. LHE accuracy improves as knowledge density grows

---

## Daniel's Direction: Stars Are Meaning-Centric, Symlinks Ensure Connection

Key architectural insight from Daniel:

Galaxy entries (stars) are **meaning-centric** -- each star represents a concept, not a surface form. The same concept exists across multiple galaxies:
- "five" in Word Galaxy (linguistic form)
- "5" in Number Galaxy (numeric form)
- "num_5" in Math Galaxy (computational form)
- Character sequence [f,i,v,e] in Character Galaxy (glyph form)

These are NOT duplicates. They are different VIEWS of the same meaning, connected by symlinks. The Save Information Principle (DUAL_CLIENT_CONTRACT_SPECIFICATION.md section 1.6) says: reference, don't duplicate.

**For LHE, this means:**
- "Weak Non-Sadism" should be a concept star in Reality Galaxy (philosophy domain)
- "non-sadism" should exist in Word Galaxy with `reality_ref` pointing to the concept
- When the four-pass fuses entities and queries galaxies, it should find the concept star via symlink navigation, not just token matching

**Current state:** Number-Word symlinks were added for math (and it works -- 20/20). The same pattern must extend to ALL domain concepts as augmentation proceeds. The augmentation ingestion pipeline must create BOTH the Reality Galaxy entry AND the Word Galaxy symlink for every domain concept it ingests.

**Language note:** Current entries have English metadata only. That's correct for now (benchmarks are in English). But the symlink architecture supports future multilingual forms -- "cinco" and "five" and "5" all symlink to `num_5`. Same principle applies to domain terms: "non-sadism" in English and "no-sadismo" in Portuguese are different Word Galaxy entries symlinking to the same Reality Galaxy concept star.

---

## Priority Actions for Codex

### Priority 1: ARC Grammar Galaxy Primitives (Immediate)

From the Universal Four-Pass Directive, add to `foundational_operations_bootstrap.py`:
```
tile_pattern       -> extract source region, repeat across grid
phase_shift        -> offset tiling by row/column index
color_remap        -> map color A to color B
object_extract     -> identify connected components in grid
object_place       -> place object at position
grid_resize        -> change output grid dimensions
conditional_fill   -> fill region based on neighbor condition
symmetry_complete  -> complete pattern by symmetry axis
border_fill        -> fill border with specified pattern
crop_region        -> extract sub-region from grid
overlay_grid       -> layer one grid on top of another
flood_fill         -> fill connected region with color
connected_components -> identify and label connected regions
```

These are Grammar Galaxy entries with `rpn_program` fields. The TRM composes them. They are NOT Python functions.

Analyze the 9 failed ARC tasks from the smoke artifact. For each, identify which compositional primitives would be needed. Add those primitives. Rerun.

### Priority 2: LHE Routing Fix (Quick Win)

Add Reality Galaxy to ALL LHE domain routes, not just physics. This is a routing change, not a knowledge change. The knowledge will come from augmentation, but the routing must be ready.

### Priority 3: LHE Open-Ended Synthesis Quality (Before Augmentation)

The `_synthesize_lhe_open_answer` method must:
1. Use `domain_hint` to select appropriate candidate extraction patterns
2. Prefer Galaxy entry `content`/`description`/`rpn_program` over surface text regex
3. For math/physics domains: look for mathematical expressions in entry metadata
4. For knowledge domains: look for proper nouns, concept names in entry metadata
5. Fall back to current regex extraction only when metadata is empty

### Priority 4: LHE Multiple-Choice Contrastive Scoring (Before Augmentation)

The `_score_lhe_option` method must:
1. Check for CONTRADICTION signals (entry says "not X", option claims X) -> score penalty
2. Check for CONFIRMATION signals (entry explicitly names the option) -> score bonus
3. Use symlink navigation: option text -> Word Galaxy -> Reality Galaxy concepts -> confirm/deny
4. Current token overlap scoring can remain as baseline, but contrastive signals should override

### Priority 5: Resume Augmentation (After Priorities 1-4)

Augmentation is what fills the galaxies. LHE will remain low until domain knowledge density grows. But priorities 1-4 ensure that when knowledge IS present, it's used effectively.

---

## What NOT to Do

1. **Do NOT add LHE question-specific logic.** No "if domain == chess then..." special cases. The four-pass + Galaxy navigation handles all domains the same way. Domain differences come from Galaxy CONTENT, not Python branches.

2. **Do NOT duplicate concepts across galaxies.** A philosophy concept goes in Reality Galaxy with symlinks from Word Galaxy. Not separate entries in each.

3. **Do NOT abandon the four-pass structure for LHE.** It's 0/10 now because galaxies are sparse, not because the structure is wrong. The structure is the same one that took math from 1/20 to 20/20.

4. **Do NOT optimize LHE scoring for the 10 smoke questions.** The scoring must be domain-agnostic and compositional. It will generalize when knowledge density grows.

---

## Success Criteria

1. ARC Grammar Galaxy has compositional transform primitives (loaded at init)
2. ARC composition_depth distribution shows more tasks above depth 1
3. ARC accuracy >= 2/10 (from adding missing primitives)
4. LHE routes include Reality Galaxy for ALL domains
5. LHE open-ended synthesis uses entry metadata, not just regex on text
6. LHE multiple-choice scoring has contrastive elimination (not just token overlap)
7. Math remains 20/20 (regression guard)
8. No numpy/cupy/scipy in hot path
9. All new primitives are Galaxy entries, not Python if/else branches

---

## Grounding in Specs

| Spec | Section | Relevance |
|------|---------|-----------|
| TERNARY_CONTRASTIVE_LEARNING_SPECIFICATION.md | Section 3 (line 591) | Contrastive +1/-1/0 for option scoring |
| DUAL_CLIENT_CONTRACT_SPECIFICATION.md | Section 1.6 | Save Information Principle -- symlinks, not duplication |
| THREE_BRAIN_SYSTEM_SPECIFICATION.md | Galaxy Universe | Reality Galaxy holds domain knowledge |
| KNOWLEDGEVERSE_SPECIFICATION.md | Region 2 | Galaxy Universe is active AI memory |
| FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md | Section 3.4 | TASK DECOMPOSE opcode for multi-step chains |

---

## The Principle

The stars are meaning-centric. "Non-sadism" is a concept. It lives in Reality Galaxy as a philosophical concept star. "non-sadism" lives in Word Galaxy as a linguistic form, symlinking to the concept. When the TRM navigates for a philosophy question, it finds the concept through any of its forms.

The four-pass decomposes the question. The Galaxy holds the knowledge. The symlinks connect the forms to the meanings. The augmentation fills the Galaxy. Each piece does one job.

Right now LHE has the structure but not the knowledge. ARC has the structure and is gaining primitives. Math has both and scores 20/20. The path is clear: add primitives, fix routing, fix synthesis quality, resume augmentation.
