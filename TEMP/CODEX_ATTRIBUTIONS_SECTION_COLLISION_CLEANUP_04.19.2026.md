# ATTRIBUTIONS.md Section-Collision Cleanup — Codex Spec

**Date**: 2026-04-19
**Owner**: Claude (spec author), Codex (executor — one commit, single clean renumber)
**Scope**: Renumber colliding top-level sections in `ATTRIBUTIONS.md`. **Additive-destruction rule applies — this is a pure structural rename; line count MUST NOT decrease.** See [`feedback_codex_additive_destruction_pattern.md`](../../home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/feedback_codex_additive_destruction_pattern.md).

---

## Context

`ATTRIBUTIONS.md` currently has **six top-level section-number collisions** that pre-date the 2026-04-18 restoration. They were present in the `c742e2dd~1` canonical version and were not introduced by any recent edit. They make the file hard to cross-reference (e.g., "see §5" is ambiguous — Software & Tools? Lexical? Datasets?).

### Current collisions

| Line | Current header | Problem |
|------|---------------|---------|
| 30 | `## 0.0 Pop Culture Influences — The Inspirational Vision` | Non-standard numbering before §0 |
| 83 | `## 5. Lexical Resources for Word Galaxy` | First §5 — displaced; belongs under Datasets |
| 99 | `## 0. Foundational Infrastructure` | Second §0 (conflicts with §0.0 above) |
| 489 | `## 1. Research Foundations` | OK |
| 1031 | `## 2. Game Industry Techniques (Repurposed)` | OK |
| 1120 | `## 3. AI/ML Foundations` | OK |
| 1415 | `## 4. Theoretical Foundations & Collaboration` | OK |
| 1720 | `## 5. Software & Tools` | Second §5 |
| 1825 | `## 5. Datasets & Corpora` | **Third §5** |
| 1994 | `## 5.6 Community & Reference Resources` | Belongs under Datasets (§5), numbering inherited from the third §5 |
| 2028 | `## 6. K3D's Novel Contributions` | First §6 |
| 2142 | `## 7. Paper Preparation` | First §7 |
| 2169 | `## 8. Citation Guidelines` | OK |
| 2209 | `## 9. Contact & Collaboration` | OK |
| 2220 | `## 6. Universal Procedural Display Stack (Future Architecture)` | **Second §6** |
| 2395 | `## 7. Carbon Impact & Future-Proofing Philosophy` | **Second §7** |
| 2677 | `## 10. License & Legal` | OK |

---

## Target numbering (after cleanup)

| Current header | Line (current) | New header | Rationale |
|----------------|----------------|------------|-----------|
| `## 0.0 Pop Culture Influences — The Inspirational Vision` | 30 | `## 0.1 Pop Culture Influences — The Inspirational Vision` | Subsection under §0 Foundational Infrastructure |
| `## 5. Lexical Resources for Word Galaxy` | 83 | `## 0.2 Lexical Resources for Word Galaxy` | Relocate under §0 (it's early foundational infrastructure, not applicable-at-§5) — see note 1 below |
| `## 0. Foundational Infrastructure` | 99 | `## 0. Foundational Infrastructure` | Keep — this is the canonical §0 header |
| `## 5. Software & Tools` | 1720 | `## 5. Software & Tools` | Keep as canonical §5 |
| `## 5. Datasets & Corpora` | 1825 | `## 6. Datasets & Corpora` | **Shift up one** to eliminate third §5 |
| `## 5.6 Community & Reference Resources` | 1994 | `## 6.6 Community & Reference Resources` | Re-parent to new §6 (was §5.6 under the removed third §5) |
| `## 6. K3D's Novel Contributions` | 2028 | `## 7. K3D's Novel Contributions` | **Shift up one** |
| `## 7. Paper Preparation` | 2142 | `## 8. Paper Preparation` | **Shift up one** |
| `## 8. Citation Guidelines` | 2169 | `## 9. Citation Guidelines` | **Shift up one** |
| `## 9. Contact & Collaboration` | 2209 | `## 10. Contact & Collaboration` | **Shift up one** |
| `## 6. Universal Procedural Display Stack (Future Architecture)` | 2220 | `## 11. Universal Procedural Display Stack (Future Architecture)` | Renumber to eliminate second §6 and put "Future Architecture" at the end of the major content block |
| `## 7. Carbon Impact & Future-Proofing Philosophy` | 2395 | `## 12. Carbon Impact & Future-Proofing Philosophy` | Renumber to eliminate second §7 |
| `## 10. License & Legal` | 2677 | `## 13. License & Legal` | **Shift up three** so License & Legal remains the final top-level section |

### Note 1 — Section 0.2 Lexical Resources placement

The current line-83 `## 5. Lexical Resources for Word Galaxy` sits *between* §0.0 (line 30) and §0 (line 99). That is already structurally inconsistent. The cleanest fix is to renumber it to `## 0.2 Lexical Resources for Word Galaxy` so it stays in its current line position as a subsection of §0 Foundational Infrastructure. **Do not move the content — only rename the heading.** If a second opinion prefers keeping Lexical under Datasets instead, flag for Daniel; do not relocate without explicit approval (that would be content restructuring, not a rename, and violates the single-commit rule).

---

## Final top-level outline after cleanup

```
## 0. Foundational Infrastructure
  ## 0.1 Pop Culture Influences — The Inspirational Vision
  ## 0.2 Lexical Resources for Word Galaxy
## 1. Research Foundations
## 2. Game Industry Techniques (Repurposed)
## 3. AI/ML Foundations
## 4. Theoretical Foundations & Collaboration
## 5. Software & Tools
## 6. Datasets & Corpora
  ## 6.6 Community & Reference Resources
## 7. K3D's Novel Contributions
## 8. Paper Preparation
## 9. Citation Guidelines
## 10. Contact & Collaboration
## 11. Universal Procedural Display Stack (Future Architecture)
## 12. Carbon Impact & Future-Proofing Philosophy
## 13. License & Legal
```

Zero collisions. 13 top-level sections (plus two subsections under §0 and one under §6).

---

## Execution rules for Codex

### Rule 1 — This is a rename ONLY

- **ONLY header-line changes.** Do not move, reorder, or restructure any content between the renamed headers.
- **Every `## N.` line change is a pure text substitution** on that one line.
- **Do not edit prose** under any section.
- **Do not change the frontmatter** or metadata block at top of file.
- **Do not touch the final `**Last Updated**:` / `**Version**:` block** in this PR (those get a separate append, see Rule 5).

### Rule 2 — Line count non-decreasing

- Current line count: **2,763** (as of 2026-04-19 after §4.4.1 + §6.4 + Material Enablers additions).
- After this cleanup commit: line count MUST be **≥ 2,763**. No line deletions permitted.
- Pre-commit check: `wc -l ATTRIBUTIONS.md` before and after the edit. Both values go in the commit message.

### Rule 3 — No cross-reference breakage

Some in-file cross-references use section numbers. Before renumbering, grep the file for section-number references:

```bash
grep -nE '§[0-9]+(\.[0-9]+)*' ATTRIBUTIONS.md
grep -nE 'section [0-9]+' ATTRIBUTIONS.md
grep -nE 'See §[0-9]' ATTRIBUTIONS.md
```

For every hit where the referenced section is being renumbered (§6 → §7, §7 → §8, etc.), update the reference to the new number in the *same* commit. Do not leave dangling references.

### Rule 4 — Test link integrity

After the rename, run:

```bash
markdown-link-check ATTRIBUTIONS.md || true
grep -nE '\]\(#' ATTRIBUTIONS.md      # anchor links
```

If any anchor link (`[text](#some-section)`) targets a renumbered section, update the anchor text to the new slug.

### Rule 5 — Commit message and log entry

Commit message:

```
docs(ATTRIBUTIONS): renumber top-level sections to eliminate 6 collisions

Pure header-rename cleanup — no content changes, no line deletions.
Line count before: <PRE>. Line count after: <POST>. Delta: <POST-PRE> (≥ 0).

Renumbered:
  5. Lexical Resources       → 0.2 Lexical Resources
  0.0 Pop Culture Influences → 0.1 Pop Culture Influences
  5. Datasets & Corpora       → 6. Datasets & Corpora
  5.6 Community              → 6.6 Community
  6. Novel Contributions      → 7. Novel Contributions
  7. Paper Preparation        → 8. Paper Preparation
  8. Citation Guidelines      → 9. Citation Guidelines
  9. Contact & Collaboration  → 10. Contact & Collaboration
  6. Universal Procedural...  → 11. Universal Procedural...
  7. Carbon Impact           → 12. Carbon Impact
  10. License & Legal         → 13. License & Legal

Cross-references updated: <N>. All anchor links verified.
Spec: TEMP/CODEX_ATTRIBUTIONS_SECTION_COLLISION_CLEANUP_04.19.2026.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

After commit lands, append a one-line entry to the file's trailing `**Major Milestones Since Last Update**:` list:

```
- ATTRIBUTIONS.md top-level section renumbering to eliminate six pre-existing collisions (April 2026)
```

That append is a separate second commit (keeps the structural rename audit-clean).

### Rule 6 — Do NOT do these things

- ❌ Do not "helpfully" consolidate or reorder sections.
- ❌ Do not fix markdown lint warnings in the same commit (do them separately if desired).
- ❌ Do not trim trailing whitespace in the same commit.
- ❌ Do not re-wrap lines or normalize spacing in the same commit.
- ❌ Do not create any new section.
- ❌ Do not touch `ATTRIBUTIONS.md` outside the header-line substitutions enumerated in the table above.

---

## Verification checklist (before PR)

- [ ] 13 top-level section headers in outline order (0, 1, 2, ..., 13)
- [ ] Zero duplicate `## N.` prefixes
- [ ] Line count after ≥ line count before
- [ ] `grep '^## 5'` returns exactly one match (`## 5. Software & Tools`)
- [ ] `grep '^## 6'` returns exactly one match (`## 6. Datasets & Corpora`)
- [ ] `grep '^## 7'` returns exactly one match (`## 7. K3D's Novel Contributions`)
- [ ] All `§N.M` cross-references resolve to an existing header
- [ ] File still opens and renders correctly in GitHub / VSCode
- [ ] Commit message contains the line-count before/after figures

---

## Why we do this before Paper A submission

Paper A will cite ATTRIBUTIONS sections by number (e.g., "See ATTRIBUTIONS §4.4.1 for semantic gravity provenance; §6.4 for external validation of the Form → Meaning architecture; §7.1 for novelty claims"). With the current collisions, those citations would be ambiguous. Fixing the collisions once, before the paper draft locks, saves us from having to regenerate every section number reference later.

---

**Estimated effort**: 30-45 min for Codex (careful text-substitution pass + cross-reference audit + link check).
**Blocks**: Paper A §3 draft referencing ATTRIBUTIONS by section number.
**Blocked by**: Nothing — can execute any time Codex is available.
**Location**: `TEMP/CODEX_ATTRIBUTIONS_SECTION_COLLISION_CLEANUP_04.19.2026.md`
