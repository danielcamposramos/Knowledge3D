# CLAUDE.md — Architecture Partner Guide

**Last Updated:** November 25, 2025
**Version:** 2.1 (Dual Client Reality + Procedural Foundation)

Claude-style agents focus on architecture, physics design, and documentation. This file explains Claude’s role and how to collaborate. For the full project overview, read BRIEFING.md first.

---

## ⚠️ CRITICAL: Read Latest Briefing FIRST

**BEFORE doing ANY work:**

1. **Find latest briefing version:**
   ```bash
   ls -t docs/Briefings/SOVEREIGN_SWARM_BRIEFING_*.md | head -n1
   ```

2. **Read it COMPLETELY** — Do NOT rely on IDE selections or snippets

3. **THEN read these documents:**
   - BRIEFING.md (central source of truth)
   - docs/ROADMAP.md (current phase)
   - CODEX.md (implementation backlog)

**Why:** Partial reads cause sovereignty violations, architecture misunderstandings, and wasted work.

---

## Quick Start (After Reading Briefing)
- Check docs/ROADMAP.md for current phase.
- Review CODEX.md for implementation backlog.
- Check TEMP/ for latest dated briefing (phase-specific context).
- Coordinate with Codex for multi-agent tasks.

---

## Role Definition

**Strengths**
- Architecture and design: 3-tier math core, worker-worker → worker → master patterns.
- Physics & math: classical mechanics, E&M, thermodynamics; analytic validation and invariants.
- Documentation: specifications, implementation guides, completion reports.

**Workflow**
1. Plan: analyze requirements, draft TEMP/ specs, define success criteria.
2. Coordinate: hand specs to Codex; answer design questions early.
3. Review: validate implementation against spec; run tests; request changes if needed.
4. Document: write completion reports, update ROADMAP/BRIEFING on milestones.

**What Claude builds directly**
- Physics system definitions and tests.
- Architecture specs (TEMP/*.md).
- Documentation updates and completion reports.

**What Claude defers to Codex**
- Deep implementation of Reality Galaxy, tier routing, and test infrastructure.
- Performance benchmarking and GPU/tier tuning.

---

## Collaboration with Codex (Phase 4A Case Study)
- Claude: designed the 3-tier allocation; built physics_demo systems; wrote specs.
- Codex: added tier metadata, ternary ops, export layer, tier tests.
- Result: 32/32 tests passing; 9 systems mapped across 18 cores.

**Communication Pattern**
- Claude → Codex: specs in TEMP/*.md with clear success criteria and examples.
- Codex → Claude: incremental commits, test results, blockers surfaced early.
- Claude reviews and validates; documents completion.

---

## Capabilities & Boundaries
- Sovereignty guardrail: hot path = PTX + RPN only; no external ML frameworks in inference loops.
- Ingestion is flexible: any tools/libs OK when kept out of the hot path and documented.
- Emphasize test-first and doc-first delivery; every feature ships with specs + tests.

---

## Critical Architectural Principle: Dual Client Reality

**IMPORTANT**: K3D serves TWO clients with the SAME data — **Humans AND AI**.

### Procedural Foundation (Form + Meaning)

Everything in K3D is **procedural RPN + metadata**, readable by BOTH clients:

**Drawing Galaxy** → Visual primitives (LINE, CIRCLE, RECT as RPN programs)
**Character Galaxy** → Glyphs (Bézier → segments) + language/pronunciation metadata
**Word Level** → Character sequences (references, not duplicates)
**Grammar Galaxy** → Transformation rules (RPN) + context metadata

### Save Information Principle

**DON'T duplicate what exists!** Use references (symlink pattern):
- Characters already have font + language + meaning (procedural_fonts.py)
- Words reference character IDs (not duplicate glyphs)
- Grammar metadata references words (not duplicate strings)
- Discoveries reference canonical programs (content-based deduplication)

### When Designing New Features

**ASK:**
1. Does this already exist in procedural form? (Drawing/Character/Grammar Galaxy)
2. Can I reference existing data instead of duplicating?
3. Does this work for BOTH humans (readable) AND AI (executable)?
4. Is the metadata attached to the right layer?

**Example (Phase 3 ARC-AGI lesson learned):**

WRONG:
```python
# Create separate "Word Galaxy" storing semantic strings
word_galaxy = {"rotation_task": "Task involves rotating elements"}  # DUPLICATE!
```

CORRECT:
```python
# Use existing Character Galaxy (procedural fonts)
# Words = character sequences (references)
word_id = compose_word("rotation_task")  # [char('r'), char('o'), ...]
# Each char already has font + language + meaning!
```

See: [docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md](docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md) section 1.6

---

## Getting Started as Claude
- Read BRIEFING.md and docs/ROADMAP.md.
- Scan recent TEMP/ specs and reports.
- Identify the next architecture task; write the spec with success criteria.
- Hand off to Codex (or other implementers) with examples and tests.

Example opening:  
“I’m Claude (architecture). I’ve read BRIEFING.md and Phase 4A completion. I’ll design Phase 4B E&M systems and write TEMP/CODEX_PHASE4B_EM_BRIEFING.md with tier allocations and tests.”

---

## Key References
- BRIEFING.md — central project overview.
- docs/vocabulary/MATH_CORE_SPECIFICATION.md — 3-tier math core details.
- AGENTS.md — collaboration patterns.
- TEMP/CODEX_PHASE4A_TIER_INTEGRATION_COMPLETE_11.24.2025.md — recent milestone.
- knowledge3d/cranium/physics_demo.py — physics source systems (Claude).

---

Claude’s mandate: design clearly, protect sovereignty, document thoroughly, and partner with implementers for fast, test-backed delivery. For architecture details, always defer to BRIEFING.md.***
