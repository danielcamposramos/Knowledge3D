# WebML Proposal Patterns (Success, Stall, and Scope)

Date: 2026-03-03
Data basis: all 16 non-PR issues in `webmachinelearning/proposals` plus comment timelines.

## 1. What "Successful" Looks Like in This Repo

There are two concrete success trajectories:

### Pattern A: Absorption into WebNN work items
Evidence: issue #2 (Operation-specific APIs)
- Maintainer/editor discussion translated requirements into WebNN PRs/issues.
- Proposal thread remained open, but work moved into normative/implementation channels.

Success markers:
1. Editor-level technical engagement.
2. Direct links to WebNN PR/issues.
3. Clear statement that requirements are being handled in the main track.

### Pattern B: Migration to dedicated incubation repo
Evidence: issues #5, #12, #13
- #5 moved practical work to `webmachinelearning/hybrid-ai`.
- #12 evolved into `webmachinelearning/webmodelcontext` and then `webmachinelearning/webmcp` with contributor onboarding.
- #13 closed as completed after move to WebMCP issue #7.

Success markers:
1. Maintainer explicitly acknowledges direction support.
2. New dedicated repository created.
3. Contributor onboarding and explainer-first process.
4. Issue either remains as index pointer or closes as completed-by-migration.

## 2. What Stalls

### Pattern C: Seed proposal with no discussion
Evidence: #6, #9, #14
- No comment thread means no process momentum.

Stall marker:
- Proposal exists but no maintainer/public feedback loop starts.

### Pattern D: Broad concept without immediate implementation path
Evidence: #3, #10 (and partially #11)
- Interesting themes but no concrete explainer/prototype plan in-thread.

Stall markers:
1. Low comment volume.
2. No milestone statement.
3. No linked repo/PR path.

### Pattern E: Scope tension with adjacent W3C groups
Evidence: #4
- Proposal intent discussed, but guidance redirects parts toward another standards venue.

Stall marker:
- Problem is plausible, but venue/scope fit remains unresolved.

## 3. "Rejected" Proposal Pattern

Closed-issue evidence is limited (only one closed issue, #13, and it is completed-by-migration).

Observed in this snapshot:
- Rejected explicitly: none
- Closed not planned: none
- Completed via migration: yes

Practical conclusion:
- The key risk is not outright rejection; it is prolonged limbo without an explainer/prototype path.

## 4. Recurring Process Signals

Strong positive signals in comments:
1. "Discussed on WG/CG call" with minutes links.
2. "Create an explainer" or "initiate prototyping" resolution language.
3. Repo creation under `webmachinelearning/*` for focused iteration.
4. Explicit maintainer scoping notes (goals/non-goals, avoid premature formal spec).

## 5. Proposal Authoring Heuristics (Derived)

Based on successful trajectories, a proposal should:
1. Be scoped to a concrete gap, not a broad manifesto.
2. Include a first prototype shape and a narrow v1 boundary.
3. Show compatibility with existing work (WebNN/WebMCP/hybrid-ai), not displacement.
4. Be ready to transition to explainer repo if maintainers request.
5. Include implementability constraints early (privacy, portability, cross-platform behavior).

## 6. PM-KR-Specific Implications

For a PM-KR proposal to land well in this ecosystem:
1. Present PM-KR as a complement layer for model/knowledge portability and transparent reasoning.
2. Start with one incubatable artifact (e.g., procedural node exchange profile + reference converter).
3. Include an explainer-friendly prototype plan and interoperability story with existing WebNN graphs.
4. Explicitly separate high-confidence measured claims from long-term projections.
