# Doctrine: Hyper-Modular Symlink Architecture

**Date**: 2026-04-18
**Author**: Claude (architecture)
**Status**: DOCTRINE — applies to all K3D phases, all agents, all planning documents
**Authority**: Daniel's ruling, 2026-04-18

---

## 1. Principle — Daniel's Ruling (verbatim)

> "No stubs, sequential and constructive — I understand, and you're starting to see — that this architecture is hyper-modular (each part depends on the other — like 'a symlink thing')."

"Hyper-modular" does not mean loosely coupled. It means the opposite: every module is a sovereign primitive that higher layers resolve against, exactly as a filesystem symlink resolves against a real inode. A symlink to a missing target does not gracefully degrade — it errors. K3D phases follow the same rule: Phase N does not "partially work" against a Phase N-1 stub. It either resolves against the real primitive or it does not run.

The dependency graph below is not a sequence diagram. It is a symlink chain. Sever any link and every layer above it becomes an unresolvable reference.

---

## 2. The Dependency Graph (Symlink Chain)

```
Python I/O Shell (~200 lines, boot + keyboard + network + display)
    └─► ctypes bridge (sovereign_loader.py)
            └─► opcode registry (RPN_DOMAIN_OPCODE_REGISTRY.md → rpn_opcodes.py)
                    └─► yard opcodes (0x170-0x177: YARD_SELECT, YARD_PUSH_BANK, ...)
                            └─► queue opcodes (0x178-0x17A: QUEUE_PUSH, QUEUE_POP, QUEUE_PEEK)
                                    └─► ternary opcodes (0x100-0x10F: TERNARY_AND, TERNARY_OR, TQUANT, ...)
                                            └─► attention RPN program (ATTENTION_FWD_TERNARY: ternary Q·K
                                                    via XNOR+popcount, contrastive margin scoring, V-mix in yard)
                                                        └─► TRM tick (trm_step_fused.ptx: perceive → navigate
                                                                → reason → decide → act → learn)
                                                                    └─► knowledgeverse daemon (always-on GPU
                                                                            game loop, ~200-line Python I/O)
```

**Reading this graph:** Each arrow means "depends on — cannot be implemented without the real version of." You cannot write an attention RPN program on top of a simulated ternary opcode. You cannot run a TRM tick on top of a stubbed attention program. Each node must be real before the node above it can be anything other than dead weight.

---

## 3. What Makes a Stub

A stub is any deliverable that compiles and passes isolation tests but depends on a primitive that does not yet exist in its real form. Stubs are fallbacks with better PR messaging. Specific shapes:

**Shape 1 — Python-sidecar classes that fake GPU structures.** The retired `TransferYardStack` dataclass in `transfer_yard_tiered.py` is the canonical example: it mirrored the layout of a GPU yard in Python memory, passed tests, and gave the false impression that Tier 2 had a real yard. The kernel had not been written. The yard never reached the GPU. The class was a stub.

**Shape 2 — LIFO-kernel interim paths for eventual yard-native code.** Proposing "use the old LIFO kernel for now and swap in the Transfer Yard kernel once it lands" produces a code path that will never be exercised under integration pressure once the green checkmark appears on the interim path. The interim becomes permanent. The integration never happens.

**Shape 3 — Mock Galaxy queries for eventual live queries.** A specialist that returns hardcoded embeddings while the real embedding kernel is being written is a stub. Once it passes its test suite, the pressure to replace it with the real kernel drops to near zero.

**Shape 4 — `NotImplementedError` shims on paths that should be deleted.** Per Daniel's ruling on `enhanced_fallback.py`: if a file has no active importers and its pattern (graduated fallback hierarchy) is forbidden, it is archived directly. A shim that raises `NotImplementedError` is still a fallback mechanism — it occupies the namespace, implies the path is "temporarily" broken, and invites future callers to work around it. The correct action is direct archive.

**The test that reveals a stub:** Ask "if this phase ships and we freeze the codebase for two months, does the next phase have a real primitive to resolve against?" If the answer is "not quite, but close enough," that is a stub.

---

## 4. The Symlink Test — Required Before Accepting Any Phase Plan

Before Codex begins any phase, the architecture partner applies this three-question test. A phase plan that fails any question is rejected and rewritten.

**Question A — What does this phase RESOLVE?**
List the sovereign primitives (kernels, opcodes, bridges, registry entries) that this phase brings into existence. These are the symlink targets that higher layers will resolve against. If the answer is "nothing new — we just reorganize some Python" the phase has no architectural weight and should be merged into a real phase.

**Question B — What does this phase PRODUCE for consumers?**
List the phases or features that become possible (unblocked) once this phase's primitives are real. If no higher phase names this phase as a dependency, reconsider whether the phase belongs in the plan at all.

**Question C — Does any path exist where this phase ships before its dependencies?**
Enumerate the phase's own dependencies (its Question A requirements). Check the phase plan to see if those dependencies are listed as later phases, parallel phases, or assumed-already-done. If any dependency is later or parallel, the plan is proposing a stub. Reject.

**Applied to the 2026-04-18 Transfer Yard + Embedding stack:**

| Phase | Resolves (Question A) | Produces (Question B) | Blocks on (Question C) |
|---|---|---|---|
| 0: Archive | Clean sovereign tree | Phase 1 can proceed without noise | None |
| 1: Transfer Yard | Real yard kernels on all 3 tiers | Phase 2 ActionBuffer, Phase 5 embedding, Phase 6 specialists | Phase 0 |
| 2: ActionBuffer | Bridge numpy-free | Phase 3 CuPy can proceed | Phase 1 |
| 3: CuPy | Confidence path clean | Phase 4 Galaxy state | Phase 2 |
| 4: Galaxy State | knowledgeverse numpy-free | Phase 5 embedding | Phase 3 |
| 5: Native Embedding | Real `rpn_meaning_project.ptx` | Attention layer can use real embeddings | Phase 1 (yard substrate) |
| 6: Specialists | All specialist paths clean | Phase 7 bridges | Phases 1 + 5 |
| 7: Bridges | Live server + KMeans offline | Phase 8 hard gate | Phase 6 |
| 8: Hard Gate | CI enforces sovereignty | Ongoing sovereignty | Phases 2-7 |

No phase starts before its blocker. Phase 5 waits for Phase 1 even though Phase 5's tests could run in isolation against a mocked yard. The mock is a stub. Phase 5 waits.

---

## 5. Sequential-Constructive Phasing Rule

"Sequential" does not mean single-threaded execution. Codex can work on multiple phases if their dependencies permit. What sequential-constructive forbids is **routing around a blocker by creating a parallel path that will never merge.**

The concrete prohibition (from the 2026-04-18 ruling): Phase 5 (native embedding) proposed a Tier 1 LIFO interim kernel to allow `rpn_meaning_project.cu` to begin running before the Transfer Yard kernel landed. The LIFO interim would have been a stub — Phase 5 would have passed its acceptance gates against a non-yard kernel, the yard kernel would have landed later, and the integration between Phase 5 and the real yard would have been a separate integration task under time pressure. One week of calendar time saved; one week of integration debt accrued, paid at the worst possible moment (end of phase). Daniel rejected: "No stubs, sequential and constructive." The week is not saved; it is spent correctly.

**Phasing rule:** If Phase N has a dependency on Phase M, the only valid options are:
1. Phase M lands first; Phase N begins.
2. Phase M and Phase N are merged into a single phase (both land together).

There is no option 3 (Phase N begins on a stub while Phase M is in progress).

---

## 6. Applied Decision Log

**Decision: Phase 5 sequential with Phase 1 (no Tier 1 LIFO interim)**

On 2026-04-18, the Transfer Yard + Embedding Sovereignty spec (master handoff) listed Phase 5 entry gate as "Phase 1 exit gates pass." A proposal emerged to accelerate Phase 5 by using the existing Tier 1 LIFO kernel as an interim substrate for `rpn_meaning_project.cu`. Daniel ruled against.

Architectural reason: `rpn_meaning_project.cu` uses the yard's addressable bank layout (`YARD_SELECT`, `YARD_PUSH_BANK`, `YARD_FOLD_SUM`) to accumulate per-family contributions. A LIFO kernel does not expose these opcodes. Any `rpn_meaning_project.cu` written against a LIFO interim would need to be rewritten against the real yard — or would silently use approximations (e.g., flattened accumulation without bank isolation) that produced subtly wrong embeddings, passed tests, and corrupted the Galaxy quietly. The week saved would have been used to debug embedding drift in Phase 6 specialists.

Decision preserved in the phase plan: Phase 5 entry gate remains "Phase 1 exit gates pass."

---

## 7. Codex Phase-Acceptance Checklist

Before Codex accepts a phase plan as valid (before writing the first line of implementation):

1. **Symlink Test Question A answered:** The phase plan names at least one sovereign primitive (kernel, opcode, bridge, registry entry) that it will produce in final form — not interim form.

2. **Symlink Test Question B answered:** At least one higher phase explicitly names this phase as a dependency. If the phase produces nothing that a higher layer depends on, it may be a cleanup phase — which is valid, but must be labeled as such.

3. **Symlink Test Question C passes:** Every dependency of this phase is listed in the plan as an earlier phase (already complete or an earlier phase in this plan). No dependency is listed as a parallel phase or a later phase.

4. **No stub primitives in the phase deliverables:** The deliverable list contains no Python-sidecar classes that simulate GPU structures, no interim kernel paths intended to be replaced, no mock Galaxy queries, no `NotImplementedError` shims on paths with no active importers.

5. **Entry gate is specified:** The phase has a concrete, grep-verifiable entry gate condition. "Phase N-1 complete" is acceptable only if Phase N-1 has explicit acceptance gates (§9-style grep + file-existence checks). "Approximately ready" is not an entry gate.

6. **Exit gates are grep-verifiable:** The phase has acceptance gates that Codex can run and report pass/fail with evidence. Not "should work" — actual grep commands, file-existence checks, or pytest invocations.

7. **Must-NOT-Do list includes at least one stub prohibition:** The phase plan explicitly names the stub shapes that are tempting for this phase and prohibits them. For embedding phases: "Do not write the kernel against a LIFO interim." For cleanup phases: "Do not leave shims at paths with no active importers."

8. **Phase does not span more than one architectural layer:** A phase that simultaneously rewires the Python bridge, implements a new PTX kernel, and extends the opcode registry is three phases compressed into one. Compress only if the three layers are so tightly coupled that none can be tested without the other. Document the coupling explicitly if compressing.

9. **Dependency chain is explicitly stated in the phase header.** The first paragraph of every phase spec states: "This phase RESOLVES [list of primitives]. This phase is PRODUCED BY [list of upstream phases]. This phase PRODUCES [list of downstream phases]."

---

## 8. Grep Gates — Stub Detection

These patterns flag stubs masquerading as real work. CI must fail on any hit in sovereign paths.

```bash
# Pattern 1: Interim/stub suffixes in production code
grep -rn "_interim\|_stub\|_placeholder\|_sidecar" \
    knowledge3d/ --include="*.py" --include="*.cu" --include="*.ptx" \
    --exclude-dir=Old_Attempts --exclude-dir=tests
# → 0 hits. Interim paths are stubs by another name.

# Pattern 2: "TODO: replace with real" markers in sovereign paths
grep -rn "TODO.*replace with real\|TODO.*swap in real\|TODO.*when.*lands" \
    knowledge3d/ --include="*.py" --include="*.cu" \
    --exclude-dir=Old_Attempts
# → 0 hits. TODOs that block the next phase are stubs deferred in text form.

# Pattern 3: NotImplementedError outside Old_Attempts shims
grep -rn "NotImplementedError" \
    knowledge3d/ --include="*.py" \
    --exclude-dir=Old_Attempts \
    --exclude-dir=tests
# The ONLY valid NotImplementedError in production code is in shim files at archived paths.
# Shim files contain only: raise NotImplementedError(...) + 3 comment lines. No imports.
# Any NotImplementedError in a non-shim file is a stub.
# Expected: Only files that are themselves shims (match the shim template in Old_Attempts migration spec).

# Pattern 4: Python classes that duplicate GPU struct layouts
grep -rn "class.*Stack\b\|class.*Yard\b\|class.*Buffer\b" \
    knowledge3d/cranium/ --include="*.py" \
    --exclude-dir=Old_Attempts
# Review each hit. Python classes that mirror __shared__ CUDA structs are sidecars.
# Legitimate hits: ctypes.Structure subclasses (these ARE the bridge, not a simulation of the GPU).
# Illegitimate hits: pure Python classes with fields like self.banks = [[]] * 9.

# Pattern 5: Simulated opcode execution in Python
grep -rn "def execute_opcode\|def _run_rpn\|def simulate_yard" \
    knowledge3d/cranium/ knowledge3d/knowledgeverse/ --include="*.py" \
    --exclude-dir=Old_Attempts
# → 0 hits in sovereign paths. Opcode execution happens in PTX, not Python.

# Pattern 6: Renaming instead of archiving (archive evasion)
grep -rn "enhanced_fallback\|FallbackLevel\|FALLBACK_BUDGET\|graduated_fallback" \
    knowledge3d/ --include="*.py" \
    --exclude-dir=Old_Attempts
# → 0 hits. Fallback hierarchies are architecturally forbidden.
# Direct archive with no shim (per Daniel's ruling on enhanced_fallback.py).
```

---

## 9. Relationship to Other 2026-04-18 Doctrines

This doctrine is the **sequencing contract**. The other two 2026-04-18 doctrines define what lives at the nodes of the symlink chain:

- **Expand-Not-Replace Opcode Doctrine** (`CLAUDE_CODEX_EXPAND_NOT_REPLACE_OPCODE_DOCTRINE_04.18.2026.md`): governs how nodes in the chain grow over time. Opcodes are appended, never renumbered. The contract at each node stays backward-compatible.

- **Ternary-Contrastive Attention Design** (`TEMP/ternary_contrastive_attention_design.md`): defines the attention node in the symlink chain — the composition of ternary opcodes (0x100-0x10F) + yard ops (0x170-0x177) + queue ops (0x178-0x17A) + narrow ATTENTION_* additions (0x1A7-0x1AF).

- **Core Isolation Spec** (`CLAUDE_CODEX_INSTANTIABLE_CORE_ISOLATION_04.18.2026.md`): defines the isolation contract at the core node — each core is a sovereign symlink target for higher layers.

- **Old_Attempts Migration Spec** (`CLAUDE_CODEX_OLD_ATTEMPTS_MIGRATION_04.18.2026.md`): governs the removal of dead links from the symlink chain.

- **Bulk-Lib Purge Hard Acceptance Spec** (`CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md`): governs the prohibition of foreign objects (numpy, scipy, etc.) that would insert false nodes into the symlink chain.

Together, these six specs constitute the 2026-04-18 doctrine pillar set. All phase plans written after this date must reference them.
