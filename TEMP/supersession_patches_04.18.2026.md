# Supersession Patches — 2026-04-18 Doctrine Corrections

**Date**: 2026-04-18
**Author**: Claude (architecture)
**Purpose**: Patch document. Existing TEMP specs are not directly edited (they are referenced from the master handoff). Instead, this document records the exact corrected content per section. Codex reads this before acting on any patched spec — the correction here takes precedence.

---

## CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md

### Section 3 — New Opcode Inventory, row for 0x1A7

**Superseded:**
```
| `0x1A7` | `ATTENTION_FWD` | Single-head attention: pop Q, K, V from three banks; push output to active bank; scale by 1/√d | P0 | `SentenceTransformer` attention layer |
```

**Corrected:**
```
| `0x1A7` | `ATTENTION_FWD_BASE` | Float32 single-head attention: pop Q, K, V from three banks; push output; scale 1/√d. Training-lane validation path only — NOT sovereign runtime. | P0 (training only) | `SentenceTransformer` attention layer (training debug) |
| `0x1A8` | `ATTENTION_FWD_TERNARY` | Sovereign attention: ternary Q·K via XNOR+popcount (2-bit packed weights), contrastive margin scoring (no softmax), V-mix via YARD_TRANSFER. Composition of TERNARY_* (0x100-0x10F) + yard ops (0x170-0x177). | P0 (sovereign runtime) | Direct sovereign replacement; see `TEMP/ternary_contrastive_attention_design.md` |
| `0x1A9` | `ATTENTION_CONTRASTIVE_MARGIN` | Pop two embedding tile vectors from two yard banks; compute contrastive margin: `dot(a,b) - threshold`; push scalar score. Used in Galaxy neighborhood scoring and Memory Palace relevance gating. | P0 | Contrastive pair-ranking (no `exp()`, no softmax) |
| `0x1AA-0x1AF` | RESERVED — attention family | Cross-reference `TEMP/ternary_contrastive_attention_design.md` before any assignment in this range. | — | — |
```

**Reason:** Daniel's ruling 2026-04-18: "We need attention mechanism, but my guess is that this is logic (model weights) and must leverage ternary logic and contrastive learning." Expand-not-replace doctrine: 0x1A7 is kept as `ATTENTION_FWD_BASE` (training validation); ternary sovereign variant added at 0x1A8. See `feedback_attention_is_ternary_plus_contrastive.md`, `feedback_expand_not_replace_opcodes.md`.

---

### Section 5 — Phase 5: Phase B Native Embedding, entry gate paragraph

**Superseded:**
```
**Entry gate**: Phase 1 exit gates pass (Transfer Yard needed for `rpn_meaning_project.cu`).
```
(The implied reading in earlier discussion was that a LIFO interim could unblock Phase 5 before Phase 1 lands.)

**Corrected:**
```
**Entry gate**: Phase 1 exit gates ALL pass (Transfer Yard real yard kernel on Tiers 1, 2, and 3 confirmed via §9.1-§9.3 acceptance gates). Phase 5 does NOT begin on a LIFO interim. `rpn_meaning_project.cu` uses YARD_SELECT (0x170), YARD_PUSH_BANK (0x171), and YARD_FOLD_SUM to accumulate per-family contributions; these opcodes require the real yard kernel. A LIFO interim would require rewriting the kernel — or producing silently wrong embeddings that pass tests and corrupt the Galaxy. Phase 5 waits for Phase 1.
```

**Reason:** Daniel's ruling 2026-04-18: "No stubs, sequential and constructive." Hyper-modular symlink doctrine: Phase N cannot begin on a stub of Phase N-1. See `feedback_hyper_modular_symlink_architecture.md`.

---

### Section 5 — Phase 5: must-not-do item, add:

**Superseded:** (item not present)

**Corrected (add to Must-NOT-Do list in §8):**
```
- ❌ Do NOT begin Phase 5 (native embedding) on a Tier 1 LIFO interim kernel. The LIFO path does not expose YARD_SELECT / YARD_PUSH_BANK / YARD_FOLD_SUM. Writing rpn_meaning_project.cu against LIFO produces a stub that will require a full kernel rewrite once Phase 1 lands — or silently corrupts embeddings. Phase 5 waits for Phase 1.
```

**Reason:** Same as above. Daniel's explicit rejection of the LIFO-interim proposal on 2026-04-18.

---

### Section 3 — Opcode registry protocol (add note at top of §3)

**Superseded:** No explicit expand-not-replace protocol stated in §3.

**Corrected (add as first paragraph of §3):**
```
**Registry protocol — expand-not-replace.** All opcode assignments in this range are append-only. No existing opcode number is removed or renumbered, even if superseded by a variant. See `CLAUDE_CODEX_EXPAND_NOT_REPLACE_OPCODE_DOCTRINE_04.18.2026.md` for the full protocol. Codex: before adding any opcode, run Gate 2 (git diff check) to confirm no existing line was removed.
```

**Reason:** Daniel's ruling 2026-04-18: "do not replace, expand." See `feedback_expand_not_replace_opcodes.md`.

---

## CLAUDE_CODEX_OLD_ATTEMPTS_MIGRATION_04.18.2026.md

### Section 3.1 — enhanced_fallback.py migration action

**Superseded:**
```
| `knowledge3d/cranium/ptx_runtime/enhanced_fallback.py` | `Old_Attempts/2026-04-18_sovereignty_potemkins/cranium/ptx_runtime/enhanced_fallback.py` | Implements a `FallbackLevel` graduated fallback hierarchy. Fallbacks are explicitly forbidden by K3D sovereignty rules: "We fail and fix — this is the goal." No graduated fallback mechanism should exist in sovereign paths. |
```

(The existing §5 Shim Replacement Strategy also applies the shim template to `enhanced_fallback.py`, implying `python -c "import knowledge3d.cranium.ptx_runtime.enhanced_fallback"` should raise `NotImplementedError`.)

**Corrected:**
```
| `knowledge3d/cranium/ptx_runtime/enhanced_fallback.py` | `Old_Attempts/2026-04-18_sovereignty_potemkins/cranium/ptx_runtime/enhanced_fallback.py` | Implements a `FallbackLevel` graduated fallback hierarchy. Fallbacks are architecturally forbidden. This file has no active importers (verified by grep in §5 import audit). Direct archive — no shim. A shim at the original path is itself a fallback placeholder: it occupies the namespace, implies the path is "temporarily" broken, and invites future callers to depend on a broken import rather than fixing their call site. Direct archive means the original path does not exist after migration — `import knowledge3d.cranium.ptx_runtime.enhanced_fallback` raises `ModuleNotFoundError`, not `NotImplementedError`. This is the correct failure mode: the module does not exist. |
```

**Reason:** Daniel's ruling 2026-04-18 (subsidiary): "Archive `enhanced_fallback.py` directly (no shim, no fallbacks)." The file has no active importers — a shim is unnecessary and is itself a mild form of fallback. See `feedback_hyper_modular_symlink_architecture.md` §3 "Shape 4 — NotImplementedError shims on paths that should be deleted."

---

### Section 5 — Shim import audit for enhanced_fallback.py

**Superseded:**
```
# Find all importers of enhanced_fallback
grep -rn "enhanced_fallback\|FallbackLevel\|FALLBACK_BUDGET" \
    knowledge3d/ --include="*.py" \
    --exclude-dir=Old_Attempts
```
(Implied: if importers are found, place a shim.)

**Corrected:**
```
# Verify enhanced_fallback.py has no active importers (prerequisite for direct archive)
grep -rn "enhanced_fallback\|FallbackLevel\|FALLBACK_BUDGET" \
    knowledge3d/ --include="*.py" \
    --exclude-dir=Old_Attempts
# Expected: 0 hits. If hits exist, those call sites must be removed (not shimmed) before archive.
# The module is architecturally forbidden — callers should not exist, and if they do, they
# are themselves sovereignty violations that must be fixed, not papered over with a shim.
# After archive: no file exists at knowledge3d/cranium/ptx_runtime/enhanced_fallback.py.
# Do NOT create a shim file at this path.
```

**Reason:** Same as above.

---

### Section 7 — Gate 2 for enhanced_fallback.py

**Superseded:**
```
python -c "import knowledge3d.cranium.ptx_runtime.enhanced_fallback" 2>&1 | grep NotImplementedError
# → must find "NotImplementedError" in output
```

**Corrected:**
```
python -c "import knowledge3d.cranium.ptx_runtime.enhanced_fallback" 2>&1 | grep ModuleNotFoundError
# → must find "ModuleNotFoundError" in output
# No shim exists at this path. The module simply does not exist.
```

**Reason:** Direct archive means `ModuleNotFoundError`, not `NotImplementedError`. A shim producing `NotImplementedError` would be a fallback placeholder.

---

## CLAUDE_CODEX_INSTANTIABLE_CORE_ISOLATION_04.18.2026.md

### Section 1 — Add "Symlink Role" paragraph after the five-point isolation contract

**Superseded:** (paragraph not present)

**Corrected (add as §1.1 immediately after the five-point list):**
```
### 1.1 Symlink Role of This Spec

Per the Hyper-Modular Symlink Doctrine (`CLAUDE_CODEX_HYPER_MODULAR_SYMLINK_DOCTRINE_04.18.2026.md`), every spec must declare what it RESOLVES and what it PRODUCES in the symlink chain.

**This spec RESOLVES:**
- Phase 0 (Archive and Isolate): `Old_Attempts/` is clean; sovereignty violations are removed from the tree.
- Queue opcodes (0x178-0x17A): QUEUE_PUSH, QUEUE_POP, QUEUE_PEEK are registered in `RPN_DOMAIN_OPCODE_REGISTRY.md` and implemented in `rpn_execute_device.cuh`. These are dependencies of the isolation contract.
- `MicroSpecialistPool.query_sm_count()`: returns the real SM count (46 for RTX 3070); used to derive the 414-core ceiling.

**This spec PRODUCES (enables, is a dependency for):**
- Phase 1 (Transfer Yard): yard kernels reference `CoreRegistry` for the `MAX_INSTANCES → query_sm_count() * 9` migration. Phase 1 Tier 2 and Tier 3 kernels use the isolated core layout specified here.
- Phase 5 (Native Embedding): `rpn_meaning_project.cu` runs as a core instance. The core isolation contract is the runtime substrate it executes inside.
- Attention RPN program: the attention composition (ternary Q·K, contrastive margin) runs inside isolated cores. QUEUE_PUSH/POP is how attention sub-problems are distributed across the nine-chain swarm.
- TRM tick (`trm_step_fused.ptx`): the TRM game loop IS a core-dispatching program. The isolation contract IS the execution model for every TRM tick.

**Higher layers that depend on this spec** cannot be stubs. If core isolation is not real, attention is not real, TRM tick is not real, and the knowledgeverse daemon is not real. This is the load-bearing node closest to the hardware.
```

**Reason:** Daniel's ruling 2026-04-18: "No stubs, sequential and constructive — this architecture is hyper-modular (each part depends on the other — like 'a symlink thing')." See `feedback_hyper_modular_symlink_architecture.md`, `CLAUDE_CODEX_HYPER_MODULAR_SYMLINK_DOCTRINE_04.18.2026.md`.

---

## CLAUDE_CODEX_PHASE_B_NATIVE_EMBEDDING_04.18.2026.md

### Section 5 — FOV/POV Usage, add forward reference after the pipeline table

**Superseded:** Section 5 ends with the integration point paragraph about `matryoshka_prefix_dot.cu`. No forward reference to attention design.

**Corrected (add as final paragraph of §5):**
```
### 5.1 Forward Reference: Ternary-Contrastive Attention

The embeddings produced by Phase B (`rpn_meaning_project.ptx`) are the direct input to the attention mechanism at the Nine-Chain Swarm and Halting Gate stages. Per Daniel's ruling (2026-04-18): "We need attention mechanism, but my guess is that this is logic (model weights) and must leverage ternary logic and contrastive learning."

The full attention design is being specified in `TEMP/ternary_contrastive_attention_design.md` (parallel architecture lane, 2026-04-18). Key pre-linkage points for Phase B implementers:

1. **Ternary weight format**: Phase B embeddings at tier_2048 are float32 during initial generation. The attention layer applies TQUANT (0x106) to quantize Q, K, V tiles to balanced ternary {-1, 0, +1} before the XNOR+popcount dot product. Phase B does not need to output ternary — the quantization happens at attention entry.

2. **Contrastive margin scoring**: The Halting Gate uses `ATTENTION_CONTRASTIVE_MARGIN` (0x1A9) over tier_2048 embeddings to detect answer duplication. Phase B's determinism guarantee (P1) is a prerequisite: bit-identical embeddings across calls are required for the deduplication check to be stable.

3. **Matryoshka tier selection at attention time**: The tier used per pipeline stage (§5 table) feeds directly into the attention tier selection. The `meta_select_matryoshka_tier` RPN meta-rule writes a `tier_signal` that the attention program reads to choose between XNOR-64 (fast, coarse) and XNOR-2048 (slow, precise).

Phase B must be complete before the attention spec can produce a working kernel. The dependency is real: attention operates on Phase B embeddings. Phase B is the symlink target; attention is the symlink consumer. This is the expand-not-replace pattern at the pipeline level: Phase B expands what `matryoshka_prefix_dot.cu` operates on; the attention layer expands what Phase B embeddings are used for.
```

**Reason:** Daniel's ruling 2026-04-18 on attention as ternary + contrastive. The forward reference anchors Phase B in the dependency chain and gives Codex implementers the parameters they need to ensure Phase B output is attention-compatible. See `feedback_attention_is_ternary_plus_contrastive.md`, `TEMP/ternary_contrastive_attention_design.md`.

---

## CLAUDE_CODEX_TRANSFER_YARD_AND_EMBEDDING_SOVEREIGNTY_04.18.2026.md

### Section 4.2 — New opcodes table, add note on opcode strategy

**Superseded:** §4.2 lists yard opcodes 0x170-0x177 without an explicit expand-not-replace protocol statement.

**Corrected (add as preamble to the opcode table in §4.2):**
```
**Opcode strategy — expand-not-replace.** These opcodes are appended to the registry, never substituted for existing opcodes. See `CLAUDE_CODEX_EXPAND_NOT_REPLACE_OPCODE_DOCTRINE_04.18.2026.md`. If a yard opcode turns out to need a ternary-first variant (e.g., `YARD_SELECT_TERNARY` where bank_id is pre-quantized), add it at the next free number — do not modify 0x170.
```

**Reason:** Daniel's ruling 2026-04-18: "do not replace, expand." The yard opcode range must be governed by the same append-only rule as all other ranges.

---

### Section 5 — Phase ordering note

**Superseded:** §5.1 and §5.2 discuss specialists and ingestion phases without explicit reference to the hyper-modular sequencing doctrine.

**Corrected (add at the top of §5 before §5.1):**
```
**Phase sequencing — sequential-constructive.** All phases in this spec follow the Hyper-Modular Symlink Doctrine (`CLAUDE_CODEX_HYPER_MODULAR_SYMLINK_DOCTRINE_04.18.2026.md`). No phase begins on a stub of a prior phase. Specifically: Phase B (native embedding, §5.2) begins only after Phase A (Transfer Yard on all tiers, §3) acceptance gates pass. No LIFO interim kernel is used to unblock Phase B early. The yard substrate is the real primitive that Phase B depends on.
```

**Reason:** Daniel's ruling 2026-04-18: "No stubs, sequential and constructive." Applied specifically to the LIFO-interim proposal that was explicitly rejected.

---

### Section 10 — Codex handoff checklist, add cross-doctrine note

**Superseded:** §10 checklist does not reference the 2026-04-18 doctrine set.

**Corrected (add as item 0 at the top of §10, before existing item 1):**
```
0. Read the six 2026-04-18 doctrine pillars before starting any implementation task in this spec. They are the overriding contracts. If any instruction in this spec conflicts with a doctrine, the doctrine wins — file an architecture note and do not proceed until resolved.
   - `CLAUDE_CODEX_HYPER_MODULAR_SYMLINK_DOCTRINE_04.18.2026.md` (sequencing contract)
   - `CLAUDE_CODEX_EXPAND_NOT_REPLACE_OPCODE_DOCTRINE_04.18.2026.md` (registry contract)
   - `TEMP/ternary_contrastive_attention_design.md` (attention design — parallel lane)
   - `CLAUDE_CODEX_INSTANTIABLE_CORE_ISOLATION_04.18.2026.md` (core isolation contract)
   - `CLAUDE_CODEX_OLD_ATTEMPTS_MIGRATION_04.18.2026.md` (archive protocol)
   - `CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md` (sovereignty gates)
```

**Reason:** The doctrine set supersedes any prior instruction on the same subject. Making item 0 explicit ensures Codex does not start implementation under assumptions that the doctrine has already corrected.

---

### Add section: Cross-Doctrines (new terminal section)

**Superseded:** (section does not exist)

**Corrected (add as §13 at the end of the spec):**

```markdown
## 13. Cross-Doctrines — Six 2026-04-18 Pillars

All work descending from this master handoff is governed by the following six doctrine documents, all issued 2026-04-18. These are the architectural contracts Codex must honor. Any implementation choice that conflicts with a doctrine requires architecture sign-off before proceeding.

| Doctrine | File | What It Governs |
|---|---|---|
| Hyper-Modular Symlink | `CLAUDE_CODEX_HYPER_MODULAR_SYMLINK_DOCTRINE_04.18.2026.md` | Phase sequencing — no stubs, no interim paths, sequential-constructive |
| Expand-Not-Replace | `CLAUDE_CODEX_EXPAND_NOT_REPLACE_OPCODE_DOCTRINE_04.18.2026.md` | Opcode registry — append-only, never renumber, never delete |
| Ternary-Contrastive Attention | `TEMP/ternary_contrastive_attention_design.md` | Attention mechanism — ternary Q·K + contrastive margin, opcode family 0x1A7-0x1AF |
| Core Isolation | `CLAUDE_CODEX_INSTANTIABLE_CORE_ISOLATION_04.18.2026.md` | Core isolation contract — 414 cores, queue opcodes, no cross-core shared memory |
| Old_Attempts Migration | `CLAUDE_CODEX_OLD_ATTEMPTS_MIGRATION_04.18.2026.md` | Archive protocol — git mv, README_WHY_ARCHIVED, direct archive for no-importer files |
| Bulk-Lib Purge Hard Acceptance | `CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md` | Sovereignty gates — 10 CI grep gates, 8-phase migration, W3C-only external alignment |

**Non-negotiable conflicts:** If any instruction anywhere (CODEX.md, AGENTS.md, older TEMP specs, inline PR comments) tells Codex to: renumber an opcode, use a stub/interim kernel, add a fallback shim to a no-importer file, import numpy in a sovereign path, or begin a phase before its dependencies are real — that instruction is superseded by the relevant doctrine above. The doctrines are the floor; all other instructions operate above the floor.
```

**Reason:** Daniel's three rulings on 2026-04-18 collectively define a doctrine stack. Making the stack explicit and terminal in the master handoff ensures Codex does not encounter contradictions silently.
