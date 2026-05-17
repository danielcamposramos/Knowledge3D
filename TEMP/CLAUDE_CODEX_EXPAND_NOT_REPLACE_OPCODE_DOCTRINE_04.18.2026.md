# Doctrine: Expand-Not-Replace Opcode Registry

**Date**: 2026-04-18
**Author**: Claude (architecture)
**Status**: DOCTRINE — applies to all opcode additions, kernel extensions, registry edits
**Authority**: Daniel's ruling, 2026-04-18

---

## 1. Principle — Daniel's Rulings (verbatim)

Ruling #1 (expansion):
> "do not replace, expand — change what's needed to make it happen."

Ruling #2 (attention mechanism):
> "We need attention mechanism, but my guess is that this is logic (model weights) and must leverage ternary logic and contrastive learning."

These two rulings arrive together because the question that triggered them was whether `ATTENTION_FWD` (0x1A7) should be dropped in favor of a pure-composition approach. The answer: neither drop nor monolithic-replace. Keep the base entry, add the sovereign ternary-contrastive sibling. The registry grows; nothing is removed.

The registry is not merely a list of opcodes. It is a **public contract**. Every PTX kernel compiled against an opcode, every RPN program that pushes an opcode value, every test that verifies opcode behavior — all are callers of that contract. Removing or renumbering an entry breaks every caller silently if the opcode space wraps around (a different opcode now occupies that number) or loudly if the kernel crashes on an unrecognized opcode. Either outcome is worse than the disk space cost of keeping the old entry.

K3D is simultaneously a new compute architecture and a new knowledge-representation paradigm. It will be extended, not replaced. The same principle that governs the architecture governs its registry.

---

## 2. The Registry Is Append-Only

**Append-only** means:
- New opcodes are always added after the highest assigned number in a range.
- Existing opcode numbers and mnemonics are never changed, even if the mnemonic was poorly chosen.
- Existing opcode numbers are never reused, even if the original opcode is superseded.
- Deprecation is permitted (see §5) but does not free the number.

The registry document is `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md`. The registry Python enum is `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py`. Both are append-only. A PR that removes a line from either is a registry violation regardless of the stated rationale.

**Why backwards-compatibility preserves momentum:**
K3D has accumulated ~18 months of RPN programs, kernel wiring, test traces, and galaxy programs that reference opcode values by number. A renumbering that "cleans up" the registry would invalidate every compiled PTX artifact, every persisted star program, every observability trace. The clean-up cost is not the editor time — it is the silent invalidation of every downstream artifact, most of which cannot be enumerated at edit time. The registry's append-only constraint is the sovereign equivalent of semantic versioning: callers have a right to expect that opcode 0x1A7 means what it said when they were written.

---

## 3. Expansion Pattern — Variant Suffixes

When an existing opcode is insufficient for a new use case, the correct action is to add a sibling opcode with a variant suffix. The original opcode keeps its number, mnemonic, and semantics. The new variant is assigned the next free number in the range.

**Canonical suffix vocabulary:**

| Suffix | Meaning | When to use |
|---|---|---|
| `_BASE` | Float32 training-lane reference implementation | When the original opcode is being paired with a sovereign variant; clarify the original's role |
| `_TERNARY` | Ternary-weighted (balanced ternary {-1, 0, +1}, 2-bit packed) | Sovereign production variant using XNOR+popcount logic |
| `_PACKED` | Bit-packed representation for high-throughput pass | When 16-bit or 8-bit packing is needed for memory bandwidth |
| `_QUANT` | TQUANT-normalized input/output (ternary quantization applied at boundary) | When an existing float op needs to interoperate with ternary pipelines |
| `_ASYNC` | Stream-ordered asynchronous execution variant | When the base op blocks and async is needed for the nine-chain swarm overlap |

A new variant does not replace the base. It extends the family. The base remains the canonical validation reference and the training-lane debug path.

---

## 4. Attention Case Study — Ruling #2 Applied

### 4.1 What Attention IS in K3D

Attention is a **composition** (an RPN program), not a monolithic opcode. This follows the "Programs before opcodes" principle established in `RPN_DOMAIN_OPCODE_REGISTRY.md`: if a behavior can be expressed as a sequence of existing opcodes, it is a program — not a new opcode. New opcodes are needed only when a behavior cannot be expressed in composition.

The composition for sovereign K3D attention draws on three existing families:

**Ternary family (0x100-0x10F):**
- `TERNARY_AND` (0x100), `TERNARY_OR` (0x101), `TERNARY_XOR` (0x103): Q·K as XNOR+popcount — weights in {-1, 0, +1} encoded as 2-bit packed uint32 (16 trits per word, BitNet b1.58 regime). At 1000× Python speed for the logic stage.
- `TQUANT` (0x106): quantize incoming float32 attention weights to balanced ternary at entry. Float32 weights are a training-lane artifact; inference operates on ternary from the start.

**Yard family (0x170-0x177):**
- `YARD_SELECT` (0x170), `YARD_PUSH_BANK` (0x171), `YARD_POP_BANK` (0x172), `YARD_PEEK_ADDR` (0x173): Q, K, V tiles live in `yards[instance][bank][slot]`. Bank 0 = Q, Bank 1 = K, Bank 2 = V (by convention; the RPN program specifies bank addresses explicitly). Mixing uses existing yard transfer ops — no external matmul library.

**Queue family (0x178-0x17A):**
- `QUEUE_PUSH` (0x178), `QUEUE_POP` (0x179): when the nine-chain swarm dispatches attention sub-problems across cores, inter-core Q/K/V tile passing uses the queue substrate — same as all other inter-core coordination.

**Contrastive margin scoring:**
The scoring mechanism is pair-ranking, not softmax. Pull matching meanings close, push non-matching meanings away. This is Christoph Dorn's "semantic gravity cohered by meaning" realized as an attention kernel. Contrastive margin avoids `exp()` — no softmax footgun, no need to import a math lib for a single transcendental. The contrastive computation is expressible as a chain of ternary comparisons and conditional ops — existing family.

### 4.2 Where New Opcodes Are Needed

The composition above handles the core computation. The following cannot be expressed in composition without significant program complexity that would exceed yard depth limits on long sequences. New opcodes are justified:

| Opcode | Mnemonic | Rationale for new opcode (not composition) |
|---|---|---|
| `0x1A7` | `ATTENTION_FWD_BASE` | Float32 single-head attention: Q·K^T / √d → softmax → V. Training-lane validation path. NOT the sovereign runtime. Reserved name clarifies the original `ATTENTION_FWD` assignment. |
| `0x1A8` | `ATTENTION_FWD_TERNARY` | Ternary Q·K via XNOR+popcount, contrastive margin scoring (pair-rank, no softmax), V-mix via yard bank transfer. The sovereign production sibling. Composition of ternary + yard + queue ops made explicit as a named opcode for observability and RPN program economy. |
| `0x1A9` | `ATTENTION_CONTRASTIVE_MARGIN` | Pop two embedding vectors (tile size, from two yard banks), compute contrastive margin score: `score = dot(a, b) - margin_threshold`. Push scalar score. Used in Galaxy neighborhood scoring and Memory Palace relevance gating. |
| `0x1AA` — `0x1AF` | Reserved | Attention family reserved range. Do NOT assign without cross-referencing `ternary_contrastive_attention_design.md` which is being finalized in a parallel architecture lane. |

**Note on the pre-existing 0x1A7 assignment:** The Bulk-Lib Purge Hard Acceptance spec (written earlier on 2026-04-18) originally assigned `ATTENTION_FWD` (single-head float32) to 0x1A7. Per Daniel's ruling, this is now `ATTENTION_FWD_BASE` at 0x1A7 (same number, renamed for clarity), with `ATTENTION_FWD_TERNARY` added at 0x1A8. The number 0x1A7 is retained; it is not dropped. This is the expand-not-replace pattern applied.

**The full attention family:** Cross-reference `TEMP/ternary_contrastive_attention_design.md` (being finalized by a parallel architecture lane as of 2026-04-18) for the complete kernel specification, XNOR+popcount implementation guide, and contrastive margin threshold setting. The opcode range 0x1A7-0x1AF is reserved exclusively for the attention family. No other domain may be assigned in this range without architecture sign-off.

### 4.3 Demonstrating Composition Over Replacement

The example that would violate this doctrine: proposing a new `ATTENTION_FWD_V2` that supersedes `ATTENTION_FWD` and asking Codex to delete the old one. The domain-compliant action:

1. Keep `ATTENTION_FWD_BASE` (0x1A7) — now the float32 training-lane reference.
2. Add `ATTENTION_FWD_TERNARY` (0x1A8) — the sovereign runtime.
3. Add `ATTENTION_CONTRASTIVE_MARGIN` (0x1A9) — the scoring primitive.
4. Register all three in `RPN_DOMAIN_OPCODE_REGISTRY.md`. Never delete 0x1A7.
5. Training pipelines continue to call 0x1A7 for validation. Inference uses the composition of 0x1A8 + 0x1A9 + ternary + yard ops.

Old callers keep working. New capabilities become available. The registry grows by 2 entries. No renumbering, no disruption.

---

## 5. Registry Edit Protocol

**When Codex adds opcodes to `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md`:**

1. **Registry first, implementation second.** The opcode is registered (name, number, one-line semantics) before any kernel code is written. This is the "Programs before opcodes" inversion applied at the planning level: you declare the contract before you implement it.

2. **Append to the correct range.** Find the relevant range section in the registry (`0x1A0-0x1BF` for the bulk-lib purge additions, `0x170-0x17F` for yard ops, etc.). Add the new entry after the highest assigned number in that range.

3. **Do not fill reserved slots out of order.** If `0x1B0-0x1B5` are marked reserved, do not assign `0x1B3` for a new opcode while leaving `0x1B0-0x1B2` unassigned. Either take the next sequential number after the last assigned, or leave the reserved block intact. Reserved blocks are future-use contracts, not free space.

4. **Update `rpn_opcodes.py` in the same commit** as the registry edit. The Python enum and the markdown document must be in sync. A PR that edits one without the other fails Gate 1 below.

5. **Deprecation (if ever necessary):** Add a `DEPRECATED:` prefix to the mnemonic description in the registry. Do NOT remove the line. Do NOT reuse the number. Add a comment pointing to the successor opcode. Deprecation is unusual — prefer keeping an opcode active over deprecating it, since deprecated entries still consume a number. The reason to deprecate rather than keep-and-ignore is observability: an opcode dispatcher that encounters a deprecated opcode can emit a warning, whereas a kept-active opcode silently passes.

6. **The Old_Attempts exception:** Moving a whole file to `Old_Attempts/` is NOT the same as removing an opcode. An implementation can be archived; its opcode-level contract persists. If the implementation moves to Old_Attempts and no replacement exists yet, the opcode entry in the registry is annotated: `UNIMPLEMENTED: implementation archived at Old_Attempts/...`. The number remains assigned.

---

## 6. Grep Gates — Registry Integrity

These patterns flag replacement attempts, renumbering, and deletion violations.

```bash
# Gate 1: Registry and Python enum are in sync for the attention range
grep -nE "0x1A[0-9A-F]" docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md | wc -l
grep -nE "0x1A[0-9A-F]" knowledge3d/cranium/ptx_runtime/rpn_opcodes.py | wc -l
# → counts must be equal. Mismatch = one was edited without the other.

# Gate 2: No opcode number deleted from the registry (check git diff)
git diff HEAD docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md | grep "^-.*0x1"
# → 0 lines starting with "-" that contain an opcode number.
# Any such line means an opcode was removed — reject the PR.

# Gate 3: ATTENTION_FWD_BASE still assigned at 0x1A7
grep -n "0x1A7.*ATTENTION_FWD_BASE\|ATTENTION_FWD_BASE.*0x1A7" \
    docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md
# → ≥1 hit. If zero, the base was deleted or renumbered — violation.

# Gate 4: ATTENTION_FWD_TERNARY assigned at 0x1A8
grep -n "0x1A8.*ATTENTION_FWD_TERNARY\|ATTENTION_FWD_TERNARY.*0x1A8" \
    docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md
# → ≥1 hit after this doctrine lands.

# Gate 5: No opcode renumbering (existing opcode at new number)
# Manual check: run git blame on RPN_DOMAIN_OPCODE_REGISTRY.md for any line
# where an opcode number changed relative to an earlier commit. Zero tolerance.
git log --oneline -5 docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md
# Review the diff of each commit. Lines that change opcode numbers (not descriptions) are violations.

# Gate 6: Reserved ranges not violated
grep -nE "0x1A[A-F]|0x1B[0-9A-F]" docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md
# Review: any assignment in 0x1AA-0x1AF (attention family reserved) must match
# the ternary_contrastive_attention_design.md spec. Any assignment in 0x1B0-0x1B5
# (reserved general) must be in a new approved spec.
```

---

## 7. Codex Checklist

1. **Before any registry edit:** Read the "Programs before opcodes" principle in `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md`. Ask: can this behavior be expressed as a composition of existing opcodes? If yes, write a program, not a new opcode.

2. **Registry entry first:** Add the opcode to `RPN_DOMAIN_OPCODE_REGISTRY.md` (number, mnemonic, one-line semantics) before writing any kernel code that uses it. The PR must include the registry edit.

3. **Python enum updated in same commit:** Update `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py` to include the new opcode constant. Never let registry and enum drift.

4. **Attention range cross-reference:** Any opcode in 0x1A7-0x1AF must be explicitly authorized by `TEMP/ternary_contrastive_attention_design.md`. Codex must read that spec before touching this range.

5. **No deletions:** Run Gate 2 (git diff check) before finalizing any registry PR. If any opcode line was removed, restore it and find an alternative (append, not replace).

6. **Old_Attempts never frees a number:** If an implementation is moved to Old_Attempts, annotate the registry entry as `UNIMPLEMENTED` but do not delete the entry or reuse the number.

7. **Report evidence:** Every Gate above must be run and reported with output in the PR. Not "gates passed" — the actual grep output or diff excerpt.

---

## 8. Relationship to Other 2026-04-18 Doctrines

This doctrine governs the **content** of nodes in the symlink chain. The Hyper-Modular Symlink Doctrine (`CLAUDE_CODEX_HYPER_MODULAR_SYMLINK_DOCTRINE_04.18.2026.md`) governs the **sequencing** of when those nodes are built. Together:

- Symlink Doctrine: "build Phase N-1 before Phase N."
- This Doctrine: "when building a node, append-only; never disrupt what callers already resolve against."

Both doctrines share a root: K3D is a living system, not a series of rewrites. Every addition must be composable with what came before.
