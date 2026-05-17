# Galaxy Star Schema — `confidence_trit` Field Specification

**Date**: 2026-04-18
**Author**: Claude (architecture, connective-tissue lane)
**Implementer**: Codex
**Authority**: Daniel's Ruling 2, 2026-04-18
**Ruling verbatim**: "The contrastive-margin parameter `m` is loaded from each star's `confidence_trit` Galaxy field via `YARD_PEEK_ADDR` inside the scoring loop — not hardcoded. High-confidence stars get tight margins, uncertain stars get wide margins. Semantic richness over speed."
**Expand-not-replace**: `confidence_trit` is a NEW field appended to the star schema. No existing fields are renamed or removed.

---

## 1. Semantics

`confidence_trit` is a per-star metadata value in balanced ternary {-1, 0, +1}:

| Trit value | Meaning | Attention margin effect |
|-----------|---------|------------------------|
| `+1` | High confidence. The star's meaning is stable and consistent — low spread across chunk projections. | Tight margin. Small `m_effective`. The attention head ranks this star's neighbors strictly. Only very similar candidates pass. |
| `0` | Unknown / default. No projection data yet, or variance in the ambiguous middle band. | Default margin `m_base`. System-level safe default. |
| `-1` | Low confidence. The star's meaning is scattered or context-dependent — high spread across chunk projections. | Wide margin. Large `m_effective`. The attention head is permissive: more candidates pass to allow for disambiguation by context. |

The design principle: **semantic richness over speed.** A wide margin for uncertain stars costs more scoring work but prevents premature pruning of valid answer paths. The GPU absorbs the cost; the Galaxy gains accuracy.

---

## 2. Storage

- **Location**: Galaxy Universe, Region 2 (active working memory VRAM), within the star metadata word.
- **Format**: 2-bit packed field in the star's metadata slot. Encoding matches the canonical K3D 2-bit trit format: `+1 → 0b10`, `0 → 0b01`, `-1 → 0b00`. Code `0b11` is treated as `+1` (forward compatibility clamp, same convention as ternary opcodes).
- **Bank**: `STAR_META_BANK` — by convention, bank 8 (the register-store bank in the yard layout). The field is at a fixed byte offset `CONFIDENCE_TRIT_OFFSET` within the bank, defined as a kernel constant.
- **Addressable**: via `YARD_PEEK_ADDR` (0x173), `bank_id = STAR_META_BANK`, `slot_id = CONFIDENCE_TRIT_OFFSET`. The peek pushes the raw 2-bit value onto the active bank. The caller decodes trit from the 2-bit encoding using the existing tquant convention.

This is NOT a BitNet 1.6-bit weight field. It is a rule-mask / metadata trit — a deliberate 3-valued category, not a quantized float. The 2-bit storage slot is a container choice for alignment; the semantic is balanced ternary.

---

## 3. Derivation — Phase B Integration

`confidence_trit` is produced by `rpn_meaning_project.cu` (Phase B) as a byproduct of chunk folding (§3.5 of `CLAUDE_CODEX_PHASE_B_NATIVE_EMBEDDING_04.18.2026.md`).

**Algorithm:**

```
// After accumulating per-chunk vectors during projection:
chunk_magnitudes[k] = L2_NORM(chunk_vec_k[0:64])  // coarse-band magnitude per chunk
mean_mag = MEAN(chunk_magnitudes)
chunk_variance = MEAN((chunk_magnitudes[k] - mean_mag)^2 for all k)

// Threshold constants (kernel __constant__ memory, tunable):
LOW_VAR_THRESHOLD  = 0.05   // programs with stable meaning across chunks
HIGH_VAR_THRESHOLD = 0.30   // programs with scattered / context-dependent meaning

IF chunk_variance < LOW_VAR_THRESHOLD:
    confidence_trit = +1   // packed as 0b10
ELIF chunk_variance > HIGH_VAR_THRESHOLD:
    confidence_trit = -1   // packed as 0b00
ELSE:
    confidence_trit = 0    // packed as 0b01
```

**Single-chunk programs** (length ≤ 69 tokens, no folding): `chunk_variance = 0` → `confidence_trit = +1` always. This is correct: a single unambiguous program has maximum self-consistency.

**Storage step**: after computing `confidence_trit`, `rpn_meaning_project.cu` writes the packed 2-bit value to the star's metadata slot in Region 2. This write is part of the same kernel pass that writes `star.embeddings.tier_2048` — no additional kernel launch.

Phase B is **not complete** until `confidence_trit` is populated for all stars in the embedding table. The attention scoring loop depends on this field. A star with no `confidence_trit` written must default to `0b01` (trit=0, default margin) — not `0b00` (trit=-1, which would incorrectly inflate all margins).

---

## 4. Integration — Attention Scoring Loop

In `ATTENTION_FWD_TERNARY` (0x1A8), the margin `m` is no longer a static integer operand. The scoring loop uses `confidence_trit` to compute `m_effective` per candidate:

```rpn
; For each candidate star at index i:
; (inside the scoring loop, after loading K[i] ternary words)

0x173  YARD_PEEK_ADDR  bank=STAR_META_BANK slot=(meta_offset_for_star_i)
                       ; push confidence_trit (2-bit encoded) of candidate star i

; Decode trit to {-1, 0, +1}:
0x74   TQUANT          ; map 0b10→+1.0, 0b01→0.0, 0b00→-1.0 (via existing tquant convention)

; Compute m_effective:
; m_effective = m_base + (1 - trit) * m_delta
;   trit=+1 → m_effective = m_base + 0        (tight)
;   trit= 0 → m_effective = m_base + m_delta  (default)
;   trit=-1 → m_effective = m_base + 2*m_delta (wide)
0x06   NEG             ; negate trit → -(trit)
0x01   ADD_SCALAR 1.0  ; 1 - trit
0x03   MUL_SCALAR m_delta
0x01   ADD_SCALAR m_base  ; m_effective on stack

; m_effective is then consumed by the CONTRASTIVE_RANK_TOPK gating for this candidate
```

The static `margin_m` operand field in 0x1A8's binary layout is now `m_base`. `m_delta` is a kernel constant (default: 4). Codex: implement the per-candidate trit read as part of the 0x1A8 kernel body; do not require callers to write this sequence manually.

---

## 5. Sleep-Time Consolidation

During sleep-time consolidation (`sleeptime.py`, Region 5 — Sleep Engine), `confidence_trit` values can be updated based on accumulated query-success traces:

- **Positive signal**: a star that was frequently the correct answer in logged traces → increment toward `+1`.
- **Negative signal**: a star that was frequently retrieved but never confirmed as correct → decrement toward `-1`.
- **No signal**: a star that was never queried → trit stays at its Phase B derivation value.

The update is a delta rule on the trit field — not gradient descent, not weight training. It is a 3-valued reputation score maintained by the sleep engine. The sleep-time update reads from Region 5 (trace logs) and writes back to Region 2 (active star metadata).

Reference: sleep-time consolidation is specified in `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md` §5 (Region 5: Sleep Engine). The `confidence_trit` update is an extension of the existing star-reputation mechanism.

**Expand-not-replace**: the sleep-time update extends the existing consolidation loop. It does not replace or remove any existing sleep-time behavior.

---

## 6. Acceptance Gates

Codex runs these before declaring `confidence_trit` complete:

```bash
# Gate 1: Phase B emits confidence_trit for all processed stars
grep -n "confidence_trit" knowledge3d/cranium/kernels/rpn_meaning_project.cu
# → at least 1 write site (the STAR_META_BANK write after chunk folding)

# Gate 2: Attention kernel reads confidence_trit per candidate
grep -n "STAR_META_BANK\|confidence_trit\|CONFIDENCE_TRIT" \
    knowledge3d/cranium/kernels/rpn_attention.cu  # or wherever 0x1A8 is implemented
# → at least 1 read site inside the scoring loop

# Gate 3: Default is trit=0 (0b01) not trit=-1 (0b00)
# Review the uninitialized-star handling code — must use 0b01 not 0b00.

# Gate 4: VEC_NORM_L2_INT8 (0x1B0) follows 0x1A8 in all RPN programs
grep -rn "0x1A8" knowledge3d/ --include="*.py" --include="*.cu" --include="*.ptx" \
    --exclude-dir=Old_Attempts
# For each hit, verify the next opcode in the program is 0x1B0.
```
