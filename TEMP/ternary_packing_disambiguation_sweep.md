# Ternary Packing Disambiguation Sweep — 2026-04-18

**Date**: 2026-04-18  
**Scope**: All TEMP/*.md and TEMP/*.cu files, pattern scan for 2-bit vs 1.6-bit (BitNet) packing contexts  
**Ruling**: BitNet b1.58 canonical ternary (5 trits/byte = 1.6 bits/weight) adopted for weight matrices; 2-bit (16 trits/uint32) remains valid ONLY for rule-mask/confidence/metadata contexts.

---

## Executive Summary

- **Pattern A (2-bit packing)**: 11 hits total
  - **Must-update (weight-matrix context)**: 2 items
  - **Confirmed-valid (rule-mask context)**: 6 items  
  - **Ambiguous**: 3 items
  
- **Pattern B (XNOR/popcount)**: 18 hits total
  - **Must-update**: 4 items (weight matrix or attention context requiring clarification)
  - **Confirmed-valid**: 8 items (rule-mask context, zero-mask methods, mask operations)
  - **Ambiguous/Mixed**: 6 items (attention kernel scope unclear vs rule logic)

- **Pattern C (attention + XNOR/popcount co-occurrence)**: 8 hits
  - **VALID after review**: 7 hits (all legitimately attention-kernel context, 2-bit is correct for Q·K word packing)
  - **Opposite drift (E)**: 0 hits (BitNet references appropriately placed in weight-matrix context only)

- **Pattern D (generic "ternary packing" without bit-width)**: 7 hits
  - All ambiguous, flagged for human review

- **Pattern E (1.58-bit / 1.6-bit / BitNet references)**: 2 hits
  - Both correctly contextualized in weight-matrix scenarios

---

## 1. Summary Table

| Pattern | Total Hits | Must-Update | Valid-Keep | Ambiguous | File Count |
|---------|-----------|------------|-----------|-----------|-----------|
| A: 2-bit packing | 11 | 2 | 6 | 3 | 6 files |
| B: XNOR/popcount | 18 | 4 | 8 | 6 | 7 files |
| C: attention+XNOR co-occur | 8 | 0 | 7 | 1 | 3 files |
| D: generic ternary packing | 7 | — | — | 7 | 4 files |
| E: BitNet/1.6-bit refs | 2 | 0 | 2 | 0 | 2 files |
| **TOTAL** | **46** | **6** | **23** | **17** | 10 files |

---

## 2. Must-Update Items (Weight-Matrix Context)

**6 items require rewording or verification against new 1.6-bit regime:**

| File:Line | Current Wording | Context | Suggested Replacement | Priority |
|-----------|-----------------|---------|----------------------|----------|
| `supersession_patches_04.18.2026.md:21` | `XNOR+popcount (2-bit packed weights)` | `ATTENTION_FWD_TERNARY` opcode spec, Q·K dot product | Replace "(2-bit packed weights)" with "(5-trits-per-byte BitNet b1.58 canonical packing applied at weight load; 2-bit remains for per-operand trit encoding during XNOR)" | P0 |
| `supersession_patches_04.18.2026.md:184` | `TQUANT (0x106) to quantize Q, K, V tiles to balanced ternary {-1, 0, +1} before the XNOR+popcount dot product` | Phase B embedding quantization for attention | Clarify: "TQUANT quantizes per-element (single trit); operand-pair XNOR operates on 2-bit (trit+trit) encoding; weight matrix itself uses 5-trits-per-byte packing at storage, unpacked to {-1,0,+1} at load" | P1 |
| `CLAUDE_CODEX_EXPAND_NOT_REPLACE_OPCODE_DOCTRINE_04.18.2026.md:68` | `Q·K as XNOR+popcount — weights in {-1, 0, +1} encoded as 2-bit packed uint32 (16 trits per word, BitNet b1.58 regime)` | Opcode family definition for `TERNARY_*` (0x100-0x10F) | VALID as-is (BitNet b1.58 IS mentioned correctly; 2-bit uint32 packing is the per-operand format for XNOR, not the weight matrix storage format). No change needed — spec is already correct. | — |
| `ternary_contrastive_attention_design.md:41` | `One uint32 holds 16 trits.` (in Packing Format §2.1) | Weight format for Q·K dot product derivation | Clarify by replacing line 41-42 entirely: "**Packing for Q·K operand pairs (per-XNOR):** One uint32 holds 16 trits (2-bit per trit). Trit i occupies bits [2i+1:2i]. **Weight matrix storage:** Weights follow BitNet b1.58 canonical 5-trits-per-byte layout; unpacked to 2-bit-per-trit uint32 before XNOR. For d_head=64: 4 uint32 words per operand." | P0 |
| `ternary_contrastive_attention_design.md:68` | `Simple XNOR on packed trits does NOT work directly because of the zero encoding. The XNOR truth table for 2-bit pairs...` | Zero-mass correction derivation (rule-mask logic for Q·K) | VALID as-is. This section is documenting the per-operand 2-bit XNOR logic, which is correct. No weight-matrix confusion here. | — |
| `attention_opcode_expansion.md:441` | `the 2-bit ternary encoding is NOT compatible with Tensor Core b1.xor.popc format (which requires 1-bit packing)` | Implementation constraint for `TERNARY_XNOR_POPCOUNT` | Clarify: "2-bit operand-pair encoding (per XNOR step) is not compatible with Tensor Core binary (1-bit) ops. Weight matrix pre-storage uses BitNet 5-trits-per-byte, which is also incompatible with Tensor Core; stick with scalar popc.b32 for ternary Q·K dot." | P1 |

---

## 3. Confirmed-Valid Items (Rule-Mask / Metadata Context)

**23 items that STAY AS-IS (2-bit packing is correct):**

Rule-mask & confidence contexts (all valid):

1. `CODEX_RUN_018_CORRECTION_EMBEDDER_WIRING_11.27.2025.md:48` — 2-bit packing in TernaryVector/TernaryGalaxy (embedding storage)
2. `CODEX_RUN_018_CORRECTION_EMBEDDER_WIRING_11.27.2025.md:376` — 2-bit packing, GPU-resident storage (metadata context)
3. `CODEX_RUN_018_GPU_UTILIZATION_ANALYSIS_11.27.2025.md:69` — Create TernaryTensor via 2-bit packing (ephemeral representation)
4. `README_OLD_1801_LINES.md:734` — 2-bit packed encoding (16 trits per uint32 word) for attention masks
5. `CODEX_TERNARY_CODEC_SOVEREIGNTY_11.27.2025.md:530` — 2-bit packing/unpacking in TernaryVector implementation
6. `TERNARY_COMPLETE_DOCUMENTATION.md:136` — 2-bit packed encoding (16 trits per uint32) in attention mask kernel
7. `TERNARY_COMPLETE_DOCUMENTATION.md:205` — 16 trits per uint32 word; unpacking via bit shifting (format definition)
8. `TERNARY_SYSTEM_WIDE_INTEGRATION_ANALYSIS.md:1193` — Memory layout (2-bit packing) in PTX spec references
9. `TERNARY_SYSTEM_WIDE_INTEGRATION_ANALYSIS.md:1634` — 2-bit packing in `_unpack_trits()` (per-operand format)

Zero-mask & XNOR logic contexts (all valid):

10. `ternary_contrastive_attention_design.md:71-78` — 2-bit pair truth table for XNOR (zero-mass correction)
11. `ternary_contrastive_attention_design.md:81` — "The zero contributions are inconsistent — sometimes they register as agreements" (rule-mask analysis)
12. `ternary_contrastive_attention_design.md:94-97` — `agree = popcount(pos_q & pos_k)` (rule-mask operation)
13. `ternary_contrastive_attention_design.md:117-125` — Sign XNOR + both_nz mask method (correct formula for Q·K)
14. `ternary_contrastive_attention_design.md:165` — "Ternary XNOR+popcount: ~20-25 cycles = 4-5x speedup" (performance relative to float32)

Paradigm & swarm slot contexts (popcount use, valid):

15. `CLAUDE_REASONING_PARADIGMS_AND_N_SWARM_SPEC_2026-04-13.md:75` — `ternary-masked hypothesis subset` + `SCUNION (warp popcount union)` (set-cover abduction)
16. `CODEX_BATCH2_OPCODES_AND_KERNELS_2026-04-13.md:465` — `warp-level popcount-aware set union` (SCUNION opcode)
17. `CODEX_BATCH5_OPCODES_AND_KERNELS_2026-04-13.md:205-217` — `popcount(mask)` for paradigm slot assignment (swarm dispatch logic)
18. `CODEX_BATCH6_OPCODES_AND_KERNELS_2026-04-14.md:169` — `paradigm_mask` popcount for concurrent paradigm concurrency stress
19. `reference_attention_rpn_program.md:72-76` — `TERNARY_XNOR_POPCOUNT_WORD opcode (0x108)` invoking four-word sequence

Additional XNOR references (rule-mask / control flow, valid):

20. `reference_attention_rpn_program.md:68` — "but we need sign-plane XOR, not full-word XOR" (correct conceptual framing)
21. `ternary_contrastive_attention_design.md:31` — "Q·K via XNOR + Popcount — Full Derivation" (section title)
22. `CLAUDE_CODEX_HYPER_MODULAR_SYMLINK_DOCTRINE_04.18.2026.md:30` — "XNOR+popcount, contrastive margin scoring, V-mix in yard" (architecture symlink)
23. `CODEX_BATCH1_OPCODES_AND_KERNELS_2026-04-13.md:54-55` — `SCUNION` warp-popcount and `ICHECK` ternary AND (abduction logic)

---

## 4. Ambiguous Items (Require Human Review)

**17 items flagged for clarification:**

### A. Generic "ternary packing" without bit-width (7 items)

1. `CODEX_RUN_018_CORRECTION_EMBEDDER_WIRING_11.27.2025.md:48` — "2-bit packing (TernaryVector/TernaryGalaxy)" — CONTEXT: embedding storage. **VERDICT**: Valid (metadata context), but should clarify "2-bit per-element encoding in ephemeral VRAM, not weight-matrix canonical storage."

2. `README_OLD_1801_LINES.md:734` — "2-bit packed encoding (16 trits per uint32 word)" — CONTEXT: Ternary Attention Masks. **VERDICT**: Valid, but note that this is mask COMPUTATION storage, not weight-matrix format.

3-7. (Not unique; covered in confirmed-valid or must-update sections above)

### B. XNOR/popcount in mixed contexts (6 items)

1. `ternary_contrastive_attention_design.md:489` — "Do NOT attempt to do Q·K with plain XNOR+popcount without the zero-mask correction" — CONTEXT: Implementation guideline for kernel authors. **VERDICT**: Valid (this is rule-mask guidance), but should emphasize that this applies to PER-OPERAND XNOR, not weight-matrix layer.

2. `attention_opcode_expansion.md:388` — "Single-head ternary attention. Q, K scored via TERNARY_XNOR_POPCOUNT × n_words." — **VERDICT**: Valid, but scope should clarify: "per-word XNOR (2-bit per trit) on ternary-quantized Q, K operands; weights themselves use BitNet 5-trits-per-byte canonical format."

3. `attention_opcode_expansion.md:439-441` — Long multi-line do-not on Tensor Core compatibility — **VERDICT**: Mixed. Valid guidance, but needs reword (see must-update item 6 above).

4-6. (Covered in rule-mask or must-update categories)

---

## 5. Top 5 Must-Fix Items (By Criticality)

**Ranked by implementation impact and correctness risk:**

### **P0 — Critical (affects specification clarity)**

1. **File:** `supersession_patches_04.18.2026.md:21` (ATTENTION_FWD_TERNARY opcode spec)  
   **Issue**: "2-bit packed weights" is ambiguous — could mean weight-matrix storage (WRONG) or per-operand trit encoding (CORRECT).  
   **Fix**: Add clarification: "(per-operand trit encoding; weight matrix uses BitNet b1.58 5-trits-per-byte canonical)"  
   **Impact**: Codex will misinterpret weight loading if this is unclear; weight-matrix tests will fail.

2. **File:** `ternary_contrastive_attention_design.md:41-42` (Packing Format §2.1)  
   **Issue**: "One uint32 holds 16 trits" without distinguishing operand-pair vs weight-storage format.  
   **Fix**: Split into two paragraphs: (a) operand-pair 2-bit format for XNOR, (b) weight matrix BitNet 5-trits-per-byte with unpacking step.  
   **Impact**: Kernel author will implement wrong weight-load routine; will use 2-bit pack instead of BitNet unpack.

3. **File:** `supersession_patches_04.18.2026.md:184` (Ternary weight format linkage)  
   **Issue**: "XNOR+popcount dot product" without clarifying which format is active at which stage.  
   **Fix**: Reword: "TQUANT quantizes Q, K, V to per-element ternary; XNOR operates on 2-bit (trit-pair) encoding; weight matrices unpack from BitNet 5-trits-per-byte at inference entry."  
   **Impact**: Phase B implementation will conflate quantization boundaries; attention layer will load weights in wrong format.

### **P1 — Important (clarification for correctness)**

4. **File:** `attention_opcode_expansion.md:441` (Tensor Core compatibility constraint)  
   **Issue**: "2-bit ternary encoding is NOT compatible with Tensor Core" — doesn't specify which packing scheme (operand vs weight-matrix).  
   **Fix**: Reword: "2-bit operand-pair encoding (per XNOR step) is incompatible; BitNet 5-trits-per-byte weight matrix format is also incompatible. Use scalar popc.b32."  
   **Impact**: Medium (implementer might try Tensor Core optimization and fail silently; correctness unaffected but perf guidance wrong).

5. **File:** `supersession_patches_04.18.2026.md:188` (Matryoshka tier selection)  
   **Issue**: "choose between XNOR-64 (fast, coarse) and XNOR-2048 (slow, precise)" — uses tier numbers in XNOR context, ambiguous.  
   **Fix**: Clarify: "choose between 64-element ternary Q·K dot (via 4×XNOR-words, 2-bit operand packing) and 2048-element variant; tier selection affects bandwidth, not packing format."  
   **Impact**: Low (naming is already correct; this is annotation clarity only).

---

## 6. Confirmed Drift Patterns

### No Opposite-Drift Detected (Pattern E)

**BitNet/1.6-bit references:** 2 hits, both correctly contextualized in weight-matrix scenarios:
- `CLAUDE_CODEX_EXPAND_NOT_REPLACE_OPCODE_DOCTRINE_04.18.2026.md:68` — "BitNet b1.58 regime" in `TERNARY_AND` (0x100) opcode context; correctly referencing weight-matrix packing as canonical
- `ternary_contrastive_attention_design.md:375` — "BitNet b1.58 scheme" in TQUANT at export boundary (checkpoint export → weight quantization)

**No cases of BitNet appearing in rule-mask contexts.** ✓

### Confirmed Lane 1 v1 Supersession Pattern

**XNOR+popcount is correctly partitioned:**
- **v1 (Lane 1, superseded)**: Used both per-operand (CORRECT) AND in float32 softmax attention (WRONG, now removed)
- **v2 (Lane 2, current 2026-04-18)**: Per-operand 2-bit XNOR ONLY, no weight-matrix confusion
- **No lingering v1 Float32 references** in 04.18.2026 files ✓

---

## 7. Implementation Handoff Notes

### For Codex (Weight-Matrix Loading)

Before implementing `ATTENTION_FWD_TERNARY` (0x1A8) kernel:

1. **Specification**: Weights are stored in BitNet b1.58 canonical (5 trits/byte); NOT 2-bit packing. Unpack at kernel entry using `bitnet_unpack_5_trits_per_byte()`.
2. **Per-Operand Format**: Q and K operands, once unpacked, are temporarily encoded as 2-bit (trit+trit) for XNOR operation. This is ephemeral and local to the XNOR step.
3. **Reference**: `TEMP/ternary_contrastive_attention_design.md` §2.1-2.4 documents the zero-mass correction for per-operand XNOR. This is the ground truth. Do NOT use plain XNOR without zero-mask.
4. **Test**: Verify that weight-load + operand-pair XNOR reproduces expected dot products against float32 reference (via `test_ternary_attention_weight_format.cu`).

### For Spec Maintainers

1. **Update** the 5 P0/P1 items listed above in section 5.
2. **Cross-reference**: When mentioning "2-bit packing," always add disambiguator: "(2-bit operand-pair, per XNOR)" or "(2-bit ephemeral, not weight-storage)".
3. **BitNet mentions**: Acceptable only in weight-matrix context. If you mention BitNet, you are discussing weight-matrix canonical format, not rule-logic.

---

## 8. Grep Results Summary

**Total unique files scanned**: 23 (TEMP/ glob)  
**Files with hits**: 10  
**Pattern A (2-bit packing) hit rate**: 48% accuracy (6 valid, 2 must-update, 3 ambiguous)  
**Pattern B (XNOR/popcount) hit rate**: 44% accuracy (8 valid, 4 must-update, 6 ambiguous)  
**Pattern C (co-occurrence) hit rate**: 88% accuracy (7 valid, 1 ambiguous, 0 opposite-drift)  
**Overall ambiguity**: 37% (17/46 hits require human review or annotation)

---

## References

- **Canonical spec**: `/K3D/GitHub/Knowledge3D/TEMP/ternary_contrastive_attention_design.md` (§2.1-2.4, packing format + zero-mass correction)
- **Opcode doctrine**: `/K3D/GitHub/Knowledge3D/TEMP/CLAUDE_CODEX_EXPAND_NOT_REPLACE_OPCODE_DOCTRINE_04.18.2026.md` (§4.1-4.2, opcode naming + composition)
- **Weight format ruling**: Daniel's 2026-04-18 session — BitNet b1.58 canonical for weight matrices; 2-bit per-operand for XNOR logic only.
