# Gate 7 — Matryoshka Weight-Matrix Pack-Order Verification Specification

**Date**: 2026-04-18
**Author**: Claude (architecture)
**Context**: Bulk-Lib Purge Hard Acceptance Gate 7
**Referenced by**: `CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md` §6 Gate 8

---

## 1. Motivation

Ternary-packed weight matrices (1.6-bit per weight, 5 trits per byte) enable Matryoshka tier-prefix truncation: a 4096-dim weight can be truncated to 64-dim by reading only the first 64/64 trits and re-normalizing. This compression is **ONLY valid if rows are stored in ascending row-index order**.

When rows are unordered, truncation silently produces wrong results — the kernel executes but computes incorrect per-tier semantics. This is a **silent correctness failure**, not a crash. Gate 7 makes pack-order mandatory and auditable.

---

## 2. Definition: Matryoshka Tier Hierarchy

| Tier | Dimension | Trits | Bytes (5:1 ratio) | Use Case |
|---|---|---|---|---|
| Tier 0 | 64 | 320 | 64 | Room-level navigation (cheap LOD) |
| Tier 1 | 128 | 640 | 128 | Shelf-level semantic search |
| Tier 2 | 512 | 2560 | 512 | Star-level identity (canonical dedup) |
| Tier 3 | 2048 | 10240 | 2048 | Full-precision reasoning |

**Pack-order rule:** When storing a full (4096-dim) matrix to VRAM:
1. Expand 5-trit-per-byte encoding to individual trits.
2. **Rows MUST be written in ascending order: row 0, row 1, ..., row N−1.**
3. Within each row, values are in column order (standard matrix layout).
4. Prefix truncation to tier-T is valid IFF row order is ascending (no shuffling, no compression).

---

## 3. Pack-Order Header Format

Every ternary-packed weight matrix buffer MUST have a **1-byte pack-order header** prepended:

```
Byte 0: pack_order_flags
  - Bit 0: ASCENDING_ROW_ORDER (1 = rows are in ascending index order, 0 = unknown/unordered)
  - Bit 1: CANONICAL_COLUMN_ORDER (1 = columns are in canonical order, 0 = mixed)
  - Bits 2-7: reserved (must be 0)

Bytes 1-4: version (uint32_be) — currently 0x00000001
Bytes 5-8: matrix_rows (uint32_be)
Bytes 9-12: matrix_cols (uint32_be)
Bytes 13-N: packed_data (5-trits-per-byte)
```

**Example valid header:**
```
0x01 0x00 0x00 0x00 0x01 0x00 0x00 0x10 0x00 0x00 0x20
 ^^^  ^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^
 flags version=1  rows=256        cols=512
```

---

## 4. Kernel-Registration Static Assert

When a weight matrix is registered with a RPN opcode or Neural Network layer, the kernel MUST verify:

```cuda
// In opcode registration code or weight loader
void validate_matryoshka_packorder(const uint8_t* weight_buffer, size_t buffer_size) {
    if (buffer_size < 13) {
        // ERROR: buffer too small for header
        throw std::runtime_error("Weight matrix buffer too small (no header)");
    }
    
    uint8_t pack_order_flags = weight_buffer[0];
    if (!(pack_order_flags & 0x01)) {
        // ERROR: ASCENDING_ROW_ORDER flag not set
        throw std::runtime_error(
            "Weight matrix pack-order violation: rows not in ascending order. "
            "Tier-prefix truncation is invalid. Repack the matrix with ASCENDING_ROW_ORDER=1."
        );
    }
    
    uint32_t version = be32(weight_buffer + 1);
    uint32_t rows = be32(weight_buffer + 5);
    uint32_t cols = be32(weight_buffer + 9);
    
    if (version != 1) {
        throw std::runtime_error("Unknown weight matrix version");
    }
    if (rows == 0 || cols == 0) {
        throw std::runtime_error("Invalid matrix dimensions");
    }
    
    // Assert: first row index == 0, last row index == rows-1, no gaps
    // (This is implicit in sequential packing, but check during load for paranoia)
}
```

---

## 5. Runtime Pack-Order Check

On weight upload (once per session, not per-inference):

```cuda
__global__ void verify_packorder_ascending(
    const uint8_t* packed_data,
    uint32_t rows, uint32_t cols,
    uint8_t* output_valid  // 1 if valid, 0 if not
) {
    // Spot-check: extract first row, last row, middle row
    // Assert they are in order (heuristic, not exhaustive)
    // This catches accidental shuffles during ingestion
    
    // For full verification, would need O(N) row-index extraction
    // For production: trust header flag + spot checks
}
```

---

## 6. Example: Correct Pack-Order Layout

**4×3 matrix (rows=4, cols=3), tier boundaries at [2, 3] (2 trit prefix for Tier 0/1, 3 for Tier 2):**

```
Input matrix (float32):
  [ 0.2  -0.5   0.1 ]  row 0
  [ 0.3   0.8  -0.2 ]  row 1
  [-0.1   0.4   0.7 ]  row 2
  [ 0.6  -0.3   0.9 ]  row 3

Ternary quantization (5 trit max = {-2, -1, 0, +1, +2}):
  [ +1  -1   0 ]  row 0
  [ +1  +1  -1 ]  row 1
  [ 0   +1  +1 ]  row 2
  [ +1  -1  +1 ]  row 3

Pack to 5-trits-per-byte (3 cols × 4 rows = 12 trits = 3 bytes after header):
  Byte layout (ascending row order):
    Row 0: +1, -1,  0     (trits 0-2)
    Row 1: +1, +1, -1     (trits 3-5)
    Row 2:  0, +1, +1     (trits 6-8)
    Row 3: +1, -1, +1     (trits 9-11)

Tier truncation (Tier 0 = first 2 trits per row):
    Row 0[0:2] = +1, -1   ✓ valid (rows in order)
    Row 1[0:2] = +1, +1   ✓ valid
    ...

Result: Tier-0 vector = [+1, -1, +1, +1, 0, +1, +1, -1] (2 trits × 4 rows)
```

If rows were shuffled (Row 3, Row 1, Row 0, Row 2), the same tier truncation would extract [+1, -1, +1, +1, +1, -1, ...] — semantically wrong.

---

## 7. Acceptance Criteria (for Gate 7 CI check)

```bash
# Grep criterion: at least one kernel or weight loader MUST reference pack-order verification
grep -rn "ASCENDING_ROW_ORDER\|pack_order_header\|validate_matryoshka_packorder" \
    knowledge3d/cranium/ptx/ \
    knowledge3d/cranium/kernels/ \
    knowledge3d/cranium/loaders/ \
    --include="*.cu" \
    --include="*.cuh" \
    --include="*.py"
# Expected: ≥3 hits (kernel registration, runtime check, loader)

# Type check: weight buffer header is always (uint8_t*, size_t)
# No bare float arrays in ternary matrix initialization
grep -rn "float.*weight\|np\.random\.randn.*weight\|torch\.randn.*weight" \
    knowledge3d/cranium/specialists/ \
    --include="*.py"
# Expected: 0 hits (all weight init via structured header buffer)
```

---

## 8. Codex Handoff

1. When writing `modular_rpn_kernel_transfer_yard.cu` (Transfer Yard spec item), include the header format check.
2. When opcode weights are registered (any opcode with learnable parameters), invoke `validate_matryoshka_packorder()`.
3. Add `gate_7_matryoshka_packorder_verification_spec.md` to the CI acceptance gates document as reference.
4. If any weight matrix is built via ingestion (e.g., in `star_crafter.py`), ensure the pack-order header is written with flag=0x01.
5. Run Gate 7 grep check as part of Phase 8 final sweep.

---

## 9. Reference

- **Transfer Yard spec**: `CLAUDE_CODEX_TRANSFER_YARD_AND_EMBEDDING_SOVEREIGNTY_04.18.2026.md` §4 (yard stack layout, ternary encoding)
- **Bulk-Lib spec**: `CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md` §6 Gate 8
- **BitNet b1.58 feedback**: `feedback_bitnet_b158_ternary_pattern.md` (5 trits per byte, weight-matrix compression)
