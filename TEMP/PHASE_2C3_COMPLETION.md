Phase 2C-3 Completion (Partial)
===============================

Date: 2025-11-18
Owner: Codex

What Changed
------------
- Added math buffer plumbing between RPN and drawing kernel (new kernel params for math buffer pointer + count).
- Added opcodes in PTX for `ROTATE_MATRIX` (0x79) and `PRECOMPUTED_PATH` (0x7A) that consume math buffer data.
- Bridge now accepts optional `math_buffer` in `execute_rpn_gpu`, copies it to a persistent GPU buffer, and passes it to the kernel.
- Tests added for rotation via math buffer and precomputed path consumption.
- RPN interop design documented (`TEMP/RPN_DRAWING_INTEROP_DESIGN.md`).

Status
------
- Core math buffer flow works for rotation and precomputed paths.
- Arc remains approximate in PTX; RPN-driven arc tessellation still pending (Phase 2D target).
- Begin-path, stroke width, set color, ternary modulate already implemented in PTX.

Tests
-----
- Added: `test_rotate_matrix_90deg`, `test_precomputed_path_triangle` (pass when kernel available).
- Existing suite: 15 passing, 1 skipped (arc), 1 xfail (latency) before arc integration.

Next Steps (Phase 2D)
---------------------
- Implement `_preprocess_rpn_math` to detect `RPN_ARC`, batch-evaluate sin/cos via RPN in parallel, build math_buffer, and replace with `PRECOMPUTED_PATH`.
- Enable `test_gpu_arc` (remove skip) and add performance test for parallel RPN arcs (<5 ms).
- (Optional) Apply stroke_width/color to rasterizer and increase segment buffer stride for per-segment color.
- Consider direct PTX invocation or fused RPN+drawing kernel to remove Python overhead for batched math.
