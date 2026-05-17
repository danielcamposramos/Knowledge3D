"""Ternary primitives — post-purge stub.

Post-purge surface (2026-04-18):
    ``ternary_vector.py`` and ``ternary_galaxy.py`` were moved to
    ``Old_Attempts/2026-04-18/knowledge3d/cranium/ternary/`` per
    ``TEMP/CLAUDE_ABSOLUTE_SOVEREIGNTY_PURGE_04.18.2026.md``. Both were
    numpy-native host-side helpers; the sovereign successors are:

        knowledge3d/cranium/ptx/ternary_ops.ptx
        knowledge3d/cranium/ptx/modular_rpn_kernel_lite_transfer_yard.ptx
        knowledge3d/cranium/ptx_runtime/rpn_math_core.py

    Daniel's BitNet b1.58 ruling (2026-04-18) supersedes the old 2-bit
    packing for weight matrices: five trits per byte (1.6 bits/weight)
    plus multiplication-free add/sub/skip kernels. See
    ``feedback_bitnet_b158_ternary_pattern.md``. Rule masks still use
    2-bit packing.

    Content-addressed frame storage (the old ``TernaryGalaxy``) is
    covered by the canonical-ID table Codex is building in Phase 7
    (see ``TEMP/CODEX_PROCEDURALIZATION_NORMALIZATION_SPEC_04.18.2026.md``).
"""

__all__: list[str] = []
