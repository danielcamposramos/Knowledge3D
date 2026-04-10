"""TRM opcode ownership tests for the sovereign GPU path.

The TRM forward-pass opcodes now live in the dedicated 0x300+ range and are
compiled as internal TRM bytecode, not modular-kernel dispatch bytes.
"""

from __future__ import annotations

import numpy as np

from knowledge3d.cranium.ptx_runtime.rpn_opcodes import (
    OP_CHECKPOINT,
    OP_POINTER_LITERAL,
    OP_ROLLBACK,
    OP_TRM_MATVEC_1024x512,
    OP_TRM_MATVEC_512x1024,
    OP_TRM_SWIGLU_1024,
    OP_TRM_SWIGLU_512,
    OP_TRM_VEC_ADD3_512,
    OP_VERIFY,
)
from knowledge3d.cranium.ptx_runtime.trm_rpn_program import (
    build_trm_refine_program,
    expected_trm_opcode_sequence,
)


def test_trm_internal_opcodes_live_in_clean_range() -> None:
    assert OP_TRM_MATVEC_512x1024 == 0x300
    assert OP_TRM_MATVEC_1024x512 == 0x301
    assert OP_TRM_VEC_ADD3_512 == 0x302
    assert OP_TRM_SWIGLU_512 == 0x303
    assert OP_TRM_SWIGLU_1024 == 0x304


def test_checkpoint_bytes_stay_on_modular_kernel_surface() -> None:
    assert OP_CHECKPOINT == 0x60
    assert OP_ROLLBACK == 0x61
    assert OP_VERIFY == 0x62
    assert OP_POINTER_LITERAL == 0x03


def test_trm_refine_program_emits_uint16_schedule() -> None:
    program = build_trm_refine_program(n_steps=1)
    raw = np.frombuffer(program.to_bytes(), dtype=np.uint16)
    assert raw.tolist() == [
        OP_TRM_VEC_ADD3_512,
        OP_TRM_MATVEC_512x1024,
        OP_TRM_SWIGLU_1024,
        OP_TRM_MATVEC_1024x512,
        OP_TRM_VEC_ADD3_512,
        OP_TRM_MATVEC_512x1024,
        OP_TRM_SWIGLU_1024,
        OP_TRM_MATVEC_1024x512,
    ]


def test_trm_expected_sequence_tracks_multiple_steps() -> None:
    seq = expected_trm_opcode_sequence(n_steps=2).tolist()
    per_step = [
        OP_TRM_VEC_ADD3_512,
        OP_TRM_MATVEC_512x1024,
        OP_TRM_SWIGLU_1024,
        OP_TRM_MATVEC_1024x512,
        OP_TRM_VEC_ADD3_512,
        OP_TRM_MATVEC_512x1024,
        OP_TRM_SWIGLU_1024,
        OP_TRM_MATVEC_1024x512,
    ]
    assert seq == per_step + per_step
