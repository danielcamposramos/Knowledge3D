import ctypes

import numpy as np
import pytest

from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import (
    RPNProgram,
)
from knowledge3d.cranium.ptx_runtime.rpn_opcodes import (
    OP_TRM_MATVEC_512x1024,
    OP_TRM_MATVEC_1024x512,
    OP_TRM_VEC_ADD3_512,
    OP_TRM_SWIGLU_1024,
)
from knowledge3d.cranium.ptx_runtime.trm_rpn_program import (
    build_trm_refine_program,
    expected_trm_opcode_sequence,
)


def test_trm_program_sequence_repeats_per_step():
    seq = expected_trm_opcode_sequence(n_steps=2)
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
    assert seq.tolist() == per_step + per_step


def test_trm_program_rejects_non_positive_steps():
    with pytest.raises(ValueError):
        build_trm_refine_program(n_steps=0)
    with pytest.raises(ValueError):
        build_trm_refine_program(n_steps=-2)


def test_rpn_program_pointer_resolution_inlines_values():
    program = RPNProgram()
    dummy_ptr = ctypes.c_void_p(0xDEADBEEFCAFEBABE)
    program.ptr(dummy_ptr)
    program.u8(42)
    raw_bytes = program.to_bytes()
    encoded_ptr = int.from_bytes(raw_bytes[:8], byteorder="little", signed=False)
    assert encoded_ptr == dummy_ptr.value
    # Subsequent conversions should not duplicate pointer writes
    second = program.to_bytes()
    assert raw_bytes == second
