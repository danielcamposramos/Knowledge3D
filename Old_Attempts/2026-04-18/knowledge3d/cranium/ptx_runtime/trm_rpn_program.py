"""
TRM ↦ RPN program builder.

This helper encodes the Tiny Recursive Model (TRM) refinement loop as an
RPN bytecode sequence that can be executed by a Tier‑3 RPN kernel once the
new TRM opcodes (0x300–0x304) are active.  The goal is to let downstream
components compile TRM updates into data rather than issuing bespoke kernel
launches.

The bytecode emitted here mirrors the execution order enumerated in the
`RPN_SOVEREIGN_AI_FRAMEWORK_V2.md` roadmap:

    1. temp  ← q + y + z
    2. hidden ← W1 @ temp          (512→1024)
    3. hidden ← swiglu(hidden)     (1024)
    4. z_new ← W2 @ hidden         (1024→512)
    5. temp2 ← y + z_new (+ 0)     (512)            # zero vector injected later
    6. hidden2 ← W3 @ temp2        (512→1024)
    7. hidden2 ← swiglu(hidden2)   (1024)
    8. y_new ← W4 @ hidden2        (1024→512)

Each refinement step repeats this stencil.  The actual data pointers
(weights, buffers, scratch space) are injected by higher layers when the
program is bound to device memory; this helper focuses purely on opcode
layout so it can be unit tested in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from .modular_rpn_engine import RPNProgram
from .rpn_opcodes import (
    OP_TRM_MATVEC_512x1024,
    OP_TRM_MATVEC_1024x512,
    OP_TRM_VEC_ADD3_512,
    OP_TRM_SWIGLU_1024,
)


@dataclass(frozen=True)
class TRMRPNTemplate:
    """Pre-computed opcode schedule for a single TRM refinement step."""

    opcodes: List[int]

    @property
    def as_uint16(self) -> np.ndarray:
        return np.asarray(self.opcodes, dtype=np.uint16)


def _single_step_template() -> TRMRPNTemplate:
    """Return the canonical opcode order for one refinement iteration."""
    return TRMRPNTemplate(
        opcodes=[
            OP_TRM_VEC_ADD3_512,      # q + y + z
            OP_TRM_MATVEC_512x1024,   # hidden = W1 @ temp
            OP_TRM_SWIGLU_1024,       # hidden = swiglu(hidden)
            OP_TRM_MATVEC_1024x512,   # z_new = W2 @ hidden
            OP_TRM_VEC_ADD3_512,      # temp2 = y + z_new (+ zero)
            OP_TRM_MATVEC_512x1024,   # hidden2 = W3 @ temp2
            OP_TRM_SWIGLU_1024,       # hidden2 = swiglu(hidden2)
            OP_TRM_MATVEC_1024x512,   # y_new = W4 @ hidden2
        ]
    )


def build_trm_refine_program(n_steps: int = 6) -> RPNProgram:
    """Create a bytecode program that performs ``n_steps`` TRM refinements.

    Args:
        n_steps: Number of recursive refinement iterations (default 6).

    Returns:
        An ``RPNProgram`` containing the opcode schedule.  The program does
        not embed device pointers yet; upstream code injects those via
        ``RPNProgram.ptr`` before execution.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    template = _single_step_template()
    program = RPNProgram()
    for _ in range(n_steps):
        for opcode in template.opcodes:
            program.u16(opcode)
    return program


def expected_trm_opcode_sequence(n_steps: int) -> np.ndarray:
    """Convenience helper for tests to inspect the raw opcode order."""
    program = build_trm_refine_program(n_steps=n_steps)
    raw = np.frombuffer(program.to_bytes(), dtype=np.uint16)
    count = n_steps * len(_single_step_template().opcodes)
    return raw[:count]


__all__ = ["TRMRPNTemplate", "build_trm_refine_program", "expected_trm_opcode_sequence"]
