from __future__ import annotations

from pathlib import Path

from knowledge3d.cranium.ptx_runtime import rpn_opcodes as op


def test_batch4_cbr_opcode_constants_match_registry_doc() -> None:
    registry = Path("docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md").read_text(encoding="utf-8")
    assert "| `0x100` | `CASE_FETCH` |" in registry
    assert "| `0x101` | `CASE_REBIND` |" in registry
    assert "| `0x102` | `CASE_REVISE` |" in registry
    assert "| `0x103` | `CASE_RETAIN_HINT` |" in registry
    assert op.OP_CASE_FETCH == 0x100
    assert op.OP_CASE_REBIND == 0x101
    assert op.OP_CASE_REVISE == 0x102
    assert op.OP_CASE_RETAIN_HINT == 0x103


def test_batch4_dispatch_and_tier_sources_reference_new_cbr_block() -> None:
    modular_kernel = Path("knowledge3d/cranium/kernels/modular_rpn_kernel.cu").read_text(encoding="utf-8")
    tiered = Path("knowledge3d/cranium/bridges/tiered_rpn.py").read_text(encoding="utf-8")
    assert "case 0x100" in modular_kernel
    assert "case 0x101" in modular_kernel
    assert "case 0x102" in modular_kernel
    assert "case 0x103" in modular_kernel
    assert "0x100" in tiered
    assert "0x101" in tiered
    assert "0x102" in tiered
    assert "0x103" in tiered
