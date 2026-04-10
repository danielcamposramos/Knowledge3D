from __future__ import annotations

from pathlib import Path

import pytest

from knowledge3d.cranium.bridges.sovereign_bridges import ModularRPNEngine as BridgeModularRPNEngine
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
from knowledge3d.cranium.ptx_runtime.rpn_opcodes import (
    OP_CHECKPOINT,
    OP_DRAW_FIELD_COEF,
    OP_DRAW_MOVE,
    OP_DRAW_REL_LINE,
    OP_TADD,
    OP_TRM_SWIGLU_1024,
    OP_TEX_BAKE,
    OP_TEX_PERLIN_NOISE,
)


def test_texture_registry_and_conflict_owners_are_explicit() -> None:
    assert OP_CHECKPOINT == 0x60
    assert OP_TRM_SWIGLU_1024 == 0x304
    assert OP_DRAW_MOVE == 0x200
    assert OP_TADD == 0x70
    assert OP_DRAW_REL_LINE == 0x211
    assert OP_DRAW_FIELD_COEF == 0x212
    assert ModularRPNEngine.OPCODES["tex_perlin_noise"] == OP_TEX_PERLIN_NOISE
    assert ModularRPNEngine.OPCODES["tex_bake"] == OP_TEX_BAKE
    assert "MOVE" not in ModularRPNEngine.OPCODES
    assert ModularRPNEngine.OPCODES["rel_line"] == OP_DRAW_REL_LINE


def test_kkrieger_grounding_is_present_in_texture_source() -> None:
    source = Path("knowledge3d/cranium/kernels/tex_noise_kernels.cu").read_text(encoding="utf-8")
    assert "0x93638245u" in source
    assert "x * x * x * (10.0f + x * (6.0f * x - 15.0f))" in source


def test_perlin_texture_smoke_and_bake() -> None:
    try:
        engine = BridgeModularRPNEngine()
        engine.bind_texture_pool(width=32, height=32, slot_count=32)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not initialise texture-capable ModularRPNEngine: {exc}")

    texture_id = engine.execute_single(
        instance_id=0,
        op_codes=[
            0x00,
            0x00,
            0x00,
            0x00,
            OP_TEX_PERLIN_NOISE,
            0x00,
            0x00,
            OP_TEX_BAKE,
        ],
        scalars=[4.0, 6.0, 1.0, 0.5, 32.0, 32.0],
        vectors=[],
    )
    width, height, values = engine.read_baked_texture(round(texture_id))
    assert (width, height) == (32, 32)
    assert len(values) == 32 * 32
    assert max(values) - min(values) > 1.0e-4
    assert abs(values[1] - values[0]) > 1.0e-5
    assert len({round(value, 6) for value in values[:128]}) > 32
