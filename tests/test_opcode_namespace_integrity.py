"""Verify the opcode namespace has no cross-domain byte conflicts."""

from knowledge3d.cranium.ptx_runtime import rpn_opcodes as op


def test_drawing_ops_in_dedicated_range() -> None:
    draw_ops = [
        op.OP_DRAW_MOVE,
        op.OP_DRAW_LINE,
        op.OP_DRAW_QUAD,
        op.OP_DRAW_CUBIC,
        op.OP_DRAW_ARC,
        op.OP_DRAW_CLOSE,
        op.OP_DRAW_STROKE,
        op.OP_DRAW_FILL,
        op.OP_DRAW_PUSH_STATE,
        op.OP_DRAW_POP_STATE,
        op.OP_DRAW_TRANSLATE,
        op.OP_DRAW_ROTATE,
        op.OP_DRAW_SCALE,
        op.OP_DRAW_SET_STROKE_COLOR,
        op.OP_DRAW_SET_FILL_COLOR,
        op.OP_DRAW_SET_LINE_WIDTH,
        op.OP_DRAW_SET_TERNARY_HINT,
        op.OP_DRAW_REL_LINE,
        op.OP_DRAW_FIELD_COEF,
        op.OP_DRAW_DOT_EMIT,
        op.OP_DRAW_VECTORDOTMAP_ENCODE,
        op.OP_DRAW_VECTORDOTMAP_DECODE,
        op.OP_DRAW_LAYER_NEW,
    ]
    for value in draw_ops:
        assert 0x200 <= value <= 0x21F, (
            f"Drawing op {hex(value)} outside dedicated range 0x200-0x21F"
        )


def test_ternary_ops_in_dedicated_range() -> None:
    ternary = [
        op.OP_TADD,
        op.OP_TMUL,
        op.OP_TNOT,
        op.OP_TCOMP,
        op.OP_TQUANT,
        op.OP_TPACK,
        op.OP_TUNPACK,
    ]
    for value in ternary:
        assert 0x70 <= value <= 0x76, (
            f"Ternary op {hex(value)} out of range 0x70-0x76"
        )


def test_trm_ops_in_dedicated_range() -> None:
    trm = [
        op.OP_TRM_MATVEC_512x1024,
        op.OP_TRM_MATVEC_1024x512,
        op.OP_TRM_VEC_ADD3_512,
        op.OP_TRM_SWIGLU_512,
        op.OP_TRM_SWIGLU_1024,
    ]
    for value in trm:
        assert 0x300 <= value <= 0x30F, (
            f"TRM op {hex(value)} outside dedicated range 0x300-0x30F"
        )


def test_cas_ops_in_dedicated_range() -> None:
    cas = [
        op.OP_POLY_COEFF,
        op.OP_POLY_BUILD,
        op.OP_POLY_ADD,
        op.OP_POLY_MUL,
        op.OP_SIMPLIFY,
        op.OP_SUBSTITUTE,
        op.OP_SOLVE_LINEAR,
        op.OP_RULE_APPLY,
    ]
    for value in cas:
        assert 0x220 <= value <= 0x25F, (
            f"CAS op {hex(value)} outside dedicated range 0x220-0x25F"
        )


def test_sas_ops_in_dedicated_range() -> None:
    sas_vals = [
        op.OP_CANONICALIZE,
        op.OP_CAS_HASH,
        op.OP_SEMANTIC_RESOLVE,
        op.OP_RULE_SELECT,
        op.OP_CONTEXTUAL_REWRITE,
        op.OP_SEMANTIC_EQUIV,
    ]
    for value in sas_vals:
        assert 0x238 <= value <= 0x25F, f"SAS opcode 0x{value:X} out of range"


def test_checkpoint_constants_exist() -> None:
    assert op.OP_CHECKPOINT == 0x60
    assert op.OP_ROLLBACK == 0x61
    assert op.OP_VERIFY == 0x62


def test_no_cross_domain_byte_conflicts() -> None:
    drawing = set(range(0x200, 0x220))
    cas = set(range(0x220, 0x260))
    ternary = {0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76}
    trm_internal = set(range(0x300, 0x310))
    checkpoint = {0x60, 0x61, 0x62}

    assert not drawing & ternary
    assert not drawing & cas
    assert not drawing & trm_internal
    assert not drawing & checkpoint
    assert not cas & ternary
    assert not cas & trm_internal
    assert not cas & checkpoint
    assert not ternary & trm_internal
    assert not ternary & checkpoint


def test_no_cross_domain_conflicts_sas() -> None:
    sas_set = {
        op.OP_CANONICALIZE,
        op.OP_CAS_HASH,
        op.OP_SEMANTIC_RESOLVE,
        op.OP_RULE_SELECT,
        op.OP_CONTEXTUAL_REWRITE,
        op.OP_SEMANTIC_EQUIV,
    }
    drawing_set = {value for key, value in vars(op).items() if key.startswith("OP_DRAW_")}
    ternary_set = {
        op.OP_TADD,
        op.OP_TMUL,
        op.OP_TNOT,
        op.OP_TCOMP,
        op.OP_TQUANT,
        op.OP_TPACK,
        op.OP_TUNPACK,
    }
    trm_set = {
        op.OP_TRM_MATVEC_512x1024,
        op.OP_TRM_MATVEC_1024x512,
        op.OP_TRM_VEC_ADD3_512,
        op.OP_TRM_SWIGLU_512,
        op.OP_TRM_SWIGLU_1024,
    }
    assert sas_set.isdisjoint(drawing_set)
    assert sas_set.isdisjoint(ternary_set)
    assert sas_set.isdisjoint(trm_set)
