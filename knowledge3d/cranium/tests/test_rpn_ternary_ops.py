import pytest

from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine


@pytest.fixture(scope="module")
def engine():
    return ModularRPNEngine()


def test_tadd_tmul(engine):
    assert engine.evaluate("1 1 tadd") == 1.0
    assert engine.evaluate("1 -1 tadd") == 0.0
    assert engine.evaluate("1 -1 tmul") == -1.0


def test_tquant_and_comp(engine):
    assert engine.evaluate("0.5 tquant") == 1.0
    assert engine.evaluate("-0.2 tquant") == 0.0
    assert engine.evaluate("0.2 0.5 tcomp") == -1.0  # 0.2 - 0.5 => -1


def test_tpack_tunpack(engine):
    # Pack (+1, -1) then unpack; top of stack should be -1, beneath +1
    engine.reset()
    result = engine.evaluate("1 -1 tpack tunpack")
    assert result == -1.0
    # Check the first unpacked trit by leaving only one push
    val_first = engine.evaluate("1 -1 tpack tunpack drop")
    assert val_first == 1.0
