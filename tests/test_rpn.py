from knowledge3d.core.legacy_rpn_python import LegacyPythonRPN


def test_rpn_basic_math():
    r = LegacyPythonRPN(max_depth=8, keep_top=5)
    assert r.eval([2, 3, '+']) == 5
    assert abs(r.eval([10, 2, '/']) - 5) < 1e-9
    assert r.eval([5, 2, '*']) == 10


def test_rpn_cosine():
    r = LegacyPythonRPN()
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    c = [1.0, 0.0, 0.0]
    assert abs(r.cosine(a, b) - 0.0) < 1e-9
    assert abs(r.cosine(a, c) - 1.0) < 1e-9

