from knowledge3d.cranium.ternary import TernaryVector, TernaryTensor


def test_ternary_vector_roundtrip():
    vals = [-1, 0, 1, -1, 1, 0, 0, -1, 1]
    tv = TernaryVector(vals)
    assert tv.length == len(vals)
    assert tv.to_python() == vals


def test_ternary_tensor_shape_check():
    tv = TernaryVector([0, 1, -1, 0])
    tensor = TernaryTensor((2, 2), tv)
    assert tensor.shape == (2, 2)
    assert tensor.to_python() == [0, 1, -1, 0]
