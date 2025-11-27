from knowledge3d.training.arc_agi.rpn_executor import ARCRPNExecutor


def test_rotate_90():
    exec_ = ARCRPNExecutor()
    grid = [[1, 2], [3, 4]]
    result = exec_.execute(grid, "1 rotate")
    assert result == [[3, 1], [4, 2]]


def test_flip_horizontal():
    exec_ = ARCRPNExecutor()
    grid = [[1, 2], [3, 4]]
    result = exec_.execute(grid, "FLIP_H")
    assert result == [[2, 1], [4, 3]]


def test_translate():
    exec_ = ARCRPNExecutor()
    grid = [[1, 0], [0, 2]]
    result = exec_.execute(grid, "1 0 TRANSLATE")
    assert result == [[0, 1], [0, 0]]


def test_recolor():
    exec_ = ARCRPNExecutor()
    grid = [[1, 2], [2, 1]]
    result = exec_.execute(grid, "1 3 RECOLOR")
    assert result == [[3, 2], [2, 3]]
