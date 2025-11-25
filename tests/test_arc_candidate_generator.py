import numpy as np

from knowledge3d.training.arc_agi.candidate_generator import CandidateGenerator


def _as_key(grid):
    return tuple(tuple(row) for row in grid)


def test_candidate_generation_infers_example_and_limits_count():
    gen = CandidateGenerator(max_candidates=20)
    input_grid = [
        [1, 0],
        [0, 0],
    ]
    # Example shows a simple translation to the right.
    train_examples = [
        {"input": [[1, 0], [0, 0]], "output": [[0, 1], [0, 0]]},
    ]

    candidates = gen.generate_candidates(input_grid, train_examples)

    # Cap respected.
    assert len(candidates) <= gen.max_candidates

    # Output grids are unique.
    outputs = [_as_key(c[0]) for c in candidates]
    assert len(outputs) == len(set(outputs))

    # Example-derived translation appears in the pool.
    expected = _as_key([[0, 1], [0, 0]])
    assert expected in outputs


def test_math_candidates_include_even_fill():
    gen = CandidateGenerator(max_candidates=20)
    input_grid = np.zeros((3, 3), dtype=int).tolist()
    candidates = gen.generate_candidates(input_grid, [])

    outputs = {_as_key(c[0]) for c in candidates}
    even_mask = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=int,
    )
    filled = even_mask.tolist()

    assert _as_key(filled) in outputs
