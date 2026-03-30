from __future__ import annotations

import math

from knowledge3d.knowledgeverse.arc3_frame_encoder import ARC3FrameEncoder
from knowledge3d.knowledgeverse.sovereign_text_embedder import embed_text_sovereign


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    norm_a = math.sqrt(sum(float(x) * float(x) for x in a) + 1.0e-12)
    norm_b = math.sqrt(sum(float(y) * float(y) for y in b) + 1.0e-12)
    return dot / (norm_a * norm_b)


def test_frame_encoder_aligned_with_galaxy():
    encoder = ARC3FrameEncoder()

    frame_up = [[0] * 8 for _ in range(8)]
    frame_up[1][4] = 2

    frame_down = [[0] * 8 for _ in range(8)]
    frame_down[6][4] = 2

    emb_up = encoder.encode(frame_up)
    emb_down = encoder.encode(frame_down)

    move_up_emb = embed_text_sovereign("up north move_up above spatial translate")
    move_down_emb = embed_text_sovereign("down south move_down below spatial translate")
    spatial_emb = embed_text_sovereign("spatial grid navigate translate object color")

    sim_up_to_move_up = _cosine(emb_up, move_up_emb)
    sim_up_to_move_down = _cosine(emb_up, move_down_emb)
    sim_down_to_move_down = _cosine(emb_down, move_down_emb)
    sim_down_to_move_up = _cosine(emb_down, move_up_emb)
    sim_to_spatial = _cosine(emb_up, spatial_emb)

    assert sim_up_to_move_up > 0.0
    assert sim_down_to_move_down > 0.0
    assert sim_up_to_move_up > sim_up_to_move_down
    assert sim_down_to_move_down > sim_down_to_move_up
    assert sim_to_spatial > 0.0


def test_arc3_action_select_dims_preserved():
    encoder = ARC3FrameEncoder()
    frame = [[0] * 8 for _ in range(8)]
    frame[3][4] = 2

    emb = encoder.encode(frame)

    assert abs(emb[10] - (4.0 / 8.0)) < 0.05
    assert abs(emb[11] - (3.0 / 8.0)) < 0.05
    assert emb[12] >= 0.0
    assert emb[13] >= 0.0
    assert 0.0 <= emb[28] <= 1.0
    assert 0.0 <= emb[29] <= 1.0
    assert 0.0 <= emb[31] <= 1.0
