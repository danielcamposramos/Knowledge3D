from __future__ import annotations

from knowledge3d.knowledgeverse.arc3_frame_encoder import ARC3FrameEncoder


def test_arc3_frame_encoder_emits_64d_geometry_packet():
    encoder = ARC3FrameEncoder()

    frame = [[0] * 8 for _ in range(8)]
    frame[1][4] = 2
    emb = encoder.encode(frame)

    assert len(emb) == 64
    assert abs(emb[10] - (4.0 / 8.0)) < 0.05
    assert abs(emb[11] - (1.0 / 8.0)) < 0.05
    assert emb[12] >= 0.0
    assert emb[13] >= 0.0
    assert 0.0 <= emb[28] <= 1.0
    assert 0.0 <= emb[29] <= 1.0
    assert 0.0 <= emb[31] <= 1.0
    assert any(abs(float(value)) > 1.0e-6 for value in emb[32:])


def test_arc3_frame_encoder_opposite_positions_separate_reserved_lanes():
    encoder = ARC3FrameEncoder()

    frame_up = [[0] * 8 for _ in range(8)]
    frame_up[1][4] = 2
    frame_down = [[0] * 8 for _ in range(8)]
    frame_down[6][4] = 2

    emb_up = encoder.encode(frame_up)
    emb_down = encoder.encode(frame_down)

    assert emb_up[11] < emb_down[11]
    assert emb_up[6] < emb_down[6]
    assert abs(sum(emb_up[32:48]) - sum(emb_down[32:48])) < 0.25
