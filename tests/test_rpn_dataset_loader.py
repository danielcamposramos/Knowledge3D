from __future__ import annotations

import numpy as np

from knowledge3d.ingestion.fonts.rpn_dataset_loader import pack_bytecodes


def test_pack_bytecodes():
    bc1 = np.array([1, 2, 3], dtype=np.uint8)
    bc2 = np.array([4, 5], dtype=np.uint8)
    packed, offsets = pack_bytecodes([bc1, bc2])
    assert packed.tolist() == [1, 2, 3, 4, 5]
    assert offsets.tolist() == [0, 3, 5]
