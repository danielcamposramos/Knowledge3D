"""
Utilities to load font→RPN datasets and convert them into GPU-ready bytecodes.
"""
import json
from pathlib import Path

import numpy as np

from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge


def load_rpn_jsonl(path: Path):
    """Yield dicts with keys {font, char, rpn} from a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def compile_rpn_entries(entries, bridge=None):
    """Compile RPN strings to bytecode arrays."""
    bridge = bridge or ProceduralDrawingBridge(matryoshka_dim=512)
    bytecodes = []
    for entry in entries:
        bc = bridge.compile_rpn_to_bytecode(entry["rpn"])
        bytecodes.append(bc)
    return bytecodes


def pack_bytecodes(bytecodes):
    """Pack a list of uint8 arrays into a contiguous buffer with offsets."""
    lens = [len(bc) for bc in bytecodes]
    offsets = [0]
    for l in lens:
        offsets.append(offsets[-1] + l)
    packed = np.frombuffer(b"".join([bytes(bc) for bc in bytecodes]), dtype=np.uint8)
    return packed, np.asarray(offsets, dtype=np.int32)
