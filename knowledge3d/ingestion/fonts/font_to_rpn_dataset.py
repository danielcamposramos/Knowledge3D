"""
Builds an RPN dataset from system fonts and characters, storing bytecode +
metadata ready for GPU ingestion (segments are generated on GPU at runtime).
"""
import argparse
import json
from pathlib import Path
from typing import List

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from knowledge3d.cranium.procedural_fonts import glyph_to_rpn
from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge


def enumerate_chars() -> List[str]:
    # Basic Latin + a few symbols; extend as needed
    chars = [chr(i) for i in range(32, 127)]
    return chars


def build_dataset(font_dir: Path, out_path: Path, stroke_width: float = 1.0, emit_bytecode_npz: Path | None = None):
    chars = enumerate_chars()
    entries = []
    bytecodes = []
    for font_path in sorted(font_dir.glob("*.ttf")):
        for ch in chars:
            rpn = glyph_to_rpn(str(font_path), ch, stroke_width=stroke_width)
            if not rpn:
                continue
            entry = {
                "font": font_path.name,
                "char": ch,
                "rpn": rpn,
            }
            entries.append(entry)
            if emit_bytecode_npz is not None:
                bc = ProceduralDrawingBridge(matryoshka_dim=512).compile_rpn_to_bytecode(rpn)
                bytecodes.append(bc)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote {len(entries)} entries to {out_path}")
    if emit_bytecode_npz is not None and bytecodes:
        lens = [len(b) for b in bytecodes]
        offsets = [0]
        for l in lens:
            offsets.append(offsets[-1] + l)
        packed = b"".join([bytes(b) for b in bytecodes])
        import numpy as np
        np.savez_compressed(emit_bytecode_npz, bytecode=np.frombuffer(packed, dtype=np.uint8), offsets=np.array(offsets, dtype=np.int32))
        print(f"Wrote packed bytecodes to {emit_bytecode_npz}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fonts", type=str, default="/usr/share/fonts/truetype/dejavu")
    parser.add_argument("--out", type=str, default="data/font_rpn_dataset.jsonl")
    parser.add_argument("--stroke-width", type=float, default=1.0)
    parser.add_argument("--emit-bytecode-npz", type=str, default=None)
    args = parser.parse_args()
    emit_npz = Path(args.emit_bytecode_npz) if args.emit_bytecode_npz else None
    build_dataset(Path(args.fonts), Path(args.out), stroke_width=args.stroke_width, emit_bytecode_npz=emit_npz)


if __name__ == "__main__":
    main()
