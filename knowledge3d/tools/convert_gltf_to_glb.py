from __future__ import annotations

"""Convert legacy JSON .gltf files into PTX-ready binary .glb assets."""

import argparse
from pathlib import Path

from pygltflib import GLTF2, BufferFormat  # type: ignore


def convert(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source glTF not found: {src}")
    if src.suffix.lower() != ".gltf":
        raise ValueError("Source file must have .gltf extension")

    gltf = GLTF2().load(src.as_posix())
    gltf.convert_buffers(BufferFormat.BINARYBLOB)
    dst.parent.mkdir(parents=True, exist_ok=True)
    gltf.save(dst.as_posix())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert JSON glTF to binary glb")
    parser.add_argument("src", type=Path, help="Path to .gltf file")
    parser.add_argument("dst", type=Path, help="Output .glb path")
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    args = parse_args()
    convert(args.src.resolve(), args.dst.resolve())


if __name__ == "__main__":  # pragma: no cover
    main()
