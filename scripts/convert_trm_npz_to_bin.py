# -*- coding: utf-8 -*-
"""convert_trm_npz_to_bin.py -- Ingestion-path converter: TRM .npz checkpoint to .bin

This script runs on the INGESTION PATH (scripts/).  Numpy is fully sovereign-exempt
here -- it is the right tool for reading .npz files.  The output .bin is consumed at
boot by the sovereign hot-path reader in knowledgeverse.py (_load_trm_weight_checkpoint),
which uses only open() + ctypes, zero numpy.

Binary layout (little-endian throughout):
    Header:
        8  bytes  -- magic b"K3DTRM01"
        4  bytes  -- u32 count (number of matrices that follow)
        4  bytes  -- u32 reserved (write 0, ignore on read)
    For each matrix in canonical order [W1, W2, W3, W4, matryoshka]:
        4  bytes  -- u32 name_len
        name_len bytes -- ASCII name string (no null terminator)
        4  bytes  -- u32 rows
        4  bytes  -- u32 cols
        rows*cols*4 bytes -- f32 row-major data

Usage:
    python scripts/convert_trm_npz_to_bin.py
    python scripts/convert_trm_npz_to_bin.py --input /path/to/trm_weights_latest.npz
    python scripts/convert_trm_npz_to_bin.py --input foo.npz --output bar.bin
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"K3DTRM01"
CANONICAL_ORDER = ("W1", "W2", "W3", "W4", "matryoshka")
EXPECTED_SHAPES: dict[str, tuple[int, int]] = {
    "W1": (1024, 512),
    "W2": (512, 1024),
    "W3": (1024, 512),
    "W4": (512, 1024),
    "matryoshka": (512, 512),
}

DEFAULT_INPUT = Path("/K3D/Knowledge3D.local/checkpoints/trm_weights_latest.npz")


def convert(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    print(f"Loading {input_path} …")
    with np.load(input_path, allow_pickle=False) as npz:
        matrices: dict[str, np.ndarray] = {}
        for name in CANONICAL_ORDER:
            if name not in npz:
                raise KeyError(
                    f"Key '{name}' missing from {input_path}. "
                    f"Available keys: {list(npz.keys())}"
                )
            arr = npz[name].astype(np.float32, copy=False)
            expected = EXPECTED_SHAPES[name]
            if arr.shape != expected:
                raise ValueError(
                    f"Shape mismatch for '{name}': got {arr.shape}, expected {expected}. "
                    "Re-train and re-save the checkpoint before converting."
                )
            matrices[name] = np.ascontiguousarray(arr, dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = len(CANONICAL_ORDER)

    total_bytes = 0
    with open(output_path, "wb") as fh:
        # Header
        header = MAGIC + struct.pack("<II", count, 0)
        fh.write(header)
        total_bytes += len(header)

        for name in CANONICAL_ORDER:
            arr = matrices[name]
            name_bytes = name.encode("ascii")
            rows, cols = arr.shape
            entry_header = struct.pack("<III", len(name_bytes), rows, cols)
            fh.write(entry_header)
            fh.write(name_bytes)
            raw = arr.tobytes()
            fh.write(raw)
            total_bytes += len(entry_header) + len(name_bytes) + len(raw)

    print(f"Written: {output_path}")
    print(f"Total bytes: {total_bytes:,}")
    for name in CANONICAL_ORDER:
        arr = matrices[name]
        print(f"  {name}: shape={arr.shape} ({arr.nbytes:,} bytes f32)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TRM .npz checkpoint to sovereign .bin format.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to .npz checkpoint (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for output .bin (default: same dir as input, named trm_weights.bin)",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    output_path: Path = args.output if args.output is not None else input_path.parent / "trm_weights.bin"

    try:
        convert(input_path, output_path)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
