# DEPRECATED: legacy pre-PTX script; kept for reference. Outputs belong in Knowledge3D.local/old_attempts.
import argparse
import json
import numpy as np
from pathlib import Path

from knowledge3d.tools.phase1_export_library import build_library_glb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default="books.json", help="JSON list: [{title,text,honesty}]")
    ap.add_argument("--positions", default="bookshelf_positions.bin", help="Float32 bin (x,y,z)*N")
    ap.add_argument("--out", default="library_room.glb")
    args = ap.parse_args()

    with open(args.books, 'r') as f:
        books = json.load(f)
    titles = [str(b.get('title', 'Untitled')) for b in books]
    pos = None
    p = Path(args.positions)
    if p.exists():
        arr = np.fromfile(str(p), dtype=np.float32)
        if arr.size % 3 == 0:
            pos = arr.reshape(-1, 3)

    build_library_glb(titles, args.out, positions=pos, books_config=books)
    print(f"Wrote {args.out} from {args.books} and {args.positions}")


if __name__ == "__main__":
    main()

