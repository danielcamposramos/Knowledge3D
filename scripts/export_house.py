from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge3d.tools.export_house import export_house_glb


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the K3D House as GLB.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("viewer/public/house.glb"),
        help="Output GLB path",
    )
    args = parser.parse_args(argv)
    summary = export_house_glb(args.output)
    print(
        f"Exported House: {summary['rooms']} rooms, "
        f"{summary['total_vertices']} vertices, "
        f"{summary['file_size_kb']:.1f} KB -> {args.output}; "
        f"content={summary['content_entries']} entries -> {summary['content_output']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
