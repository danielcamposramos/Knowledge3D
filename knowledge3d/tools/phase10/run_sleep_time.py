from __future__ import annotations

import argparse
from ...cranium.phase10.sleep_time_compute import SleepTimeCompute  # type: ignore


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Run Sleep-Time Compute & Permanent Materialization")
    ap.add_argument("--house", required=True, help="Path to House GLB")
    ap.add_argument("--galaxy", required=True, help="Path to Galaxy GLB")
    ap.add_argument("--output", default=None, help="Output House GLB path (stub)")
    ap.add_argument("--material_dir", default=None, help="Directory for materialized objects")
    args = ap.parse_args()
    stc = SleepTimeCompute(house_path=args.house, galaxy_path=args.galaxy, output_path=args.output, material_dir=args.material_dir)
    stc.run()


if __name__ == "__main__":  # pragma: no cover
    main()

