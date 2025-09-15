from __future__ import annotations

import argparse
from ...cranium.phase10.baby_curriculum import BabyCurriculum  # type: ignore


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Baby-style curriculum trainer")
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--modality", default="text")
    args = ap.parse_args()
    bc = BabyCurriculum()
    bc.stage = int(args.stage)
    res = bc.introduce_modality(str(args.modality))
    if res is not None:
        print(res)


if __name__ == "__main__":  # pragma: no cover
    main()

