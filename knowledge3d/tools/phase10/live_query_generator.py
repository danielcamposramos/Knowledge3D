from __future__ import annotations

import argparse
import json
from typing import List


FACTS = {
    # From README.md and SPAWN_CONTEXT.md
    "shape:text": "tetrahedron",
    "shape:image": "cube",
    "shape:audio": "octahedron",
    "shape:video": "icosahedron",
    "shape:mixed": "dodecahedron",
    "ray:length": "content",  # aka content size
    "ray:thickness": "resolution",
    "ray:color": "modality",
}


def default_queries_for_stage(stage: int, num: int) -> List[str]:
    base = []
    if stage == 1:
        base = [
            "What shape represents text in the Galaxy?",
            "What does ray thickness encode?",
            "What is the simplest 3D shape for text?",
            "What shape represents image in the Galaxy?",
            "What does ray length encode?",
        ]
    elif stage == 2:
        base = [
            "What shape represents text+image in the Galaxy?",
            "What does ray color encode?",
            "Which shape is used for video?",
            "Which shape is used for audio?",
            "What modality does cube represent?",
        ]
    else:
        base = [
            "In the Galaxy, how are modalities encoded?",
            "How do rays represent content and resolution?",
            "Map text, image, audio, video to shapes",
            "What does ray color represent?",
            "What shape is mixed modality?",
        ]
    # Truncate or repeat to match num
    out: List[str] = []
    while len(out) < num:
        for q in base:
            if len(out) >= num:
                break
            out.append(q)
    return out


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Generate live queries for co-creation")
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--num_queries", type=int, default=5)
    args = ap.parse_args()
    qs = default_queries_for_stage(int(args.stage), int(args.num_queries))
    print(json.dumps(qs, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()

