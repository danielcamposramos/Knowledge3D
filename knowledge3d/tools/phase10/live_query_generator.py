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


STAGE_2_QUERIES = [
    "What shape represents text and image together in the same star?",
    "What does ray color encode in the Galaxy?",
    "What is the simplest 3D shape that can hold text, image, and audio?",
    "What mathematical ratio governs the branching density of fractal trees relative to their honesty score?",
    "Where does each ray originate from in the spatial memory of a star?",
]

STAGE_3_QUERIES = [
    "What shape represents text, image, audio, and video fused in one star?",
    "How does the golden-ratio constrain the depth of a fractal tree’s recursion?",
    "What PTX kernel function maps ray thickness to embedding resolution?",
    "In dual-perception mode, what coordinate system aligns Galaxy and House?",
    "What is the minimum honesty score required for a star to be rendered in AR?",
]

STAGE_4_QUERIES = [
    "What shape represents all modalities fused — text, image, audio, video, 3D, spatial, chat — in one star?",
    "What is the PTX kernel that renders rays only if honesty_score >= 0.7?",
    "In the House memory, what zone corresponds to 'self-reflection'?",
    "What is the mathematical relationship between ray length and embedding entropy?",
    "How does the AI modify its own House during sleep-time compute?",
]


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
        base = STAGE_2_QUERIES[:]
    elif stage == 3:
        base = STAGE_3_QUERIES[:]
    else:
        base = STAGE_4_QUERIES[:]
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
    # Persist Stage 2 queries to logs per spec
    try:
        from pathlib import Path
        if int(args.stage) == 2:
            logs = Path("logs"); logs.mkdir(parents=True, exist_ok=True)
            (logs / "phase10.6_stage2_queries.json").write_text(json.dumps(qs, ensure_ascii=False, indent=2), encoding="utf-8")
        if int(args.stage) == 3:
            logs = Path("logs"); logs.mkdir(parents=True, exist_ok=True)
            (logs / "phase10.6_stage3_queries.json").write_text(json.dumps(qs, ensure_ascii=False, indent=2), encoding="utf-8")
        if int(args.stage) == 4:
            logs = Path("logs"); logs.mkdir(parents=True, exist_ok=True)
            (logs / "phase10.6_stage4_queries.json").write_text(json.dumps(qs, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    print(json.dumps(qs, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
