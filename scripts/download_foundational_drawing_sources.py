#!/usr/bin/env python3
"""Download foundational drawing-course source pages for ingestion prep.

This script intentionally downloads only public course/reference pages and
stores HTML snapshots plus a manifest in Knowledge3D.local.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SOURCES = [
    {
        "source_id": "pikuma_3d_graphics_course",
        "category": "course",
        "url": "https://pikuma.com/courses/learn-3d-computer-graphics-programming",
        "access": "mixed_paid",
    },
    {
        "source_id": "pikuma_course_catalog",
        "category": "course",
        "url": "https://pikuma.com/courses",
        "access": "mixed_paid",
    },
    {
        "source_id": "learnvern_blender_course",
        "category": "course",
        "url": "https://www.learnvern.com/blender-course",
        "access": "free_registration",
    },
    {
        "source_id": "learnvern_blender_interface",
        "category": "course",
        "url": "https://www.learnvern.com/blender-course/understanding-interface",
        "access": "free_registration",
    },
    {
        "source_id": "learnvern_blender_intro",
        "category": "course",
        "url": "https://www.learnvern.com/blender-course/introduction-to-blender",
        "access": "free_registration",
    },
    {
        "source_id": "blenderguru_donut_v4",
        "category": "course",
        "url": "https://www.blenderguru.com/tutorials/blender-4-beginner-donut-tutorial",
        "access": "free",
    },
    {
        "source_id": "blender_manual_curves",
        "category": "reference",
        "url": "https://docs.blender.org/manual/en/latest/modeling/curves/index.html",
        "access": "free",
    },
    {
        "source_id": "blender_manual_curve_structure",
        "category": "reference",
        "url": "https://docs.blender.org/manual/en/latest/modeling/curves/structure.html",
        "access": "free",
    },
    {
        "source_id": "blender_manual_mesh_transform",
        "category": "reference",
        "url": "https://docs.blender.org/manual/en/latest/modeling/meshes/editing/mesh/transform/index.html",
        "access": "free",
    },
    {
        "source_id": "pomax_bezier_primer",
        "category": "reference",
        "url": "https://pomax.github.io/bezierinfo/",
        "access": "free_open",
    },
    {
        "source_id": "scratchapixel_vector_math",
        "category": "reference",
        "url": "https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/math-operations-on-points-and-vectors.html",
        "access": "free_open",
    },
    {
        "source_id": "scratchapixel_matrix_ops",
        "category": "reference",
        "url": "https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/transforming-points-and-vectors.html",
        "access": "free_open",
    },
]


def _fetch(url: str) -> tuple[str, bytes]:
    req = Request(
        url,
        headers={
            "User-Agent": "Knowledge3D-Bootstrap/1.0 (+https://github.com/)",
        },
    )
    with urlopen(req, timeout=30) as resp:
        return str(resp.geturl()), resp.read()


def main() -> int:
    root = Path("/K3D/Knowledge3D.local/datasets/foundational_drawing_sources")
    raw = root / "raw_html"
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.jsonl"
    tsv_path = root / "sources.tsv"

    tsv_lines = ["source_id\tcategory\turl\taccess"]
    rows: list[dict[str, object]] = []

    for source in SOURCES:
        source_id = str(source["source_id"])
        out = raw / f"{source_id}.html"
        status = "ok"
        final_url = str(source["url"])
        data = b""
        error = ""

        try:
            final_url, data = _fetch(final_url)
            out.write_bytes(data)
        except (HTTPError, URLError, TimeoutError) as exc:
            status = "failed"
            error = str(exc)
            out.write_text("", encoding="utf-8")

        sha = hashlib.sha256(data).hexdigest() if data else hashlib.sha256(b"").hexdigest()
        row = {
            "source_id": source_id,
            "category": source["category"],
            "url": source["url"],
            "final_url": final_url,
            "access": source["access"],
            "status": status,
            "bytes": len(data),
            "sha256": sha,
            "error": error,
        }
        rows.append(row)
        tsv_lines.append(
            "\t".join(
                [
                    source_id,
                    str(source["category"]),
                    str(source["url"]),
                    str(source["access"]),
                ]
            )
        )

    tsv_path.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")
    with manifest_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")

    ok = sum(1 for row in rows if row["status"] == "ok")
    print(f"Downloaded {ok}/{len(rows)} sources to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

