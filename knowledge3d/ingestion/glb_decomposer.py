"""Decompose GLB assets into meaning-centric stars with procedural visual RPN."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import struct
from typing import Any

from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar, SurfaceForm

try:  # pragma: no cover - optional dependency for the ingestion tool
    from pygltflib import GLTF2
except Exception:  # pragma: no cover
    GLTF2 = None  # type: ignore


_COMPONENT_SIZES = {
    5121: 1,  # UNSIGNED_BYTE
    5123: 2,  # UNSIGNED_SHORT
    5125: 4,  # UNSIGNED_INT
    5126: 4,  # FLOAT
}

_ACCESSOR_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
}

_STRUCT_CODES = {
    5121: "B",
    5123: "H",
    5125: "I",
    5126: "f",
}


def _read_blob(gltf: Any, glb_path: Path) -> bytes:
    blob = gltf.binary_blob()
    if blob:
        return bytes(blob)
    if not gltf.buffers:
        return b""
    uri = str(gltf.buffers[0].uri or "").strip()
    if uri.startswith("data:"):
        _, encoded = uri.split(",", 1)
        return base64.b64decode(encoded)
    if not uri:
        return b""
    return (glb_path.parent / uri).read_bytes()


def _extract_accessor(gltf: Any, accessor_index: int, blob: bytes) -> list[tuple[float, ...]]:
    accessor = gltf.accessors[accessor_index]
    view = gltf.bufferViews[accessor.bufferView]
    stride = int(view.byteStride or 0)
    component_size = _COMPONENT_SIZES[int(accessor.componentType)]
    component_count = _ACCESSOR_COMPONENTS[str(accessor.type)]
    item_size = component_size * component_count
    read_stride = stride or item_size
    offset = int(view.byteOffset or 0) + int(accessor.byteOffset or 0)
    count = int(accessor.count or 0)
    struct_code = "<" + (_STRUCT_CODES[int(accessor.componentType)] * component_count)
    rows: list[tuple[float, ...]] = []
    for index in range(count):
        start = offset + index * read_stride
        chunk = blob[start:start + item_size]
        if len(chunk) < item_size:
            break
        rows.append(tuple(float(value) for value in struct.unpack(struct_code, chunk)))
    return rows


def _extract_indices(gltf: Any, accessor_index: int, blob: bytes) -> list[int]:
    rows = _extract_accessor(gltf, accessor_index, blob)
    return [int(row[0]) for row in rows]


def _bbox(vertices: list[tuple[float, float, float]]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not vertices:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    xs = [vertex[0] for vertex in vertices]
    ys = [vertex[1] for vertex in vertices]
    zs = [vertex[2] for vertex in vertices]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _detect_primitive(vertices: list[tuple[float, float, float]], indices: list[int]) -> str | None:
    tri_count = len(indices) // 3
    if len(vertices) in {8, 24} and tri_count == 12:
        return "cube"
    if len(vertices) >= 20 and tri_count >= 24:
        radii = []
        for x, y, z in vertices:
            radii.append((x * x + y * y + z * z) ** 0.5)
        if radii:
            avg = sum(radii) / float(len(radii))
            if avg > 1e-5 and max(abs(radius - avg) for radius in radii) < avg * 0.25:
                return "sphere"
    return None


def _primitive_program(
    *,
    primitive_kind: str | None,
    vertices: list[tuple[float, float, float]],
    indices: list[int],
) -> str:
    bbox_min, bbox_max = _bbox(vertices)
    size_x = bbox_max[0] - bbox_min[0]
    size_y = bbox_max[1] - bbox_min[1]
    size_z = bbox_max[2] - bbox_min[2]
    if primitive_kind == "cube":
        size = max(size_x, size_y, size_z)
        return f"{size:.6f} GEN_CUBE"
    if primitive_kind == "sphere":
        radius = max(size_x, size_y, size_z) / 2.0
        return f"{radius:.6f} 12 16 GEN_UV_SPHERE"
    tokens = ["MESH_BEGIN"]
    for x, y, z in vertices:
        tokens.append(f"{x:.6f}")
        tokens.append(f"{y:.6f}")
        tokens.append(f"{z:.6f}")
        tokens.append("VERTEX3")
    for cursor in range(0, len(indices), 3):
        face = indices[cursor:cursor + 3]
        if len(face) < 3:
            continue
        tokens.extend([str(int(face[0])), str(int(face[1])), str(int(face[2])), "TRI_FACE"])
    tokens.append("MESH_END")
    return " ".join(tokens)


def decompose_glb_to_stars(
    glb_path: str | Path,
    *,
    domain: str = "Workshop/Assets",
    language: str = "en",
) -> list[MeaningCentricStar]:
    if GLTF2 is None:
        raise RuntimeError("pygltflib is required for GLB decomposition")
    path = Path(glb_path)
    gltf = GLTF2().load(str(path))
    blob = _read_blob(gltf, path)
    stars: list[MeaningCentricStar] = []
    mesh_list = list(gltf.meshes or [])
    for mesh_index, mesh in enumerate(mesh_list):
        primitives = list(mesh.primitives or [])
        for primitive_index, primitive in enumerate(primitives):
            attributes = primitive.attributes or {}
            position_accessor = getattr(attributes, "POSITION", None)
            if position_accessor is None:
                continue
            vertices = [
                (float(row[0]), float(row[1]), float(row[2]))
                for row in _extract_accessor(gltf, int(position_accessor), blob)
                if len(row) >= 3
            ]
            indices = _extract_indices(gltf, int(primitive.indices), blob) if primitive.indices is not None else list(range(len(vertices)))
            primitive_kind = _detect_primitive(vertices, indices)
            visual_rpn = _primitive_program(
                primitive_kind=primitive_kind,
                vertices=vertices,
                indices=indices,
            )
            base_name = str(mesh.name or path.stem or f"mesh_{mesh_index}_{primitive_index}").strip()
            stars.append(
                MeaningCentricStar(
                    meaning_class="concept",
                    meaning_rpn="3D_CONSTRUCTION",
                    domain=domain,
                    visual_rpn=visual_rpn,
                    confidence=1,
                    polarity=1,
                    surface_forms={
                        language.lower(): SurfaceForm(
                            word_ref=base_name,
                            char_refs=[char for char in base_name],
                        )
                    },
                    meta_refs=[f"glb:{path.name}", f"mesh:{mesh_index}", f"primitive:{primitive_index}"],
                )
            )
    return stars


def write_stars_jsonl(stars: list[MeaningCentricStar], output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for star in stars:
            handle.write(json.dumps(star.to_dict(), ensure_ascii=True, separators=(",", ":")) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Decompose GLB assets into meaning-centric procedural stars.")
    parser.add_argument("input", type=Path, help="Input .glb path")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSONL output path")
    parser.add_argument("--domain", default="Workshop/Assets", help="Meaning star domain")
    parser.add_argument("--language", default="en", help="Surface-form language tag")
    args = parser.parse_args(argv)

    stars = decompose_glb_to_stars(args.input, domain=args.domain, language=args.language)
    if args.output is not None:
        write_stars_jsonl(stars, args.output)
    else:
        for star in stars:
            print(star.to_dict())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
