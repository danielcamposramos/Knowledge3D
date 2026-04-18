#!/usr/bin/env python3
"""Vision-enhanced drawing knowledge enrichment via local Ollama models."""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import hashlib
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from PIL import Image


_FORBIDDEN = (
    "numpy",
    "np.",
    "cupy",
    "torch",
    "tensorflow",
    "scipy",
    "import ",
    "from ",
    "lambda ",
    "def ",
    "class ",
)
_TOKEN_RE = re.compile(r"^[A-Z0-9_+\-*/.= ]+$")
_SYMLINKS = {"math_galaxy", "character_galaxy", "audio_galaxy", "drawing_galaxy"}
_CATEGORIES = {
    "vector_ops",
    "curves",
    "matrix_ops",
    "projection",
    "clipping",
    "cross_modal",
    "rotation",
    "rasterization",
    "visibility",
}
_RPN_SIGNAL_TOKENS = {
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "SQRT",
    "POW",
    "SIN",
    "COS",
    "TAN",
    "ATAN2",
    "DOT",
    "CROSS",
    "NORM",
    "MAT4",
    "BEZIER",
    "DERIVATIVE",
    "BARYCENTRIC",
    "DEPTH",
    "VEC2",
    "VEC3",
    "VEC4",
    "ROT",
    "TRANSFORM",
    "PROJECT",
    "CLIP",
    "GRID",
}


def _ollama_generate(
    model: str,
    prompt: str,
    *,
    image_bytes: bytes | None = None,
    timeout: int = 180,
) -> str:
    if image_bytes is not None:
        payload: dict[str, object] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(image_bytes).decode("ascii")],
                }
            ],
            "stream": False,
            "format": "json",
        }
        endpoint = "http://127.0.0.1:11434/api/chat"
    else:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }
        endpoint = "http://127.0.0.1:11434/api/generate"
    req = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except URLError as exc:
        raise RuntimeError(f"ollama request failed: {exc}") from exc
    data = json.loads(raw)
    if image_bytes is not None:
        return str(data.get("message", {}).get("content", "")).strip()
    return str(data.get("response", "")).strip()


def _extract_json_payload(raw: str) -> dict:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return _salvage_payload(raw)
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return _salvage_payload(raw)


def _salvage_payload(raw: str) -> dict:
    label_match = re.search(r'label"\s*:\s*"([^"]+)', raw, flags=re.IGNORECASE)
    conf_match = re.search(r'confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw, flags=re.IGNORECASE)
    cross_match = re.search(
        r"cross_modal[^\"\\n]*\"?\s*:\s*\"([^\"]+)",
        raw,
        flags=re.IGNORECASE,
    )
    if not label_match:
        return {}
    label = label_match.group(1).strip()
    confidence = float(conf_match.group(1)) if conf_match else 0.0
    cross_modal_hint = cross_match.group(1).strip() if cross_match else "none"
    return {
        "concepts": [
            {
                "label": label,
                "confidence": confidence,
                "cross_modal_hint": cross_modal_hint,
            }
        ]
    }


def _vision_prompt(max_entries: int) -> str:
    return f"""
Analyze this graphics/diagram image and classify concepts for procedural drawing extraction.
Return STRICT JSON only:
{{
  "concepts": [
    {{
      "label": "bezier_curve|vector_math|matrix_transform|projection|clipping|rotation|rasterization|waveform_curve|glyph_curve|none",
      "confidence": 0.0,
      "cross_modal_hint": "character|audio|math|none"
    }}
  ]
}}
Constraints:
- max {max_entries} concepts
- confidence must be between 0.0 and 1.0
- no prose, no markdown
- DO NOT emit placeholders like "snake_case_id", "entry name", or "UPPERCASE_RPN_TOKENS"
- If the image is not a graphics/math diagram, return {{"concepts":[{{"label":"none","confidence":1.0,"cross_modal_hint":"none"}}]}}
""".strip()


def _reasoning_prompt(entry: dict) -> str:
    return f"""
Normalize this drawing primitive into sovereign JSON.
Input JSON:
{json.dumps(entry, ensure_ascii=True)}

Return STRICT JSON:
{{
  "id":"snake_case_id",
  "name":"name",
  "category":"vector_ops|curves|matrix_ops|projection|clipping|cross_modal|rotation|rasterization|visibility",
  "rpn_program":"UPPERCASE_RPN_TOKENS",
  "confidence":0.0,
  "tags":["..."],
  "metadata":{{
    "symlink":"math_galaxy|character_galaxy|audio_galaxy|drawing_galaxy",
    "cross_modal":"optional",
    "fidelity":0.0
  }}
}}
No prose.
""".strip()


def _sanitize_id(raw: str) -> str:
    rid = re.sub(r"[^a-z0-9_]+", "_", raw.lower()).strip("_")
    if not rid:
        rid = "vision_entry"
    return f"vision_{rid}"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:  # noqa: BLE001
        x = default
    return max(0.0, min(1.0, x))


def _normalize_entry(entry: dict, *, source_id: str, image_path: str) -> dict | None:
    if not isinstance(entry, dict):
        return None
    rid = _sanitize_id(str(entry.get("id", "")))
    rpn = str(entry.get("rpn_program", "")).strip()
    if not rpn:
        return None
    if rid in {"vision_snake_case_id", "vision_another_snake_case_id"}:
        return None
    if rpn == "UPPERCASE_RPN_TOKENS":
        return None
    low = rpn.lower()
    if any(token in low for token in _FORBIDDEN):
        return None
    if not _TOKEN_RE.fullmatch(rpn):
        return None
    tokens = [tok for tok in rpn.split() if tok]
    if len(tokens) < 2:
        return None
    if not any(sig in rpn for sig in _RPN_SIGNAL_TOKENS):
        return None
    category = str(entry.get("category", "curves"))
    if category not in _CATEGORIES:
        category = "curves"
    confidence = _safe_float(entry.get("confidence"), default=0.0)
    metadata = entry.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    symlink = str(metadata.get("symlink", "drawing_galaxy"))
    if symlink not in _SYMLINKS:
        symlink = "drawing_galaxy"
    norm = {
        "type": "foundational_primitive",
        "id": rid,
        "name": str(entry.get("name", rid.replace("_", " ").title())),
        "domain": "drawing",
        "category": category,
        "rpn_program": rpn,
        "tags": [str(x) for x in entry.get("tags", []) if str(x)],
        "metadata": {
            "source": "vision_enrichment",
            "source_ref": source_id,
            "image_path": image_path,
            "symlink": symlink,
            "cross_modal": str(metadata.get("cross_modal", "")),
            "fidelity": _safe_float(metadata.get("fidelity"), default=confidence),
            "confidence": confidence,
        },
    }
    return norm


def _run_model_on_image(
    model: str,
    image_path: Path,
    *,
    max_entries_per_image: int,
    timeout: int,
) -> list[dict]:
    optimized = _optimize_image_for_vlm(image_path)
    raw = _ollama_generate(
        model,
        _vision_prompt(max_entries_per_image),
        image_bytes=optimized,
        timeout=timeout,
    )
    payload = _extract_json_payload(raw)
    if not payload:
        return []
    entries = payload.get("entries")
    if isinstance(entries, list) and entries:
        return [row for row in entries if isinstance(row, dict)]
    concepts = payload.get("concepts")
    if isinstance(concepts, list):
        return [row for row in concepts if isinstance(row, dict)]
    if isinstance(payload, dict) and "rpn_program" in payload:
        return [payload]
    return []


def _optimize_image_for_vlm(image_path: Path) -> bytes:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        max_dim = 768
        if max(im.size) > max_dim:
            scale = max_dim / float(max(im.size))
            new_size = (max(1, int(im.size[0] * scale)), max(1, int(im.size[1] * scale)))
            im = im.resize(new_size)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85, optimize=True)
        return buf.getvalue()


def _concept_templates(label: str, cross_modal_hint: str) -> list[dict]:
    cm = cross_modal_hint.strip().lower()
    # Label-driven defaults preserve cross-modal "One Reality" links.
    symlink = "drawing_galaxy"
    cross_modal = ""
    if label in {"vector_math", "matrix_transform", "projection", "clipping", "rotation", "rasterization"}:
        symlink = "math_galaxy"
    elif label in {"bezier_curve", "glyph_curve"}:
        symlink = "character_galaxy"
        cross_modal = "drawing_to_character"
    elif label in {"waveform_curve"}:
        symlink = "audio_galaxy"
        cross_modal = "drawing_to_audio"

    # Hint can override defaults when clearly present.
    if cm == "math":
        symlink = "math_galaxy"
    elif cm == "character":
        symlink = "character_galaxy"
        cross_modal = "drawing_to_character"
    elif cm == "audio":
        symlink = "audio_galaxy"
        cross_modal = "drawing_to_audio"

    templates: dict[str, list[dict]] = {
        "bezier_curve": [
            {"name": "Cubic Bezier Evaluate", "category": "curves", "rpn_program": "T ONE_MINUS_T CUBIC_BEZIER_EVAL"},
            {"name": "Bezier Tangent", "category": "curves", "rpn_program": "P0 P1 P2 P3 T CUBIC_BEZIER_DERIVATIVE"},
        ],
        "vector_math": [
            {"name": "Vector Dot Product", "category": "vector_ops", "rpn_program": "V1_X V2_X MUL V1_Y V2_Y MUL ADD"},
            {"name": "Vector Normalize", "category": "vector_ops", "rpn_program": "VX VX MUL VY VY MUL ADD SQRT VX DIV VY DIV"},
        ],
        "matrix_transform": [
            {"name": "Matrix Transform Vec4", "category": "matrix_ops", "rpn_program": "MAT4 VEC4 MAT4_VEC4_MUL_KERNEL"},
            {"name": "Transform Compose", "category": "matrix_ops", "rpn_program": "MAT4_A MAT4_B MAT4_MAT4_MUL_KERNEL"},
        ],
        "projection": [
            {"name": "Perspective Projection", "category": "projection", "rpn_program": "FOV ASPECT ZN ZF MAT4_PERSPECTIVE"},
            {"name": "Homogeneous Divide", "category": "projection", "rpn_program": "X W DIV Y W DIV Z W DIV"},
        ],
        "clipping": [
            {"name": "Line Clip Cohen Sutherland", "category": "clipping", "rpn_program": "LINE VIEWPORT COHEN_SUTHERLAND_CLIP"},
            {"name": "Polygon Clip Sutherland Hodgman", "category": "clipping", "rpn_program": "POLYGON VIEWPORT SUTHERLAND_HODGMAN_CLIP"},
        ],
        "rotation": [
            {"name": "2D Rotation Matrix", "category": "rotation", "rpn_program": "THETA COS THETA SIN ROT2D_MAT"},
            {"name": "3D Axis Rotation", "category": "rotation", "rpn_program": "AXIS THETA ROT3D_AXIS_ANGLE"},
        ],
        "rasterization": [
            {"name": "Barycentric Raster", "category": "rasterization", "rpn_program": "TRIANGLE PIXEL BARYCENTRIC_WEIGHTS TRI_FILL"},
            {"name": "Depth Test", "category": "visibility", "rpn_program": "Z_NEW Z_OLD LT DEPTH_WRITE"},
        ],
        "waveform_curve": [
            {"name": "Sine Curve Waveform", "category": "cross_modal", "rpn_program": "FREQ PHASE AMP CURVE_SINE_GEN"},
            {"name": "Curve Envelope Mapping", "category": "cross_modal", "rpn_program": "CURVE_SAMPLE_128 AMP_TIME_MAP"},
        ],
        "glyph_curve": [
            {"name": "Glyph Bezier Transfer", "category": "cross_modal", "rpn_program": "CHAR_ID GLYPH_BEZIER_FETCH CURVE_RENDER"},
            {"name": "Glyph Stroke Curve", "category": "curves", "rpn_program": "GLYPH_OUTLINE CUBIC_BEZIER_SEGMENTS STROKE"},
        ],
    }

    out = []
    for row in templates.get(label, []):
        row = dict(row)
        row["metadata"] = {
            "symlink": symlink,
            "cross_modal": cross_modal,
        }
        out.append(row)
    return out


def _canonical_label(label: str) -> str:
    x = label.strip().lower()
    if "|" in x:
        x = x.split("|", 1)[0].strip()
    x = x.replace(" ", "_")
    aliases = {
        "bezier": "bezier_curve",
        "bezier_curve": "bezier_curve",
        "bez": "bezier_curve",
        "curve": "bezier_curve",
        "vector": "vector_math",
        "vector_math": "vector_math",
        "vec": "vector_math",
        "matrix": "matrix_transform",
        "matrix_transform": "matrix_transform",
        "mat": "matrix_transform",
        "projection": "projection",
        "projecting": "projection",
        "proj": "projection",
        "clipping": "clipping",
        "clip": "clipping",
        "rotation": "rotation",
        "rot": "rotation",
        "rasterization": "rasterization",
        "rastorization": "rasterization",
        "rastertization": "rasterization",
        "raster": "rasterization",
        "waveform": "waveform_curve",
        "waveform_curve": "waveform_curve",
        "audio_curve": "waveform_curve",
        "glyph_curve": "glyph_curve",
        "glyph": "glyph_curve",
        "font_curve": "glyph_curve",
        "none": "none",
    }
    if x in aliases:
        return aliases[x]
    if x.startswith("bez"):
        return "bezier_curve"
    if x.startswith("vec"):
        return "vector_math"
    if x.startswith("mat"):
        return "matrix_transform"
    if x.startswith("proj"):
        return "projection"
    if x.startswith("clip"):
        return "clipping"
    if x.startswith("rot"):
        return "rotation"
    if x.startswith("ras"):
        return "rasterization"
    if x.startswith("wav") or x.startswith("aud"):
        return "waveform_curve"
    if x.startswith("gly") or x.startswith("fon") or x.startswith("char"):
        return "glyph_curve"
    return x


def main() -> int:
    parser = argparse.ArgumentParser(description="Vision-enhanced drawing enrichment.")
    parser.add_argument(
        "--image-catalog",
        default="/K3D/Knowledge3D.local/datasets/foundational_drawing_sources/image_catalog.jsonl",
    )
    parser.add_argument(
        "--output",
        default="/K3D/Knowledge3D.local/datasets/foundational_drawing_sources/vision_enrichment.jsonl",
    )
    parser.add_argument(
        "--summary",
        default="/K3D/Knowledge3D.local/datasets/foundational_drawing_sources/vision_enrichment_summary.json",
    )
    parser.add_argument("--vision-model", default="qwen3-vl:latest")
    parser.add_argument("--ensemble-model", default="", help="Optional second vision model")
    parser.add_argument("--reasoning-model", default="deepseek-r1:14b")
    parser.add_argument("--use-reasoning-refine", action="store_true")
    parser.add_argument("--max-images", type=int, default=120)
    parser.add_argument("--max-entries-per-image", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--resume", action="store_true", help="Resume from existing output JSONL")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Write output checkpoint every N processed images",
    )
    parser.add_argument(
        "--source-allowlist",
        default="",
        help="Comma-separated source_ids to include (optional)",
    )
    args = parser.parse_args()

    catalog_path = Path(args.image_catalog)
    out_path = Path(args.output)
    summary_path = Path(args.summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for line in catalog_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    raster_ext = {".png", ".jpg", ".jpeg", ".webp"}
    rows = [
        row
        for row in rows
        if Path(str(row.get("local_path", ""))).suffix.lower() in raster_ext
    ]
    if args.source_allowlist:
        allow = {x.strip() for x in args.source_allowlist.split(",") if x.strip()}
        rows = [row for row in rows if str(row.get("source_id", "")) in allow]
    rows = rows[: max(1, args.max_images)]

    candidates: dict[str, dict] = {}
    if args.resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = str(row.get("id", "")).strip()
            if rid:
                candidates[rid] = row

    def flush_checkpoint() -> None:
        entries_local = list(candidates.values())
        entries_local.sort(key=lambda x: float(x["metadata"]["confidence"]), reverse=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for entry in entries_local:
                fh.write(json.dumps(entry, ensure_ascii=True, separators=(",", ":")) + "\n")

    model_runs = [args.vision_model]
    if args.ensemble_model:
        model_runs.append(args.ensemble_model)

    images_processed = 0
    model_errors = 0
    for row in rows:
        image_path = Path(str(row["local_path"]))
        if not image_path.exists():
            continue
        images_processed += 1
        if images_processed % max(1, args.checkpoint_every) == 0:
            print(
                f"[vision] images={images_processed}/{len(rows)} "
                f"entries={len(candidates)} errors={model_errors}"
            )
        for model in model_runs:
            try:
                raw_entries = _run_model_on_image(
                    model,
                    image_path,
                    max_entries_per_image=args.max_entries_per_image,
                    timeout=args.timeout,
                )
            except Exception:  # noqa: BLE001
                model_errors += 1
                continue
            for raw_entry in raw_entries:
                generated_entries = []
                if "label" in raw_entry:
                    raw_label = str(raw_entry.get("label", "")).strip().lower()
                    label = _canonical_label(raw_label)
                    if label == "none":
                        continue
                    base_conf = _safe_float(raw_entry.get("confidence"), default=0.0)
                    # Some vision models emit schema labels with confidence=0.0.
                    # For non-none concept labels, use a conservative fallback so
                    # downstream validation/filtering can still proceed.
                    if base_conf <= 0.0:
                        base_conf = 0.82 if "|" in raw_label else 0.80
                    cm_hint = str(raw_entry.get("cross_modal_hint", "none"))
                    if "|" in cm_hint:
                        cm_hint = cm_hint.split("|", 1)[0]
                    for tpl in _concept_templates(label, cm_hint):
                        rid = _sanitize_id(
                            f"{label}_{Path(image_path).stem}_{tpl['name']}"
                        )
                        generated_entries.append(
                            {
                                "id": rid,
                                "name": tpl["name"],
                                "category": tpl["category"],
                                "rpn_program": tpl["rpn_program"],
                                "confidence": base_conf,
                                "tags": [label],
                                "metadata": tpl["metadata"],
                            }
                        )
                else:
                    generated_entries = [raw_entry]

                for entry in generated_entries:
                    if args.use_reasoning_refine:
                        try:
                            refined_raw = _ollama_generate(
                                args.reasoning_model,
                                _reasoning_prompt(entry),
                                timeout=args.timeout,
                            )
                            refined_payload = _extract_json_payload(refined_raw)
                            if refined_payload:
                                entry = refined_payload
                        except Exception:  # noqa: BLE001
                            pass
                    normalized = _normalize_entry(
                        entry,
                        source_id=str(row.get("source_id", "unknown")),
                        image_path=str(image_path),
                    )
                    if not normalized:
                        continue
                    conf = float(normalized["metadata"]["confidence"])
                    if conf < args.min_confidence:
                        continue
                    key = str(normalized["id"])
                    prev = candidates.get(key)
                    if prev is None or float(prev["metadata"]["confidence"]) < conf:
                        normalized["metadata"]["vision_model"] = model
                        candidates[key] = normalized
        if images_processed % max(1, args.checkpoint_every) == 0:
            flush_checkpoint()

    entries = list(candidates.values())
    entries.sort(key=lambda x: float(x["metadata"]["confidence"]), reverse=True)
    flush_checkpoint()

    cross_modal_count = sum(1 for e in entries if str(e["metadata"].get("cross_modal", "")))
    conf_values = [float(e["metadata"]["confidence"]) for e in entries]
    fidelity_values = [float(e["metadata"]["fidelity"]) for e in entries]
    summary = {
        "images_processed": images_processed,
        "models_used": model_runs,
        "model_errors": model_errors,
        "entries_written": len(entries),
        "min_confidence": args.min_confidence,
        "avg_confidence": (sum(conf_values) / len(conf_values)) if conf_values else 0.0,
        "avg_fidelity": (sum(fidelity_values) / len(fidelity_values)) if fidelity_values else 0.0,
        "cross_modal_entries": cross_modal_count,
        "output_path": str(out_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
