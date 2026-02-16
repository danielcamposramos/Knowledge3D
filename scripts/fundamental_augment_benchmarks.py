#!/usr/bin/env python3
"""Fundamental benchmark augmentation for Galaxy knowledge construction.

PURPOSE:
  Build foundational benchmark-derived payloads for Knowledgeverse population.
  This is augmentation-time construction, not inference-time execution.

WHEN TO USE:
  - Initial/foundational world bootstrap.
  - Knowledge expansion after new benchmark sources.
  - Re-augmentation after Ollama model upgrades.

NOT FOR:
  - PTX hot-path inference.
  - Runtime daemon specialist solving.

ARCHITECTURE:
  - Ollama is central in augmentation pipeline (mandatory by default).
  - Outputs JSONL payload rows consumed by `scripts/fundamental_ingest_payloads.py`.
  - Ingestion phase applies symlink compression (form->meaning refs).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from knowledge3d.ingestion.ollama_manager import OllamaModelManager

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-']{1,31}")

STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "into", "what", "which", "when", "where", "how", "why",
    "your", "their", "there", "about", "would", "could", "should", "after", "before", "while", "under", "over", "have",
    "has", "had", "were", "was", "are", "is", "be", "been", "being", "also", "than", "then", "into", "onto", "just",
    "more", "most", "some", "such", "only", "very", "much", "many", "each", "other", "another", "through", "across",
    "between", "because", "therefore", "however", "whereas", "option", "options", "answer", "question", "choose", "best",
}

MATH_KEYWORDS = {
    "algebra": "X Y ADD",
    "equation": "A B EQ",
    "solve": "X ISOLATE",
    "factor": "EXPR FACTOR",
    "polynomial": "POLY ROOTS",
    "geometry": "A B DIST",
    "triangle": "A B C TRIANGLE_AREA",
    "circle": "R R MUL PI MUL",
    "angle": "A B C ANGLE",
    "calculus": "F X DIFF",
    "derivative": "F X DIFF",
    "integral": "F X INTEGRATE",
    "limit": "F X LIMIT",
    "probability": "EVENT_SPACE COUNT DIV",
    "combinatorics": "N K COMB",
    "number": "N FACT",
    "mod": "A B MOD",
}

REALITY_KEYWORDS = {
    "physics": "M A MUL",
    "force": "M A MUL",
    "velocity": "DX DT DIV",
    "acceleration": "DV DT DIV",
    "energy": "0.5 M MUL V V MUL MUL",
    "kinetic": "0.5 M MUL V V MUL MUL",
    "power": "WORK TIME DIV",
    "pressure": "F A DIV",
    "density": "M VOLUME DIV",
    "biology": "STATE TRANSITION APPLY",
    "chemical": "REACTION_RATE C1 C2 MUL",
    "temperature": "Q C MUL DIV",
}

ARC_PATTERN_TEMPLATES = {
    "identity": "GRID COPY",
    "rotate_or_transpose": "GRID ROTATE_90",
    "mirror": "GRID MIRROR_H",
    "color_remap": "GRID MAP_COLORS",
    "shape_resize": "GRID RESIZE",
    "object_count_change": "GRID COMPONENT_REWRITE",
    "compositional": "GRID ROTATE_90 MAP_COLORS",
}


@dataclass
class AugmentStats:
    processed: int = 0
    entries: int = 0


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _safe_read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _flatten_grid(grid: list[list[Any]]) -> list[int]:
    out: list[int] = []
    for row in grid:
        if not isinstance(row, list):
            continue
        for cell in row:
            try:
                out.append(int(cell))
            except Exception:
                out.append(0)
    return out


def _grid_shape(grid: list[list[Any]]) -> tuple[int, int]:
    if not isinstance(grid, list):
        return (0, 0)
    h = len(grid)
    w = len(grid[0]) if h > 0 and isinstance(grid[0], list) else 0
    return (h, w)


def _palette(grid: list[list[Any]]) -> set[int]:
    return set(_flatten_grid(grid))


def _count_components(grid: list[list[Any]]) -> int:
    h, w = _grid_shape(grid)
    if h == 0 or w == 0:
        return 0
    visited = [[False for _ in range(w)] for _ in range(h)]

    def cell(x: int, y: int) -> int:
        try:
            return int(grid[y][x])
        except Exception:
            return 0

    count = 0
    for y in range(h):
        for x in range(w):
            if visited[y][x] or cell(x, y) == 0:
                continue
            count += 1
            stack = [(x, y)]
            visited[y][x] = True
            while stack:
                cx, cy = stack.pop()
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if nx < 0 or ny < 0 or nx >= w or ny >= h:
                        continue
                    if visited[ny][nx] or cell(nx, ny) == 0:
                        continue
                    visited[ny][nx] = True
                    stack.append((nx, ny))
    return count


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for m in WORD_RE.finditer(text.lower()):
        tok = m.group(0)
        if tok in STOPWORDS or len(tok) < 2:
            continue
        tokens.append(tok)
    return tokens


def _classify_math_family(text: str) -> str:
    lowered = text.lower()
    if any(k in lowered for k in ("triangle", "circle", "angle", "geometry", "perimeter", "area")):
        return "geometry"
    if any(k in lowered for k in ("derivative", "integral", "limit", "calculus")):
        return "calculus"
    if any(k in lowered for k in ("probability", "combinatorics", "arrangements", "choose")):
        return "combinatorics"
    if any(k in lowered for k in ("prime", "divisor", "mod", "number theory")):
        return "number_theory"
    return "algebra"


def _rpn_from_keywords(text: str, mapping: dict[str, str], default: str) -> str:
    lowered = text.lower()
    for key, rpn in mapping.items():
        if key in lowered:
            return rpn
    return default


def _iter_arc_tasks(dataset_root: Path, max_tasks: int) -> list[dict[str, Any]]:
    candidates = [
        dataset_root / "exams" / "arc-src" / "data" / "evaluation",
        dataset_root / "arc_agi" / "ARC-AGI-master" / "data" / "evaluation",
        dataset_root / "arc_agi_2" / "evaluation",
        Path("/K3D/Knowledge3D.local/datasets/exams/arc-src/data/evaluation"),
        Path("/K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/evaluation"),
        Path("/K3D/Knowledge3D.local/datasets/arc_agi_2/evaluation"),
        Path("../Knowledge3D.local/datasets/exams/arc-src/data/evaluation"),
    ]
    files: list[Path] = []
    for root in candidates:
        if root.exists():
            files = sorted(root.glob("*.json"))
            if files:
                break

    out: list[dict[str, Any]] = []
    if max_tasks <= 0:
        max_tasks = len(files)
    for path in files[:max_tasks]:
        payload = _safe_read_json(path)
        if not isinstance(payload, dict):
            continue
        train = payload.get("train")
        test = payload.get("test")
        if not isinstance(train, list) or not isinstance(test, list):
            continue
        out.append({"id": path.stem, "train": train, "test": test})
    return out


def _infer_arc_family(task: dict[str, Any]) -> str:
    train = task.get("train", [])
    if not isinstance(train, list) or not train:
        return "compositional"

    in_shapes: list[tuple[int, int]] = []
    out_shapes: list[tuple[int, int]] = []
    palette_change = False
    object_delta = False

    for ex in train:
        if not isinstance(ex, dict):
            continue
        inp = ex.get("input")
        out = ex.get("output")
        if not isinstance(inp, list) or not isinstance(out, list):
            continue
        in_shape = _grid_shape(inp)
        out_shape = _grid_shape(out)
        in_shapes.append(in_shape)
        out_shapes.append(out_shape)
        if _palette(inp) != _palette(out):
            palette_change = True
        if _count_components(inp) != _count_components(out):
            object_delta = True

    if in_shapes and out_shapes and all(a == b for a, b in zip(in_shapes, out_shapes)):
        if palette_change and object_delta:
            return "compositional"
        if palette_change:
            return "color_remap"
        return "mirror"

    if in_shapes and out_shapes and any((a[0], a[1]) == (b[1], b[0]) for a, b in zip(in_shapes, out_shapes)):
        return "rotate_or_transpose"

    if object_delta:
        return "object_count_change"

    if in_shapes and out_shapes and any(a != b for a, b in zip(in_shapes, out_shapes)):
        return "shape_resize"

    return "identity"


def _build_arc_entries(task: dict[str, Any], include_ollama_hint: str | None) -> list[dict[str, Any]]:
    task_id = str(task.get("id", "arc_task"))
    family = _infer_arc_family(task)
    train = task.get("train", [])

    palettes_in: set[int] = set()
    palettes_out: set[int] = set()
    shapes_in: list[tuple[int, int]] = []
    shapes_out: list[tuple[int, int]] = []
    objects_in: list[int] = []
    objects_out: list[int] = []

    for ex in train:
        if not isinstance(ex, dict):
            continue
        inp = ex.get("input")
        out = ex.get("output")
        if not isinstance(inp, list) or not isinstance(out, list):
            continue
        palettes_in.update(_palette(inp))
        palettes_out.update(_palette(out))
        shapes_in.append(_grid_shape(inp))
        shapes_out.append(_grid_shape(out))
        objects_in.append(_count_components(inp))
        objects_out.append(_count_components(out))

    rpn = ARC_PATTERN_TEMPLATES.get(family, ARC_PATTERN_TEMPLATES["compositional"])
    base_meta = {
        "source": "benchmark_augmentation_arc",
        "task_id": task_id,
        "pattern_family": family,
        "palette_in": sorted(palettes_in),
        "palette_out": sorted(palettes_out),
        "shape_in": [list(s) for s in shapes_in[:4]],
        "shape_out": [list(s) for s in shapes_out[:4]],
        "object_count_in": objects_in[:4],
        "object_count_out": objects_out[:4],
        "cross_modal": ["drawing", "grammar", "math", "3d_objects", "reality"],
        "confidence": 0.88,
    }
    if include_ollama_hint:
        base_meta["ollama_hint"] = include_ollama_hint

    token = _sha(task_id + family)
    return [
        {
            "galaxy": "Grammar",
            "entry": {
                "id": f"arc_rule_{task_id}_{token}",
                "name": f"ARC rule {task_id}",
                "domain": "grammar",
                "category": "arc_aug_rule",
                "rpn_program": rpn,
                "metadata": {**base_meta, "symlink": "drawing_galaxy"},
            },
        },
        {
            "galaxy": "Drawing",
            "entry": {
                "id": f"arc_sig_{task_id}_{token}",
                "name": f"ARC signature {task_id}",
                "domain": "drawing",
                "category": "arc_visual_signature",
                "rpn_program": "GRID FEATURE_EXTRACT",
                "metadata": {**base_meta, "symlink": "grammar_galaxy"},
            },
        },
        {
            "galaxy": "Math",
            "entry": {
                "id": f"arc_math_{task_id}_{token}",
                "name": f"ARC shape math {task_id}",
                "domain": "math",
                "category": "arc_shape_constraints",
                "rpn_program": "H_IN W_IN H_OUT W_OUT DELTA_SHAPE",
                "metadata": {**base_meta, "symlink": "drawing_galaxy"},
            },
        },
        {
            "galaxy": "3DObjects",
            "entry": {
                "id": f"arc_spatial_{task_id}_{token}",
                "name": f"ARC spatial transform {task_id}",
                "domain": "3d_objects",
                "category": "arc_spatial_transform",
                "rpn_program": "GRID_TO_MESH TRANSFORM_APPLY MESH_TO_GRID",
                "metadata": {**base_meta, "symlink": "drawing_galaxy"},
            },
        },
    ]


def _iter_math_records(dataset_root: Path, max_problems: int) -> list[dict[str, Any]]:
    roots = [
        dataset_root / "math_competitions",
        Path("/K3D/Knowledge3D.local/datasets/math_competitions"),
        Path("../Knowledge3D.local/datasets/math_competitions"),
    ]
    files: list[tuple[str, Path]] = []
    for root in roots:
        files.extend(
            [
                ("AMC", root / "amc_problems.json"),
                ("AIME", root / "aime_problems.json"),
                ("IMO", root / "imo_problems.json"),
            ]
        )
    out: list[dict[str, Any]] = []
    if max_problems <= 0:
        max_problems = 10**9

    for competition, file_path in files:
        if not file_path.exists():
            continue
        payload = _safe_read_json(file_path)
        records = payload if isinstance(payload, list) else []
        for idx, row in enumerate(records):
            if len(out) >= max_problems:
                return out
            if not isinstance(row, dict):
                continue
            text = str(row.get("problem_text") or row.get("question") or "").strip()
            answer = str(row.get("answer", "")).strip()
            if not text:
                continue
            out.append(
                {
                    "id": str(row.get("id") or f"{competition.lower()}_{idx}"),
                    "competition": competition,
                    "problem_text": text,
                    "answer": answer,
                }
            )
    return out


def _build_math_entries(row: dict[str, Any], include_ollama_hint: str | None) -> list[dict[str, Any]]:
    pid = str(row.get("id", "math_problem"))
    text = str(row.get("problem_text", ""))
    answer = str(row.get("answer", "")).strip()
    family = _classify_math_family(text)
    rpn = _rpn_from_keywords(text, MATH_KEYWORDS, "A B EQ SOLVE")

    meta = {
        "source": "benchmark_augmentation_math",
        "problem_id": pid,
        "competition": str(row.get("competition", "unknown")),
        "family": family,
        "confidence": 0.9,
        "cross_modal": ["math", "grammar", "word"],
        "supervision_answer": answer or None,
    }
    if include_ollama_hint:
        meta["ollama_hint"] = include_ollama_hint

    token = _sha(pid + family)
    return [
        {
            "galaxy": "Math",
            "entry": {
                "id": f"math_tpl_{pid}_{token}",
                "name": f"{family} template {pid}",
                "domain": "math",
                "category": "benchmark_template",
                "rpn_program": rpn,
                "metadata": {**meta, "symlink": "grammar_galaxy"},
            },
        },
        {
            "galaxy": "Grammar",
            "entry": {
                "id": f"math_rule_{pid}_{token}",
                "name": f"Math reasoning rule {pid}",
                "domain": "grammar",
                "category": "math_reasoning_rule",
                "rpn_program": "QUERY PARSE_TEMPLATE APPLY_SOLVER",
                "metadata": {**meta, "symlink": "math_galaxy"},
            },
        },
    ]


def _iter_lhe_rows(dataset_root: Path, max_questions: int) -> list[dict[str, Any]]:
    candidates = [
        dataset_root / "last_humanity_exam" / "questions.json",
        dataset_root / "exams" / "hle-src" / "questions.json",
        Path("/K3D/Knowledge3D.local/datasets/last_humanity_exam/questions.json"),
        Path("/K3D/Knowledge3D.local/datasets/exams/hle-src/questions.json"),
        Path("../Knowledge3D.local/datasets/last_humanity_exam/questions.json"),
        Path("../Knowledge3D.local/datasets/exams/hle-src/questions.json"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        payload = _safe_read_json(path)
        if isinstance(payload, dict):
            records = payload.get("questions", [])
        else:
            records = payload if isinstance(payload, list) else []
        if not isinstance(records, list):
            continue
        out: list[dict[str, Any]] = []
        limit = max_questions if max_questions > 0 else len(records)
        for idx, row in enumerate(records[:limit]):
            if not isinstance(row, dict):
                continue
            text = str(row.get("question_text") or row.get("question") or "").strip()
            if not text:
                continue
            out.append(
                {
                    "id": str(row.get("id") or f"lhe_{idx}"),
                    "question_text": text,
                    "domain": str(row.get("domain") or row.get("subject") or "multi"),
                    "options": row.get("options") if isinstance(row.get("options"), list) else [],
                    "correct_answer": str(row.get("correct_answer") or row.get("answer") or "").strip(),
                }
            )
        return out
    return []


def _iter_mmlu_rows(dataset_root: Path, max_questions: int) -> list[dict[str, Any]]:
    roots = [
        Path("/K3D/K3D_llama_cpp/datasets/MMLU/data"),
        dataset_root / "MMLU" / "data",
        dataset_root / "global_benchmarks" / "mmlu" / "data",
    ]
    split_root = None
    for root in roots:
        test = root / "test"
        if test.exists():
            split_root = test
            break
    if split_root is None:
        return []

    rows: list[dict[str, Any]] = []
    limit = max_questions if max_questions > 0 else 10**9
    for csv_file in sorted(split_root.glob("*_test.csv")):
        subject = csv_file.stem.replace("_test", "")
        with csv_file.open("r", encoding="utf-8", errors="ignore") as handle:
            reader = csv.reader(handle)
            for idx, row in enumerate(reader):
                if len(rows) >= limit:
                    return rows
                if len(row) < 6:
                    continue
                question = str(row[0]).strip()
                if not question:
                    continue
                rows.append(
                    {
                        "id": f"mmlu_{subject}_{idx}",
                        "subject": subject,
                        "question_text": question,
                        "options": [str(row[1]).strip(), str(row[2]).strip(), str(row[3]).strip(), str(row[4]).strip()],
                        "correct_answer": str(row[5]).strip() if len(row) >= 6 else "",
                    }
                )
    return rows


def _subject_to_domain(subject: str) -> str:
    s = subject.lower()
    if any(k in s for k in ("physics", "chemistry", "biology", "astronomy", "engineering")):
        return "reality"
    if any(k in s for k in ("math", "algebra", "geometry", "calculus", "statistics", "econometrics")):
        return "math"
    if any(k in s for k in ("logic", "philosophy", "law", "history", "government", "religion", "ethics")):
        return "grammar"
    return "word"


def _build_question_entries(
    row: dict[str, Any],
    *,
    source_name: str,
    include_ollama_hint: str | None,
) -> list[dict[str, Any]]:
    qid = str(row.get("id", "question"))
    text = str(row.get("question_text", ""))
    subject = str(row.get("subject") or row.get("domain") or "multi")
    answer = str(row.get("correct_answer") or row.get("answer") or "").strip()
    domain = _subject_to_domain(subject)

    grammar_id = f"{source_name}_rule_{qid}_{_sha(text)}"
    grammar_entry = {
        "galaxy": "Grammar",
        "entry": {
            "id": grammar_id,
            "name": f"{source_name.upper()} reasoning {qid}",
            "domain": "grammar",
            "category": f"{source_name}_reasoning_rule",
            "rpn_program": "QUESTION PARSE CONTEXT_ALIGN OPTION_SELECT",
            "metadata": {
                "source": f"benchmark_augmentation_{source_name}",
                "question_id": qid,
                "subject": subject,
                "target_domain": domain,
                "cross_modal": ["word", "grammar", "math", "reality"],
                "confidence": 0.82,
                "supervision_answer": answer or None,
                **({"ollama_hint": include_ollama_hint} if include_ollama_hint else {}),
            },
        },
    }

    entries = [grammar_entry]
    if domain == "math":
        entries.append(
            {
                "galaxy": "Math",
                "entry": {
                    "id": f"{source_name}_math_{qid}_{_sha(subject)}",
                    "name": f"{source_name.upper()} math bridge {qid}",
                    "domain": "math",
                    "category": f"{source_name}_math_bridge",
                    "rpn_program": _rpn_from_keywords(text, MATH_KEYWORDS, "A B EQ"),
                    "metadata": {
                        "source": f"benchmark_augmentation_{source_name}",
                        "question_id": qid,
                        "subject": subject,
                        "symlink": "grammar_galaxy",
                        "confidence": 0.8,
                    },
                },
            }
        )
    elif domain == "reality":
        entries.append(
            {
                "galaxy": "Reality",
                "entry": {
                    "id": f"{source_name}_reality_{qid}_{_sha(subject)}",
                    "name": f"{source_name.upper()} reality bridge {qid}",
                    "domain": "reality",
                    "category": f"{source_name}_reality_bridge",
                    "rpn_program": _rpn_from_keywords(text, REALITY_KEYWORDS, "STATE TRANSITION APPLY"),
                    "metadata": {
                        "source": f"benchmark_augmentation_{source_name}",
                        "question_id": qid,
                        "subject": subject,
                        "symlink": "grammar_galaxy",
                        "confidence": 0.8,
                    },
                },
            }
        )

    return entries


def _build_word_entries(counter: Counter[str], *, max_word_entries: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for token, freq in counter.most_common(max_word_entries):
        char_refs = [f"char_u{ord(ch):04x}" for ch in token[:24]]
        out.append(
            {
                "galaxy": "Word",
                "entry": {
                    "id": f"word_bench_{_sha(token)}",
                    "name": token,
                    "domain": "word",
                    "category": "benchmark_lexeme",
                    "rpn_program": f"WORD {token} TOKEN",
                    "metadata": {
                        "source": "benchmark_augmentation",
                        "frequency": int(freq),
                        "symlink": "character_galaxy",
                        "char_refs": char_refs,
                        "cross_modal": ["word", "grammar"],
                        "confidence": 0.78,
                    },
                },
            }
        )
    return out


def _compose_prompt(kind: str, payload: dict[str, Any]) -> str:
    if kind == "arc":
        return (
            "Extract concise transformation hints for this ARC task as one sentence with keywords. "
            "Focus on family, palette behavior, shape changes, and object-count behavior.\n"
            f"Task ID: {payload.get('id')}\n"
            f"Family guess: {payload.get('family')}"
        )
    if kind == "math":
        answer = str(payload.get("answer", "")).strip()
        return (
            "You are building K3D galaxy-ready procedural knowledge. "
            "Given a math problem and gold answer, output concise procedural hints (not full chain-of-thought) "
            "with pattern family, variables, and RPN-friendly operator sequence.\n"
            f"Problem: {payload.get('text', '')[:800]}\n"
            f"Gold answer: {answer[:120]}\n"
            "Return one compact paragraph with key tokens."
        )
    if kind == "qa":
        answer = str(payload.get("answer", "")).strip()
        options = payload.get("options")
        options_line = ""
        if isinstance(options, list) and options:
            compact = [str(opt).strip()[:80] for opt in options if str(opt).strip()]
            if compact:
                options_line = "\nOptions: " + " | ".join(compact[:8])
        return (
            "You are building K3D galaxy-ready reasoning templates. "
            "Given a question plus gold answer, produce compact procedural reasoning anchors, key entities, and "
            "target galaxy hints. Avoid long narrative.\n"
            f"Question: {payload.get('text', '')[:800]}"
            f"{options_line}\n"
            f"Gold answer: {answer[:160]}\n"
            "Return one compact paragraph with entity tokens and operation hints."
        )
    return (
        "Extract one concise reasoning template for this question. "
        "Mention whether it is math, reality, or language dominant.\n"
        f"Question: {payload.get('text', '')[:600]}"
    )


def _maybe_ollama_hint(
    manager: OllamaModelManager | None,
    *,
    enabled: bool,
    model: str,
    prompt: str,
    timeout: float,
    state: dict[str, int],
    stride: int,
    min_len: int = 8,
) -> str | None:
    if not enabled or manager is None:
        return None
    seen = state.get("seen", 0)
    state["seen"] = seen + 1
    if stride > 1 and (seen % stride) != 0:
        return None
    budget = state.get("budget", 0)
    if budget <= 0:
        return None
    res = manager.query(model=model, prompt=prompt, timeout=timeout)
    state["budget"] = budget - 1
    text = (res.output or "").strip()
    if res.returncode != 0 or len(text) < min_len:
        return None
    return text[:800]


def _append_rows(
    rows: list[dict[str, Any]],
    new_rows: Iterable[dict[str, Any]],
    seen_ids: dict[str, set[str]],
    stats_by_galaxy: Counter[str],
    stats_by_source: Counter[str],
) -> None:
    for row in new_rows:
        if not isinstance(row, dict):
            continue
        galaxy = str(row.get("galaxy", "")).strip()
        entry = row.get("entry")
        if not galaxy or not isinstance(entry, dict):
            continue
        entry_id = str(entry.get("id", "")).strip()
        if not entry_id:
            continue
        bucket = seen_ids.setdefault(galaxy, set())
        if entry_id in bucket:
            continue
        bucket.add(entry_id)
        rows.append({"galaxy": galaxy, "entry": entry})
        stats_by_galaxy[galaxy] += 1
        meta = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        stats_by_source[str(meta.get("source", "unknown"))] += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("../Knowledge3D.local/datasets"))
    parser.add_argument("--output", type=Path, default=Path("../Knowledge3D.local/datasets/external_payloads/benchmark_augmentation_payload.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("../Knowledge3D.local/datasets/external_payloads/benchmark_augmentation_report.json"))

    parser.add_argument("--max-arc-tasks", type=int, default=400, help="ARC tasks to augment (<=0 skips ARC)")
    parser.add_argument("--max-math-problems", type=int, default=2000, help="Math records to augment (<=0 skips Math)")
    parser.add_argument("--max-lhe-questions", type=int, default=2500, help="LHE questions to augment (<=0 skips LHE)")
    parser.add_argument("--max-mmlu-questions", type=int, default=2000, help="MMLU questions to augment (<=0 skips MMLU)")
    parser.add_argument("--max-word-entries", type=int, default=50000)

    parser.add_argument(
        "--skip-ollama-enrichment",
        action="store_true",
        help="EMERGENCY ONLY: skip Ollama enrichment (breaks standard augmentation pipeline).",
    )
    parser.add_argument("--ollama-model", default="llama3.2")
    parser.add_argument("--ollama-timeout", type=float, default=45.0)
    parser.add_argument("--ollama-stride", type=int, default=50, help="Call Ollama once every N records (1 = every record)")
    parser.add_argument("--max-ollama-calls", type=int, default=200)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    seen_ids: dict[str, set[str]] = {}
    stats_by_galaxy: Counter[str] = Counter()
    stats_by_source: Counter[str] = Counter()
    token_counter: Counter[str] = Counter()

    arc_stats = AugmentStats()
    math_stats = AugmentStats()
    lhe_stats = AugmentStats()
    mmlu_stats = AugmentStats()

    ollama_enabled = not bool(args.skip_ollama_enrichment)
    if not ollama_enabled:
        print(
            "[augment] WARNING: --skip-ollama-enrichment is active. "
            "This bypasses mandatory enrichment and should be used only for emergency diagnostics."
        )
    ollama_state = {"budget": int(max(0, args.max_ollama_calls)), "seen": 0}
    manager_ctx = OllamaModelManager(default_timeout=float(args.ollama_timeout)) if ollama_enabled else None

    if manager_ctx is None:
        manager_cm = None
    else:
        manager_cm = manager_ctx

    try:
        manager = manager_cm.__enter__() if manager_cm is not None else None

        if args.max_arc_tasks > 0:
            arc_tasks = _iter_arc_tasks(args.dataset_root, args.max_arc_tasks)
            for task in arc_tasks:
                family = _infer_arc_family(task)
                hint = _maybe_ollama_hint(
                    manager,
                    enabled=ollama_enabled,
                    model=args.ollama_model,
                    prompt=_compose_prompt("arc", {"id": task.get("id"), "family": family}),
                    timeout=args.ollama_timeout,
                    state=ollama_state,
                    stride=max(1, int(args.ollama_stride)),
                )
                new_rows = _build_arc_entries(task, hint)
                _append_rows(rows, new_rows, seen_ids, stats_by_galaxy, stats_by_source)
                arc_stats.processed += 1
                arc_stats.entries += len(new_rows)
                for ex in task.get("train", []):
                    if isinstance(ex, dict):
                        token_counter.update(_tokenize(json.dumps(ex, ensure_ascii=False)))

        if args.max_math_problems > 0:
            math_rows = _iter_math_records(args.dataset_root, args.max_math_problems)
            for row in math_rows:
                text = str(row.get("problem_text", ""))
                hint = _maybe_ollama_hint(
                    manager,
                    enabled=ollama_enabled,
                    model=args.ollama_model,
                    prompt=_compose_prompt("math", {"text": text, "answer": row.get("answer", "")}),
                    timeout=args.ollama_timeout,
                    state=ollama_state,
                    stride=max(1, int(args.ollama_stride)),
                )
                new_rows = _build_math_entries(row, hint)
                _append_rows(rows, new_rows, seen_ids, stats_by_galaxy, stats_by_source)
                math_stats.processed += 1
                math_stats.entries += len(new_rows)
                token_counter.update(_tokenize(text))

        if args.max_lhe_questions > 0:
            lhe_rows = _iter_lhe_rows(args.dataset_root, args.max_lhe_questions)
            for row in lhe_rows:
                text = str(row.get("question_text", ""))
                hint = _maybe_ollama_hint(
                    manager,
                    enabled=ollama_enabled,
                    model=args.ollama_model,
                    prompt=_compose_prompt(
                        "qa",
                        {
                            "text": text,
                            "answer": row.get("correct_answer", ""),
                            "options": row.get("options", []),
                        },
                    ),
                    timeout=args.ollama_timeout,
                    state=ollama_state,
                    stride=max(1, int(args.ollama_stride)),
                )
                new_rows = _build_question_entries(row, source_name="lhe", include_ollama_hint=hint)
                _append_rows(rows, new_rows, seen_ids, stats_by_galaxy, stats_by_source)
                lhe_stats.processed += 1
                lhe_stats.entries += len(new_rows)
                token_counter.update(_tokenize(text))
                for opt in row.get("options", []):
                    token_counter.update(_tokenize(str(opt)))

        if args.max_mmlu_questions > 0:
            mmlu_rows = _iter_mmlu_rows(args.dataset_root, args.max_mmlu_questions)
            for row in mmlu_rows:
                text = str(row.get("question_text", ""))
                hint = _maybe_ollama_hint(
                    manager,
                    enabled=ollama_enabled,
                    model=args.ollama_model,
                    prompt=_compose_prompt(
                        "qa",
                        {
                            "text": text,
                            "answer": row.get("correct_answer", ""),
                            "options": row.get("options", []),
                        },
                    ),
                    timeout=args.ollama_timeout,
                    state=ollama_state,
                    stride=max(1, int(args.ollama_stride)),
                )
                new_rows = _build_question_entries(row, source_name="mmlu", include_ollama_hint=hint)
                _append_rows(rows, new_rows, seen_ids, stats_by_galaxy, stats_by_source)
                mmlu_stats.processed += 1
                mmlu_stats.entries += len(new_rows)
                token_counter.update(_tokenize(text))
                for opt in row.get("options", []):
                    token_counter.update(_tokenize(str(opt)))

        # Add compact benchmark lexicon rows for Character->Word symlink flows.
        word_rows = _build_word_entries(token_counter, max_word_entries=max(0, int(args.max_word_entries)))
        _append_rows(rows, word_rows, seen_ids, stats_by_galaxy, stats_by_source)

    finally:
        if manager_cm is not None:
            manager_cm.__exit__(None, None, None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    report = {
        "dataset_root": str(args.dataset_root),
        "output": str(args.output),
        "total_payload_rows": len(rows),
        "ollama": {
            "enabled": bool(ollama_enabled),
            "model": args.ollama_model,
            "calls_budget": int(args.max_ollama_calls),
            "calls_used": int(max(0, args.max_ollama_calls - ollama_state.get("budget", 0))),
            "stride": int(max(1, args.ollama_stride)),
        },
        "augmented": {
            "arc": {"processed": arc_stats.processed, "entries": arc_stats.entries},
            "math": {"processed": math_stats.processed, "entries": math_stats.entries},
            "lhe": {"processed": lhe_stats.processed, "entries": lhe_stats.entries},
            "mmlu": {"processed": mmlu_stats.processed, "entries": mmlu_stats.entries},
            "word_entries": int(stats_by_galaxy.get("Word", 0)),
        },
        "stats_by_galaxy": dict(sorted(stats_by_galaxy.items())),
        "stats_by_source": dict(sorted(stats_by_source.items())),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[augment] rows={len(rows)} output={args.output}")
    print(f"[augment] report={args.report}")
    for galaxy, count in sorted(stats_by_galaxy.items()):
        print(f"[augment] {galaxy}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
