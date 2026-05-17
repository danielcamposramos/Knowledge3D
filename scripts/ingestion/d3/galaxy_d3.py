#!/usr/bin/env python3
"""D3 additive dedup + RPN proceduralization + Matryoshka staging pipeline.

This script consumes the byte-stable D2 staged outputs and produces the D3
staging artifacts under ``scripts/ingestion/staging/D3_dedup/``.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge3d.cranium.sovereign import loader
from knowledge3d.skills.infix_to_rpn import infix_to_rpn
from scripts.ingestion.audit.galaxy_audit import (  # type: ignore[attr-defined]
    _collect_procedural_payload,
    _content_hash,
    _has_matryoshka_payload,
    _looks_canonical_id,
    _non_placeholder_text,
    render_report as audit_render_report,
    run_scan as audit_run_scan,
)
from scripts.ingestion.normalize.galaxy_normalize import (  # type: ignore[attr-defined]
    _append_target_to_path,
    _expected_inverse_kind,
    _get_path_value,
    _inverse_field_path,
    _set_path_value,
)


DEFAULT_D2_STAGE_DIR = REPO_ROOT / "scripts" / "ingestion" / "staging" / "D2_normalize"
DEFAULT_STAGE_DIR = REPO_ROOT / "scripts" / "ingestion" / "staging" / "D3_dedup"
DEFAULT_TEMP_REPORT_PATH = REPO_ROOT / "TEMP" / "CODEX_D3_FINAL_REPORT_04.19.2026.md"
REPRO_COMMAND = "bash scripts/ingestion/d3/run.sh"
NULL_TEXT_VALUES = {"", "none", "null", "n/a"}
PROCEDURAL_FIELDS = ("rpn_program", "meaning_rpn", "visual_rpn", "audio_rpn", "behavior_rpn")
RAW_BATCH_FIELDS = ("content", "description", "summary", "name", "action", "condition", "answer_text")
WORD_CHARACTER_GALAXIES = {"Word", "Character"}
MATRYOSHKA_DIMENSIONS = (64, 128, 512, 2048)
MATRYOSHKA_DIM = 2048
PACKED_TRITS_STRIDE = (MATRYOSHKA_DIM + 4) // 5
KNOWN_OPCODE_MAP = {
    "+": 0x0A,
    "-": 0x0B,
    "*": 0x0C,
    "/": 0x0D,
    "^": 0x0E,
    "sqrt": 0x14,
    "exp": 0x15,
    "log": 0x16,
    "sin": 0x18,
    "cos": 0x19,
    "tan": 0x1A,
    "asin": 0x1B,
    "acos": 0x1C,
    "atan": 0x1D,
    "abs": 0x27,
    "gt": 0x28,
    "ceil": 0x29,
    "lt": 0x2A,
    "floor": 0x2B,
    "eq": 0x2C,
    "round": 0x2D,
    "max": 0x2E,
    "min": 0x2F,
    "dup": 0x32,
    "swap": 0x33,
    "drop": 0x34,
    "mod": 0x38,
    "log2": 0x39,
    "log10": 0x3A,
    "store": 0xB3,
    "recall": 0xB4,
    "gradient": 0xB6,
    "branch": 0xB0,
    "loop": 0xB1,
    "next": 0xB2,
    "normalize": 0xC1,
    "TERNARY_QUANT": 0x170,
    "TERNARY_DEQUANT": 0x171,
    "BATCH_DCT": 0x176,
    "BATCH_MDCT": 0x177,
    "RESHAPE_TO_BLOCKS": 0x178,
    "BLOCKS_TO_GRID": 0x179,
}
AUTHORITY_ORDER = {
    "curated": 0,
    "w3c cg": 1,
    "cg": 1,
    "registry": 2,
    "pm-kr": 2,
    "auto": 3,
    "heuristic": 4,
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                yield json.loads(text)


def _deep_copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _field_kind_from_path(field_path: str) -> str:
    if ".surface_forms." in field_path and field_path.endswith(".word_ref"):
        return "surface_forms.*.word_ref"
    if ".surface_forms." in field_path and field_path.endswith(".char_refs"):
        return "surface_forms.*.char_refs"
    return field_path.split(".")[-1]


def _numeric_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = _non_placeholder_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_label(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return " ".join(stripped.lower().split())


def _source_authority_rank(value: Any) -> int:
    text = _non_placeholder_text(value).lower()
    if not text:
        return AUTHORITY_ORDER["heuristic"]
    for needle, rank in AUTHORITY_ORDER.items():
        if needle in text:
            return rank
    return AUTHORITY_ORDER["auto"]


def _first_non_placeholder(row: dict[str, Any], paths: list[tuple[str, ...]]) -> str:
    for path in paths:
        current: Any = row
        for part in path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(part)
        text = _non_placeholder_text(current)
        if text:
            return text
    return ""


def _row_identity_values(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("id", "star_id"):
        candidate = row.get(key)
        if isinstance(candidate, str) and candidate.strip():
            values.append(candidate.strip())
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in ("meaning_star_id", "meaning_ref", "canonical_id", "meta_id"):
            candidate = metadata.get(key)
            if isinstance(candidate, str) and candidate.strip():
                values.append(candidate.strip())
        meaning_star = metadata.get("meaning_star")
        if isinstance(meaning_star, dict):
            candidate = meaning_star.get("star_id")
            if isinstance(candidate, str) and candidate.strip():
                values.append(candidate.strip())
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _pick_canonical_id(member_ids: list[str]) -> str:
    k3d_ids = sorted(member_id for member_id in member_ids if member_id.startswith("k3d-"))
    if k3d_ids:
        return k3d_ids[0]
    canonical = sorted(member_id for member_id in member_ids if _looks_canonical_id(member_id))
    if canonical:
        return canonical[0]
    return sorted(member_ids)[0]


def _sync_row_ids(row: dict[str, Any], canonical_id: str) -> dict[str, Any]:
    synced = _deep_copy_json(row)
    synced["id"] = canonical_id
    if "star_id" in synced:
        synced["star_id"] = canonical_id
    metadata = synced.get("metadata")
    if isinstance(metadata, dict):
        if "meaning_star_id" in metadata:
            metadata["meaning_star_id"] = canonical_id
        if "meaning_ref" in metadata and isinstance(metadata["meaning_ref"], str):
            metadata["meaning_ref"] = canonical_id
        meaning_star = metadata.get("meaning_star")
        if isinstance(meaning_star, dict):
            meaning_star["star_id"] = canonical_id
    return synced


def _best_procedural_text(existing: Any, candidate: Any) -> Any:
    existing_text = _non_placeholder_text(existing)
    candidate_text = _non_placeholder_text(candidate)
    if not existing_text:
        return candidate
    if not candidate_text:
        return existing
    existing_score = (len(existing_text.split()), len(existing_text), existing_text)
    candidate_score = (len(candidate_text.split()), len(candidate_text), candidate_text)
    return candidate if candidate_score < existing_score else existing


def _merge_lists(existing: list[Any], candidate: list[Any]) -> list[Any]:
    combined = list(existing) + list(candidate)
    keyed: dict[str, Any] = {}
    for item in combined:
        key = _stable_json(item)
        keyed[key] = item
    ordered_keys = sorted(keyed)
    return [keyed[key] for key in ordered_keys]


def _merge_values(path: tuple[str, ...], existing: Any, candidate: Any, slot_sources: dict[str, str], source_id: str) -> Any:
    path_key = ".".join(path)
    leaf = path[-1] if path else ""
    if existing is None:
        if candidate is not None:
            slot_sources[path_key] = source_id
        return _deep_copy_json(candidate)
    if candidate is None:
        return existing
    if isinstance(existing, dict) and isinstance(candidate, dict):
        merged: dict[str, Any] = {}
        for key in sorted(set(existing) | set(candidate)):
            if key in existing and key in candidate:
                merged[key] = _merge_values(path + (key,), existing[key], candidate[key], slot_sources, source_id)
            elif key in existing:
                merged[key] = _deep_copy_json(existing[key])
            else:
                merged[key] = _deep_copy_json(candidate[key])
                slot_sources[".".join(path + (key,))] = source_id
        return merged
    if isinstance(existing, list) and isinstance(candidate, list):
        merged = _merge_lists(existing, candidate)
        if merged != existing:
            slot_sources[path_key] = source_id
        return merged
    if leaf in {"id", "star_id", "meaning_star_id", "meaning_ref"}:
        return existing
    if leaf == "source":
        existing_rank = _source_authority_rank(existing)
        candidate_rank = _source_authority_rank(candidate)
        if candidate_rank < existing_rank:
            slot_sources[path_key] = source_id
            return candidate
        return existing
    if "confidence" in leaf:
        existing_num = _numeric_or_none(existing)
        candidate_num = _numeric_or_none(candidate)
        if existing_num is None:
            slot_sources[path_key] = source_id
            return candidate
        if candidate_num is not None and candidate_num > existing_num:
            slot_sources[path_key] = source_id
            return candidate
        return existing
    if leaf in {"created_at", "first_seen"}:
        if str(candidate) < str(existing):
            slot_sources[path_key] = source_id
            return candidate
        return existing
    if leaf in {"last_updated", "updated_at", "updated"}:
        if str(candidate) > str(existing):
            slot_sources[path_key] = source_id
            return candidate
        return existing
    if leaf in PROCEDURAL_FIELDS:
        chosen = _best_procedural_text(existing, candidate)
        if chosen != existing:
            slot_sources[path_key] = source_id
        return chosen
    if isinstance(existing, str) or isinstance(candidate, str):
        existing_text = _non_placeholder_text(existing)
        candidate_text = _non_placeholder_text(candidate)
        if not existing_text and candidate_text:
            slot_sources[path_key] = source_id
            return candidate
        return existing
    if isinstance(existing, (int, float)) and isinstance(candidate, (int, float)):
        return existing
    return existing


def _pick_primary_label(row: dict[str, Any]) -> str:
    candidates = [
        ("name",),
        ("summary",),
        ("description",),
        ("content",),
        ("metadata", "query_anchor"),
        ("metadata", "embedding_text"),
        ("metadata", "semantics"),
        ("metadata", "meaning_star", "meaning_rpn"),
    ]
    return _first_non_placeholder(row, candidates)


def _primary_language_family(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        language_profile = metadata.get("language_profile")
        if isinstance(language_profile, dict):
            language = _non_placeholder_text(language_profile.get("language"))
            if language:
                return language
    row_id = _non_placeholder_text(row.get("id"))
    if row_id.startswith("word_bench_"):
        return "unknown"
    if row_id.startswith("word_"):
        parts = row_id.split("_")
        if len(parts) >= 3:
            return parts[1]
    return "unknown"


def _normalized_star_type_for_meaning_key(row: dict[str, Any]) -> str:
    star_type = _non_placeholder_text(
        row.get("category") or row.get("kind") or row.get("type") or row.get("domain") or row.get("galaxy")
    )
    if star_type.startswith("benchmark_"):
        return star_type[len("benchmark_") :]
    return star_type


def _meaning_group_key(row: dict[str, Any]) -> str:
    content_hash = _content_hash(row, _collect_procedural_payload(row))
    concept_id = _first_non_placeholder(
        row,
        [
            ("metadata", "concept_id"),
            ("metadata", "meaning_star_id"),
            ("metadata", "meaning_ref"),
        ],
    )
    if concept_id:
        basis = {
            "star_type": _normalized_star_type_for_meaning_key(row),
            "concept_id": concept_id,
            "language": _primary_language_family(row),
        }
        return hashlib.sha256(_stable_json(basis).encode("utf-8")).hexdigest()
    semantic_text = _normalize_label(
        _first_non_placeholder(
            row,
            [
                ("condition",),
                ("action",),
                ("content",),
                ("summary",),
                ("description",),
                ("name",),
                ("metadata", "query_anchor"),
                ("metadata", "embedding_text"),
                ("metadata", "semantics"),
            ],
        )
    )
    if semantic_text:
        basis = {
            "star_type": _normalized_star_type_for_meaning_key(row),
            "semantic_text": semantic_text,
            "language": _primary_language_family(row),
        }
        return hashlib.sha256(_stable_json(basis).encode("utf-8")).hexdigest()
    return content_hash


def _payload_type_for_row(row: dict[str, Any]) -> str:
    category = _non_placeholder_text(row.get("category") or row.get("kind") or row.get("type")).lower()
    metadata = row.get("metadata")
    content = _non_placeholder_text(row.get("content"))
    if isinstance(metadata, dict) and isinstance(metadata.get("arc_primitive_plan"), list):
        return "glyph_bezier"
    if "formula" in category or "equation" in category:
        return "math_expression"
    if "language_profile" in category or "multilingual_word" in category or "meaning_symlink" in category:
        return "text_sequence"
    if "rule" in category or row.get("domain") in {"grammar", "interactive_game_strategy"}:
        return "grammar_rule"
    if "audio" in category:
        return "audio_spec"
    if any(ch in content for ch in "=^+-*/()") and any(ch.isdigit() for ch in content):
        return "math_expression"
    return "unknown"


def _build_text_sequence_rpn(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    query_anchor = _non_placeholder_text((metadata or {}).get("query_anchor"))
    content = _non_placeholder_text(row.get("content"))
    name = _non_placeholder_text(row.get("name"))
    source_text = query_anchor or content or name or _non_placeholder_text(row.get("id"))
    tokens = [token for token in source_text.replace("|", " ").replace("/", " ").split() if token]
    if not tokens:
        return ""
    return " ".join(tokens[:64] + ["COMPOSE"])


def _build_visual_rpn(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("arc_primitive_plan"), list):
        ops: list[str] = []
        for item in metadata["arc_primitive_plan"]:
            if isinstance(item, dict):
                op_name = _non_placeholder_text(item.get("op"))
                if op_name:
                    ops.append(op_name.upper())
        task_id = _non_placeholder_text(metadata.get("arc_task_id"))
        if ops:
            suffix = [f"ARC_TASK_{task_id}"] if task_id else []
            return " ".join(ops + suffix + ["VISUAL_COMPOSE"])
    task_id = _non_placeholder_text(row.get("task_id"))
    if task_id:
        train_examples = row.get("train_examples")
        example_token = f"TRAIN_EXAMPLES_{int(train_examples)}" if isinstance(train_examples, int) else ""
        pieces = [f"ARC_PATTERN_{task_id.upper()}"]
        if example_token:
            pieces.append(example_token)
        pieces.append("VISUAL_COMPOSE")
        return " ".join(pieces)
    return ""


def _build_grammar_rpn(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    condition = _non_placeholder_text(row.get("condition"))
    action = _non_placeholder_text(row.get("action"))
    query_anchor = _non_placeholder_text((metadata or {}).get("query_anchor"))
    semantics = _non_placeholder_text((metadata or {}).get("semantics"))
    content = _non_placeholder_text(row.get("content"))
    if condition or action:
        pieces = []
        if condition:
            pieces.extend(condition.split())
        if action:
            pieces.extend(action.split())
        pieces.append("APPLY_RULE")
        return " ".join(pieces)
    source = query_anchor or semantics or content
    if source:
        return " ".join(source.split()[:64] + ["RULE_COMPOSE"])
    return ""


def _build_math_rpn(row: dict[str, Any]) -> str:
    source = _non_placeholder_text(row.get("content")) or _non_placeholder_text(row.get("summary")) or _non_placeholder_text(row.get("description"))
    if not source:
        return ""
    try:
        tokens = infix_to_rpn(source)
    except Exception:
        tokens = []
    return " ".join(tokens)


def _convert_raw_row(row: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    payload_type = _payload_type_for_row(row)
    converted = _deep_copy_json(row)
    metadata = converted.setdefault("metadata", {}) if isinstance(converted.get("metadata"), dict) else {}
    if payload_type == "math_expression":
        program = _build_math_rpn(converted)
        if program:
            converted["rpn_program"] = program
            metadata["payload_type"] = payload_type
            return True, converted
        return False, converted
    if payload_type == "glyph_bezier":
        program = _build_visual_rpn(converted)
        if program:
            converted["visual_rpn"] = program
            metadata["payload_type"] = payload_type
            return True, converted
        return False, converted
    if payload_type == "text_sequence":
        program = _build_text_sequence_rpn(converted)
        if program:
            converted["meaning_rpn"] = program
            metadata["payload_type"] = payload_type
            return True, converted
        return False, converted
    if payload_type == "grammar_rule":
        program = _build_grammar_rpn(converted)
        if program:
            converted["behavior_rpn"] = program
            metadata["payload_type"] = payload_type
            return True, converted
        return False, converted
    if payload_type == "audio_spec":
        source = _non_placeholder_text(converted.get("content")) or _non_placeholder_text(converted.get("summary"))
        if source:
            converted["audio_rpn"] = " ".join(source.split()[:64] + ["AUDIO_COMPOSE"])
            metadata["payload_type"] = payload_type
            return True, converted
        return False, converted
    source = (
        _non_placeholder_text(converted.get("content"))
        or _non_placeholder_text(converted.get("description"))
        or _non_placeholder_text(converted.get("summary"))
        or _non_placeholder_text(converted.get("name"))
        or _non_placeholder_text((metadata or {}).get("query_anchor"))
    )
    if source:
        converted["meaning_rpn"] = " ".join(source.split()[:64] + ["MEANING_COMPOSE"])
        metadata["payload_type"] = "unknown"
        return True, converted
    return False, converted


def _tokenize_for_embedding(program: str) -> list[str]:
    return [token for token in str(program or "").replace("×", "*").replace("÷", "/").split() if token]


def _canonical_opcode_for_token(token: str) -> int | None:
    if token in KNOWN_OPCODE_MAP:
        return KNOWN_OPCODE_MAP[token]
    lowered = token.lower()
    if lowered in KNOWN_OPCODE_MAP:
        return KNOWN_OPCODE_MAP[lowered]
    return None


def _program_for_embedding(row: dict[str, Any]) -> str:
    payload = _collect_procedural_payload(row)
    if payload:
        for field in (
            "rpn_program",
            "meaning_rpn",
            "visual_rpn",
            "audio_rpn",
            "behavior_rpn",
            "metadata.meaning_star.meaning_rpn",
            "metadata.meaning_star.visual_rpn",
            "metadata.meaning_star.audio_rpn",
            "metadata.meaning_star.behavior_rpn",
        ):
            if field in payload:
                return payload[field]
    return ""


def _load_rpn_smoke_module() -> None:
    module_path = REPO_ROOT / "knowledge3d" / "cranium" / "ptx" / "modular_rpn_kernel.ptx"
    loader.load_module_from_file(str(module_path))


class MatryoshkaAccumulator:
    def __init__(
        self,
        *,
        source_path: Path,
        ptx_path: Path,
        basis_source_path: Path,
        basis_ptx_path: Path,
    ) -> None:
        self.source_path = source_path
        self.ptx_path = ptx_path
        self.basis_source_path = basis_source_path
        self.basis_ptx_path = basis_ptx_path
        self.module = None
        self.kernel = None
        self.basis_module = None
        self.basis_fingerprint_kernel = None
        self.basis_token_id_kernel = None
        self.d_basis = None
        self.d_token_indices = None
        self.d_token_ids = None
        self.d_masses = None
        self.d_output = None
        self.d_vocab_token_ids = None
        self.max_tokens = 0
        self.basis_stride = PACKED_TRITS_STRIDE
        self.vocab: list[str] = []
        self.token_to_index: dict[str, int] = {}
        self.token_bytes_blob: bytes = b""
        self.token_offsets: Any = None
        self.vocab_token_ids_host: Any = None

    def _compile_one(self, source_path: Path, ptx_path: Path) -> None:
        if ptx_path.exists() and ptx_path.stat().st_mtime >= source_path.stat().st_mtime:
            return
        ptx_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "nvcc",
            "-ptx",
            "-O3",
            str(source_path),
            "-o",
            str(ptx_path),
        ]
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    def compile(self) -> None:
        self._compile_one(self.source_path, self.ptx_path)
        self._compile_one(self.basis_source_path, self.basis_ptx_path)

    def prepare_vocab(self, token_lists: list[list[str]]) -> None:
        vocab = sorted({token for tokens in token_lists for token in tokens})
        self.vocab = vocab
        self.token_to_index = {token: index for index, token in enumerate(vocab)}
        byte_chunks: list[bytes] = []
        offsets: list[int] = [0]
        total = 0
        for token in vocab:
            encoded = token.encode("utf-8")
            byte_chunks.append(encoded)
            total += len(encoded)
            offsets.append(total)
        self.token_bytes_blob = b"".join(byte_chunks)
        self.token_offsets = (ctypes.c_uint32 * len(offsets))(*offsets)

    def _derive_vocab_on_gpu(self) -> None:
        vocab_count = len(self.vocab)

        byte_blob_size = len(self.token_bytes_blob)
        d_token_bytes = loader.gpu_malloc(max(byte_blob_size, 1))
        if byte_blob_size:
            loader.memcpy_htod(
                d_token_bytes,
                ctypes.create_string_buffer(self.token_bytes_blob),
                byte_blob_size,
            )
        offsets_size = ctypes.sizeof(self.token_offsets)
        d_token_offsets = loader.gpu_malloc(offsets_size)
        loader.memcpy_htod(d_token_offsets, self.token_offsets, offsets_size)

        basis_bytes = vocab_count * PACKED_TRITS_STRIDE
        self.d_basis = loader.gpu_malloc(basis_bytes)
        basis_params = [
            d_token_bytes,
            d_token_offsets,
            ctypes.c_int(vocab_count),
            self.d_basis,
        ]
        loader.launch(
            self.basis_fingerprint_kernel,
            (vocab_count, 1, 1),
            (64, 1, 1),
            basis_params,
        )

        ids_bytes = vocab_count * ctypes.sizeof(ctypes.c_uint32)
        self.d_vocab_token_ids = loader.gpu_malloc(ids_bytes)
        id_threads = 256
        id_blocks = (vocab_count + id_threads - 1) // id_threads
        id_params = [
            d_token_bytes,
            d_token_offsets,
            ctypes.c_int(vocab_count),
            self.d_vocab_token_ids,
        ]
        loader.launch(
            self.basis_token_id_kernel,
            (id_blocks, 1, 1),
            (id_threads, 1, 1),
            id_params,
        )

        id_host = (ctypes.c_uint32 * vocab_count)()
        loader.memcpy_dtoh(id_host, self.d_vocab_token_ids, ids_bytes)
        for index, token in enumerate(self.vocab):
            canonical = _canonical_opcode_for_token(token)
            if canonical is not None:
                id_host[index] = ctypes.c_uint32(canonical).value
        loader.memcpy_htod(self.d_vocab_token_ids, id_host, ids_bytes)
        self.vocab_token_ids_host = id_host

        loader.gpu_free(d_token_bytes)
        loader.gpu_free(d_token_offsets)

    def load(self) -> None:
        if not self.vocab:
            return
        self.compile()
        self.module = loader.load_module_from_file(str(self.ptx_path))
        self.kernel = loader.get_function(self.module, "matryoshka_accumulator")
        self.basis_module = loader.load_module_from_file(str(self.basis_ptx_path))
        self.basis_fingerprint_kernel = loader.get_function(
            self.basis_module, "procedural_basis_fingerprint"
        )
        self.basis_token_id_kernel = loader.get_function(
            self.basis_module, "procedural_token_id"
        )
        self._derive_vocab_on_gpu()

    def ensure_buffers(self, max_tokens: int) -> None:
        if max_tokens <= self.max_tokens:
            return
        self.max_tokens = max_tokens
        byte_count_u32 = max_tokens * ctypes.sizeof(ctypes.c_uint32)
        byte_count_u8 = max_tokens * ctypes.sizeof(ctypes.c_uint8)
        if self.d_token_indices is not None:
            loader.gpu_free(self.d_token_indices)
            loader.gpu_free(self.d_token_ids)
            loader.gpu_free(self.d_masses)
        if self.d_output is not None:
            loader.gpu_free(self.d_output)
        self.d_token_indices = loader.gpu_malloc(byte_count_u32)
        self.d_token_ids = loader.gpu_malloc(byte_count_u32)
        self.d_masses = loader.gpu_malloc(byte_count_u8)
        self.d_output = loader.gpu_malloc(MATRYOSHKA_DIM * ctypes.sizeof(ctypes.c_int8))

    def embed(self, tokens: list[str]) -> bytes:
        if not tokens:
            return bytes(MATRYOSHKA_DIM)
        self.ensure_buffers(len(tokens))
        token_indices = (ctypes.c_uint32 * len(tokens))()
        token_ids = (ctypes.c_uint32 * len(tokens))()
        masses = (ctypes.c_uint8 * len(tokens))()
        for index, token in enumerate(tokens):
            vocab_index = self.token_to_index[token]
            token_indices[index] = vocab_index
            numeric_id = int(self.vocab_token_ids_host[vocab_index])
            token_ids[index] = numeric_id
            is_canonical = (numeric_id & 0x80000000) == 0
            masses[index] = 64 if is_canonical else 32
        loader.memcpy_htod(self.d_token_indices, token_indices, ctypes.sizeof(token_indices))
        loader.memcpy_htod(self.d_token_ids, token_ids, ctypes.sizeof(token_ids))
        loader.memcpy_htod(self.d_masses, masses, ctypes.sizeof(masses))
        params = [
            self.d_basis,
            ctypes.c_int(self.basis_stride),
            self.d_token_indices,
            self.d_token_ids,
            self.d_masses,
            ctypes.c_int(len(tokens)),
            self.d_output,
        ]
        loader.launch(self.kernel, (1, 1, 1), (256, 1, 1), params)
        output = (ctypes.c_int8 * MATRYOSHKA_DIM)()
        loader.memcpy_dtoh(output, self.d_output, ctypes.sizeof(output))
        return bytes((value + 256) % 256 for value in output)

    def close(self) -> None:
        for ptr_name in (
            "d_basis",
            "d_token_indices",
            "d_token_ids",
            "d_masses",
            "d_output",
            "d_vocab_token_ids",
        ):
            ptr = getattr(self, ptr_name, None)
            if ptr is not None:
                try:
                    loader.gpu_free(ptr)
                except Exception:
                    pass
                setattr(self, ptr_name, None)


class D3Pipeline:
    def __init__(self, *, d2_stage_dir: Path, stage_dir: Path, temp_report_path: Path) -> None:
        self.d2_stage_dir = d2_stage_dir
        self.stage_dir = stage_dir
        self.temp_report_path = temp_report_path
        self.normalized_input_dir = self.d2_stage_dir / "normalized"
        self.d2_violations_path = self.d2_stage_dir / "re_audit" / "violations.jsonl"
        self.kernel_source_path = REPO_ROOT / "scripts" / "ingestion" / "d3" / "matryoshka_accumulator.cu"
        self.basis_kernel_source_path = REPO_ROOT / "scripts" / "ingestion" / "d3" / "procedural_basis.cu"
        self.temp_run_dir = self.stage_dir.parent / ".D3_dedup_run_a"

    def _reset_stage_dir(self, path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    def _connect(self, path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(str(path))
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=MEMORY")
        return connection

    def _init_db(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE source_rows (
                row_id TEXT PRIMARY KEY,
                galaxy TEXT NOT NULL,
                file_name TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                group_key TEXT NOT NULL,
                row_json TEXT NOT NULL
            );
            CREATE TABLE merged_rows (
                row_id TEXT PRIMARY KEY,
                galaxy TEXT NOT NULL,
                file_name TEXT NOT NULL,
                row_json TEXT NOT NULL,
                member_ids_json TEXT NOT NULL,
                member_count INTEGER NOT NULL
            );
            CREATE TABLE final_rows (
                row_id TEXT PRIMARY KEY,
                galaxy TEXT NOT NULL,
                file_name TEXT NOT NULL,
                row_json TEXT NOT NULL
            );
            """
        )
        connection.execute("CREATE INDEX idx_source_group ON source_rows (group_key)")
        connection.execute("CREATE INDEX idx_merged_galaxy ON merged_rows (galaxy, row_id)")
        connection.execute("CREATE INDEX idx_final_galaxy ON final_rows (galaxy, row_id)")
        connection.commit()

    def _load_d2_source_rows(self, connection: sqlite3.Connection) -> int:
        total_rows = 0
        for path in sorted(self.normalized_input_dir.glob("*.jsonl")):
            galaxy = path.stem
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if not text:
                        continue
                    row = json.loads(text)
                    row_id = _row_identity_values(row)[0]
                    content_hash = _content_hash(row, _collect_procedural_payload(row))
                    group_key = _meaning_group_key(row)
                    connection.execute(
                        """
                        INSERT INTO source_rows (row_id, galaxy, file_name, content_hash, group_key, row_json)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row_id,
                            galaxy,
                            path.name,
                            content_hash,
                            group_key,
                            json.dumps(row, ensure_ascii=False, sort_keys=True),
                        ),
                    )
                    total_rows += 1
        connection.commit()
        return total_rows

    def _apply_b1_bidirectional_closure(self, connection: sqlite3.Connection, output_path: Path) -> list[dict[str, Any]]:
        source_rows: dict[str, dict[str, Any]] = {}
        for row_id, galaxy, file_name, row_json in connection.execute(
            "SELECT row_id, galaxy, file_name, row_json FROM source_rows"
        ):
            source_rows[row_id] = {
                "galaxy": galaxy,
                "file_name": file_name,
                "row": json.loads(row_json),
            }
        additions: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for violation in _iter_jsonl(self.d2_violations_path):
            if violation.get("violation_kind") != "unidirectional_ref":
                continue
            row_id = str(violation.get("row_id") or "").strip()
            target_id = str(violation.get("target_id") or "").strip()
            field_path = str(violation.get("field_path") or "").strip()
            if not row_id or not target_id or not field_path:
                continue
            source_kind = _field_kind_from_path(field_path)
            inverse_kind = _expected_inverse_kind(source_kind)
            if not inverse_kind:
                continue
            inverse_field_path = _inverse_field_path(field_path, source_kind)
            target_entry = source_rows.get(target_id)
            if target_entry is None:
                continue
            target_row = target_entry["row"]
            current_value = _get_path_value(target_row, inverse_field_path)
            already_present = False
            if isinstance(current_value, list):
                already_present = row_id in current_value
            elif isinstance(current_value, str):
                already_present = current_value == row_id
            if already_present:
                continue
            _append_target_to_path(target_row, inverse_field_path, row_id)
            target_entry["row"] = target_row
            source_rows[target_id] = target_entry
            key = (target_id, inverse_field_path, row_id)
            if key in seen:
                continue
            seen.add(key)
            additions.append(
                {
                    "reproduction_command": REPRO_COMMAND,
                    "source_row_id": row_id,
                    "source_field_path": field_path,
                    "target_row_id": target_id,
                    "inverse_field_path": inverse_field_path,
                    "inverse_kind": inverse_kind,
                }
            )
        connection.execute("DELETE FROM source_rows")
        for row_id, entry in sorted(source_rows.items()):
            row = entry["row"]
            galaxy = entry["galaxy"]
            file_name = entry["file_name"]
            content_hash = _content_hash(row, _collect_procedural_payload(row))
            group_key = _meaning_group_key(row)
            connection.execute(
                """
                INSERT INTO source_rows (row_id, galaxy, file_name, content_hash, group_key, row_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    galaxy,
                    file_name,
                    content_hash,
                    group_key,
                        json.dumps(row, ensure_ascii=False, sort_keys=True),
                    ),
                )
        connection.commit()
        additions.sort(key=lambda row: (row["target_row_id"], row["inverse_field_path"], row["source_row_id"]))
        _write_jsonl(output_path, additions)
        return additions

    def _merge_group(self, members: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        member_ids = [member["row_id"] for member in members]
        canonical_id = _pick_canonical_id(member_ids)
        sorted_members = sorted(
            members,
            key=lambda member: (
                0 if member["row_id"] == canonical_id else 1,
                _source_authority_rank((member["row"].get("metadata") or {}).get("source") if isinstance(member["row"].get("metadata"), dict) else ""),
                member["row_id"],
            ),
        )
        merged = _sync_row_ids(sorted_members[0]["row"], canonical_id)
        slot_sources: dict[str, str] = {"id": canonical_id}
        for member in sorted_members[1:]:
            merged = _merge_values((), merged, _sync_row_ids(member["row"], canonical_id), slot_sources, member["row_id"])
        aliases = sorted(member_id for member_id in member_ids if member_id != canonical_id)
        if aliases:
            merged["aliases"] = sorted(set(list(merged.get("aliases", [])) + aliases))
        provenance = merged.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}
        provenance["sources"] = sorted(set(member_ids))
        merged["provenance"] = provenance
        metadata = merged.get("metadata")
        if isinstance(metadata, dict):
            confidence_sources = []
            for member in sorted_members:
                candidate_metadata = member["row"].get("metadata")
                if isinstance(candidate_metadata, dict):
                    source = _non_placeholder_text(candidate_metadata.get("source")) or member["row_id"]
                    confidence = _numeric_or_none(candidate_metadata.get("confidence"))
                    if confidence is not None:
                        confidence_sources.append({"source": source, "confidence": confidence})
            if confidence_sources:
                metadata["confidence_sources"] = sorted(confidence_sources, key=lambda item: (item["source"], item["confidence"]))
        galaxy = sorted_members[0]["galaxy"]
        file_name = sorted_members[0]["file_name"]
        if any(member["galaxy"] in WORD_CHARACTER_GALAXIES for member in sorted_members):
            preferred = sorted(
                (member for member in sorted_members if member["galaxy"] in WORD_CHARACTER_GALAXIES),
                key=lambda member: (member["galaxy"], member["file_name"], member["row_id"]),
            )
            galaxy = preferred[0]["galaxy"]
            file_name = preferred[0]["file_name"]
        merged["galaxy"] = galaxy
        join_map = {
            "reproduction_command": REPRO_COMMAND,
            "merged_row_id": canonical_id,
            "galaxy": galaxy,
            "member_ids": sorted(member_ids),
            "aliases": aliases,
            "slot_sources": slot_sources,
        }
        return merged, join_map

    def _apply_b2_additive_dedup(self, connection: sqlite3.Connection, merged_output_path: Path, join_map_path: Path) -> dict[str, int]:
        connection.execute("DELETE FROM merged_rows")
        merged_rows: list[dict[str, Any]] = []
        join_map_rows: list[dict[str, Any]] = []
        source_total = 0
        covered_ids: set[str] = set()
        groups = connection.execute(
            """
            SELECT group_key
            FROM source_rows
            GROUP BY group_key
            ORDER BY group_key
            """
        )
        for (group_key,) in groups:
            members: list[dict[str, Any]] = []
            cursor = connection.execute(
                """
                SELECT row_id, galaxy, file_name, row_json
                FROM source_rows
                WHERE group_key = ?
                ORDER BY row_id, galaxy, file_name
                """,
                (group_key,),
            )
            for row_id, galaxy, file_name, row_json in cursor:
                members.append(
                    {
                        "row_id": row_id,
                        "galaxy": galaxy,
                        "file_name": file_name,
                        "row": json.loads(row_json),
                    }
                )
            source_total += len(members)
            merged, join_map = self._merge_group(members)
            covered_ids.update(join_map["member_ids"])
            merged_rows.append(merged)
            join_map_rows.append(join_map)
            connection.execute(
                """
                INSERT INTO merged_rows (row_id, galaxy, file_name, row_json, member_ids_json, member_count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    merged["id"],
                    merged["galaxy"],
                    join_map["galaxy"] + ".jsonl",
                    json.dumps(merged, ensure_ascii=False, sort_keys=True),
                    json.dumps(join_map["member_ids"], ensure_ascii=False, sort_keys=True),
                    len(join_map["member_ids"]),
                ),
            )
        connection.commit()
        merged_rows.sort(key=lambda row: (row.get("galaxy", ""), row.get("id", "")))
        join_map_rows.sort(key=lambda row: (row["galaxy"], row["merged_row_id"]))
        _write_jsonl(merged_output_path, merged_rows)
        _write_jsonl(join_map_path, join_map_rows)
        duplicate_groups = 0
        absorbed_rows = 0
        for row in join_map_rows:
            member_count = len(row["member_ids"])
            if member_count > 1:
                duplicate_groups += 1
                absorbed_rows += member_count
        return {
            "source_row_count": source_total,
            "merged_row_count": len(merged_rows),
            "duplicate_groups": duplicate_groups,
            "duplicate_row_count": absorbed_rows,
            "dedup_coverage_count": len(covered_ids),
        }

    def _apply_b3_raw_to_rpn(self, connection: sqlite3.Connection, upgrades_path: Path, queue_path: Path) -> dict[str, int]:
        _load_rpn_smoke_module()
        connection.execute("DELETE FROM final_rows")
        upgrade_rows: list[dict[str, Any]] = []
        queue_rows: list[dict[str, Any]] = []
        success_count = 0
        queued_count = 0
        raw_rows = 0
        for row_id, galaxy, file_name, row_json in connection.execute(
            "SELECT row_id, galaxy, file_name, row_json FROM merged_rows ORDER BY galaxy, row_id"
        ):
            row = json.loads(row_json)
            procedural = _collect_procedural_payload(row)
            if procedural:
                connection.execute(
                    "INSERT INTO final_rows (row_id, galaxy, file_name, row_json) VALUES (?, ?, ?, ?)",
                    (row_id, galaxy, file_name, json.dumps(row, ensure_ascii=False, sort_keys=True)),
                )
                continue
            raw_rows += 1
            payload_type = _payload_type_for_row(row)
            succeeded, converted = _convert_raw_row(row)
            if succeeded and _collect_procedural_payload(converted):
                success_count += 1
                procedural_payload = _collect_procedural_payload(converted)
                chosen_field = next(iter(sorted(procedural_payload)))
                upgrade_rows.append(
                    {
                        "reproduction_command": REPRO_COMMAND,
                        "row_id": row_id,
                        "galaxy": galaxy,
                        "payload_type": payload_type,
                        "conversion_status": "success",
                        "target_field": chosen_field,
                        "program": procedural_payload[chosen_field],
                    }
                )
                connection.execute(
                    "INSERT INTO final_rows (row_id, galaxy, file_name, row_json) VALUES (?, ?, ?, ?)",
                    (row_id, galaxy, file_name, json.dumps(converted, ensure_ascii=False, sort_keys=True)),
                )
                continue
            queued_count += 1
            original_payload_hash = hashlib.sha256(_stable_json(row).encode("utf-8")).hexdigest()
            queue_rows.append(
                {
                    "reproduction_command": REPRO_COMMAND,
                    "source_row_id": row_id,
                    "payload_type": payload_type,
                    "failure_reason": f"{payload_type}_parse" if payload_type != "unknown" else "unknown_type",
                    "target_phase": "next_proceduralization_phase",
                    "original_payload_hash": original_payload_hash,
                }
            )
            upgrade_rows.append(
                {
                    "reproduction_command": REPRO_COMMAND,
                    "row_id": row_id,
                    "galaxy": galaxy,
                    "payload_type": payload_type,
                    "conversion_status": "queued",
                    "target_field": "",
                    "program": "",
                }
            )
        connection.commit()
        upgrade_rows.sort(key=lambda row: (row["conversion_status"], row["galaxy"], row["row_id"]))
        queue_rows.sort(key=lambda row: (row["payload_type"], row["source_row_id"]))
        _write_jsonl(upgrades_path, upgrade_rows)
        _write_jsonl(queue_path, queue_rows)
        return {
            "raw_row_count": raw_rows,
            "success_count": success_count,
            "queued_count": queued_count,
        }

    def _apply_b4_matryoshka(self, connection: sqlite3.Connection, bin_path: Path) -> dict[str, int]:
        pending_rows: list[dict[str, Any]] = []
        token_lists: list[list[str]] = []
        for row_id, galaxy, file_name, row_json in connection.execute(
            "SELECT row_id, galaxy, file_name, row_json FROM final_rows ORDER BY galaxy, row_id"
        ):
            row = json.loads(row_json)
            if galaxy not in WORD_CHARACTER_GALAXIES:
                continue
            if _has_matryoshka_payload(row):
                continue
            program = _program_for_embedding(row)
            tokens = _tokenize_for_embedding(program)
            pending_rows.append({"row_id": row_id, "galaxy": galaxy, "file_name": file_name, "row": row, "tokens": tokens})
            token_lists.append(tokens)
        accumulator = MatryoshkaAccumulator(
            source_path=self.kernel_source_path,
            ptx_path=self.stage_dir / "matryoshka_accumulator.ptx",
            basis_source_path=self.basis_kernel_source_path,
            basis_ptx_path=self.stage_dir / "procedural_basis.ptx",
        )
        generated_count = 0
        prefix_sample_count = 0
        prefix_failures = 0
        if pending_rows:
            accumulator.prepare_vocab(token_lists)
            accumulator.load()
            with bin_path.open("wb") as handle:
                for item in pending_rows:
                    vector = accumulator.embed(item["tokens"])
                    offset = handle.tell()
                    handle.write(vector)
                    row = item["row"]
                    matryoshka = row.get("matryoshka")
                    if not isinstance(matryoshka, dict):
                        matryoshka = {}
                    matryoshka["storage"] = "matryoshka_embeddings.bin"
                    matryoshka["dtype"] = "int8"
                    matryoshka["offset"] = offset
                    matryoshka["length"] = MATRYOSHKA_DIM
                    for dim in MATRYOSHKA_DIMENSIONS:
                        descriptor = {
                            "storage": "matryoshka_embeddings.bin",
                            "offset": offset,
                            "length": dim,
                            "prefix_of": 2048,
                        }
                        matryoshka[f"dim_{dim}"] = descriptor
                        row[f"embedding_{dim}"] = descriptor
                    row["matryoshka"] = matryoshka
                    connection.execute(
                        "UPDATE final_rows SET row_json = ? WHERE row_id = ?",
                        (json.dumps(row, ensure_ascii=False, sort_keys=True), item["row_id"]),
                    )
                    generated_count += 1
                    if prefix_sample_count < 1000:
                        prefix_sample_count += 1
                        if vector[:64] != vector[:2048][:64] or vector[:128] != vector[:2048][:128] or vector[:512] != vector[:2048][:512]:
                            prefix_failures += 1
            connection.commit()
            accumulator.close()
        else:
            bin_path.parent.mkdir(parents=True, exist_ok=True)
            bin_path.write_bytes(b"")
        return {
            "generated_count": generated_count,
            "prefix_sample_count": prefix_sample_count,
            "prefix_failures": prefix_failures,
        }

    def _write_final_rows(self, connection: sqlite3.Connection, merged_stars_path: Path, split_dir: Path) -> int:
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
        grouped: dict[str, list[dict[str, Any]]] = {}
        merged_rows: list[dict[str, Any]] = []
        for galaxy, file_name, row_json in connection.execute(
            "SELECT galaxy, file_name, row_json FROM final_rows ORDER BY galaxy, row_id"
        ):
            row = json.loads(row_json)
            merged_rows.append(row)
            grouped.setdefault(file_name, []).append(row)
        _write_jsonl(merged_stars_path, merged_rows)
        for file_name, rows in sorted(grouped.items()):
            output_path = split_dir / file_name
            rows.sort(key=lambda row: str(row.get("id", "")))
            _write_jsonl(output_path, rows)
        return len(merged_rows)

    def _render_final_report(self, run_dir: Path, stats: dict[str, int], manifest: list[tuple[str, str]]) -> str:
        audit_census_path = run_dir / "re_audit_d3" / "galaxy_census.jsonl"
        audit_violations_path = run_dir / "re_audit_d3" / "violations.jsonl"
        census_rows = _read_jsonl(audit_census_path)
        totals = Counter()
        for row in census_rows:
            for key in (
                "entry_count",
                "duplicate_row_count",
                "matryoshka_missing_count",
                "raw_payload_count",
                "unidirectional_site_count",
            ):
                totals[key] += int(row.get(key, 0) or 0)
        violation_counts = Counter()
        for row in _iter_jsonl(audit_violations_path):
            violation_counts[row["violation_kind"]] += 1
        lines = [
            "# D3 FINAL REPORT",
            "",
            f"Reproduction command: `{REPRO_COMMAND}`",
            "",
            "## Scope",
            "",
            f"- D2 source root: `{self.normalized_input_dir}`",
            f"- D3 stage root: `{self.stage_dir}`",
            "",
            "## Batch Results",
            "",
            f"- B1 closure additions: {stats['b1_edge_additions']}",
            f"- B2 merged rows: {stats['b2_merged_row_count']} from {stats['b2_source_row_count']} D2 rows",
            f"- B2 dedup coverage: {stats['b2_dedup_coverage_count']}/{stats['b2_source_row_count']}",
            f"- B3 raw rows converted: {stats['b3_success_count']}",
            f"- B3 rows queued for next proceduralization phase: {stats['b3_queued_count']}",
            f"- B4 generated Matryoshka embeddings: {stats['b4_generated_count']}",
            f"- B4 prefix-property samples: {stats['b4_prefix_sample_count']} with failures={stats['b4_prefix_failures']}",
            "",
            "## Acceptance Gates",
            "",
            f"- `duplicate_row_count`: {totals['duplicate_row_count']}",
            f"- `missing_matryoshka`: {totals['matryoshka_missing_count']}",
            f"- `raw_payload`: {totals['raw_payload_count']}",
            f"- `unidirectional_site_count`: {totals['unidirectional_site_count']}",
            "",
            "## Violation Counts",
            "",
        ]
        for kind, count in sorted(violation_counts.items()):
            lines.append(f"- `{kind}`: {count}")
        lines.extend(
            [
                "",
                "## Artifact Hashes",
                "",
            ]
        )
        for rel_path, digest in manifest:
            lines.append(f"- `{rel_path}` `{digest}`")
        return "\n".join(lines) + "\n"

    def _build_hash_manifest(self, run_dir: Path) -> list[tuple[str, str]]:
        manifest_paths = [
            run_dir / "patched_symlink_edges.jsonl",
            run_dir / "B2_merged_stars.jsonl",
            run_dir / "dedup_join_map.jsonl",
            run_dir / "rpn_upgrades.jsonl",
            run_dir / "pending_proceduralization_queue.jsonl",
            run_dir / "matryoshka_embeddings.bin",
            run_dir / "merged_stars.jsonl",
            run_dir / "re_audit_d3" / "galaxy_census.jsonl",
            run_dir / "re_audit_d3" / "violations.jsonl",
            run_dir / "re_audit_d3" / "RE_AUDIT_REPORT.md",
        ]
        manifest = []
        for path in manifest_paths:
            manifest.append((str(path.relative_to(run_dir)), _sha256_path(path)))
        return manifest

    def _execute_single_run(self, run_dir: Path) -> tuple[dict[str, int], list[tuple[str, str]]]:
        self._reset_stage_dir(run_dir)
        db_path = run_dir / ".d3.sqlite3"
        connection = self._connect(db_path)
        try:
            self._init_db(connection)
            source_count = self._load_d2_source_rows(connection)
            b1_rows = self._apply_b1_bidirectional_closure(connection, run_dir / "patched_symlink_edges.jsonl")
            b2_stats = self._apply_b2_additive_dedup(
                connection,
                run_dir / "B2_merged_stars.jsonl",
                run_dir / "dedup_join_map.jsonl",
            )
            b3_stats = self._apply_b3_raw_to_rpn(
                connection,
                run_dir / "rpn_upgrades.jsonl",
                run_dir / "pending_proceduralization_queue.jsonl",
            )
            b4_stats = self._apply_b4_matryoshka(connection, run_dir / "matryoshka_embeddings.bin")
            final_count = self._write_final_rows(connection, run_dir / "merged_stars.jsonl", run_dir / "merged_by_galaxy")
        finally:
            connection.close()
            if db_path.exists():
                db_path.unlink()
        re_audit_dir = run_dir / "re_audit_d3"
        audit_run_scan(run_dir / "merged_by_galaxy", re_audit_dir)
        audit_render_report(run_dir / "merged_by_galaxy", re_audit_dir, re_audit_dir / "RE_AUDIT_REPORT.md")
        manifest = self._build_hash_manifest(run_dir)
        stats = {
            "source_count": source_count,
            "final_count": final_count,
            "b1_edge_additions": len(b1_rows),
            "b2_source_row_count": b2_stats["source_row_count"],
            "b2_merged_row_count": b2_stats["merged_row_count"],
            "b2_duplicate_groups": b2_stats["duplicate_groups"],
            "b2_duplicate_row_count": b2_stats["duplicate_row_count"],
            "b2_dedup_coverage_count": b2_stats["dedup_coverage_count"],
            "b3_raw_row_count": b3_stats["raw_row_count"],
            "b3_success_count": b3_stats["success_count"],
            "b3_queued_count": b3_stats["queued_count"],
            "b4_generated_count": b4_stats["generated_count"],
            "b4_prefix_sample_count": b4_stats["prefix_sample_count"],
            "b4_prefix_failures": b4_stats["prefix_failures"],
        }
        report_text = self._render_final_report(run_dir, stats, manifest)
        _write_text(run_dir / "D3_FINAL_REPORT.md", report_text)
        manifest.append(("D3_FINAL_REPORT.md", _sha256_path(run_dir / "D3_FINAL_REPORT.md")))
        hash_lines = [f"{digest}  {rel_path}" for rel_path, digest in manifest]
        _write_text(run_dir / "hashes.txt", "\n".join(hash_lines) + "\n")
        return stats, manifest

    def _compare_manifests(self, left: list[tuple[str, str]], right: list[tuple[str, str]]) -> None:
        left_map = dict(left)
        right_map = dict(right)
        if left_map != right_map:
            diffs = []
            for key in sorted(set(left_map) | set(right_map)):
                if left_map.get(key) != right_map.get(key):
                    diffs.append(f"{key}: {left_map.get(key)} != {right_map.get(key)}")
            raise RuntimeError("D3 determinism check failed:\n" + "\n".join(diffs))

    def _publish(self, source_dir: Path) -> None:
        if self.stage_dir.exists():
            shutil.rmtree(self.stage_dir)
        shutil.copytree(source_dir, self.stage_dir)

    def _finalize_published_stage(self, stats: dict[str, int]) -> None:
        re_audit_dir = self.stage_dir / "re_audit_d3"
        audit_run_scan(self.stage_dir / "merged_by_galaxy", re_audit_dir)
        audit_render_report(self.stage_dir / "merged_by_galaxy", re_audit_dir, re_audit_dir / "RE_AUDIT_REPORT.md")
        manifest = self._build_hash_manifest(self.stage_dir)
        report_text = self._render_final_report(self.stage_dir, stats, manifest)
        _write_text(self.stage_dir / "D3_FINAL_REPORT.md", report_text)
        manifest.append(("D3_FINAL_REPORT.md", _sha256_path(self.stage_dir / "D3_FINAL_REPORT.md")))
        hash_lines = [f"{digest}  {rel_path}" for rel_path, digest in manifest]
        _write_text(self.stage_dir / "hashes.txt", "\n".join(hash_lines) + "\n")
        shutil.copyfile(self.stage_dir / "D3_FINAL_REPORT.md", self.temp_report_path)

    def run(self, *, verify_twice: bool) -> None:
        primary_stats, primary_manifest = self._execute_single_run(self.temp_run_dir)
        if verify_twice:
            _, verify_manifest = self._execute_single_run(self.temp_run_dir)
            self._compare_manifests(primary_manifest, verify_manifest)
        self._publish(self.temp_run_dir)
        self._finalize_published_stage(primary_stats)
        if self.temp_run_dir.exists():
            shutil.rmtree(self.temp_run_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the D3 additive-dedup staging pipeline.")
    parser.add_argument("--d2-stage-dir", type=Path, default=DEFAULT_D2_STAGE_DIR)
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    parser.add_argument("--temp-report-path", type=Path, default=DEFAULT_TEMP_REPORT_PATH)
    parser.add_argument("--no-verify-twice", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = D3Pipeline(
        d2_stage_dir=args.d2_stage_dir,
        stage_dir=args.stage_dir,
        temp_report_path=args.temp_report_path,
    )
    pipeline.run(verify_twice=not args.no_verify_twice)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
