"""Runtime ingest helpers for external Galaxy surfaces.

These loaders keep heavy local data access out of the hot query module while
still making large local knowledge surfaces available to the sovereign GPU
runtime. They are intentionally read-only and cache their outputs per-process.
"""

from __future__ import annotations

from functools import lru_cache
import gzip
import json
from pathlib import Path
import zipfile
from typing import Any

import numpy as np


BOOKS_V5_CANDIDATES = (
    Path("/K3D/Knowledge3D.local/galaxies/books_v5_clean2"),
    Path("../Knowledge3D.local/galaxies/books_v5_clean2"),
)


def resolve_books_v5_root() -> Path | None:
    for candidate in BOOKS_V5_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _normalize_embedding(values: list[float]) -> list[float]:
    norm = sum(value * value for value in values) ** 0.5
    if norm <= 1e-8:
        return [0.0 for _ in values]
    return [float(value / norm) for value in values]


def _project_embedding128_to16(values: np.ndarray) -> list[float]:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.size == 0:
        return [0.0] * 16
    if vector.size < 16:
        padded = np.zeros(16, dtype=np.float32)
        padded[: vector.size] = vector
        return _normalize_embedding(padded.tolist())
    if vector.size == 16:
        return _normalize_embedding(vector.astype(np.float32).tolist())
    block = max(1, vector.size // 16)
    reduced: list[float] = []
    for offset in range(0, vector.size, block):
        chunk = vector[offset : offset + block]
        if chunk.size == 0:
            continue
        reduced.append(float(chunk.mean()))
        if len(reduced) == 16:
            break
    if len(reduced) < 16:
        reduced.extend(float(vector[i]) for i in range(len(reduced), min(vector.size, 16)))
    if len(reduced) < 16:
        reduced.extend([0.0] * (16 - len(reduced)))
    return _normalize_embedding(reduced[:16])


def _domain_to_runtime_galaxy(domain: str, artifact_type: str) -> str:
    lowered_domain = str(domain or "").strip().lower()
    lowered_type = str(artifact_type or "").strip().lower()
    if lowered_domain in {
        "physics",
        "chemistry",
        "biology",
        "reality",
        "science",
    }:
        return "Reality"
    if lowered_domain in {
        "math",
        "mathematics",
        "algebra",
        "calculus",
        "geometry",
        "trigonometry",
        "number_theory",
        "statistics",
        "combinatorics",
    }:
        return "Math"
    if lowered_domain in {"grammar", "language", "linguistics"}:
        return "Grammar"
    if lowered_type in {"formula", "theorem", "proof", "equation"}:
        return "Math"
    return "Reality"


def _artifact_query_anchor(row: dict[str, Any], metadata: dict[str, Any]) -> str:
    fields: list[str] = []
    for value in (
        row.get("name"),
        row.get("conclusion"),
        row.get("raw_block"),
        row.get("lhs"),
        row.get("rhs"),
        row.get("source"),
        metadata.get("title"),
        metadata.get("domain"),
    ):
        if isinstance(value, str) and value.strip():
            fields.append(value.strip())
    symbol_bindings = row.get("symbol_bindings")
    if isinstance(symbol_bindings, dict):
        for key, payload in symbol_bindings.items():
            if isinstance(payload, dict):
                meaning = str(payload.get("meaning", "")).strip()
                if meaning:
                    fields.append(f"{key} {meaning}")
    conditions = row.get("conditions")
    if isinstance(conditions, list):
        fields.extend(str(item).strip() for item in conditions if str(item).strip())
    return " ".join(fields).strip()


@lru_cache(maxsize=1)
def load_books_runtime_entries() -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    root = resolve_books_v5_root()
    grouped: dict[str, list[dict[str, Any]]] = {}
    stats = {"books": 0, "artifacts": 0}
    if root is None:
        return grouped, stats

    for book_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        artifacts_path = book_dir / "artifacts.jsonl"
        embeddings_path = book_dir / "embeddings_128.npy"
        metadata_path = book_dir / "metadata.json"
        if not artifacts_path.exists() or not embeddings_path.exists():
            continue
        metadata = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
        try:
            embeddings = np.load(embeddings_path, mmap_mode="r")
        except Exception:
            continue
        stats["books"] += 1
        with artifacts_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                page_number = int(row.get("page_number") or 1)
                embedding_index = page_number - 1
                if not (0 <= embedding_index < len(embeddings)):
                    embedding_index = idx
                if not (0 <= embedding_index < len(embeddings)):
                    break
                runtime_galaxy = _domain_to_runtime_galaxy(
                    str(row.get("domain") or metadata.get("domain") or ""),
                    str(row.get("artifact_type") or ""),
                )
                answer = (
                    str(row.get("conclusion") or "").strip()
                    or str(row.get("raw_block") or "").strip()
                    or str(row.get("name") or "").strip()
                )
                query_anchor = _artifact_query_anchor(row, metadata)
                entry = {
                    "id": str(row.get("artifact_id") or f"{book_dir.name}_{idx}"),
                    "name": str(row.get("name") or row.get("artifact_id") or f"{book_dir.name}_{idx}"),
                    "domain": str(row.get("domain") or metadata.get("domain") or runtime_galaxy).lower(),
                    "category": str(row.get("artifact_type") or "book_artifact"),
                    "content": answer,
                    "summary": answer,
                    "description": f"{row.get('artifact_type', 'artifact')} from {metadata.get('title') or book_dir.name}",
                    "embedding16": _project_embedding128_to16(embeddings[embedding_index]),
                    "metadata": {
                        "meaning_ref": str(row.get("artifact_id") or f"{book_dir.name}_{idx}"),
                        "subject": str(row.get("domain") or metadata.get("domain") or runtime_galaxy).lower(),
                        "subfield": str(metadata.get("title") or book_dir.name),
                        "book_id": str(row.get("book_id") or metadata.get("book_id") or book_dir.name),
                        "book_title": str(metadata.get("title") or book_dir.name),
                        "artifact_type": str(row.get("artifact_type") or "book_artifact"),
                        "artifact_source": "books_v5_clean2",
                        "query_anchor": query_anchor,
                        "answer": answer,
                        "conditions": list(row.get("conditions", []))
                        if isinstance(row.get("conditions"), list)
                        else [],
                        "conditions_rpn": list(row.get("conditions_rpn", []))
                        if isinstance(row.get("conditions_rpn"), list)
                        else [],
                        "formula_rpn": str(row.get("rpn") or "").strip(),
                        "conclusion_rpn": str(row.get("conclusion_rpn") or "").strip(),
                        "lhs": str(row.get("lhs") or "").strip(),
                        "rhs": str(row.get("rhs") or "").strip(),
                        "symbol_bindings": row.get("symbol_bindings") if isinstance(row.get("symbol_bindings"), dict) else {},
                        "var_mapping": row.get("var_mapping") if isinstance(row.get("var_mapping"), dict) else {},
                        "source": str(row.get("source") or "").strip(),
                        "page_number": row.get("page_number"),
                        "page_embedding_index": embedding_index,
                        "confidence": 0.84,
                    },
                    "runtime_galaxy": runtime_galaxy,
                }
                grouped.setdefault(runtime_galaxy, []).append(entry)
                stats["artifacts"] += 1
    return grouped, stats


def _word_entry(
    *,
    entry_id: str,
    lemma: str,
    language: str,
    definition: str,
    forms: list[str],
    sources: list[str],
    pos: list[str] | None = None,
    symlinks: list[str] | None = None,
) -> dict[str, Any]:
    pos_list = [str(item).strip() for item in (pos or []) if str(item).strip()]
    form_list = [str(item).strip() for item in forms if str(item).strip()]
    return {
        "id": entry_id,
        "name": lemma,
        "domain": "word",
        "category": "multilingual_word",
        "content": definition or lemma,
        "summary": definition or lemma,
        "description": f"{language} lexical entry for {lemma}",
        "metadata": {
            "meaning_ref": entry_id,
            "subject": "language",
            "subfield": language,
            "language": language,
            "word_id": entry_id,
            "char_sequence": lemma,
            "definition": definition or lemma,
            "forms": form_list,
            "sources": list(dict.fromkeys(sources)),
            "parts_of_speech": pos_list,
            "symlinks": list(dict.fromkeys(symlinks or [])),
            "query_anchor": " ".join(
                [lemma, language, definition or "", *form_list, *pos_list]
            ).strip(),
            "answer": definition or lemma,
            "confidence": 0.81,
        },
    }


def _grammar_entry(language: str, description: str, aliases: list[str], sources: list[str]) -> dict[str, Any]:
    entry_id = f"grammar_language_{language}_core"
    return {
        "id": entry_id,
        "name": f"{language} grammar core",
        "domain": "grammar",
        "category": "language_rule",
        "content": description,
        "summary": description,
        "description": f"Core grammar routing entry for {language}.",
        "metadata": {
            "meaning_ref": entry_id,
            "subject": "language",
            "subfield": language,
            "language": language,
            "language_tags": [language],
            "aliases": aliases,
            "sources": sources,
            "query_anchor": " ".join([language, description, *aliases]).strip(),
            "answer": description,
            "confidence": 0.8,
        },
    }


@lru_cache(maxsize=1)
def load_language_runtime_entries() -> dict[str, list[dict[str, Any]]]:
    datasets_root = Path("/K3D/Knowledge3D.local/datasets")
    entries_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    def upsert(
        *,
        lemma: str,
        language: str,
        definition: str,
        forms: list[str],
        source: str,
        pos: list[str] | None = None,
    ) -> None:
        key = (language, lemma)
        existing = entries_by_key.get(key)
        if existing is None:
            entries_by_key[key] = {
                "lemma": lemma,
                "language": language,
                "definition": definition,
                "forms": list(dict.fromkeys(forms + [lemma])),
                "sources": [source],
                "pos": list(dict.fromkeys(pos or [])),
            }
            return
        if definition and not existing["definition"]:
            existing["definition"] = definition
        existing["forms"] = list(dict.fromkeys(existing["forms"] + forms + [lemma]))
        existing["sources"] = list(dict.fromkeys(existing["sources"] + [source]))
        existing["pos"] = list(dict.fromkeys(existing["pos"] + (pos or [])))

    spanish_path = datasets_root / "lexicons" / "spanish" / "kaikki.org-dictionary-Spanish.jsonl"
    if spanish_path.exists():
        with spanish_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if idx >= 256:
                    break
                row = json.loads(line)
                lemma = str(row.get("word") or "").strip()
                if not lemma:
                    continue
                senses = row.get("senses") if isinstance(row.get("senses"), list) else []
                glosses = senses[0].get("glosses") if senses and isinstance(senses[0], dict) else []
                definition = str(glosses[0]).strip() if glosses else ""
                upsert(
                    lemma=lemma,
                    language="es",
                    definition=definition,
                    forms=[],
                    source="kaikki_es",
                    pos=[str(row.get("pos") or "").strip()],
                )

    portuguese_path = datasets_root / "lexicons" / "portuguese_br" / "kaikki.org-dictionary-Portuguese.jsonl.gz"
    if portuguese_path.exists():
        with gzip.open(portuguese_path, "rt", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if idx >= 256:
                    break
                row = json.loads(line)
                lemma = str(row.get("word") or "").strip()
                if not lemma:
                    continue
                senses = row.get("senses") if isinstance(row.get("senses"), list) else []
                glosses = senses[0].get("glosses") if senses and isinstance(senses[0], dict) else []
                definition = str(glosses[0]).strip() if glosses else ""
                forms = [
                    str(item.get("form")).strip()
                    for item in row.get("forms", [])
                    if isinstance(item, dict) and str(item.get("form")).strip()
                ]
                upsert(
                    lemma=lemma,
                    language="pt",
                    definition=definition,
                    forms=forms,
                    source="kaikki_pt",
                    pos=[str(row.get("pos") or "").strip()],
                )

    cedict_path = datasets_root / "lexicons" / "zh" / "cedict_1_0_ts_utf-8_mdbg.zip"
    if cedict_path.exists():
        with zipfile.ZipFile(cedict_path) as archive:
            names = archive.namelist()
            if names:
                with archive.open(names[0]) as handle:
                    loaded = 0
                    for raw in handle:
                        line = raw.decode("utf-8", "ignore").strip()
                        if not line or line.startswith("#"):
                            continue
                        if "/" not in line or "[" not in line:
                            continue
                        parts = line.split("/")
                        if len(parts) < 2:
                            continue
                        head = parts[0].strip()
                        gloss = parts[1].strip()
                        segments = head.split()
                        if len(segments) < 2:
                            continue
                        lemma = segments[1].strip()
                        if not lemma:
                            continue
                        upsert(
                            lemma=lemma,
                            language="zh",
                            definition=gloss,
                            forms=[segments[0].strip()],
                            source="cedict_zh",
                            pos=["lexicon"],
                        )
                        loaded += 1
                        if loaded >= 256:
                            break

    word_entries: list[dict[str, Any]] = []
    surfaces: dict[str, list[str]] = {}
    for payload in entries_by_key.values():
        entry_id = f"word_{payload['language']}_{payload['lemma']}"
        word_entries.append(
            _word_entry(
                entry_id=entry_id,
                lemma=payload["lemma"],
                language=payload["language"],
                definition=payload["definition"],
                forms=payload["forms"],
                sources=payload["sources"],
                pos=payload["pos"],
            )
        )
        for form in payload["forms"]:
            normalized = form.lower()
            surfaces.setdefault(normalized, []).append(entry_id)

    if word_entries:
        for entry in word_entries:
            metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            forms = metadata.get("forms") if isinstance(metadata.get("forms"), list) else []
            symlinks: list[str] = []
            for form in forms:
                symlinks.extend(surfaces.get(str(form).lower(), []))
            symlinks = [item for item in dict.fromkeys(symlinks) if item != entry["id"]]
            metadata["symlinks"] = symlinks

    grammar_entries = [
        _grammar_entry(
            "en",
            "English grammar routing and lexical lookup surface.",
            ["english grammar", "english lexicon", "en words"],
            ["runtime_language_enrichment"],
        ),
        _grammar_entry(
            "es",
            "Spanish grammar routing and lexical lookup surface.",
            ["spanish grammar", "spanish lexicon", "es words"],
            ["kaikki_es"],
        ),
        _grammar_entry(
            "pt",
            "Portuguese grammar routing and lexical lookup surface.",
            ["portuguese grammar", "portuguese lexicon", "pt words"],
            ["kaikki_pt"],
        ),
        _grammar_entry(
            "zh",
            "Chinese grammar routing and lexical lookup surface.",
            ["chinese grammar", "chinese lexicon", "zh words"],
            ["cedict_zh"],
        ),
    ]

    return {
        "Word": word_entries,
        "Grammar": grammar_entries,
    }
