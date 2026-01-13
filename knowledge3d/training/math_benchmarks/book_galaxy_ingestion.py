"""
Book → Galaxy ingestion (Phase: multi-benchmark coverage expansion).

This module implements a *minimal*, sovereignty-aligned ingestion scaffold for
turning math books (PDFs / pre-extracted JSON page dumps) into "Book Galaxies"
that TRM can later navigate.

Vocabulary alignment (see `docs/vocabulary/`):
- Dual Client Contract: shared reality, form + meaning.
- Foundational Knowledge Spec: symlink pattern (do not duplicate glyphs).
- K3D Node Spec: procedural-first, meaning-first identity.

Important constraints:
- Ingestion is *not* hot path: it may use richer tooling, but **outputs must be
  written to `K3D_LOCAL_DIR` / Knowledge3D.local and not committed**.
- Inference remains sovereign: PTX + RPN + Galaxy only.

Current scope (intentionally small):
- Ingest book pages from an existing JSON export (list[{page, content}]) OR a
  plain-text file.
- Tokenise pages via WordGalaxy ("reading" primitive) and store char-codepoint
  sequences as symlink-style form references.
- Produce per-page 128D embeddings (RPNEmbeddingEngine CPU prototype) and stable
  3D positions derived from embeddings (no swarm dependency).
- Skip OCR-only PDFs for now by requiring text-bearing inputs (JSON/text).

Future work (out of scope here):
- PDF parsing via PyMuPDF bridge (requires `pymupdf`) and image → VectorDotMap.
- Theorem/definition extraction into structured knowledge graphs.
- Consolidation into GLB (`extras.k3d`) or House objects during SleepTime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime as _dt
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine
from knowledge3d.cranium.word_galaxy import WordGalaxy
from knowledge3d.training.math_benchmarks.rpn_parser import RPNParser


def _utc_now_iso() -> str:
    # Use timezone-aware UTC datetime (Python 3.13 deprecates naive utcnow()).
    # Python 3.11+ provides `datetime.UTC`; fall back to `timezone.utc`.
    tz = getattr(_dt, "UTC", _dt.timezone.utc)
    return _dt.datetime.now(tz).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slugify(value: str) -> str:
    keep: List[str] = []
    for ch in value.strip().lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in {" ", "_", "-", "."}:
            keep.append("_")
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "book"


def _default_local_dir() -> Path:
    """
    Return the local (non-repo) output directory.

    Prefer `K3D_LOCAL_DIR` when present, otherwise fall back to the canonical
    sibling folder `../Knowledge3D.local` relative to the repo root.
    """
    env = os.getenv("K3D_LOCAL_DIR")
    if env:
        return Path(env)
    return (Path(__file__).resolve().parents[4] / "Knowledge3D.local").resolve()


_PAGE_TEXT_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    # Some PDF text extractors (including Poppler's pdftotext) can emit control
    # characters in place of punctuation. "\x03" has been observed to represent
    # an equals sign in some textbooks (e.g., Euler formula lines).
    ("\x03", "="),
    # Common unicode normalization for PDF exports.
    ("−", "-"),
    ("–", "-"),
    ("—", "-"),
    ("×", "*"),
    ("∙", "*"),
    ("·", "*"),
    ("÷", "/"),
    ("∕", "/"),
    ("∗", "*"),
    ("＝", "="),
    ("≈", "="),
    ("≃", "="),
    ("≅", "="),
    ("≤", "<="),
    ("≥", ">="),
)


def _normalize_page_text(text: str) -> str:
    out = str(text or "")
    out = out.replace("\r", " ")
    for a, b in _PAGE_TEXT_REPLACEMENTS:
        out = out.replace(a, b)
    return out.strip()


@dataclass(frozen=True)
class BookGalaxyMetadata:
    book_id: str
    title: str
    author: Optional[str] = None
    domain: Optional[str] = None
    source_path: Optional[str] = None
    ingestion_utc: str = field(default_factory=_utc_now_iso)
    page_count: int = 0
    template_count: int = 0
    artifact_count: int = 0
    schema_version: str = "book_galaxy_v0"


@dataclass(frozen=True)
class BookGalaxyPage:
    page_number: int
    text: str
    token_count: int
    tokens: List[Dict[str, Any]]
    embedding_index: int
    position_3d: Tuple[float, float, float]


class BookGalaxyIngester:
    """
    Minimal book ingester: JSON/text → BookGalaxy artifacts under K3D_LOCAL_DIR.
    """

    def __init__(
        self,
        *,
        local_dir: Optional[Path] = None,
        word_galaxy: Optional[WordGalaxy] = None,
        embedding_engine: Optional[RPNEmbeddingEngine] = None,
        build_token_index: bool = True,
        token_index_min_token_len: int = 3,
        max_token_index_keys: int = 100_000,
        max_pages_per_token: int = 64,
        build_template_index: bool = True,
        template_index_min_token_len: int = 3,
        max_template_index_keys: int = 50_000,
        max_templates_per_token: int = 64,
        build_artifact_index: bool = True,
        artifact_index_min_token_len: int = 3,
        max_artifact_index_keys: int = 50_000,
        max_artifacts_per_token: int = 64,
    ) -> None:
        self.local_dir = Path(local_dir) if local_dir else _default_local_dir()
        self.word_galaxy = word_galaxy or WordGalaxy()
        self.embedding_engine = embedding_engine or RPNEmbeddingEngine(embedding_dim=128)
        self._rpn_parser = RPNParser()
        self._build_token_index = bool(build_token_index)
        self._token_index_min_token_len = max(1, int(token_index_min_token_len))
        self._max_token_index_keys = max(0, int(max_token_index_keys))
        self._max_pages_per_token = max(0, int(max_pages_per_token))
        self._build_template_index = bool(build_template_index)
        self._template_index_min_token_len = max(1, int(template_index_min_token_len))
        self._max_template_index_keys = max(0, int(max_template_index_keys))
        self._max_templates_per_token = max(0, int(max_templates_per_token))
        self._build_artifact_index = bool(build_artifact_index)
        self._artifact_index_min_token_len = max(1, int(artifact_index_min_token_len))
        self._max_artifact_index_keys = max(0, int(max_artifact_index_keys))
        self._max_artifacts_per_token = max(0, int(max_artifacts_per_token))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def ingest_json_pages(
        self,
        *,
        json_path: str | Path,
        title: Optional[str] = None,
        author: Optional[str] = None,
        domain: Optional[str] = None,
        book_id: Optional[str] = None,
        max_pages: Optional[int] = None,
        out_dir: Optional[str | Path] = None,
        lenient: bool = True,
    ) -> Path:
        """
        Ingest an extracted JSON page dump (list[{page, content}] or list[str]).
        """
        json_path = Path(json_path)
        raw = _load_json_lenient(json_path) if lenient else json.loads(json_path.read_text(encoding="utf-8"))

        pages: List[Tuple[int, str]] = []
        if isinstance(raw, list):
            for idx, item in enumerate(raw):
                if isinstance(item, dict):
                    page_num = int(item.get("page", idx + 1))
                    content = str(item.get("content", ""))
                else:
                    page_num = idx + 1
                    content = str(item)
                pages.append((page_num, content))
        else:
            raise ValueError("JSON must be a list of pages (dicts or strings).")

        return self._ingest_pages(
            pages=pages,
            source_path=str(json_path),
            title=title or json_path.stem,
            author=author,
            domain=domain,
            book_id=book_id or _slugify(title or json_path.stem),
            max_pages=max_pages,
            out_dir=out_dir,
        )

    def ingest_json_dir(
        self,
        *,
        json_dir: str | Path,
        glob: str = "*.json",
        domain: Optional[str] = None,
        author: Optional[str] = None,
        max_books: Optional[int] = None,
        max_pages: Optional[int] = None,
        out_root: Optional[str | Path] = None,
        lenient: bool = True,
    ) -> List[Path]:
        """
        Ingest a directory of JSON page dumps.

        This is an ingestion convenience wrapper intended for bulk conversion of
        existing text exports (e.g., Advanced Maths JSON dumps) into per-book
        Book Galaxy artifacts under a common root directory.
        """
        json_dir = Path(json_dir)
        if not json_dir.exists():
            raise FileNotFoundError(str(json_dir))
        if not json_dir.is_dir():
            raise NotADirectoryError(str(json_dir))

        out_root_path = Path(out_root) if out_root else (self.local_dir / "galaxies" / "books")
        out_root_path.mkdir(parents=True, exist_ok=True)

        results: List[Path] = []
        skipped: List[str] = []
        limit = max(0, int(max_books)) if max_books is not None else None
        for idx, path in enumerate(sorted(json_dir.glob(glob))):
            if limit is not None and idx >= limit:
                break
            title = path.stem.replace("_", " ").replace(".", " ").strip()
            book_id = _slugify(path.stem)
            out_dir = out_root_path / book_id
            try:
                results.append(
                    self.ingest_json_pages(
                        json_path=path,
                        title=title,
                        author=author,
                        domain=domain,
                        book_id=book_id,
                        max_pages=max_pages,
                        out_dir=out_dir,
                        lenient=lenient,
                    )
                )
            except Exception as exc:
                detail = str(exc).strip().replace("\n", " ")
                if len(detail) > 120:
                    detail = detail[:117] + "..."
                skipped.append(f"{path.name}: {exc.__class__.__name__}({detail})")
                continue
        if skipped:
            # Ingestion is best-effort; report skipped sources for manual cleanup.
            msg = "\n".join(skipped[:20])
            if len(skipped) > 20:
                msg += f"\n... (+{len(skipped) - 20} more)"
            print(f"[BookGalaxyIngester] Skipped {len(skipped)} JSON files due to parse errors:\n{msg}")
        return results

    def ingest_text_file(
        self,
        *,
        text_path: str | Path,
        title: Optional[str] = None,
        author: Optional[str] = None,
        domain: Optional[str] = None,
        book_id: Optional[str] = None,
        out_dir: Optional[str | Path] = None,
        split_on_formfeed: bool = True,
        max_pages: Optional[int] = None,
    ) -> Path:
        """
        Ingest a plain text file as pages.

        If `split_on_formfeed` is True, splits on '\\f' (form feed) to approximate
        pages; otherwise the whole file becomes one page.
        """
        text_path = Path(text_path)
        text = text_path.read_text(encoding="utf-8", errors="replace")
        if split_on_formfeed and "\f" in text:
            chunks = text.split("\f")
        else:
            chunks = [text]
        pages = [(i + 1, chunk) for i, chunk in enumerate(chunks)]
        return self._ingest_pages(
            pages=pages,
            source_path=str(text_path),
            title=title or text_path.stem,
            author=author,
            domain=domain,
            book_id=book_id or _slugify(title or text_path.stem),
            max_pages=max_pages,
            out_dir=out_dir,
        )

    def ingest_pdf_pdftotext(
        self,
        *,
        pdf_path: str | Path,
        title: Optional[str] = None,
        author: Optional[str] = None,
        domain: Optional[str] = None,
        book_id: Optional[str] = None,
        out_dir: Optional[str | Path] = None,
        max_pages: Optional[int] = None,
        layout: bool = False,
    ) -> Path:
        """
        Ingest a text-bearing PDF using Poppler's `pdftotext`.

        This is ingestion-only (not hot path). OCR-only PDFs are intentionally
        out-of-scope for now, so a failing `pdftotext` extraction is treated as
        an error.
        """
        pdf_path = Path(pdf_path)
        args = ["pdftotext"]
        if layout:
            args.append("-layout")
        args.extend([str(pdf_path), "-"])
        proc = subprocess.run(args, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "").strip().replace("\n", " ")
            raise RuntimeError(f"pdftotext failed for {pdf_path}: {msg[:200]}")
        text = proc.stdout or ""
        chunks = text.split("\f") if "\f" in text else [text]
        pages = [(i + 1, chunk) for i, chunk in enumerate(chunks)]
        return self._ingest_pages(
            pages=pages,
            source_path=str(pdf_path),
            title=title or pdf_path.stem,
            author=author,
            domain=domain,
            book_id=book_id or _slugify(title or pdf_path.stem),
            max_pages=max_pages,
            out_dir=out_dir,
        )

    # ------------------------------------------------------------------ #
    # Core ingestion
    # ------------------------------------------------------------------ #
    def _ingest_pages(
        self,
        *,
        pages: Sequence[Tuple[int, str]],
        source_path: Optional[str],
        title: str,
        author: Optional[str],
        domain: Optional[str],
        book_id: str,
        max_pages: Optional[int],
        out_dir: Optional[str | Path],
    ) -> Path:
        pages = list(pages)
        if max_pages is not None:
            pages = pages[: max(0, int(max_pages))]

        if not any(text.strip() for _, text in pages):
            raise ValueError("No text found in pages (OCR-only sources are skipped in this phase).")

        output_dir = Path(out_dir) if out_dir else (self.local_dir / "galaxies" / "books" / book_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = BookGalaxyMetadata(
            book_id=book_id,
            title=title,
            author=author,
            domain=domain,
            source_path=source_path,
            page_count=len(pages),
        )

        # Build embeddings + positions (stable, deterministic).
        #
        # IMPORTANT (memory safety):
        # Do NOT accumulate full `page_records` in RAM. Each page contains a full
        # token payload with per-token `char_sequence`, which becomes enormous for
        # large books. Instead, stream-write `pages.jsonl` / `pages_text.jsonl`
        # while iterating pages.
        embeddings: List[np.ndarray] = []
        positions: List[Tuple[float, float, float]] = []
        # Token → pages index for cheap runtime lookup.
        #
        # IMPORTANT (memory safety):
        # A naive "token -> set(page_num)" can explode RAM for large books due to:
        #  - huge unique token counts (dictionary growth),
        #  - huge per-token sets (many pages),
        #  - per-token updates done per occurrence (time + memory).
        #
        # Use a bounded, append-only representation: token -> sorted list of page numbers.
        # This keeps memory predictable and prevents CPU OOM during ingestion.
        token_index_pages: Dict[str, List[int]] = {}
        token_index_truncated_keys = 0
        token_index_truncated_pages = 0
        template_count = 0
        template_index_sets: Dict[str, set[str]] = {}
        template_index_truncated_keys = 0
        template_index_truncated_ids = 0
        artifact_count = 0
        artifact_index_sets: Dict[str, set[str]] = {}
        artifact_index_truncated_keys = 0
        artifact_index_truncated_ids = 0

        # Optional Math Galaxy symlinks (ingestion-only; avoid hard dependency).
        try:
            from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
        except Exception:
            MATH_GALAXY = None

        pages_path = output_dir / "pages.jsonl"
        # Lightweight page text file for runtime query (avoid loading full token payloads).
        pages_text_path = output_dir / "pages_text.jsonl"
        templates_path = output_dir / "templates.jsonl"
        artifacts_path = output_dir / "artifacts.jsonl"

        templates_handle = None
        artifacts_handle = None

        # Optional articulated artifacts: structured theorem/definition/formula blocks.
        # This is ingestion-side enrichment; runtime consumers can choose to use it.
        articulator = None
        try:
            from knowledge3d.training.math_benchmarks.sovereign_knowledge_articulator import (
                OllamaRoleExtractor,
                RoleExtractionConfig,
                SovereignKnowledgeArticulator,
            )

            def _env_flag(name: str, default: bool = False) -> bool:
                raw = os.getenv(name)
                if raw is None:
                    return bool(default)
                return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

            def _env_first(*names: str) -> Optional[str]:
                for n in names:
                    v = os.getenv(n)
                    if v is not None and str(v).strip():
                        return str(v)
                return None

            def _env_int(name: str, default: int) -> int:
                raw = os.getenv(name)
                if raw is None or not raw.strip():
                    return int(default)
                try:
                    return int(float(raw.strip()))
                except Exception:
                    return int(default)

            def _env_float(name: str, default: float) -> float:
                raw = os.getenv(name)
                if raw is None or not raw.strip():
                    return float(default)
                try:
                    return float(raw.strip())
                except Exception:
                    return float(default)

            # Option A: ingestion-time semantic role extraction. Default is OFF so
            # tests and baseline ingestions do not require a local Ollama service.
            role_extractor = None
            enable_role_llm = _env_flag("K3D_BOOK_ROLE_LLM_ENABLE", _env_flag("K3D_ROLE_LLM_ENABLE", False))
            if enable_role_llm:
                cache_path = _env_first("K3D_BOOK_ROLE_LLM_CACHE_PATH", "K3D_ROLE_LLM_CACHE_PATH")
                if not cache_path:
                    cache_path = str(output_dir / "role_llm_cache.jsonl")
                cfg = RoleExtractionConfig(
                    enabled=True,
                    model=_env_first("K3D_BOOK_ROLE_LLM_MODEL", "K3D_ROLE_LLM_MODEL") or "granite4:tiny-h",
                    fallback_model=_env_first(
                        "K3D_BOOK_ROLE_LLM_FALLBACK_MODEL", "K3D_ROLE_LLM_FALLBACK_MODEL"
                    ) or "qwen2.5:14b",
                    timeout_s=_env_float("K3D_BOOK_ROLE_LLM_TIMEOUT_S", _env_float("K3D_ROLE_LLM_TIMEOUT_S", 30.0)),
                    max_context_chars=_env_int("K3D_BOOK_ROLE_LLM_MAX_CONTEXT_CHARS", _env_int("K3D_ROLE_LLM_MAX_CONTEXT_CHARS", 1200)),
                    cache_path=cache_path,
                    restart_between_calls=_env_flag("K3D_BOOK_ROLE_LLM_RESTART", _env_flag("K3D_ROLE_LLM_RESTART", False)),
                    prefer_http=_env_flag("K3D_BOOK_ROLE_LLM_PREFER_HTTP", _env_flag("K3D_ROLE_LLM_PREFER_HTTP", True)),
                    fallback_on_unknown=_env_flag(
                        "K3D_BOOK_ROLE_LLM_FALLBACK_ON_UNKNOWN",
                        _env_flag("K3D_ROLE_LLM_FALLBACK_ON_UNKNOWN", True),
                    ),
                    http_url=_env_first("K3D_BOOK_ROLE_LLM_HTTP_URL", "K3D_ROLE_LLM_HTTP_URL")
                    or "http://127.0.0.1:11434/api/generate",
                )
                role_extractor = OllamaRoleExtractor(config=cfg)

            articulator = SovereignKnowledgeArticulator(parser=self._rpn_parser, role_extractor=role_extractor)
        except Exception:
            articulator = None

        try:
            with pages_path.open("w", encoding="utf-8") as pages_handle, pages_text_path.open(
                "w", encoding="utf-8"
            ) as pages_text_handle:
                for embed_idx, (page_num, text) in enumerate(pages):
                    clean_text = _normalize_page_text(text)
                    tokens = self.word_galaxy.tokenize(clean_text)
                    token_payload: List[Dict[str, Any]] = []
                    page_index_keys: set[str] = set()
                    for tok in tokens:
                        char_seq = self.word_galaxy.compose_from_text(tok.token)
                        math_symbol_ref = None
                        if MATH_GALAXY is not None:
                            try:
                                entry = MATH_GALAXY.lookup(tok.token)
                                if entry is not None:
                                    math_symbol_ref = entry.symbol
                            except Exception:
                                math_symbol_ref = None
                        tok_dict: Dict[str, Any] = {
                            "token": tok.token,
                            "normalized": tok.normalized,
                            "category": tok.category,
                            "value": tok.value,
                            "rpn_literal": tok.rpn_literal,
                            "role": tok.role,
                            "char_sequence": char_seq,
                            "math_symbol_ref": math_symbol_ref,
                        }
                        token_payload.append(tok_dict)
                        if self._build_token_index:
                            key = tok.normalized
                            # Keep indexing permissive: the WordGalaxy uses richer categories
                            # (noun/verb/number/etc). Runtime retrieval is cheap and benefits from
                            # indexing those tokens too. Avoid pure symbol tokens like "=" or "*".
                            if (
                                key
                                and len(key) >= self._token_index_min_token_len
                                and re.search(r"[A-Za-z0-9]", key)
                            ):
                                page_index_keys.add(key)
                            # Symlink pattern: also index the canonical Math Galaxy symbol
                            # when we can resolve it (e.g., "cos" -> "\\cos").
                            if (
                                math_symbol_ref
                                and len(math_symbol_ref) >= self._token_index_min_token_len
                                and re.search(r"[A-Za-z0-9]", math_symbol_ref)
                            ):
                                page_index_keys.add(math_symbol_ref)

                    # Update token→pages index once per token per page (bounded).
                    if self._build_token_index and page_index_keys:
                        page_number_int = int(page_num)
                        for key in page_index_keys:
                            pages_list = token_index_pages.get(key)
                            if pages_list is None:
                                if self._max_token_index_keys and len(token_index_pages) >= self._max_token_index_keys:
                                    token_index_truncated_keys += 1
                                    continue
                                token_index_pages[key] = [page_number_int]
                                continue
                            # Avoid duplicates; pages are ingested in order.
                            if pages_list and pages_list[-1] == page_number_int:
                                continue
                            if self._max_pages_per_token and len(pages_list) >= self._max_pages_per_token:
                                token_index_truncated_pages += 1
                                continue
                            pages_list.append(page_number_int)

                    # Ingestion-time: extract equation templates (lhs = rhs) for runtime retrieval.
                    try:
                        page_templates = _extract_equation_templates(
                            clean_text,
                            parser=self._rpn_parser,
                            book_id=book_id,
                            page_number=int(page_num),
                            source=str(source_path or ""),
                            domain_hint=domain,
                        )
                    except Exception:
                        page_templates = []
                    if page_templates:
                        if templates_handle is None:
                            templates_handle = templates_path.open("w", encoding="utf-8")
                        for item in page_templates:
                            templates_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                            template_count += 1
                            template_id = str(item.get("template_id") or "")
                            if not template_id:
                                continue
                            # Index by tokens from both LHS (concept cue) and RHS (operation cue).
                            if self._build_template_index:
                                for part in (str(item.get("lhs") or ""), str(item.get("rhs") or "")):
                                    for t in self.word_galaxy.tokenize(part):
                                        norm = str(getattr(t, "normalized", "") or "")
                                        if (
                                            not norm
                                            or len(norm) < self._template_index_min_token_len
                                            or not re.search(r"[A-Za-z0-9]", norm)
                                        ):
                                            continue
                                        ids = template_index_sets.get(norm)
                                        if ids is None:
                                            if self._max_template_index_keys and len(template_index_sets) >= self._max_template_index_keys:
                                                template_index_truncated_keys += 1
                                                continue
                                            ids = set()
                                            template_index_sets[norm] = ids
                                        if self._max_templates_per_token and len(ids) >= self._max_templates_per_token:
                                            template_index_truncated_ids += 1
                                            continue
                                        ids.add(template_id)
                                        # Also index the canonical Math Galaxy symbol, if any.
                                        if MATH_GALAXY is not None:
                                            raw = str(getattr(t, "token", "") or "") or norm
                                            try:
                                                sym = MATH_GALAXY.lookup(raw)
                                            except Exception:
                                                sym = None
                                            if sym is not None:
                                                canon = str(sym.symbol or "")
                                                if (
                                                    canon
                                                    and len(canon) >= self._template_index_min_token_len
                                                    and re.search(r"[A-Za-z0-9]", canon)
                                                ):
                                                    ids2 = template_index_sets.get(canon)
                                                    if ids2 is None:
                                                        if (
                                                            self._max_template_index_keys
                                                            and len(template_index_sets) >= self._max_template_index_keys
                                                        ):
                                                            template_index_truncated_keys += 1
                                                        else:
                                                            ids2 = set()
                                                            template_index_sets[canon] = ids2
                                                    if ids2 is not None:
                                                        if self._max_templates_per_token and len(ids2) >= self._max_templates_per_token:
                                                            template_index_truncated_ids += 1
                                                        else:
                                                            ids2.add(template_id)

                    # Articulate knowledge artifacts per page (streaming).
                    if articulator is not None:
                        try:
                            page_artifacts = articulator.articulate_page(
                                page_number=int(page_num),
                                text=clean_text,
                                book_id=book_id,
                                domain=domain,
                            )
                        except Exception:
                            page_artifacts = []
                        if page_artifacts:
                            if artifacts_handle is None:
                                artifacts_handle = artifacts_path.open("w", encoding="utf-8")
                            for art in page_artifacts:
                                artifacts_handle.write(json.dumps(asdict(art), ensure_ascii=False) + "\n")
                                artifact_count += 1
                                # Index by terms likely to appear in user queries.
                                for part in (
                                    art.name,
                                    " ".join(art.conditions or []),
                                    str(art.lhs or ""),
                                    str(art.rhs or ""),
                                ):
                                    if not self._build_artifact_index:
                                        continue
                                    for t in self.word_galaxy.tokenize(part):
                                        norm = str(getattr(t, "normalized", "") or "")
                                        if (
                                            not norm
                                            or len(norm) < self._artifact_index_min_token_len
                                            or not re.search(r"[A-Za-z0-9]", norm)
                                        ):
                                            continue
                                        ids = artifact_index_sets.get(norm)
                                        if ids is None:
                                            if self._max_artifact_index_keys and len(artifact_index_sets) >= self._max_artifact_index_keys:
                                                artifact_index_truncated_keys += 1
                                                continue
                                            ids = set()
                                            artifact_index_sets[norm] = ids
                                        if self._max_artifacts_per_token and len(ids) >= self._max_artifacts_per_token:
                                            artifact_index_truncated_ids += 1
                                            continue
                                        ids.add(str(art.artifact_id))
                                        # Also index the canonical Math Galaxy symbol, if any.
                                        if MATH_GALAXY is not None:
                                            raw = str(getattr(t, "token", "") or "") or norm
                                            try:
                                                sym = MATH_GALAXY.lookup(raw)
                                            except Exception:
                                                sym = None
                                            if sym is not None:
                                                canon = str(sym.symbol or "")
                                                if (
                                                    canon
                                                    and len(canon) >= self._artifact_index_min_token_len
                                                    and re.search(r"[A-Za-z0-9]", canon)
                                                ):
                                                    ids2 = artifact_index_sets.get(canon)
                                                    if ids2 is None:
                                                        if (
                                                            self._max_artifact_index_keys
                                                            and len(artifact_index_sets) >= self._max_artifact_index_keys
                                                        ):
                                                            artifact_index_truncated_keys += 1
                                                        else:
                                                            ids2 = set()
                                                            artifact_index_sets[canon] = ids2
                                                    if ids2 is not None:
                                                        if self._max_artifacts_per_token and len(ids2) >= self._max_artifacts_per_token:
                                                            artifact_index_truncated_ids += 1
                                                        else:
                                                            ids2.add(str(art.artifact_id))

                    emb = self.embedding_engine.embed_sentence(clean_text)
                    embeddings.append(emb)

                    pos = _embedding_to_position(emb)
                    positions.append(pos)

                    page_record = BookGalaxyPage(
                        page_number=int(page_num),
                        text=clean_text,
                        token_count=len(token_payload),
                        tokens=token_payload,
                        embedding_index=embed_idx,
                        position_3d=pos,
                    )

                    pages_handle.write(json.dumps(asdict(page_record), ensure_ascii=False) + "\n")
                    pages_text_handle.write(
                        json.dumps(
                            {
                                "page_number": page_record.page_number,
                                "text": page_record.text,
                                "embedding_index": page_record.embedding_index,
                                "position_3d": page_record.position_3d,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        finally:
            if templates_handle is not None:
                templates_handle.close()
            if artifacts_handle is not None:
                artifacts_handle.close()

        # Persist metadata + optional indices.
        meta: Dict[str, Any] = asdict(metadata)
        meta["template_count"] = int(template_count)
        meta["artifact_count"] = int(artifact_count)
        meta["token_index_enabled"] = bool(self._build_token_index)
        meta["token_index_min_token_len"] = int(self._token_index_min_token_len)
        meta["token_index_max_keys"] = int(self._max_token_index_keys)
        meta["token_index_max_pages_per_token"] = int(self._max_pages_per_token)
        meta["token_index_truncated_keys"] = int(token_index_truncated_keys)
        meta["token_index_truncated_pages"] = int(token_index_truncated_pages)
        meta["template_index_enabled"] = bool(self._build_template_index)
        meta["template_index_min_token_len"] = int(self._template_index_min_token_len)
        meta["template_index_max_keys"] = int(self._max_template_index_keys)
        meta["template_index_max_templates_per_token"] = int(self._max_templates_per_token)
        meta["template_index_truncated_keys"] = int(template_index_truncated_keys)
        meta["template_index_truncated_template_ids"] = int(template_index_truncated_ids)
        meta["artifact_index_enabled"] = bool(self._build_artifact_index)
        meta["artifact_index_min_token_len"] = int(self._artifact_index_min_token_len)
        meta["artifact_index_max_keys"] = int(self._max_artifact_index_keys)
        meta["artifact_index_max_artifacts_per_token"] = int(self._max_artifacts_per_token)
        meta["artifact_index_truncated_keys"] = int(artifact_index_truncated_keys)
        meta["artifact_index_truncated_artifact_ids"] = int(artifact_index_truncated_ids)
        (output_dir / "metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if template_count > 0:
            if self._build_template_index and template_index_sets:
                tmpl_index = {k: sorted(v) for k, v in template_index_sets.items()}
                (output_dir / "template_index.json").write_text(
                    json.dumps(tmpl_index, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            else:
                try:
                    (output_dir / "template_index.json").unlink()
                except Exception:
                    pass
        else:
            # Avoid leaving empty optional files around.
            try:
                if templates_path.exists():
                    templates_path.unlink()
            except Exception:
                pass
            try:
                (output_dir / "template_index.json").unlink()
            except Exception:
                pass

        if artifact_count > 0:
            if self._build_artifact_index and artifact_index_sets:
                art_index = {k: sorted(v) for k, v in artifact_index_sets.items()}
                (output_dir / "artifact_index.json").write_text(
                    json.dumps(art_index, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            else:
                try:
                    (output_dir / "artifact_index.json").unlink()
                except Exception:
                    pass
        else:
            try:
                if artifacts_path.exists():
                    artifacts_path.unlink()
            except Exception:
                pass
            try:
                (output_dir / "artifact_index.json").unlink()
            except Exception:
                pass

        # Save arrays in binary form to keep disk size small.
        emb_mat = np.vstack(embeddings).astype(np.float32) if embeddings else np.zeros((0, 128), dtype=np.float32)
        pos_mat = np.asarray(positions, dtype=np.float32) if positions else np.zeros((0, 3), dtype=np.float32)
        np.save(output_dir / "embeddings_128.npy", emb_mat)
        np.save(output_dir / "positions_3d.npy", pos_mat)

        if self._build_token_index:
            token_index = {k: v for k, v in sorted(token_index_pages.items(), key=lambda kv: kv[0])}
        else:
            token_index = {}
        (output_dir / "token_index.json").write_text(
            json.dumps(token_index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return output_dir


_EQ_PATTERNS: Tuple[re.Pattern[str], ...] = (
    # Basic equations: "lhs = rhs"
    # Keep this intentionally permissive; RHS parsing filters non-math noise.
    re.compile(r"(?P<lhs>[^=\n]{1,80})\s*=\s*(?P<rhs>[^\n]{1,160})"),
    # Derivatives: d/dx(expr)=rhs
    re.compile(
        r"d/d(?P<var>[a-zA-Z])\s*\(?\s*(?P<lhs>[^\n]{1,80}?)\s*\)?\s*=\s*(?P<rhs>[^\n]{1,160}?)"
    ),
)


def _extract_equation_templates(
    text: str,
    *,
    parser: RPNParser,
    book_id: str,
    page_number: int,
    source: str,
    domain_hint: Optional[str],
    max_templates: int = 32,
) -> List[Dict[str, Any]]:
    """
    Extract parseable (lhs = rhs) templates from a single page of text.

    This is intentionally conservative: only templates whose RHS parses into a
    valid RPN program are emitted.
    """
    if not text:
        return []

    out: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for pat in _EQ_PATTERNS:
        for m in pat.finditer(text):
            lhs = str(m.groupdict().get("lhs") or "").strip()
            rhs = str(m.groupdict().get("rhs") or "").strip()
            if not lhs or not rhs:
                continue
            if len(lhs) > 80 or len(rhs) > 200:
                continue
            key = (lhs, rhs)
            if key in seen:
                continue
            try:
                rpn = parser.infix_to_rpn(rhs)
            except Exception:
                continue
            rpn = _normalize_template_rpn(rpn)
            if not _is_valid_template_rpn(rpn):
                continue
            seen.add(key)
            template_id = f"{book_id}_p{int(page_number)}_eq{len(out)}"
            out.append(
                {
                    "template_id": template_id,
                    "book_id": book_id,
                    "page_number": int(page_number),
                    "lhs": lhs,
                    "rhs": rhs,
                    "rpn": rpn,
                    "domain": domain_hint,
                    "source": source,
                }
            )
            if len(out) >= int(max_templates):
                return out

    return out


_TEMPLATE_OPS: set[str] = {
    # Arithmetic
    "+",
    "-",
    "*",
    "/",
    "pow",
    # Unary / functions
    "sqrt",
    "abs",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "log",
    "ln",
    "exp",
    "floor",
    "ceil",
    # Discrete / combinatorics (supported by ModularRPNEngine even if not in validator)
    "gcd",
    "factorial",
    "binomial",
}


def _normalize_template_rpn(program: str) -> str:
    """
    Convert multi-letter identifiers into single-letter placeholders.

    Rationale:
    - Runtime execution expects numeric literals + opcodes; during TTC we can bind
      variables to numbers. Keeping variables single-letter maintains compatibility
      with the existing RPN validator and avoids embedding book-specific token names
      into the hot-path surface area.
    """
    tokens = [t for t in str(program or "").strip().split() if t]
    if not tokens:
        return ""

    mapping: Dict[str, str] = {}
    pool = list("xyzabcdefghijklmnopqrstuvw")  # stable order, avoids common i/j confusion
    out: List[str] = []

    for tok in tokens:
        lower = tok.lower()
        # Numbers
        try:
            float(tok)
            out.append(tok)
            continue
        except Exception:
            pass
        # Known opcodes
        if lower in _TEMPLATE_OPS:
            out.append(lower)
            continue
        # Already a single-letter variable
        if len(tok) == 1 and tok.isalpha():
            out.append(tok.lower())
            continue
        # Identifier → map to a single letter
        key = tok.strip()
        if key not in mapping:
            if not pool:
                return ""
            mapping[key] = pool.pop(0)
        out.append(mapping[key])

    return " ".join(out)


def _is_valid_template_rpn(program: str) -> bool:
    """
    Slightly stricter-than-runtime validation for stored templates.

    We only store templates that are likely executable after variable binding.
    """
    program = str(program or "").strip()
    if not program:
        return False
    for tok in program.split():
        lower = tok.lower()
        try:
            float(tok)
            continue
        except Exception:
            pass
        if lower in _TEMPLATE_OPS:
            continue
        if len(tok) == 1 and tok.isalpha():
            continue
        return False
    return True


def _embedding_to_position(embedding_128: np.ndarray) -> Tuple[float, float, float]:
    """
    Map a 128D embedding to a stable 3D coordinate.

    This mirrors `LanguageSwarmProcessor._embedding_to_position` but keeps this
    module dependency-light (no swarm bridge).
    """
    embedding = np.asarray(embedding_128, dtype=np.float32).reshape(-1)
    if embedding.shape[0] != 128:
        raise ValueError("Expected embedding of shape (128,)")
    bins = np.array_split(embedding, 3)
    coords = np.array([segment.mean() for segment in bins], dtype=np.float32)
    coords -= coords.min()
    denom = float(coords.max()) or 1.0
    coords = coords / denom
    return (float(coords[0]), float(coords[1]), float(coords[2]))


def _load_json_lenient(path: Path) -> Any:
    """
    Load a JSON file with a minimal "escape repair" fallback.

    Some of the legacy text exports contain backslashes that are not valid JSON
    escapes (e.g., LaTeX fragments like ``\\alpha`` or ``\\c`` inside strings). We
    repair these by doubling backslashes that are not part of a valid escape.

    This is ingestion-only and keeps the pipeline usable without requiring a
    re-export of large libraries.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = re.sub(r"\\(?![\\\\/\"bfnrtu])", r"\\\\", text)
        return json.loads(repaired)


def _build_arg_parser():
    import argparse

    ap = argparse.ArgumentParser(description="Ingest a math book into a Book Galaxy (local artifacts)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--json", dest="json_path", help="Path to JSON page dump (list[{page, content}] or list[str])")
    src.add_argument("--json-dir", dest="json_dir", help="Directory containing JSON page dumps (bulk ingest)")
    src.add_argument("--pdf", dest="pdf_path", help="Path to a text-bearing PDF (uses pdftotext)")
    src.add_argument("--text", dest="text_path", help="Path to plain text file (optionally split on formfeed)")

    ap.add_argument("--title", default=None)
    ap.add_argument("--author", default=None)
    ap.add_argument("--domain", default=None)
    ap.add_argument("--book-id", default=None)
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--out-dir", default=None, help="Override output dir (defaults under K3D_LOCAL_DIR/galaxies/books/<book_id>)")
    ap.add_argument("--out-root", default=None, help="Output root for --json-dir (defaults under K3D_LOCAL_DIR/galaxies/books/)")
    ap.add_argument("--glob", default="*.json", help="Glob for --json-dir (default: *.json)")
    ap.add_argument("--max-books", type=int, default=None, help="Max books for --json-dir")
    ap.add_argument("--no-formfeed-split", action="store_true", help="Treat input text as a single page")
    ap.add_argument("--pdf-layout", action="store_true", help="Use pdftotext -layout when ingesting PDFs")
    ap.add_argument(
        "--no-token-index",
        action="store_true",
        help="Disable token→pages index generation (reduces RAM usage during ingestion).",
    )
    ap.add_argument(
        "--token-index-min-len",
        type=int,
        default=3,
        help="Minimum normalized token length to include in token index (default: 3).",
    )
    ap.add_argument(
        "--token-index-max-keys",
        type=int,
        default=100_000,
        help="Cap unique token keys in token index to prevent CPU OOM (default: 100000).",
    )
    ap.add_argument(
        "--token-index-max-pages",
        type=int,
        default=64,
        help="Cap pages stored per token key to prevent CPU OOM (default: 64).",
    )
    ap.add_argument(
        "--no-template-index",
        action="store_true",
        help="Disable template token index generation (reduces RAM usage during ingestion).",
    )
    ap.add_argument(
        "--template-index-min-len",
        type=int,
        default=3,
        help="Minimum normalized token length to include in template index (default: 3).",
    )
    ap.add_argument(
        "--template-index-max-keys",
        type=int,
        default=50_000,
        help="Cap unique token keys in template index to prevent CPU OOM (default: 50000).",
    )
    ap.add_argument(
        "--template-index-max-ids",
        type=int,
        default=64,
        help="Cap stored template ids per token key to prevent CPU OOM (default: 64).",
    )
    ap.add_argument(
        "--no-artifact-index",
        action="store_true",
        help="Disable articulated artifact token index generation (reduces RAM usage during ingestion).",
    )
    ap.add_argument(
        "--artifact-index-min-len",
        type=int,
        default=3,
        help="Minimum normalized token length to include in artifact index (default: 3).",
    )
    ap.add_argument(
        "--artifact-index-max-keys",
        type=int,
        default=50_000,
        help="Cap unique token keys in artifact index to prevent CPU OOM (default: 50000).",
    )
    ap.add_argument(
        "--artifact-index-max-ids",
        type=int,
        default=64,
        help="Cap stored artifact ids per token key to prevent CPU OOM (default: 64).",
    )
    return ap


def main() -> None:  # pragma: no cover
    ap = _build_arg_parser()
    args = ap.parse_args()

    ingester = BookGalaxyIngester(
        build_token_index=not bool(args.no_token_index),
        token_index_min_token_len=int(args.token_index_min_len),
        max_token_index_keys=int(args.token_index_max_keys),
        max_pages_per_token=int(args.token_index_max_pages),
        build_template_index=not bool(args.no_template_index),
        template_index_min_token_len=int(args.template_index_min_len),
        max_template_index_keys=int(args.template_index_max_keys),
        max_templates_per_token=int(args.template_index_max_ids),
        build_artifact_index=not bool(args.no_artifact_index),
        artifact_index_min_token_len=int(args.artifact_index_min_len),
        max_artifact_index_keys=int(args.artifact_index_max_keys),
        max_artifacts_per_token=int(args.artifact_index_max_ids),
    )

    if args.json_path:
        out = ingester.ingest_json_pages(
            json_path=args.json_path,
            title=args.title,
            author=args.author,
            domain=args.domain,
            book_id=args.book_id,
            max_pages=args.max_pages,
            out_dir=args.out_dir,
        )
        print(f"[BookGalaxyIngester] Wrote book galaxy to: {out}")
        return

    if args.json_dir:
        outs = ingester.ingest_json_dir(
            json_dir=args.json_dir,
            glob=args.glob,
            domain=args.domain,
            author=args.author,
            max_books=args.max_books,
            max_pages=args.max_pages,
            out_root=args.out_root,
        )
        print(f"[BookGalaxyIngester] Wrote {len(outs)} book galaxies under: {args.out_root or (ingester.local_dir / 'galaxies' / 'books')}")
        return
    if args.pdf_path:
        out = ingester.ingest_pdf_pdftotext(
            pdf_path=args.pdf_path,
            title=args.title,
            author=args.author,
            domain=args.domain,
            book_id=args.book_id,
            out_dir=args.out_dir,
            max_pages=args.max_pages,
            layout=bool(args.pdf_layout),
        )
        print(f"[BookGalaxyIngester] Wrote book galaxy to: {out}")
        return

    out = ingester.ingest_text_file(
        text_path=args.text_path,
        title=args.title,
        author=args.author,
        domain=args.domain,
        book_id=args.book_id,
        out_dir=args.out_dir,
        split_on_formfeed=not bool(args.no_formfeed_split),
        max_pages=args.max_pages,
    )
    print(f"[BookGalaxyIngester] Wrote book galaxy to: {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
