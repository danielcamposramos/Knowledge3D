"""
Book Galaxy Library - lightweight runtime access to ingested book galaxies.

This module complements `book_galaxy_ingestion.py`:
- Ingestion writes page-level artifacts under `K3D_LOCAL_DIR/galaxies/books/<book_id>/`
- Runtime needs a cheap way to query those artifacts without loading full token payloads.

Design goals (aligned with `docs/vocabulary/`):
- Galaxy-first: knowledge lives in Book Galaxies (local artifacts), not hardcoded.
- Dual-client contract: every hit is anchored to a concrete page/location.
- Keep inference-side deps minimal: stdlib only (no numpy).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _default_books_root() -> Path:
    local_dir = os.getenv("K3D_LOCAL_DIR")
    if local_dir:
        return Path(local_dir) / "galaxies" / "books"
    # Repo root is 4 parents up from this file.
    return (Path(__file__).resolve().parents[4] / "Knowledge3D.local" / "galaxies" / "books").resolve()


@dataclass(frozen=True)
class BookGalaxyHit:
    book_id: str
    title: str
    domain: Optional[str]
    page_number: int
    score: float
    excerpt: str
    position_3d: Optional[Tuple[float, float, float]] = None


@dataclass(frozen=True)
class BookGalaxyTemplateHit:
    book_id: str
    title: str
    domain: Optional[str]
    page_number: int
    score: float
    lhs: str
    rhs: str
    rpn: str


@dataclass(frozen=True)
class BookGalaxyArtifactHit:
    book_id: str
    title: str
    domain: Optional[str]
    page_number: int
    score: float
    artifact_id: str
    artifact_type: str
    name: str
    conditions: List[str]
    conditions_rpn: List[str]
    symbol_bindings: Dict[str, Any]
    lhs: Optional[str]
    rhs: Optional[str]
    rpn: Optional[str]
    conclusion: Optional[str]
    conclusion_rpn: Optional[str]
    derived_rpns: List[Dict[str, Any]]
    var_mapping: Dict[str, str]


class BookGalaxyLibrary:
    """
    Load and query Book Galaxy artifacts produced by `BookGalaxyIngester`.

    Query strategy (v0):
    - Lexical overlap using `token_index.json` (normalized WordGalaxy tokens).
    - Fetch page text excerpts from `pages_text.jsonl`.
    """

    def __init__(self, *, books_root: Optional[str | Path] = None, max_books: Optional[int] = None) -> None:
        self.books_root = Path(books_root) if books_root else _default_books_root()
        self.max_books = int(max_books) if max_books is not None else None
        self._books: Dict[str, "_BookStore"] = {}

    def discover(self) -> None:
        if not self.books_root.exists():
            return
        if not self.books_root.is_dir():
            return
        dirs = [p for p in sorted(self.books_root.iterdir()) if p.is_dir()]
        if self.max_books is not None:
            dirs = dirs[: max(0, int(self.max_books))]
        for path in dirs:
            book_id = path.name
            if book_id in self._books:
                continue
            meta_path = path / "metadata.json"
            token_index_path = path / "token_index.json"
            pages_text_path = path / "pages_text.jsonl"
            if not meta_path.exists():
                continue
            self._books[book_id] = _BookStore(
                book_id=book_id,
                root=path,
                metadata_path=meta_path,
                token_index_path=token_index_path if token_index_path.exists() else None,
                pages_text_path=pages_text_path if pages_text_path.exists() else None,
            )

    def list_books(self) -> List[Dict[str, Any]]:
        self.discover()
        out: List[Dict[str, Any]] = []
        for store in self._books.values():
            meta = store.metadata
            out.append(
                {
                    "book_id": store.book_id,
                    "title": meta.get("title"),
                    "author": meta.get("author"),
                    "domain": meta.get("domain"),
                    "page_count": meta.get("page_count"),
                }
            )
        return out

    def search(
        self,
        *,
        normalized_tokens: Sequence[str],
        top_k: int = 5,
        max_pages_per_book: int = 8,
        min_token_hits: int = 2,
    ) -> List[BookGalaxyHit]:
        """
        Search across books by token overlap.

        Args:
            normalized_tokens: WordGalaxy-normalized tokens from the query.
            top_k: number of hits to return globally.
            max_pages_per_book: cap candidate pages from each book to keep it fast.
            min_token_hits: require at least this many matched tokens for a page to be considered.
        """
        self.discover()
        toks = [t for t in normalized_tokens if t]
        if not toks or not self._books:
            return []

        hits: List[BookGalaxyHit] = []
        for store in self._books.values():
            page_scores: Dict[int, int] = {}
            for tok in toks:
                for page in store.pages_for_token(tok):
                    page_scores[page] = page_scores.get(page, 0) + 1

            if not page_scores:
                continue

            # Rank pages in this book by overlap.
            ranked = sorted(page_scores.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            per_book = 0
            for page_num, score_int in ranked:
                if score_int < int(min_token_hits):
                    break
                excerpt, pos = store.page_excerpt(page_num)
                hits.append(
                    BookGalaxyHit(
                        book_id=store.book_id,
                        title=str(store.metadata.get("title") or store.book_id),
                        domain=store.metadata.get("domain"),
                        page_number=int(page_num),
                        score=float(score_int),
                        excerpt=excerpt,
                        position_3d=pos,
                    )
                )
                per_book += 1
                if per_book >= int(max_pages_per_book):
                    break

        hits.sort(key=lambda h: (h.score, h.book_id, h.page_number), reverse=True)
        return hits[: max(0, int(top_k))]

    def search_templates(
        self,
        *,
        normalized_tokens: Sequence[str],
        top_k: int = 5,
        max_templates_per_book: int = 16,
        min_token_hits: int = 2,
    ) -> List[BookGalaxyTemplateHit]:
        """
        Search across books for extracted equation templates (lhs = rhs).

        This relies on optional artifacts produced by `BookGalaxyIngester`:
        - `templates.jsonl`
        - `template_index.json`
        """
        self.discover()
        toks = [t for t in normalized_tokens if t]
        if not toks or not self._books:
            return []

        hits: List[BookGalaxyTemplateHit] = []
        for store in self._books.values():
            if not store.has_templates:
                continue
            template_scores: Dict[str, int] = {}
            for tok in toks:
                for template_id in store.templates_for_token(tok):
                    template_scores[template_id] = template_scores.get(template_id, 0) + 1

            if not template_scores:
                continue

            ranked = sorted(template_scores.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            desired: List[Tuple[str, int]] = []
            for template_id, score_int in ranked:
                if score_int < int(min_token_hits):
                    break
                desired.append((template_id, score_int))
                if len(desired) >= int(max_templates_per_book):
                    break

            records = store.template_records([template_id for template_id, _ in desired])
            for template_id, score_int in desired:
                tmpl = records.get(template_id)
                if tmpl is None:
                    continue
                hits.append(
                    BookGalaxyTemplateHit(
                        book_id=store.book_id,
                        title=str(store.metadata.get("title") or store.book_id),
                        domain=store.metadata.get("domain"),
                        page_number=int(tmpl.get("page_number") or 0),
                        score=float(score_int),
                        lhs=str(tmpl.get("lhs") or ""),
                        rhs=str(tmpl.get("rhs") or ""),
                        rpn=str(tmpl.get("rpn") or ""),
                    )
                )

        hits.sort(key=lambda h: (h.score, h.book_id, h.page_number), reverse=True)
        return hits[: max(0, int(top_k))]

    def search_artifacts(
        self,
        *,
        normalized_tokens: Sequence[str],
        top_k: int = 5,
        max_artifacts_per_book: int = 24,
        min_token_hits: int = 2,
    ) -> List[BookGalaxyArtifactHit]:
        """
        Search across books for articulated artifacts (theorem/definition/formula blocks).

        This relies on optional artifacts produced by `BookGalaxyIngester`:
        - `artifacts.jsonl`
        - `artifact_index.json`
        """
        self.discover()
        toks = [t for t in normalized_tokens if t]
        if not toks or not self._books:
            return []

        hits: List[BookGalaxyArtifactHit] = []
        for store in self._books.values():
            if not store.has_artifacts:
                continue
            artifact_scores: Dict[str, int] = {}
            for tok in toks:
                for artifact_id in store.artifacts_for_token(tok):
                    artifact_scores[artifact_id] = artifact_scores.get(artifact_id, 0) + 1

            if not artifact_scores:
                continue

            ranked = sorted(artifact_scores.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            desired: List[Tuple[str, int]] = []
            for artifact_id, score_int in ranked:
                if score_int < int(min_token_hits):
                    break
                desired.append((artifact_id, score_int))
                if len(desired) >= int(max_artifacts_per_book):
                    break

            records = store.artifact_records([artifact_id for artifact_id, _ in desired])
            for artifact_id, score_int in desired:
                art = records.get(artifact_id)
                if art is None:
                    continue
                hits.append(
                    BookGalaxyArtifactHit(
                        book_id=store.book_id,
                        title=str(store.metadata.get("title") or store.book_id),
                        domain=store.metadata.get("domain"),
                        page_number=int(art.get("page_number") or 0),
                        score=float(score_int),
                        artifact_id=str(art.get("artifact_id") or artifact_id),
                        artifact_type=str(art.get("artifact_type") or ""),
                        name=str(art.get("name") or ""),
                        conditions=list(art.get("conditions") or []),
                        conditions_rpn=list(art.get("conditions_rpn") or []),
                        symbol_bindings=dict(art.get("symbol_bindings") or {}),
                        lhs=(str(art.get("lhs")) if art.get("lhs") is not None else None),
                        rhs=(str(art.get("rhs")) if art.get("rhs") is not None else None),
                        rpn=(str(art.get("rpn")) if art.get("rpn") is not None else None),
                        conclusion=(str(art.get("conclusion")) if art.get("conclusion") is not None else None),
                        conclusion_rpn=(str(art.get("conclusion_rpn")) if art.get("conclusion_rpn") is not None else None),
                        derived_rpns=list(art.get("derived_rpns") or []),
                        var_mapping=dict(art.get("var_mapping") or {}),
                    )
                )

        hits.sort(key=lambda h: (h.score, h.book_id, h.page_number), reverse=True)
        return hits[: max(0, int(top_k))]


class _BookStore:
    def __init__(
        self,
        *,
        book_id: str,
        root: Path,
        metadata_path: Path,
        token_index_path: Optional[Path],
        pages_text_path: Optional[Path],
    ) -> None:
        self.book_id = book_id
        self.root = root
        self.metadata_path = metadata_path
        self.token_index_path = token_index_path
        self.pages_text_path = pages_text_path
        self.templates_path = (root / "templates.jsonl") if (root / "templates.jsonl").exists() else None
        self.template_index_path = (root / "template_index.json") if (root / "template_index.json").exists() else None
        self.artifacts_path = (root / "artifacts.jsonl") if (root / "artifacts.jsonl").exists() else None
        self.artifact_index_path = (root / "artifact_index.json") if (root / "artifact_index.json").exists() else None
        self._metadata: Optional[Dict[str, Any]] = None
        self._token_index: Optional[Dict[str, List[int]]] = None
        # Page text can be huge; never pre-load by default. We either:
        # - Preload excerpts if the file is small enough, or
        # - Scan on-demand with a bounded cache.
        self._page_excerpt_preloaded: Optional[Dict[int, Tuple[str, Optional[Tuple[float, float, float]]]]] = None
        self._page_excerpt_preload_attempted: bool = False
        self._page_excerpt_cache: "OrderedDict[Tuple[int, int], Tuple[str, Optional[Tuple[float, float, float]]]]" = (
            OrderedDict()
        )
        self._template_index: Optional[Dict[str, List[str]]] = None
        self._templates: Optional[Dict[str, Dict[str, Any]]] = None  # Deprecated: avoid full-file load for OOM safety.
        self._template_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._artifact_index: Optional[Dict[str, List[str]]] = None
        self._artifacts: Optional[Dict[str, Dict[str, Any]]] = None  # Deprecated: avoid full-file load for OOM safety.
        self._artifact_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    @staticmethod
    def _debug_enabled() -> bool:
        return os.getenv("K3D_BOOK_DEBUG", "").strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return int(default)
        try:
            return int(str(raw).strip())
        except Exception:
            return int(default)

    @staticmethod
    def _should_skip_json(path: Path, *, env_name: str, default_mb: int) -> bool:
        """
        Returns True if `path` should not be loaded due to size controls.

        Semantics (consistent across indices):
        - `0` disables loading entirely.
        - Positive values cap file size in MB (default varies by index).
        - Negative values disable the cap (attempt to load regardless).
        """
        max_mb = _BookStore._env_int(env_name, default_mb)
        if max_mb == 0:
            return True
        if max_mb < 0:
            return False
        try:
            size_bytes = path.stat().st_size
        except Exception:
            size_bytes = 0
        return size_bytes > max_mb * 1024 * 1024

    def _cache_put(self, cache: "OrderedDict[str, Dict[str, Any]]", key: str, value: Dict[str, Any]) -> None:
        max_items = self._env_int("K3D_BOOK_RECORD_CACHE_MAX", 2048)
        if max_items <= 0:
            return
        cache[key] = value
        cache.move_to_end(key, last=True)

    def _cache_put_page_excerpt(
        self,
        cache: "OrderedDict[Tuple[int, int], Tuple[str, Optional[Tuple[float, float, float]]]]",
        key: Tuple[int, int],
        value: Tuple[str, Optional[Tuple[float, float, float]]],
    ) -> None:
        max_items = self._env_int("K3D_BOOK_PAGE_CACHE_MAX", 512)
        if max_items <= 0:
            return
        cache[key] = value
        cache.move_to_end(key, last=True)
        while len(cache) > max_items:
            cache.popitem(last=False)

    def _page_excerpt_source_path(self) -> Optional[Path]:
        if self.pages_text_path and self.pages_text_path.exists():
            return self.pages_text_path
        pages_path = self.root / "pages.jsonl"
        if pages_path.exists():
            return pages_path
        return None

    def _parse_page_line(
        self, *, item: Dict[str, Any], max_chars: int
    ) -> Optional[Tuple[int, Tuple[str, Optional[Tuple[float, float, float]]]]]:
        try:
            page_num = int(item.get("page_number"))
        except Exception:
            return None
        text = str(item.get("text") or "")
        clean = " ".join(text.split())
        if len(clean) > int(max_chars):
            clean = clean[: max(0, int(max_chars) - 1)] + "…"
        pos = item.get("position_3d")
        pos3 = None
        if isinstance(pos, (list, tuple)) and len(pos) == 3:
            try:
                pos3 = (float(pos[0]), float(pos[1]), float(pos[2]))
            except Exception:
                pos3 = None
        return page_num, (clean, pos3)

    def _maybe_preload_page_excerpts(self) -> None:
        """
        Preload page excerpts into memory *only* if the source file is small.

        This prevents CPU OOM when books have large `pages_text.jsonl`.
        """
        if self._page_excerpt_preload_attempted:
            return
        self._page_excerpt_preload_attempted = True

        src = self._page_excerpt_source_path()
        if src is None:
            self._page_excerpt_preloaded = {}
            return

        max_mb = self._env_int("K3D_BOOK_PAGES_TEXT_PRELOAD_MAX_MB", 8)
        if max_mb == 0:
            # Explicitly disable preload.
            self._page_excerpt_preloaded = None
            return
        if max_mb < 0:
            # Negative disables cap (use with caution).
            max_bytes = None
        else:
            max_bytes = max_mb * 1024 * 1024

        try:
            size_bytes = src.stat().st_size
        except Exception:
            size_bytes = 0

        if max_bytes is not None and size_bytes > max_bytes:
            if self._debug_enabled():
                print(
                    f"[BookGalaxyLibrary] skip pages preload for {self.book_id}: "
                    f"{size_bytes / (1024 * 1024):.1f}MB > {max_mb}MB (K3D_BOOK_PAGES_TEXT_PRELOAD_MAX_MB)"
                )
            self._page_excerpt_preloaded = None
            return

        # Safe enough to preload excerpts (not full pages) for fast lookups.
        preloaded: Dict[int, Tuple[str, Optional[Tuple[float, float, float]]]] = {}
        try:
            with src.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    parsed = self._parse_page_line(item=item, max_chars=240)
                    if parsed is None:
                        continue
                    page_num, payload = parsed
                    preloaded[page_num] = payload
        except Exception:
            # Fall back to scan mode.
            self._page_excerpt_preloaded = None
            return

        self._page_excerpt_preloaded = preloaded
        # Note: we intentionally preload only short excerpts (default 240 chars),
        # and any further memory bounding is handled via `K3D_BOOK_PAGES_TEXT_PRELOAD_MAX_MB`.

    @property
    def metadata(self) -> Dict[str, Any]:
        if self._metadata is None:
            self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return self._metadata

    @property
    def token_index(self) -> Dict[str, List[int]]:
        if self._token_index is None:
            self._token_index = {}
            if not self.token_index_path or not self.token_index_path.exists():
                return self._token_index

            # Guard against memory OOM: skip loading huge token indices.
            #
            # - `0` disables token-index loading entirely.
            # - Positive values cap file size in MB (default 64).
            # - Negative values disable the cap (attempt to load regardless).
            max_mb = self._env_int("K3D_BOOK_TOKEN_INDEX_MAX_MB", 64)
            if max_mb == 0:
                return self._token_index

            if max_mb > 0:
                try:
                    size_bytes = self.token_index_path.stat().st_size
                except Exception:
                    size_bytes = 0
                if size_bytes > max_mb * 1024 * 1024:
                    if self._debug_enabled():
                        print(
                            f"[BookGalaxyLibrary] skip token_index for {self.book_id}: "
                            f"{size_bytes / (1024 * 1024):.1f}MB > {max_mb}MB (K3D_BOOK_TOKEN_INDEX_MAX_MB)"
                        )
                    return self._token_index

            raw = json.loads(self.token_index_path.read_text(encoding="utf-8"))
            index: Dict[str, List[int]] = {}
            for k, v in raw.items():
                if isinstance(k, str) and isinstance(v, list):
                    try:
                        index[k] = [int(x) for x in v]
                    except Exception:
                        continue
            self._token_index = index
        return self._token_index

    def pages_for_token(self, normalized_token: str) -> List[int]:
        return self.token_index.get(normalized_token, [])

    @property
    def has_templates(self) -> bool:
        return bool(self.templates_path and self.template_index_path)

    @property
    def has_artifacts(self) -> bool:
        return bool(self.artifacts_path and self.artifact_index_path)

    @property
    def template_index(self) -> Dict[str, List[str]]:
        if self._template_index is None:
            idx: Dict[str, List[str]] = {}
            if self.template_index_path and self.template_index_path.exists():
                if self._should_skip_json(self.template_index_path, env_name="K3D_BOOK_TEMPLATE_INDEX_MAX_MB", default_mb=64):
                    if self._debug_enabled():
                        try:
                            size_bytes = self.template_index_path.stat().st_size
                        except Exception:
                            size_bytes = 0
                        max_mb = self._env_int("K3D_BOOK_TEMPLATE_INDEX_MAX_MB", 64)
                        print(
                            f"[BookGalaxyLibrary] skip template_index for {self.book_id}: "
                            f"{size_bytes / (1024 * 1024):.1f}MB > {max_mb}MB (K3D_BOOK_TEMPLATE_INDEX_MAX_MB)"
                        )
                    self._template_index = idx
                    return self._template_index
                raw = json.loads(self.template_index_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    for k, v in raw.items():
                        if isinstance(k, str) and isinstance(v, list):
                            idx[k] = [str(x) for x in v if x]
            self._template_index = idx
        return self._template_index

    def templates_for_token(self, normalized_token: str) -> List[str]:
        return self.template_index.get(normalized_token, [])

    @property
    def artifact_index(self) -> Dict[str, List[str]]:
        if self._artifact_index is None:
            idx: Dict[str, List[str]] = {}
            if self.artifact_index_path and self.artifact_index_path.exists():
                if self._should_skip_json(self.artifact_index_path, env_name="K3D_BOOK_ARTIFACT_INDEX_MAX_MB", default_mb=64):
                    if self._debug_enabled():
                        try:
                            size_bytes = self.artifact_index_path.stat().st_size
                        except Exception:
                            size_bytes = 0
                        max_mb = self._env_int("K3D_BOOK_ARTIFACT_INDEX_MAX_MB", 64)
                        print(
                            f"[BookGalaxyLibrary] skip artifact_index for {self.book_id}: "
                            f"{size_bytes / (1024 * 1024):.1f}MB > {max_mb}MB (K3D_BOOK_ARTIFACT_INDEX_MAX_MB)"
                        )
                    self._artifact_index = idx
                    return self._artifact_index
                raw = json.loads(self.artifact_index_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    for k, v in raw.items():
                        if isinstance(k, str) and isinstance(v, list):
                            idx[k] = [str(x) for x in v if x]
            self._artifact_index = idx
        return self._artifact_index

    def artifacts_for_token(self, normalized_token: str) -> List[str]:
        return self.artifact_index.get(normalized_token, [])

    def _scan_jsonl_records(self, *, path: Path, id_field: str, wanted: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        want = {w for w in wanted if w}
        if not want:
            return {}
        found: Dict[str, Dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not want:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                rec_id = str(item.get(id_field) or "")
                if rec_id and rec_id in want:
                    found[rec_id] = item
                    want.remove(rec_id)
        return found

    def template_records(self, template_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        if not self.templates_path or not self.templates_path.exists():
            return {}
        hits: Dict[str, Dict[str, Any]] = {}
        missing: List[str] = []
        for template_id in template_ids:
            if not template_id:
                continue
            cached = self._template_cache.get(template_id)
            if cached is not None:
                hits[template_id] = cached
                self._template_cache.move_to_end(template_id, last=True)
                continue
            missing.append(template_id)
        if missing:
            scanned = self._scan_jsonl_records(path=self.templates_path, id_field="template_id", wanted=missing)
            for key, value in scanned.items():
                hits[key] = value
                self._cache_put(self._template_cache, key, value)
        return hits

    def template_record(self, template_id: str) -> Optional[Dict[str, Any]]:
        if not template_id:
            return None
        return self.template_records([template_id]).get(template_id)

    def artifact_records(self, artifact_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        if not self.artifacts_path or not self.artifacts_path.exists():
            return {}
        hits: Dict[str, Dict[str, Any]] = {}
        missing: List[str] = []
        for artifact_id in artifact_ids:
            if not artifact_id:
                continue
            cached = self._artifact_cache.get(artifact_id)
            if cached is not None:
                hits[artifact_id] = cached
                self._artifact_cache.move_to_end(artifact_id, last=True)
                continue
            missing.append(artifact_id)
        if missing:
            scanned = self._scan_jsonl_records(path=self.artifacts_path, id_field="artifact_id", wanted=missing)
            for key, value in scanned.items():
                hits[key] = value
                self._cache_put(self._artifact_cache, key, value)
        return hits

    def artifact_record(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        if not artifact_id:
            return None
        return self.artifact_records([artifact_id]).get(artifact_id)

    def page_excerpt(self, page_number: int, *, max_chars: int = 240) -> Tuple[str, Optional[Tuple[float, float, float]]]:
        page_num = int(page_number)
        max_chars_i = int(max_chars)

        # Try (small) preload first.
        self._maybe_preload_page_excerpts()
        if self._page_excerpt_preloaded is not None:
            text, pos = self._page_excerpt_preloaded.get(page_num, ("", None))
            if max_chars_i != 240:
                # Preload stores 240-char excerpts; if caller wants a different size,
                # fall back to scan mode for correctness.
                pass
            else:
                return text, pos

        # Cache lookup (keyed by requested max_chars).
        cache_key = (page_num, max_chars_i)
        cached = self._page_excerpt_cache.get(cache_key)
        if cached is not None:
            self._page_excerpt_cache.move_to_end(cache_key, last=True)
            return cached

        src = self._page_excerpt_source_path()
        if src is None:
            value = ("", None)
            self._cache_put_page_excerpt(self._page_excerpt_cache, cache_key, value)
            return value

        # Scan-on-demand: read JSONL until we find the requested page.
        found_value: Tuple[str, Optional[Tuple[float, float, float]]] = ("", None)
        try:
            with src.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    parsed = self._parse_page_line(item=item, max_chars=max_chars_i)
                    if parsed is None:
                        continue
                    num, payload = parsed
                    if num == page_num:
                        found_value = payload
                        break
        except Exception:
            found_value = ("", None)

        self._cache_put_page_excerpt(self._page_excerpt_cache, cache_key, found_value)
        return found_value
