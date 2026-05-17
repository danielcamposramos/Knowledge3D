"""Ingest WINE — Tablet surface for document ingestion (PDF / URL / file).

Queues a source for proceduralization. Never imports the proceduralizer
directly. Never performs reasoning. Emits a TabletEnvelope with
surface_kind=INGEST; the daemon stamps an ingest_id and drops temporary
stars into the Knowledgeverse's temporary-star region. Sleeptime later
transmutes worthy temporary stars into House knowledge.

See: TEMP/CLAUDE_INGEST_WINE_SPEC_04.20.2026.md
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from knowledge3d.bridge.headless_tablet import (
    ROUTE_POLICY_ALL_LIVE_GALAXIES,
    SURFACE_KIND_INGEST,
    TabletEnvelope,
    TabletIngest,
)


# Broad — ingestion can cross domains (a PDF of physics, a web page of
# grammar, an audio transcript). Keep all default live galaxies biased in.
# LOD + frustum cull handle working-memory management on GPU per
# feedback_no_knowledge_caps.md.
INGEST_ROUTE_GALAXIES: tuple[str, ...] = (
    "Drawing",
    "Character",
    "Word",
    "Number",
    "Grammar",
    "Math",
    "Reality",
    "Audio",
    "3DObjects",
    "Tool",
)


# --- Gate constants (input validation) -------------------------------------

# Max size of a source_uri string (not the document — just the URI itself).
INGEST_MAX_URI_BYTES: int = 4 * 1024

# Supported MIME classes. This is a coarse gate only — per-MIME parsers
# live in knowledge3d/ingestion/ and are selected by the proceduralizer
# pipeline, not by this WINE.
INGEST_SUPPORTED_MIME: frozenset[str] = frozenset({
    "application/pdf",
    "text/plain",
    "text/markdown",
    "text/html",
    "application/json",
    "image/png",
    "image/jpeg",
    "audio/wav",
    "audio/mpeg",
})

# Max chunk count per ingest (downstream, the proceduralizer enforces
# per-chunk cost gates). This cap protects the receipt/queue path from
# a single 100k-chunk submission starving everything else.
INGEST_MAX_CHUNKS: int = 4096

# Supported URI schemes.
_ALLOWED_SCHEMES: frozenset[str] = frozenset({"file", "https", "http", "s3"})

_ALLOWED_CHUNKING_KEYS: frozenset[str] = frozenset({"strategy", "size", "overlap"})


def _validate_ingest_input(
    source_uri: Any,
    mime: Any,
    chunking: Any,
) -> dict[str, Any] | None:
    """Validate ingest input. Returns error dict on failure, None on success.

    Pure Python type/length checks — NOT reasoning logic.
    Per feedback_python_dispatch_is_not_a_line_item.md: these are I/O gates.
    """
    if not isinstance(source_uri, str) or not source_uri.strip():
        return {"status": "error", "error": "missing_source_uri"}
    try:
        encoded_uri = source_uri.encode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return {"status": "error", "error": "missing_source_uri"}
    if len(encoded_uri) > INGEST_MAX_URI_BYTES:
        return {"status": "error", "error": "uri_too_long"}
    # Check URI scheme
    colon_pos = source_uri.find(":")
    if colon_pos <= 0:
        return {"status": "error", "error": "unsupported_scheme"}
    scheme = source_uri[:colon_pos].lower().strip()
    if scheme not in _ALLOWED_SCHEMES:
        return {"status": "error", "error": "unsupported_scheme"}
    if not isinstance(mime, str) or mime not in INGEST_SUPPORTED_MIME:
        return {"status": "error", "error": "unsupported_mime"}
    if chunking is not None:
        if not isinstance(chunking, dict):
            return {"status": "error", "error": "bad_chunking"}
        for key in chunking:
            if key not in _ALLOWED_CHUNKING_KEYS:
                return {"status": "error", "error": "bad_chunking"}
        size = chunking.get("size")
        if size is not None and not isinstance(size, int):
            return {"status": "error", "error": "bad_chunking"}
        overlap = chunking.get("overlap")
        if overlap is not None and not isinstance(overlap, int):
            return {"status": "error", "error": "bad_chunking"}
    return None


def build_ingest_route(
    *,
    specialist: str = "ingest",
    domain_hint: str | None = None,
    galaxies: Sequence[str] | None = None,
    route_policy: str = ROUTE_POLICY_ALL_LIVE_GALAXIES,
) -> dict[str, Any]:
    """Route descriptor for an ingest dispatch.

    Mirrors build_math_route(). INGEST does not run the composed head;
    it runs the proceduralizer-feeder chain. Specialist is fixed at
    "ingest" — the sovereign core recognizes this as a queue-write lane.
    """
    route: dict[str, Any] = {
        "specialist": str(specialist or "ingest"),
        "route_policy": str(route_policy or ROUTE_POLICY_ALL_LIVE_GALAXIES),
    }
    if domain_hint is not None and str(domain_hint).strip():
        route["domain_hint"] = str(domain_hint).strip()
    galaxy_names = [str(name) for name in (galaxies or INGEST_ROUTE_GALAXIES) if str(name).strip()]
    if galaxy_names:
        route["galaxy_names"] = galaxy_names
    return route


def build_ingest_task(
    *,
    task_id: str,
    source_uri: str,
    mime: str,
    chunking: Mapping[str, Any] | None = None,
    lang_hint: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build (task_payload, route_payload) for a single ingest job.

    Args:
        task_id: Caller-supplied identifier (CLI / daemon generates UUID
            if caller omits). Used as the ingest_id seed.
        source_uri: file:// or https:// URI of the document. Must be
            reachable from the daemon process. No raw bytes in the
            envelope — the fetch happens inside the ingest pipeline.
        mime: IANA MIME type. Must be in INGEST_SUPPORTED_MIME.
        chunking: Optional overrides for chunker policy:
            {"strategy": "pdf_pages"|"md_headers"|"fixed_chars",
             "size": int, "overlap": int}.
            Default policy is inferred from mime inside the proceduralizer.
        lang_hint: Optional BCP-47 language hint (e.g. "pt-BR", "en").
            Used by the proceduralizer to bias multilingual embeddings;
            NOT a filter (knowledge is meaning-centric per MEMORY.md).
        metadata: Free-form caller metadata — provenance, user tags, etc.
    """
    envelope = TabletIngest.ingest_task(
        task_id=task_id,
        source_uri=source_uri,
        mime=mime,
        chunking=chunking,
        lang_hint=lang_hint,
        metadata=metadata,
    )
    return dict(envelope.task), build_ingest_route(
        specialist=envelope.specialist,
        domain_hint=envelope.domain_hint,
        galaxies=list(envelope.galaxies) if envelope.galaxies else None,
        route_policy=envelope.route_policy,
    )


def ingest_envelope(
    *,
    task_id: str,
    source_uri: str,
    mime: str,
    chunking: Mapping[str, Any] | None = None,
    lang_hint: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TabletEnvelope:
    """Return the full TabletEnvelope for an ingest job.

    Mirrors math_wine.math_dataset_envelope(). This is the factory the
    daemon and CLI call; TabletIngest.ingest_task does the actual
    construction.
    """
    return TabletIngest.ingest_task(
        task_id=task_id,
        source_uri=source_uri,
        mime=mime,
        chunking=chunking,
        lang_hint=lang_hint,
        metadata=metadata,
    )
