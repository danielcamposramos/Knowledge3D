"""Ingest a scan manifest into meaning-centric stars and optional galaxies."""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from typing import Any

from knowledge3d.knowledgeverse.galaxy_manager import GalaxyManager

from .augmentation_providers import AugmentationProvider, AugmentationResult, create_provider
from .content_to_stars import result_to_star, write_stars_jsonl


def _entry_hash(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:8]


def _star_id_for_entry(entry: dict[str, Any]) -> str:
    name = str(entry.get("name", "entry")).strip().lower().replace(" ", "_")
    return f"ingested_{name or 'entry'}_{_entry_hash(str(entry.get('path', '')))}"


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_text_file(path: Path, *, limit: int = 16000) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[:limit]


def _read_document_file(path: Path, *, limit: int = 8000) -> str:
    raw = path.read_bytes()[:limit]
    decoded = raw.decode("utf-8", errors="ignore")
    compact = " ".join(decoded.split())
    if compact:
        return f"Document {path.stem}: {compact[:limit]}"
    return f"Document {path.stem} ({path.suffix.lower()}) size={path.stat().st_size} bytes"


def load_entry_content(entry: dict[str, Any]) -> str:
    """Load file content into a provider-ready string."""
    file_path = Path(str(entry.get("path", "")).strip())
    content_type = str(entry.get("content_type", "")).strip().lower()
    if content_type in {"text", "code", "structured", "tabular"}:
        return _read_text_file(file_path)
    if content_type == "document":
        return _read_document_file(file_path)
    if content_type in {"image", "audio", "video"}:
        size_bytes = int(file_path.stat().st_size) if file_path.exists() else 0
        return (
            f"{content_type.title()} asset: {file_path.name}\n"
            f"Path: {file_path}\n"
            f"Size bytes: {size_bytes}\n"
            f"Domain hint: {entry.get('domain_hint', 'General')}"
        )
    return ""


def ingest_entry(
    entry: dict[str, Any],
    provider: AugmentationProvider,
    output_dir: str | Path,
    *,
    galaxy_manager: GalaxyManager | None = None,
) -> dict[str, Any]:
    """Ingest a single manifest entry."""
    output_dir = Path(output_dir)
    file_path = Path(str(entry.get("path", "")).strip())
    content_type = str(entry.get("content_type", "")).strip().lower()
    if content_type not in {"document", "text", "structured", "tabular", "code", "image", "audio", "video"}:
        return {
            "path": str(file_path),
            "status": "skipped",
            "reason": f"unsupported type: {content_type}",
        }
    if not file_path.exists():
        return {
            "path": str(file_path),
            "status": "skipped",
            "reason": "missing file",
        }

    content = load_entry_content(entry)
    context = dict(entry)
    result: AugmentationResult = provider.augment(content, context)
    star = result_to_star(
        result,
        star_id=_star_id_for_entry(entry),
        meta_refs=[f"source:{file_path}"],
    )
    metadata = {
        "augmentation": {
            "summary": result.summary,
            "entities": list(result.entities),
            "relationships": list(result.relationships),
            "provider": result.provider,
            "confidence": float(result.confidence),
            "source_path": str(file_path),
        }
    }
    galaxy_status = "not_persisted"
    galaxy_name = result.domain
    if galaxy_manager is not None:
        galaxy_status = galaxy_manager.store_meaning_star(
            galaxy_name,
            star,
            metadata=metadata,
        )
    return {
        "path": str(file_path),
        "status": "ingested",
        "star_id": star.star_id,
        "domain": result.domain,
        "house_room": star.house_room,
        "provider": result.provider,
        "galaxy": galaxy_name,
        "galaxy_status": galaxy_status,
        "star": star,
    }


def ingest_manifest(
    manifest_path: str | Path,
    *,
    provider: AugmentationProvider,
    output_dir: str | Path,
    galaxy_manager: GalaxyManager | None = None,
) -> dict[str, Any]:
    """Ingest all entries from a manifest."""
    manifest = load_manifest(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict[str, Any]] = []
    stars = []
    for entry in list(manifest.get("entries", []) or []):
        row = ingest_entry(entry, provider, output_dir, galaxy_manager=galaxy_manager)
        star = row.pop("star", None)
        if star is not None:
            stars.append(star)
        report_rows.append(row)
    stars_path = write_stars_jsonl(stars, output_dir / "stars.jsonl")
    report = {
        "manifest": str(Path(manifest_path)),
        "provider": provider.provider_name,
        "total_entries": len(report_rows),
        "ingested": sum(1 for row in report_rows if row.get("status") == "ingested"),
        "skipped": sum(1 for row in report_rows if row.get("status") == "skipped"),
        "stars_path": str(stars_path),
        "entries": report_rows,
    }
    report_path = output_dir / "ingestion_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest JSON produced by scan_content.")
    parser.add_argument("--provider", default="auto", help="ollama, claude, gpt, or auto.")
    parser.add_argument("--model", default=None, help="Override provider model name.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for stars/report output.")
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=Path("../Knowledge3D.local/galaxies"),
        help="Galaxy storage root when persisting stars.",
    )
    parser.add_argument("--no-persist", action="store_true", help="Skip Galaxy persistence and only write JSONL.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    provider = create_provider(args.provider, model=args.model) if args.model else create_provider(args.provider)
    if not provider.is_available() and str(args.provider).strip().lower() != "auto":
        raise RuntimeError(f"Provider '{args.provider}' is not available.")
    galaxy_manager = None if args.no_persist else GalaxyManager(storage_root=args.storage_root)
    report = ingest_manifest(
        args.manifest,
        provider=provider,
        output_dir=args.output_dir,
        galaxy_manager=galaxy_manager,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
