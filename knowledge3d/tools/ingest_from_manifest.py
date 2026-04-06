"""Ingest a scan manifest into meaning-centric stars and optional galaxies."""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from typing import Any

from knowledge3d.knowledgeverse.galaxy_manager import GalaxyManager

from .content_to_stars import write_stars_jsonl
from .knowledge_proceduralizer import packet_to_star, proceduralize_text_content


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
    *,
    provider_name: str,
    model: str | None,
    model_profile: str,
    capture_dir: str | Path | None,
    output_dir: str | Path,
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
    receipt, request = proceduralize_text_content(
        content=content,
        source_id=str(entry.get("id") or _star_id_for_entry(entry)),
        domain_hint=str(entry.get("domain_hint") or "General"),
        source_path=str(file_path),
        context_chunks=[f"name:{entry.get('name', '')}", f"content_type:{content_type}"],
        model=model,
        provider=provider_name,
        model_profile=model_profile,
        capture_dir=capture_dir,
        source_kind="manifest",
    )
    if receipt.parsed_bundle.ingest_action != "augment" or not receipt.parsed_bundle.knowledge_packets:
        failure_code = str(receipt.failure_code or "").strip()
        retry_after_utc = str(receipt.retry_after_utc or "").strip()
        return {
            "path": str(file_path),
            "status": "stopped_plan_limit" if failure_code == "plan_limit_consumed" else "skipped",
            "reason": receipt.parsed_bundle.ingest_action,
            "provider": receipt.provider,
            "failure_code": failure_code,
            "retry_after_utc": retry_after_utc,
            "receipt": receipt.to_dict(),
        }
    primary_packet = receipt.parsed_bundle.knowledge_packets[0]
    star = packet_to_star(primary_packet, request)
    metadata = {"proceduralizer": receipt.to_dict()}
    galaxy_status = "not_persisted"
    galaxy_name = primary_packet.domain
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
        "domain": primary_packet.domain,
        "house_room": star.house_room,
        "provider": receipt.provider,
        "galaxy": galaxy_name,
        "galaxy_status": galaxy_status,
        "star": star,
        "receipt": receipt.to_dict(),
    }


def ingest_manifest(
    manifest_path: str | Path,
    *,
    provider_name: str,
    model: str | None,
    model_profile: str,
    capture_dir: str | Path | None,
    output_dir: str | Path,
    galaxy_manager: GalaxyManager | None = None,
) -> dict[str, Any]:
    """Ingest all entries from a manifest."""
    manifest = load_manifest(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict[str, Any]] = []
    stars = []
    stopped_due_to_plan_limit = False
    retry_after_utc = ""
    for entry in list(manifest.get("entries", []) or []):
        row = ingest_entry(
            entry,
            provider_name=provider_name,
            model=model,
            model_profile=model_profile,
            capture_dir=capture_dir,
            output_dir=output_dir,
            galaxy_manager=galaxy_manager,
        )
        star = row.pop("star", None)
        if star is not None:
            stars.append(star)
        report_rows.append(row)
        receipt = row.get("receipt")
        if isinstance(receipt, dict) and str(receipt.get("failure_code") or "").strip() == "plan_limit_consumed":
            stopped_due_to_plan_limit = True
            retry_after_utc = str(receipt.get("retry_after_utc") or "").strip()
            break
    stars_path = write_stars_jsonl(stars, output_dir / "stars.jsonl")
    report = {
        "manifest": str(Path(manifest_path)),
        "provider": provider_name,
        "model_profile": model_profile,
        "model": model,
        "total_entries": len(report_rows),
        "ingested": sum(1 for row in report_rows if row.get("status") == "ingested"),
        "skipped": sum(1 for row in report_rows if row.get("status") == "skipped"),
        "stopped_due_to_plan_limit": bool(stopped_due_to_plan_limit),
        "retry_after_utc": retry_after_utc,
        "stars_path": str(stars_path),
        "entries": report_rows,
    }
    report_path = output_dir / "ingestion_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest JSON produced by scan_content.")
    parser.add_argument("--provider", default="ollama", help="Default transport provider; ollama is canonical.")
    parser.add_argument("--model-profile", default="quality", help="Proceduralizer model profile.")
    parser.add_argument("--model", default=None, help="Override provider model name.")
    parser.add_argument("--capture-dir", type=Path, default=None, help="Optional directory for request/response captures.")
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
    galaxy_manager = None if args.no_persist else GalaxyManager(storage_root=args.storage_root)
    report = ingest_manifest(
        args.manifest,
        provider_name=str(args.provider).strip().lower(),
        model=str(args.model).strip() or None,
        model_profile=str(args.model_profile).strip().lower(),
        capture_dir=args.capture_dir,
        output_dir=args.output_dir,
        galaxy_manager=galaxy_manager,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 75 if bool(report.get("stopped_due_to_plan_limit")) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
