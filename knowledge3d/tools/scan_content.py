"""Scan file paths into a simple ingestion manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "document",
    ".txt": "text",
    ".md": "text",
    ".json": "structured",
    ".jsonl": "structured",
    ".csv": "tabular",
    ".py": "code",
    ".ts": "code",
    ".js": "code",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".svg": "image",
    ".mp3": "audio",
    ".wav": "audio",
    ".mp4": "video",
}


def guess_domain(path: Path) -> str:
    """Best-effort domain hint from the file path."""
    parts = "/".join(str(part).lower() for part in path.parts)
    if "math" in parts or "algebra" in parts or "calculus" in parts:
        return "Mathematics"
    if "physics" in parts or "kinematic" in parts or "thermo" in parts:
        return "Physics"
    if "biology" in parts or "/bio" in parts or "ecology" in parts:
        return "Biology"
    if "language" in parts or "grammar" in parts or "lingu" in parts:
        return "Language"
    if "tool" in parts or "engineering" in parts or "workshop" in parts:
        return "Tools"
    if "art" in parts or "visual" in parts or "draw" in parts or "gallery" in parts:
        return "Visual"
    if "audio" in parts or "music" in parts or "sound" in parts:
        return "Audio"
    return "General"


def scan_file(path: Path) -> dict[str, Any] | None:
    """Produce a manifest entry for a single supported file."""
    try:
        resolved = path.expanduser().resolve()
    except Exception:
        resolved = path
    if not resolved.exists() or not resolved.is_file():
        return None
    extension = resolved.suffix.lower()
    content_type = SUPPORTED_EXTENSIONS.get(extension)
    if not content_type:
        return None
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "name": resolved.stem,
        "extension": extension,
        "content_type": content_type,
        "size_bytes": int(stat.st_size),
        "domain_hint": guess_domain(resolved),
    }


def scan_content(sources: Iterable[Path]) -> dict[str, Any]:
    """Build a manifest from file paths."""
    entries: list[dict[str, Any]] = []
    for source in sources:
        entry = scan_file(Path(source))
        if entry is not None:
            entries.append(entry)
    entries.sort(key=lambda row: str(row.get("path", "")))
    by_type: dict[str, int] = {}
    for entry in entries:
        key = str(entry["content_type"])
        by_type[key] = int(by_type.get(key, 0)) + 1
    return {
        "version": 1,
        "total_files": len(entries),
        "by_type": dict(sorted(by_type.items())),
        "entries": entries,
    }


def _iter_paths_from_stdin() -> list[Path]:
    return [
        Path(line.strip())
        for line in sys.stdin.read().splitlines()
        if line.strip()
    ]


def _iter_paths_from_list_file(list_path: Path) -> list[Path]:
    return [
        Path(line.strip())
        for line in list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _iter_paths_from_dir(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", dest="list_path", type=Path, help="File containing one path per line.")
    parser.add_argument("--dir", dest="dir_path", type=Path, help="Directory to scan recursively.")
    parser.add_argument("--output", type=Path, help="Write manifest JSON to this path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    sources: list[Path]
    if args.dir_path is not None:
        sources = _iter_paths_from_dir(args.dir_path)
    elif args.list_path is not None:
        sources = _iter_paths_from_list_file(args.list_path)
    else:
        sources = _iter_paths_from_stdin()
    manifest = scan_content(sources)
    rendered = json.dumps(manifest, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        sys.stdout.write(rendered + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
