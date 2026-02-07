#!/usr/bin/env python3
"""Generate additional drawing primitives from local source snapshots via Ollama.

This runs in ingestion space and writes normalized JSONL entries that can be
loaded by DrawingGalaxy bootstrap as optional defaults.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


def _html_to_text(html: str) -> str:
    text = re.sub(r"<script[\\s\\S]*?</script>", " ", html, flags=re.IGNORECASE)
    text = re.sub(r"<style[\\s\\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def _build_prompt(text_snippet: str, source_id: str, max_entries: int) -> str:
    return f"""
You are generating sovereign drawing primitives for Knowledge3D.
Output STRICT JSON only with this schema:
{{
  "entries": [
    {{
      "id": "string_snake_case",
      "name": "human readable name",
      "category": "vector_ops|curves|matrix_ops|projection|clipping|cross_modal",
      "rpn_program": "UPPERCASE_TOKENS_AND_NUMBERS",
      "tags": ["tag1","tag2"],
      "metadata": {{
        "symlink": "math_galaxy|character_galaxy|audio_galaxy|drawing_galaxy",
        "cross_modal": "optional",
        "source_ref": "{source_id}"
      }}
    }}
  ]
}}

Rules:
- Maximum {max_entries} entries.
- No prose, no markdown, no comments.
- Use procedural drawing/math operations.
- Keep ids unique and stable.
- Do not include unsafe code or external API calls.

Source excerpt:
{text_snippet}
""".strip()


def _run_ollama(model: str, prompt: str) -> str:
    cmd = ["ollama", "run", model, prompt]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ollama run failed")
    return proc.stdout.strip()


def _extract_entries(raw: str) -> list[dict[str, Any]]:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        return []
    out: list[dict[str, Any]] = []
    for row in entries:
        if not isinstance(row, dict):
            continue
        if "id" not in row or "rpn_program" not in row:
            continue
        rid = re.sub(r"[^a-z0-9_]+", "_", str(row["id"]).lower()).strip("_")
        if not rid:
            continue
        normalized = {
            "type": "foundational_primitive",
            "id": f"ollama_{rid}",
            "name": str(row.get("name", rid.replace("_", " ").title())),
            "domain": "drawing",
            "category": str(row.get("category", "llm_enrichment")),
            "rpn_program": str(row["rpn_program"]).strip(),
            "tags": [str(x) for x in row.get("tags", []) if str(x)],
            "metadata": row.get("metadata", {}) if isinstance(row.get("metadata", {}), dict) else {},
        }
        normalized["metadata"].setdefault("source", "ollama_enrichment")
        out.append(normalized)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate drawing enrichment entries with Ollama.")
    parser.add_argument("--model", default="qwen2.5:14b", help="Ollama model name")
    parser.add_argument(
        "--manifest",
        default="../Knowledge3D.local/datasets/foundational_drawing_sources/manifest.jsonl",
        help="Source manifest path",
    )
    parser.add_argument(
        "--raw-dir",
        default="../Knowledge3D.local/datasets/foundational_drawing_sources/raw_html",
        help="Downloaded HTML directory",
    )
    parser.add_argument(
        "--output",
        default="../Knowledge3D.local/datasets/foundational_drawing_sources/ollama_enrichment.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--max-sources", type=int, default=4, help="How many sources to process")
    parser.add_argument("--max-entries-per-source", type=int, default=12, help="Entries per source")
    parser.add_argument("--snippet-chars", type=int, default=6000, help="Text characters per source")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    raw_dir = Path(args.raw_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("status") == "ok":
            rows.append(row)
    rows = rows[: max(1, args.max_sources)]

    all_entries: list[dict[str, Any]] = []
    for row in rows:
        source_id = str(row["source_id"])
        html_path = raw_dir / f"{source_id}.html"
        if not html_path.exists():
            continue
        text = _html_to_text(html_path.read_text(encoding="utf-8", errors="ignore"))
        snippet = text[: args.snippet_chars]
        prompt = _build_prompt(snippet, source_id, args.max_entries_per_source)
        raw = _run_ollama(args.model, prompt)
        entries = _extract_entries(raw)
        for entry in entries:
            entry.setdefault("metadata", {})
            entry["metadata"]["source_ref"] = source_id
        all_entries.extend(entries)

    dedup: dict[str, dict[str, Any]] = {}
    for entry in all_entries:
        dedup[entry["id"]] = entry

    with out_path.open("w", encoding="utf-8") as fh:
        for entry in dedup.values():
            fh.write(json.dumps(entry, ensure_ascii=True, separators=(",", ":")) + "\n")

    print(f"wrote {len(dedup)} entries to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

