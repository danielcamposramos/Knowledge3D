"""Cached MCP web-search client for ingestion-path differentiation work."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MCP_ENDPOINT = "http://localhost:8502/mcp"
DEFAULT_CACHE_DIR = REPO_ROOT / "scripts" / "ingestion" / "staging" / "D3_dedup" / "cache" / "web_search"
_SEARCH_RESULT_RE = re.compile(
    r"^\d+\.\s+\*\*(?P<title>.*?)\*\*\s*\n\s*(?P<snippet>.*?)\n\s*(?P<url>https?://\S+)\s*$",
    re.DOTALL,
)


class WebSearchUnavailable(RuntimeError):
    """Raised when the MCP web_search tool is unavailable and no cache is present."""


def _cache_path(query: str, *, cache_dir: Path) -> Path:
    digest = hashlib.sha256(str(query).encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def _parse_tool_output(raw_output: str) -> list[dict[str, str]]:
    text = str(raw_output or "").strip()
    if not text:
        return []
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if re.match(r"^\d+\.\s+\*\*", line):
            if current:
                blocks.append("\n".join(current).strip())
                current = []
        if current or re.match(r"^\d+\.\s+\*\*", line):
            current.append(line.rstrip())
    if current:
        blocks.append("\n".join(current).strip())
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for block in blocks:
        match = _SEARCH_RESULT_RE.match(block)
        if not match:
            continue
        url = str(match.group("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        results.append(
            {
                "url": url,
                "title": str(match.group("title") or "").strip(),
                "snippet": str(match.group("snippet") or "").strip(),
            }
        )
    return results


def _extract_frames(body: str) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    text = str(body or "")
    if ("data:" in text and text.lstrip().startswith("event:")) or text.startswith("data:"):
        for line in text.splitlines():
            if line.startswith("data: "):
                try:
                    frames.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    continue
        return frames
    try:
        frames.append(json.loads(text))
    except json.JSONDecodeError:
        for line in text.splitlines():
            if line.startswith("data: "):
                try:
                    frames.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    continue
    return frames


def _tool_text_from_frames(frames: list[dict[str, Any]]) -> str:
    fragments: list[str] = []
    for frame in frames:
        result = frame.get("result")
        if not isinstance(result, dict):
            continue
        content = result.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if text:
                fragments.append(text)
    return "\n".join(fragment for fragment in fragments if fragment).strip()


def _call_mcp_web_search(
    query: str,
    *,
    endpoint: str,
    max_results: int,
    timeout: float,
) -> str:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    session = requests.Session()
    try:
        init_resp = session.post(
            endpoint,
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "knowledge3d-b7", "version": "1.0"},
                },
            },
            timeout=timeout,
        )
        init_resp.raise_for_status()
        session_id = init_resp.headers.get("Mcp-Session-Id") or init_resp.headers.get("mcp-session-id")
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        session.post(
            endpoint,
            headers=headers,
            json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            timeout=timeout,
        )
        call_resp = session.post(
            endpoint,
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "web_search",
                    "arguments": {"query": query, "num_results": int(max_results)},
                },
            },
            timeout=timeout,
        )
        call_resp.raise_for_status()
        text = _tool_text_from_frames(_extract_frames(call_resp.text))
        return text
    except requests.RequestException as exc:
        raise WebSearchUnavailable(f"web_search unavailable for query={query!r}: {exc}") from exc
    finally:
        session.close()


def web_search(
    query: str,
    max_results: int = 5,
    *,
    endpoint: str = DEFAULT_MCP_ENDPOINT,
    cache_dir: Path | None = None,
    timeout: float = 30.0,
) -> list[dict[str, str]]:
    resolved_cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(query, cache_dir=resolved_cache_dir)
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return list(cached.get("results") or [])
    raw_output = _call_mcp_web_search(
        query,
        endpoint=endpoint,
        max_results=max_results,
        timeout=timeout,
    )
    results = _parse_tool_output(raw_output)
    payload = {
        "query": query,
        "max_results": int(max_results),
        "results": results,
        "raw_output": raw_output,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return results


__all__ = ["DEFAULT_CACHE_DIR", "DEFAULT_MCP_ENDPOINT", "WebSearchUnavailable", "web_search"]
