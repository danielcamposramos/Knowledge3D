"""Minimal Wikipedia-shaped ingestion records for attribution-aware bulk imports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WikipediaAttribution:
    source_name: str
    source_url: str
    license_name: str = "CC-BY-SA 4.0"


@dataclass(frozen=True)
class WikipediaIngestRecord:
    title: str
    summary: str
    domain: str
    attribution: WikipediaAttribution


def build_wikipedia_record(title: str, summary: str, url: str, *, domain: str = "General") -> WikipediaIngestRecord:
    return WikipediaIngestRecord(
        title=str(title).strip(),
        summary=str(summary).strip(),
        domain=str(domain).strip() or "General",
        attribution=WikipediaAttribution(source_name="Wikipedia", source_url=str(url).strip()),
    )


__all__ = ["WikipediaAttribution", "WikipediaIngestRecord", "build_wikipedia_record"]
