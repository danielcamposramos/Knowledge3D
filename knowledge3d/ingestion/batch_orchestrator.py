"""Batch ingestion orchestrator for corpus preparation."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Any

from knowledge3d.knowledgeverse.sovereignty_firewall import SovereigntyFirewall
from knowledge3d.knowledgeverse.stargate import IngestionStargate

from .corpus_manifest import CorpusEntry, CorpusManifest, CorpusTier, CorpusType


class BatchOrchestrator:
    """Orchestrate staged corpus ingestion with dependency enforcement."""

    def __init__(self, manifest: CorpusManifest, stargate: IngestionStargate):
        self.manifest = manifest
        self.stargate = stargate
        self.firewall = SovereigntyFirewall()

    async def ingest_entry(
        self,
        entry: CorpusEntry,
        max_attempts: int = 3,
        backoff_base: float = 0.5,
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        """Ingest one entry with bounded retry logic."""
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            try:
                return await self._ingest_entry_once(entry=entry, timeout=timeout)
            except Exception as exc:
                last_error = exc
                if attempt >= max_attempts - 1:
                    break
                delay = backoff_base * (2**attempt)
                await asyncio.sleep(delay)
        assert last_error is not None
        raise last_error

    async def _ingest_entry_once(self, entry: CorpusEntry, timeout: float) -> dict[str, Any]:
        source_path = Path(entry.path)
        if not source_path.exists():
            raise FileNotFoundError(f"Corpus source not found: {entry.path}")

        if entry.corpus_type == CorpusType.CODE:
            feeder_candidate = entry.metadata.get("feeder_path")
            if isinstance(feeder_candidate, str) and feeder_candidate:
                self.stargate.validate_feeder_script(feeder_candidate)

        job_id = self.stargate.submit_ingestion_job(
            data_path=entry.path,
            data_type=entry.corpus_type.value,
            target_galaxies=entry.target_galaxies,
            metadata={
                **entry.metadata,
                "entry_id": entry.id,
                "tier": entry.tier.value,
            },
        )
        result = self.stargate.wait_for_job(job_id=job_id, timeout=timeout)
        entry.ingested = True
        embed_count = result.get("embeddings_stored", result.get("embedding_count"))
        entry.embedding_count = int(embed_count) if isinstance(embed_count, int) else None
        return result

    async def ingest_tier(
        self,
        tier: int | CorpusTier,
        max_parallel: int = 4,
    ) -> dict[str, dict[str, Any]]:
        """Ingest all entries for one tier while respecting dependencies."""
        tier_enum = tier if isinstance(tier, CorpusTier) else CorpusTier(int(tier))
        entries = self.manifest.get_by_tier(tier_enum)
        results: dict[str, dict[str, Any]] = {}
        completed: set[str] = {entry_id for entry_id, e in self.manifest.entries.items() if e.ingested}
        pending_ids: set[str] = {entry.id for entry in entries}

        semaphore = asyncio.Semaphore(max(1, int(max_parallel)))

        async def ingest_with_limit(entry: CorpusEntry) -> tuple[str, dict[str, Any]]:
            async with semaphore:
                try:
                    result = await self.ingest_entry(entry)
                    return entry.id, result
                except Exception as exc:
                    return entry.id, {"error": str(exc)}

        while pending_ids:
            ready_entries = [
                self.manifest.entries[entry_id]
                for entry_id in sorted(pending_ids)
                if all(dep_id in completed for dep_id in self.manifest.entries[entry_id].dependencies)
            ]
            if not ready_entries:
                unresolved = {
                    entry_id: self.manifest.entries[entry_id].dependencies
                    for entry_id in sorted(pending_ids)
                }
                for entry_id, deps in unresolved.items():
                    missing = [dep for dep in deps if dep not in completed]
                    results[entry_id] = {
                        "error": (
                            "Unresolved dependencies: "
                            f"{missing if missing else deps}"
                        )
                    }
                pending_ids.clear()
                break

            batch = await asyncio.gather(*(ingest_with_limit(entry) for entry in ready_entries))
            for entry_id, result in batch:
                results[entry_id] = result
                pending_ids.remove(entry_id)
                if "error" not in result:
                    completed.add(entry_id)

        return results

    async def ingest_all(self, max_parallel: int = 4) -> dict[str, Any]:
        """Ingest all tiers in order Tier 1 -> Tier 2 -> Tier 3."""
        stats_before = self.manifest.get_stats()
        all_results: dict[str, dict[str, dict[str, Any]]] = {}
        for tier in (
            CorpusTier.TIER_1_FOUNDATIONAL,
            CorpusTier.TIER_2_DOMAIN,
            CorpusTier.TIER_3_INTEGRATION,
        ):
            all_results[f"tier_{tier.value}"] = await self.ingest_tier(
                tier=tier,
                max_parallel=max_parallel,
            )
        stats_after = self.manifest.get_stats()
        return {
            "stats_before": stats_before,
            "stats_after": stats_after,
            "results": all_results,
        }

    def snapshot_manifest(self) -> list[dict[str, Any]]:
        """Return serializable manifest snapshot for logging/reporting."""
        return [asdict(entry) for entry in self.manifest.get_ingestion_order()]
