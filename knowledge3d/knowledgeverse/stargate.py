"""Knowledgeverse Ingestion Stargate helpers.

This module wires sovereignty checks into feeder validation and candidate
program validation before crystallization into sovereign runtime assets.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Mapping

from knowledge3d.ingestion.enrichment_pipeline import EnrichmentPipeline
from knowledge3d.ingestion.k3d_transformer import K3DTransformer

from .galaxy_manager import GalaxyManager
from .ring_buffer import RingBuffer
from .sovereignty_firewall import SovereigntyFirewall
from .temporal_metadata import TemporalMetadataManager


class IngestionStargate:
    """Ingestion entrypoint with sovereignty and temporal tracking."""

    def __init__(
        self,
        stargate_region: object | None = None,
        manifest_version: str = "unknown",
        ring_size_mb: int = 512,
        galaxy_manager: GalaxyManager | None = None,
        enrichment_pipeline: EnrichmentPipeline | None = None,
        k3d_transformer: K3DTransformer | None = None,
        storage_root: str | Path = "../Knowledge3D.local/stargate_jobs",
        use_local_models: bool = False,
    ):
        self.region = stargate_region
        self.ring_buffer = RingBuffer(size_mb=ring_size_mb)
        self.temporal_manager = TemporalMetadataManager(
            manifest_version=manifest_version,
            region_id="ingestion_stargate",
        )
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.galaxy_manager = galaxy_manager or GalaxyManager()
        self.enrichment_pipeline = enrichment_pipeline or EnrichmentPipeline(
            use_local_models=use_local_models
        )
        self.k3d_transformer = k3d_transformer or K3DTransformer(self.galaxy_manager)
        self.pending_jobs: list[dict[str, object]] = []
        self.completed_jobs: list[dict[str, object]] = []
        self._jobs_by_id: dict[str, dict[str, object]] = {}

    def validate_feeder_script(self, feeder_path: str | Path) -> None:
        is_valid, violations = SovereigntyFirewall.validate_feeder_imports(feeder_path)
        if not is_valid:
            joined = "; ".join(violations)
            raise RuntimeError(f"Feeder validation failed: {joined}")

    def validate_candidate_program(self, candidate_rpn: Mapping[str, object]) -> None:
        is_valid, reason = SovereigntyFirewall.validate_rpn_output(candidate_rpn)
        if not is_valid:
            raise RuntimeError(f"RPN candidate validation failed: {reason}")

    def submit_ingestion_job(
        self,
        data_path: str,
        data_type: str = "auto",
        target_galaxies: list[str] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> str:
        """Submit ingestion job and return temporal job ID."""
        temporal = self.temporal_manager.create_metadata()
        job_id = temporal.event_id
        job = {
            "job_id": job_id,
            "data_path": data_path,
            "data_type": data_type,
            "status": "pending",
            "target_galaxies": list(target_galaxies or []),
            "metadata": dict(metadata or {}),
            "temporal": {
                "event_id": temporal.event_id,
                "timestamp": temporal.timestamp,
                "lamport_clock": temporal.lamport_clock,
                "vector_clock": temporal.vector_clock,
                "parent_event_id": temporal.parent_event_id,
                "manifest_version": temporal.manifest_version,
            },
        }
        self.pending_jobs.append(job)
        self._jobs_by_id[job_id] = job
        self._trigger_feeder(job)
        return job_id

    def wait_for_job(self, job_id: str, timeout: float = 300.0) -> dict[str, object]:
        """Wait until a job transitions to completed/error or timeout."""
        deadline = time.monotonic() + max(timeout, 0.0)
        while time.monotonic() <= deadline:
            job = self._jobs_by_id.get(job_id)
            if job is None:
                raise KeyError(f"Unknown ingestion job id: {job_id}")
            status = str(job.get("status", "pending"))
            if status in {"completed", "error"}:
                return job
            time.sleep(0.01)
        raise TimeoutError(f"Ingestion job timed out: {job_id}")

    def _trigger_feeder(self, job: dict[str, object]) -> None:
        """Execute ingestion + crystallization synchronously for this runtime."""
        try:
            crystallization = self._execute_ingestion_pipeline(job)
            job["status"] = "completed"
            job["crystallization_result"] = crystallization
            job.update(crystallization)
        except Exception as exc:
            job["status"] = "error"
            job["error"] = str(exc)
        self.completed_jobs.append(job)

    def _execute_ingestion_pipeline(self, job: dict[str, object]) -> dict[str, object]:
        """Run feeder -> enrichment -> sovereign crystallization."""
        data_path = str(job.get("data_path", ""))
        metadata = dict(job.get("metadata", {}))
        target_galaxies = list(job.get("target_galaxies", []))
        if not target_galaxies:
            target_galaxies = ["Grammar"]
        if "domain" not in metadata:
            metadata["domain"] = str(job.get("data_type", "general"))
        metadata["source_path"] = data_path

        content = self._load_job_content(data_path)
        enriched = self.enrichment_pipeline.enrich_document(
            content=content,
            metadata=metadata,
        )
        crystallization = self.k3d_transformer.crystallize_enrichment_to_galaxies(
            enrichment_result=enriched,
            target_galaxies=target_galaxies,
        )

        self._persist_job_artifacts(
            job_id=str(job.get("job_id", "unknown")),
            enriched=enriched,
            crystallization=crystallization,
        )
        return crystallization

    def _load_job_content(self, data_path: str) -> str:
        """Load source content for ingestion; fallback to path literal if missing."""
        path = Path(data_path)
        if not path.exists():
            return f"MISSING_SOURCE {data_path}"

        if path.is_dir():
            parts: list[str] = []
            for child in sorted(path.rglob("*")):
                if not child.is_file():
                    continue
                try:
                    text = child.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if text.strip():
                    parts.append(f"## FILE: {child.name}\n{text[:64_000]}")
                if len(parts) >= 32:
                    break
            return "\n\n".join(parts) if parts else f"EMPTY_DIRECTORY {data_path}"

        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return f"UNREADABLE_SOURCE {data_path}"

    def _persist_job_artifacts(
        self,
        job_id: str,
        enriched: dict[str, object],
        crystallization: dict[str, object],
    ) -> None:
        """Persist ingestion artifacts for audit and reruns."""
        job_dir = self.storage_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        def _default_serializer(value: object) -> object:
            shape = getattr(value, "shape", None)
            if shape is not None:
                return {"ndarray_shape": list(shape)}
            if hasattr(value, "tolist"):
                try:
                    return value.tolist()
                except Exception:
                    return str(value)
            return str(value)

        (job_dir / "enriched.json").write_text(
            json.dumps(enriched, indent=2, sort_keys=True, default=_default_serializer),
            encoding="utf-8",
        )
        (job_dir / "crystallization.json").write_text(
            json.dumps(crystallization, indent=2, sort_keys=True),
            encoding="utf-8",
        )
