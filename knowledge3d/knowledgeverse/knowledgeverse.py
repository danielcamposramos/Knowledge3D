"""Knowledgeverse runtime harness for benchmark and integration scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .foundational_galaxy_bootstrap import populate_always_on_foundational_galaxies
from .galaxy_manager import GalaxyManager
from .shadow_copy import ShadowCopyLearning
from .sleeptime import SleepTimeConsolidation
from .stargate import IngestionStargate
from .trm_navigator import TRMNavigator


@dataclass
class KnowledgeverseMetrics:
    """Runtime metrics surface consumed by integration checks."""

    ptx_fallback_rate: float = 0.0


class Knowledgeverse:
    """Minimal runtime assembly for current Knowledgeverse MVP flows."""

    DEFAULT_GALAXIES: tuple[str, ...] = (
        "Drawing",
        "Character",
        "Word",
        "Grammar",
        "Math",
        "Reality",
        "Audio",
        "3DObjects",
        "Tool",
    )

    def __init__(
        self,
        *,
        manifest_version: str = "kv-2026-02-06",
        storage_root: str | Path = "../Knowledge3D.local",
        galaxy_storage_root: str | Path | None = None,
        audit_index_path: str | Path | None = None,
        sleeptime_journal_path: str | Path | None = None,
        stargate_storage_root: str | Path | None = None,
        eager_load_default_galaxies: bool = True,
        bootstrap_foundational_galaxies: bool = True,
    ):
        self.manifest_version = manifest_version
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.metrics = KnowledgeverseMetrics(ptx_fallback_rate=0.0)

        galaxy_root = (
            Path(galaxy_storage_root)
            if galaxy_storage_root is not None
            else self.storage_root / "galaxies"
        )
        self.galaxy_manager = GalaxyManager(storage_root=galaxy_root)
        self.galaxy_manager.set_knowledgeverse(self)
        self.foundational_bootstrap_summary: dict[str, Any] = {}
        if bootstrap_foundational_galaxies:
            self.foundational_bootstrap_summary = populate_always_on_foundational_galaxies(
                self.galaxy_manager
            )

        stargate_root = (
            Path(stargate_storage_root)
            if stargate_storage_root is not None
            else self.storage_root / "stargate_jobs"
        )
        self.stargate = IngestionStargate(
            manifest_version=self.manifest_version,
            galaxy_manager=self.galaxy_manager,
            storage_root=stargate_root,
        )

        self.trm_navigator = TRMNavigator(knowledgeverse=self)
        self.specialist_router = self.trm_navigator.specialist_router
        self.navigator_specialist = self.trm_navigator.navigator_specialist

        audit_index = (
            Path(audit_index_path)
            if audit_index_path is not None
            else self.storage_root / "audit_index.json"
        )
        self.shadow_copy = ShadowCopyLearning(
            trm_manager=self.trm_navigator,
            index_path=audit_index,
            manifest_version=self.manifest_version,
        )

        sleeptime_journal = (
            Path(sleeptime_journal_path)
            if sleeptime_journal_path is not None
            else self.storage_root / "logs" / "sleeptime_journal.jsonl"
        )
        self.sleeptime = SleepTimeConsolidation(
            knowledgeverse=self,
            journal_path=sleeptime_journal,
        )
        self._default_galaxies_loaded = False
        if eager_load_default_galaxies:
            self.ensure_default_galaxies_loaded()

    def ensure_default_galaxies_loaded(self, *, force: bool = False) -> dict[str, int]:
        """
        Ensure all default galaxies are present in the active universe.

        This enforces the single-world contract for training/benchmark runs:
        all default galaxies are loaded and queryable in every session.
        """
        if self._default_galaxies_loaded and not force:
            return {name: len(self.galaxy_manager.get_galaxy(name).entries) for name in self.DEFAULT_GALAXIES}

        counts: dict[str, int] = {}
        for galaxy_name in self.DEFAULT_GALAXIES:
            galaxy = self.galaxy_manager.get_galaxy(galaxy_name)
            counts[galaxy_name] = len(getattr(galaxy, "entries", []))
        self._default_galaxies_loaded = True
        return counts

    def log_event(
        self,
        event_type: str,
        event_data: dict[str, Any],
        parent_event_id: str | None = None,
    ) -> str:
        """Record an event into Shadow Copy compressed audit."""
        event_id = self.shadow_copy.record_event(
            event_type=event_type,
            event_data=event_data,
            parent_event_id=parent_event_id,
        )
        try:
            specialist = str(event_data.get("specialist", "grammar"))
            query = str(event_data.get("query") or event_data.get("prompt") or event_type)
            lowered = event_type.lower()
            success = ("success" in lowered) or (
                "fail" not in lowered and float(event_data.get("confidence", 0.0)) >= 0.65
            )
            self.trm_navigator.learn_from_feedback(
                query=query,
                specialist=specialist,
                success=success,
                confidence=float(event_data.get("confidence", 0.0) or 0.0),
                domain_hint=str(event_data.get("domain_hint") or event_data.get("domain") or ""),
            )
            self.trm_navigator.save_weights()
        except Exception:
            # Feedback learning should never block event recording.
            pass
        return event_id

    def query(
        self,
        prompt: str,
        *,
        specialist: str = "auto",
        domain_hint: str | None = None,
    ) -> dict[str, Any]:
        """Unified query entrypoint with centralized specialist routing."""
        return self.trm_navigator.navigate_and_compose(
            query=prompt,
            specialist=specialist,
            domain_hint=domain_hint,
            use_enriched=True,
        )
