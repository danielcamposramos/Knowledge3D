"""Transform external enrichment outputs into K3D sovereign artifacts."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

from knowledge3d.knowledgeverse.galaxy_manager import GalaxyManager


@dataclass
class RPNProgram:
    """Minimal sovereign program envelope for ingestion transformation."""

    name: str
    program: str
    domain: str


class K3DTransformer:
    """Transform pattern/concept extraction results into galaxy entries."""

    def __init__(self, galaxy_manager: GalaxyManager):
        self.galaxy_manager = galaxy_manager

    @staticmethod
    def _clean_id(value: str) -> str:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        return f"k3d_{digest}"

    def transform_pattern_to_rpn(self, pattern_data: dict[str, Any], domain: str) -> RPNProgram:
        name = str(pattern_data.get("name", "pattern"))
        template = str(pattern_data.get("rpn_template", "")).strip()
        if not template:
            step_summary = " ".join(str(x) for x in pattern_data.get("transformation_steps", []))
            if not step_summary:
                step_summary = "noop"
            template = f"{step_summary} exec"
        return RPNProgram(name=name, program=template, domain=domain)

    def transform_concept_to_galaxy_entry(
        self,
        concept: str,
        domain: str,
        embedding: list[float] | None = None,
    ) -> dict[str, Any]:
        entry_id = self._clean_id(f"{domain}:{concept}")
        return {
            "id": entry_id,
            "name": concept,
            "domain": domain,
            "embedding": embedding or [],
            "rpn_program": f"{concept} concept_register",
            "metadata": {
                "source": "llm_extraction",
                "timestamp": time.time(),
            },
        }

    def crystallize_enrichment_to_galaxies(
        self,
        enrichment_result: dict[str, Any],
        target_galaxies: list[str],
    ) -> dict[str, Any]:
        domain = str(enrichment_result.get("metadata", {}).get("domain", "general"))
        patterns = list(enrichment_result.get("patterns", []))
        related = list(enrichment_result.get("related_concepts", []))
        embeddings = enrichment_result.get("embeddings", {})

        rpn_programs: list[RPNProgram] = []
        for pattern in patterns:
            if isinstance(pattern, dict):
                rpn_programs.append(self.transform_pattern_to_rpn(pattern, domain=domain))

        created_entries = 0
        for galaxy_name in target_galaxies:
            for rpn in rpn_programs:
                entry = {
                    "id": self._clean_id(f"{galaxy_name}:{rpn.name}:{rpn.program}"),
                    "name": rpn.name,
                    "domain": rpn.domain,
                    "rpn_program": rpn.program,
                    "metadata": {
                        "source": "pattern",
                        "galaxy": galaxy_name,
                    },
                }
                self.galaxy_manager.add_entry(galaxy_name, entry)
                created_entries += 1

            # Concepts are materialized in first target galaxy only to avoid spam.
            if galaxy_name == target_galaxies[0]:
                for concept in related:
                    concept_entry = self.transform_concept_to_galaxy_entry(
                        concept=str(concept),
                        domain=domain,
                    )
                    self.galaxy_manager.add_entry(galaxy_name, concept_entry)
                    created_entries += 1

        embedding_count = 0
        if isinstance(embeddings, dict):
            for value in embeddings.values():
                try:
                    embedding_count += len(value)
                except Exception:
                    pass

        return {
            "galaxy_entries_created": created_entries,
            "rpn_programs_created": len(rpn_programs),
            "target_galaxies": list(target_galaxies),
            "embeddings_stored": embedding_count,
        }
