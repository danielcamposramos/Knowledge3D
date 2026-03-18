from __future__ import annotations

import json
from pathlib import Path

from knowledge3d.knowledgeverse.galaxy_manager import GalaxyManager
from knowledge3d.tools.augmentation_providers import AugmentationProvider, AugmentationResult
from knowledge3d.tools.ingest_from_manifest import ingest_manifest


class _FakeProvider(AugmentationProvider):
    provider_name = "fake"

    def augment(self, content: str, context: dict[str, object]) -> AugmentationResult:
        domain = str(context.get("domain_hint", "General"))
        name = str(context.get("name", "entry"))
        return AugmentationResult(
            summary=name,
            entities=[],
            relationships=[],
            domain=domain,
            meaning_rpn_hint=f"{domain.upper()} CONTENT ENTRY",
            taxonomy_refs=[f"concept_{domain.lower()}"] if domain != "General" else [],
            surface_forms={"en": name.replace("_", " ").title(), "pt": name.replace("_", " ").title()},
            confidence=0.8,
            provider=self.provider_name,
            raw_response="{}",
        )

    def classify(self, content: str) -> str:
        return "General"

    def is_available(self) -> bool:
        return True


def test_ingest_manifest_produces_stars_and_report(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    text_path = docs_dir / "math_notes.txt"
    json_path = docs_dir / "biology.json"
    text_path.write_text("derivative of x^2", encoding="utf-8")
    json_path.write_text('{"species":"fern"}', encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "total_files": 2,
                "by_type": {"structured": 1, "text": 1},
                "entries": [
                    {
                        "path": str(text_path),
                        "name": "math_notes",
                        "extension": ".txt",
                        "content_type": "text",
                        "size_bytes": text_path.stat().st_size,
                        "domain_hint": "Mathematics",
                    },
                    {
                        "path": str(json_path),
                        "name": "biology",
                        "extension": ".json",
                        "content_type": "structured",
                        "size_bytes": json_path.stat().st_size,
                        "domain_hint": "Biology",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    galaxy_manager = GalaxyManager(storage_root=tmp_path / "galaxies")
    report = ingest_manifest(
        manifest_path,
        provider=_FakeProvider(),
        output_dir=tmp_path / "out",
        galaxy_manager=galaxy_manager,
    )

    assert report["ingested"] == 2
    stars_path = tmp_path / "out" / "stars.jsonl"
    assert stars_path.exists()
    lines = [line for line in stars_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    first_star = json.loads(lines[0])
    assert galaxy_manager.load_meaning_star("Mathematics", first_star["star_id"]) is not None
