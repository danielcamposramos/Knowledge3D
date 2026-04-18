from __future__ import annotations

import json
from pathlib import Path

import pytest

from knowledge3d.ingestion.encapsulate_exporter import EncapsulateExporter
from knowledge3d.ingestion.encapsulate_importer import EncapsulateImporter
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse

_ENCAPSULATE_REPO = Path(__file__).resolve().parents[3] / "encapsulate"


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def test_import_capsule_source_tree_creates_entries(tmp_path: Path):
    cst_payload = {
        "capsuleSourceUriLineRef": "@stream44.studio/demo/capsule:12",
        "source": {
            "capsuleName": "@stream44.studio/demo/capsule",
            "moduleFilepath": "/tmp/capsule.ts",
            "importStackLine": 12,
        },
        "spineContracts": {
            "#@stream44.studio/encapsulate/spine-contracts/CapsuleSpineContract.v0": {
                "#": {
                    "realm": {"type": "Literal", "value": "global"},
                    "username": {"type": "String", "value": "daniel"},
                    "hello": {"type": "Function", "value": "STACK PUSH_HELLO"},
                }
            }
        },
    }
    crt_payload = {
        "references": {
            "@stream44.studio/demo/capsule:12": [
                {
                    "capsuleSourceUriLineRef": "@stream44.studio/demo/helper:3",
                    "relation": "CAPSULE_IMPORT",
                }
            ]
        }
    }
    cst_path = _write_json(tmp_path / "demo.csts.json", cst_payload)
    crt_path = _write_json(tmp_path / "demo.crts.json", crt_payload)

    kv = Knowledgeverse(storage_root=tmp_path / "kv", eager_load_default_galaxies=True)
    importer = EncapsulateImporter(galaxy_manager=kv.galaxy_manager)
    result = importer.import_capsule_source_tree(cst_path, crt_path)

    assert result["capsules_processed"] == 1
    assert result["entries_created"] >= 4
    assert result["property_entries_created"] == 3
    assert result["symlink_entries_created"] >= 1

    grammar_entries = kv.galaxy_manager.get_galaxy("Grammar").entries
    math_entries = kv.galaxy_manager.get_galaxy("Math").entries

    assert any(
        entry.get("metadata", {}).get("source") == "encapsulate"
        and entry.get("metadata", {}).get("encapsulate_property_type") in {"Literal", "String"}
        for entry in grammar_entries
    )
    assert any(
        entry.get("metadata", {}).get("encapsulate_property_type") == "Function"
        for entry in math_entries
    )
    assert any(
        entry.get("metadata", {}).get("link_type") == "CAPSULE_IMPORT"
        for entry in grammar_entries
    )


def test_export_galaxy_entries_and_round_trip_import(tmp_path: Path):
    entries = [
        {
            "id": "math_linear_1",
            "name": "math_template_linear_equation_ax_plus_b_eq_c_v1",
            "rpn_program": "{c} {b} - {a} /",
            "metadata": {
                "source": "math_specialist_bootstrap",
                "encapsulate_property_type": "Function",
                "capsule_import_refs": ["@stream44.studio/demo/base:1"],
            },
        },
        {
            "id": "grammar_phrase_1",
            "name": "word_problem_twice",
            "value": "twice means multiply by 2",
            "metadata": {"encapsulate_property_type": "String"},
        },
    ]

    exporter = EncapsulateExporter()
    exported = exporter.export_galaxy_to_capsule_tree(
        entries,
        output_dir=tmp_path / "exported",
        capsule_name="@k3d/demo/interop",
        include_sit=True,
    )

    cst_path = Path(exported["cst_path"])
    crt_path = Path(exported["crt_path"])
    sit_path = Path(exported["sit_path"])
    assert cst_path.exists()
    assert crt_path.exists()
    assert sit_path.exists()

    cst_obj = json.loads(cst_path.read_text(encoding="utf-8"))
    crt_obj = json.loads(crt_path.read_text(encoding="utf-8"))
    sit_obj = json.loads(sit_path.read_text(encoding="utf-8"))

    assert "spineContracts" in cst_obj
    assert "references" in crt_obj
    assert "rootCapsule" in sit_obj
    assert "capsuleInstances" in sit_obj
    assert sit_obj["rootCapsule"]["capsuleSourceUriLineRefInstanceId"]

    kv = Knowledgeverse(storage_root=tmp_path / "roundtrip_kv", eager_load_default_galaxies=True)
    importer = EncapsulateImporter(galaxy_manager=kv.galaxy_manager)
    imported = importer.import_capsule_source_tree(cst_path, crt_path)

    assert imported["capsules_processed"] == 1
    assert imported["entries_created"] >= 2


def test_import_real_encapsulate_artifacts_when_available(tmp_path: Path):
    cst_files = sorted(_ENCAPSULATE_REPO.rglob("*.csts.json"))
    if not cst_files:
        pytest.skip("No generated .csts.json artifacts found in encapsulate repo.")

    cst_path = cst_files[0]
    crt_candidate = Path(str(cst_path).replace(".csts.json", ".crts.json"))
    crt_path = crt_candidate if crt_candidate.exists() else None

    kv = Knowledgeverse(storage_root=tmp_path / "real_kv", eager_load_default_galaxies=True)
    importer = EncapsulateImporter(galaxy_manager=kv.galaxy_manager)
    result = importer.import_capsule_source_tree(cst_path, crt_path)

    assert result["capsules_processed"] >= 1
    assert "entries_created" in result

