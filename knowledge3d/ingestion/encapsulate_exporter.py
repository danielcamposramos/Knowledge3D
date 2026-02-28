"""Export Knowledge3D galaxy entries into Encapsulate CST/CRT/SIT JSON files."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from knowledge3d.knowledgeverse.galaxy_manager import GalaxyManager
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse

_VALID_PROP_NAME = re.compile(r"[^A-Za-z0-9_]+")

_ENCAPSULATE_PROPERTY_TYPES = {
    "Function",
    "GetterFunction",
    "SetterFunction",
    "String",
    "Mapping",
    "Literal",
    "Constant",
    "StructInit",
    "StructDispose",
    "Init",
    "Dispose",
}


def _safe_property_name(raw: str, *, fallback: str) -> str:
    text = _VALID_PROP_NAME.sub("_", raw.strip())
    text = text.strip("_")
    return text or fallback


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def _normalize_entries(galaxy_entries: Any) -> list[dict[str, Any]]:
    if isinstance(galaxy_entries, dict):
        if "entry" in galaxy_entries and isinstance(galaxy_entries.get("entry"), dict):
            return [galaxy_entries["entry"]]
        out = [value for value in galaxy_entries.values() if isinstance(value, dict)]
        return out
    if isinstance(galaxy_entries, list):
        out: list[dict[str, Any]] = []
        for item in galaxy_entries:
            if isinstance(item, dict) and isinstance(item.get("entry"), dict):
                out.append(item["entry"])
            elif isinstance(item, dict):
                out.append(item)
        return out
    return []


def _property_type_for_entry(entry: dict[str, Any]) -> str:
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    hinted = str(metadata.get("encapsulate_property_type") or "").strip()
    if hinted in _ENCAPSULATE_PROPERTY_TYPES:
        return hinted
    if entry.get("rpn_program"):
        return "Function"
    if isinstance(entry.get("value"), str):
        return "String"
    return "Literal"


def _property_value_for_entry(entry: dict[str, Any], prop_type: str) -> Any:
    if prop_type == "Function":
        if isinstance(entry.get("rpn_program"), str) and entry["rpn_program"].strip():
            return entry["rpn_program"]
        return "NOOP"
    if "value" in entry:
        value = entry.get("value")
    else:
        value = entry.get("name")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(k): v
            for k, v in value.items()
            if isinstance(k, (str, int, float, bool))
        }
    if isinstance(value, (list, tuple)):
        return list(value)
    return repr(value)


def _extract_capsule_import_refs(entry: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    for key in ("capsule_import_refs", "encapsulate_refs", "call_refs"):
        value = metadata.get(key)
        if isinstance(value, list):
            refs.update(str(item).strip() for item in value if str(item).strip())
        elif isinstance(value, str) and value.strip():
            refs.add(value.strip())

    rpn_program = str(entry.get("rpn_program") or "").strip()
    if rpn_program.startswith("CALL "):
        parts = rpn_program.split(maxsplit=1)
        if len(parts) == 2 and parts[1].strip():
            refs.add(parts[1].strip())
    return refs


class EncapsulateExporter:
    """Export bridge from K3D entries to Encapsulate CST/CRT/SIT artifacts."""

    def __init__(
        self,
        *,
        galaxy_manager: GalaxyManager | None = None,
        storage_root: str | Path = "../Knowledge3D.local",
    ):
        if galaxy_manager is None:
            kv = Knowledgeverse(storage_root=storage_root, eager_load_default_galaxies=True)
            self.galaxy_manager = kv.galaxy_manager
        else:
            self.galaxy_manager = galaxy_manager

    def export_galaxy_to_capsule_tree(
        self,
        galaxy_entries: Any,
        output_dir: str | Path,
        *,
        capsule_name: str = "k3d.exported.capsule",
        include_sit: bool = False,
    ) -> dict[str, str]:
        entries = _normalize_entries(galaxy_entries)
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        capsule_source_uri_line_ref = f"k3d://{capsule_name}:0"
        properties: dict[str, dict[str, Any]] = {}
        capsule_import_refs: set[str] = set()

        used_prop_names: set[str] = set()
        for idx, entry in enumerate(entries):
            metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            raw_name = str(
                metadata.get("property_name")
                or entry.get("name")
                or entry.get("id")
                or f"property_{idx + 1}"
            )
            prop_name = _safe_property_name(raw_name, fallback=f"property_{idx + 1}")
            if prop_name in used_prop_names:
                prop_name = f"{prop_name}_{idx + 1}"
            used_prop_names.add(prop_name)

            prop_type = _property_type_for_entry(entry)
            prop_value = _property_value_for_entry(entry, prop_type)
            properties[prop_name] = {
                "type": prop_type,
                "value": prop_value,
            }
            capsule_import_refs.update(_extract_capsule_import_refs(entry))

        cst = {
            "cacheBustVersion": 1,
            "capsuleSourceUriLineRef": capsule_source_uri_line_ref,
            "source": {
                "capsuleName": capsule_name,
                "moduleFilepath": f"galaxy://{capsule_name}.json",
                "importStackLine": 0,
            },
            "spineContracts": {
                "#k3d/encapsulate/spine-contracts/CapsuleSpineContract.v1": {
                    "#": properties
                }
            },
            "ambientReferences": {
                "source": "knowledge3d_export",
                "entry_count": len(entries),
            },
        }

        crt_refs = sorted(ref for ref in capsule_import_refs if ref and ref != capsule_source_uri_line_ref)
        crt = {
            "capsuleSourceUriLineRef": capsule_source_uri_line_ref,
            "references": {
                capsule_source_uri_line_ref: [
                    {"capsuleSourceUriLineRef": ref, "relation": "CAPSULE_IMPORT"}
                    for ref in crt_refs
                ]
            },
            "capsuleReferences": crt_refs,
        }

        safe_name = _safe_property_name(capsule_name, fallback="k3d_capsule")
        cst_path = output_root / f"{safe_name}.csts.json"
        crt_path = output_root / f"{safe_name}.crts.json"
        cst_path.write_text(json.dumps(cst, indent=2, ensure_ascii=True), encoding="utf-8")
        crt_path.write_text(json.dumps(crt, indent=2, ensure_ascii=True), encoding="utf-8")

        outputs = {"cst_path": str(cst_path), "crt_path": str(crt_path)}

        if include_sit:
            root_instance_id = _sha256(capsule_source_uri_line_ref)
            capsules: dict[str, dict[str, str]] = {
                capsule_name: {"capsuleSourceUriLineRef": capsule_source_uri_line_ref}
            }
            capsule_instances: dict[str, dict[str, str]] = {
                root_instance_id: {
                    "capsuleName": capsule_name,
                    "capsuleSourceUriLineRef": capsule_source_uri_line_ref,
                    "parentCapsuleSourceUriLineRefInstanceId": "",
                }
            }
            for ref in crt_refs:
                child_name = ref.split(":", 1)[0]
                capsules.setdefault(child_name, {"capsuleSourceUriLineRef": ref})
                child_instance_id = _sha256(f"{root_instance_id}:{ref}")
                capsule_instances[child_instance_id] = {
                    "capsuleName": child_name,
                    "capsuleSourceUriLineRef": ref,
                    "parentCapsuleSourceUriLineRefInstanceId": root_instance_id,
                }
            sit = {
                "rootCapsule": {
                    "capsuleSourceUriLineRef": capsule_source_uri_line_ref,
                    "capsuleSourceUriLineRefInstanceId": root_instance_id,
                },
                "capsules": capsules,
                "capsuleInstances": capsule_instances,
            }
            sit_path = output_root / f"{safe_name}.sit.json"
            sit_path.write_text(json.dumps(sit, indent=2, ensure_ascii=True), encoding="utf-8")
            outputs["sit_path"] = str(sit_path)

        return outputs


def export_galaxy_to_capsule_tree(
    galaxy_entries: Any,
    output_dir: str | Path,
    *,
    galaxy_manager: GalaxyManager | None = None,
    storage_root: str | Path = "../Knowledge3D.local",
    capsule_name: str = "k3d.exported.capsule",
    include_sit: bool = False,
) -> dict[str, str]:
    """Convenience function wrapper for EncapsulateExporter."""
    exporter = EncapsulateExporter(galaxy_manager=galaxy_manager, storage_root=storage_root)
    return exporter.export_galaxy_to_capsule_tree(
        galaxy_entries=galaxy_entries,
        output_dir=output_dir,
        capsule_name=capsule_name,
        include_sit=include_sit,
    )

