"""Import Encapsulate CST/CRT JSON structures into Knowledge3D galaxies.

This module provides an ingestion-time bridge between Christoph Dorn's
Encapsulate data model and K3D's Galaxy Universe representation.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from knowledge3d.knowledgeverse.galaxy_manager import GalaxyManager
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse

_FUNCTION_TYPES = {
    "Function",
    "GetterFunction",
    "SetterFunction",
    "Init",
    "Dispose",
    "StructInit",
    "StructDispose",
}

_VALUE_TYPES = {"String", "Literal", "Constant"}


def _normalize_property_type(raw: Any) -> str:
    text = str(raw or "").strip()
    if "." in text:
        text = text.split(".")[-1]
    return text


def _stable_entry_id(*parts: str) -> str:
    payload = "||".join(parts)
    digest = hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:20]
    return f"encap_{digest}"


def _looks_like_capsule_ref(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if text.startswith("k3d://"):
        return True
    if text.startswith("@") and ":" in text:
        return True
    return "/" in text and ":" in text


def _iter_cst_documents(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if "spineContracts" in payload:
            return [payload]
        docs = [value for value in payload.values() if isinstance(value, dict) and "spineContracts" in value]
        return docs
    if isinstance(payload, list):
        return [value for value in payload if isinstance(value, dict) and "spineContracts" in value]
    return []


def _extract_capsule_identity(cst_doc: dict[str, Any]) -> tuple[str, str]:
    capsule_ref = str(
        cst_doc.get("capsuleSourceUriLineRef")
        or cst_doc.get("capsuleSourceLineRef")
        or ""
    ).strip()
    source = cst_doc.get("source") if isinstance(cst_doc.get("source"), dict) else {}
    capsule_name = str(source.get("capsuleName") or "").strip()
    if not capsule_name and capsule_ref:
        capsule_name = capsule_ref.split(":", 1)[0]
    if not capsule_ref and capsule_name:
        capsule_ref = f"{capsule_name}:0"
    if not capsule_name:
        capsule_name = "encapsulate/unknown_capsule"
    return capsule_ref, capsule_name


def _iter_property_definitions(cst_doc: dict[str, Any]) -> list[tuple[str, str, str, dict[str, Any]]]:
    results: list[tuple[str, str, str, dict[str, Any]]] = []
    spine_contracts = cst_doc.get("spineContracts")
    if not isinstance(spine_contracts, dict):
        return results

    seen: set[tuple[str, str, str]] = set()
    for spine_contract_uri, spine_contract_def in spine_contracts.items():
        if not isinstance(spine_contract_def, dict):
            continue

        direct_properties = spine_contract_def.get("#")
        if isinstance(direct_properties, dict):
            for prop_name, prop_def in direct_properties.items():
                if not isinstance(prop_def, dict):
                    continue
                if not prop_def.get("type"):
                    continue
                key = (str(spine_contract_uri), "#", str(prop_name))
                if key in seen:
                    continue
                seen.add(key)
                results.append((str(spine_contract_uri), "#", str(prop_name), prop_def))

        property_contracts = spine_contract_def.get("propertyContracts")
        if not isinstance(property_contracts, dict):
            continue
        for property_contract_uri, contract_def in property_contracts.items():
            if not isinstance(contract_def, dict):
                continue
            properties = contract_def.get("properties")
            if not isinstance(properties, dict):
                continue
            for prop_name, prop_def in properties.items():
                if not isinstance(prop_def, dict):
                    continue
                if not prop_def.get("type"):
                    continue
                key = (str(spine_contract_uri), str(property_contract_uri), str(prop_name))
                if key in seen:
                    continue
                seen.add(key)
                results.append((str(spine_contract_uri), str(property_contract_uri), str(prop_name), prop_def))
    return results


def _to_stored_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(k): _to_stored_value(v)
            for k, v in value.items()
            if isinstance(k, (str, int, float, bool))
        }
    if isinstance(value, (list, tuple)):
        return [_to_stored_value(item) for item in value]
    return repr(value)


def _collect_capsule_refs(value: Any, *, max_depth: int = 8) -> set[str]:
    refs: set[str] = set()
    if max_depth <= 0:
        return refs
    if isinstance(value, str):
        if _looks_like_capsule_ref(value):
            refs.add(value)
        return refs
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and _looks_like_capsule_ref(key):
                refs.add(key)
            refs.update(_collect_capsule_refs(item, max_depth=max_depth - 1))
        return refs
    if isinstance(value, (list, tuple, set)):
        for item in value:
            refs.update(_collect_capsule_refs(item, max_depth=max_depth - 1))
    return refs


def _parse_crt_dependencies(crt_doc: dict[str, Any], fallback_from_ref: str) -> dict[str, set[str]]:
    dependencies: dict[str, set[str]] = {}
    if not isinstance(crt_doc, dict):
        return dependencies

    for key, value in crt_doc.items():
        if key in {"capsuleSourceUriLineRef", "capsuleSourceLineRef", "cacheBustVersion"}:
            continue
        refs = _collect_capsule_refs(value)
        if not refs:
            continue
        from_ref = key if _looks_like_capsule_ref(key) else fallback_from_ref
        dependencies.setdefault(from_ref, set()).update(refs)

    # Fallback for CRT documents with only nested lists and no explicit source key.
    if fallback_from_ref:
        top_refs = _collect_capsule_refs(crt_doc)
        if top_refs:
            dependencies.setdefault(fallback_from_ref, set()).update(top_refs)

    # Clean obvious self references.
    for from_ref, refs in list(dependencies.items()):
        clean = {ref for ref in refs if ref != from_ref}
        if clean:
            dependencies[from_ref] = clean
        else:
            dependencies.pop(from_ref, None)
    return dependencies


class EncapsulateImporter:
    """Ingestion bridge for Encapsulate CST/CRT payloads."""

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

    def import_capsule_source_tree(
        self,
        cst_path: str | Path,
        crt_path: str | Path | None = None,
        *,
        namespace: str = "christoph_encapsulate",
        include_function_properties: bool = True,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        cst_obj = json.loads(Path(cst_path).read_text(encoding="utf-8"))
        crt_obj: dict[str, Any] = {}
        if crt_path is not None and Path(crt_path).exists():
            crt_obj = json.loads(Path(crt_path).read_text(encoding="utf-8"))

        created_entries: list[dict[str, Any]] = []
        property_entries = 0
        symlink_entries = 0
        capsules_processed = 0
        seen_ids: set[str] = set()

        cst_docs = _iter_cst_documents(cst_obj)
        crt_docs = _iter_cst_documents(crt_obj) if crt_obj else []

        for idx, cst_doc in enumerate(cst_docs):
            capsules_processed += 1
            capsule_ref, capsule_name = _extract_capsule_identity(cst_doc)

            crt_doc = crt_docs[idx] if idx < len(crt_docs) else crt_obj
            dependency_map = _parse_crt_dependencies(crt_doc, capsule_ref)
            capsule_refs = sorted(dependency_map.get(capsule_ref, set()))

            for spine_contract_uri, property_contract_uri, prop_name, prop_def in _iter_property_definitions(cst_doc):
                property_type = _normalize_property_type(prop_def.get("type"))
                if property_type in _FUNCTION_TYPES and not include_function_properties:
                    continue

                entry_id = _stable_entry_id(
                    namespace,
                    capsule_name,
                    spine_contract_uri,
                    property_contract_uri,
                    prop_name,
                    property_type,
                )
                if entry_id in seen_ids:
                    continue
                seen_ids.add(entry_id)

                metadata = {
                    "source": "encapsulate",
                    "namespace": namespace,
                    "capsule_name": capsule_name,
                    "capsule_source_uri_line_ref": capsule_ref,
                    "spine_contract_uri": spine_contract_uri,
                    "property_contract_uri": property_contract_uri,
                    "property_name": prop_name,
                    "encapsulate_property_type": property_type,
                    "capsule_import_refs": capsule_refs,
                }
                entry: dict[str, Any] = {
                    "id": entry_id,
                    "name": f"{capsule_name}::{prop_name}",
                    "pattern_type": "encapsulate_property",
                    "metadata": metadata,
                }

                raw_value = _to_stored_value(prop_def.get("value"))
                if property_type in _FUNCTION_TYPES:
                    if isinstance(raw_value, str) and raw_value.strip():
                        entry["rpn_program"] = raw_value
                    else:
                        entry["rpn_program"] = f"ENCAPSULATE_CALL {capsule_name}::{prop_name}"
                elif property_type in _VALUE_TYPES or property_type == "Mapping":
                    entry["value"] = raw_value
                else:
                    entry["value"] = raw_value

                target_galaxy = "Math" if property_type in _FUNCTION_TYPES else "Grammar"
                if not dry_run:
                    self.galaxy_manager.add_entry(target_galaxy, entry)
                created_entries.append({"galaxy": target_galaxy, "entry": entry})
                property_entries += 1

            for from_ref, targets in dependency_map.items():
                for target_ref in sorted(targets):
                    if target_ref == from_ref:
                        continue
                    link_id = _stable_entry_id(namespace, "symlink", from_ref, target_ref)
                    if link_id in seen_ids:
                        continue
                    seen_ids.add(link_id)
                    symlink_entry = {
                        "id": link_id,
                        "name": f"{from_ref} -> {target_ref}",
                        "pattern_type": "capsule_import",
                        "rpn_program": f"CALL {target_ref}",
                        "metadata": {
                            "source": "encapsulate",
                            "namespace": namespace,
                            "link_type": "CAPSULE_IMPORT",
                            "from_capsule_source_uri_line_ref": from_ref,
                            "to_capsule_source_uri_line_ref": target_ref,
                        },
                    }
                    if not dry_run:
                        self.galaxy_manager.add_entry("Grammar", symlink_entry)
                    created_entries.append({"galaxy": "Grammar", "entry": symlink_entry})
                    symlink_entries += 1

        return {
            "cst_path": str(cst_path),
            "crt_path": str(crt_path) if crt_path else None,
            "capsules_processed": capsules_processed,
            "entries_created": len(created_entries),
            "property_entries_created": property_entries,
            "symlink_entries_created": symlink_entries,
            "entries": created_entries,
            "namespace": namespace,
            "dry_run": dry_run,
        }


def import_capsule_source_tree(
    cst_path: str | Path,
    crt_path: str | Path | None = None,
    *,
    galaxy_manager: GalaxyManager | None = None,
    storage_root: str | Path = "../Knowledge3D.local",
    namespace: str = "christoph_encapsulate",
    include_function_properties: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Convenience function wrapper for EncapsulateImporter."""
    importer = EncapsulateImporter(galaxy_manager=galaxy_manager, storage_root=storage_root)
    return importer.import_capsule_source_tree(
        cst_path=cst_path,
        crt_path=crt_path,
        namespace=namespace,
        include_function_properties=include_function_properties,
        dry_run=dry_run,
    )

