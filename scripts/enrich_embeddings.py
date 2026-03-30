from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
from typing import Any

from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine


def _normalize_embedding(values: list[float]) -> list[float]:
    if not values:
        return []
    norm_sq = 0.0
    for value in values:
        norm_sq += float(value) * float(value)
    if norm_sq <= 1e-16:
        return [0.0 for _ in values]
    inv_norm = 1.0 / (norm_sq ** 0.5)
    return [float(value) * inv_norm for value in values]


def _coerce_embedding16(values: Any) -> list[float]:
    if not isinstance(values, (list, tuple)):
        return []
    padded = [0.0] * 16
    width = min(16, len(values))
    for index in range(width):
        try:
            padded[index] = float(values[index])
        except Exception:
            padded[index] = 0.0
    return _normalize_embedding(padded)


def _entry_embedding_text(entry: dict[str, Any]) -> str:
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    query_anchor = metadata.get("query_anchor")
    if isinstance(query_anchor, str) and query_anchor.strip():
        return query_anchor.strip()
    fields: list[str] = []
    for value in (
        metadata.get("question"),
        metadata.get("answer"),
        entry.get("name"),
        entry.get("content"),
        entry.get("summary"),
        entry.get("description"),
        metadata.get("semantics"),
    ):
        if isinstance(value, str) and value.strip():
            fields.append(value.strip())
    for key in ("aliases", "keywords", "forms"):
        value = metadata.get(key)
        if isinstance(value, list):
            fields.extend(str(item).strip() for item in value if str(item).strip())
    if fields:
        return " ".join(fields).strip()
    return json.dumps(entry, ensure_ascii=True, sort_keys=True)[:256]


def _store_embedding16(entry: dict[str, Any], embedding16: list[float]) -> None:
    normalized = _coerce_embedding16(embedding16)
    if not normalized:
        return
    entry["embedding16"] = list(normalized)
    metadata = entry.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        entry["metadata"] = metadata
    metadata["embedding16"] = list(normalized)


def _needs_embedding(entry: dict[str, Any]) -> bool:
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    for candidate in (
        entry.get("embedding16"),
        entry.get("embedding"),
        metadata.get("embedding16"),
        metadata.get("embedding"),
    ):
        if _coerce_embedding16(candidate):
            return False
    return True


def _latest_checkpoint_path(storage_root: Path) -> Path:
    return storage_root / "checkpoints" / "galaxy_consolidated_latest.json"


def _house_state_path(storage_root: Path) -> Path:
    return storage_root / "house" / "galaxy_state.bin"


def _load_payload(storage_root: Path) -> tuple[dict[str, Any], Path]:
    binary_path = _house_state_path(storage_root)
    latest_path = _latest_checkpoint_path(storage_root)
    if binary_path.exists():
        with binary_path.open("rb") as handle:
            payload = pickle.load(handle)
        return dict(payload), binary_path
    if latest_path.exists():
        with latest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return dict(payload), latest_path
    raise FileNotFoundError(f"no checkpoint found under {storage_root}")


def enrich_embeddings(storage_root: str | Path) -> dict[str, Any]:
    root = Path(storage_root)
    payload, loaded_from = _load_payload(root)
    galaxies = payload.get("galaxies")
    if not isinstance(galaxies, dict):
        raise ValueError("checkpoint_missing_galaxies")

    engine = RPNEmbeddingEngine(embedding_dim=16)
    enriched = 0
    galaxy_count = 0
    for galaxy_name, entries in galaxies.items():
        if not isinstance(entries, list):
            continue
        galaxy_count += 1
        for entry in entries:
            if not isinstance(entry, dict) or not _needs_embedding(entry):
                continue
            _store_embedding16(entry, list(engine.embed_sentence(_entry_embedding_text(entry))))
            enriched += 1

    binary_path = _house_state_path(root)
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    with binary_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    latest_path = _latest_checkpoint_path(root)
    if latest_path.exists() and latest_path.is_symlink():
        target_path = latest_path.resolve()
    else:
        target_path = latest_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str),
        encoding="utf-8",
    )

    return {
        "storage_root": str(root),
        "loaded_from": str(loaded_from),
        "saved_binary": str(binary_path),
        "saved_json": str(target_path),
        "galaxies": int(galaxy_count),
        "enriched": int(enriched),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich checkpoint entries with pre-computed embedding16 values.")
    parser.add_argument(
        "--storage-root",
        default="/K3D/Knowledge3D.local",
        help="Knowledge3D.local root containing checkpoints/ and house/",
    )
    args = parser.parse_args()
    summary = enrich_embeddings(args.storage_root)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
