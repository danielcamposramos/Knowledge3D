from __future__ import annotations

"""Build PTX-ready language galaxies from existing lexicon/audio JSONL corpora."""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np  # type: ignore
from pygltflib import (  # type: ignore
    ARRAY_BUFFER,
    Accessor,
    Asset,
    Buffer,
    BufferView,
    GLTF2,
    Mesh,
    Node,
    Primitive,
    Scene,
)


@dataclass
class StarRecord:
    star_id: str
    name: str
    embedding: np.ndarray
    modalities: Sequence[str] = field(default_factory=list)
    tags: Sequence[str] = field(default_factory=list)
    payload: Dict[str, object] = field(default_factory=dict)


def _load_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if isinstance(data, dict):
                records.append(data)
            if limit is not None and len(records) >= limit:
                break
    return records


def _to_star(record: Dict[str, object], *, default_language: str) -> Optional[StarRecord]:
    star_id = str(record.get("id") or record.get("star_id") or "")
    if not star_id:
        return None
    embedding = record.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        return None
    emb_array = np.asarray(embedding, dtype=np.float32)
    if emb_array.ndim != 1:
        return None

    name = str(record.get("name") or star_id)
    modalities = record.get("modality_fusion")
    if not isinstance(modalities, (list, tuple)):
        modalities = ["text"]
    tags = record.get("tags")
    if not isinstance(tags, (list, tuple)):
        tags = []
    payload: Dict[str, object] = {
        "language": default_language,
        "source": record.get("lexicon_entry", {}).get("source")
        if isinstance(record.get("lexicon_entry"), dict)
        else record.get("source"),
    }
    if "lexicon_entry" in record and isinstance(record["lexicon_entry"], dict):
        lex_entry = record["lexicon_entry"]
        payload["lemma"] = lex_entry.get("lemma")
        payload["definition"] = lex_entry.get("definition")
    if "lexicon_audio" in record and isinstance(record["lexicon_audio"], dict):
        audio_entry = record["lexicon_audio"]
        payload["transcript"] = audio_entry.get("transcript")
        payload["audio_path"] = audio_entry.get("audio_path")

    return StarRecord(
        star_id=star_id,
        name=name,
        embedding=emb_array,
        modalities=list(modalities),
        tags=list(tags),
        payload=payload,
    )


def _collect_stars(paths: Sequence[Path], *, language: str, limit: Optional[int]) -> List[StarRecord]:
    stars: List[StarRecord] = []
    for path in paths:
        records = _load_jsonl(path, limit)
        for rec in records:
            star = _to_star(rec, default_language=language)
            if star is not None:
                stars.append(star)
        if limit is not None and len(stars) >= limit:
            stars = stars[:limit]
            break
    if not stars:
        raise ValueError("No valid stars were parsed from the provided inputs")
    return stars


def _compute_positions(embeddings: np.ndarray, *, seed: int = 0) -> np.ndarray:
    if embeddings.shape[0] == 0:
        raise ValueError("Cannot compute positions for empty embedding array")
    # Centre embeddings
    centred = embeddings - embeddings.mean(axis=0, keepdims=True)
    if centred.shape[1] < 3:
        pad = np.zeros((centred.shape[0], 3 - centred.shape[1]), dtype=np.float32)
        coords = np.concatenate([centred, pad], axis=1)
        return coords.astype(np.float32)
    # PCA via SVD (deterministic for given content)
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    components = vt[:3, :].T
    coords = centred @ components
    return coords.astype(np.float32)


def _build_glb(
    stars: Sequence[StarRecord],
    positions: np.ndarray,
    embeddings: np.ndarray,
    *,
    label: str,
    language: str,
    zone: str,
) -> GLTF2:
    ids = [star.star_id for star in stars]
    names = [star.name for star in stars]
    metadata = []
    modality_counts: Dict[str, int] = {}
    for star in stars:
        metadata.append(
            {
                "name": star.name,
                "tags": list(star.tags),
                "modalities": list(star.modalities),
                "payload": star.payload,
            }
        )
        for mod in star.modalities:
            modality_counts[mod] = modality_counts.get(mod, 0) + 1

    pos_bytes = positions.astype(np.float32).tobytes()
    emb_bytes = embeddings.astype(np.float32).tobytes()
    blob = pos_bytes + emb_bytes

    buffer = Buffer(byteLength=len(blob))
    view_pos = BufferView(buffer=0, byteOffset=0, byteLength=len(pos_bytes), target=ARRAY_BUFFER)
    view_emb = BufferView(buffer=0, byteOffset=len(pos_bytes), byteLength=len(emb_bytes))

    accessor = Accessor(
        bufferView=0,
        byteOffset=0,
        componentType=5126,
        count=len(stars),
        type="VEC3",
        max=positions.max(axis=0).tolist(),
        min=positions.min(axis=0).tolist(),
    )

    extras = {
        "language": language,
        "label": label,
        "zone": zone,
        "modalities": modality_counts,
        "k3d": {
            "ids": ids,
            "names": names,
            "vectorsView": 0,
            "embeddingsView": 1,
            "embeddingDims": int(embeddings.shape[1]),
            "metadata": metadata,
            "neighbors": [[] for _ in stars],
        },
    }

    primitive = Primitive(attributes={"POSITION": 0}, mode=0, extras=extras)
    mesh = Mesh(primitives=[primitive])
    node = Node(mesh=0)
    scene = Scene(nodes=[0])

    gltf = GLTF2(
        asset=Asset(generator="knowledge3d.language_galaxy_builder"),
        buffers=[buffer],
        bufferViews=[view_pos, view_emb],
        accessors=[accessor],
        meshes=[mesh],
        nodes=[node],
        scenes=[scene],
        scene=0,
    )
    gltf.set_binary_blob(blob)
    return gltf


def _write_manifest(
    path: Path,
    *,
    stars: Sequence[StarRecord],
    positions: np.ndarray,
    embeddings: np.ndarray,
    language: str,
    label: str,
    zone: str,
) -> None:
    if not path:
        return
    bbox = {
        "min": positions.min(axis=0).tolist(),
        "max": positions.max(axis=0).tolist(),
    }
    manifest = {
        "language": language,
        "label": label,
        "zone": zone,
        "count": len(stars),
        "embedding_dim": int(embeddings.shape[1]),
        "bounding_box": bbox,
        "modalities": sorted({mod for star in stars for mod in star.modalities}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_language_galaxy(args: argparse.Namespace) -> None:
    input_paths = [Path(p).resolve() for p in args.input]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input JSONL not found: {path}")

    stars = _collect_stars(input_paths, language=args.language_id, limit=args.limit)
    embeddings = np.vstack([star.embedding for star in stars]).astype(np.float32)
    positions = _compute_positions(embeddings, seed=args.seed)

    glb = _build_glb(
        stars,
        positions,
        embeddings,
        label=args.label,
        language=args.language_id,
        zone=args.zone,
    )
    output_path = Path(args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    glb.save(output_path.as_posix())

    if args.manifest:
        _write_manifest(
            Path(args.manifest).resolve(),
            stars=stars,
            positions=positions,
            embeddings=embeddings,
            language=args.language_id,
            label=args.label,
            zone=args.zone,
        )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PTX-ready language galaxy from JSONL corpora")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Path to JSONL star corpus (repeat for multiple sources)",
    )
    parser.add_argument("--out", required=True, help="Output GLB path")
    parser.add_argument("--language-id", required=True, help="Language identifier (e.g., pt-BR, en-US)")
    parser.add_argument("--label", default="Language Galaxy", help="Display label stored in extras")
    parser.add_argument(
        "--zone",
        default="Zone 2 (Language Quadrant)",
        help="Logical zone placement stored in extras",
    )
    parser.add_argument("--manifest", help="Optional manifest JSON output path")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit across all inputs")
    parser.add_argument("--seed", type=int, default=0, help="Seed for PCA tie-breaking")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:  # pragma: no cover
    args = parse_args(argv)
    build_language_galaxy(args)


if __name__ == "__main__":  # pragma: no cover
    main()

