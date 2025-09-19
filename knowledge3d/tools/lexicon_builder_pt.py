from __future__ import annotations

"""Build Portuguese lexicon stars from OpenWordNet-PT."""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence

from knowledge3d.tools.lexicon.common import build_star

DEFAULT_RELATION_PREFIX = re.compile(r"^wn30_[a-z]+_(.+)$")

POS_BY_TYPE = {
    "NounSynset": "noun",
    "VerbSynset": "verb",
    "AdjectiveSynset": "adjective",
    "AdjectiveSatelliteSynset": "adjective",
    "AdverbSynset": "adverb",
}


def normalise_relation(name: str) -> str:
    match = DEFAULT_RELATION_PREFIX.match(name)
    key = match.group(1) if match else name
    key = re.sub("Of$", "", key)
    key = re.sub(r"([a-z])([A-Z])", r"\1_\2", key).lower()
    return key.replace("__", "_")


def extract_relations(payload: Mapping[str, object]) -> Dict[str, List[str]]:
    relations: Dict[str, List[str]] = {}
    for key, raw_value in payload.items():
        if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes)):
            continue
        if key.startswith("word_") or key.startswith("gloss_") or key.startswith("example_"):
            continue
        if key in {"wn30_synsetId", "rdf_type", "doc_id", "word_count_pt", "word_count_en"}:
            continue
        relation_key = normalise_relation(key)
        targets: List[str] = []
        for item in raw_value:
            if isinstance(item, Mapping):
                target = (
                    item.get("target_synset")
                    or item.get("target")
                    or item.get("synset")
                    or item.get("name")
                )
                if isinstance(target, str) and target:
                    targets.append(target)
            elif isinstance(item, str):
                targets.append(item)
        if targets:
            relations[relation_key] = targets
    return relations


def guess_pos(types: Sequence[object]) -> Optional[str]:
    for item in types:
        if not isinstance(item, str):
            continue
        for type_name, pos in POS_BY_TYPE.items():
            if item.endswith(type_name):
                return pos
    return None


def iter_stars(zip_path: Path, limit: Optional[int]) -> Iterator[MutableMapping[str, object]]:
    import zipfile

    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = [name for name in zf.namelist() if name.endswith(".jsonl") and "dump/wn-" in name]
        for member in sorted(members):
            with zf.open(member) as handle:
                for line in handle:
                    if not line:
                        break
                    record = json.loads(line)
                    payload = record.get("_source") or {}
                    words_pt = payload.get("word_pt") or []
                    if not words_pt:
                        continue
                    synset_id = str(payload.get("doc_id") or payload.get("wn30_synsetId", [""])[0])
                    definitions = payload.get("gloss_pt") or payload.get("gloss_en") or []
                    definition = None
                    if isinstance(definitions, list) and definitions:
                        definition = definitions[0]
                    elif isinstance(definitions, str):
                        definition = definitions
                        definitions = [definitions]
                    examples = payload.get("example_pt") or payload.get("example_en") or []
                    if isinstance(examples, str):
                        examples = [examples]
                    translations = payload.get("word_en") or []
                    frames = payload.get("wn30_frame") or []
                    pos = guess_pos(payload.get("rdf_type") or [])
                    relations = extract_relations(payload)
                    for lemma in words_pt:
                        if not isinstance(lemma, str) or not lemma.strip():
                            continue
                        lemma_clean = lemma.strip()
                        synonyms = [w for w in words_pt if isinstance(w, str) and w != lemma]
                        extra: Dict[str, object] = {
                            "lexicon_entry": {
                                "language": "pt",
                                "lemma": lemma_clean,
                                "pos": pos,
                                "sense_id": synset_id,
                                "synset_id": synset_id,
                                "definition": definition,
                                "definitions": definitions,
                                "examples": examples,
                                "synonyms": synonyms,
                                "translations": translations,
                                "frames": frames,
                                "source": {
                                    "dataset": "openwordnet-pt",
                                    "archive_member": Path(member).name,
                                },
                            }
                        }
                        embedding_parts = [lemma_clean, definition or "", " ".join(translations)]
                        star = build_star(
                            language="pt",
                            source="openwordnet-pt",
                            lemma=lemma_clean,
                            pos=pos,
                            sense_ref=synset_id,
                            definition=definition,
                            embedding_parts=embedding_parts,
                            relations=relations,
                            extra=extra,
                        )
                        yield star
                        count += 1
                        if limit and count >= limit:
                            return


def build(args: argparse.Namespace) -> None:
    from knowledge3d.tools.lexicon.common import write_jsonl

    records = iter_stars(args.own_zip, args.limit)
    write_jsonl(args.out, records)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Portuguese lexicon stars")
    parser.add_argument(
        "--own-zip",
        type=Path,
        default=Path("../Knowledge3D.local/datasets/lexicons/portuguese_br/openwordnet-pt.zip"),
        help="Path to openwordnet-pt zip",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("viewer/public/galaxy/working/lexicon_pt_openwordnet.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for generated stars")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:  # pragma: no cover
    args = parse_args(argv)
    build(args)


if __name__ == "__main__":  # pragma: no cover
    main()
