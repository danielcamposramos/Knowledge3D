from __future__ import annotations

"""Build English lexicon stars from English WordNet 2024."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional

import yaml

from knowledge3d.tools.lexicon.common import build_star

WORDNET_PREFIX = "english-wordnet-2024-edition/src/yaml/"
ENTRIES_PREFIX = WORDNET_PREFIX + "entries-"
SYNSET_PREFIXES = ("noun.", "verb.", "adj.", "adv.")
DESCRIPTIVE_KEYS = {"definition", "example", "members", "partOfSpeech", "ili", "wikidata"}

POS_MAP = {
    "n": "noun",
    "v": "verb",
    "a": "adjective",
    "s": "adjective",
    "r": "adverb",
}


def load_synsets(zip_path: Path) -> Dict[str, MutableMapping[str, object]]:
    import zipfile

    synsets: Dict[str, MutableMapping[str, object]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            if not name.startswith(WORDNET_PREFIX):
                continue
            stem = name[len(WORDNET_PREFIX) :]
            if not stem.endswith(".yaml"):
                continue
            if not stem.startswith(SYNSET_PREFIXES):
                continue
            with zf.open(name) as handle:
                data = yaml.safe_load(handle) or {}
            for synset_id, payload in data.items():
                if isinstance(payload, dict):
                    synsets[synset_id] = payload
    return synsets


def extract_relations(payload: Mapping[str, object]) -> Dict[str, List[str]]:
    relations: Dict[str, List[str]] = {}
    for key, raw_value in payload.items():
        if key in DESCRIPTIVE_KEYS:
            continue
        if not isinstance(raw_value, list):
            continue
        items: List[str] = []
        for entry in raw_value:
            if isinstance(entry, str):
                items.append(entry)
            elif isinstance(entry, Mapping):
                target = (
                    str(entry.get("target_synset") or entry.get("synset") or entry.get("target") or "")
                )
                if target:
                    items.append(target)
        if items:
            relations[key] = items
    return relations


def iter_entries(zip_path: Path, synsets: Mapping[str, Mapping[str, object]], limit: Optional[int]) -> Iterator[MutableMapping[str, object]]:
    import zipfile

    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        entry_names = [name for name in zf.namelist() if name.startswith(ENTRIES_PREFIX)]
        for name in sorted(entry_names):
            with zf.open(name) as handle:
                data = yaml.safe_load(handle) or {}
            if not isinstance(data, Mapping):
                continue
            for lemma, pos_map in data.items():
                if not isinstance(pos_map, Mapping):
                    continue
                for pos_code, info in pos_map.items():
                    if not isinstance(info, Mapping):
                        continue
                    pronunciations = []
                    for entry in info.get("pronunciation", []) or []:
                        if isinstance(entry, Mapping):
                            value = entry.get("value")
                            if isinstance(value, str) and value:
                                pronunciations.append(value)
                    forms = []
                    for form in info.get("form", []) or []:
                        if isinstance(form, str):
                            forms.append(form)
                    senses = info.get("sense") or []
                    if not isinstance(senses, list):
                        continue
                    for sense in senses:
                        if not isinstance(sense, Mapping):
                            continue
                        synset_id = str(sense.get("synset") or "")
                        sense_id = str(sense.get("id") or synset_id or lemma)
                        synset_payload = synsets.get(synset_id, {})
                        definitions = synset_payload.get("definition") or []
                        definition = definitions[0] if definitions else None
                        examples = synset_payload.get("example") or []
                        members = synset_payload.get("members") or []
                        ili = synset_payload.get("ili")
                        relations = extract_relations(synset_payload)
                        entry_payload: Dict[str, object] = {
                            "lexicon_entry": {
                                "language": "en",
                                "lemma": lemma,
                                "pos": POS_MAP.get(pos_code, pos_code),
                                "sense_id": sense_id,
                                "synset_id": synset_id or None,
                                "definition": definition,
                                "definitions": definitions,
                                "examples": examples,
                                "synonyms": members,
                                "pronunciations": pronunciations,
                                "forms": forms,
                                "ili": ili,
                                "source": {
                                    "dataset": "english-wordnet-2024",
                                    "archive_member": Path(name).name,
                                },
                                "sense_meta": {k: v for k, v in sense.items() if k not in {"id", "synset"}},
                            }
                        }
                        embedding_parts = [lemma, definition or "", " ".join(members), sense_id]
                        star = build_star(
                            language="en",
                            source="wordnet2024",
                            lemma=lemma,
                            pos=POS_MAP.get(pos_code, pos_code),
                            sense_ref=sense_id,
                            definition=definition,
                            embedding_parts=embedding_parts,
                            relations=relations,
                            extra=entry_payload,
                        )
                        yield star
                        count += 1
                        if limit and count >= limit:
                            return


def build(args: argparse.Namespace) -> None:
    synsets = load_synsets(args.wordnet_zip)
    records = iter_entries(args.wordnet_zip, synsets, args.limit)
    from knowledge3d.tools.lexicon.common import write_jsonl

    write_jsonl(args.out, records)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build English lexicon stars")
    parser.add_argument(
        "--wordnet-zip",
        type=Path,
        default=Path("../Knowledge3D.local/datasets/lexicons/english/english-wordnet-2024-edition.zip"),
        help="Path to english-wordnet-2024 zip",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("viewer/public/galaxy/working/lexicon_en_wordnet.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for generated stars")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:  # pragma: no cover
    args = parse_args(argv)
    build(args)


if __name__ == "__main__":  # pragma: no cover
    main()
