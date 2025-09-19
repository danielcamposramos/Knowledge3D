from __future__ import annotations

"""Build Spanish lexicon stars from Kaikki.org dictionary dumps."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence

from knowledge3d.tools.lexicon.common import build_star

POS_MAP = {
    "adj": "adjective",
    "adv": "adverb",
    "noun": "noun",
    "verb": "verb",
    "pron": "pronoun",
    "det": "determiner",
    "prep": "preposition",
    "phrase": "phrase",
    "interj": "interjection",
}

RELATION_KEYS = {
    "synonyms": "synonyms",
    "antonyms": "antonyms",
    "hypernyms": "hypernyms",
    "hyponyms": "hyponyms",
    "coordinate_terms": "coordinate_terms",
    "meronyms": "meronyms",
    "holonyms": "holonyms",
    "derived": "derived",
    "related": "related_terms",
    "see_also": "see_also",
}


def extract_words(items: Sequence[object]) -> List[str]:
    words: List[str] = []
    for item in items:
        if isinstance(item, str):
            words.append(item)
        elif isinstance(item, Mapping):
            value = item.get("word") or item.get("gloss") or item.get("sense")
            if isinstance(value, str):
                words.append(value)
    return words


def sense_relations(sense: Mapping[str, object]) -> Dict[str, List[str]]:
    relations: Dict[str, List[str]] = {}
    for raw_key, rel_key in RELATION_KEYS.items():
        payload = sense.get(raw_key)
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
            entries = extract_words(payload)
            if entries:
                relations[rel_key] = entries
    links = sense.get("links")
    if isinstance(links, Sequence):
        targets = []
        for link in links:
            if isinstance(link, Sequence) and len(link) >= 1:
                target = link[0]
                if isinstance(target, str):
                    targets.append(target)
        if targets:
            relations.setdefault("links", targets)
    return relations


def iter_stars(path: Path, limit: Optional[int]) -> Iterator[MutableMapping[str, object]]:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line:
                break
            entry = json.loads(line)
            if entry.get("lang_code") != "es":
                continue
            lemma = entry.get("word")
            if not isinstance(lemma, str):
                continue
            pos = POS_MAP.get(entry.get("pos"), entry.get("pos"))
            pronunciations = []
            for sound in entry.get("sounds", []) or []:
                if isinstance(sound, Mapping):
                    ipa = sound.get("ipa")
                    if isinstance(ipa, str) and ipa:
                        pronunciations.append(ipa)
            etymology = entry.get("etymology_text")
            forms = []
            for form in entry.get("forms", []) or []:
                if isinstance(form, Mapping):
                    value = form.get("form") or form.get("word")
                    if isinstance(value, str):
                        forms.append(value)
            senses = entry.get("senses") or []
            if not isinstance(senses, list):
                continue
            for sense in senses:
                if not isinstance(sense, Mapping):
                    continue
                glosses = sense.get("glosses") or []
                definition = None
                if isinstance(glosses, list) and glosses:
                    definition = glosses[0]
                elif isinstance(glosses, str):
                    definition = glosses
                    glosses = [glosses]
                examples = []
                for example in sense.get("examples", []) or []:
                    if isinstance(example, str):
                        examples.append(example)
                    elif isinstance(example, Mapping):
                        text = example.get("text")
                        if isinstance(text, str):
                            examples.append(text)
                relations = sense_relations(sense)
                synonyms = relations.get("synonyms", [])
                sense_id = sense.get("id") or f"{lemma}-{len(glosses)}"
                extra: Dict[str, object] = {
                    "lexicon_entry": {
                        "language": "es",
                        "lemma": lemma,
                        "pos": pos,
                        "sense_id": sense_id,
                        "definition": definition,
                        "glosses": glosses,
                        "examples": examples,
                        "synonyms": synonyms,
                        "pronunciations": pronunciations,
                        "forms": forms,
                        "etymology": etymology,
                        "tags": sense.get("tags"),
                        "source": {
                            "dataset": "kaikki.org-spanish",
                            "sense_id": sense_id,
                        },
                    }
                }
                embedding_parts = [lemma, definition or "", " ".join(synonyms)]
                star = build_star(
                    language="es",
                    source="kaikki",
                    lemma=lemma,
                    pos=pos,
                    sense_ref=str(sense_id),
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

    records = iter_stars(args.dictionary, args.limit)
    write_jsonl(args.out, records)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Spanish lexicon stars")
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=Path("../Knowledge3D.local/datasets/lexicons/spanish/kaikki.org-dictionary-Spanish.jsonl"),
        help="Path to kaikki.org Spanish dictionary JSONL",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("viewer/public/galaxy/working/lexicon_es_kaikki.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for generated stars")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:  # pragma: no cover
    args = parse_args(argv)
    build(args)


if __name__ == "__main__":  # pragma: no cover
    main()
