from __future__ import annotations

"""Build Mandarin lexicon stars from CC-CEDICT."""

import argparse
import re
from io import TextIOWrapper
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, MutableMapping, Optional

from knowledge3d.tools.lexicon.common import build_star

ENTRY_RE = re.compile(r"^(?P<traditional>\S+)\s+(?P<simplified>\S+)\s+\[(?P<pinyin>[^\]]+)\]\s+/(?P<defs>.+)/$")
VARIANT_RE = re.compile(r"variant of ([^/;]+)")
MEASURE_RE = re.compile(r"CL:([^/]+)")


def parse_definitions(def_blob: str) -> Dict[str, List[str]]:
    raw_defs = [item.strip() for item in def_blob.split("/") if item.strip()]
    variants: List[str] = []
    measure_words: List[str] = []
    senses: List[str] = []
    for definition in raw_defs:
        match_var = VARIANT_RE.search(definition)
        if match_var:
            variants.append(match_var.group(1).strip())
        match_measure = MEASURE_RE.search(definition)
        if match_measure:
            measure_words.append(match_measure.group(1).strip())
        senses.append(definition)
    return {
        "definitions": senses,
        "variants": variants,
        "measure_words": measure_words,
    }


def iter_stars(zip_path: Path, limit: Optional[int]) -> Iterator[MutableMapping[str, object]]:
    import zipfile

    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("cedict_ts.u8") as binary:
            handle = TextIOWrapper(binary, encoding="utf-8")
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                match = ENTRY_RE.match(line)
                if not match:
                    continue
                trad = match.group("traditional")
                simp = match.group("simplified")
                pinyin = match.group("pinyin")
                components = parse_definitions(match.group("defs"))
                definitions = components["definitions"]
                definition = definitions[0] if definitions else None
                relations: Dict[str, List[str]] = {}
                if components["variants"]:
                    relations["variant_of"] = components["variants"]
                extra: Dict[str, object] = {
                    "lexicon_entry": {
                        "language": "zh",
                        "lemma": simp,
                        "pos": None,
                        "sense_id": f"cedict:{trad}:{pinyin}",
                        "definition": definition,
                        "definitions": definitions,
                        "pronunciations": [pinyin],
                        "traditional": trad,
                        "measure_words": components["measure_words"],
                        "source": {
                            "dataset": "cc-cedict",
                        },
                    }
                }
                embedding_parts = [simp, definition or "", trad, pinyin]
                star = build_star(
                    language="zh",
                    source="cedict",
                    lemma=simp,
                    pos=None,
                    sense_ref=f"cedict-{trad}-{pinyin}",
                    definition=definition,
                    embedding_parts=embedding_parts,
                    relations=relations,
                    extra=extra,
                    tags=["script:han"],
                )
                yield star
                count += 1
                if limit and count >= limit:
                    return


def build(args: argparse.Namespace) -> None:
    from knowledge3d.tools.lexicon.common import write_jsonl

    records = iter_stars(args.cedict_zip, args.limit)
    write_jsonl(args.out, records)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Mandarin lexicon stars")
    parser.add_argument(
        "--cedict-zip",
        type=Path,
        default=Path("../Knowledge3D.local/datasets/lexicons/zh/cedict_1_0_ts_utf-8_mdbg.zip"),
        help="Path to CC-CEDICT zip",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("viewer/public/galaxy/working/lexicon_zh_cedict.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for generated stars")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:  # pragma: no cover
    args = parse_args(argv)
    build(args)


if __name__ == "__main__":  # pragma: no cover
    main()
