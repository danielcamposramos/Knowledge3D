"""Universal Dependencies Grammar Galaxy builder for Phase 7.A.1 Slice 4.

Ingestion-path only. Parses selected UD treebanks with the standard library,
aggregates language-level construction/morphology evidence, and emits
meaning-centric grammar template/rule stars composed from existing RPN words.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from knowledge3d.ingestion.canonical_lookup import (
    canonical_grammar_template_id,
    canonical_slug,
)
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar


DEFAULT_UD_ROOT = Path("/K3D/K3D_llama_cpp/datasets/ud/ud-treebanks-v2.14")
LOCAL_GRAMMAR_PATH = Path("/K3D/Knowledge3D.local/assets/grammar/UD_GRAMMAR_STARS.jsonl")

CANONICAL_TREEBANKS = {
    "en": "UD_English-EWT",
    "pt": "UD_Portuguese-Bosque",
    "es": "UD_Spanish-GSD",
    "fr": "UD_French-GSD",
    "de": "UD_German-GSD",
    "it": "UD_Italian-ISDT",
    "ja": "UD_Japanese-GSD",
    "zh": "UD_Chinese-GSD",
    "ru": "UD_Russian-SynTagRus",
}

BASE_TEMPLATE_KEYS = ("copula", "periphrastic_explanation")
CORE_CONSTRUCTIONS = ("nsubj_obj", "det_noun", "amod_noun", "case_oblique", "aux_verb")


@dataclass(frozen=True)
class UDToken:
    token_id: int
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: int
    deprel: str


@dataclass
class UDGrammarBuild:
    stars: dict[str, dict]
    canonical_entries: list[dict[str, object]]
    treebank_stats: dict[str, dict[str, int]]


def iter_conllu_sentences(path: str | Path) -> Iterator[list[UDToken]]:
    sentence: list[UDToken] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                if sentence:
                    yield sentence
                    sentence = []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) != 10:
                continue
            token_id = cols[0]
            if "-" in token_id or "." in token_id:
                continue
            try:
                tid = int(token_id)
                head = int(cols[6]) if cols[6].isdigit() else 0
            except ValueError:
                continue
            sentence.append(
                UDToken(
                    token_id=tid,
                    form=cols[1],
                    lemma=cols[2],
                    upos=cols[3],
                    xpos=cols[4],
                    feats=cols[5],
                    head=head,
                    deprel=cols[7],
                )
            )
    if sentence:
        yield sentence


def _sentence_by_id(sentence: list[UDToken]) -> dict[int, UDToken]:
    return {token.token_id: token for token in sentence}


def _construction_keys(sentence: list[UDToken]) -> set[str]:
    by_id = _sentence_by_id(sentence)
    relations = {token.deprel for token in sentence}
    keys: set[str] = set()
    if "nsubj" in relations and "obj" in relations:
        keys.add("nsubj_obj")
    if any(token.deprel == "det" and by_id.get(token.head, token).upos == "NOUN" for token in sentence):
        keys.add("det_noun")
    if any(token.deprel == "amod" and by_id.get(token.head, token).upos in {"NOUN", "PROPN"} for token in sentence):
        keys.add("amod_noun")
    if any(token.deprel == "case" and by_id.get(token.head, token).deprel in {"obl", "nmod"} for token in sentence):
        keys.add("case_oblique")
    if any(token.deprel == "aux" for token in sentence):
        keys.add("aux_verb")
    if any(token.deprel == "cop" for token in sentence):
        keys.add("copula")
    return keys


def _word_order(sentence: list[UDToken]) -> str:
    positions: dict[str, int] = {}
    for token in sentence:
        if token.deprel == "nsubj":
            positions.setdefault("S", token.token_id)
        elif token.upos in {"VERB", "AUX"}:
            positions.setdefault("V", token.token_id)
        elif token.deprel == "obj":
            positions.setdefault("O", token.token_id)
    if set(positions) >= {"S", "V", "O"}:
        return "".join(label for label, _pos in sorted(positions.items(), key=lambda item: item[1]))
    return ""


def _aggregate_files(paths: Iterable[Path]) -> dict:
    stats = {
        "sentences": 0,
        "tokens": 0,
        "construction_counts": Counter(),
        "deprel_counts": Counter(),
        "upos_counts": Counter(),
        "lemma_feats": defaultdict(Counter),
        "word_orders": Counter(),
    }
    for path in paths:
        for sentence in iter_conllu_sentences(path):
            stats["sentences"] += 1
            stats["tokens"] += len(sentence)
            stats["construction_counts"].update(_construction_keys(sentence))
            order = _word_order(sentence)
            if order:
                stats["word_orders"][order] += 1
            for token in sentence:
                stats["deprel_counts"][token.deprel] += 1
                stats["upos_counts"][token.upos] += 1
                if token.feats and token.feats != "_":
                    stats["lemma_feats"][(token.lemma, token.upos)].update(token.feats.split("|"))
    return stats


def _treebank_files(ud_root: Path, treebank: str) -> list[Path]:
    root = ud_root / treebank
    files = sorted(root.glob("*-ud-train.conllu"))
    if not files:
        files = sorted(root.glob("*.conllu"))
    return files


def _template_star(language: str, template_key: str, meaning_rpn: str, *, metadata: Mapping[str, object]) -> dict:
    star = MeaningCentricStar(
        star_id=canonical_grammar_template_id(language, template_key),
        meaning_class="rule",
        domain=f"Grammar/{language}",
        meaning_rpn=meaning_rpn,
        lod_class="LOD_SUMMARY",
    )
    payload = star.to_dict()
    payload.update(
        {
            "name": f"{language} {template_key.replace('_', ' ')} grammar template",
            "language": language,
            "template_key": template_key,
            "kind": "grammar_template",
            "metadata": dict(metadata),
        }
    )
    return payload


def _rule_star(language: str, rule_key: str, meaning_rpn: str, *, metadata: Mapping[str, object]) -> dict:
    star_id = f"grammar_rule_{language}_{canonical_slug(rule_key)}"
    star = MeaningCentricStar(
        star_id=star_id,
        meaning_class="rule",
        domain=f"Grammar/{language}",
        meaning_rpn=meaning_rpn,
        lod_class="LOD_SUMMARY",
    )
    payload = star.to_dict()
    payload.update(
        {
            "name": f"{language} {rule_key.replace('_', ' ')} grammar rule",
            "language": language,
            "rule_key": rule_key,
            "kind": "grammar_rule",
            "metadata": dict(metadata),
        }
    )
    return payload


def _canonical_template_entry(language: str, template_key: str) -> dict[str, object]:
    return {
        "kind": "grammar_template",
        "key": f"{language}:{template_key}",
        "star_id": canonical_grammar_template_id(language, template_key),
        "metadata": {"language": language, "template": template_key},
    }


def _canonical_rule_entry(language: str, rule_key: str, star_id: str, metadata: Mapping[str, object]) -> dict[str, object]:
    return {
        "kind": "grammar_rule",
        "key": f"{language}:{rule_key}",
        "star_id": star_id,
        "metadata": dict(metadata),
    }


def build_language_grammar(language: str, treebank: str, paths: list[Path]) -> tuple[list[dict], list[dict[str, object]], dict[str, int]]:
    stats = _aggregate_files(paths)
    rows: list[dict] = []
    canonical_entries: list[dict[str, object]] = []

    for key in BASE_TEMPLATE_KEYS:
        meaning = (
            "MEANING_DECOMPOSE GRAMMAR_SELECT WORD_RESOLVE MORPHOLOGY_APPLY SURFACE_EMIT"
            if key == "periphrastic_explanation"
            else "MEANING_DECOMPOSE GRAMMAR_SELECT COPULA_BIND WORD_RESOLVE SURFACE_EMIT"
        )
        row = _template_star(
            language,
            key,
            meaning,
            metadata={"treebank": treebank, "source": "phase7_ud_seed"},
        )
        rows.append(row)
        canonical_entries.append(_canonical_template_entry(language, key))

    construction_counts: Counter = stats["construction_counts"]
    for key in CORE_CONSTRUCTIONS:
        if construction_counts.get(key, 0) <= 0 and key != "copula":
            continue
        row = _template_star(
            language,
            key,
            f"MEANING_DECOMPOSE GRAMMAR_SELECT {key.upper()} WORD_RESOLVE SURFACE_EMIT",
            metadata={
                "treebank": treebank,
                "count": int(construction_counts.get(key, 0)),
                "source": "universal_dependencies",
            },
        )
        rows.append(row)
        canonical_entries.append(_canonical_template_entry(language, key))

    for upos, count in stats["upos_counts"].most_common(12):
        rule_key = f"upos_{upos.lower()}"
        row = _rule_star(
            language,
            rule_key,
            f"LEMMA_RECALL UPOS_{upos} MORPHOLOGY_APPLY STORE",
            metadata={"treebank": treebank, "upos": upos, "count": int(count)},
        )
        rows.append(row)
        canonical_entries.append(_canonical_rule_entry(language, rule_key, row["star_id"], row["metadata"]))

    for order, count in stats["word_orders"].most_common(3):
        rule_key = f"word_order_{order.lower()}"
        row = _rule_star(
            language,
            rule_key,
            f"GRAMMAR_SELECT WORD_ORDER_{order} SURFACE_EMIT",
            metadata={"treebank": treebank, "word_order": order, "count": int(count)},
        )
        rows.append(row)
        canonical_entries.append(_canonical_rule_entry(language, rule_key, row["star_id"], row["metadata"]))

    return rows, canonical_entries, {"sentences": int(stats["sentences"]), "tokens": int(stats["tokens"]), "rows": len(rows)}


def build_ud_grammar_galaxy(
    *,
    ud_root: str | Path = DEFAULT_UD_ROOT,
    treebanks: Mapping[str, str] | None = None,
) -> UDGrammarBuild:
    ud_root = Path(ud_root)
    selected = dict(treebanks or CANONICAL_TREEBANKS)
    rows: dict[str, dict] = {}
    canonical_entries: list[dict[str, object]] = []
    treebank_stats: dict[str, dict[str, int]] = {}
    for language, treebank in selected.items():
        paths = _treebank_files(ud_root, treebank)
        if not paths:
            raise FileNotFoundError(f"ud_treebank_missing:{language}:{treebank}:{ud_root}")
        language_rows, language_entries, stats = build_language_grammar(language, treebank, paths)
        for row in language_rows:
            rows[row["star_id"]] = row
        canonical_entries.extend(language_entries)
        treebank_stats[language] = stats
    return UDGrammarBuild(stars=rows, canonical_entries=canonical_entries, treebank_stats=treebank_stats)


def write_ud_grammar_build(build: UDGrammarBuild, output_path: str | Path = LOCAL_GRAMMAR_PATH) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for star_id in sorted(build.stars):
            handle.write(json.dumps(build.stars[star_id], ensure_ascii=False) + "\n")
    output_path.with_suffix(output_path.suffix + ".canonical.json").write_text(
        json.dumps(build.canonical_entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_path.with_suffix(output_path.suffix + ".meta.json").write_text(
        json.dumps(
            {
                "star_count": len(build.stars),
                "canonical_entry_count": len(build.canonical_entries),
                "treebank_stats": build.treebank_stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def register_ud_grammar_canonical_entries(lookup, build: UDGrammarBuild) -> int:
    count = 0
    for entry in build.canonical_entries:
        lookup.register(
            kind=str(entry["kind"]),
            key=str(entry["key"]),
            star_id=str(entry["star_id"]),
            metadata=dict(entry.get("metadata") or {}),
        )
        count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build local UD Grammar Galaxy stars.")
    parser.add_argument("--ud-root", default=str(DEFAULT_UD_ROOT))
    parser.add_argument("--out", default=str(LOCAL_GRAMMAR_PATH))
    args = parser.parse_args(argv)
    build = build_ud_grammar_galaxy(ud_root=args.ud_root)
    out = write_ud_grammar_build(build, args.out)
    print(
        f"grammar_stars={len(build.stars)} "
        f"canonical_entries={len(build.canonical_entries)} "
        f"languages={','.join(sorted(build.treebank_stats))} "
        f"out={out}"
    )
    return 0


__all__ = [
    "BASE_TEMPLATE_KEYS",
    "CANONICAL_TREEBANKS",
    "DEFAULT_UD_ROOT",
    "LOCAL_GRAMMAR_PATH",
    "UDGrammarBuild",
    "UDToken",
    "build_language_grammar",
    "build_ud_grammar_galaxy",
    "iter_conllu_sentences",
    "register_ud_grammar_canonical_entries",
    "write_ud_grammar_build",
]


if __name__ == "__main__":
    raise SystemExit(main())
