#!/usr/bin/env python3
"""Load H19 and B3 stars into an initialized Knowledgeverse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse  # noqa: E402
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar  # noqa: E402
from knowledge3d.tools.benchmark_health_check import load_questions  # noqa: E402


DEFAULT_STORAGE_ROOT = Path("/K3D/Knowledge3D.local")
DEFAULT_MEANING_PATH = DEFAULT_STORAGE_ROOT / "galaxies" / "meaning_layer_stars.jsonl"
DEFAULT_MMLU_PATH = DEFAULT_STORAGE_ROOT / "galaxies" / "proceduralized_mmlu_val_10.jsonl"
DEFAULT_GSM8K_PATH = DEFAULT_STORAGE_ROOT / "galaxies" / "proceduralized_gsm8k_train_10.jsonl"
DEFAULT_COUNTS = {
    "arc": 10,
    "math": 20,
    "gsm8k": 10,
    "lhe": 10,
    "mmlu": 50,
}
TOKEN_RE = re.compile(r"[a-z0-9]+")
ROUTED_MEANING_GALAXIES = ("Math", "Reality", "Drawing", "Grammar", "Language")
GALAXY_BY_DOMAIN = {
    "mathematics": "Math",
    "physics": "Reality",
    "biology": "Reality",
    "language": "Language",
    "tools": "Tool",
    "visual": "Drawing",
    "audio": "Tool",
    "general": "Language",
}
_MATH_LEMMAS = {
    "addition",
    "add",
    "plus",
    "subtraction",
    "subtract",
    "minus",
    "multiplication",
    "multiply",
    "times",
    "division",
    "divide",
    "quotient",
    "remainder",
    "modulo",
    "mod",
    "equation",
    "inequality",
    "derivative",
    "integral",
    "function",
    "algebra",
    "geometry",
    "calculus",
    "trigonometry",
    "trigonometric",
    "combinatorics",
    "counting",
    "number",
    "sum",
    "product",
    "fraction",
    "ratio",
    "exponent",
    "power",
    "logarithm",
    "sequence",
    "series",
    "polynomial",
    "matrix",
    "vector",
    "determinant",
    "permutation",
    "permutations",
    "combination",
    "combinations",
    "binomial",
    "factorial",
    "probability",
    "statistics",
    "theorem",
    "proof",
    "axiom",
    "hypothesis",
    "statistic",
    "average",
    "mean",
    "median",
    "variance",
    "formula",
    "computation",
    "arithmetic",
    "calculate",
    "compute",
    "solve",
    "equal",
    "greater",
    "less",
    "infinite",
    "finite",
    "zero",
    "prime",
    "square",
    "cube",
    "root",
    "factor",
    "coefficient",
    "constant",
    "variable",
    "graph",
    "coordinate",
    "angle",
    "triangle",
    "circle",
    "area",
    "volume",
    "perimeter",
    "circumference",
    "diameter",
    "radius",
    "symmetry",
    "proportion",
    "percentage",
    "decimal",
    "integer",
    "prealgebra",
    "precalculus",
    "number_theory",
    "prime",
    "divisibility",
    "gcd",
    "lcm",
    "congruence",
    "slope",
    "midpoint",
    "distance",
    "radius",
    "diameter",
    "circumference",
    "area",
    "volume",
}
_PHYSICS_LEMMAS = {
    "force",
    "energy",
    "mass",
    "velocity",
    "acceleration",
    "gravity",
    "momentum",
    "pressure",
    "temperature",
    "wave",
    "frequency",
    "light",
    "electric",
    "magnetic",
    "atom",
    "molecule",
    "element",
    "compound",
    "reaction",
    "density",
    "friction",
    "inertia",
    "orbit",
    "field",
}
_VISUAL_LEMMAS = {
    "line",
    "circle",
    "rectangle",
    "square",
    "triangle",
    "shape",
    "color",
    "pattern",
    "grid",
    "pixel",
    "image",
    "draw",
    "sketch",
}
_GRAMMAR_LEMMAS = {
    "grammar",
    "language",
    "linguistic",
    "linguistics",
    "verb",
    "noun",
    "adjective",
    "adverb",
    "pronoun",
    "syntax",
    "sentence",
    "phrase",
    "clause",
    "alphabet",
    "character",
    "word",
    "syllable",
    "phoneme",
    "morpheme",
    "punctuation",
}
_STOPWORD_LEMMAS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "can",
    "could",
    "must",
    "need",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "out",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "and",
    "but",
    "or",
    "if",
    "while",
    "that",
    "this",
    "these",
    "those",
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "you",
    "your",
    "he",
    "him",
    "his",
    "she",
    "her",
    "it",
    "its",
    "they",
    "them",
    "their",
    "what",
    "which",
    "who",
    "whom",
    "whose",
}

_MATH_EXECUTION_SPECS = [
    (
        {"addition", "add", "plus", "sum"},
        {
            "operation_key": "addition",
            "rpn_program": "ARG0 ARG1 +",
            "template_ref": "math_template_arithmetic_chain_gpu",
            "math_type": "Prealgebra",
            "keywords": ["addition", "sum", "plus", "add"],
            "semantics": "add two values after navigating through the language meaning layer",
        },
    ),
    (
        {"subtraction", "subtract", "minus", "difference"},
        {
            "operation_key": "subtraction",
            "rpn_program": "ARG0 ARG1 -",
            "template_ref": "math_template_arithmetic_chain_gpu",
            "math_type": "Prealgebra",
            "keywords": ["subtraction", "difference", "minus", "subtract"],
            "semantics": "subtract one value from another after language-side meaning lookup",
        },
    ),
    (
        {"multiplication", "multiply", "times", "product"},
        {
            "operation_key": "multiplication",
            "rpn_program": "ARG0 ARG1 *",
            "template_ref": "math_template_arithmetic_chain_gpu",
            "math_type": "Prealgebra",
            "keywords": ["multiplication", "product", "times", "multiply"],
            "semantics": "multiply values after navigating from language meaning to math execution",
        },
    ),
    (
        {"division", "divide", "quotient"},
        {
            "operation_key": "division",
            "rpn_program": "ARG0 ARG1 /",
            "template_ref": "math_template_arithmetic_chain_gpu",
            "math_type": "Prealgebra",
            "keywords": ["division", "quotient", "divide"],
            "semantics": "divide one value by another after language meaning resolution",
        },
    ),
    (
        {"factorial"},
        {
            "operation_key": "factorial",
            "rpn_program": "ARG0 factorial",
            "template_ref": "math_template_factorial_gpu",
            "math_type": "Counting & Probability",
            "keywords": ["factorial", "n!", "permutation"],
            "semantics": "compute factorial values inside the Math galaxy after language navigation",
        },
    ),
    (
        {"combination", "combinations", "binomial"},
        {
            "operation_key": "combination",
            "rpn_program": "ARG0 ARG1 binom",
            "template_ref": "math_template_binomial_gpu",
            "math_type": "Counting & Probability",
            "keywords": ["combination", "choose", "binomial coefficient"],
            "semantics": "compute combinations after following the language meaning bridge",
        },
    ),
    (
        {"permutation", "permutations"},
        {
            "operation_key": "permutation",
            "rpn_program": "ARG0 factorial ARG0 ARG1 - factorial /",
            "template_ref": "math_template_permutation_gpu",
            "math_type": "Counting & Probability",
            "keywords": ["permutation", "arrangement", "ordered selection"],
            "semantics": "compute permutations after meaning lookup in the Language galaxy",
        },
    ),
    (
        {"equation", "solve", "algebra"},
        {
            "operation_key": "equation_solve",
            "rpn_program": "ARG2 ARG1 - ARG0 /",
            "template_ref": "math_template_linear_equation_ax_plus_b_eq_c_gpu",
            "math_type": "Algebra",
            "keywords": ["equation", "solve", "linear equation", "algebra"],
            "semantics": "solve linear equations after language-side navigation into Math execution",
        },
    ),
    (
        {"gcd"},
        {
            "operation_key": "gcd",
            "rpn_program": "ARG0 ARG1 gcd",
            "template_ref": "math_template_gcd_gpu",
            "math_type": "Number Theory",
            "keywords": ["gcd", "greatest common divisor", "greatest common factor"],
            "semantics": "compute greatest common divisors after meaning lookup",
        },
    ),
    (
        {"lcm"},
        {
            "operation_key": "lcm",
            "rpn_program": "ARG0 ARG1 * ARG0 ARG1 gcd / abs",
            "template_ref": "math_template_lcm_gpu",
            "math_type": "Number Theory",
            "keywords": ["lcm", "least common multiple"],
            "semantics": "compute least common multiples after following the language meaning bridge",
        },
    ),
    (
        {"remainder", "modulo", "mod"},
        {
            "operation_key": "remainder",
            "rpn_program": "ARG0 ARG1 mod",
            "template_ref": "math_template_remainder_gpu",
            "math_type": "Number Theory",
            "keywords": ["remainder", "modulo", "mod"],
            "semantics": "compute modular remainders after language meaning navigation",
        },
    ),
    (
        {"sequence", "series"},
        {
            "operation_key": "sequence_rule",
            "rpn_program": "LOOKUP_SEQUENCE_RULE",
            "template_ref": "math_template_arithmetic_nth_term_gpu",
            "math_type": "Precalculus",
            "keywords": ["sequence", "series", "nth term", "progression"],
            "semantics": "navigate from the language meaning of sequences into Math galaxy execution rules",
        },
    ),
]


def load_stars_from_jsonl(path: Path) -> list[MeaningCentricStar]:
    stars: list[MeaningCentricStar] = []
    if not path.exists():
        return stars
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if any(key in payload for key in ("star_id", "meaning_rpn", "surface_forms")):
                stars.append(MeaningCentricStar.from_dict(payload))
            else:
                stars.append(MeaningCentricStar.from_galaxy_entry(payload))
    return stars


def _tokenize_text(text: str) -> set[str]:
    normalized = str(text or "").lower().replace("_", " ").replace("-", " ")
    return {token for token in TOKEN_RE.findall(normalized) if token}


def _primary_english_lemma(star: MeaningCentricStar) -> str:
    surface = star.surface_forms.get("en")
    if surface is None:
        return ""
    word_ref = str(surface.word_ref or "").strip().lower()
    if re.fullmatch(r"[a-z]{2}_.+", word_ref):
        return word_ref.split("_", 1)[1]
    return word_ref


def _primary_meaning_tokens(star: MeaningCentricStar) -> set[str]:
    meaning_tokens = str(star.meaning_rpn or "").split()
    if meaning_tokens[:1] != ["SYNSET"]:
        return _tokenize_text(star.meaning_rpn)
    primary_tokens: list[str] = []
    for token in meaning_tokens[2:]:
        if token == "DEF":
            break
        if token.startswith("LANG_"):
            continue
        primary_tokens.append(token)
    return _tokenize_text(" ".join(primary_tokens))


def _star_reference_tokens(star: MeaningCentricStar) -> set[str]:
    tokens: set[str] = set()
    tokens.update(_tokenize_text(_primary_english_lemma(star)))
    tokens.update(_primary_meaning_tokens(star))
    return tokens


def _matches_domain_tokens(tokens: set[str], vocabulary: set[str]) -> bool:
    return not tokens.isdisjoint(vocabulary)


def collect_benchmark_keywords(counts: dict[str, int] | None = None) -> set[str]:
    keywords: set[str] = set()
    for suite, count in dict(counts or DEFAULT_COUNTS).items():
        for row in load_questions(suite, count):
            keywords.update(_tokenize_text(str(row.get("question") or "")))
            keywords.update(_tokenize_text(str(row.get("expected") or "")))
            payload = dict(row.get("payload") or {})
            for key in ("question_text", "problem_text", "correct_answer", "correct_letter", "subject"):
                keywords.update(_tokenize_text(str(payload.get(key) or "")))
            options = payload.get("options")
            if isinstance(options, list):
                for option in options:
                    keywords.update(_tokenize_text(str(option)))
    return keywords


def _english_surface_text(star: MeaningCentricStar) -> str:
    return _primary_english_lemma(star)


def _surface_form_aliases(star: MeaningCentricStar) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()
    for language, surface in sorted(star.surface_forms.items()):
        word_ref = str(surface.word_ref or "").strip().replace("_", " ")
        if not word_ref:
            continue
        alias = word_ref.lower()
        if alias in seen:
            continue
        seen.add(alias)
        aliases.append(alias)
        if language == "en" and alias.startswith("en "):
            trimmed = alias[3:].strip()
            if trimmed and trimmed not in seen:
                seen.add(trimmed)
                aliases.append(trimmed)
    return aliases


def _math_execution_spec(tokens: set[str]) -> dict[str, Any] | None:
    for vocabulary, spec in _MATH_EXECUTION_SPECS:
        if not tokens.isdisjoint(vocabulary):
            return dict(spec)
    return None


def _language_math_bridge_id(star: MeaningCentricStar) -> str:
    return f"math_exec_{star.star_id}"


def build_language_math_bridge_entry(star: MeaningCentricStar) -> dict[str, Any]:
    english_lemma = _english_surface_text(star) or star.star_id
    aliases = _surface_form_aliases(star)
    tokens = _star_reference_tokens(star)
    spec = _math_execution_spec(tokens) or {
        "operation_key": "compose_from_rules",
        "rpn_program": "",
        "template_ref": "",
        "math_type": "Intermediate Algebra",
        "keywords": sorted(tokens)[:16],
        "semantics": "navigate from language meaning into Math galaxy execution rules",
    }
    bridge_id = _language_math_bridge_id(star)
    query_anchor_parts = [english_lemma]
    query_anchor_parts.extend(spec.get("keywords", []))
    query_anchor_parts.append("math execution rule")
    metadata = {
        "ingest_source": "meaning_layer",
        "bridge_role": "language_to_math_execution",
        "language_star_ref": star.star_id,
        "math_operation_key": spec.get("operation_key", "compose_from_rules"),
        "template_ref": str(spec.get("template_ref", "")).strip(),
        "math_type": str(spec.get("math_type", "Intermediate Algebra")).strip(),
        "query_anchor": " ".join(part for part in query_anchor_parts if str(part).strip()),
        "aliases": aliases,
        "keywords": list(spec.get("keywords", [])),
        "surface_form_languages": sorted(star.surface_forms.keys()),
        "semantics": str(spec.get("semantics", "")).strip(),
        "direct_eval": False,
        "symlink_to": star.star_id,
    }
    return {
        "id": bridge_id,
        "name": f"{english_lemma} execution bridge",
        "domain": "math",
        "category": "rule",
        "content": f"Math execution bridge for the language meaning '{english_lemma}'.",
        "summary": f"{english_lemma} math execution bridge",
        "description": "Language meaning routes into Math galaxy execution via symlink.",
        "rpn_program": str(spec.get("rpn_program", "")).strip(),
        "answer_text": "",
        "symlink_to": star.star_id,
        "metadata": metadata,
    }


def star_matches_keywords(star: MeaningCentricStar, keywords: set[str]) -> bool:
    if not keywords:
        return True
    candidate_tokens = _star_reference_tokens(star)
    return not candidate_tokens.isdisjoint(keywords)


def _star_quality(star: MeaningCentricStar) -> tuple[int, int, int, int, int]:
    return (
        len(star.surface_forms),
        len(star.taxonomy_refs) + len(star.meta_refs) + len(star.grammar_refs) + len(star.reality_refs),
        1 if star.meaning_rpn else 0,
        1 if star.domain else 0,
        1 if star.galaxy_ref else 0,
    )


def dedup_stars(stars: list[MeaningCentricStar]) -> list[MeaningCentricStar]:
    deduped: dict[str, MeaningCentricStar] = {}
    for star in stars:
        current = deduped.get(star.star_id)
        if current is None or _star_quality(star) > _star_quality(current):
            deduped[star.star_id] = star
    return [deduped[key] for key in sorted(deduped.keys())]


def _is_stopword_star(star: MeaningCentricStar) -> bool:
    lemma = _primary_english_lemma(star)
    if not lemma:
        return False
    normalized = lemma.strip().lower()
    if normalized in _STOPWORD_LEMMAS:
        return True
    tokens = _tokenize_text(normalized)
    return bool(tokens) and tokens.issubset(_STOPWORD_LEMMAS)


def _route_meaning_star_to_galaxy(star: MeaningCentricStar) -> str:
    tokens = _star_reference_tokens(star)
    if not tokens:
        return "Language"
    if _matches_domain_tokens(tokens, _GRAMMAR_LEMMAS):
        return "Grammar"
    if _matches_domain_tokens(tokens, _PHYSICS_LEMMAS):
        return "Reality"
    if _matches_domain_tokens(tokens, _VISUAL_LEMMAS):
        return "Drawing"
    if _matches_domain_tokens(tokens, _MATH_LEMMAS):
        return "Math"
    return "Language"


def _existing_foundation_lemmas(knowledgeverse: Knowledgeverse) -> set[str]:
    lemmas: set[str] = set()
    manager = knowledgeverse.galaxy_manager
    ignored_prefixes = ("synset_", "mmlu_", "gsm8k_", "benchmark_", "lhe_", "arc_")
    for path in sorted(manager.storage_root.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entry_id = str(entry.get("id", "")).strip().lower()
                if not entry_id or any(entry_id.startswith(prefix) for prefix in ignored_prefixes):
                    continue
                metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
                name = str(entry.get("name") or metadata.get("meaning_title_en") or "")
                summary = str(entry.get("summary") or metadata.get("summary") or "")
                lemmas.update(_tokenize_text(entry_id))
                lemmas.update(_tokenize_text(name))
                lemmas.update(_tokenize_text(summary))
    return lemmas


def _should_skip_as_duplicate(star: MeaningCentricStar, foundation_lemmas: set[str]) -> bool:
    if not foundation_lemmas:
        return False
    if _route_meaning_star_to_galaxy(star) == "Math":
        return False
    lemma_tokens = _tokenize_text(_primary_english_lemma(star))
    if not lemma_tokens:
        return False
    return lemma_tokens.issubset(foundation_lemmas)


def _entry_list_ref(galaxy: Any) -> list[dict[str, Any]] | None:
    entries = getattr(galaxy, "entries", None)
    if isinstance(entries, list):
        return entries
    extra_entries = getattr(galaxy, "_extra_entries", None)
    if isinstance(extra_entries, list):
        return extra_entries
    return None


def _is_synset_entry(entry: dict[str, Any]) -> bool:
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    entry_id = str(entry.get("id", "")).strip()
    meaning_star_id = str(metadata.get("meaning_star_id", "")).strip()
    ingest_source = str(metadata.get("ingest_source", "")).strip().lower()
    symlink_to = str(entry.get("symlink_to") or metadata.get("symlink_to") or "").strip()
    return (
        ingest_source == "meaning_layer"
        or entry_id.startswith("synset_")
        or meaning_star_id.startswith("synset_")
        or symlink_to.startswith("synset_")
    )


def _purge_existing_synset_entries(knowledgeverse: Knowledgeverse) -> dict[str, int]:
    manager = knowledgeverse.galaxy_manager
    removed: dict[str, int] = {}
    touched = False
    for galaxy_name in ROUTED_MEANING_GALAXIES:
        galaxy = manager.get_galaxy(galaxy_name)
        entries = _entry_list_ref(galaxy)
        if entries is None:
            continue
        before = len(entries)
        entries[:] = [entry for entry in entries if not (isinstance(entry, dict) and _is_synset_entry(entry))]
        removed_count = before - len(entries)
        if removed_count > 0:
            removed[galaxy_name] = removed_count
            manager._dirty_galaxies.add(galaxy_name)
            touched = True
    if touched:
        manager._entry_text_cache.clear()
        manager._specialist_entry_cache.clear()
        if manager._knowledgeverse is not None and hasattr(manager._knowledgeverse, "invalidate_gpu_galaxy_binding"):
            try:
                manager._knowledgeverse.invalidate_gpu_galaxy_binding()
            except Exception:
                pass
    return removed


def _invalidate_galaxy_caches(manager: Any) -> None:
    manager._entry_text_cache.clear()
    manager._specialist_entry_cache.clear()
    if manager._knowledgeverse is not None and hasattr(manager._knowledgeverse, "invalidate_gpu_galaxy_binding"):
        try:
            manager._knowledgeverse.invalidate_gpu_galaxy_binding()
        except Exception:
            pass


def _bulk_upsert_entries(manager: Any, galaxy_name: str, entries: list[dict[str, Any]]) -> dict[str, int]:
    if not entries:
        return {"inserted": 0, "updated": 0}
    galaxy = manager.get_galaxy(galaxy_name)
    entry_list = _entry_list_ref(galaxy)
    if entry_list is None:
        raise RuntimeError(f"Galaxy {galaxy_name} does not expose a mutable entry list")

    id_to_index: dict[str, int] = {}
    for index, current in enumerate(entry_list):
        if not isinstance(current, dict):
            continue
        entry_id = manager._entry_identifier(current)
        if entry_id and entry_id not in id_to_index:
            id_to_index[entry_id] = index

    counts = {"inserted": 0, "updated": 0}
    for entry in entries:
        entry_dict = dict(entry)
        entry_id = manager._entry_identifier(entry_dict)
        if entry_id and entry_id in id_to_index:
            entry_list[id_to_index[entry_id]] = entry_dict
            counts["updated"] += 1
            continue
        if entry_id:
            id_to_index[entry_id] = len(entry_list)
        entry_list.append(entry_dict)
        counts["inserted"] += 1

    manager._dirty_galaxies.add(galaxy_name)
    _invalidate_galaxy_caches(manager)
    return counts


def select_meaning_layer_stars(
    meaning_stars: list[MeaningCentricStar],
    *,
    benchmark_keywords: set[str],
    full_load: bool = False,
    min_languages: int = 5,
    foundation_lemmas: set[str] | None = None,
    filter_stats: dict[str, int] | None = None,
) -> list[MeaningCentricStar]:
    stats: dict[str, int] = {}
    stats["available"] = len(meaning_stars)

    multilingual = [star for star in meaning_stars if len(star.surface_forms) >= max(1, int(min_languages))]
    stats["min_languages"] = int(min_languages)
    stats["after_min_languages"] = len(multilingual)
    stats["removed_for_min_languages"] = stats["available"] - stats["after_min_languages"]

    no_stopwords = [star for star in multilingual if not _is_stopword_star(star)]
    stats["after_stopwords"] = len(no_stopwords)
    stats["stopwords_removed"] = stats["after_min_languages"] - stats["after_stopwords"]

    foundation = set(foundation_lemmas or set())
    no_duplicates = [star for star in no_stopwords if not _should_skip_as_duplicate(star, foundation)]
    stats["after_foundation_dedup"] = len(no_duplicates)
    stats["foundation_duplicates_removed"] = stats["after_stopwords"] - stats["after_foundation_dedup"]

    keyword_filtered = no_duplicates
    if not full_load:
        keyword_filtered = [
            star
            for star in no_duplicates
            if _route_meaning_star_to_galaxy(star) != "Language" or star_matches_keywords(star, benchmark_keywords)
        ]
    stats["after_keyword_filter"] = len(keyword_filtered)
    stats["keyword_filter_removed"] = stats["after_foundation_dedup"] - stats["after_keyword_filter"]

    deduped = dedup_stars(keyword_filtered)
    stats["after_dedup"] = len(deduped)
    stats["dedup_removed"] = stats["after_keyword_filter"] - stats["after_dedup"]

    deduped.sort(key=lambda star: (len(star.surface_forms), _star_quality(star), star.star_id), reverse=True)
    stats["after_selection"] = len(deduped)

    if filter_stats is not None:
        filter_stats.update(stats)
    return deduped


def target_galaxy_for_star(star: MeaningCentricStar, fallback: str = "Language") -> str:
    if str(star.star_id).startswith("synset_"):
        return _route_meaning_star_to_galaxy(star)
    domain = str(star.domain or "").strip().lower()
    for key, galaxy_name in GALAXY_BY_DOMAIN.items():
        if key in domain:
            return galaxy_name
    if any(ref.startswith("concept_mathematics") for ref in star.taxonomy_refs):
        return "Math"
    if any(ref.startswith("concept_language") for ref in star.taxonomy_refs):
        return "Language"
    return fallback


def count_persisted_entries(storage_root: Path) -> int:
    total = 0
    galaxy_root = storage_root / "galaxies"
    if not galaxy_root.exists():
        return 0
    for path in galaxy_root.glob("*.jsonl"):
        with path.open("r", encoding="utf-8") as handle:
            total += sum(1 for line in handle if line.strip())
    return total


def ingest_enriched_galaxy(
    knowledgeverse: Knowledgeverse,
    *,
    meaning_path: Path = DEFAULT_MEANING_PATH,
    mmlu_path: Path = DEFAULT_MMLU_PATH,
    gsm8k_path: Path = DEFAULT_GSM8K_PATH,
    full_load: bool = False,
    benchmark_counts: dict[str, int] | None = None,
    min_languages: int = 5,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    manager = knowledgeverse.galaxy_manager
    emit = progress or (lambda _message: None)
    benchmark_keywords = collect_benchmark_keywords(benchmark_counts)
    foundation_lemmas = _existing_foundation_lemmas(knowledgeverse)

    emit(f"Meaning layer: loading stars from {meaning_path}")
    meaning_all = load_stars_from_jsonl(meaning_path)
    meaning_filter_stats: dict[str, int] = {}
    meaning_selected = select_meaning_layer_stars(
        meaning_all,
        benchmark_keywords=benchmark_keywords,
        full_load=full_load,
        min_languages=min_languages,
        foundation_lemmas=foundation_lemmas,
        filter_stats=meaning_filter_stats,
    )
    emit(
        "Meaning layer: selected "
        f"{len(meaning_selected)} / {len(meaning_all)} stars "
        f"(min_languages={min_languages}, full_load={bool(full_load)})"
    )
    mmlu_stars = dedup_stars(load_stars_from_jsonl(mmlu_path))
    gsm8k_stars = dedup_stars(load_stars_from_jsonl(gsm8k_path))
    emit(f"Proceduralized stars: MMLU={len(mmlu_stars)} GSM8K={len(gsm8k_stars)}")

    counts: dict[str, dict[str, int]] = {}
    removed_synsets: dict[str, int] = {}
    staged_entries: dict[str, list[dict[str, Any]]] = {}

    with manager.bulk_disk_sync():
        removed_synsets = _purge_existing_synset_entries(knowledgeverse)
        emit(
            "Meaning layer: purged existing generated entries "
            + json.dumps(removed_synsets, ensure_ascii=False, sort_keys=True)
        )
        for star in meaning_selected:
            galaxy_name = _route_meaning_star_to_galaxy(star)
            if galaxy_name == "Math":
                bridge_entry = build_language_math_bridge_entry(star)
                language_entry = star.to_galaxy_entry(
                    entry_id=star.star_id,
                    galaxy_name="Language",
                    category="meaning_star",
                    metadata={
                        "ingest_source": "meaning_layer",
                        "math_galaxy_ref": bridge_entry["id"],
                        "math_bridge_role": "meaning_navigation",
                    },
                )
                staged_entries.setdefault("Language", []).append(language_entry)
                staged_entries.setdefault("Math", []).append(bridge_entry)
                continue

            staged_entries.setdefault(galaxy_name, []).append(
                star.to_galaxy_entry(
                    entry_id=star.star_id,
                    galaxy_name=galaxy_name,
                    category="meaning_star",
                    metadata={"ingest_source": "meaning_layer"},
                )
            )

        emit(
            "Meaning layer: staged entries "
            + json.dumps(
                {name: len(values) for name, values in sorted(staged_entries.items())},
                ensure_ascii=False,
                sort_keys=True,
            )
        )

        for galaxy_name, entries in sorted(staged_entries.items()):
            emit(f"Meaning layer: writing {len(entries)} entries to {galaxy_name}")
            galaxy_counts = _bulk_upsert_entries(manager, galaxy_name, entries)
            counts[galaxy_name] = galaxy_counts

        for star in mmlu_stars + gsm8k_stars:
            galaxy_name = target_galaxy_for_star(star, fallback="Math")
            entry = star.to_galaxy_entry(
                entry_id=star.star_id,
                galaxy_name=galaxy_name,
                category="meaning_star",
            )
            status = manager.upsert_entry(galaxy_name, entry)
            bucket = counts.setdefault(galaxy_name, {"inserted": 0, "updated": 0})
            bucket[status] = int(bucket.get(status, 0)) + 1

    emit("Meaning layer: ingest complete")
    return {
        "foundation_bootstrap": dict(getattr(knowledgeverse, "foundational_bootstrap_summary", {}) or {}),
        "benchmark_keyword_count": len(benchmark_keywords),
        "meaning_stars_available": len(meaning_all),
        "meaning_stars_loaded": len(meaning_selected),
        "meaning_filter_stats": meaning_filter_stats,
        "foundation_lemma_count": len(foundation_lemmas),
        "removed_existing_synset_entries": removed_synsets,
        "proceduralized_mmlu_loaded": len(mmlu_stars),
        "proceduralized_gsm8k_loaded": len(gsm8k_stars),
        "galaxy_ingest_status": counts,
        "total_galaxy_entries": count_persisted_entries(knowledgeverse.storage_root),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    parser.add_argument("--meaning-jsonl", type=Path, default=DEFAULT_MEANING_PATH)
    parser.add_argument("--mmlu-jsonl", type=Path, default=DEFAULT_MMLU_PATH)
    parser.add_argument("--gsm8k-jsonl", type=Path, default=DEFAULT_GSM8K_PATH)
    parser.add_argument("--full-load", action="store_true")
    parser.add_argument("--min-languages", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    knowledgeverse = Knowledgeverse(storage_root=args.storage_root)
    summary = ingest_enriched_galaxy(
        knowledgeverse,
        meaning_path=args.meaning_jsonl,
        mmlu_path=args.mmlu_jsonl,
        gsm8k_path=args.gsm8k_jsonl,
        full_load=bool(args.full_load),
        min_languages=int(args.min_languages),
        progress=lambda message: print(message, flush=True),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
