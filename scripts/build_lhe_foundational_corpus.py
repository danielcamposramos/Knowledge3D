#!/usr/bin/env python3
"""Build a deterministic foundational corpus for LHE-style benchmark support.

The generator emits JSONL payload rows compatible with
`scripts/fundamental_ingest_payloads.py`. The corpus is meaning-first:

- Reality Galaxy: canonical concept meaning
- Word Galaxy: linguistic access form, symlinked to Reality
- Math Galaxy: optional formalization surface
- Grammar Galaxy: optional reasoning surface

The builder is deterministic by default. `--with-ollama` is accepted for future
augmentation-time expansion, but this baseline implementation intentionally
stays deterministic and self-contained.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT = Path("../Knowledge3D.local/fundamental_augmentation/lhe_foundational_corpus.jsonl")
BOOTSTRAP_TAG = "lhe_foundational_corpus_v1"

DEFAULT_DOMAIN_ALLOCATIONS: dict[str, int] = {
    "mathematics": 448,
    "physics": 288,
    "cs_ai": 288,
    "biology_medicine": 256,
    "chemistry": 192,
    "humanities_social_science": 256,
    "engineering": 160,
    "other": 160,
}

_FORBIDDEN_SMOKE_IDS: tuple[str, ...] = (
    "6687ffb1091058ff19128813",
    "668825f80a642802bdfeadfa",
    "668828540a642802bdfeadfc",
    "669402b41dcb3d5a1ef9e951",
    "6696c3734c196f1af6a16fcb",
    "66b2c7c9795022550abb176b",
    "66b727d367968fa27f2dddda",
    "66b827b9b64deaedfbb997a2",
    "66b91693d86bff9a12fc1f99",
    "66ba5510db07a3a9151be0d2",
)

_FORBIDDEN_SMOKE_STRINGS: tuple[str, ...] = (
    "Weak Non-Sadism",
    "yeyo",
    "Z+Z+Z+Z+Z",
    "Katie kicked the knotted kite string, knowing it would take skill to unknot the tangled mess.",
    "\\mathcal{A}^{\\alpha-}(X)",
    "Rxf3, Rf1#",
)


@dataclass(frozen=True)
class SubfieldBlueprint:
    key: str
    label: str
    focus: str
    core_terms: tuple[str, ...]
    modifiers: tuple[str, ...]
    qualifiers: tuple[str, ...]
    reason_modes: tuple[str, ...]


@dataclass(frozen=True)
class DomainBlueprint:
    key: str
    label: str
    subject: str
    target: int
    subfields: tuple[SubfieldBlueprint, ...]
    formal_stride: int
    grammar_stride: int


def _slug(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _title_like(text: str) -> str:
    if text.isupper():
        return text
    replacements = {
        "qft": "QFT",
        "gelu": "GELU",
        "crispr": "CRISPR",
        "pde": "PDE",
        "pdes": "PDEs",
        "emd": "EMD",
    }
    words = []
    for token in text.split():
        lowered = token.lower()
        if lowered in replacements:
            words.append(replacements[lowered])
        elif token.startswith("\\"):
            words.append(token)
        else:
            words.append(token.capitalize())
    return " ".join(words)


def _dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def _scale_allocations(
    *,
    target_concepts: int,
    domains: list[str],
) -> dict[str, int]:
    base_total = sum(DEFAULT_DOMAIN_ALLOCATIONS[name] for name in domains)
    if base_total <= 0:
        return {}
    raw = {
        name: (target_concepts * DEFAULT_DOMAIN_ALLOCATIONS[name]) / base_total
        for name in domains
    }
    floor_counts = {name: int(value) for name, value in raw.items()}
    remainder = target_concepts - sum(floor_counts.values())
    ranked = sorted(
        domains,
        key=lambda name: (raw[name] - floor_counts[name], DEFAULT_DOMAIN_ALLOCATIONS[name]),
        reverse=True,
    )
    for name in ranked[:remainder]:
        floor_counts[name] += 1
    return floor_counts


def _distribute(total: int, buckets: int) -> list[int]:
    base = total // buckets
    remainder = total % buckets
    return [base + (1 if idx < remainder else 0) for idx in range(buckets)]


def _domain_alias(domain_key: str) -> str:
    aliases = {
        "mathematics": "math",
        "physics": "physics",
        "cs_ai": "cs",
        "biology_medicine": "bio",
        "chemistry": "chem",
        "humanities_social_science": "humanities",
        "engineering": "eng",
        "other": "other",
    }
    return aliases.get(domain_key, domain_key)


def _reserve_unique_id(candidate: str, *, suffix: str, used: set[str]) -> str:
    if candidate not in used:
        used.add(candidate)
        return candidate
    alternate = f"{candidate}_{suffix}"
    if alternate not in used:
        used.add(alternate)
        return alternate
    counter = 2
    while True:
        numbered = f"{alternate}_{counter}"
        if numbered not in used:
            used.add(numbered)
            return numbered
        counter += 1


def _build_aliases(label: str, subfield: str, subject: str) -> list[str]:
    lowered = label.lower()
    aliases = [
        lowered,
        lowered.replace("-", " "),
        lowered.replace(" and ", " & "),
        f"{lowered} in {subfield.replace('_', ' ')}",
        f"{lowered} in {subject}",
    ]
    if "," not in lowered:
        aliases.append(lowered.replace(" ", "_"))
    return _dedupe_keep_order(aliases)


def _is_formalizable(domain: DomainBlueprint, index: int, label: str) -> bool:
    if any(token in label.lower() for token in ("notation", "cipher", "mate", "opening")):
        return True
    return index % max(1, domain.formal_stride) == 0


def _is_reasonable_for_grammar(domain: DomainBlueprint, index: int, label: str) -> bool:
    if any(
        token in label.lower()
        for token in (
            "ethic",
            "principle",
            "theorem",
            "logic",
            "cipher",
            "notation",
            "figure",
            "irony",
            "paradox",
            "sarcasm",
            "pun",
            "metaphor",
            "allusion",
        )
    ):
        return True
    return index % max(1, domain.grammar_stride) == 0


def _relationship_triplets(
    *,
    concept_id: str,
    word_id: str,
    math_id: str | None,
    grammar_id: str | None,
) -> list[dict[str, str]]:
    out = [
        {"from": concept_id, "relation": "lexicalized_as", "to": word_id},
        {"from": word_id, "relation": "names", "to": concept_id},
    ]
    if math_id:
        out.append({"from": math_id, "relation": "formalizes", "to": concept_id})
    if grammar_id:
        out.append({"from": grammar_id, "relation": "reasons_about", "to": concept_id})
    return out


def _formal_rpn(label: str, subfield: str) -> str:
    label_slug = _slug(label)
    return f"LOOKUP {label_slug} CONTEXT {subfield} FORMALIZE"


def _grammar_rpn(mode: str, label: str, subfield: str) -> str:
    return f"{mode.upper()} LOOKUP {_slug(label)} CONTEXT {subfield} CONTRASTIVE_VERIFY"


def _make_reality_entry(
    *,
    domain: DomainBlueprint,
    subfield: SubfieldBlueprint,
    concept_id: str,
    word_id: str,
    math_id: str | None,
    grammar_id: str | None,
    label: str,
    aliases: list[str],
    related_concepts: list[str],
) -> dict[str, Any]:
    display_name = _title_like(label)
    related_text = ", ".join(related_concepts[:4])
    definition = (
        f"{display_name} is a foundational concept in {subfield.label.replace('_', ' ')} "
        f"within {domain.subject}, used to reason about {subfield.focus}."
    )
    content = (
        f"{display_name} anchors meaning for {subfield.label.replace('_', ' ')} questions. "
        f"It is commonly linked to {related_text}. "
        f"The concept is represented canonically in Reality and accessed lexically through {word_id}."
    )
    summary = f"{display_name} is a core {domain.subject} concept for {subfield.focus}."
    keywords = _dedupe_keep_order(
        [display_name, subfield.label.replace("_", " "), domain.subject, *aliases[:4], *related_concepts[:3]]
    )
    symlinks = [word_id]
    if math_id:
        symlinks.append(math_id)
    if grammar_id:
        symlinks.append(grammar_id)
    return {
        "id": concept_id,
        "name": display_name,
        "domain": domain.subject,
        "category": subfield.key,
        "content": content,
        "description": definition,
        "summary": summary,
        "rpn_program": f"LOOKUP {word_id}",
        "metadata": {
            "source": "lhe_foundational_corpus",
            "subject": domain.subject,
            "subfield": subfield.key,
            "definition": definition,
            "related_concepts": related_concepts,
            "symlinks": symlinks,
            "confidence": 0.92,
            "bootstrap": BOOTSTRAP_TAG,
            "aliases": aliases,
            "keywords": keywords,
            "notes": f"Meaning-first canonical concept for {subfield.focus}.",
            "embedding_text": f"{display_name}. {definition} Related concepts: {related_text}.",
            "semantics": f"{display_name} denotes a concept for {subfield.focus}.",
            "usage_conditions": (
                f"Use when a question requires {subfield.focus} knowledge in {domain.subject}."
            ),
            "entities": [
                {"name": display_name, "content": definition},
                {"name": subfield.label.replace("_", " "), "content": subfield.focus},
            ],
            "relationships": _relationship_triplets(
                concept_id=concept_id,
                word_id=word_id,
                math_id=math_id,
                grammar_id=grammar_id,
            ),
            "domain": domain.subject,
            "category": subfield.key,
        },
    }


def _make_word_entry(
    *,
    domain: DomainBlueprint,
    subfield: SubfieldBlueprint,
    concept_id: str,
    word_id: str,
    label: str,
    aliases: list[str],
) -> dict[str, Any]:
    display_name = label.lower()
    description = (
        f"{display_name} is the linguistic surface form that resolves to {concept_id} "
        f"in {domain.subject}."
    )
    return {
        "id": word_id,
        "name": display_name,
        "domain": "word",
        "category": f"{domain.key}_lexeme",
        "content": description,
        "description": description,
        "summary": f"Lexical form for {label}.",
        "rpn_program": f"LOOKUP {concept_id}",
        "metadata": {
            "meaning_ref": concept_id,
            "aliases": _dedupe_keep_order([display_name, *aliases]),
            "subject": domain.subject,
            "subfield": subfield.key,
            "bootstrap": BOOTSTRAP_TAG,
            "source": "lhe_foundational_corpus",
            "keywords": [display_name, subfield.label.replace("_", " "), domain.subject],
            "embedding_text": f"{display_name}. lexical form for {label} in {domain.subject}.",
            "notes": f"Word-to-meaning symlink for {label}.",
            "semantics": f"Lexical access form for {label}.",
            "usage_conditions": f"Resolve text mentions of {display_name} to {concept_id}.",
            "domain": "word",
            "category": f"{domain.key}_lexeme",
        },
    }


def _make_math_entry(
    *,
    domain: DomainBlueprint,
    subfield: SubfieldBlueprint,
    concept_id: str,
    math_id: str,
    label: str,
    related_concepts: list[str],
) -> dict[str, Any]:
    display_name = f"{_title_like(label)} formalization"
    formal_statement = (
        f"{label} formal relation in {subfield.label.replace('_', ' ')} "
        f"over symbols tied to {', '.join(related_concepts[:2])}."
    )
    return {
        "id": math_id,
        "name": display_name,
        "domain": "math",
        "category": f"{domain.key}_{subfield.key}_formalization",
        "content": formal_statement,
        "description": formal_statement,
        "summary": f"Formalization layer for {label}.",
        "rpn_program": _formal_rpn(label=label, subfield=subfield.key),
        "metadata": {
            "formalizes_ref": concept_id,
            "subject": domain.subject,
            "subfield": subfield.key,
            "bootstrap": BOOTSTRAP_TAG,
            "source": "lhe_foundational_corpus",
            "aliases": [f"{label} formalization", f"{label} rule"],
            "keywords": [label, subfield.label.replace("_", " "), "formalization"],
            "embedding_text": f"{label}. formalization in {subfield.label.replace('_', ' ')}.",
            "notes": "Formal symbolic surface for structured reasoning.",
            "semantics": f"Formalizes {label} for reusable symbolic reasoning.",
            "usage_conditions": f"Use when the task requires formal structure for {label}.",
            "domain": "math",
            "category": f"{domain.key}_{subfield.key}_formalization",
            "entities": [{"name": label, "content": formal_statement}],
            "relationships": [{"from": math_id, "relation": "formalizes", "to": concept_id}],
        },
    }


def _make_grammar_entry(
    *,
    domain: DomainBlueprint,
    subfield: SubfieldBlueprint,
    concept_id: str,
    grammar_id: str,
    label: str,
    mode: str,
) -> dict[str, Any]:
    display_name = f"{_title_like(label)} reasoning"
    semantics = (
        f"{mode.replace('_', ' ')} reasoning rule for {label} in "
        f"{subfield.label.replace('_', ' ')}."
    )
    return {
        "id": grammar_id,
        "rule_id": grammar_id,
        "name": display_name,
        "domain": "grammar",
        "category": f"{domain.key}_{subfield.key}_reasoning",
        "pattern_type": "lhe_foundational_reasoning",
        "content": semantics,
        "description": semantics,
        "summary": f"Reasoning rule for {label}.",
        "rpn_program": _grammar_rpn(mode=mode, label=label, subfield=subfield.key),
        "metadata": {
            "reasons_about_ref": concept_id,
            "subject": domain.subject,
            "subfield": subfield.key,
            "bootstrap": BOOTSTRAP_TAG,
            "source": "lhe_foundational_corpus",
            "semantics": semantics,
            "usage_conditions": (
                f"Apply when a task requires {mode.replace('_', ' ')} over {label}."
            ),
            "aliases": [f"{label} reasoning", f"{label} interpretation"],
            "keywords": [label, mode.replace("_", " "), "reasoning"],
            "embedding_text": f"{label}. reasoning rule for {mode.replace('_', ' ')}.",
            "notes": "Grammar-layer reasoning scaffold for domain interpretation.",
            "domain": "grammar",
            "category": f"{domain.key}_{subfield.key}_reasoning",
            "entities": [{"name": label, "content": semantics}],
            "relationships": [{"from": grammar_id, "relation": "reasons_about", "to": concept_id}],
        },
    }


def _generate_labels(subfield: SubfieldBlueprint, target: int) -> list[tuple[str, str, str]]:
    generated: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    variant_specs = (
        lambda base, modifier, qualifier: base,
        lambda base, modifier, qualifier: f"{modifier} {base}",
        lambda base, modifier, qualifier: f"{base} {qualifier}",
        lambda base, modifier, qualifier: f"{modifier} {base} {qualifier}",
    )
    for builder in variant_specs:
        for base in subfield.core_terms:
            for modifier in ("", *subfield.modifiers):
                for qualifier in ("", *subfield.qualifiers):
                    if builder is variant_specs[0] and (modifier or qualifier):
                        continue
                    if builder is variant_specs[1] and (not modifier or qualifier):
                        continue
                    if builder is variant_specs[2] and (modifier or not qualifier):
                        continue
                    if builder is variant_specs[3] and (not modifier or not qualifier):
                        continue
                    label = builder(base, modifier, qualifier).strip()
                    key = label.lower()
                    if not label or key in seen:
                        continue
                    seen.add(key)
                    generated.append((label, base, qualifier or modifier or base))
                    if len(generated) >= target:
                        return generated
    return generated[:target]


def _build_rows_for_domain(
    domain: DomainBlueprint,
    target: int,
    *,
    used_ids: dict[str, set[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_subfield = _distribute(target, len(domain.subfields))
    for subfield, subfield_target in zip(domain.subfields, per_subfield, strict=True):
        labels = _generate_labels(subfield, subfield_target)
        for index, (label, base_term, relation_seed) in enumerate(labels):
            alias_domain = _domain_alias(domain.key)
            label_slug = _slug(label)
            concept_id = _reserve_unique_id(
                f"concept_{alias_domain}_{label_slug}",
                suffix=subfield.key,
                used=used_ids.setdefault("Reality", set()),
            )
            word_id = _reserve_unique_id(
                f"word_{label_slug}",
                suffix=subfield.key,
                used=used_ids.setdefault("Word", set()),
            )
            math_id = None
            grammar_id = None
            if _is_formalizable(domain, index, label):
                if domain.key == "physics" and label.lower() == "gamma matrices":
                    math_id = _reserve_unique_id(
                        "math_gamma_matrix_clifford_relation",
                        suffix=subfield.key,
                        used=used_ids.setdefault("Math", set()),
                    )
                else:
                    math_id = _reserve_unique_id(
                        f"math_{alias_domain}_{label_slug}",
                        suffix=subfield.key,
                        used=used_ids.setdefault("Math", set()),
                    )
            if _is_reasonable_for_grammar(domain, index, label):
                grammar_id = _reserve_unique_id(
                    f"grammar_{alias_domain}_{label_slug}_reasoning",
                    suffix=subfield.key,
                    used=used_ids.setdefault("Grammar", set()),
                )

            aliases = _build_aliases(label, subfield.key, domain.subject)
            related_concepts = _dedupe_keep_order(
                [base_term, relation_seed, *subfield.core_terms[:3], *subfield.qualifiers[:2]]
            )
            reality_entry = _make_reality_entry(
                domain=domain,
                subfield=subfield,
                concept_id=concept_id,
                word_id=word_id,
                math_id=math_id,
                grammar_id=grammar_id,
                label=label,
                aliases=aliases,
                related_concepts=related_concepts,
            )
            word_entry = _make_word_entry(
                domain=domain,
                subfield=subfield,
                concept_id=concept_id,
                word_id=word_id,
                label=label,
                aliases=aliases,
            )
            rows.append({"galaxy": "Reality", "entry": reality_entry})
            rows.append({"galaxy": "Word", "entry": word_entry})
            if math_id is not None:
                rows.append(
                    {
                        "galaxy": "Math",
                        "entry": _make_math_entry(
                            domain=domain,
                            subfield=subfield,
                            concept_id=concept_id,
                            math_id=math_id,
                            label=label,
                            related_concepts=related_concepts,
                        ),
                    }
                )
            if grammar_id is not None:
                mode = subfield.reason_modes[index % len(subfield.reason_modes)]
                rows.append(
                    {
                        "galaxy": "Grammar",
                        "entry": _make_grammar_entry(
                            domain=domain,
                            subfield=subfield,
                            concept_id=concept_id,
                            grammar_id=grammar_id,
                            label=label,
                            mode=mode,
                        ),
                    }
                )
    return rows


def _build_domain_blueprints() -> dict[str, DomainBlueprint]:
    return {
        "mathematics": DomainBlueprint(
            key="mathematics",
            label="Mathematics",
            subject="mathematics",
            target=448,
            formal_stride=2,
            grammar_stride=5,
            subfields=(
                SubfieldBlueprint(
                    key="algebraic_topology",
                    label="algebraic_topology",
                    focus="topological invariants, bordism, and classifying constructions",
                    core_terms=(
                        "homology group",
                        "cohomology ring",
                        "bordism class",
                        "classifying space",
                        "fundamental class",
                        "spectral sequence",
                        "homotopy fiber",
                        "characteristic class",
                    ),
                    modifiers=("relative", "equivariant", "stable", "generalized", "compactly_supported"),
                    qualifiers=("criterion", "decomposition", "computation", "duality", "obstruction"),
                    reason_modes=("invariant_tracking", "dimension_check", "topological_reduction"),
                ),
                SubfieldBlueprint(
                    key="lie_theory",
                    label="lie_theory",
                    focus="Lie groups, Lie algebras, symmetries, and representation structure",
                    core_terms=(
                        "Lie algebra",
                        "Lie group",
                        "root system",
                        "weight lattice",
                        "Cartan subalgebra",
                        "highest weight module",
                        "Weyl chamber",
                        "representation ring",
                    ),
                    modifiers=("semisimple", "nilpotent", "compact", "reductive", "graded"),
                    qualifiers=("classification", "decomposition", "invariant", "criterion", "correspondence"),
                    reason_modes=("symmetry_reasoning", "representation_decomposition", "structure_comparison"),
                ),
                SubfieldBlueprint(
                    key="moduli_and_elliptic",
                    label="moduli_and_elliptic",
                    focus="moduli, elliptic curves, families of varieties, and deformation data",
                    core_terms=(
                        "moduli space",
                        "elliptic curve",
                        "torsion subgroup",
                        "sheaf cohomology",
                        "stack quotient",
                        "stable bundle",
                        "deformation functor",
                        "abelian variety",
                    ),
                    modifiers=("derived", "stable", "compactified", "rational", "arithmetic"),
                    qualifiers=("parameterization", "degeneration", "classification", "invariant", "family"),
                    reason_modes=("family_tracking", "parameter_reasoning", "deformation_control"),
                ),
                SubfieldBlueprint(
                    key="functional_analysis",
                    label="functional_analysis",
                    focus="operators, Sobolev spaces, PDE solutions, and distributional limits",
                    core_terms=(
                        "Sobolev space",
                        "Banach space",
                        "Hilbert space",
                        "distribution",
                        "bounded operator",
                        "weak derivative",
                        "elliptic PDE",
                        "semigroup solution",
                    ),
                    modifiers=("weighted", "anisotropic", "compact", "self_adjoint", "nonlinear"),
                    qualifiers=("estimate", "regularity", "criterion", "extension", "approximation"),
                    reason_modes=("operator_reasoning", "regularity_check", "norm_estimation"),
                ),
                SubfieldBlueprint(
                    key="number_theory",
                    label="number_theory",
                    focus="arithmetic structure, Galois actions, p-adics, and L-functions",
                    core_terms=(
                        "Galois group",
                        "p-adic field",
                        "L-function",
                        "modular form",
                        "Hecke algebra",
                        "Selmer group",
                        "ideal class group",
                        "local field",
                    ),
                    modifiers=("abelian", "ramified", "automorphic", "local", "global"),
                    qualifiers=("criterion", "correspondence", "bound", "lifting", "decomposition"),
                    reason_modes=("arithmetic_transfer", "local_global_reasoning", "galois_tracking"),
                ),
                SubfieldBlueprint(
                    key="combinatorics",
                    label="combinatorics",
                    focus="counting, generating functions, partition theory, and discrete structure",
                    core_terms=(
                        "generating function",
                        "partition identity",
                        "matroid rank",
                        "poset interval",
                        "graph minor",
                        "symmetric function",
                        "Young tableau",
                        "incidence algebra",
                    ),
                    modifiers=("weighted", "enumerative", "labeled", "unlabeled", "asymptotic"),
                    qualifiers=("recurrence", "bijection", "criterion", "expansion", "enumeration"),
                    reason_modes=("counting_argument", "bijection_reasoning", "recurrence_build"),
                ),
                SubfieldBlueprint(
                    key="differential_geometry",
                    label="differential_geometry",
                    focus="manifolds, curvature, bundles, and geometric flow structure",
                    core_terms=(
                        "Riemannian metric",
                        "connection form",
                        "geodesic flow",
                        "curvature tensor",
                        "principal bundle",
                        "symplectic form",
                        "contact structure",
                        "minimal surface",
                    ),
                    modifiers=("compact", "Kahler", "hyperbolic", "conformal", "integrable"),
                    qualifiers=("classification", "stability", "criterion", "normal_form", "deformation"),
                    reason_modes=("geometric_constraint", "curvature_reasoning", "bundle_transport"),
                ),
                SubfieldBlueprint(
                    key="algebra_and_logic",
                    label="algebra_and_logic",
                    focus="formal algebra, proof systems, and structural logical semantics",
                    core_terms=(
                        "model category",
                        "proof system",
                        "predicate calculus",
                        "set theoretic hierarchy",
                        "Boolean algebra",
                        "topos object",
                        "Grothendieck universe",
                        "compactness theorem",
                    ),
                    modifiers=("intuitionistic", "classical", "categorical", "effective", "higher_order"),
                    qualifiers=("semantics", "proof", "criterion", "translation", "consistency"),
                    reason_modes=("proof_composition", "semantic_verification", "axiom_tracking"),
                ),
            ),
        ),
        "physics": DomainBlueprint(
            key="physics",
            label="Physics",
            subject="physics",
            target=288,
            formal_stride=2,
            grammar_stride=4,
            subfields=(
                SubfieldBlueprint(
                    key="relativistic_quantum",
                    label="relativistic_quantum",
                    focus="spinors, gamma matrices, and relativistic wave structure",
                    core_terms=(
                        "gamma matrices",
                        "Clifford algebra",
                        "Dirac spinor",
                        "chirality projector",
                        "Lorentz generator",
                        "spin representation",
                        "propagator pole",
                        "mass shell",
                    ),
                    modifiers=("massive", "massless", "covariant", "chiral", "Euclidean"),
                    qualifiers=("relation", "identity", "decomposition", "constraint", "representation"),
                    reason_modes=("symmetry_reasoning", "covariance_check", "operator_linking"),
                ),
                SubfieldBlueprint(
                    key="compactification_string",
                    label="compactification_string",
                    focus="compactification, extra dimensions, and Kaluza-Klein spectra",
                    core_terms=(
                        "compactification",
                        "Kaluza-Klein mode",
                        "Calabi-Yau manifold",
                        "orbifold sector",
                        "moduli stabilization",
                        "brane tension",
                        "flux compactification",
                        "string vacuum",
                    ),
                    modifiers=("warped", "supersymmetric", "effective", "heterotic", "dual"),
                    qualifiers=("spectrum", "constraint", "vacuum", "stability", "reduction"),
                    reason_modes=("dimensional_reduction", "spectrum_reasoning", "vacuum_selection"),
                ),
                SubfieldBlueprint(
                    key="qft_gauge",
                    label="qft_gauge",
                    focus="quantum fields, gauge symmetries, and path-integral structure",
                    core_terms=(
                        "partition function",
                        "gauge field",
                        "renormalization group",
                        "path integral",
                        "Wilson loop",
                        "anomaly coefficient",
                        "Green function",
                        "vacuum expectation value",
                    ),
                    modifiers=("nonabelian", "thermal", "effective", "topological", "perturbative"),
                    qualifiers=("flow", "identity", "constraint", "expansion", "matching"),
                    reason_modes=("field_theory_composition", "symmetry_breaking", "renormalization_tracking"),
                ),
                SubfieldBlueprint(
                    key="statistical_mechanics",
                    label="statistical_mechanics",
                    focus="ensembles, fluctuations, critical phenomena, and entropy structure",
                    core_terms=(
                        "free energy",
                        "critical exponent",
                        "partition ensemble",
                        "order parameter",
                        "correlation length",
                        "phase transition",
                        "fluctuation theorem",
                        "transfer matrix",
                    ),
                    modifiers=("canonical", "microcanonical", "nonequilibrium", "continuous", "discrete"),
                    qualifiers=("scaling", "criterion", "behavior", "estimate", "duality"),
                    reason_modes=("ensemble_reasoning", "critical_scaling", "entropy_balance"),
                ),
                SubfieldBlueprint(
                    key="condensed_matter",
                    label="condensed_matter",
                    focus="collective excitations, band structure, and material order",
                    core_terms=(
                        "band gap",
                        "Fermi surface",
                        "quasiparticle mode",
                        "topological insulator",
                        "order parameter",
                        "spin chain",
                        "lattice defect",
                        "Berry phase",
                    ),
                    modifiers=("strongly_correlated", "topological", "superconducting", "ferromagnetic", "two_dimensional"),
                    qualifiers=("phase", "transition", "response", "transport", "invariant"),
                    reason_modes=("material_response", "band_reasoning", "phase_diagnosis"),
                ),
                SubfieldBlueprint(
                    key="classical_fields",
                    label="classical_fields",
                    focus="electromagnetic and continuum field descriptions",
                    core_terms=(
                        "differential form electromagnetism",
                        "field tensor",
                        "Maxwell equation",
                        "stress energy tensor",
                        "fluid vorticity",
                        "wave packet",
                        "boundary condition",
                        "Green kernel",
                    ),
                    modifiers=("source_free", "retarded", "covariant", "time_dependent", "boundary_driven"),
                    qualifiers=("solution", "constraint", "propagation", "conservation", "matching"),
                    reason_modes=("field_balance", "boundary_reasoning", "propagation_tracking"),
                ),
            ),
        ),
        "cs_ai": DomainBlueprint(
            key="cs_ai",
            label="CS/AI",
            subject="computer_science",
            target=288,
            formal_stride=3,
            grammar_stride=4,
            subfields=(
                SubfieldBlueprint(
                    key="deep_learning",
                    label="deep_learning",
                    focus="transformer internals, activation behavior, and representation flow",
                    core_terms=(
                        "transformer architecture",
                        "attention head",
                        "residual stream",
                        "layer norm",
                        "activation function",
                        "GELU nonlinearity",
                        "embedding matrix",
                        "positional encoding",
                    ),
                    modifiers=("causal", "bidirectional", "sparse", "shared", "normalized"),
                    qualifiers=("analysis", "interaction", "ablation", "behavior", "stability"),
                    reason_modes=("representation_tracing", "mechanistic_interpretation", "activation_selection"),
                ),
                SubfieldBlueprint(
                    key="complexity_theory",
                    label="complexity_theory",
                    focus="complexity classes, hardness, and reducibility",
                    core_terms=(
                        "P versus NP",
                        "NP completeness",
                        "polynomial reduction",
                        "space hierarchy",
                        "interactive proof",
                        "approximation scheme",
                        "communication complexity",
                        "randomized class",
                    ),
                    modifiers=("deterministic", "nondeterministic", "probabilistic", "parameterized", "oracle"),
                    qualifiers=("bound", "collapse", "criterion", "reduction", "separation"),
                    reason_modes=("reduction_reasoning", "class_separation", "resource_accounting"),
                ),
                SubfieldBlueprint(
                    key="cryptography",
                    label="cryptography",
                    focus="classical ciphers, modern cryptography, and adversarial security reasoning",
                    core_terms=(
                        "substitution cipher",
                        "Vigenere cipher",
                        "frequency analysis",
                        "one time pad",
                        "zero knowledge proof",
                        "public key cryptography",
                        "hash preimage",
                        "elliptic curve cryptography",
                    ),
                    modifiers=("classical", "symmetric", "asymmetric", "adaptive", "chosen_ciphertext"),
                    qualifiers=("attack", "security", "proof", "recovery", "distinguishability"),
                    reason_modes=("contrastive_elimination", "attack_surface_reasoning", "evidence_decryption"),
                ),
                SubfieldBlueprint(
                    key="algorithms",
                    label="algorithms",
                    focus="algorithmic structure, optimization, and discrete problem solving",
                    core_terms=(
                        "dynamic programming",
                        "greedy algorithm",
                        "graph traversal",
                        "minimum cut",
                        "shortest path",
                        "priority queue",
                        "suffix automaton",
                        "convex hull",
                    ),
                    modifiers=("amortized", "parallel", "streaming", "approximate", "randomized"),
                    qualifiers=("analysis", "invariant", "optimization", "proof", "decomposition"),
                    reason_modes=("algorithm_selection", "invariant_tracking", "optimization_reasoning"),
                ),
                SubfieldBlueprint(
                    key="formal_languages",
                    label="formal_languages",
                    focus="grammars, automata, parsing, and symbolic computation",
                    core_terms=(
                        "context free grammar",
                        "pushdown automaton",
                        "finite automaton",
                        "regular expression",
                        "parser combinator",
                        "type system",
                        "lambda calculus",
                        "proof assistant",
                    ),
                    modifiers=("deterministic", "probabilistic", "typed", "higher_order", "minimal"),
                    qualifiers=("equivalence", "derivation", "normalization", "recognition", "inference"),
                    reason_modes=("grammar_composition", "parse_verification", "symbolic_normalization"),
                ),
                SubfieldBlueprint(
                    key="systems_ai",
                    label="systems_ai",
                    focus="distributed systems, agents, and learning system interfaces",
                    core_terms=(
                        "consensus protocol",
                        "distributed training",
                        "vector database",
                        "retrieval pipeline",
                        "memory hierarchy",
                        "agent coordination",
                        "control loop",
                        "inference server",
                    ),
                    modifiers=("fault_tolerant", "stateful", "stateless", "streaming", "hierarchical"),
                    qualifiers=("latency", "consistency", "recovery", "scheduling", "orchestration"),
                    reason_modes=("system_coordination", "latency_budgeting", "reliability_reasoning"),
                ),
            ),
        ),
        "biology_medicine": DomainBlueprint(
            key="biology_medicine",
            label="Biology/Medicine",
            subject="biology_medicine",
            target=256,
            formal_stride=4,
            grammar_stride=4,
            subfields=(
                SubfieldBlueprint(
                    key="molecular_biology",
                    label="molecular_biology",
                    focus="transcription, translation, and molecular regulation",
                    core_terms=(
                        "transcription factor",
                        "messenger RNA",
                        "ribosome assembly",
                        "protein folding",
                        "enzyme kinetics",
                        "allosteric site",
                        "signal peptide",
                        "post translational modification",
                    ),
                    modifiers=("regulatory", "mitochondrial", "cytosolic", "cooperative", "allosteric"),
                    qualifiers=("pathway", "control", "binding", "turnover", "response"),
                    reason_modes=("causal_pathway_reasoning", "molecular_binding", "regulation_tracking"),
                ),
                SubfieldBlueprint(
                    key="genetics_and_crispr",
                    label="genetics_and_crispr",
                    focus="genetic variation, CRISPR editing, and inheritance",
                    core_terms=(
                        "CRISPR guide RNA",
                        "Cas nuclease",
                        "gene regulation",
                        "epigenetic marker",
                        "allele frequency",
                        "genome editing",
                        "repair template",
                        "dominant trait",
                    ),
                    modifiers=("somatic", "germline", "targeted", "repair_mediated", "population"),
                    qualifiers=("editing", "inheritance", "selection", "specificity", "screen"),
                    reason_modes=("genetic_intervention", "inheritance_reasoning", "variant_mapping"),
                ),
                SubfieldBlueprint(
                    key="neuroscience",
                    label="neuroscience",
                    focus="neural signaling, circuits, and plasticity",
                    core_terms=(
                        "synaptic plasticity",
                        "action potential",
                        "neurotransmitter release",
                        "cortical column",
                        "hippocampal replay",
                        "dopamine pathway",
                        "axon guidance",
                        "receptor potential",
                    ),
                    modifiers=("excitatory", "inhibitory", "cortical", "hippocampal", "dopaminergic"),
                    qualifiers=("dynamics", "modulation", "coding", "adaptation", "circuit"),
                    reason_modes=("circuit_reasoning", "signal_modulation", "plasticity_tracking"),
                ),
                SubfieldBlueprint(
                    key="immunology",
                    label="immunology",
                    focus="innate and adaptive immune response",
                    core_terms=(
                        "major histocompatibility complex",
                        "antigen presentation",
                        "B cell maturation",
                        "T cell activation",
                        "cytokine signaling",
                        "immune tolerance",
                        "antibody affinity",
                        "memory response",
                    ),
                    modifiers=("adaptive", "innate", "mucosal", "systemic", "autoimmune"),
                    qualifiers=("cascade", "selection", "response", "recognition", "escape"),
                    reason_modes=("immune_matching", "response_selection", "recognition_reasoning"),
                ),
                SubfieldBlueprint(
                    key="pharmacology",
                    label="pharmacology",
                    focus="dose response, receptor binding, and therapeutic action",
                    core_terms=(
                        "dose response curve",
                        "receptor binding",
                        "agonist efficacy",
                        "drug metabolism",
                        "therapeutic index",
                        "half life",
                        "bioavailability",
                        "toxicity profile",
                    ),
                    modifiers=("competitive", "noncompetitive", "oral", "intravenous", "sustained_release"),
                    qualifiers=("model", "interaction", "clearance", "exposure", "safety"),
                    reason_modes=("dose_reasoning", "binding_selection", "safety_constraint"),
                ),
            ),
        ),
        "chemistry": DomainBlueprint(
            key="chemistry",
            label="Chemistry",
            subject="chemistry",
            target=192,
            formal_stride=3,
            grammar_stride=4,
            subfields=(
                SubfieldBlueprint(
                    key="organic_chemistry",
                    label="organic_chemistry",
                    focus="functional groups, stereochemistry, and reaction mechanisms",
                    core_terms=(
                        "functional group",
                        "SN2 mechanism",
                        "electrophilic addition",
                        "stereocenter inversion",
                        "aromatic substitution",
                        "carbonyl condensation",
                        "retrosynthetic step",
                        "protecting group",
                    ),
                    modifiers=("enantioselective", "concerted", "stepwise", "activated", "aromatic"),
                    qualifiers=("pathway", "selectivity", "transition_state", "intermediate", "control"),
                    reason_modes=("mechanism_selection", "stereochemical_reasoning", "retrosynthetic_linking"),
                ),
                SubfieldBlueprint(
                    key="inorganic_chemistry",
                    label="inorganic_chemistry",
                    focus="coordination chemistry, ligands, and inorganic structure",
                    core_terms=(
                        "coordination complex",
                        "ligand field splitting",
                        "octahedral complex",
                        "oxidation state",
                        "crystal field theory",
                        "spin state",
                        "bridging ligand",
                        "metal orbital",
                    ),
                    modifiers=("high_spin", "low_spin", "octahedral", "tetrahedral", "mixed_valence"),
                    qualifiers=("stability", "splitting", "assignment", "interaction", "geometry"),
                    reason_modes=("coordination_reasoning", "oxidation_assignment", "geometry_selection"),
                ),
                SubfieldBlueprint(
                    key="physical_chemistry",
                    label="physical_chemistry",
                    focus="thermodynamics, kinetics, and spectroscopy",
                    core_terms=(
                        "reaction coordinate",
                        "activation energy",
                        "partition function",
                        "rate constant",
                        "enthalpy change",
                        "entropy production",
                        "spectroscopic line",
                        "equilibrium constant",
                    ),
                    modifiers=("reversible", "irreversible", "transition_state", "vibrational", "electronic"),
                    qualifiers=("estimate", "dependence", "profile", "balance", "constraint"),
                    reason_modes=("energy_balance", "kinetic_reasoning", "spectral_interpretation"),
                ),
                SubfieldBlueprint(
                    key="analytical_chemistry",
                    label="analytical_chemistry",
                    focus="measurement, calibration, chromatography, and titration",
                    core_terms=(
                        "titration endpoint",
                        "calibration curve",
                        "chromatography peak",
                        "mass spectrometry fragment",
                        "limit of detection",
                        "standard addition",
                        "retention time",
                        "signal baseline",
                    ),
                    modifiers=("quantitative", "qualitative", "isocratic", "gradient", "internal_standard"),
                    qualifiers=("measurement", "correction", "resolution", "uncertainty", "separation"),
                    reason_modes=("measurement_reasoning", "signal_discrimination", "calibration_alignment"),
                ),
            ),
        ),
        "humanities_social_science": DomainBlueprint(
            key="humanities_social_science",
            label="Humanities/Social Science",
            subject="humanities_social_science",
            target=256,
            formal_stride=5,
            grammar_stride=3,
            subfields=(
                SubfieldBlueprint(
                    key="ethics_population",
                    label="ethics_population",
                    focus="normative ethics, population ethics, and welfare aggregation",
                    core_terms=(
                        "utilitarianism",
                        "deontology",
                        "virtue ethics",
                        "population ethics",
                        "Arrhenius impossibility theorem",
                        "non-sadism principle",
                        "egalitarian welfare",
                        "person affecting view",
                    ),
                    modifiers=("classical", "average", "prioritarian", "person_affecting", "axiomatic"),
                    qualifiers=("principle", "criterion", "theorem", "tradeoff", "constraint"),
                    reason_modes=("contrastive_elimination", "axiom_comparison", "normative_balance"),
                ),
                SubfieldBlueprint(
                    key="logic_and_philosophy",
                    label="logic_and_philosophy",
                    focus="predicate logic, modal reasoning, and philosophical analysis",
                    core_terms=(
                        "modal logic",
                        "predicate logic",
                        "proof theory",
                        "possible world semantics",
                        "necessity operator",
                        "soundness theorem",
                        "completeness theorem",
                        "reference relation",
                    ),
                    modifiers=("classical", "modal", "intuitionistic", "epistemic", "formal"),
                    qualifiers=("argument", "semantics", "proof", "criterion", "paradox"),
                    reason_modes=("argument_decomposition", "semantic_verification", "modal_tracking"),
                ),
                SubfieldBlueprint(
                    key="legal_theory",
                    label="legal_theory",
                    focus="legal principles, interpretation, and institutional reasoning",
                    core_terms=(
                        "legal precedent",
                        "statutory interpretation",
                        "proportionality test",
                        "due process",
                        "burden of proof",
                        "strict scrutiny",
                        "jurisdictional limit",
                        "constitutional principle",
                    ),
                    modifiers=("comparative", "constitutional", "administrative", "criminal", "civil"),
                    qualifiers=("analysis", "balancing", "interpretation", "threshold", "review"),
                    reason_modes=("case_comparison", "norm_hierarchy", "interpretive_balance"),
                ),
                SubfieldBlueprint(
                    key="rhetoric_language_figures",
                    label="rhetoric_language_figures",
                    focus="figurative language, irony, metaphor, and literary interpretation",
                    core_terms=(
                        "irony",
                        "sarcasm",
                        "pun",
                        "paradox",
                        "oxymoron",
                        "allusion",
                        "personification",
                        "metaphor",
                        "hyperbole",
                        "understatement",
                        "euphemism",
                    ),
                    modifiers=("dramatic", "verbal", "situational", "extended", "contextual"),
                    qualifiers=("interpretation", "signal", "marker", "contrast", "reading"),
                    reason_modes=("figurative_interpretation", "irony_detection", "contrastive_reading"),
                ),
                SubfieldBlueprint(
                    key="social_theory",
                    label="social_theory",
                    focus="institutions, norms, and collective behavior",
                    core_terms=(
                        "institutional norm",
                        "social contract",
                        "collective action",
                        "public reason",
                        "legitimacy claim",
                        "deliberative process",
                        "power relation",
                        "moral psychology",
                    ),
                    modifiers=("collective", "institutional", "public", "normative", "strategic"),
                    qualifiers=("analysis", "equilibrium", "failure", "constraint", "coordination"),
                    reason_modes=("institutional_reasoning", "norm_conflict_resolution", "collective_choice"),
                ),
            ),
        ),
        "engineering": DomainBlueprint(
            key="engineering",
            label="Engineering",
            subject="engineering",
            target=160,
            formal_stride=3,
            grammar_stride=4,
            subfields=(
                SubfieldBlueprint(
                    key="circuits_and_signal",
                    label="circuits_and_signal",
                    focus="circuits, transfer functions, and signal behavior",
                    core_terms=(
                        "transfer function",
                        "Kirchhoff current law",
                        "Bode plot",
                        "feedback amplifier",
                        "filter cutoff",
                        "sampling theorem",
                        "impedance matching",
                        "state variable filter",
                    ),
                    modifiers=("linear", "nonlinear", "analog", "digital", "closed_loop"),
                    qualifiers=("response", "stability", "design", "analysis", "constraint"),
                    reason_modes=("signal_chain_reasoning", "stability_check", "design_tradeoff"),
                ),
                SubfieldBlueprint(
                    key="control_systems",
                    label="control_systems",
                    focus="control loops, stability, and state estimation",
                    core_terms=(
                        "PID controller",
                        "state observer",
                        "controllability matrix",
                        "Kalman filter",
                        "phase margin",
                        "root locus",
                        "state feedback",
                        "robust stability",
                    ),
                    modifiers=("adaptive", "robust", "optimal", "continuous", "discrete"),
                    qualifiers=("design", "estimate", "margin", "criterion", "response"),
                    reason_modes=("feedback_reasoning", "state_estimation", "control_tuning"),
                ),
                SubfieldBlueprint(
                    key="fluid_and_thermal",
                    label="fluid_and_thermal",
                    focus="fluid flow, heat transfer, and thermodynamic cycles",
                    core_terms=(
                        "Navier Stokes flow",
                        "boundary layer",
                        "Reynolds number",
                        "heat exchanger",
                        "Rankine cycle",
                        "compressible flow",
                        "turbulent regime",
                        "thermal resistance",
                    ),
                    modifiers=("laminar", "turbulent", "compressible", "steady", "transient"),
                    qualifiers=("regime", "balance", "loss", "estimate", "design"),
                    reason_modes=("flow_regime_reasoning", "thermal_balance", "cycle_analysis"),
                ),
                SubfieldBlueprint(
                    key="structures_and_materials",
                    label="structures_and_materials",
                    focus="structural behavior, materials, and geotechnical foundations",
                    core_terms=(
                        "stress strain curve",
                        "fatigue crack",
                        "buckling mode",
                        "shear force diagram",
                        "foundation settlement",
                        "elastic modulus",
                        "composite laminate",
                        "geotechnical bearing capacity",
                    ),
                    modifiers=("elastic", "plastic", "reinforced", "composite", "geotechnical"),
                    qualifiers=("failure", "criterion", "response", "design", "assessment"),
                    reason_modes=("load_path_reasoning", "failure_diagnosis", "material_selection"),
                ),
            ),
        ),
        "other": DomainBlueprint(
            key="other",
            label="Other",
            subject="other",
            target=160,
            formal_stride=4,
            grammar_stride=3,
            subfields=(
                SubfieldBlueprint(
                    key="chess",
                    label="chess",
                    focus="chess notation, mating patterns, and strategic motifs",
                    core_terms=(
                        "algebraic chess notation",
                        "back rank mate",
                        "smothered mate",
                        "fork tactic",
                        "pin motif",
                        "discovered attack",
                        "minor piece ending",
                        "queen sacrifice",
                    ),
                    modifiers=("forced", "tactical", "thematic", "classical", "defensive"),
                    qualifiers=("pattern", "sequence", "resource", "conversion", "defense"),
                    reason_modes=("contrastive_elimination", "tactical_projection", "mate_pattern_matching"),
                ),
                SubfieldBlueprint(
                    key="classical_ciphers",
                    label="classical_ciphers",
                    focus="substitution ciphers, code breaking, and letter-frequency reasoning",
                    core_terms=(
                        "Caesar cipher",
                        "substitution cipher",
                        "Vigenere cipher",
                        "frequency analysis",
                        "monoalphabetic mapping",
                        "cipher alphabet",
                        "plaintext recovery",
                        "key schedule",
                    ),
                    modifiers=("classical", "manual", "rotational", "polyalphabetic", "frequency_based"),
                    qualifiers=("attack", "recovery", "mapping", "constraint", "deduction"),
                    reason_modes=("cipher_elimination", "frequency_mapping", "pattern_recovery"),
                ),
                SubfieldBlueprint(
                    key="logic_puzzles",
                    label="logic_puzzles",
                    focus="deduction patterns, grid puzzles, and constraint propagation",
                    core_terms=(
                        "constraint propagation",
                        "logic grid puzzle",
                        "truth teller puzzle",
                        "Sudoku candidate",
                        "knight and knave puzzle",
                        "Latin square",
                        "set intersection clue",
                        "elimination table",
                    ),
                    modifiers=("iterative", "deductive", "symbolic", "grid_based", "contradiction_based"),
                    qualifiers=("strategy", "step", "invariant", "resolution", "clue"),
                    reason_modes=("deduction_chain", "constraint_filtering", "contradiction_search"),
                ),
                SubfieldBlueprint(
                    key="scientific_miscellany",
                    label="scientific_miscellany",
                    focus="cross-domain reference knowledge that appears in difficult exams",
                    core_terms=(
                        "measurement convention",
                        "reference frame",
                        "taxonomy rule",
                        "classification code",
                        "notation standard",
                        "symbol legend",
                        "terminology contrast",
                        "evidence hierarchy",
                    ),
                    modifiers=("canonical", "historical", "domain_specific", "cross_domain", "comparative"),
                    qualifiers=("mapping", "translation", "interpretation", "criterion", "reference"),
                    reason_modes=("reference_alignment", "notation_resolution", "domain_bridge_reasoning"),
                ),
            ),
        ),
    }


DOMAIN_BLUEPRINTS = _build_domain_blueprints()


def build_lhe_foundational_payload(
    *,
    target_concepts: int = 2048,
    domains: list[str] | None = None,
    with_ollama: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = domains or list(DEFAULT_DOMAIN_ALLOCATIONS.keys())
    allocations = _scale_allocations(target_concepts=target_concepts, domains=selected)
    rows: list[dict[str, Any]] = []
    domain_counts: Counter[str] = Counter()
    subfield_counts: Counter[str] = Counter()
    used_ids: dict[str, set[str]] = {}
    for domain_key in selected:
        blueprint = DOMAIN_BLUEPRINTS[domain_key]
        domain_rows = _build_rows_for_domain(
            blueprint,
            allocations[domain_key],
            used_ids=used_ids,
        )
        rows.extend(domain_rows)
        domain_counts[domain_key] = allocations[domain_key]
        for row in domain_rows:
            entry = row["entry"]
            if row["galaxy"] == "Reality":
                subfield_counts[str(entry.get("category", ""))] += 1

    galaxy_counts: Counter[str] = Counter(row["galaxy"] for row in rows)
    manifest = {
        "bootstrap": BOOTSTRAP_TAG,
        "target_concepts": int(target_concepts),
        "concept_family_count": int(sum(domain_counts.values())),
        "row_count": len(rows),
        "domains": selected,
        "domain_concept_counts": dict(domain_counts),
        "subfield_counts": dict(subfield_counts),
        "galaxy_row_counts": dict(galaxy_counts),
        "with_ollama_requested": bool(with_ollama),
        "deterministic_only": True,
        "forbidden_smoke_ids": list(_FORBIDDEN_SMOKE_IDS),
        "forbidden_smoke_strings": list(_FORBIDDEN_SMOKE_STRINGS),
        "representative_ids": {
            "reality": ["concept_math_homology_group", "concept_physics_gamma_matrices", "concept_humanities_non_sadism_principle"],
            "word": ["word_homology_group", "word_gamma_matrices", "word_non_sadism_principle"],
            "math": ["math_gamma_matrix_clifford_relation"],
            "grammar": ["grammar_humanities_non_sadism_principle_reasoning"],
        },
    }
    return rows, manifest


def _validate_payload(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    ids_by_galaxy: dict[str, set[str]] = {}
    concept_ids: set[str] = set()
    row_blob = json.dumps(rows, ensure_ascii=True, sort_keys=True)
    for forbidden in _FORBIDDEN_SMOKE_IDS:
        if forbidden in row_blob:
            raise ValueError(f"forbidden benchmark id leaked into corpus: {forbidden}")
    for forbidden in _FORBIDDEN_SMOKE_STRINGS:
        if forbidden in row_blob:
            raise ValueError(f"forbidden benchmark string leaked into corpus: {forbidden}")

    for row in rows:
        galaxy = str(row.get("galaxy", "")).strip()
        entry = row.get("entry") if isinstance(row.get("entry"), dict) else None
        if not galaxy or entry is None:
            raise ValueError("invalid payload row")
        entry_id = str(entry.get("id", "")).strip()
        if not entry_id:
            raise ValueError("entry missing id")
        ids_by_galaxy.setdefault(galaxy, set())
        if entry_id in ids_by_galaxy[galaxy]:
            raise ValueError(f"duplicate id in {galaxy}: {entry_id}")
        ids_by_galaxy[galaxy].add(entry_id)
        if galaxy == "Reality":
            concept_ids.add(entry_id)

    for row in rows:
        galaxy = str(row["galaxy"])
        entry = row["entry"]
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        if galaxy == "Word":
            meaning_ref = str(metadata.get("meaning_ref", "")).strip()
            if meaning_ref not in concept_ids:
                raise ValueError(f"word entry points to missing reality concept: {meaning_ref}")
        if galaxy == "Math":
            formalizes_ref = str(metadata.get("formalizes_ref", "")).strip()
            if formalizes_ref not in concept_ids:
                raise ValueError(f"math entry points to missing reality concept: {formalizes_ref}")
        if galaxy == "Grammar":
            reasons_about_ref = str(metadata.get("reasons_about_ref", "")).strip()
            if reasons_about_ref not in concept_ids:
                raise ValueError(f"grammar entry points to missing reality concept: {reasons_about_ref}")

    if int(manifest["concept_family_count"]) != len(ids_by_galaxy.get("Reality", set())):
        raise ValueError("manifest concept count mismatch")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSONL file compatible with fundamental_ingest_payloads.py",
    )
    parser.add_argument(
        "--domain",
        action="append",
        default=None,
        help="Repeatable domain filter. Defaults to all supported domains.",
    )
    parser.add_argument(
        "--target-concepts",
        type=int,
        default=2048,
        help="Total concept families to generate across the selected domains.",
    )
    parser.add_argument(
        "--with-ollama",
        action="store_true",
        default=False,
        help="Reserved for future augmentation-time enrichment. Deterministic baseline ignores it.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional explicit manifest output path. Defaults to <output>.manifest.json",
    )
    parser.add_argument(
        "--dry-summary",
        action="store_true",
        default=False,
        help="Print the manifest summary and do not write files.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    selected = [item.strip() for item in (args.domain or []) if str(item).strip()]
    selected = selected or list(DEFAULT_DOMAIN_ALLOCATIONS.keys())
    unknown = [name for name in selected if name not in DOMAIN_BLUEPRINTS]
    if unknown:
        raise SystemExit(f"unknown domain(s): {', '.join(sorted(unknown))}")

    rows, manifest = build_lhe_foundational_payload(
        target_concepts=int(args.target_concepts),
        domains=selected,
        with_ollama=bool(args.with_ollama),
    )
    _validate_payload(rows, manifest)

    if args.with_ollama:
        print("[lhe-foundational] with_ollama requested; deterministic baseline emitted unchanged")

    if args.dry_summary:
        print(json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True))
        return 0

    output = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest is not None else output.with_suffix(".manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")

    print(f"[lhe-foundational] concepts={manifest['concept_family_count']}")
    print(f"[lhe-foundational] rows={manifest['row_count']}")
    print(f"[lhe-foundational] output={output}")
    print(f"[lhe-foundational] manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
