from __future__ import annotations

from knowledge3d.knowledgeverse.meaning_first_reasoning import (
    MeaningAtom,
    fuse_meaning_atoms,
    meaning_atoms_from_evidence_rows,
    meaning_atoms_from_parse_entities,
)


def test_meaning_atoms_from_evidence_rows_extracts_meaning_ref_and_symlinks() -> None:
    rows = [
        {
            "rank_weight": 0.8,
            "row": {
                "entry": {
                    "id": "word_gelu_nonlinearity",
                    "name": "gelu nonlinearity",
                    "description": "Lexical form for GELU nonlinearity.",
                    "metadata": {
                        "meaning_ref": "concept_cs_gelu_nonlinearity",
                        "subject": "computer_science",
                        "subfield": "deep_learning",
                        "aliases": ["gelu activation"],
                        "symlinks": ["concept_cs_gelu_nonlinearity"],
                        "word_refs": ["word_gelu_nonlinearity"],
                        "related_concepts": ["transformer architecture"],
                        "semantics": "Lexical access form for GELU nonlinearity.",
                    },
                }
            },
        }
    ]
    atoms = meaning_atoms_from_evidence_rows(rows)
    assert len(atoms) == 1
    atom = atoms[0]
    assert atom.concept_ref == "concept_cs_gelu_nonlinearity"
    assert atom.subject == "computer_science"
    assert "gelu activation" in atom.forms
    assert "word_gelu_nonlinearity" in atom.symlinks


def test_meaning_atoms_from_parse_entities_preserves_source_pass() -> None:
    atoms = meaning_atoms_from_parse_entities(
        [
            {
                "value": "gamma matrices",
                "raw": "gamma matrices",
                "role": "goal",
                "source_pass": "fusion",
                "confidence": 0.75,
            }
        ]
    )
    assert len(atoms) == 1
    assert atoms[0].source_pass == "fusion"
    assert atoms[0].canonical_name == "gamma matrices"


def test_meaning_atoms_from_evidence_rows_prefer_answer_bearing_field_content() -> None:
    rows = [
        {
            "rank_weight": 0.9,
            "row": {
                "entry": {
                    "id": "math_gamma_bivector_sandwich_factor",
                    "name": "gamma sandwich factor",
                    "description": "Relevant proportionality factor for antisymmetrized gamma matrices.",
                    "metadata": {
                        "meaning_ref": "concept_physics_gamma_factor",
                        "subject": "physics",
                        "subfield": "qft",
                        "aliases": ["antisymmetrized gamma matrices"],
                        "formalizes_ref": "concept_physics_gamma_factor",
                        "semantics": "symbolic factor for gamma contraction identity",
                    },
                }
            },
            "fields": {
                "content": "\\(-((d - 2k)^2) + d\\)",
                "description": "Relevant proportionality factor for antisymmetrized gamma matrices.",
            },
        }
    ]
    atoms = meaning_atoms_from_evidence_rows(rows)
    assert len(atoms) == 1
    atom = atoms[0]
    assert atom.summary == "\\(-((d - 2k)^2) + d\\)"
    assert "\\(-((d - 2k)^2) + d\\)" in atom.forms
    assert "concept_physics_gamma_factor" in atom.symlinks


def test_fuse_meaning_atoms_merges_forms_and_confidence() -> None:
    atoms = [
        MeaningAtom(
            atom_id="a",
            concept_ref="concept_alpha",
            canonical_name="Alpha",
            domain="math",
            subject="mathematics",
            subfield="algebra",
            source_pass="evidence",
            confidence=0.6,
            forms=("Alpha",),
            related_refs=("beta",),
            symlinks=("word_alpha",),
            semantics="alpha concept",
            summary="Alpha summary",
        ),
        MeaningAtom(
            atom_id="b",
            concept_ref="concept_alpha",
            canonical_name="Alpha Concept",
            domain="math",
            subject="mathematics",
            subfield="algebra",
            source_pass="fusion",
            confidence=0.8,
            forms=("Alpha Concept", "alpha"),
            related_refs=("gamma",),
            symlinks=("word_alpha_alt",),
            semantics="",
            summary="Longer Alpha summary",
        ),
    ]
    fused = fuse_meaning_atoms(atoms)
    assert len(fused) == 1
    atom = fused[0]
    assert atom.confidence == 0.8
    assert "alpha" in tuple(value.lower() for value in atom.forms)
    assert "gamma" in atom.related_refs
