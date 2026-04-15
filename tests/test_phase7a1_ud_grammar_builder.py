from pathlib import Path

from knowledge3d.ingestion.canonical_lookup import canonical_grammar_template_id
from knowledge3d.ingestion.grammar.ud_grammar_builder import (
    CANONICAL_TREEBANKS,
    build_ud_grammar_galaxy,
    iter_conllu_sentences,
    register_ud_grammar_canonical_entries,
    write_ud_grammar_build,
)


class FakeCanonicalLookup:
    def __init__(self):
        self.calls = []

    def register(self, *, kind, key, star_id, metadata):
        self.calls.append({"kind": kind, "key": key, "star_id": star_id, "metadata": metadata})
        return star_id


def _write_treebank(root: Path, language: str = "en", treebank: str = "UD_English-EWT") -> Path:
    tb = root / treebank
    tb.mkdir(parents=True)
    path = tb / f"{language}_sample-ud-train.conllu"
    path.write_text(
        "\n".join(
            [
                "# sent_id = 1",
                "1\tI\tI\tPRON\tPRP\tNumber=Sing|Person=1\t2\tnsubj\t_\t_",
                "2\tlike\tlike\tVERB\tVBP\tTense=Pres\t0\troot\t_\t_",
                "3\tthe\tthe\tDET\tDT\tDefinite=Def\t4\tdet\t_\t_",
                "4\tcats\tcat\tNOUN\tNNS\tNumber=Plur\t2\tobj\t_\t_",
                "",
                "# sent_id = 2",
                "1\tSaudades\tsaudades\tNOUN\tNN\tNumber=Plur\t0\troot\t_\t_",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_conllu_parser_yields_sentences(tmp_path):
    path = _write_treebank(tmp_path)
    sentences = list(iter_conllu_sentences(path))

    assert len(sentences) == 2
    assert sentences[0][0].form == "I"
    assert sentences[0][3].deprel == "obj"


def test_ud_grammar_builder_emits_templates_rules_and_canonical_entries(tmp_path):
    _write_treebank(tmp_path)
    build = build_ud_grammar_galaxy(ud_root=tmp_path, treebanks={"en": "UD_English-EWT"})

    assert "grammar_template_en_copula" in build.stars
    assert "grammar_template_en_periphrastic_explanation" in build.stars
    assert "grammar_template_en_nsubj_obj" in build.stars
    assert "grammar_template_en_det_noun" in build.stars
    assert "grammar_rule_en_upos_noun" in build.stars
    assert build.stars["grammar_template_en_periphrastic_explanation"]["meaning_rpn"] == (
        "MEANING_DECOMPOSE GRAMMAR_SELECT WORD_RESOLVE MORPHOLOGY_APPLY SURFACE_EMIT"
    )
    assert build.treebank_stats["en"]["sentences"] == 2
    assert build.treebank_stats["en"]["tokens"] == 5
    assert {
        (entry["kind"], entry["key"], entry["star_id"])
        for entry in build.canonical_entries
    } >= {
        ("grammar_template", "en:copula", canonical_grammar_template_id("en", "copula")),
        ("grammar_template", "en:periphrastic_explanation", canonical_grammar_template_id("en", "periphrastic_explanation")),
    }


def test_ud_grammar_registers_canonical_entries(tmp_path):
    _write_treebank(tmp_path)
    build = build_ud_grammar_galaxy(ud_root=tmp_path, treebanks={"en": "UD_English-EWT"})
    lookup = FakeCanonicalLookup()

    count = register_ud_grammar_canonical_entries(lookup, build)

    assert count == len(build.canonical_entries)
    assert any(call["kind"] == "grammar_rule" for call in lookup.calls)
    assert any(call["key"] == "en:periphrastic_explanation" for call in lookup.calls)


def test_ud_grammar_build_writes_local_artifacts(tmp_path):
    _write_treebank(tmp_path)
    build = build_ud_grammar_galaxy(ud_root=tmp_path, treebanks={"en": "UD_English-EWT"})
    out = tmp_path / "UD_GRAMMAR_STARS.jsonl"

    written = write_ud_grammar_build(build, out)

    assert written == out
    assert out.exists()
    assert out.with_suffix(".jsonl.canonical.json").exists()
    assert out.with_suffix(".jsonl.meta.json").exists()


def test_phase7_treebank_selection_names_nine_languages():
    assert CANONICAL_TREEBANKS == {
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
