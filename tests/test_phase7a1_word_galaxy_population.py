import gzip
import json
from pathlib import Path

from knowledge3d.ingestion.canonical_lookup import canonical_char_star_id, canonical_word_star_id
from knowledge3d.ingestion.universal_knowledge.dbnary_ingester import (
    LexicalRecord,
    iter_dbnary_records,
    lexical_record_to_word_star,
    merge_lexical_records_into_omw,
)
from knowledge3d.ingestion.universal_knowledge.kaikki_ingester import iter_kaikki_records
from knowledge3d.ingestion.universal_knowledge.multilingual_meanings import SynsetEntry
from scripts.run_phase7a1_unified_ingestion import (
    build_unified_ingestion,
    find_dangling_refs,
    write_unified_ingestion,
)


DEJAVU_SANS = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def _write_omw(root: Path) -> Path:
    lang_dir = root / "eng"
    lang_dir.mkdir(parents=True)
    (lang_dir / "wn-data-eng.tab").write_text(
        "\n".join(
            [
                "00000001-n\teng:lemma\tcat",
                "00000001-n\teng:def\tcat\tA small domesticated carnivorous mammal.",
            ]
        ),
        encoding="utf-8",
    )
    return root


def _manifest(path: Path = DEJAVU_SANS) -> dict:
    return {
        "fonts": [
            {
                "path": str(path),
                "file": path.name,
                "font_index": 0,
                "family": "DejaVu Sans",
                "style": "Book",
                "scripts": ["latn"],
                "codepoint_count": 26,
                "codepoint_ranges": [["U+0061", "U+007A"]],
            }
        ]
    }


def test_dbnary_parser_extracts_lexical_entries(tmp_path):
    ttl = tmp_path / "en_dbnary_ontolex.ttl"
    ttl.write_text(
        """
eng:cat__Noun__1
        rdf:type               lexinfo:Noun , ontolex:LexicalEntry , ontolex:Word;
        rdfs:label             "cat"@en;
        dbnary:partOfSpeech    "Noun";
        lime:language          "en" .

eng:__cf_cat__Noun__1
        rdf:type             ontolex:Form;
        ontolex:phoneticRep  "/kæt/"@en-fonipa;
        ontolex:writtenRep   "cat"@en .
""",
        encoding="utf-8",
    )

    records = list(iter_dbnary_records(ttl))

    assert records == [LexicalRecord(source="dbnary", language="en", lemma="cat", pos="noun")]


def test_kaikki_parser_extracts_records_from_jsonl_gz(tmp_path):
    path = tmp_path / "kaikki.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "word": "gato",
                    "lang_code": "pt",
                    "pos": "noun",
                    "senses": [{"glosses": ["cat"]}],
                    "sounds": [{"ipa": "/ˈɡatu/"}],
                    "etymology_text": "From Latin cattus.",
                }
            )
            + "\n"
        )

    records = list(iter_kaikki_records(path))

    assert records[0].source == "kaikki"
    assert records[0].language == "pt"
    assert records[0].definitions == ("cat",)
    assert records[0].pronunciations == ("/ˈɡatu/",)


def test_merge_records_preserves_omw_primary_and_creates_misses_only():
    synsets = {
        "00000001-n": SynsetEntry(
            synset_id="00000001-n",
            pos="n",
            lemmas={"en": ["cat"]},
        )
    }
    records = [
        LexicalRecord(source="dbnary", language="en", lemma="cat", pos="noun"),
        LexicalRecord(source="dbnary", language="en", lemma="dog", pos="noun"),
    ]

    result = merge_lexical_records_into_omw(synsets, records)

    assert result.merged_count == 1
    assert canonical_word_star_id("en", "dog") in result.new_word_stars
    assert canonical_word_star_id("en", "cat") not in result.new_word_stars


def test_lexical_record_word_star_carries_component_refs():
    star = lexical_record_to_word_star(LexicalRecord(source="kaikki", language="pt", lemma="gato"))

    assert star.star_id == "word_pt_gato"
    assert star.component_refs == [
        canonical_char_star_id("g"),
        canonical_char_star_id("a"),
        canonical_char_star_id("t"),
        canonical_char_star_id("o"),
    ]


def test_unified_runner_merges_sources_and_reports_no_dangling_refs(tmp_path):
    assert DEJAVU_SANS.exists(), "required_system_font_missing:DejaVuSans.ttf"
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    grammar_path = tmp_path / "UD_GRAMMAR_STARS.jsonl"
    grammar_path.write_text("", encoding="utf-8")
    grammar_path.with_suffix(".jsonl.canonical.json").write_text("[]", encoding="utf-8")
    omw_path = _write_omw(tmp_path / "omw")
    dbnary_path = tmp_path / "en_dbnary_ontolex.ttl"
    dbnary_path.write_text(
        """
eng:dog__Noun__1
        rdf:type               lexinfo:Noun , ontolex:LexicalEntry , ontolex:Word;
        rdfs:label             "dog"@en;
        dbnary:partOfSpeech    "Noun";
        lime:language          "en" .
""",
        encoding="utf-8",
    )

    result = build_unified_ingestion(
        manifest_path=manifest_path,
        grammar_star_path=grammar_path,
        omw_path=omw_path,
        dbnary_paths=[dbnary_path],
        kaikki_paths=[],
    )

    assert "word_en_cat" in result.stars
    assert "word_en_dog" in result.stars
    assert "synset_00000001_n" in result.stars["word_en_cat"]["taxonomy_refs"]
    assert "word_en_cat" in result.stars["char_c"]["composite_of"]
    assert result.dangling_refs == []
    out = write_unified_ingestion(result, tmp_path / "unified.jsonl")
    assert out.exists()


def test_dangling_ref_report_is_strict():
    rows = {
        "word_en_gap": {
            "star_id": "word_en_gap",
            "domain": "Word/en",
            "component_refs": ["char_missing"],
        }
    }

    assert find_dangling_refs(rows) == [
        {"source": "word_en_gap", "field": "component_refs", "target": "char_missing"}
    ]
