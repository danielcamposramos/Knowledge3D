"""Seed meaning-centric stars that act as domain anchors."""

from __future__ import annotations

from .meaning_star import MeaningCentricStar, SurfaceForm


def _char_refs(text: str, language: str) -> list[str]:
    refs: list[str] = []
    for char in text:
        if char.isspace():
            continue
        if char.isascii() and char.isalnum():
            refs.append(f"char_{char.lower()}")
        else:
            refs.append(f"char_{language}_u{ord(char):04x}")
    return refs


def _surface_form(language: str, ref: str, text: str) -> SurfaceForm:
    return SurfaceForm(word_ref=ref, char_refs=_char_refs(text, language))


def _seed_word_entry(entry_id: str, word: str, language: str) -> dict[str, object]:
    return {
        "id": entry_id,
        "name": word,
        "domain": "word",
        "category": "seed_lexeme",
        "content": word,
        "summary": word,
        "description": word,
        "rpn_program": f"LOOKUP {entry_id}",
        "metadata": {
            "language": language,
            "bootstrap": "meaning_star_seed_v1",
            "forms": [word],
        },
    }


def _seed_star(
    *,
    star_id: str,
    meaning_rpn: str,
    domain: str,
    english: tuple[str, str],
    portuguese: tuple[str, str],
    japanese: tuple[str, str],
    visual_rpn: str,
    taxonomy_refs: list[str],
    component_refs: list[str],
    house_position: tuple[float, float, float],
) -> MeaningCentricStar:
    return MeaningCentricStar(
        star_id=star_id,
        meaning_class="concept",
        meaning_rpn=meaning_rpn,
        domain=domain,
        taxonomy_refs=taxonomy_refs,
        surface_forms={
            "en": _surface_form("en", english[0], english[1]),
            "pt": _surface_form("pt", portuguese[0], portuguese[1]),
            "ja": _surface_form("ja", japanese[0], japanese[1]),
        },
        visual_rpn=visual_rpn,
        behavior_rpn="DOMAIN_CENTER STABLE ATTRACT",
        grammar_refs=["grammar_domain_anchor"],
        meta_refs=["meta_prefer_domain_center", "meta_expand_related_meaning"],
        house_position=house_position,
        house_room=domain,
        confidence=1,
        polarity=1,
        component_refs=component_refs,
    )


def build_seed_stars() -> list[MeaningCentricStar]:
    return [
        _seed_star(
            star_id="concept_mathematics",
            meaning_rpn="MATHEMATICS NUMBER PATTERN PROOF DOMAIN_CENTER",
            domain="Library/Mathematics",
            english=("seed_word_en_mathematics", "mathematics"),
            portuguese=("seed_word_pt_matematica", "matematica"),
            japanese=("seed_word_ja_sugaku", "数学"),
            visual_rpn="CIRCLE GRID SIGMA STACK",
            taxonomy_refs=["concept_language", "concept_tool", "concept_growth"],
            component_refs=["num_0", "num_1", "num_2", "word_zero", "word_one", "word_two"],
            house_position=(0.0, 0.0, 0.0),
        ),
        _seed_star(
            star_id="concept_physics",
            meaning_rpn="PHYSICS FORCE MOTION FIELD MEASURE DOMAIN_CENTER",
            domain="Library/Physics",
            english=("seed_word_en_physics", "physics"),
            portuguese=("seed_word_pt_fisica", "fisica"),
            japanese=("seed_word_ja_butsuri", "物理"),
            visual_rpn="ARROW WAVE ORBIT FIELD",
            taxonomy_refs=["concept_mathematics", "concept_tool", "concept_growth"],
            component_refs=[
                "reality_anchor_college_physics_core",
                "reality_law_newton_second",
                "reality_law_velocity",
                "reality_constant_speed_of_light",
            ],
            house_position=(3.0, 0.0, 0.0),
        ),
        _seed_star(
            star_id="concept_chemistry",
            meaning_rpn="CHEMISTRY ATOM BOND REACTION TRANSFORM DOMAIN_CENTER",
            domain="Library/Chemistry",
            english=("seed_word_en_chemistry", "chemistry"),
            portuguese=("seed_word_pt_quimica", "quimica"),
            japanese=("seed_word_ja_kagaku", "化学"),
            visual_rpn="HEXAGON BOND LATTICE FLASK",
            taxonomy_refs=["concept_physics", "concept_biology", "concept_tool"],
            component_refs=[
                "reality_anchor_college_chemistry_core",
                "reality_law_ideal_gas",
                "reality_chemistry_stoichiometry",
                "reality_constant_avogadro_constant",
            ],
            house_position=(6.0, 0.0, 0.0),
        ),
        _seed_star(
            star_id="concept_biology",
            meaning_rpn="BIOLOGY CELL ORGANISM GROWTH EVOLUTION DOMAIN_CENTER",
            domain="Library/Biology",
            english=("seed_word_en_biology", "biology"),
            portuguese=("seed_word_pt_biologia", "biologia"),
            japanese=("seed_word_ja_seibutsu", "生物"),
            visual_rpn="CELL TREE SPIRAL LEAF",
            taxonomy_refs=["concept_chemistry", "concept_growth", "concept_self_reflection"],
            component_refs=[
                "reality_anchor_college_biology_core",
                "reality_biology_cell_theory",
                "reality_biology_genetics_inheritance",
                "reality_biology_evolution_selection",
            ],
            house_position=(9.0, 0.0, 0.0),
        ),
        _seed_star(
            star_id="concept_language",
            meaning_rpn="LANGUAGE SYMBOL MEANING GRAMMAR COMMUNICATE DOMAIN_CENTER",
            domain="Library/Languages",
            english=("seed_word_en_language", "language"),
            portuguese=("seed_word_pt_linguagem", "linguagem"),
            japanese=("seed_word_ja_gengo", "言語"),
            visual_rpn="GLYPH TREE BRIDGE SPEECH",
            taxonomy_refs=["concept_mathematics", "concept_sound", "concept_self_reflection"],
            component_refs=["word_zero", "word_one", "word_two", "cipher_language_profile_en_common"],
            house_position=(12.0, 0.0, 0.0),
        ),
        _seed_star(
            star_id="concept_tool",
            meaning_rpn="TOOL ACTION INSTRUMENT CONSTRUCTION APPLY DOMAIN_CENTER",
            domain="Workshop/Tools",
            english=("seed_word_en_tool", "tool"),
            portuguese=("seed_word_pt_ferramenta", "ferramenta"),
            japanese=("seed_word_ja_dougu", "道具"),
            visual_rpn="HAMMER GEAR HANDLE ALIGN",
            taxonomy_refs=["concept_mathematics", "concept_physics", "concept_visual_art"],
            component_refs=["rotate_90", "reflect_horizontal", "translate_one"],
            house_position=(15.0, 0.0, 0.0),
        ),
        _seed_star(
            star_id="concept_growth",
            meaning_rpn="GROWTH CHANGE SCALE FEEDBACK ADAPT DOMAIN_CENTER",
            domain="Knowledge Gardens/Growth",
            english=("seed_word_en_growth", "growth"),
            portuguese=("seed_word_pt_crescimento", "crescimento"),
            japanese=("seed_word_ja_seichou", "成長"),
            visual_rpn="SEED SPROUT TREE BRANCH",
            taxonomy_refs=["concept_biology", "concept_mathematics", "concept_self_reflection"],
            component_refs=[
                "math_concept_compound_growth",
                "reality_biology_ecology_populations",
                "reality_biology_homeostasis",
            ],
            house_position=(18.0, 0.0, 0.0),
        ),
        _seed_star(
            star_id="concept_visual_art",
            meaning_rpn="VISUAL FORM COLOR COMPOSITION SEE DOMAIN_CENTER",
            domain="Gallery/Visual",
            english=("seed_word_en_visual_art", "visual art"),
            portuguese=("seed_word_pt_arte_visual", "arte visual"),
            japanese=("seed_word_ja_bijutsu", "美術"),
            visual_rpn="FRAME PALETTE CURVE BALANCE",
            taxonomy_refs=["concept_language", "concept_tool", "concept_sound"],
            component_refs=["rotate_90", "rotate_180", "mirror_main_diagonal"],
            house_position=(21.0, 0.0, 0.0),
        ),
        _seed_star(
            star_id="concept_sound",
            meaning_rpn="SOUND RHYTHM FREQUENCY RESONANCE LISTEN DOMAIN_CENTER",
            domain="Gallery/Audio",
            english=("seed_word_en_sound", "sound"),
            portuguese=("seed_word_pt_som", "som"),
            japanese=("seed_word_ja_oto", "音"),
            visual_rpn="WAVE SPIRAL PULSE RING",
            taxonomy_refs=["concept_language", "concept_visual_art", "concept_self_reflection"],
            component_refs=["audio_seed_waveform", "audio_seed_rhythm", "audio_seed_harmonic"],
            house_position=(24.0, 0.0, 0.0),
        ),
        _seed_star(
            star_id="concept_self_reflection",
            meaning_rpn="SELF REFLECTION MEMORY META OBSERVE DOMAIN_CENTER",
            domain="Bathtub/Observatory",
            english=("seed_word_en_self_reflection", "self reflection"),
            portuguese=("seed_word_pt_autorreflexao", "autorreflexao"),
            japanese=("seed_word_ja_jikohansei", "自己反省"),
            visual_rpn="MIRROR EYE ORBIT LOOP",
            taxonomy_refs=["concept_language", "concept_growth", "concept_sound"],
            component_refs=["meta_prefer_domain_center", "meta_expand_related_meaning", "concept_mathematics"],
            house_position=(27.0, 0.0, 0.0),
        ),
    ]


SEED_STARS = build_seed_stars()


def seed_star_entries() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for star in SEED_STARS:
        entries.append(
            star.to_galaxy_entry(
                entry_id=star.star_id,
                name=star.star_id.replace("concept_", "").replace("_", " ").title(),
                galaxy_name="Meaning",
            )
        )
    return entries


def seed_word_entries() -> list[dict[str, object]]:
    seen: set[str] = set()
    entries: list[dict[str, object]] = []
    labels = {
        "seed_word_en_mathematics": ("mathematics", "en"),
        "seed_word_pt_matematica": ("matematica", "pt"),
        "seed_word_ja_sugaku": ("数学", "ja"),
        "seed_word_en_physics": ("physics", "en"),
        "seed_word_pt_fisica": ("fisica", "pt"),
        "seed_word_ja_butsuri": ("物理", "ja"),
        "seed_word_en_chemistry": ("chemistry", "en"),
        "seed_word_pt_quimica": ("quimica", "pt"),
        "seed_word_ja_kagaku": ("化学", "ja"),
        "seed_word_en_biology": ("biology", "en"),
        "seed_word_pt_biologia": ("biologia", "pt"),
        "seed_word_ja_seibutsu": ("生物", "ja"),
        "seed_word_en_language": ("language", "en"),
        "seed_word_pt_linguagem": ("linguagem", "pt"),
        "seed_word_ja_gengo": ("言語", "ja"),
        "seed_word_en_tool": ("tool", "en"),
        "seed_word_pt_ferramenta": ("ferramenta", "pt"),
        "seed_word_ja_dougu": ("道具", "ja"),
        "seed_word_en_growth": ("growth", "en"),
        "seed_word_pt_crescimento": ("crescimento", "pt"),
        "seed_word_ja_seichou": ("成長", "ja"),
        "seed_word_en_visual_art": ("visual art", "en"),
        "seed_word_pt_arte_visual": ("arte visual", "pt"),
        "seed_word_ja_bijutsu": ("美術", "ja"),
        "seed_word_en_sound": ("sound", "en"),
        "seed_word_pt_som": ("som", "pt"),
        "seed_word_ja_oto": ("音", "ja"),
        "seed_word_en_self_reflection": ("self reflection", "en"),
        "seed_word_pt_autorreflexao": ("autorreflexao", "pt"),
        "seed_word_ja_jikohansei": ("自己反省", "ja"),
    }
    for entry_id, (word, language) in labels.items():
        if entry_id in seen:
            continue
        seen.add(entry_id)
        entries.append(_seed_word_entry(entry_id, word, language))
    return entries


__all__ = [
    "SEED_STARS",
    "build_seed_stars",
    "seed_star_entries",
    "seed_word_entries",
]
