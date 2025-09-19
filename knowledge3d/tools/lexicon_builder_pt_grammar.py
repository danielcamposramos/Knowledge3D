from __future__ import annotations

"""Build European Portuguese grammar stars from curated rules."""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional

from knowledge3d.tools.lexicon.common import build_star

GRAMMAR_RULES: List[Dict[str, object]] = [
    {
        "lemma": "Pronomes pessoais (sujeito)",
        "definition": "Lista dos pronomes pessoais do português europeu: eu, tu, ele/ela, nós, vós, eles/elas.",
        "synonyms": ["pronomes sujeitos"],
        "examples": ["Nós falamos português todos os dias."],
    },
    {
        "lemma": "Contração de preposição com artigo",
        "definition": "As preposições 'a', 'de', 'em' e 'por' contraem-se com os artigos definidos: ao, do, no, pelo, etc.",
        "examples": ["Vou ao mercado", "Vim do trabalho", "Estou no centro", "Passei pelo jardim"],
    },
    {
        "lemma": "Tempo verbal: Presente do indicativo",
        "definition": "Tempo usado para ações habituais ou verdades universais. Exemplo com o verbo 'falar': eu falo, tu falas, ele fala, nós falamos, vós falais, eles falam.",
        "synonyms": ["presente simples"],
        "examples": ["Eles falam baixo na biblioteca."],
    },
    {
        "lemma": "Tempo verbal: Pretérito perfeito",
        "definition": "Expressa uma ação concluída no passado. Exemplo com 'comer': eu comi, tu comeste, ele comeu, nós comemos, vós comestes, eles comeram.",
        "examples": ["Ontem comemos tarde."],
    },
    {
        "lemma": "Colocação pronominal: Próclise",
        "definition": "Uso do pronome antes do verbo em contextos com palavras negativas ou pronomes relativos: 'Não me digas isso.'",
        "examples": ["Ninguém me avisou.", "Quem te disse isso?"],
    },
    {
        "lemma": "Colocação pronominal: Ênclise",
        "definition": "Uso do pronome depois do verbo quando a oração começa pelo verbo no infinitivo, gerúndio ou imperativo afirmativo: 'Diz-me a verdade.'",
        "examples": ["Fazer-te-ei um favor.", "Sentando-se, descansou."],
    },
    {
        "lemma": "Colocação pronominal: Mesóclise",
        "definition": "Uso do pronome no meio do verbo, típico do futuro do indicativo ou condicional: 'Dir-te-ei a verdade.'",
        "examples": ["Far-se-á justiça."],
    },
    {
        "lemma": "Subjuntivo presente",
        "definition": "Modo usado para hipóteses ou desejos. Exemplo com 'ser': que eu seja, que tu sejas, que ele seja, que nós sejamos, que vós sejais, que eles sejam.",
        "synonyms": ["presente do conjuntivo"],
        "examples": ["Espero que sejas feliz."],
    },
    {
        "lemma": "Uso do artigo definido",
        "definition": "No português europeu, usa-se frequentemente artigo definido antes de nomes próprios: 'A Maria chegou cedo.'",
        "examples": ["O João vai ao cinema."],
    },
    {
        "lemma": "Infinitivo pessoal",
        "definition": "Forma do infinitivo que concorda com o sujeito: 'para nós fazermos', 'para eles fazerem'.",
        "examples": ["É importante vocês estudarem.", "Antes de partirem, arrumem a casa."],
    },
    {
        "lemma": "Gerúndio",
        "definition": "Indica ação em progresso: 'Estou a trabalhar.' Em Portugal, usa-se frequentemente a perífrase 'estar a + infinitivo'.",
        "examples": ["Eles estão a conversar."],
    },
    {
        "lemma": "Perífrase verbal: Estar prestes a",
        "definition": "Expressa iminência: 'Estou prestes a sair.'",
        "examples": ["A equipa estava prestes a vencer."],
    },
    {
        "lemma": "Plural dos nomes terminados em -ão",
        "definition": "As terminações variam: -ões (pão → pães), -ãos (mão → mãos), -ães (cão → cães).",
        "examples": ["Os cães correram atrás dos pães."],
    },
    {
        "lemma": "Pret. imperfeito do indicativo",
        "definition": "Expressa ações habituais no passado: 'falava', 'comia', 'partia'.",
        "examples": ["Quando era criança, jogava na rua."],
    },
    {
        "lemma": "Futuro do presente",
        "definition": "Expressa ações futuras: 'falarei', 'comeremos'.",
        "examples": ["Amanhã viajaremos cedo."],
    },
    {
        "lemma": "Futuro do conjuntivo",
        "definition": "Usado em orações condicionais: 'se eu for, se tu fores, se ele for, se nós formos, se vós fordes, se eles forem'.",
        "examples": ["Quando chegares, avisa-me."],
    },
    {
        "lemma": "Imperativo afirmativo",
        "definition": "Formado a partir do presente do subjuntivo (exceto as formas tu e vós, derivadas do presente do indicativo sem o 's'): 'fala tu', 'falai vós'.",
        "examples": ["Fala mais devagar.", "Façam favor de entrar."],
    },
    {
        "lemma": "Imperativo negativo",
        "definition": "Usa as formas do presente do subjuntivo precedidas de 'não': 'não fales', 'não faleis'.",
        "examples": ["Não saias sem casaco."],
    },
    {
        "lemma": "Advérbios terminados em -mente",
        "definition": "Formados a partir do feminino do adjetivo: 'rápido' → 'rapidamente'.",
        "examples": ["Ela respondeu calmamente."],
    },
    {
        "lemma": "Uso de 'onde' e 'aonde'",
        "definition": "'Onde' indica permanência; 'aonde' movimento com 'a': 'Onde moras?', 'Aonde vais?'.",
        "examples": ["Aonde vais depois do trabalho?"],
    },
    {
        "lemma": "Crase",
        "definition": "Acontece quando a preposição 'a' encontra o artigo 'a/as': 'Vou à escola', 'Cheguei às três'.",
        "examples": ["Entregou o presente à irmã."],
    },
]


def build(args: argparse.Namespace) -> None:
    from knowledge3d.tools.lexicon.common import write_jsonl

    records = []
    for rule in GRAMMAR_RULES:
        lemma = rule["lemma"]
        definition = rule.get("definition", "")
        synonyms = rule.get("synonyms", []) or []
        examples = rule.get("examples", []) or []
        relations: Dict[str, List[str]] = {}
        extra: Dict[str, object] = {
            "lexicon_entry": {
                "language": "pt",
                "lemma": lemma,
                "pos": "grammar",
                "sense_id": lemma,
                "definition": definition,
                "glosses": [definition],
                "examples": examples,
                "synonyms": list(synonyms),
                "source": {
                    "dataset": "pt-pt-grammar-curated",
                },
            }
        }
        embedding_parts = [lemma, definition, " ".join(examples)]
        star = build_star(
            language="pt",
            source="pt-pt-grammar",
            lemma=lemma,
            pos="grammar",
            sense_ref=lemma,
            definition=definition,
            embedding_parts=embedding_parts,
            relations=relations,
            extra=extra,
            tags=["variant:pt-PT", "topic:grammar"],
        )
        records.append(star)
    write_jsonl(args.out, records)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build European Portuguese grammar stars")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("viewer/public/galaxy/working/lexicon_pt_pt_grammar.jsonl"),
        help="Output JSONL path",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:  # pragma: no cover
    args = parse_args(argv)
    build(args)


if __name__ == "__main__":  # pragma: no cover
    main()
