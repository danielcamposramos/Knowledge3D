"""Meaning-centric foundational math star crafter."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import sys
from typing import Any

from knowledge3d.cranium.sovereign_matryoshka_embedder import (
    SovereignMatryoshkaTextEmbedder,
    get_sovereign_matryoshka_text_embedder,
)

try:  # pragma: no cover - optional dependency
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover - fallback for lean envs
    BaseModel = object  # type: ignore[assignment]

    def Field(default: Any = None, **_: Any) -> Any:
        return default


_MEANING_STAR_PATH = Path(__file__).resolve().parents[1] / "knowledgeverse" / "meaning_star.py"
_MEANING_STAR_SPEC = importlib.util.spec_from_file_location("k3d_meaning_star_standalone", _MEANING_STAR_PATH)
if _MEANING_STAR_SPEC is None or _MEANING_STAR_SPEC.loader is None:  # pragma: no cover - defensive
    raise RuntimeError(f"unable_to_load_meaning_star_module:{_MEANING_STAR_PATH}")
_MEANING_STAR_MODULE = importlib.util.module_from_spec(_MEANING_STAR_SPEC)
sys.modules.setdefault("k3d_meaning_star_standalone", _MEANING_STAR_MODULE)
_MEANING_STAR_SPEC.loader.exec_module(_MEANING_STAR_MODULE)
MeaningCentricStar = _MEANING_STAR_MODULE.MeaningCentricStar
SurfaceForm = _MEANING_STAR_MODULE.SurfaceForm
compute_star_id = _MEANING_STAR_MODULE.compute_star_id
wrap_galaxy_entry_with_meaning_star = _MEANING_STAR_MODULE.wrap_galaxy_entry_with_meaning_star


ROLE_UNKNOWN = "unknown"
ROLE_ROUTER = "router"
ROLE_EXECUTOR = "executor"
ROLE_ANSWER = "answer"

STAR_TYPE_CHARACTER = 2
STAR_TYPE_GRAMMAR = 3
STAR_TYPE_MATH = 5

STAR_FLAG_ACTIVE = 0x01
STAR_FLAG_LEARNABLE = 0x02

PROGRAM_ENTRY_ALIGN = 8
PROGRAM_TABLE_SENTINEL_BYTES = 8


class StarCraftSurfaceDraft(BaseModel):  # type: ignore[misc]
    word: str = Field(default="")
    char_refs: list[str] = Field(default_factory=list)


class StarCraftPlan(BaseModel):  # type: ignore[misc]
    entry_id: str = Field(default="")
    galaxy: str = Field(default="")
    name: str = Field(default="")
    meaning_class: str = Field(default="concept")
    meaning_rpn: str = Field(default="")
    domain: str = Field(default="")
    taxonomy_refs: list[str] = Field(default_factory=list)
    meta_refs: list[str] = Field(default_factory=list)
    grammar_refs: list[str] = Field(default_factory=list)
    component_refs: list[str] = Field(default_factory=list)
    composite_of: list[str] = Field(default_factory=list)
    surface_forms: dict[str, StarCraftSurfaceDraft] = Field(default_factory=dict)


@dataclass(frozen=True)
class ProgramSpec:
    program_id: str
    opcode_names: tuple[str, ...]
    bytecode: bytes
    meaning_rpn: str


@dataclass
class ProgramTableLayout:
    payload: bytes
    offsets: dict[str, int]
    lengths: dict[str, int]
    opcode_counts: dict[str, int]

    @property
    def size_bytes(self) -> int:
        return len(self.payload)


@dataclass
class _CraftedEntry:
    entry_id: str
    galaxy: str
    name: str
    category: str
    star_type: int
    selection_role: str
    answer_eligible: bool
    layer_id: int
    route_family: str
    route_policy: dict[str, Any]
    meaning_star: MeaningCentricStar
    embedding_stack: dict[int, list[float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    extra_top_level: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = {
            "id": self.entry_id,
            "name": self.name,
            "galaxy": self.galaxy,
            "domain": self.meaning_star.domain,
            "category": self.category,
            "star_type": self.star_type,
            "selection_role": self.selection_role,
            "answer_eligible": self.answer_eligible,
            "layer_id": self.layer_id,
            "route_family": self.route_family,
            "route_policy": dict(self.route_policy),
            "embedding": list(self.embedding_stack.get(64) or []),
            "embedding_tier_64": list(self.embedding_stack.get(64) or []),
            "embedding_tier_128": list(self.embedding_stack.get(128) or []),
            "embedding_tier_512": list(self.embedding_stack.get(512) or []),
            "embedding_tier_2048": list(self.embedding_stack.get(2048) or []),
            "embedding_max": list(self.embedding_stack.get(2048) or []),
            "flags": STAR_FLAG_ACTIVE | STAR_FLAG_LEARNABLE,
            "taxonomy_refs": list(self.meaning_star.taxonomy_refs),
            "meta_refs": list(self.meaning_star.meta_refs),
            "grammar_refs": list(self.meaning_star.grammar_refs),
            "component_refs": list(self.meaning_star.component_refs),
            "composite_of": list(self.meaning_star.composite_of),
            "surface_forms": {
                language: surface.word_ref
                for language, surface in self.meaning_star.surface_forms.items()
                if surface.word_ref
            },
            "sovereign_route_exempt": True,
        }
        row.update(dict(self.extra_top_level))
        wrapped = wrap_galaxy_entry_with_meaning_star(row, self.meaning_star)
        wrapped["metadata"] = dict(wrapped.get("metadata") or {}) | {"sovereign_route_exempt": True} | dict(self.metadata)
        return wrapped


_DIGIT_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}

_OPERATOR_SPECS = [
    ("math_operator_addition", "Addition", "plus", "+", "add_program", ("sum", "addition")),
    ("math_operator_subtraction", "Subtraction", "minus", "-", "sub_program", ("subtract", "difference")),
    ("math_operator_multiplication", "Multiplication", "times", "*", "mul_program", ("multiply", "product")),
    ("math_operator_division", "Division", "divided by", "/", "div_program", ("divide", "quotient")),
    ("math_operator_power", "Power", "to the power of", "^", "pow_program", ("power", "exponent")),
]

_PROGRAM_SPECS = [
    ProgramSpec(
        program_id="rpn_program_addition",
        opcode_names=("PUSH_OPERAND_0", "PUSH_OPERAND_1", "ADD", "STORE_RESULT", "RET"),
        bytecode=bytes((0x10, 0x11, 0x20, 0x30, 0xFF)),
        meaning_rpn="OPERAND_0 OPERAND_1 ADD STORE_RESULT RET",
    ),
    ProgramSpec(
        program_id="rpn_program_subtraction",
        opcode_names=("PUSH_OPERAND_0", "PUSH_OPERAND_1", "SUB", "STORE_RESULT", "RET"),
        bytecode=bytes((0x10, 0x11, 0x21, 0x30, 0xFF)),
        meaning_rpn="OPERAND_0 OPERAND_1 SUB STORE_RESULT RET",
    ),
    ProgramSpec(
        program_id="rpn_program_multiplication",
        opcode_names=("PUSH_OPERAND_0", "PUSH_OPERAND_1", "MUL", "STORE_RESULT", "RET"),
        bytecode=bytes((0x10, 0x11, 0x22, 0x30, 0xFF)),
        meaning_rpn="OPERAND_0 OPERAND_1 MUL STORE_RESULT RET",
    ),
    ProgramSpec(
        program_id="rpn_program_division",
        opcode_names=("PUSH_OPERAND_0", "PUSH_OPERAND_1", "DIV", "STORE_RESULT", "RET"),
        bytecode=bytes((0x10, 0x11, 0x23, 0x30, 0xFF)),
        meaning_rpn="OPERAND_0 OPERAND_1 DIV STORE_RESULT RET",
    ),
    ProgramSpec(
        program_id="rpn_program_power",
        opcode_names=("PUSH_OPERAND_0", "PUSH_OPERAND_1", "POW", "STORE_RESULT", "RET"),
        bytecode=bytes((0x10, 0x11, 0x24, 0x30, 0xFF)),
        meaning_rpn="OPERAND_0 OPERAND_1 POW STORE_RESULT RET",
    ),
]

_GRAMMAR_SPECS = [
    ("grammar_binary_op_infix", "Binary Op Infix", "{operand_1} {operator} {operand_2}", "EXPR OP EXPR PACK_OPERANDS"),
    ("grammar_binary_op_infix_eq", "Binary Op Infix Eq", "{operand_1} {operator} {operand_2} = {result}", "EXPR OP EXPR RESULT VERIFY_OR_SOLVE"),
    ("grammar_question_what_is", "Question What Is", "what is {expression}", "WHAT IS EXPR EXTRACT_QUERY"),
    ("grammar_english_binary_op", "English Binary Op", "{word_operand_1} {word_operator} {word_operand_2}", "WORD_OPERAND WORD_OPERATOR WORD_OPERAND PACK_OPERANDS"),
    ("grammar_trailing_question_mark", "Trailing Question Mark", "{expression}?", "EXPR STRIP_QUESTION MARK_QUERY"),
]

_KBO_PRECEDENCE_RANKS = {
    "f": 240,
    "g": 224,
    "h": 208,
    "x": 176,
    "y": 160,
    "z": 144,
    "a": 96,
    "b": 80,
    "c": 64,
    "d": 48,
}


def _pad_program_entry(bytecode: bytes, opcode_count: int) -> bytes:
    header = len(bytecode).to_bytes(2, "little") + int(opcode_count).to_bytes(2, "little")
    payload = header + bytes(bytecode)
    padding = (-len(payload)) % PROGRAM_ENTRY_ALIGN
    if padding:
        payload += b"\x00" * padding
    return payload


def build_default_star_crafter_program_table() -> ProgramTableLayout:
    payload = bytearray(PROGRAM_TABLE_SENTINEL_BYTES)
    offsets: dict[str, int] = {}
    lengths: dict[str, int] = {}
    opcode_counts: dict[str, int] = {}
    for spec in _PROGRAM_SPECS:
        entry = _pad_program_entry(spec.bytecode, len(spec.opcode_names))
        offsets[spec.program_id] = len(payload)
        lengths[spec.program_id] = len(spec.bytecode)
        opcode_counts[spec.program_id] = len(spec.opcode_names)
        payload.extend(entry)
    return ProgramTableLayout(
        payload=bytes(payload),
        offsets=offsets,
        lengths=lengths,
        opcode_counts=opcode_counts,
    )


class StarCrafter:
    """Deterministic foundational star crafter scaffold."""

    def __init__(
        self,
        *,
        provider: Any | None = None,
        program_table: ProgramTableLayout | None = None,
        embedder: SovereignMatryoshkaTextEmbedder | None = None,
    ) -> None:
        self.provider = provider
        self.program_table = program_table or build_default_star_crafter_program_table()
        self._embedder = embedder or get_sovereign_matryoshka_text_embedder()
        self._entries: dict[str, _CraftedEntry] = {}

    def _register(self, entry: _CraftedEntry) -> None:
        entry.embedding_stack = self._embedder.embed_stack(self._semantic_text(entry))
        self._entries[entry.entry_id] = entry

    @property
    def embedder(self) -> SovereignMatryoshkaTextEmbedder:
        return self._embedder

    def _semantic_text(self, entry: _CraftedEntry) -> str:
        metadata = dict(entry.metadata or {})
        parts: list[str] = []
        aliases = metadata.get("aliases")
        if isinstance(aliases, (list, tuple)):
            clean_aliases = [str(item).strip() for item in aliases if str(item).strip()]
            parts.extend(clean_aliases)
            parts.extend(clean_aliases)
        elif str(aliases or "").strip():
            parts.extend([str(aliases).strip(), str(aliases).strip()])
        for key in ("keywords",):
            value = metadata.get(key)
            if isinstance(value, (list, tuple)):
                parts.extend(str(item).strip() for item in value if str(item).strip())
            elif str(value or "").strip():
                parts.append(str(value).strip())
        for value in (
            entry.name,
            entry.entry_id,
            entry.meaning_star.domain,
            entry.meaning_star.meaning_rpn,
            metadata.get("description"),
            entry.meaning_star.behavior_rpn,
        ):
            text = str(value or "").strip()
            if text:
                parts.append(text)
        return " ".join(parts)

    def _meaning_star(
        self,
        *,
        meaning_class: str,
        meaning_rpn: str,
        domain: str,
        taxonomy_refs: list[str] | None = None,
        meta_refs: list[str] | None = None,
        grammar_refs: list[str] | None = None,
        component_refs: list[str] | None = None,
        composite_of: list[str] | None = None,
        surface_forms: dict[str, SurfaceForm] | None = None,
        behavior_rpn: str | None = None,
    ) -> MeaningCentricStar:
        return MeaningCentricStar(
            star_id=compute_star_id(meaning_rpn, meaning_class, domain),
            meaning_class=meaning_class,
            meaning_rpn=meaning_rpn,
            domain=domain,
            taxonomy_refs=list(taxonomy_refs or []),
            meta_refs=list(meta_refs or []),
            grammar_refs=list(grammar_refs or []),
            component_refs=list(component_refs or []),
            composite_of=list(composite_of or []),
            surface_forms=dict(surface_forms or {}),
            behavior_rpn=behavior_rpn,
        )

    def _append_ref(self, entry_id: str, field_path: str, ref_id: str) -> None:
        entry = self._entries[entry_id]
        star = entry.meaning_star
        if field_path == "taxonomy_refs":
            if ref_id not in star.taxonomy_refs:
                star.taxonomy_refs.append(ref_id)
            return
        if field_path == "meta_refs":
            if ref_id not in star.meta_refs:
                star.meta_refs.append(ref_id)
            return
        if field_path == "grammar_refs":
            if ref_id not in star.grammar_refs:
                star.grammar_refs.append(ref_id)
            return
        if field_path == "component_refs":
            if ref_id not in star.component_refs:
                star.component_refs.append(ref_id)
            return
        if field_path == "composite_of":
            if ref_id not in star.composite_of:
                star.composite_of.append(ref_id)
            return
        if field_path.startswith("surface_forms.") and field_path.endswith(".word_ref"):
            _, language, _ = field_path.split(".", 2)
            form = star.surface_forms.setdefault(language, SurfaceForm(word_ref="", char_refs=[]))
            form.word_ref = ref_id
            return
        if field_path.startswith("surface_forms.") and field_path.endswith(".char_refs"):
            _, language, _ = field_path.split(".", 2)
            form = star.surface_forms.setdefault(language, SurfaceForm(word_ref="", char_refs=[]))
            if ref_id not in form.char_refs:
                form.char_refs.append(ref_id)
            return
        raise ValueError(f"unsupported_star_crafter_link_field:{field_path}")

    def _link(self, left_id: str, right_id: str, forward_kind: str, backward_kind: str) -> None:
        self._append_ref(left_id, forward_kind, right_id)
        self._append_ref(right_id, backward_kind, left_id)

    def craft_digit_stars(self) -> None:
        for value, word in _DIGIT_WORDS.items():
            digit_id = f"concept_digit_{word}"
            word_id = f"word_digit_{word}_en"
            char_id = f"char_digit_{value}"
            star = self._meaning_star(
                meaning_class="concept",
                meaning_rpn=f"{value} INTEGER DIGIT",
                domain="Math/Number/Integer/Digit",
                taxonomy_refs=["concept_integer", "concept_number"],
                grammar_refs=["grammar_binary_op_infix", "grammar_binary_op_infix_eq", "grammar_question_what_is"],
                surface_forms={"en": SurfaceForm(word_ref=word_id, char_refs=[char_id])},
            )
            self._register(
                _CraftedEntry(
                    entry_id=digit_id,
                    galaxy="Math",
                    name=f"Digit {value}",
                    category="meaning_digit",
                    star_type=STAR_TYPE_MATH,
                    selection_role=ROLE_ANSWER,
                    answer_eligible=True,
                    layer_id=2,
                    route_family="MATH",
                    route_policy={"branch_topk": 0},
                    meaning_star=star,
                    metadata={
                        "description": f"concept digit {value} numeric value {word} integer number",
                        "aliases": [word, str(value), word, str(value), f"digit {value}", f"number {word}"],
                        "keywords": ["digit", "integer", "number", str(value), word],
                    },
                )
            )
            form_star = self._meaning_star(
                meaning_class="concept",
                meaning_rpn=f"{word.upper()} NUMERIC_WORD",
                domain="Word/English/NumberWord",
                taxonomy_refs=[digit_id],
                surface_forms={"en": SurfaceForm(word_ref=word_id, char_refs=[char_id])},
            )
            self._register(
                _CraftedEntry(
                    entry_id=word_id,
                    galaxy="Word",
                    name=word,
                    category="meaning_number_word",
                    star_type=STAR_TYPE_CHARACTER,
                    selection_role=ROLE_UNKNOWN,
                    answer_eligible=False,
                    layer_id=1,
                    route_family="",
                    route_policy={},
                    meaning_star=form_star,
                    metadata={
                        "sovereign_route_exempt": True,
                        "description": f"English number word {word} symlink for digit {value}",
                        "aliases": [word, str(value), f"number {word}", f"digit {value}"],
                        "keywords": ["word", "number", "digit", word, str(value)],
                    },
                )
            )
            char_star = self._meaning_star(
                meaning_class="form",
                meaning_rpn=f"{value} ASCII_DIGIT_FORM",
                domain="Word/Character/ASCII/Digit",
                taxonomy_refs=[digit_id],
                surface_forms={"en": SurfaceForm(word_ref=char_id, char_refs=[char_id])},
            )
            self._register(
                _CraftedEntry(
                    entry_id=char_id,
                    galaxy="Word",
                    name=str(value),
                    category="meaning_digit_char",
                    star_type=STAR_TYPE_CHARACTER,
                    selection_role=ROLE_UNKNOWN,
                    answer_eligible=False,
                    layer_id=1,
                    route_family="",
                    route_policy={},
                    meaning_star=char_star,
                    metadata={
                        "sovereign_route_exempt": True,
                        "description": f"ASCII digit character {value} symlink for numeric digit {word}",
                        "aliases": [str(value), word, f"digit {value}"],
                        "keywords": ["character", "digit", str(value), word],
                    },
                )
            )
            self._link(digit_id, word_id, "surface_forms.en.word_ref", "taxonomy_refs")
            self._link(digit_id, char_id, "surface_forms.en.char_refs", "taxonomy_refs")

    def craft_rpn_program_stars(self) -> None:
        for spec in _PROGRAM_SPECS:
            star = self._meaning_star(
                meaning_class="meta",
                meaning_rpn=spec.meaning_rpn,
                domain="Programs/Math/Arithmetic",
                taxonomy_refs=["concept_arithmetic_operator", "concept_binary_operation"],
            )
            self._register(
                _CraftedEntry(
                    entry_id=spec.program_id,
                    galaxy="Math",
                    name=spec.program_id.replace("rpn_program_", "").replace("_", " "),
                    category="meaning_rpn_program",
                    star_type=STAR_TYPE_MATH,
                    selection_role=ROLE_EXECUTOR,
                    answer_eligible=False,
                    layer_id=4,
                    route_family="MATH",
                    route_policy={"requires_validator": False, "branch_topk": 0},
                    meaning_star=star,
                    metadata={
                        "description": f"RPN bytecode program for {spec.program_id.replace('rpn_program_', '').replace('_', ' ')} arithmetic execution",
                        "aliases": [
                            spec.program_id.replace("rpn_program_", "").replace("_", " "),
                            *[name.lower() for name in spec.opcode_names],
                        ],
                        "keywords": ["rpn", "program", "bytecode", "arithmetic", *[name.lower() for name in spec.opcode_names]],
                    },
                    extra_top_level={
                        "meta_rule_addr": int(self.program_table.offsets[spec.program_id]),
                        "program_flags": 0x01,
                        "program_length": int(self.program_table.lengths[spec.program_id]),
                        "program_opcode_count": int(self.program_table.opcode_counts[spec.program_id]),
                    },
                )
            )

    def craft_operator_stars(self) -> None:
        program_by_alias = {
            "add_program": "rpn_program_addition",
            "sub_program": "rpn_program_subtraction",
            "mul_program": "rpn_program_multiplication",
            "div_program": "rpn_program_division",
            "pow_program": "rpn_program_power",
        }
        for entry_id, name, word, symbol, program_alias, extra_terms in _OPERATOR_SPECS:
            program_id = program_by_alias[program_alias]
            word_id = f"word_operator_{word.replace(' ', '_')}_en"
            escaped_symbol = symbol.encode("unicode_escape").decode("ascii").replace("\\", "_")
            symbol_id = f"char_operator_{escaped_symbol}"
            meaning_rpn = f"{name.upper()} BINARY_OPERATOR {symbol}"
            star = self._meaning_star(
                meaning_class="action",
                meaning_rpn=meaning_rpn,
                domain="Math/Operator/Arithmetic",
                taxonomy_refs=["concept_arithmetic_operator", "concept_binary_operation"],
                meta_refs=[program_id],
                grammar_refs=[rule_id for rule_id, _, _, _ in _GRAMMAR_SPECS],
                surface_forms={"en": SurfaceForm(word_ref=word_id, char_refs=[symbol_id])},
                behavior_rpn=meaning_rpn,
            )
            self._register(
                _CraftedEntry(
                    entry_id=entry_id,
                    galaxy="Math",
                    name=name,
                    category="meaning_operator",
                    star_type=STAR_TYPE_MATH,
                    selection_role=ROLE_EXECUTOR,
                    answer_eligible=True,
                    layer_id=2,
                    route_family="MATH",
                    route_policy={"requires_validator": False, "branch_topk": 0},
                    meaning_star=star,
                    metadata={
                        "description": (
                            f"{name} arithmetic operator: {word}, symbol {symbol}, "
                            f"terms {' '.join(extra_terms)}; executes {program_id}"
                        ),
                        "aliases": [word, name.lower(), word, name.lower(), *list(extra_terms), symbol, program_id],
                        "keywords": ["operator", "arithmetic", "binary", name.lower(), word, symbol, *list(extra_terms)],
                    },
                    extra_top_level={
                        "meta_rule_addr": int(self.program_table.offsets[program_id]),
                        "program_flags": 0x01,
                        "program_length": int(self.program_table.lengths[program_id]),
                        "program_opcode_count": int(self.program_table.opcode_counts[program_id]),
                    },
                )
            )
            word_star = self._meaning_star(
                meaning_class="concept",
                meaning_rpn=f"{word.upper()} OPERATOR_WORD",
                domain="Word/English/OperatorWord",
                taxonomy_refs=[entry_id],
                surface_forms={"en": SurfaceForm(word_ref=word_id, char_refs=[symbol_id])},
            )
            self._register(
                _CraftedEntry(
                    entry_id=word_id,
                    galaxy="Word",
                    name=word,
                    category="meaning_operator_word",
                    star_type=STAR_TYPE_CHARACTER,
                    selection_role=ROLE_UNKNOWN,
                    answer_eligible=False,
                    layer_id=1,
                    route_family="",
                    route_policy={},
                    meaning_star=word_star,
                    metadata={
                        "sovereign_route_exempt": True,
                        "description": f"English operator word {word} symlink for {name.lower()} arithmetic",
                        "aliases": [word, name.lower(), symbol, *list(extra_terms)],
                        "keywords": ["word", "operator", "arithmetic", word, name.lower(), symbol, *list(extra_terms)],
                    },
                )
            )
            symbol_star = self._meaning_star(
                meaning_class="form",
                meaning_rpn=f"{symbol} OPERATOR_SYMBOL",
                domain="Word/Character/ASCII/Operator",
                taxonomy_refs=[entry_id],
                surface_forms={"en": SurfaceForm(word_ref=symbol_id, char_refs=[symbol_id])},
            )
            self._register(
                _CraftedEntry(
                    entry_id=symbol_id,
                    galaxy="Word",
                    name=symbol,
                    category="meaning_operator_symbol",
                    star_type=STAR_TYPE_CHARACTER,
                    selection_role=ROLE_UNKNOWN,
                    answer_eligible=False,
                    layer_id=1,
                    route_family="",
                    route_policy={},
                    meaning_star=symbol_star,
                    metadata={
                        "sovereign_route_exempt": True,
                        "description": f"Operator symbol {symbol} symlink for {name.lower()} arithmetic",
                        "aliases": [symbol, word, name.lower(), *list(extra_terms)],
                        "keywords": ["symbol", "operator", "arithmetic", symbol, word, name.lower(), *list(extra_terms)],
                    },
                )
            )
            self._link(entry_id, word_id, "surface_forms.en.word_ref", "taxonomy_refs")
            self._link(entry_id, symbol_id, "surface_forms.en.char_refs", "taxonomy_refs")
            self._link(entry_id, program_id, "meta_refs", "taxonomy_refs")

    def craft_grammar_rule_stars(self) -> None:
        concept_targets = [f"concept_digit_{word}" for word in _DIGIT_WORDS.values()]
        concept_targets.extend(entry_id for entry_id, *_ in _OPERATOR_SPECS)
        for rule_id, name, pattern, meaning_rpn in _GRAMMAR_SPECS:
            star = self._meaning_star(
                meaning_class="meta",
                meaning_rpn=meaning_rpn,
                domain="Grammar/Math/Arithmetic",
                taxonomy_refs=["concept_binary_operation"],
                grammar_refs=list(concept_targets),
            )
            self._register(
                _CraftedEntry(
                    entry_id=rule_id,
                    galaxy="Grammar",
                    name=name,
                    category="meaning_grammar_rule",
                    star_type=STAR_TYPE_GRAMMAR,
                    selection_role=ROLE_ROUTER,
                    answer_eligible=False,
                    layer_id=3,
                    route_family="GRAMMAR",
                    route_policy={
                        "requires_executor": True,
                        "requires_validator": False,
                        "answer_gate": False,
                        "branch_topk": 2,
                    },
                    meaning_star=star,
                    metadata={
                        "description": f"Grammar rule {name}: {pattern}; {meaning_rpn}",
                        "aliases": [name.lower(), pattern, meaning_rpn.lower()],
                        "keywords": ["grammar", "math", "parse", "infix", "operator", "operand"],
                    },
                )
            )
            for target in concept_targets:
                self._link(rule_id, target, "grammar_refs", "taxonomy_refs")

    def craft_swarm_perf_calibration_star(self) -> None:
        star = self._meaning_star(
            meaning_class="meta",
            meaning_rpn="SWARM PERF CALIBRATION N_HINT SAMPLE_COUNT PEAK_UTILITY",
            domain="Meta/Swarm/Performance",
            taxonomy_refs=["concept_binary_operation"],
        )
        self._register(
            _CraftedEntry(
                entry_id="swarm_perf_calibration",
                galaxy="Meta",
                name="Swarm Perf Calibration",
                category="meaning_swarm_perf_calibration",
                star_type=STAR_TYPE_GRAMMAR,
                selection_role=ROLE_UNKNOWN,
                answer_eligible=False,
                layer_id=4,
                route_family="META",
                route_policy={"requires_validator": False, "branch_topk": 0},
                meaning_star=star,
                metadata={
                    "description": "Sleep-time sovereign calibration surface for adaptive N-chain worker selection.",
                    "aliases": ["swarm perf calibration", "adaptive n calibration", "lane perf calibration"],
                    "keywords": ["swarm", "performance", "calibration", "adaptive", "n-chain", "sleep"],
                    "calibration_schema": {
                        "n_hint": "u32",
                        "sample_count_total": "u32",
                        "last_tick_epoch": "u32",
                        "utility_peak_q20": "u32",
                    },
                },
                extra_top_level={
                    "program_flags": 0x00,
                    "program_length": 0,
                    "program_opcode_count": 0,
                },
            )
        )

    def craft_reasoning_kbo_precedence_star(self) -> None:
        star = self._meaning_star(
            meaning_class="meta",
            meaning_rpn="KBO PRECEDENCE F G H X Y Z A B C D",
            domain="Meta/Reasoning/Ordering",
            taxonomy_refs=["concept_binary_operation"],
        )
        self._register(
            _CraftedEntry(
                entry_id="reasoning_kbo_precedence",
                galaxy="Meta",
                name="Reasoning KBO Precedence",
                category="meaning_reasoning_kbo_precedence",
                star_type=STAR_TYPE_GRAMMAR,
                selection_role=ROLE_UNKNOWN,
                answer_eligible=False,
                layer_id=4,
                route_family="META",
                route_policy={"requires_validator": False, "branch_topk": 0},
                meaning_star=star,
                metadata={
                    "description": "Canonical precedence source for bounded Batch 3 KBO ordering and ordered rewriting.",
                    "aliases": ["kbo precedence", "reasoning precedence", "ordered rewrite precedence"],
                    "keywords": ["reasoning", "kbo", "precedence", "rewrite", "superposition", "ordering"],
                    "precedence_ranks": dict(_KBO_PRECEDENCE_RANKS),
                    "symbol_ids": {
                        "f": 1,
                        "g": 2,
                        "h": 3,
                        "x": 4,
                        "y": 5,
                        "z": 6,
                        "a": 7,
                        "b": 8,
                        "c": 9,
                        "d": 10,
                    },
                },
                extra_top_level={
                    "program_flags": 0x00,
                    "program_length": 0,
                    "program_opcode_count": 0,
                },
            )
        )

    def craft_all(self) -> list[dict[str, Any]]:
        self.craft_digit_stars()
        self.craft_rpn_program_stars()
        self.craft_operator_stars()
        self.craft_grammar_rule_stars()
        self.craft_swarm_perf_calibration_star()
        self.craft_reasoning_kbo_precedence_star()
        return [self._entries[key].to_row() for key in sorted(self._entries)]


def build_foundational_star_crafter_outputs(
    *,
    provider: Any | None = None,
    program_table: ProgramTableLayout | None = None,
    embedder: SovereignMatryoshkaTextEmbedder | None = None,
) -> list[dict[str, Any]]:
    crafter = StarCrafter(provider=provider, program_table=program_table, embedder=embedder)
    return crafter.craft_all()


__all__ = [
    "ProgramSpec",
    "ProgramTableLayout",
    "StarCraftPlan",
    "StarCraftSurfaceDraft",
    "StarCrafter",
    "build_default_star_crafter_program_table",
    "build_foundational_star_crafter_outputs",
]
