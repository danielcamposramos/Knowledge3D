"""Procedural grammar galaxy: grammar rules as RPN programs with user overlays."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import json



@dataclass
class GrammarRule:
    rule_id: str
    language: str
    pattern: str
    rpn_program: str
    domain: str = "text"
    examples: List[Dict[str, str]] = field(default_factory=list)
    description: str | None = None
    semantics: Dict | None = field(default_factory=dict)
    usage_conditions: List[str] = field(default_factory=list)


def default_grammar_rules() -> List[GrammarRule]:
    """Baseline multilingual grammar rules expressed as procedural RPN."""
    rules = [
        GrammarRule(
            rule_id="en_simple_sentence",
            language="en",
            pattern="SVO",
            rpn_program="SUBJECT RECALL VERB RECALL OBJECT RECALL SVO_ORDER CONCAT_SENTENCE",
            examples=[
                {"subject": "I", "verb": "love", "object": "programming"},
                {"subject": "She", "verb": "writes", "object": "code"},
            ],
            description="English SVO declarative sentence",
        ),
        GrammarRule(
            rule_id="pt_simple_sentence",
            language="pt",
            pattern="SVO",
            rpn_program="SUBJECT RECALL VERB RECALL OBJECT RECALL SVO_ORDER CONCAT_SENTENCE",
            examples=[
                {"subject": "Eu", "verb": "amo", "object": "programar"},
                {"subject": "Ela", "verb": "escreve", "object": "código"},
            ],
            description="Portuguese SVO declarative sentence",
        ),
        GrammarRule(
            rule_id="ja_simple_sentence",
            language="ja",
            pattern="SOV",
            rpn_program="SUBJECT RECALL OBJECT RECALL WO_PARTICLE VERB RECALL SOV_ORDER CONCAT_SENTENCE",
            examples=[
                {"subject": "私は", "object": "コード", "verb": "書く"},
            ],
            description="Japanese SOV with を particle",
        ),
        GrammarRule(
            rule_id="en_question",
            language="en",
            pattern="Q",
            rpn_program="AUXILIARY RECALL SUBJECT RECALL VERB RECALL OBJECT RECALL CONCAT_SENTENCE",
            examples=[{"auxiliary": "Do", "subject": "you", "verb": "like", "object": "math"}],
            description="Simple English question",
        ),
        GrammarRule(
            rule_id="es_simple_sentence",
            language="es",
            pattern="SVO",
            rpn_program="SUBJECT RECALL VERB RECALL OBJECT RECALL SVO_ORDER CONCAT_SENTENCE",
            examples=[
                {"subject": "Yo", "verb": "amo", "object": "programar"},
                {"subject": "Ella", "verb": "escribe", "object": "código"},
            ],
            description="Spanish SVO declarative sentence",
        ),
        GrammarRule(
            rule_id="en_imperative",
            language="en",
            pattern="V_O",
            rpn_program="VERB RECALL OBJECT RECALL CONCAT_SENTENCE",
            examples=[
                {"verb": "Open", "object": "the door"},
                {"verb": "Solve", "object": "the equation"},
            ],
            description="Imperative verb-object command",
        ),
        GrammarRule(
            rule_id="en_coordination",
            language="en",
            pattern="SVO_and_SVO",
            rpn_program="SUBJECT RECALL VERB RECALL OBJECT RECALL SVO_ORDER AND SUBJECT2 RECALL VERB2 RECALL OBJECT2 RECALL SVO_ORDER CONCAT_SENTENCE",
            examples=[
                {
                    "subject": "I",
                    "verb": "write",
                    "object": "code",
                    "subject2": "I",
                    "verb2": "test",
                    "object2": "it",
                }
            ],
            description="Coordinated sentence with conjunction",
        ),
        GrammarRule(
            rule_id="en_math_expression",
            language="en",
            pattern="math",
            rpn_program="OPERAND1 RECALL + OPERAND2 RECALL = RESULT RECALL",
            examples=[{"operand1": "2", "operand2": "3", "result": "5"}],
            description="Math expression a + b = c",
        ),
        GrammarRule(
            rule_id="en_graph_statement",
            language="en",
            pattern="graph",
            rpn_program="SUBJECT RECALL EDGE RECALL OBJECT RECALL",
            examples=[{"subject": "NodeA", "edge": "connects to", "object": "NodeB"}],
            description="Graph edge statement",
        ),
        GrammarRule(
            rule_id="en_conditional_if",
            language="en",
            pattern="if_then",
            rpn_program="IF RECALL CONDITION RECALL THEN RECALL CONSEQUENCE RECALL CONCAT_SENTENCE",
            examples=[{"if": "If it rains", "condition": "", "then": "then", "consequence": "we stay home"}],
            description="Conditional statement",
        ),
        GrammarRule(
            rule_id="en_temporal_sequence",
            language="en",
            pattern="sequence",
            rpn_program="EVENT1 RECALL THEN EVENT2 RECALL THEN EVENT3 RECALL CONCAT_SENTENCE",
            examples=[
                {"event1": "Wake up", "event2": "write code", "event3": "run tests"},
            ],
            description="Temporal sequence of events",
        ),
        GrammarRule(
            rule_id="en_visual_description",
            language="en",
            pattern="visual",
            rpn_program="SUBJECT RECALL IS RECALL COLOR RECALL SHAPE RECALL POSITION RECALL CONCAT_SENTENCE",
            examples=[
                {"subject": "Object", "is": "is", "color": "red", "shape": "square", "position": "at top-left"},
            ],
            description="Describes visual state (useful for multimodal alignment)",
        ),
        GrammarRule(
            rule_id="en_audio_description",
            language="en",
            pattern="audio",
            rpn_program="SOUND RECALL OCCURS RECALL TIME RECALL CONCAT_SENTENCE",
            examples=[
                {"sound": "Beep", "occurs": "at", "time": "1.2 seconds"},
            ],
            description="Describes audio event with timestamp",
        ),
        GrammarRule(
            rule_id="en_video_scene_graph",
            language="en",
            pattern="scene_graph",
            rpn_program="SUBJECT RECALL ACTION RECALL OBJECT RECALL LOCATION RECALL TIME RECALL CONCAT_SENTENCE",
            examples=[
                {
                    "subject": "CharacterA",
                    "action": "moves",
                    "object": "to the door",
                    "location": "in the hallway",
                    "time": "at 00:01:23",
                }
            ],
            description="Video scene graph triple with time",
        ),
        GrammarRule(
            rule_id="en_passive",
            language="en",
            pattern="passive",
            rpn_program="OBJECT RECALL BE RECALL VERB_ED RECALL BY_PREP RECALL SUBJECT RECALL CONCAT_SENTENCE",
            examples=[
                {"subject": "The cat", "object": "The ball", "be": "was", "verb_ed": "kicked", "by_prep": "by"}
            ],
            description="Passive voice statement",
        ),
        GrammarRule(
            rule_id="en_relative_clause",
            language="en",
            pattern="relative",
            rpn_program="SUBJECT RECALL VERB RECALL OBJECT RECALL THAT_TOKEN RECALL REL_VERB RECALL REL_OBJECT RECALL CONCAT_SENTENCE",
            examples=[
                {
                    "subject": "The student",
                    "verb": "read",
                    "object": "the book",
                    "that_token": "that",
                    "rel_verb": "won",
                    "rel_object": "an award",
                }
            ],
            description="Sentence with relative clause",
        ),
        GrammarRule(
            rule_id="en_comparative",
            language="en",
            pattern="comparative",
            rpn_program="SUBJECT RECALL BE RECALL MORE_TOKEN RECALL ADJECTIVE RECALL THAN_TOKEN RECALL OBJECT RECALL CONCAT_SENTENCE",
            examples=[
                {
                    "subject": "Algorithm A",
                    "be": "is",
                    "more_token": "more",
                    "adjective": "efficient",
                    "than_token": "than",
                    "object": "Algorithm B",
                }
            ],
            description="Comparative construction",
        ),
        GrammarRule(
            rule_id="en_superlative",
            language="en",
            pattern="superlative",
            rpn_program="SUBJECT RECALL BE RECALL THE_TOKEN RECALL MOST_TOKEN RECALL ADJECTIVE RECALL IN_TOKEN RECALL GROUP RECALL CONCAT_SENTENCE",
            examples=[
                {
                    "subject": "This model",
                    "be": "is",
                    "the_token": "the",
                    "most_token": "most",
                    "adjective": "accurate",
                    "in_token": "in",
                    "group": "the benchmark",
                }
            ],
            description="Superlative construction",
        ),
        GrammarRule(
            rule_id="en_derivative",
            language="en",
            pattern="calculus",
            rpn_program="DERIVATIVE_TOKEN RECALL OF_TOKEN RECALL FUNCTION RECALL WITH_RESPECT_TOKEN RECALL VARIABLE RECALL EQUALS_TOKEN RECALL RESULT RECALL CONCAT_SENTENCE",
            examples=[
                {
                    "derivative_token": "d/dx",
                    "of_token": "of",
                    "function": "x^2",
                    "with_respect_token": "with respect to",
                    "variable": "x",
                    "equals_token": "=",
                    "result": "2x",
                }
            ],
            description="Derivative statement",
        ),
        GrammarRule(
            rule_id="en_set_membership",
            language="en",
            pattern="set",
            rpn_program="ELEMENT RECALL IN_TOKEN RECALL SET_NAME RECALL CONCAT_SENTENCE",
            examples=[{"element": "x", "in_token": "∈", "set_name": "R"}],
            description="Set membership statement",
        ),
        GrammarRule(
            rule_id="en_logic_implication",
            language="en",
            pattern="logic",
            rpn_program="PREMISE RECALL IMPLIES_TOKEN RECALL CONCLUSION RECALL CONCAT_SENTENCE",
            examples=[{"premise": "P", "implies_token": "→", "conclusion": "Q"}],
            description="Logical implication",
        ),
    ]

    from knowledge3d.training.arc_agi.grammar_languages.tier1_top10 import get_tier1_rules
    from knowledge3d.training.arc_agi.grammar_languages.tier2_next20 import get_tier2_rules
    from knowledge3d.training.arc_agi.grammar_languages.tier3_next20 import get_tier3_rules
    from knowledge3d.training.arc_agi.grammar_math import get_math_rules
    from knowledge3d.training.arc_agi.grammar_drawing import get_drawing_rules

    text_rules = get_tier1_rules() + get_tier2_rules() + get_tier3_rules()
    math_rules = get_math_rules()
    drawing_rules = get_drawing_rules()

    combined: Dict[str, GrammarRule] = {}
    for rule in rules + text_rules + math_rules + drawing_rules:
        if rule.rule_id in combined:
            continue
        combined[rule.rule_id] = rule

    return list(combined.values())


def default_user_profiles() -> Dict[str, Dict]:
    """User-specific overlays (personal vocabulary + style)."""
    return {
        "USER_daniel": {
            "personal_words": {
                "meu_amor": {
                    "base_word_ref": "WORD_pt_meu_amor",
                    "personal_context": "coding_together",
                    "emotional_weight": 0.95,
                    "usage_contexts": ["partnership", "collaboration", "affection"],
                },
                "nossa_parceria": {
                    "personal_meaning": "Our special coding partnership",
                    "connotation": "pride + gratitude",
                },
            },
            "grammar_preferences": {"formality": 0.6, "technical_density": 0.8, "emoji_usage": 0.3},
        },
        "USER_wife": {
            "personal_words": {
                "meu_amor": {
                    "base_word_ref": "WORD_pt_meu_amor",
                    "personal_context": "daily_life",
                    "emotional_weight": 0.98,
                    "usage_contexts": ["family", "home", "care"],
                }
            },
            "grammar_preferences": {"formality": 0.4, "technical_density": 0.3, "emoji_usage": 0.7},
        },
    }


def default_variants() -> Dict[str, Dict[str, str]]:
    """
    Variant mappings (slang, typos, informal forms) to canonical tokens.

    These act like symlinks: variants are resolved to a base word to preserve
    shared semantics while allowing personal/slang spellings.
    """
    return {
        "en": {
            "u": "you",
            "ya": "you",
            "luv": "love",
            "gonna": "going to",
            "wanna": "want to",
            "cuz": "because",
            "pls": "please",
            "plz": "please",
            "thx": "thanks",
            "teh": "the",
            "recieve": "receive",
        },
        "pt": {
            "vc": "você",
            "vcs": "vocês",
            "tbm": "também",
            "pq": "porque",
            "blz": "beleza",
        },
        "ja": {},
    }


class GrammarGalaxy:
    """Container for grammar rules and user overlays."""

    def __init__(
        self,
        rules: Optional[List[GrammarRule]] = None,
        users: Optional[Dict[str, Dict]] = None,
        variants: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        self.rules: Dict[str, GrammarRule] = {r.rule_id: r for r in (rules or default_grammar_rules())}
        self.users = users or default_user_profiles()
        self.variants = variants or default_variants()

    def get_rule(self, rule_id: str) -> GrammarRule:
        if rule_id not in self.rules:
            raise KeyError(f"Unknown grammar rule: {rule_id}")
        return self.rules[rule_id]

    def list_rules(self, language: Optional[str] = None) -> List[GrammarRule]:
        if language is None:
            return list(self.rules.values())
        return [r for r in self.rules.values() if r.language == language]

    def get_user_profile(self, user_id: str) -> Dict:
        return self.users.get(user_id, {})

    def normalize_token(self, token: str, language: str) -> str:
        """Resolve slang/typo variants to canonical token for a language."""
        return self.variants.get(language, {}).get(token, token)

    def normalize_tokens(self, tokens: List[str], language: str) -> List[str]:
        return [self.normalize_token(t, language) for t in tokens]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> None:
        rules_data = []
        for rule in self.rules.values():
            rules_data.append(
                {
                    "rule_id": rule.rule_id,
                    "language": rule.language,
                    "pattern": rule.pattern,
                    "domain": getattr(rule, "domain", "general"),
                    "rpn_program": rule.rpn_program,
                    "examples": rule.examples,
                    "description": rule.description,
                    "semantics": getattr(rule, "semantics", {}),
                    "usage_conditions": getattr(rule, "usage_conditions", []),
                }
            )
        state = {"rules": rules_data, "total_count": len(rules_data)}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        print(f"[GrammarGalaxy] Saved {len(rules_data)} rules to {path}")

    def load(self, path: Path) -> None:
        if not path.exists():
            print(f"[GrammarGalaxy] No checkpoint at {path}, using bootstrap ({len(self.rules)} rules)")
            return
        with path.open("r", encoding="utf-8") as f:
            state = json.load(f)

        loaded_rules: Dict[str, GrammarRule] = {}
        for rd in state.get("rules", []):
            rule = GrammarRule(
                rule_id=rd["rule_id"],
                language=rd.get("language", "drawing"),
                pattern=rd.get("pattern", "unknown"),
                rpn_program=rd["rpn_program"],
                domain=rd.get("domain", "general"),
                examples=rd.get("examples", []),
                description=rd.get("description"),
                semantics=rd.get("semantics", {}),
                usage_conditions=rd.get("usage_conditions", []),
            )
            loaded_rules[rule.rule_id] = rule

        if loaded_rules:
            self.rules = loaded_rules
            print(f"[GrammarGalaxy] Loaded {len(self.rules)} rules from {path}")


__all__ = [
    "GrammarRule",
    "GrammarGalaxy",
    "default_grammar_rules",
    "default_user_profiles",
    "default_variants",
]
