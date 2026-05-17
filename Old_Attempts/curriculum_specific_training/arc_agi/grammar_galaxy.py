"""Procedural grammar galaxy: grammar rules as RPN programs with user overlays."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

from knowledge3d.cranium.math_galaxy import get_math_galaxy

# Sovereign successor to the numpy-native host bridge archived on 2026-04-18
# lives at ``knowledge3d.cranium.bridges.cosine_similarity_bridge``. We import
# lazily so that non-GPU contexts (pure CPU tooling, tests) can still load this
# module — the bridge is only constructed when ``_get_cosine_bridge`` is hit.
CosineSimilarityBridge = None  # type: ignore[assignment]


@dataclass
class GrammarRule:
    rule_id: str
    language: str
    pattern: str
    rpn_program: str
    domain: str = "text"
    # Symlink references (canonical knowledge, no duplication)
    symbol_refs: List[int] = field(default_factory=list)  # Math Galaxy codepoints
    word_refs: List[str] = field(default_factory=list)    # Word/lexicon IDs
    examples: List[Dict[str, str]] = field(default_factory=list)
    description: str | None = None
    semantics: Dict | None = field(default_factory=dict)
    usage_conditions: List[str] = field(default_factory=list)
    is_canonical: bool = False

    def validate_symbol_refs(self) -> bool:
        """Ensure symbol_refs all exist in Math Galaxy."""
        math_galaxy = get_math_galaxy()
        return all(math_galaxy.get(cp) is not None for cp in self.symbol_refs)


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
        storage_path: Optional[Path] = None,
        snapshot: Optional[bytes] = None,
    ):
        self.storage_path = Path(storage_path) if storage_path else Path("/K3D/Knowledge3D.local/galaxies/grammar")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.rules: Dict[str, GrammarRule] = {r.rule_id: r for r in (rules or default_grammar_rules())}
        # Load persisted state only when bootstrap rules are used and no snapshot was provided
        if snapshot is None and rules is None and self._rules_file().exists():
            self.load(self._rules_file())
        # Snapshot overrides persisted state (no file I/O)
        if snapshot is not None:
            self._load_from_snapshot(snapshot)
        self.users = users or default_user_profiles()
        self.variants = variants or default_variants()
        # Lazy-init GPU bridge: many call sites (tests, CPU tooling) only need the
        # word-sequence matcher and should not require CUDA availability.
        self.cosine_bridge = None
        self._local_discoveries: Dict[str, Dict] = {}
        self._discovery_threshold = 0.6
        self._promotion_threshold = 0.7
        self._min_usage_for_promotion = 3
        self._current_epoch = 0

    def _get_cosine_bridge(self):
        if self.cosine_bridge is None:
            from knowledge3d.cranium.bridges.cosine_similarity_bridge import (
                CosineSimilarityBridge as _SovereignCosineBridge,
            )
            self.cosine_bridge = _SovereignCosineBridge()
        return self.cosine_bridge

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

    def has_rule(self, rule_id: str) -> bool:
        return rule_id in self.rules

    def _rules_file(self) -> Path:
        return self.storage_path / "grammar_galaxy.json"

    def add_rule(self, rule: GrammarRule, persist: bool = True) -> bool:
        """
        Add a grammar rule after validating symlink integrity.

        Returns False when the rule_id already exists to avoid duplication.
        """
        if not rule.validate_symbol_refs():
            raise ValueError(f"Invalid symbol_refs for rule {rule.rule_id}")
        if rule.rule_id in self.rules:
            return False

        self.rules[rule.rule_id] = rule
        if persist:
            try:
                self.save(self._rules_file())
            except Exception as exc:  # Persistence failures should surface for debugging
                print(f"[GrammarGalaxy] Failed to persist {rule.rule_id}: {exc}")
        return True

    # ------------------------------------------------------------------ #
    # Cross-modality discovery + promotion
    # ------------------------------------------------------------------ #
    def observe_pattern(
        self,
        visual_embedding: List[float],
        text_embedding: List[float],
        context: str,
    ) -> Optional[str]:
        """
        Observe cross-modal correlation. If strong, synthesize and propose a tentative rule.
        """
        if not visual_embedding or not text_embedding:
            return None

        scores = self._get_cosine_bridge().compute_similarities([visual_embedding], text_embedding)
        correlation = scores[0] if scores else 0.0

        if correlation >= self._discovery_threshold:
            rule_rpn = self._synthesize_rule_rpn(visual_embedding, text_embedding)
            return self.propose_rule(rule_rpn, context, correlation)
        return None

    def _synthesize_rule_rpn(
        self,
        visual_emb: List[float],
        text_emb: List[float],
    ) -> str:
        """Synthesize an RPN description of a cross-modal mapping."""
        vis_top = sorted(enumerate(visual_emb), key=lambda x: abs(x[1]), reverse=True)[:8]
        txt_top = sorted(enumerate(text_emb), key=lambda x: abs(x[1]), reverse=True)[:8]

        parts: List[str] = []
        for (vi, vv), (ti, tv) in zip(vis_top, txt_top):
            denom = vv if abs(vv) > 1e-6 else 1e-6
            weight = tv / denom
            parts.append(f"DIM_{vi} {weight:.4f} MUL DIM_{ti} STORE")

        return " ".join(parts) + " CROSS_MODAL_RULE"

    def propose_rule(self, rpn_program: str, context: str, confidence: float = 0.0) -> str:
        """
        Add tentative rule to local discovery space.
        """
        rule_id = f"DISC_{hash(rpn_program) & 0xFFFFFF:06x}"
        if rule_id in self.rules:
            return rule_id

        rule_embedding = self._compile_rule_to_embedding(rpn_program)

        self._local_discoveries[rule_id] = {
            "rpn_program": rpn_program,
            "embedding": rule_embedding,
            "context": context,
            "usage_count": 0,
            "success_count": 0,
            "quality_score": confidence,
            "created_epoch": getattr(self, "_current_epoch", 0),
        }
        return rule_id

    def validate_usage(self, rule_id: str, success: bool) -> float:
        """
        Update quality score for a rule (local discovery or canonical).
        """
        if rule_id not in self._local_discoveries:
            if rule_id in self.rules:
                return 1.0
            return 0.0

        rule = self._local_discoveries[rule_id]
        rule["usage_count"] += 1
        if success:
            rule["success_count"] += 1

        if rule["usage_count"] > 0:
            rule["quality_score"] = rule["success_count"] / rule["usage_count"]

        self._try_promote(rule_id)
        return rule["quality_score"]

    def _try_promote(self, rule_id: str) -> bool:
        """
        Promote a discovery to shared rules if thresholds are met.
        """
        if rule_id not in self._local_discoveries:
            return False

        rule = self._local_discoveries[rule_id]
        if rule["quality_score"] >= self._promotion_threshold and rule["usage_count"] >= self._min_usage_for_promotion:
            new_rule = GrammarRule(
                rule_id=rule_id,
                language="discovered",
                pattern="cross_modal",
                rpn_program=rule["rpn_program"],
                domain="discovered",
                description=f"Discovered from: {rule['context']}",
                is_canonical=False,
            )
            self.rules[rule_id] = new_rule
            del self._local_discoveries[rule_id]
            print(
                f"[GRAMMAR PROMOTE] {rule_id} promoted "
                f"(quality={rule['quality_score']:.2f}, usage={rule['usage_count']})"
            )
            return True
        return False

    def query_similar(self, embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """
        Find top-k rules similar to a provided embedding (canonical + discoveries).
        """
        scores: List[Tuple[str, float]] = []
        for rule_id, rule in self.rules.items():
            rule_emb = self._get_rule_embedding(rule)
            if not rule_emb:
                continue
            sim = self._get_cosine_bridge().compute_similarities([embedding], rule_emb)[0]
            scores.append((rule_id, sim))

        for rule_id, rule_data in self._local_discoveries.items():
            sim = self._get_cosine_bridge().compute_similarities([embedding], rule_data["embedding"])[0]
            scores.append((rule_id, sim))

        scores.sort(key=lambda x: -x[1])
        return scores[:k]

    def _compile_rule_to_embedding(self, rpn_program: str) -> List[float]:
        """Compile an RPN program into a lightweight embedding (hash-based)."""
        tokens = rpn_program.split()
        embedding = [0.0] * 128
        for i, token in enumerate(tokens):
            idx = hash(token) % 128
            embedding[idx] += 1.0 / (i + 1)

        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        return embedding

    def _get_rule_embedding(self, rule: GrammarRule) -> Optional[List[float]]:
        cache_key = f"_emb_{rule.rule_id}"
        if hasattr(rule, cache_key):
            return getattr(rule, cache_key)
        emb = self._compile_rule_to_embedding(rule.rpn_program)
        setattr(rule, cache_key, emb)
        return emb

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def to_snapshot(self) -> bytes:
        """
        Serialize current rule set for worker transfer (memory-only).
        """
        data = {
            "rules": {
                rule_id: {
                    "rule_id": rule.rule_id,
                    "rpn_program": rule.rpn_program,
                    "pattern": rule.pattern,
                    "language": rule.language,
                    "domain": getattr(rule, "domain", "general"),
                }
                for rule_id, rule in self.rules.items()
            },
            "promoted_count": len([r for r in self.rules.values() if not getattr(r, "is_canonical", True)]),
        }
        return json.dumps(data).encode("utf-8")

    def _load_from_snapshot(self, snapshot: bytes) -> None:
        """
        Load grammar state from snapshot bytes (no disk I/O).
        """
        try:
            data = json.loads(snapshot.decode("utf-8"))
        except Exception as exc:
            print(f"[GrammarGalaxy] Failed to load snapshot: {exc}")
            return

        for rule_id, rule_data in data.get("rules", {}).items():
            if rule_id in self.rules:
                continue
            self.rules[rule_id] = GrammarRule(
                rule_id=rule_data["rule_id"],
                rpn_program=rule_data["rpn_program"],
                pattern=rule_data.get("pattern", "unknown"),
                language=rule_data.get("language", "en"),
                domain=rule_data.get("domain", "general"),
            )

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
                    "is_canonical": getattr(rule, "is_canonical", False),
                    "symbol_refs": getattr(rule, "symbol_refs", []),
                    "word_refs": getattr(rule, "word_refs", []),
                }
            )
        parameters = getattr(self, "_parameters", {})
        state = {"rules": rules_data, "total_count": len(rules_data), "parameters": parameters}
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
                symbol_refs=rd.get("symbol_refs", []),
                word_refs=rd.get("word_refs", []),
                examples=rd.get("examples", []),
                description=rd.get("description"),
                semantics=rd.get("semantics", {}),
                usage_conditions=rd.get("usage_conditions", []),
                is_canonical=rd.get("is_canonical", False),
            )
            loaded_rules[rule.rule_id] = rule

        if loaded_rules:
            self.rules = loaded_rules
            print(f"[GrammarGalaxy] Loaded {len(self.rules)} rules from {path}")

        parameters = state.get("parameters", {})
        if parameters:
            setattr(self, "_parameters", parameters)

    def merge_discoveries(self, other_discoveries: Dict[str, Dict]) -> int:
        """
        Merge discoveries from a worker back into the main galaxy.
        """
        merged = 0
        for rule_id, rule_data in other_discoveries.items():
            if rule_id in self._local_discoveries:
                existing = self._local_discoveries[rule_id]
                existing["usage_count"] += rule_data.get("usage_count", 0)
                existing["success_count"] += rule_data.get("success_count", 0)
                if existing["usage_count"] > 0:
                    existing["quality_score"] = existing["success_count"] / existing["usage_count"]
            else:
                self._local_discoveries[rule_id] = rule_data
            merged += 1
            self._try_promote(rule_id)
        return merged

    # ------------------------------------------------------------------ #
    # Galaxy-based word-sequence matching ("reading")
    # ------------------------------------------------------------------ #
    def match_word_sequence(self, word_entries: List[object], *, extra_rules: Optional[List[GrammarRule]] = None) -> List[Dict]:
        """
        Match grammar rules against a sequence of WordEntry-like objects.

        This supports Galaxy-aligned reading rules which store a "word_sequence"
        pattern inside `rule.semantics`:

        - semantics["pattern_type"] == "word_sequence"
        - semantics["word_pattern"] == List[Dict] describing token constraints
        """
        rules = list(self.rules.values())
        if extra_rules:
            rules.extend(extra_rules)

        matches: List[Dict] = []
        for rule in rules:
            semantics = getattr(rule, "semantics", {}) or {}
            if semantics.get("pattern_type") != "word_sequence":
                continue
            pattern = semantics.get("word_pattern")
            if not isinstance(pattern, list) or not pattern:
                continue
            captures = semantics.get("captures", {})
            match_mode = semantics.get("match_mode", "window")
            if match_mode == "subsequence":
                max_skip = int(semantics.get("max_skip", 2))
                skip_categories = semantics.get("skip_categories", ["stopword", "symbol"])
                found = _match_word_pattern_subsequence(
                    word_entries,
                    pattern,
                    max_skip=max(0, max_skip),
                    skip_categories={str(c).lower() for c in (skip_categories or [])},
                )
            else:
                found = _match_word_pattern(word_entries, pattern)

            for m in found:
                score = float(len(pattern))
                matches.append(
                    {
                        "rule": rule,
                        "captures": m.get("captures", {}),
                        "start": m.get("start"),
                        "end": m.get("end"),
                        "score": score,
                        "pattern": pattern,
                        "capture_schema": captures,
                    }
                )
        matches.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return matches


def _match_word_pattern(word_entries: List[object], word_pattern: List[Dict]) -> List[Dict[str, object]]:
    """
    Sliding-window matcher for a word_pattern over WordEntry-like tokens.

    Each WordEntry is expected to expose `.normalized` (lowercase token),
    `.category`, and optionally `.value` and `.rpn_literal`.
    """
    results: List[Dict[str, object]] = []
    n = len(word_entries)
    m = len(word_pattern)
    if m == 0 or n == 0 or m > n:
        return results

    for start in range(0, n - m + 1):
        captures: Dict[str, object] = {}
        ok = True
        for offset, constraint in enumerate(word_pattern):
            entry = word_entries[start + offset]
            normalized = getattr(entry, "normalized", "").lower()
            category = getattr(entry, "category", "")

            word_eq = constraint.get("word")
            if isinstance(word_eq, str) and normalized != word_eq.lower():
                ok = False
                break

            word_in = constraint.get("word_in")
            if isinstance(word_in, list) and normalized not in {str(w).lower() for w in word_in}:
                ok = False
                break

            cat = constraint.get("category")
            if isinstance(cat, str) and category != cat:
                ok = False
                break
            cat_in = constraint.get("category_in")
            if isinstance(cat_in, list) and category not in {str(c).lower() for c in cat_in}:
                ok = False
                break

            capture = constraint.get("capture")
            if isinstance(capture, str) and capture:
                value = getattr(entry, "value", None)
                rpn_literal = getattr(entry, "rpn_literal", None)
                captures[capture] = {
                    "token": getattr(entry, "token", ""),
                    "normalized": normalized,
                    "category": category,
                    "value": value,
                    "rpn_literal": rpn_literal,
                }

        if ok:
            results.append({"captures": captures, "start": start, "end": start + m})

    return results


def _match_word_pattern_subsequence(
    word_entries: List[object],
    word_pattern: List[Dict],
    *,
    max_skip: int,
    skip_categories: set[str],
) -> List[Dict[str, object]]:
    """
    Match `word_pattern` as an in-order subsequence within a limited skip budget.

    This improves robustness for patterns like "gave 3 apples to" where a noun
    may appear between the amount and the preposition.
    """
    results: List[Dict[str, object]] = []
    seen: set[tuple] = set()
    n = len(word_entries)
    m = len(word_pattern)
    if m == 0 or n == 0:
        return results

    for start in range(n):
        captures: Dict[str, object] = {}
        idx = start
        consumed_start: Optional[int] = None
        consumed_end: Optional[int] = None
        ok = True

        for constraint in word_pattern:
            skips = 0
            matched = False
            while idx < n:
                entry = word_entries[idx]
                normalized = getattr(entry, "normalized", "").lower()
                category = str(getattr(entry, "category", "")).lower()

                # Try to match constraint at this position.
                word_eq = constraint.get("word")
                word_in = constraint.get("word_in")
                cat = constraint.get("category")
                cat_in = constraint.get("category_in")

                word_ok = True
                if isinstance(word_eq, str):
                    word_ok = normalized == word_eq.lower()
                if isinstance(word_in, list):
                    word_ok = normalized in {str(w).lower() for w in word_in}

                cat_ok = True
                if isinstance(cat, str):
                    cat_ok = category == cat.lower()
                if isinstance(cat_in, list):
                    cat_ok = category in {str(c).lower() for c in cat_in}

                if word_ok and cat_ok:
                    capture = constraint.get("capture")
                    if isinstance(capture, str) and capture:
                        value = getattr(entry, "value", None)
                        rpn_literal = getattr(entry, "rpn_literal", None)
                        captures[capture] = {
                            "token": getattr(entry, "token", ""),
                            "normalized": normalized,
                            "category": category,
                            "value": value,
                            "rpn_literal": rpn_literal,
                        }
                    matched = True
                    if consumed_start is None:
                        consumed_start = idx
                    consumed_end = idx + 1
                    idx += 1
                    break

                # Not matched: decide whether we can skip this token.
                if category in skip_categories and skips < max_skip:
                    skips += 1
                    idx += 1
                    continue

                # Cannot skip further: fail this start.
                ok = False
                break

            if not ok or not matched:
                ok = False
                break

        if ok and consumed_start is not None and consumed_end is not None:
            # Deduplicate identical matches that can be found from multiple start offsets.
            cap_key = tuple(sorted((k, str(v.get("normalized", "")), str(v.get("value", ""))) for k, v in captures.items()))
            key = (consumed_start, consumed_end, cap_key)
            if key in seen:
                continue
            seen.add(key)
            results.append({"captures": captures, "start": consumed_start, "end": consumed_end})

    return results

    # ------------------------------------------------------------------ #
    # Introspection helpers
    # ------------------------------------------------------------------ #
    def get_high_confidence_rules(self, min_score: float = 0.70) -> List[Dict]:
        """
        Return grammar rules with an implicit confidence (canonical rules assumed high).

        Discovered rules may carry a quality_score attribute. Canonical rules default to 1.0.
        """
        selected: List[Dict] = []
        for rule in self.rules.values():
            score = getattr(rule, "quality_score", 1.0)
            if score >= min_score:
                selected.append(
                    {
                        "id": rule.rule_id,
                        "rpn_program": rule.rpn_program,
                        "quality_score": score,
                    }
                )
        return selected


__all__ = [
    "GrammarRule",
    "GrammarGalaxy",
    "get_grammar_galaxy",
    "default_grammar_rules",
    "default_user_profiles",
    "default_variants",
]

# Singleton accessor to avoid repeated disk loads across workers.
_grammar_galaxy: GrammarGalaxy | None = None


def get_grammar_galaxy() -> GrammarGalaxy:
    global _grammar_galaxy
    if _grammar_galaxy is None:
        _grammar_galaxy = GrammarGalaxy()
    return _grammar_galaxy
