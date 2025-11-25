"""Smoke tests for grammar galaxy → executor."""

from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
from knowledge3d.training.arc_agi.grammar_normalizer import GrammarNormalizer
from knowledge3d.training.arc_agi.grammar_executor import GrammarRPNExecutor


def run_tests():
    galaxy = GrammarGalaxy()
    exec = GrammarRPNExecutor()
    norm = GrammarNormalizer(galaxy)

    tests = [
        ("en_simple_sentence", {"subject": "I", "verb": "love", "object": "programming"}, "I love programming"),
        ("pt_simple_sentence", {"subject": "Eu", "verb": "amo", "object": "código"}, "Eu amo código"),
        ("ja_simple_sentence", {"subject": "私は", "object": "コード", "verb": "書く"}, "私は コード を 書く"),
        ("en_question", {"auxiliary": "Do", "subject": "you", "verb": "like", "object": "math"}, "Do you like math"),
        ("es_simple_sentence", {"subject": "Yo", "verb": "amo", "object": "programar"}, "Yo amo programar"),
        ("en_imperative", {"verb": "Open", "object": "the door"}, "Open the door"),
        ("en_math_expression", {"operand1": "2", "operand2": "3", "result": "5"}, "2 + 3 = 5"),
        ("en_graph_statement", {"subject": "NodeA", "edge": "connects to", "object": "NodeB"}, "NodeA connects to NodeB"),
        ("en_conditional_if", {"if": "If it rains", "condition": "", "then": "then", "consequence": "we stay home"}, "If it rains then we stay home"),
        ("en_temporal_sequence", {"event1": "Wake up", "event2": "write code", "event3": "run tests"}, "Wake up then write code then run tests"),
        (
            "en_visual_description",
            {"subject": "Object", "is": "is", "color": "red", "shape": "square", "position": "at top-left"},
            "Object is red square at top-left",
        ),
        (
            "en_audio_description",
            {"sound": "Beep", "occurs": "at", "time": "1.2 seconds"},
            "Beep at 1.2 seconds",
        ),
        (
            "en_video_scene_graph",
            {
                "subject": "CharacterA",
                "action": "moves",
                "object": "to the door",
                "location": "in the hallway",
                "time": "at 00:01:23",
            },
            "CharacterA moves to the door in the hallway at 00:01:23",
        ),
        (
            "en_passive",
            {"subject": "The cat", "object": "The ball", "be": "was", "verb_ed": "kicked", "by_prep": "by"},
            "The ball was kicked by The cat",
        ),
        (
            "en_relative_clause",
            {
                "subject": "The student",
                "verb": "read",
                "object": "the book",
                "that_token": "that",
                "rel_verb": "won",
                "rel_object": "an award",
            },
            "The student read the book that won an award",
        ),
        (
            "en_comparative",
            {
                "subject": "Algorithm A",
                "be": "is",
                "more_token": "more",
                "adjective": "efficient",
                "than_token": "than",
                "object": "Algorithm B",
            },
            "Algorithm A is more efficient than Algorithm B",
        ),
        (
            "en_superlative",
            {
                "subject": "This model",
                "be": "is",
                "the_token": "the",
                "most_token": "most",
                "adjective": "accurate",
                "in_token": "in",
                "group": "the benchmark",
            },
            "This model is the most accurate in the benchmark",
        ),
        (
            "en_derivative",
            {
                "derivative_token": "d/dx",
                "of_token": "of",
                "function": "x^2",
                "with_respect_token": "with respect to",
                "variable": "x",
                "equals_token": "=",
                "result": "2x",
            },
            "d/dx of x^2 with respect to x = 2x",
        ),
        ("en_set_membership", {"element": "x", "in_token": "∈", "set_name": "R"}, "x ∈ R"),
        ("en_logic_implication", {"premise": "P", "implies_token": "→", "conclusion": "Q"}, "P → Q"),
        (
            "en_coordination",
            {
                "subject": "I",
                "verb": "write",
                "object": "code",
                "subject2": "I",
                "verb2": "test",
                "object2": "it",
            },
            "I write code and I test it",
        ),
    ]

    passed = 0
    for rule_id, ctx, expected in tests:
        rule = galaxy.get_rule(rule_id)
        out = exec.execute(rule.rpn_program, ctx, user_style=galaxy.get_user_profile("USER_daniel").get("grammar_preferences"))
        if out == expected or out.startswith(expected):
            status = "✅"
            passed += 1
        else:
            status = "❌"
        print(f"{status} {rule_id}: {out}")

    total = len(tests)
    print(f"\n{passed}/{total} tests passed ({passed/total*100:.1f}%)")

    # Normalizer smoke: slang/typo → canonical
    slang_line = "u luv teh code"
    norm_line = norm.normalize_text(slang_line, "en")
    expected_norm = "you love the code"
    if norm_line == expected_norm:
        print("✅ normalizer:", norm_line)
    else:
        print("❌ normalizer:", norm_line, "(expected:", expected_norm, ")")


if __name__ == "__main__":
    run_tests()
