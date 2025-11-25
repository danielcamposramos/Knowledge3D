"""Test semantic parser fallback to grammar rule lookup."""

from knowledge3d.training.arc_agi.semantic_parser import SemanticParser
from knowledge3d.training.arc_agi.semantic_compiler import SemanticToRPNCompiler
from knowledge3d.training.arc_agi.rpn_executor import ARCRPNExecutor


def run():
    parser = SemanticParser(language="en")
    compiler = SemanticToRPNCompiler()
    executor = ARCRPNExecutor()

    instruction = "I love programming"
    sem = parser.parse(instruction)
    print("Semantic:", sem)
    rpn = compiler.compile(sem)
    print("RPN:", rpn)
    # Grammar-rule semantics are not executed by RPN executor; verify passthrough
    assert sem["action"] == "grammar_rule"
    assert sem["rule_id"] == "en_simple_sentence"
    print("✅ grammar rule matched")


if __name__ == "__main__":
    run()
