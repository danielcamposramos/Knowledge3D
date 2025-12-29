"""Execute grammar RPN programs into sentences."""

from __future__ import annotations

from typing import Dict, List


class GrammarRPNExecutor:
    """Simple stack-based executor for grammar RPN programs."""

    def execute(self, rpn_program: str, context: Dict[str, str], user_style: Dict | None = None) -> str:
        """
        Execute RPN grammar program into a sentence string.

        Supported ops:
            SUBJECT/VERB/OBJECT/AUXILIARY: push from context
            WO_PARTICLE: push "を"
            SVO_ORDER/SOV_ORDER: reorder stack items accordingly
            CONJUGATE_VERB/APPLY_TENSE: placeholder (no-op for now)
            CONCAT_SENTENCE: join tokens with spaces
        """
        tokens = rpn_program.split()
        stack: List[str] = []
        out: List[str] = []

        idx = 0
        while idx < len(tokens):
            tok = tokens[idx]
            if tok == "SUBJECT":
                stack.append(context.get("subject", ""))
            elif tok == "VERB":
                stack.append(context.get("verb", ""))
            elif tok == "OBJECT":
                stack.append(context.get("object", ""))
            elif tok == "AUXILIARY":
                stack.append(context.get("auxiliary", ""))
            elif tok == "SUBJECT2":
                stack.append(context.get("subject2", ""))
            elif tok == "VERB2":
                stack.append(context.get("verb2", ""))
            elif tok == "OBJECT2":
                stack.append(context.get("object2", ""))
            elif tok == "OPERAND1":
                stack.append(context.get("operand1", ""))
            elif tok == "OPERAND2":
                stack.append(context.get("operand2", ""))
            elif tok == "RESULT":
                stack.append(context.get("result", ""))
            elif tok == "EDGE":
                stack.append(context.get("edge", ""))
            elif tok == "IF":
                stack.append(context.get("if", "If"))
            elif tok == "THEN":
                stack.append(context.get("then", "then"))
            elif tok == "CONDITION":
                stack.append(context.get("condition", ""))
            elif tok == "CONSEQUENCE":
                stack.append(context.get("consequence", ""))
            elif tok == "EVENT1":
                stack.append(context.get("event1", ""))
            elif tok == "EVENT2":
                stack.append(context.get("event2", ""))
            elif tok == "EVENT3":
                stack.append(context.get("event3", ""))
            elif tok == "IS":
                stack.append(context.get("is", "is"))
            elif tok == "COLOR":
                stack.append(context.get("color", ""))
            elif tok == "SHAPE":
                stack.append(context.get("shape", ""))
            elif tok == "POSITION":
                stack.append(context.get("position", ""))
            elif tok == "SOUND":
                stack.append(context.get("sound", ""))
            elif tok == "OCCURS":
                stack.append(context.get("occurs", "at"))
            elif tok == "TIME":
                stack.append(context.get("time", ""))
            elif tok == "ACTION":
                stack.append(context.get("action", ""))
            elif tok == "LOCATION":
                stack.append(context.get("location", ""))
            elif tok == "BE":
                stack.append(context.get("be", "is"))
            elif tok == "VERB_ED":
                stack.append(context.get("verb_ed", ""))
            elif tok == "BY_PREP":
                stack.append(context.get("by_prep", "by"))
            elif tok == "THAT_TOKEN":
                stack.append(context.get("that_token", "that"))
            elif tok == "REL_VERB":
                stack.append(context.get("rel_verb", ""))
            elif tok == "REL_OBJECT":
                stack.append(context.get("rel_object", ""))
            elif tok == "MORE_TOKEN":
                stack.append(context.get("more_token", "more"))
            elif tok == "ADJECTIVE":
                stack.append(context.get("adjective", ""))
            elif tok == "THAN_TOKEN":
                stack.append(context.get("than_token", "than"))
            elif tok == "THE_TOKEN":
                stack.append(context.get("the_token", "the"))
            elif tok == "MOST_TOKEN":
                stack.append(context.get("most_token", "most"))
            elif tok == "IN_TOKEN":
                stack.append(context.get("in_token", "in"))
            elif tok == "GROUP":
                stack.append(context.get("group", ""))
            elif tok == "DERIVATIVE_TOKEN":
                stack.append(context.get("derivative_token", "d/dx"))
            elif tok == "OF_TOKEN":
                stack.append(context.get("of_token", "of"))
            elif tok == "FUNCTION":
                stack.append(context.get("function", "f(x)"))
            elif tok == "WITH_RESPECT_TOKEN":
                stack.append(context.get("with_respect_token", "with respect to"))
            elif tok == "VARIABLE":
                stack.append(context.get("variable", "x"))
            elif tok == "EQUALS_TOKEN":
                stack.append(context.get("equals_token", "="))
            elif tok == "RESULT":
                stack.append(context.get("result", ""))
            elif tok == "ELEMENT":
                stack.append(context.get("element", ""))
            elif tok == "SET_NAME":
                stack.append(context.get("set_name", ""))
            elif tok == "IMPLIES_TOKEN":
                stack.append(context.get("implies_token", "→"))
            elif tok == "AND":
                out.append("and")
            elif tok == "PREMISE":
                stack.append(context.get("premise", ""))
            elif tok == "CONCLUSION":
                stack.append(context.get("conclusion", ""))
            elif tok == "WO_PARTICLE":
                stack.append("を")
            elif tok == "SVO_ORDER":
                # Expect top-3 = object, verb, subject (due to push order above)
                obj = stack.pop() if stack else ""
                verb = stack.pop() if stack else ""
                subj = stack.pop() if stack else ""
                out.extend([subj, verb, obj])
            elif tok == "SOV_ORDER":
                # Support optional particle between object and verb.
                if len(stack) >= 4 and stack[-2] == "を":
                    verb = stack.pop()
                    particle = stack.pop()
                    obj = stack.pop()
                    subj = stack.pop()
                    out.extend([subj, obj, particle, verb])
                else:
                    obj = stack.pop() if stack else ""
                    verb = stack.pop() if stack else ""
                    subj = stack.pop() if stack else ""
                    out.extend([subj, obj, verb])
            elif tok == "CONCAT_SENTENCE":
                # Finalize
                pass
            elif tok == "CONCAT":
                # Deprecated; treat as space join of stack
                pass
            elif tok == "RECALL":
                # No-op placeholder for compatibility.
                pass
            elif tok == "CONJUGATE_VERB":
                # Placeholder: leave verb unchanged
                pass
            elif tok == "APPLY_TENSE":
                # Placeholder: no tense change
                pass
            elif tok == "CONCAT_SENTENCE":
                pass
            elif tok == "CONCAT_PARAGRAPH" or tok == "CONCAT_DOCUMENT":
                pass
            else:
                # Literal token
                stack.append(tok)
            idx += 1

        # If out is empty, fall back to stacked tokens.
        if not out:
            out = [tok for tok in stack if tok]

        sentence = " ".join([t for t in out if t])

        # Apply simple style touches (formality/punctuation/emoji)
        if user_style:
            emoji_usage = float(user_style.get("emoji_usage", 0.0))
            if emoji_usage > 0.5:
                sentence = sentence + " 🙂"
        return sentence.strip()


__all__ = ["GrammarRPNExecutor"]
