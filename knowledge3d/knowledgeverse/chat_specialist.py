"""
Chat Specialist: Internal TRM specialist for conversational I/O.

Provides standard LLM-compatible interface (chat messages) while maintaining
full PTX sovereignty. Acts as compatibility layer between standard chat formats
and K3D's procedural spatial reasoning.
"""

from __future__ import annotations

from typing import Any

from knowledge3d.knowledgeverse.specialist_base import SpecialistBase


class ChatSpecialist(SpecialistBase):
    """
    Internal specialist for chat-like interactions.

    Key design:
    - Accepts standard LLM chat format (messages: [{role, content}])
    - Routes internally using Galaxy navigation + PTX/RPN
    - Returns standard LLM response format
    - Zero external LLM calls (fully sovereign)

    This is a "compatibility layer" - standard I/O, sovereign processing.
    """

    def __init__(
        self,
        knowledgeverse,
        parent: SpecialistBase | None = None,
        **kwargs
    ):
        super().__init__(
            name="chat",
            domain="conversational",
            parent=parent,
            **kwargs
        )
        self.knowledgeverse = knowledgeverse
        self.conversation_history: list[dict[str, str]] = []

    def _query_galaxy(self, galaxy_name: str, semantic_query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Query a named galaxy through the current Knowledgeverse API.

        Returns normalized entries (bare entry dicts), regardless of whether the
        underlying query returns wrapped {"entry": ...} records.
        """
        manager = getattr(self.knowledgeverse, "galaxy_manager", None)
        if manager is None:
            return []
        try:
            specialist = galaxy_name.lower()
            results = manager.query(
                query_text=semantic_query,
                specialist=specialist,
                top_k=top_k,
                galaxies=[galaxy_name],
            )
        except Exception:
            return []

        normalized: list[dict[str, Any]] = []
        for item in results or []:
            if isinstance(item, dict) and isinstance(item.get("entry"), dict):
                normalized.append(item["entry"])
            elif isinstance(item, dict):
                normalized.append(item)
        return normalized

    def process_chat_message(
        self,
        messages: list[dict[str, str]],
        use_enriched: bool = True
    ) -> str:
        """
        Process chat message(s) using sovereign Galaxy navigation.

        Args:
            messages: Standard chat format [{"role": "user", "content": "..."}]
            use_enriched: Whether to use enriched Galaxy content

        Returns:
            Standard LLM response (string)
        """
        # Extract latest user message
        user_message = self._extract_user_message(messages)
        if not user_message:
            return "I need a question or instruction to help you."

        # Route internally to answer
        response = self._answer_using_galaxy(user_message, use_enriched)

        # Record conversation
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return response

    def _extract_user_message(self, messages: list[dict[str, str]]) -> str:
        """Extract latest user message from chat format."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return str(msg.get("content", "")).strip()
        return ""

    def _answer_using_galaxy(self, query: str, use_enriched: bool) -> str:
        """
        Answer query using sovereign Galaxy navigation.

        Internal routing:
        1. Classify query type (question, instruction, math, visual, etc.)
        2. Query relevant galaxies
        3. Compose RPN program if needed
        4. Format response in natural language

        NO external LLM calls - fully sovereign!
        """
        # Step 1: Classify query intent
        query_type = self._classify_query(query)

        # Step 2: Route to appropriate Galaxy navigation
        if query_type == "math":
            response = self._answer_math_query(query, use_enriched)
        elif query_type == "visual":
            response = self._answer_visual_query(query, use_enriched)
        elif query_type == "transformation":
            response = self._answer_transformation_query(query, use_enriched)
        elif query_type == "definition":
            response = self._answer_definition_query(query, use_enriched)
        else:
            # General query - multi-galaxy search
            response = self._answer_general_query(query, use_enriched)

        return response

    def _classify_query(self, query: str) -> str:
        """
        Classify query type using simple pattern matching.

        (Could be enhanced with a small classifier trained on Galaxy data)
        """
        query_lower = query.lower()

        # Math keywords
        if any(kw in query_lower for kw in [
            "calculate", "compute", "solve", "derivative", "integral",
            "equation", "formula", "sum", "product", "factor"
        ]):
            return "math"

        # Visual keywords
        if any(kw in query_lower for kw in [
            "draw", "shape", "color", "grid", "pattern", "visual",
            "image", "rotate", "flip", "line", "circle", "rectangle"
        ]):
            return "visual"

        # Transformation keywords
        if any(kw in query_lower for kw in [
            "transform", "change", "convert", "apply", "map", "rule"
        ]):
            return "transformation"

        # Definition keywords
        if any(kw in query_lower for kw in [
            "what is", "define", "meaning", "explain", "describe"
        ]):
            return "definition"

        return "general"

    def _answer_math_query(self, query: str, use_enriched: bool) -> str:
        """Answer math query using Math Galaxy."""
        # Query Math Galaxy for relevant symbols/formulas
        math_results = self._query_galaxy("Math", query, top_k=5)

        if not math_results:
            return "I don't have enough mathematical knowledge to answer this yet. Could you rephrase or provide more context?"

        # Format response using retrieved math symbols
        response_parts = ["Based on my mathematical knowledge:\n"]
        for idx, result in enumerate(math_results[:3], 1):
            symbol_id = result.get("id", "unknown")
            rpn_program = result.get("rpn_program", "")
            if rpn_program:
                response_parts.append(f"{idx}. {symbol_id}: {rpn_program}")

        return "\n".join(response_parts)

    def _answer_visual_query(self, query: str, use_enriched: bool) -> str:
        """Answer visual query using Drawing Galaxy."""
        # Query Drawing Galaxy for visual primitives
        drawing_results = self._query_galaxy("Drawing", query, top_k=5)

        if not drawing_results:
            return "I don't have visual primitives for this yet. Could you describe it differently?"

        # Format response
        response_parts = ["Visual elements I found:\n"]
        for idx, result in enumerate(drawing_results[:3], 1):
            primitive_id = result.get("id", "unknown")
            rpn_program = result.get("rpn_program", "")
            if rpn_program:
                response_parts.append(f"{idx}. {primitive_id}: {rpn_program}")

        return "\n".join(response_parts)

    def _answer_transformation_query(self, query: str, use_enriched: bool) -> str:
        """Answer transformation query using Grammar Galaxy."""
        # Query Grammar Galaxy for transformation rules
        grammar_results = self._query_galaxy("Grammar", query, top_k=5)

        if not grammar_results:
            return "I don't have transformation rules for this yet."

        # Format response
        response_parts = ["Transformation rules I found:\n"]
        for idx, result in enumerate(grammar_results[:3], 1):
            rule_id = result.get("id", "unknown")
            rpn_program = result.get("rpn_program", "")
            if rpn_program:
                response_parts.append(f"{idx}. {rule_id}: {rpn_program}")

        return "\n".join(response_parts)

    def _answer_definition_query(self, query: str, use_enriched: bool) -> str:
        """Answer definition query using Word/Character Galaxies."""
        # Query Word Galaxy for definitions
        word_results = self._query_galaxy("Word", query, top_k=3)

        if not word_results:
            return "I don't have a definition for this yet."

        # Format response
        response_parts = ["Definitions I found:\n"]
        for idx, result in enumerate(word_results[:3], 1):
            word_id = result.get("id", "unknown")
            metadata = result.get("metadata", {})
            definition = metadata.get("definition", "No definition available")
            response_parts.append(f"{idx}. {word_id}: {definition}")

        return "\n".join(response_parts)

    def _answer_general_query(self, query: str, use_enriched: bool) -> str:
        """
        Answer general query using multi-galaxy search.

        Query all relevant galaxies and compose response.
        """
        all_results = []

        # Query multiple galaxies
        for galaxy_name in ["Drawing", "Math", "Grammar", "Word", "Reality"]:
            try:
                results = self._query_galaxy(galaxy_name, query, top_k=2)
                for result in results:
                    result["source_galaxy"] = galaxy_name
                    all_results.append(result)
            except Exception:
                # Skip if galaxy not available
                continue

        if not all_results:
            return "I don't have enough knowledge to answer this yet. My knowledge base is still growing!"

        # Format multi-galaxy response
        response_parts = ["Based on my knowledge across multiple domains:\n"]
        for idx, result in enumerate(all_results[:5], 1):
            galaxy = result.get("source_galaxy", "unknown")
            entry_id = result.get("id", "unknown")
            rpn_program = result.get("rpn_program", "")
            if rpn_program:
                response_parts.append(f"{idx}. [{galaxy}] {entry_id}: {rpn_program}")

        return "\n".join(response_parts)

    def answer_multiple_choice(
        self,
        question_text: str,
        options: list[str],
        use_enriched: bool = True,
        galaxy_scope: list[str] | None = None,
    ) -> str:
        """
        Answer multiple-choice question using simple heuristics.

        Used by MMLU benchmark. Routes to best matching option.

        NOTE: This is a simplified version for initial testing.
        Can be enhanced with Galaxy navigation in future iterations.
        """
        # Simple sovereignty-safe scoring with scoped Galaxy evidence.
        option_scores = {}
        question_lower = question_text.lower()
        scope = self._normalize_scope(galaxy_scope) or ["Grammar", "Word", "Math", "Reality", "Drawing"]
        evidence_blob_parts: list[str] = []
        for galaxy_name in scope:
            evidence = self._query_galaxy(galaxy_name, question_text, top_k=3)
            for row in evidence:
                evidence_blob_parts.append(
                    f"{row.get('id','')} {row.get('name','')} {row.get('category','')} {row.get('domain','')}"
                )
                metadata = row.get("metadata")
                if isinstance(metadata, dict):
                    evidence_blob_parts.append(str(metadata))
        evidence_blob = " ".join(evidence_blob_parts).lower()

        for option in options:
            score = 0
            option_lower = option.lower()

            # Score based on keyword overlap
            option_words = set(option_lower.split())
            question_words = set(question_lower.split())
            overlap = option_words & question_words
            score += len(overlap)

            # Score against scoped Galaxy evidence.
            if evidence_blob:
                score += sum(1 for tok in option_words if tok and tok in evidence_blob)

            # Prefer numeric answers for math questions
            if any(kw in question_lower for kw in ['calculate', 'compute', 'find']):
                if any(char.isdigit() for char in option):
                    score += 1

            option_scores[option] = score

        # Return option with highest score
        if option_scores:
            best_option = max(option_scores.items(), key=lambda x: x[1])[0]
            return best_option

        # Fallback: return first option
        return options[0] if options else ""

    def _normalize_scope(self, scope: list[str] | None) -> list[str]:
        if not scope:
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in scope:
            name = str(item).strip()
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(name)
        return out

    def get_stats(self) -> dict[str, Any]:
        """Get conversation statistics."""
        return {
            "specialist_name": self.name,
            "domain": self.domain,
            "conversation_turns": len(self.conversation_history) // 2,
            "query_count": self.query_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
        }
