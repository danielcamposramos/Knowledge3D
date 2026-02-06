"""Document enrichment pipeline for ingestion-time knowledge synthesis."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from .numbered_context import NumberedContextProvider
from .ollama_manager import OllamaModelManager

SYSTEM_PROMPT = (
    "You are a Knowledge3D ingestion extractor.\n"
    "Return ONLY valid JSON that matches the requested schema.\n"
    "No markdown fences, no prose, no commentary."
)


class EnrichmentPipeline:
    """Generate matryoshka embeddings, deduplicate content and extract patterns."""

    def __init__(self, use_local_models: bool = False):
        self.use_local_models = use_local_models
        self.embedding_cache: dict[str, dict[int, np.ndarray]] = {}
        self.symlink_registry: dict[str, str] = {}

    @staticmethod
    def _content_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def generate_matryoshka_embedding(self, content: str) -> dict[int, np.ndarray]:
        """Generate deterministic multi-resolution embeddings."""
        digest = self._content_hash(content)
        if digest in self.embedding_cache:
            return self.embedding_cache[digest]

        seed = int(digest[:16], 16) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        base = rng.normal(0.0, 1.0, size=(2048,)).astype(np.float32)
        norm = float(np.linalg.norm(base))
        if norm > 0:
            base = base / norm

        embeddings = {
            64: base[:64].copy(),
            128: base[:128].copy(),
            512: base[:512].copy(),
            2048: base.copy(),
        }
        self.embedding_cache[digest] = embeddings
        return embeddings

    def find_or_create_symlink(self, content: str) -> str:
        """Return canonical entry id for duplicated content."""
        digest = self._content_hash(content)[:16]
        if digest in self.symlink_registry:
            return self.symlink_registry[digest]
        entry_id = f"entry_{digest}"
        self.symlink_registry[digest] = entry_id
        return entry_id

    @staticmethod
    def _domain_model(domain: str) -> str:
        mapping = {
            "math": "deepseek-r1:7b",
            "visual": "gemma3:latest",
            "physics": "qwen2.5:14b",
            "logic": "deepseek-r1:7b",
            "reasoning": "qwen2.5:14b",
            "computer_science": "qwen2.5:14b",
        }
        return mapping.get(domain, "qwen2.5:14b")

    @staticmethod
    def _strip_json_fence(response: str) -> str:
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    def _extract_patterns_heuristic(self, content: str, domain: str) -> list[dict[str, Any]]:
        text = content.lower()
        patterns: list[dict[str, Any]] = []
        if domain == "math":
            if "derivative" in text:
                patterns.append(
                    {
                        "name": "derivative_pattern",
                        "program": "x power_rule derivative",
                        "domain": domain,
                    }
                )
            if "integral" in text:
                patterns.append(
                    {
                        "name": "integral_pattern",
                        "program": "x antiderivative evaluate",
                        "domain": domain,
                    }
                )
        if domain == "logic":
            if "if" in text and "then" in text:
                patterns.append(
                    {
                        "name": "implication_pattern",
                        "program": "premise conclusion imply",
                        "domain": domain,
                    }
                )
        if domain == "visual":
            if "rotate" in text:
                patterns.append(
                    {
                        "name": "rotation_pattern",
                        "program": "shape angle rotate",
                        "domain": domain,
                    }
                )
        return patterns

    @staticmethod
    def _parse_patterns_from_llm_response(response: str, domain: str) -> list[dict[str, Any]]:
        del domain
        if not response:
            return []
        response = EnrichmentPipeline._strip_json_fence(response)
        try:
            decoded = json.loads(response)
            if isinstance(decoded, dict) and isinstance(decoded.get("patterns"), list):
                return [item for item in decoded["patterns"] if isinstance(item, dict)]
            if isinstance(decoded, list):
                return [item for item in decoded if isinstance(item, dict)]
            if isinstance(decoded, dict):
                return [decoded]
        except json.JSONDecodeError:
            return []
        return []

    def _format_chunks(self, chunks: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for chunk in chunks:
            lines.append(
                f"--- chunk {chunk['chunk_id']} "
                f"(lines {chunk['line_start']}-{chunk['line_end']}) ---"
            )
            lines.append(chunk["content"])
        return "\n".join(lines)

    def _pattern_schema(self, domain: str) -> dict[str, Any]:
        base = {
            "patterns": [
                {
                    "name": "string",
                    "input_type": "string",
                    "transformation_steps": ["string"],
                    "output_type": "string",
                    "rpn_template": "string",
                }
            ]
        }
        if domain == "visual":
            base["patterns"][0]["input_type"] = "grid"
            base["patterns"][0]["output_type"] = "grid"
        return base

    def _query_structured_patterns(
        self,
        content: str,
        domain: str,
    ) -> list[dict[str, Any]]:
        context_provider = NumberedContextProvider(content=content, chunk_size=2000)
        initial = context_provider.get_initial_context(num_chunks=1)
        schema = self._pattern_schema(domain)

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Task: Extract reusable {domain} patterns.\n"
            "You may request more chunks with JSON: "
            '{"request_more": true, "chunk_ids": [2,3]}.\n\n'
            f"Schema:\n{json.dumps(schema, indent=2)}\n\n"
            f"Context:\n{self._format_chunks(initial['provided_chunks'])}\n\n"
            "Return JSON only."
        )
        model = self._domain_model(domain)
        with OllamaModelManager(default_timeout=180.0) as manager:
            manager.load_model(model)
            first = manager.query(model=model, prompt=prompt, timeout=180.0)
            if first.returncode != 0:
                return []
            response = self._strip_json_fence(first.output)

            try:
                maybe = json.loads(response) if response else {}
            except json.JSONDecodeError:
                maybe = {}

            if isinstance(maybe, dict) and maybe.get("request_more"):
                chunk_ids = maybe.get("chunk_ids", [])
                if isinstance(chunk_ids, list):
                    extra_chunks = context_provider.get_chunks(
                        [int(c) for c in chunk_ids if isinstance(c, int) or str(c).isdigit()]
                    )
                else:
                    extra_chunks = []
                second_prompt = (
                    f"{SYSTEM_PROMPT}\n\n"
                    f"Task: Continue extracting {domain} patterns.\n"
                    f"Schema:\n{json.dumps(schema, indent=2)}\n\n"
                    f"Additional Context:\n{self._format_chunks(extra_chunks)}\n\n"
                    "Return JSON only."
                )
                second = manager.query(model=model, prompt=second_prompt, timeout=180.0)
                if second.returncode != 0:
                    return []
                return self._parse_patterns_from_llm_response(
                    response=second.output,
                    domain=domain,
                )

            return self._parse_patterns_from_llm_response(
                response=response,
                domain=domain,
            )

    def extract_procedural_patterns(self, content: str, domain: str) -> list[dict[str, Any]]:
        """Extract reusable patterns either heuristically or with local models."""
        if not self.use_local_models:
            return self._extract_patterns_heuristic(content=content, domain=domain)
        llm_patterns = self._query_structured_patterns(content=content, domain=domain)
        if llm_patterns:
            return llm_patterns
        return self._extract_patterns_heuristic(content=content, domain=domain)

    def _find_related_concepts(self, content: str) -> list[str]:
        if not self.use_local_models:
            return []
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            "Task: list up to 10 related concepts.\n"
            'Return JSON: {"related_concepts": ["..."]}\n\n'
            f"Content:\n{content[:1500]}"
        )
        with OllamaModelManager(default_timeout=120.0) as manager:
            model = "qwen2.5:14b"
            manager.load_model(model)
            response = manager.query(model=model, prompt=prompt, timeout=120.0).output
        stripped = self._strip_json_fence(response)
        try:
            parsed = json.loads(stripped)
            concepts = parsed.get("related_concepts", [])
            if isinstance(concepts, list):
                return [str(c).strip() for c in concepts if str(c).strip()][:10]
        except json.JSONDecodeError:
            pass
        return [token.strip() for token in stripped.split(",") if token.strip()][:10]

    def enrich_document(self, content: str, metadata: dict[str, Any]) -> dict[str, Any]:
        """Run full enrichment pipeline for one document payload."""
        entry_id = self.find_or_create_symlink(content)
        embeddings = self.generate_matryoshka_embedding(content)
        domain = str(metadata.get("domain", "general"))
        patterns = self.extract_procedural_patterns(content=content, domain=domain)
        related = self._find_related_concepts(content)
        return {
            "entry_id": entry_id,
            "embeddings": embeddings,
            "patterns": patterns,
            "related_concepts": related,
            "metadata": metadata,
        }
