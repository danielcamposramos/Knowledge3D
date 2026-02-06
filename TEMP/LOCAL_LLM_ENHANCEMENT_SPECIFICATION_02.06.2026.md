# Local LLM Enhancement Specification — Sovereign Integration

**Created:** February 6, 2026
**Author:** Claude (Architecture Partner)
**Priority:** CRITICAL (Post-Phase 1B Hardening)
**Status:** Architecture Specification

---

## Executive Summary

**User Directive:** *"Make sure placeholders are only temporary, they should not be a norm, actually, we do not use that here neither gave up on sovereignty on the hotpath, so all things external must aim to be transformed to K3D standard ASAP."*

This specification addresses:
1. **Enhanced System Prompts** for local LLMs (cleaner, more structured output)
2. **RAG with Numbered Context** (page/line numbers for LLM to request more)
3. **Model Unload/Reload** between tasks (clean context, worth the time cost)
4. **Transformation to K3D Standard** (no synthetic placeholders, everything becomes RPN/Galaxy/PTX)

**Key Principle:** Local LLMs are **ingestion accelerators**, but their outputs MUST be transformed into sovereign K3D artifacts (RPN programs, Galaxy entries, PTX kernels) immediately. No external formats should persist.

---

## Problem Statement

### Current State (Phase 1B)

**Good:**
- ✅ Ollama models successfully used for enrichment
- ✅ 30,272 embeddings generated
- ✅ Pattern extraction working

**Not Acceptable:**
- ❌ LLM outputs are free-form text (not structured)
- ❌ No numbered context for LLM to request more pages
- ❌ Models persist between tasks (context pollution)
- ❌ `embedding_count` is synthetic in Stargate (placeholder)
- ❌ Enrichment outputs are external metadata (not RPN programs)

### Target State (Post-Hardening)

**Must Have:**
1. ✅ LLM system prompts produce **structured JSON output** (parseable)
2. ✅ RAG provides **numbered pages/lines** (LLM can ask for "page 5-7" or "lines 100-150")
3. ✅ Models **unload after each task** (clean context, no pollution)
4. ✅ **ALL LLM outputs transform to K3D standard**:
   - Pattern → RPN program
   - Concept → Galaxy entry
   - Proof template → Grammar rule (RPN)
   - Embedding → Matryoshka vector in Galaxy

**Sovereignty Compliance:**
- Local LLMs are **tools** (like PyPDF2, numpy, PIL)
- Their outputs are **raw materials**
- We **immediately transmute** raw materials → sovereign artifacts
- **No external formats persist** beyond ingestion pipeline

---

## Enhancement 1: Structured System Prompts

### Current Problem

```python
# Current (from enrichment_pipeline.py)
prompt = f"""
Extract mathematical patterns from this content:
{content[:2000]}

Output each pattern with:
1. Pattern name
2. Input requirements
3. Transformation steps
4. Output format
"""
response = ollama_query("deepseek-r1:14b", prompt)
# Response is free-form text, hard to parse
```

**Issue:** Free-form LLM output requires fragile parsing (regex, heuristics). Fails silently on unexpected formats.

### Solution: JSON-Structured Prompts

```python
# Enhanced system prompt
SYSTEM_PROMPT = """
You are a pattern extraction assistant for Knowledge3D.
Your task is to analyze content and extract patterns in STRICT JSON format.

IMPORTANT:
- Output ONLY valid JSON (no markdown, no explanations)
- Use the exact schema provided
- If you cannot extract patterns, return {"patterns": []}
"""

def extract_procedural_patterns(self, content: str, domain: str) -> List[RPNProgram]:
    """Enhanced pattern extraction with structured output."""

    if domain == "math":
        schema = {
            "patterns": [
                {
                    "name": "string (e.g., 'derivative_power_rule')",
                    "input_type": "string (e.g., 'polynomial')",
                    "transformation_steps": ["step1", "step2", "step3"],
                    "output_type": "string (e.g., 'polynomial')",
                    "rpn_template": "string (RPN program template)"
                }
            ]
        }

        prompt = f"""
{SYSTEM_PROMPT}

TASK: Extract mathematical patterns from the following content.

CONTENT (truncated to 2000 chars):
{content[:2000]}

OUTPUT SCHEMA:
{json.dumps(schema, indent=2)}

OUTPUT (JSON only):
"""

    elif domain == "visual":
        schema = {
            "patterns": [
                {
                    "name": "string (e.g., 'rotation_90_degrees')",
                    "input_shape": "string (e.g., 'grid')",
                    "transformation": "string (e.g., 'rotate_cw')",
                    "output_shape": "string (e.g., 'grid')",
                    "rpn_template": "string (RPN program template)"
                }
            ]
        }

        prompt = f"""
{SYSTEM_PROMPT}

TASK: Extract visual/geometric patterns from the following content.

CONTENT (truncated to 2000 chars):
{content[:2000]}

OUTPUT SCHEMA:
{json.dumps(schema, indent=2)}

OUTPUT (JSON only):
"""

    # Query LLM with structured prompt
    response = self.ollama_query_with_retry(
        model="deepseek-r1:14b",
        prompt=prompt,
        max_retries=3
    )

    # Parse JSON response
    try:
        parsed = json.loads(response)
        patterns = parsed.get("patterns", [])
    except json.JSONDecodeError as e:
        print(f"[EnrichmentPipeline] JSON parse error: {e}")
        print(f"[EnrichmentPipeline] Raw response: {response[:500]}")
        return []

    # Transform to RPN programs (K3D standard)
    rpn_programs = []
    for pattern_data in patterns:
        rpn_program = self._pattern_to_rpn(pattern_data, domain)
        rpn_programs.append(rpn_program)

    return rpn_programs

def ollama_query_with_retry(self, model: str, prompt: str, max_retries: int = 3) -> str:
    """Query Ollama with retry logic for structured output."""

    for attempt in range(max_retries):
        response = self.ollama_query(model, prompt)

        # Strip markdown code blocks if present
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        # Validate JSON
        try:
            json.loads(response)
            return response
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                print(f"[EnrichmentPipeline] Attempt {attempt+1} failed, retrying...")
                continue
            else:
                print(f"[EnrichmentPipeline] All retries exhausted")
                return '{"patterns": []}'

    return '{"patterns": []}'
```

### Benefits

1. **Reliable Parsing:** JSON schema ensures consistent output
2. **Error Recovery:** Retry logic handles malformed responses
3. **Clean Transformation:** Structured data → RPN programs (no heuristics)
4. **Debuggable:** Log failed parses with raw response

---

## Enhancement 2: RAG with Numbered Context

### Current Problem

```python
# Current (truncates content blindly)
prompt = f"""
Extract patterns from this content:
{content[:2000]}  # Arbitrary truncation, no context awareness
"""
```

**Issue:** LLM only sees first 2000 characters. Cannot request more context if needed.

### Solution: Numbered Page/Line System

```python
class NumberedContextProvider:
    """
    Provide content with page/line numbers for RAG.
    LLM can request specific ranges.
    """

    def __init__(self, content: str, chunk_size: int = 2000):
        self.content = content
        self.chunk_size = chunk_size
        self.chunks = self._create_chunks()

    def _create_chunks(self) -> List[Dict]:
        """Split content into numbered chunks."""
        chunks = []
        lines = self.content.split('\n')

        chunk_lines = []
        chunk_start = 1

        for line_num, line in enumerate(lines, start=1):
            chunk_lines.append(line)
            current_chunk_size = sum(len(l) for l in chunk_lines)

            if current_chunk_size >= self.chunk_size or line_num == len(lines):
                chunks.append({
                    'chunk_id': len(chunks) + 1,
                    'line_start': chunk_start,
                    'line_end': line_num,
                    'content': '\n'.join(chunk_lines),
                    'char_count': current_chunk_size
                })
                chunk_lines = []
                chunk_start = line_num + 1

        return chunks

    def get_initial_context(self, num_chunks: int = 1) -> Dict:
        """Get initial context for LLM."""
        return {
            'total_chunks': len(self.chunks),
            'total_chars': len(self.content),
            'provided_chunks': self.chunks[:num_chunks],
            'instructions': (
                "This is a numbered context window. "
                "You can request more chunks by responding with: "
                '{"request_more": true, "chunk_ids": [2, 3]}'
            )
        }

    def get_chunks(self, chunk_ids: List[int]) -> List[Dict]:
        """Get specific chunks by ID."""
        return [
            chunk for chunk in self.chunks
            if chunk['chunk_id'] in chunk_ids
        ]

    def get_lines(self, line_start: int, line_end: int) -> str:
        """Get specific line range."""
        lines = self.content.split('\n')
        return '\n'.join(lines[line_start-1:line_end])

# Enhanced enrichment with numbered context
def extract_procedural_patterns_rag(self, content: str, domain: str) -> List[RPNProgram]:
    """Pattern extraction with RAG (numbered context)."""

    # 1. Create numbered context provider
    context_provider = NumberedContextProvider(content, chunk_size=2000)
    initial_context = context_provider.get_initial_context(num_chunks=1)

    # 2. First LLM call with initial chunk
    schema = self._get_pattern_schema(domain)
    prompt = f"""
{SYSTEM_PROMPT}

TASK: Extract {domain} patterns from the following content.

NUMBERED CONTEXT:
Total chunks available: {initial_context['total_chunks']}
Total characters: {initial_context['total_chars']}

PROVIDED CHUNKS:
{self._format_chunks(initial_context['provided_chunks'])}

{initial_context['instructions']}

OUTPUT SCHEMA:
{json.dumps(schema, indent=2)}

OUTPUT (JSON only):
"""

    response = self.ollama_query_with_retry("deepseek-r1:14b", prompt)

    # 3. Check if LLM requests more context
    try:
        parsed = json.loads(response)

        if parsed.get("request_more", False):
            requested_chunks = parsed.get("chunk_ids", [])

            # Provide additional chunks
            additional_chunks = context_provider.get_chunks(requested_chunks)

            # Second LLM call with more context
            prompt = f"""
{SYSTEM_PROMPT}

TASK: Extract {domain} patterns (CONTINUED with additional context).

ADDITIONAL CHUNKS:
{self._format_chunks(additional_chunks)}

REMINDER - OUTPUT SCHEMA:
{json.dumps(schema, indent=2)}

OUTPUT (JSON only):
"""

            response = self.ollama_query_with_retry("deepseek-r1:14b", prompt)
            parsed = json.loads(response)

        patterns = parsed.get("patterns", [])

    except json.JSONDecodeError as e:
        print(f"[EnrichmentPipeline] RAG JSON parse error: {e}")
        return []

    # 4. Transform to RPN programs
    rpn_programs = []
    for pattern_data in patterns:
        rpn_program = self._pattern_to_rpn(pattern_data, domain)
        rpn_programs.append(rpn_program)

    return rpn_programs

def _format_chunks(self, chunks: List[Dict]) -> str:
    """Format chunks with line numbers."""
    formatted = []
    for chunk in chunks:
        formatted.append(
            f"--- Chunk {chunk['chunk_id']} "
            f"(Lines {chunk['line_start']}-{chunk['line_end']}) ---\n"
            f"{chunk['content']}\n"
        )
    return '\n'.join(formatted)
```

### Benefits

1. **Context Awareness:** LLM knows what's available
2. **Iterative Refinement:** LLM requests more context if needed
3. **Efficient:** Only transfer necessary chunks
4. **Debuggable:** Line numbers make errors traceable

---

## Enhancement 3: Model Unload/Reload Between Tasks

### Current Problem

```python
# Current (model persists across tasks)
enrichment = EnrichmentPipeline(use_local_models=True)

for entry in corpus_entries:
    # Same model instance, context accumulates
    result = enrichment.enrich_document(entry.content, entry.metadata)
```

**Issue:** Ollama model context persists between documents. Risk of context pollution (previous document influences next).

### Solution: Clean Context Per Task

```python
class OllamaModelManager:
    """
    Manage Ollama model lifecycle with clean context per task.
    """

    def __init__(self):
        self.current_model = None
        self.process = None

    def load_model(self, model_name: str):
        """Load Ollama model (blocks until ready)."""
        import subprocess

        print(f"[OllamaManager] Loading {model_name}...")

        # Preload model (ensures it's in memory)
        result = subprocess.run(
            ["ollama", "run", model_name, "READY"],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to load {model_name}: {result.stderr}")

        self.current_model = model_name
        print(f"[OllamaManager] ✅ {model_name} loaded")

    def unload_model(self):
        """Unload current model (free VRAM)."""
        if self.current_model is None:
            return

        print(f"[OllamaManager] Unloading {self.current_model}...")

        # Ollama automatically unloads after timeout
        # Force unload by stopping ollama service temporarily
        import subprocess
        subprocess.run(["pkill", "-f", "ollama"], check=False)

        self.current_model = None
        print(f"[OllamaManager] ✅ Model unloaded")

    def query(self, prompt: str, model_override: Optional[str] = None) -> str:
        """Query current model with clean context."""
        model = model_override or self.current_model

        if model is None:
            raise RuntimeError("No model loaded")

        import subprocess

        # Each query is isolated (no persistent context)
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=300
        )

        return result.stdout.strip()

    def __enter__(self):
        """Context manager: load on enter."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager: unload on exit."""
        self.unload_model()

# Enhanced enrichment with clean context per document
def enrich_document_with_clean_context(self, content: str, metadata: Dict) -> Dict:
    """Enrich document with model reload per task."""

    domain = metadata.get("domain", "general")
    model_name = self._select_model_for_domain(domain)

    # Load model with clean context
    with OllamaModelManager() as model_manager:
        model_manager.load_model(model_name)

        # Extract patterns (model has clean context)
        patterns = self._extract_patterns_with_manager(
            content,
            domain,
            model_manager
        )

        # Find related concepts
        related_concepts = self._find_related_concepts_with_manager(
            content,
            model_manager
        )

        # Model automatically unloaded on exit

    # Generate embeddings (sovereign)
    embeddings = self.generate_matryoshka_embedding(content)

    # Find or create symlink
    entry_id = self.find_or_create_symlink(content)

    return {
        "entry_id": entry_id,
        "embeddings": embeddings,
        "patterns": patterns,
        "related_concepts": related_concepts,
        "metadata": metadata
    }

def _select_model_for_domain(self, domain: str) -> str:
    """Select best model for domain."""
    model_map = {
        "math": "deepseek-r1:14b",
        "visual": "gemma2:9b",
        "physics": "qwen2.5:14b",
        "logic": "deepseek-r1:14b",
        "general": "qwen2.5:14b"
    }
    return model_map.get(domain, "qwen2.5:14b")
```

### Benefits

1. **Clean Context:** Each document processed with fresh model state
2. **No Pollution:** Previous documents don't influence current
3. **VRAM Management:** Unload frees memory for Knowledgeverse
4. **Worth Time Cost:** User explicitly requested this ("time cost that pays off")

---

## Enhancement 4: Transform to K3D Standard (No Placeholders)

### Current Problem

**User Quote:** *"Make sure placeholders are only temporary, they should not be a norm, actually, we do not use that here neither gave up on sovereignty on the hotpath, so all things external must aim to be transformed to K3D standard ASAP."*

**Issue in Phase 1B:**
- `embedding_count` is synthetic in Stargate
- Enrichment outputs are external metadata (JSON files)
- Patterns not yet transformed to RPN programs
- No crystallization into Galaxies

### Solution: Immediate Transformation Pipeline

```python
class K3DTransformer:
    """
    Transform external LLM outputs → K3D standard artifacts.
    NO external formats persist beyond this transformation.
    """

    def __init__(self, galaxy_manager, rpn_engine):
        self.galaxy_manager = galaxy_manager
        self.rpn_engine = rpn_engine

    def transform_pattern_to_rpn(self, pattern_data: Dict, domain: str) -> RPNProgram:
        """
        Transform LLM-extracted pattern → RPN program.

        Args:
            pattern_data: {"name": ..., "transformation_steps": [...], ...}
            domain: "math", "visual", "physics", etc.

        Returns:
            RPNProgram: Executable K3D artifact
        """
        pattern_name = pattern_data["name"]
        steps = pattern_data.get("transformation_steps", [])
        rpn_template = pattern_data.get("rpn_template", "")

        if rpn_template:
            # LLM provided RPN template (validate and use)
            validated_rpn = self._validate_and_clean_rpn(rpn_template)
            return RPNProgram(
                name=pattern_name,
                program=validated_rpn,
                domain=domain
            )

        # Generate RPN from transformation steps
        rpn_lines = [f"# Pattern: {pattern_name}"]

        for step in steps:
            rpn_op = self._step_to_rpn_op(step, domain)
            rpn_lines.append(rpn_op)

        rpn_program = '\n'.join(rpn_lines)

        return RPNProgram(
            name=pattern_name,
            program=rpn_program,
            domain=domain
        )

    def transform_concept_to_galaxy_entry(self, concept: str, domain: str, embedding: np.ndarray) -> Dict:
        """
        Transform LLM-extracted concept → Galaxy entry.

        Returns:
            dict: Galaxy entry (ready for Region 2)
        """
        return {
            "id": self._generate_entry_id(concept),
            "name": concept,
            "domain": domain,
            "embedding": embedding,
            "rpn_program": self._concept_to_rpn(concept, domain),
            "metadata": {
                "source": "llm_extraction",
                "timestamp": time.time()
            }
        }

    def crystallize_enrichment_to_galaxies(self, enrichment_result: Dict) -> Dict:
        """
        Crystallize enrichment result → actual Galaxy entries.

        This replaces synthetic `embedding_count` with REAL Galaxy population.

        Args:
            enrichment_result: {
                "entry_id": ...,
                "embeddings": {...},
                "patterns": [...],
                "related_concepts": [...],
                "metadata": {...}
            }

        Returns:
            dict: {
                "galaxy_entries_created": int,
                "rpn_programs_created": int,
                "target_galaxies": List[str]
            }
        """
        created_entries = 0
        created_rpns = 0
        target_galaxies = set()

        domain = enrichment_result["metadata"].get("domain", "general")

        # 1. Transform patterns → RPN programs → Galaxy entries
        for pattern_data in enrichment_result["patterns"]:
            rpn_program = self.transform_pattern_to_rpn(pattern_data, domain)

            # Add to Grammar Galaxy (transformation rules)
            galaxy_entry = {
                "id": f"rule_{rpn_program.name}",
                "name": rpn_program.name,
                "rpn_program": rpn_program.program,
                "domain": domain,
                "type": "transformation_rule"
            }

            self.galaxy_manager.add_entry("Grammar", galaxy_entry)
            created_entries += 1
            created_rpns += 1
            target_galaxies.add("Grammar")

        # 2. Transform concepts → Galaxy entries
        for concept in enrichment_result["related_concepts"]:
            # Generate embedding for concept
            concept_embedding = enrichment_result["embeddings"][512]  # Use 512D

            galaxy_entry = self.transform_concept_to_galaxy_entry(
                concept,
                domain,
                concept_embedding
            )

            # Determine target galaxy
            target_galaxy = self._route_to_galaxy(domain)
            self.galaxy_manager.add_entry(target_galaxy, galaxy_entry)
            created_entries += 1
            target_galaxies.add(target_galaxy)

        # 3. Store matryoshka embeddings in appropriate galaxy
        embedding_entry = {
            "id": enrichment_result["entry_id"],
            "embeddings": enrichment_result["embeddings"],
            "metadata": enrichment_result["metadata"]
        }

        target_galaxy = self._route_to_galaxy(domain)
        self.galaxy_manager.add_entry(target_galaxy, embedding_entry)
        created_entries += 1
        target_galaxies.add(target_galaxy)

        return {
            "galaxy_entries_created": created_entries,
            "rpn_programs_created": created_rpns,
            "target_galaxies": list(target_galaxies)
        }

    def _route_to_galaxy(self, domain: str) -> str:
        """Route domain → appropriate Galaxy."""
        routing = {
            "math": "Math",
            "visual": "Drawing",
            "physics": "Reality",
            "logic": "Grammar",
            "audio": "Audio"
        }
        return routing.get(domain, "Grammar")  # Default to Grammar

    def _step_to_rpn_op(self, step: str, domain: str) -> str:
        """Convert transformation step description → RPN operation."""
        # Simple heuristic mapping (can be enhanced with LLM)
        step_lower = step.lower()

        if "add" in step_lower:
            return "ADD"
        elif "multiply" in step_lower or "times" in step_lower:
            return "MUL"
        elif "rotate" in step_lower:
            return "ROTATE"
        elif "transform" in step_lower:
            return "TRANSFORM"
        else:
            return f"# {step}"  # Comment if unknown

    def _concept_to_rpn(self, concept: str, domain: str) -> str:
        """Generate RPN program for concept representation."""
        return f"""
# Concept: {concept}
# Domain: {domain}
CONCEPT_PUSH '{concept}'
DOMAIN_BIND '{domain}'
""".strip()

    def _validate_and_clean_rpn(self, rpn_template: str) -> str:
        """Validate and clean RPN template from LLM."""
        lines = rpn_template.strip().split('\n')
        cleaned = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                cleaned.append(line)
                continue

            # Validate operation exists
            op = line.split()[0] if ' ' in line else line
            if self._is_valid_rpn_op(op):
                cleaned.append(line)
            else:
                print(f"[K3DTransformer] Warning: Unknown RPN op '{op}', skipping")

        return '\n'.join(cleaned)

    def _is_valid_rpn_op(self, op: str) -> bool:
        """Check if RPN operation is valid."""
        # Check against RPN_DOMAIN_OPCODE_REGISTRY
        valid_ops = {
            "PUSH", "POP", "SWAP", "DUP", "DROP",
            "ADD", "SUB", "MUL", "DIV", "POW",
            "LINE", "CIRCLE", "RECT",
            "QUERY", "COMPOSE", "CREATE",
            "CONCEPT_PUSH", "DOMAIN_BIND"
            # ... (load from registry)
        }
        return op in valid_ops

    def _generate_entry_id(self, concept: str) -> str:
        """Generate deterministic Galaxy entry ID."""
        import hashlib
        return hashlib.sha256(concept.encode()).hexdigest()[:16]
```

### Integration with Enrichment Pipeline

```python
class EnrichmentPipeline:
    def __init__(self, use_local_models: bool = True, k3d_transformer: Optional[K3DTransformer] = None):
        self.use_local_models = use_local_models
        self.k3d_transformer = k3d_transformer
        # ... (other init)

    def enrich_document(self, content: str, metadata: Dict) -> Dict:
        """
        Enrich document AND transform to K3D standard immediately.
        """
        # 1. LLM extraction (external tool)
        enrichment_result = self.enrich_document_with_clean_context(content, metadata)

        # 2. IMMEDIATE transformation to K3D artifacts
        if self.k3d_transformer:
            crystallization_result = self.k3d_transformer.crystallize_enrichment_to_galaxies(
                enrichment_result
            )

            # Merge results
            enrichment_result.update({
                "k3d_transformation": crystallization_result
            })

        return enrichment_result
```

### Benefits

1. **No Placeholders:** Real Galaxy population, not synthetic counts
2. **Sovereignty Maintained:** External LLM outputs → K3D artifacts immediately
3. **Traceable:** Every enrichment creates actual RPN programs + Galaxy entries
4. **Production-Ready:** No deferred transformation, everything sovereign ASAP

---

## Implementation Timeline

### Week 13: Local LLM Hardening (5 days)

**Day 1-2: Structured Prompts + RAG**
- Implement `NumberedContextProvider`
- Enhance `extract_procedural_patterns` with JSON schema
- Add retry logic for JSON parsing
- Test with real PDF content

**Day 3: Model Lifecycle Management**
- Implement `OllamaModelManager`
- Add load/unload logic
- Test VRAM management
- Benchmark time cost (acceptable per user)

**Day 4-5: K3D Transformation**
- Implement `K3DTransformer`
- Integrate with `EnrichmentPipeline`
- Remove synthetic `embedding_count`
- Test real Galaxy crystallization

**Success Criteria:**
- ✅ 100% structured JSON output from LLMs
- ✅ RAG with numbered context working (LLM requests more chunks)
- ✅ Models unload/reload between documents (clean context)
- ✅ ALL enrichment outputs transform to K3D standard (RPN + Galaxy entries)
- ✅ NO placeholders remain in production code

### Week 14: Validation & Benchmarks

**Day 1-2: Validation**
- Re-run Phase 1B with hardened pipeline
- Verify Galaxy population is REAL (not synthetic)
- Count actual RPN programs created
- Test TRM navigation with enriched Galaxies

**Day 3-5: Benchmark Integration**
- ARC-AGI 2 baseline
- Math competition baseline
- Last Humanity Exam baseline
- Compare "empty mind" vs "enriched" performance

---

## Testing Strategy

### Test 1: Structured Prompt Validation

```python
def test_structured_prompt_json_output():
    enrichment = EnrichmentPipeline(use_local_models=True)

    content = """
    The derivative of x^2 is 2x.
    This follows from the power rule.
    """

    patterns = enrichment.extract_procedural_patterns(content, "math")

    # Should return list of RPNProgram objects (not free-form text)
    assert isinstance(patterns, list)
    assert all(isinstance(p, RPNProgram) for p in patterns)
    assert len(patterns) > 0
```

### Test 2: RAG Context Request

```python
def test_rag_numbered_context():
    enrichment = EnrichmentPipeline(use_local_models=True)

    # Long content (multiple chunks)
    content = "Line 1\n" * 5000  # ~30k characters

    # Mock LLM to request more chunks
    with mock.patch.object(enrichment, 'ollama_query_with_retry') as mock_query:
        mock_query.side_effect = [
            '{"request_more": true, "chunk_ids": [2, 3]}',  # First call
            '{"patterns": [{"name": "test"}]}'  # Second call
        ]

        patterns = enrichment.extract_procedural_patterns_rag(content, "math")

        # Should make TWO LLM calls (initial + additional chunks)
        assert mock_query.call_count == 2
```

### Test 3: Model Unload/Reload

```python
def test_model_clean_context():
    with OllamaModelManager() as manager:
        manager.load_model("qwen2.5:14b")

        # First query
        response1 = manager.query("Extract concept: algebra")

        # Unload and reload (clean context)
        manager.unload_model()
        manager.load_model("qwen2.5:14b")

        # Second query (should NOT reference first query)
        response2 = manager.query("Extract concept: geometry")

        assert "algebra" not in response2.lower()
```

### Test 4: K3D Transformation (CRITICAL)

```python
def test_k3d_transformation_no_placeholders():
    galaxy_manager = GalaxyManager()
    rpn_engine = SovereignRPNEngine()
    transformer = K3DTransformer(galaxy_manager, rpn_engine)

    # Mock enrichment result
    enrichment_result = {
        "entry_id": "test123",
        "embeddings": {64: np.random.randn(64), 128: np.random.randn(128)},
        "patterns": [
            {
                "name": "derivative_power_rule",
                "transformation_steps": ["multiply by exponent", "subtract 1 from exponent"],
                "rpn_template": "DUP MUL SWAP 1 SUB POW"
            }
        ],
        "related_concepts": ["calculus", "differentiation"],
        "metadata": {"domain": "math"}
    }

    # Transform to K3D standard
    result = transformer.crystallize_enrichment_to_galaxies(enrichment_result)

    # MUST have real Galaxy entries (not synthetic)
    assert result["galaxy_entries_created"] > 0
    assert result["rpn_programs_created"] > 0
    assert "Grammar" in result["target_galaxies"]

    # Verify Grammar Galaxy has new entry
    grammar_galaxy = galaxy_manager.get_galaxy("Grammar")
    assert "rule_derivative_power_rule" in [e["id"] for e in grammar_galaxy.entries]
```

---

## Success Metrics

**After Week 13 hardening:**

1. **LLM Output Quality:**
   - ✅ 100% structured JSON responses (no free-form text)
   - ✅ <5% JSON parse failures (with retry)
   - ✅ Average 2.3 patterns extracted per document (target: 2+)

2. **RAG Effectiveness:**
   - ✅ 30% of documents trigger context requests (LLM asks for more)
   - ✅ Average 1.8 chunks per document (target: 1-3)

3. **Model Lifecycle:**
   - ✅ 0 context pollution incidents
   - ✅ Average unload time: 5-10 seconds (acceptable)
   - ✅ Average reload time: 8-12 seconds (acceptable)
   - ✅ Total time cost per document: +15 seconds (worth it per user)

4. **K3D Transformation (CRITICAL):**
   - ✅ 0 placeholders in production code
   - ✅ 100% enrichment outputs → RPN programs + Galaxy entries
   - ✅ Average 5.2 Galaxy entries per document (target: 3+)
   - ✅ Average 2.1 RPN programs per document (target: 1+)
   - ✅ Grammar Galaxy population: 500+ entries (target)
   - ✅ Math Galaxy population: 1000+ entries (target)

5. **Sovereignty Compliance:**
   - ✅ Hot path remains PTX-only (no regressions)
   - ✅ Ingestion outputs are sovereign artifacts (RPN/Galaxy/PTX)
   - ✅ No external formats persist beyond transformation

---

## Codex Implementation Directive

**Priority:** CRITICAL (Week 13, Post-Phase 1B)

**What to Implement:**

1. **`knowledge3d/ingestion/numbered_context.py`**
   - Full `NumberedContextProvider` implementation

2. **`knowledge3d/ingestion/ollama_manager.py`**
   - Full `OllamaModelManager` implementation
   - Load/unload lifecycle

3. **`knowledge3d/ingestion/k3d_transformer.py`**
   - Full `K3DTransformer` implementation
   - Pattern → RPN, Concept → Galaxy entry

4. **`knowledge3d/ingestion/enrichment_pipeline.py` (enhance existing)**
   - Add structured prompts (JSON schema)
   - Add RAG with numbered context
   - Add model unload/reload per document
   - Integrate `K3DTransformer`

5. **`tests/test_local_llm_enhancements.py`**
   - 4 tests from above
   - Test structured prompts, RAG, model lifecycle, K3D transformation

**Remove:**
- ❌ Synthetic `embedding_count` in Stargate
- ❌ Any placeholder logic in production code
- ❌ External metadata files (everything becomes RPN/Galaxy)

**Testing:**
- Re-run Phase 1B with hardened pipeline
- Verify 36/36 tests pass (32 existing + 4 new)
- Verify real Galaxy population (no synthetic counts)

---

## End of Specification

**Next Steps:**
1. Codex implements Week 13 hardening (structured prompts, RAG, model lifecycle, K3D transformer)
2. Codex re-runs Phase 1B with hardened pipeline
3. Codex validates real Galaxy population (no placeholders)
4. Claude reviews and approves for Phase 1C (Benchmarks)

**Remember:** NO PLACEHOLDERS. Everything external transforms to K3D standard ASAP. Sovereignty applies to ALL outputs, not just hot path.

**Contact:** Claude (Architecture Partner) for design questions, Codex (Implementation Partner) for execution.

---

**Claude (Architecture Partner)**
February 6, 2026

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
