# Kimi Agent Swarm Result
**Task:** Design a "smart proceduralizer" architecture for the K3D project. It must enrich existing Galaxy entries (Grammar.jsonl, proceduralized_*.jsonl) using Ollama cloud models (gpt-oss, qwen3.5:397b-cloud)
**Mode:** thinking
**Time:** 2026-04-20 17:55:08

────────────────────────────────────────────────────────────────

## Sub-Agent A: ENRICHMENT CORRECTNESS PERSPECTIVE. Design the per-entry prompt chain that takes one raw Grammar/proceduralized entry and produces the enriched version. Cover: (1) prompt template fed to the LLM with the opcode catalog + arg_keys contract + symlink rules; (2) schema validator that rejects LLM outputs failing the binder contract; (3) retry/repair strategy when output is malformed (ask the LLM to fix specific validation errors); (4) dedup/conflict strategy when two entries synthesize to the same eval_program but with different supervision answers; (5) symlink composition rules (meaning-first, bidirectional, canonical IDs per K3D spec). Specifically address: how does the LLM know the difference between a correct and incorrect eval_program without executing it? What static checks can we do before runtime validation?

**Sub-Agent A Output: Enrichment Correctness Perspective**

## 1. Prompt Template Architecture

### System Prompt (Injected Context)
```xml
<K3D_PROCEDURALIZER_CONTEXT>
  <DOMAIN>MATH_RPN</DOMAIN>
  <BINDER_CONTRACT>
    <ARG_KEYS_PATTERN>^[a-z][a-z0-9_]*$</ARG_KEYS_PATTERN>
    <PLACEHOLDER_FORMAT>ARG_{KEY.upper()}</PLACEHOLDER_FORMAT>
    <REPLACEMENT_RULE>program.replace(f"ARG_{key.upper()}", f"{numeric_value:g}")</REPLACEMENT_RULE>
    <OUTPUT_SCHEMA>eval_program must be space-delimited RPN tokens</OUTPUT_SCHEMA>
  </BINDER_CONTRACT>
  
  <OPCODE_CATALOG subset="math_core">
    <!-- Injected from docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md §11 -->
    <OPCODE id="0x1101" symbol="+" arity="2" type="numeric" unit_preserving="true"/>
    <OPCODE id="0x1102" symbol="-" arity="2" type="numeric"/>
    <OPCODE id="0x1103" symbol="*" arity="2" type="numeric"/>
    <OPCODE id="0x1104" symbol="/" arity="2" type="numeric" guard="divisor!=0"/>
    <OPCODE id="0x1105" symbol="POW" arity="2" type="numeric"/>
    <OPCODE id="0x1201" symbol="SQRT" arity="1" type="numeric" guard=">=0"/>
    <OPCODE id="0x1301" symbol="DUP" arity="1" effect="stack+1"/>
    <OPCODE id="0x1302" symbol="SWAP" arity="2" effect="reorder"/>
  </OPCODE_CATALOG>

  <SYMLINK_RULES>
    <CANONICAL_ID_FORMAT>sha256(domain:meaning_normalized)[:16]</CANONICAL_ID_FORMAT>
    <DIRECTIONALITY>bidirectional</DIRECTIONALITY>
    <MEANING_EXTRACTION>Strip ARG_* placeholders, hash remaining opcode sequence</MEANING_EXTRACTION>
  </SYMLINK_RULES>

  <VALIDATION_REQUIREMENTS>
    <STATIC>Stack depth final=1, no underflow, all tokens valid</STATIC>
    <SEMANTIC>If supervision_answer provided, symbolic evaluation must match</SEMANTIC>
  </VALIDATION_REQUIREMENTS>
</K3D_PROCEDURALIZER_CONTEXT>
```

### User Prompt Template (Per-Entry)
```xml
<ENTRY_INPUT>
  <RAW_GRAMMAR id="{entry_id}">{original_text}</RAW_GRAMMAR>
  <PROCEDURALIZED_RPN>{natural_language_rpn_or_null}</PROCEDURALIZED_RPN>
  <SUPERVISION answer="{gold_answer}" available="{bool}"/>
  <CONTEXT domain="{domain}" difficulty="{level}"/>
</ENTRY_INPUT>

<TASK>
Convert the proceduralized description into executable RPN following the binder contract.
1. Identify arg_keys from problem text (e.g., "number" → "n", "original_amount" → "original_amount")
2. Construct eval_program using ARG_N, ARG_ORIGINAL_AMOUNT, etc. as placeholders
3. Ensure final stack depth = 1
4. Populate rule_strength (0.0-1.0) based on ambiguity
5. List superior_to entry IDs that this supersedes (if known)

Static correctness checks (perform before output):
- Verify every operation has sufficient stack depth
- Check that ARG_* placeholders match declared arg_keys
- Ensure no undefined opcodes
- If supervision_answer exists: simulate with symbolic values to verify logic
</TASK>

<OUTPUT_FORMAT>
```json
{
  "entry_id": "string",
  "arg_keys": ["string"],
  "eval_program": "ARG_N 2 / ARG_N +",
  "rule_strength": 0.95,
  "superior_to": ["entry_id_1", "entry_id_2"],
  "meaning_hash": "sha256_prefix",
  "symlink_targets": ["canonical_meaning_id"],
  "static_check_passed": true,
  "validation_notes": "string"
}
```
</OUTPUT_FORMAT>
```

## 2. Schema Validator (Binder Contract Enforcer)

```python
# enrichment/validators/binder_contract.py
import re
from typing import List, Tuple, Optional

class BinderContractValidator:
    def __init__(self, opcode_registry_path: str = "docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md"):
        self.opcodes = self._load_opcodes(opcode_registry_path)
        self.arg_pattern = re.compile(r'^ARG_([A-Z][A-Z0-9_]*)$')
        
    def validate(self, entry: dict) -> Tuple[bool, List[str], Optional[dict]]:
        """
        Returns: (is_valid, error_messages, repaired_entry)
        """
        errors = []
        
        # 1. Arg Keys Contract
        if not entry.get('arg_keys'):
            errors.append("MISSING_ARG_KEYS: Must declare query slots")
        else:
            for key in entry['arg_keys']:
                if not re.match(r'^[a-z][a-z0-9_]*$', key):
                    errors.append(f"INVALID_ARG_KEY: '{key}' must match ^[a-z][a-z0-9_]*$")
        
        # 2. Placeholder Resolution Check
        program = entry.get('eval_program', '')
        declared_placeholders = {f"ARG_{k.upper()}" for k in entry.get('arg_keys', [])}
        used_placeholders = set(self.arg_pattern.findall(program))
        
        missing = declared_placeholders - used_placeholders
        if missing:
            errors.append(f"UNUSED_ARGS: {missing} declared but not used")
            
        undeclared = used_placeholders - declared_placeholders
        if undeclared:
            errors.append(f"UNDECLARED_ARGS: {undeclared} used but not in arg_keys")
        
        # 3. RPN Static Analysis (Stack Simulation)
        tokens = program.split()
        stack_depth = 0
        max_depth = 0
        
        for token in tokens:
            if self.arg_pattern.match(token):
                stack_depth += 1
            elif token in self.opcodes:
                arity = self.opcodes[token]['arity']
                if stack_depth < arity:
                    errors.append(f"STACK_UNDERFLOW: '{token}' requires {arity}, have {stack_depth}")
                stack_depth = stack_depth - arity + 1  # Pop arity, push result
            else:
                errors.append(f"UNKNOWN_OPCODE: '{token}' not in registry")
            
            max_depth = max(max_depth, stack_depth)
        
        if stack_depth != 1:
            errors.append(f"INVALID_FINAL_STACK: depth={stack_depth}, expected=1")
        
        # 4. Defeasibility Schema
        if not isinstance(entry.get('rule_strength'), (int, float)):
            errors.append("INVALID_RULE_STRENGTH: must be numeric")
        elif not (0.0 <= entry['rule_strength'] <= 1.0):
            errors.append("RULE_STRENGTH_RANGE: must be in [0.0, 1.0]")
            
        return (len(errors) == 0, errors, None)
    
    def _load_opcodes(self, path: str) -> dict:
        # Parse registry markdown §11
        # Returns: {symbol: {arity: int, id: hex, ...}}
        pass
```

## 3. Retry/Repair Strategy

```python
# enrichment/pipeline/repair_orchestrator.py

class RepairStrategy:
    MAX_RETRIES = 3
    ESCALATION_MODELS = [
        "qwen3.5:397b-cloud",      # Tier 1: Fast/Cheap
        "kimi-k2-thinking:cloud",   # Tier 2: Reasoning
        "kimi-k2:1t-cloud"          # Tier 3: Sovereign correctness
    ]
    
    def repair_entry(self, entry: dict, errors: List[str], attempt: int = 0):
        if attempt >= self.MAX_RETRIES:
            return self._quarantine(entry, errors)
        
        # Structured error feedback
        repair_prompt = f"""
        <VALIDATION_ERRORS>
        {chr(10).join(f"<ERROR type='{e.split(':')[0]}'>{e}</ERROR>" for e in errors)}
        </VALIDATION_ERRORS>
        
        <ORIGINAL_ATTEMPT>
        {json.dumps(entry, indent=2)}
        </ORIGINAL_ATTEMPT>
        
        <REPAIR_INSTRUCTIONS>
        1. Fix specific validation errors listed above
        2. If STACK_UNDERFLOW: Ensure sufficient DUP/SWAP operations or reorder operations
        3. If UNDECLARED_ARGS: Add missing keys to arg_keys or fix placeholder spelling
        4. If UNKNOWN_OPCODE: Replace with registered equivalent from catalog
        </REPAIR_INSTRUCTIONS>
        """
        
        model = self.ESCALATION_MODELS[min(attempt, len(self.ESCALATION_MODELS)-1)]
        
        # Call Ollama MCP
        response

────────────────────────────────────────────────────────────────

## Sub-Agent B: INGESTION SCALABILITY PERSPECTIVE. Design the batch pipeline for enriching 103k Grammar + ~10k proceduralized entries through Ollama cloud models. Cover: (1) batching strategy (one entry per call vs N-entry batch prompt); (2) which model for which entry class (gpt-oss for short/simple, qwen3.5:397b-cloud for complex, kimi-k2-thinking for edge cases); (3) caching / idempotency so re-runs skip already-enriched entries (content hash on raw input); (4) observability — how to track success rate per entry source, per model, per benchmark split; (5) failure modes and how to isolate them (LLM hallucinates opcode not in registry, LLM emits arg_keys that don't match query numbers, LLM fabricates a symlink target that doesn't exist); (6) backpressure — how to handle Ollama cloud rate limits. Specifically address: what's the minimum viable pipeline (one script, one day of runtime) vs the ideal (incremental, re-runnable, resumable)?

**Sub-Agent B: Ingestion Scalability Perspective**

## 1. Batching Strategy: Adaptive Token-Budget Batching

**Decision**: Use **heterogeneous batching** by entry class rather than uniform batch sizes.

| Entry Class | Batch Size | Rationale | Prompt Structure |
|-------------|------------|-----------|------------------|
| **Grammar** (rule_strength/superior_to) | 20 entries | Low token per entry (~200), deterministic output | JSONL array format with `entry_id` prefix |
| **Math Proceduralization** | 1-2 entries | High token (natural language RPN), high variance in output length | Individual XML-wrapped prompt with `<input_program>` tags |
| **Edge Cases** (supervision misalignment) | 1 entry | Requires chain-of-thought reasoning, context-heavy | Zero-shot with retrieval-augmented context |

**Token Budget Math**:
- Target: 4k input + 2k output per batch (75% of 8k context for safety)
- Grammar entry avg: 150 tokens → 20 × 150 = 3k + instructions = 3.8k ✓
- Math entry avg: 800 tokens (natural language RPN) → 2 × 800 = 1.6k + examples = 2.4k ✓

**Implementation Detail**:
```python
# ingestion/batcher.py:42
def calculate_batch(entries: List[RawEntry]) -> List[Batch]:
    buckets = defaultdict(list)
    for e in entries:
        est_tokens = estimate_tokens(e.raw_program)  # tiktoken approximation
        complexity = classify_complexity(e)  # heuristic: presence of "equals_total_sum" etc.
        buckets[complexity].append((e, est_tokens))
    
    # Greedy bin packing per complexity class
    for complexity, items in buckets.items():
        yield from pack_bins(items, max_tokens=4000)
```

## 2. Model Routing Matrix

| Entry Characteristics | Assigned Model | Fallback | Routing Logic |
|----------------------|----------------|----------|---------------|
| `len(raw_text) < 500` AND `opcode_count < 3` | `gpt-oss:120b-cloud` | `qwen3.5:397b-cloud` | Fast path for Grammar entries |
| `rpn_program` contains natural language verbs | `qwen3.5:397b-cloud` | `deepseek-v3.1:671b-cloud` | Code/math translation task |
| `supervision_answer` exists but `query_hash` mismatch | `kimi-k2-thinking:cloud` | Manual queue | Ambiguous alignment requiring reasoning |
| Validation failed (hallucinated opcode) | `kimi-k2-thinking:cloud` (retry) | Dead letter | Re-process with temperature 0.1 + registry context |

**Router Location**: `ingestion/router/entry_classifier.py` — lightweight heuristic (regex for "divide_by_two" etc.) + embedding similarity to known patterns.

## 3. Caching & Idempotency

**Content Hash Strategy**:
- Hash input: `SHA256(canonical_json(entry.raw_input) + "|" + ENRICHMENT_SCHEMA_VERSION)`
- Store in: `state/enrichment_checkpoint.sqlite` (table: `enrichment_jobs`)

**Schema**:
```sql
-- state/schema.sql:15
CREATE TABLE enrichment_jobs (
    content_hash BLOB PRIMARY KEY,  -- 32 bytes SHA256
    entry_source VARCHAR(50),       -- 'grammar' or 'proceduralized_math'
    assigned_model VARCHAR(100),    -- 'qwen3.5:397b-cloud' etc.
    status ENUM('pending', 'success', 'failed', 'quarantined'),
    output_path VARCHAR(255),       -- s3:// or local path to enriched JSON
    validation_errors JSON,         -- array of error codes if failed
    attempt_count TINYINT DEFAULT 0,
    last_attempt TIMESTAMP,
    INDEX (status, entry_source)    -- for resumption queries
);
```

**Idempotency Flow**:
1. Pre-flight: Check hash. If `status='success'`, skip entirely.
2. If `status='failed'` AND `attempt_count < 3`, retry with exponential backoff.
3. If `attempt_count >= 3`, route to `kimi-k2-thinking` or manual queue.

**Resume Capability**:
```bash
# Resume from last checkpoint
python -m ingestion.pipeline --resume-from-checkpoint state/enrichment_checkpoint.sqlite
```

## 4. Observability Implementation

**Metrics Architecture**: Push-based via MCP to `kimi_swarm/ask_cloud` for aggregation.

**Key Metrics**:
- `enrichment_throughput_entries_per_second{model, source}`
- `validation_failure_rate{failure_type, model}`  // types: hallucinated_opcode, arg_mismatch, symlink_dne
- `ollama_latency_seconds{model, quantile="p99"}`
- `backpressure_wait_seconds`  // time spent in rate limit backoff

**Tracing**: Each entry gets a `trace_id` (UUIDv4) propagated through:

────────────────────────────────────────────────────────────────

## Synthesis

Here is the unified architecture specification, synthesizing correctness guarantees with scalable ingestion.

---

# K3D Smart Proceduralizer Architecture v1.0

## Executive Summary
Transform 113k dead entries (Grammar + proceduralized math) into sovereign, GPU-executable PTX templates using cloud LLMs. The architecture resolves the **supervision_answer miskeying** (problem_id vs query text) through a pre-flight alignment index, enforces **binder contract compliance** via stack-symbolic validation, and generates **meaning-star symlinks** for defeasible reasoning.

---

## Part 1: Correctness-First Design (The "Sovereign Validator" Path)

**Philosophy**: Zero Python fallbacks. Every entry must pass PTX-lowering validation before write. Single-entry processing to maximize traceability.

### 1.1 Prompt Chain (Per-Entry)

```xml
<K3D_SOVEREIGN_CONTEXT version="1.0">
  <BINDER_CONTRACT>
    <!-- Critical: Must match knowledgeverse.py:5931 replacement logic -->
    <PLACEHOLDER_PATTERN>ARG_{KEY.upper()}</PLACEHOLDER_PATTERN>
    <REPLACEMENT_SEMANTICS>literal_string_replace</REPLACEMENT_SEMANTICS>
    <ARG_KEY_REGEX>^[a-z][a-z0-9_]{0,15}$</ARG_KEY_REGEX>
  </BINDER_CONTRACT>
  
  <OPCODE_REGISTRY path="docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md" section="11">
    <!-- PTX-mapped opcodes only -->
    <OP symbol="+" ptx="add.f64" arity="2" stack_effect="-1"/>
    <OP symbol="DUP" ptx="mov.f64" arity="1" stack_effect="+1"/>
  </OPCODE_REGISTRY>

  <SUPERVISION_ALIGNMENT>
    <!-- Resolution for the 600-entry mismatch -->
    <QUERY_HASH_SHA256>canonical_query_text</QUERY_HASH_SHA256>
    <ALIGNMENT_LOOKUP>problem_id → query_hash</ALIGNMENT_LOOKUP>
  </SUPERVISION_ALIGNMENT>
</K3D_SOVEREIGN_CONTEXT>
```

**User Prompt Structure** (Single-entry XML):
```xml
<ENTRY uuid="{entry_id}" source="{grammar|proceduralized}">
  <RAW_TEXT>{original}</RAW_TEXT>
  <PROCEDURALIZED_RPN>{nl_rpn_or_null}</PROCEDURALIZED_RPN>
  <SUPERVISION problem_id="{pid}" answer="{val}" query_hash="{hash}"/>
  
  <TASK>
    1. Extract arg_keys from query text (e.g., "number" → "n")
    2. Translate natural language RPN to executable RPN using ARG_N placeholders
    3. Perform static stack simulation: verify final_depth=1, no underflow
    4. If supervision_answer provided: perform symbolic execution
       - Replace ARG_* with symbolic variables (σ0, σ1...)
       - Verify symbolic result equals supervision_answer expression
    5. Compute meaning_hash: SHA256(normalized_opcode_sequence)[0:16]
       - Normalized = eval_program with ARG_* replaced by "ARG"
  </TASK>
</ENTRY>
```

### 1.2 Three-Tier Validation Gate

```python
# enrichment/gates/sovereign_validator.py
class SovereignValidator:
    def validate(self, entry: EnrichedEntry) -> ValidationReport:
        # Gate 1: Static Binder Contract
        stack = []
        for token in entry.eval_program.split():
            if token.startswith("ARG_"):
                stack.append(SymbolicVar(token))  # Type tracking
            elif op := self.opcodes.get(token):
                if len(stack) < op.arity:
                    raise StackUnderflow(token)
                args = [stack.pop() for _ in range(op.arity)]
                stack.append(self.infer_return_type(op, args))
        
        if len(stack) != 1:
            raise InvalidFinalStack(len(stack))
        
        # Gate 2: Supervision Symbolic Verification (The "Correctness" Check)
        if entry.supervision_answer:
            symbolic_result = self.symbolic_execute(
                entry.eval_program, 
                entry.arg_keys,
                entry.supervision_answer  # Gold value
            )
            if not symbolic_match(symbolic_result, entry.supervision_answer):
                raise LogicMismatch(f"Program evaluates to {symbolic_result}, expected {entry.supervision_answer}")
        
        # Gate 3: PTX Lowering Check (Sovereignty Guarantee)
        try:
            ptx_kernel = self.lower_to_ptx(entry.eval_program)
            entry.ptx_hash = hash(ptx_kernel)
        except LoweringError as e:
            raise NonSovereignCode(f"Cannot lower to PTX: {e}")
        
        return ValidationReport(passed=True, ptx_hash=entry.ptx_hash)
```

### 1.3 Retry Escalation (Correctness Path)

| Attempt | Model | Temperature | Strategy |
|---------|-------|---------------|----------|
| 1 | `qwen3.5:397b-cloud` | 0.2 | Generate with strict XML schema |
| 2 | `kimi-k2-thinking:cloud` | 0.1 | Repair specific validation errors (stack underflow, arg mismatch) |
| 3 | `kimi-k2:1t-cloud` | 0.0 | Full reasoning trace + PTX lowering validation |

**Repair Prompt Template**:
```xml
<REPAIR_REQUEST>
  <VALIDATION_ERROR type="STACK_UNDERFLOW" token="SWAP" needed="2" had="1"/>
  <HINT>Insert DUP before SWAP to ensure sufficient stack depth</HINT>
  <PARTIAL_PROGRAM>ARG_N 2 / SWAP</PARTIAL_PROGRAM>
</REPAIR_REQUEST>
```

---

## Part 2: Scale-First Design (The "Throughput" Path)

**Philosophy**: Maximize cloud token utilization via adaptive batching. Isolate failures without blocking the pipeline.

### 2.1 Adaptive Batching Strategy

```python
# ingestion/batcher.py
class AdaptiveBatcher:
    def batch_entries(self, entries: List[RawEntry]) -> Iterator[Batch]:
        """
        Heterogeneous batching based on complexity heuristics.
        """
        buckets = {
            'grammar_simple': [],   # rule_strength only
            'grammar_complex': [],  # superior_to graph edges
            'math_translation': [], # NL → RPN conversion
            'supervision_align': [] # Requires query_hash resolution
        }
        
        for e in entries:
            if e.source == 'grammar' and not e.has_dependencies:
                buckets['grammar_simple'].append(e)
            elif 'divide_by_two' in str(e.raw_rpn) or 'equals_total_sum' in str(e.raw_rpn):
                buckets['math_translation'].append(e)
            else:
                buckets['grammar_complex'].append(e)
        
        # Batch sizes: Grammar=25, Math=2 (token budget: 4k input/2k output)
        yield from self.greedy_pack(buckets['grammar_simple'], max_tokens=4000, max_items=25)
        yield from self.greedy_pack(buckets['math_translation'], max_tokens=4000, max_items=2)
```

### 2.2 Model Routing Matrix (Scale-Optimized)

| Entry Class | Primary | Fallback | Rationale |
|-------------|---------|----------|-----------|
| Grammar (simple) | `gpt-oss:120b-cloud` | `qwen3.5:397b-cloud` | High throughput, deterministic schema |
| Grammar (complex graph) | `qwen3.5:397b-cloud` | `deepseek-v3.1:671b-cloud` | Needs defeasibility reasoning |
| Math (NL→RPN) | `qwen3.5:397b-cloud` | `kimi-k2-thinking:cloud` | Code translation task |
| Supervision mismatch | `kimi-k2-thinking:cloud` | Manual queue | Query-to-problem_id alignment |

### 2.3 Caching & Idempotency (Content-Addressable)

```sql
-- state/enrichment_cache.sql
CREATE TABLE enrichment_cache (
    input_hash BINARY(32) PRIMARY KEY,  -- SHA256(canonical_json(entry))
    schema_version INT NOT NULL,        -- For cache invalidation
    output_json JSON NOT NULL,
    ptx_hash BINARY(32),                -- Sovereign validation proof
    model_used VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_lookup ON enrichment_cache(input_hash, schema_version);
```

**Resume Logic**:
```bash
$ python -m ingestion.pipeline --checkpoint-db state/enrichment_cache.sqlite --resume
# Skips entries where input_hash exists and schema_version matches current
```

### 2.4 Failure Isolation & Backpressure

```python
# ingestion/failure_isolation.py
class CircuitBreaker:
    def __init__(self):
        self.ollama_limits = {
            'qwen3.5:397b-cloud': RateLimit(rpm=60, tpm=10000),
            'kimi-k2:1t-cloud': RateLimit(rpm=10, tpm=2000)
        }
    
    def handle_validation_failure(self, entry, error):
        if error.type == 'HALLUCINATED_OPCODE':
            # Quarantine for manual review (registry append-only)
            self.quarantine(entry, reason='opcode_not_in_registry')
        elif error.type == 'ARG_MISMATCH':
            # Auto-retry with higher model
            self.retry_with_model(entry, model='kimi-k2-thinking:cloud')
        elif error.type == 'RATE_LIMIT':
            # Exponential backoff with jitter
            self.backpressure_wait(error.retry_after)
```

---

## Part 3: Merged Final Architecture (Unified Spec)

### 3.1 Conflict Resolution Summary

| Conflict | Resolution | Rationale |
|----------|-----------|-----------|
| **Batch vs Single** | **Streaming batches with per-item validation** | Batching for throughput (B), but validation runs per-entry before commit (A). Failed entries are ejected from batch without failing the whole batch. |
| **Prompt Format** | **Hybrid: JSONL for Grammar, XML for Math** | Grammar entries are homogeneous (schema only), XML for math provides explicit structure for NL→RPN translation. |
| **Repair Strategy** | **Escalation with cache invalidation** | First fail updates cache with `status=failed`, retry uses higher-tier model and replaces cache entry. |
| **Supervision Mismatch** | **Pre-computed alignment index** | Build `query_hash → problem_id` lookup table in preprocessing (neither A nor B addressed this explicitly). |

### 3.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                     │
├─────────────────────────────────────────────────────────────┤
│  Stage 0: Preprocessing (CPU)                               │
│  ├── Build supervision_index: query_hash → (problem_id,   │
│  │   answer) for the 600 gold entries                       │
│  └── Canonicalize raw entries (normalize text, hash)        │
├─────────────────────────────────────────────────────────────┤
│  Stage 1: Batching & Routing (CPU)                          │
│  ├── Classify: grammar_simple | grammar_complex | math_nl   │
│  ├── Check cache (content_hash hit → skip)                  │
│  └── Route to appropriate model queue                       │
├─────────────────────────────────────────────────────────────┤
│  Stage 2: Cloud Enrichment (Ollama MCP @ localhost:8502)    │
│  ├── Batch submit to models per routing matrix              │
│  ├── Apply temperature: 0.2 (grammar) / 0.1 (math)          │
│  └── Return raw LLM output                                   │
├─────────────────────────────────────────────────────────────┤
│  Stage 3: Sovereign Validation (GPU-ready checks)           │
│  ├── Binder Contract Validator (stack simulation)             │
│  ├── Symbolic Execution (if supervision_answer available)   │
│  │   └── Use supervision_index to match query→answer        │
│  ├── PTX Lowering Check (dry-run)                           │
│  └── Meaning Star Generation (bidirectional symlinks)         │
├─────────────────────────────────────────────────────────────┤
│  Stage 4: Commit or Repair                                  │
│  ├── PASS: Write to enriched.jsonl + symlink index          │
│  ├── FAIL (repairable): Escalate model, retry Stage 2-3     │
│  │   (max 3 attempts, cache attempt_count)                 │
│  └── FAIL (fatal): Quarantine to manual_review/             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Key Implementation Details (Codex Spec)

#### A. Supervision Alignment Fix (The "Missing Link")
**Problem**: 600 entries have `supervision_answer` keyed by `problem_id`, but the math binder matches by query text.

**Solution**:
```python
# preprocessing/build_supervision_index.py
def align_supervision():
    """
    Create bidirectional lookup before enrichment begins.
    """
    index = {}
    for entry in raw_entries:
        if entry.problem_id in gold_dataset:
            # Create content-addressable key
            query_hash = sha256
