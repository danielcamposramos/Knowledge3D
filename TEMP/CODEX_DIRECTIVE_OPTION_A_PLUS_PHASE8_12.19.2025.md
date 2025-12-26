# CODEX DIRECTIVE: Option A (books_v5) + Phase 8 (Multi-Step Chaining)

**Date:** December 19, 2025
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation Partner)
**Priority:** HIGH
**Context:** Phase 7 complete, implementing both data quality fix AND compositional reasoning

---

## Objective

Implement **TWO major improvements** in sequence:

1. **Option A (books_v5):** Re-ingest with enhanced articulator to populate semantic role metadata
2. **Phase 8:** Multi-step theorem chaining for compositional problem-solving

**Timeline:** ~6-8 hours total (3-4 hours each)
**Expected impact:** MATH 3.0% → **10-15%**, AMC 0.5% → **4-8%**

---

## Part 1: Option A - Books v5 Re-Ingestion with LLM-Assisted Extraction (4-6 hours)

### Goal

**Populate `symbol_bindings[*].meaning` with semantic roles** during ingestion using local Ollama models as extraction assistants.

### Problem Summary (From Phase 7)

**Current state (books_v4):**
- Artifacts: 1,329
- With `symbol_bindings`: 1,122 (84.4%) ✅ structure exists
- With semantic meanings: 0 (0.0%) ❌ all = "unknown"

**Root cause:** Mathematical prose is implicit - patterns like "height h" or "legs a and b" rarely appear literally. Need semantic understanding of context.

**Impact:** Stage 3 semantic binding can't match roles → falls back to variable-name heuristics

**User insight:** "It's impossible to do so from books with programs" → Leverage local Ollama models during ingestion time.

### Architectural Compliance ✅

**CRITICAL:** Using LLMs during ingestion is **FULLY COMPLIANT** with sovereignty:
- ✅ Ingestion = one-time offline process (not hot inference path)
- ✅ Result = Galaxy entries in VRAM (sovereign at inference time)
- ✅ Hot path = PTX + Galaxy only (zero external dependencies)

**Analogy:** Using a compiler (external tool) to generate machine code (sovereign artifact).

### GPU Constraints (CRITICAL)

**User directive:** One GPU only, sequential execution, model restart between tasks.

**Implementation requirements:**
1. **Sequential processing:** Process artifacts one at a time
2. **Model restart:** Restart Ollama between artifacts to clear context
3. **No parallelization:** Never run multiple LLM calls concurrently
4. **Clean state:** Each artifact gets fresh model context

### Solution: Enhanced Articulator with LLM Assistance

**File:** `knowledge3d/training/math_benchmarks/sovereign_knowledge_articulator.py`

**Strategy:** Use local Ollama models to extract semantic roles from artifact text.

**Available models (from user's system):**
- `qwen3:8b` - Fast general-purpose (5.2 GB)
- `deepseek-r1:latest` - Reasoning focused (5.2 GB)
- `ministral-3:latest` - Efficient (6.0 GB)
- `deepseek-ocr:latest` - Visual/OCR (6.7 GB) - for equation images

**Model selection:** `qwen3:8b` (fastest, good accuracy, fits GPU)

### Implementation: LLM-Assisted Role Extraction

**Add Ollama client to articulator:**

```python
import subprocess
import json
from typing import Dict, List, Optional

class SovereignKnowledgeArticulator:
    def __init__(self, use_llm_extraction: bool = True):
        self.use_llm_extraction = use_llm_extraction
        # ... existing init ...

    def _extract_symbol_bindings_with_llm(
        self,
        artifact_text: str,
        latex_content: str,
        variables: List[str]
    ) -> Dict[str, Dict]:
        """
        Use local Ollama LLM to extract semantic roles for variables.

        CRITICAL: Sequential processing with model restart between calls.

        Args:
            artifact_text: Plain text context from PDF
            latex_content: LaTeX formulas containing variables
            variables: List of variables found in formulas (e.g., ['r', 'h', 'a'])

        Returns:
            {
                "r": {"meaning": "radius", "domain": "positive_real"},
                "h": {"meaning": "height", "domain": "positive_real"},
                "a": {"meaning": "leg", "domain": "positive_real"}
            }
        """
        if not self.use_llm_extraction or not variables:
            return self._extract_symbol_bindings_fallback(artifact_text, latex_content, variables)

        bindings = {}

        # CRITICAL: Process each variable SEQUENTIALLY (GPU constraint)
        for var in variables:
            # Restart Ollama between calls to clear context
            self._restart_ollama()

            # Extract role for this variable
            role, domain = self._query_llama_for_variable_role(
                var, artifact_text, latex_content
            )

            bindings[var] = {
                "meaning": role or "unknown",
                "domain": domain or "real"
            }

        return bindings

    def _restart_ollama(self):
        """
        Restart Ollama to clear model context (GPU memory cleanup).

        CRITICAL: Ensures each artifact extraction starts with clean state.
        """
        try:
            # Stop existing Ollama process
            subprocess.run(["pkill", "-f", "ollama"], check=False, capture_output=True)

            # Brief pause for cleanup
            import time
            time.sleep(1)

            # Ollama will auto-restart on next query
        except Exception as e:
            # Non-critical: Ollama will restart anyway
            pass

    def _query_llama_for_variable_role(
        self,
        var: str,
        artifact_text: str,
        latex_content: str
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Query Ollama qwen3:8b to extract semantic role for a variable.

        CRITICAL: Single query per call, model restarts between calls.

        Returns:
            (role, domain) tuple, e.g., ("radius", "positive_real")
        """
        # Build extraction prompt
        prompt = self._build_role_extraction_prompt(var, artifact_text, latex_content)

        # Query Ollama (blocking call, respects GPU constraint)
        try:
            result = subprocess.run(
                [
                    "ollama", "run", "qwen3:8b",
                    prompt
                ],
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout per variable
            )

            if result.returncode == 0:
                response = result.stdout.strip()
                return self._parse_llm_response(response)
            else:
                # Fallback on error
                return self._fallback_role_inference(var, artifact_text)

        except subprocess.TimeoutExpired:
            # Fallback on timeout
            return self._fallback_role_inference(var, artifact_text)
        except Exception as e:
            # Fallback on any error
            return self._fallback_role_inference(var, artifact_text)

    def _build_role_extraction_prompt(
        self,
        var: str,
        artifact_text: str,
        latex_content: str
    ) -> str:
        """
        Build prompt for LLM to extract variable semantic role.

        Design: Zero-shot extraction with structured output.
        """
        # Limit context to avoid GPU memory issues
        context_snippet = artifact_text[:1000] if len(artifact_text) > 1000 else artifact_text

        prompt = f"""You are extracting semantic roles of mathematical variables from textbook content.

CONTEXT:
{context_snippet}

FORMULA:
{latex_content}

VARIABLE: {var}

TASK: What is the semantic meaning of variable '{var}' in this context?

Choose from: radius, diameter, height, width, length, base, leg, hypotenuse, area, volume, side, angle, coefficient, constant, unknown

Respond with ONLY the role name (e.g., "radius") or "unknown" if unclear.
Do not add explanations."""

        return prompt

    def _parse_llm_response(self, response: str) -> tuple[Optional[str], Optional[str]]:
        """
        Parse LLM response to extract role and infer domain.

        Expected response: "radius" or "height" or "unknown"
        """
        role = response.strip().lower()

        # Validate role is in expected set
        valid_roles = {
            "radius", "diameter", "height", "width", "length", "base",
            "leg", "hypotenuse", "area", "volume", "side", "angle",
            "coefficient", "constant", "unknown"
        }

        if role not in valid_roles:
            role = "unknown"

        # Infer domain from role
        domain = "real"
        if role in ["radius", "diameter", "height", "width", "length", "area", "volume"]:
            domain = "positive_real"
        elif role == "angle":
            domain = "angle"

        return (role, domain)

    def _fallback_role_inference(
        self,
        var: str,
        artifact_text: str
    ) -> tuple[str, str]:
        """
        Fallback to regex-based role inference if LLM fails.

        Same logic as original Option A regex patterns.
        """
        text_lower = artifact_text.lower()

        # Simple pattern matching
        patterns = {
            "radius": [rf"radius\s+{var}\b", rf"{var}\s+is\s+(?:the\s+)?radius"],
            "height": [rf"height\s+{var}\b", rf"{var}\s+is\s+(?:the\s+)?height"],
            "leg": [rf"legs?\s+{var}\b"],
            "hypotenuse": [rf"hypotenuse\s+{var}\b"],
        }

        for role, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, text_lower):
                    domain = "positive_real" if role in ["radius", "height"] else "real"
                    return (role, domain)

        # Convention-based fallback
        if "circle" in text_lower and var == "r":
            return ("radius", "positive_real")
        if "triangle" in text_lower and var in ["a", "b"]:
            return ("leg", "positive_real")

        return ("unknown", "real")

    def _extract_symbol_bindings(self, artifact_text: str, latex_content: str) -> Dict[str, Dict]:
        """
        Main entry point: Extract variables and infer semantic roles.

        Uses LLM assistance if enabled, falls back to regex otherwise.
        """
        # Extract variables from LaTeX (existing logic)
        variables = self._find_variables_in_latex(latex_content)

        # Use LLM-assisted extraction
        return self._extract_symbol_bindings_with_llm(
            artifact_text, latex_content, variables
        )
```

### Implementation Steps (LLM-Assisted)

**Step 1: Enhance Articulator with Ollama Integration (~1 hour)**

1. Add Ollama client methods to `SovereignKnowledgeArticulator`:
   - `_extract_symbol_bindings_with_llm()` - main LLM extraction loop
   - `_restart_ollama()` - model restart for clean context
   - `_query_llama_for_variable_role()` - single variable extraction
   - `_build_role_extraction_prompt()` - prompt construction
   - `_parse_llm_response()` - response validation
   - `_fallback_role_inference()` - regex fallback if LLM fails

2. Test on 10 sample artifacts:
   ```bash
   # Extract first 10 artifacts from any book
   head -n 10 /K3D/Knowledge3D.local/galaxies/books_v4/la_done_right/artifacts.jsonl > /tmp/test_artifacts.jsonl

   # Test LLM extraction (sequential, model restarts between each)
   PYTHONPATH=. python3 -c "
   from knowledge3d.training.math_benchmarks.sovereign_knowledge_articulator import SovereignKnowledgeArticulator
   import json

   articulator = SovereignKnowledgeArticulator(use_llm_extraction=True)

   with open('/tmp/test_artifacts.jsonl') as f:
       for i, line in enumerate(f, 1):
           artifact = json.loads(line)
           bindings = articulator._extract_symbol_bindings(
               artifact['raw_text'],
               artifact['latex_content']
           )
           print(f'Artifact {i}: {bindings}')
   "
   ```

3. Verify:
   - ✅ Ollama restarts between variables (check GPU memory stable)
   - ✅ Roles extracted: at least 50% non-unknown on samples
   - ✅ Fallback works if Ollama unavailable

**Step 2: Re-Ingest 23 Books with LLM Extraction (~2-3 hours)**

**CRITICAL:** Sequential processing, one artifact at a time, model restart per variable.

**Estimated time per book:**
- ~60 artifacts per book (average from books_v4)
- ~3 variables per artifact (average)
- ~30 seconds per variable (Ollama query + restart)
- **~90 minutes per book** (60 artifacts × 3 variables × 0.5 min)

**Total ingestion time:** ~23 books × 1.5 hours = **~35 hours** ⚠️

**Optimization:** Batch similar artifacts, skip re-extraction if cached:

```python
# Add caching to avoid re-extracting same patterns
class SovereignKnowledgeArticulator:
    def __init__(self, use_llm_extraction: bool = True, cache_file: str = None):
        self.role_cache = {}  # Cache: (var, context_hash) → role
        self.cache_file = cache_file
        if cache_file and os.path.exists(cache_file):
            with open(cache_file) as f:
                self.role_cache = json.load(f)

    def _query_llama_for_variable_role(self, var, artifact_text, latex_content):
        # Check cache first
        context_hash = hashlib.md5(
            (artifact_text[:500] + latex_content).encode()
        ).hexdigest()
        cache_key = f"{var}_{context_hash}"

        if cache_key in self.role_cache:
            return self.role_cache[cache_key]  # Cache hit

        # Query LLM (sequential, restart)
        role, domain = self._query_llama_for_variable_role_uncached(...)

        # Cache result
        self.role_cache[cache_key] = (role, domain)
        if self.cache_file:
            with open(self.cache_file, 'w') as f:
                json.dump(self.role_cache, f)

        return (role, domain)
```

**Optimized timeline:** ~8-12 hours (with caching, many artifacts have similar contexts)

**Ingestion command with LLM extraction:**

```bash
# Re-ingest all books to books_v5 with LLM-assisted role extraction
BOOKS_V5_DIR="/K3D/Knowledge3D.local/galaxies/books_v5"
mkdir -p "$BOOKS_V5_DIR"

# Cache file for role extractions (speeds up similar patterns)
CACHE_FILE="/tmp/llm_role_extraction_cache.json"

# List of 23 books (from Phase 6)
BOOKS=(
  "Linear.Algebra.Done.Right.pdf:la_done_right:linear_algebra"
  "advcalc.pdf:advanced_calculus:calculus"
  "dmoi3.pdf:dmoi3:discrete_math"
  "transition_v104.pdf:transition_v104:transitions"
  "Area_and_Volume.pdf:areavol:geometry"
  "setsOfNumbers.pdf:numbersets:number_theory"
  "physicalQuantities.pdf:physquantities:physics"
  "Kappraff-Math-Gems-III-Part-1.pdf:mathgems:problem_solving"
  "9780134689579_sc.pdf:multivariable_calc:calculus"
  "AdvancedCalculusI_II.pdf:advanced_calc_1_2:calculus"
  "Advanced-Calculus-Patton-Bullett.pdf:advanced_calc_alt:calculus"
  "HildebrandMethods.pdf:numerical_analysis:applied_math"
  "Stehlik-Shortest-Shortcut-v31c.pdf:shortestshortcut:competition_math"
  "Advanced-Calculus-Robert-Wrede.pdf:wrede_calculus:calculus"
  "hildebrand.pdf:hildebrand:advanced_math"
  "MATH 2F05.pdf:math_2f05:university_course"
  "Manning.Math.for.Programmers.2020.11.pdf:math_for_programmers:applied_math"
  "Undergraduate Texts in Mathematics - Basic concepts of algebraic topology - Croom.pdf:algebraic_topology:topology"
  "3.3. Reverse Polish - Intermediate.pdf:rpn_intermediate:rpn_methods"
  "ReversePolishNotatonMethod.pdf:rpn_method:rpn_methods"
  "Orland_MfP_MEAP_V02_ch1.pdf:orland_math_prog:applied_math"
  "advmathprog.pdf:adv_math_programming:optimization"
  "Stavely_python_ebook.pdf:stavely_python:programming"
)

# Ingest each book SEQUENTIALLY (GPU constraint)
for book_spec in "${BOOKS[@]}"; do
  IFS=':' read -r pdf_name book_id domain <<< "$book_spec"

  echo "=== Ingesting $book_id with LLM extraction (this will take ~30-60 min) ==="

  # Run ingestion with LLM extraction enabled
  PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
    -c "
from pathlib import Path
from knowledge3d.training.math_benchmarks.book_galaxy_ingestion import BookGalaxyIngester
from knowledge3d.training.math_benchmarks.sovereign_knowledge_articulator import SovereignKnowledgeArticulator

pdf_path = '/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Advanced Maths/$pdf_name'
book_id = '$book_id'
title = '$book_id'
domain = '$domain'
output_dir = Path('$BOOKS_V5_DIR')

# Create articulator with LLM extraction + caching
articulator = SovereignKnowledgeArticulator(
    use_llm_extraction=True,
    cache_file='$CACHE_FILE'
)

# Create ingester with LLM-enabled articulator
ingester = BookGalaxyIngester(
    local_dir=output_dir,
    articulator=articulator
)

# Convert PDF to JSON pages
json_path = ingester.pdf_to_json_pages(pdf_path=pdf_path, title=title)

# Ingest with LLM extraction (sequential, model restarts between variables)
result_dir = ingester.ingest_json_pages(
    json_path=json_path,
    title=title,
    book_id=book_id,
    domain=domain
)

print(f'Ingested {book_id}: {result_dir}')
"

  echo "=== $book_id complete ==="
done

echo "=== All 23 books ingested to books_v5 with LLM-extracted semantic roles ==="
```

**Progress tracking:** Run in tmux/screen session, monitor with:
```bash
# Check cache growth (indicates extraction progressing)
wc -l $CACHE_FILE

# Check GPU usage
nvidia-smi

# Check Ollama restarts (should see frequent process cycling)
watch -n 5 'ps aux | grep ollama'
```

**Step 3: Validate Metadata Quality (~30 min)**

```bash
# Check symbol_bindings semantic coverage in books_v5
find "$BOOKS_V5_DIR" -name "artifacts.jsonl" -exec cat {} \; | \
  python3 -c "
import json, sys
total = 0
with_bindings = 0
non_unknown = 0
meanings = {}

for line in sys.stdin:
    obj = json.loads(line)
    total += 1
    if obj.get('symbol_bindings'):
        with_bindings += 1
        for var, info in obj['symbol_bindings'].items():
            meaning = info.get('meaning', 'unknown')
            meanings[meaning] = meanings.get(meaning, 0) + 1
            if meaning != 'unknown':
                non_unknown += 1

print(f'Total artifacts: {total}')
print(f'With bindings: {with_bindings} ({100*with_bindings/total:.1f}%)')
print(f'Non-unknown meanings: {non_unknown}')
print(f'Top meanings: {sorted(meanings.items(), key=lambda x: -x[1])[:10]}')
"
```

**Success criteria:**
- ✅ Non-unknown meanings: > 40% (vs 0% in books_v4)
- ✅ Top meanings include: radius, height, leg, hypotenuse, base, area, volume
- ✅ "unknown" percentage < 60% (vs 100% in books_v4)

**Step 4: Benchmark books_v5 (~40 min)**

**CRITICAL:** Run benchmarks AFTER all book ingestion complete (sequential constraint).

```bash
# MATH dataset (200 problems)
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/run_sovereign_math_benchmarks.py \
  --use-trm-navigator --disable-retrieval --datasets math \
  --max-problems 200 --shuffle --shuffle-seed 123 --thinking-budget 8 \
  --shadow-readonly --load-all-galaxies \
  --enable-book-galaxies \
  --book-galaxy-root /K3D/Knowledge3D.local/galaxies/books_v5 \
  --book-max-books 64 --book-top-k 5 --verbose \
  > /tmp/math_option_a_books_v5_200_seed123.log 2>&1

# AMC-AIME dataset (200 problems)
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/run_sovereign_math_benchmarks.py \
  --use-trm-navigator --disable-retrieval --datasets amc_aime \
  --max-problems 200 --shuffle --shuffle-seed 123 --thinking-budget 8 \
  --shadow-readonly --load-all-galaxies \
  --enable-book-galaxies \
  --book-galaxy-root /K3D/Knowledge3D.local/galaxies/books_v5 \
  --book-max-books 64 --book-top-k 5 --verbose \
  > /tmp/amc_aime_option_a_books_v5_200_seed123.log 2>&1
```

**Expected results (Option A with LLM extraction):**
- **MATH:** 3.0% → **6-10%** (semantic binding active with high-quality metadata)
- **AMC-AIME:** 0.5% → **3-5%** (better role matching)
- **Semantic coverage:** 0% → **50-70%** (LLM extraction better than regex)

---

### Optional: Visual Content Extraction with deepseek-ocr

**Use case:** Extract semantic roles from equation images/diagrams in PDFs.

**Model:** `deepseek-ocr:latest` (6.7 GB)

**When to use:**
- PDFs with scanned pages (no OCR text)
- Diagrams with variable labels (e.g., triangle diagram with "a", "b", "c" labeled)
- Equation images (extracted as images, not LaTeX)

**Implementation addition:**

```python
def _extract_from_image(self, image_path: str, variables: List[str]) -> Dict[str, str]:
    """
    Use deepseek-ocr to extract variable semantics from equation images.

    CRITICAL: Sequential processing, model restart between images.
    """
    self._restart_ollama()

    prompt = f"""Analyze this mathematical diagram/equation image.

Variables present: {', '.join(variables)}

For each variable, identify its semantic role (radius, height, leg, etc.).

Respond in JSON format:
{{"r": "radius", "h": "height", "a": "leg"}}"""

    result = subprocess.run(
        ["ollama", "run", "deepseek-ocr:latest", "--image", image_path, prompt],
        capture_output=True,
        text=True,
        timeout=60
    )

    # Parse JSON response
    return json.loads(result.stdout)
```

**Note:** Only implement if books have significant visual content. Most math textbooks have LaTeX text, so text-based extraction (qwen3:8b) should be sufficient.

---

## Part 2: Phase 8 - Multi-Step Theorem Chaining (3-4 hours)

### Goal

**Enable TRM to chain multiple theorems/formulas** for compositional problem-solving.

### Problem Summary

**Current limitation:** Single-step reasoning only
- Can apply one formula per problem
- Can't decompose multi-step problems
- Can't chain prerequisite theorems

**Example failure:**
```
Problem: "Circle with circumference 20, find area"

Current approach (single-step):
- Search for "area formula" → πr²
- But we don't know r!
- Fails with no_rule_match or wrong_computation

Needed approach (multi-step):
- Step 1: Circumference = 2πr → solve for r → r = 20/(2π) = 3.183
- Step 2: Area = πr² → π(3.183)² = 31.83
- Result: Correct answer via chaining
```

### Strategy: Shadow Copy Chaining

**Architecture:** Use existing Shadow Copy mechanism for theorem composition

**Implementation location:** `knowledge3d/training/math_benchmarks/trm_galaxy_reader.py`

### Implementation Steps

**Step 1: Add Multi-Step Candidate Generator (~1.5 hours)**

```python
def _generate_multi_step_candidates(
    self,
    problem_text: str,
    problem_state: Dict,
    max_steps: int = 3
) -> List[Tuple[str, Dict, str]]:
    """
    Generate multi-step RPN programs by chaining book artifacts.

    Strategy:
    1. Identify missing variables (e.g., "need r but have circumference")
    2. Search for intermediate formulas (circumference → r)
    3. Chain: intermediate formula + target formula
    4. Generate composed RPN program

    Returns:
        List of (rpn_program, metadata, seed_source) tuples
    """
    candidates = []

    # Step 1: Get target formulas (what we're trying to compute)
    target_candidates, target_meta, _ = self._generate_book_galaxy_candidates(
        problem_text, problem_state
    )

    # Step 2: For each target, check if it has unbound variables
    for target_rpn, target_info in zip(target_candidates, target_meta):
        missing_vars = self._find_missing_variables(target_rpn, problem_state)

        if not missing_vars:
            continue  # All variables bound, no chaining needed

        # Step 3: Search for intermediate formulas for missing variables
        for var in missing_vars:
            intermediate_candidates = self._search_for_variable(
                var, problem_text, problem_state
            )

            for inter_rpn, inter_info in intermediate_candidates:
                # Step 4: Compose intermediate + target
                composed_rpn = self._compose_rpn_programs(
                    inter_rpn, target_rpn, var
                )

                if self._validate_composed_program(composed_rpn):
                    metadata = {
                        "steps": [inter_info, target_info],
                        "intermediate_var": var,
                        "composed": True
                    }
                    candidates.append((composed_rpn, metadata, "book_multi_step"))

    return candidates


def _find_missing_variables(self, rpn_program: str, problem_state: Dict) -> List[str]:
    """
    Find variables in RPN program that aren't in problem_state.

    Example:
        rpn_program = "π r 2 pow *"  (πr²)
        problem_state = {"numbers": [20], "context": "circumference"}
        Returns: ["r"]  (r is needed but not available)
    """
    # Parse RPN for variable tokens
    tokens = rpn_program.split()
    variables = [t for t in tokens if t.isalpha() and len(t) == 1]

    # Check which aren't in problem_state
    available = set(problem_state.get("variables", {}).keys())
    missing = [v for v in variables if v not in available]

    return missing


def _search_for_variable(
    self,
    var: str,
    problem_text: str,
    problem_state: Dict
) -> List[Tuple[str, Dict]]:
    """
    Search for formulas that solve for the missing variable.

    Example:
        var = "r"
        problem_text = "circle with circumference 20"
        Returns: [("2 π * r *", {...})]  (circumference = 2πr, can solve for r)
    """
    # Search book artifacts for formulas involving this variable
    search_query = f"{var} formula {problem_text}"

    artifacts = self._search_book_artifacts(search_query, top_k=5)

    candidates = []
    for artifact in artifacts:
        # Check if artifact can compute this variable
        if var in artifact.symbol_bindings:
            # Extract RPN that isolates this variable
            rpn = self._extract_variable_formula(artifact, var)
            if rpn:
                candidates.append((rpn, {"artifact_id": artifact.artifact_id}))

    return candidates


def _compose_rpn_programs(
    self,
    intermediate_rpn: str,
    target_rpn: str,
    shared_var: str
) -> str:
    """
    Compose two RPN programs by substituting intermediate result.

    Example:
        intermediate_rpn = "circumference 2 π * /"  (r = C/(2π))
        target_rpn = "π r 2 pow *"  (A = πr²)
        shared_var = "r"

        Result: "circumference 2 π * / 2 pow π *"  (A = π(C/(2π))²)
    """
    # Step 1: Compute intermediate result (r)
    inter_tokens = intermediate_rpn.split()

    # Step 2: Replace shared_var in target with intermediate result
    target_tokens = target_rpn.split()
    composed = []

    for token in target_tokens:
        if token == shared_var:
            # Substitute intermediate computation
            composed.extend(inter_tokens)
        else:
            composed.append(token)

    return " ".join(composed)
```

**Step 2: Integrate with TTC (~1 hour)**

```python
# In _test_time_compute()

def _test_time_compute(self, problem_text, problem_state, seed_candidates):
    # ... existing single-step logic ...

    # NEW: Add multi-step candidates
    multi_step_candidates = self._generate_multi_step_candidates(
        problem_text, problem_state, max_steps=2
    )

    # Merge with single-step candidates
    all_candidates = seed_candidates + multi_step_candidates

    # Score all candidates (multi-step gets small boost)
    for candidate, metadata, source in all_candidates:
        score = self._score_candidate(candidate, metadata)

        if source == "book_multi_step":
            score *= 1.2  # 20% boost for compositional reasoning

        # ... rest of TTC logic ...
```

**Step 3: Test Multi-Step Logic (~30 min)**

```python
# Add test case in tests/test_book_galaxy_templates.py

def test_trm_reader_multi_step_chaining():
    """Test multi-step theorem chaining."""
    # Create artifact for circumference → radius
    circ_artifact = KnowledgeArtifact(
        artifact_id="circ_to_radius",
        conclusion_rpn="circumference 2 π * /",  # r = C/(2π)
        symbol_bindings={"r": {"meaning": "radius"}}
    )

    # Create artifact for radius → area
    area_artifact = KnowledgeArtifact(
        artifact_id="circle_area",
        conclusion_rpn="π r 2 pow *",  # A = πr²
        symbol_bindings={"r": {"meaning": "radius"}}
    )

    # Problem: "Circle with circumference 20, find area"
    reader = TRMGalaxyReader(...)
    candidates = reader._generate_multi_step_candidates(
        "Circle with circumference 20, find area",
        {"numbers": [20], "context": "circle circumference area"}
    )

    # Should generate composed program
    assert len(candidates) > 0
    composed_rpn = candidates[0][0]

    # Verify composition: should compute r from C, then A from r
    assert "20 2 π * /" in composed_rpn  # r = 20/(2π)
    assert "2 pow π *" in composed_rpn   # A = πr²
```

**Step 4: Benchmark with Multi-Step (~40 min)**

```bash
# Run with books_v5 + multi-step chaining
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/run_sovereign_math_benchmarks.py \
  --benchmark MATH \
  --max-problems 200 \
  --shuffle --shuffle-seed 123 \
  --use-trm-navigator \
  --enable-book-galaxies \
  --book-galaxy-root /K3D/Knowledge3D.local/galaxies/books_v5 \
  > /tmp/math_phase8_multistep_books_v5_200_seed123.log 2>&1
```

**Expected results (books_v5 + multi-step):**
- MATH: 5-7% → **10-15%** (compositional problems solved)
- AMC-AIME: 2-3% → **5-8%** (multi-step geometry/algebra)

---

## Success Criteria

### Option A (books_v5) Success
- ✅ Symbol_bindings semantic coverage: 0% → **40-60%**
- ✅ Top meanings include: radius, height, leg, hypotenuse, base
- ✅ MATH accuracy: 3.0% → **5-7%** (Stage 3 binding active)
- ✅ AMC accuracy: 0.5% → **2-3%**

### Phase 8 (Multi-Step) Success
- ✅ Multi-step candidates generated for chained problems
- ✅ Composed RPN programs validate (stack-shape correct)
- ✅ MATH accuracy: 5-7% → **10-15%** (2-3× from Option A)
- ✅ AMC accuracy: 2-3% → **5-8%** (2-3× from Option A)

### Combined Success (Option A + Phase 8)
- ✅ MATH accuracy: 3.0% → **10-15%** (3-5× improvement)
- ✅ AMC accuracy: 0.5% → **5-8%** (10-16× improvement)
- ✅ Multi-modal improvement: single-step + multi-step both working
- ✅ Clean architecture: metadata + reasoning both enhanced

---

## Timeline & Deliverables (LLM-Assisted)

**Part 1: Option A (books_v5 with LLM extraction)**
- Articulator Ollama integration: **1 hour**
- Testing on sample artifacts: **30 min**
- Re-ingestion (23 books, sequential, LLM per variable): **8-12 hours** ⚠️
  - Optimized with caching: Context-similar artifacts reuse extractions
  - Progress tracked via cache file growth
  - Can pause/resume (cache persists)
- Metadata validation: **30 min**
- Benchmarks (MATH + AMC): **40 min**
- **Subtotal: ~11-15 hours** (overnight run recommended)

**Part 2: Phase 8 (Multi-Step Chaining)**
- Multi-step generator: **1.5 hours**
- TTC integration: **1 hour**
- Testing + benchmarks: **1 hour**
- **Subtotal: ~3.5 hours**

**Total: ~14-18 hours** (Part 1 can run overnight, Part 2 next day)

**Deliverables:**
1. Enhanced `SovereignKnowledgeArticulator` with Ollama LLM integration:
   - `_extract_symbol_bindings_with_llm()` - main extraction loop
   - `_restart_ollama()` - model restart for clean context
   - `_query_llama_for_variable_role()` - single variable extraction
   - `_fallback_role_inference()` - regex fallback
   - Role extraction cache (persisted to `/tmp/llm_role_extraction_cache.json`)

2. books_v5 directory with LLM-extracted semantic metadata:
   - 23 books re-ingested
   - symbol_bindings with 50-70% non-unknown meanings
   - `/K3D/Knowledge3D.local/galaxies/books_v5/`

3. Multi-step chaining in `TRMGalaxyReader`:
   - `_generate_multi_step_candidates()` - theorem composition
   - `_find_missing_variables()` - prerequisite detection
   - `_compose_rpn_programs()` - RPN chaining

4. Benchmark logs:
   - `/tmp/math_option_a_books_v5_200_seed123.log` (Option A results)
   - `/tmp/amc_aime_option_a_books_v5_200_seed123.log`
   - `/tmp/math_phase8_multistep_books_v5_200_seed123.log` (Phase 8 results)

5. Completion report: `TEMP/CODEX_OPTION_A_PHASE8_COMPLETE_12.19.2025.md`

---

## Risk Mitigation

**Risk 1: Articulator role inference < 40% coverage**
- Mitigation: Start with high-confidence patterns (context mentions)
- Fallback: Convention-based inference (shape → variable conventions)
- Acceptable: Even 30-40% coverage is 30-40× better than 0%

**Risk 2: Multi-step composition produces invalid RPN**
- Mitigation: Use existing RPN validator (stack-shape checking)
- Fallback: Only compose if both programs validate independently
- Safety: Invalid compositions filtered before TTC

**Risk 3: Multi-step candidates don't improve accuracy**
- Mitigation: Add logging to track multi-step selection rate
- Fallback: Multi-step can be disabled via flag if needed
- Learning: Even negative result teaches us about problem structure

---

## LLM-Assisted Ingestion Architecture Summary

### Why This Approach Works ✅

**Problem:** Mathematical prose is implicit:
```
"For a right triangle, if the legs have lengths a and b, and c is the hypotenuse..."
```
→ Regex can't extract "a=leg, b=leg, c=hypotenuse" from this implicit phrasing.

**Solution:** LLM semantic understanding:
- Qwen3:8b reads context and infers variable roles
- Sequential processing (one variable at a time) respects GPU constraint
- Model restart between variables ensures clean context
- Caching avoids re-extracting similar patterns

**Result:** 50-70% semantic coverage (vs 0% with regex, 2.9% with hot-path inference)

### Sovereignty Compliance ✅

**CRITICAL:** Using LLMs during ingestion is **FULLY SOVEREIGN**:

| Phase | Tools Allowed | Sovereignty Check |
|-------|---------------|-------------------|
| **Ingestion** (offline) | ✅ Ollama, numpy, pandas, ANY tool | N/A (one-time process) |
| **Galaxy Storage** (VRAM) | ✅ PTX kernels, Galaxy entries | ✅ Sovereign |
| **Inference** (hot path) | ✅ PTX + Galaxy ONLY | ✅ Sovereign |

**Analogy:** Using GCC (external compiler) to generate assembly code (sovereign artifact).

**Result:** books_v5 galaxies are **pure VRAM entries** at inference time. Zero external dependencies.

### GPU Constraint Compliance ✅

**User directive:** One GPU only, sequential execution, model restart between tasks.

**Implementation guarantees:**
1. ✅ **Sequential processing:** One artifact at a time
2. ✅ **Model restart:** `pkill ollama` between variables clears GPU context
3. ✅ **No parallelization:** Single `ollama run` call at a time
4. ✅ **Progress tracking:** Cache file shows extraction progress
5. ✅ **Pause/resume:** Can stop and restart (cache persists)

**Timeline realistic:** 8-12 hours for 23 books (overnight run).

### Alternative Models Available 📋

**If qwen3:8b underperforms, try these alternatives:**

1. **deepseek-r1:latest** (5.2 GB) - Reasoning-focused, better for implicit role inference
2. **ministral-3:latest** (6.0 GB) - Efficient, good at structured extraction
3. **gemma3:12b** (8.1 GB) - Larger model, more accurate but slower

**To switch models:** Change `"qwen3:8b"` → `"deepseek-r1:latest"` in `_query_llama_for_variable_role()`

**To use deepseek-ocr for visual content:** Implement `_extract_from_image()` if PDFs have diagrams.

---

## Authorization

**Proceed with both Option A and Phase 8 in sequence:**

1. **Start with Option A** (books_v5 LLM-assisted re-ingestion)
   - Implement Ollama integration in articulator
   - Test on 10 sample artifacts first
   - Run full ingestion overnight (8-12 hours)
   - Validate semantic coverage ≥ 50%
   - Benchmark MATH/AMC (target: 6-10% MATH, 3-5% AMC)

2. **Then implement Phase 8** (multi-step chaining)
   - Build on improved metadata from books_v5
   - Test compositional reasoning with better variable binding
   - Benchmark with multi-step enabled
   - Target: 10-15% MATH, 5-8% AMC

**User directive:** "Let's do as Codex suggested, but let's implement both ideas" + "leverage my local Ollama... to aid our quest"

**GPU constraint:** Sequential only, model restart between tasks, one GPU.

---

**Architect:** Claude (Architecture Partner)
**Date:** December 19, 2025
**Status:** READY FOR EXECUTION (Codex proceed with Option A LLM integration, then Phase 8)

**Expected total timeline:** ~14-18 hours (Option A overnight, Phase 8 next day)
**Expected final accuracy:** MATH 10-15% (3-5× from Phase 7), AMC 5-8% (10-16× from Phase 7)
