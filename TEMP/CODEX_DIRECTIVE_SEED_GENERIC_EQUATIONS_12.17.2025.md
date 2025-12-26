# CODEX DIRECTIVE: Seed Galaxy with Generic Equations

**Date:** December 17, 2025
**Priority:** CRITICAL
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation Lead)

---

## Context

**Current State:**
- Reality Galaxy loaded but only has 3 physics systems (projectile_2d, point_charge_2d, lc_circuit)
- Math Galaxy exists but lacks cross-domain generic equations
- Accuracy: 1% on shuffled GSM8K (overfitted to task-specific rules)
- Root cause: Missing generic mathematical relationships that apply across domains

**Goal:** Populate Galaxy Universe with cross-domain generic equations so TRM can find isomorphic patterns (e.g., physics rate equation solves math word problem).

---

## Implementation Task

### File to Create

**`knowledge3d/cranium/generic_equations.py`**

This file defines the 7 core generic equations and provides a loader to populate Math/Reality Galaxy.

```python
"""
Generic cross-domain equations for true generalization.

These are NOT task-specific. They're universal mathematical relationships
that apply across physics, economics, geometry, biology, etc.
"""

GENERIC_EQUATIONS = {
    "rate_time_distance": {
        "formula": "distance = rate × time",
        "rpn": "rate time *",
        "domains": ["physics", "math", "economics"],
        "variables": ["rate", "time", "distance"],
        "isomorphic_to": [
            "money = rate × duration",
            "work = power × time",
            "distance = speed × time",
            "pages = pages_per_day × days"
        ],
        "examples": [
            "Tom reads 5 pages per day for 6 days → 5 6 * = 30 pages",
            "Car travels 60 mph for 3 hours → 60 3 * = 180 miles",
            "Worker earns $15/hour for 8 hours → 15 8 * = $120"
        ]
    },

    "work_rate_time": {
        "formula": "work = rate × time",
        "rpn": "rate time *",
        "domains": ["physics", "economics", "biology"],
        "variables": ["rate", "time", "work"],
        "isomorphic_to": [
            "earnings = wage × hours",
            "production = output_rate × time",
            "growth = growth_rate × time",
            "items = items_per_hour × hours"
        ],
        "examples": [
            "Factory produces 20 widgets/hour for 5 hours → 20 5 * = 100 widgets",
            "Baker makes 12 cookies/batch for 4 batches → 12 4 * = 48 cookies"
        ]
    },

    "area_rectangle": {
        "formula": "area = length × width",
        "rpn": "length width *",
        "domains": ["geometry", "physics", "economics"],
        "variables": ["length", "width", "area"],
        "isomorphic_to": [
            "cost = price × quantity",
            "total = count × value",
            "items = rows × columns"
        ],
        "examples": [
            "Garden is 8 feet by 6 feet → 8 6 * = 48 square feet",
            "Buy 5 apples at $3 each → 5 3 * = $15"
        ]
    },

    "unit_conversion": {
        "formula": "target = source × conversion_factor",
        "rpn": "source factor *",
        "domains": ["physics", "chemistry", "math"],
        "variables": ["source", "factor", "target"],
        "examples": [
            "1000 grams to kilograms → 1000 0.001 * = 1 kg",
            "120 minutes to hours → 120 0.01666 * = 2 hours",
            "500 cents to dollars → 500 0.01 * = $5"
        ]
    },

    "fair_share": {
        "formula": "share = total / count",
        "rpn": "total count /",
        "domains": ["math", "economics"],
        "variables": ["total", "count", "share"],
        "isomorphic_to": [
            "average = sum / count",
            "rate = total / time",
            "per_item = total_cost / quantity"
        ],
        "examples": [
            "12 cookies shared among 4 people → 12 4 / = 3 each",
            "150 miles in 3 hours → 150 3 / = 50 mph",
            "$60 for 5 items → 60 5 / = $12 per item"
        ]
    },

    "total_from_parts": {
        "formula": "total = sum(parts)",
        "rpn": "part1 part2 + part3 + ...",
        "domains": ["math", "physics", "economics"],
        "variables": ["part1", "part2", "part3", "total"],
        "examples": [
            "John has 5 apples, Mary has 3 → 5 3 + = 8 apples",
            "Monday 10 pages, Tuesday 15 pages, Wednesday 8 pages → 10 15 + 8 + = 33 pages"
        ]
    },

    "remaining": {
        "formula": "remaining = total - used",
        "rpn": "total used -",
        "domains": ["math", "economics", "physics"],
        "variables": ["total", "used", "remaining"],
        "isomorphic_to": [
            "left = had - gave",
            "balance = initial - spent"
        ],
        "examples": [
            "Had 20 candies, gave 8 away → 20 8 - = 12 left",
            "Started with $100, spent $35 → 100 35 - = $65 remaining"
        ]
    }
}


class GenericEquationGalaxy:
    """
    Loader for generic cross-domain equations into Galaxy Universe.
    """

    def __init__(self, galaxy_universe=None):
        """
        Initialize with optional GalaxyUniverse instance.

        Args:
            galaxy_universe: GalaxyUniverse instance (if None, creates standalone)
        """
        self.galaxy_universe = galaxy_universe
        self.equations = GENERIC_EQUATIONS

    def load_equations(self, target_galaxy_name="reality"):
        """
        Load generic equations into specified galaxy.

        Args:
            target_galaxy_name: "reality" or "math" galaxy to populate

        Returns:
            Number of equations loaded
        """
        if not self.galaxy_universe:
            print("[WARN] No GalaxyUniverse provided, cannot load equations")
            return 0

        count = 0
        for eq_id, eq_data in self.equations.items():
            # Create galaxy entry for each equation
            entry = {
                "id": eq_id,
                "type": "generic_equation",
                "formula": eq_data["formula"],
                "rpn": eq_data["rpn"],
                "domains": eq_data["domains"],
                "variables": eq_data.get("variables", []),
                "isomorphic_patterns": eq_data.get("isomorphic_to", []),
                "examples": eq_data.get("examples", [])
            }

            # Add to galaxy
            self.galaxy_universe.add_entry(
                galaxy_name=target_galaxy_name,
                entry=entry
            )
            count += 1

        print(f"[GENERIC_EQUATIONS] Loaded {count} equations into {target_galaxy_name} galaxy")
        return count

    def query_by_domain(self, domain: str):
        """
        Query equations by domain.

        Args:
            domain: Domain to query ("physics", "math", "economics", etc.)

        Returns:
            List of equations matching domain
        """
        matching = []
        for eq_id, eq_data in self.equations.items():
            if domain in eq_data["domains"]:
                matching.append({
                    "id": eq_id,
                    "formula": eq_data["formula"],
                    "rpn": eq_data["rpn"]
                })
        return matching

    def query_by_variables(self, variables: list):
        """
        Query equations by required variables.

        Args:
            variables: List of variable names (e.g., ["rate", "time"])

        Returns:
            List of equations involving those variables
        """
        matching = []
        for eq_id, eq_data in self.equations.items():
            eq_vars = set(eq_data.get("variables", []))
            query_vars = set(variables)
            if query_vars.issubset(eq_vars):
                matching.append({
                    "id": eq_id,
                    "formula": eq_data["formula"],
                    "rpn": eq_data["rpn"]
                })
        return matching

    def report(self):
        """
        Generate report of loaded equations.

        Returns:
            Dictionary with equation counts by domain
        """
        domain_counts = {}
        for eq_data in self.equations.values():
            for domain in eq_data["domains"]:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

        return {
            "total_equations": len(self.equations),
            "domains": domain_counts,
            "equation_ids": list(self.equations.keys())
        }
```

---

## Integration Points

### 1. Wire into GalaxyUniverse Loader

**File:** `knowledge3d/cranium/word_galaxy.py` (or wherever GalaxyUniverse is initialized)

Add generic equations loading to the initialization sequence:

```python
def load_all_galaxies(self):
    """Load all default galaxies including generic equations."""
    # Existing galaxy loads...
    self.load_drawing_galaxy()
    self.load_character_galaxy()
    self.load_reality_galaxy()
    # ... etc ...

    # NEW: Load generic equations into Reality/Math Galaxy
    from knowledge3d.cranium.generic_equations import GenericEquationGalaxy
    eq_loader = GenericEquationGalaxy(galaxy_universe=self)
    eq_loader.load_equations(target_galaxy_name="reality")

    print(f"[GALAXY_UNIVERSE] Generic equations loaded: {eq_loader.report()}")
```

### 2. Enable TRM to Query Generic Equations

**File:** `knowledge3d/training/math_benchmarks/trm_galaxy_reader.py`

Add method to query generic equations during exploration:

```python
def explore_generic_equations(self, problem_text: str) -> List[Dict]:
    """
    Query generic equations based on problem context.

    Args:
        problem_text: Problem text to analyze

    Returns:
        List of relevant generic equations
    """
    from knowledge3d.cranium.generic_equations import GenericEquationGalaxy

    # Extract concepts from problem
    concepts = self.extract_concepts(problem_text)  # e.g., ["rate", "time"]

    # Query equations by variables
    eq_galaxy = GenericEquationGalaxy()
    relevant_equations = eq_galaxy.query_by_variables(concepts)

    return relevant_equations
```

### 3. Record Cross-Domain Usage in Shadow Copy

**File:** `knowledge3d/training/arc_agi/dual_shadow_copy.py`

Add tracking for cross-domain equation usage:

```python
def record_cross_domain_usage(
    self,
    equation_id: str,
    source_domain: str,
    target_problem_type: str,
    success: bool
):
    """
    Record when a cross-domain equation was used.

    Example: physics rate equation solving math word problem.
    """
    entry = {
        "type": "cross_domain",
        "equation_id": equation_id,
        "source_domain": source_domain,
        "target_type": target_problem_type,
        "success": success,
        "timestamp": time.time()
    }
    self.library.append(entry)

    if success:
        # Strengthen cross-domain connection
        key = f"{equation_id}_{target_problem_type}"
        self.cross_domain_scores[key] = self.cross_domain_scores.get(key, 0) + 1
```

---

## Validation Steps

### Step 1: Verify Equations Load

```bash
bash scripts/k3d_env.sh run python3 -c "
from knowledge3d.cranium.generic_equations import GenericEquationGalaxy
eq_galaxy = GenericEquationGalaxy()
report = eq_galaxy.report()
print(f'Loaded {report[\"total_equations\"]} equations')
print(f'Domains: {report[\"domains\"]}')
print(f'IDs: {report[\"equation_ids\"]}')
"
```

**Expected output:**
```
Loaded 7 equations
Domains: {'physics': 3, 'math': 7, 'economics': 5, 'biology': 1, 'chemistry': 1, 'geometry': 2}
IDs: ['rate_time_distance', 'work_rate_time', 'area_rectangle', 'unit_conversion', 'fair_share', 'total_from_parts', 'remaining']
```

### Step 2: Test Cross-Domain Query

```bash
bash scripts/k3d_env.sh run python3 -c "
from knowledge3d.cranium.generic_equations import GenericEquationGalaxy
eq_galaxy = GenericEquationGalaxy()

# Query by domain
physics_eqs = eq_galaxy.query_by_domain('physics')
print(f'Physics equations: {len(physics_eqs)}')

# Query by variables
rate_eqs = eq_galaxy.query_by_variables(['rate', 'time'])
print(f'Rate+time equations: {len(rate_eqs)}')
for eq in rate_eqs:
    print(f'  {eq[\"id\"]}: {eq[\"formula\"]}')
"
```

**Expected output:**
```
Physics equations: 3
Rate+time equations: 2
  rate_time_distance: distance = rate × time
  work_rate_time: work = rate × time
```

### Step 3: Run Shuffled GSM8K with Generic Equations

```bash
bash scripts/k3d_env.sh run python3 scripts/run_sovereign_math_benchmarks.py \
    --use-trm-navigator \
    --load-all-galaxies \
    --datasets gsm8k \
    --max-problems 200 \
    --shuffle \
    --shuffle-seed 123 \
    --shadow-readonly
```

**Expected improvements:**
- Accuracy: 1% → 3-5% (cross-domain equations provide new paths)
- Cross-domain usage: 0% → 10-20% (trace shows physics → math transfers)
- no_rule_match: Should decrease as generic equations provide fallback patterns

### Step 4: Verify Cross-Domain Traces

After benchmark run, check logs for cross-domain usage:

```bash
grep "CROSS_DOMAIN" /K3D/Knowledge3D.local/logs/math_benchmark_*.log | head -10
```

**Expected:**
```
[CROSS_DOMAIN] rate_time_distance (physics) → GSM8K rate problem (success)
[CROSS_DOMAIN] work_rate_time (economics) → GSM8K work problem (success)
[CROSS_DOMAIN] area_rectangle (geometry) → GSM8K multiplication (success)
```

---

## Success Criteria

### Generic Equations Loaded
- [ ] `generic_equations.py` created with 7 equations
- [ ] GenericEquationGalaxy class implemented with loader/query methods
- [ ] Equations appear in Reality/Math Galaxy report
- [ ] Can query by domain (physics, math, economics)
- [ ] Can query by variables (rate+time, total+count)

### Integration
- [ ] GalaxyUniverse loads generic equations automatically
- [ ] TRMGalaxyReader can explore generic equations
- [ ] Shadow copy records cross-domain usage

### Validation
- [ ] Equations load without errors
- [ ] Query methods return expected results
- [ ] Shuffled GSM8K accuracy improves (1% → 3-5%)
- [ ] Cross-domain usage appears in traces (>0%)
- [ ] no_rule_match decreases (exploration finds more paths)

---

## Implementation Notes

**Sovereignty compliance:**
- ✅ Generic equations are sovereign (hardcoded in Python, loaded to Galaxy VRAM)
- ✅ No external APIs or libraries in hot path
- ✅ RPN programs executable by PTX kernels

**Galaxy-First design:**
- Equations live in Galaxy Universe (not hardcoded in solver logic)
- TRM learns which equations to query (not manually dispatched)
- Shadow copy records successful cross-domain transfers

**Multi-curriculum contribution:**
- These equations help math benchmarks NOW
- Will help ARC-AGI visual tasks (spatial rate problems)
- Will help physics simulations (same equations, different domain)

---

## Expected Timeline

- **Hour 1:** Create `generic_equations.py` with 7 equations + loader
- **Hour 2:** Wire into GalaxyUniverse initialization + TRMGalaxyReader exploration
- **Hour 3:** Add shadow copy tracking for cross-domain usage
- **Hour 4:** Validation (run benchmarks, verify improvements)

---

## Handoff to Codex

**Codex:** Implement Phase 1 (seed generic equations) as specified above.

**After completion:**
1. Report equation counts and domains loaded
2. Show query test results (domain + variable queries)
3. Run shuffled GSM8K (200 problems) and report accuracy change
4. Show 5 example cross-domain usage traces from logs

**Then:** Return to Claude for architecture review before proceeding to Phase 2 (test-time compute infrastructure).

---

**Architect:** Claude (Architecture Partner)
**Implementer:** Codex (Implementation Lead)

**Status:** Ready for implementation
**Priority:** CRITICAL - Foundation for test-time compute
