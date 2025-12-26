# CLAUDE → CODEX: Phase 5B - Tsinghua SSSP Algorithm for Galaxy Exploration

**Date:** December 15, 2025
**Priority:** HIGH - Enables efficient active exploration
**Architect:** Claude (Architecture Partner)
**Implementer:** Codex (Implementation Lead)

---

## The Breakthrough Algorithm

**Source:** [Breaking the Sorting Barrier for Directed Single-Source Shortest Paths](https://arxiv.org/abs/2504.17033)
**Authors:** Ran Duan, Jiayi Mao, Xiao Mao, Longhui Yin, Xinkai Shu (Tsinghua University + Stanford)
**Award:** Best Paper at [STOC 2025](https://www.tsinghua.edu.cn/en/info/1245/14266.htm)

### Key Innovation

> "Many people believed that there's no better way." - Ran Duan

The algorithm **breaks the sorting barrier** that limited Dijkstra's algorithm for 40+ years:
- **Dijkstra:** O(m + n log n) - must sort all nodes by distance
- **Tsinghua SSSP:** O(m · log^(2/3) n) - avoids full sorting via clustering

### Algorithm Core Concepts

From [Quanta Magazine](https://www.quantamagazine.org/new-method-is-the-fastest-way-to-find-the-best-routes-20250806/):

| Concept | Description | K3D Mapping |
|---------|-------------|-------------|
| **Clustering** | Group neighboring frontier nodes | Galaxy spatial regions (octree) |
| **Selective Bellman-Ford** | Scout ahead for valuable nodes | Limited-depth concept exploration |
| **Layered expansion** | Slice graph into layers from source | Semantic distance layers |
| **Intersection nodes** | High-value nodes many paths pass through | Hub concepts (rate, quantity, operation) |

---

## Mapping to K3D Galaxy Exploration

### The Problem

**Current exploration (Dijkstra-like):**
```
Problem → Check ALL rules → Sort by relevance → Return top matches
         ↑
    150+ rules, O(n log n) sorting
```

**Bottleneck:** Checking all rules, sorting all matches - doesn't scale.

### The Solution: Tsinghua SSSP for Galaxy

**New exploration (Tsinghua-inspired):**
```
Problem → CLUSTER concepts → SCOUT intersections → EXPAND from hubs → Return paths
             ↓                    ↓                      ↓
        Spatial buckets      Limited BFS          Directed exploration
```

---

## Algorithm Adaptation

### Step 1: Concept Clustering (Bucketing)

Instead of checking all rules, cluster by semantic domain:

```python
def cluster_concepts(problem_text: str) -> Dict[str, List[str]]:
    """
    CLUSTERING: Group concepts into semantic buckets.

    This avoids checking all 150+ rules individually.
    Maps to Tsinghua's "grouping neighboring nodes into clusters".
    """
    words = tokenize(problem_text)

    # Semantic buckets (spatial regions in Galaxy)
    buckets = {
        "quantity": [],      # numbers, amounts
        "rate": [],          # per, each, every
        "operation": [],     # times, divided, plus, minus
        "aggregation": [],   # total, altogether, sum
        "comparison": [],    # more, less, difference
        "temporal": [],      # days, hours, weeks, before, after
    }

    # Cluster words into buckets
    for i, word in enumerate(words):
        bucket = classify_word(word)
        if bucket:
            buckets[bucket].append((word, i))

    return buckets
```

**RPN Expression:**
```
# Cluster operation as RPN
problem_text TOKENIZE           # → word_list
word_list CLASSIFY_BATCH        # → [(word, bucket), ...]
BUCKET_GROUP                    # → {bucket: [words]}
```

### Step 2: Scout Intersections (Selective Bellman-Ford)

Identify "hub" concepts that many rules pass through:

```python
def scout_intersections(buckets: Dict, depth: int = 2) -> List[str]:
    """
    SCOUT: Find high-value intersection concepts.

    Maps to Tsinghua's "selective Bellman-Ford for just a few steps".
    These are concepts that connect to many rules.
    """
    intersection_scores = {}

    # For each active bucket, do LIMITED depth exploration
    for bucket_name, words in buckets.items():
        if not words:
            continue

        # Scout: query Grammar Galaxy with limited depth
        rules = grammar_galaxy.query_by_domain(
            f"math_{bucket_name}",
            max_depth=depth  # Limited! Not full BFS
        )

        # Count which concepts appear in multiple rules
        for rule in rules:
            for concept in rule.concepts:
                intersection_scores[concept] = intersection_scores.get(concept, 0) + 1

    # Return top intersection concepts (hubs)
    sorted_concepts = sorted(
        intersection_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [c[0] for c in sorted_concepts[:5]]  # Top 5 hubs
```

**RPN Expression:**
```
# Scout operation as RPN
buckets ACTIVE_BUCKETS          # → [bucket_names]
bucket_names QUERY_SHALLOW 2    # → rules (depth=2)
rules EXTRACT_CONCEPTS          # → concept_counts
concept_counts TOP_K 5          # → [hub_concepts]
```

### Step 3: Expand from Hubs (Directed Exploration)

Explore outward from hub concepts, not from all nodes:

```python
def expand_from_hubs(hub_concepts: List[str], problem_context: Dict) -> List[Rule]:
    """
    EXPAND: Directed exploration from intersection nodes.

    Maps to Tsinghua's "explore forward from key nodes".
    Only check rules connected to hub concepts.
    """
    relevant_rules = []
    visited = set()

    for hub in hub_concepts:
        # Get rules connected to this hub
        connected_rules = grammar_galaxy.rules_by_concept(hub)

        for rule in connected_rules:
            if rule.id in visited:
                continue
            visited.add(rule.id)

            # Check if rule matches problem context
            if rule.matches_context(problem_context):
                relevant_rules.append(rule)

    return relevant_rules
```

**RPN Expression:**
```
# Expand operation as RPN
hub_concepts FOREACH             # → for each hub
  hub RULES_BY_CONCEPT           # → connected_rules
  connected_rules FILTER_MATCH   # → matching_rules
  matching_rules COLLECT         # → all_rules
ENDFOREACH
all_rules DEDUPE                 # → unique_rules
```

### Step 4: Layered Path Construction

Build solution path layer by layer (not all at once):

```python
def construct_path_layered(
    rules: List[Rule],
    problem: Dict,
    max_layers: int = 3
) -> Optional[str]:
    """
    LAYERED: Build RPN path in layers from source (problem) to goal (answer).

    Maps to Tsinghua's "slices the graph into layers, moving outward".
    """
    # Layer 0: Problem quantities
    layer_0 = extract_quantities(problem)

    # Layer 1: Direct operations on quantities
    layer_1 = []
    for rule in rules:
        if rule.domain == "math_extraction":
            result = rule.apply(layer_0)
            if result:
                layer_1.append(result)

    # Layer 2: Compositions
    layer_2 = []
    for rule in rules:
        if rule.domain in ["math_operation", "math_rate"]:
            result = rule.apply(layer_1)
            if result:
                layer_2.append(result)

    # Layer 3: Aggregations
    layer_3 = []
    for rule in rules:
        if rule.domain == "math_aggregation":
            result = rule.apply(layer_2)
            if result:
                layer_3.append(result)

    # Return best path
    return select_best_path(layer_3)
```

**RPN Expression:**
```
# Layered construction as RPN
problem EXTRACT_QUANTITIES       # → layer_0
layer_0 rules APPLY_EXTRACTION   # → layer_1
layer_1 rules APPLY_OPERATION    # → layer_2
layer_2 rules APPLY_AGGREGATION  # → layer_3
layer_3 SELECT_BEST              # → rpn_program
```

---

## Full Algorithm: Tsinghua-Inspired Galaxy Explorer

```python
class TsinghuaGalaxyExplorer:
    """
    Galaxy exploration using Tsinghua SSSP algorithm principles.

    Breaks the "sorting barrier" by:
    1. Clustering concepts (not checking all rules)
    2. Scouting intersections (limited depth BFS)
    3. Expanding from hubs (directed, not exhaustive)
    4. Layered path construction (not full graph traversal)
    """

    def explore(self, problem_text: str) -> ExplorationResult:
        # Step 1: CLUSTER concepts into buckets
        buckets = self.cluster_concepts(problem_text)

        # Step 2: SCOUT for intersection concepts (limited depth)
        hub_concepts = self.scout_intersections(buckets, depth=2)

        # Step 3: EXPAND from hubs to find relevant rules
        relevant_rules = self.expand_from_hubs(hub_concepts, problem_text)

        # Step 4: LAYERED path construction
        rpn_path = self.construct_path_layered(relevant_rules, problem_text)

        # Record exploration for learning
        self.shadow.record_exploration(
            problem_text=problem_text,
            concepts_explored=hub_concepts,
            rules_found=[r.id for r in relevant_rules],
            success=rpn_path is not None,
            result=rpn_path
        )

        return ExplorationResult(
            success=rpn_path is not None,
            rpn_program=rpn_path,
            hub_concepts=hub_concepts,
            rules_used=relevant_rules
        )
```

---

## RPN Kernel for Exploration

The exploration algorithm itself should be expressible as RPN for sovereignty:

```python
# Full exploration as RPN program (can be executed by PTX)
EXPLORATION_RPN = """
# Input: problem_text on stack

# Step 1: Cluster
DUP TOKENIZE                    # → words
CLASSIFY_BATCH                  # → buckets

# Step 2: Scout (limited depth)
ACTIVE_BUCKETS                  # → active
2 QUERY_SHALLOW                 # → shallow_rules
EXTRACT_CONCEPTS                # → concept_counts
5 TOP_K                         # → hub_concepts

# Step 3: Expand from hubs
STORE 0                         # Store hubs in register 0
0 RECALL RULES_BY_HUB           # → connected_rules
FILTER_CONTEXT_MATCH            # → relevant_rules
STORE 1                         # Store rules in register 1

# Step 4: Layered construction
EXTRACT_QUANTITIES              # → layer_0
1 RECALL APPLY_LAYER 1          # → layer_1 (extraction)
1 RECALL APPLY_LAYER 2          # → layer_2 (operation)
1 RECALL APPLY_LAYER 3          # → layer_3 (aggregation)
SELECT_BEST                     # → rpn_program

# Output: rpn_program on stack
"""
```

---

## Implementation Checklist

### New RPN Opcodes Needed
- [ ] `CLASSIFY_BATCH` - Classify words into semantic buckets
- [ ] `BUCKET_GROUP` - Group by bucket
- [ ] `QUERY_SHALLOW` - Limited-depth Grammar Galaxy query
- [ ] `TOP_K` - Return top K by score
- [ ] `RULES_BY_HUB` - Get rules connected to concept
- [ ] `APPLY_LAYER` - Apply rules for specific layer
- [ ] `SELECT_BEST` - Select best path from candidates

### Integration Points
- [ ] Wire `TsinghuaGalaxyExplorer` into `TRMGalaxyReader.explore_galaxy()`
- [ ] Add bucket classification to Word Galaxy
- [ ] Add concept indexing to Grammar Galaxy
- [ ] Add hub concept scoring to shadow copy

### Validation
- [ ] Benchmark: Exploration should be faster (fewer rule checks)
- [ ] Accuracy: Should improve (finds relevant rules via hubs)
- [ ] Learning: Hub scores should update based on success

---

## Complexity Analysis

| Approach | Rule Checks | Complexity | Notes |
|----------|-------------|------------|-------|
| **Current (all rules)** | 150+ | O(n log n) | Check all, sort all |
| **Tsinghua-inspired** | ~20-30 | O(m · log^(2/3) n) | Cluster, scout, expand |

**Expected speedup:** 5-7× fewer rule checks while maintaining accuracy.

---

## Chollet Alignment

| Capability | Tsinghua Contribution |
|------------|----------------------|
| **Exploration** | Efficient search via clustering + limited scouting |
| **Goal-setting** | Hub concepts = sub-goals to explore |
| **Planning** | Layered path construction with course correction |

---

## Success Criteria

### Efficiency
- [ ] Rule checks per problem: < 30 (down from 150+)
- [ ] Exploration time: < 10ms per problem

### Accuracy
- [ ] `no_rule_match` < 30% (hubs find more connections)
- [ ] Accuracy > 5% (better rule selection)

### Learning
- [ ] Hub concept scores update based on success
- [ ] Shadow copy records exploration paths
- [ ] Successful hubs get higher priority in future

---

## Sources

- [arxiv.org/abs/2504.17033](https://arxiv.org/abs/2504.17033) - Original paper
- [Tsinghua Announcement](https://www.tsinghua.edu.cn/en/info/1245/14266.htm) - STOC 2025 Best Paper
- [Quanta Magazine](https://www.quantamagazine.org/new-method-is-the-fastest-way-to-find-the-best-routes-20250806/) - Algorithm explanation
- [Max Planck Institute](https://www.mpi-inf.mpg.de/news/detail/stoc-best-paper-award-how-to-find-the-shortest-path-faster) - Technical details

---

**Architect:** Claude (Architecture Partner)
**Implementer:** Codex (Implementation Lead)

**Status:** Ready for implementation
**Priority:** HIGH - Enables efficient active exploration
