# PM-KR X3D Adaptive Reasoning Component

**Version**: 0.1 (Initial Draft)
**Status**: PM-KR Community Group Working Draft
**Date**: March 26, 2026
**Authors**: PM-KR Community Group (Daniel Campos Ramos, Chair; Milton Ponson, Co-Chair)
**Liaison**: Web3D Consortium (Don Brutzman, Advisory Committee Representative)
**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Reference Implementation)

**Normative References**:
- ISO/IEC 19775-1:2023 (X3D Architecture and Base Components, Version 4.0)
- PM-KR X3D Procedural Memory Component v0.1 (docs/w3c/x3d/PM_KR_X3D_PROCEDURAL_MEMORY_COMPONENT.md)
- PM-KR X3D Avatar Embodiment Specification v0.1 (docs/w3c/x3d/PM_KR_X3D_AVATAR_SPECIFICATION.md)
- PM-KR X3D Ontology v0.1 (docs/w3c/x3d/PM_KR_X3D_ONTOLOGY.md)
- Adaptive Reasoning Budget Specification v1.0 (docs/vocabulary/ADAPTIVE_REASONING_BUDGET_SPECIFICATION.md)
- Hyper-Parallel Processing Specification v1.0 (docs/vocabulary/HYPER_PARALLEL_PROCESSING.md)
- Knowledgeverse Specification v5.1 (docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Scope and Design Rationale](#2-scope-and-design-rationale)
3. [Component Definition: AdaptiveReasoning](#3-component-definition-adaptivereasoning)
4. [Abstract Node Types](#4-abstract-node-types)
5. [Concrete Node Reference](#5-concrete-node-reference)
6. [Budget Visualization](#6-budget-visualization)
7. [Decomposition Tree Interchange](#7-decomposition-tree-interchange)
8. [OWL 2 Ontology Extension](#8-owl-2-ontology-extension)
9. [Relationship to Existing X3D Components](#9-relationship-to-existing-x3d-components)
10. [Conformance](#10-conformance)
11. [Examples](#11-examples)

---

## 1. Introduction

### 1.1 Purpose

This document defines the **AdaptiveReasoning** component for X3D, extending the KnowledgeNavigation component (PM-KR X3D Procedural Memory Component §10) and the AvatarEmbodiment component (PM-KR X3D Avatar Embodiment Specification) with node types for:

- **Ternary-gated computation budgets** that dynamically control reasoning depth based on knowledge confidence signals.
- **Recursive sub-task decomposition trees** serialized as X3D scene graph structures for interchange, visualization, and audit.
- **Parallel-to-serial scheduling state** for representing the execution queue, worker assignments, and priority ordering.
- **Budget visualization** as 3D geometry within the avatar's Cranial Galaxy, making the reasoning process inspectable by human observers.

### 1.2 Motivation

The PM-KR Procedural Memory Component defines knowledge nodes and agent entities. The Avatar Embodiment Specification defines the avatar's body and cranial Galaxy. Neither addresses HOW MUCH reasoning the agent should apply to a given query. The Adaptive Reasoning Budget Specification (docs/vocabulary/ADAPTIVE_REASONING_BUDGET_SPECIFICATION.md) defines the computational governance model. This X3D component provides the **scene graph representation** for that governance model — enabling:

1. **Interchange**: Export/import reasoning configurations between K3D systems.
2. **Visualization**: Render budget allocation, decomposition trees, and scheduling state as 3D geometry inside the Cranial Galaxy.
3. **Audit**: Inspect past reasoning episodes by loading their decomposition trees as X3D scenes.
4. **Federation**: Share reasoning sub-tasks between Houses by exchanging X3D-encoded decomposition nodes.

---

## 2. Scope and Design Rationale

### 2.1 What This Component Defines

1. **Node types** for budget configuration, sub-task decomposition, priority scheduling, and intermediate result persistence.
2. **Field semantics** for ternary signals, budget parameters, decomposition depth, and worker assignments.
3. **Visualization mapping** from budget state to 3D geometry.
4. **OWL 2 classes** extending the PM-KR X3D Ontology with adaptive reasoning concepts.

### 2.2 What This Component Does Not Define

- GPU kernel implementations for budget computation (sovereign execution concern).
- Runtime scheduling algorithms (the work-stealing protocol is an implementation detail).
- Training procedures for learned budget allocation (future extension).
- Specific RPN opcode definitions (see RPN Domain Opcode Registry).

---

## 3. Component Definition: AdaptiveReasoning

### 3.1 Component Overview

| Property | Value |
|----------|-------|
| **Component name** | AdaptiveReasoning |
| **Component level** | 1 (single level) |
| **Dependencies** | KnowledgeNavigation (level 1), AvatarEmbodiment (level 1), ProceduralMemory (level 1) |
| **New abstract types** | X3DReasoningNode, X3DBudgetNode |
| **New concrete types** | ReasoningBudget, DecompositionTask, TaskQueue, BudgetVisualization, IntermediateResult |

### 3.2 Support Levels

| Level | Nodes Required |
|-------|---------------|
| **Level 1** | ReasoningBudget, DecompositionTask, TaskQueue, IntermediateResult, BudgetVisualization |

---

## 4. Abstract Node Types

### 4.1 X3DReasoningNode

```
X3DReasoningNode : X3DChildNode {
    SFString [in,out] queryId       ""
    SFInt32  [in,out] ternarySignal 0      # {-1, 0, +1}
    SFFloat  [in,out] confidence    0.0    # [0.0, 1.0]
    SFTime   [in,out] timestamp     0
}
```

Base type for all adaptive reasoning nodes. Every reasoning node is associated with a query (via `queryId`) and carries the ternary signal and confidence at the time of creation.

### 4.2 X3DBudgetNode

```
X3DBudgetNode : X3DReasoningNode {
    SFInt32  [in,out] budgetAllocated  5    # Total iterations allocated
    SFInt32  [in,out] budgetRemaining  5    # Iterations remaining
    SFInt32  [in,out] budgetMinimum    5    # Minimum before halting allowed
    SFInt32  [in,out] iterationsUsed   0    # Iterations completed
}
```

Base type for nodes that carry budget state.

---

## 5. Concrete Node Reference

### 5.1 ReasoningBudget

```
ReasoningBudget : X3DBudgetNode {
    SFInt32  [in,out] baseIterations       5     # B_base
    SFFloat  [in,out] minimumFraction      0.5   # min_fraction for B_min
    SFInt32  [in,out] decompositionThreshold 20   # T_decomp
    SFInt32  [in,out] maxDecompositionDepth 8     # D_max
    SFInt32  [in,out] activeWorkers        9     # Current active swarm workers
    SFString [in,out] memoryWatermark      "GREEN"  # GREEN/YELLOW/ORANGE/RED
    SFFloat  [in,out] aspirationLevel      0.85  # Convergence aspiration
    MFNode   [in,out] decompositionTasks   []    # DecompositionTask children
    SFNode   [in,out] taskQueue            NULL  # TaskQueue for serialization
    SFNode   [in,out] visualization        NULL  # BudgetVisualization
}
```

**Description**: Top-level configuration node for the Adaptive Reasoning Budget. One ReasoningBudget node exists per active query in the agent's reasoning context.

**Field Semantics**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `baseIterations` | SFInt32 | 5 | B_base — the 87th percentile convergence point |
| `minimumFraction` | SFFloat | 0.5 | Fraction of B(q) that constitutes B_min |
| `decompositionThreshold` | SFInt32 | 20 | B(q) above which decomposition is triggered |
| `maxDecompositionDepth` | SFInt32 | 8 | Maximum recursion depth for sub-task decomposition |
| `activeWorkers` | SFInt32 | 9 | Number of swarm workers currently active |
| `memoryWatermark` | SFString | "GREEN" | Current Knowledgeverse memory pressure level |
| `aspirationLevel` | SFFloat | 0.85 | Convergence threshold (adjusted by ternary signal) |
| `decompositionTasks` | MFNode | [] | Child DecompositionTask nodes forming the decomposition tree |
| `taskQueue` | SFNode | NULL | TaskQueue for serialized overflow tasks |
| `visualization` | SFNode | NULL | BudgetVisualization for 3D rendering |

**Budget Computation**: The allocated budget is computed as:
```
budgetAllocated = baseIterations × 2^(1 − ternarySignal)
budgetMinimum = max(baseIterations, budgetAllocated × minimumFraction)
```

**Watermark Adjustment**: When `memoryWatermark` is not GREEN:
- YELLOW: budgetAllocated × 0.75, maxDecompositionDepth = min(D_max, 4)
- ORANGE: budgetAllocated × 0.50, maxDecompositionDepth = min(D_max, 2)
- RED: budgetAllocated = budgetMinimum, maxDecompositionDepth = 0

### 5.2 DecompositionTask

```
DecompositionTask : X3DBudgetNode {
    SFString [in,out] taskId              ""
    SFString [in,out] parentTaskId        ""     # Empty for root task
    SFInt32  [in,out] decompositionDepth  0
    SFString [in,out] strategy            "direct"  # direct/mathematical/multihop/dialectical/temporal
    SFString [in,out] status              "pending" # pending/active/completed/failed
    SFFloat  [in,out] priority            0.0
    SFBool   [in,out] onCriticalPath      FALSE
    SFInt32  [in,out] assignedWorker      -1    # Swarm worker index, -1 if unassigned
    MFString [in,out] dependsOn           []    # taskIds this task depends on
    MFNode   [in,out] subtasks            []    # Child DecompositionTask nodes
    SFNode   [in,out] result              NULL  # IntermediateResult when completed
    SFString [in,out] galaxyDomain        ""    # Target Galaxy for this sub-task
    SFString [in,out] meaningProgram      ""    # RPN program defining the sub-query
}
```

**Description**: Represents a single node in the recursive decomposition tree. Can contain child `subtasks` forming the tree structure. Each task carries its own budget, ternary signal, priority, and result.

**Field Semantics**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `taskId` | SFString | "" | Unique identifier for this task |
| `parentTaskId` | SFString | "" | Parent task (empty for root) |
| `decompositionDepth` | SFInt32 | 0 | Depth in the decomposition tree |
| `strategy` | SFString | "direct" | Decomposition strategy used |
| `status` | SFString | "pending" | Current execution status |
| `priority` | SFFloat | 0.0 | Computed priority for scheduling |
| `onCriticalPath` | SFBool | FALSE | Whether this task is on the DAG critical path |
| `assignedWorker` | SFInt32 | -1 | Index of swarm worker executing this task |
| `dependsOn` | MFString | [] | Task IDs that must complete before this task |
| `subtasks` | MFNode | [] | Child decomposition tasks (recursive) |
| `result` | SFNode | NULL | IntermediateResult node when task completes |
| `galaxyDomain` | SFString | "" | Which Galaxy domain this sub-task targets |
| `meaningProgram` | SFString | "" | RPN program defining the sub-query |

**Decomposition Strategies**:

| Strategy | Description | Typical σ Trigger |
|----------|-------------|-------------------|
| `direct` | No decomposition — resolve by Galaxy navigation | σ = +1 |
| `mathematical` | Break proof into lemmas with dependency edges | σ = −1, math domain |
| `multihop` | Break multi-domain query into domain-specific sub-queries | σ = 0, multiple galaxies |
| `dialectical` | Break contradiction into arguments-for, arguments-against, synthesis | σ = −1, contradiction |
| `temporal` | Break process into initial → transitions → final states | σ = 0, temporal domain |

### 5.3 TaskQueue

```
TaskQueue : X3DChildNode {
    MFNode   [in,out] pending      []    # DecompositionTask nodes awaiting execution
    MFNode   [in,out] active       []    # DecompositionTask nodes currently executing
    MFNode   [in,out] completed    []    # DecompositionTask nodes finished
    SFInt32  [in,out] maxParallel  9     # Maximum parallel execution slots
    SFInt32  [in,out] activeCount  0     # Current number of active tasks
    SFBool   [in,out] saturated    FALSE # TRUE when activeCount >= maxParallel
}
```

**Description**: Priority queue for managing sub-task execution when parallel capacity is exceeded. Tasks in `pending` are ordered by priority. When a worker completes its current task, it pulls the highest-priority pending task.

### 5.4 IntermediateResult

```
IntermediateResult : X3DReasoningNode {
    SFString [in,out] starId           ""     # Content-addressed star ID
    SFString [in,out] meaningProgram   ""     # RPN program of the result
    SFString [in,out] visualProgram    ""     # Visualization RPN
    SFInt32  [in,out] layer            2      # PM-KR layer (1-4)
    SFFloat  [in,out] confidence       0.0
    SFString [in,out] galaxyDomain     ""
    SFString [in,out] provenance       ""     # JSON: parent_query, depth, budget_used, worker_id
    MFString [in,out] symlinkRefs      []     # References to other stars used
    MFString [in,out] taxonomyRefs     []     # Discovered taxonomy links
    SFBool   [in,out] persisted        FALSE  # TRUE when written to Galaxy
}
```

**Description**: An intermediate reasoning result that will be (or has been) persisted as a MeaningCentricStar in the Knowledgeverse. The `persisted` field indicates whether the write has occurred. Content-addressed `starId` enables deduplication.

### 5.5 BudgetVisualization

```
BudgetVisualization : X3DGroupingNode {
    SFFloat  [in,out] sphereRadius     1.0    # Proportional to budgetAllocated
    SFColor  [in,out] sphereColor      0.3 0.6 1.0  # Blue = normal, yellow = pressure, red = exhausted
    SFFloat  [in,out] sphereOpacity    0.3
    MFNode   [in,out] traceGeometry    []     # NavigationTrace geometry for decomposition branches
    MFNode   [in,out] queueDots        []     # Dots orbiting for serialized tasks
    SFBool   [in,out] converged        FALSE  # Triggers merge animation when TRUE
    SFFloat  [in,out] pulseFrequency   1.0    # Hz — faster under memory pressure
}
```

**Description**: 3D visualization of the budget state, rendered inside the avatar's Cranial Galaxy. The sphere radius grows with budget allocation, branches show decomposition, orbiting dots show serialized queue depth, and convergence triggers a merge animation.

---

## 6. Budget Visualization

### 6.1 Visual Language

| Visual Element | Budget State | Geometry |
|---------------|-------------|----------|
| **Sphere radius** | Budget allocated (B(q)) | Sphere around TRM core, radius ∝ B(q) |
| **Sphere color** | Memory watermark | Blue (GREEN), Yellow (YELLOW), Orange (ORANGE), Red (RED) |
| **Branching traces** | Sub-task decomposition | Lines from sphere center outward, one per sub-task |
| **Trace color** | Sub-task ternary signal | Green (+1), Yellow (0), Red (−1) |
| **Orbiting dots** | Serialized queue | Small spheres orbiting, count = pending tasks |
| **Dot speed** | Task priority | Faster orbit = higher priority |
| **Merge flash** | Convergence | All traces collapse to center, white flash |
| **Sphere dimming** | Budget exhaustion | Opacity decreases as budget_remaining → 0 |
| **Pulse frequency** | Memory pressure | Faster pulse under higher watermark levels |

### 6.2 Dual-Client Visualization

**Human client**: Sees the 3D sphere, traces, and animations. Intuitively understands: big sphere = hard problem, many branches = decomposed, fast orbits = queued.

**AI client**: Reads the numeric fields (budgetAllocated, iterationsUsed, ternarySignal, priority) from the node's `extras.k3d` metadata. Uses these for scheduling, federation, and audit.

Same data, different perception — maintaining the Dual-Client Contract.

---

## 7. Decomposition Tree Interchange

### 7.1 Scene Graph Encoding

A decomposition tree is encoded as nested DecompositionTask nodes:

```xml
<ReasoningBudget queryId="q_goldbach" ternarySignal="-1"
                 budgetAllocated="20" baseIterations="5"
                 maxDecompositionDepth="8">
  <DecompositionTask taskId="t_root" strategy="mathematical"
                     decompositionDepth="0" status="completed"
                     ternarySignal="-1" budgetAllocated="20">

    <DecompositionTask taskId="t_even_decomp" strategy="mathematical"
                       decompositionDepth="1" status="completed"
                       ternarySignal="0" budgetAllocated="10"
                       dependsOn='"t_prime_lookup"'
                       galaxyDomain="Math">
      <IntermediateResult starId="abc123..." layer="3"
                          confidence="0.82" persisted="true"
                          meaningProgram="RECALL_even RECALL_prime_pair SUM EQ"/>
    </DecompositionTask>

    <DecompositionTask taskId="t_prime_lookup" strategy="direct"
                       decompositionDepth="1" status="completed"
                       ternarySignal="1" budgetAllocated="5"
                       galaxyDomain="Math">
      <IntermediateResult starId="def456..." layer="2"
                          confidence="0.98" persisted="true"
                          meaningProgram="RECALL_sieve_eratosthenes EXEC"/>
    </DecompositionTask>

    <DecompositionTask taskId="t_synthesis" strategy="dialectical"
                       decompositionDepth="1" status="completed"
                       ternarySignal="-1" budgetAllocated="20"
                       dependsOn='"t_even_decomp" "t_prime_lookup"'
                       galaxyDomain="Math">
      <IntermediateResult starId="ghi789..." layer="4"
                          confidence="0.71" persisted="true"
                          meaningProgram="RECALL_partial_result RECALL_counterexample_search EVAL"/>
    </DecompositionTask>
  </DecompositionTask>
</ReasoningBudget>
```

### 7.2 glTF Encoding

In glTF, the decomposition tree is stored in `extras.k3d.reasoning`:

```json
{
  "extras": {
    "k3d": {
      "reasoning": {
        "queryId": "q_goldbach",
        "ternarySignal": -1,
        "budgetAllocated": 20,
        "budgetUsed": 18,
        "decompositionTree": {
          "taskId": "t_root",
          "strategy": "mathematical",
          "subtasks": [
            {
              "taskId": "t_even_decomp",
              "ternarySignal": 0,
              "budgetAllocated": 10,
              "result": { "starId": "abc123...", "confidence": 0.82 }
            },
            {
              "taskId": "t_prime_lookup",
              "ternarySignal": 1,
              "budgetAllocated": 5,
              "result": { "starId": "def456...", "confidence": 0.98 }
            }
          ]
        }
      }
    }
  }
}
```

---

## 8. OWL 2 Ontology Extension

### 8.1 New Classes

```turtle
@prefix k3d:  <https://knowledge3d.org/ontology/> .
@prefix k3da: <https://knowledge3d.org/ontology/agent/> .

k3d:AdaptiveReasoningBudget a owl:Class ;
    rdfs:subClassOf k3d:Process ;
    rdfs:label "Adaptive Reasoning Budget"@en ;
    rdfs:comment """Computational governance controlling reasoning depth,
    decomposition, and scheduling based on ternary knowledge signals."""@en .

k3d:DecompositionTask a owl:Class ;
    rdfs:subClassOf k3d:Process ;
    rdfs:label "Decomposition Task"@en ;
    rdfs:comment """A node in the recursive decomposition tree. Carries its
    own budget, ternary signal, and result."""@en .

k3d:TaskQueue a owl:Class ;
    rdfs:label "Task Queue"@en ;
    rdfs:comment """Priority queue for serialized sub-tasks when parallel
    capacity is exceeded."""@en .

k3d:IntermediateResult a owl:Class ;
    rdfs:subClassOf k3d:MeaningCentricStar ;
    rdfs:label "Intermediate Result"@en ;
    rdfs:comment """A reasoning result persisted as a star. Content-addressed
    for deduplication. Produced by sub-task decomposition."""@en .

k3d:DecompositionStrategy a owl:Class ;
    owl:oneOf ( k3d:DirectStrategy k3d:MathematicalStrategy
                k3d:MultihopStrategy k3d:DialecticalStrategy
                k3d:TemporalStrategy ) ;
    rdfs:label "Decomposition Strategy"@en .
```

### 8.2 New Properties

```turtle
k3d:budgetAllocated a owl:DatatypeProperty ;
    rdfs:domain k3d:AdaptiveReasoningBudget ;
    rdfs:range xsd:nonNegativeInteger .

k3d:budgetRemaining a owl:DatatypeProperty ;
    rdfs:domain k3d:AdaptiveReasoningBudget ;
    rdfs:range xsd:nonNegativeInteger .

k3d:budgetMinimum a owl:DatatypeProperty ;
    rdfs:domain k3d:AdaptiveReasoningBudget ;
    rdfs:range xsd:nonNegativeInteger .

k3d:compositeSignal a owl:DatatypeProperty ;
    rdfs:domain k3d:AdaptiveReasoningBudget ;
    rdfs:range k3d:TritValue .

k3d:decompositionDepth a owl:DatatypeProperty ;
    rdfs:domain k3d:DecompositionTask ;
    rdfs:range xsd:nonNegativeInteger .

k3d:hasSubtask a owl:ObjectProperty ;
    rdfs:domain k3d:DecompositionTask ;
    rdfs:range k3d:DecompositionTask .

k3d:dependsOn a owl:ObjectProperty, owl:TransitiveProperty ;
    rdfs:domain k3d:DecompositionTask ;
    rdfs:range k3d:DecompositionTask .

k3d:producesResult a owl:ObjectProperty ;
    rdfs:domain k3d:DecompositionTask ;
    rdfs:range k3d:IntermediateResult .

k3d:usesStrategy a owl:ObjectProperty ;
    rdfs:domain k3d:DecompositionTask ;
    rdfs:range k3d:DecompositionStrategy .

k3d:onCriticalPath a owl:DatatypeProperty ;
    rdfs:domain k3d:DecompositionTask ;
    rdfs:range xsd:boolean .

k3d:taskPriority a owl:DatatypeProperty ;
    rdfs:domain k3d:DecompositionTask ;
    rdfs:range xsd:float .
```

### 8.3 SPARQL Query: Find Deep Reasoning Episodes

```sparql
PREFIX k3d: <https://knowledge3d.org/ontology/>

SELECT ?query ?depth ?budgetUsed ?signal WHERE {
    ?budget a k3d:AdaptiveReasoningBudget ;
            k3d:queryId ?query ;
            k3d:compositeSignal ?signal ;
            k3d:budgetAllocated ?budgetUsed .
    ?budget k3d:hasSubtask+ ?deepTask .
    ?deepTask k3d:decompositionDepth ?depth .
    FILTER(?depth >= 3)
}
ORDER BY DESC(?depth)
```

---

## 9. Relationship to Existing X3D Components

| Existing Component | Relationship | Integration Point |
|-------------------|-------------|-------------------|
| **KnowledgeNavigation** | AdaptiveReasoning wraps navigation with budget control | AgentEntity receives ReasoningBudget as reasoning governor |
| **AvatarEmbodiment** | Budget state visualized in CranialGalaxy | BudgetVisualization is child of CranialGalaxy |
| **ProceduralMemory** | IntermediateResults are ProceduralMemoryNodes | IntermediateResult extends MeaningCentricStar (via PM-KR) |
| **Grouping** | DecompositionTask uses standard X3D parent-child nesting | Subtasks are MFNode children |
| **Time** | Budget iteration timing uses TimeSensor | TimeSensor drives budget countdown visualization |

---

## 10. Conformance

### 10.1 Producer Conformance

A conforming AdaptiveReasoning producer MUST:

**Level 1:**
- Emit valid ReasoningBudget nodes with correct budget computation from ternary signal
- Emit IntermediateResult nodes for all completed sub-tasks
- Set `persisted="true"` only after the result has been written to Knowledgeverse

**Level 2:**
- Emit DecompositionTask trees with correct depth and dependency edges
- Emit TaskQueue with priority ordering
- Produce valid glTF `extras.k3d.reasoning` blocks

**Level 3:**
- Emit BudgetVisualization with correct visual mapping
- Produce dual-client visualization (3D geometry + metadata)

### 10.2 Consumer Conformance

A conforming AdaptiveReasoning consumer MUST:

**Level 1:**
- Parse ReasoningBudget and its fields
- Reconstruct budget computation from stored parameters
- Retrieve IntermediateResult stars from Knowledgeverse by starId

**Level 2:**
- Reconstruct DecompositionTask trees including dependencies
- Evaluate critical path from dependency DAG
- Replay scheduling from TaskQueue state

---

## 11. Examples

### 11.1 Simple Query (σ = +1): "What is 2 + 3?"

```xml
<ReasoningBudget queryId="q_simple_add" ternarySignal="1"
                 budgetAllocated="5" budgetMinimum="5"
                 iterationsUsed="2" memoryWatermark="GREEN">
  <!-- No decomposition needed — direct resolution -->
  <DecompositionTask taskId="t_root" strategy="direct"
                     status="completed" ternarySignal="1"
                     assignedWorker="0" budgetAllocated="5">
    <IntermediateResult starId="a1b2c3..."
                        meaningProgram="2 3 ADD"
                        confidence="1.0" layer="2"
                        persisted="true" galaxyDomain="Math"/>
  </DecompositionTask>
</ReasoningBudget>
```

Budget: 5 iterations allocated, 2 used. No decomposition. Single worker.

### 11.2 Complex Query (σ = −1): Multi-Domain Contradiction

```xml
<ReasoningBudget queryId="q_paradox" ternarySignal="-1"
                 budgetAllocated="20" budgetMinimum="10"
                 iterationsUsed="18" maxDecompositionDepth="8"
                 activeWorkers="9" memoryWatermark="GREEN">

  <DecompositionTask taskId="t_root" strategy="dialectical"
                     decompositionDepth="0" status="completed"
                     ternarySignal="-1" budgetAllocated="20">

    <!-- Arguments FOR -->
    <DecompositionTask taskId="t_pro" strategy="multihop"
                       decompositionDepth="1" status="completed"
                       ternarySignal="0" budgetAllocated="10"
                       assignedWorker="1" onCriticalPath="true">
      <IntermediateResult starId="pro123..." confidence="0.76"
                          persisted="true" layer="3"
                          galaxyDomain="Reality"/>
    </DecompositionTask>

    <!-- Arguments AGAINST -->
    <DecompositionTask taskId="t_con" strategy="multihop"
                       decompositionDepth="1" status="completed"
                       ternarySignal="-1" budgetAllocated="20"
                       assignedWorker="3" onCriticalPath="true">

      <!-- Sub-decomposition: AGAINST required deeper analysis -->
      <DecompositionTask taskId="t_con_physics" strategy="direct"
                         decompositionDepth="2" status="completed"
                         ternarySignal="1" budgetAllocated="5"
                         assignedWorker="5" galaxyDomain="Reality">
        <IntermediateResult starId="phys789..." confidence="0.94"
                            persisted="true" layer="2"/>
      </DecompositionTask>

      <DecompositionTask taskId="t_con_logic" strategy="dialectical"
                         decompositionDepth="2" status="completed"
                         ternarySignal="-1" budgetAllocated="20"
                         assignedWorker="7"
                         dependsOn='"t_con_physics"'
                         galaxyDomain="Grammar">
        <IntermediateResult starId="logic012..." confidence="0.68"
                            persisted="true" layer="3"/>
      </DecompositionTask>

      <IntermediateResult starId="con456..." confidence="0.72"
                          persisted="true" layer="3"
                          symlinkRefs='"phys789..." "logic012..."'/>
    </DecompositionTask>

    <!-- Synthesis -->
    <DecompositionTask taskId="t_synth" strategy="dialectical"
                       decompositionDepth="1" status="completed"
                       ternarySignal="0" budgetAllocated="10"
                       dependsOn='"t_pro" "t_con"'
                       assignedWorker="0" onCriticalPath="true">
      <IntermediateResult starId="synth345..." confidence="0.71"
                          persisted="true" layer="4"
                          symlinkRefs='"pro123..." "con456..."'/>
    </DecompositionTask>
  </DecompositionTask>

  <TaskQueue maxParallel="9" activeCount="4" saturated="false">
    <!-- No overflow in this example — all tasks fit in 9 workers -->
  </TaskQueue>

  <BudgetVisualization sphereRadius="2.0" sphereColor="0.3 0.6 1.0"
                       sphereOpacity="0.3" converged="true"
                       pulseFrequency="1.0"/>
</ReasoningBudget>
```

Budget: 20 iterations allocated (σ = −1), 18 used. Dialectical decomposition: FOR, AGAINST (with sub-decomposition into physics + logic), SYNTHESIS. All 6 intermediate results persisted as stars with symlink references.
