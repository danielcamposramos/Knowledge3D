# Kimi Agent Swarm Result
**Task:** Prepare a K3D canonical knowledge ingestion plan for Algebraic Modeling Languages (AML) and Solvers. Produce a markdown plan that lists every concept we must register as a meaning star, with star_id, 
**Mode:** thinking
**Time:** 2026-04-13 00:53:11

────────────────────────────────────────────────────────────────

## Sub-Agent A: Map every AML language and modeling-formulation concept in scope (AIMMS, AMPL, GAMS, Pyomo, JuMP, MathProg, Mosel, OPL, MiniZinc, Gekko, FlopC++, OptimJ, GBOML) to meaning stars. For each: star_id (canonical form), domain path under Math/Optimization/Modeling, meaning_rpn as a declarative STORE/RECALL program capturing its syntactic primitives (sets, indices, parameters, variables, constraints, objectives, sums/products over index sets), and a word star list across the 9 K3D languages. Identify which AMLs are "specialized commercial", which are "embedded in general-purpose language", and which are open-source. Produce the inheritance graph (taxonomy_refs) tying everything to a root `concept_algebraic_modeling_language` star.

The user wants a detailed K3D (Knowledge 3D) canonical knowledge ingestion plan for Algebraic Modeling Languages (AMLs) and related concepts. I need to map every AML in scope to "meaning stars" following a specific schema.

Key requirements:
1. **Scope**: 12 AMLs (AIMMS, AMPL, GAMS, Pyomo, JuMP, MathProg, Mosel, OPL, MiniZinc, Gekko, FlopC++, OptimJ, GBOML)
2. **Problem classes**: LP, MILP, NLP, QCP, MINLP, SAT, SMT, CSP, MP with equilibrium constraints, global optimization, stochastic optimization
3. **Solver architecture**: model-data separation, solver links, data exchange facilities, General Problem Solver
4. **Output format**: Markdown table with ~80-120 stars
5. **Schema per star**:
   - star_id (canonical form)
   - meaning_class (concept, relation, action, property, meta, form)
   - domain path (e.g., Math/Optimization/Modeling)
   - meaning_rpn (declarative STORE/RECALL program using only existing opcodes)
   - surface_forms across 9 languages (en/pt/es/fr/de/it/ja/zh/ru)
   - taxonomy_refs (inheritance graph)
   - math symbol refs
   - Classification: "specialized commercial", "embedded in general-purpose language", "open-source"

Constraints:
- Use ONLY existing opcodes: STORE, RECALL, GALAXY_LOOKUP, OP_BRANCH, TQUANT/TCOMP, arithmetic/comparison ops
- No new opcodes
- RPN must capture syntactic primitives: sets, indices, parameters, variables, constraints, objectives, sums/products over index sets
- Include dangling-ref risk list

Let me structure this:

**Root concept**: `concept_algebraic_modeling_language`

**AML Categories**:
1. Specialized commercial: AIMMS, AMPL, GAMS, Mosel, OPL
2. Embedded in general-purpose language: Pyomo (Python), JuMP (Julia), FlopC++ (C++), OptimJ (Java), Gekko (Python), MathProg (GNU MathProg is stand-alone but often embedded? Actually MathProg is GNU's modeling language - I'll classify as specialized open-source), MiniZinc (specialized but open-source)
3. Open-source: Pyomo, JuMP, MiniZinc, GBOML, MathProg (GNU), Gekko

Wait, need to be careful:
- **AIMMS**: Commercial, specialized
- **AMPL**: Commercial (A Mathematical Programming Language), specialized
- **GAMS**: Commercial (General Algebraic Modeling System), specialized
- **Pyomo**: Open-source, embedded in Python
- **JuMP**: Open-source, embedded in Julia
- **MathProg**: GNU MathProg/GMPL - open-source, specialized (similar to AMPL)
- **Mosel**: Commercial (FICO), specialized
- **OPL**: Commercial (IBM), specialized
- **MiniZinc**: Open-source, specialized (constraint modeling)
- **Gekko**: Open-source, embedded in Python (dynamic optimization)
- **FlopC++**: Open-source, embedded in C++ (COIN-OR)
- **OptimJ**: Commercial? I think OptimJ is from OptimJ.com, likely commercial but embedded in Java. Actually OptimJ is a commercial product but embedded.
- **GBOML**: Open-source (Green Bounded Optimization Modeling Language) - specialized

So classification:
- **Specialized Commercial**: AIMMS, AMPL, GAMS, Mosel, OPL, OptimJ (commercial)
- **Embedded Open-source**: Pyomo, JuMP, FlopC++, Gekko
- **Specialized Open-source**: MiniZinc, MathProg (GNU), GBOML

Now for the RPN patterns. The RPN needs to capture:
- SETS: index sets, domains
- INDICES: i, j, t
- PARAMETERS: data constants
- VARIABLES: decision variables (continuous, integer, binary)
- CONSTRAINTS: equations/inequalities
- OBJECTIVES: minimize/maximize
- AGGREGATION: sum, product over index sets
- MODEL: container

Common RPN structure using available opcodes:
- STORE/RECALL for variable binding
- GALAXY_LOOKUP for referencing existing math stars
- TQUANT for quantifiers (∀, ∃) - though TQUANT is ternary comparison, maybe not for quantifiers. Actually TQUANT is for comparisons, TCOMP might be for composite? The prompt says TQUANT/TCOMP (ternary comparison).
- OP_BRANCH for conditional/model structure

Wait, the prompt says: "RPN composition pattern using ONLY existing opcodes (STORE/RECALL/GALAXY_LOOKUP/OP_BRANCH/TQUANT/TCOMP). No new opcodes."

So I need to compose these to represent AML concepts.

Let me think about how to represent:
- **Sets**: STORE set_definition RECALL
- **Sum over set**: GALAXY_LOOKUP(sum_symbol) GALAXY_LOOKUP(index_set) OP_BRANCH STORE?

Actually, looking at the prompt: "meaning_rpn as a declarative STORE/RECALL program capturing its syntactic primitives"

So it's more like a bytecode representation of the language's primitive operations.

For example, for "Set declaration":
```
STORE(set_id) GALAXY_LOOKUP(math_set) RECALL(index) OP_BRANCH(domain_expr)
```

But I need to be consistent. Let me define patterns:

**Pattern for Index Sets**:
```
STORE(set_name) GALAXY_LOOKUP(concept_index_set) GALAXY_LOOKUP(syntax_declaration) OP_BRANCH
```

**Pattern for Summation**:
```
GALAXY_LOOKUP(math_summation) STORE(index_var) GALAXY_LOOKUP(set_domain) OP_BRANCH STORE(expression) RECALL GALAXY_LOOKUP(math_addition) OP_BRANCH
```

Actually, RPN (Reverse Polish Notation) means operands come before operators.

So:
- `STORE` takes a value and a name: `value name STORE` → stores value under name
- `RECALL` takes a name: `name RECALL` → pushes value
- `GALAXY_LOOKUP` takes a star_id: `star_id GALAXY_LOOKUP` → pushes the star reference

For a sum like ∑_{i∈I} x_i:
```
I GALAXY_LOOKUP(concept_index_set) STORE(i) 
x i RECALL GALAXY_LOOKUP(syntax_subscript) OP_BRANCH 
GALAXY_LOOKUP(math_summation) OP_BRANCH
```

But this is getting complex. The prompt asks for "meaning_rpn sketch" - so I can provide a simplified sketch showing the composition pattern.

Let me organize the stars:

**Category 1: Root and Meta Concepts**
1. concept_algebraic_modeling_language (root)
2. concept_aml_specialized_commercial
3. concept_aml_embedded_general_purpose
4. concept_aml_open_source
5. concept_model_data_separation
6. concept_solver_link
7. concept_data_exchange_facility

**Category 2: Individual AMLs (12 stars)**
8. concept_aml_aimms
9. concept_aml_ampl
10. concept_aml_gams
11. concept_aml_pyomo
12. concept_aml_jump
13. concept_aml_mathprog
14. concept_aml_mosel
15. concept_aml_opl
16. concept_aml_minizinc
17. concept_aml_gekko
18. concept_aml_flopcpp
19. concept_aml_optimj
20. concept_aml_gboml

**Category 3: Syntactic Primitives (shared across AMLs)**
21. concept_aml_set_declaration
22. concept_aml_parameter_declaration
23. concept_aml_variable_declaration
24. concept_aml_constraint_declaration
25. concept_aml_objective_declaration
26. concept_aml_index_reference
27. concept_aml_summation_expression
28. concept_aml_product_expression
29. concept_aml_indexing_expression
30. concept_aml_model_block

**Category 4: Problem Classes**
31. concept_problem_class_lp
32. concept_problem_class_milp
33. concept_problem_class_nlp
34. concept_problem_class_qcp
35. concept_problem_class_minlp
36. concept_problem_class_sat
37. concept_problem_class_smt
38. concept_problem_class_csp
39. concept_problem_class_mpec (mathematical program with equilibrium constraints)
40. concept_problem_class_global_optimization
41. concept_problem_class_stochastic_optimization

**Category 5: Solver Architecture**
42. concept_solver_link
43. concept_data_exchange_facility
44. concept_general_problem_solver
45. concept_model_instance
46. concept_symbolic_model
47. concept_numeric_data

**Category 6: Mathematical Symbols/Operators (as references)**
These might be references to existing math stars, not new stars, but I should list the refs:
- math_summation (Σ)
- math_product (Π)
- math_less_than_equal (≤)
- math_greater_than_equal (≥)
- math_equal (=)
- math_integral (∫)
- math_partial_derivative (∂)
- math_objective_function
- math_constraint
- math_variable
- math_parameter

**Category 7: Language-Specific Surface Forms**
Each AML needs surface forms in 9 languages.

Wait, the user wants "surface_forms map {lang: {word_ref: "word_{lang}_{lemma}"}}" - so I need to provide the word references, not the actual words, or indicate if untranslatable.

For many AMLs, the names are proper nouns (acronyms) and remain the same across languages:
- AIMMS, AMPL, GAMS, Pyomo, JuMP, MathProg, Mosel, OPL, MiniZinc, Gekko, FlopC++, OptimJ, GBOML

These should be marked as proper nouns with `untanslatable_languages` or just use the same string.

**Category 8: Relations between concepts**
- relation_implements (AML implements problem class)
- relation_compiles_to
- relation_embedded_in (for embedded languages)

Let me calculate total:
- Root: 1
- Meta categories: 3 (commercial, embedded, open-source)
- Individual AMLs: 12
- Syntactic primitives: ~10
- Problem classes: 11
- Solver arch: ~6
- Relations: ~5
Total: ~38. I need 80-120. So I need to expand.

I should expand:
- Each AML should have its specific dialect features (e.g., Pyomo ConcreteModel vs AbstractModel, JuMP @variable vs @constraint)
- Specific constraint types (linear, quadratic, nonlinear)
- Variable types (continuous, integer, binary)
- Specific problem class variants
- Solver interface concepts
- Data formats (GAMS GDX, AMPL .mod/.dat, etc.)

Let me expand the list:

**Additional AML-specific features:**

48. concept_aimms_identifier_declaration (specific to AIMMS)
49. concept_aml_concrete_model (Pyomo)
50. concept_aml_abstract_model (Pyomo)
51. concept_aml_macro (JuMP @macro)
52. concept_aml_domain_specific_language (general pattern)
53. concept_aml_syntax_sum (AMPL-style sum{})
54. concept_aml_syntax_forall (MiniZinc, OPL constraint forall)
55. concept_aml_syntax_subject_to
56. concept_aml_syntax_minimize
57. concept_aml_syntax_maximize

**Variable types:**
58. concept_aml_variable_continuous
59. concept_aml_variable_integer
60. concept_aml_variable_binary
61. concept_aml_variable_semicontinuous
62. concept_aml_variable_sos (Special Ordered Sets)

**Constraint types:**
63. concept_aml_constraint_equality
64. concept_aml_constraint_inequality_le
65. concept_aml_constraint_inequality_ge
66. concept_aml_constraint_nonlinear
67. concept_aml_constraint_logical (MiniZinc)

**Problem class specifics:**
68. concept_optimization_linear
69. concept_optimization_mixed_integer
70. concept_optimization_quadratic
71. concept_optimization_quadratically_constrained
72. concept_optimization_nonlinear
73. concept_optimization_stochastic
74. concept_optimization_robust
75. concept_optimization_multi_objective

**Solver interactions:**
76. concept_solver_gurobi
77. concept_solver_cplex
78. concept_solver_glpk
79. concept_solver_ipopt
80. concept_solver_coin_or

**Data separation:**
81. concept_aml_model_file
82. concept_aml_data_file
83. concept_aml_instance_file

**Math symbols as explicit stars (if not existing):**
These should reference existing math stars, but I need to ensure they are linked.

Let me organize into a comprehensive table.

**Structure of the markdown output:**

1. **Header**: Explanation of the ingestion plan
2. **Taxonomy Overview**: Brief description of the inheritance graph
3. **Star Table**: The main table with ~80-120 rows containing:
   - star_id
   - meaning_class
   - domain_path
   - meaning_rpn (sketch)
   - surface_forms (summary or reference)
   - taxonomy_refs
   - math_symbol_refs
   - notes (classification, etc.)
4. **Dangling Reference Risk List**: Potential missing dependencies

**RPN Pattern Examples:**

For a set declaration in AML:
```
"index_set" STORE GALAXY_LOOKUP(concept_set) GALAXY_LOOKUP(math_index_set) OP_BRANCH
```

For a summation:
```
GALAXY_LOOKUP(math_summation) STORE(index) GALAXY_LOOKUP(domain) OP_BRANCH RECALL(expression) OP_BRANCH
```

For variable declaration:
```
STORE(var_name) GALAXY_LOOKUP(concept_decision_variable) GALAXY_LOOKUP(type_continuous) OP_BRANCH
```

For constraint:
```
STORE(constraint_name) RECALL(lhs_expr) RECALL(rhs_expr) GALAXY_LOOKUP(math_less_than_equal) OP_BRANCH GALAXY_LOOKUP(concept_constraint) OP_BRANCH
```

**Surface Forms Strategy:**

For proper nouns (AMPL, GAMS, etc.): Use the same string across all languages, mark as `proper_noun: true` or indicate in untranslatable_languages.

For concepts like "set declaration

────────────────────────────────────────────────────────────────

## Sub-Agent B: Taxonomize solver software and problem classes. For each solver type (linear, nonlinear, integer, mixed-integer, SAT, SMT, constraint satisfaction, shortest path, min spanning tree, ODE/DAE, Boolean quantified, polynomial systems): propose a `concept_solver_{class}` meaning star with domain Math/Optimization/Solver, meta_refs pointing at the model class it consumes, and executable-sketch meaning_rpn that dispatches STORE of problem form → GALAXY_LOOKUP for applicable algorithm → OP_BRANCH on feasibility. Include historical concepts (General Problem Solver, matrix-generator-report-writer era, UIMP, MGRW). Identify bidirectional links from `concept_solver_*` to AML stars (which AML can produce input for which solver).

The user wants me to act as Sub-Agent B in a Kimi Agent Swarm. My specific sub-task is to "Taxonomize solver software and problem classes" according to the detailed instructions provided.

Key requirements:
1. For each solver type (linear, nonlinear, integer, mixed-integer, SAT, SMT, constraint satisfaction, shortest path, min spanning tree, ODE/DAE, Boolean quantified, polynomial systems), propose a `concept_solver_{class}` meaning star.
2. These stars should have:
   - Domain: Math/Optimization/Solver
   - meta_refs pointing at the model class it consumes
   - executable-sketch meaning_rpn that dispatches: STORE of problem form → GALAXY_LOOKUP for applicable algorithm → OP_BRANCH on feasibility
3. Include historical concepts: General Problem Solver, matrix-generator-report-writer era, UIMP, MGRW
4. Identify bidirectional links from `concept_solver_*` to AML stars (which AML can produce input for which solver)

The broader context mentions preparing a K3D canonical knowledge ingestion plan for AML and Solvers, but my specific focus is the solver taxonomy part.

I need to produce a markdown plan with:
- star_id
- meaning_class (should be "concept")
- domain path
- meaning_rpn sketch
- surface_forms across 9 languages (en/pt/es/fr/de/it/ja/zh/ru)
- bidirectional symlinks to existing math/word stars
- parent/child taxonomy_refs
- math symbol refs
- RPN composition pattern using only: STORE/RECALL/GALAXY_LOOKUP/OP_BRANCH/TQUANT/TCOMP

I should create a table of stars (~80-120 total, but my portion is specifically the solver taxonomy), plus a dangling-ref risk list.

Let me structure this:

**Solver Types to cover:**
1. Linear Programming (LP) Solver
2. Mixed-Integer Linear Programming (MILP) Solver
3. Nonlinear Programming (NLP) Solver
4. Mixed-Integer Nonlinear Programming (MINLP) Solver
5. Quadratically Constrained Programming (QCP) Solver
6. Satisfiability (SAT) Solver
7. Satisfiability Modulo Theories (SMT) Solver
8. Constraint Satisfaction Problem (CSP) Solver
9. Shortest Path Solver
10. Minimum Spanning Tree (MST) Solver
11. Ordinary Differential Equation (ODE) Solver
12. Differential-Algebraic Equation (DAE) Solver
13. Boolean Quantified Formula (QBF) Solver
14. Polynomial Systems Solver
15. Global Optimization Solver
16. Stochastic Optimization Solver
17. Mathematical Programming with Equilibrium Constraints (MPEC) Solver
18. General Problem Solver (Historical - GPS)
19. Matrix-Generator-Report-Writer (MGRW) - Historical
20. UIMP (Universal Integer Mathematical Programming) - Historical

**AML Stars to link to** (from Sub-Agent A presumably, but I need to reference them):
- AIMMS, AMPL, GAMS, Gekko, MathProg, Mosel, OPL, Pyomo, JuMP, FlopC++, OptimJ, GBOML, MiniZinc

**Problem Class Stars to reference** (these should be the meta_refs):
- concept_problem_lp, concept_problem_milp, concept_problem_nlp, concept_problem_minlp, concept_problem_qcp, concept_problem_sat, concept_problem_smt, concept_problem_csp, concept_problem_shortest_path, concept_problem_mst, concept_problem_ode, concept_problem_dae, concept_problem_qbf, concept_problem_polynomial_system, concept_problem_global_opt, concept_problem_stochastic_opt, concept_problem_mpec

**RPN Pattern for each solver concept:**
The instruction says: "STORE of problem form → GALAXY_LOOKUP for applicable algorithm → OP_BRANCH on feasibility"

So something like:
`STORE(problem_instance) GALAXY_LOOKUP(algorithm_space) OP_BRANCH(feasible, infeasible)`

Or more specifically:
`STORE problem_form_id GALAXY_LOOKUP solver_algorithm_space OP_BRANCH feasible_path infeasible_path`

Wait, RPN typically works with stack. So:
`problem_form STORE algorithm_space GALAXY_LOOKUP feasible_result infeasible_result OP_BRANCH`

But the instruction says: "STORE of problem form → GALAXY_LOOKUP for applicable algorithm → OP_BRANCH on feasibility"

So the meaning_rpn sketch should look like:
`[problem_form_ref] STORE [algorithm_lookup_key] GALAXY_LOOKUP [feasible_label] [infeasible_label] OP_BRANCH`

Or perhaps:
`STORE(problem_form) GALAXY_LOOKUP(solver_algorithms) OP_BRANCH(feasibility_check)`

But since it's RPN (Reverse Polish Notation), it would be:
`problem_form STORE solver_space GALAXY_LOOKUP feasible_branch infeasible_branch OP_BRANCH`

Let me check the available opcodes: STORE/RECALL/GALAXY_LOOKUP/OP_BRANCH/TQUANT/TCOMP

So valid RPN sequences:
- `STORE` (pops value, stores at address)
- `RECALL` (pushes value from address)
- `GALAXY_LOOKUP` (looks up in some space)
- `OP_BRANCH` (conditional branch)
- `TQUANT` (ternary quantifier?)
- `TCOMP` (ternary comparison?)

Actually, looking at the context, it seems these are custom opcodes for the K3D system, not standard RPN. The sketch should be descriptive but use these tokens.

The pattern requested: "STORE of problem form → GALAXY_LOOKUP for applicable algorithm → OP_BRANCH on feasibility"

So I'll write: `[problem_form] STORE [algorithm_domain] GALAXY_LOOKUP [feasible_handler] [infeasible_handler] OP_BRANCH`

**Surface forms:**
Need to provide for 9 languages: en/pt/es/fr/de/it/ja/zh/ru

For solver types:
- LP Solver: Linear Programming Solver / Resolvedor de Programação Linear / Solucionador de Programación Lineal / Solveur de Programmation Linéaire / Lineare Optimierung Solver / Risolutore di Programmazione Lineare / 線形計画法ソルバー / 线性规划求解器 / Решатель линейного программирования
- MILP Solver: Mixed-Integer Linear Programming Solver / ... 
- etc.

**Historical concepts:**
- General Problem Solver (GPS) - Newell & Simon 1959
- Matrix-Generator-Report-Writer (MGRW) - 1960s-70s era commercial software architecture
- UIMP - Universal Integer Mathematical Programming system

**Bidirectional links:**
Need to specify which AML can target which solver. For example:
- AMPL → CPLEX, Gurobi, IPOPT, MINOS, etc.
- GAMS → CONOPT, MINOS, CPLEX, etc.
- MiniZinc → Gecode, Chuffed, OR-Tools (CSP/SAT solvers)
- Pyomo/JuMP → Various
- OPL → CPLEX CP Optimizer
- etc.

But since I'm creating the solver concepts, I should note which AML stars link to them.

**Table Structure:**
I'll create a markdown table with columns:
- star_id
- meaning_class
- domain_path
- meaning_rpn_sketch
- surface_forms (abbreviated or structured)
- meta_refs (problem class it consumes)
- taxonomy_refs (parent/child)
- aml_links (bidirectional symlinks to AML concepts)

Wait, the user wants "bidirectional links from `concept_solver_*` to AML stars". So I should include a column or section showing these links.

Also need math symbol refs where applicable.

Let me draft the content:

**Solver Concepts:**

1. **concept_solver_lp** (Linear Programming)
   - Domain: Math/Optimization/Solver
   - Meaning RPN: `problem_form STORE Math/Algorithm/LP GALAXY_LOOKUP solution_feasible solution_infeasible OP_BRANCH`
   - Meta refs: concept_problem_lp
   - AML links: concept_aml_ampl, concept_aml_gams, concept_aml_cvxx, concept_aml_jump, concept_aml_pyomo, etc.
   - Math symbols: ≤, ≥, =, ∑

2. **concept_solver_milp** (Mixed-Integer Linear)
   - Meta refs: concept_problem_milp
   - AML links: All major AMLs

3. **concept_solver_nlp** (Nonlinear)
   - Meta refs: concept_problem_nlp
   - AML links: AMPL, GAMS, Pyomo, JuMP, etc.

4. **concept_solver_minlp** (Mixed-Integer Nonlinear)
   - Meta refs: concept_problem_minlp

5. **concept_solver_qcp** (Quadratically Constrained)
   - Meta refs: concept_problem_qcp

6. **concept_solver_sat** (Boolean Satisfiability)
   - Meta refs: concept_problem_sat
   - AML links: concept_aml_minizinc (to some extent), concept_aml_opl

7. **concept_solver_smt** (Satisfiability Modulo Theories)
   - Meta refs: concept_problem_smt

8. **concept_solver_csp** (Constraint Satisfaction)
   - Meta refs: concept_problem_csp
   - AML links: concept_aml_minizinc, concept_aml_opl

9. **concept_solver_shortest_path** (Shortest Path Algorithms)
   - Meta refs: concept_problem_shortest_path (graph theory)
   - AML links: concept_aml_ampl (network flows), concept_aml_gams

10. **concept_solver_mst** (Minimum Spanning Tree)
    - Meta refs: concept_problem_mst

11. **concept_solver_ode** (Ordinary Differential Equations)
    - Meta refs: concept_problem_ode
    - AML links: concept_aml_gekko (specialized)

12. **concept_solver_dae** (Differential-Algebraic Equations)
    - Meta refs: concept_problem_dae

13. **concept_solver_qbf** (Quantified Boolean Formulas)
    - Meta refs: concept_problem_qbf

14. **concept_solver_polynomial** (Polynomial Systems)
    - Meta refs: concept_problem_polynomial_system

15. **concept_solver_global** (Global Optimization)
    - Meta refs: concept_problem_global_opt

16. **concept_solver_stochastic** (Stochastic Optimization)
    - Meta refs: concept_problem_stochastic_opt

17. **concept_solver_mpec** (Mathematical Programming with Equilibrium Constraints)
    - Meta refs: concept_problem_mpec

18. **concept_solver_historical_gps** (General Problem Solver)
    - Historical, cognitive architecture
    - Domain: Math/Optimization/Solver/Historical

19. **concept_solver_historical_mgrw** (Matrix-Generator-Report-Writer)
    - Historical architecture
    - Domain: Math/Optimization/Solver/Historical/Architecture

20. **concept_solver_historical_uimp** (Universal Integer Mathematical Programming)
    - Historical system

That's 20 solver concepts. The user mentioned ~80-120 stars total for the full plan, but my sub-task is specifically the solver taxonomy. I should provide a comprehensive list.

Wait, I should also include specific algorithm implementations or categories? The prompt says "solver type" and lists categories. I think the categories are the main ones, but I could include sub-types like:
- concept_solver_lp_simplex
- concept_solver_lp_interior_point
- concept_solver_milp_branch_and_bound
- concept_solver_nlp_interior_point
- concept_solver_nlp_sqp

But the prompt asks for "Taxonomize solver software and problem classes" and lists the high-level types. I'll stick to the high-level types but include the historical ones and ensure I have the full list from the prompt.

Actually, looking at the list again: "linear, nonlinear, integer, mixed-integer, SAT, SMT, constraint satisfaction, shortest path, min spanning tree, ODE/DAE, Boolean quantified, polynomial systems"

That's:
1. Linear
2. Nonlinear
3. Integer (this might be pure integer programming, subset of MILP)
4. Mixed-integer (MILP/MINLP)
5. SAT
6. SMT
7. Constraint satisfaction
8. Shortest path
9. Min spanning tree
10. ODE
11. DAE
12. Boolean quantified (QBF)
13. Polynomial systems

Plus the historical ones: GPS, MGRW, UIMP

And I should probably include QCP, Global, Stochastic, MPEC as mentioned in the broader context.

So I'll structure the table with these columns:
- star_id
- meaning_class
- domain_path
- meaning_rpn_sketch
- meta_refs (problem class consumed)
- taxonomy_refs (parent: concept_solver_optimization or concept_solver_math)
- surface_forms (JSON-like or structured text for the 9 languages)
- aml_symlinks (which AML stars link here)

For surface forms, I'll provide the English lemma and indicate translations exist for the others, or provide a compact representation like: `{en: "LP solver", pt: "resolvedor PL", ...}`

For the RPN, I need to be consistent. The pattern is:
`STORE problem_form GALAXY_LOOKUP algorithm_space OP_BRANCH feasible infeasible`

Or more precisely using the available opcodes:
`problem_instance STORE solver_algorithm_class GALAXY_LOOKUP feasible_solution infeasible_solution OP_BRANCH`

Wait, OP_BRANCH typically takes a condition and two addresses. So:
`condition feasible_branch infeasible_branch OP_BRANCH`

Or in stack-based:
`condition feasible_branch infeasible_branch OP_BRANCH` (pops 3, pushes 0, jumps)

But the problem form needs to be stored first, then looked up... Actually, the sketch is declarative of the process.

Let me write it as: `[problem_form_ref] STORE [algorithm_space] GALAXY_LOOKUP [feasible] [infeasible] OP_BRANCH`

This shows the sequence: store the form, lookup applicable algorithm, branch on feasibility.

**Dangling-ref risk list:**
This should list any references to stars that might not exist yet or are assumed to be created by Sub-Agent A (the AML stars, problem class stars, math symbol stars).

For example:
- concept_problem_lp (assumed exists)
- concept_aml_ampl (assumed created by Sub-Agent A)
- Math/Algorithm/LP (domain path assumed

────────────────────────────────────────────────────────────────

## Synthesis

# K3D Canonical Knowledge Ingestion Plan: Algebraic Modeling Languages & Solvers

## Executive Summary

This plan registers **100 canonical stars** covering the complete AML-Solver ecosystem:
