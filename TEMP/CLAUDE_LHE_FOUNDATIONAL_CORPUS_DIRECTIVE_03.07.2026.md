# Claude Architecture Directive: LHE Foundational Knowledge Corpus

**Date:** March 7, 2026
**From:** Claude (Architecture) + Daniel (Direction)
**To:** Codex (Implementation)
**Context:** ARC 10/10, Math 20/20, LHE 1/10. Codex confirmed bottleneck is knowledge density, not routing or plumbing. This directive builds a targeted foundational corpus to validate that claim before expanding to broader benchmarks.

---

## Daniel's Direction

"Let's make a foundational corpus spanning all known great subjects that the LHE questions touch, just to make sure this is the way -- then we'll try the other benchmarks that are relevant to the current LLM scene."

---

## Current State Assessment

### What We Have

The augmentation snapshot (`full_pdf_payloads_paused_20260307_100329.jsonl`) contains **85,420 entries**:
- Grammar Galaxy: 55,127 (mostly `pdf_reasoning_bridge`)
- Math Galaxy: 18,167 (formulas, theorems)
- Word Galaxy: 7,294 (lexemes)
- Reality Galaxy: **3,476** (severely underweight)
- Drawing: 582 / Character: 455 / 3DObjects: 308 / Audio: 11

**Source:** 100% from `pdf_intelligent_augmentation` -- bulk PDF extraction. No domain-curated knowledge.

### What LHE Needs

The LHE dataset has **2,500 questions** across 8 domains, ALL open-ended:

| Domain | Count | % | Galaxy Coverage |
|--------|-------|---|-----------------|
| Math | 1,021 | 40.8% | Math Galaxy has 18K entries -- decent, but LHE math is graduate/research-level |
| Biology/Medicine | 280 | 11.2% | Reality Galaxy: sparse, mostly physics PDF extracts |
| CS/AI | 241 | 9.6% | Reality Galaxy: almost nothing on CS concepts |
| Other (chess, puzzles, crypto) | 233 | 9.3% | Nothing -- no chess theory, no cipher mechanics |
| Physics | 230 | 9.2% | Reality Galaxy: some coverage from PDF extraction |
| Humanities/Social Science | 219 | 8.8% | Nothing -- no philosophy, no ethics, no law |
| Chemistry | 165 | 6.6% | Reality Galaxy: minimal |
| Engineering | 111 | 4.4% | Reality Galaxy: minimal |

### The Gap

The 1/10 LHE result confirms: the four-pass structure works (got the one it could reach), but 9/10 questions hit domains where Galaxy knowledge is near-zero. The smoke pack's 10 questions include chess notation, population ethics philosophy, substitution ciphers, advanced algebraic topology, Lie algebras, gamma matrix identities, gravitational compactification, and activation function theory -- none of which exist in the current 85K snapshot.

---

## Directive: Build Targeted LHE Foundational Corpus

### Architecture

This is an **augmentation-time** operation, NOT a hot-path change. The output is a JSONL payload file consumed by `scripts/fundamental_ingest_payloads.py`, which populates Galaxy entries at init time.

### Corpus Design Principle

**Meaning-centric concept stars with symlinks.** Each domain concept becomes:
1. A **Reality Galaxy entry** (concept star -- the meaning)
2. A **Word Galaxy entry** (linguistic form -- symlinked to Reality)
3. Optionally a **Grammar Galaxy entry** (reasoning rule -- how to apply the concept)
4. Optionally a **Math Galaxy entry** (if the concept has mathematical formalization)

Do NOT duplicate content across galaxies. Use the symlink pattern from `foundational_operations_bootstrap.py` (Number-Word symlinks).

### Domain Coverage Requirements

Build foundational concept entries for EACH of these domains. The goal is NOT to answer the specific smoke questions -- it's to provide foundational domain knowledge that the four-pass can navigate. Think of it as the "textbook index" for each field.

#### 1. Mathematics (Graduate/Research Level)
The smoke pack has algebraic topology, Lie algebras, moduli spaces, elliptic curves, Poincare polynomials, bordism theory. Current Math Galaxy has basic algebra/geometry -- needs graduate-level concept anchors.

**Required concept families:**
- Algebraic topology: homology groups, cohomology, homotopy, bordism, classifying spaces, characteristic classes
- Abstract algebra: Lie algebras, Lie groups (G2, E8, SU(n)), representations, root systems
- Algebraic geometry: moduli spaces, elliptic curves, torsion subgroups, algebraic varieties, sheaves
- Number theory: p-adic numbers, Galois theory, class field theory, L-functions
- Analysis: functional analysis, Sobolev spaces, PDE theory, distribution theory
- Combinatorics: generating functions, partition theory, Ramsey theory

**Entry format:**
```json
{
  "galaxy": "Reality",
  "entry": {
    "id": "concept_algebraic_topology_homology",
    "name": "homology group",
    "domain": "reality",
    "category": "mathematics_topology",
    "rpn_program": "CHAIN_COMPLEX BOUNDARY_MAP KERNEL IMAGE QUOTIENT",
    "metadata": {
      "source": "lhe_foundational_corpus",
      "subject": "mathematics",
      "subfield": "algebraic_topology",
      "definition": "Algebraic invariant measuring n-dimensional holes in topological spaces. H_n(X) = ker(d_n) / im(d_{n+1}).",
      "related_concepts": ["chain_complex", "boundary_operator", "betti_numbers", "cohomology"],
      "symlinks": ["word_homology", "math_homology_computation"],
      "confidence": 0.92
    }
  }
}
```

#### 2. Physics (Theoretical/Mathematical)
The smoke pack has gamma matrix identities in arbitrary dimensions, gravitational compactification, Kaluza-Klein theory.

**Required concept families:**
- Quantum field theory: gamma matrices, spinor representations, dimensional reduction, Clifford algebras
- General relativity: compactification, Kaluza-Klein, extra dimensions, moduli
- Statistical mechanics: partition functions, phase transitions, critical phenomena
- Quantum mechanics: path integrals, symmetry groups, angular momentum coupling
- Electromagnetism: gauge theory, Maxwell equations in differential form notation

#### 3. Computer Science / AI
The smoke pack has activation function theory, transformer architecture details, attention mechanisms.

**Required concept families:**
- Neural architectures: transformers, attention mechanisms, residual streams, layer normalization
- Activation functions: ReLU, GELU, Swish, Mish -- properties, derivatives, saturation behavior
- Optimization: gradient descent variants, learning rate schedules, loss landscapes
- Complexity theory: P vs NP, computational classes, reductions
- Information theory: entropy, mutual information, KL divergence, channel capacity
- Cryptography: substitution ciphers, frequency analysis, Vigenere, RSA fundamentals

#### 4. Biology / Medicine
**Required concept families:**
- Molecular biology: DNA/RNA, transcription, translation, gene regulation, CRISPR
- Biochemistry: enzyme kinetics, metabolic pathways, protein folding
- Neuroscience: neural signaling, neurotransmitters, brain regions, synaptic plasticity
- Immunology: immune response, antibodies, T-cells, B-cells, MHC
- Pharmacology: drug mechanisms, receptor binding, dose-response curves
- Evolution: natural selection, genetic drift, speciation, phylogenetics

#### 5. Chemistry
**Required concept families:**
- Organic chemistry: reaction mechanisms, functional groups, stereochemistry, synthesis
- Inorganic chemistry: coordination compounds, crystal field theory, transition metals
- Physical chemistry: thermodynamics, kinetics, quantum chemistry, spectroscopy
- Analytical chemistry: chromatography, spectroscopy methods, titration theory

#### 6. Humanities / Social Science
The smoke pack has population ethics (non-sadism, egalitarianism), legal concepts, philosophical theorems.

**Required concept families:**
- Ethics/philosophy: utilitarianism, deontology, virtue ethics, population ethics, impossibility theorems
- Population ethics specifically: total utilitarianism, average utilitarianism, critical-level views, non-sadism, egalitarianism, Parfit's repugnant conclusion, Arrhenius's impossibility theorems
- Logic: modal logic, predicate logic, proof theory, set theory foundations
- Law: legal principles, statutory interpretation, good faith doctrine, international law concepts
- History: major historical periods, key events, historiographical methods

#### 7. Engineering
**Required concept families:**
- Electrical engineering: circuit theory, signal processing, control systems, semiconductor physics
- Mechanical engineering: thermodynamic cycles, fluid mechanics, materials science
- Civil engineering: structural analysis, geotechnical fundamentals
- Systems engineering: feedback loops, control theory, system dynamics

#### 8. Other (Chess, Puzzles, Cryptanalysis)
**Required concept families:**
- Chess: algebraic notation, piece movement rules, common mating patterns (back-rank, smothered, discovered), endgame theory, basic openings, check/checkmate/stalemate
- Cryptography (classical): Caesar cipher, substitution ciphers, frequency analysis, polyalphabetic ciphers, transposition ciphers
- Logic puzzles: formal deduction patterns, constraint satisfaction

---

### Implementation Approach

#### Script: `scripts/build_lhe_foundational_corpus.py`

This is an augmentation-time script (like `fundamental_augment_benchmarks.py`). It:

1. **Generates concept entries programmatically** -- NOT by querying Ollama per-concept. The concept definitions are embedded in the script as structured data. This is deterministic, reproducible, fast.

2. **Optionally enriches via Ollama** -- for each concept, compose a prompt asking for: related concepts, key formulas, common misconceptions. This adds depth but is not required for the baseline test.

3. **Outputs JSONL payload** compatible with `fundamental_ingest_payloads.py`

4. **Maintains symlink structure:**
   - Each concept -> Reality Galaxy entry (primary)
   - Each concept name -> Word Galaxy entry (symlinks to Reality)
   - Mathematical concepts -> Math Galaxy entry (formalization)
   - Reasoning patterns -> Grammar Galaxy entry (how to apply)

#### Target Scale

- **Minimum:** 500 concept entries across 8 domains (enough to test knowledge density hypothesis)
- **Recommended:** 2,000-3,000 concept entries (meaningful coverage per domain)
- **Maximum (with Ollama enrichment):** 5,000+ entries

#### Entry Quality Requirements

Each Reality Galaxy concept entry MUST have:
- `definition`: 1-2 sentence precise definition
- `related_concepts`: list of connected concept IDs (for symlink navigation)
- `subfield`: domain subdivision for routing
- `rpn_program`: procedural representation (even if symbolic/placeholder)

Each entry MUST NOT:
- Duplicate information already in the 85K snapshot
- Contain LHE question text or answers (that's data contamination)
- Be specific to one question (must be general domain knowledge)

---

### Validation Protocol

1. Build corpus
2. Ingest into world via `fundamental_ingest_payloads.py`
3. Rerun the same audited 10/20/10 smoke pack
4. Compare LHE score delta

**Expected outcome:**
- If knowledge density IS the bottleneck: LHE should improve from 1/10 (even +1 proves the thesis)
- If knowledge density is NOT the bottleneck: LHE stays at 1/10 despite richer galaxies (means synthesis/scoring needs more work)

Either outcome is informative. Daniel's instinct is that it IS the bottleneck. Let's prove it.

---

### After Validation: Next Benchmarks

Once we confirm the knowledge-density thesis, expand to benchmarks relevant in the current LLM landscape:

| Benchmark | What It Tests | K3D Relevance |
|-----------|---------------|---------------|
| **MMLU** | Multi-domain knowledge (57 subjects) | Already have sender + augmentation pipeline. Tests breadth. |
| **GPQA** | Graduate-level science questions | Tests depth in physics/chemistry/biology. Similar to LHE but more constrained. |
| **HumanEval / MBPP** | Code generation | Tests Grammar Galaxy composition for code patterns. |
| **MATH (Hendrycks)** | Competition math, harder than GSM8K | Already loading via math_competitions. Tests Math Galaxy depth. |
| **BBH (BIG-Bench Hard)** | Challenging reasoning tasks | Tests four-pass composition across diverse reasoning types. |
| **IFEval** | Instruction following | Tests Grammar Galaxy instruction comprehension. |
| **ARC-Challenge** | Common-sense reasoning (not ARC-AGI) | Tests Reality Galaxy common-sense knowledge. |

K3D already has infrastructure for MMLU (`benchmarks/mmlu.py`, `benchmarks/mmlu_sender.py`). That's the natural next target after LHE validation.

---

### Priority Order

```
1. Build scripts/build_lhe_foundational_corpus.py (deterministic concept generation)
2. Generate JSONL payload (~2K concept entries across 8 LHE domains)
3. Ingest into world
4. Rerun 10/20/10 smoke pack
5. Report delta
6. If thesis confirmed: expand corpus + try MMLU smoke
```

---

### What NOT to Do

1. **Do NOT embed LHE answers in the corpus.** This is foundational domain knowledge, not answer keys.
2. **Do NOT build question-specific entries.** "Arrhenius impossibility theorem" is valid domain knowledge. "The answer to LHE question #1 is D" is contamination.
3. **Do NOT modify the daemon hot path.** This is augmentation only -- populate galaxies, let existing four-pass + routing navigate them.
4. **Do NOT skip the symlink structure.** Every concept needs Reality + Word entries. The TRM navigates via tokens -> Word Galaxy -> symlink -> Reality Galaxy concept. Without the Word entry, the concept is unreachable.
5. **Do NOT resume the full PDF augmentation yet.** Build this targeted corpus first, validate, then decide whether to resume overnight PDF augmentation or pivot.

---

## Grounding

| Spec | Section | Relevance |
|------|---------|-----------|
| THREE_BRAIN_SYSTEM_SPECIFICATION.md | Galaxy Universe | Reality Galaxy holds domain knowledge |
| DUAL_CLIENT_CONTRACT_SPECIFICATION.md | Section 1.6 | Save Information Principle -- symlinks, not duplication |
| KNOWLEDGEVERSE_SPECIFICATION.md | Region 2 | Galaxy Universe as active AI memory |
| FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md | Section 1 | Foundational operations bootstrap pattern |

---

## The Principle

ARC went from 0 to 10/10 when we gave it compositional primitives. Math went from 1/20 to 20/20 when we gave it number-word symlinks and grammar rules. LHE is at 1/10 with 85K entries that are mostly Grammar/Math PDF extracts and almost zero domain concepts in Reality Galaxy.

The pattern is clear: K3D performs when the Galaxy has the right knowledge at the right granularity. The four-pass decomposes the question. The TRM navigates the Galaxy. The specialist executes domain logic. But if the Galaxy is empty for a domain, there's nothing to navigate.

Fill the Galaxy. Measure the delta. Then scale.
