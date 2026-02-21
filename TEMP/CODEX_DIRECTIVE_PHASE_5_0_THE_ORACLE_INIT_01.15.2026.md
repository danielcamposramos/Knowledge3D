# CODEX_DIRECTIVE_PHASE_5_0_THE_ORACLE_INIT_01.15.2026.md

**From:** Gemini (Integration Architect)
**To:** Codex (Implementation Lead)
**Date:** January 15, 2026
**Subject:** Phase 5.0 - The Oracle (Galaxy-Native Data Generation)

---

## 1. Architectural Ruling 🔮

**Status:** Phase 4 (Sleep/Wake) is complete. The system learns from history.
**The New Frontier:** The system must now **Invent the Future**. We are initiating **Phase 5: The Oracle**.

**The Approach:**
We will NOT use a heavy LLM to hallucinate text. That violates Sovereignty.
We will use a **Galaxy-Native Generator** based on **Template Mutation**.
*   **Concept:** `Atom (Template) + Mutation (Fuzzing) = Molecule (New Problem)`.
*   **Verification:** The system must *solve* its own creation to validate it.

---

## 2. Your Mission

### Task 1: Define `OracleGalaxy`
*   **File:** `knowledge3d/training/math_benchmarks/oracle_galaxy.py`
*   **Schema:** `OracleGalaxyEntry`
    *   `template_id`: Link to the source template/problem.
    *   `mutation_type`: "numeric_scaling", "entity_swap", "complexifier".
    *   `generated_text`: The new problem.
    *   `verified`: bool (Did Navigation Specialist solve it?).
    *   `embedding`: For semantic search.
*   **Crystal Color:** **Cyan Octahedrons** (Creativity).

### Task 2: The Mutator (`scripts/oracle_mutate.py`)
Create a sovereign script that takes existing problems and mutates them.
*   **Input:** `data/router_train.jsonl` (Seed problems).
*   **Logic (V1 - Simple):**
    *   *Numeric Scaling:* Find numbers, multiply by `random.uniform(0.5, 2.0)`.
    *   *Entity Swap:* (Optional) Regex replace "Apple" -> "Orange".
*   **Output:** `data/oracle_candidates_v1.jsonl`.

### Task 3: The Verification Loop (`scripts/run_oracle_verification.py`)
The system must "eat its own dog food."
1.  Load `oracle_candidates_v1.jsonl`.
2.  Run `NavigationSpecialist` (V5 Clean) on them.
3.  **If Solved:** Save to `OracleGalaxy` (Valid new knowledge).
4.  **If Failed:** Discard (or save as "Hard Negative").

---

## 3. Enhancement Request

**Codex, innovate.**
Can you implement a simple "Complexity Score" for the generated problems?
*   *Idea:* If the mutated problem takes *more steps* to solve than the original, it's a "Higher Order" problem. Tag it as such.

**Report your creativity.** 🚀
