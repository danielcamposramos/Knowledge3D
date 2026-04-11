Based on the provided K3D vocabulary specifications, here is the architecture validation for the ARC-3 fresh weights + discriminative embedding plan.

### 1. Does resetting TRM weights while keeping Galaxy violate any K3D architectural principle?

**Verdict: Compliant (with conditions)**

**Analysis:**
This action aligns with the **Separation of Concerns** principle defined in the `THREE_BRAIN_SYSTEM_SPECIFICATION.md`. The architecture explicitly distinguishes between **Reasoning** (Cranium/TRM), **Active Memory** (Galaxy), and **Persistence** (House).

*   **Separation of Concerns:** Section 1.2 states: *"Reasoning ≠ Memory ≠ Persistence."* Resetting the TRM weights (Reasoning) while retaining Galaxy stars (Memory) adheres to this separation. You are resetting the "processor" without wiping the "RAM."
*   **Rollback Capability:** Section 1 (Critical Architecture Paradigm) states: *"The brain model persists across wake cycles as a versioned entity with rollback capability."* Resetting to a fresh checkpoint is architecturally valid as a "rollback" event to ensure stability or clear corrupted learning states.
*   **Continuous Improvement Caution:** Section 1 (Critical Architecture Paradigm) also notes: *"K3D is... a living, always-on, embodied AI that perfects itself during idle time."* While a reset is allowed via rollback, it interrupts the *"Shadow Copy learning mechanism enables continuous self-improvement during inference (no external training loops)"* (Abstract).
    *   **Recommendation:** Frame this operation as a **Versioned Rollback** rather than a "retrain." Ensure the `adaptive_swarm` checkpoint is versioned in the House so the system can trace provenance (Section 2.1: *"House = Memory Palace... Provenance"*).

### 2. Is the directional mapping (ACTION1=north, ACTION2=south) architecturally sound?

**Verdict: Violation**

**Analysis:**
Hardcoding semantic mappings in Python violates two core pillars of the K3D architecture: **Python's Role** and **Meaning-Centricity**.

*   **Python's Role:** `THREE_BRAIN_SYSTEM_SPECIFICATION.md` (Critical Architecture Paradigm) explicitly states: *"Python = Boot + I/O only (~200 lines target). ALL reasoning happens on GPU via PTX kernels."* Mapping `ACTION1` to the semantic concept "north" in Python constitutes reasoning logic residing in the boot layer, not the GPU kernel.
*   **Meaning at the Center:** `MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md` Section 1.2 states: *"A star represents a concept, not a word... The meaning star IS the concept."*
    *   **Correction:** `ACTION1` should not map to a string `"north"`. Instead, `ACTION1` should reference a **Star ID** (e.g., `star_id_concept_north`) stored in the Galaxy. The *meaning* of North must exist as a Meaning-Centric Star (with visual, spatial, and relational references), not as a Python string literal.
    *   **Game State:** The GAME state should provide the *contextual vector* (e.g., current orientation), but the *semantic label* must be resolved by the TRM navigating the Galaxy stars, not by Python assignment.

### 3. Will the fresh TRM learn meaningful routing patterns from the contrastive embeddings, or is 32 dimensions too few?

**Verdict: High Risk / Architectural Deviation**

**Analysis:**
While 32 dimensions may offer speed, it deviates from the validated embedding specifications required for sovereign reasoning.

*   **Embedding Dimensions:** `SOVEREIGN_TRAINING_SPECIFICATION.md` Section 2.2 (`MultiModalGridEmbedder`) specifies: *"512-dim embedding (Matryoshka, adjustable 64-2048)."*
    *   The proposed **32-float** embedding is below the documented minimum adjustable range (64).
    *   **Risk:** Section 2.1 states intelligence is achieved *"Through Procedures, Not Parameters"* using multimodal embeddings (Video + Audio → Ternary). Compressing this multimodal semantic data into 32 dimensions risks **semantic collapse**, where distinct concepts (e.g., "north" vs. "blocked") become indistinguishable in cosine similarity space.
*   **Ternary Readiness:** `THREE_BRAIN_SYSTEM_SPECIFICATION.md` (Critical Architecture Paradigm) notes: *"Ternary-ready registers — all intermediate results carry value + confidence + polarity."* A 32-float normalized embedding lacks the explicit ternary structure (-1/0/+1) recommended for intermediate results, potentially reducing the efficiency of the *"semantic gravity"* described in `MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md`.
*   **Recommendation:** Align with the Sovereign Training Spec minimum of **64 dimensions** to ensure sufficient capacity for the discriminative task (ACTION1-4 + States + Colors) without violating the validated architecture.

### 4. Should the semantic color names be stored in Galaxy as knowledge rather than hardcoded in Python?

**Verdict: Mandatory Requirement**

**Analysis:**
Hardcoding color names in Python is a direct violation of the **Meaning-Centric Star Schema**.

*   **Concept vs. Word:** `MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md` Section 1.3 states: *"This is not metaphor — it's architecture. The star `concept_cat` exists once... Reasoning about cats uses the MEANING, not the English word 'cat'."*
    *   Applying this to colors: `"red"` must not be a string. It must be a `MeaningCentricStar` (e.g., `concept_red`) containing visual RPN (wavelengths), cultural references, and relational data.
*   **Language Agnosticism:** Section 1.2 states this design makes K3D reasoning *"language-agnostic by construction."* Hardcoding English names ("red", "green") in Python binds the reasoning to English surface forms, breaking language agnosticism.
*   **Implementation:**
    *   **House:** Store `concept_red`, `concept_green`, `concept_blue` as persistent stars.
    *   **Galaxy:** Load these stars into VRAM during initialization.
    *   **Python:** Pass only the `star_id` references to the GPU. The TRM must resolve the semantic meaning via embedding similarity against these color stars, not via string matching.

### Summary of Required Corrections

| Component | Current Plan | Architectural Requirement | Spec Reference |
| :--- | :--- | :--- | :--- |
| **TRM Reset** | Reset weights, keep Galaxy | **Compliant** (if versioned as Rollback) | `THREE_BRAIN` §1 (Rollback) |
| **Directions** | Python `ACTION1="north"` | **Violation** (Must be Galaxy Star IDs) | `THREE_BRAIN` §1 (Python Role) |
| **Embeddings** | 32-float normalized | **Risk** (Min 64-dim per Spec) | `SOVEREIGN_TRAINING` §2.2 |
| **Colors** | Python strings ("red") | **Violation** (Must be Meaning Stars) | `MEANING_CENTRIC` §1.2 |