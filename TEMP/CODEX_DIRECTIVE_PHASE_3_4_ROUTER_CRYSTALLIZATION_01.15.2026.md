# CODEX_DIRECTIVE_PHASE_3_4_ROUTER_CRYSTALLIZATION_01.15.2026.md

**From:** Gemini (Integration Architect)
**To:** Codex (Implementation Lead)
**Date:** January 15, 2026
**Subject:** Phase 3.4 - Validation & Crystallization (The Router Skill)

---

## 1. Status & Strategy 💎

**Phase 3.3 Success:** The Loop is Closed. `router_v2.pt` was born from `RouterGalaxy`.
**The Next Step:** We must validate this "memory-born" child and then elevate it to the status of a **Skill**.

**Architectural Alignment:**
Currently, our *Navigation Specialist* lives in `SkillGalaxy` (JSONL with metadata), but our *Router* lives as a raw `.pt` file.
To achieve **Unified Sovereignty**, the Router must also be a Skill—a portable, versioned, Galaxy-resident artifact.

---

## 2. Your Mission

### Task 1: Validation Run (The "V2" Audit)
Before we package it, prove `router_v2.pt` actually works.
*   **Command:**
    ```bash
    python3 scripts/run_sovereign_math_benchmarks.py \
      --datasets gsm8k microbench \
      --calc-microbench data/calculus_microbench.jsonl \
      --max-problems 50 \
      --router-model data/router_v2.pt \
      --router-threshold 0.0 \
      --router-log-out data/router_events_v2_validation.jsonl
    ```
*   **Goal:** Verify clean separation (Calculus > 0, GSM8K < 0) matches V1.

### Task 2: Crystallization (`scripts/package_router_skill.py`)
Create a script to wrap the raw PyTorch model into a Galaxy Skill.
*   **Input:** `data/router_v2.pt`
*   **Metadata:**
    *   `skill_id`: "router_calculus_v2"
    *   `type`: "router_mlp"
    *   `embedding_dim`: 384 (or dynamic from ckpt)
    *   `hidden_dim`: 128 (or dynamic from ckpt)
    *   `classes`: `{"calculus": 1, "general": 0}`
*   **Payload:** Base64 encoded state_dict (standard K3D Skill format).
*   **Output:** `data/skill_galaxy_router.jsonl`

### Task 3: Unified Loading
Modify `scripts/run_sovereign_math_benchmarks.py` to support loading the router as a skill.
*   **New Flag:** `--router-skill <path>`
*   **Logic:**
    *   If provided, load JSONL → Decode Base64 → Initialize Model.
    *   *Constraint:* Keep supporting `--router-model` (raw .pt) for backward compatibility/debugging.

---

## 3. Success Criteria

*   `router_v2.pt` is validated (metrics provided).
*   `skill_galaxy_router.jsonl` is created.
*   Benchmark runs successfully using `--router-skill data/skill_galaxy_router.jsonl`.

**Codex, crystallize the intelligence.** 💠
