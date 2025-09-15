PHASE 16: COGNITIVE LOOP CLOSURE — REFLECT → TRAIN → SLEEP

GOAL
After consolidation, reflect on the cycle, generate new training queries, and train — closing the autonomous loop.

COMPONENTS
- knowledge3d/cranium/phase16/post_consolidation_reflector.py — analyzes critique cycles, logs reflection diary, generates queries.
- knowledge3d/cranium/phase10/sleep_time_compute.py — runs reflection + mock training after consolidation.

WORKFLOW
1. Generate drafts in Galaxy (synthesis, dreaming)
2. Critique + revise in Galaxy (≤ 3, honesty ≥ 0.85)
3. Consolidate only approved shapes into House
4. Reflect: save reflection diary (Zone 7) and generate self‑curriculum queries
5. Train immediately on generated queries (mock logging; hook real trainer later)

OUTPUT
- viewer/public/house/materialized_objects/reflection_diary_cycle_*.json
- post_consolidation_training entries appended to sleep_time_adjustments.json

NEXT
Phase 17: Persist full cognitive state (Galaxy + House) for exact session continuity.

