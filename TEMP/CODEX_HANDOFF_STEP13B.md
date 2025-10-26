# Codex Handoff — Step 13-B (Phase 0)

## 1. Test Failure Taxonomy
| Category | Tests Affected | Symptoms |
|----------|----------------|----------|
| Action Buffer API | `tests/test_step12_action_buffer_integration.py` (21/21) | `ThinkingTagBridge` lacks 288-byte struct fields, modal bitfields, serialization helpers, curiosity/novelty metrics. |
| Cognitive Pipeline / State Trace | `tests/test_step12_cognitive_pipeline.py` (29/29) | Missing `get_state_trace_report`, `export_state_trace`, pruning, percentile stats, error annotations. |
| Dynamic LOD / Saliency | `tests/test_step12_dynamic_lod.py` (7/15 collected; 7 failed) | No `dynamic_lod_kernel`, saliency buffers, Morton threshold tuning, fallback hooks. |
| Misc (already passing) | 8 tests | Basic smoke tests that rely solely on existing mocks. |

## 2. Bridge Surface Gap (Expected vs Available)
- **State Trace**: `get_state_trace_report`, `export_state_trace`, `clear_state_trace`, `prune_state_trace`, percentile stats, error tagging.
- **Action Buffer**: 288-byte struct mirroring `ActionBuffer`, modal signature bitfields, curiosity scoring, serialization/deserialization helpers.
- **Dynamic LOD**: `dynamic_lod_kernel`, saliency buffer population, Morton saliency metrics, fallback behaviour under OOM.
- **Auxiliary Hooks**: Novelty metrics, JSON export path (`export_state_trace`), concurrency-safe population.

## 3. Fixture Strategy Recommendation
- **Preferred**: Add a shared augmentation helper (e.g. `tests.utils.bridges.ensure_step12_surface`) that patches any `ThinkingTagBridge` instance with the Step 12 mock surface. Call this helper in:
  - `tests/conftest.py` bridge fixture.
  - Direct instantiations within Step 12 test modules.
- **Rationale**: Keeps test-only behaviour isolated, avoids modifying runtime bridge in production, and supports both import paths.
- **Hybrid Option**: If long-term alignment desired, expose a light `Step12MockBridge` in `tests.utils` and update Step 12 tests to consume it via fixtures.

## 4. Priority Queue for Fixes
1. Implement augmentation helper + patch `tests/test_step12_*` modules to use it (unblocks majority of failures).
2. Re-run `pytest tests/test_step12_cognitive_pipeline.py` to verify state-trace surface.
3. Address dynamic LOD mocks (`dynamic_lod_kernel`, saliency buffers) and re-test `tests/test_step12_dynamic_lod.py`.
4. Validate action buffer metrics (confidence/curiosity) with `tests/test_step12_action_buffer_integration.py`.
5. Full suite re-run `pytest tests/test_step12_*.py` and update `reports/all_issues_found.md`.

— Codex (Step 13-B handoff)
