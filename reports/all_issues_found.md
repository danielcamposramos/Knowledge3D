# Phase 0 Test Failures (Step 12 Suite)

| Test Module | Count | Key Symptoms |
|-------------|-------|--------------|
| `tests/test_step12_action_buffer_integration.py` | 21 | ActionBuffer attributes missing (size, modal bits, serialization) due to minimal bridge implementation. |
| `tests/test_step12_cognitive_pipeline.py` | 29 | State trace APIs absent/incomplete; percentile/statistics expectations unmet; error-handling hooks not mocked. |
| `tests/test_step12_dynamic_lod.py` | 7  | Dynamic LOD tuners and saliency metrics not exposed by fallback bridge; kernel hooks missing. |

## Root Causes
1. `knowledge3d.cranium.bridges.sovereign_bridges` no longer exports `ThinkingTagBridge`; tests fallback to `ptx_runtime.thinking_tag_bridge`, which is heavier than mocks but still misses several fields when called in isolation.
2. Global test fixtures (`conftest.py`) mock only a subset of bridge behaviours. Step 12 suites instantiate the bridge directly, bypassing the enriched fixture and exposing missing APIs.
3. Dynamic LOD helpers (`dynamic_lod_kernel`, saliency buffers) are not initialised by default when running outside the full runtime.

## Proposed Remediations
- Introduce a shared helper to augment `ThinkingTagBridge` instances with Step 12 mock behaviours (state trace, action buffer, LOD).
- Update Step 12 tests to import the helper via `tests.utils` to avoid duplicating mock code.
- Optionally export a lightweight `ThinkingTagBridge` stub from `sovereign_bridges` to maintain compatibility with historical imports.

## Artifacts
- Detailed log: `reports/phase0_results.txt`

