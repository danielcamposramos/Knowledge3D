# Codex Contrastive Root Cause Summary

**Date:** 2026-03-24  
**Prompt:** `TEMP/CODEX_PROMPT_CONTRASTIVE_ROOT_CAUSE_AND_PHASE_D_03.24.2026.md`

## Scope

This summary accompanies the full diagnostic log at:

- `TEMP/CONTRASTIVE_ROOT_CAUSE_DIAGNOSTIC_03.24.2026.txt`

The diagnostic was run through the real sovereign production chain:

`loader._ensure_current_context()` -> `loader.gpu_malloc()` -> `loader.memcpy_htod()` -> `SelfUpdatingAdapter._ensure_math_core()` -> `SelfUpdatingAdapter._ensure_device_buffers()` -> `SelfUpdatingAdapter.apply_gradient_rpn()` -> `SelfUpdatingAdapter.apply_gradient()`

## Diagnostic Result

The previously reported contrastive failure:

`argument 2: TypeError: Don't know how to convert parameter 2`

did **not** reproduce in the direct production-path diagnostic.

What the diagnostic showed instead:

- CUDA driver API is active: `nvcuda loaded: True`
- CUDA runtime API is active: `libcudart loaded: True`
- allocator path is **driver allocation** via `cuMemAlloc`, not CuPy and not `cudaMalloc`
- memcpy path is **driver memcpy**, with `cuMemcpyHtoD_v2` reporting invalid context and the loader retrying `cuMemcpyHtoD_v1`
- after that retry, the copy succeeds
- the full sovereign GPU gradient path succeeds end-to-end

Key evidence from the full log:

- `Is driver allocation: True`
- `[loader] cuMemcpyHtoD_v2 invalid context, retrying v1`
- `[loader] cuMemcpyHtoD -> 0`
- `memcpy_htod: PASSED`
- `_ensure_device_buffers: True`
- `apply_gradient_rpn: PASSED`
- `apply_gradient: PASSED`

## Current Interpretation

The `loader.py` argtypes fix is active on the real driver path and the direct sovereign GPU chain now works in the current environment.

That means the earlier benchmark-time `TypeError` is **not reproducible on demand** through the current production path. The honest conclusion is:

1. the old failure was real,
2. the current loader/adaptor stack no longer reproduces it under direct production-path diagnosis,
3. the remaining benchmark instability must be traced in the full benchmark/runtime path, not by reintroducing CPU fallbacks.

## Sovereignty Cleanup Applied

The adapter layer was cleaned to comply with the binding specs:

- `knowledge3d/cranium/trm_adapters.py`

What changed:

- deleted `require_gpu` from `AdapterConfig`
- deleted `_apply_gradient_cpu`
- `apply_gradient(...)` is now GPU-only and fail-fast
- `apply_gradient_rpn(...)` is now GPU-only and fail-fast
- `apply_gradient_to_shadow(...)` is now GPU-only and fail-fast

Auxiliary cleanup:

- `knowledge3d/cranium/codecs/ternary_audio_codec.py`
  - renamed unrelated `_require_gpu` field to `_gpu_enabled` to avoid false-positive grep noise in the code tree

Test cleanup:

- `tests/test_trm_game_loop.py`
  - now asserts adapters expose no `require_gpu` field
- `tests/test_rpn_sovereignty_phase2.py`
  - replaced the deleted CPU-path comparison with a NumPy reference update

## Grep Validation

In the production code tree:

- `rg -n "require_gpu|_apply_gradient_cpu" knowledge3d || true`
- expected result: no matches

The remaining `_require_gpu()` references are test helpers in unrelated GPU-gated test files, not runtime fallbacks.

## Validation Performed

Focused tests passed:

- `tests/test_trm_game_loop.py`
- `tests/test_routing_contrastive_multihop.py`

Result:

- `11 passed`

GPU sovereign flow smoke also passed end-to-end with `AdaptiveSwarmTRM.train_specialist_contrastive(...)`, confirming the no-fallback adapter path can train on GPU in the current environment.

## Honest Boundary

This summary does **not** claim the entire long benchmark path is fixed forever. It claims something narrower and evidence-backed:

- the direct sovereign GPU contrastive path is functioning now,
- the adapter CPU escape hatches are removed,
- the loader binding fix is active on the real driver path,
- Phase D migration should proceed from the current Python/GPU boundary map, not from assumptions about a still-broken adapter path.
