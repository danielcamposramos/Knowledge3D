# Phase 7.6 Live Server Purge Spec — 2026-04-18

**Target:** `/K3D/GitHub/Knowledge3D/knowledge3d/bridge/live_server.py`
**Goal:** eliminate 16 numpy + 1 torch import sites; remove
`live_server.py` carve-out from `scripts/sovereignty_preflight.sh`.

## 1. Executive summary

`live_server.py` is the IRC/WebSocket Tablet bridge. All 17 offending
sites fall into three disposition buckets:

- **(A) Sovereign-required** — live `/goto` / `/llm rag` / `/ask` / `/trit-*`
  paths: trivial argmax + TF-IDF cosine math. Replace with stdlib
  (`math`, `heapq`, `ctypes`, `bytearray`) OR raise `NotImplementedError`
  pointing to Qdrant MCP for the autonomy [suggest] block.
- **(B) Ingestion-side** — `/think` torch checkpoint load + `.npy`
  embedding I/O + gltf f32 unpack. Relocate to a new
  `knowledge3d/ingestion/embedding_io.py` (ingestion tree is NOT hot
  path) and lazy-import from there. Call sites stay sovereign.
- **(C) Dead/mock code** — `/trit-overlay` / `/trit-inspect` /
  `/trit-path` / `/trit-depth` generate mock data via `np.random.*`.
  Replace with stdlib `random` + list comprehensions + `bytearray` +
  `ctypes.c_uint32`.

**Outcome:** `live_server.py` contains zero `numpy|cupy|scipy|sympy|torch`
imports. Preflight carve-out is removed.

## 2. Per-site disposition table

| # | Line | Site | Class | Replacement | LOC |
|---|------|------|-------|-------------|-----|
| 1 | 129 | `import numpy as _np` (TF-IDF init) | A | Drop `_np`; keep `self._TFIDF`; delete `self._NP`. | -3 |
| 2 | 945 | `import torch as _t` (/think) | B | Move into `knowledge3d/ingestion/embedding_io.py::load_thinking_tag_embedder`. | ~8 |
| 3 | 1872 | `import numpy as _np` (/sleep) | B | Move to `embedding_io.load_npy_rows(path)`. | ~5 |
| 4 | 1896 | `_np.load(epath)` | B | Same module: `load_npy_row(path, i)`. | 2 |
| 5 | 1908 | `_np.load(epath)` | B | Same as #4. | 2 |
| 6 | 2097 | `import numpy as _np` (gltf unpack) | A | Drop `_np`; keep `struct`. Use `list(_st.unpack('<'+'f'*n, data))`. | -2 |
| 7 | 2124 | `_np.array(..., dtype=float32)` | A | Covered by #6. | 0 |
| 8 | 2459 | `import numpy as _np` (autonomy suggest) | A→C | `NotImplementedError("autonomy-link-suggest: use qdrant-find; Phase 7.6 spec")`. | -6 |
| 9 | 2462 | `_np.fill_diagonal(S,-1)` | A | Removed with #8. | 0 |
| 10 | 2463 | `_np.unravel_index(_np.argmax(S), S.shape)` | A | Removed with #8. | 0 |
| 11 | 2540 | `import numpy as np` (TF-IDF /goto cosine) | A | sklearn CSR iteration via `.data`/`.indices`; `math.sqrt`. | ~12 |
| 12 | 2544 | `row_norms = np.sqrt(...).A1` | A | `[math.sqrt(sum(v*v for v in X.getrow(i).data)) for i in range(X.shape[0])]`. | 1 |
| 13 | 2548 | `np.where(denom==0, 1.0, denom)` | A | Inline `(rnorm * qnorm) or 1.0` in per-row loop. | 1 |
| 14 | 2648 | `import numpy as np` (path-label sims) | A | Same pattern as #11. | ~6 |
| 15 | 2652 | `row_norms = np.sqrt(...).A1` | A | Same as #12. | 1 |
| 16 | 2869 | `import numpy as _np` + `argsort` (/llm rag) | A | `heapq.nlargest(k, range(n), key=scores.__getitem__)`. | ~2 |
| 17 | 2899 | `import numpy as _np` + `argsort` (/ask) | A | Same as #16 with k=8. | ~2 |
| 18 | 3132 | `np.random.choice([-1,0,1], n)` (/trit-overlay) | C | `[random.choice((-1,0,1)) for _ in range(n)]`. | 2 |
| 19 | 3190 | same (/trit-inspect) | C | Same as #18. | 2 |
| 20 | 3234 | same (/trit-path) | C | Same as #18. | 2 |
| 21 | 3274/3279/3280/3283-4 | `np.random.randn` + `np.linalg.norm` (/trit-depth) | C | `random.gauss(0,1)` + `math.sqrt(sum(x*x))`. | ~10 |
| 22 | 3330/3333/3339 | `_pack_trits_helper`: `np.zeros(uint32)` + bit-OR | A | `(ctypes.c_uint32 * n)()` + `packed[w] = (packed[w] \| (bits<<shift)) & 0xFFFFFFFF`. | ~4 |
| 23 | 3344/3345 | `_trits_to_rgba`: `np.zeros((n,4), uint8)` | A | `bytearray(n*4)` + per-row slice writes. | ~6 |

## 3. Order of operations (batches)

**Batch 1 — Trivial stdlib math** (sites 6, 7, 16, 17, 22, 23)
Pure mechanical. Import smoke after.

**Batch 2 — Mock benchmark data** (sites 18, 19, 20, 21)
Verify `TritInspector` / `AdaptiveTernaryDepth` accept lists first.

**Batch 3 — Ingestion relocation** (sites 2, 3, 4, 5)
Create `knowledge3d/ingestion/embedding_io.py`.

**Batch 4 — TF-IDF sovereign refactor** (sites 1, 11–15)

**Batch 5 — Dead-code escalation** (sites 8, 9, 10)

**Batch 6 — Preflight unlock**
Remove the `live_server.py` carve-out lines from
`scripts/sovereignty_preflight.sh`.

## 4. Verification plan

1. `grep -nE '^[[:space:]]*(import|from)[[:space:]]+(numpy|cupy|scipy|sympy|torch)' knowledge3d/bridge/live_server.py` → **zero** lines.
2. `bash scripts/sovereignty_preflight.sh` → full-tree clean.
3. `python -c "from knowledge3d.bridge import live_server"` → clean import.
4. Smoke-test tablet commands: `/goto`, `/llm rag`, `/ask`, `/trit-overlay`,
   `/think_path`, `/sleep` (if .npy present), `/think` (if pth present).
5. Run existing regression tests (`tests/bridge/test_headless_tablet.py`,
   `tests/test_tablet_sovereign_query.py`).
6. Pre-commit hook runs preflight on live_server.py with no carve-out.

## 5. Risks / rollback

- **TF-IDF correctness (Batch 4):** if per-row cosine diverges >1e-4
  from old numpy path, fall back to Batch 5's NotImplementedError
  strategy and route `/goto` to Qdrant MCP.
- **`TritInspector` internals (Batch 2):** if they require numpy
  internally, they're cranium hot-path violations and must be purged
  before Batch 2 lands. Grep `knowledge3d/cranium/tools/` first.
- **Batches are independent commits:** revert individually. Batch 6
  (preflight unlock) lands last; can be reverted alone to re-enable the
  carve-out without touching the code.

## 6. Files touched

- `knowledge3d/bridge/live_server.py` (primary edit target)
- `knowledge3d/ingestion/embedding_io.py` (new, created in Batch 3)
- `scripts/sovereignty_preflight.sh` (Batch 6 carve-out removal)
- `knowledge3d/cranium/tools/trit_inspector.py` (inspect for list-input)
- `knowledge3d/cranium/tools/adaptive_ternary_depth.py` (inspect for list-input)

---

Authored after the Plan sub-agent's live_server.py walk.
Execution tracked in the Phase 7.6 todo chain.
