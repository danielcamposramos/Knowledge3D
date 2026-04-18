# Claude → Codex Spec: Old_Attempts Migration

**Date**: 2026-04-18
**Author**: Claude (architecture)
**Implementer**: Codex
**Daniel's Ruling (verbatim)**: "We don't need to rename, just move out of our way to 'Old_Attempts' folder, where all wrong attempts go in the end (who knows when this can be reused elsewhere...)."

---

## 1. Philosophy — Old_Attempts Is a Deliberate Archive, Not /dev/null

Old_Attempts is a graveyard in the dignified sense. Code placed here was wrong for K3D's current architecture, but Daniel explicitly leaves the door open: "who knows when this can be reused elsewhere." Preservation is intentional.

**What this means in practice:**
- File structure inside Old_Attempts mirrors the source path it came from (not flattened).
- Each migrated file or folder gets a `README_WHY_ARCHIVED.md` alongside it explaining: what made it wrong, what might make it right later, and who archived it.
- Files are MOVED, not deleted. git history follows with `git mv` so provenance is preserved.
- `Old_Attempts/` is permanently excluded from all grep-based sovereignty acceptance gates, bulk-lib audit scans, and CI sovereignty linter passes.

**The shim rule:** When sovereign code still imports a migrated module, leave a one-line Python shim at the original path. The shim does nothing except raise a loud error pointing to Old_Attempts. No silent fallthrough, no accidental reactivation.

---

## 2. Destination Layout

`Old_Attempts/` already exists at:
```
/K3D/GitHub/Knowledge3D/Old_Attempts/
```

Use dated theme subfolders for the current migration wave:

```
Old_Attempts/
├── README.md                          (already exists — update to add new batch)
├── DEPRECATED.md                      (already exists)
│
├── 2026-04-18_sovereignty_potemkins/
│   ├── README_WHY_ARCHIVED.md
│   ├── cranium/ptx_runtime/
│   │   ├── sovereign_multi_modal_embedder.py
│   │   └── multi_modal_world_generator.py
│   └── cranium/ptx_runtime/
│       └── enhanced_fallback.py
│
├── 2026-04-18_legacy_embedders/
│   ├── README_WHY_ARCHIVED.md
│   └── core/
│       └── legacy_rpn_python.py
│
└── 2026-04-18_transfer_yard_python_sidecar/
    ├── README_WHY_ARCHIVED.md
    └── (TransferYardStack class — extracted from transfer_yard_tiered.py, not a whole file move)
```

---

## 3. Migration Manifest — Files to Move RIGHT NOW

### 3.1 Potemkin Sovereign Files (highest priority — name says sovereign, behaviour is not)

| Original Path | Destination | Why |
|---|---|---|
| `knowledge3d/cranium/ptx_runtime/sovereign_multi_modal_embedder.py` | `Old_Attempts/2026-04-18_sovereignty_potemkins/cranium/ptx_runtime/sovereign_multi_modal_embedder.py` | Imports `SentenceTransformer`, `numpy`, `cv2`, `PIL` at module level. The word "sovereign" in the name is a misnomer — this is the opposite of sovereign. Replaced by `rpn_meaning_project.ptx` (Phase B spec). |
| `knowledge3d/cranium/ptx_runtime/multi_modal_world_generator.py` | `Old_Attempts/2026-04-18_sovereignty_potemkins/cranium/ptx_runtime/multi_modal_world_generator.py` | Imports `SentenceTransformer` (line 11) and re-exports `SovereignMultiModalEmbedder` (line 23). Depends on the above file. Same violation class. |
| `knowledge3d/cranium/ptx_runtime/enhanced_fallback.py` | `Old_Attempts/2026-04-18_sovereignty_potemkins/cranium/ptx_runtime/enhanced_fallback.py` | Implements a `FallbackLevel` graduated fallback hierarchy. Fallbacks are explicitly forbidden by K3D sovereignty rules: "We fail and fix — this is the goal." No graduated fallback mechanism should exist in sovereign paths. |

### 3.2 Legacy Python RPN Core

| Original Path | Destination | Why |
|---|---|---|
| `knowledge3d/core/legacy_rpn_python.py` | `Old_Attempts/2026-04-18_legacy_embedders/core/legacy_rpn_python.py` | The filename itself is the verdict. PTX kernels + Transfer Yard are the RPN execution substrate. Python RPN calculator is the wrong path superseded by `rpn_math_core.py` + `modular_rpn_kernel_transfer_yard.ptx`. |

### 3.3 Python TransferYardStack Sidecar (class extraction, not whole-file move)

The class `TransferYardStack` inside `knowledge3d/cranium/bridges/transfer_yard_tiered.py` (approx lines 28-78) is a Python-side simulation of what the GPU kernel should do. It pre-dates the real Transfer Yard kernel and was never intended to be permanent.

**Action**: Extract the class body and save it as `Old_Attempts/2026-04-18_transfer_yard_python_sidecar/transfer_yard_stack_class.py`, then delete the class from `transfer_yard_tiered.py`. This is NOT a whole-file move — the rest of `transfer_yard_tiered.py` is the live Tier 2 bridge and stays.

### 3.4 Deferred: Legacy PTX Kernel (conditional on Transfer Yard landing)

| File | Condition | Action when condition met |
|---|---|---|
| `knowledge3d/cranium/ptx/modular_rpn_kernel_lite.ptx` | After `modular_rpn_kernel_transfer_yard.ptx` compiles and all Tier 1 acceptance gates pass | Move to `Old_Attempts/2026-04-18_legacy_kernels/ptx/modular_rpn_kernel_lite.ptx` |
| `knowledge3d/cranium/kernels/modular_rpn_kernel.cu` (original LIFO source) | Same condition | Move to same folder |

Codex: track this deferred item in the PR. Do not move these PTX/CU files until the replacement is proven live.

---

## 4. README_WHY_ARCHIVED.md Template

Each subfolder in Old_Attempts MUST have one. Use this template exactly:

```markdown
# README_WHY_ARCHIVED

**Archived**: 2026-04-18
**Archived by**: Codex (per spec CLAUDE_CODEX_OLD_ATTEMPTS_MIGRATION_04.18.2026.md)
**Ruling**: Daniel Campos Ramos

## What went wrong

[2-4 sentences: the specific architectural violation or wrong approach]

## What might make it right later

[2-4 sentences: under what conditions this code or pattern could be revisited — or "None identified" if truly dead]

## Replaced by

[File path(s) of the live replacement, or "Pending" if not yet written]

## Call sites that were rewired

[List of file:line that previously imported this module, now pointing to the replacement or raising NotImplementedError via shim]
```

---

## 5. Shim Replacement Strategy

For every file in §3 that has active importers, Codex places a **one-line shim** at the original path. The shim raises an error with a clear message.

### Shim template

```python
# ARCHIVED — see Old_Attempts/2026-04-18_sovereignty_potemkins/...
# This module was a sovereignty violation (SentenceTransformer / fallback logic / Python RPN).
# Replacement: knowledge3d/cranium/rpn_meaning_projector.py (Phase B spec)
raise NotImplementedError(
    "sovereign_multi_modal_embedder was moved to Old_Attempts/ on 2026-04-18. "
    "See Old_Attempts/2026-04-18_sovereignty_potemkins/README_WHY_ARCHIVED.md. "
    "Use knowledge3d/cranium/rpn_meaning_projector.project(meaning_rpn) instead."
)
```

The shim is a `.py` file containing ONLY the raise statement and the three comment lines above it. No imports. No logic.

### Import audit (Codex: run these before placing shims)

```bash
# Find all importers of sovereign_multi_modal_embedder
grep -rn "sovereign_multi_modal_embedder\|SovereignMultiModalEmbedder" \
    knowledge3d/ --include="*.py" \
    --exclude-dir=Old_Attempts
# Expected: multi_modal_world_generator.py (which is itself being archived), possibly answer_ranker.py

# Find all importers of multi_modal_world_generator
grep -rn "multi_modal_world_generator\|MultiModalWorldGenerator" \
    knowledge3d/ --include="*.py" \
    --exclude-dir=Old_Attempts

# Find all importers of enhanced_fallback
grep -rn "enhanced_fallback\|FallbackLevel\|FALLBACK_BUDGET" \
    knowledge3d/ --include="*.py" \
    --exclude-dir=Old_Attempts

# Find all importers of legacy_rpn_python
grep -rn "legacy_rpn_python" \
    knowledge3d/ --include="*.py" \
    --exclude-dir=Old_Attempts

# Find all importers of TransferYardStack
grep -rn "TransferYardStack" \
    knowledge3d/ --include="*.py" \
    --exclude-dir=Old_Attempts
```

For each call site found: update the import to the sovereign replacement, or if no replacement exists yet, leave the shim in place and file a note in the PR listing pending rewires.

---

## 6. Audit Exclusion

`Old_Attempts/` MUST be excluded from every sovereignty check. Codex confirms each exclusion point:

### 6.1 Bulk-lib audit grep commands

Every grep gate in the bulk-lib audit spec (`CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md`) uses `--exclude-dir=Old_Attempts`. Example:

```bash
grep -rn "import numpy\|from numpy" knowledge3d/ --exclude-dir=Old_Attempts
# Old_Attempts may contain any number of numpy imports — that is correct and expected.
```

### 6.2 CI sovereignty linter

Add to `.gitattributes` or the linter config:

```
Old_Attempts/** linguist-vendored
```

And in whatever CI sovereignty script exists (likely `tests/test_*_sovereignty_grep.py`):

```python
EXCLUDED_DIRS = ["Old_Attempts", "tests", "scripts/ingestion"]
```

### 6.3 Core isolation acceptance gates

The grep commands in `CLAUDE_CODEX_INSTANTIABLE_CORE_ISOLATION_04.18.2026.md` §6 all operate on `knowledge3d/cranium/` — Old_Attempts is outside that tree, so exclusion is automatic. Codex: verify this is the case before running.

---

## 7. Acceptance Gates

### Gate 1 — Files moved (not deleted)
```bash
test -f Old_Attempts/2026-04-18_sovereignty_potemkins/cranium/ptx_runtime/sovereign_multi_modal_embedder.py
test -f Old_Attempts/2026-04-18_sovereignty_potemkins/cranium/ptx_runtime/multi_modal_world_generator.py
test -f Old_Attempts/2026-04-18_sovereignty_potemkins/cranium/ptx_runtime/enhanced_fallback.py
test -f Old_Attempts/2026-04-18_legacy_embedders/core/legacy_rpn_python.py
# All four must exist.
```

### Gate 2 — Shims raise NotImplementedError
```bash
python -c "import knowledge3d.cranium.ptx_runtime.sovereign_multi_modal_embedder" 2>&1 | grep NotImplementedError
# → must find "NotImplementedError" in output
python -c "import knowledge3d.cranium.ptx_runtime.enhanced_fallback" 2>&1 | grep NotImplementedError
# → same
```

### Gate 3 — No silent imports of archived modules from live sovereign paths
```bash
grep -rn "sovereign_multi_modal_embedder\|multi_modal_world_generator\|enhanced_fallback\|legacy_rpn_python\|TransferYardStack" \
    knowledge3d/ --include="*.py" --exclude-dir=Old_Attempts
# All hits must be either:
#   (a) shim files at the original path (containing only the raise), or
#   (b) test files referencing the archived path explicitly for documentation purposes.
# No live production code may import archived modules.
```

### Gate 4 — Audit excludes Old_Attempts
```bash
# Run the Phase 1 bulk-lib audit command with exclusion:
grep -rn "import numpy\|from numpy" knowledge3d/ --exclude-dir=Old_Attempts | wc -l
# Must be less than the pre-migration count.
# Then run without exclusion to confirm Old_Attempts DOES contain numpy refs:
grep -rn "import numpy\|from numpy" Old_Attempts/ | wc -l
# Must be > 0 (confirms archiving happened and exclusion is meaningful).
```

### Gate 5 — README_WHY_ARCHIVED.md present in each subfolder
```bash
test -f Old_Attempts/2026-04-18_sovereignty_potemkins/README_WHY_ARCHIVED.md
test -f Old_Attempts/2026-04-18_legacy_embedders/README_WHY_ARCHIVED.md
test -f Old_Attempts/2026-04-18_transfer_yard_python_sidecar/README_WHY_ARCHIVED.md
# All three must exist with non-empty content.
```

---

## 8. Codex Handoff Checklist

1. Run the import audit grep commands in §5 before touching any files — record every call site.
2. Create folder structure: `Old_Attempts/2026-04-18_sovereignty_potemkins/cranium/ptx_runtime/` and sibling folders.
3. `git mv` each file to its destination per §3.1 and §3.2 (preserves git history).
4. Extract `TransferYardStack` class from `transfer_yard_tiered.py`, save to `Old_Attempts/2026-04-18_transfer_yard_python_sidecar/transfer_yard_stack_class.py`, delete from source.
5. Write `README_WHY_ARCHIVED.md` in each subfolder per §4 template.
6. Place shim `.py` files at original paths per §5 template.
7. Update every call site found in step 1 — either rewire to sovereign replacement or document as pending.
8. Add `--exclude-dir=Old_Attempts` to all grep gates in the sovereignty test files.
9. Add `Old_Attempts/** linguist-vendored` to `.gitattributes`.
10. Run all five acceptance gates (§7). Report pass/fail.
