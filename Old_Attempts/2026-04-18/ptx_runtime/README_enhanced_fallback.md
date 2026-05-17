# Archived: enhanced_fallback.py

**Original Path**: `knowledge3d/cranium/ptx_runtime/enhanced_fallback.py`  
**Archive Date**: 2026-04-18  
**Reason**: K3D no-fallbacks rule (Batch 11 mandate): all fallback code is legacy

## Why Archived

K3D Mandate (March 31, 2026):
> "No Python fallbacks. EVER. Not in hot path, not in sleep-time, NOWHERE. We fix or we fix."

This module implements a "graduated fallback hierarchy"—a design pattern incompatible with K3D's zero-tolerance fallback rule. If a kernel or PTX operation fails, K3D must propagate the error and fix the code; fallback paths hide design problems.

**All fallback code is now archived.**

## Replacement Strategy

- If a kernel fails, trace the error and fix the kernel (not the fallback)
- Errors are traced via note-taking (observability mandate, April 15, 2026)
- See: MEMORY.md "Take Notes Everywhere — Observability"

---

*This file was moved as part of no-fallbacks enforcement. See feedback_no_fallbacks_ever_including_sleeptime.md.*
