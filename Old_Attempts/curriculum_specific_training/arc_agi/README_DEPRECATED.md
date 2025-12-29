# ARC-AGI Training Module — DEPRECATED

**Date Deprecated**: December 27, 2025
**Reason**: Violates "one model to process them all" architecture principle

## What Was Wrong

This module treated ARC-AGI as a SEPARATE model with:
- Duplicate galaxy implementations (drawing, grammar, math_symbol)
- Separate sleeptime consolidator
- Separate RPN executors

## What Replaced It

Unified architecture:
- `knowledge3d/cranium/procedural_galaxy.py` — Drawing Galaxy (serves ALL curricula)
- `knowledge3d/cranium/word_galaxy.py` — Grammar Galaxy (serves ALL curricula)
- `knowledge3d/cranium/math_galaxy.py` — Math Galaxy (serves ALL curricula)
- `knowledge3d/cranium/sleep_time_consolidator.py` — Unified consolidation

## Migration Path

If you need ARC-AGI training:
- Use `knowledge3d/training/curriculum_loaders/arc_agi_loader.py` for data loading
- Use `knowledge3d/training/unified_trainer.py` for training
- Use cranium/* galaxies (not local copies)

## Files Archived

- `drawing_galaxy.py` (10,617 lines)
- `grammar_galaxy.py` (38,223 lines)
- `math_symbol_galaxy.py` (17,781 lines)
- `sleeptime_consolidator.py` (11,744 lines)
- 40+ other files

Total: 34,337 lines of duplicated code
