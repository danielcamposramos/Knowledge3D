# Language Plan (Early Multilingual + Ancient)

Goals
- Introduce multilingual data early and grow steadily.
- Favor languages with text-line representations (vectorizable via embeddings).
- Permit one pictographic exception: Sumerian cuneiform (with transliteration fallback).

Current status
- Included in care pack: EN, PT-BR, ES.
- Embeddings model: `sentence-transformers/all-MiniLM-L6-v2` (works well for EN, reasonably for multi-language; can switch to `paraphrase-multilingual-MiniLM-L12-v2` when needed).

Near-term additions
- Modern: FR, DE, IT, ZH (simplified), JA, KO.
- Ancient (text-line capable with Unicode): Latin, Classical Greek (polytonic), Sanskrit (Devanāgarī), Biblical Hebrew.
- Exception: Sumerian cuneiform.
  - Phase 1: Use scholarly transliteration (Latin characters, e.g., `e2`, `lugal`, diacritics) to keep data vectorizable.
  - Phase 2: Optional glyph geometry showcase (Three.js layer), separate from embeddings.

How to extend
1) Add translated self-knowledge lines to the care pack in `knowledge3d/tools/build_ai_books.py` (function `self_knowledge_lines`).
2) Re-run: `python -m knowledge3d.tools.build_ai_books`.
3) Generate a small GLB for fast iteration:
   - `python -m k3dgen --text data/ai_care_multilang.txt --gltf viewer/public/ai_care_multilang.umap.glb --k 5 --reducer umap --emb-precision f16`
4) Add the new house to `viewer/public/condo.json`.

Notes
- Keep early increments small (≤1k lines per language) to preserve agility.
- Tag lines with a language prefix like `[pt]`, `[es]`, `[la]` for readability in the viewer.
- For scripts with complex diacritics, ensure your editor/viewer serves UTF‑8 and a font that supports the range.

