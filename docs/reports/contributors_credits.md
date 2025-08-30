# Knowledge3D — Contributors & Credits

This document acknowledges collaborators (human and AI) who shaped this phase.

## Human in the Loop
- Daniel Campos Ramos — Vision, direction, and orchestration; provided prompts, curation, and strategic decisions that guided functionality and scope.

## AI Partners
- Codex (OpenAI) — Hands-on coding agent integrating live bridge, Tablet apps, Agentic Browser, multilingual model routing, and viewer extensions (AI suggestions, layers, LOD, gazetteer/aliases). Implemented training/eval scripts, logs, and scoreboard wiring.
- Perplexity AI — Recommended lightweight cross‑lingual slot extraction via hybrid rule+gazetteer+affix, and prioritized steps for open‑vocabulary goto. Influenced the gazetteer canonicalization, alias ingestion, and TF‑IDF hybrid.
- DeepSeek — Provided guidance on LOD strategy: multi-level decimation, screen-space thresholds, hysteresis, and opacity fades. Informed devicePixelRatio and FOV-aware LOD and ease-in-out quadratic blending for seamless transitions.

## Recent Advancements (This Phase)
- Agentic Browser with Wikipedia search; session logs for training.
- Tablet: Live Stats app, Layers app, Galaxy expand/freeze controls.
- Live Server: model-to-action routing with multilingual parsing; goto via gazetteer + TF‑IDF + alias enrichment; `goto_resolution` logs.
- Viewer: SmartGraph extensions (AISuggestionManager [mock], DynamicLayerManager, LODRenderer with hysteresis + eased fades), Layers overlay and Tablet panel; alias enrichment on load.
- Tools: `build_aliases.py` for Wikipedia redirects.

## Notes
- All work aligns to the K3D research, ROADMAP, and CODEX tasks. Logs produced via the live server are treated as training data and not committed.

