# Early LLM Evaluation Plan

Learning Goals
- Spatial grounding: map text concepts to neighborhoods; reason via neighbor hops.
- Explainability: articulate path steps with cosine similarity and labels.
- Door usage: propose and use doors for intentional transitions.
- Reflection: pause, summarize state, and suggest next actions.

Objective Tests (Phase 1)
- GOTO: Given a label substring, produce a navigation intent and arrive via neighbors (<= N hops). Metric: success rate, median hops.
- CLUSTER EXPLAIN: Pick two clusters; explain contrast using labels and local neighbors. Metric: rated clarity.
- DOOR OPEN: Choose a relevant door label; open and justify. Metric: door validity, justification relevance.
- REFLECTION: Provide a coherent house reflection (nodes, avg degree, hubs). Metric: completeness vs. reflect_glb baseline.

Artifacts
- Houses: UMAP GLBs (256/1k/4k/full), guidance variants with doors/masks.
- Logs: JSONL sessions; reflections in `docs/reports/reflections/` via `reflect_glb.py`.

Harness (offline scaffolding)
- Task generator and baseline routes: `knowledge3d/tools/reflect_glb.py` and (optional) `eval_tasks.py` for random pair navigation checks.
- Manual prompts: `/ask-thoughts`, `/whoami`, `goto <label>` in viewer.

Scoring
- Success/hops for GOTO, door validity ratio, reflection completeness vs. stats, human rating for cluster explain.

Path to ARK-AGI
- Build curriculum from small trees to full content; increase complexity (longer hops, multi-door reasoning, multilingual labels).
- Integrate environment interactions later (Phase 2/3), preserving explainability at each step.

