Wikipedia Sweep Evaluator

Purpose
- Sanity‑check non‑math routing by asking the fused head to summarize many Wikipedia‑style lines.
- Reports total prompts, non‑empty responses, and how often math/RPN traces appear (should be ~0).

Run
- GPU env: `conda run -n k3d-cranium env PYTHONPATH=. python -m knowledge3d.tools.wiki_sweep_evaluator --max-lines 0 --summarize`
- Output report: `logs/wiki_sweep_report.json`

Implementation Notes
- If the source passage isn’t already in the tablet corpus, the head uses an extractive fallback for prompts like `Summarize: <text>`.
- Corpus hydration recommended: convert curated paragraphs into `viewer/public/galaxy/working/wikipedia_corpus.jsonl` so summaries are sourced through the tablet path before fallback.

