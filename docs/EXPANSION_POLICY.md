Balanced Expansion Policy (Modality + Topics)
=============================================

Principles
----------
- Balance by count: For every Galaxy build, aim for roughly equal counts per modality (text, images, audio, video, 3D) and per topic cluster (e.g., vehicles, sports, gardens, tools, animals).
- Coherence first: Choose and reuse the same topic set across modalities so neighborhoods align by meaning, not by file type.
- GPU‑only retrieval: Use FAISS‑GPU or cuML for kNN where possible; otherwise build small, then unify (neighbors recomputed globally on GPU).
- Smart fallback: If open‑source data is exhausted in a modality/topic, proceed with a controlled imbalance and document the gap (prefer fewer but coherent samples over noisy matches).
- Live seeding payload cap: When galaxies get large, cap seeding WS payload via `K3D_SEED_GRAPH_MAX` to keep runs stable.

Sources
-------
- Local datasets under `/K3D/K3D_llama_cpp/datasets` (large raw/curated) and `../Knowledge3D.local/datasets` (curated/working sets).
- Text generation via local Ollama (e.g., `exaone3.5:latest`, `granite3.3:8b`) to fill topic slots while keeping lines short and grounded.
- Images (COCO), audio (Clotho/AudioCaps), video (VATEX/MSR‑VTT/WebVid), 3D (glTF samples and TRELLIS/Hunyuan adapters).

Process
-------
1) Filter per‑modality CSV+meta by topic keywords (see `knowledge3d/tools/filter_modal_csv.py`).
2) Cap per‑topic counts (e.g., 50 each) and merge per‑modality CSV+meta.
3) Build per‑modality GLBs (`knowledge3d.tools.trellis_adapter to-k3d`), allowing small k with fallback neighbors for tiny sets.
4) Unify GLBs to a single Galaxy with a shared embedding space and add cross‑modal edges.
5) Generate grounded RLWHF data with exaone (or other local models) that enforces honesty + error‑feedback.
6) Merge RLWHF datasets with de‑dup by `query`, retrain small policy, evaluate, and log results.

Degradation Rules
-----------------
- If topic X lacks enough examples in modality Y, log counts and continue; prefer dropping to a smaller balanced cardinality over skewing by a large margin.
- When further balancing is impossible due to public data limits, permit controlled imbalance but clearly annotate it in build notes.

