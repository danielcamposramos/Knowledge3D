# GPT‑OSS Style Datasets (Open, Replicable)

Intent
- Replicate the kinds of public data used by open LLMs (e.g., Dolma/OLMo, RedPajama, SlimPajama, FineWeb) so our Galaxy contains rich, diverse text akin to GPT‑OSS development corpora — without touching closed/proprietary sources.

Categories (examples; all open‑licensed, but verify per source)
- Web crawl (cleaned):
  - FineWeb (LAION) — filtered CommonCrawl
  - SlimPajama — cleaned RefinedWeb
  - RedPajama – CommonCrawl component
- Wikipedia:
  - enwiki snapshots via `wikipedia` HF dataset
- QA/forums:
  - StackExchange dumps (CC‑BY‑SA 4.0)
  - OpenAssistant/OASST1 (for dialogue style, not base pretraining)
- Scientific
  - arXiv abstracts (via arxiv or SemanticScholar subsets; check license)
  - PubMed abstracts (non‑commercial restrictions in some subsets)
- Books/public domain
  - Project Gutenberg (PG‑19 subset), BookCorpusOpen
- News
  - RealNews‑like open alternatives (check HF datasets)
- Code (optional, as text)
  - The Stack v2 (BigCode)

K3D Integration
- Fetch text with `knowledge3d.tools.hf_fetch_text` or local dumps → `*.txt` (one line per entry)
- Embed on GPU in Galaxy build:
  - `scripts/k3d_env.sh run python -m knowledge3d.tools.build_galaxy ... textlines:../Knowledge3D.local/datasets/<name>.txt ... --reducer umap`

Notes
- Licenses: Keep a manifest (dataset, version/date, license, URL) alongside the text file.
- Storage: place raw under `/K3D/K3D_llama_cpp/datasets` when large; symlink curated subsets into `../Knowledge3D.local/datasets`.
- Multimodal: continue mixing image (COCO/WIT), audio (Clotho/AudioCaps), video (VATEX/MSR‑VTT) as you’ve already done.

References
- Dolma (OLMo): https://huggingface.co/datasets/allenai/dolma
- RedPajama: https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T
- SlimPajama: https://huggingface.co/datasets/cerebras/SlimPajama-627B
- FineWeb: https://huggingface.co/datasets/GAIA-SDC/FineWeb
