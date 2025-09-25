# Hugging Face Cache Workflow

The TransformerLab desktop client downloads datasets into the standard Hugging Face
cache at `~/.cache/huggingface/datasets`. The Knowledge3D toolchain harvests those
already-downloaded assets so they appear inside the Galaxy/House memory stack
without duplicating the original downloads.

## Folder Layout

Each dataset is cached under `~/.cache/huggingface/datasets/<namespace>___<name>/`
with nested configuration → version → hash folders. The raw files (Arrow, Parquet,
media assets, `dataset_info.json`, etc.) live inside the hashed leaf directory.

```
~/.cache/huggingface/datasets/
└── knkarthick___samsum/
    └── default/
        └── 0.0.0/
            └── 6b929ff10edec703164e3ddb2e94aae058c9ab5f/
                ├── dataset_info.json
                ├── samsum-train.arrow
                └── ...
```

## Converting Cache Entries into Galaxy Corpora

Use `knowledge3d/tools/ingest_hf_cache.py` to sweep the cache and emit a
Knowledge3D-compatible corpus:

```bash
# Optional (for audio datasets): ensure TorchCodec is available inside k3d-cranium
conda run -n k3d-cranium python -m pip install torchcodec==0.6.0

conda run -n k3d-cranium PYTHONPATH=. \
  python knowledge3d/tools/ingest_hf_cache.py --max-per-split 200
```

This command writes two files under `viewer/public/galaxy/working/`:

* `hf_cache_corpus.jsonl` – prompt/answer records with provenance
* `hf_cache_manifest.json` – per-dataset statistics (splits and sample counts)

You can restrict the sweep to specific datasets with `--datasets namespace___name`.

## Building a Galaxy GLB

After generating the JSONL corpus, run the standard learning-memory builder to
create a PTX-ready GLB and manifest for the tablet:

```bash
conda run -n k3d-cranium PYTHONPATH=. \
  python knowledge3d/tools/learning_memory_builder.py \
    --input viewer/public/galaxy/working/hf_cache_corpus.jsonl \
    --out viewer/public/galaxy/hf_cache_memory.glb \
    --manifest viewer/public/galaxy/hf_cache_memory.json \
    --label "HF Cache Knowledge"
```

The fused head automatically loads `hf_cache_corpus.jsonl` (alongside the time,
math, and Wikipedia corpora), so summaries, honesty notes, and concept tags now
reflect the harvested open datasets.

## Notes

* The pipeline respects Hugging Face’s download cache—no extra network transfer.
* Audio datasets require the `torchcodec` dependency (we pin `torchcodec==0.6.0`
  for compatibility with PyTorch 2.5). Install it before harvesting speech corpora.
* Large GLBs stay well below the repository upload cap (current GLB ≈20 MB).
