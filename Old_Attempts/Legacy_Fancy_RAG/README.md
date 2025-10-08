# Legacy Fancy RAG Manifests

The pre-PTX workflow produced many assets (GLBs, sidecar `.k3d`, training logs, viewer exports). They are now stored in the local workspace:

```
../Knowledge3D.local/old_attempts/legacy_fancy_rag/
```

This directory only lists what existed, so we can audit or migrate code without bloating the repo. Pair these manifests with the reproduction recipes under `Large_Assets_Kitchen/`.

## Manifests
- `manifests/examples.txt` — legacy demo houses, CSVs, and generators.
- `manifests/viewer_public.txt` — viewer exports (houses, galaxy snapshots, workshops, etc.).
- `manifests/docs_reports.txt` — benchmark summaries, live logs, and training transcripts.
- `manifests/binaries.txt` — helper binaries (`book_mesh.bin`, `embedding.npy`, etc.) removed from version control.

Feel free to append notes when you migrate a legacy artifact into the PTX pipeline. Otherwise keep this directory read-only.
