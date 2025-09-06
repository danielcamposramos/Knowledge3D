# Large Assets and Replication

Some generated assets exceed GitHub’s size limits. We keep heavy datasets in a sibling local folder and document exact steps to reproduce.

Local Folders (not in repo)
- `../Knowledge3D.local/logs/` — live-mode logs
- `../Knowledge3D.local/models/` — trained models (e.g., intent.pkl)
- `../Knowledge3D.local/datasets/` — large datasets and exports

## 80k AI Compendium (local)
1) Build corpus (80k lines, deduped across repo docs, repos, wiki):
```bash
python3 -m knowledge3d.tools.build_repo_corpus --max-lines 60000
python3 -m knowledge3d.tools.fetch_wiki_corpus --default-topics --max-lines 30000 --min-len 80
python3 - <<'PY'
# Merge + dedupe to 80k: writes data/ai_compendium_80k.txt
from pathlib import Path
import re
seen=set(); out=[]
for p in ['data/ai_compendium_full.txt','data/ai_repos_corpus.txt','data/ai_wiki_corpus.txt','data/ai_compendium.txt']:
  for ln in Path(p).read_text(encoding='utf-8').splitlines():
    s=ln.strip();
    if s and s not in seen: seen.add(s); out.append(s)
if len(out)<80000:
  import itertools
  for ln in itertools.islice(out, 80000-len(out)): out.append(ln)
Path('data/ai_compendium_80k.txt').write_text('\n'.join(out[:80000])+'\n',encoding='utf-8')
print('Wrote',len(out[:80000]))
PY
```
2) Vectorize (512d hashed, L2):
```bash
python3 -m knowledge3d.tools.text_to_vectors \
  --text data/ai_compendium_80k.txt \
  --out ../Knowledge3D.local/datasets/ai_compendium_80k_vectors.csv \
  --dims 512
```
3) Generate GLBs (PCA→3D, f16 embeddings):
```bash
python3 -m k3dgen ../Knowledge3D.local/datasets/ai_compendium_80k_vectors.csv \
  --gltf ../Knowledge3D.local/datasets/ai_compendium.80k.pca.glb --k 5 --reducer pca --emb-precision f16
python3 -m knowledge3d.tools.mark_doors \
  --input ../Knowledge3D.local/datasets/ai_compendium.80k.pca.glb \
  --output ../Knowledge3D.local/datasets/ai_compendium.80k.pca.doors.glb \
  --doors 1280 --trail true
```
4) View locally by adding entries to `viewer/public/condo.json` pointing to local server paths.

## Knowledge Gardens
- Build lightweight ontology demo: `python3 -m knowledge3d.tools.gardens --gltf viewer/public/knowledge_garden.glb`

## 500k / 1M Compendium (GPU)

1) Build mega text (dedupe + fill to target):
```bash
python -m knowledge3d.tools.build_mega_corpus --target 1000000 --out data/ai_compendium_1m.txt
```
2) Vectorize to CSV (CPU HashingVectorizer is fine):
```bash
python -m knowledge3d.tools.text_to_vectors \
  --text data/ai_compendium_1m.txt \
  --out ../Knowledge3D.local/datasets/ai_compendium_1m_vectors.csv \
  --dims 512
```
3) Build GPU GLBs (IVF‑PQ):
```bash
conda run -n k3d-rapids \
  env K3D_ACCEL=gpu K3D_FAISS_DEVICE=gpu K3D_ACCEL_LOG=1 \
      K3D_FAISS_PQ_M=16 K3D_FAISS_PQ_BITS=8 \
  python -m k3dgen ../Knowledge3D.local/datasets/ai_compendium_1m_vectors.csv \
    --gltf ../Knowledge3D.local/datasets/ai_compendium.1m.umap.ivfpq.glb \
    --k 10 --reducer umap --ann ivfpq --emb-precision f16
```
Sharding option (when memory is tight):
```bash
# Split your vectors CSV externally (e.g., by head/tail) and build multiple 500k GLBs
```

## Tablet Exams (ARC‑AGI + HLE)
- Use `viewer/public/training/` samples as a template to mount real datasets from local disks (keep large sets in `../Knowledge3D.local/datasets`).
- Fetch sources locally:
  - ARC tasks: `git clone https://github.com/fchollet/ARC.git ../Knowledge3D.local/datasets/exams/arc-src`
- HLE sample export: `python3 -m knowledge3d.tools.export_hle_sample --count 50` (requires `pip install datasets`)
- Note: HLE on Hugging Face is gated; run `huggingface-cli login` and accept access at https://huggingface.co/datasets/cais/hle before exporting.
- Build index combining ARC + HLE: `python3 -m knowledge3d.tools.build_exams_index --max-arc 200`
- Serve local datasets with CORS: `python3 -m knowledge3d.tools.serve_datasets --port 8766`
- In the viewer, the Tablet Exams app will attempt to load `http://127.0.0.1:8766/exams_index.json` automatically.

## Verification
- We publish checksums for public artifacts under `docs/reports/`.
