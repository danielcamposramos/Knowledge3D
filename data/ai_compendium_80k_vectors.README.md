The file `data/ai_compendium_80k_vectors.csv` is intentionally not tracked (large >99MB).

Recreate locally
1) Build the 80k text corpus (or use your own): see `docs/LARGE_ASSETS.md` → “80k AI Compendium (local)”.
2) Convert text → vectors (CPU‑friendly):
   
   ```bash
   python3 -m knowledge3d.tools.text_to_vectors \
     --text data/ai_compendium_80k.txt \
     --out ../Knowledge3D.local/datasets/ai_compendium_80k_vectors.csv \
     --dims 512
   ```

3) Optional: Generate a GLB for the viewer directly from the vectors:
   
   ```bash
   python3 -m k3dgen ../Knowledge3D.local/datasets/ai_compendium_80k_vectors.csv \
     --gltf viewer/public/ai_compendium.80k.pca.glb --k 5 --reducer pca --emb-precision f16
   ```

Storage policy
- Keep large datasets under `../Knowledge3D.local/datasets/` and never commit them.

