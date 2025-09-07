Multimodal “Baby” Session (100k) — Plan

Principle
- Small, specialized brain regions; memory stored in the House (glTF/GLB + extras.k3d). Use open datasets and existing strong encoders (CLIP/CLAP) for perception; keep logic simple and trainable (intent classifier, RSSM dynamics, RPN policies).

Modalities (seed encoders)
- Text: HF encoder for intent training; K3D vectors can be HashingVectorizer for fast protos.
- Image: OpenCLIP (GPU) → embeddings; thumbnails for tooltips.
- Audio: LAION‑CLAP (GPU) → embeddings.
- Video: frame sampling via PyAV, OpenCLIP per‑frame then aggregated.

Pipeline (one session)
1) Ingest (100k total lines/records across modalities)
   - Text: WIT captions → `wit.sample.txt`
   - Image: WIT images + OpenCLIP → `*.clip.csv` + `*.meta.json`
   - Video: `ingest_video.py` → `video.sample.clip.csv` + thumbs + meta
   - Audio: `ingest_audio.py` → `audio.sample.clap.csv` + meta
2) Build GLBs
   - Start with PCA positions (fast) → GLBs for viewer validation.
   - Then UMAP (GPU cuML) for near LOD fidelity.
3) Viewer & Logs
   - Add GLBs to condo; run viewer.
   - Generate logs via `knowledge3d.tools.multi_instance` (500–1000 messages).
4) Train logic
   - Intent (HF): `knowledge3d.models.intent_hf train ...`
   - World dynamics (RSSM): `knowledge3d.models.world_model.train ...`
   - Keep RPN for precise internal inference & control.
5) Sleep‑Compute
   - Consolidate: save models + rotate logs; restart live server.
6) Validate
   - Intent: held‑out accuracy.
   - RSSM: next‑step 3D MSE; sanity rollouts.

Iteration
- Add HunyuanWorld scenes as house rooms via the adapter; repeat logging + training with richer data.

