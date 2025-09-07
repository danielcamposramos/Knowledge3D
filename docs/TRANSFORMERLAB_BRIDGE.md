# TransformerLab Bridge for K3D

This document outlines how to connect TransformerLab’s training/evaluation stack to K3D’s memory‑first, multimodal pipeline.

Objectives
- Drive K3D ingestion/builds from a UI similar to TransformerLab’s dataset/model panels.
- Train small adapters/LoRAs for K3D skills (intent classification, policy heads, composition) using TransformerLab’s finetune harnesses.
- Visualize & select training data from K3D logs; ship results back as lightweight adapters.

Components
- Dataset Orchestrator (K3D)
  - Fetchers: `knowledge3d/tools/hf_fetch_multimodal.py`, `hf_export_urls.py`, `yt_batch.py`
  - Ingestors: `ingest_wit.py`, `ingest_audio.py`, `ingest_video.py`
  - Builder: `build_multimodal_50k.py` + `k3dgen`
- Adapter Trainers (TransformerLab)
  - LoRA/QLoRA for HF models; MLX for Apple, vLLM/Llama.cpp for inference
  - Evaluation panels and dataset selection UI
- Bridge Service (K3D stub)
  - `knowledge3d/tools/tlab_bridge.py`: simple CLI hooks to launch K3D fetch/build and to summarize logs as JSON

Flows
- Curate multimodal datasets (50k target each) → Build GLBs → Launch adapter training from selected K3D logs
- Return adapters to House via a door/app, and activate in Cranium (threshold‑gated)

Next Steps
- Add a small HTTP layer to `tlab_bridge.py` (FastAPI) to provide endpoints TransformerLab can call.
- Map K3D replay logs (../Knowledge3D.local/logs) into TransformerLab’s dataset viewer.
