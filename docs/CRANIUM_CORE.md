K3D Cranium Core — Single Unified Head (All Modalities)
======================================================

Decision
--------
- Adopt a single in‑process, multi‑modal core head that runs entirely inside the Cranium runtime. No external wrappers or tools. All modalities are first‑class and memory‑native to K3D’s 256‑D Galaxy space.
- Navigation head is mandatory (not optional). TTS is also a first‑class head — not a python wrapper — producing in‑game voice directly from the model.

Why
---
- Standard AI stacks glue external LLMs and tools with siloed memories. K3D is different: one unified memory (Galaxy/House) + one model head that learns to read/write that memory.
- Low‑dimension, high‑density memory (256‑D) lets us keep the entire system small, fast, and auditable.

Architecture (v1)
-----------------
Core Head
- Small transformer with cross‑attention to K3D memory tokens at every layer.
- Inputs:
  - Text tokens (user query, optional system prompt)
  - Memory tokens (retrieved neighbors in 256‑D)
  - Modal stems (added over time):
    - Image (tiny ViT patch embed)
    - Audio (mel‑spec + CNN stem)
    - Video (frame‑level patch stem + temporal pool)
    - 3D (PointNet‑lite over sampled vectors from extras.k3d.vectors)
  - All stems project into the 256‑D space for alignment.
- Heads:
  - Text head (answers)
  - Navigation head (predict next node(s), citations) — REQUIRED
  - TTS head (waveform from latent acoustic tokens) — FIRST-CLASS

### Latest Implementation Notes (2025-09-25)

- NVRTC/PTX path now guards kernel launches with synchronization and shared context locks so geometry generation faults surface immediately instead of crashing the driver.
- The fused head consumes fused embeddings directly for House and Learning lookups, materializes new shapes into the House manifest, and records every generation through the tablet/learning memory contract.
- Media retrieval spans all modalities: image/audio/video assets are indexed from House metadata and materialized inventories, embedded (OpenCLIP/torchaudio/image histograms) on demand, and logged back into learning memory for later sessions.
- A lightweight video embedding pipeline samples frames and either runs OpenCLIP or falls back to colour histograms, keeping cross-modal retrieval self-contained.

Memory Adapter (GPU)
- Retrieval/top‑K over 256‑D memory. v1 uses RAPIDS cuML; v2 adds a custom CUDA/Triton/PTX kernel for batched dot/L2 + top‑K.

Training Objectives
- Reward‑weighted SFT (existing RLWHF) on the text head.
- Contrastive alignment: text/image/audio/video/3D ↔ memory tokens.
- Navigation supervision: predict labels used + next nodes.
- TTS alignment: latent acoustic tokens and waveform loss (for voice output).

Phases
------
Phase A (Now): Navigation + Text
- Implement the core head with text input and memory cross‑attention.
- Add the navigation head with citation/next‑node prediction.
- Replace the existing compose_generate path with the core head (feature flag `K3D_CORE_HEAD=1`).

Phase B: Add Modal Stems
- Image/audio/3D stems + contrastive alignment to 256‑D; video via frame sampling.

Phase C: GPU KNN Kernel
- Custom top‑K kernel (CUDA/Triton/PTX) to remove FAISS dependency while staying GPU‑native.

Phase D: First‑Class TTS
- Integrate an in‑process neural TTS head (Index‑TTS class models as reference). No external wrapper. The head generates waveform for in‑space voice and diary “voice notes”.

Acceptance Criteria
- All heads and stems are loaded and run inside the Cranium; no subprocesses or network calls.
- Retrieval is GPU‑native and in‑process.
- The model learns to navigate and answer grounded in the Galaxy; TTS produces playable audio inside the game.
