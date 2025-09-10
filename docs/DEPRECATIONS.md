Deprecated (Legacy) Patterns
============================

The following approaches are no longer valid for K3D core runs:

1) External LLM wrappers as primary generation paths
- Why: Contradicts the single‑interpreter design. The Cranium must run one in‑process core head that conditions on K3D memory.
- Replacement: Cranium Core Head (docs/CRANIUM_CORE.md) with text + navigation (Phase A), then multi‑modal stems (Phase B), and first‑class TTS (Phase D).

2) TTS as a python wrapper/tool
- Why: TTS must be first‑class; the Cranium head should generate speech directly (no subprocess, no network). Diary “voice notes” are integral.
- Replacement: In‑process neural TTS head (Phase D).

3) CPU fallbacks for training/inference
- Why: Violates performance and design constraints; we require GPU‑native operation.
- Replacement: Strict GPU enforcement (`K3D_STRICT_GPU=1`), with CUDA/Triton/PTX paths where needed.

4) Over‑reliance on external high‑dimensional encoders
- Why: K3D uses a low‑dimensional, high‑density 256‑D space. Additional stems must project into this shared space, not introduce separate high‑D silos.
- Replacement: In‑repo stems mapping into 256‑D K3D space via contrastive alignment.

