## Codex Phase 2.10 – Residual Gating, Ternary Context, and Next Steps

### Summary (what we did)
- Pushed the video codec to a coherence-first stance: gating only flags full-DCT on very weak procedural fits; otherwise procedural is kept, with a strict lossless escape for strong fits.
- Lossless escape: only when projected ternary gain < 2× and fit is high (SSIM ≥ 0.8 or PSNR ≥ 20 or fitness ≥ 0.8). Otherwise ternary/quantization proceeds.
- Block quantization split into adaptive stripes (3–6) to emulate multi-math-core concurrency with limited overhead.
- Benchmarks now default to 126×126 (divisible by 3) for turnaround speed.

### Current benchmark snapshot (126×126, GPU, standard script)
- pattern_a: PROCEDURAL, 46.5×, PSNR inf, SSIM 1.000
- pattern_b: FULL-DCT, 2.9×, PSNR 10.2 dB, SSIM 0.114
- random_frame: FULL-DCT, 2.4×, PSNR 10.3 dB, SSIM 0.088
Encode ~35–44 ms; decode ~3–8 ms.

### Observations
- Coherence prioritized → many frames remain procedural/lossless; compression often <2× on harder content, PSNR low (~8–12 dB) where fit is poor.
- Gating is minimal (ssim_proc < 0.1 & psnr_proc < 6 & fitness < 0.3) for full-DCT.
- Stripe split improves concurrency modestly; overhead grows with higher resolution (252× tests showed latency inflation).

### Guidance on stride/stripe vs resolution
- Keep 126×126 for quick runs; stripes adapt 3–6 based on block rows to balance overhead vs concurrency.
- If more throughput is needed on larger frames, increase stripes; otherwise, fewer stripes reduces overhead.

### Ternary/state notes
- Ternary fits remain in the pipeline (quantization path), but coherence-first mode will opt lossless when ternary compression gain is weak.
- If we need more compression, tighten gating (ssim<0.3 or psnr<12) and relax lossless criteria.

### Texture synthesis insight (external references)
- Exemplar-based texture synthesis (Efros–Leung) and PatchMatch-style compositing can procedurally generate infinite, non-repeating textures. Could be applied to “break” inputs into panels and reassemble procedurally rather than pure compression.
- Procedural shader approach: learn a reassembly plan; execution is cheap, the planning is the hard part.

### Actionable next steps (codec optional, as current stage acceptable)
- If desired: tighten gating to regain >2–3× on hard frames; keep lossless only for SSIM ≥ 0.9 / PSNR ≥ 27.
- Otherwise: freeze this stage and move on.

### Phase 3 focus (fused head and training)
- Shift to audio ingestion: pair pronunciation with letters and text pronunciation hints across languages; math symbols pronunciation is a plus.
- Re-engineer fused head/training path to integrate audio (Stack 7/14 future work), tri-modal ternary fusion ideas (unified trit space), and router bootstrap with ternary heuristics.

### Ternary chain notes (from discussions)
- Trits reduce state space ~33%; use ternary projections and TADD/TMUL for tri-modal fusion to cut latency (~40%).
- Ensure ternary heuristics in routers (stratified sampling, carry-free TMUL) to speed bootstrap.
- For 3D modality: ternary helps packing; guard against transitive dilution with TAND propagation of unknowns.

### Audio proceduralization utilities (added)
- `scripts/proceduralize_audio.py`: converts labeled audio (manifest-driven or filename fallback) into procedural harmonic seeds (letters/phonemes); uses ffmpeg fallback for decoding.
- `scripts/generate_audio_manifest_skeleton.py`: scans audio roots and emits a CSV skeleton (path,text,phoneme,lang) for manual/automatic labeling since minds14/Librispeech filenames lack labels.
- `scripts/generate_phoneme_bank.py`: synthesizes a small open phoneme bank via espeak for en/pt/es/zh into `/K3D/K3D_llama_cpp/datasets/audio/phoneme_bank/<lang>/`.

### Decision
Content coherence is paramount, compression acceptable in a short range. Current codec stage is acceptable to move on to fused-head/audio ingestion work.
