---
Author: Claude (architecture partner)
Date: 2026-04-20
Branch: codex/batch11-knowledge-waves-observability-game2d-2026-04-15
Scope: Procedural audio + video codec sovereignty audit, with actionable spec for P6 (House permanence + 3D crafting lane)
Supersedes-notes: extends CODEC_SOVEREIGNTY_COMPLETE_11.27.2025.md (audio) and CLAUDE_ARC_CODEC_INTEGRATION_11.24.2025.md (ARC3 frames)
---

# Codec Sovereignty Audit — April 2026

## 1. Executive Verdict

| Lane | Status | Gate |
|---|---|---|
| **Audio** | ✅ Sovereign (hot path clean) | Promote two Stage 0 opcodes to keep it that way |
| **Video / Frame** | ⚠️ Partial (encoder sovereign, codec incomplete) | Ship Frame→RPN bridge + sprite + temporal kernels |

Audio survived the 2025-11 sovereignty pass intact. Video is where the real work is.

### 2026-04-20 landing update

Parallel-lane partners (3× sub-agents, 1× ollama kimi_swarm) delivered drafts
within a 90-min window; Claude synthesized with the ternary-first retarget
(BitNet b1.58 5-trits/byte, zero-trit skip) per Daniel's 2026-04-20 directive.

Landed:

- **Opcodes minted**: 0x217-0x21F (DotMap), 0x240-0x24F (JPEG line-scan),
  0x250-0x25F (Audio FFT), 0x260-0x26F (Frame codec), 0x270-0x27F (Projection).
  See [rpn_opcodes.py](../knowledge3d/cranium/ptx_runtime/rpn_opcodes.py).
- **Kernel files**: [dotmap_codec.cu](../knowledge3d/cranium/codecs/kernels/dotmap_codec.cu),
  [idct_8x8.cu](../knowledge3d/cranium/codecs/kernels/idct_8x8.cu),
  [jpeg_scan.cu](../knowledge3d/cranium/codecs/kernels/jpeg_scan.cu),
  [huff_decode.cu](../knowledge3d/cranium/codecs/kernels/huff_decode.cu),
  [frame_codec.cu](../knowledge3d/cranium/codecs/kernels/frame_codec.cu),
  [audio_fft.cu](../knowledge3d/cranium/codecs/kernels/audio_fft.cu) (sub-agent),
  [projection_screen.cu](../knowledge3d/cranium/codecs/kernels/projection_screen.cu).
- **Dispatch**: 72 new tokens registered in `ModularRPNEngine.OPCODES` (277 total).
- **Registry**: §11 reservation rows landed for all 5 ranges.

Remaining for Codex (implementation lane):

- PTX compile + nvcc -arch=sm_86 for each .cu file; wire into `loader.launch()`.
- Round-trip tests per audit §5 (64×64 DotMap byte-equal, 256×256 JPEG ≤ 1-step delta, FFT parity 1e-5 vs scipy).
- Retrofit `arc3_frame_encoder.cu` to dual-emit (keep 64-D embedding, add DotMap program).
- Author Python ring-buffer + PTS state machine for projection playback (companion to projection_screen.cu).

---

## 2. Audio — Findings

### 2.1 What is in place

- **Galaxy**: `/K3D/Knowledge3D.local/galaxies/Audio.jsonl` — 3,144 rows, all procedural RPN (e.g. `32.703196 F_STORE T F MUL TWO_PI MUL SIN` for MIDI-24 sine). Zero raw WAV samples.
- **PTX kernels**: [codec_ops.ptx](../knowledge3d/cranium/ptx/codec_ops.ptx) (`ternary_quant`, `ternary_dequant`, `mdct_forward`, `imdct_inverse`, `batch_mdct`). Archived CUDA source: [ternary_mdct.cu](../knowledge3d/cranium/codecs/kernels/ternary_mdct.cu).
- **ctypes bridge**: [ternary_codec_ops.py](../knowledge3d/cranium/codecs/ternary_codec_ops.py) — no numpy / cupy / scipy.
- **Ingestion isolation**: `librosa`, `ffmpeg`, `wave` all confined to [knowledge3d/ingestion/language/](../knowledge3d/ingestion/language/) and [knowledge3d/tools/audio_pipeline.py](../knowledge3d/tools/audio_pipeline.py). `knowledge3d/skills/audio.py` uses `torchaudio` only as an optional embedding skill with a deterministic hash fallback.

### 2.2 Gaps

- **Stage 0 opcodes not promoted**: `OP_FFT_FORWARD`, `OP_FFT_INVERSE`, `OP_AUDIO_TO_SPECTROGRAM` in [RPN_DOMAIN_OPCODE_REGISTRY.md](../docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md) §6.5 are still candidates. The codec today composes `OP_MATVEC_F32` + `OP_BATCH_*` + `OP_REDUCE_*` for spectrograms. Works, but a TRM craft-lane that generates audio papers or spectrogram demos will want dedicated opcodes.
- **No audio opcode range reservation**: §11 of the registry has not been checked-in for audio. Reserve before dispatching kernel work (see opcode reservation protocol memory).

### 2.3 Spec — Audio Codec Promotion (P6-adjacent)

**Reserve first** in `RPN_DOMAIN_OPCODE_REGISTRY.md §11`:

```
0xA100-0xA10F  AUDIO_FFT_*           (forward / inverse / window / bit-reverse)
0xA110-0xA11F  AUDIO_SPECTROGRAM_*   (mel, linear, log, framewise)
0xA120-0xA12F  AUDIO_ENVELOPE_*      (ADSR, LFO, AM, FM)
0xA130-0xA13F  AUDIO_MIX_*           (bus, gain, pan, crossfade)
```

**Then implement** (Codex):
1. `audio_fft.cu` (radix-2 Stockham, 256/512/1024/2048 sizes).
2. `audio_spectrogram.cu` wrapping FFT + mel-filter bank. Reuse `batch_mdct` for MDCT variants.
3. Wire opcodes in [knowledge3d/cranium/ptx_runtime/rpn_opcodes.py](../knowledge3d/cranium/ptx_runtime/rpn_opcodes.py).
4. Add tests in [knowledge3d/cranium/tests/test_sovereign_ternary_audio_codec.py](../knowledge3d/cranium/tests/test_sovereign_ternary_audio_codec.py) for the new opcodes (determinism + parity vs scipy ground truth on a numpy-using *test* harness).

**Success gate**: one TRM-navigated spectrogram query resolves end-to-end with `KNOWLEDGEVERSE.rpn_exec` only, zero numpy in the dispatch trace.

---

## 3. Video / Frame — Findings

### 3.1 What is in place

- **Sovereign encoder**: [arc3_frame_encoder.cu](../knowledge3d/cranium/cuda/arc3_frame_encoder.cu) → 64-D embedding (color-mass / moments / occupancy). Launched from [arc3_frame_encoder.py](../knowledge3d/knowledgeverse/arc3_frame_encoder.py) via ctypes; no cv2/PIL/numpy in hot path.
- **Sovereign mask opcode**: [rpn_frame.cu](../knowledge3d/cranium/cuda/rpn_frame.cu) — `op_frame` + `op_biduce` for preserved/missing/incompatible mask inference.
- **Sovereign galaxies**: [Drawing.jsonl](../data/runtime_execution_journal_batch/galaxies/Drawing.jsonl) and 3DObjects.jsonl are 100 % RPN — no pixel arrays anywhere. Primitives match `PROCEDURAL_VISUAL_SPECIFICATION.md` (MOVE, LINE, QUAD, CUBIC, ARC, FILL, STROKE …).
- **WINE transport**: [game2d_wine.py](../knowledge3d/tablet/wine/game2d_wine.py) and [arc3_wine.py](../knowledge3d/tablet/wine/arc3_wine.py) pass frames as opaque `Any` through to the Tablet — no pixel ops in the envelope.

### 3.2 Gaps (the whole P6 reason)

**Gap V-1 — Frame → RPN drawing bridge is missing.**
The current ARC3 path is *analysis* (frame → 64-D neural embedding). For the dual-client contract, the Tablet must also emit a *procedural draw program* that reconstructs the frame — `CELL x y COLOR_i RECT_FILL` sequences for ARC grids, vector-trace programs for the LHE/Drawing curriculum. Without this, an AI that "watches" a frame cannot *craft* the frame back as a book illustration or a paper figure.

**Gap V-2 — Sprite / tile composition kernel is missing.**
GAME_2D needs `SPRITE_DRAW(id, x, y, palette)` and `SPRITE_BATCH_COMPOSE(frame_id, sprite_list)`. Today every GAME_2D frame re-encodes a full grid; with a sprite kernel, a frame = `(background_rpn, diff_list)` which is what multi-frame temporal RPN will need.

**Gap V-3 — Temporal frame-delta codec is missing.**
No kernel emits `FRAME_DELTA(prev_frame_id, diff_ops)` or `MOTION_VECTOR(dx, dy, region)`. A movie = a list of draw programs composed with deltas. Today a video is just N independent encoded frames — which is duplicative storage and blocks any "how does this scene evolve" reasoning.

**Gap V-4 — Morton-encoded pixel indexing is missing for dense frames.**
LHE and Drawing both have a Morton octree for 3-D; 2-D frame pixels use linear addressing. A 2-D Morton kernel would make diff-region queries hit locality and close the symmetry with the octree pipeline.

**Gap V-5 — [skills/video.py](../knowledge3d/skills/video.py) drift.**
Imports `imageio`, `PIL`, `numpy`, `torch`, `open_clip`. It is ingestion-only today, but the functionality it provides (frame sampling + embedding) overlaps with `arc3_frame_encoder.cu`. Once the sovereign encoder covers generic video, this skill should be demoted to ingestion-only helper or removed.

### 3.3 Spec — Video / Frame Codec Completion (P6 core)

**Reserve** in `RPN_DOMAIN_OPCODE_REGISTRY.md §11` (opcode reservation protocol — pre-register before lanes fork):

```
0xD200-0xD20F  FRAME_DRAW_*          (RECT, LINE, CIRCLE_FILL, PALETTE_SET, CELL_FILL)
0xD210-0xD21F  FRAME_SPRITE_*        (LOAD, DRAW, BATCH_COMPOSE, PALETTE_BIND)
0xD220-0xD22F  FRAME_DELTA_*         (DIFF, REGION_DIFF, COMPOSE, MOTION_VECTOR)
0xD230-0xD23F  FRAME_MORTON_*        (ENCODE_2D, DECODE_2D, RANGE_QUERY)
0xD240-0xD24F  FRAME_ANALYZE_*       (ENCODE_64D — wraps existing arc3_frame_encoder)
```

These sit under the CRAFT super-range (0xD000-0xD3FF tentative) the kimi_swarm returned for P6.

**Then implement** (Codex, in order):

1. **Frame→RPN bridge** ([knowledge3d/cranium/cuda/frame_to_rpn.cu](../knowledge3d/cranium/cuda/frame_to_rpn.cu), new):
   - Input: `int grid[H][W]` (ARC3-style palette indices).
   - Output: compact RPN program stream emitting `CELL_FILL` runs (row-wise run-length with palette headers).
   - Must be deterministic (same grid → byte-identical program).
   - Ingestion helper lives in [knowledge3d/ingestion/frames/](../knowledge3d/ingestion/) (new dir) — OK to use Python there; hot-path reconstruction runs the emitted program through `rpn_exec`.

2. **Sprite kernel** (`sprite_compose.cu`): atlas-backed sprite draws with palette remapping; target ≥ 10 k sprite draws per CognitionTick on the 3070.

3. **Frame delta kernel** (`frame_delta.cu`): `prev_frame_rpn ⊕ curr_frame_rpn → delta_rpn`; region-based (64×64 tiles) for locality.

4. **Morton 2-D** (`morton_2d.cu`): standard Z-order on `(x, y)` with 16-bit deinterleave; integrate with the existing frustum kernel for 2-D ROI culling.

5. **ARC3 integration**: update [arc3_frame_encoder.py](../knowledge3d/knowledgeverse/arc3_frame_encoder.py) to *also* emit the Frame→RPN program alongside the 64-D embedding — so each frame lands in the Galaxy as a draw program (procedural truth) + embedding (routing shortcut).

6. **skills/video.py demotion**: move to [knowledge3d/ingestion/video/](../knowledge3d/ingestion/) and strip any hot-path callers. Hot-path video = sovereign encoder + Frame→RPN bridge + sprite/delta composition.

**Success gate** (for the live daemon):
- ARC3 game frame ingested → Galaxy row has **both** `rpn_program` (draw ops) and `embedding` (64-D).
- Reconstruct test: `rpn_exec(row.rpn_program)` produces a grid byte-equal to the source (no numpy).
- Multi-frame test: two consecutive frames + `FRAME_DELTA` compose to the second — verified on GPU, no host reconstruction.
- Sovereignty grep: zero `import cv2|PIL|imageio|av|pyav` in anything under `knowledge3d/cranium/` or `knowledge3d/knowledgeverse/`.

---

## 4. Dual-Client Contract Check

The contract (DUAL_CLIENT_CONTRACT_SPECIFICATION.md) requires: humans see rendered output, AI reads RPN — and those two paths must be derivable from the **same** procedural source.

- Audio today: ✅ both clients. Humans get audio via iMDCT → waveform; AI reads the RPN oscillator program.
- Video today: ⚠️ only analysis is sovereign. The human side works because WINE passes the frame to the viewer as-is; the AI side stops at a 64-D embedding — it cannot *regenerate* or *author* frames. Gap V-1 above is the contract violation.

P6 closes the contract for video.

---

## 5. Interaction with Active Priority Stack

- **P0 (depythonize knowledgeverse GEMV)**: unchanged — independent.
- **P1 (embodiment MVP — perceive / act / symlinks)**: the perceive kernel can consume the new `FRAME_ANALYZE_ENCODE_64D` opcode; lock the opcode range now so the two lanes don't collide.
- **P2 (pipeline → live hydration)**: once Frame→RPN lands, ARC3 games proceduralized in past batches need a migration pass to add `rpn_program` to their Galaxy rows.
- **P5 (adaptive tick)**: frame-delta composition fits naturally into the WorldTick budget — measure per-frame cost before deciding tick credit.
- **P6 (House permanence + 3D crafting)**: this spec **is** the video-codec half of P6. CRAFT opcodes and FRAME opcodes live in adjacent ranges on purpose — TRM needs both to draw a figure in a paper.

---

## 6. Work Items for Codex (when unlocked)

In strict order — each item has clear DoD so there is no stub risk:

| # | Item | File(s) | DoD |
|---|---|---|---|
| 1 | Reserve opcode ranges | `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md §11` | PR lands with ranges 0xA100-0xA13F + 0xD200-0xD24F marked `RESERVED — P6 codec` |
| 2 | `audio_fft.cu` + spectrogram wrapper | `knowledge3d/cranium/cuda/` | unit test: FFT-256 ≤ 1e-4 vs scipy on random input; zero numpy in kernel launcher |
| 3 | Wire audio opcodes | `rpn_opcodes.py` | TRM probe resolves `mel_spectrogram(audio_wave_midi_24)` via RPN only |
| 4 | `frame_to_rpn.cu` + ctypes | `knowledge3d/cranium/cuda/` | ARC3 grid round-trip: grid → RPN → grid is byte-equal |
| 5 | Sprite + delta + Morton-2D | `knowledge3d/cranium/cuda/` | two-frame composition test passes on GPU, no host reconstruction |
| 6 | ARC3 encoder dual-emit | `knowledge3d/knowledgeverse/arc3_frame_encoder.py` | galaxy row contains `rpn_program` + `embedding` |
| 7 | Demote `skills/video.py` | move to `knowledge3d/ingestion/video/` | grep `knowledge3d/cranium|knowledge3d/knowledgeverse` for `cv2|PIL|imageio` returns empty |

No stubs. Each kernel must pass determinism + parity tests. If a kernel is not ready, do not merge; do not emit a `TODO` placeholder — block the lane instead. (Rule: "we fail and fix.")

---

## 7. Open Questions (for Daniel)

1. **CRAFT super-range**: kimi_swarm proposed 0xD000-0xD3FF. I split it — D000-D1FF for structural CRAFT opcodes (papers/books/shelves), D200-D24F for frame codec. OK, or should frame codecs live under a separate F-prefix?
2. **Video skills demotion**: is the open_clip-backed video embedding still needed for external-query routing, or is the 64-D sovereign encoder enough once Frame→RPN lands?
3. **Audio paper crafting**: should TRM be able to emit audio *papers* (spectrogram figures + captions as procedural drawings), or is audio only a leaf modality for now?

Answers from Daniel drive ordering of audio-opcode work vs. frame-codec work.
