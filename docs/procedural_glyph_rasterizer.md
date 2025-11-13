# Procedural Glyph Rasterizer – Sovereign Design Brief

**Objective**: replace the CPU/PIL glyph generator used by `train_atomic_character.py` with a GPU-resident procedural rasterizer that treats font programs as the ground truth. The kernel streams Bézier data (procedural instructions) directly into the GPU, produces glyph tensors on demand, and hands them off to `GPUCNNTrainer` without staging ~0.6 GB numpy arrays per character.

## Motivations
- **Philosophy alignment**: Font files are already compact procedural programs (~50 KB). Rasterizing them into static numpy blobs (~589 MB) violates the “store how-to-reconstruct, not raw data” mandate.
- **Resource utilization**: RTX 3060 (12 GB VRAM) sits idle while host RAM explodes under 15 parallel CPU jobs. Rendering on-device eliminates the host bottleneck and keeps kernels busy.
- **Pipeline consistency**: New procedural kernels mirror the Matryoshka/TRM/Sovereign stack. Python remains a thin I/O wrapper; all heavy lifting happens inside PTX.

## Kernel Specification: `procedural_glyph_rasterizer.ptx`
| Component | Description |
| --- | --- |
| **Inputs** | Packed control points (quadratic/cubic Béziers), per-contour metadata (winding, on/off-curve flags), glyph metrics, augmentation seeds. Multiple glyph descriptors stream in one batch. |
| **Execution** | 1. Load descriptor → evaluate curves using de Casteljau or implicit SDF on shared memory tiles.<br>2. Apply oversampling + coverage accumulation to produce anti-aliased masks.<br>3. Inline augmentation (scale, shear, rotation, jitter, PDF-noise) via parameterized transforms.<br>4. Emit float32 tensors (N×64×64×C) in global memory. |
| **Outputs** | Device pointer(s) to rendered glyph batches plus optional signed-distance/coverage maps for downstream procedural effects. |
| **Integration Hooks** | - Accepts descriptor buffers from a Python bridge (`knowledge3d/cranium/bridges/procedural_glyph_bridge.py`).<br>- Streams results into `GPUCNNTrainer.train_batch` without host copies (trainer already expects device pointers).<br>- Reuses existing PTX helpers (`glyph_resonator`, `conv2d_3x3`, `batchnorm_backward`, `sgd_optimizer`). |

## Streaming Dataset Refactor
1. **Descriptor Cache**: Parse fonts once (CPU) into compact descriptors (glyph id → control-point buffer). Store under `/K3D/Knowledge3D.local/procedural_fonts/` alongside manifests.
2. **Generator**: `train_atomic_character.py` swaps the `np.array(positives + negatives, ...)` block with a streaming generator that batches descriptors and calls the rasterizer kernel just-in-time.
3. **Trainer Interface**: Extend `GPUCNNTrainer.train_batch` to accept either host numpy arrays (legacy) or device pointers (procedural). During migration we can keep both paths behind a flag.
4. **Augmentation**: Move PDF-style noise, blur, rotation, etc., into PTX so no CPU PIL/OpenCV dependency remains.

## Future Bitmap Ingestion
The same pipeline can emit bitmap snapshots for archive/annotation without violating sovereignty: instead of storing raster outputs, check in the procedural descriptors plus deterministic seeds. When external bitmap ingestion is needed, run the PTX kernel offline to regenerate datasets.

## Next Steps
1. Implement `knowledge3d/cranium/bridges/procedural_glyph_bridge.py` to marshal descriptors + launch kernels.
2. Author `procedural_glyph_rasterizer.cu` and compile to PTX (align with existing build system under `knowledge3d/cranium/kernels/`).
3. Refactor `train_atomic_character.py` to detect descriptor availability and stream batches; keep the legacy path under `--legacy-raster` for comparison.
4. Re-run the atomic character suite (problematic Latin, punctuation, math) once the streaming path is stable.
