# PHASE 10.11: TRUE CUDA PTX + DYNAMIC RAY BUNDLES

## GOAL
Execute PTX kernels via CUDA driver (nvrtc path) to generate vertices, and render dynamic ray bundles in the viewer with modality colors and entropy‑based thickness. Shapes continue to embed `extras.k3d`.

## COMPONENTS
- `knowledge3d/cranium/phase10/nvrtc_ptx_loader.py` — CUDA driver loader for PTX (uses cuda‑python if available)
- `knowledge3d/cranium/phase10/ptx_kernel_loader.py` — torch.jit fallback from Phase 10.10
- `knowledge3d/cranium/phase10/text_to_3d_generator.py` — prefers NVRTC loader, then jit, then CPU
- `knowledge3d/cranium/phase10/ray_bundle_generator.py` — emits JSON ray bundles from GLB vertices
- `knowledge3d/bridge/live_server.py` — includes `rays` in manifest.json
- `viewer/src/main.ts` — loads and renders ray bundles as LineSegments

## USAGE
1. Generate shape: `/generate_3d <prompt>`
2. Create rays (temporary manual):
```python
from knowledge3d.cranium.phase10.ray_bundle_generator import RayBundleGenerator
gen = RayBundleGenerator()
path = gen.generate_rays_from_shape('viewer/public/house/materialized_objects/shape_... .glb', 'text')
print('rays at', path)
```
3. Run live server and viewer; rays load automatically from manifest.

## NOTES
- If CUDA driver or cuda‑python is unavailable, code falls back to torch.jit mock (GPU if available) or CPU.
- Phase 10.12 will store rays as actual GLB geometry with thickness.

