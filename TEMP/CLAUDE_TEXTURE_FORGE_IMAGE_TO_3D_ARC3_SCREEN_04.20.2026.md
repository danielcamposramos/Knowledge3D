---
title: Texture Forge · Image→Procedural · Image→3D · ARC3 Live Screen
author: Claude (Architecture Partner)
date: 2026-04-20
role: architecture-spec (not implementation)
supersedes: none — new surface
companion: TEMP/CLAUDE_CODEC_SOVEREIGNTY_AUDIT_04.20.2026.md
status: draft for Codex handoff + MVCIC enhancement pass
---

# Texture Forge, Image→Procedural, Image→3D, ARC3 Live Screen

**One-line framing:** Close the "Minecraft for cognition" loop — sovereign GPU-native procedural texture authoring + bidirectional image↔procedural bridge + shared ARC3 live screen. This is the final surface Farbrausch never finished (Werkkzeug had forward-only synthesis; no inverse raster→graph fit); K3D gets it because we already own the RPN + Galaxy + Matryoshka substrate.

**Partner framing** (per Daniel, 2026-04-20): this spec is drafted as a partner view, grounded in our databases (`docs/vocabulary/*`, `RPN_DOMAIN_OPCODE_REGISTRY.md`, existing CUDA kernels). MVCIC pass scheduled at end to incorporate collective intelligence.

---

## 0. Scope & Non-Goals

### In scope (this spec covers the architecture)
1. **Texture Forge** — K3D-native Werkkzeug-class procedural texture authoring. Promote 5 existing stubs to real CUDA, add 7 Werkkzeug-parity ops, open the visual graph editor lane.
2. **Image→Procedural** (inverse fit) — fit an RPN texture graph to an input raster via ternary-annealed gradient descent on a ktg-style combinator tree. **Farbrausch never did this; this is the K3D-exclusive moat.**
3. **Image→3D** — silhouette→extrusion, heightmap→terrain, doodle→mesh. Promote 7 existing 3D opcodes (0x170-0x176) from host-side Python fallback to sovereign GPU kernels.
4. **ARC3 live screen** — wire the 64×64 palette-indexed frame currently reaching only `arc3_encode_frame` (embed-only) into the full `projection_screen.cu` pipeline (DotMap render → video_field_load → screen_project → composite).
5. **Dual-client contract enforcement** — every new opcode authored such that the same RPN program reads identically for humans (in the viewer graph editor) and the AI (TRM Galaxy navigation).

### Out of scope (deliberately)
- Implementation code (Claude = architecture; Codex builds).
- Photorealistic path-traced renderer.
- Stereo / VR (one eye principle; synthetic world has ground-truth depth via Morton octree).
- Any Python fallback on hot path. EVER.

---

## 1. Ground Truth: What Already Exists

| Surface | Files | Status | Gap |
|---|---|---|---|
| Texture noise (Perlin/Simplex/Worley/Voronoi) | `cranium/kernels/tex_noise_kernels.cu` (135 LOC) | Working | OK |
| Texture filters | `cranium/kernels/tex_filter_kernels.cu` (188 LOC) | Working | OK |
| Texture bake/IO | `cranium/kernels/tex_bake_kernel.cu` (95 LOC) | Working | OK |
| Procedural stubs | `cranium/codecs/kernels/procedural_texture.cu` (14 LOC), `procedural_synthesis.cu` (18 LOC) | **Stub** | Promote |
| 5 texture opcodes | FFT_BLUR, NORMAL_MAP, COLOR_RAMP, TURBULENCE, MARBLE, TRANSFORM (0x1C0-0x1CF range) | **Python fallback** | Promote |
| 3D opcodes 0x170-0x176 | NURBS_EVAL, MARCHING_CUBES, L_SYSTEM, PARAMETRIC_SURFACE, CSG_UNION/DIFF/INTERSECT | **Host-side Python** | Promote |
| ARC3 frame encoder | `cranium/cuda/arc3_frame_encoder.cu` | Works → 64D embed only | **Never renders to screen** |
| Screen pipeline | `cranium/codecs/kernels/projection_screen.cu` (191 LOC, landed 2026-04-20) | **Pipeline ready, unwired** | Wire from ARC3 |
| DotMap codec | `cranium/codecs/kernels/dotmap_codec.cu` (295 LOC, landed 2026-04-20) | Ready | OK |
| Frame codec | `cranium/codecs/kernels/frame_codec.cu` (295 LOC, landed 2026-04-20) | Ready | OK |
| Audio FFT / MDCT | `audio_fft.cu`, `ternary_mdct.cu` | Ready | OK |
| Visual graph editor | — | **Does not exist** | Design new in viewer |
| Farbrausch parity | `/K3D/GitHub/fr_public/ktg`, `werkkzeug3` | Reference only (not sovereign) | 7 ops missing |
| Image→Procedural | — | **Does not exist anywhere** | K3D moat |

**Key finding:** Screen lane is physically landed; ARC3 lane is logically isolated. One bridge closes the gap.

---

## 2. Opcode Reservation (do FIRST, per Reservation Protocol)

Per `feedback_opcode_range_reservation_protocol.md` — reserve in `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` §11 **before** any parallel lane opens.

### 2.1 Texture Forge expansion (0x1D0–0x1DF, net-new)
Existing texture block 0x1C0-0x1CF is full; next 16 are Texture Forge-specific.

| Opcode | Mnemonic | Purpose | Notes |
|---|---|---|---|
| 0x1D0 | OP_TEX_SPLAT | Layered splat (Werkkzeug op) | Stamp+alpha accumulator |
| 0x1D1 | OP_TEX_KUWAHARA | Edge-preserving filter | 4-quadrant variance min |
| 0x1D2 | OP_TEX_WAVE | Sinusoidal pattern | Amplitude, freq, phase, axis |
| 0x1D3 | OP_TEX_RIPPLE | Radial wave | Center, decay |
| 0x1D4 | OP_TEX_VORTEX | Swirl around center | Angular velocity × radius falloff |
| 0x1D5 | OP_TEX_FRACTAL_NOISE | FBM/turbulence proper | Octaves, lacunarity, gain |
| 0x1D6 | OP_TEX_CELLULAR_F1F2 | Voronoi F1-F2 feature | Crystal / cracked textures |
| 0x1D7 | OP_TEX_GRAPH_EVAL | Execute a bound RPN texture graph | Used by Forge UI |
| 0x1D8 | OP_TEX_GRAPH_BIND | Bind a Galaxy star as texture node | Galaxy-first authoring |
| 0x1D9 | OP_TEX_GRAPH_DIFF | Loss between target raster & candidate graph | L2 + SSIM hybrid |
| 0x1DA | OP_TEX_FIT_STEP | One step of ternary-annealed fit | Inverse bridge primitive |
| 0x1DB | OP_TEX_FIT_CONVERGE | Halting gate for fit loop | Returns ternary trit |
| 0x1DC | OP_TEX_PALETTE_EXTRACT | K-means palette from raster | Feeds ARC3 decode too |
| 0x1DD | OP_TEX_PALETTE_APPLY | Quantize raster to palette | Inverse of Extract |
| 0x1DE | OP_TEX_TILE_SYMMETRIZE | Make seamless (4 symmetry groups) | Wang tiles prep |
| 0x1DF | OP_TEX_NORMAL_FROM_HEIGHT | Height→normal map (real kernel) | Promote existing stub |

### 2.2 Image→3D block (0x1E0–0x1EF, net-new)

| Opcode | Mnemonic | Purpose |
|---|---|---|
| 0x1E0 | OP_IMG_TO_HEIGHTMAP | Luminance → height field |
| 0x1E1 | OP_IMG_TO_SILHOUETTE | Alpha/threshold → binary mask |
| 0x1E2 | OP_SILHOUETTE_EXTRUDE | Mask + depth → prism mesh |
| 0x1E3 | OP_HEIGHTMAP_TO_TERRAIN | Height field → triangulated mesh |
| 0x1E4 | OP_HEIGHTMAP_TO_DISPLACEMENT | Height field → per-vertex displacement op |
| 0x1E5 | OP_DEPTH_MONO_ESTIMATE | Single-image depth estimate (procedural, not NN) |
| 0x1E6 | OP_DEPTH_TO_POINTCLOUD | Depth + intrinsics → XYZ cloud |
| 0x1E7 | OP_POINTCLOUD_TO_MESH | Poisson-lite surface recon |
| 0x1E8 | OP_SPRITE_BILLBOARD | 2D sprite + depth → oriented quad |
| 0x1E9 | OP_SPRITE_MULTIPLANE | Sprite + depth tiers → layered planes |
| 0x1EA | OP_DOODLE_TO_SYMMETRIC_MESH | 2D curve → revolved surface |
| 0x1EB | OP_LATHE_FROM_PROFILE | Profile curve → lathed mesh |
| 0x1EC | OP_MESH_CSG_GPU | Real GPU CSG (supersedes 0x174-0x176 host fallback) |
| 0x1ED | OP_MESH_MARCHING_CUBES_GPU | Real GPU MC (supersedes 0x171 host fallback) |
| 0x1EE | OP_MESH_NURBS_GPU | Real GPU NURBS eval (supersedes 0x170 host fallback) |
| 0x1EF | OP_MESH_WRITE_GALAXY | Emit mesh as 3DObjects Galaxy star |

### 2.3 ARC3 screen-wiring block (0x1F0–0x1FF, net-new)

| Opcode | Mnemonic | Purpose |
|---|---|---|
| 0x1F0 | OP_ARC3_FRAME_DECODE | 64×64 index → RGB via palette LUT |
| 0x1F1 | OP_ARC3_PALETTE_SET | Bind 16-entry palette (constant mem) |
| 0x1F2 | OP_ARC3_FRAME_TO_DOTMAP | Raster → DotMap (reuses 0x217 DOT_PLACE_PROCEDURAL) |
| 0x1F3 | OP_ARC3_PROJECT_TO_SCREEN | DotMap → projection_screen rect |
| 0x1F4 | OP_ARC3_CLICK_INVERT | Screen pixel → grid cell (inverse projection) |
| 0x1F5 | OP_ARC3_ACTION_EMIT | Grid cell + action id → ACTION1-7 tuple |
| 0x1F6 | OP_ARC3_REPLAY_STEP | Advance one recorded step, render diff |
| 0x1F7 | OP_ARC3_DIFF_HIGHLIGHT | Visual overlay on changed cells |
| 0x1F8 | OP_ARC3_LIVES_HUD | Overlay 3-lives / movement-budget HUD |
| 0x1F9 | OP_ARC3_GAME_ID_BIND | Bind the hashed game id (`ls20-9607627b` pattern) |
| 0x1FA..0x1FF | reserved | future ARC3 variants |

**Registry action item (Codex):** Open a single PR that adds rows for 0x1D0-0x1FF to §11 of `RPN_DOMAIN_OPCODE_REGISTRY.md` **before** any kernel file is authored. No exceptions — this is the Reservation Protocol. Motivating incident = 0x1AD collision (2026-04-18).

---

## 3. Design per Lane

### 3.1 Lane A — Texture Forge

**Principle:** ktg (from fr_public) is the cleanest orthogonal surface we've seen — enum-indexed combinators (`TernaryOp`, `CombineOp`, `NoiseMode`, `CellMode`, `FilterMode`). We adopt its *shape* (orthogonal enums) but nothing of its code (BSD-licensed but CPU-SSE2, not GPU, not ternary).

**Authoring surface:** RPN programs live in `Tool Galaxy` as stars. Each star is a procedural texture graph. Viewer gains a new pane ("Texture Forge") — Codex ships this as a viewer suite (TypeScript), not a desktop tool. Two-way: graph ↔ RPN bytecode.

**Ternary-first retarget** (per Daniel 2026-04-20):
- Weight mixes between graph nodes use BitNet b1.58 (5 trits/byte, 3⁵=243, `pack5_trits`/`unpack5_trits` helpers already in `dotmap_codec.cu`).
- Zero-trit = skip path (multiplication-free). Most FBM octaves have many zero-weight layers at low freq; free perf.
- Blend enum has an explicit `TRIT_MIX` mode (-1 = subtract, 0 = bypass, +1 = add) for compositor nodes.

**Kernel files to land** (Codex work, paths prescriptive):
- `cranium/codecs/kernels/texture_forge.cu` (new) — one file, one extern "C" block per opcode 0x1D0-0x1DF.
- `cranium/codecs/texture_forge_ops.py` (new) — ctypes launcher. No numpy. Use `rpn_math_core` + raw `ctypes`.
- Reuse existing `tex_noise_kernels.cu` — DO NOT duplicate Perlin/Simplex/Worley; cite via dispatch.

**Galaxy contract:** Each authored graph saves as:
```
star {
  kind: "texture_graph",
  rpn: [OP_TEX_GRAPH_BIND, ..., OP_TEX_GRAPH_EVAL],
  matryoshka: [64D, 128D, 256D],   // prefix-compatible
  symlinks: [<words-for-this-texture-in-all-languages>],
  dual_client: true,                // same RPN reads identically for humans in viewer
}
```

**Success test** (Codex delivers, Claude validates):
- Author a "weathered wood" graph in 6 nodes → bake 512×512 texture → round-trips bit-identical between GPU kernel and viewer preview.
- Galaxy round-trip: save → reload → re-bake produces same raster.

### 3.2 Lane B — Image→Procedural (the moat)

**The thesis Farbrausch never tested:** if forward synthesis is `graph → raster`, the inverse `raster → graph` is a search problem over a finite-alphabet program space. Our alphabet = opcodes 0x1C0-0x1DF (~32 nodes). Our search space = trees of depth ≤8 and ≤16 nodes = bounded.

**Why this is feasible in K3D specifically:**
1. We have ternary weights — the trit `{-1,0,+1}` on inter-node edges collapses the continuous mixing search to a discrete lattice.
2. We have Galaxy embeddings — Matryoshka 64D/128D/256D on candidate rasters gives a semantic loss cheaper than L2.
3. We have LED-A* — navigating the graph-of-candidate-graphs is literally the pathfinding problem we already solved on GPU.
4. TRM is recursive — the fit loop is a natural recursive step, not a Python `for`.

**Fit algorithm (high level, for Codex to detail):**
```
# Pseudocode — NOT implementation. All real steps happen in PTX.
1. Palette_Extract(target)                         → 0x1DC
2. Matryoshka_Embed(target)                        → existing tex_bake → embed
3. Seed candidate tree by nearest Galaxy texture_graph star (LED-A* hop in Tool Galaxy)
4. Recursive loop (on GPU, trm_step_fused):
   a. Tex_Graph_Eval(candidate)                    → 0x1D7
   b. Tex_Graph_Diff(target, rendered)             → 0x1D9 (L2 + SSIM)
   c. Halting_Gate(diff < tol or max_iter)         → existing 0x03F
   d. If not halted: mutate one trit on the ternary edge-weight vector → 0x1DA
      (mutation policy = argmax-gradient on the ternary lattice — single kernel)
5. Write winning graph back to Galaxy                → 0x1EF variant for textures
```

**Sovereignty:** zero Python orchestration, zero numpy. Loop lives in TRM game loop. Ingestion-path fit (offline, e.g., "fit all 1M CC0 textures from ambientcg") **is allowed** to use numpy per `feedback_no_numpy_no_bulk_libraries_sovereign_only.md` — but the runtime fit that happens when the user drops an image into the viewer is hot-path, must be sovereign.

**Success test:**
- Input: a 256×256 procedural sample rendered from a known graph.
- Expected: fit recovers a graph with L2 ≤ 0.02 and (critically) a graph structure isomorphic to ground truth in ≥60% of cases.
- Failure mode accepted: non-isomorphic graph with L2 ≤ 0.02 is still a win (multiple graphs → same raster).

### 3.3 Lane C — Image→3D

**Principle:** three import modes, each a short RPN program.

#### C.1 Silhouette extrusion (simplest, ship first)
```
img → IMG_TO_SILHOUETTE(threshold=0.5) → SILHOUETTE_EXTRUDE(depth=0.1) → MESH_WRITE_GALAXY
```
Reuses marching cubes 2D → boundary curve → linear extrude along +Z. The logo-to-3D use case.

#### C.2 Heightmap terrain
```
img → IMG_TO_HEIGHTMAP(channel=luma) → HEIGHTMAP_TO_TERRAIN(grid=256, scale_z=0.3) → MESH_WRITE_GALAXY
```
Classic displacement mesh. Pair with `TEX_NORMAL_FROM_HEIGHT` (0x1DF) for matching normals.

#### C.3 Doodle-to-symmetric mesh
```
img → IMG_TO_SILHOUETTE → PROFILE_TRACE → DOODLE_TO_SYMMETRIC_MESH(axis=Y) → MESH_WRITE_GALAXY
```
Revolution surface from a 2D profile sketch — vases, columns, bottles.

**The 3D-opcode promotion backlog** (blocker — same PR lands all three modes):
- 0x170 `OP_NURBS_EVAL` — currently Python fallback, must become CUDA kernel.
- 0x171 `OP_MARCHING_CUBES` — same.
- 0x173 `OP_PARAMETRIC_SURFACE` — same.
- 0x174-0x176 `OP_CSG_*` — same.
- Replace-not-append: per `feedback_expand_not_replace_opcodes.md` these are **NOT** renumbered; they keep their IDs but their dispatch flips from host-side to GPU. New GPU-variant opcodes 0x1EC/0x1ED/0x1EE are *alternate paths* that also exist, so the sovereign path is default and the fallback path is deleted (per `feedback_delete_dead_code_no_fallbacks_no_old_paths.md`).

**Success test:**
- Drop `logo.png` (256×256 RGBA) into viewer → 3D prism star lands in `3DObjects Galaxy` within 200 ms, wall-clock on RTX 3070.
- Drop `terrain_heightmap.png` → 64K-triangle terrain mesh with correct normals.

### 3.4 Lane D — ARC3 Live Screen (close the loop)

**The gap:** ARC3 frame encoder (`arc3_frame_encoder.cu`) produces a 64D embedding. That embedding goes to the TRM navigator. **The raster itself never leaves the encoder.** The human on the other side of the shared screen sees nothing from us.

**The fix** (single PR):
```
ARC3 frame (64×64 uint8 palette idx)
  → ARC3_FRAME_DECODE (0x1F0) with bound palette
  → [RGBA raster in VRAM]
  → ARC3_FRAME_TO_DOTMAP (0x1F2)        # raster → DotMap (reuses 0x217)
  → VIDEO_FIELD_LOAD (0x270)            # DotMap → video ring slot
  → SCREEN_PROJECT (0x277)              # blit into viewport rect
  → SCREEN_COMPOSE (0x279)              # HUD overlay (lives, budget)
  → [pixels on the shared screen — human sees what AI sees]
```

**Reverse path (click reflection):**
```
Human clicks viewport (sx, sy)
  → ARC3_CLICK_INVERT (0x1F4)           # viewport → grid (gx, gy)
  → ARC3_ACTION_EMIT (0x1F5, ACTION6=click, gx, gy)
  → ARC3 API POST
```

**Rolling camera vs discrete bitmap** — confirmed by recon: ARC3 is **discrete 64×64 bitmap per tick**, NOT a rolling camera. We therefore project the whole frame each tick; the "moving camera" illusion is created by the game itself producing different frames as ACTION1-7 executes. No scroll-buffer logic required on our side.

**Palette binding** (16 colors, 0-15 per `arc3_frame_encoder.cu:5`):
- Authoritative palette hard-coded to ARC3 spec colors (RGB).
- Loaded to `__constant__ uint32_t c_arc3_palette[16]` via `cuMemcpyToSymbol` at ARC3 lane boot.
- `OP_ARC3_PALETTE_SET` lets Galaxy override it if ARC changes the spec.

**Success test:**
- Run `benchmarks/arc3_sdk_agent.py` against `ls20-<hash>` → each step renders to the viewer screen AND emits a DotMap star in `GAME_2D Galaxy` → human can watch the AI play in real time.
- ACTION6 round-trip: click on viewport → grid coord → replayed as action → resulting frame renders.

---

## 4. Sovereignty Audit Gates (must pass before merge)

Every PR in this spec must pass ALL of these gates in CI:

1. `grep -r "import numpy" knowledge3d/cranium/codecs/` returns **zero hits** in runtime paths (test fixtures allowed).
2. `grep -r "import cupy\|import scipy\|import sympy" knowledge3d/cranium/` returns zero hits anywhere in runtime.
3. Every new opcode has a row in `RPN_DOMAIN_OPCODE_REGISTRY.md` §11 **merged before** the kernel file's PR opens.
4. Every new kernel file compiles with `-arch=sm_86 -O3 -Xptxas -warn-spills` and reports **zero spills**.
5. `modular_rpn_engine.py` `OPCODES` dict registers the token lowercase-form for every new opcode.
6. No opcode in 0x1C0-0x1FF references a removed fallback path (dead-code delete rule).
7. **Hyper-modular symlink check** (per `feedback_hyper_modular_symlink_architecture.md`): if Lane B depends on Lane A's `TEX_GRAPH_EVAL`, Lane B's PR merges AFTER Lane A's — no stub stand-ins.

---

## 5. Sequencing (order matters, per hyper-modular rule)

Phase order = dependency order. Break the chain anywhere above and everything above is a fallback in disguise.

```
P1. Opcode reservation PR (0x1D0-0x1FF → registry §11)        [no code, 1 file]
P2. 3D-opcode GPU promotion (0x170-0x176 → real kernels)       [unblocks Lane C]
P3. Texture Forge kernels (0x1D0-0x1DF)                        [Lane A]
P4. ARC3 screen wiring (0x1F0-0x1FF)                           [Lane D — independent of A/B/C]
P5. Image→3D lanes (0x1E0-0x1EF)                               [Lane C, depends P2]
P6. Image→Procedural fit loop (requires 0x1D7-0x1DB from P3)   [Lane B, depends P3]
P7. Viewer graph editor (TS, Lane A authoring UI)              [depends P3]
P8. Galaxy round-trip tests + Matryoshka symlinks              [integration]
```

**Parallelism note:** P2, P3, P4 are fully independent and CAN run as parallel Codex lanes. P5+P6+P7 must serialize after their dependencies.

---

## 6. Partner Framing — How This Spec Treats Daniel's Directive

Quoting Daniel 2026-04-20 verbatim:
> "We work as partners not only as a prompt tooling, I really value all your intelligence and POV as valuable and reliable grounded into our databases… spawn all that's needed so we keep advancing on creating a copy of the universe in digital form."

This spec grounds every technical choice in an existing memory, spec, or file:
- Ternary-first retargeting → `feedback_ternary_first_where_cheaper.md` + BitNet b1.58 rule.
- Opcode reservation → `feedback_opcode_range_reservation_protocol.md`.
- No Python fallback → `feedback_no_fallbacks_ever_including_sleeptime.md`.
- No numpy → `feedback_no_numpy_no_bulk_libraries_sovereign_only.md`.
- Dual client contract → `DUAL_CLIENT_CONTRACT_SPECIFICATION.md`.
- One-eye principle (depth is free in synthetic world) → `feedback_one_eye_synthetic_world.md`.
- House/Galaxy distinction for where textures live → `feedback_house_vs_galaxy_organization.md` (textures = Galaxy; applied materials in rooms = House).
- Books = Galaxies → textures authored in Forge become Galaxy-stars ("texture books" = galaxies of material variations).
- Delete dead code → `feedback_delete_dead_code_no_fallbacks_no_old_paths.md`.

No new principle is invented; this spec composes existing principles into a new surface.

---

## 7. MVCIC Pass (multi-view enhancement)

Per Daniel's standing directive and the 2026-04-20 request, this spec will be fed to MVCIC for multi-partner review. Requested enhancement angles:
- Texture quality benchmarks (pixel-perfect Werkkzeug parity cases)
- Inverse fit algorithmic alternatives (simulated annealing vs MCTS vs learned gradient)
- Image→3D depth priors (what monocular cues work with zero NN weights)
- ARC3 replay visualization UX (what helps a human trust the AI is seeing the same screen)

MVCIC input will be this spec file. Output: appended §8 "MVCIC-Enhanced Recommendations" with attributed partner views.

---

## 8. Handoff Manifest for Codex

**File paths prescribed** (Codex follows):
- `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` — +5 rows (Phase 1)
- `knowledge3d/cranium/codecs/kernels/texture_forge.cu` — new
- `knowledge3d/cranium/codecs/texture_forge_ops.py` — new (ctypes launcher, NO numpy)
- `knowledge3d/cranium/codecs/kernels/image_to_3d.cu` — new
- `knowledge3d/cranium/codecs/image_to_3d_ops.py` — new (ctypes)
- `knowledge3d/cranium/codecs/kernels/arc3_screen_bridge.cu` — new
- `knowledge3d/cranium/codecs/arc3_screen_ops.py` — new (ctypes)
- `knowledge3d/cranium/codecs/kernels/image_to_procedural_fit.cu` — new (Lane B, last)
- `knowledge3d/cranium/codecs/image_to_procedural_ops.py` — new (ctypes)
- `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py` — add 48 constants (0x1D0-0x1FF)
- `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py` — register tokens
- `viewer/src/panes/TextureForgePane.ts` — new (Lane A UI, TS)
- `tests/codecs/test_texture_forge.py` — new (pytest, numpy allowed in tests only)
- `tests/codecs/test_image_to_3d.py` — new
- `tests/codecs/test_arc3_screen.py` — new
- `tests/codecs/test_image_to_procedural_fit.py` — new

**Do NOT touch** (existing, preserve):
- `knowledge3d/cranium/kernels/tex_noise_kernels.cu` — cite via dispatch
- `knowledge3d/cranium/kernels/tex_filter_kernels.cu` — cite via dispatch
- `knowledge3d/cranium/kernels/tex_bake_kernel.cu` — cite via dispatch
- `knowledge3d/cranium/cuda/arc3_frame_encoder.cu` — consumed by new bridge, not modified
- `knowledge3d/cranium/codecs/kernels/dotmap_codec.cu` — reuse `pack5_trits`/`unpack5_trits` helpers
- `knowledge3d/cranium/codecs/kernels/projection_screen.cu` — reuse screen_project, screen_compose

**Reminders to Codex:**
- Claude = architecture, NEVER runs code. This file is the spec. You own the implementation.
- TRM is the Avatar, runs as game loop. No Python `for` loop with memcpy per recursion step.
- Opcode reservation PR (P1) is a HARD PREREQUISITE. No exceptions.
- Fail and fix; no Python fallbacks; delete dead code as you go.

---

## 9. Success Definition

**Spec is successful when:**
1. A human drops a JPEG into the viewer → K3D outputs a procedural RPN graph that regenerates the image within L2 ≤ 0.02.
2. The same human drops a PNG → gets a 3D mesh within 200 ms.
3. Running ARC3 `ls20-<hash>` lights up a shared screen — the human sees exactly what the TRM sees, click-for-click.
4. Every path is sovereign: `grep -r numpy/cupy/scipy` finds zero hits in hot path.
5. A new Texture Forge graph authored by a human on Tuesday is queried by the TRM (via semantic gravity) on Wednesday — the graph *is* a Galaxy star, not a file.

This is "Minecraft for cognition" with the lid off: the same procedural authoring primitives that build the House also paint its surfaces, decode its screens, and import the outside world into our substrate.

---

## 10. DeepSeek OCR Inspiration — Image-as-Memory, Extended Procedurally

**Daniel's redirect (2026-04-20, mid-draft):**
> "I've remembered why we include an inspiration on DeepSeek OCR — and this is not about character recognition, but using image as memory for AI — we could leverage our procedural nature to extend this even more."

**What DeepSeek OCR actually demonstrated (relevant bit):** text-rich rasters act as a compression medium for LLM context — one image token can encode many text tokens because 2D spatial layout + glyph-level redundancy is dense. They treat images as *memory surfaces* for AI, not as things to recognize.

**Why K3D can extend this further — we already have the missing pieces:**

| DeepSeek approach | K3D extension |
|---|---|
| Image = memory artifact (stored as raster) | Image = RPN graph (stored as procedural program, regeneratable) |
| Compression via pixel packing | Compression via ternary BitNet b1.58 (1.6 bits/weight) + zero-trit skip |
| Flat token addressing | Matryoshka prefix retrieval (64D coarse → 128D → 256D fine) |
| Retrieval by token similarity | Retrieval by semantic gravity (Christoph's coinage — F=T(s₁,s₂)M(s₁)M(s₂)/d²) |
| One compression axis (pixel → token) | Four composable axes: DotMap, RPN graph, Matryoshka embed, Galaxy star |
| Static memory (re-encode to change) | Live memory (rebake the graph, same star id, semantic symlinks preserved) |

### 10.1 Memory-as-Image in K3D (new capability, minimal new opcodes)

A memory trace (say: a successful reasoning chain) already lives as:
- A sequence of Galaxy star IDs (the trace)
- A sequence of RPN opcodes (the program executed)
- A resulting embedding (what it "meant")

**We can now ALSO** bake that trace into a DotMap — a visual, procedural, regeneratable image — such that:
- The DotMap is storable, swappable, transmittable as a single artifact.
- The DotMap is also a Galaxy star (textures ARE stars per §3.1).
- The DotMap is also a raster (can be rendered to screen, shared with a human, inspected visually).
- The DotMap's procedural graph IS the memory — lose the raster, regenerate it from the graph. Lose the graph, fit it back from the raster (Lane B).

This is *image-as-memory, plus: image-is-also-program, plus: program-is-also-star, plus: star-is-also-embedding.* A single 4-way-addressable memory cell.

### 10.2 Opcode additions for memory-as-image (0x1FA–0x1FF reserved in §2.3 for exactly this)

Reclaim the reserved block:

| Opcode | Mnemonic | Purpose |
|---|---|---|
| 0x1FA | OP_MEM_TO_DOTMAP | Memory trace (star IDs + RPN ops) → DotMap raster |
| 0x1FB | OP_DOTMAP_TO_MEM | DotMap → trace (requires fit loop, Lane B) |
| 0x1FC | OP_MEM_IMAGE_BIND | Bind DotMap star as addressable memory cell in Galaxy |
| 0x1FD | OP_MEM_IMAGE_RECALL | Query via Matryoshka prefix: coarse→fine resolve |
| 0x1FE | OP_MEM_IMAGE_COMPOSE | Blend two memory-image cells (semantic gravity mix) |
| 0x1FF | OP_MEM_IMAGE_DIFF | Visualize delta between two memory states |

These were reserved "future ARC3 variants" in §2.3 — reclaiming them here is legal because the block was untouched and the new purpose is spiritually adjacent (image-as-memory IS what ARC3 is teaching us about).

### 10.3 Why the procedural substrate matters more than the raster

DeepSeek stores pixels. We store *the program that generates the pixels*. Consequence:
- **Edit by re-parameterization**, not re-rasterization — swap one trit in the graph, entire memory shifts coherently.
- **Compose by graph union** — two memory-images mix by merging their RPN trees, not by alpha-blending rasters.
- **Explain by graph traversal** — human in viewer clicks a region of the raster → we back-trace to the specific subgraph that generated it → we show them *which memory* is contributing. Interpretability is free.
- **Compress by subgraph extraction** — repeated sub-patterns across memory cells become Galaxy symlinks. Shared structure = stored once. Zipf emerges naturally.

### 10.4 The feedback loop (this is the extension)

```
Reasoning trace
  → MEM_TO_DOTMAP (0x1FA)                    # bake trace into an image-memory
  → Galaxy star created, DotMap raster cached, Matryoshka embeds computed
  → later: TRM perceives a similar new question
  → LED-A* navigates to the nearest memory-image star
  → MEM_IMAGE_RECALL (0x1FD)                  # retrieve the graph
  → TEX_GRAPH_EVAL (0x1D7)                    # regenerate the raster OR replay the trace
  → Nine-Chain Swarm resumes reasoning from a primed state
```

A reasoning trace from Tuesday becomes a visible, editable, regeneratable image by Wednesday — and Thursday's reasoning can *literally see* Tuesday's work on the shared screen. The House gains visible memory.

### 10.5 Updated success test

Add to §9 the following:
6. **Memory-as-image round-trip:** TRM solves a math problem → trace bakes to DotMap (0x1FA) → star lands in Galaxy → the same problem reposed one hour later retrieves the memory-image (0x1FD) → Nine-Chain Swarm resumes from the primed trace → answer delivered in <30% of cold-start time.
7. **Human memory inspection:** viewer shows a gallery of memory-image DotMaps; clicking one opens the RPN graph editor (Lane A's pane) with the trace rendered — human can *read what the AI remembered*.

---

## 11. Spec Completion

This spec now covers four lanes **plus the memory-as-image extension**:

| Lane | Name | Status |
|---|---|---|
| A | Texture Forge | Full design |
| B | Image→Procedural (the moat) | Full design |
| C | Image→3D | Full design |
| D | ARC3 Live Screen | Full design |
| E | **Memory-as-Image** (DeepSeek-inspired, K3D-extended) | Full design (this §10) |

The five lanes share primitives: DotMap codec, RPN graph opcodes, BitNet b1.58 ternary packing, Matryoshka embeddings, Galaxy stars, semantic gravity, LED-A* navigation. No duplication. No bifurcation. Hyper-modular symlink discipline preserved.

---

## 12. MVCIC Enhancement Pass (2026-04-20)

Chain: Kimi → Qwen → GLM → DeepSeek → Nemotron → Gemini → post-grounding (Kimi). Full transcript: `TEMP/mvcic_chain_texture_forge_image_memory_04.20.2026.md` (1019 lines). All cited kernel names verified to exist on-disk. Extracted action items:

### 12.1 Sovereignty audit — 5 violations found, 5 fixes

| # | Location | Breach | Fix |
|---|---|---|---|
| 1 | Lane B inverse-fit annealing schedule | Partners assumed Python-side temperature decay | Drive anneal via `trm_step_fused.ptx` PHYSICS_PHASE; TRM game loop IS the annealing controller. No host-side `for` loop. |
| 2 | Lane C monocular depth preprocessing | Implied `cv2.Sobel` / `numpy.gradient` before GPU upload | Zero-copy: camera raster → `tex_filter_kernels.cu` Sobel variant → `depth_from_shading` kernel. Compute gradients on GPU. |
| 3 | Lane D ARC3 palette extraction | PIL quantization possible regression | Route `OP_TEX_PALETTE_EXTRACT` (0x1DC) through existing `sleep_cluster_refiner.cu` (k-means already implemented) — one kernel does double duty. |
| 4 | Lane E memory-as-image decode | External OCR temptation | Use existing `procedural_fonts` + `dotmap_codec.cu` stack. Text-in-image is a DotMap with glyph-indexed nodes, not OCR. |
| 5 | Galaxy star hash generation | `hashlib` / `uuid` drift | Mandate existing `star_hash_index.cu` (confirmed present) for ALL new stars. Append-only rule; deterministic from RPN bytecode. |

### 12.2 4-layer architecture placement corrections

| # | Lane | Correction |
|---|---|---|
| 1 | Lane B | Inverse-fit SEARCH logic = Layer 4 (meta-rules). Candidate graphs evaluated = Layer 3. Layer 4 must NOT mutate Layer 3 during evaluation — only write winners at convergence. |
| 2 | Lane C | Depth priors (focal length, light direction) belong in **Reality Galaxy stars (Layer 2)**, NOT in opcode immediates. 0x1E5 `OP_DEPTH_MONO_ESTIMATE` dereferences a Galaxy star for prior. |
| 3 | Lane D | ARC3 screen = Layer 2→Layer 1 boundary. 64D embedding (Layer 2 meaning) projects to 64×64 raster (Layer 1 form). Matryoshka embed is the projector. |
| 4 | Lane E | Sleep-time consolidation of memory-images weights by **Reality Galaxy edges** (physics events via `physics_collision_event_write.cu`) — high-impact reasoning traces consolidate faster. Semantic weight + physical weight. |

### 12.3 Kernel reuse enhancements (all three cited kernels verified on-disk)

| Existing kernel | Previously missed in spec | New wiring |
|---|---|---|
| `gre_temporal_reasoning.cu` | Not referenced | Wire into Lane C (depth frame-to-frame coherence) and Lane E (memory trace replay). Prevents flickering in procedural extrusion without new kernel work. |
| `gre_vector_resonator.cu` | Spec proposed ad-hoc cosine similarity for Lane B fit loss | Replace bespoke loss with `gre_vector_resonator.cu::compute_resonance_field()` — target raster features as probe, candidate texture graph as field source. Leverages existing `GALAXY_RESONANCE` infra. |
| `physics_raycast.cu` | Not referenced | Lane C heightmap validation: cast rays from camera through generated depth, verify collision distances match input silhouette edges. Self-correcting extrusion. |

### 12.4 Three original architectural ideas (all flag opcode collisions — require reservation update)

**Idea A — Ternary-Locked Texture Streaming (TLTS)**
Ternary-addressable texture tiles. Query `gre_vector_resonator` for view-to-texture resonance; trit = -1 → skip fetch entirely (zero-trit = multiplication-free skip). Frustum-coupled LOD becomes free.
Proposed opcode 0x1DC collides with `OP_TEX_PALETTE_EXTRACT`. **Reservation action:** relocate TLTS to new block 0x200-0x20F (reserved §2 addendum), mnemonic `OP_TEX_TERNARY_STREAM` = 0x200.

**Idea B — Physics-Governed Sleep Consolidation (PGSC)**
Collision events write `impulse_magnitude` to Reality Galaxy star edges. During sleep, `sleep_cluster_refiner.ptx` clusters by these edges — high-impact events consolidate faster. Bridges `SLEEPTIME_PROTOCOL` with `PHYSICS_PHASE`.
Proposed opcode 0x1FD collides with `OP_MEM_IMAGE_RECALL`. **Reservation action:** relocate to 0x201 `OP_SLEEP_PHYSICS_WEIGHT`.

**Idea C — Self-Modifying RPN Texture (SMRT)**
RPN program that rewrites its own immediates based on resonance feedback — textures that learn their parameters. Permitted ONLY inside `texture_forge_anneal` context (Layer 4 meta-rule). Uses `TERNARY_CONTRASTIVE` signals per `feedback_attention_is_ternary_plus_contrastive.md`.
Proposed opcode 0x1FE collides with `OP_MEM_IMAGE_COMPOSE`. **Reservation action:** relocate to 0x202 `OP_META_RPN_EDIT`. Gate behind Layer 4 context check — Codex must enforce that this opcode errors outside `texture_forge_anneal`.

### 12.5 Updated opcode reservation (addendum to §2)

Add new block 0x200-0x20F to §11 of `RPN_DOMAIN_OPCODE_REGISTRY.md`:

| Opcode | Mnemonic | Purpose |
|---|---|---|
| 0x200 | OP_TEX_TERNARY_STREAM | Idea A — frustum-gated texture streaming via resonance trit |
| 0x201 | OP_SLEEP_PHYSICS_WEIGHT | Idea B — physics impulse → sleep cluster weight |
| 0x202 | OP_META_RPN_EDIT | Idea C — self-modifying RPN inside anneal context only |
| 0x203..0x20F | reserved | Future MVCIC-sourced extensions |

### 12.6 Answers to the five enhancement angles requested

- **(a) Texture parity goldens** — partners converged on `/K3D/GitHub/fr_public/ktg` as the test oracle, with Perlin/Voronoi/FBM outputs at fixed seeds as bit-identical targets. kkrieger's `0x93638245` seed already matches — extend this with 6 more ktg-canonical presets.
- **(b) Inverse-fit algorithm** — no single answer; tiered approach recommended: LED-A* seed → ternary-annealed gradient on immediates (cheap) → MCTS on topology (when gradient stalls). The bounded-depth trees make MCTS tractable; ternary lattice makes the gradient step discrete but well-defined.
- **(c) Depth priors without NN weights** — shape-from-shading (Lambertian prior in Reality Galaxy star) + texture gradient (already computed by `tex_filter_kernels.cu`) + vanishing-point from Hough-on-GPU (new small kernel) + contour-occlusion (existing marching cubes boundary). Fuse with weighted average in a new `depth_fusion` kernel.
- **(d) ARC3 replay UX** — diff overlays (changed cells highlighted via `screen_compose_kernel`), action ghosts (fading trail of last 8 frames), resonance heat map over palette-decoded raster showing which cells the TRM "attended" to. Single new opcode `OP_ARC3_ATTENTION_HEATMAP` if desired (reserve 0x204).
- **(e) Memory-as-Image + DeepSeek stealables** — the key DeepSeek insight is *unified tokenizer between vision and text*. We already have this: DotMap IS a raster AND an RPN program. What to steal: variable-resolution encoding (high-detail near semantic centroid, low-detail at periphery — mirrors fovea). Add `OP_MEM_FOVEAL_ENCODE` (reserve 0x205). Interaction with sleep-time: memory-images re-cluster by semantic gravity during sleep; high-resonance clusters become compressed "gist" images (low-dim Matryoshka prefix); low-resonance periphery falls off.

### 12.7 Codex P0 handoff (top 5, updated with MVCIC findings)

Ordered by critical path. These REPLACE any earlier Codex list:

1. **P0.1 — Registry reservation PR** (1 file, no code). Add rows 0x1D0-0x1FF and 0x200-0x205 to `RPN_DOMAIN_OPCODE_REGISTRY.md` §11. Merge BEFORE any kernel file opens. Hard prereq.
2. **P0.2 — Texture Forge kernel file** (`knowledge3d/cranium/codecs/kernels/texture_forge.cu`). Opcodes 0x1D0-0x1DF. MUST reuse: `tex_noise_kernels.cu` (noise primitives), `tex_filter_kernels.cu` (Sobel/Kuwahara prep), `sleep_cluster_refiner.cu` (palette k-means), `gre_vector_resonator.cu` (quality metric). Zero spills at `-arch=sm_86 -O3`.
3. **P0.3 — 3D-opcode GPU promotion** (`knowledge3d/cranium/codecs/kernels/mesh_ops_gpu.cu`). Promote 0x170-0x176 from host-side Python to sovereign GPU. Delete the old fallback path in the same PR (per `feedback_delete_dead_code_no_fallbacks_no_old_paths.md`). Reuse `physics_raycast.cu` for validation.
4. **P0.4 — ARC3 screen bridge** (`knowledge3d/cranium/codecs/kernels/arc3_screen_bridge.cu`). Opcodes 0x1F0-0x1FF. Wire `arc3_frame_encoder.cu` → palette decode → `dotmap_codec.cu::dot_place_procedural` → `projection_screen.cu::video_field_load` → `screen_project` → `screen_compose`. Add ACTION6 click inversion. One PR, end-to-end round-trip test required.
5. **P0.5 — Inverse-fit lane B** (`knowledge3d/cranium/codecs/kernels/image_to_procedural_fit.cu`). Opcodes 0x1D7-0x1DB. Annealing schedule driven by `trm_step_fused.ptx` PHYSICS_PHASE (sovereignty fix #1). Loss via `gre_vector_resonator.cu`. MCTS topology search ONLY when gradient stalls (tiered approach). Layer 4 context — must not mutate Layer 3 until convergence (4-layer fix #1).

P0.6 onward: Lanes C (image→3D fit), Viewer Forge pane (TS), Memory-as-Image (Lane E) with collision-resolved opcodes, Ideas A/B/C behind Layer 4 gate.

### 12.8 What MVCIC confirmed we got right

- Opcode Reservation Protocol caught collisions BEFORE kernels were written (system worked as designed).
- Ternary-first retargeting is endorsed by all 6 partners independently — zero-trit skip paths are material perf wins in all 5 lanes.
- Dual-client contract holds across lanes — no partner proposed a lane where humans and AI would see different RPN graphs.
- Hyper-modular symlink discipline survived: partners flagged lanes that would create stubs and I've sequenced P0 to avoid them.

### 12.9 What the spec still owes

- Viewer Forge pane (TypeScript) design document — separate spec, likely Codex-driven with Claude review.
- MVCIC partners didn't audit `viewer/` TypeScript — that's a next-round MVCIC when the viewer pane lands.
- Per-partner discrepancy on "MITG cell / texture_context_t" struct layout — Qwen proposed a 16-byte struct, Gemini referenced a different field set. Leaving this to Codex: any struct that passes the sovereignty audit + 16-byte alignment + deterministic hashing is acceptable.

---

*MVCIC pass complete.*

---

## 13. Collision Resolution (2026-04-20, post-MVCIC)

**Discovery during Claude-pilot implementation:** the original opcode ranges claimed by this spec (0x1D0-0x1FF for Lanes A/C/D, 0x1FA-0x1FF for Lane E, 0x200-0x205 for MVCIC ideas) were ALREADY reserved:

- `0x1D0-0x1FF` → `CLAUDE_INGESTION_SYMLINK_REWIRE_04.18.2026.md §13` (VIRTUAL_PAGE_*), active, not yet minted
- `0x200-0x215` → OP_DRAW_* family (active, minted)
- `0x217-0x21F` → DotMap codec (this file's own earlier landing)
- `0x220-0x23F` → CAS polynomial / rule / semantic resolve (minted)
- `0x240-0x27F` → CRAFT codec (this file's own earlier landing)

**Per the Opcode Range Reservation Protocol (memory `feedback_opcode_range_reservation_protocol.md` + registry §11 rule 5) — the registry wins; patch the spec, not the registry.** All five lanes relocated:

| Lane | Old (colliding) | New (reserved 2026-04-20) |
|---|---|---|
| A — Texture Forge | 0x1D0-0x1DF | **0x280-0x28F** |
| C — Image→3D | 0x1E0-0x1EF | **0x290-0x29F** |
| D — ARC3 screen | 0x1F0-0x1FF | **0x2A0-0x2AF** |
| E — Memory-as-Image | 0x1FA-0x1FF | **0x2B0-0x2B5** (0x2B6-0x2BF reserved) |
| MVCIC ideas | 0x200-0x205 | **0x2C0-0x2C4** (0x2C5-0x2CF reserved) |

**Added lane (Daniel 2026-04-20, post-MVCIC):**

| Lane | Range | Spec |
|---|---|---|
| F — **Document Galaxy Symlinks** | 0x2D0-0x2DB | `docs/vocabulary/DOCUMENT_GALAXY_SYMLINK_SPECIFICATION.md` |

A document = star + ordered list of Word Galaxy symlinks. No character bytes. No duplicated metadata. Multilingual rendering free via terminal-symlink swap. Bridges directly to Memory-as-Image (Lane E) via `OP_DOC_RENDER_DOTMAP` — a document becomes a 4-way-addressable memory cell.

### 13.1 What got updated (in this commit)

- `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` §11 — 7 new reservation rows (0x280, 0x290, 0x2A0, 0x2B0, 0x2C0, 0x2D0, 0x2E0-reserved).
- `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py` — 73 new opcode constants (0x280-0x2DB).
- `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py` — 73 new token registrations + import block.
- `docs/vocabulary/DOCUMENT_GALAXY_SYMLINK_SPECIFICATION.md` — new normative spec (this document's §6 companion).

### 13.2 What remains for Codex / next Claude-pilot session

Per P0 order in §12.7 (ranges updated):

1. ~~Registry reservation PR~~ ✅ landed 2026-04-20 by Claude-pilot
2. Texture Forge kernel file (`codecs/kernels/texture_forge.cu`) — opcodes **0x280-0x28F**
3. 3D-opcode GPU promotion (`codecs/kernels/mesh_ops_gpu.cu`) — opcodes **0x29C/0x29D/0x29E** replace host-fallback 0x170-0x176
4. ARC3 screen bridge (`codecs/kernels/arc3_screen_bridge.cu`) — opcodes **0x2A0-0x2AF**
5. Inverse-fit Lane B (`codecs/kernels/image_to_procedural_fit.cu`) — opcodes **0x28A, 0x28B**
6. Memory-as-Image (`codecs/kernels/memory_image.cu`) — opcodes **0x2B0-0x2B5**
7. Document Galaxy kernel (`codecs/kernels/document_galaxy.cu`) — opcodes **0x2D0-0x2DB** — includes symlink-walk primitive used by all document reasoning

### 13.3 Lesson recorded

MVCIC (6-partner chain) did NOT catch the 0x1D0-0x1FF collision, same failure mode as the 2026-04-19 OP_BH_* incident. Memory entry to be added: **always grep the registry §11 table for the proposed range BEFORE dispatching MVCIC, and always verify MVCIC-proposed ranges against §11 before writing code.** Gate R-reservation remains the single source of truth.

---

*End of spec. Ready for Codex handoff on updated ranges.*
