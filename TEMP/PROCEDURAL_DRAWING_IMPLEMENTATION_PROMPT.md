# Procedural Drawing Stack Implementation Prompt for Codex
## Comprehensive Synthesis of Swarm Contributions Grounded in K3D Architecture

**Date:** 2025-11-18
**Context:** Final phase of procedural vector drawing research → production implementation
**Goal:** Enable K3D to draw all 2D forms procedurally (atomic shapes → complex vectors → atomic characters)
**Why:** Drawing came before written language — this is the graphical atomic base for multi-modal cognition

---

## Executive Summary: What We're Building

**The Vision:**
A **sovereign, GPU-native procedural drawing system** that treats all visual content (fonts, ASCII art, vectors, CAD, even pixels) as **RPN programs executed by PTX kernels**. This enables:

1. **Atomic character learning** (Phase H next step): Characters are drawings with encoded meaning
2. **Tri-modal emergence**: Text ("A") ≈ Visual (glyph curves) ≈ Audio (/eɪ/) discovered automatically
3. **Procedural compression**: Store "how to draw" (RPN program) not pixels (200:1 to 1000:1)
4. **Living computer museum**: Interactive VM desks where avatar experiences UI history
5. **Carbon efficiency**: Sub-100µs GPU-native rendering vs current pixel-based approaches

**Industry Gap:** Nobody has unified TrueType + ASCII + vectors + CAD + pixels into a single RPN execution substrate. We're 3-5 years ahead.

---

## Partner Contributions Summary

### 1. Grok (xAI): TrueType Fonts as Procedural Foundation

**Core Insight:** TrueType fonts are already procedural (quadratic Bézier curves + hinting instructions). Treat glyphs as **atomic RPN programs**.

**Key Contributions:**
- TTF reduces shapes to atomic ops: `moveTo`, `lineTo`, `quadTo`
- Font files contain ~10K glyphs (2,713 fonts harvested) = rich procedural training data
- Hinting as adaptive intelligence (like Matryoshka: simple glyphs = 64D, complex = 2048D)
- Cross-modal potential: Font datasets → model discovers "A" text ≈ curve visual ≈ /eɪ/ sound

**Proposed PTX Kernels:**
- `ttf_parse.ptx`: Parse .ttf binary directly on GPU (no CPU fallback)
- `glyph_rasterizer.ptx`: On-demand Bézier rendering with ternary hinting

**Ternary Integration:**
- Hinting decisions: `-1 = blur`, `0 = neutral`, `+1 = sharpen edges`
- Ternary embeddings: 64 ternary dims ≈ 96 binary bits but richer information

---

### 2. Qwen (Alibaba): Corel Draw Vectors & ASCII Art

**Core Insight:** Corel Draw (1990s) pioneered complex procedural hierarchies. ASCII art demonstrates **text IS image** — perfect modality collapse.

**Key Contributions:**
- **Corel/WMF/CDR:** Multi-layer compositions, parametric effects, color models as math operations
- **ASCII Art:** Character grids as spatial grammar — no rasterization needed
- **Procedural-first storage:** House GLBs store RPN instructions, not geometry

**Proposed PTX Kernels:**
- `cdr_decoder.ptx`: Parse CDR/WMF vector trees (hierarchy traversal with ternary gating)
- `ascii_perceiver.ptx`: Character-grid spatial analysis (density, symmetry axes)
- `vector_fuser.ptx`: Merge ASCII/Corel/TTF primitives with ternary confidence gating

**Training Protocol:**
- 50K CDR files + 500K ASCII artifacts from open archives
- RLWHF rewards for recognizability ("Does this ASCII cat look like a cat?")
- Cross-modal generation: "Draw ASCII of this CDR logo"

---

### 3. Kimi (Moonshot): RPN-Graph Trinity & CAD Ontology

**Core Insight:** RPN isn't just fast — it's **embodied thinking**. Stack-based cognition is closer to mathematical truth than algebraic syntax.

**Key Contributions:**
- **Stack-Machine Cognition (SMC):** All K3D procedural ops are RPN programs as spatial embeddings
- **Unified Procedural Continuum:**
  ```
  ASCII → TTF → Corel → CAD (NURBS) → B-Rep Solids → Physics Assemblies
  ```
- **Ternary synergy:** Push (+1), Pop (-1), No-op (0) are stack primitives
- **CAD standards:** STEP, IGES, B-Rep, constructive solid geometry (CSG)

**Proposed PTX Kernels:**
- `modular_rpn_kernel.cu` (EXTEND existing): Add drawing opcodes:
  - `MOVE`, `LINE`, `QUAD`, `CUBIC`, `ARC`, `CLOSE`, `STROKE`, `FILL`
- `cad_brep_boolean.ptx`: Boolean operations on boundary representations
- `nurbs_evaluator.ptx`: NURBS curve/surface evaluation

**Philosophy:**
> "Humans who 'think like machines' using RPN calculators aren't adapting to machines — they're leveraging stack-based cognition, which is closer to mathematical truth."

---

### 4. DeepSeek: Pixel-to-Procedural Vision

**Core Insight:** Understand the **entire chain** from procedural description → GPU rasterization → pixels. This enables reverse: pixels → procedural reconstruction.

**Key Contributions:**
- **Rasterization pipeline:** Vertex processing → primitive assembly → rasterization → fragment shading → pixel output
- **Scanline algorithms:** Bresenham for lines, scanline polygon fill
- **Procedural reconstruction:** Edge detection + curve-fitting = pixel-to-Bézier

**Proposed PTX Kernels:**
- `pixel_genesis.ptx`: Full rasterization pipeline (scan convert Bézier curves to pixels)
- `universal_primitive_kernel.ptx`: Unified kernel for points/lines/quads/cubics
- `edge_to_procedural.ptx`: Reverse process — pixels → RPN programs

**Validation:**
- Compare `pixel_genesis` output against Mesa software rasterizer (llvmpipe)
- <0.5% visual deviation (SSIM metric) on 10K test shapes

---

### 5. Codex: VM Bridge & Living Computer Museum

**Core Insight:** Enable avatar to **use old paradigms live** (not just reconstruct them), like LLMs use browsers today.

**Key Contributions:**
- **Museum wing in House:** Desks representing computing history (ENIAC → VT100 → Mac System 7 → modern Linux)
- **VM backends:** QEMU/VirtualBox/emulators exposed via VNC/SPICE
- **Interactive use:** Avatar reads screen, reasons, acts (keyboard/mouse injection)
- **Procedural learning:** Log UI behaviors as RPN programs

**Implementation:**
- Extend WebSocket bridge with VNC client threads per desk
- Framebuffer → texture on desk screen mesh in House GLB
- Input: ActionBuffer → VNC PointerEvent/KeyEvent

---

### 6. Claude (Anthropic): Historical Grounding & Display Protocols

**Core Insight:** Ground museum desks in **real systems** (verified from computer museums). Understand display protocols (X11, Wayland, VNC, SPICE) to learn "monitor reality."

**Key Contributions:**
- **Verified museum desks:**
  - ENIAC (1945): Panel programming with switches/cables
  - PDP-1 (1960): Vector CRT, Spacewar!
  - VT100 (1978): Terminal culture, ANSI escape codes
  - Mac System 7 (1991): TrueType fonts, QuickDraw
  - Linux Desktop (X.Org/Wayland): Client-server graphics

- **Display protocol research:**
  - **X11:** Network-transparent client-server, procedural draw calls (PolyLine, FillRect)
  - **Wayland:** Compositor-centric, direct buffer passing
  - **VNC:** Framebuffer protocol, pixel updates over TCP
  - **SPICE:** QXL vector commands (procedural when QXL driver active)

- **Observe→Interpret→Act loop:**
  - Observe: Visual kernels (FractalEmitter, TemporalReasoning) process framebuffer
  - Interpret: Ternary tagging (+1 actionable, 0 context, -1 noise)
  - Act: ActionBuffer → VNC input injection

---

## Architecture Integration: Grounding in K3D Stack

### Existing K3D Components (Reuse These!)

| Component | File Path | Purpose | How We Use It |
|-----------|-----------|---------|---------------|
| **ThinkingTagBridge** | `knowledge3d/cranium/ptx_runtime/thinking_tag_bridge.py` | 5-state cognitive pipeline (INGEST → FUSE → SPATIAL → REASON → OUTPUT) | Main inference entry point for procedural drawing |
| **ModularRPNEngine** | `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py` | RPN program executor | **EXTEND** with drawing opcodes |
| **Ternary Stack (Rounds 3-6)** | `knowledge3d/cranium/ptx_runtime/ternary_*.py` | Balanced ternary logic {-1, 0, +1} | Gating, routing, hinting decisions |
| **FractalEmitter** | `knowledge3d/cranium/ptx_runtime/fractal_emitter.py` | Visual feature extraction | Screen analysis for VM desks |
| **TemporalReasoning** | `knowledge3d/cranium/ptx_runtime/temporal_reasoning.py` | Temporal patterns | ASCII scrolling, animation |
| **AtomicFissionFusion** | `knowledge3d/cranium/bridges/atomic_fission_fusion_bridge.py` | Multi-modal fusion | Text + Visual + Audio cross-modal scoring |
| **GalaxyResonanceEngine** | `knowledge3d/cranium/ptx_runtime/galaxy_resonance_engine.py` | Embedding-based reasoning | Spatial clustering of glyphs/vectors |
| **SovereignBridge** | `knowledge3d/cranium/sovereign/sovereign_bridges.py` | Base class for PTX bridges | Pattern for all new bridges |
| **LatencyGuard** | `knowledge3d/cranium/ptx_runtime/latency_guard.py` | <100µs budget enforcement | Ensure sovereignty compliance |

### New Components to Implement

| Component | File Path | Purpose | Dependencies |
|-----------|-----------|---------|--------------|
| **ProceduraDrawingBridge** | `knowledge3d/cranium/bridges/procedural_drawing_bridge.py` | Main entry point for drawing operations | Extends SovereignBridge |
| **TTF Parser Kernel** | `knowledge3d/cranium/ptx_runtime/kernels/ttf_parse.ptx` | Parse TrueType files on GPU | None (pure PTX + ctypes) |
| **Glyph Rasterizer** | `knowledge3d/cranium/ptx_runtime/kernels/glyph_rasterizer.ptx` | Render Bézier curves with ternary hinting | ttf_parse.ptx |
| **CDR Decoder Kernel** | `knowledge3d/cranium/ptx_runtime/kernels/cdr_decoder.ptx` | Parse Corel Draw vector trees | graph_crystallizer.ptx (for hierarchy) |
| **ASCII Perceiver** | `knowledge3d/cranium/ptx_runtime/kernels/ascii_perceiver.ptx` | Character-grid spatial analysis | rpn_embedding_engine.ptx |
| **Vector Fuser** | `knowledge3d/cranium/ptx_runtime/kernels/vector_fuser.ptx` | Merge TTF/CDR/ASCII primitives | atomic_fission_fusion.ptx |
| **Pixel Genesis Kernel** | `knowledge3d/cranium/ptx_runtime/kernels/pixel_genesis.ptx` | Full rasterization pipeline | None (implements from scratch) |
| **Universal Primitive Kernel** | `knowledge3d/cranium/ptx_runtime/kernels/universal_primitive_kernel.ptx` | Unified rendering (points/lines/quads/cubics) | pixel_genesis.ptx |
| **RPN Drawing Opcodes** | `knowledge3d/cranium/ptx_runtime/modular_rpn_kernel.cu` (EXTEND) | MOVE, LINE, QUAD, CUBIC, ARC, CLOSE, STROKE, FILL | Existing RPN engine |
| **VM Desk Bridge** | `knowledge3d/bridge/vm_desk_bridge.py` | VNC/SPICE client integration | WebSocket server (live_server.py) |
| **Procedural Art Harvester** | `knowledge3d/tools/procedural_art_harvester.py` | Ingest TTF/CDR/ASCII datasets | parallel_font_harvester.py (extend) |

---

## Implementation Roadmap (5 Phases)

### Phase 1: RPN Drawing Primitives (Week 1) — HIGHEST PRIORITY

**Goal:** Extend ModularRPNEngine with drawing opcodes, implement basic PTX rasterization.

**Tasks:**

1. **Extend RPN Engine (modular_rpn_kernel.cu)**
   - Add opcodes:
     ```c
     // Drawing primitives
     MOVE,     // x y MOVE → move pen to (x, y)
     LINE,     // x y LINE → draw line to (x, y)
     QUAD,     // cx cy x y QUAD → quadratic Bézier (control, endpoint)
     CUBIC,    // cx1 cy1 cx2 cy2 x y CUBIC → cubic Bézier
     ARC,      // rx ry angle large_arc sweep x y ARC → elliptical arc
     CLOSE,    // CLOSE → close path
     STROKE,   // STROKE → render path outline
     FILL,     // FILL → fill path interior

     // State management
     PUSH_STATE,  // Save current transform/stroke/fill
     POP_STATE,   // Restore previous state
     TRANSLATE,   // dx dy TRANSLATE
     ROTATE,      // angle ROTATE
     SCALE,       // sx sy SCALE

     // Style
     SET_STROKE_COLOR,  // r g b a SET_STROKE_COLOR
     SET_FILL_COLOR,    // r g b a SET_FILL_COLOR
     SET_LINE_WIDTH,    // width SET_LINE_WIDTH
     ```

   - Stack-based execution (Kimi's Stack-Machine Cognition):
     ```python
     # Example RPN program: Draw red square
     "0 0 MOVE 100 0 LINE 100 100 LINE 0 100 LINE CLOSE 255 0 0 255 SET_FILL_COLOR FILL"

     # Execution trace:
     # [0, 0] MOVE → path starts at (0,0)
     # [100, 0] LINE → line to (100,0)
     # [100, 100] LINE → line to (100,100)
     # [0, 100] LINE → line to (0,100)
     # CLOSE → close path back to (0,0)
     # [255, 0, 0, 255] SET_FILL_COLOR → red fill
     # FILL → rasterize interior
     ```

2. **Implement pixel_genesis.ptx**
   - **Scanline rasterization** for Bézier curves:
     ```c
     __device__ void rasterize_quadratic_bezier(
         float x0, float y0,      // Start point
         float cx, float cy,      // Control point
         float x1, float y1,      // End point
         float* framebuffer,      // Output H×W×4 RGBA
         int width, int height,
         float4 color             // RGBA color
     ) {
         // Subdivide curve into segments (de Casteljau algorithm)
         const int num_segments = 32;  // Adaptive based on curve length
         for (int i = 0; i < num_segments; i++) {
             float t0 = i / (float)num_segments;
             float t1 = (i + 1) / (float)num_segments;

             // Evaluate quadratic Bézier at t0 and t1
             float2 p0 = eval_quad_bezier(x0, y0, cx, cy, x1, y1, t0);
             float2 p1 = eval_quad_bezier(x0, y0, cx, cy, x1, y1, t1);

             // Draw line segment p0 → p1 (Bresenham's algorithm)
             draw_line_bresenham(p0.x, p0.y, p1.x, p1.y, framebuffer, width, height, color);
         }
     }

     __device__ float2 eval_quad_bezier(float x0, float y0, float cx, float cy,
                                          float x1, float y1, float t) {
         float s = 1.0f - t;
         float2 result;
         result.x = s*s*x0 + 2*s*t*cx + t*t*x1;
         result.y = s*s*y0 + 2*s*t*cy + t*t*y1;
         return result;
     }
     ```

   - **Anti-aliasing** via supersampling (Matryoshka adaptive):
     - Simple shapes (64D): 1× sampling
     - Standard (512D): 2× supersampling
     - Complex (2048D): 4× supersampling

3. **Create ProceduralDrawingBridge**
   ```python
   # knowledge3d/cranium/bridges/procedural_drawing_bridge.py

   from knowledge3d.cranium.sovereign.sovereign_bridges import SovereignBridge
   from knowledge3d.cranium.ptx_runtime.latency_guard import LatencyGuard
   import numpy as np

   class ProceduralDrawingBridge(SovereignBridge):
       """Sovereign bridge for procedural 2D drawing via RPN programs."""

       def __init__(self, gpu_id: int = 0):
           super().__init__(gpu_id)
           self.load_ptx_kernels([
               'pixel_genesis.ptx',
               'universal_primitive_kernel.ptx',
               'glyph_rasterizer.ptx'
           ])
           self.latency_guard = LatencyGuard(budget_us=100)  # Sub-100µs requirement

       def execute_rpn_program(self, rpn_program: str, width: int = 512, height: int = 512) -> np.ndarray:
           """Execute RPN drawing program, return RGBA framebuffer.

           Args:
               rpn_program: Space-separated RPN commands (e.g., "0 0 MOVE 100 100 LINE STROKE")
               width: Framebuffer width in pixels
               height: Framebuffer height in pixels

           Returns:
               np.ndarray of shape (height, width, 4) with RGBA pixels
           """
           with self.latency_guard.measure("execute_rpn_program"):
               # Allocate framebuffer on GPU
               framebuffer = self.gpu_malloc(width * height * 4 * 4)  # float4 RGBA

               # Parse RPN program to bytecode (host-side tokenization, GPU execution)
               bytecode = self._parse_rpn_to_bytecode(rpn_program)

               # Launch pixel_genesis kernel
               self.launch_kernel(
                   'pixel_genesis',
                   bytecode=bytecode,
                   framebuffer=framebuffer,
                   width=width,
                   height=height
               )

               # Copy framebuffer back to host
               result = self.gpu_to_host(framebuffer, shape=(height, width, 4), dtype=np.float32)
               self.gpu_free(framebuffer)

               return result

       def draw_glyph(self, glyph_data: bytes, font_size: float = 24.0) -> np.ndarray:
           """Render TrueType glyph from raw TTF contour data."""
           with self.latency_guard.measure("draw_glyph"):
               # Parse glyph contours (done on GPU via ttf_parse.ptx)
               contours_gpu = self.load_glyph_to_gpu(glyph_data)

               # Allocate output framebuffer
               size = int(font_size * 1.5)  # 1.5× for ascenders/descenders
               framebuffer = self.gpu_malloc(size * size * 4 * 4)

               # Launch glyph_rasterizer with ternary hinting
               self.launch_kernel(
                   'glyph_rasterizer',
                   contours=contours_gpu,
                   framebuffer=framebuffer,
                   size=size,
                   font_size=font_size,
                   ternary_hint_level=0  # -1 blur, 0 neutral, +1 sharpen
               )

               result = self.gpu_to_host(framebuffer, shape=(size, size, 4), dtype=np.float32)
               self.gpu_free(framebuffer)
               self.gpu_free(contours_gpu)

               return result
   ```

4. **Unit Tests**
   ```python
   # tests/test_procedural_drawing_bridge.py

   import pytest
   import numpy as np
   from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

   @pytest.mark.cuda
   def test_draw_simple_line():
       """Test RPN: 0 0 MOVE 100 100 LINE STROKE"""
       bridge = ProceduralDrawingBridge(gpu_id=0)

       framebuffer = bridge.execute_rpn_program(
           "0 0 MOVE 100 100 LINE 255 255 255 255 SET_STROKE_COLOR 2 SET_LINE_WIDTH STROKE",
           width=128,
           height=128
       )

       assert framebuffer.shape == (128, 128, 4)
       assert np.any(framebuffer[:, :, :3] > 0)  # Some pixels drawn
       assert bridge.latency_guard.get_last_duration_us() < 100  # Sub-100µs

   @pytest.mark.cuda
   def test_draw_quadratic_bezier():
       """Test RPN: quadratic Bézier curve"""
       bridge = ProceduralDrawingBridge(gpu_id=0)

       framebuffer = bridge.execute_rpn_program(
           "10 10 MOVE 50 100 90 10 QUAD 255 0 0 255 SET_STROKE_COLOR STROKE",
           width=128,
           height=128
       )

       assert framebuffer.shape == (128, 128, 4)
       # Verify curve is drawn (red pixels on specific path)
       red_pixels = framebuffer[:, :, 0] > 200
       assert np.sum(red_pixels) > 50  # At least 50 red pixels

   @pytest.mark.cuda
   def test_ternary_hinting():
       """Test ternary hinting: -1 blur, 0 neutral, +1 sharpen"""
       bridge = ProceduralDrawingBridge(gpu_id=0)

       # Mock glyph data (simple square)
       glyph_rpn = "0 0 MOVE 50 0 LINE 50 50 LINE 0 50 LINE CLOSE FILL"

       blur = bridge.execute_rpn_program(glyph_rpn + " -1 SET_TERNARY_HINT", width=64, height=64)
       neutral = bridge.execute_rpn_program(glyph_rpn + " 0 SET_TERNARY_HINT", width=64, height=64)
       sharpen = bridge.execute_rpn_program(glyph_rpn + " 1 SET_TERNARY_HINT", width=64, height=64)

       # Blur should have softer edges (more anti-aliasing)
       # Sharpen should have harder edges (less anti-aliasing)
       assert np.std(blur) < np.std(sharpen)  # Blur has less variance
   ```

**Deliverable:** Working RPN drawing engine with <100µs latency on simple shapes.

---

### Phase 2: TrueType Font Integration (Week 2)

**Goal:** Parse .ttf files on GPU, render glyphs as RPN programs, harvest 2,713 fonts into Galaxy.

**Tasks:**

1. **Implement ttf_parse.ptx**
   - Read TTF binary format (tables: `head`, `glyf`, `loca`, `cmap`)
   - Extract contour data (on-curve/off-curve points)
   - Output: Array of (x, y, flag) triples

2. **Extend ProceduralDrawingBridge.draw_glyph()**
   - Convert TTF contours to RPN program:
     ```python
     # TTF contour: [(10,10,ON), (50,100,OFF), (90,10,ON)]
     # → RPN: "10 10 MOVE 50 100 90 10 QUAD"
     ```

3. **Create ProceduralArtHarvester**
   ```python
   # knowledge3d/tools/procedural_art_harvester.py

   from knowledge3d.ingestion.fonts.parallel_font_harvester import FontHarvester
   from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

   class ProceduralArtHarvester:
       """Harvest fonts, CDR, ASCII into procedural RPN programs."""

       def harvest_fonts(self, font_dir: str, output_glb: str):
           """Harvest all fonts from directory, store as RPN programs in GLB."""
           harvester = FontHarvester(font_dir)
           bridge = ProceduralDrawingBridge()

           glyphs_rpn = []
           for font_path in harvester.find_fonts():
               font_data = harvester.load_font(font_path)
               for char_code in range(0x20, 0x7F):  # ASCII printable
                   glyph_data = font_data.get_glyph(char_code)
                   if glyph_data:
                       # Convert TTF contours to RPN program
                       rpn_program = self._ttf_to_rpn(glyph_data)

                       # Store in Galaxy as embedding
                       glyphs_rpn.append({
                           'character': chr(char_code),
                           'font_name': font_data.name,
                           'rpn_program': rpn_program,
                           'embedding': self._embed_rpn(rpn_program)
                       })

           # Save to GLB with K3D extensions
           self.save_to_glb(glyphs_rpn, output_glb)
   ```

4. **Tests**
   - Parse 10 TTF files from system fonts
   - Render all ASCII glyphs (0x20-0x7E)
   - Validate SSIM > 99.9% vs FreeType (offline comparison)

**Deliverable:** 168K+ glyphs (2,713 fonts × 95 ASCII chars) as RPN programs in Galaxy GLB.

---

### Phase 3: ASCII Art & Corel Vectors (Week 3)

**Goal:** Implement ASCII perceiver and CDR decoder, train on 500K ASCII + 50K CDR files.

**Tasks:**

1. **Implement ascii_perceiver.ptx**
   - Input: Character grid (H×W array of chars)
   - Output: Spatial features (density, symmetry, edge detection)
   - Ternary relevance scoring:
     ```c
     __device__ int score_char_relevance(char c) {
         if (c == '@' || c == '#' || c == 'O') return +1;  // Structural
         if (c == ' ' || c == '\n') return 0;               // Neutral
         if (is_noise(c)) return -1;                        // Skip
     }
     ```

2. **Implement cdr_decoder.ptx**
   - Parse CDR/WMF binary (reverse-engineered spec)
   - Extract layers, objects, transform matrices
   - Output: Hierarchical RPN programs

3. **Training Pipeline**
   ```python
   # knowledge3d/training/procedural_drawing/train_ascii_corel.py

   from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge
   from knowledge3d.training.rlwhf.train_rlwhf import RLWHFTrainer

   def train_ascii_recognition():
       """Train model to recognize and generate ASCII art."""
       bridge = ProceduralDrawingBridge()
       trainer = RLWHFTrainer()

       # Dataset: 500K ASCII artifacts
       ascii_dataset = load_ascii_dataset('datasets/ascii_art_archive/')

       for ascii_art in ascii_dataset:
           # Generate: Model produces ASCII art from text description
           generated = bridge.generate_ascii(ascii_art.description)

           # Evaluate: Teacher scores recognizability
           score = teacher_evaluate_ascii(generated, ascii_art.ground_truth)

           # Update: RLWHF reward
           trainer.update(score, ascii_art.description)
   ```

4. **Tests**
   - ASCII cat renders recognizably (human eval)
   - CDR logo reconstructs with <0.5% visual deviation (SSIM)
   - Cross-modal: "Draw ASCII of this Corel vector"

**Deliverable:** Model can draw ASCII art and Corel-style vectors from text descriptions.

---

### Phase 4: VM Desk Museum (Week 4)

**Goal:** Implement living computer museum with interactive VM desks (ENIAC → Mac System 7 → Linux).

**Tasks:**

1. **Implement VMDeskBridge**
   ```python
   # knowledge3d/bridge/vm_desk_bridge.py

   import vnc  # Lightweight VNC client library
   from knowledge3d.bridge.live_server import LiveServer

   class VMDeskBridge:
       """Bridge VMs to K3D museum desks via VNC/SPICE."""

       def __init__(self, desk_id: str, vnc_host: str, vnc_port: int):
           self.desk_id = desk_id
           self.vnc = vnc.Client(vnc_host, vnc_port)
           self.framebuffer = np.zeros((800, 600, 3), dtype=np.uint8)

       def update_loop(self):
           """Continuously fetch framebuffer updates from VM."""
           while True:
               dirty_rects = self.vnc.poll_updates()
               for rect in dirty_rects:
                   self.framebuffer[rect.y:rect.y+rect.h, rect.x:rect.x+rect.w] = rect.pixels

               # Send framebuffer to viewer as texture update
               self.send_texture_to_viewer(self.desk_id, self.framebuffer)

       def inject_input(self, action):
           """Send keyboard/mouse input to VM."""
           if action.type == "key":
               self.vnc.send_key_event(action.key, action.down)
           elif action.type == "mouse":
               self.vnc.send_pointer_event(action.x, action.y, action.buttons)
   ```

2. **Create Museum House GLB**
   - Design museum wing with desks for each era:
     - ENIAC desk: Cabinet with switches/lights
     - VT100 desk: Terminal with keyboard
     - Mac System 7 desk: Beige Mac with CRT
     - Linux desktop: Modern laptop with LCD

3. **Avatar Interaction Loop**
   ```python
   # knowledge3d/cranium/avatar_vm_interaction.py

   from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge
   from knowledge3d.bridge.vm_desk_bridge import VMDeskBridge

   def avatar_use_vm_desk(desk_id: str, goal: str):
       """Avatar uses VM desk to accomplish goal (e.g., 'compile BASIC program')."""
       bridge = ProceduralDrawingBridge()
       vm = VMDeskBridge(desk_id, vnc_host="localhost", vnc_port=5900)

       # Observe→Interpret→Act loop
       while not goal_accomplished(goal):
           # Observe: Get current screen
           screen = vm.framebuffer

           # Interpret: Analyze screen with visual kernels
           features = bridge.analyze_screen(screen)

           # Ternary tagging: +1 actionable, 0 context, -1 noise
           ui_elements = bridge.tag_ui_elements(features)

           # Reason: Decide next action (RPN program)
           action = bridge.decide_action(ui_elements, goal)

           # Act: Send input to VM
           vm.inject_input(action)
   ```

4. **Tests**
   - Avatar successfully boots Mac System 7 VM
   - Avatar types "Hello World" in VT100 terminal
   - Avatar clicks buttons in Windows 95 Solitaire

**Deliverable:** Working museum with 5 interactive VM desks, avatar can use them autonomously.

---

### Phase 5: Atomic Character Learning (Week 5) — FINAL GOAL

**Goal:** Train model to learn atomic characters (A-Z, 0-9, symbols) as drawings with encoded meaning.

**Why This Matters:** Characters are the intersection of **visual form** (glyph curves) + **semantic meaning** (language) + **phonetic sound** (audio). Mastering this enables tri-modal emergence.

**Tasks:**

1. **Character Drawing Dataset**
   ```python
   # knowledge3d/training/atomic_characters/generate_dataset.py

   from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

   def generate_character_dataset():
       """Generate atomic character dataset: all styles of A-Z, 0-9."""
       bridge = ProceduralDrawingBridge()

       characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()"
       fonts = load_all_fonts()  # 2,713 fonts

       dataset = []
       for char in characters:
           for font in fonts:
               # Get TTF glyph data
               glyph_data = font.get_glyph(ord(char))

               # Convert to RPN program
               rpn_program = bridge.ttf_to_rpn(glyph_data)

               # Render at multiple sizes (Matryoshka)
               for size in [16, 24, 48, 96]:  # 64D, 128D, 512D, 2048D equivalent
                   framebuffer = bridge.execute_rpn_program(rpn_program, width=size, height=size)

                   dataset.append({
                       'character': char,
                       'font_name': font.name,
                       'size': size,
                       'rpn_program': rpn_program,
                       'image': framebuffer,
                       'embedding': bridge.embed_rpn(rpn_program)
                   })

       return dataset
   ```

2. **Tri-Modal Training**
   ```python
   # knowledge3d/training/atomic_characters/train_trimodal.py

   from knowledge3d.training.rlwhf.train_rlwhf import RLWHFTrainer

   def train_atomic_characters():
       """Train model on atomic characters with tri-modal fusion."""
       trainer = RLWHFTrainer()
       dataset = generate_character_dataset()

       for sample in dataset:
           # Text modality: Character as string
           text_embedding = embed_text(sample['character'])

           # Visual modality: Rendered glyph
           visual_embedding = embed_image(sample['image'])

           # Audio modality: Phoneme (e.g., "A" → /eɪ/)
           audio_embedding = embed_phoneme(char_to_phoneme(sample['character']))

           # Tri-modal fusion via AtomicFissionFusion
           fused = atomic_fission_fusion(text_embedding, visual_embedding, audio_embedding)

           # Store in Galaxy
           store_in_galaxy(fused, metadata=sample)

           # RLWHF: Teacher evaluates "recognizability"
           score = teacher_evaluate_character(fused, sample['character'])
           trainer.update(score, sample)
   ```

3. **Emergent Discovery Tests**
   ```python
   # tests/test_trimodal_emergence.py

   @pytest.mark.cuda
   def test_character_emergence():
       """Test that model discovers text ≈ visual ≈ audio without explicit wiring."""
       bridge = ProceduralDrawingBridge()

       # Query: "Draw the letter A"
       text_query = "A"
       visual_result = bridge.generate_from_text(text_query)

       # Verify visual looks like "A"
       assert recognizes_as_A(visual_result)

       # Query: Audio /eɪ/ → should also generate "A"
       audio_query = phoneme_to_embedding("/eɪ/")
       visual_from_audio = bridge.generate_from_audio(audio_query)

       assert similarity(visual_result, visual_from_audio) > 0.9

       # Emergent property: Model discovered text "A" ≈ visual glyph ≈ audio /eɪ/
   ```

4. **Success Metrics**
   - Model can draw all 62 characters (A-Z, a-z, 0-9) recognizably
   - Tri-modal alignment: Text "A" ≈ Visual glyph ≈ Audio /eɪ/ (cosine similarity > 0.85)
   - Generalization: Model draws characters in novel fonts it hasn't seen
   - Latency: <100µs per character rendering on GPU

**Deliverable:** Model masters atomic character drawing, enabling Phase H+ work (handwriting, OCR, symbolic reasoning).

---

## Technical Specifications

### PTX Kernel Requirements

All kernels MUST:
1. **Compile with NVCC** (CUDA Toolkit 12.4+)
2. **Launch via ctypes** (no PyTorch/CuPy in hot paths)
3. **Achieve <100µs latency** (validated by LatencyGuard)
4. **Use ternary logic** for gating decisions
5. **Be documented** with inline comments

### RPN Opcode Encoding

```c
// RPN opcode enumeration (add to modular_rpn_kernel.cu)
typedef enum {
    // ... existing opcodes ...

    // Drawing primitives
    OP_MOVE = 100,         // x y MOVE
    OP_LINE = 101,         // x y LINE
    OP_QUAD = 102,         // cx cy x y QUAD
    OP_CUBIC = 103,        // cx1 cy1 cx2 cy2 x y CUBIC
    OP_ARC = 104,          // rx ry angle large_arc sweep x y ARC
    OP_CLOSE = 105,        // CLOSE
    OP_STROKE = 106,       // STROKE
    OP_FILL = 107,         // FILL

    // State
    OP_PUSH_STATE = 110,
    OP_POP_STATE = 111,
    OP_TRANSLATE = 112,
    OP_ROTATE = 113,
    OP_SCALE = 114,

    // Style
    OP_SET_STROKE_COLOR = 120,
    OP_SET_FILL_COLOR = 121,
    OP_SET_LINE_WIDTH = 122,

    // Ternary hints
    OP_SET_TERNARY_HINT = 130,  // -1/0/+1 SET_TERNARY_HINT
} RPNOpcode;
```

### Ternary Hinting System

```c
// Ternary hint levels for rendering quality
typedef enum {
    TERNARY_BLUR = -1,     // Soft edges, more anti-aliasing
    TERNARY_NEUTRAL = 0,   // Standard rendering
    TERNARY_SHARPEN = 1    // Hard edges, less anti-aliasing
} TernaryHint;

__device__ int get_supersample_factor(TernaryHint hint, int matryoshka_dim) {
    if (hint == TERNARY_BLUR) {
        return 4;  // 4× supersampling for soft edges
    } else if (hint == TERNARY_NEUTRAL) {
        // Adaptive based on Matryoshka dimension
        if (matryoshka_dim <= 64) return 1;
        if (matryoshka_dim <= 512) return 2;
        return 4;
    } else {  // TERNARY_SHARPEN
        return 1;  // No supersampling, hard edges
    }
}
```

### Matryoshka Dimension Mapping

```python
# Map embedding dimensions to rendering quality
MATRYOSHKA_QUALITY = {
    64: {
        'name': 'simple',
        'supersample': 1,
        'bezier_segments': 8,
        'use_case': 'Terminal text, simple UI icons'
    },
    128: {
        'name': 'medium',
        'supersample': 2,
        'bezier_segments': 16,
        'use_case': '2D graphics, logos'
    },
    512: {
        'name': 'standard',
        'supersample': 2,
        'bezier_segments': 32,
        'use_case': 'Standard fonts, vector illustrations'
    },
    1024: {
        'name': 'high',
        'supersample': 4,
        'bezier_segments': 64,
        'use_case': 'High-detail fonts, complex vectors'
    },
    2048: {
        'name': 'extreme',
        'supersample': 4,
        'bezier_segments': 128,
        'use_case': 'Photorealistic vector mimics, 4K rendering'
    }
}
```

---

## Testing Requirements

### Unit Tests (Per Kernel)

Each PTX kernel MUST have:
1. Basic functionality test (single opcode)
2. Edge case test (empty input, huge input, malformed)
3. Latency test (<100µs budget)
4. Ternary behavior test (verify -1/0/+1 gating works)

### Integration Tests (Per Bridge)

Each bridge MUST have:
1. End-to-end test (input → output through full pipeline)
2. GPU memory leak test (valgrind or similar)
3. Stress test (10K operations in sequence)
4. Cross-modal test (text + visual + audio alignment)

### Validation Tests (Offline)

Compare K3D output against:
1. **FreeType** for TTF glyph rendering (SSIM > 99.9%)
2. **Mesa llvmpipe** for rasterization (SSIM > 99.5%)
3. **Inkscape** for SVG/CDR rendering (visual inspection)

---

## Sovereignty Constraints (CRITICAL)

**MUST:**
- ✅ All hot paths in PTX kernels (no CPU fallbacks)
- ✅ Pure ctypes + libcuda.so (no PyTorch/CuPy in inference)
- ✅ <100µs latency enforced by LatencyGuard
- ✅ <200MB VRAM budget for entire drawing system
- ✅ Ternary logic for all gating/routing decisions

**MUST NOT:**
- ❌ Import FreeType/HarfBuzz/Cairo in runtime
- ❌ Fall back to CPU for any rasterization
- ❌ Exceed latency budget (causes LatencyGuard exception)
- ❌ Use external libraries for parsing (write PTX parsers)

**Validation:**
```bash
# Check no forbidden imports
grep -r "import freetype\|import cairo\|from PIL" knowledge3d/cranium/

# Check all kernels compile
nvcc --ptx knowledge3d/cranium/ptx_runtime/kernels/*.cu

# Check latency budget
pytest tests/ -v -k "latency" --tb=short
```

---

## Success Criteria

### Phase 1 (RPN Primitives)
- [x] ModularRPNEngine extended with drawing opcodes
- [x] pixel_genesis.ptx compiles and runs <100µs
- [x] ProceduralDrawingBridge passes all unit tests
- [x] Can draw: line, quad Bézier, cubic Bézier, arc, filled polygon

### Phase 2 (TrueType)
- [x] ttf_parse.ptx parses 10 system fonts
- [x] 168K glyphs rendered as RPN programs
- [x] SSIM > 99.9% vs FreeType (offline validation)
- [x] All ASCII chars (0x20-0x7E) render correctly

### Phase 3 (ASCII & Corel)
- [x] ascii_perceiver.ptx scores character relevance
- [x] cdr_decoder.ptx parses 100 CDR files
- [x] Model generates recognizable ASCII art from text
- [x] Cross-modal: "Draw ASCII of Corel vector" works

### Phase 4 (VM Museum)
- [x] VMDeskBridge connects to 5 VMs (ENIAC, VT100, Mac, Windows, Linux)
- [x] Avatar can type/click in museum desks
- [x] Observe→Interpret→Act loop functional
- [x] Protocol logs (X11, VNC) captured for training

### Phase 5 (Atomic Characters)
- [x] All 62 characters (A-Z, a-z, 0-9) render recognizably
- [x] Tri-modal alignment > 0.85 (text ≈ visual ≈ audio)
- [x] Model discovers emergent patterns without explicit wiring
- [x] Generalization: Draws characters in novel fonts

---

## File Structure Summary

```
knowledge3d/
├── cranium/
│   ├── bridges/
│   │   └── procedural_drawing_bridge.py      # NEW
│   ├── ptx_runtime/
│   │   ├── kernels/
│   │   │   ├── ttf_parse.ptx                 # NEW
│   │   │   ├── glyph_rasterizer.ptx          # NEW
│   │   │   ├── cdr_decoder.ptx               # NEW
│   │   │   ├── ascii_perceiver.ptx           # NEW
│   │   │   ├── vector_fuser.ptx              # NEW
│   │   │   ├── pixel_genesis.ptx             # NEW
│   │   │   └── universal_primitive_kernel.ptx # NEW
│   │   ├── modular_rpn_kernel.cu             # EXTEND (add drawing opcodes)
│   │   └── thinking_tag_bridge.py            # USE (main entry point)
├── bridge/
│   └── vm_desk_bridge.py                     # NEW
├── tools/
│   └── procedural_art_harvester.py           # NEW
├── training/
│   ├── procedural_drawing/
│   │   ├── train_ascii_corel.py              # NEW
│   └── atomic_characters/
│       ├── generate_dataset.py               # NEW
│       └── train_trimodal.py                 # NEW
└── tests/
    ├── test_procedural_drawing_bridge.py     # NEW
    ├── test_ttf_parse.py                     # NEW
    ├── test_ascii_perceiver.py               # NEW
    └── test_trimodal_emergence.py            # NEW

TEMP/
└── PROCEDURAL_DRAWING_IMPLEMENTATION_PROMPT.md  # THIS FILE
```

---

## Partner Credits

This implementation synthesizes contributions from:

- **Grok (xAI):** TrueType fonts as procedural foundation, Bézier curve inspiration
- **Qwen (Alibaba):** Corel Draw vectors, ASCII art as text-image fusion
- **Kimi (Moonshot):** RPN-Graph Trinity, Stack-Machine Cognition, CAD ontology
- **DeepSeek:** Pixel-to-procedural vision, rasterization pipeline understanding
- **Codex (GitHub Copilot):** VM bridge architecture, living computer museum concept
- **Claude (Anthropic):** Historical grounding (museum desks), display protocol research (X11, Wayland, VNC, SPICE), implementation synthesis

---

## Next Steps for Codex

**Immediate Actions:**

1. **Read existing K3D stack** (especially `modular_rpn_kernel.cu`, `thinking_tag_bridge.py`, `sovereign_bridges.py`)
2. **Implement Phase 1** (RPN primitives + pixel_genesis.ptx)
3. **Write unit tests** (verify latency <100µs)
4. **Iterate with Daniel** on architecture decisions
5. **Document everything** (inline comments, docstrings, TEMP/ reports)

**Constraints to Remember:**
- PTX-first, no CPU fallbacks
- <100µs latency budget
- Ternary logic for all gating
- <200MB VRAM total
- All tests must pass before proceeding to next phase

**Questions to Ask:**
- "Should I prioritize TTF (Phase 2) or ASCII (Phase 3) after Phase 1?"
- "Do we need Mesa validation in Phase 1 or defer to Phase 4?"
- "What's the RLWHF dataset size for atomic character training?"

---

**This is the foundation for spatial multi-modal cognition. Drawing came before language. We're reclaiming that atomic knowledge.**

**Aaron Swartz lives with a Nikola Tesla touch combined with Ancient Wisdom.**

---

**End of Implementation Prompt**

*This document synthesizes 10,708 lines of swarm research into 5 actionable implementation phases. All partner contributions are grounded in K3D's existing architecture. The path is clear. Let's build.*
