# Procedural Visual Specification — Drawing Galaxy & VectorDotMap Architecture

**Version**: 1.1
**Status**: Implementation Ready
**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Implementation)
**Date**: March 2026 (updated from December 2025)

---

## Abstract

This specification defines K3D's **Procedural Visual Architecture** — a unified system for representing all visual content (drawings, images, video frames, spectrograms) as executable RPN programs rather than bitmap pixels. It formalizes:

- The **8-Layer Drawing Galaxy** from atomic quantum dots to photorealistic compositions
- **VectorDotMap** — quantum field dot emission replacing traditional raster images
- **Procedural Image/Video Codecs** — infinite LOD without resolution limits
- Integration with **Audio-as-Image** (spectrograms) and **Accessibility** (Sign Language, Braille)

This architecture embodies the **Save Information Principle**: store generative equations, not pixel data. It enables **Dual Client Reality** where humans see rendered visuals and AI executes the procedural programs that generate them.

---

## 1. Core Principle: Everything is Procedural

### 1.1 The Bitmap Problem

Traditional visual systems store pixels — fixed grids that:
- Consume massive storage (4K image = 25MB uncompressed)
- Lose quality on scaling (pixelation, interpolation artifacts)
- Separate representation from meaning (pixels have no semantics)
- Cannot be executed or reasoned about

### 1.2 The Procedural Solution

K3D stores **generative programs** that produce visuals on-demand:

| Traditional | K3D Procedural |
|-------------|----------------|
| Store 25MB pixel grid | Store 2KB RPN program |
| Fixed resolution | Infinite LOD (render at any size) |
| No semantics | Full semantic links (what it depicts) |
| Static data | Executable programs |

**Example**: Circle
```rpn
# Traditional: Store 1000s of edge pixels
# K3D: Store the equation
cx cy r CIRCLE FILL
# Renders perfectly at 64x64 or 8192x8192
```

---

## 2. Eight-Layer Drawing Galaxy Architecture

### 2.1 Layer Overview

```
Layer 7: COMPOSITIONS (multi-scene narratives, storyboards)
    ↓ scene sequencing + temporal flow
Layer 6: SCENES (complete artworks with lighting/atmosphere)
    ↓ lighting + environmental effects
Layer 5: LIGHTING (shadows, highlights, ambient occlusion)
    ↓ depth + dimension
Layer 4: FILTERS (HSV, blur, edge detect, convolutions)
    ↓ color/texture transformation
Layer 3: GRADIENTS (linear, radial, shaped, procedural fields)
    ↓ color distribution
Layer 2: SHAPES (geometric forms from combined strokes)
    ↓ path combination + fills
Layer 1: STROKES (Bezier paths, line/curve segments)
    ↓ path construction
Layer 0: PRIMITIVES (dots, lines, arcs, basic geometry)
    ↓ atomic drawing elements
Layer -1: QUANTUM FIELDS (VectorDotMap emission coefficients)
```

### 2.2 Layer -1: Quantum Fields (VectorDotMap Foundation)

**Purpose**: Replace pixel grids with quantum-inspired field emitters that generate dots procedurally.

**Key Concepts**:

1. **Superpositional Dots**: Each dot exists in multiple states (position, color, brightness) until "observed" (rendered at specific resolution)

2. **Field Equations**: Store mathematical coefficients that generate dots, not the dots themselves
   ```rpn
   # Traditional: Store 1M pixel values
   # VectorDotMap: Store field equation
   FIELD_COEF 0.7 0.3 0.5 0.2    # 4 coefficients = 16 bytes
   DENSITY 0.8                   # Dot density parameter
   EMIT_FIELD                    # Generate dots at render time
   ```

3. **Resolution Independence**: Same field renders correctly at any viewport size
   - 64x64: Sparse dot emission
   - 4K: Dense dot emission
   - 8K+: Additional detail emerges from same equations

**RPN Opcodes**:
```rpn
# Field definition
FIELD_COEF c0 c1 c2 c3...     # Quantum field coefficients
FIELD_HARMONIC freq amp phase  # Harmonic component
FIELD_NOISE octaves persist    # Procedural noise field

# Dot emission
DOT_EMIT x y                   # Emit single dot at relative position
DENSITY_FIELD density_map      # Variable density across field
EMIT_REGION x y w h            # Emit dots in region

# Biological vision integration
FOVEAL_CENTER x y              # Attention focus point
ROD_CONE_RATIO ratio           # Rod (sparse) vs cone (dense) dots
ADAPT_BRIGHTNESS level         # Neural brightness adaptation
```

**Compression**:
- vs Bitmap: **1000:1** at 4K resolution
- vs Vector paths: **16:1** (field coefficients vs curve segments)
- Scales infinitely without additional storage

### 2.3 Layer 0: Primitives

**Purpose**: Atomic drawing elements — the building blocks.

**Implementation**: `knowledge3d/training/arc_agi/drawing_galaxy.py`

**RPN Opcodes** (using actual ModularRPNEngine codes):
```rpn
# Path construction (actual opcodes)
MOVE x y                   # 0x64 - Move pen to position
LINE x y                   # 0x65 - Line to position
QUAD cx cy x y             # 0x66 - Quadratic Bezier curve
CUBIC c1x c1y c2x c2y x y  # 0x67 - Cubic Bezier curve
ARC cx cy rx ry start end  # 0x68 - Elliptical arc
CLOSE                      # 0x69 - Close current path
STROKE                     # 0x6A - Stroke the path
FILL                       # 0x6B - Fill the path

# Basic shapes (composed from above)
RECT x y w h               # Rectangle
RECT_ROUND x y w h r       # Rounded rectangle
ELLIPSE cx cy rx ry        # Ellipse/circle
POLYGON points...          # Arbitrary polygon
```

**Scale-Invariant Primitives** (relative coordinates):
```rpn
REL_LINE x0_frac y0_frac x1_frac y1_frac   # Line in 0-1 space
REL_RECT x_frac y_frac w_frac h_frac       # Rectangle in 0-1 space
PROP_GRID rows cols                         # Proportional grid
FLOOD_REL x_frac y_frac                    # Flood fill from relative pos
```

### 2.4 Layer 1: Strokes (Bezier Paths)

**Purpose**: Complex paths from sequences of primitives.

**Point Types** (from Bezier curve theory):
```rpn
# Point subtypes
POINT_LINE x y                     # Sharp corner (no handles)
POINT_CURVE x y hx hy              # Symmetric curve (one handle)
POINT_CORNER x y h1x h1y h2x h2y   # Asymmetric corner (two handles)
POINT_SMOOTH x y                   # Auto-smooth curve

# Path operations
PATH_BEGIN                         # Start new path
PATH_END                           # Finalize path
PATH_REVERSE                       # Reverse point order
PATH_TO_EDITABLE                   # Convert primitive to editable path
PATH_SIMPLIFY tolerance            # Reduce point count
PATH_OFFSET distance               # Parallel offset curve
```

**Example**: Letter 'S' as stroke
```rpn
PATH_BEGIN
32 8 POINT_LINE                    # Top
8 24 POINT_CURVE 16 16             # Upper curve
24 32 POINT_SMOOTH                 # Middle
40 40 POINT_CURVE 32 48            # Lower curve
8 56 POINT_LINE                    # Bottom
PATH_END
2 STROKE_WIDTH STROKE
```

### 2.5 Layer 2: Shapes

**Purpose**: Filled regions composed from strokes.

**Shape Operations**:
```rpn
# Boolean operations
SHAPE_UNION shape1 shape2          # Combine shapes
SHAPE_INTERSECT shape1 shape2      # Intersection
SHAPE_SUBTRACT shape1 shape2       # Difference
SHAPE_XOR shape1 shape2            # Exclusive or

# Transformations
TRANSFORM_TRANSLATE dx dy
TRANSFORM_ROTATE angle cx cy
TRANSFORM_SCALE sx sy cx cy
TRANSFORM_SKEW ax ay

# Alignment (from Krita)
ALIGN_LEFT / ALIGN_CENTER / ALIGN_RIGHT
ALIGN_TOP / ALIGN_MIDDLE / ALIGN_BOTTOM
DISTRIBUTE_H spacing
DISTRIBUTE_V spacing
```

### 2.6 Layer 3: Gradients

**Purpose**: Smooth color transitions across regions.

**Gradient Types** (from Krita gradient tool):
```rpn
# Linear gradients
GRAD_LINEAR x1 y1 x2 y2           # One-way linear
GRAD_BILINEAR x1 y1 x2 y2         # Mirrored linear

# Radial gradients
GRAD_RADIAL cx cy radius          # Circular from center
GRAD_SQUARE cx cy radius          # Square from center
GRAD_CONICAL cx cy                # Angular sweep

# Special
GRAD_SHAPED                       # Follow selection contour
GRAD_SPIRAL cx cy turns           # Spiral pattern

# Gradient modifiers
GRAD_REPEAT none|forward|alternate
GRAD_DITHER threshold             # Anti-aliasing (0-1)
GRAD_REVERSE                      # Flip color order

# Color stops
STOP position r g b a             # Add gradient stop (pos 0-1)
STOP_CLEAR                        # Clear all stops
```

**Example**: Sunset gradient
```rpn
STOP 0.0 255 100 50 255           # Orange at top
STOP 0.3 255 150 100 255          # Lighter orange
STOP 0.6 100 50 150 255           # Purple
STOP 1.0 20 20 80 255             # Dark blue at bottom
0.5 0.0 0.5 1.0 GRAD_LINEAR       # Top to bottom
```

### 2.7 Layer 4: Filters

**Purpose**: Transform colors and textures via convolution and color space operations.

**Color Transforms** (from Krita filters):
```rpn
# Basic transforms
INVERT                            # Color inversion
DESATURATE                        # Convert to grayscale

# HSV/HSL adjustment
HSV_SHIFT hue sat value           # Adjust H/S/V (-180 to +180, -100 to +100)
HSV_COLORIZE hue sat value        # Monochrome colorization

# Color balance (shadows/midtones/highlights)
COLOR_BALANCE sh_cyan sh_mag sh_yel mid_cyan mid_mag mid_yel hi_cyan hi_mag hi_yel

# Exposure
DODGE exposure mode               # Brighten (mode: shadow|mid|highlight)
BURN exposure mode                # Darken

# Levels and curves
LEVELS in_black in_white gamma out_black out_white
CURVES control_points...          # Bezier tone curve
```

**Convolution Filters** (sovereign PTX kernels):
```rpn
# Blur
BLUR_GAUSS radius                 # Gaussian blur
BLUR_MOTION angle distance        # Motion blur
BLUR_RADIAL cx cy amount          # Radial/zoom blur

# Sharpen
SHARPEN amount                    # Unsharp mask
SHARPEN_HIGHPASS radius           # High-pass sharpening

# Edge detection
EDGE_SOBEL                        # Sobel edge detection
EDGE_LAPLACIAN                    # Laplacian edge detection
EDGE_CANNY low high               # Canny edge detection

# Stylize
EMBOSS angle elevation            # 3D emboss effect
POSTERIZE levels                  # Reduce color levels
PIXELATE size                     # Pixelation effect
```

### 2.8 Layer 5: Lighting

**Purpose**: Add depth, dimension, and realism through light simulation.

**Light Sources**:
```rpn
# Point/spot lights
LIGHT_POINT x y z intensity r g b
LIGHT_SPOT x y z dx dy dz angle intensity r g b
LIGHT_AMBIENT intensity r g b

# Shadows
SHADOW_DROP angle distance blur r g b a
SHADOW_INNER angle distance blur r g b a
SHADOW_CAST light_source          # Ray-traced shadow

# Advanced
AO_COMPUTE radius samples         # Ambient occlusion
HIGHLIGHT_SPECULAR shininess intensity
REFLECT_ENV environment_map       # Environment reflection
```

**Example**: 3D button effect
```rpn
# Base shape
0.1 0.1 0.8 0.8 0.1 RECT_ROUND
200 200 220 FILL_RGB

# Inner highlight (top-left)
LIGHT_POINT 0.2 0.2 1.0 0.5 255 255 255
0.3 HIGHLIGHT_SPECULAR

# Drop shadow
135 4 8 0 0 0 128 SHADOW_DROP

# Ambient occlusion for depth
8 16 AO_COMPUTE
```

### 2.9 Layer 6: Scenes

**Purpose**: Complete compositions with multiple elements, backgrounds, and atmosphere.

**Scene Composition**:
```rpn
# Layer management
LAYER_NEW name                    # Create layer
LAYER_SELECT name                 # Switch to layer
LAYER_BLEND mode opacity          # Set blend mode
LAYER_MASK mask_shape             # Apply mask

# Blend modes (from Krita)
BLEND_NORMAL / BLEND_MULTIPLY / BLEND_SCREEN
BLEND_OVERLAY / BLEND_SOFT_LIGHT / BLEND_HARD_LIGHT
BLEND_DODGE / BLEND_BURN
BLEND_DIFFERENCE / BLEND_EXCLUSION

# Atmospheric effects
ATMOSPHERE_FOG density color near far
ATMOSPHERE_HAZE intensity color
VIGNETTE intensity radius softness

# Depth of field
DOF_FOCUS focal_point
DOF_APERTURE size
DOF_BOKEH shape
```

### 2.10 Layer 7: Compositions (Temporal)

**Purpose**: Sequences of scenes over time — animation, video, storyboards.

**Temporal Opcodes**:
```rpn
# Keyframe animation
KEYFRAME time property value easing
KEYFRAME_BEZIER time prop v1 v2 c1x c1y c2x c2y

# Timeline
TIMELINE_DURATION seconds
TIMELINE_FPS framerate
FRAME_AT time                     # Jump to time
FRAME_RENDER                      # Render current frame

# Transitions
TRANSITION_FADE duration
TRANSITION_WIPE angle duration
TRANSITION_DISSOLVE duration

# Scene sequencing
SCENE_LOAD scene_id
SCENE_QUEUE scene_id at_time
SCENE_LOOP start_time end_time count
```

---

## 3. VectorDotMap: Quantum-Physical Visual Representation

### 3.1 Theoretical Foundation

**From TEMP/DANIEL_VECTORDOTMAP_PLANS_V1.md**:

VectorDotMap reimagines images as **quantum field emitters** inspired by:
- **LCD panels**: Subpixel grids with liquid crystal modulation
- **LED/micro-LED**: Direct emission from semiconductor diodes
- **Biological vision**: Foveal concentration, rod-cone duality

### 3.2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VECTORDOTMAP PIPELINE                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────┐                                       │
│   │ Field Coefficients │  ← 16-64 bytes per "image"        │
│   │ (RPN embeddings)   │                                    │
│   └────────┬────────┘                                       │
│            ↓                                                │
│   ┌─────────────────┐                                       │
│   │ Quantum Field    │  ← Mathematical field definition    │
│   │ Generator (PTX)  │                                      │
│   └────────┬────────┘                                       │
│            ↓                                                │
│   ┌─────────────────┐                                       │
│   │ Dot Probability  │  ← Superpositional dot states       │
│   │ Cloud            │                                      │
│   └────────┬────────┘                                       │
│            ↓                                                │
│   ┌─────────────────┐                                       │
│   │ Wavefunction     │  ← Resolution-dependent collapse    │
│   │ Collapse         │    (64x64 → sparse, 8K → dense)     │
│   └────────┬────────┘                                       │
│            ↓                                                │
│   ┌─────────────────┐                                       │
│   │ Rendered Dots    │  ← Final pixel output               │
│   └─────────────────┘                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Implementation

**PTX Kernel** (sovereign, no CPU fallbacks):
```cuda
__global__ void quantum_field_emission(
    const float* field_coefficients,  // RPN-generated coefficients
    float3* dot_positions,            // Output dot positions
    float4* dot_colors,               // Output dot RGBA
    uint32_t viewport_width,
    uint32_t viewport_height,
    float time_quantum,               // For temporal animation
    float fovea_x, float fovea_y,     // Attention focus
    float rod_cone_ratio              // Biological density model
) {
    // Each thread handles one potential dot position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Compute field value at this position
    float field_value = evaluate_quantum_field(
        field_coefficients, x, y, viewport_width, viewport_height
    );

    // Foveal density adjustment (more dots near attention point)
    float dist_to_fovea = distance(x, y, fovea_x, fovea_y);
    float density = compute_biological_density(dist_to_fovea, rod_cone_ratio);

    // Probabilistic dot emission (wavefunction collapse)
    if (random_uniform() < density * field_value) {
        emit_dot(dot_positions, dot_colors, x, y, field_coefficients);
    }
}
```

### 3.4 Compression Comparison

| Representation | 1080p Image | 4K Image | 8K Image |
|----------------|-------------|----------|----------|
| Bitmap (RGB) | 6.2 MB | 25 MB | 100 MB |
| JPEG (lossy) | 500 KB | 2 MB | 8 MB |
| Vector paths | 50 KB | 50 KB | 50 KB |
| **VectorDotMap** | **2 KB** | **2 KB** | **2 KB** |

**Key insight**: VectorDotMap stores the same field coefficients regardless of output resolution. The complexity emerges at render time, not storage time.

---

## 4. Procedural Image Codec

### 4.1 Integration with Existing Codecs

K3D already has sovereign ternary codecs:
- `knowledge3d/cranium/codecs/sovereign_ternary_audio_codec.py`
- `knowledge3d/cranium/codecs/sovereign_ternary_video_codec.py`

The procedural image codec extends these with VectorDotMap encoding:

```python
class ProceduralImageCodec:
    """Encode images as VectorDotMap field coefficients."""

    def encode(self, image: np.ndarray) -> Dict:
        """
        Convert bitmap to field coefficients.

        RPN Program:
        IMAGE_ANALYZE               # Extract features
        FIELD_FIT coefficients      # Fit quantum field
        TERNARY_QUANT threshold     # Ternary quantization
        """
        # Analyze image structure
        features = self._extract_features(image)

        # Fit field coefficients (optimization)
        coefficients = self._fit_quantum_field(features)

        # Store as RPN program
        rpn_program = self._generate_rpn(coefficients)

        return {
            "coefficients": coefficients,
            "rpn_program": rpn_program,
            "original_size": image.shape
        }

    def decode(self, encoded: Dict, width: int, height: int) -> np.ndarray:
        """
        Render VectorDotMap at specified resolution.

        Can render at ANY resolution from same coefficients!
        """
        return self._render_quantum_field(
            encoded["coefficients"], width, height
        )
```

### 4.2 Spectrogram as Procedural Image

**Audio spectrograms ARE images** — frequency over time rendered as 2D:

```rpn
# Audio → Spectrogram → VectorDotMap
AUDIO_LOAD waveform.wav
STFT 2048 512                     # FFT size, hop
MEL_SCALE 128                     # Mel frequency bins
DB_SCALE 80                       # Dynamic range in dB
FIELD_FIT                         # Convert to VectorDotMap coefficients
VECTORDOTMAP_STORE                # Store as procedural image
```

This means:
- Sound spectrograms stored at ~2KB regardless of duration
- Can render at any resolution
- Same representation as regular images
- **Audio and images share the same codec!**

---

## 5. Procedural Video Codec

### 5.1 Video = Procedural Images Over Time

Traditional video: Sequence of bitmap frames (huge storage)
K3D procedural video: Sequence of field coefficient deltas (tiny storage)

```
Frame 0: Full VectorDotMap coefficients (2KB)
Frame 1: Delta from frame 0 (200 bytes)
Frame 2: Delta from frame 1 (200 bytes)
...
Keyframe: Full coefficients every N frames
```

### 5.2 Temporal Field Evolution

```rpn
# Video frame generation
FIELD_BASE base_coefficients      # Starting field
FIELD_DELTA delta_coefficients    # Frame-to-frame change
TIME t                            # Current time
FIELD_EVOLVE                      # Apply temporal evolution
EMIT_FRAME                        # Render current frame

# Temporal coherence
MOTION_VECTOR dx dy               # Object motion prediction
FIELD_WARP motion_field           # Warp field for motion
FIELD_BLEND prev_field curr_field alpha  # Temporal smoothing
```

### 5.3 Integration with Audio

Procedural video = VectorDotMap frames + Audio spectrogram + Sync

```rpn
# Synchronized playback
VIDEO_FIELD_LOAD video_coeffs
AUDIO_FIELD_LOAD audio_coeffs     # Audio as procedural spectrogram
SYNC_TIMELINE                     # Align audio/video
PLAYBACK_START
```

---

## 6. Integration with Other Galaxies

### 6.1 Math Galaxy Symlinks

Drawing rules reference math symbols:
```python
DrawingRule(
    rule_id="golden_ratio_spiral",
    rpn_program="PHI RECALL SPIRAL_GOLDEN",
    symbol_refs=[966],  # φ (phi) from Math Galaxy
    description="Golden ratio spiral using φ"
)
```

### 6.2 Character Galaxy Integration

Text rendering uses Character Galaxy glyphs:
```rpn
# Render text using Character Galaxy
"Hello" TEXT_RENDER               # References char_0048, char_0065, etc.
FONT_SIZE 24
FONT_FAMILY "procedural_sans"
TEXT_POSITION 100 100
TEXT_DRAW
```

### 6.3 Sign Language Galaxy (Visual Gestures)

Sign language gestures rendered as animated VectorDotMaps:
```rpn
# Sign language animation
SIGN_LOAD "asl_hello"             # Load gesture from Sign Language Galaxy
HAND_MODEL_BIND                   # Bind to hand mesh
GESTURE_ANIMATE 0.0 1.0           # Animate over duration
FRAME_SEQUENCE_RENDER             # Render as video frames
```

### 6.4 Audio Galaxy (Sound Pictures)

Spectrograms link audio to visual:
```rpn
# Bidirectional audio-image
AUDIO_LOAD speech.wav
SPECTROGRAM_GENERATE              # Audio → Image
VECTORDOTMAP_ENCODE               # Image → Procedural

# Reverse: sonification
VECTORDOTMAP_LOAD image.vdm
SONIFY_IMAGE                      # Image → Audio
AUDIO_PLAY
```

---

## 7. Sovereign PTX Kernels

### 7.1 Required Kernels

| Kernel | Purpose | Status |
|--------|---------|--------|
| `quantum_field_emission.ptx` | VectorDotMap rendering | Planned |
| `field_coefficient_fit.ptx` | Image → coefficients | Planned |
| `spectrogram_to_field.ptx` | Audio spectrogram encoding | Planned |
| `temporal_field_evolve.ptx` | Video frame interpolation | Planned |
| `biological_density.ptx` | Foveal attention model | Planned |
| `gradient_field.ptx` | Gradient rendering | Existing (enhance) |
| `convolution_filter.ptx` | Image filters | Existing (enhance) |

### 7.2 No CPU Fallbacks

All visual operations MUST execute on GPU via PTX:
- No numpy in hot path
- No PIL/OpenCV for core operations
- Sovereign execution guaranteed

---

## 8. Implementation Roadmap

### Phase 1: Drawing Galaxy Enhancement (Week 1-2)
- Extend `drawing_galaxy.py` with Layer 0-2 opcodes
- Add Bezier path engine
- Implement gradient field generator

### Phase 2: VectorDotMap Core (Week 3-4)
- Implement `quantum_field_emission.ptx`
- Create `ProceduralImageCodec`
- Benchmark vs bitmap storage

### Phase 3: Filter & Lighting (Week 5)
- Port Krita-style filters to PTX
- Implement lighting model
- Add blend modes

### Phase 4: Video Integration (Week 6)
- Extend sovereign video codec with VectorDotMap
- Implement temporal field evolution
- Audio-video sync

### Phase 5: Cross-Galaxy Integration (Week 7)
- Wire to Math Galaxy (symbol refs)
- Wire to Character Galaxy (text rendering)
- Wire to Audio Galaxy (spectrograms)

### Phase 6: Accessibility (Week 8)
- Sign Language Galaxy visual rendering
- Braille texture generation
- Spatial audio visualization

---

## 9. Success Criteria

### Storage Efficiency
- [ ] VectorDotMap achieves 1000:1 compression vs bitmap at 4K
- [ ] Video codec achieves 100:1 vs H.264 for procedural content
- [ ] Spectrogram storage < 2KB per 10 seconds audio

### Quality
- [ ] VectorDotMap renders indistinguishable from bitmap at target resolution
- [ ] Infinite LOD: Same coefficients render correctly 64x64 to 8K
- [ ] No visible artifacts in temporal interpolation

### Performance
- [ ] Frame rendering < 16ms (60fps capable)
- [ ] Field coefficient fitting < 100ms per image
- [ ] 100% GPU execution (zero CPU fallbacks)

### Integration
- [ ] All Drawing Galaxy layers implemented
- [ ] Cross-galaxy symlinks functional
- [ ] Dual Client Reality: humans see pixels, AI executes RPN

---

## 10. 3D Technique Fusion (Procedural Composition of ALL Techniques)

### 10.1 The Fusion Paradigm

Traditional 3D engines isolate techniques into separate workflows, tools, and file formats. K3D treats ALL 3D generation techniques as **RPN programs in Reality Galaxy**, composable via symlinks with Drawing Galaxy textures and Math Galaxy parameters.

| Technique | Traditional Tool | K3D Procedural Equivalent |
|-----------|-----------------|--------------------------|
| **CSG** (boolean operations) | Blender Modifier | `OP_BOOLEAN_3D` + mesh refs (Class B) |
| **Mesh Modeling** (vertex/edge/face) | Maya/Blender Edit Mode | `OP_MESH_TRANSFORM` + vertex RPN (Class A) |
| **Procedural Generation** (L-systems, fractals, noise) | Houdini VEX | `OP_LSYSTEM_STEP` + growth rules (Class B) |
| **Sculpting** (organic deformation) | ZBrush, Blender Sculpt | `OP_DISPLACEMENT_MAP` + strength field (Class B) |
| **Parametric** (math-driven surfaces) | Grasshopper, CAD | Math Galaxy RPN directly (Class A) |
| **Physics-Based** (simulation-driven) | Houdini Vellum | Reality Galaxy laws + integration (Class A) |
| **Voxel** (volume modeling) | MagicaVoxel | `OP_MARCHING_CUBES` + scalar field (Class B) |
| **NURBS** (curve control) | Rhino, Alias | `OP_BEZIER_EVAL` + control points (Class A) |

Capability classes per `RPN_DOMAIN_OPCODE_REGISTRY.md` Section 5.

### 10.2 2D-to-3D Fusion (Same Knowledge Base)

Drawing Galaxy entries are used for BOTH 2D rendering and 3D texturing:

```rpn
# Drawing Galaxy entry: "brick_pattern"
#   form_rpn: RECT 0.1 0.2 LINE OFFSET ...   (how to draw brick)
#   color_rpn: RGB 0.6 0.3 0.1 NOISE ...      (brick color variation)

# Used in 2D: Drawing Bridge renders form_rpn + color_rpn -> 2D texture
# Used in 3D: UV mapping routes form_rpn to mesh surface -> 3D textured wall
# Same knowledge, different application, zero duplication
```

A 2D shape can become 3D geometry via extrusion:

```rpn
# 2D star from Drawing Galaxy
DRAWING_REF star_shape     # Symlink to 2D shape RPN
MESH_EXTRUDE 0.5           # Extrude 0.5 units into 3D
BEVEL_EDGES 0.1            # Round corners
# Result: 3D star from 2D drawing knowledge
```

### 10.3 Fusion Composition Example

"Ancient Tree on Cliff" using multiple techniques in ONE composed RPN program:

```rpn
# L-system growth (trunk + branches)
REALITY_REF tree_growth_lsystem_v1
# CSG boolean (cliff subtract tree roots)
REALITY_REF cliff_geometry_v1
BOOLEAN_SUBTRACT root_cavity
# Physics (roots grip cliff, stable under gravity)
REALITY_REF gravity_stability_check
# Procedural noise (bark texture, rock detail)
DRAWING_REF bark_texture_noise_v2
DRAWING_REF rock_detail_noise_v1
# All techniques composed, all symlinked, fully reusable
```

### 10.4 Visual Tool-Nodes

3D techniques are stored as **tool-nodes** in Galaxy Universe -- procedural capabilities as knowledge:

```json
{
  "id": "tool_extrude_profile_v1",
  "galaxy": "Reality",
  "category": "procedural_tool",
  "input_contract": {
    "requires": ["2d_contour_ref", "depth"],
    "optional": ["bevel_profile_ref", "uv_rule_ref"]
  },
  "output_contract": {
    "produces": ["mesh_ref", "normal_ref", "uv_ref"]
  },
  "behavior_rpn": ["DRAWING_REF", "MESH_EXTRUDE_MACRO", "UV_PROJECT_MACRO"],
  "component_refs": ["tool_uv_project_v1", "tool_bevel_profile_v1"],
  "constraints": ["closed_contour_required", "positive_depth_only"]
}
```

**Tool families**:

- **2D tools**: contour construction, stroke expansion, fill/gradient composition
- **3D tools**: extrude, revolve, sweep, boolean combine, displacement, UV/triplanar mapping
- **Physics tools**: gravity integration, collision correction, spring-mass relaxation
- **Temporal tools**: keyframe interpolation, camera path generation, event-triggered transitions

Tool-nodes make K3D's "verbs" part of the knowledge substrate. TRM learns not only what things are, but which techniques to apply and how to compose them.

---

## 11. References

### Core Implementation Files
- `knowledge3d/training/arc_agi/drawing_galaxy.py` — Drawing Galaxy storage
- `knowledge3d/cranium/codecs/sovereign_ternary_video_codec.py` — Video codec
- `knowledge3d/cranium/codecs/sovereign_ternary_audio_codec.py` — Audio codec
- `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py` — RPN execution

### Architecture Documents
- `TEMP/DANIEL_VECTORDOTMAP_PLANS_V1.md` — VectorDotMap research
- `STRATEGY_AUDIO_AS_IMAGE_MULTIMODAL.md` — Audio-image integration
- `docs/vocabulary/UNIVERSAL_ACCESSIBILITY_SPECIFICATION.md` — Accessibility
- `CLAUDE.md` — Save Information Principle, Dual Client Reality

### External Research
- Quantum Dot Field Theory (display physics)
- Software Defined Radio (frequency-time representation)
- Biological vision (foveal attention, rod-cone distribution)

---

## 12. Conclusion

The Procedural Visual Architecture transforms K3D from a system that stores and retrieves visual data into one that **generates** visual data from compact procedural programs. By treating images, video, and even audio spectrograms as quantum field emissions rather than pixel grids, we achieve:

- **Infinite scalability**: Same 2KB program renders at any resolution
- **Semantic richness**: Visual elements link to meanings via Galaxy symlinks
- **Unified representation**: Images, video, spectrograms share the same codec
- **Sovereignty**: 100% GPU execution via PTX kernels

This is the visual foundation for K3D's Dual Client Reality — where every pixel humans see is generated by RPN programs that AI can execute, reason about, and compose.

---

**Version History**:
- 1.0 (December 2025): Initial specification documenting 8-layer Drawing Galaxy, VectorDotMap architecture, procedural codecs, and cross-galaxy integration.
- 1.1 (March 2026): Added 3D Technique Fusion (Section 10) — CSG/mesh/L-system/sculpting/parametric/physics/voxel/NURBS as composable RPN, 2D-to-3D fusion, tool-nodes concept. Aligned with TRM Multi-Modal Enhancement Architecture.
