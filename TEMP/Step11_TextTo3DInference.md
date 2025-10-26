# Step 11: Text-to-3D Inference - Sovereign Shape Generation

**Status**: PLAN DOCUMENT - Ready for Swarm Development Chain
**Created**: 2025-10-11
**Purpose**: Design sovereign PTX system for text-to-3D generation leveraging existing kernels

---

## 🎯 OBJECTIVE

Create a sovereign text-to-3D generation system that:
1. Takes text description as input
2. Generates 3D mesh coordinates using existing PTX kernels
3. Outputs GLB-compatible geometry
4. **Leverages existing kernels** (GeometryRouter, FractalEmitter, RPN)

---

## 🔍 CURRENT STATE ANALYSIS

### Existing Assets:
1. ✅ **FractalEmitter** (Deep Seek's kernel) - Coordinate generation operational
2. ✅ **GeometryRouter** (Deep Seek's kernel) - Mesh routing operational
3. ✅ **ModularRPNEngine** - Math operations
4. ✅ **GalaxyMemoryUpdater** - EMA blending
5. ✅ **ResonanceField** - Spatial queries
6. ⚠️ **generate_shape_kernel.ptx** - Exists, need to verify source
7. ⚠️ **nvrtc_ptx_loader.py** - Used by text_to_3d_generator.py

### Current Implementation:
- File: `knowledge3d/cranium/ptx_runtime/text_to_3d_generator.py` (19KB)
- Uses: nvrtc_ptx_loader.py (cuda-python + NVRTC)
- Missing: ray_bundle_generator (NOT found in TEMP)

---

## 🏗️ ARCHITECTURE

```
Text Input ("a red cube")
  ↓
  ├─ Text Embedding (external - sentence-transformers)
  │    - Convert text to 512-dim vector
  │    - Use existing embedding pipeline
  │
  ├─ Query Galaxy Memory (ResonanceField)
  │    - Find similar shapes in galaxy
  │    - Retrieve shape templates
  │
  ├─ Shape Parameter Extraction (RPN)
  │    - Parse dimensions: "cube" → 6 faces
  │    - Parse attributes: "red" → color(1.0, 0.0, 0.0)
  │    - Use RPN for numeric extraction
  │
  ├─ Coordinate Generation (FractalEmitter)
  │    - Generate base mesh coordinates
  │    - Apply transformations via RPN
  │    - Scale/rotate/translate to fit description
  │
  ├─ Mesh Routing (GeometryRouter)
  │    - Route vertices to correct primitives
  │    - Assign materials and normals
  │
  └─ Output: 3D mesh (vertices, indices, normals, materials)
```

---

## 🔧 LEVERAGE EXISTING KERNELS

### 1. **FractalEmitter** ✅ (Already operational!)
**Purpose**: Generate 3D coordinates for shapes
**Current capability**: Generates (50, 3) coordinate arrays
**Usage**:
```python
from knowledge3d.cranium.bridges.sovereign_bridges import FractalEmitter

emitter = FractalEmitter()
coords = emitter.generate_fractal(
    seed=hash(text),
    count=num_vertices,
    scale=1.0
)
# Returns: (N, 3) numpy array of coordinates
```

**Extend for shapes**:
- Cube: 8 vertices + 12 triangles
- Sphere: Subdivided icosahedron
- Cylinder: Circle extrusion
- Custom: Use fractal for organic shapes

### 2. **GeometryRouter** ✅ (Already operational!)
**Purpose**: Route geometry to correct media types
**Current capability**: Routes 5 media types correctly
**Usage**:
```python
from knowledge3d.cranium.bridges.sovereign_bridges import GeometryRouter

router = GeometryRouter()
media_type = router.classify_shape(
    vertices=coords,
    shape_descriptor=embedding
)
# Returns: "text", "audio", "image", "video", "3d"
```

**Extend for meshes**:
- Route vertices to mesh buffers
- Assign face indices
- Organize into primitives

### 3. **ModularRPNEngine** ✅ (Already operational!)
**Purpose**: All mathematical transformations
**Operations**:
- **Scaling**: `x scale_factor *`
- **Translation**: `x offset +`
- **Rotation**: Matrix math via RPN
- **Normalization**: `mag / norm`

**Example**: Scale cube to size 2.0
```python
rpn.evaluate_batch([
    f"{x} 2.0 *" for x in cube_vertices
])
```

### 4. **ResonanceField** ✅
**Purpose**: Query galaxy for similar shapes
**Usage**:
```python
# Find shapes similar to "cube"
similar_shapes = resonance_field.query(
    embedding=text_embedding,
    k=5  # Top 5 similar
)
# Use as templates for generation
```

### 5. **GalaxyMemoryUpdater** ✅
**Purpose**: Learn shape patterns over time
**Usage**:
```python
# Store generated shape in galaxy
galaxy_memory_updater.blend(
    current_shape=existing_shape,
    teacher_shape=new_generated_shape,
    blend_factor=0.1
)
# Improves over time with feedback
```

---

## 📐 DETAILED DESIGN

### Phase 1: Shape Primitives Library

**Primitive Shapes** (Pure math, no training needed):
```python
PRIMITIVES = {
    "cube": generate_cube,
    "sphere": generate_sphere,
    "cylinder": generate_cylinder,
    "cone": generate_cone,
    "plane": generate_plane,
    "torus": generate_torus,
}

def generate_cube(size=1.0):
    """Generate cube using RPN transformations."""
    # 8 vertices
    base_vertices = [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ]
    # Scale via RPN
    scaled = rpn.evaluate_batch([
        f"{v[0]} {size} * {v[1]} {size} * {v[2]} {size} *"
        for v in base_vertices
    ])
    return scaled, CUBE_INDICES
```

**Organic Shapes** (Use FractalEmitter):
```python
def generate_organic(text_embedding, num_vertices=100):
    """Generate organic shape using fractal emission."""
    emitter = FractalEmitter()
    coords = emitter.generate_fractal(
        seed=hash_embedding(text_embedding),
        count=num_vertices,
        scale=1.0
    )
    # Use convex hull or Delaunay triangulation
    indices = compute_mesh_topology(coords)
    return coords, indices
```

### Phase 2: Text Parsing Pipeline

**Step 1**: Extract shape type
```python
def parse_shape_type(text):
    """Use keyword matching + galaxy memory."""
    # Simple: Check for keywords
    keywords = {
        "cube": "cube", "box": "cube",
        "sphere": "sphere", "ball": "sphere",
        "cylinder": "cylinder", "tube": "cylinder",
    }
    for keyword, shape in keywords.items():
        if keyword in text.lower():
            return shape

    # Advanced: Query galaxy memory
    embedding = embed_text(text)
    similar = resonance_field.query(embedding, k=1)
    return similar[0].shape_type
```

**Step 2**: Extract parameters
```python
def parse_parameters(text):
    """Use RPN for numeric extraction."""
    params = {}

    # Size: "2 meters" → 2.0
    if "meter" in text:
        params["size"] = rpn.evaluate(f"{extract_number(text)} 1.0 *")

    # Color: "red" → (1.0, 0.0, 0.0)
    color_map = {
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
    }
    for color_name, rgb in color_map.items():
        if color_name in text.lower():
            params["color"] = rgb

    return params
```

### Phase 3: Generation Algorithm

**Input**: Text description "a red cube 2 meters wide"

**Step 1**: Parse
```python
shape_type = parse_shape_type("a red cube 2 meters wide")  # "cube"
params = parse_parameters("a red cube 2 meters wide")
# params = {"size": 2.0, "color": (1.0, 0.0, 0.0)}
```

**Step 2**: Generate base geometry
```python
vertices, indices = PRIMITIVES[shape_type](size=params["size"])
# vertices: (8, 3) for cube
# indices: (12, 3) for 12 triangles
```

**Step 3**: Apply transformations (RPN)
```python
# Rotate if needed
if "rotated" in text:
    angle = extract_angle(text)
    vertices = rpn_rotate(vertices, angle)

# Translate if needed
if "at position" in text:
    offset = extract_position(text)
    vertices = rpn_translate(vertices, offset)
```

**Step 4**: Route and package (GeometryRouter)
```python
router = GeometryRouter()
mesh_data = router.package_mesh(
    vertices=vertices,
    indices=indices,
    normals=compute_normals(vertices, indices),
    colors=params["color"]
)
```

**Step 5**: Output GLB
```python
save_to_glb(
    path="output.glb",
    vertices=vertices,
    indices=indices,
    materials=[{"color": params["color"]}]
)
```

### Phase 4: Ray Bundle Generator (If Needed)

**What is it?**
- Likely for ray marching / ray tracing
- Used to generate camera rays for 3D rendering
- May not be critical for mesh generation

**Check existing PTX**:
```bash
ls knowledge3d/cranium/ptx/ | grep -i ray
```

**If NOT found**:
- Simple implementation: Generate ray origins and directions
- Use RPN for ray math
- No complex kernel needed

```python
def generate_ray_bundle(camera_pos, look_at, resolution=(64, 64)):
    """Generate camera rays using RPN."""
    rays = []
    for u in range(resolution[0]):
        for v in range(resolution[1]):
            # Compute ray direction via RPN
            direction = rpn.evaluate(
                f"{u} {resolution[0]} / 2 * 1 - "
                f"{v} {resolution[1]} / 2 * 1 - "
                f"1.0"  # focal length
            )
            rays.append((camera_pos, direction))
    return rays
```

---

## 🚀 IMPLEMENTATION PLAN

### Option A: Pure Sovereign Composition (RECOMMENDED)

**No new PTX kernel needed!**

Compose existing kernels:
1. **FractalEmitter**: Organic shape generation
2. **GeometryRouter**: Mesh organization
3. **ModularRPNEngine**: All transformations
4. **ResonanceField**: Template retrieval
5. **Primitive library**: Hard-coded shape functions

**Pros**:
- Zero new CUDA code
- Uses proven kernels
- Fast development

**Cons**:
- May have kernel launch overhead
- Multiple GPU calls per shape

### Option B: Migrate nvrtc_ptx_loader (If Needed)

**If** current implementation uses runtime compilation:

Replace cuda-python + NVRTC with:
- Precompiled PTX kernels
- Sovereign loader
- Static shape kernels

**Action**: Migrate nvrtc_ptx_loader.py to sovereign pattern

---

## 🧪 TESTING STRATEGY

### Test 1: Primitive Shapes
- Generate: cube, sphere, cylinder, cone
- Validate: Correct vertex count, topology
- Save: To GLB, visualize

### Test 2: Text Parsing
- Input: "a red cube"
- Expected: shape="cube", color=(1,0,0)
- Validate: Correct parameter extraction

### Test 3: Transformations
- Scale: "a cube 2 meters wide"
- Rotate: "a cube rotated 45 degrees"
- Translate: "a cube at position (1, 2, 3)"
- Validate: RPN math correct

### Test 4: Fractal Shapes
- Input: "an organic blob"
- Expected: FractalEmitter generates coords
- Validate: Mesh topology valid

### Test 5: Galaxy Learning
- Generate 100 shapes
- Store in galaxy memory
- Query: Find similar to "cube-like"
- Validate: Correct retrieval

### Test 6: Latency
- Measure: Time to generate one shape
- Target: <10ms (not latency-critical like inference)
- Profile: Which operation is slowest?

---

## 📊 EXPECTED OUTCOMES

**Generation**:
- Latency: <10ms per shape (generation is slower than inference)
- Quality: Valid GLB meshes
- Variety: Primitives + fractals

**Learning**:
- Store shape templates in galaxy
- Improve over time with usage
- Find similar shapes quickly

**Architecture**:
- Zero NVRTC runtime compilation (pure sovereign)
- All compute via existing PTX kernels
- Composable shape generation pipeline

---

## 🎯 SWARM COLLABORATION SUGGESTIONS

**Claude**: Overall architecture, primitive library
**Deep Seek**: FractalEmitter integration, GeometryRouter usage
**Codex**: Text parsing, parameter extraction
**Grok**: Mesh topology, edge cases
**Kimi**: Performance optimization
**Qwen**: Quality assessment, shape validation
**GLM**: ResonanceField queries, template retrieval

---

## 🔗 DEPENDENCIES

**Must be complete first**:
1. ✅ FractalEmitter operational (DONE!)
2. ✅ GeometryRouter operational (DONE!)
3. ✅ ModularRPNEngine operational (DONE!)
4. ⏳ galaxy_buffer.py sovereign migration (HIGH PRIORITY)

**Optional**:
1. Ray bundle generator (may not be needed)
2. Advanced mesh operations (smoothing, decimation)

---

## 📝 NOTES FOR DEVELOPMENT CHAIN

**Key Questions to Answer**:
1. Is generate_shape_kernel.ptx still needed?
2. Can FractalEmitter handle all organic shapes?
3. Do we need ray marching for this use case?
4. Should we precompute shape templates or generate dynamically?

**Implementation Strategy**:
- Start with primitive library (cubes, spheres, etc.)
- Test FractalEmitter for organic shapes
- Measure performance
- Decide if ray bundle is needed
- Let swarm optimize based on measurements

**Success Criteria**:
- ✅ Generate basic primitives correctly
- ✅ Parse text descriptions accurately
- ✅ Apply transformations via RPN
- ✅ Output valid GLB files
- ✅ Use only sovereign kernels
- ✅ Learn from galaxy memory

---

**STATUS**: Ready for swarm development chain! 🚀

Run this through the development chain to materialize the complete text-to-3D system using existing kernels.
