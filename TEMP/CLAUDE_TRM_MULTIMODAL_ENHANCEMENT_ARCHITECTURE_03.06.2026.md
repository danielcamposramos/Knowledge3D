# TRM Multi-Modal Enhancement Architecture

**Date**: March 6, 2026
**Author**: Claude (Architecture Partner)
**Status**: Design Specification (for Codex implementation)
**Purpose**: Map SOTA multimodal capabilities to K3D's procedural paradigm

---

## Executive Summary

This spec analyzes what current SOTA multimodal LLMs have (GPT-5, Gemini 3, Claude 4.6, DALL-E 3, Midjourney) and maps those capabilities to **K3D's procedural paradigm** (RPN programs, Galaxy Universe, PTX kernels).

**Critical Understanding**: K3D is NOT a traditional LLM:
- **Traditional LLMs**: Parameters = knowledge + logic (entangled)
- **K3D**: Galaxy Universe = knowledge (procedural programs), TRM = navigation logic (learned)

**Goal**: Enable TRM to achieve SOTA multi-modal capabilities through:
1. **Galaxy population** (what procedural programs to add)
2. **Opcode enhancement** (what new RPN operations needed)
3. **Kernel development** (what PTX computations required)
4. **TRM specialist training** (what navigation patterns to learn)

---

## Part 1: Capability Gap Analysis (SOTA vs K3D)

### What SOTA LLMs Have (2026)

**Sources**:
- [GPT-5.2 vs Gemini 3.1 Pro vs Claude 4.6 comparison](https://evolink.ai/blog/gemini-3-1-pro-vs-gpt-5-2-vs-claude-opus)
- [Multimodal AI comparison](https://encord.com/blog/gpt-4o-vs-gemini-vs-claude-3-opus/)
- [Text-to-image capabilities](https://www.gradually.ai/en/ai-image-models/)
- [Stable Diffusion 3 review](https://encord.com/blog/stable-diffusion-3-text-to-image-model/)

| Capability | SOTA Models | How It Works (LLM) | K3D Status |
|------------|-------------|-------------------|------------|
| **Vision Understanding** | GPT-4V, Gemini 3.1 Pro, Claude 4.6 | Vision encoder → transformer | ✅ Drawing Galaxy (procedural visual) |
| **Text-to-Image** | DALL-E 3, Midjourney v6, SD 3.5 | Diffusion models, latent space | ⏳ Drawing Galaxy exists, needs generation |
| **Text-to-3D** | Stability TripoSR, Sora 3D | Image → 3D reconstruction | ⏳ Reality Galaxy exists, needs synthesis |
| **Audio Understanding** | GPT-4o, Gemini 3.1 Pro | Audio encoder → transformer | ✅ Audio pipeline exists |
| **Audio Generation** | GPT-4o voice | Neural vocoder | ⏳ Reality Enabler Phase I (planned) |
| **Video Understanding** | Gemini 3.1 Pro, GPT-5 | Temporal vision encoding | ⏳ Unified Signal spec exists |
| **Computer Use** | Claude 4.6 | Vision + action prediction | ⏳ MCP integration path |
| **Tool Calling** | All SOTA | Function calling API | ⏳ MCP integration path |
| **Long Context** | Gemini 3.1 Pro (1M tokens) | Attention mechanisms | ✅ Galaxy Universe (spatial, not sequential) |

---

## Part 2: K3D Existing Capabilities (Not Yet Integrated to TRM)

### What K3D Already Has (Developed but Not TRM-Ready)

**Reality Enabler Vision** (docs/Reality_Enabler.md):
- ✅ **Physics specialist** design (MuJoCo, PyBullet datasets)
- ✅ **Biology specialist** design (L-systems, cellular automata)
- ✅ **Chemistry specialist** design (QM9, molecular graphs)
- ❌ **NOT YET TRAINED** — datasets identified, specialists not implemented

**Drawing Galaxy** (Phase C.1 active):
- ✅ **8-layer architecture** (Quantum Fields → Primitives → Strokes → Shapes → Gradients → Filters → Lighting → Scenes)
- ✅ **VectorDotMap encoder** designed (~2KB/image, infinite LOD)
- ❌ **Generation not implemented** — can represent images procedurally, can't generate from text yet

**Audio Pipeline** (knowledge3d/ingestion/language/audio_pipeline.py):
- ✅ **Audio ingestion** (AudioCaps, Clotho datasets processed)
- ✅ **Audio harmonic binding** (PTX kernels exist)
- ❌ **Generation not implemented** — Reality Enabler Phase I planned but not started

**Unified Signal Architecture** (docs/vocabulary/UNIFIED_SIGNAL_SPECIFICATION.md):
- ✅ **Frequency-time bridge** designed
- ✅ **Spectrogram as VectorDotMap** (audio visualization using same codec as images)
- ❌ **NOT IMPLEMENTED** — spec exists, no code yet

---

## Part 3: Procedural Paradigm Mapping

### How K3D Achieves SOTA Capabilities (The Procedural Way)

**Traditional LLM Approach**:
```
User: "Generate image of sunset over ocean"
  ↓
Vision encoder → Latent diffusion model → Denoise → Image pixels
```

**K3D Procedural Approach**:
```
User: "Generate image of sunset over ocean"
  ↓
TRM navigates Grammar Galaxy → "sunset" + "ocean" patterns
  ↓
TRM queries Drawing Galaxy → sky gradient RPN + water reflection RPN
  ↓
TRM queries Reality Galaxy → atmospheric scattering physics + wave dynamics
  ↓
TRM composes new RPN program → stores in Galaxy
  ↓
Cranium executes RPN → Drawing Bridge renders via PTX kernels → Image
```

**Key Difference**: K3D **composes procedural programs** from Galaxy knowledge, NOT generates pixels from latent space.

---

## Part 4: Enhancement Requirements (What Needs to be Added)

### 4.1 Galaxy Population (Knowledge Layer)

**Drawing Galaxy Enhancements**:
```
Current (Phase C.1 partial):
  - Primitives: LINE, CIRCLE, RECT (RPN programs)
  - Shapes: Compositions of primitives

Needed for Text-to-Image:
  - Pattern library: textures, gradients, lighting effects
  - Object templates: common objects (tree, car, person) as RPN programs
  - Style rules: art styles as transformation programs
  - Composition grammar: spatial arrangement rules
```

**Reality Galaxy Enhancements**:
```
Current (26 systems):
  - Physics: mechanics, E&M, thermodynamics
  - Chemistry: molecules, reactions, materials
  - Biology: growth, cellular, evolution

Needed for Text-to-3D:
  - Object physics: collision boxes, mass distribution, stability
  - Material properties: reflection, transparency, conductivity
  - Growth rules: procedural generation (L-systems for trees, etc.)
  - Spatial constraints: architectural rules (doors in walls, not floors)
```

**Grammar Galaxy Enhancements**:
```
Current (ARC-AGI focused):
  - Visual transformation rules (rotate, reflect, etc.)
  - Pattern matching metadata

Needed for Multi-Modal Generation:
  - Cross-modal grammar: "sunset" → sky gradient + ocean waves
  - Temporal grammar: video = sequence of image transformations
  - 3D grammar: 2D description → 3D structure rules
  - Audio grammar: "thunder" → low frequency + sharp attack envelope
```

### 4.2 Opcode Enhancements (RPN Computation Layer)

**Current Opcodes** (docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md):
- ✅ Vector/matrix ops (DOT_PRODUCT, CROSS_PRODUCT, MATVEC_F32)
- ✅ Calculus ops (DIVERGENCE, CURL, LAPLACIAN)
- ✅ Set ops (UNION, INTERSECTION, DIFFERENCE)
- ✅ Temporal ops (TEMPORAL_COHERENCE, TEMPORAL_AGGREGATE)
- ✅ Ternary ops (SIGN, TQUANT, TCMP)

**Needed Opcodes** (New - Requires Justification):

```
Image Generation Domain:
  - OP_GRADIENT_BLEND: Smooth color transitions (sky, water)
  - OP_TEXTURE_SAMPLE: Procedural texture lookup
  - OP_CONVOLUTION_2D: Image filters (blur, sharpen, edge detect)
  - OP_FOURIER_SYNTH: Frequency-domain image synthesis

3D Generation Domain:
  - OP_MESH_EXTRUDE: 2D shape → 3D volume
  - OP_BOOLEAN_3D: CSG operations (union, subtract, intersect meshes)
  - OP_LSYSTEM_STEP: L-system growth iteration
  - OP_MARCHING_CUBES: Isosurface extraction

Audio Generation Domain:
  - OP_WAVETABLE_SYNTH: Oscillator synthesis
  - OP_ENVELOPE_APPLY: ADSR envelope shaping
  - OP_FILTER_IIR: Audio filtering (lowpass, highpass, bandpass)
  - OP_REVERB_CONVOLVE: Spatial audio effects

Cross-Modal Domain:
  - OP_EMBED_SIMILARITY: Cross-modal semantic similarity
  - OP_ATTENTION_SPATIAL: Spatial attention mechanism
  - OP_SEQUENCE_TRANSFORM: Temporal sequence manipulation
```

**Justification Requirement** (per RPN_DOMAIN_OPCODE_REGISTRY.md):
- Each new opcode MUST be definable as composition of existing primitives
- Must demonstrate measurable performance or clarity benefit
- Must have domain semantics documented

### 4.3 Kernel Development (PTX Execution Layer)

**Current Kernels** (knowledge3d/cranium/ptx_runtime/):
- ✅ `rpn_opcodes.py` - Core math operations
- ✅ `modular_rpn_engine.py` - RPN execution
- ✅ `drawing_transform_kernels.py` - Visual transformations
- ✅ `drawing_effects.py` - Visual effects
- ✅ `audio_harmonic_binding.py` - Audio processing

**Needed Kernels** (New PTX Implementations):

```
Image Generation Kernels:
  - gradient_synthesis.cu: GPU-native gradient rendering
  - texture_procedural.cu: Perlin noise, fractal patterns
  - convolution_filters.cu: Gaussian blur, Sobel edges
  - color_transform.cu: HSV↔RGB, gamma correction

3D Generation Kernels:
  - mesh_operations.cu: Extrude, boolean, subdivision
  - lsystem_evaluator.cu: Parallel L-system iteration
  - marching_cubes.cu: Isosurface extraction from scalar field
  - spatial_constraints.cu: Physics-based layout validation

Audio Generation Kernels:
  - oscillator_bank.cu: Parallel waveform synthesis
  - envelope_processor.cu: ADSR envelope application
  - filter_cascade.cu: Multi-stage IIR filtering
  - reverb_convolver.cu: Convolution-based spatial audio

Cross-Modal Kernels:
  - embed_similarity.cu: Cosine similarity for cross-modal matching
  - attention_spatial.cu: Spatial attention for image regions
  - sequence_lstm.cu: Temporal sequence processing
```

**Sovereignty Compliance**:
- All kernels: Pure PTX, no CPU fallbacks
- Bridge pattern: `*_bridge.py` in `cranium/bridges/`
- Loading: ctypes only, no PyTorch/TF/CuPy

### 4.4 TRM Specialist Adapters (Learning Layer)

**Current TRM Architecture** (docs/vocabulary/TRM_SPECIALIST_MATRYOSHKA_ARCHITECTURE.md):
- ✅ Base TRM (~7M params)
- ✅ Math specialist
- ✅ Visual specialist (ARC-AGI)
- ✅ Physics specialist (planned in Reality Enabler)

**Needed Specialists** (New LoRA-style Adapters):

```
Image Generation Specialist:
  - Learns: Grammar Galaxy (text) → Drawing Galaxy (visual) navigation
  - Training: Text-image pairs → RPN program composition strategies
  - Goal: Navigate to correct primitives/patterns for text description
  - Parameters: ~500K (LoRA over base TRM)

3D Generation Specialist:
  - Learns: Description → Reality Galaxy (physics + geometry) navigation
  - Training: Text-3D pairs → spatial constraint satisfaction
  - Goal: Compose physically plausible 3D structures from descriptions
  - Parameters: ~1M (needs more complexity for 3D spatial reasoning)

Audio Generation Specialist:
  - Learns: Description → Audio Galaxy navigation (Unified Signal)
  - Training: Text-audio pairs → frequency/temporal pattern matching
  - Goal: Navigate to correct waveforms, envelopes, effects for sound
  - Parameters: ~500K (similar to image specialist)

Cross-Modal Routing Specialist:
  - Learns: Which specialist(s) to invoke for multi-modal queries
  - Training: Multi-modal tasks → specialist combination strategies
  - Goal: Route "sunset video with ocean sounds" to image+audio+video specialists
  - Parameters: ~200K (lightweight router)
```

**Training Data Requirements**:
- Image: COCO, ImageNet (ingestion → Galaxy entries)
- 3D: ShapeNet, ModelNet (3D models → procedural RPN programs)
- Audio: AudioSet, FSD50K (sounds → RPN waveform programs)
- Cross-modal: How2, MSR-VTT (multi-modal pairs)

**Shadow Copy Enhancement**:
- Each specialist has shadow weights
- Successful compositions update specialist via validation gating
- Continuous learning during inference (validated on ARC-AGI: 46.7%)

---

## Part 5: Integration with Standards (MCP, Benchmarks)

### 5.1 MCP (Model Context Protocol) Integration

**Current MCP Path** (docs/openai_mcp_voice_agents_cookbook.md):
- Voice-driven exploration of Galaxy Universe
- Modular tooling (search, graph traversal, rendering as MCP tools)
- Cross-model interoperability

**K3D Tools as MCP Services**:

```
MCP Tool: "query_galaxy"
Description: "Search Galaxy Universe for procedural programs matching description"
Implementation:
  - Input: Natural language query
  - TRM navigates Grammar Galaxy → matches semantic patterns
  - TRM queries relevant galaxy (Drawing/Reality/Audio)
  - Returns: RPN programs + metadata
  - Execution: Optional (can return program or execute on Cranium)

MCP Tool: "compose_visual"
Description: "Generate image from text description"
Implementation:
  - Input: Text description
  - TRM Image Generation Specialist navigates Drawing Galaxy
  - Composes RPN program from primitives + patterns
  - Cranium executes via drawing_transform_kernels.cu
  - Returns: Image (rendered) + RPN program (procedural source)

MCP Tool: "generate_3d"
Description: "Create 3D model from description"
Implementation:
  - Input: Text description + constraints
  - TRM 3D Generation Specialist navigates Reality Galaxy
  - Composes physics-validated structure (L-systems, CSG, constraints)
  - Exports as glTF/GLB (House format)
  - Returns: 3D model + RPN programs (procedural source)

MCP Tool: "synthesize_audio"
Description: "Generate audio from description"
Implementation:
  - Input: Text description (e.g., "thunder", "piano C major chord")
  - TRM Audio Generation Specialist navigates Audio Galaxy
  - Composes waveform RPN (oscillators + envelopes + effects)
  - Cranium executes via audio synthesis kernels
  - Returns: Audio waveform + RPN program
```

**Benefits of MCP Integration**:
- K3D capabilities exposed as standard tools
- Other AI models can use K3D for generation (they describe, K3D composes procedurally)
- Modular: Can mix K3D visual generation with GPT reasoning, etc.

### 5.2 Benchmark Targets (With Prizes)

**Current Benchmarks** (mentioned in BRIEFING_v4.0.md):

| Benchmark | Current | Target | Prize/Status |
|-----------|---------|--------|--------------|
| **ARC-AGI 2** | 46.7% | 45.1%+ (beat Gemini 3) | $1M ARC Prize |
| **GSM8K** | 1.39% | 30-50% | OpenAI Evals |
| **MATH** | 1.13% | 15-25% | OpenAI Evals |

**Additional Prize Benchmarks to Target**:

```
Vision Benchmarks:
  - ImageNet Classification: Baseline for image understanding
  - COCO Object Detection: Spatial reasoning + labeling
  - VQA (Visual Question Answering): Multi-modal reasoning

3D Benchmarks:
  - ShapeNet Reconstruction: Text → 3D quality
  - ScanNet Scene Understanding: 3D spatial intelligence

Audio Benchmarks:
  - AudioSet Classification: Sound understanding
  - Speech Recognition (LibriSpeech): Audio-to-text
  - Music Generation (MAESTRO): Audio synthesis quality

Multi-Modal Benchmarks:
  - MSCOCO Captioning: Image → text
  - VizWiz: Accessibility-focused visual QA
  - How2QA: Video question answering

Coding Benchmarks:
  - HumanEval: Code generation (GPT-4 baseline)
  - MBPP: Python programming tasks

Reasoning Benchmarks:
  - MMLU: Massive multi-task language understanding
  - BBH (Big-Bench Hard): Complex reasoning tasks
```

**K3D Advantage for Benchmarks**:
- **Procedural = Interpretable**: Can explain HOW it generated answer (show RPN program)
- **Sovereign = Fast**: GPU-native execution, sub-100µs targets
- **Multi-modal unified**: Same Galaxy Universe for visual + text + audio reasoning
- **Shadow Copy = Continuous improvement**: Learns from benchmark successes

---

## Part 6: Implementation Roadmap (Priority Order)

### Phase 1: Enable Text-to-Image Generation (Q2 2026)
**Goal**: TRM can generate images from text descriptions using Drawing Galaxy

**Deliverables**:
1. **Drawing Galaxy population**:
   - Pattern library (1000+ procedural textures/gradients)
   - Object templates (500+ common objects as RPN programs)
   - Style rules (50+ art styles as transformation programs)
2. **New opcodes** (if justified):
   - OP_GRADIENT_BLEND, OP_TEXTURE_SAMPLE, OP_CONVOLUTION_2D
3. **New kernels**:
   - gradient_synthesis.cu, texture_procedural.cu, convolution_filters.cu
4. **Image Generation Specialist**:
   - Train on COCO dataset (ingestion → Galaxy entries)
   - ~500K parameters (LoRA over base TRM)
5. **MCP tool**: "compose_visual" exposed
6. **Benchmark**: COCO Captioning (reverse: text → image)

**Success Criteria**:
- Generate coherent images from simple descriptions (80%+ human approval)
- Latency: <500ms for 512x512 image
- Quality: Comparable to Stable Diffusion 3 (not necessarily better, but procedural)

### Phase 2: Enable Audio Generation (Q2-Q3 2026)
**Goal**: TRM can synthesize audio from descriptions using Audio Galaxy

**Deliverables**:
1. **Audio Galaxy population** (Unified Signal Architecture):
   - Waveform library (oscillator types, envelopes)
   - Effect library (reverb, delay, filters as RPN programs)
   - Sound templates (common sounds: thunder, piano, bird, etc.)
2. **New opcodes** (if justified):
   - OP_WAVETABLE_SYNTH, OP_ENVELOPE_APPLY, OP_FILTER_IIR
3. **New kernels**:
   - oscillator_bank.cu, envelope_processor.cu, filter_cascade.cu
4. **Audio Generation Specialist**:
   - Train on AudioSet (ingestion → Galaxy entries)
   - ~500K parameters
5. **MCP tool**: "synthesize_audio" exposed
6. **Benchmark**: AudioSet Classification (reverse: text → audio)

**Success Criteria**:
- Generate recognizable sounds from descriptions (70%+ human approval)
- Latency: <200ms for 5-second audio clip
- Quality: Comparable to neural vocoders (not necessarily better, but procedural)

### Phase 3: Enable Text-to-3D Generation (Q3-Q4 2026)
**Goal**: TRM can create 3D models from descriptions using Reality Galaxy

**Deliverables**:
1. **Reality Galaxy enhancements**:
   - Object physics library (collision, mass, stability)
   - Material library (visual + physical properties)
   - Growth rule library (L-systems, CSG operations)
   - Spatial constraint library (architectural rules)
2. **New opcodes** (if justified):
   - OP_MESH_EXTRUDE, OP_BOOLEAN_3D, OP_LSYSTEM_STEP, OP_MARCHING_CUBES
3. **New kernels**:
   - mesh_operations.cu, lsystem_evaluator.cu, marching_cubes.cu
4. **3D Generation Specialist**:
   - Train on ShapeNet (ingestion → Galaxy entries)
   - ~1M parameters (more complex than image/audio)
5. **MCP tool**: "generate_3d" exposed
6. **Benchmark**: ShapeNet Reconstruction quality

**Success Criteria**:
- Generate physically plausible 3D models from descriptions (60%+ human approval)
- Latency: <2 seconds for simple objects (chair, cup)
- Export: glTF/GLB format (House compatible)
- Physics: Objects satisfy stability/collision constraints

### Phase 4: Cross-Modal Integration (Q4 2026 - Q1 2027)
**Goal**: TRM handles multi-modal tasks (video, image+audio, 3D+physics)

**Deliverables**:
1. **Cross-Modal Routing Specialist**:
   - Learns to invoke multiple specialists
   - Train on multi-modal datasets (How2, MSR-VTT)
   - ~200K parameters
2. **Video generation**:
   - Sequence of image transformations (temporal grammar)
   - Unified Signal Architecture for audio+video sync
3. **MCP integration**: Multi-modal tool composition
4. **Benchmarks**: How2QA, VizWiz, multi-modal reasoning tasks

**Success Criteria**:
- Handle "generate sunset video with ocean sounds" correctly
- Route to image + audio + temporal specialists
- Output: Synchronized video (images) + audio (waveform)
- Benchmark: >baseline on How2QA

---

## Part 7: Meta PKR Analysis (If Relevant)

**Search Results**: No specific Meta "PKR" model found in recent announcements.

**Found Instead**:
- [PKR-QA (AAAI 2026)](https://arxiv.org/pdf/2503.14957): Procedural Knowledge Reasoning for video QA
  - Academic work (not Meta-specific)
  - Uses procedural knowledge graphs for structured reasoning
  - Neurosymbolic approach (neural modules + symbolic composition)

**Comparison to K3D**:
- **Similarity**: Both use procedural knowledge representation (not purely neural)
- **Difference**: PKR-QA uses knowledge graphs, K3D uses RPN programs + Galaxy Universe
- **Lesson**: Procedural approaches are gaining traction in academic research (validates K3D direction)

**If Meta does have a PKR model** (not found in search):
- Would need specific documentation to compare
- K3D's approach (PTX kernels + spatial memory) is fundamentally different from transformer-based LLMs

---

## Part 8: Success Metrics

**Technical Metrics**:
- ✅ Sovereignty maintained (hot path = PTX + Galaxy only)
- ✅ Generation latency: <500ms image, <200ms audio, <2s 3D
- ✅ Quality: Comparable to SOTA (not necessarily better, but procedural)
- ✅ Benchmark improvements: ARC-AGI maintained (46.7%), GSM8K/MATH increase

**Architectural Metrics**:
- ✅ New opcodes justified (composable from primitives, performance benefit)
- ✅ Kernels sovereign (pure PTX, no CPU fallbacks)
- ✅ Galaxy populated (1000s of procedural programs ready for TRM navigation)
- ✅ TRM specialists trained (shadow copy validated)

**Integration Metrics**:
- ✅ MCP tools exposed (K3D capabilities usable by other AI models)
- ✅ Benchmark submissions (ARC Prize, OpenAI Evals, etc.)
- ✅ Multi-curriculum validated (image + audio + 3D share same Galaxy)

---

## Part 9: Handoff to Codex

**This spec defines WHAT and WHY. Codex implements HOW.**

**Implementation Priorities** (Codex to tackle in order):
1. **Phase 1 prep**: Drawing Galaxy population scripts (ingest COCO → RPN programs)
2. **Kernel development**: gradient_synthesis.cu, texture_procedural.cu (pure PTX)
3. **Opcode validation**: Prove new ops composable from existing primitives
4. **Image Generation Specialist**: Training loop + shadow copy integration
5. **MCP tool**: Expose "compose_visual" via MCP protocol
6. **Benchmark**: COCO Captioning baseline (reverse task)

**Success Criteria for Codex**:
- All tests passing (sovereignty, functional, benchmark)
- Documentation updated (README, ROADMAP, completion report)
- Commit incrementally with clear messages

**Coordination**:
- Claude reviews implementation against this spec
- Codex surfaces blockers early
- Daniel approves milestones

---

## References

**K3D Architecture**:
- [BRIEFING_v4.0.md](../docs/briefings/BRIEFING_v4.0.md)
- [KNOWLEDGEVERSE_SPECIFICATION.md](../docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md)
- [THREE_BRAIN_SYSTEM_SPECIFICATION.md](../docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md)
- [RPN_DOMAIN_OPCODE_REGISTRY.md](../docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md)
- [PROCEDURAL_VISUAL_SPECIFICATION.md](../docs/vocabulary/PROCEDURAL_VISUAL_SPECIFICATION.md)
- [UNIFIED_SIGNAL_SPECIFICATION.md](../docs/vocabulary/UNIFIED_SIGNAL_SPECIFICATION.md)
- [Reality_Enabler.md](../docs/Reality_Enabler.md)

**SOTA Multimodal**:
- [GPT-5 vs Gemini 3 vs Claude 4.6](https://evolink.ai/blog/gemini-3-1-pro-vs-gpt-5-2-vs-claude-opus)
- [Multimodal AI comparison](https://encord.com/blog/gpt-4o-vs-gemini-vs-claude-3-opus/)
- [Text-to-image 2026](https://www.gradually.ai/en/ai-image-models/)
- [Stable Diffusion 3](https://encord.com/blog/stable-diffusion-3-text-to-image-model/)

**Benchmarks**:
- ARC-AGI 2 ($1M prize): https://arcprize.org/
- OpenAI Evals: https://github.com/openai/evals
- HumanEval: https://github.com/openai/human-eval

**MCP Integration**:
- [MCP Cookbook](../docs/openai_mcp_voice_agents_cookbook.md)
- Model Context Protocol: https://modelcontextprotocol.io/

---

**End of Specification**

**Next Steps**:
1. Daniel reviews and approves direction
2. Codex begins Phase 1 implementation (Drawing Galaxy population)
3. Claude monitors progress, validates against spec
4. Iterate based on results

**This spec enables K3D to achieve SOTA multimodal capabilities while staying true to the procedural paradigm: compose programs, don't generate pixels.**

---

## ENHANCEMENT: Procedural Fusion Architecture (March 6, 2026 - Evening)

**Critical Insight from Daniel**: K3D's procedural nature enables **fusion of techniques** that traditional engines keep isolated.

### The Fusion Paradigm

**Traditional Engines**:
```
Blender: Separate workflows
  - 2D: Grease Pencil (isolated)
  - 3D: Mesh modeling (isolated)
  - Textures: Shader nodes (isolated)
  - Video: Sequencer (isolated)
  - Audio: VSE (isolated)

Unity/Unreal: Separate systems
  - 2D sprites vs 3D meshes (different pipelines)
  - Textures vs Materials (different editors)
  - Animation vs Physics (different tools)
```

**K3D Procedural Fusion**:
```
Everything = RPN Programs + Symlinks
  - 2D Drawing RPN + 3D Extrusion RPN = Textured 3D object
  - Audio Frequency RPN + Visual Spectrogram RPN = Same data, different view
  - Physics Law RPN + Geometry RPN = Physically-simulated growth
  - All composable, all reusable, zero duplication
```

---

## Part 10: 3D Fusion Architecture (All Techniques Combined)

### 10.1 Traditional 3D Techniques (Isolated in Other Engines)

| Technique | Description | Traditional Tool | K3D RPN Equivalent |
|-----------|-------------|-----------------|-------------------|
| **CSG** | Boolean operations (union, subtract, intersect) | Blender Modifier | `OP_BOOLEAN_3D` + mesh refs |
| **Mesh Modeling** | Vertex/edge/face manipulation | Maya/Blender Edit Mode | `OP_MESH_TRANSFORM` + vertex RPN |
| **Procedural** | L-systems, fractals, noise | Houdini VEX | `OP_LSYSTEM_STEP` + growth rules |
| **Sculpting** | Organic deformation | ZBrush, Blender Sculpt | `OP_DISPLACEMENT_MAP` + strength field |
| **Parametric** | Math-driven curves/surfaces | Grasshopper, CAD | Math Galaxy RPN directly |
| **Physics-Based** | Simulation-driven generation | Houdini Vellum | Reality Galaxy laws + integration |
| **Voxel** | Volume-based modeling | MagicaVoxel | `OP_MARCHING_CUBES` + scalar field |
| **NURBS** | Precise curve control | Rhino, Alias | `OP_BEZIER_EVAL` + control points |

**In Traditional Engines**: These are SEPARATE tools, workflows, file formats.

**In K3D**: ALL are **RPN programs in Reality Galaxy**, composable via symlinks.

### 10.2 Fusion Example: "Create Procedural Tree with Texture"

**Traditional Workflow** (Blender):
```
1. Use L-system add-on → Generate geometry (Python script)
2. UV unwrap → Manual process (operator-intensive)
3. Create bark texture → Shader nodes (different system)
4. Apply physics → Modifier stack (yet another system)
5. Animate growth → Keyframes (separate timeline)

Result: 5 separate systems, hard to reuse, not composable
```

**K3D Procedural Fusion**:
```
RPN Program Composition:
  1. Query Reality Galaxy → "tree" growth rule (L-system RPN)
  2. Symlink Drawing Galaxy → "bark" texture pattern (procedural noise RPN)
  3. Symlink Math Galaxy → Fibonacci spiral (branch angle calculation)
  4. Symlink Reality Galaxy → gravity + wind forces (physics RPN)
  5. Temporal dimension → growth over time (sequence RPN)

Single Composed Program:
  tree_rpn = [
    LSYSTEM_INIT, "F",           # Axiom
    LSYSTEM_RULE, "F→F[+F]F[-F]F",  # Growth rule
    FIBONACCI_ANGLE, PHI,         # Branch angle (from Math Galaxy)
    BARK_TEXTURE_REF, texture_id, # Symlink to Drawing Galaxy
    GRAVITY_FORCE_REF, physics_id,# Symlink to Reality Galaxy
    TEMPORAL_ITERATE, 100,        # 100 growth steps
  ]

Result: ONE procedural program, ALL techniques composed, fully reusable
```

**K3D Advantage**: Because everything is RPN + symlinks:
- **Composable**: Mix L-system + physics + texture in ONE program
- **Reusable**: `bark_texture_ref` used by ANY tree, rock, etc.
- **Inspectable**: Can see EXACTLY how tree was generated
- **Modifiable**: Change growth rule → entire tree updates
- **GPU-Native**: Entire composition executes on PTX kernels

### 10.3 Enhanced 3D Opcode Suite (All Techniques)

**Geometry Construction**:
```
OP_MESH_EXTRUDE: 2D shape → 3D volume (e.g., letter 'A' → 3D text)
OP_MESH_REVOLVE: 2D profile → 3D rotation (e.g., vase, cup)
OP_MESH_SWEEP: 2D shape along 3D path (e.g., pipe, rope)
OP_MESH_SUBDIVIDE: Add geometry detail (smooth surfaces)
```

**Boolean Operations (CSG)**:
```
OP_BOOLEAN_UNION: Combine meshes (A ∪ B)
OP_BOOLEAN_SUBTRACT: Cut shapes (A - B)
OP_BOOLEAN_INTERSECT: Keep overlap (A ∩ B)
OP_BOOLEAN_XOR: Symmetric difference
```

**Procedural Generation**:
```
OP_LSYSTEM_STEP: L-system iteration (trees, plants, fractals)
OP_NOISE_3D: Perlin/Simplex noise (terrain, clouds, organic shapes)
OP_FRACTAL_SUBDIVIDE: Fractal detail addition
OP_VORONOI_3D: Cell-based patterns (rocks, skin, foam)
```

**Deformation & Sculpting**:
```
OP_DISPLACEMENT_MAP: Height-based deformation (terrain from heightmap)
OP_LATTICE_DEFORM: Control cage deformation (bend, twist)
OP_SMOOTH_LAPLACIAN: Mesh smoothing (sculpt-style)
OP_INFLATE: Volume expansion (balloon effect)
```

**Parametric & Mathematical**:
```
OP_BEZIER_SURFACE: NURBS-like smooth surfaces
OP_CATMULL_CLARK: Subdivision surface (smooth from coarse)
OP_IMPLICIT_SURFACE: f(x,y,z) = 0 evaluation (blobs, metaballs)
OP_MARCHING_CUBES: Isosurface extraction (volume → mesh)
```

**Physics-Driven**:
```
OP_PHYSICS_SIMULATE: Rigid body dynamics (objects falling, colliding)
OP_SOFTBODY_STEP: Elastic deformation (cloth, jelly)
OP_FLUID_ADVECT: Liquid simulation (water, lava)
OP_PARTICLE_EMIT: Particle systems (fire, smoke, sparks)
```

**UV & Texturing**:
```
OP_UV_PROJECT: Automatic UV unwrap (cylindrical, spherical, planar)
OP_UV_OPTIMIZE: Pack UV islands efficiently
OP_TEXTURE_BAKE: Procedural → bitmap texture
OP_TRIPLANAR_MAP: Seamless 3D texture projection
```

**All Opcodes**: Composable via RPN, symlinked to avoid duplication.

### 10.4 2D ↔ 3D Fusion (Same Knowledge Base)

**Key Insight**: Drawing Galaxy knowledge used for BOTH 2D and 3D:

```
Drawing Galaxy Entry: "brick_pattern"
  - form_rpn: [RECT, 0.1, 0.2, LINE, OFFSET, ...] (how to draw brick)
  - color_rpn: [RGB, 0.6, 0.3, 0.1, NOISE, ...] (brick color variation)

Used in 2D:
  - Render brick pattern as 2D image
  - Drawing Bridge executes form_rpn + color_rpn
  - Output: 2D texture

Used in 3D:
  - Reference brick_pattern as material texture
  - UV mapping → form_rpn determines pattern placement
  - Bump mapping → form_rpn LINE edges = surface detail
  - Output: 3D textured wall

Same Knowledge, Different Application - Zero Duplication
```

**2D Drawing → 3D Extrusion Example**:
```
Drawing Galaxy: "star_shape"
  - RPN: [MOVE, 0, 0, LINE, 1, 0, ROTATE, 72°, LOOP, 5]

3D Composition:
  - star_3d_rpn = [
      DRAWING_REF, "star_shape",  # Symlink to 2D shape
      MESH_EXTRUDE, 0.5,          # Extrude 0.5 units
      BEVEL_EDGES, 0.1,           # Round corners
    ]

Result: 3D star created from 2D drawing knowledge
```

**Traditional Engines**: Separate SVG import, manual conversion, lost editability.
**K3D**: Direct symlink, fully procedural, always editable.

---

## Part 11: Unified Signal Architecture (Waves = Visual)

### 11.1 The Fundamental Unity

**All Waves Are Frequency Over Time**:

```
Sound Wave:
  - Frequency: 20 Hz - 20 kHz
  - Representation: amplitude(t) at each frequency
  - Visualization: Spectrogram (frequency on Y, time on X, amplitude as color)

Radio Wave:
  - Frequency: 3 kHz - 300 GHz
  - Representation: signal(t) at carrier frequency
  - Visualization: Same spectrogram format

Light Wave:
  - Frequency: 430-770 THz (visible spectrum)
  - Representation: intensity(λ) per wavelength
  - Visualization: Spectrum analyzer or image (spatial frequencies)

Electromagnetic Spectrum:
  - All are EM waves, just different frequencies
  - ALL visualizable as frequency-over-time
  - ALL processable with SAME procedural tools
```

**K3D Unified Signal Specification** (already exists in `docs/vocabulary/UNIFIED_SIGNAL_SPECIFICATION.md`):
- Treats audio, RF, video, light as **frequency components over time**
- Spectrogram = VectorDotMap (same procedural codec as images)
- Cross-modal discovery (find connections between sounds and images)

### 11.2 Audio ↔ Visual Fusion

**Spectrogram as Visual Data**:
```
Audio RPN Program: "thunder"
  - Waveform: [OSCILLATOR, 100Hz, NOISE, ENVELOPE, sharp_attack]
  - Executes → Audio sample

Visualize:
  - FFT → Frequency spectrum
  - Plot frequency(Y) vs time(X) vs amplitude(color)
  - Result: Spectrogram (image!)

Drawing Galaxy can RENDER this:
  - Drawing RPN: [GRADIENT, low_freq → high_freq, RECT_MAP]
  - Same VectorDotMap codec used for ANY image
```

**Visual → Audio (Reverse)**:
```
Image with vertical lines (e.g., barcode):
  - Spatial frequency analysis (Fourier transform)
  - Interpret vertical lines as audio frequencies
  - Synthesize waveform: line spacing → pitch
  - Result: Audio from image pattern
```

**This is NOT science fiction** - it's already done in:
- Spectral audio editing (iZotope RX, Audacity spectrogram view)
- Data sonification (translate images to sound)
- Optical sound (barcode music, visual audio synthesis)

**K3D Advantage**: Because Audio + Drawing share the **same procedural RPN substrate**:
- Audio programs can be VISUALIZED automatically
- Visual patterns can be SONIFIED automatically
- Same FFT kernels work on audio + image data
- Cross-modal composition is NATIVE, not a hack

### 11.3 Unified Signal Opcodes

**Frequency Domain Operations** (Work on Audio, RF, Light, Images):
```
OP_FFT_FORWARD: Time/space → frequency domain
OP_FFT_INVERSE: Frequency → time/space domain
OP_FREQUENCY_FILTER: Bandpass/lowpass/highpass (works on audio, RF, image)
OP_SPECTRAL_MASK: Frequency-selective gating
OP_PHASE_SHIFT: Phase manipulation (audio effects, image filters)
```

**Temporal Operations** (Work on Audio, Video, Time-Series):
```
OP_TEMPORAL_RESAMPLE: Change sample rate (audio) or frame rate (video)
OP_TEMPORAL_STRETCH: Time dilation (slow-motion video, audio pitch shift)
OP_TEMPORAL_WINDOW: Extract time slice (audio segment, video frame range)
OP_TEMPORAL_CONVOLVE: Reverb (audio), motion blur (video)
```

**Cross-Modal Operations**:
```
OP_AUDIO_TO_SPECTROGRAM: Waveform → visual frequency plot
OP_SPECTROGRAM_TO_AUDIO: Visual → waveform (reverse synthesis)
OP_IMAGE_TO_SPECTRUM: Spatial frequencies → 1D spectrum
OP_SPECTRUM_TO_IMAGE: Frequency data → visual representation
```

**Visualization Operations**:
```
OP_WAVEFORM_PLOT: Audio → time-domain graph (oscilloscope view)
OP_SPECTRUM_PLOT: Frequency → magnitude graph (spectrum analyzer)
OP_PHASE_PLOT: Frequency → phase graph (phase meter)
OP_WATERFALL_PLOT: Spectrogram with time scrolling (radio SDR display)
```

**All Unified**: Same RPN opcodes work on audio, RF, video, image data because they're ALL frequency-over-time.

### 11.4 Video as Temporal Image Composition

**Traditional Video Models**:
```
Diffusion-based (Sora, Runway):
  - Generate frames via latent diffusion
  - Temporal coherence via attention mechanisms
  - Each frame = separate generation (expensive)

Interpolation-based:
  - Generate keyframes
  - Interpolate between them
  - Limited to smooth motion
```

**K3D Procedural Video**:
```
Video = Sequence of Transformations (RPN Programs)

Example: "Ball bouncing"
  - Frame 0: ball_position = [0, 10]
  - Physics RPN: [GRAVITY, -9.8, INTEGRATE_VELOCITY, dt]
  - Frame 1: ball_position = physics_rpn(previous_state)
  - Frame 2: ball_position = physics_rpn(previous_state)
  - ...

Result: Video = ONE procedural program executed over time
        Not "generate 30 frames" but "run physics for 30 steps"
```

**Symlinked Composition**:
```
Video: "Sunset over ocean with birds flying"

Composition RPN:
  - Background: [SKY_GRADIENT_REF, "sunset", OCEAN_WAVES_REF, physics_id]
  - Objects: [BIRD_ANIMATION_REF, "seagull", FLOCK_BEHAVIOR_REF, boid_rules]
  - Temporal: [TEMPORAL_SEQUENCE, 0→300 frames, LIGHT_ANGLE, animate]

Symlinks:
  - SKY_GRADIENT_REF → Drawing Galaxy (2D pattern)
  - OCEAN_WAVES_REF → Reality Galaxy (fluid physics)
  - BIRD_ANIMATION_REF → Biology Galaxy (wing movement L-system)
  - FLOCK_BEHAVIOR_REF → Biology Galaxy (boid flocking rules)

Result: Video composed from existing knowledge, ZERO duplication
        Change "sunset" to "dawn" → entire video updates
        Change bird species → flock behavior reused
```

**K3D Advantage over Traditional Video Models**:
- **Consistency**: Physics ensures motion is realistic (not hallucinated)
- **Editability**: Change any component → video updates procedurally
- **Efficiency**: No frame-by-frame generation, just temporal execution
- **Composability**: Mix 2D (sky) + 3D (birds) + physics (waves) seamlessly

---

## Part 12: Enhanced Opcode + Knowledge Architecture

### 12.1 The Two-Layer Enhancement

**User's Insight**: "We need opcodes + knowledge"

**Opcodes** (Computation Layer - PTX Kernels):
- What CAN be computed (primitive operations)
- Examples: FFT, mesh boolean, L-system step, gradient blend

**Knowledge** (Galaxy Universe - RPN Programs):
- What IS known (combinations, patterns, techniques)
- Examples: "brick_pattern" RPN, "tree_growth" L-system, "thunder" waveform

**Synergy**: Opcodes enable computation, Knowledge provides composition strategies.

### 12.2 Galaxy Population Strategy (Knowledge Layer)

**Drawing Galaxy** (2D + Textures):
```
Patterns (1000+ entries):
  - Geometric: brick, tile, hexagon, checkerboard
  - Organic: wood grain, marble, skin, bark
  - Procedural: Perlin noise, Voronoi, fractal
  - Artistic: hatching, stippling, watercolor

Each Entry:
  - form_rpn: How to draw pattern
  - color_rpn: How to color pattern
  - metadata: Style tags, usage hints
  - symlinks: References to primitives (LINE, CIRCLE, etc.)
```

**Reality Galaxy** (3D + Physics):
```
Objects (500+ entries):
  - Natural: tree, rock, cloud, mountain
  - Architectural: wall, door, window, roof
  - Mechanical: gear, spring, lever, pulley
  - Organic: bone, muscle, blood vessel, neuron

Each Entry:
  - geometry_rpn: How to generate shape (L-system, CSG, parametric)
  - physics_rpn: How object behaves (gravity, collision, elasticity)
  - material_rpn: Visual + physical properties
  - growth_rpn: How object evolves over time (optional)
  - symlinks: References to Drawing Galaxy (textures), Math Galaxy (parameters)
```

**Audio Galaxy** (Sound + Music):
```
Sounds (200+ entries):
  - Environmental: thunder, rain, wind, fire
  - Musical: piano, violin, drum, synthesizer
  - Effects: reverb, delay, chorus, distortion
  - Voice: phonemes, prosody, emotion

Each Entry:
  - waveform_rpn: Oscillator + envelope + harmonics
  - effect_rpn: Processing chain (filter, reverb, etc.)
  - temporal_rpn: How sound evolves over time
  - visualization_rpn: Spectrogram representation (symlink to Drawing Galaxy)
  - symlinks: References to Math Galaxy (frequencies), Reality Galaxy (physics of sound)
```

**Grammar Galaxy** (Cross-Modal Rules):
```
Composition Rules (100+ entries):
  - "sunset" → [SKY_GRADIENT, orange→purple, LIGHT_ANGLE, low]
  - "tree" → [LSYSTEM_GROWTH, trunk+branches, BARK_TEXTURE, brown]
  - "thunder" → [LOW_FREQUENCY, 100Hz, SHARP_ATTACK, reverb]
  - "bouncing ball" → [GRAVITY_PHYSICS, elastic_collision, SPHERE_MESH]

Each Entry:
  - input: Semantic description (text, tags)
  - output: RPN program composition strategy
  - constraints: Physical laws, aesthetic rules
  - symlinks: References to Drawing/Reality/Audio galaxies
```

**Total Knowledge Base**: 1800+ procedural programs, ALL symlinked (zero duplication).

### 12.3 Opcode Justification (Computation Layer)

**New Opcodes MUST**:
1. Be composable from existing primitives (for verification)
2. Demonstrate measurable performance benefit vs pure composition
3. Have clear domain semantics

**Example Justification: OP_LSYSTEM_STEP**

**Composable from Existing**:
```
L-system step CAN be done with existing ops:
  - OP_BRANCH (conditional)
  - OP_LOOP (iteration)
  - OP_STORE/RECALL (state management)
  - String operations (pattern matching)

But it requires 100+ RPN instructions per iteration
```

**Performance Benefit**:
```
Pure composition: 100+ RPN ops × 10 iterations = 1000 ops
OP_LSYSTEM_STEP: Specialized kernel, 10 iterations = 10 ops (100× faster)
```

**Domain Semantics**:
```
L-system is FUNDAMENTAL to procedural generation:
  - Trees, plants, fractals, architecture
  - Used across biology, computer graphics, urban planning
  - Domain-specific opcode justified
```

**Result**: OP_LSYSTEM_STEP is a valid opcode target because it has clear procedural semantics and likely acceleration value, but it must still pass the promotion pipeline and is not assumed to be implemented in the current PTX runtime surface.

---

## Part 13: Revised Implementation Roadmap (Fusion-Aware)

### Phase 1: Drawing + Reality Fusion (Q2 2026)

**Goal**: 2D patterns automatically become 3D textures and geometry

**Deliverables**:
1. **Drawing Galaxy population** (1000+ patterns)
   - Geometric, organic, procedural, artistic
   - Each pattern usable as 2D image OR 3D texture
2. **Reality Galaxy population** (500+ objects)
   - Natural, architectural, mechanical, organic
   - Each object references Drawing Galaxy for textures (symlink)
3. **Fusion opcodes**:
   - OP_MESH_EXTRUDE (2D → 3D)
   - OP_UV_PROJECT (automatic texture mapping)
   - OP_TRIPLANAR_MAP (seamless 3D texturing)
4. **Fusion kernels**:
   - mesh_extrude.cu, uv_project.cu, triplanar_map.cu
5. **TRM 2D/3D Fusion Specialist**:
   - Learns when to use 2D vs 3D vs combined
   - Train on paired data (image + 3D model)
   - ~700K parameters

**Success**: "Generate brick wall" creates:
- 2D brick texture (Drawing Galaxy)
- 3D wall geometry (Reality Galaxy mesh)
- Texture applied via UV mapping (automatic)
- Physically plausible (gravity, collision)
- ALL from ONE composed RPN program

### Phase 2: Audio ↔ Visual Fusion (Q2-Q3 2026)

**Goal**: Sounds are visualizable, images are sonifiable (Unified Signal)

**Deliverables**:
1. **Audio Galaxy population** (200+ sounds)
   - Environmental, musical, effects, voice
   - Each sound has visualization_rpn (spectrogram)
2. **Unified Signal opcodes**:
   - OP_FFT_FORWARD/INVERSE
   - OP_AUDIO_TO_SPECTROGRAM / OP_SPECTROGRAM_TO_AUDIO
   - OP_FREQUENCY_FILTER (works on audio + images)
3. **Unified Signal kernels**:
   - fft_transform.cu, spectrogram_render.cu, frequency_filter.cu
4. **TRM Audio/Visual Fusion Specialist**:
   - Learns cross-modal patterns (thunder sound = dark jagged spectrogram)
   - Train on audio-visual paired data
   - ~500K parameters

**Success**: "Visualize thunder"
- Generates thunder waveform (Audio Galaxy)
- Automatically renders spectrogram (Drawing Galaxy symlink)
- Same RPN program, two outputs (audio + visual)

### Phase 3: 3D Multi-Technique Fusion (Q3 2026)

**Goal**: Combine ALL 3D techniques in single compositions

**Deliverables**:
1. **Reality Galaxy enhancements**:
   - CSG library (boolean operation templates)
   - Procedural library (L-systems, fractals, noise)
   - Physics library (simulation-driven shapes)
   - All combinable via symlinks
2. **3D Fusion opcodes**:
   - OP_BOOLEAN_UNION/SUBTRACT/INTERSECT
   - OP_LSYSTEM_STEP
   - OP_NOISE_3D
   - OP_MARCHING_CUBES
3. **3D Fusion kernels**:
   - boolean_operations.cu, lsystem_evaluator.cu, noise_3d.cu, marching_cubes.cu
4. **TRM 3D Composer Specialist**:
   - Learns to combine techniques (CSG + procedural + physics)
   - Train on complex 3D models (decomposed into techniques)
   - ~1M parameters

**Success**: "Generate ancient tree on cliff"
- L-system growth (trunk + branches)
- CSG boolean (cliff subtract tree roots)
- Physics simulation (roots grip cliff, stable)
- Procedural noise (bark texture, rock detail)
- ALL techniques composed in ONE RPN program

### Phase 4: Video as Temporal Fusion (Q3-Q4 2026)

**Goal**: Video = temporal execution of fused 2D/3D/audio programs

**Deliverables**:
1. **Temporal Grammar Galaxy**:
   - Animation rules (how objects move/change over time)
   - Sequence composition (shot 1 → shot 2 transitions)
   - Symlinks to all other galaxies
2. **Video opcodes**:
   - OP_TEMPORAL_SEQUENCE
   - OP_TEMPORAL_INTERPOLATE
   - OP_PHYSICS_INTEGRATE (time-stepping)
3. **Video kernels**:
   - temporal_sequence.cu, interpolation.cu, physics_integrate.cu
4. **TRM Temporal Composer Specialist**:
   - Learns shot composition, transitions, temporal coherence
   - Train on video datasets (decomposed into temporal programs)
   - ~800K parameters

**Success**: "Generate sunset timelapse with birds"
- Sky gradient animates (Drawing Galaxy, temporal)
- Birds flock (Biology Galaxy, boid rules, temporal)
- Ocean waves (Reality Galaxy, fluid sim, temporal)
- Audio ambience (Audio Galaxy, wind + waves)
- ONE composed program executed over time → video

---

## Part 14: The Fusion Advantage (K3D vs Traditional)

### Why K3D Can Do This (Others Can't)

**1. Everything is Procedural RPN**:
```
Traditional: Pixels (image), vertices (3D), samples (audio) - different data types
K3D: RPN programs for ALL - same data type, composable
```

**2. Symlink Architecture (Save Information Principle)**:
```
Traditional: Duplicate assets (texture file copied to 10 materials)
K3D: Reference once (texture_id symlinked by 10 materials) - zero duplication
```

**3. Unified Computation (PTX Kernels)**:
```
Traditional: CPU for physics, GPU for rendering, separate audio engine
K3D: ALL on GPU via PTX kernels - same execution substrate
```

**4. Spatial Memory (Galaxy Universe)**:
```
Traditional: Sequential access (load texture → load mesh → load physics)
K3D: Spatial access (nearby in 3D space = related concepts) - navigate, not load
```

**5. TRM Navigation (Learned Composition)**:
```
Traditional: Hardcoded pipelines (3D pipeline ≠ 2D pipeline)
K3D: TRM learns to compose from ANY galaxy - flexible, adaptive
```

### Compositions Impossible in Traditional Engines

**Example 1: "Visualize Music as Growing Tree"**
```
K3D Composition:
  - Audio Galaxy: music waveform → FFT → frequency spectrum
  - Math Galaxy: frequency → branch angle (map 100Hz-1kHz to 0-90°)
  - Biology Galaxy: L-system growth (branch rule from frequency)
  - Drawing Galaxy: bark texture (brightness from amplitude)
  - Reality Galaxy: physics (tree sways with bass frequencies)

Result: Tree that GROWS and MOVES based on music frequencies

Traditional: Would require:
  - Custom audio analysis code
  - Custom procedural generation code
  - Custom physics integration
  - Manual scripting to connect all pieces
  - Brittle, non-reusable, hard to modify
```

**Example 2: "Generate 3D Object from Sound"**
```
K3D Composition:
  - Audio Galaxy: thunder waveform
  - Unified Signal: waveform → spectrogram (frequency over time)
  - Drawing Galaxy: spectrogram rendered as heightmap
  - Reality Galaxy: heightmap → marching cubes → 3D mesh
  - Physics validation: ensure mesh is stable (not floating)

Result: 3D sculpture shaped by sound frequencies

Traditional: Would require separate tools for EACH step, manual export/import between them
```

**Example 3: "Animate Texture Based on Physics"**
```
K3D Composition:
  - Reality Galaxy: water simulation (fluid dynamics)
  - Drawing Galaxy: water surface as normal map (waves = texture)
  - Geometry Galaxy: flat plane mesh
  - Temporal: physics simulation over time
  - Material: normal map applied to mesh, updates each frame

Result: Animated water texture driven by real fluid simulation

Traditional: Pre-baked texture sequence (not real-time, huge storage) OR separate physics engine (not integrated)
```

---

## Part 15: Success Metrics (Fusion-Aware)

**Technical**:
- ✅ Cross-galaxy composition works (2D+3D+audio+physics in ONE program)
- ✅ Symlinks validated (reference counting, zero duplication)
- ✅ Unified Signal proven (audio ↔ visual bidirectional)
- ✅ All techniques composable (CSG + L-system + physics + noise in one object)

**Performance**:
- ✅ Fusion latency: <1 second for complex compositions (tree with texture + physics)
- ✅ GPU memory: <500MB for 1000+ procedural programs (via symlinks)
- ✅ Execution speed: 100k+ RPN ops/sec (PTX kernel efficiency)

**Benchmarks**:
- ✅ 3D generation: ShapeNet quality (but procedural, editable)
- ✅ Audio generation: AudioSet quality (but visualizable)
- ✅ Video generation: Comparable to Sora (but physically consistent)

**Unique Capabilities** (Can K3D do what NO other engine can?):
- ✅ Visualize any audio as spectrogram (automatic, native)
- ✅ Compose 3D from mixed techniques (CSG + procedural + physics)
- ✅ 2D texture → 3D geometry (automatic extrusion + UV mapping)
- ✅ Video from temporal physics (not generated frames, simulated reality)
- ✅ Cross-modal discovery (find sounds similar to image patterns)

---

**End of Enhancement**

**Summary**: K3D's procedural fusion enables compositions impossible in traditional engines because:
1. **Everything is RPN** (composable)
2. **Symlinks save information** (zero duplication)
3. **Unified Signal** (audio = visual = frequency over time)
4. **All techniques available** (3D CSG + L-system + physics + noise)
5. **TRM learns composition** (not hardcoded pipelines)

**This is K3D's unique competitive advantage.**

---

## Part 16: Codex Grounding Addendum (What Is Actually Executable Today)

This section grounds the fusion vision in the current runtime surface so implementation stays inside K3D's real sovereignty boundaries.

### 16.1 Current Sovereign Runtime Surface

**What exists in code now**:

- `167` opcode constants in `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py`
- `63` kernel files in `knowledge3d/cranium/kernels/`
  - `26` PTX artifacts
  - `37` CUDA source files
- **Tiered execution stack**:
  - Tier 1: `knowledge3d/cranium/bridges/lightweight_rpn.py`
  - Tier 2: `knowledge3d/cranium/bridges/sovereign_bridges.py`
  - Tier 3: `knowledge3d/cranium/bridges/advanced_rpn.py`
  - Dispatcher: `knowledge3d/cranium/bridges/tiered_rpn.py`
  - High-level compiler/wrapper: `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py`
  - Sovereign CUDA loader: `knowledge3d/cranium/sovereign/loader.py`

**What is test-backed today**:

- Tier 1 arithmetic, comparisons, math ops, and stack ops
- Tier 2 cooperative/vector ops:
  - memcpy
  - fill
  - reductions
  - matvec
  - vector relu
  - vector multiply
  - sigmoid
  - entropy
- Tier 3 matrix ops:
  - matmul
  - determinant
  - inverse
  - trace
- Ternary ops:
  - `tadd`
  - `tmul`
  - `tquant`
  - `tcomp`
  - `tpack`
  - `tunpack`
- Programmable memory surface:
  - `STORE`
  - `RECALL`
  - friendly `STORE_X` / `RECALL_X` expansion in `ModularRPNEngine`

### 16.2 Important Boundary: Not All `ptx_runtime/` Is Hot-Path Clean

The sovereign core is real, but the broader tree is mixed.

**Safe hot-path assumption**:
- Tiered RPN stack
- Sovereign loader
- Explicit PTX-backed bridges used by the tiered engines

**Do NOT assume all of these are hot-path-safe just because of the directory name**:
- `knowledge3d/cranium/ptx_runtime/*`
- `knowledge3d/cranium/bridges/*`

Some non-core modules still import `numpy`, `cupy`, or `torch`. That is acceptable for tooling, ingestion, and non-hot-path helpers, but not for the sovereign execution loop.

### 16.3 Three Classes of Multimodal Capability

To avoid architectural drift, multimodal features should be classified into exactly three buckets:

**Class A: Executable Now**
- Can be expressed with current opcodes and bridges.
- Example:
  - drawing primitives
  - STORE/RECALL-driven temporal state
  - vector/matrix transforms
  - ternary routing
  - simple procedural animation

**Class B: Representable Now, Kernel Later**
- Can be encoded immediately as Galaxy recipes or Grammar macros.
- Deserves a dedicated opcode only after usage frequency and performance justify it.
- Example:
  - L-system expansion
  - spectrogram pipelines
  - triplanar mapping
  - mesh extrusion from 2D contours
  - boid update rules

**Class C: Research / Not Yet Admitted**
- Requires new data structures, too many host-side assumptions, or unproven performance.
- Example:
  - full volumetric remeshing pipelines
  - robust production-grade cloth/fluids
  - heavy differentiable rendering loops

**Rule**: Every new multimodal proposal must state which class it belongs to.

---

## Part 17: Procedural Tools as First-Class Knowledge

Daniel's correction is crucial: the 3D/video/audio vision is not only "objects in galaxies". It is also **techniques themselves stored as reusable procedural means**.

### 17.1 The Missing Unit: Tool-Nodes

Traditional engines store:
- assets
- modifiers
- scripts
- node graphs

K3D should additionally store:
- **tool-nodes**

A tool-node is a procedural capability represented as knowledge, not just code.

**Examples**:
- `tool_extrude_profile_v1`
- `tool_revolve_profile_v1`
- `tool_triplanar_material_v1`
- `tool_lsystem_branching_v1`
- `tool_spectrogram_render_v1`
- `tool_temporal_camera_orbit_v1`

### 17.2 Tool-Node Contract

Each tool-node should be stored as a Galaxy entry with:

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

This matters because K3D's advantage is not only generating final artifacts. It is making the **means of generation** reusable, inspectable, and symlinkable.

### 17.3 Tool Families

**2D tool family**:
- contour construction
- stroke expansion
- fill/gradient composition
- spectral plotting

**3D tool family**:
- extrude
- revolve
- sweep
- boolean combine
- displacement from scalar field
- UV and triplanar mapping

**Signal tool family**:
- waveform synthesis
- FFT projection
- filter chain
- spectrogram render
- image sonification

**Temporal tool family**:
- keyframe interpolation
- time-window slicing
- camera path generation
- event-triggered state transition

**Physics tool family**:
- gravity integration
- collision correction
- boid steering
- spring-mass relaxation
- field sampling

### 17.4 Why Tool-Nodes Matter

Without tool-nodes, K3D risks re-creating traditional engines:
- assets in one place
- code elsewhere
- undocumented composition logic

With tool-nodes:
- the engine's "verbs" become part of the knowledge substrate
- TRM learns not only what things are, but which techniques to apply
- imported legacy workflows can be transmuted into PM-KR-compatible procedural means

---

## Part 18: Video Should Be Treated as Scene-Time, Not Frame-Time

The earlier enhancement is directionally correct, but the implementation target should be stricter:

**Video in K3D is not a list of frames.**
**Video in K3D is a scene program evaluated over time.**

### 18.1 Five-Layer Temporal Video Contract

Every procedural video composition should separate:

1. **Scene Layer**
   - what exists in the world
   - objects, materials, sky, terrain, fluids

2. **Dynamics Layer**
   - what changes state
   - physics, growth, flocking, deformation, lighting drift

3. **Camera Layer**
   - point of view
   - orbit, pan, zoom, cuts, focus

4. **Render Layer**
   - how to turn state into view
   - image formation, shading, spectrogram, overlays

5. **Audio Layer**
   - waveform or event-driven sound state
   - ambience, source positioning, signal synthesis

This keeps video procedural and editable at the correct semantic level.

### 18.2 Canonical Example: "Sunset Ocean Timelapse"

```text
scene_rpn:
  SKY_REF sunset_gradient_v2
  WATER_REF ocean_surface_v4
  BIRD_REF seagull_flock_v1

dynamics_rpn:
  LIGHT_ANGLE animate_low_to_horizon
  WAVE_STATE fluid_iterate
  FLOCK_STATE boid_step

camera_rpn:
  ORBIT coastline_path_v1
  ZOOM slow_in

render_rpn:
  DRAWING_RENDER sky
  REALITY_RENDER water
  DRAWING_OVERLAY glare

audio_rpn:
  WIND_REF coastal_wind_v2
  SURF_REF shore_break_v1
  BIRD_AUDIO_REF gull_call_v1
```

This should be the target representation, not "generate 300 frames".

### 18.3 Video Benchmarking Must Reflect Procedural Value

For K3D, video quality is not only visual plausibility.

It must also measure:
- determinism of regeneration
- editability of a single component
- reuse of referenced tool-nodes
- cross-modal consistency between audio and image
- compression ratio versus frame storage

That is a stronger benchmark than pure perceptual similarity.

---

## Part 19: Opcode Admission and Promotion Pipeline

The earlier spec correctly says "we need opcodes + knowledge". The missing implementation rule is **when** a recipe becomes an opcode.

### 19.1 Promotion Stages

**Stage 0: Galaxy Recipe**
- Store the technique as Grammar/Reality/Drawing/Audio knowledge only.
- No new opcode.
- Prove semantic usefulness first.

**Stage 1: Macro Surface**
- Add a stable macro or token expansion in the high-level compiler.
- Example:
  - `STORE_X`
  - `RECALL_DISC`
  - future `SPECTROGRAM_MACRO`
- Still no new kernel.

**Stage 2: Opcode Candidate**
- Measure repeated usage.
- Profile cost of recipe/macro execution.
- Show that dedicated opcode reduces complexity materially.

**Stage 3: PTX Kernel Admission**
- Add kernel only after:
  - composability proof
  - measurable speedup
  - clean domain semantics
  - test coverage
  - sovereignty review

### 19.2 Admission Rubric

A new opcode is justified only if all are true:

1. **Recipe already exists**
   - it has already been expressed as composition

2. **Frequency is high**
   - appears in many tool-nodes or object recipes

3. **Speedup is meaningful**
   - not 5 percent, but enough to matter architecturally

4. **Semantics are stable**
   - same meaning across use cases

5. **It reduces graph complexity**
   - fewer references, cleaner composition, easier TRM routing

### 19.3 Immediate Consequence for This Spec

The following should be treated as **target names**, not assumed-to-exist runtime ops:

- `OP_MESH_EXTRUDE`
- `OP_BOOLEAN_UNION`
- `OP_BOOLEAN_SUBTRACT`
- `OP_BOOLEAN_INTERSECT`
- `OP_LSYSTEM_STEP`
- `OP_MARCHING_CUBES`
- `OP_FFT_FORWARD`
- `OP_AUDIO_TO_SPECTROGRAM`
- `OP_SPECTROGRAM_TO_AUDIO`

They are valid design targets, but must pass the promotion pipeline above.

---

## Part 20: Revised Phase Order (Grounded)

The roadmap should slightly shift from "add opcodes first" to "toolify first, promote later".

### Phase 0: Toolify Existing Surface

Before adding new opcodes:
- populate Galaxy with tool-nodes
- encode multimodal recipes using current opcodes where possible
- create benchmark tasks for:
  - 2D -> 3D extrusion recipes
  - audio -> spectrogram recipes
  - temporal scene programs

### Phase 1: Draw-Extrude-Texture Minimum Viable Fusion

The first deliverable should not be generic text-to-image.
It should be:

**"2D contour + procedural material -> textured 3D object"**

Why:
- directly demonstrates PM-KR symlink value
- reuses Drawing + Reality + Math together
- stays closer to currently available runtime primitives
- produces a stronger K3D-native demo than yet another image generator

### Phase 2: Spectrogram / Signal Fusion

Next deliverable:

**"sound <-> visual spectrum <-> surface displacement"**

Why:
- directly validates Unified Signal
- lets audio and drawing share one procedural substrate
- creates a clean bridge into video later

### Phase 3: Temporal Scene Programs

Only after the above:
- camera tracks
- scene-time deltas
- procedural clips
- audio-visual sync via shared signal semantics

### Phase 4: Opcode Promotion

Promote the highest-value recipes to kernels after usage data exists.

This order is slower intellectually, but faster in practice because it avoids inventing an oversized opcode surface before the tool grammar stabilizes.

---

## Part 21: K3D-Specific Benchmarks (Not Borrowed from Diffusion Models)

K3D should not only chase conventional multimodal benchmarks. It should define metrics that reflect procedural superiority.

### 21.1 New Metrics

**Deterministic Rebuild Score**
- same source program regenerates same artifact

**Variant Edit Cost**
- number of changed refs/tokens required to create a meaningful variant

**Symlink Reuse Ratio**
- percentage of final artifact sourced by references rather than duplication

**Cross-Modal Reuse Count**
- how many modalities reuse the same underlying procedural source

**Temporal Compression Ratio**
- scene-time program size versus equivalent frame-by-frame storage

### 21.2 Why These Matter

These metrics reveal value that diffusion benchmarks hide:
- editability
- inspectability
- knowledge reuse
- long-term maintainability
- sovereign compression

That is where K3D should win.

---

## Part 22: Final Implementation Principle

The strongest version of this architecture is:

**knowledge first**
-> **tools as knowledge**
-> **recipes before opcodes**
-> **opcodes before kernels only when justified**
-> **temporal scene programs instead of frame generators**

That keeps the multimodal expansion faithful to PM-KR and to K3D's real runtime constraints.
