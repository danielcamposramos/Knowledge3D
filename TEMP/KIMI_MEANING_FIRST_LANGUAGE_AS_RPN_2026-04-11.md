# Kimi Agent Swarm Result
**Task:** K3D architecture: design how MEANING-CENTRIC stars + LANGUAGE-AS-RPN-MATH should work. Daniel's vision: TRM stores ONLY RPN problem-solving strategies (one generic model, no domain-specific naming lik
**Mode:** instant
**Time:** 2026-04-11 16:07:36

────────────────────────────────────────────────────────────────

## Sub-Agent A: Design the LANGUAGE-AS-RPN-MATH composition path. How does TRM produce a sentence in language X for a meaning M, given: meaning star carries language-agnostic semantic content; grammar galaxy carries RPN transformation rules per language; word galaxies (one per language) carry surface-form stars symlinked to meaning. The composition must be PURE RPN MATH (no Python string formatting, no translation tables). Specify: the sentence-synthesis RPN program structure, how Grammar Galaxy rules dispatch by target language, how the saudades-edge-case (single-language meaning) is handled without breaking universality, how meaning_rpn for word stars stays human-readable AND machine-executable, how this differs from current LLM approach. Be specific about opcodes and program structure. Reference existing K3D specs: FOUNDATIONAL_KNOWLEDGE, MEANING_CENTRIC_STAR_SCHEMA, RPN_DOMAIN_OPCODE_REGISTRY, DUAL_CLIENT_CONTRACT.</sub_task_a>
<parameter name="sub_task_b">Design the LAYER-0 INSTANCING + KNOWLEDGE LOD path. How does the same drawing primitive set (Layer 0 of FOUNDATIONAL_KNOWLEDGE_SPECIFICATION) instance: (1) a glyph for character 'a', (2) a wall for House room, (3) a book mesh, (4) a hologram in Knowledgeverse view, (5) a UI element. All from the SAME Galaxy assets via GPU instancing (one VBO loaded, N instance transforms). Then: how does LOD + Frustum Culling (already used for visual rendering) ALSO apply to KNOWLEDGE itself in the Knowledgeverse? Stars at distance get coarse representation; only nearby stars expand to full RPN bytecode. Specify: how a Book object in the House transitions from 'closed 3D asset' (Layer 0 mesh + binding) to 'opened Galaxy' (its contents become loaded stars in Knowledgeverse via symlinks), how the same RPN drawing programs render BOTH the House book AND the stars when opened, how this sustains 'one model fits all modalities' for TRM. Reference existing K3D specs: SPATIAL_GENERAL_INTELLIGENCE, KNOWLEDGEVERSE, MEMORY_TABLET, THREE_BRAIN_SYSTEM.</sub_task_b>
<parameter name="think">true

I'll design the LANGUAGE-AS-RPN-MATH composition path, focusing on pure RPN mathematics for sentence synthesis.

## RPN Sentence-Synthesis Program Structure

### Core Program Layout

```
┌─────────────────────────────────────────────────────────┐
│  RPN_SENTENCE_SYNTHESIS_PROGRAM (meaning_rpn field)     │
├─────────────────────────────────────────────────────────┤
│  [MEANING_STAR_REF] [GRAMMAR_GALAXY_REF] [LANG_TAG]     │
│  MEANING_EXTRACT    GRAMMAR_SELECT      WORD_RESOLVE    │
│  ─────────────────────────────────────────────────────  │
│  [morpheme*] ... [morpheme*]  ← RPN morpheme stack ops  │
│  MORPHEME_SEQUENCE  SYNTAX_VERIFY  SURFACE_EMIT          │
└─────────────────────────────────────────────────────────┘
```

### Specific Opcodes (from RPN_DOMAIN_OPCODE_REGISTRY)

| Opcode | Hex | Stack Effect | Purpose |
|--------|-----|--------------|---------|
| `MEANING_LOAD` | `0xM101` | `meaning_id → meaning_star*` | Dereference meaning star from embedding |
| `GRAMMAR_FETCH` | `0xG201` | `lang_tag grammar_galaxy* → grammar_rules*` | Load RPN transform rules for target |
| `WORD_RESOLVE` | `0xW301` | `meaning_star* lang_tag → word_star*` | Find surface form in word galaxy |
| `MORPHEME_PUSH` | `0xM401` | `word_star* → morpheme_stack*` | Push to construction stack |
| `MORPHEME_POP` | `0xM402` | `morpheme_stack* → word_star*` | Pop for assembly |
| `MORPHEME_SWAP` | `0xM403` | `a b → b a` | Reorder morphemes (syntax) |
| `MORPHEME_DUP` | `0xM404` | `a → a a` | Copy for agreement |
| `SYNTAX_COMPOSE` | `0xS501` | `morpheme*[n] n → composed_phrase*` | Apply grammar rules |
| `AGREEMENT_APPLY` | `0xS502` | `phrase* grammar_rule* → phrase*` | Gender/number/case |
| `SURFACE_RENDER` | `0xR601` | `composed_phrase* → utf8_stream*` | Final string emission |

### Grammar Galaxy: Language-Specific RPN Rules

**File: `galaxies/grammar/universal.syntax`**

```
; Grammar rules stored as RPN subroutines, dispatched by LANG_TAG hash

GRAMMAR_SUBROUTINE_TABLE:
  [LANG_TAG_HASH_0]: @ENGLISH_SVO
  [LANG_TAG_HASH_1]: @JAPANESE_SOV
  [LANG_TAG_HASH_2]: @ARABIC_VSO
  ...
  [LANG_TAG_HASH_N]: @PORTUGUESE_SAUDADES_SPECIAL  ; edge case handler

@ENGLISH_SVO:          ; RPN subroutine
  MORPHEME_POP MORPHEME_POP MORPHEME_POP   ; S V O on stack
  SWAP                                     ; O S V → want S first
  ROT                                      ; S V O → S O V
  SWAP                                     ; S O V → S V O
  SYNTAX_COMPOSE 3                         ; compose 3 morphemes
  AGREEMENT_APPLY @ENGLISH_AGREEMENT
  RET

@JAPANESE_SOV:
  MORPHEME_POP MORPHEME_POP MORPHEME_POP
  ROT ROT                                  ; S V O → O V S
  SWAP                                     ; O V S → O S V  
  ROT                                      ; O S V → S O V
  SYNTAX_COMPOSE 3
  AGREEMENT_APPLY @JAPANESE_AGREEMENT
  RET
```

**Dispatch mechanism (pure RPN math):**

```
; GRAMMAR_SELECT implementation - no Python conditionals
lang_tag grammar_galaxy_addr
DUP2                    ; duplicate both
HASH_MOD TABLE_SIZE     ; hash lang_tag → index
MUL WORD_SIZE           ; byte offset
ADD                     ; grammar_galaxy_addr + offset
LOAD                    ; fetch subroutine address
CALL                    ; execute language-specific rules
```

## The Saudades Edge Case: Single-Language Meanings

### Problem
Portuguese "saudades" has no English equivalent. How to maintain universality without breaking the "meaning first" architecture?

### Solution: Canonical Surface Form as Self-Symlink

**File: `stars/meaning/saudades_0xM7A3F2.star`**

```json
{
  "meaning_id": "0xM7A3F2",
  "embedding": "[vector for cosine nav]",
  "meaning_rpn": {
    "program": "MEANING_LOAD SELF_REF LANG_CHECK PORTUGUESE_FALLBACK SURFACE_RENDER",
    "canonical_surface": {
      "language": "pt-BR",
      "form": "saudades",
      "type": "UNTRANSLATABLE_ROOT"
    }
  },
  "word_galaxy_links": {
    "pt-BR": "@word/pt/saudades",           ; direct hit
    "en-US": "@meaning/saudades_0xM7A3F2",  ; SELF-REFERENCE symlink
    "ja-JP": "@meaning/saudades_0xM7A3F2",  ; SELF-REFERENCE symlink
    "es-ES": "@word/es/anhelo"               ; approximate (different meaning)
  }
}
```

### RPN Handling in WORD_RESOLVE

```
WORD_RESOLVE implementation:
  meaning_star* lang_tag
  DUP2
  WORD_GALAXY_LOOKUP        ; attempt lookup in target word galaxy
  DUP IS_NULL               ; check if result is null/self-ref
  [BRANCH]
    POP POP                 ; discard null, keep originals
    CANONICAL_EXTRACT       ; extract canonical surface
    UNTRANSLATABLE_MARK     ; mark with provenance
  [ELSE]
    ; normal word resolution continues
  [END]
```

**Result for English query:** `"saudades" [pt-BR]` — surface form carries source language tag, allowing TRM to either:
- Use loanword directly
- Generate periphrastic approximation via RPN: `"longing for something lost that may never return"`

## Human-Readable AND Machine-Executable `meaning_rpn`

### Dual-Representation Format

**File: `stars/meaning/example.star` — field `meaning_rpn`**

```json
{
  "meaning_rpn": {
    "human_comment": [
      "; TRUTH: proposition P holds for entity E",
      "; MOOD: speaker believes P strongly",
      "; EVIDENCE: speaker witnessed P directly"
    ],
    "machine_program": [
      "0xM101",           // MEANING_LOAD
      "0x7A3F2",          // meaning_id (truth)
      "0xG201",           // GRAMMAR_FETCH  
      "0xA0B1C2",         // grammar_galaxy_ref
      "0x0001",           // lang_tag (English)
      "0xW301",           // WORD_RESOLVE
      "0xM401",           // MORPHEME_PUSH
      "0xM404",           // MORPHEME_DUP (for emphasis)
      "0xS502",           // AGREEMENT_APPLY
      "0x0003",           // @EVIDENTIAL_SYSTEM
      "0xR601"            // SURFACE_RENDER
    ],
    "symbolic_form": "TRUTH[witnessed]+strong_belief → English surface",
    "verification_hash": "0x..."
  }
}
```

### Execution Path (no Python string ops)

```
TRM inference:
1. Embedding lookup → meaning star address
2. RPN VM loads machine_program bytes
3. Each opcode: stack manipulation only
4. Final SURFACE_RENDER: UTF-8 byte stream to output buffer
   - No f-strings, no .format(), no template substitution
   - Pure: byte[] → byte[] transformation via RPN stack
```

## Contrast with Current LLM Approach

| Aspect | LLM Approach | K3D LANGUAGE-AS-RPN-MATH |
|--------|-----------|---------------------------|
| **Storage** | Trillions of weights memorize language patterns | TRM stores only RPN strategies; knowledge in Galaxy programs |
| **Translation** | Implicit in attention layers; English-centric bias | Explicit RPN dispatch; no privileged language |
| **Untranslatables** | Forced approximation or hallucination | Self-referential symlinks preserve authenticity |
| **Composition** | Statistical next-token prediction | Deterministic stack-based morpheme assembly |
| **Verifiability** | Black-box neural activation | Every step: inspectable RPN stack state |
| **Human-AI joint use** | Different representations (weights vs. text) | **Same RPN programs** — dual client contract |
| **New language** | Retrain entire model | Add grammar subroutine to Galaxy; TRM unchanged |

## Integration with K3D Specs

### FOUNDATIONAL_KNOWLEDGE
- Grammar galaxies are Layer 2+ knowledge: procedural rules, not static data

### MEANING_CENTRIC_STAR_SCHEMA  
- `meaning_rpn` field replaces `natural_language_description`
- Same star serves TRM inference AND human inspection

### RPN_DOMAIN_OPCODE_REGISTRY
- Language opcodes (`0xM4xx`, `0xS5xx`, `0xR6xx`) extend domain registry
- Same VM executes math, spatial, AND linguistic programs

### DUAL_CLIENT_CONTRACT
- Human reads symbolic_form: `"TRUTH[witnessed]+strong_belief"`
- Machine executes identical `machine_program` bytes
- No translation layer between human and AI representation

## Complete Synthesis Example

**Input:** `meaning_id=0xM7A3F2` (saudades-like concept), `lang_tag=en-US`

**RPN execution trace:**

```
Initial stack: [meaning_id: 0x7A3F2, lang: 0x0001]

MEANING_LOAD      → [meaning_star*@0x7A3F2]
GRAMMAR_FETCH     → [meaning_star*, grammar_rules*@ENGLISH_SVO]
WORD_RESOLVE      → [meaning_star*, word_star*→SELF_REF]  ; detects untranslatable
  → branches to CANONICAL_EXTRACT
  → [meaning_star*, canonical: "saudades", provenance: "pt-BR"]
MORPHEME_PUSH     → ["saudades"]
MORPHEME_DUP      → ["saudades", "saudades"]  ; emphasis via duplication
SYNTAX_COMPOSE 2  → ["saudades saudades"]     ; or periphrastic expansion
SURFACE_RENDER    → UTF-8: "saudades" [pt-BR]  ; or: "aching longing..."
```

**Output:** `"saudades"` with provenance metadata, or periphrastic construction via extended RPN subroutine for explanatory mode.

────────────────────────────────────────────────────────────────

## Sub-Agent B: Design the Layer 0 rendering and instantiation system — implement LINE/CIRCLE/RECT primitives as unified GPU instancing for all visual elements (glyphs, architecture, furniture, holograms), define LOD/FOV application to knowledge structures in both Knowledgeverse and House, and specify the 3D book object system with Galaxy loading via symlinks.

## Sub-Agent B: Layer 0 Rendering & Instantiation System Design

---

### 1. Unified GPU Instancing Architecture

**Core Principle:** Single instancing pipeline for ALL visual elements—no special-case renderers.

#### 1.1 Primitive Definition (`/src/layer0/primitives.h`)

```cpp
// Lines 1-45
enum class PrimitiveType : uint8_t {
    LINE    = 0,  // 2-vertex strip, width in screen pixels
    CIRCLE  = 1,  // center + radius, tessellated to N-gon
    RECT    = 2,  // axis-aligned or rotated, corner radius optional
    COUNT
};

// Unified instance data — 32 bytes, GPU-friendly
struct alignas(16) PrimitiveInstance {
    vec4  transform;   // x,y = position, z = scale, w = rotation
    vec4  color;       // RGBA, linear color space
    vec2  extent;      // width/height or radius/inner_radius
    uint32_t flags;    // primitive type [0:1], corner radius [2:15], etc.
    uint32_t userData; // material index, glyph codepoint, galaxy ID
};
```

#### 1.2 Multi-Tier Instancing (`/src/layer0/batch_renderer.cpp`)

| Tier | Use Case | Batch Size | Update Frequency |
|------|----------|-----------|------------------|
| **Static** | Architecture, furniture bases | 10K-100K instances | Never / load time |
| **Semi-static** | Books on shelves, decor | 1K-10K instances | Occasional (symlink swap) |
| **Dynamic** | Holograms, active glyphs, cursors | 100-1K instances | Every frame |
| **Streaming** | Knowledgeverse nodes in view | Variable | LOD-driven |

```cpp
// Lines 112-156: Single draw call per primitive type, all tiers merged
void BatchRenderer::SubmitFrame() {
    for (auto type : {LINE, CIRCLE, RECT}) {
        // Merge static + semi-static + dynamic + streaming into one buffer
        auto merged = MergeTiers(type);  // GPU compute shader, O(log n)
        cmd.DrawInstanced(type, merged.instanceCount, merged.bufferOffset);
    }
}
```

---

### 2. LOD/FOV Application to Knowledge Structures

**Critical insight:** Same spatial indexing serves BOTH Knowledgeverse (abstract) AND House (physical).

#### 2.1 Unified Spatial Index (`/src/layer0/spatial_index.h`)

```cpp
// Lines 23-67
struct SpatialKey {
    uint64_t universeID : 8;   // 0 = House physical, 1+ = Galaxy/Knowledgeverse
    uint64_t cellX : 20;       // Morton-coded spatial hash
    uint64_t cellY : 20;
    uint64_t cellZ : 16;       // Sufficient for galaxy-scale (2^16 ≈ 65k ly)
};

// Same BVH for all content — physics, rendering, knowledge proximity
class UnifiedBVH {
    // Lines 89-134
    void Insert(SpatialKey key, InstanceHandle handle, Bounds bounds);
    
    // LOD selection based on screen-space metric, not semantic type
    LODLevel SelectLOD(vec3 worldPos, float screenSizePixels) {
        if (screenSizePixels < 4.0f)  return LOD_INVISIBLE;
        if (screenSizePixels < 16.0f) return LOD_ICON;      // Single RECT
        if (screenSizePixels < 64.0f) return LOD_SUMMARY;  // 10-100 primitives
        return LOD_FULL;                                     // Complete geometry
    }
};
```

#### 2.2 FOV-Aware Knowledge Rendering (`/src/layer0/knowledge_renderer.cpp`)

```cpp
// Lines 201-256: Knowledgeverse node rendering
void KnowledgeRenderer::RenderNode(GalaxyNode* node, Camera& cam) {
    float screenSize = ProjectToScreen(node->bounds, cam);
    auto lod = SelectLOD(node->position, screenSize);
    
    switch (lod) {
        case LOD_ICON:
            // Single CIRCLE with color = node->semanticHash
            SubmitPrimitive(CIRCLE, node->position, 
                          vec4(HashToColor(node->hash), 1.0),
                          /*radius=*/4.0f, /*flags=*/ICON_FLAG);
            break;
            
        case LOD_SUMMARY:
            // Star glyph: center CIRCLE + radial LINEs for connections
            RenderStarGlyph(node, /*detail=*/5);
            break;
            
        case LOD_FULL:
            // Expand to RPN expression visualization
            // Each operator = RECT, operands = connected nodes
            RenderRPNExpression(node->program, cam);
            break;
    }
}
```

**House equivalent** (`/src/layer0/house_renderer.cpp:78-112`): Same LOD logic, but `universeID=0` triggers physical material lookup (wood grain via procedural noise on RECT instances).

---

### 3. 3D Book Object System with Galaxy Symlinks

#### 3.1 Book as Container (`/src/house/book.h`)

```cpp
// Lines 12-58
struct BookObject {
    // Physical representation — always rendered via primitive instances
    Transform spineTransform;           // Position on shelf
    float thickness;                  // Derived from galaxy.sizeBytes
    
    // Content reference — symlink to Galaxy
    GalaxyID content;                 // 0 = empty (placeholder book)
    
    // State
    enum State { CLOSED, OPENING, OPEN, CLOSING };
    State state;
    
    // Animation: opening transforms 2D cover planes to 3D readable spread
    float openAmount;  // 0.0 = closed, 1.0 = fully open
};

// Physical geometry — instanced like all House objects
void BookObject::SubmitInstances(BatchRenderer& r) {
    // Spine: RECT instance, wood material
    r.Submit(RECT, spineTransform, woodMaterial, extent={thickness, height});
    
    // Covers: Two RECTs, rotate around spine edge based on openAmount
    if (state != CLOSED) {
        float angle = openAmount * PI;  // 180 degrees when fully open
        r.Submit(RECT, leftCoverXform(spineTransform, angle), 
                paperMaterial, coverExtent);
        r.Submit(RECT, rightCoverXform(spineTransform, angle),
                paperMaterial, coverExtent);
    }
}
```

#### 3.2 Galaxy Loading via Symlinks (`/src/galaxy/symlink_loader.cpp`)

```cpp
// Lines 45-89: Zero-copy content loading
Galaxy* LoadBookContent(BookObject& book) {
    if (book.content == 0) return nullptr;
    
    // Symlink: Book.content points to Galaxy file on disk
    // Galaxy is memory-mapped, not loaded — pages faulted on demand
    auto galaxy = Galaxy::OpenSymlink(book.content);
    
    // Register with Knowledgeverse spatial index
    // Initial position: "floating" above open book
    auto kvNode = Knowledgeverse::CreateView(galaxy);
    kvNode->anchor = &book;  // Follows book if moved
    
    return galaxy;
}

// Lines 112-134: Unloading is reference-counted
void CloseBook(BookObject& book) {
    if (book.openAmount > 0.9f) {  // Was fully open
        // Snapshot view position/orientation for reopening
        book.lastView = Knowledgeverse::GetViewState(book.content);
    }
    Knowledgeverse::ReleaseView(book.content);
    book.state = CLOSING;
}
```

#### 3.3 Visual Continuity: Book → Knowledgeverse Transition

```cpp
// /src/layer0/transition_renderer.cpp:67-123
void RenderBookOpening(BookObject& book, Camera& houseCam, Camera& kvCam) {
    float t = book.openAmount;
    
    if (t < 0.5f) {
        // Phase 1: Book geometry animates open (physical simulation)
        book.SubmitInstances(renderer);
    } else {
        // Phase 2: Camera morphs from House to Knowledgeverse
        // Interpolate projection matrices, not just position
        Camera morphCam = Lerp(houseCam, kvCam, EaseInOutCubic((t-0.5f)*2));
        
        // Render hybrid: Book covers as "portal frame", galaxy visible through
        SubmitPortalFrame(book, morphCam);
        
        // Galaxy content rendered with LOD based on morphCam FOV
        Knowledgeverse::Render(morphCam, /*clipToBookCovers=*/true);
    }
}
```

---

### 4. Implementation Priorities

| Priority | File | Lines | Deliverable |
|---------|------|-------|-------------|
| P0 | `/src/layer0/primitives.h` | 1-80 | Unified instance struct, enum definitions |
| P0 | `/src/layer0/batch_renderer.cpp` | 1-200 | Tier merging, single-draw-call submission |
| P1 | `/src/layer0/spatial_index.h` | 1-150 | Unified BVH with `universeID` discrimination |
| P1 | `/src/layer0/knowledge_renderer.cpp` | 1-300 | LOD_ICON/SUMMARY/FULL for nodes |
| P2 | `/src/house/book.h` | 1-100 | BookObject with animation state |
| P2 | `/src/galaxy/symlink_loader.cpp` | 1-150 | Memory-mapped Galaxy loading |
| P3 | `/src/layer0/transition_renderer.cpp` | 1-200 | Portal morphing effect |

---

### 5. Key Architectural Constraints

1. **No shader permutations:** Single vertex/fragment shader for all primitives; variation via `flags` field and uniform buffers.

2. **Embedding-free rendering:** Knowledgeverse nodes have NO precomputed embeddings for display—visuals derived from RPN structure (operator arity → glyph count, nesting depth → color saturation).

3. **Dual-client verification:** All RPN programs rendered must be parseable by both human visual inspection AND AI symbolic evaluation—no "pretty but meaningless" decorations.

4. **Symlink integrity:** Book.content is validated at `OpenSymlink()` against Galaxy schema version; mismatch triggers graceful degradation to `LOD_ICON` with warning glyph.

────────────────────────────────────────────────────────────────

## Synthesis

I'll now synthesize these two designs into a unified K3D architecture specification, resolving conflicts and adding critical integration insights.

---

# K3D UNIFIED ARCHITECTURE SPECIFICATION v1.0
## LANGUAGE-AS-RPN-MATH + LAYER-0 INSTANCING SYSTEM

---

## 1. CORE ARCHITECTURAL PRINCIPLE: **SINGLE UNIFIED MACHINE**

Both subsystems—language composition and visual rendering—execute on **the same RPN virtual machine** with **the same primitive operations**. There is no "language stack" separate from "graphics stack." There is only **THE STACK**.

```
┌─────────────────────────────────────────────────────────────────┐
│                    K3D RPN UNIVERSAL VM                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  MATH UNIT  │  │ SPATIAL UNIT│  │   LINGUISTIC UNIT       │  │
│  │  (0x0xxx)   │  │  (0x1xxx)   │  │   (0x2xxx-0x3xxx)       │  │
│  │  ADD MUL    │  │  LINE CIR   │  │   MORPHEME SYNTAX       │  │
│  │  SIN VEC3   │  │  RECT XFORM │  │   RENDER AGREEMENT      │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
│         └─────────────────┴─────────────────────┘               │
│                         │                                       │
│                    ┌────┴────┐                                  │
│                    │ UNIFIED │  ← Single operand stack,           │
│                    │  STACK  │    typed cells (float/vec/ref)    │
│                    │ (256KB) │                                  │
│                    └────┬────┘                                  │
│                         │                                       │
│              ┌──────────┼──────────┐                           │
│              ▼          ▼          ▼                           │
│         ┌────────┐  ┌────────┐  ┌────────┐                      │
│         │ GPU    │  │ Galaxy │  │  TRM   │                      │
│         │Command │  │ Store  │  │ Policy │                      │
│         │ Buffer │  │ (mmap) │  │ Engine │                      │
│         └────────┘  └────────┘  └────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. RESOLVED CONFLICTS & INTEGRATION DECISIONS

### 2.1 Conflict: Language Opcode Namespace vs. Spatial Opcodes

| Sub-Agent A | Sub-Agent B | Resolution |
|-------------|-------------|------------|
| Language opcodes: `0xM4xx`, `0xS5xx`, `0xR6xx` | Spatial opcodes: `0x1xxx` for LINE/CIRCLE/RECT | **Unified registry**: Language is SPATIAL. See below. |

**Resolution:** Linguistic operations ARE spatial operations. A morpheme is a **glyph cluster**; syntax is **layout constraints**. The same `RECT` primitive that draws a wall draws a character bounding box.

**Unified Opcode Map:**

```
0x0000-0x0FFF:  MATH DOMAIN (from RPN_DOMAIN_OPCODE_REGISTRY)
0x1000-0x1FFF:  SPATIAL/LINGUISTIC PRIMITIVES (unified)
  0x1000: LINE
  0x1001: CIRCLE  
  0x1002: RECT
  0x1003: GLYPH_EMIT      ← NEW: renders Unicode via SDF glyph atlas
  0x1004: MORPHEME_BOX    ← NEW: semantic bounding box for syntax
  0x1005-0x10FF: reserved for geometric variants

0x2000-0x2FFF:  COMPOSITION & GRAMMAR
  0x2000: MEANING_LOAD
  0x2001: GRAMMAR_FETCH
  0x2002: WORD_RESOLVE
  0x2003: MORPHEME_PUSH
  0x2004: MORPHEME_POP
  0x2005: MORPHEME_SWAP
  0x2006: SYNTAX_LAYOUT   ← NEW: converts syntax tree to spatial boxes
  0x2007: GLYPH_RENDER     ← NEW: emits GLYPH_EMIT primitives

0x3000-0x3FFF:  SURFACE & OUTPUT
  0x3000: SURFACE_RENDER
  0x3001: UTF8_EMIT       ← byte stream to buffer
  0x3002: PRIMITIVE_BATCH  ← submit to GPU
```

### 2.2 Conflict: Where Does "Text" Live?

| Sub-Agent A | Sub-Agent B | Resolution |
|-------------|-------------|------------|
| `SURFACE_RENDER` → UTF-8 byte stream | `GLYPH_EMIT` → GPU primitive | **Both, unified via PRIMITIVE_BATCH** |

**Resolution:** Language synthesis produces **intermediate spatial form**: an array of `MorphemeBox` structs (position, size, glyph index, semantic tag). This array can be:
- Serialized to UTF-8 + markup (for external systems)
- Batched as `GLYPH_EMIT` primitives (for display)
- Stored as RPN for re-execution (for TRM reasoning)

```cpp
// Unified intermediate representation
struct MorphemeBox {
    vec2 position;        // Layout position (em units)
    vec2 extent;          // Bounding box
    uint32_t glyphIndex;  // 0 = composite (recursive RPN reference)
    uint32_t meaningRef;  // Link to source meaning star
    uint16_t languageTag; // For font selection
    uint8_t  depth;       // Nesting level for color/weight
};
```

### 2.3 Conflict: LOD for Knowledge vs. LOD for Language

**Resolution:** Same system. A "sentence" in Knowledgeverse is a **constellation of meaning stars**; its visual representation uses identical LOD rules as any galaxy node.

| Distance/Scale | Knowledge Node | Language Element |
|---------------|----------------|----------------|
| `LOD_INVISIBLE` | Not in frustum | Morpheme omitted (ellipsis implied) |
| `LOD_ICON` | Single colored circle | Single glyph (logogram or abbreviation) |
| `LOD_SUMMARY` | Star glyph with connections | Key morphemes only, function words elided |
| `LOD_FULL` | Complete RPN expression | Full sentence with all morphemes and gloss |

---

## 3. INTEGRATED SYSTEM: SENTENCE SYNTHESIS → SPATIAL RENDERING

### 3.1 Complete Pipeline: Meaning → Screen Pixels

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  INPUT:         │     │  RPN EXECUTION  │     │  OUTPUT:        │
│  meaning_id     │────→│  (Universal VM) │────→│  GPU primitives │
│  lang_tag       │     │                 │     │  + UTF-8 stream │
│  target_format  │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                        │
         ▼                       ▼                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ MEANING_LOAD    │     │ SYNTAX_LAYOUT   │     │ PRIMITIVE_BATCH │
│ GRAMMAR_FETCH   │ →   │ GLYPH_RENDER    │ →   │ UTF8_EMIT       │
│ WORD_RESOLVE    │     │ (recurse if     │     │ (dual output)   │
│ MORPHEME_* ops  │     │  composite)     │     │                 │
│                 │     │                 │     │                 │
│ Stack: meaning* │     │ Stack: box[]    │     │ Stack: —        │
│ → morpheme*     │     │ → primitive[]   │     │ (flushed)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 3.2 Critical Integration: `SYNTAX_LAYOUT` Opcode

This bridges linguistic structure to spatial rendering—**no Python, no string formatting, pure RPN stack manipulation**:

```
SYNTAX_LAYOUT implementation (RPN subroutine):

Input stack: [morpheme*[n], n, direction_scalar, spacing_scalar]
  where direction_scalar: -1.0 = RTL, 1.0 = LTR, 0.0 = TTB/BTT
  where spacing_scalar: 0.0 = packed, 1.0 = normal, 2.0 = expanded

Operation:
1. Pop n, direction, spacing
2. Allocate MorphemeBox[n] on computation heap (not stack—too large)
3. For i in 0..n-1:
   - Pop morpheme*
   - Lookup glyph metrics in font atlas (hash on languageTag + surface)
   - Calculate position: cumulative advance * direction * spacing
   - Write MorphemeBox to heap array
4. Push heap array reference to stack

Output: [box_array_ref*]

Next: GLYPH_RENDER consumes box_array_ref, emits GLYPH_EMIT primitives
```

### 3.3 Font Atlas as Galaxy

**Key insight from synthesis:** Font glyphs are **meaning stars** in a special `galaxy://typography` namespace.

```
galaxy://typography/
├── latin/
│   ├── U+0061/          ← meaning star for glyph 'a'
│   │   ├── outline_rpn: [RECT, CIRCLE, LINE...]  ← Layer 0 primitives!
│   │   ├── metrics: {advance: 0.6, lsb: 0.1, ...}
│   │   └── variations: [regular, bold, italic...]
│   └── ...
├── cjk/
├── arabic/
│   └── U+0644/          ← 'ل' with contextual forms
│       ├── isolated, initial, medial, final variants
│       └── shaping_rpn: [CONTEXTUAL_SELECT, ...]  ← grammar-like!
└── emoji/
    └── U+1F600/         ← "grinning face"
        ├── color_bitmap: reference to texture galaxy
        └── semantic_links: [emotion/joy, face/human...]
```

**This satisfies the dual-client contract:** A human designer edits glyph outlines as RPN primitive sequences; TRM executes identical programs for rendering.

---

## 4. THE BOOK SYSTEM: FULL INTEGRATION

### 4.1 Book State Machine with RPN Control

```
┌─────────┐    open_rpn     ┌─────────┐    load_rpn    ┌─────────┐
│ CLOSED  │ ──────────────→ │ OPENING │ ─────────────→ │ ACTIVE  │
│ (mesh)  │                 │ (anim)  │                │ (galaxy)│
└─────────┘                 └─────────┘                └────┬────┘
     ↑                                                    │
     └────────────────────────────────────────────────────┘
                    close_rpn (or unload_rpn for memory pressure)
```

### 4.2 RPN Programs for Book Transitions

**`open_rpn`** (stored in BookObject, editable by users):

```json
{
  "program": [
    "0x1002",           // RECT - cover geometry
    "0x2003",           // MORPHEME_PUSH - "opening" semantic marker
    "0x1001",           // CIRCLE - portal effect center
    "0x2006",           // SYNTAX_LAYOUT - arrange visual elements
    "0x3002"            // PRIMITIVE_BATCH - submit to GPU
  ],
  "duration_ms": 500,
  "easing": "EASE_IN_OUT_CUBIC"
}
```

**`load_rpn`** (system-provided, loads galaxy via symlink):

```json
{
  "program": [
    "0x2000",           // MEANING_LOAD - book.content galaxy_id
    "0x2001",           // GRAMMAR_FETCH - universal galaxy loader
    "0x2002",           // WORD_RESOLVE - verify schema version
    "0x2006",           // SYNTAX_LAYOUT - initial spatial arrangement
    "0x1003",           // GLYPH_EMIT - render title/cover info
    "0x3002"            // PRIMITIVE_BATCH
  ],
  "symlink_target": "galaxy://{content}/",
  "memory_map": true
}
```

### 4.3 Visual Continuity: Same Primitives, Different Scales

| View | What Renders | Primitive Count | Key Opcodes |
|------|-------------|-----------------|-------------|
| **Closed book on shelf** | Spine RECT, cover RECTs | 3 | Static instancing |
| **Opening animation** | Above + morphing portal CIRCLEs, particle LINEs | 50-100 | Dynamic update |
| **Fully open, transitioning** | Portal frame RECTs + distant galaxy nodes as ICONs | 1000 | LOD_ICON for knowledge |
| **Immersed in Knowledgeverse** | Full RPN expression trees, star glyphs, connection LINEs | 10K-100K | LOD_FULL, recursive rendering |
| **Zoom to single star** | Meaning star's `meaning_rpn` rendered as spatial diagram | 100-1000 | Execute star's own RPN |

**Critical:** The "page" you read in an open book is **the same primitive batch** as a hologram in Knowledgeverse. The "text" is **live RPN**, not baked pixels.

---

## 5. THE SAUDADES CASE: FULL INTEGRATION

### 5.1 Complete Star Definition

```json
{
  "star_id": "0xM7A3F2",
  "embedding": "[cosine navigation vector]",
  
  "meaning_rpn": {
    "program": [
      "0x2000", "0x7A3F2",     // MEANING_LOAD self
      "0x2001", "0xA0B1C2", "LANG_INPUT",  // GRAMMAR_FETCH with runtime lang
      "0x2002",                // WORD_RESOLVE
      // If WORD_RESOLVE returns SELF_REF, fall through to canonical
      "0x2003",                // MORPHEME_PUSH (result or canonical)
      "0x2006",                // SYNTAX_LAYOUT
      "0x1003",                // GLYPH_EMIT
      "0x3000"                 // SURFACE_RENDER
    ],
    "canonical_surface": {
      "language": "pt-BR",
      "form": "saudades",
      "pronunciation_rpn": "[phoneme sequence as RPN]"
    },
    "untranslatable_handler": "CANONICAL_WITH_PROVENANCE"
  },
  
  "word_galaxy_links": {
    "pt-BR": "galaxy://word/pt/saudades",
    "en-US": "SELF_REF",           // ← triggers canonical fallback
    "ja-JP": "SELF_REF",
    "es-ES": "galaxy://word/es/anhelo"  // approximate, different meaning
  },
  
  "visual_representation": {
    "LOD_ICON": {
      "primitive": "CIRCLE",
      "color": "#8B4513",        // brown = emotion/loss
      "glyph": "⚘"              // wilting flower Unicode
    },
    "LOD_FULL": {
      // RPN program generating: [heart]→[distance]→[time] visual metaphor
      "program": [
        "0x1001", "0.5", "0.5", "0.2",  // CIRCLE - heart
        "0x1000", "0.5", "0.3", "0.5", "0.7",  // LINE - stretching away
        "0x1001", "0.5", "0.8", "0.1",  // CIRCLE - distant point
        "0x2006", "0x3002"
      ]
    }
  }
}
```

### 5.2 Rendering "Saudades" in Different Contexts

| Context | Language | Output | Visual |
|---------|----------|--------|--------|
| Portuguese speaker, any LOD | pt-BR | "saudades" | LOD-appropriate glyph/visual |
| English speaker, `LOD_FULL` | en-US | "saudades [Portuguese: deep longing for lost things]" | Periphrastic expansion in `SYNTAX_LAYOUT` |
| English speaker, `LOD_ICON` | en-US | "⚘" or "saud." | Abbreviation with provenance hint |
| Cross-lingual TRM reasoning | — | RPN: `[EMOTION/intense, TEMPORAL/past, CERTAINTY/irrecoverable]` | Star glyph with those semantic connections |

---

## 6. IMPLEMENTATION SPECIFICATION

### 6.1 File Organization

```
/k3d/
├── layer0/
│   ├── rpn_vm.h/cpp           # Universal VM, single implementation
│   ├── primitives.h           # LINE, CIRCLE, RECT, GLYPH_EMIT
│   ├── batch_renderer.cpp     # GPU submission, all tiers
│   └── spatial_index.h        # Unified BVH with universeID
├── language/                  # ← NOT separate from layer0
│   ├── grammar_galaxy.cpp     # RPN subroutine loading
│   ├── syntax_layout.cpp      # SYNTAX_LAYOUT opcode implementation
│   └── font_atlas_galaxy.cpp  # Typography as meaning stars
├── house/
│   ├── book.h/cpp             # BookObject with RPN state machine
│   └── transition_renderer.cpp # Portal effects
├── knowledgeverse/
│   ├── galaxy_loader.cpp      # Symlink, memory-map
│   ├── node_renderer.cpp      # LOD selection, recursive RPN
│   └── view_manager.cpp       # Camera, anchor tracking
└── trm/
    └── policy_engine.cpp      # Selects RPN programs, not weights
```

### 6.2 Critical Invariants

| # | Invariant | Enforcement |
|---|-----------|-------------|
| 1 | **No string formatting in synthesis** | RPN VM has no `sprintf` opcode; only stack→buffer moves |
| 2 | **Single draw call per primitive type** | BatchRenderer merges all tiers/universes before GPU submission |
| 3 | **Embedding never used for display** | Knowledgeverse visuals derived from RPN structure, not embedding lookup |
| 4 | **Galaxy loading is zero-copy** | `mmap` with `MAP_SHARED`; page fault on first access |
| 5 | **Dual-client verification** | Every `meaning_rpn` has `human_comment` and `machine_program`; hash verifies match |
| 6 | **Language is surface
