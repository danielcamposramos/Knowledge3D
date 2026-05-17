# PM-KR RPN-to-X3D Scene Graph Emission

**Version**: 0.1 (Initial Draft)
**Status**: PM-KR Community Group Working Draft
**Date**: March 26, 2026
**Authors**: PM-KR Community Group (Daniel Campos Ramos, Chair; Milton Ponson, Co-Chair)
**Liaison**: Web3D Consortium (Don Brutzman, Advisory Committee Representative)
**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Reference Implementation)

**Normative References**:
- ISO/IEC 19775-1:2023 (X3D Architecture and Base Components, Version 4.0)
- ISO/IEC 19775-2:2023 (X3D Scene Access Interface, SAI)
- ISO/IEC 19776-1 (X3D XML Encoding)
- ISO/IEC 19776-2 (X3D ClassicVRML Encoding)
- ISO/IEC 19776-3 (X3D Compressed Binary Encoding)
- PM-KR X3D Procedural Memory Component v0.1 (docs/w3c/x3d/PM_KR_X3D_PROCEDURAL_MEMORY_COMPONENT.md)
- RPN Domain Opcode Registry v0.1 (docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md)
- Foundational Knowledge Specification v1.0 (docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md)
- Dual-Client Contract Specification v1.1 (docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md)
- Knowledgeverse Specification v5.1 (docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Scope and Architecture](#2-scope-and-architecture)
3. [Emission Model: From Galaxy to Scene Graph](#3-emission-model-from-galaxy-to-scene-graph)
4. [Drawing Opcode Emission](#4-drawing-opcode-emission)
5. [Mesh Opcode Emission](#5-mesh-opcode-emission)
6. [Transform Emission](#6-transform-emission)
7. [Material and Appearance Emission](#7-material-and-appearance-emission)
8. [House Artifact Trigger Pipeline](#8-house-artifact-trigger-pipeline)
9. [Galaxy-to-Scene Composition](#9-galaxy-to-scene-composition)
10. [X3D Serialization Targets](#10-x3d-serialization-targets)
11. [Scene Access Interface (SAI) Integration](#11-scene-access-interface-sai-integration)
12. [Dual-Client Emission](#12-dual-client-emission)
13. [Cross-Layer Reference Emission](#13-cross-layer-reference-emission)
14. [glTF Round-Trip Preservation](#14-gltf-round-trip-preservation)
15. [Conformance](#15-conformance)
16. [Examples](#16-examples)

---

## 1. Introduction

### 1.1 Purpose

This document specifies how PM-KR's RPN programs --- stored as Galaxy knowledge entries and originating from House artifacts --- **emit X3D-compliant scene graphs**. It defines the mapping from every RPN opcode tier to concrete X3D node types, the emission pipeline from House artifact activation through Galaxy resolution to serialized X3D output, and the conformance criteria for emitters.

The PM-KR X3D Procedural Memory Component (companion document) defines how PM-KR knowledge nodes **exist within** X3D scene graphs. This document defines how PM-KR knowledge nodes **produce** X3D scene graphs.

The distinction is fundamental: PM-KR knowledge is programs. Programs execute. Execution produces output. That output is an X3D scene graph.

### 1.2 Motivation

X3D scene graphs are traditionally authored declaratively: a human or tool creates a static XML/JSON/ClassicVRML file describing geometry, appearance, and behavior. Script nodes and PROTOs add procedural capability, but the dominant paradigm remains declarative authoring.

PM-KR inverts this: knowledge IS executable programs. A book in the House is a 3D artifact; when loaded, its contents become a Galaxy. Each Galaxy entry carries `formProgram` and `meaningProgram` fields that are RPN instruction sequences. When the system needs to present, explain, or reason about that knowledge, the RPN programs execute and emit scene graph fragments.

This means:
- A math symbol emits its own visual form as X3D geometry (IndexedLineSet from drawing opcodes).
- A 3D object emits its mesh as X3D geometry (IndexedFaceSet/IndexedTriangleSet from mesh opcodes).
- A grammar rule emits its transformation as an X3D scene showing input → output.
- A House room emits its spatial layout as a complete X3D sub-scene.
- An agent's perception emits as an X3D frustum visualization.

The RPN engine is the scene graph factory. X3D is the interchange format.

### 1.3 Terminology

| Term | Definition | Source |
|------|-----------|--------|
| **Emission** | The process of executing an RPN program and producing X3D scene graph nodes from the execution output | This document |
| **Emitter** | A conformant implementation that executes RPN programs and produces X3D output | This document |
| **Source Galaxy** | The Galaxy containing the PM-KR entry whose RPN program is being executed | Knowledgeverse Spec §2.1 |
| **Trigger Artifact** | A House object (typically glTF/GLB with `extras.k3d`) that activates RPN emission when loaded or interacted with | This document |
| **Intermediate Representation (IR)** | The in-memory data structures produced by RPN execution before X3D serialization: `MeshBuffer`, `Path2D`, drawing command lists | This document |
| **Scene Fragment** | A self-contained X3D node tree (rooted at a Transform or Group) emitted by a single RPN program execution | This document |

---

## 2. Scope and Architecture

### 2.1 What This Document Defines

1. **Opcode-to-X3D mapping**: Every K3D RPN drawing and mesh opcode maps to specific X3D geometry, appearance, and grouping nodes.
2. **Emission pipeline**: The five-stage pipeline from House artifact trigger through Galaxy resolution, RPN execution, IR construction, to X3D serialization.
3. **Composition rules**: How scene fragments from multiple Galaxy entries compose into complete X3D scenes via canonical references.
4. **Serialization targets**: X3D XML (.x3d), ClassicVRML (.x3dv), JSON (.x3dj), and compressed binary (.x3db) output.
5. **SAI integration**: How emitted scenes connect to the X3D Scene Access Interface for live scene manipulation.
6. **Round-trip preservation**: How PM-KR metadata survives X3D → glTF → X3D conversions.

### 2.2 What This Document Does Not Define

- GPU kernel implementation of RPN execution (that is K3D's sovereign runtime concern).
- Specific compression ratios or performance targets (those are implementation-dependent).
- Network transport protocols for scene distribution (X3D Networking component handles that).
- Agent behavior during emission (the agent decides WHEN to emit; this document defines HOW).

### 2.3 Architectural Position

```
┌─────────────────────────────────────────────────────────────────┐
│                        House (SSD/Disk)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Book.glb │  │ Shelf.glb│  │ Tablet   │  │ Room.glb │       │
│  │ extras:  │  │ extras:  │  │ extras:  │  │ extras:  │       │
│  │  k3d:    │  │  k3d:    │  │  k3d:    │  │  k3d:    │       │
│  │  galaxy: │  │  galaxy: │  │  galaxy: │  │  galaxy: │       │
│  │  "Math"  │  │  "Tool"  │  │  "Meta"  │  │  "House" │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │              │              │              │             │
└───────┼──────────────┼──────────────┼──────────────┼─────────────┘
        │              │              │              │
        v              v              v              v
┌─────────────────────────────────────────────────────────────────┐
│                    Galaxy Universe (VRAM)                        │
│                                                                 │
│  Galaxy entry:                                                  │
│    canonicalId: "char:U+2211"                                   │
│    layer: "form"                                                │
│    formProgram: "32 8 MOVE 8 32 LINE 32 56 LINE STROKE"        │
│    embedding16: [0.82, -0.15, ...]                              │
│                                                                 │
│         │ RPN Execution                                         │
│         v                                                       │
│  ┌──────────────┐                                               │
│  │ RPN Engine   │  Stack machine executes formProgram           │
│  │ (PTX / Host) │  Produces: Path2D / MeshBuffer / DrawCmds    │
│  └──────┬───────┘                                               │
│         │ Intermediate Representation                           │
│         v                                                       │
│  ┌──────────────┐                                               │
│  │ X3D Emitter  │  IR → X3D node tree                          │
│  │              │  Drawing → IndexedLineSet                     │
│  │              │  Mesh → IndexedFaceSet                        │
│  │              │  Transform → Transform                        │
│  │              │  Material → Appearance + Material             │
│  └──────┬───────┘                                               │
│         │                                                       │
└─────────┼───────────────────────────────────────────────────────┘
          │
          v
┌─────────────────────────────────────────────────────────────────┐
│                    X3D Scene Graph Output                        │
│                                                                 │
│  Serialization targets:                                         │
│    .x3d   (XML)                                                 │
│    .x3dv  (ClassicVRML)                                         │
│    .x3dj  (JSON)                                                │
│    .x3db  (Compressed Binary)                                   │
│    .glb   (glTF with extras.k3d — round-trip)                   │
│                                                                 │
│  Live targets:                                                  │
│    SAI (Scene Access Interface) — runtime scene manipulation    │
│    DOM (browser projection via domOps)                          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.4 Five-Stage Emission Pipeline

Every RPN-to-X3D emission follows five stages:

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| **1. Trigger** | House artifact interaction or query | Identify Galaxy entries to emit | List of canonical IDs + RPN programs |
| **2. Resolve** | Canonical IDs | Resolve canonical references, load symlinked entries | Complete dependency graph of PM-KR entries |
| **3. Execute** | RPN programs + resolved references | Stack machine execution | Intermediate Representation (IR) |
| **4. Construct** | IR (MeshBuffer, Path2D, draw commands) | Map IR to X3D node types | In-memory X3D node tree |
| **5. Serialize** | X3D node tree | Encode to target format | .x3d / .x3dv / .x3dj / .x3db / .glb |

Stages 1-3 are K3D runtime concerns (sovereignty domain). Stages 4-5 are interchange concerns (this document's primary focus). Stage 3 produces a clean boundary: the IR is the contract between sovereign execution and X3D emission.

---

## 3. Emission Model: From Galaxy to Scene Graph

### 3.1 Galaxy Entry as Scene Fragment Source

Each Galaxy entry with a non-NULL `formProgram` or `meaningProgram` is a potential scene fragment source. When emitted, a single Galaxy entry produces one X3D scene fragment:

```
Galaxy Entry                    X3D Scene Fragment
─────────────                   ──────────────────
canonicalId ──────────────────→ Transform DEF="{canonicalId}"
layer ────────────────────────→ MetadataString name="pmkr:layer"
galaxy ───────────────────────→ parent GalaxyGroup galaxyName="{galaxy}"
formProgram ──────────────────→ child geometry nodes (execution output)
meaningProgram ───────────────→ MetadataString name="pmkr:meaningOpcodes"
embedding ────────────────────→ MetadataFloat name="pmkr:embedding16"
canonicalRefs ────────────────→ child CanonicalReference nodes
position (from Morton code) ──→ Transform translation="{x} {y} {z}"
```

### 3.2 Emission Contexts

RPN programs emit X3D scene fragments in four distinct contexts:

**Context A: Visual Form Emission**
- Source: `formProgram` field of ProceduralFormNode entries
- Purpose: Produce the visual/auditory form of a knowledge node
- Output: Geometry nodes (IndexedLineSet, IndexedFaceSet, Text) with Appearance
- Example: Character glyph "∑" emits its Bézier stroke as IndexedLineSet

**Context B: Semantic Structure Emission**
- Source: `meaningProgram` field of ProceduralMeaningNode entries
- Purpose: Produce a scene representing semantic relationships
- Output: Grouping nodes containing positioned references to form nodes
- Example: Word "derivative" emits positioned character references + relationship visualization

**Context C: Rule Visualization Emission**
- Source: `meaningProgram` field of ProceduralRulesNode entries
- Purpose: Produce a scene showing input → transformation → output
- Output: Scene with "before" and "after" sub-graphs connected by visual transformation indicators
- Example: Grammar rule "plural_regular_s" emits: `[dog]` → `[dog][s]`

**Context D: Spatial Construction Emission**
- Source: `formProgram` field of 3DObjects Galaxy or House construction entries
- Purpose: Produce 3D mesh geometry for House rooms, furniture, objects
- Output: Complete mesh sub-scenes (IndexedFaceSet, IndexedTriangleSet) with materials
- Example: Bookshelf object emits its mesh geometry with wood material

### 3.3 The IR Contract

The Intermediate Representation is the boundary between RPN execution (sovereign) and X3D emission (interchange). The IR consists of three data types:

**Path2D**: Ordered sequence of 2D path commands produced by drawing opcodes.
```
Commands: MoveTo(x,y) | LineTo(x,y) | QuadTo(cx,cy,x,y) | CubicTo(c1x,c1y,c2x,c2y,x,y)
          | ArcTo(cx,cy,r,startAngle,endAngle) | Close
Properties: strokeColor, fillColor, lineWidth
```

**MeshBuffer**: Indexed triangle mesh produced by mesh opcodes.
```
Arrays: vertices[(x,y,z)...], normals[(nx,ny,nz)...], uvs[(u,v)...], triangles[(i0,i1,i2)...]
Properties: transform (4x4 matrix)
```

**DrawCommandList**: Sequence of high-level drawing commands produced by drawing opcodes when a 2D→3D extrusion is not requested.
```
Commands: SetStrokeColor(r,g,b,a) | SetFillColor(r,g,b,a) | SetLineWidth(w)
          | StrokePath(Path2D) | FillPath(Path2D)
```

Each RPN program execution produces one or more of these IR types. The X3D emitter consumes these IR types and produces the corresponding X3D nodes.

---

## 4. Drawing Opcode Emission

### 4.1 Drawing Opcodes to X3D Mapping

The drawing opcodes (Tier 2, 0x64--0x77) are the primary form-emission opcodes. They produce 2D path geometry that maps to X3D line and face geometry.

**Table 4.1 --- Drawing opcode to X3D node mapping**

| RPN Opcode | Hex | IR Output | X3D Node(s) |
|------------|-----|-----------|-------------|
| `MOVE` | 0x64 | `Path2D.MoveTo(x,y)` | Start new polyline segment in Coordinate |
| `LINE` | 0x65 | `Path2D.LineTo(x,y)` | Extend polyline in Coordinate |
| `BEZIER` | 0x67 | `Path2D.CubicTo(...)` | Tessellated to line segments in Coordinate |
| `ARC` | 0x68 | `Path2D.ArcTo(...)` | Tessellated to line segments in Coordinate |
| `CLOSE` | 0x69 | `Path2D.Close()` | Close polyline (last point connects to first) |
| `STROKE` | 0x6A | `DrawCmd.StrokePath(path)` | **IndexedLineSet** with Coordinate, Color |
| `FILL` | 0x6B | `DrawCmd.FillPath(path)` | **IndexedFaceSet** with Coordinate, Color (triangulated) |
| `SET_STROKE_COLOR` | 0x75 | `strokeColor = (r,g,b,a)` | Color node on IndexedLineSet |
| `SET_FILL_COLOR` | 0x76 | `fillColor = (r,g,b,a)` | Material diffuseColor on IndexedFaceSet |
| `SET_LINE_WIDTH` | 0x77 | `lineWidth = w` | LineProperties linewidthScaleFactor |

### 4.2 STROKE Emission: IndexedLineSet

When a `STROKE` opcode fires, the accumulated path becomes an X3D IndexedLineSet. Each `MoveTo` starts a new polyline; each `LineTo` extends it.

**Emission rule:**

```
STROKE(Path2D) →
  <Shape>
    <Appearance>
      <Material emissiveColor="{strokeColor.rgb}"/>
      <LineProperties linewidthScaleFactor="{lineWidth}"/>
    </Appearance>
    <IndexedLineSet coordIndex="{path_to_coord_indices}">
      <Coordinate point="{all_path_points_as_3D}" />
      <Color color="{per_vertex_colors_if_gradient}" />
    </IndexedLineSet>
  </Shape>
```

**Path-to-coordinate conversion:**

1. Each `MoveTo(x,y)` → new Coordinate point at `(x, y, 0)`, starts new polyline segment.
2. Each `LineTo(x,y)` → new Coordinate point at `(x, y, 0)`, extends current polyline.
3. Each `CubicTo(c1,c2,end)` → tessellate to N line segments (adaptive: chord length < 0.5 units), append as LineTo points.
4. Each `ArcTo(center,r,a0,a1)` → tessellate to N line segments (adaptive: angular step < 10°), append as LineTo points.
5. `Close` → append coordIndex pointing back to first point of current polyline.
6. Polyline segments separated by `-1` in coordIndex.

**Z-coordinate**: Drawing opcodes produce 2D geometry. The emitter places all points at `z=0` in the local coordinate system. The parent Transform positions the fragment in 3D space.

**Example --- Summation symbol (∑) form program emission:**

RPN program: `32 8 MOVE 8 32 LINE 32 56 LINE 56 56 LINE 32 32 MOVE 8 32 LINE 32 8 LINE STROKE`

Execution trace:
1. `32 8 MOVE` → MoveTo(32, 8)
2. `8 32 LINE` → LineTo(8, 32)
3. `32 56 LINE` → LineTo(32, 56)
4. `56 56 LINE` → LineTo(56, 56) — polyline 0 complete
5. `32 32 MOVE` → MoveTo(32, 32)
6. `8 32 LINE` → LineTo(8, 32)
7. `32 8 LINE` → LineTo(32, 8) — polyline 1 complete
8. `STROKE` → emit IndexedLineSet

Emitted X3D:
```xml
<Shape>
  <Appearance>
    <Material emissiveColor="0 0 0"/>
    <LineProperties linewidthScaleFactor="1.0"/>
  </Appearance>
  <IndexedLineSet coordIndex="0 1 2 3 -1 4 5 6 -1">
    <Coordinate point="32 8 0, 8 32 0, 32 56 0, 56 56 0, 32 32 0, 8 32 0, 32 8 0"/>
  </IndexedLineSet>
</Shape>
```

### 4.3 FILL Emission: IndexedFaceSet

When a `FILL` opcode fires, the accumulated closed path becomes an X3D IndexedFaceSet. The path MUST be closed (explicitly via `CLOSE` or implicitly if the last point equals the first). Open paths with `FILL` MUST be auto-closed by the emitter.

**Emission rule:**

```
FILL(Path2D) →
  <Shape>
    <Appearance>
      <Material diffuseColor="{fillColor.rgb}" transparency="{1-fillColor.a}"/>
    </Appearance>
    <IndexedFaceSet coordIndex="{triangulated_indices}" solid="false" creaseAngle="3.14159">
      <Coordinate point="{all_path_points_as_3D}" />
      <TextureCoordinate point="{normalized_uv}" />
    </IndexedFaceSet>
  </Shape>
```

**Triangulation**: The emitter MUST triangulate the closed path polygon for IndexedFaceSet. Ear-clipping or constrained Delaunay triangulation is RECOMMENDED. The triangulation MUST handle:
- Convex polygons (trivial fan triangulation).
- Concave polygons (ear-clipping required).
- Self-intersecting paths (split into non-intersecting sub-polygons first).

**UV generation**: For filled forms, the emitter SHOULD generate UV coordinates by normalizing the path bounding box to [0,1]×[0,1]. This enables the Dual-Client texture contract: UV Map 0 maps to human-readable texture, UV Map 1 maps to machine-readable texture.

### 4.4 Combined STROKE + FILL

A single path MAY receive both `STROKE` and `FILL`, producing two X3D Shape nodes grouped under a common parent:

```xml
<Group>
  <!-- Fill -->
  <Shape>
    <Appearance><Material diffuseColor="0.9 0.9 0.9"/></Appearance>
    <IndexedFaceSet coordIndex="0 1 2 3 -1" solid="false">
      <Coordinate DEF="SHARED_COORDS" point="..."/>
    </IndexedFaceSet>
  </Shape>
  <!-- Stroke -->
  <Shape>
    <Appearance>
      <Material emissiveColor="0 0 0"/>
      <LineProperties linewidthScaleFactor="2.0"/>
    </Appearance>
    <IndexedLineSet coordIndex="0 1 2 3 0 -1">
      <Coordinate USE="SHARED_COORDS"/>
    </IndexedLineSet>
  </Shape>
</Group>
```

The `DEF/USE` pattern for shared Coordinate nodes aligns with X3D's node reuse mechanism and PM-KR's Save Information Principle.

### 4.5 Curve Tessellation Parameters

Bézier curves and arcs require tessellation into line segments for X3D IndexedLineSet. The emitter MUST support configurable tessellation quality:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `maxChordError` | 0.5 | (0, inf) | Maximum distance from chord to true curve (units) |
| `maxAngularStep` | 10.0 | (0, 180) | Maximum angular step for arc tessellation (degrees) |
| `minSegments` | 4 | [2, 256] | Minimum segments per curve |
| `maxSegments` | 64 | [4, 4096] | Maximum segments per curve |

Higher-quality tessellation produces more Coordinate points but smoother curves. The emitter SHOULD select quality based on the LOD level of the emitting knowledge node.

---

## 5. Mesh Opcode Emission

### 5.1 Mesh Opcodes to X3D Mapping

The mesh opcodes (defined in `mesh_opcodes.py`) produce 3D geometry directly. These are the primary opcodes for House construction and 3DObjects Galaxy entries.

**Table 5.1 --- Mesh opcode to X3D node mapping**

| RPN Opcode | IR Output | X3D Node |
|------------|-----------|----------|
| `MESH_BEGIN` | Start new MeshBuffer | (begin accumulating geometry) |
| `VERTEX3` | Append vertex (x,y,z) | Coordinate point entry |
| `NORMAL3` | Append normal (nx,ny,nz) | Normal vector entry |
| `UV2` | Append UV (u,v) | TextureCoordinate point entry |
| `TRI_FACE` | Append triangle (i0,i1,i2) | coordIndex triplet in IndexedTriangleSet |
| `QUAD_FACE` | Append quad (i0,i1,i2,i3) | coordIndex quad in IndexedFaceSet |
| `FACE_NORMAL` | Compute face normal | Normal vector entry (auto-computed) |
| `MESH_END` | Finalize MeshBuffer | Complete geometry node |
| `GEN_PLANE` | Generate plane mesh | IndexedFaceSet (subdivided quad) |
| `GEN_CUBE` | Generate cube mesh | **Box** geometry OR IndexedFaceSet |
| `GEN_UV_SPHERE` | Generate UV sphere | **Sphere** geometry OR IndexedFaceSet |
| `GEN_CYLINDER` | Generate cylinder | **Cylinder** geometry OR IndexedFaceSet |
| `GEN_CONE` | Generate cone | **Cone** geometry OR IndexedFaceSet |
| `GEN_TORUS` | Generate torus mesh | IndexedFaceSet (no X3D primitive) |
| `GEN_ICOSPHERE` | Generate icosphere mesh | IndexedFaceSet (no X3D primitive) |
| `CSG_UNION` | Boolean union of two meshes | IndexedFaceSet (computed result) |
| `CSG_SUBTRACT` | Boolean subtraction | IndexedFaceSet (computed result) |
| `CSG_INTERSECT` | Boolean intersection | IndexedFaceSet (computed result) |
| `EXTRUDE` | Extrude Path2D along Z | **Extrusion** OR IndexedFaceSet |
| `LATHE` | Revolve Path2D around Y | **Extrusion** (circular spine) OR IndexedFaceSet |

### 5.2 Explicit Mesh Emission: MESH_BEGIN/VERTEX3/TRI_FACE/MESH_END

The most direct emission path: RPN opcodes explicitly define vertex positions, normals, UVs, and face indices.

**Emission rule:**

```
MESH_BEGIN ... VERTEX3 ... TRI_FACE ... MESH_END →
  <Shape>
    <Appearance>
      <Material diffuseColor="{material_color}"/>
    </Appearance>
    <IndexedTriangleSet index="{all_tri_indices}" solid="true">
      <Coordinate point="{all_vertices}" />
      <Normal vector="{all_normals}" />
      <TextureCoordinate point="{all_uvs}" />
    </IndexedTriangleSet>
  </Shape>
```

When QUAD_FACE opcodes are present, the emitter MUST use IndexedFaceSet instead of IndexedTriangleSet, with `-1` face separators in coordIndex.

### 5.3 Primitive Generator Emission

Generator opcodes (`GEN_CUBE`, `GEN_UV_SPHERE`, etc.) produce complete meshes from parameters. The emitter has two strategies:

**Strategy A: X3D Primitive Nodes (Preferred when available)**

X3D defines built-in geometry primitives that match several generators:

| Generator | X3D Primitive | Conditions |
|-----------|--------------|------------|
| `GEN_CUBE(size)` | `<Box size="{s} {s} {s}"/>` | No post-transform CSG |
| `GEN_UV_SPHERE(r,stacks,slices)` | `<Sphere radius="{r}"/>` | No post-transform CSG |
| `GEN_CYLINDER(r,h,segs,caps)` | `<Cylinder radius="{r}" height="{h}"/>` | No post-transform CSG |
| `GEN_CONE(r,h,segs)` | `<Cone bottomRadius="{r}" height="{h}"/>` | No post-transform CSG |

Using X3D primitives is PREFERRED because:
- Smaller file size (parameters vs. full vertex data).
- X3D browsers can optimize rendering for known primitives.
- Aligns with X3D's declarative philosophy.

**Strategy B: Explicit IndexedFaceSet (When primitives are insufficient)**

Required when:
- The mesh has been modified by CSG operations.
- The mesh has been transformed by MAT4_APPLY.
- The generator has no X3D primitive equivalent (GEN_TORUS, GEN_ICOSPHERE, GEN_PLANE).
- Custom UV mapping or normals differ from primitive defaults.

### 5.4 CSG Emission

CSG (Constructive Solid Geometry) opcodes produce computed mesh results. X3D 4.0 includes the CADGeometry component with Boolean operations, but support is limited. The emitter MUST emit the computed IndexedFaceSet result:

```
CSG_UNION(meshA, meshB) →
  <Shape>
    <Appearance>...</Appearance>
    <IndexedFaceSet coordIndex="{computed_union_indices}" solid="true" creaseAngle="1.0">
      <Coordinate point="{computed_union_vertices}" />
      <Normal vector="{computed_union_normals}" />
    </IndexedFaceSet>
  </Shape>
```

The emitter MAY additionally emit the CADGeometry BooleanOperation metadata for browsers that support it, as a progressive enhancement:

```xml
<MetadataSet name="pmkr:csg">
  <MetadataString name="operation" value='"union"'/>
  <MetadataString name="operandA" value='"mesh_bookshelf_body"'/>
  <MetadataString name="operandB" value='"mesh_bookshelf_shelf"'/>
</MetadataSet>
```

### 5.5 Extrusion Emission

The `EXTRUDE` opcode takes a Path2D cross-section and extrudes it along the Z axis. This maps directly to X3D's Extrusion node:

```
EXTRUDE(path, depth) →
  <Shape>
    <Appearance>...</Appearance>
    <Extrusion
      crossSection="{path_points_2D}"
      spine="0 0 0, 0 0 {depth}"
      solid="true"
      creaseAngle="1.0"
      beginCap="true"
      endCap="true" />
  </Shape>
```

**Path2D to crossSection mapping:**
- Each `MoveTo(x,y)` starts a new cross-section contour.
- Each `LineTo(x,y)` adds a point to the cross-section.
- Curves are tessellated to line segments (per §4.5 parameters).
- The cross-section MUST be closed (last point = first point).

### 5.6 Lathe Emission

The `LATHE` opcode revolves a Path2D profile around the Y axis. This maps to X3D Extrusion with a circular spine:

```
LATHE(path, segments) →
  <Shape>
    <Appearance>...</Appearance>
    <Extrusion
      crossSection="{path_points_2D}"
      spine="{circular_spine_points}"
      solid="true"
      creaseAngle="1.0" />
  </Shape>
```

The spine is a circle of `segments` points in the XZ plane:
```
spine = [(cos(2πi/n), 0, sin(2πi/n)) for i in 0..n]
```

---

## 6. Transform Emission

### 6.1 Transform Opcodes to X3D

Matrix transform opcodes produce X3D Transform nodes that position, orient, and scale emitted geometry.

**Table 6.1 --- Transform opcode to X3D mapping**

| RPN Opcode | IR Output | X3D Field |
|------------|-----------|-----------|
| `MAT4_TRANSLATE(tx,ty,tz)` | Translation matrix | `<Transform translation="{tx} {ty} {tz}">` |
| `MAT4_SCALE(sx,sy,sz)` | Scale matrix | `<Transform scale="{sx} {sy} {sz}">` |
| `MAT4_ROTATE_X(angle)` | Rotation matrix | `<Transform rotation="1 0 0 {angle}">` |
| `MAT4_ROTATE_Y(angle)` | Rotation matrix | `<Transform rotation="0 1 0 {angle}">` |
| `MAT4_ROTATE_Z(angle)` | Rotation matrix | `<Transform rotation="0 0 1 {angle}">` |
| `MAT4_MUL(A,B)` | Composed matrix | Nested Transform nodes OR decomposed TRS |
| `MAT4_APPLY(M, mesh)` | Transformed mesh | Transform wrapping geometry |

### 6.2 Matrix Decomposition

When a composed matrix (`MAT4_MUL` result) is applied, the emitter SHOULD decompose it into Translation, Rotation, Scale (TRS) for X3D Transform:

```
MAT4_MUL(translate, MAT4_MUL(rotate, scale)) →
  <Transform
    translation="{tx} {ty} {tz}"
    rotation="{axis_x} {axis_y} {axis_z} {angle}"
    scale="{sx} {sy} {sz}">
    <!-- child geometry -->
  </Transform>
```

If the matrix contains shear or cannot be cleanly decomposed, the emitter MUST use a Matrix4 node (X3D 4.0 RigidBodyPhysics component) or emit pre-transformed vertex data in the Coordinate node.

### 6.3 Spatial Positioning from Morton Code

Galaxy entries carry semantic embeddings that map to 3D positions via Morton space-filling curves. The emitter MUST position each scene fragment's root Transform at the Morton-decoded position:

```
entry.embedding → morton_encode → (x, y, z)
→ <Transform translation="{x} {y} {z}" DEF="{canonicalId}">
    <!-- emitted geometry from formProgram -->
  </Transform>
```

This preserves the spatial semantics invariant: **semantically similar knowledge is spatially proximate** in the emitted X3D scene.

---

## 7. Material and Appearance Emission

### 7.1 Color Opcodes to X3D Appearance

RPN drawing color opcodes map to X3D Material and Appearance nodes:

```
SET_STROKE_COLOR(r, g, b, a) →
  <Appearance>
    <Material emissiveColor="{r} {g} {b}" transparency="{1-a}"/>
    <LineProperties linewidthScaleFactor="{current_lineWidth}"/>
  </Appearance>

SET_FILL_COLOR(r, g, b, a) →
  <Appearance>
    <Material diffuseColor="{r} {g} {b}" transparency="{1-a}"/>
  </Appearance>
```

### 7.2 Default Materials

When no color opcodes precede STROKE or FILL, the emitter MUST apply default materials:

| Operation | Default Material |
|-----------|-----------------|
| STROKE | `emissiveColor="0 0 0"` (black lines) |
| FILL | `diffuseColor="0.8 0.8 0.8"` (light gray) |
| Mesh (GEN_*) | `diffuseColor="0.7 0.7 0.7"` (neutral gray) |

### 7.3 PhysicalMaterial (PBR) Emission

For mesh geometry representing physical objects (House furniture, containers, instruments), the emitter SHOULD use X3D 4.0's PhysicalMaterial node:

```xml
<Appearance>
  <PhysicalMaterial
    baseColor="{rgb}"
    metallic="{0..1}"
    roughness="{0..1}" />
</Appearance>
```

Material properties MAY be encoded in the Galaxy entry's metadata or derived from the domain (e.g., `domain="wood"` → roughness=0.8, metallic=0.0).

---

## 8. House Artifact Trigger Pipeline

### 8.1 Artifact Structure

A House artifact is a glTF/GLB object with `extras.k3d` metadata that identifies its associated Galaxy and knowledge content:

```json
{
  "nodes": [{
    "name": "MathBook",
    "translation": [5.0, 1.2, 3.0],
    "extras": {
      "k3d": {
        "type": "book",
        "galaxy": "Math",
        "canonicalId": "artifact:math:calculus_textbook",
        "contentManifest": [
          "char:U+2211", "char:U+222B", "char:U+2202",
          "word:en:derivative", "word:en:integral",
          "rule:calculus:power_rule", "rule:calculus:chain_rule"
        ],
        "loadBehavior": "on_interact"
      }
    }
  }]
}
```

### 8.2 Trigger Events

House artifacts trigger emission in three ways:

**Trigger A: Load-Time Emission**
- `loadBehavior: "on_load"` — emit when House loads.
- Used for: Room structure, permanent furniture, always-visible objects.
- The artifact's geometry (from glTF mesh) is emitted as X3D geometry.
- The artifact's `contentManifest` entries are NOT emitted yet (they load into Galaxy).

**Trigger B: Interaction Emission**
- `loadBehavior: "on_interact"` — emit when agent or user interacts.
- Used for: Books, journals, tools, instruments.
- On interaction: resolve all `contentManifest` entries from Galaxy, execute their RPN programs, emit the combined scene fragment.
- A book "opening" emits its content Galaxy as an X3D scene.

**Trigger C: Query Emission**
- `loadBehavior: "on_query"` — emit as part of a knowledge query result.
- Used for: Knowledge nodes retrieved by the composed head pipeline.
- The query result (top-K scored candidates) triggers emission of each candidate's scene fragment.

### 8.3 Content Manifest Resolution

The `contentManifest` is a list of canonical IDs. Resolution follows:

1. For each canonical ID in manifest:
   a. Look up the entry in the source Galaxy (named by `galaxy` field).
   b. Resolve all `canonicalRefs` recursively (depth-first, fail-fast on missing).
   c. Collect the complete dependency closure.
2. Topologically sort the closure (lower layers first: form → meaning → rules → meta_rules).
3. Execute RPN programs in topological order (forms first, so meanings can reference emitted forms).
4. Compose scene fragments into a single X3D sub-scene rooted at a GalaxyGroup.

**Example --- Math book opening:**

```
contentManifest: ["char:U+2211", "word:en:summation", "rule:calculus:sum_notation"]

Resolution:
  char:U+2211 → ProceduralFormNode (no dependencies)
  word:en:summation → ProceduralMeaningNode
    → depends on: char:U+2211 (via charRefs) — already resolved
  rule:calculus:sum_notation → ProceduralRulesNode
    → depends on: char:U+2211 (via symbolRefs), word:en:summation (via wordRefs) — already resolved

Emission order: char:U+2211, word:en:summation, rule:calculus:sum_notation
Each emits its scene fragment; combined into GalaxyGroup.
```

---

## 9. Galaxy-to-Scene Composition

### 9.1 Single Galaxy Emission

A complete Galaxy emits as a single X3D GalaxyGroup containing all its entries:

```xml
<GalaxyGroup DEF="MATH_GALAXY" galaxyName="Math" galaxyType="default" entryCount="152">
  <!-- Layer 1: Form nodes (characters, symbols) -->
  <Transform translation="10.5 23.1 -5.3">
    <ProceduralFormNode DEF="CHAR_2211" canonicalId="char:U+2211" galaxy="Math">
      <RPNProgram containerField="formProgram"
          opcodes="32 8 MOVE 8 32 LINE 32 56 LINE 56 56 LINE STROKE"/>
      <!-- Emitted geometry (from RPN execution): -->
      <Shape>
        <Appearance><Material emissiveColor="0 0 0"/></Appearance>
        <IndexedLineSet coordIndex="0 1 2 3 -1">
          <Coordinate point="32 8 0, 8 32 0, 32 56 0, 56 56 0"/>
        </IndexedLineSet>
      </Shape>
    </ProceduralFormNode>
  </Transform>

  <!-- Layer 2: Meaning nodes referencing Layer 1 -->
  <Transform translation="12.3 24.0 -4.8">
    <ProceduralMeaningNode DEF="WORD_SUMMATION"
        canonicalId="word:en:summation" galaxy="Math"
        charRefs='"char:U+2211"'>
      <RPNProgram containerField="meaningProgram"
          opcodes="LOAD_CONTEXT RECALL_ITERATOR RECALL_BOUNDS APPLY_SUM NORMALIZE"/>
      <!-- No geometry duplication: references char:U+2211 via CanonicalReference -->
      <CanonicalReference targetId="char:U+2211" targetLayer="form" role="symbol_ref"/>
    </ProceduralMeaningNode>
  </Transform>

  <!-- Layer 3: Rules nodes referencing Layers 1 and 2 -->
  <Transform translation="14.1 25.2 -4.2">
    <ProceduralRulesNode DEF="RULE_SUM_NOTATION"
        canonicalId="rule:calculus:sum_notation" galaxy="Math"
        symbolRefs='"char:U+2211"' wordRefs='"word:en:summation"'>
      <RPNProgram containerField="meaningProgram"
          opcodes="RECALL_LOWER_BOUND RECALL_UPPER_BOUND RECALL_EXPRESSION ITERATE_SUM STORE"/>
    </ProceduralRulesNode>
  </Transform>
</GalaxyGroup>
```

### 9.2 Multi-Galaxy Scene

A complete House scene contains multiple galaxies, each as a GalaxyGroup, all within an AgentMemoryPalace:

```xml
<AgentMemoryPalace DEF="MY_HOUSE" palaceId="house:default">
  <!-- Room: Library -->
  <Transform translation="0 0 0">
    <GalaxyGroup galaxyName="Math" galaxyType="default">
      <!-- Math Galaxy entries (emitted forms + references) -->
    </GalaxyGroup>
    <GalaxyGroup galaxyName="Grammar" galaxyType="default">
      <!-- Grammar Galaxy entries -->
    </GalaxyGroup>
    <!-- Room geometry (furniture, walls, floor) from mesh opcodes -->
    <Shape>
      <Appearance><PhysicalMaterial baseColor="0.6 0.4 0.2" roughness="0.8"/></Appearance>
      <IndexedFaceSet coordIndex="...">
        <Coordinate point="..."/>
      </IndexedFaceSet>
    </Shape>
  </Transform>

  <!-- Room: Knowledge Garden -->
  <Transform translation="50 0 0">
    <GalaxyGroup galaxyName="Reality" galaxyType="default">
      <!-- Reality Galaxy entries -->
    </GalaxyGroup>
  </Transform>
</AgentMemoryPalace>
```

### 9.3 Cross-Galaxy References

When a Layer 3 rule in Grammar Galaxy references a Layer 1 symbol in Character Galaxy, the emitter MUST resolve the reference across galaxy boundaries using canonical IDs:

```xml
<!-- In Character Galaxy -->
<ProceduralFormNode DEF="CHAR_2211" canonicalId="char:U+2211" galaxy="Character" .../>

<!-- In Grammar Galaxy (different GalaxyGroup) -->
<ProceduralRulesNode DEF="RULE_SUM" canonicalId="rule:calculus:sum_notation" galaxy="Grammar"
    symbolRefs='"char:U+2211"'>
  <CanonicalReference targetId="char:U+2211" targetLayer="form" role="symbol_ref"/>
</ProceduralRulesNode>
```

The X3D `DEF/USE` mechanism handles intra-scene references. For inter-scene references (knowledge in separate X3D files), the emitter MUST use X3D `Inline` nodes:

```xml
<Inline url='"CharacterGalaxy.x3d"' />
<!-- RULE_SUM can now reference CHAR_2211 from the inlined scene -->
```

---

## 10. X3D Serialization Targets

### 10.1 XML Encoding (.x3d)

The primary interchange format. Human-readable, validatable against X3D DTD/Schema.

**Header:**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<X3D version='4.0' profile='ProceduralMemoryInterchange'>
  <head>
    <component name='ProceduralMemory' level='2'/>
    <component name='KnowledgeNavigation' level='1'/>
    <meta name='title' content='{scene_title}'/>
    <meta name='generator' content='K3D PM-KR Emitter v0.1'/>
    <meta name='pmkr:conformance' content='Level B (Sovereign Runtime)'/>
    <meta name='pmkr:sourceGalaxies' content='Math Grammar Character'/>
    <meta name='pmkr:entryCount' content='352'/>
  </head>
  <Scene>
    <!-- Emitted content -->
  </Scene>
</X3D>
```

### 10.2 ClassicVRML Encoding (.x3dv)

Compact text format compatible with VRML97-era tools:

```vrml
#X3D V4.0 utf8
PROFILE ProceduralMemoryInterchange
COMPONENT ProceduralMemory:2
COMPONENT KnowledgeNavigation:1
META "title" "MathGalaxy"
META "generator" "K3D PM-KR Emitter v0.1"

DEF CHAR_2211 ProceduralFormNode {
  canonicalId "char:U+2211"
  galaxy "Character"
  formProgram RPNProgram {
    opcodes "32 8 MOVE 8 32 LINE 32 56 LINE STROKE"
    tier "standard"
  }
}
```

### 10.3 JSON Encoding (.x3dj)

Machine-friendly format for web applications:

```json
{
  "X3D": {
    "@version": "4.0",
    "@profile": "ProceduralMemoryInterchange",
    "head": {
      "component": [
        {"@name": "ProceduralMemory", "@level": 2}
      ],
      "meta": [
        {"@name": "generator", "@content": "K3D PM-KR Emitter v0.1"}
      ]
    },
    "Scene": {
      "-children": [
        {
          "ProceduralFormNode": {
            "@DEF": "CHAR_2211",
            "@canonicalId": "char:U+2211",
            "@galaxy": "Character",
            "-formProgram": {
              "RPNProgram": {
                "@opcodes": "32 8 MOVE 8 32 LINE 32 56 LINE STROKE",
                "@tier": "standard"
              }
            }
          }
        }
      ]
    }
  }
}
```

### 10.4 Compressed Binary Encoding (.x3db)

For performance-critical interchange (large galaxies with thousands of entries). RPN programs encode as raw byte sequences using opcode hex values:

```
[4-byte program length][opcode bytes...]
```

Example: `32 8 MOVE 8 32 LINE STROKE`
→ `[7] [0x20] [0x08] [0x64] [0x08] [0x20] [0x65] [0x6A]`

(Where numeric literals are encoded as their byte values and opcodes as their hex codes from the registry.)

### 10.5 glTF Encoding (.glb)

For K3D-native round-trip, the emitter produces glTF with `extras.k3d` metadata preserving all PM-KR fields. See §14 for round-trip preservation rules.

---

## 11. Scene Access Interface (SAI) Integration

### 11.1 SAI Overview

The X3D Scene Access Interface (ISO/IEC 19775-2) defines how external programs create, modify, and query nodes in a live X3D scene. PM-KR emission integrates with SAI for runtime scene manipulation.

### 11.2 Live Emission via SAI

When the TRM agent navigates to new knowledge during reasoning, it emits scene fragments into the live X3D scene via SAI:

```
Agent perceives query → navigates Galaxy → retrieves candidates →
  for each candidate:
    1. SAI.createNode("ProceduralFormNode")
    2. SAI.setField(node, "canonicalId", candidate.canonicalId)
    3. SAI.setField(node, "formProgram", candidate.formProgram)
    4. Execute formProgram → IR → create geometry nodes via SAI
    5. SAI.addChild(sceneRoot, transformNode)
```

### 11.3 Event Model Integration

Emitted scene fragments connect to X3D's event model via ROUTEs:

```xml
<!-- Knowledge query sensor -->
<KnowledgeQuery DEF="QUERY_SENSOR" queryRadius="5.0" enabled="true"/>

<!-- When query results change, update visualization -->
<ROUTE fromNode="QUERY_SENSOR" fromField="resultCount"
       toNode="RESULT_DISPLAY" toField="set_whichChoice"/>
```

### 11.4 Incremental Emission

For large galaxies, the emitter SHOULD support incremental emission:

1. **LOD-based**: Emit geometry at the current LOD level only. High-LOD details emit on demand.
2. **Frustum-based**: Only emit scene fragments for knowledge nodes currently in the agent's frustum.
3. **Priority-based**: Emit high-confidence, high-relevance entries first; fill in lower-priority entries progressively.

This aligns with the KnowledgeLOD and KnowledgeFrustum nodes from the PM-KR component spec.

---

## 12. Dual-Client Emission

### 12.1 Dual Texture Emission

Every emitted ProceduralFormNode carries two representations via DualClientTexture:

**Human texture (UV Map 0):**
- The emitter renders the form program output at 512x512 resolution.
- Readable fonts (14--18pt equivalent), proper anti-aliasing, aesthetic colors.
- Stored as ImageTexture referencing a PNG/JPEG.

**Machine texture (UV Map 1):**
- The emitter encodes the form program output at 256x256 resolution.
- Maximum information density, small fonts (6--8pt), structured layout.
- Encoding: `text_as_image` (default), `embedding_grid`, or `rpn_encoded`.

```xml
<ProceduralFormNode DEF="CHAR_2211" canonicalId="char:U+2211">
  <!-- Emitted geometry (shared between clients) -->
  <Shape>
    <IndexedLineSet coordIndex="0 1 2 3 -1">
      <Coordinate point="32 8 0, 8 32 0, 32 56 0, 56 56 0"/>
    </IndexedLineSet>
  </Shape>
  <!-- Dual-client textures -->
  <DualClientTexture containerField="dualTexture">
    <ImageTexture containerField="humanTexture" url='"textures/char_2211_human.png"'/>
    <ImageTexture containerField="machineTexture" url='"textures/char_2211_machine.png"'/>
  </DualClientTexture>
</ProceduralFormNode>
```

### 12.2 Geometry Identity Guarantee

Both the human-client rendering (geometry + UV Map 0 texture) and the machine-client data (geometry + UV Map 1 texture + RPN program + embedding) derive from the same canonical source: the `formProgram`. The emitter MUST NOT produce divergent geometry for the two clients.

### 12.3 Emission Mode Selection

The emitter supports three output modes:

| Mode | Output | Use Case |
|------|--------|----------|
| `human` | Geometry + Appearance + UV Map 0 textures | Visual rendering for human viewers |
| `machine` | Geometry + UV Map 1 textures + metadata + embeddings | AI/machine consumption |
| `dual` (default) | Both representations in same scene | Full PM-KR interchange |

---

## 13. Cross-Layer Reference Emission

### 13.1 Reference-Preserving Emission

The emitter MUST preserve the symlink reference structure. When emitting a meaning node that references form nodes, the emitter MUST NOT inline the form node's geometry. Instead:

```xml
<!-- Form node emitted once -->
<ProceduralFormNode DEF="CHAR_2211" canonicalId="char:U+2211" galaxy="Character">
  <!-- Geometry emitted from formProgram -->
</ProceduralFormNode>

<!-- Meaning node references via CanonicalReference (no geometry duplication) -->
<ProceduralMeaningNode DEF="WORD_SUM" canonicalId="word:en:summation" galaxy="Word">
  <CanonicalReference targetId="char:U+2211" targetLayer="form" role="symbol_ref"/>
  <!-- If the meaning node needs to DISPLAY the symbol, it uses X3D USE: -->
  <ProceduralFormNode USE="CHAR_2211"/>
</ProceduralMeaningNode>
```

### 13.2 Compression Metrics Emission

The emitter SHOULD report compression metrics in the scene header:

```xml
<meta name='pmkr:totalNodes' content='352'/>
<meta name='pmkr:uniqueCanonicalSources' content='152'/>
<meta name='pmkr:referenceCount' content='1847'/>
<meta name='pmkr:estimatedSavingsBytes' content='2847296'/>
<meta name='pmkr:compressionRatio' content='69.3'/>
```

---

## 14. glTF Round-Trip Preservation

### 14.1 X3D → glTF Mapping

When emitting to glTF (for K3D-native House storage), PM-KR fields map to `extras.k3d`:

| X3D PM-KR Field | glTF Location |
|------------------|--------------|
| `canonicalId` | `node.extras.k3d.canonicalId` |
| `layer` | `node.extras.k3d.layer` |
| `galaxy` | `node.extras.k3d.galaxy` |
| `formProgram.opcodes` | `node.extras.k3d.formProgram` |
| `meaningProgram.opcodes` | `node.extras.k3d.meaningProgram` |
| `embedding.value` | `node.extras.k3d.embedding16` |
| `canonicalRefs[].targetId` | `node.extras.k3d.refs[]` |
| `confidence` | `node.extras.k3d.confidence` |
| `provenance` | `node.extras.k3d.provenance` |
| Geometry (Coordinate, IndexedFaceSet) | Standard glTF mesh/accessor |
| Material (diffuseColor, PhysicalMaterial) | Standard glTF material (PBR) |
| Transform (translation, rotation, scale) | Standard glTF node TRS |

### 14.2 glTF → X3D Reconstruction

When loading a glTF file with `extras.k3d` metadata, the importer:

1. Reads `extras.k3d` fields from each glTF node.
2. Reconstructs PM-KR node types based on `layer` field.
3. Re-parses `formProgram` / `meaningProgram` strings as RPNProgram opcodes.
4. Reconstructs CanonicalReference nodes from `refs[]` arrays.
5. Maps glTF mesh data to X3D IndexedFaceSet/IndexedTriangleSet.
6. Maps glTF PBR material to X3D PhysicalMaterial.
7. Wraps in appropriate GalaxyGroup based on `galaxy` field.

### 14.3 Round-Trip Invariant

For any PM-KR scene S:
```
X3D_emit(S) → glTF → X3D_import → S'
```
S' MUST preserve:
- All canonical IDs, layer classifications, and galaxy assignments.
- All RPN program opcodes (byte-exact string comparison).
- All canonical reference target IDs and roles.
- All embedding vectors (float32 precision).
- Geometry topology (vertex count, face count, connectivity).

S' MAY differ in:
- Vertex position precision (float32 rounding).
- Material color precision (float32 rounding).
- Node ordering within GalaxyGroup (topological order SHOULD be preserved but is not required).

---

## 15. Conformance

### 15.1 Emitter Conformance Levels

**Level 1: Drawing Emitter**
- Emits IndexedLineSet from STROKE opcodes.
- Emits IndexedFaceSet from FILL opcodes with triangulation.
- Handles MOVE, LINE, CLOSE, STROKE, FILL, SET_STROKE_COLOR, SET_FILL_COLOR.
- Produces valid X3D XML encoding.
- Preserves canonicalId in emitted nodes.

**Level 2: Full Geometry Emitter**
- Level 1 plus:
- Emits mesh geometry from all mesh opcodes (VERTEX3, TRI_FACE, GEN_*, CSG_*, EXTRUDE, LATHE).
- Emits Transform nodes from matrix opcodes.
- Emits PhysicalMaterial from color context.
- Produces valid X3D XML, ClassicVRML, and JSON encodings.
- Preserves all PM-KR metadata (layer, galaxy, embedding, canonicalRefs).

**Level 3: Full PM-KR Emitter**
- Level 2 plus:
- Emits DualClientTexture pairs.
- Emits GalaxyGroup composition from multi-galaxy scenes.
- Emits AgentEntity, AgentSwarm, AgentMemoryPalace.
- Supports incremental emission (LOD, Frustum, Priority).
- Supports SAI live emission.
- Produces all five encoding formats (.x3d, .x3dv, .x3dj, .x3db, .glb).
- Supports glTF round-trip preservation.
- Reports compression metrics.

### 15.2 Conformance Tests

An emitter claiming Level N conformance MUST pass:

| Test | Level | Verification |
|------|-------|-------------|
| Drawing emission | 1 | STROKE produces valid IndexedLineSet with correct coordIndex |
| Fill emission | 1 | FILL produces valid IndexedFaceSet with correct triangulation |
| Color emission | 1 | SET_STROKE_COLOR → Material emissiveColor |
| Curve tessellation | 1 | BEZIER/ARC tessellated within maxChordError |
| Mesh emission | 2 | MESH_BEGIN/VERTEX3/TRI_FACE/MESH_END → valid IndexedTriangleSet |
| Primitive emission | 2 | GEN_CUBE → Box (or equivalent IndexedFaceSet) |
| CSG emission | 2 | CSG_UNION → valid watertight IndexedFaceSet |
| Extrusion emission | 2 | EXTRUDE → valid Extrusion or IndexedFaceSet |
| Transform emission | 2 | MAT4_TRANSLATE → Transform translation |
| Reference preservation | 2 | CanonicalReference nodes present, no content duplication |
| Dual texture emission | 3 | DualClientTexture with humanTexture and machineTexture |
| Galaxy composition | 3 | Multiple GalaxyGroup nodes with cross-galaxy references |
| Incremental emission | 3 | LOD-filtered scene fragment count < total entry count |
| glTF round-trip | 3 | X3D → glTF → X3D preserves all PM-KR fields |
| SAI emission | 3 | createNode + setField produces live scene graph updates |

---

## 16. Examples

### 16.1 Drawing Galaxy Entry → X3D Line Art

**Source:** Character Galaxy entry for plus sign (+).

**Galaxy entry:**
```python
{
    "canonicalId": "char:U+002B",
    "layer": "form",
    "galaxy": "Character",
    "formProgram": "16 32 MOVE 48 32 LINE 32 16 MOVE 32 48 LINE STROKE",
    "embedding16": [0.5, 0.3, -0.2, 0.8, ...]
}
```

**RPN execution trace:**
1. `16 32 MOVE` → MoveTo(16, 32) — horizontal bar start
2. `48 32 LINE` → LineTo(48, 32) — horizontal bar end
3. `32 16 MOVE` → MoveTo(32, 16) — vertical bar start
4. `32 48 LINE` → LineTo(32, 48) — vertical bar end
5. `STROKE` → emit

**Emitted X3D (XML):**
```xml
<Transform DEF="char_U_002B" translation="5.2 12.8 -3.1">
  <ProceduralFormNode
      canonicalId="char:U+002B"
      galaxy="Character"
      domain="arithmetic"
      confidence="1.0">
    <RPNProgram containerField="formProgram"
        opcodes="16 32 MOVE 48 32 LINE 32 16 MOVE 32 48 LINE STROKE"
        tier="standard"
        stackDepth="69"/>
    <MetadataFloat containerField="embedding"
        name="embedding16"
        value="0.5 0.3 -0.2 0.8 0.1 -0.4 0.6 0.2 -0.1 0.7 0.4 -0.3 0.5 0.1 -0.2 0.9"/>
    <!-- Emitted geometry from formProgram execution -->
    <Shape>
      <Appearance>
        <Material emissiveColor="0 0 0"/>
        <LineProperties linewidthScaleFactor="1.0"/>
      </Appearance>
      <IndexedLineSet coordIndex="0 1 -1 2 3 -1">
        <Coordinate point="16 32 0, 48 32 0, 32 16 0, 32 48 0"/>
      </IndexedLineSet>
    </Shape>
  </ProceduralFormNode>
</Transform>
```

### 16.2 Mesh Galaxy Entry → X3D Solid Geometry

**Source:** 3DObjects Galaxy entry for a bookshelf.

**Galaxy entry:**
```python
{
    "canonicalId": "object:furniture:bookshelf_small",
    "layer": "form",
    "galaxy": "3DObjects",
    "formProgram": "1.0 2.0 0.3 GEN_CUBE 0.3 0 0 MAT4_TRANSLATE MAT4_APPLY "
                   "0.9 0.02 0.28 GEN_CUBE 0 0.5 0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
                   "0.9 0.02 0.28 GEN_CUBE 0 1.0 0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
                   "0.9 0.02 0.28 GEN_CUBE 0 1.5 0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION"
}
```

**Emitted X3D (XML):**
```xml
<Transform DEF="obj_bookshelf_small" translation="2.0 0.0 8.5">
  <ProceduralFormNode
      canonicalId="object:furniture:bookshelf_small"
      galaxy="3DObjects"
      domain="furniture">
    <RPNProgram containerField="formProgram"
        opcodes="1.0 2.0 0.3 GEN_CUBE 0.3 0 0 MAT4_TRANSLATE MAT4_APPLY
                 0.9 0.02 0.28 GEN_CUBE 0 0.5 0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION
                 0.9 0.02 0.28 GEN_CUBE 0 1.0 0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION
                 0.9 0.02 0.28 GEN_CUBE 0 1.5 0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION"
        tier="standard"/>
    <!-- Emitted geometry (CSG result — explicit mesh) -->
    <Shape>
      <Appearance>
        <PhysicalMaterial baseColor="0.55 0.35 0.18" roughness="0.8" metallic="0.0"/>
      </Appearance>
      <IndexedFaceSet coordIndex="..." solid="true" creaseAngle="1.0">
        <Coordinate point="..."/>
        <Normal vector="..."/>
        <TextureCoordinate point="..."/>
      </IndexedFaceSet>
    </Shape>
  </ProceduralFormNode>
</Transform>
```

### 16.3 Extrusion from 2D Path

**Source:** Drawing Galaxy glyph extruded to 3D for House display.

**Galaxy entry:**
```python
{
    "canonicalId": "display:char:U+2211:3d",
    "layer": "form",
    "galaxy": "Drawing",
    "formProgram": "32 8 MOVE 8 32 LINE 32 56 LINE 56 56 LINE CLOSE 0.5 EXTRUDE"
}
```

**Emitted X3D:**
```xml
<Shape>
  <Appearance>
    <PhysicalMaterial baseColor="0.9 0.85 0.7" roughness="0.3" metallic="0.5"/>
  </Appearance>
  <Extrusion
    crossSection="32 8, 8 32, 32 56, 56 56, 32 8"
    spine="0 0 0, 0 0 0.5"
    solid="true"
    creaseAngle="1.0"
    beginCap="true"
    endCap="true" />
</Shape>
```

### 16.4 Complete Book Artifact → X3D Scene

**Scenario:** User interacts with a math book in the House library. The book's content (a Galaxy of calculus knowledge) emits as an X3D scene.

**Trigger artifact (glTF):**
```json
{
  "name": "CalculusBook",
  "extras": {
    "k3d": {
      "type": "book",
      "galaxy": "Math",
      "canonicalId": "artifact:math:calculus_intro",
      "contentManifest": [
        "char:U+2211", "char:U+222B", "char:U+2202",
        "word:en:derivative", "word:en:integral", "word:en:limit",
        "rule:calculus:power_rule", "rule:calculus:chain_rule",
        "rule:calculus:sum_notation",
        "meta_rule:pedagogy:scaffold_calculus"
      ],
      "loadBehavior": "on_interact"
    }
  }
}
```

**Emitted X3D scene (complete):**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<X3D version='4.0' profile='ProceduralMemoryInterchange'>
  <head>
    <component name='ProceduralMemory' level='2'/>
    <component name='KnowledgeNavigation' level='1'/>
    <meta name='title' content='CalculusBook Contents'/>
    <meta name='generator' content='K3D PM-KR Emitter v0.1'/>
    <meta name='pmkr:sourceArtifact' content='artifact:math:calculus_intro'/>
    <meta name='pmkr:sourceGalaxies' content='Math Character Word Grammar'/>
    <meta name='pmkr:entryCount' content='10'/>
    <meta name='pmkr:compressionRatio' content='42.7'/>
  </head>
  <Scene>
    <!-- Spatial index for knowledge queries within this book -->
    <SpatialKnowledgeIndex DEF="BOOK_INDEX" indexType="morton" maxDepth="6"/>

    <!-- Layer 1: Form nodes (canonical sources, emitted once) -->
    <GalaxyGroup galaxyName="Character" galaxyType="default">
      <Transform translation="0 0 0">
        <ProceduralFormNode DEF="CHAR_2211" canonicalId="char:U+2211" galaxy="Character">
          <RPNProgram containerField="formProgram"
              opcodes="32 8 MOVE 8 32 LINE 32 56 LINE 56 56 LINE STROKE"/>
          <Shape>
            <Appearance><Material emissiveColor="0 0 0"/></Appearance>
            <IndexedLineSet coordIndex="0 1 2 3 -1">
              <Coordinate point="32 8 0, 8 32 0, 32 56 0, 56 56 0"/>
            </IndexedLineSet>
          </Shape>
        </ProceduralFormNode>
      </Transform>
      <Transform translation="2 0 0">
        <ProceduralFormNode DEF="CHAR_222B" canonicalId="char:U+222B" galaxy="Character">
          <RPNProgram containerField="formProgram"
              opcodes="24 8 MOVE 20 16 16 24 BEZIER 16 40 20 48 BEZIER STROKE"/>
          <Shape>
            <Appearance><Material emissiveColor="0 0 0"/></Appearance>
            <IndexedLineSet coordIndex="0 1 2 3 4 5 6 7 -1">
              <Coordinate point="24 8 0, 22 12 0, 20 16 0, 18 20 0,
                                 16 28 0, 16 32 0, 18 40 0, 20 48 0"/>
            </IndexedLineSet>
          </Shape>
        </ProceduralFormNode>
      </Transform>
      <Transform translation="4 0 0">
        <ProceduralFormNode DEF="CHAR_2202" canonicalId="char:U+2202" galaxy="Character">
          <RPNProgram containerField="formProgram"
              opcodes="40 32 MOVE 32 48 24 32 BEZIER 32 16 40 32 BEZIER STROKE"/>
          <Shape>
            <Appearance><Material emissiveColor="0 0 0"/></Appearance>
            <IndexedLineSet coordIndex="0 1 2 3 4 5 6 -1">
              <Coordinate point="40 32 0, 36 40 0, 32 48 0, 28 40 0,
                                 28 24 0, 32 16 0, 40 32 0"/>
            </IndexedLineSet>
          </Shape>
        </ProceduralFormNode>
      </Transform>
    </GalaxyGroup>

    <!-- Layer 2: Meaning nodes (reference Layer 1 via symlinks) -->
    <GalaxyGroup galaxyName="Word" galaxyType="default">
      <Transform translation="10 0 0">
        <ProceduralMeaningNode DEF="WORD_DERIVATIVE"
            canonicalId="word:en:derivative" galaxy="Word" domain="calculus"
            charRefs='"char:U+0064" "char:U+0065" "char:U+0072" "char:U+0069"
                      "char:U+0076" "char:U+0061" "char:U+0074" "char:U+0069"
                      "char:U+0076" "char:U+0065"'>
          <RPNProgram containerField="meaningProgram"
              opcodes="LOAD_CONTEXT RECALL_FUNCTION RECALL_VARIABLE LIMIT_DELTA APPLY_QUOTIENT"/>
          <CanonicalReference targetId="char:U+2202" targetLayer="form" role="symbol_ref"/>
        </ProceduralMeaningNode>
      </Transform>
      <Transform translation="12 0 0">
        <ProceduralMeaningNode DEF="WORD_INTEGRAL"
            canonicalId="word:en:integral" galaxy="Word" domain="calculus">
          <RPNProgram containerField="meaningProgram"
              opcodes="LOAD_CONTEXT RECALL_INTEGRAND RECALL_BOUNDS APPLY_ANTIDERIVATIVE"/>
          <CanonicalReference targetId="char:U+222B" targetLayer="form" role="symbol_ref"/>
        </ProceduralMeaningNode>
      </Transform>
    </GalaxyGroup>

    <!-- Layer 3: Rules (reference Layers 1 and 2) -->
    <GalaxyGroup galaxyName="Grammar" galaxyType="default">
      <Transform translation="20 0 0">
        <ProceduralRulesNode DEF="RULE_POWER"
            canonicalId="rule:calculus:power_rule" galaxy="Grammar" domain="calculus"
            pattern="d/dx x^n = n*x^(n-1)" language="*" ruleStrength="1"
            wordRefs='"word:en:derivative"'>
          <RPNProgram containerField="meaningProgram"
              opcodes="RECALL_EXPONENT DUP 1 SUB STORE RECALL_BASE SWAP MUL STORE"/>
        </ProceduralRulesNode>
      </Transform>
      <Transform translation="22 0 0">
        <ProceduralRulesNode DEF="RULE_CHAIN"
            canonicalId="rule:calculus:chain_rule" galaxy="Grammar" domain="calculus"
            pattern="d/dx f(g(x)) = f'(g(x)) * g'(x)" language="*" ruleStrength="1"
            wordRefs='"word:en:derivative"'>
          <RPNProgram containerField="meaningProgram"
              opcodes="RECALL_OUTER RECALL_INNER DUP STORE APPLY_DERIVATIVE SWAP
                       APPLY_DERIVATIVE MUL STORE"/>
        </ProceduralRulesNode>
      </Transform>
    </GalaxyGroup>

    <!-- Layer 4: Meta-rules -->
    <GalaxyGroup galaxyName="Grammar" galaxyType="meta">
      <Transform translation="30 0 0">
        <ProceduralMetaRulesNode DEF="META_SCAFFOLD_CALC"
            canonicalId="meta_rule:pedagogy:scaffold_calculus" galaxy="Grammar"
            category="pedagogy"
            ruleRefs='"rule:calculus:power_rule" "rule:calculus:chain_rule"
                      "rule:calculus:sum_notation"'>
          <RPNProgram containerField="meaningProgram"
              opcodes="RECALL_STUDENT_LEVEL 3 LT BRANCH
                       RECALL_RULE_POWER APPLY
                       RECALL_RULE_CHAIN APPLY"/>
          <RPNProgram containerField="condition"
              opcodes="RECALL_CONTEXT_DOMAIN @0 EQ"
              stringConstants='"calculus"'/>
        </ProceduralMetaRulesNode>
      </Transform>
    </GalaxyGroup>
  </Scene>
</X3D>
```

### 16.5 Live Agent Emission via SAI

**Scenario:** TRM agent reasons about "What is the derivative of x²?" and emits scene fragments during reasoning.

```
// Pseudocode: SAI-based live emission during agent reasoning

// 1. Agent receives query
query = "What is the derivative of x²?"
embedding = embed(query)  // 16-dim semantic vector

// 2. Spatial navigation finds relevant knowledge
candidates = spatialIndex.query(embedding, radius=5.0, galaxyFilter=["Math", "Grammar"])

// 3. Agent emits retrieved candidates into live scene
for candidate in candidates:
    // Create PM-KR node via SAI
    node = SAI.createNode("ProceduralFormNode")
    SAI.setField(node, "canonicalId", candidate.canonicalId)
    SAI.setField(node, "galaxy", candidate.galaxy)

    // Execute formProgram to produce geometry
    ir = rpnEngine.execute(candidate.formProgram)
    geometry = emitGeometry(ir)  // IR → X3D geometry nodes

    // Position at Morton-decoded location
    transform = SAI.createNode("Transform")
    SAI.setField(transform, "translation", mortonDecode(candidate.embedding))
    SAI.addChild(transform, node)
    SAI.addChild(node, geometry)

    // Add to live scene
    SAI.addChild(sceneRoot, transform)

// 4. Agent navigates to power rule
pathfinder.setGoal(mortonDecode(embed("power rule")))
pathNodes = pathfinder.getPathNodes()

// 5. Agent applies rule and emits answer
answerNode = SAI.createNode("ProceduralMeaningNode")
SAI.setField(answerNode, "canonicalId", "answer:derivative:x_squared")
SAI.setField(answerNode, "meaningProgram", "2 RECALL_X MUL")  // 2x
```

---

## Appendix A: Complete Opcode-to-X3D Quick Reference

**Drawing Opcodes (Tier 2: Standard)**

| Opcode | Hex | X3D Output |
|--------|-----|-----------|
| MOVE | 0x64 | Coordinate point (new polyline) |
| LINE | 0x65 | Coordinate point (extend polyline) |
| BEZIER | 0x67 | Tessellated Coordinate points |
| ARC | 0x68 | Tessellated Coordinate points |
| CLOSE | 0x69 | coordIndex back to first point |
| STROKE | 0x6A | IndexedLineSet + Material(emissive) + LineProperties |
| FILL | 0x6B | IndexedFaceSet + Material(diffuse) + TextureCoordinate |
| SET_STROKE_COLOR | 0x75 | Material emissiveColor |
| SET_FILL_COLOR | 0x76 | Material diffuseColor |
| SET_LINE_WIDTH | 0x77 | LineProperties linewidthScaleFactor |

**Mesh Opcodes (mesh_opcodes.py)**

| Opcode | X3D Output |
|--------|-----------|
| MESH_BEGIN | (begin IndexedTriangleSet/IndexedFaceSet accumulation) |
| VERTEX3 | Coordinate point |
| NORMAL3 | Normal vector |
| UV2 | TextureCoordinate point |
| TRI_FACE | IndexedTriangleSet index |
| QUAD_FACE | IndexedFaceSet coordIndex (with -1 separator) |
| MESH_END | (finalize geometry node) |
| GEN_PLANE | IndexedFaceSet (subdivided quad) |
| GEN_CUBE | Box OR IndexedFaceSet |
| GEN_UV_SPHERE | Sphere OR IndexedFaceSet |
| GEN_CYLINDER | Cylinder OR IndexedFaceSet |
| GEN_CONE | Cone OR IndexedFaceSet |
| GEN_TORUS | IndexedFaceSet |
| GEN_ICOSPHERE | IndexedFaceSet |
| CSG_UNION | IndexedFaceSet (computed) |
| CSG_SUBTRACT | IndexedFaceSet (computed) |
| CSG_INTERSECT | IndexedFaceSet (computed) |
| EXTRUDE | Extrusion OR IndexedFaceSet |
| LATHE | Extrusion (circular spine) OR IndexedFaceSet |

**Transform Opcodes (mesh_opcodes.py)**

| Opcode | X3D Output |
|--------|-----------|
| MAT4_IDENTITY | (identity — no Transform emitted) |
| MAT4_TRANSLATE | Transform translation |
| MAT4_SCALE | Transform scale |
| MAT4_ROTATE_X | Transform rotation="1 0 0 {angle}" |
| MAT4_ROTATE_Y | Transform rotation="0 1 0 {angle}" |
| MAT4_ROTATE_Z | Transform rotation="0 0 1 {angle}" |
| MAT4_MUL | Nested Transform OR decomposed TRS |
| MAT4_APPLY | Transform wrapping child geometry |

---

## Appendix B: Emission Pipeline Implementation Reference

**K3D Implementation Files:**

| File | Role in Emission Pipeline |
|------|--------------------------|
| `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py` | Drawing opcode definitions (hex values, token names) |
| `knowledge3d/cranium/ptx_runtime/mesh_opcodes.py` | Mesh opcode definitions, MeshBuffer, Path2D, primitives, CSG, transforms |
| `knowledge3d/cranium/ptx_runtime/mesh_engine.py` | MeshRPNEngine: host-side RPN mesh executor (Stage 3: Execute) |
| `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py` | ModularRPNEngine: host-side RPN drawing executor (Stage 3: Execute) |
| `knowledge3d/tools/gltf_export.py` | MeshBuffer → glTF node conversion (Stage 5: Serialize to .glb) |
| `knowledge3d/tools/export_house.py` | House → glTF scene export (Stage 1+5: Trigger + Serialize) |

**New files needed for X3D emission (implementation targets for Codex):**

| File | Role |
|------|------|
| `knowledge3d/tools/x3d_emitter.py` | IR → X3D node tree construction (Stage 4) |
| `knowledge3d/tools/x3d_serializer.py` | X3D node tree → .x3d/.x3dv/.x3dj/.x3db (Stage 5) |
| `knowledge3d/tools/x3d_galaxy_emitter.py` | Galaxy → X3D GalaxyGroup emission (Stage 1-5 combined) |
| `knowledge3d/tools/x3d_roundtrip.py` | glTF ↔ X3D round-trip with PM-KR preservation (§14) |

---

## Appendix C: Document History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-03-26 | Initial draft. Complete opcode-to-X3D mapping, five-stage emission pipeline, House artifact trigger, Galaxy-to-scene composition, all serialization targets, SAI integration, dual-client emission, glTF round-trip, conformance levels. |

---

**End of Document**

*This specification is a working draft of the PM-KR Community Group (W3C). It defines how PM-KR's executable knowledge programs produce X3D-compliant scene graphs for interchange between procedural memory systems. Feedback should be directed to the PM-KR Community Group mailing list.*
