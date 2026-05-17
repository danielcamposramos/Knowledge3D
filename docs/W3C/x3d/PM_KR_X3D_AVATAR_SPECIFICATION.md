# PM-KR X3D Avatar Embodiment Specification

**Version**: 0.1 (Initial Draft)
**Status**: PM-KR Community Group Working Draft
**Date**: March 26, 2026
**Authors**: PM-KR Community Group (Daniel Campos Ramos, Chair; Milton Ponson, Co-Chair)
**Liaison**: Web3D Consortium (Don Brutzman, Advisory Committee Representative)
**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Reference Implementation)

**Normative References**:
- ISO/IEC 19775-1:2023 (X3D Architecture and Base Components, Version 4.0)
- ISO/IEC 19774:2019 (HAnim — Humanoid Animation, Version 2.0)
- ISO/IEC 19776-1 (X3D XML Encoding)
- ISO/IEC 19776-2 (X3D ClassicVRML Encoding)
- ISO/IEC 19776-3 (X3D JSON Encoding)
- PM-KR X3D Procedural Memory Component v0.1 (docs/w3c/x3d/PM_KR_X3D_PROCEDURAL_MEMORY_COMPONENT.md)
- PM-KR X3D RPN Scene Emission Specification v0.1 (docs/w3c/x3d/PM_KR_RPN_X3D_SCENE_EMISSION.md)
- Avatar Embodiment Specification v1.0 (docs/vocabulary/AVATAR_EMBODIMENT_SPECIFICATION.md)
- Three Brain System Specification v1.1 (docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md)
- Hyper-Parallel Processing Specification v1.0 (docs/vocabulary/HYPER_PARALLEL_PROCESSING.md)
- Dual-Client Contract Specification v1.1 (docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md)
- Spatial General Intelligence Specification v1.0 (docs/vocabulary/SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md)
- Memory Tablet Specification v1.0 (docs/vocabulary/MEMORY_TABLET_SPECIFICATION.md)
- Knowledgeverse Specification v5.1 (docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Scope and Design Rationale](#2-scope-and-design-rationale)
3. [Concepts](#3-concepts)
4. [Component Definition: AvatarEmbodiment](#4-component-definition-avatarembodiment)
5. [Abstract Node Types](#5-abstract-node-types)
6. [Concrete Node Reference](#6-concrete-node-reference)
7. [HAnim Integration](#7-hanim-integration)
8. [Cranial Galaxy Rendering](#8-cranial-galaxy-rendering)
9. [Dual-Client Avatar Texturing](#9-dual-client-avatar-texturing)
10. [Interaction Nodes](#10-interaction-nodes)
11. [Cognitive State Visualization](#11-cognitive-state-visualization)
12. [AvatarEmbodiment Profile Extension](#12-avatarembodiment-profile-extension)
13. [glTF Interoperability](#13-gltf-interoperability)
14. [Relationship to Existing X3D Components](#14-relationship-to-existing-x3d-components)
15. [Conformance](#15-conformance)
16. [Examples](#16-examples)

---

## 1. Introduction

### 1.1 Purpose

This document defines the **AvatarEmbodiment** component for X3D, extending the HAnim component (ISO/IEC 19774:2019) and the KnowledgeNavigation component (PM-KR X3D Procedural Memory Component §10) with node types for:

- **Embodied agent avatars** with HAnim-compliant skeletons that host live procedural knowledge workspaces inside their cranial volumes.
- **Unified human/AI body architecture** where both entity types share the same skeletal topology, dual-client surface contract, and interaction model.
- **Cranial Galaxy rendering** — the visualization of an agent's internal Galaxy Universe as a bounded 3D volume inside the skull segment of an HAnim Humanoid.
- **Cognitive state visualization** — external indicators of internal reasoning processes (specialist activation, navigation traces, convergence).
- **Spatial interaction primitives** — reach, grasp, hold, point, and share actions anchored to HAnim sites.

### 1.2 Motivation

The HAnim specification (ISO/IEC 19774:2019) defines a comprehensive humanoid animation framework with hierarchical joints, segments, sites, and displacers across four Levels of Articulation (LOA 0--4). HAnim humanoids can be animated, skinned, and rendered in X3D browsers. However, HAnim treats humanoids as **passive articulated geometry** — bodies that external systems animate.

PM-KR's Three Brain System paradigm (Three Brain System Specification §2) requires a fundamentally different relationship between body and mind: **the agent IS the body**. The TRM (Tiny Recursive Model) does not "control" an HAnim humanoid from outside; it lives inside the humanoid's skull as a Galaxy Universe, perceives through the humanoid's eyes (frustum cull from eye position), and acts through the humanoid's hands (spatial interaction at hand sites).

This specification bridges HAnim's mature body standard with PM-KR's embodied cognition model:

1. **HAnim provides the body.** Standard joint hierarchy (LOA-2 minimum, 71 joints), segment geometry, site markers, skin mesh with weighted deformation. All existing HAnim content, motion capture libraries, and animation tools remain compatible.

2. **PM-KR provides the mind.** The AgentEntity node (KnowledgeNavigation component, Level 3) hosts the agent's cognitive architecture. This specification defines how AgentEntity attaches to an HAnim Humanoid via three K3D extension joints and how the agent's internal Galaxy manifests as a renderable volume inside the skull.

3. **Dual-client texturing provides shared perception.** Every avatar surface carries both human-readable appearance (UV Map 0) and machine-readable semantic data (UV Map 1), ensuring that a human viewer and an AI agent looking at the same avatar agree on identity, position, and state.

### 1.3 Terminology

The keywords **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**, **SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **MAY**, and **OPTIONAL** in this document are to be interpreted as described in RFC 2119.

| Term | Definition | Source |
|------|-----------|--------|
| **Avatar** | An HAnim Humanoid inhabited by an entity (human or AI) in a PM-KR scene | This spec §3.1 |
| **Cranial Galaxy** | The Galaxy Universe spatially located inside an avatar's cranial volume | Avatar Embodiment Spec §4 |
| **Cranial Volume** | The bounded 3D region inside the skull segment above the skullbase joint | HAnim §skull segment |
| **Driver** | The entity (human input or TRM game loop) that produces joint transforms per frame | This spec §3.2 |
| **HAnim** | ISO/IEC 19774 Humanoid Animation standard for X3D | ISO/IEC 19774:2019 |
| **LOA** | Level of Articulation — HAnim's joint count classification (LOA 0--4) | HAnim §4 |
| **Site** | A named point on the body surface (HAnim HAnimSite) used for attachment | HAnim §7 |
| **Dual-Client** | Architecture where human and synthetic clients consume the same node | Dual-Client Contract §1.3 |
| **TRM** | Tiny Recursive Model (~7M params) — the autonomous cognitive entity | Three Brain System Spec §2 |
| **Game Loop** | Continuous perceive→navigate→reason→decide→act→learn cycle on GPU | Three Brain System Spec §3 |
| **Specialist** | A LoRA-like domain-biased adapter within the TRM swarm | Hyper-Parallel Processing Spec §1 |
| **Halting Gate** | Convergence checker (ternary: agree/disagree/uncertain) ending the reasoning loop | Sovereign NSI Spec §4.2 |

---

## 2. Scope and Design Rationale

### 2.1 What This Specification Defines

1. **AvatarEmbodiment component** with three support levels providing avatar body, cranial rendering, and interaction nodes.
2. **K3D joint extensions** to HAnim (k3d_cranial_origin, k3d_tablet_grip, k3d_thought_emitter) as HAnimJoint specializations.
3. **CranialGalaxy node** — a bounded rendering container that maps a GalaxyGroup into the skull segment's coordinate space.
4. **AvatarInteraction node** — spatial interaction primitives (reach, grasp, hold, point, share) anchored to HAnim sites.
5. **CognitiveStateIndicator node** — external visualization of internal agent state attached to the thought_emitter site.
6. **DualClientAvatarAppearance node** — the dual-texture contract applied to avatar skin and clothing meshes.
7. **Conformance criteria** for avatar producers and consumers.

### 2.2 What This Specification Does Not Define

- HAnim joint semantics, segment topology, or skin weighting — those are defined by ISO/IEC 19774:2019.
- AgentEntity, AgentSwarm, or AgentMemoryPalace node semantics — those are defined by PM-KR X3D Procedural Memory Component §10.
- Specific animation sequences, motion capture formats, or inverse kinematics algorithms.
- Network protocols for multi-user avatar synchronization (Doors protocol is out of scope for scene format).
- GPU kernel implementations or PTX compilation — those are sovereign runtime concerns.

### 2.3 Design Rationale: Extension, Not Replacement

This specification follows the X3D extension philosophy: **add, do not modify**. No existing HAnim node type is redefined. The K3D joints are new HAnimJoint instances with specific `name` values that HAnim-unaware browsers can render as standard joints. The CranialGalaxy is a new grouping node that HAnim-unaware browsers skip. Dual-client texturing adds a second texture set that browsers without PM-KR support simply do not sample.

This means:

- An existing HAnim browser renders a PM-KR avatar as a normal humanoid (body visible, cranial galaxy hidden, single-texture rendering). Full backward compatibility.
- A PM-KR-aware browser additionally renders the cranial galaxy, samples dual-client textures, and enables interaction primitives. Full forward capability.

---

## 3. Concepts

### 3.1 One Body, Two Drivers

Every avatar in a PM-KR scene uses the same body definition regardless of whether a human or an AI inhabits it:

```
HAnim Humanoid (LOA-2 minimum)
├── Joint Hierarchy (71+ joints, standard HAnim names)
│   └── K3D Extension Joints (3 additional)
├── Skin Mesh (weighted to joints)
│   └── Dual-Client Texturing (UV Map 0 + UV Map 1)
├── Segments (body parts between joints)
│   └── skull segment contains Cranial Volume
├── Sites (named surface points)
│   └── Interaction anchors (hand tips, skull vertex)
└── Metadata (PM-KR identity: canonicalId, entityType)
```

**Human-driven avatar**: Human input (keyboard, mouse, VR controller) produces joint transforms each frame. Perception originates from the camera placed between `l_eyeball_joint` and `r_eyeball_joint`. The cranial volume is empty.

**AI-driven avatar**: The TRM game loop (`trm_step_fused.ptx`) produces joint transforms each frame as procedural animation computed from cognitive state. Perception originates from frustum culling at the avatar's position and `skullbase` orientation. The cranial volume contains a live CranialGalaxy.

The body does not know which driver controls it. Any animation designed for one avatar type works on the other. Any interaction mechanism works between any combination of avatar types.

### 3.2 Cranial Galaxy as Renderable Volume

The defining architectural innovation of PM-KR avatars is that the AI's internal cognitive workspace — the Galaxy Universe (Knowledgeverse Spec §2.1) — is a real, renderable 3D volume located inside the avatar's skull. It is NOT metadata attached to a node. It is NOT a separate scene. It is geometry INSIDE the skull segment, positioned relative to `k3d_cranial_origin`, scaled to fit within the cranial bounding box.

This has concrete X3D implications:

- The CranialGalaxy node is a **bounded grouping node** (like BoundedPhysicsModel or GeoLOD) whose children (GalaxyGroup from ProceduralMemory component) are transformed into skull-local coordinates.
- The skull segment mesh can be rendered with variable transparency (opaque by default, translucent in diagnostic mode, invisible in teaching mode) to control cranial galaxy visibility.
- Knowledge stars inside the cranial galaxy are standard PM-KR ProceduralFormNode instances, rendered as small luminous particles.
- Navigation traces (LED-A* paths) inside the cranial galaxy are IndexedLineSet geometry connecting star positions.

### 3.3 Dual-Client Avatar Perception

Per the Dual-Client Contract (§2.3), every visible surface on the avatar carries two texture layers:

- **UV Map 0 (Human-readable)**: Standard visual appearance — skin, clothing, facial features, material properties. This is what X3D browsers render by default.
- **UV Map 1 (Machine-readable)**: Semantic data — joint weight visualization, segment boundaries, interaction zone masks, expression blend weights, gaze direction encoding. This is what AI agents decode when perceiving another avatar.

Both maps derive from the same underlying procedural source (the avatar IS a PM-KR node). The human sees the rendered form; the AI reads the encoded meaning. Both agree on identity, position, and state.

### 3.4 Interaction as Spatial Primitives

Avatar interaction in PM-KR is fundamentally spatial: entities interact by moving through 3D space, reaching toward objects, grasping them, holding them, pointing at locations, and sharing held objects. These are NOT UI abstractions. They are physical actions in the HAnim joint space:

- **Reach**: IK chain from shoulder through elbow to wrist extends hand toward target position.
- **Grasp**: Hand joints close around object at interaction site; physics constraint binds object to hand.
- **Hold**: Object remains attached to hand site (k3d_tablet_grip or hand tip) across frames.
- **Point**: Arm extends, index finger (if LOA ≥ 3) or hand (LOA-2) orients toward target.
- **Share**: One avatar presents held object toward another avatar's interaction radius.

---

## 4. Component Definition: AvatarEmbodiment

### 4.1 Component Name

`AvatarEmbodiment`

### 4.2 Component Overview

The AvatarEmbodiment component provides node types for representing PM-KR-aware humanoid avatars in X3D scenes. It extends the HAnim component with cognitive embodiment (cranial galaxy), dual-client surface rendering, spatial interaction primitives, and cognitive state visualization.

### 4.3 Component Levels

**Table 4.1 — AvatarEmbodiment component support levels**

| Level | Prerequisites | Nodes Added | Description |
|-------|--------------|-------------|-------------|
| 1 | HAnim:1, ProceduralMemory:1, Core:1, Grouping:1 | AvatarBody, CranialGalaxy, K3DJointExtension | Basic avatar body with cranial galaxy and K3D extension joints |
| 2 | HAnim:1, ProceduralMemory:2, Core:1, Grouping:1, Texturing:2 | DualClientAvatarAppearance, AvatarInteraction, HeldObject | Dual-client surface rendering and spatial interaction primitives |
| 3 | HAnim:1, KnowledgeNavigation:3, ProceduralMemory:2, Core:1, Grouping:2 | CognitiveStateIndicator, SpecialistVisualization, NavigationTrace | Cognitive state rendering and specialist activity visualization |

---

## 5. Abstract Node Types

### 5.1 X3DAvatarNode

Base abstract type for all PM-KR avatar nodes. Extends X3DChildNode and X3DBoundedObject.

```
X3DAvatarNode : X3DChildNode, X3DBoundedObject {
  SFString [in,out] canonicalId     ""           # Stable PM-KR entity identifier
  SFString [in,out] entityType      "human"      # "human" | "ai_trm" | "ai_assistant" | "ai_service"
  SFNode   [in,out] humanoid        NULL         [HAnimHumanoid]    # The HAnim body
  SFNode   [in,out] metadata        NULL         [X3DMetadataObject]
  SFBool   [in,out] bboxDisplay     FALSE
  SFBool   [in,out] visible         TRUE
  SFVec3f  []       bboxCenter      0 0 0        (-inf,inf)
  SFVec3f  []       bboxSize        -1 -1 -1     [0,inf) or -1 -1 -1
}
```

**Behavioral rules:**

1. The `canonicalId` field MUST be unique within the scene. It identifies this entity for PM-KR queries, canonical references, and persistent storage.

2. The `entityType` field determines driver expectations:
   - `"human"`: Driven by external input (keyboard, VR controller). No cranial galaxy.
   - `"ai_trm"`: Driven by TRM game loop. Cranial galaxy REQUIRED.
   - `"ai_assistant"`: Simplified AI (subset of TRM). Cranial galaxy OPTIONAL.
   - `"ai_service"`: Headless service entity (no visible avatar, present for interaction routing).

3. The `humanoid` field MUST reference a valid HAnimHumanoid node at LOA-2 or higher.

### 5.2 X3DCognitiveNode

Abstract type for nodes that represent internal cognitive state visualization.

```
X3DCognitiveNode : X3DChildNode {
  SFNode   [in,out] avatar          NULL         [X3DAvatarNode]    # Parent avatar
  SFBool   [in,out] active          TRUE         # Whether this visualization is currently rendering
  SFFloat  [in,out] opacity         0.0          [0,1]              # Rendering opacity (0 = hidden)
  SFNode   [in,out] metadata        NULL         [X3DMetadataObject]
}
```

**Behavioral rules:**

1. Cognitive nodes are only meaningful for avatars with `entityType` containing `"ai"`. For human avatars, cognitive nodes are present but MUST render with `opacity` 0.0 (maintaining structural parity without visual artifacts).

2. The `opacity` field controls the visibility of the cognitive visualization. Default is 0.0 (hidden) to preserve cognitive privacy. It is set to non-zero values under specific conditions (diagnostic mode, teaching mode, self-inspection).

---

## 6. Concrete Node Reference

### 6.1 AvatarBody (Level 1)

The primary avatar node. Wraps an HAnimHumanoid with PM-KR metadata, cranial galaxy reference, and interaction capabilities.

```
AvatarBody : X3DAvatarNode {
  SFString [in,out] canonicalId     ""
  SFString [in,out] entityType      "human"
  SFNode   [in,out] humanoid        NULL         [HAnimHumanoid]
  SFNode   [in,out] agent           NULL         [AgentEntity]      # PM-KR agent (AI only)
  SFNode   [in,out] cranialGalaxy   NULL         [CranialGalaxy]    # Internal brain (AI only)
  SFNode   [in,out] dualAppearance  NULL         [DualClientAvatarAppearance]
  MFNode   [in,out] heldObjects     []           [HeldObject]       # Currently held items
  MFNode   [in,out] interactions    []           [AvatarInteraction] # Active interactions
  SFNode   [in,out] cognitiveState  NULL         [CognitiveStateIndicator]
  SFString [in,out] homeHouse       ""           # Persistent House location URI
  SFInt32  [in,out] specialistCount 0            [0,64]             # Active specialist adapters
  SFString [in,out] currentState    "idle"       # "idle"|"perceiving"|"navigating"|"reasoning"|"converged"|"acting"|"teaching"|"sleeping"
  SFNode   [in,out] metadata        NULL         [X3DMetadataObject]
  SFBool   [in,out] bboxDisplay     FALSE
  SFBool   [in,out] visible         TRUE
  SFVec3f  []       bboxCenter      0 0 0        (-inf,inf)
  SFVec3f  []       bboxSize        -1 -1 -1     [0,inf) or -1 -1 -1
}
```

**Behavioral rules:**

1. When `entityType` is `"ai_trm"`, the `agent` field MUST reference a valid AgentEntity node (PM-KR X3D Procedural Memory Component §10.9). The agent's `body` field SHOULD reference back to this AvatarBody's `humanoid` node, establishing the bidirectional link between cognitive architecture and physical body.

2. When `entityType` is `"ai_trm"`, the `cranialGalaxy` field MUST be present and non-NULL. The CranialGalaxy node MUST be spatially contained within the skull segment of the referenced HAnimHumanoid.

3. When `entityType` is `"human"`, the `agent` and `cranialGalaxy` fields MUST be NULL. Human consciousness is external to the scene graph.

4. The `currentState` field reflects the TRM game loop phase (for AI avatars) or user activity state (for human avatars). State changes drive procedural animation selection — each state maps to a body animation pattern (see §11).

5. `heldObjects` contains objects currently attached to the avatar's hand sites. The Memory Tablet (Memory Tablet Specification) is the canonical held object but tools, books, and other graspable House objects are valid entries.

6. The `humanoid` node MUST contain the three K3D extension joints defined in §7.2, present at all LOA levels regardless of `entityType`. This maintains skeletal parity between avatar types.

### 6.2 CranialGalaxy (Level 1)

A bounded rendering container that maps an agent's GalaxyGroup into the skull segment coordinate space.

```
CranialGalaxy : X3DGroupingNode {
  MFNode   [in]     addChildren                  [X3DChildNode]
  MFNode   [in]     removeChildren               [X3DChildNode]
  MFNode   [in,out] children        []           [X3DChildNode]   # GalaxyGroup nodes
  SFVec3f  [in,out] cranialOrigin   0 0 0        (-inf,inf)      # Position of k3d_cranial_origin
  SFFloat  [in,out] cranialRadius   0.1          (0,inf)          # Skull bounding radius in meters
  SFFloat  [in,out] galaxyExtent    200.0        (0,inf)          # Galaxy coordinate extent (units)
  SFFloat  [in,out] scaleFactor     0.0005       (0,inf)          # galaxyExtent → cranialRadius
  SFInt32  [in,out] galaxyCount     0            [0,inf)          # Number of loaded galaxies
  SFInt32  [in,out] entryCount      0            [0,inf)          # Total knowledge entries
  SFString [in,out] visibilityMode  "hidden"     # "hidden"|"self"|"diagnostic"|"teaching"
  SFFloat  [in,out] skullTransparency 1.0        [0,1]            # Skull mesh transparency override
  SFNode   [in,out] starAppearance  NULL         [X3DAppearanceNode] # Default appearance for stars
  SFNode   [in,out] traceAppearance NULL         [X3DAppearanceNode] # Appearance for navigation traces
  SFNode   [in,out] metadata        NULL         [X3DMetadataObject]
  SFBool   [in,out] bboxDisplay     FALSE
  SFBool   [in,out] visible         TRUE
  SFVec3f  []       bboxCenter      0 0 0        (-inf,inf)
  SFVec3f  []       bboxSize        -1 -1 -1     [0,inf) or -1 -1 -1
}
```

**Behavioral rules:**

1. The CranialGalaxy MUST be a descendant of the skull segment in the HAnimHumanoid's joint/segment hierarchy. Its world-space bounding box MUST NOT extend outside the skull segment's bounding geometry.

2. The coordinate mapping from Galaxy space to skull-local space is:

   ```
   skull_local_position = cranialOrigin + scaleFactor * galaxy_position
   ```

   where `scaleFactor = cranialRadius / galaxyExtent`. Default: 0.1m / 200 units = 0.0005 m/unit. A knowledge star at Galaxy position (100, 50, −30) maps to skull-local position (0.05, 0.025, −0.015) meters relative to `k3d_cranial_origin`.

3. `children` SHOULD contain GalaxyGroup nodes (PM-KR X3D Procedural Memory Component §6.12). Each GalaxyGroup represents one loaded Galaxy (Math, Grammar, Drawing, etc.). The children of GalaxyGroup nodes are standard PM-KR procedural nodes, rendered as luminous particles at their spatial positions.

4. `visibilityMode` controls rendering:
   - `"hidden"`: Children are not rendered. Skull mesh is opaque. Default for privacy.
   - `"self"`: Children are rendered only for the owning agent's perception pipeline. Other entities see opaque skull.
   - `"diagnostic"`: Children are rendered for all authorized observers. Skull mesh transparency set to `skullTransparency`.
   - `"teaching"`: Children are rendered for all entities within interaction radius. Skull mesh fully transparent.

5. `skullTransparency` overrides the skull segment mesh material's transparency when `visibilityMode` is `"diagnostic"` or `"teaching"`. Value of 0.0 means fully opaque (hidden), 1.0 means fully transparent (visible through skull).

6. `starAppearance` provides the default rendering appearance for knowledge stars (ProceduralFormNode instances) inside the cranial galaxy when no per-star appearance is specified. Conforming implementations SHOULD render stars as small luminous point sprites or billboarded quads.

7. When the Galaxy Universe is live (agent is active), stars SHOULD be rendered with brightness proportional to their activation level. Stars currently being navigated (part of the active LED-A* path) SHOULD be rendered brighter than inactive stars.

### 6.3 K3DJointExtension (Level 1)

Three K3D-specific joints added to the HAnim skeleton. These are standard HAnimJoint instances with reserved `name` values.

```
K3DJointExtension : HAnimJoint {
  # Inherits all HAnimJoint fields
  # The 'name' field MUST be one of the three K3D-reserved names
  SFString [in,out] name            ""           # MUST be "k3d_cranial_origin" | "k3d_tablet_grip" | "k3d_thought_emitter"
  SFString [in,out] k3dPurpose      ""           # Purpose description for non-K3D browsers
}
```

**Joint definitions:**

**`k3d_cranial_origin`** (child of `skullbase`)
- Position: Center of cranial volume (approximately at geometric center of brain, ~4cm above and ~2cm anterior to skullbase)
- Purpose: Origin point for CranialGalaxy coordinate system. The Galaxy Universe's (0,0,0) maps to this joint's world position.
- For human avatars: Present but no CranialGalaxy attached (maintains skeletal parity).
- Transform: Identity rotation. Translation only (offsets from skullbase to brain center).

**`k3d_tablet_grip`** (child of `l_radiocarpal` or `r_radiocarpal`)
- Position: Palm center, oriented for tablet holding (screen facing away from palm).
- Purpose: Memory Tablet attachment point. The Tablet's local origin snaps to this joint's world transform.
- Default hand: Left (`l_radiocarpal`). Configurable per avatar preference.
- Transform: Rotation to orient a planar object (the Tablet) parallel to the palm surface with normal pointing away from the palm.

**`k3d_thought_emitter`** (child of `skullbase`)
- Position: 15cm above `skull_vertex` site (i.e., above the head).
- Purpose: Mounting point for CognitiveStateIndicator (§6.6). Visual indicator of internal cognitive state.
- For human avatars: Optional status indicator (e.g., "typing", "away").
- Transform: Translation only (15cm +Y from skull_vertex).

**Backward compatibility:** An HAnim browser that does not recognize K3D joint names MUST treat them as standard HAnimJoint nodes. They appear in the joint hierarchy, participate in skinning, but have no special behavior. This is the standard HAnim extensibility mechanism — unknown joint names are not errors.

### 6.4 DualClientAvatarAppearance (Level 2)

Dual-texture contract applied to avatar skin, clothing, and accessory meshes.

```
DualClientAvatarAppearance : X3DAppearanceNode {
  SFNode   [in,out] humanAppearance   NULL       [X3DAppearanceNode]  # UV Map 0: human-readable
  SFNode   [in,out] machineAppearance NULL       [X3DAppearanceNode]  # UV Map 1: machine-readable
  SFString [in,out] textureChannel0   "TEXCOORD_0"  # glTF attribute for human texture
  SFString [in,out] textureChannel1   "TEXCOORD_1"  # glTF attribute for machine texture
  SFNode   [in,out] metadata          NULL       [X3DMetadataObject]
}
```

**Behavioral rules:**

1. `humanAppearance` provides the standard visual appearance for human viewers. This MUST contain an Appearance node with Material and ImageTexture (or equivalent) using UV Map 0 coordinates. Resolution SHOULD be 512×512 or higher per segment.

2. `machineAppearance` provides the semantic data layer for AI consumers. This MUST contain an Appearance node with ImageTexture (or equivalent) using UV Map 1 coordinates. Resolution SHOULD be 256×256 per segment. The texture encodes:
   - **R channel**: Joint weight index (which joint most influences this texel)
   - **G channel**: Segment boundary (hard mask identifying body part)
   - **B channel**: Interaction zone (mask identifying interactive regions — hand, face, etc.)
   - **A channel**: State encoding (expression blend weight, speech phoneme, gaze angle)

3. A conforming PM-KR browser MUST sample both texture channels during rendering. A non-PM-KR browser SHOULD fall back to `humanAppearance` only. This fallback is automatic if the browser uses only TEXCOORD_0.

4. **Guaranteed identity** (Dual-Client Contract §2.1): Both appearances MUST be derived from the same canonical avatar source. The human appearance is the visual rendering of the same procedural body model that the machine appearance semantically encodes.

### 6.5 AvatarInteraction (Level 2)

A spatial interaction primitive anchored to HAnim sites.

```
AvatarInteraction : X3DChildNode {
  SFString [in,out] interactionType  ""          # "reach"|"grasp"|"hold"|"point"|"share"|"release"|"use"
  SFNode   [in,out] sourceAvatar     NULL        [AvatarBody]        # Avatar performing the action
  SFNode   [in,out] sourceSite       NULL        [HAnimSite]         # HAnim site anchor for action origin
  SFVec3f  [in,out] targetPosition   0 0 0       (-inf,inf)          # World-space target of action
  SFNode   [in,out] targetObject     NULL        [X3DChildNode]      # Scene object being interacted with
  SFNode   [in,out] targetAvatar     NULL        [AvatarBody]        # Other avatar (for share/communicate)
  SFFloat  [in,out] interactionRadius 3.0        (0,inf)             # Maximum distance for interaction (meters)
  SFBool   [in,out] isActive         FALSE       # TRUE while interaction is in progress
  SFFloat  [in,out] progress         0.0         [0,1]               # Completion progress (0 = started, 1 = done)
  SFTime   [in,out] startTime        0           # When interaction began
  SFNode   [in,out] metadata         NULL        [X3DMetadataObject]
}
```

**Behavioral rules:**

1. Interaction types map to HAnim joint chains and animation patterns:

   | Type | Joint Chain | Description |
   |------|------------|-------------|
   | `"reach"` | shoulder → elbow → wrist | Extend arm toward `targetPosition` via IK |
   | `"grasp"` | wrist → finger joints | Close hand around `targetObject` at interaction site |
   | `"hold"` | (static) | Object attached to hand site; persists across frames |
   | `"point"` | shoulder → elbow → wrist → (finger if LOA ≥ 3) | Arm extends, digit orients toward `targetPosition` |
   | `"share"` | (hold) + orientation toward `targetAvatar` | Present held object facing another avatar |
   | `"release"` | finger joints open | Detach object from hand site |
   | `"use"` | action-specific | Trigger object-specific behavior (open book, activate tool) |

2. Both human and AI avatars use the same interaction types with the same joint chains. The driver (human input vs. TRM game loop) produces different IK targets, but the skeletal mechanics are identical. This is the Interaction Symmetry Invariant (Avatar Embodiment Spec §12.4).

3. `interactionRadius` defines the maximum distance between `sourceSite` and `targetPosition` for the interaction to be valid. Objects beyond this radius cannot be reached. Default is 3.0 meters (approximate arm's reach plus one step for a standard humanoid).

4. When `isActive` is TRUE, the avatar's animation system SHOULD produce joint transforms that move the joint chain toward the interaction target. The `progress` field tracks completion (0 = just started, 1 = fully completed).

### 6.6 HeldObject (Level 2)

An object currently held by an avatar, attached to a hand site.

```
HeldObject : X3DChildNode {
  SFNode   [in,out] object          NULL         [X3DChildNode]     # The held scene object
  SFNode   [in,out] gripSite        NULL         [HAnimSite]        # Hand site to attach to
  SFString [in,out] gripType        "palm"       # "palm"|"pinch"|"wrap"|"rest"
  SFVec3f  [in,out] localOffset     0 0 0        (-inf,inf)         # Offset from grip site
  SFRotation [in,out] localRotation 0 0 1 0      # Rotation relative to grip site
  SFBool   [in,out] isHeld          TRUE         # FALSE = released, pending removal
  SFNode   [in,out] metadata        NULL         [X3DMetadataObject]
}
```

**Behavioral rules:**

1. When `isHeld` is TRUE, the `object` node's world transform MUST be computed as: `gripSite.worldTransform * Translation(localOffset) * Rotation(localRotation)`. The object moves with the hand.

2. The canonical HeldObject is the Memory Tablet (Memory Tablet Specification). The Tablet attaches to `k3d_tablet_grip` with `gripType="palm"`, screen normal pointing away from palm.

3. When an avatar holds a book (House object), the book's content Galaxy MAY be loaded into the agent's CranialGalaxy working memory. The HeldObject's `object` field references the book's scene graph node; the content loading is a runtime behavior triggered by the `"use"` AvatarInteraction.

### 6.7 CognitiveStateIndicator (Level 3)

Visual representation of an agent's internal cognitive state, mounted at `k3d_thought_emitter`.

```
CognitiveStateIndicator : X3DCognitiveNode {
  SFNode   [in,out] avatar          NULL         [AvatarBody]
  SFBool   [in,out] active          TRUE
  SFFloat  [in,out] opacity         0.0          [0,1]
  SFString [in,out] currentPhase    "idle"       # "idle"|"perceiving"|"navigating"|"reasoning"|"converged"|"acting"|"teaching"|"sleeping"
  SFColor  [in,out] phaseColor      1 1 1        [0,1]              # Color of indicator for current phase
  SFFloat  [in,out] intensity       0.0          [0,1]              # Brightness/activity level
  SFNode   [in,out] idleAppearance        NULL   [X3DAppearanceNode]  # Dim steady glow
  SFNode   [in,out] perceivingAppearance  NULL   [X3DAppearanceNode]  # Brief pulse
  SFNode   [in,out] navigatingAppearance  NULL   [X3DAppearanceNode]  # Orbiting particles
  SFNode   [in,out] reasoningAppearance   NULL   [X3DAppearanceNode]  # Active multi-color glow
  SFNode   [in,out] convergedAppearance   NULL   [X3DAppearanceNode]  # Single bright flash
  SFNode   [in,out] actingAppearance      NULL   [X3DAppearanceNode]  # Directional beam
  SFNode   [in,out] teachingAppearance    NULL   [X3DAppearanceNode]  # Expanding ring
  SFNode   [in,out] sleepingAppearance    NULL   [X3DAppearanceNode]  # Slow pulsing glow
  SFNode   [in,out] metadata        NULL         [X3DMetadataObject]
}
```

**Behavioral rules:**

1. The indicator is positioned at `k3d_thought_emitter` (15cm above skull vertex). When `opacity` > 0, the appropriate appearance node for `currentPhase` is rendered at that position.

2. The `currentPhase` field MUST match the parent AvatarBody's `currentState` field. When the agent's TRM game loop transitions phase, both fields update simultaneously.

3. Phase-to-appearance mapping:

   | Phase | Default Visual | Semantic Meaning |
   |-------|---------------|-----------------|
   | `idle` | Dim steady glow (warm white) | Agent is awake but not actively reasoning |
   | `perceiving` | Brief outward pulse | Frustum cull / Morton query in progress |
   | `navigating` | Orbiting particles (3-5 dots) | LED-A* pathfinding through Galaxy |
   | `reasoning` | Multi-color glow (specialist colors) | Nine-Chain Swarm processing |
   | `converged` | Single bright flash (white) | Halting Gate converged — answer found |
   | `acting` | Directional beam toward target | Agent performing spatial action |
   | `teaching` | Expanding ring animation | Agent sharing reasoning (cranial galaxy visible) |
   | `sleeping` | Slow pulsing glow (blue) | Sleep-time consolidation in progress |

4. If no per-phase appearance is provided, the implementation SHOULD generate a default particle effect at `k3d_thought_emitter` position using `phaseColor` and `intensity`.

### 6.8 SpecialistVisualization (Level 3)

Renders the Nine-Chain Swarm workers as visible elements inside the CranialGalaxy.

```
SpecialistVisualization : X3DCognitiveNode {
  SFNode   [in,out] avatar          NULL         [AvatarBody]
  SFBool   [in,out] active          TRUE
  SFFloat  [in,out] opacity         0.0          [0,1]
  SFNode   [in,out] swarm           NULL         [AgentSwarm]       # PM-KR swarm reference
  SFInt32  [in,out] workerCount     9            [1,64]
  MFString [in,out] workerNames     []           # ["math","grammar","visual","chat","physics","logic","spatial","temporal","meta"]
  MFColor  [in,out] workerColors    []           # Domain-coded colors (one per worker)
  MFFloat  [in,out] workerActivation []          [0,1]              # Per-worker activation level
  SFFloat  [in,out] orbitRadius     0.03         (0,inf)            # Orbit radius around TRM core (skull-local meters)
  SFFloat  [in,out] orbitSpeed      1.0          (0,inf)            # Orbits per second (when active)
  SFNode   [in,out] coreAppearance  NULL         [X3DAppearanceNode] # TRM core (center point)
  SFNode   [in,out] workerAppearance NULL        [X3DAppearanceNode] # Default worker point
  SFNode   [in,out] metadata        NULL         [X3DMetadataObject]
}
```

**Behavioral rules:**

1. Workers are rendered as small luminous particles orbiting the TRM core position (which coincides with `k3d_cranial_origin`). Each worker's position at time *t* follows a Lissajous orbit pattern unique to its index, ensuring visual distinctness.

2. `workerActivation` is a per-worker float array (length = `workerCount`). Each value indicates how actively that specialist is contributing to the current reasoning task:
   - 0.0: Idle (dim or invisible)
   - 0.5: Background processing (moderate glow)
   - 1.0: Primary contributor (bright, faster orbit)

3. `workerColors` provides domain-coded colors. Default assignment:

   | Index | Name | Default Color | RGB |
   |-------|------|--------------|-----|
   | 0 | math | Blue | (0.2, 0.4, 1.0) |
   | 1 | grammar | Green | (0.2, 0.8, 0.3) |
   | 2 | visual | Red | (1.0, 0.3, 0.2) |
   | 3 | chat | Yellow | (1.0, 0.9, 0.2) |
   | 4 | physics | Cyan | (0.2, 0.9, 0.9) |
   | 5 | logic | Purple | (0.7, 0.2, 0.9) |
   | 6 | spatial | Orange | (1.0, 0.6, 0.1) |
   | 7 | temporal | Pink | (0.9, 0.4, 0.6) |
   | 8 | meta (Jarvis) | White | (1.0, 1.0, 1.0) |

4. The SpecialistVisualization is only rendered when the parent CranialGalaxy's `visibilityMode` is not `"hidden"` AND `opacity` > 0. It respects the same visibility constraints as the cranial galaxy itself.

### 6.9 NavigationTrace (Level 3)

Renders LED-A* pathfinding traces through the CranialGalaxy as glowing line geometry.

```
NavigationTrace : X3DCognitiveNode {
  SFNode   [in,out] avatar          NULL         [AvatarBody]
  SFBool   [in,out] active          TRUE
  SFFloat  [in,out] opacity         0.0          [0,1]
  MFVec3f  [in,out] pathNodes       []           (-inf,inf)         # Galaxy-space positions along path
  MFFloat  [in,out] nodeFocus       []           [0,1]              # LED focus value per node (brightness)
  SFColor  [in,out] traceColor      0.3 0.8 1.0  [0,1]             # Trace line color
  SFFloat  [in,out] traceWidth      2.0          (0,inf)            # Line width in pixels
  SFFloat  [in,out] fadeTime        3.0          (0,inf)            # Seconds before completed trace fades
  SFBool   [in,out] showDeadEnds    FALSE        # Render abandoned path branches
  SFColor  [in,out] deadEndColor    0.5 0.2 0.2  [0,1]             # Color for dead-end branches
  SFNode   [in,out] seedAppearance  NULL         [X3DAppearanceNode] # Navigation origin point
  SFNode   [in,out] focusAppearance NULL         [X3DAppearanceNode] # Destination (brightest) point
  SFNode   [in,out] metadata        NULL         [X3DMetadataObject]
}
```

**Behavioral rules:**

1. `pathNodes` contains Galaxy-space positions that form the navigation path. These are transformed to skull-local coordinates via the parent CranialGalaxy's coordinate mapping before rendering.

2. The path is rendered as an IndexedLineSet (or equivalent) connecting consecutive path nodes. Each node's brightness is modulated by `nodeFocus[i]`: the LED-A* focus value where 1.0 indicates the destination node and values decrease with path distance.

3. The first node in `pathNodes` (seed) is rendered with `seedAppearance`. The last node (focus) is rendered with `focusAppearance`. Interior nodes are rendered as progressively brightening points along the trace.

4. When a navigation path completes (agent reaches convergence), the trace SHOULD fade over `fadeTime` seconds. During the fade, `opacity` decreases linearly from its current value to 0.

5. If `showDeadEnds` is TRUE, path branches that were explored but abandoned (LED-A* pruned paths) are rendered in `deadEndColor` with reduced width. This is useful in teaching mode to show students why the agent rejected certain reasoning paths.

---

## 7. HAnim Integration

### 7.1 Humanoid Requirements

A PM-KR avatar's HAnimHumanoid MUST satisfy:

1. **LOA-2 minimum** (71 joints). This provides sufficient articulation for spine, limbs, neck, and head. Higher LOAs (3 = finger articulation, 4 = facial muscle groups) are RECOMMENDED for expressive interaction.

2. **Standard HAnim joint names** per ISO/IEC 19774:2019 Annex A. The K3D extension joints (§6.3) use the `k3d_` prefix to avoid collision with current and future HAnim standard names.

3. **Skin mesh** with weighted vertex deformation (HAnimHumanoid `skin` field). The skin mesh carries the dual-client textures (§9).

4. **Sites** at minimum: `skull_vertex`, `l_hand_tip`, `r_hand_tip`, `navel`. Additional HAnim standard sites are RECOMMENDED.

### 7.2 K3D Extension Joint Placement

The three K3D joints are inserted into the standard HAnim hierarchy at specific locations:

```xml
<HAnimJoint DEF='SKULLBASE' name='skullbase'>
  <!-- Standard HAnim children -->
  <HAnimJoint DEF='L_EYEBALL' name='l_eyeball_joint'/>
  <HAnimJoint DEF='R_EYEBALL' name='r_eyeball_joint'/>
  <HAnimJoint DEF='JAW' name='temporomandibular'/>

  <!-- K3D extensions -->
  <HAnimJoint DEF='K3D_CRANIAL' name='k3d_cranial_origin'
    center='0.0 0.04 -0.02'
    k3dPurpose='Galaxy Universe origin inside skull'>
    <!-- CranialGalaxy attaches here for AI avatars -->
  </HAnimJoint>

  <HAnimJoint DEF='K3D_EMITTER' name='k3d_thought_emitter'
    center='0.0 0.25 0.0'
    k3dPurpose='Cognitive state indicator above head'>
    <!-- CognitiveStateIndicator attaches here -->
  </HAnimJoint>
</HAnimJoint>

<!-- In the hand chain: -->
<HAnimJoint DEF='L_WRIST' name='l_radiocarpal'>
  <HAnimJoint DEF='K3D_GRIP' name='k3d_tablet_grip'
    center='0.0 -0.03 0.0'
    k3dPurpose='Memory Tablet attachment on palm'>
    <!-- HeldObject (Tablet) attaches here -->
  </HAnimJoint>
  <!-- Standard HAnim finger joints follow -->
</HAnimJoint>
```

### 7.3 Segment-to-Galaxy Mapping

The skull segment in HAnim defines the cranial bounding volume. The CranialGalaxy's `cranialRadius` SHOULD be derived from the skull segment's bounding box:

```
skull_segment.bboxSize = (width, height, depth)
cranialRadius = min(width, height, depth) / 2.0 * 0.85
```

The 0.85 factor ensures the Galaxy stays within the skull boundary with margin for rendering artifacts.

---

## 8. Cranial Galaxy Rendering

### 8.1 Rendering Pipeline

When a CranialGalaxy is visible (visibilityMode ≠ "hidden"), the rendering pipeline adds these passes:

1. **Skull transparency pass**: Set skull segment mesh material transparency to `skullTransparency`. This makes the cranial volume visible through a translucent skull.

2. **Star rendering pass**: Each ProceduralFormNode child of the contained GalaxyGroup nodes is rendered as a luminous particle at its Galaxy-space position, transformed to skull-local coordinates via the CranialGalaxy's scale mapping.

3. **Trace rendering pass**: Active NavigationTrace nodes render their path geometry inside the cranial volume.

4. **Specialist rendering pass**: SpecialistVisualization renders orbiting worker particles around the TRM core position.

5. **Clipping pass**: All cranial content is clipped to the skull segment bounding sphere. Nothing renders outside the skull boundary (Cranial Containment Invariant — Avatar Embodiment Spec §12.3).

### 8.2 Star Rendering

Knowledge stars inside the CranialGalaxy SHOULD be rendered as:

- **Point sprites** (PointSet with per-point color/size): Most efficient for thousands of stars.
- **Billboarded quads** (Billboard node with textured quad): Better visual quality, higher cost.
- **Volumetric particles** (ParticleSystem): Best visual quality for small star counts.

Star size SHOULD be modulated by:
- **Activation level**: Stars in the active navigation path are larger.
- **Confidence**: Higher-confidence knowledge stars are brighter.
- **Recency**: Recently accessed stars have a brief glow-up effect.

### 8.3 Performance Considerations

A CranialGalaxy may contain hundreds of thousands of knowledge entries (e.g., 247,889 entries across 19 galaxies — see Phase D.3 report). Rendering all stars as individual nodes would overwhelm any scene graph.

Conforming implementations SHOULD:
- Use GPU instancing (one draw call for all stars, per-instance color/position/size).
- Apply LOD within the cranial galaxy: only render the N nearest or most active stars.
- Use the same frustum culling and LOD mechanisms that the agent itself uses for perception — the visualization IS the agent's cognitive field-of-view.

---

## 9. Dual-Client Avatar Texturing

### 9.1 UV Map Layout

Avatar meshes carry two UV coordinate sets:

**TEXCOORD_0 (UV Map 0) — Human Appearance:**

| Body Region | Texture Content | Minimum Resolution |
|-------------|----------------|-------------------|
| Head/Face | Skin tone, eyes, lips, brows, hair attachment | 512×512 |
| Torso | Skin or clothing material | 512×512 |
| Arms | Skin, sleeve transitions | 256×256 |
| Hands | Skin detail, nail | 256×256 |
| Legs | Skin or clothing material | 256×256 |
| Feet | Skin, shoe material | 256×256 |

**TEXCOORD_1 (UV Map 1) — Machine Semantic Data:**

| Body Region | Texture Content | Resolution |
|-------------|----------------|-----------|
| Head/Face | Expression blend weights (R), gaze direction (G), speech phoneme (B), segment mask (A) | 256×256 |
| Torso | Joint weight index (R), segment boundary (G), interaction zone (B), state (A) | 256×256 |
| Limbs | Joint weight index (R), segment boundary (G), interaction zone (B), state (A) | 128×128 |

### 9.2 Machine Texture Encoding

The machine-readable texture (UV Map 1) encodes per-texel semantic data in RGBA channels:

| Channel | Encoding | Range | Description |
|---------|----------|-------|-------------|
| R | Joint weight index | 0–255 | Index of the joint with highest influence weight at this texel |
| G | Segment boundary | 0/255 | 255 = segment boundary edge, 0 = interior |
| B | Interaction zone | 0–255 | Zone classification: 0=none, 1=hand, 2=face, 3=torso, 4=foot, 5=tablet_grip |
| A | State encoding | 0–255 | Context-dependent: facial region = expression blend, hand = grip state |

An AI agent perceiving another avatar decodes this texture to understand:
- Which body part it is looking at (segment boundary + joint index).
- Whether that part is an interaction target (interaction zone).
- What state that part is in (state encoding: is the hand open or gripping? is the face smiling or neutral?).

### 9.3 Identity Guarantee

Both UV maps derive from the same underlying avatar body model. The human appearance and machine semantic data MUST be consistent:
- If the human texture shows the left hand gripping an object, the machine texture's interaction zone MUST encode grip state for the corresponding texels.
- If the human texture shows the head turned 30° right, the machine texture's gaze direction encoding MUST reflect the same orientation.

---

## 10. Interaction Nodes

### 10.1 Interaction Flow

A complete avatar interaction follows this sequence:

```
1. PROXIMITY  → Entities within interactionRadius
2. ATTENTION  → Source avatar's skullbase orients toward target
3. REACH      → AvatarInteraction(type="reach") extends arm
4. GRASP      → AvatarInteraction(type="grasp") closes hand
5. HOLD       → HeldObject created, attached to grip site
6. USE/SHARE  → AvatarInteraction(type="use"|"share")
7. RELEASE    → AvatarInteraction(type="release") opens hand
```

### 10.2 Tablet Interaction Protocol

The Memory Tablet follows a specific interaction protocol:

```xml
<!-- Tablet pickup -->
<AvatarInteraction interactionType='grasp'
  sourceAvatar='TRM_AVATAR' sourceSite='L_HAND_TIP'
  targetObject='MEMORY_TABLET' isActive='true'/>

<!-- Tablet held -->
<HeldObject object='MEMORY_TABLET' gripSite='K3D_GRIP'
  gripType='palm' isHeld='true'/>

<!-- Show tablet to another avatar -->
<AvatarInteraction interactionType='share'
  sourceAvatar='TRM_AVATAR' sourceSite='K3D_GRIP'
  targetAvatar='HUMAN_AVATAR' isActive='true'/>
```

### 10.3 Book Interaction Protocol

When an avatar opens a book (a House artifact with `extras.k3d.contentManifest`):

1. **Reach + Grasp**: Avatar reaches to shelf, grasps book at book's interaction site.
2. **Hold**: Book attached to hand as HeldObject.
3. **Use (open)**: AvatarInteraction(type="use") triggers book open animation.
4. **Content load**: Book's content manifest triggers Galaxy loading into the agent's CranialGalaxy (for AI avatars) or visual content rendering on book pages (for human avatars).
5. **Release**: Book returns to shelf or is placed on a surface.

---

## 11. Cognitive State Visualization

### 11.1 State Machine

The AI avatar's cognitive state follows a strict state machine driven by the TRM game loop:

```
                    ┌──────────────┐
                    │   sleeping   │
                    └──────┬───────┘
                           │ wake
                    ┌──────▼───────┐
              ┌─────│     idle     │◄────────────────────┐
              │     └──────┬───────┘                      │
              │ query      │ query                        │ done
              │     ┌──────▼───────┐                      │
              │     │  perceiving  │                      │
              │     └──────┬───────┘                      │
              │            │                              │
              │     ┌──────▼───────┐                      │
              │     │  navigating  │                      │
              │     └──────┬───────┘                      │
              │            │                              │
              │     ┌──────▼───────┐  not converged       │
              │     │  reasoning   │──────────┐           │
              │     └──────┬───────┘          │           │
              │            │ converged  ┌─────▼────┐      │
              │     ┌──────▼───────┐    │ navigating│     │
              │     │  converged   │    │ (iterate) │     │
              │     └──────┬───────┘    └──────────┘      │
              │            │                              │
              │     ┌──────▼───────┐                      │
              └────►│   acting     │──────────────────────┘
                    └──────────────┘
```

### 11.2 External Indicators (CognitiveStateIndicator)

The thought emitter (above head) provides non-intrusive external state. Other entities can observe the indicator to know:
- Is the AI thinking? (navigating/reasoning phases)
- Has it found an answer? (converged flash)
- Is it available for interaction? (idle glow)
- Is it consolidating? (sleeping pulse)

This is the spatial equivalent of a "typing indicator" — but for cognition.

### 11.3 Internal Indicators (CranialGalaxy + SpecialistVisualization)

When cranial galaxy is visible, observers can see:
- Which Galaxy neighborhoods are active (bright clusters of stars)
- Which specialists are contributing (colored orbiting workers)
- What path the reasoning took (navigation traces between stars)
- When convergence occurred (traces merge to single point, flash)

This enables:
- **Teaching**: An AI can show a student how it solved a problem by making its cranial galaxy visible.
- **Debugging**: A developer can watch knowledge navigation in real-time.
- **AI-to-AI**: One agent can observe another's reasoning (with permission) for collaborative problem-solving.

---

## 12. AvatarEmbodiment Profile Extension

### 12.1 Profile Extension Name

`AvatarEmbodimentExtension`

This is an extension to the ProceduralMemoryInterchange profile (PM-KR X3D Procedural Memory Component §11), not a standalone profile.

### 12.2 Additional Component Requirements

**Table 12.1 — AvatarEmbodimentExtension additional components**

| Component | Minimum Level | Requirement | Rationale |
|-----------|--------------|-------------|-----------|
| AvatarEmbodiment | 1 | REQUIRED | Avatar body, cranial galaxy, K3D joints |
| HAnim | 1 | REQUIRED | Humanoid body standard |
| AvatarEmbodiment | 2 | RECOMMENDED | Dual-client appearance, interaction, held objects |
| AvatarEmbodiment | 3 | OPTIONAL | Cognitive visualization, specialist rendering, traces |
| PointingDeviceSensor | 1 | RECOMMENDED | Click/touch interaction with avatar surfaces |
| KeyDeviceSensor | 1 | RECOMMENDED | Keyboard input for human avatar driver |

### 12.3 Combined Profile Table

When combined with ProceduralMemoryInterchange, the full avatar-capable profile requires:

| Component | Level | Source |
|-----------|-------|--------|
| Core | 1 | X3D base |
| Grouping | 2 | X3D base |
| Shape | 2 | X3D base |
| Rendering | 3 | X3D base |
| Texturing | 2 | X3D base |
| Lighting | 2 | X3D base |
| Metadata | 1 | X3D base |
| ProceduralMemory | 2 | PM-KR |
| KnowledgeNavigation | 3 | PM-KR |
| HAnim | 1 | HAnim |
| AvatarEmbodiment | 1+ | This spec |

---

## 13. glTF Interoperability

### 13.1 glTF/GLB Export

PM-KR avatars export to glTF/GLB using standard glTF skin/animation plus K3D metadata in `extras`:

```json
{
  "nodes": [{
    "name": "TRM_Avatar",
    "skin": 0,
    "extras": {
      "k3d": {
        "nodeType": "AvatarBody",
        "canonicalId": "avatar:trm:primary",
        "entityType": "ai_trm",
        "hanim": {
          "loa": 2,
          "jointCount": 74,
          "k3dExtensions": [
            "k3d_cranial_origin",
            "k3d_tablet_grip",
            "k3d_thought_emitter"
          ]
        },
        "cranialGalaxy": {
          "cranialRadius": 0.1,
          "galaxyExtent": 200.0,
          "scaleFactor": 0.0005,
          "galaxyCount": 11,
          "entryCount": 247889,
          "visibilityMode": "hidden"
        },
        "dualClient": {
          "humanTexture": "TEXCOORD_0",
          "machineTexture": "TEXCOORD_1"
        },
        "currentState": "idle",
        "specialistCount": 9
      }
    }
  }],
  "skins": [{
    "joints": [0, 1, 2, "...71 HAnim joints + 3 K3D joints..."],
    "skeleton": 0,
    "inverseBindMatrices": 0
  }],
  "meshes": [{
    "primitives": [{
      "attributes": {
        "POSITION": 0,
        "NORMAL": 1,
        "TEXCOORD_0": 2,
        "TEXCOORD_1": 3,
        "JOINTS_0": 4,
        "WEIGHTS_0": 5
      }
    }]
  }]
}
```

### 13.2 Round-Trip Preservation

Per PM-KR X3D RPN Scene Emission Specification §14, glTF round-trip MUST preserve:
- All `extras.k3d` metadata (avatar identity, entity type, cranial galaxy config).
- Both TEXCOORD sets (dual-client textures).
- All skin joints including K3D extensions (joint names preserved in node names).
- Animation data for procedural body animation.

### 13.3 glTF Extensions

PM-KR avatars MAY use these glTF extensions:
- `KHR_materials_unlit` — for cranial galaxy star rendering (emissive particles).
- `KHR_texture_transform` — for UV Map 1 coordinate adjustments.
- `EXT_mesh_gpu_instancing` — for efficient cranial galaxy star rendering (thousands of instances).

---

## 14. Relationship to Existing X3D Components

### 14.1 HAnim Component (ISO/IEC 19774)

AvatarEmbodiment extends HAnim, it does not replace it. The relationship:

| HAnim Provides | AvatarEmbodiment Adds |
|---------------|----------------------|
| HAnimHumanoid (skeleton root) | AvatarBody (PM-KR wrapper with agent link) |
| HAnimJoint (71+ standard joints) | K3DJointExtension (3 additional joints) |
| HAnimSegment (body parts) | CranialGalaxy (content inside skull segment) |
| HAnimSite (surface points) | AvatarInteraction (interaction primitives at sites) |
| HAnimDisplacer (morph targets) | CognitiveStateIndicator (state-driven appearance) |
| Skin mesh (deformable geometry) | DualClientAvatarAppearance (two UV maps) |

### 14.2 KnowledgeNavigation Component (PM-KR)

AvatarEmbodiment uses KnowledgeNavigation's agent nodes:

| KnowledgeNavigation Provides | AvatarEmbodiment Uses It For |
|-----------------------------|------------------------------|
| AgentEntity | The cognitive architecture inside AvatarBody |
| AgentSwarm | Referenced by SpecialistVisualization for worker rendering |
| AgentMemoryPalace | The House where the avatar lives |
| SpatialIndex | Frustum culling and LOD from avatar position |
| KnowledgePath | LED-A* traces rendered by NavigationTrace |

### 14.3 ProceduralMemory Component (PM-KR)

AvatarEmbodiment renders ProceduralMemory nodes inside the cranial galaxy:

| ProceduralMemory Provides | AvatarEmbodiment Uses It For |
|--------------------------|------------------------------|
| GalaxyGroup | Children of CranialGalaxy (Galaxy domains) |
| ProceduralFormNode | Knowledge stars rendered as particles |
| DualClientTexture | Extended as DualClientAvatarAppearance for body surfaces |
| CanonicalReference | Avatar identity and cross-scene reference |

### 14.4 Navigation Component (X3D Base)

Standard X3D Navigation provides Viewpoint, NavigationInfo, LOD. AvatarEmbodiment uses these for:
- Camera placement between eyeball joints (Viewpoint bound to eye midpoint).
- LOD on avatar body (distant avatars render at lower detail).
- NavigationInfo for human avatar locomotion constraints.

---

## 15. Conformance

### 15.1 Conformance Levels

**Level 1 — Basic Avatar Producer:**
A system that produces X3D content containing AvatarBody nodes with HAnimHumanoid references and K3D extension joints MUST:
- Produce valid HAnimHumanoid at LOA-2 or higher.
- Include all three K3D extension joints with correct names and parent joints.
- Include CranialGalaxy for `entityType="ai_trm"` avatars.
- Preserve `canonicalId` across export/reimport.

**Level 2 — Dual-Client Avatar Producer:**
A Level 1 producer that additionally MUST:
- Produce DualClientAvatarAppearance with both UV maps.
- Encode machine texture per §9.2 channel specification.
- Include AvatarInteraction nodes for documented interaction types.
- Include HeldObject for Tablet attachment.

**Level 3 — Full Avatar Producer:**
A Level 2 producer that additionally MUST:
- Produce CognitiveStateIndicator with per-phase appearances.
- Produce SpecialistVisualization with per-worker colors and activation.
- Produce NavigationTrace for active reasoning paths.
- Support all four CranialGalaxy visibility modes.

**Level 1 — Basic Avatar Consumer:**
A system that consumes X3D content containing AvatarBody nodes MUST:
- Render the HAnimHumanoid body with standard HAnim skinning.
- Render K3D extension joints as standard joints (no special behavior required).
- Display `currentState` as metadata (no visualization required).

**Level 2 — Dual-Client Avatar Consumer:**
A Level 1 consumer that additionally MUST:
- Sample both TEXCOORD_0 and TEXCOORD_1 for PM-KR-aware rendering.
- Expose machine texture data via API for agent perception.
- Enable/disable AvatarInteraction based on proximity.

**Level 3 — Full Avatar Consumer:**
A Level 2 consumer that additionally MUST:
- Render CranialGalaxy with star particles, navigation traces, specialist orbits.
- Animate CognitiveStateIndicator per phase transitions.
- Support CranialGalaxy visibility mode transitions.
- Clip all cranial content to skull bounding volume.

### 15.2 Backward Compatibility

A conforming implementation MUST degrade gracefully when processing PM-KR avatar content:

| Browser Capability | Avatar Rendering |
|-------------------|-----------------|
| HAnim only (no PM-KR) | Body renders normally; K3D joints as standard joints; no cranial galaxy; single texture |
| HAnim + ProceduralMemory (no AvatarEmbodiment) | Body renders; GalaxyGroup nodes visible but not cranially bounded |
| Full PM-KR | Complete avatar rendering with cranial galaxy, dual textures, cognitive indicators |

---

## 16. Examples

### 16.1 Complete AI Avatar (X3D XML Encoding)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<X3D profile='ProceduralMemoryInterchange' version='4.0'>
  <head>
    <component name='AvatarEmbodiment' level='3'/>
    <component name='HAnim' level='1'/>
    <component name='KnowledgeNavigation' level='3'/>
    <meta name='title' content='TRM Avatar — Primary AI Entity'/>
    <meta name='creator' content='PM-KR Community Group'/>
  </head>
  <Scene>

    <!-- === HAnim Humanoid Body === -->
    <HAnimHumanoid DEF='TRM_HUMANOID' name='TRM' version='2.0' loa='2'
      info='"authorName" "PM-KR" "authorEmail" "pmkr@w3.org"'>

      <!-- Root joint -->
      <HAnimJoint DEF='HUMANOID_ROOT' name='humanoid_root' center='0 0.87 0'>
        <HAnimSegment DEF='SACRUM' name='sacrum'>
          <Shape>
            <Appearance><Material diffuseColor='0.8 0.7 0.6'/></Appearance>
            <IndexedFaceSet coordIndex='0 1 2 -1'>
              <Coordinate point='0 0 0, 0.1 0 0, 0 0.1 0'/>
              <TextureCoordinate point='0 0, 1 0, 0 1'/>
            </IndexedFaceSet>
          </Shape>
        </HAnimSegment>

        <!-- Spine chain (abbreviated) -->
        <HAnimJoint DEF='VL5' name='vl5' center='0 1.05 -0.01'>
          <HAnimJoint DEF='VL3' name='vl3' center='0 1.12 -0.01'>
            <HAnimJoint DEF='VL1' name='vl1' center='0 1.19 -0.01'>
              <HAnimJoint DEF='VT10' name='vt10' center='0 1.27 0.0'>
                <HAnimJoint DEF='VT6' name='vt6' center='0 1.35 0.01'>
                  <HAnimJoint DEF='VT1' name='vt1' center='0 1.44 0.02'>

                    <!-- Left arm chain -->
                    <HAnimJoint DEF='L_STERNOCLAVICULAR' name='l_sternoclavicular' center='0.08 1.44 0.0'>
                      <HAnimJoint DEF='L_ACROMIOCLAVICULAR' name='l_acromioclavicular' center='0.15 1.44 0.0'>
                        <HAnimJoint DEF='L_SHOULDER' name='l_shoulder' center='0.2 1.44 0.0'>
                          <HAnimJoint DEF='L_ELBOW' name='l_elbow' center='0.2 1.14 0.0'>
                            <HAnimJoint DEF='L_WRIST' name='l_radiocarpal' center='0.2 0.87 0.0'>

                              <!-- K3D: Tablet grip (left hand) -->
                              <HAnimJoint DEF='K3D_TABLET_GRIP' name='k3d_tablet_grip'
                                center='0.0 -0.03 0.0'/>

                            </HAnimJoint>
                          </HAnimJoint>
                        </HAnimJoint>
                      </HAnimJoint>
                    </HAnimJoint>

                    <!-- Right arm chain (abbreviated, mirrors left) -->
                    <HAnimJoint DEF='R_STERNOCLAVICULAR' name='r_sternoclavicular' center='-0.08 1.44 0.0'>
                      <!-- ... mirrors left arm ... -->
                    </HAnimJoint>

                    <!-- Neck and head -->
                    <HAnimJoint DEF='VC4' name='vc4' center='0 1.5 0.01'>
                      <HAnimJoint DEF='VC2' name='vc2' center='0 1.54 0.01'>
                        <HAnimJoint DEF='SKULLBASE' name='skullbase' center='0 1.58 0.01'>

                          <HAnimSegment DEF='SKULL' name='skull'>
                            <!-- Skull mesh with dual-client textures -->
                            <Shape>
                              <DualClientAvatarAppearance>
                                <Appearance containerField='humanAppearance'>
                                  <Material diffuseColor='0.8 0.7 0.6'/>
                                  <ImageTexture url='"avatar_head_human.png"'/>
                                </Appearance>
                                <Appearance containerField='machineAppearance'>
                                  <ImageTexture url='"avatar_head_machine.png"'/>
                                </Appearance>
                              </DualClientAvatarAppearance>
                              <IndexedFaceSet coordIndex='...'>
                                <Coordinate point='...'/>
                                <TextureCoordinate point='...'/>
                              </IndexedFaceSet>
                            </Shape>
                          </HAnimSegment>

                          <!-- Standard HAnim head joints -->
                          <HAnimJoint DEF='L_EYEBALL' name='l_eyeball_joint' center='0.03 1.64 0.07'/>
                          <HAnimJoint DEF='R_EYEBALL' name='r_eyeball_joint' center='-0.03 1.64 0.07'/>
                          <HAnimJoint DEF='JAW' name='temporomandibular' center='0 1.58 0.05'/>

                          <!-- K3D: Cranial origin (brain center) -->
                          <HAnimJoint DEF='K3D_CRANIAL' name='k3d_cranial_origin'
                            center='0.0 0.04 -0.02'>

                            <!-- CranialGalaxy: the AI's brain -->
                            <CranialGalaxy DEF='TRM_BRAIN'
                              cranialOrigin='0 0 0'
                              cranialRadius='0.085'
                              galaxyExtent='200.0'
                              scaleFactor='0.000425'
                              galaxyCount='11'
                              entryCount='247889'
                              visibilityMode='hidden'
                              skullTransparency='0.7'>

                              <!-- Default galaxies loaded -->
                              <GalaxyGroup galaxyName='Math' entryCount='15234'/>
                              <GalaxyGroup galaxyName='Grammar' entryCount='42891'/>
                              <GalaxyGroup galaxyName='Drawing' entryCount='8102'/>
                              <GalaxyGroup galaxyName='Character' entryCount='51023'/>
                              <GalaxyGroup galaxyName='Word' entryCount='67432'/>
                              <GalaxyGroup galaxyName='Number' entryCount='12089'/>
                              <GalaxyGroup galaxyName='Reality' entryCount='23401'/>
                              <GalaxyGroup galaxyName='Audio' entryCount='5123'/>
                              <GalaxyGroup galaxyName='3DObjects' entryCount='9876'/>
                              <GalaxyGroup galaxyName='Tool' entryCount='4521'/>
                              <GalaxyGroup galaxyName='Meta-Navigation' entryCount='8197'/>

                            </CranialGalaxy>

                            <!-- Specialist Visualization -->
                            <SpecialistVisualization DEF='TRM_SPECIALISTS'
                              opacity='0.0'
                              workerCount='9'
                              workerNames='"math" "grammar" "visual" "chat" "physics" "logic" "spatial" "temporal" "meta"'
                              workerColors='0.2 0.4 1.0, 0.2 0.8 0.3, 1.0 0.3 0.2, 1.0 0.9 0.2, 0.2 0.9 0.9, 0.7 0.2 0.9, 1.0 0.6 0.1, 0.9 0.4 0.6, 1.0 1.0 1.0'
                              workerActivation='0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.3'
                              orbitRadius='0.03'
                              orbitSpeed='1.0'/>

                            <!-- Navigation Trace (empty, populated during reasoning) -->
                            <NavigationTrace DEF='TRM_TRACE'
                              opacity='0.0'
                              traceColor='0.3 0.8 1.0'
                              traceWidth='2.0'
                              fadeTime='3.0'
                              showDeadEnds='false'/>

                          </HAnimJoint>

                          <!-- K3D: Thought emitter (above head) -->
                          <HAnimJoint DEF='K3D_EMITTER' name='k3d_thought_emitter'
                            center='0.0 0.25 0.0'>

                            <!-- Cognitive state indicator -->
                            <CognitiveStateIndicator DEF='TRM_INDICATOR'
                              currentPhase='idle'
                              phaseColor='1.0 0.9 0.7'
                              intensity='0.3'
                              opacity='0.8'>
                              <Appearance containerField='idleAppearance'>
                                <Material emissiveColor='1.0 0.9 0.7' transparency='0.3'/>
                              </Appearance>
                              <Appearance containerField='reasoningAppearance'>
                                <Material emissiveColor='0.4 0.6 1.0' transparency='0.0'/>
                              </Appearance>
                              <Appearance containerField='convergedAppearance'>
                                <Material emissiveColor='1.0 1.0 1.0' transparency='0.0'/>
                              </Appearance>
                              <Appearance containerField='sleepingAppearance'>
                                <Material emissiveColor='0.2 0.3 0.8' transparency='0.5'/>
                              </Appearance>
                            </CognitiveStateIndicator>

                          </HAnimJoint>

                        </HAnimJoint> <!-- skullbase -->
                      </HAnimJoint> <!-- vc2 -->
                    </HAnimJoint> <!-- vc4 -->

                  </HAnimJoint> <!-- vt1 -->
                </HAnimJoint> <!-- vt6 -->
              </HAnimJoint> <!-- vt10 -->
            </HAnimJoint> <!-- vl1 -->
          </HAnimJoint> <!-- vl3 -->
        </HAnimJoint> <!-- vl5 -->

        <!-- Leg chains (abbreviated) -->
        <HAnimJoint DEF='L_HIP' name='l_hip' center='0.1 0.87 0.0'>
          <!-- ... standard HAnim leg chain ... -->
        </HAnimJoint>
        <HAnimJoint DEF='R_HIP' name='r_hip' center='-0.1 0.87 0.0'>
          <!-- ... standard HAnim leg chain ... -->
        </HAnimJoint>

      </HAnimJoint> <!-- humanoid_root -->

      <!-- Sites -->
      <HAnimSite DEF='SKULL_VERTEX' name='skull_vertex_tip' center='0 1.72 0.0'/>
      <HAnimSite DEF='L_HAND_TIP' name='l_hand_tip' center='0.2 0.82 0.0'/>
      <HAnimSite DEF='R_HAND_TIP' name='r_hand_tip' center='-0.2 0.82 0.0'/>
      <HAnimSite DEF='NAVEL' name='navel_pt' center='0 1.0 0.08'/>

    </HAnimHumanoid>

    <!-- === PM-KR Agent (cognitive architecture) === -->
    <AgentEntity DEF='TRM_AGENT'
      agentId='trm:primary'
      agentType='trm'
      parameterCount='7000000'
      maxRecursionSteps='9'
      convergenceThreshold='0.01'
      isActive='true'>
      <HAnimHumanoid USE='TRM_HUMANOID' containerField='body'/>
      <GalaxyGroup USE='TRM_BRAIN' containerField='internalState'/>
      <AgentMemoryPalace DEF='TRM_HOUSE' palaceId='house:default'
        entryCount='247889' galaxyCount='19'
        containerField='memoryPalace'>
        <!-- House rooms as children -->
      </AgentMemoryPalace>
      <AgentSwarm DEF='TRM_SWARM' workerCount='9'
        specialistNames='"math" "grammar" "visual" "chat" "physics" "logic" "spatial" "temporal" "meta"'
        convergenceMode='one_mind'/>
    </AgentEntity>

    <!-- === Avatar Body (PM-KR wrapper) === -->
    <AvatarBody DEF='TRM_AVATAR'
      canonicalId='avatar:trm:primary'
      entityType='ai_trm'
      currentState='idle'
      homeHouse='house:default'
      specialistCount='9'>
      <HAnimHumanoid USE='TRM_HUMANOID' containerField='humanoid'/>
      <AgentEntity USE='TRM_AGENT' containerField='agent'/>
      <CranialGalaxy USE='TRM_BRAIN' containerField='cranialGalaxy'/>
      <CognitiveStateIndicator USE='TRM_INDICATOR' containerField='cognitiveState'/>

      <!-- Held object: Memory Tablet -->
      <HeldObject gripType='palm' isHeld='true'
        localOffset='0.0 0.0 0.05' localRotation='1 0 0 1.57'>
        <Transform containerField='object'>
          <Shape>
            <Appearance>
              <Material diffuseColor='0.1 0.1 0.12' specularColor='0.5 0.5 0.5'/>
            </Appearance>
            <Box size='0.20 0.15 0.01'/>
          </Shape>
        </Transform>
        <HAnimSite USE='K3D_TABLET_GRIP' containerField='gripSite'/>
      </HeldObject>

    </AvatarBody>

  </Scene>
</X3D>
```

### 16.2 Human Avatar (Minimal)

```xml
<AvatarBody DEF='HUMAN_PLAYER'
  canonicalId='avatar:human:player1'
  entityType='human'
  currentState='idle'>

  <HAnimHumanoid DEF='PLAYER_HUMANOID' name='Player1' version='2.0' loa='2'>
    <HAnimJoint DEF='P_ROOT' name='humanoid_root' center='0 0.87 0'>
      <!-- Standard HAnim skeleton (LOA-2) -->
      <!-- K3D extension joints present but unused -->
      <!-- ... -->
      <HAnimJoint DEF='P_SKULLBASE' name='skullbase' center='0 1.58 0.01'>
        <HAnimJoint name='k3d_cranial_origin' center='0 0.04 -0.02'/>
        <HAnimJoint name='k3d_thought_emitter' center='0 0.25 0'/>
        <!-- ... standard head joints ... -->
      </HAnimJoint>
    </HAnimJoint>
  </HAnimHumanoid>

  <!-- No agent, no cranialGalaxy (human avatar) -->
  <!-- agent field: NULL -->
  <!-- cranialGalaxy field: NULL -->

  <DualClientAvatarAppearance containerField='dualAppearance'>
    <Appearance containerField='humanAppearance'>
      <Material diffuseColor='0.85 0.72 0.58'/>
      <ImageTexture url='"player1_skin.png"'/>
    </Appearance>
    <Appearance containerField='machineAppearance'>
      <ImageTexture url='"player1_semantic.png"'/>
    </Appearance>
  </DualClientAvatarAppearance>

</AvatarBody>
```

### 16.3 Teaching Mode — AI Shows Reasoning

```xml
<!-- Transition cranial galaxy to teaching mode -->
<TimeSensor DEF='TEACH_CLOCK' cycleInterval='0.5' loop='false'/>

<!-- Make skull transparent and galaxy visible -->
<ROUTE fromNode='TEACH_CLOCK' fromField='fraction_changed'
       toNode='TRM_BRAIN' toField='skullTransparency'/>

<Script DEF='TEACH_MODE'>
  <field accessType='inputOnly' name='activate' type='SFBool'/>
  <field accessType='outputOnly' name='visMode' type='SFString'/>
  <![CDATA[
    function activate(val) {
      if (val) visMode = 'teaching';
      else visMode = 'hidden';
    }
  ]]>
</Script>
<ROUTE fromNode='TEACH_MODE' fromField='visMode'
       toNode='TRM_BRAIN' toField='visibilityMode'/>
```

---

## Appendix A: K3D Joint Extension Summary

| Joint Name | Parent | Default Center (relative) | Purpose |
|-----------|--------|--------------------------|---------|
| `k3d_cranial_origin` | `skullbase` | (0, 0.04, −0.02) | CranialGalaxy coordinate origin |
| `k3d_tablet_grip` | `l_radiocarpal` | (0, −0.03, 0) | Memory Tablet attachment |
| `k3d_thought_emitter` | `skullbase` | (0, 0.25, 0) | Cognitive state indicator mount |

## Appendix B: Cognitive Phase Color Defaults

| Phase | Default Color | RGB | Emissive Intensity |
|-------|--------------|-----|-------------------|
| idle | Warm white | (1.0, 0.9, 0.7) | 0.3 |
| perceiving | Light cyan | (0.6, 0.9, 1.0) | 0.6 |
| navigating | Sky blue | (0.3, 0.6, 1.0) | 0.7 |
| reasoning | Deep blue | (0.2, 0.4, 1.0) | 0.9 |
| converged | Pure white | (1.0, 1.0, 1.0) | 1.0 |
| acting | Gold | (1.0, 0.8, 0.2) | 0.8 |
| teaching | Soft green | (0.4, 1.0, 0.6) | 0.7 |
| sleeping | Deep indigo | (0.2, 0.2, 0.6) | 0.2 |

## Appendix C: Machine Texture Channel Encoding Reference

| Channel | Encoding | Value 0 | Value 128 | Value 255 |
|---------|----------|---------|-----------|-----------|
| R | Joint weight index | humanoid_root | vt1 | last joint |
| G | Segment boundary | Interior texel | — | Boundary edge |
| B | Interaction zone | No zone | — | Tablet grip zone |
| A | State | Neutral/rest | Mid-expression | Full activation |

## Appendix D: Document History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-03-26 | Initial draft. AvatarEmbodiment component, CranialGalaxy, K3D joints, dual-client texturing, interaction nodes, cognitive visualization, conformance levels. |

---

**End of Document**
