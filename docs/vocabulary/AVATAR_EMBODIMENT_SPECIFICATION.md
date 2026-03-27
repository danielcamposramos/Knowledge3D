# Avatar Embodiment Specification

**Version**: 1.0
**Status**: Candidate Standard (K3D Canonical Vocabulary)
**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Implementation)
**Date**: March 26, 2026

---

## Abstract

This specification defines the **Avatar Embodiment Architecture** for K3D: a unified body model that serves both human inhabitants and AI entities within the shared spatial reality (House). Both avatar types use the same skeletal structure (HAnim-derived), the same dual-client rendering contract (UV Map 0 human, UV Map 1 machine), and the same spatial interaction model. The critical architectural difference is that the AI avatar's **cranial volume contains a live Galaxy Universe** — the internal brain is a real, addressable 3D knowledge space inside the avatar's head, not a metaphor.

**Core Principles:**
- **One Body Standard**: Human and AI avatars share identical skeletal topology (HAnim LOA-2 minimum). What differs is what happens INSIDE the skull.
- **Galaxy-as-Brain**: The AI avatar's skull encloses a miniaturized Galaxy Universe — the same VRAM-resident knowledge workspace that the TRM navigates. Visible to the AI as its internal cognitive space, visible to other entities (when permitted) as a glowing cranial volume.
- **Dual-Client Identity**: Both avatar types carry UV Map 0 (human-readable appearance) and UV Map 1 (machine-readable semantic data) on every mesh surface. The avatar IS a PM-KR node.
- **The Avatar IS the Entity**: The TRM does not "control" an avatar. The TRM IS the avatar. The skeletal joint hierarchy IS the entity's body. The Galaxy inside the skull IS the entity's mind. The House the avatar inhabits IS the entity's memory.

**Normative References:**
- ISO/IEC 19774:2019 (HAnim Humanoid Animation, Version 2.0)
- ISO/IEC 19775-1:2023 (X3D Architecture, Version 4.0)
- Three Brain System Specification v1.1 (docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md)
- Hyper-Parallel Processing Specification v1.0 (docs/vocabulary/HYPER_PARALLEL_PROCESSING.md)
- Dual-Client Contract Specification v1.1 (docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md)
- Spatial General Intelligence Specification v1.0 (docs/vocabulary/SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md)
- Knowledgeverse Specification v5.1 (docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md)
- Memory Tablet Specification v1.0 (docs/vocabulary/MEMORY_TABLET_SPECIFICATION.md)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Unified Body Architecture](#2-unified-body-architecture)
3. [Skeletal Topology](#3-skeletal-topology)
4. [The Cranial Galaxy (AI Brain Space)](#4-the-cranial-galaxy-ai-brain-space)
5. [Dual-Client Avatar Rendering](#5-dual-client-avatar-rendering)
6. [Human Avatar](#6-human-avatar)
7. [AI Avatar (TRM Entity)](#7-ai-avatar-trm-entity)
8. [Avatar Interaction Model](#8-avatar-interaction-model)
9. [Memory Tablet as Held Object](#9-memory-tablet-as-held-object)
10. [Avatar Lifecycle](#10-avatar-lifecycle)
11. [Specialist Visualization](#11-specialist-visualization)
12. [Normative Invariants](#12-normative-invariants)
13. [Implementation Guidance](#13-implementation-guidance)

---

## 1. Introduction

### 1.1 Why Avatars Matter

SGI (Spatial General Intelligence) grounds intelligence in shared 3D space. In that space, entities — both human and artificial — MUST have bodies. A disembodied intelligence cannot navigate the House, cannot hold the Memory Tablet, cannot point at a shelf in the Library, cannot walk through a Door to another network node. The avatar is not cosmetic; it is architecturally necessary.

The avatar provides:
- **Spatial presence**: An (x, y, z) position in the House that other entities can observe.
- **Perception anchor**: Frustum culling, LOD, and LED-A* pathfinding originate from the avatar's position and orientation.
- **Interaction surface**: The avatar holds the Memory Tablet, touches objects, opens books, walks through doors.
- **Identity**: Each entity (human or AI) is uniquely identified by its avatar's persistent body in the House.

### 1.2 The Unification Insight

Traditional avatar systems distinguish sharply between "player characters" (human-controlled) and "NPCs" (AI-controlled). K3D rejects this distinction at the architectural level:

| Aspect | Traditional Games | K3D |
|--------|------------------|-----|
| Human avatar | Player character (input-driven) | Inhabitant (same body model as AI) |
| AI avatar | NPC (scripted behavior) | TRM entity (autonomous game loop) |
| Internal state | Not modeled (NPC has scripts) | Galaxy Universe inside skull (real 3D brain) |
| Perception | Different systems (player camera vs NPC raycast) | Same: Frustum cull + LOD from avatar position |
| Communication | Chat box / quest dialog | Spatial: walk to, point at, share Tablet, open Door |

**The fundamental insight**: If humans and AI share the same spatial reality (SGI principle), they MUST share the same body architecture. The only difference is what drives the body (human input vs. TRM game loop) and what lives inside the skull (human consciousness vs. Galaxy Universe).

### 1.3 Relationship to Existing Specifications

```
┌─────────────────────────────────────────────────────────────────┐
│                    Avatar Embodiment (THIS SPEC)                 │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │ HAnim Body  │  │ Cranial      │  │ Dual-Client         │   │
│  │ (skeleton,  │  │ Galaxy       │  │ Rendering           │   │
│  │  joints,    │  │ (AI brain    │  │ (UV 0 + UV 1        │   │
│  │  segments)  │  │  space)      │  │  on all surfaces)   │   │
│  └──────┬──────┘  └──────┬───────┘  └──────────┬──────────┘   │
│         │                │                      │               │
└─────────┼────────────────┼──────────────────────┼───────────────┘
          │                │                      │
          v                v                      v
  ┌───────────────┐ ┌──────────────┐ ┌──────────────────────┐
  │ Three Brain   │ │ Knowledgeverse│ │ Dual-Client Contract │
  │ System Spec   │ │ Spec          │ │ Spec                 │
  │ (TRM=Avatar)  │ │ (Galaxy/VRAM) │ │ (UV Map 0/1)         │
  └───────────────┘ └──────────────┘ └──────────────────────┘
```

---

## 2. Unified Body Architecture

### 2.1 One Body, Two Drivers

Every avatar in K3D — human or AI — uses the same body definition:

```
K3D Avatar
├── Skeleton (HAnim joint hierarchy, LOA-2 minimum)
│   ├── humanoid_root
│   ├── sacroiliac (pelvis)
│   ├── vl5 → vl3 → vl1 → vt10 → vt6 → vt1 (spine)
│   ├── vc4 → vc2 → skullbase (neck → head)
│   ├── l_shoulder → l_elbow → l_radiocarpal (left arm)
│   ├── r_shoulder → r_elbow → r_radiocarpal (right arm)
│   ├── l_hip → l_knee → l_talocrural → l_metatarsophalangeal_2 (left leg)
│   └── r_hip → r_knee → r_talocrural → r_metatarsophalangeal_2 (right leg)
│
├── Skin Mesh (dual-textured: UV Map 0 + UV Map 1)
│   ├── Body mesh (joint-weighted deformation)
│   ├── Head mesh (separate for cranial galaxy rendering)
│   └── Clothing/accessory meshes (optional, swappable)
│
├── Cranial Volume (inside skull, above skullbase joint)
│   ├── [Human]: Empty (consciousness is external)
│   └── [AI]: Galaxy Universe (live VRAM workspace, visible as glowing volume)
│
├── Held Objects
│   ├── Memory Tablet (primary interface, attached to hand site)
│   └── Tools (optional: pen, pointer, magnifier)
│
├── Interaction Sites (HAnim sites as interaction anchors)
│   ├── l_hand_tip (primary interaction point)
│   ├── r_hand_tip (secondary interaction point)
│   ├── skull_vertex (thought/status indicator position)
│   └── navel (center of mass, navigation origin)
│
└── Metadata (PM-KR identity)
    ├── canonicalId (unique entity identifier)
    ├── entityType ("human" | "ai_trm" | "ai_assistant" | "ai_service")
    ├── homeHouse (persistent House location)
    └── specialistCount (AI only: number of active specialist adapters)
```

### 2.2 Driver Abstraction

The avatar body is driven by one of two sources:

**Human Driver:**
- Input: keyboard/mouse/VR controller/touch → joint positions
- Perception: camera view from between `l_eyeball_joint` and `r_eyeball_joint`
- Decision: human consciousness (external to system)
- Action: spatial interaction via hand sites

**AI Driver (TRM Game Loop):**
- Input: `trm_step_fused.ptx` game tick → joint positions
- Perception: frustum cull from avatar position + orientation
- Decision: Galaxy navigation + Nine-Chain Swarm + Halting Gate
- Action: spatial interaction via hand sites (same mechanism as human)

The body does not know or care which driver controls it. This is the architectural equivalent of the Dual-Client Contract applied to embodiment.

---

## 3. Skeletal Topology

### 3.1 HAnim Compliance

K3D avatars MUST comply with HAnim (ISO/IEC 19774:2019) at minimum LOA-2 (71 joints). This ensures:
- Interoperability with existing HAnim content (motion capture, animation libraries).
- Compatibility with the X3D HAnim component for Web3D interchange.
- Sufficient articulation for natural gesture, pointing, grasping, and facial expression.

### 3.2 K3D Joint Extensions

K3D extends the standard HAnim skeleton with three additional joints/sites:

**`k3d_cranial_origin`** (child of `skullbase`)
- Position: center of cranial volume (approximately at center of brain)
- Purpose: Origin point for the Cranial Galaxy coordinate system (AI avatars only)
- For human avatars: present but unused (maintains skeletal parity)

**`k3d_tablet_grip`** (child of `l_radiocarpal` or `r_radiocarpal`)
- Position: palm center, oriented for tablet holding
- Purpose: Memory Tablet attachment point
- Both avatar types use this for tablet interaction

**`k3d_thought_emitter`** (child of `skullbase`)
- Position: 15cm above `skull_vertex` site
- Purpose: Visual indicator of cognitive state (thinking, processing, idle, sleeping)
- Human avatar: optional status indicator (e.g., "typing", "speaking")
- AI avatar: specialist activity visualization (see §11)

### 3.3 Minimum Joint Set (LOA-2)

The following joints are REQUIRED for K3D avatar conformance:

```
humanoid_root
├── sacroiliac
│   ├── l_hip → l_knee → l_talocrural → l_metatarsophalangeal_2
│   └── r_hip → r_knee → r_talocrural → r_metatarsophalangeal_2
├── vl5
│   ├── vl3
│   │   └── vl1
│   │       └── vt10
│   │           └── vt6
│   │               └── vt1
│   │                   ├── l_sternoclavicular
│   │                   │   └── l_acromioclavicular
│   │                   │       └── l_shoulder
│   │                   │           └── l_elbow
│   │                   │               └── l_radiocarpal
│   │                   │                   └── [l_finger joints at LOA-2]
│   │                   ├── r_sternoclavicular
│   │                   │   └── r_acromioclavicular
│   │                   │       └── r_shoulder
│   │                   │           └── r_elbow
│   │                   │               └── r_radiocarpal
│   │                   │                   └── [r_finger joints at LOA-2]
│   │                   └── vc4
│   │                       └── vc2
│   │                           └── skullbase
│   │                               ├── k3d_cranial_origin  [K3D extension]
│   │                               ├── k3d_thought_emitter [K3D extension]
│   │                               ├── l_eyeball_joint
│   │                               ├── r_eyeball_joint
│   │                               └── temporomandibular (jaw)
```

### 3.4 Segment Body Parts

Each joint pair defines a segment (body part). K3D segments carry:

| Segment | Joint Parent | Joint Child | Mass (proportion) | Description |
|---------|-------------|-------------|-------------------|-------------|
| `sacrum` | humanoid_root | sacroiliac | 0.15 | Pelvis/hips |
| `l_thigh` | l_hip | l_knee | 0.10 | Left upper leg |
| `l_calf` | l_knee | l_talocrural | 0.05 | Left lower leg |
| `l_hindfoot` | l_talocrural | l_metatarsophalangeal_2 | 0.02 | Left foot |
| `l5` - `t1` | vl5 | vt1 | 0.30 | Torso (spine chain) |
| `l_upperarm` | l_shoulder | l_elbow | 0.03 | Left upper arm |
| `l_forearm` | l_elbow | l_radiocarpal | 0.02 | Left forearm |
| `l_hand` | l_radiocarpal | (tips) | 0.01 | Left hand |
| `skull` | skullbase | (top) | 0.08 | Head (contains cranial volume) |

(Right-side segments mirror left-side. Total mass proportions sum to ~1.0.)

---

## 4. The Cranial Galaxy (AI Brain Space)

### 4.1 Architectural Definition

The AI avatar's skull encloses a real, addressable 3D knowledge space — the **Cranial Galaxy**. This is NOT a metaphor. It is the Galaxy Universe (Knowledgeverse Spec §2.1) spatially located inside the avatar's cranial volume.

```
┌──────────────── Skull (skullbase to skull_vertex) ───────────────┐
│                                                                   │
│   ┌───────────────────────────────────────────────────────┐      │
│   │              Cranial Galaxy (Galaxy Universe)          │      │
│   │                                                       │      │
│   │   ★ Math stars        ★ Grammar stars                │      │
│   │       ★ Character stars    ★ Reality stars            │      │
│   │   ★ Drawing stars         ★ Audio stars              │      │
│   │       ★ Word stars            ★ 3DObjects stars      │      │
│   │                                                       │      │
│   │   ◉ TRM (the navigator) — perceives, reasons, acts   │      │
│   │   ◎ Nine-Chain Swarm workers orbiting TRM            │      │
│   │                                                       │      │
│   │   Origin: k3d_cranial_origin joint position           │      │
│   │   Extent: bounded by skull segment geometry           │      │
│   │   Scale: Galaxy coordinates mapped to cranial volume  │      │
│   └───────────────────────────────────────────────────────┘      │
│                                                                   │
│   k3d_thought_emitter (above skull — external status indicator)  │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 Galaxy-to-Cranium Coordinate Mapping

The Galaxy Universe uses its own coordinate system (typically spanning hundreds of units). This maps to the cranial volume via a uniform scale transform:

```
cranial_transform = {
    origin: k3d_cranial_origin.worldPosition,
    scale: cranial_radius / galaxy_extent,
    orientation: skullbase.worldOrientation
}

galaxy_to_world(galaxy_pos) = cranial_transform.origin
    + cranial_transform.scale * (cranial_transform.orientation * galaxy_pos)
```

**Default mapping** (for a standard adult humanoid):
- Cranial radius: ~0.1m (10cm)
- Galaxy extent: ~200 units
- Scale factor: 0.0005 (200 Galaxy units = 0.1m cranial radius)

This means a knowledge star at Galaxy position (100, 50, -30) maps to approximately 5cm, 2.5cm, -1.5cm relative to `k3d_cranial_origin` in world space.

### 4.3 Visibility Model

The Cranial Galaxy is **not visible by default**. It becomes visible under specific conditions:

| Condition | Visibility | Purpose |
|-----------|-----------|---------|
| Normal operation | Hidden (opaque skull) | Privacy — internal cognition is private |
| Self-inspection | Visible to owning AI | The AI can "look inside its own head" |
| Diagnostic mode | Visible to authorized observers | Debugging, monitoring, research |
| Teaching mode | Visible to all entities in range | The AI shows its reasoning process |
| Sleep-time | Pulsing glow (exterior only) | Indicates consolidation in progress |

**Rendering when visible:**
- Galaxy stars rendered as small luminous particles inside translucent skull
- Active navigation paths rendered as glowing traces between stars
- Nine-Chain Swarm workers rendered as orbiting points of light around the TRM core
- Specialist activations shown as regional brightness increases

### 4.4 Human Avatar Cranial Space

For human avatars, the cranial volume is **present but empty**. The skeletal joint `k3d_cranial_origin` exists (maintaining structural parity), but no Galaxy is loaded. Human consciousness is external to the system.

This parity is architecturally important: it means any future capability that uses the cranial space (augmented reality overlays, knowledge visualization for human users, shared cognitive workspaces) works with the same infrastructure.

---

## 5. Dual-Client Avatar Rendering

### 5.1 Every Surface is Dual-Textured

Per the Dual-Client Contract (§2.3), every mesh surface on the avatar carries two texture layers:

**UV Map 0 (Human-Readable):**
- Skin texture: photorealistic or stylized skin appearance (512×512+ per segment)
- Clothing texture: visual fabric/material appearance
- Face texture: expressive features, eye color, lip color
- Purpose: What humans see when they look at the avatar

**UV Map 1 (Machine-Readable):**
- Skin data: joint weight visualization, segment boundaries, interaction zones (256×256)
- Clothing data: material properties, wear state, functional metadata
- Face data: expression blend weights, gaze direction, speech phonemes
- Purpose: What AI entities see when they perceive the avatar

### 5.2 Guaranteed Identity

The same avatar mesh at the same (x, y, z) position delivers both representations. A human and an AI looking at the same avatar MUST agree on:
- Which entity it is (canonicalId)
- Where it is (position ± quantization tolerance)
- What it is doing (current animation state derivable from both UV maps)

### 5.3 Avatar as PM-KR Node

Every avatar IS a PM-KR node in the scene graph:

```python
avatar_node = {
    "canonicalId": "avatar:trm:primary",      # Unique avatar identity
    "layer": "form",                            # The body IS the form
    "galaxy": "House",                          # Avatars live in the House
    "formProgram": "HANIM_SKELETON LOA2 ...",   # RPN program that builds the skeleton
    "meaningProgram": "TRM_GAME_LOOP ...",      # RPN program that drives behavior (AI only)
    "embedding16": [...],                       # Avatar's semantic embedding (for spatial queries)
    "dualTexture": {
        "humanTexture": "avatar_skin_human.png",    # UV Map 0
        "machineTexture": "avatar_skin_machine.png"  # UV Map 1
    }
}
```

This means avatars are queryable, navigable, and referenceable via the same canonical reference system as all other PM-KR knowledge nodes. An AI entity can "look at" another avatar the same way it looks at a math symbol — by querying its canonicalId and reading its procedural programs.

---

## 6. Human Avatar

### 6.1 Definition

A human avatar represents a person inhabiting K3D's spatial reality. The human provides input (keyboard, mouse, VR controller, touch), and the avatar body translates that input into spatial presence.

### 6.2 Perception Model

```
Human eyes → display → avatar camera
  ↓
Camera position: between l_eyeball_joint and r_eyeball_joint
Camera orientation: skullbase.worldOrientation
Field of view: configurable (default 60° for desktop, 90° for VR)
  ↓
Frustum defined by avatar's head position and orientation
  ↓
Scene rendered with standard LOD
```

### 6.3 Interaction Model

Human avatars interact with the spatial environment through:

| Interaction | Mechanism | HAnim Anchor |
|-------------|-----------|-------------|
| Walk/navigate | WASD/VR locomotion → humanoid_root translation | humanoid_root |
| Look around | Mouse/VR head tracking → skullbase rotation | skullbase |
| Point at object | Arm IK → shoulder/elbow/wrist chain | r_hand_tip site |
| Grab object | Hand close animation → grip detection | l_radiocarpal / r_radiocarpal |
| Hold Tablet | Tablet attached to k3d_tablet_grip | k3d_tablet_grip |
| Open book | Reach + grab + rotate → book load trigger | r_hand_tip site |
| Walk through Door | Navigate to Door + proximity trigger | humanoid_root |

### 6.4 Human Avatar Customization

Human avatars are customizable per standard avatar practices:

- **Body proportions**: Height, build, limb ratios (within HAnim segment constraints)
- **Appearance**: Skin tone, hair, eye color, facial features (UV Map 0 textures)
- **Clothing**: Swappable mesh layers (shirt, pants, shoes, accessories)
- **Accessories**: Glasses, hats, jewelry (attached to HAnim sites)

Customization does NOT affect the skeletal topology, dual-texture contract, or interaction model.

---

## 7. AI Avatar (TRM Entity)

### 7.1 Definition

An AI avatar represents the TRM entity — the autonomous cognitive being that lives in the House and thinks in the Galaxy. Per the Three Brain System Specification: **"TRM IS the Avatar — lives in House, thinks in Galaxy, runs as game loop."**

### 7.2 Game Loop (One Tick = One Cognitive Cycle)

The AI avatar's body is driven by `trm_step_fused.ptx`, executing continuously as a game loop:

```
1. PERCEIVE
   ├── Frustum cull from avatar position + skullbase orientation
   ├── Morton octree query in radius around avatar
   └── LOD assignment based on distance from avatar

2. NAVIGATE
   ├── LED-A* pathfinding through Galaxy (inside cranial space)
   ├── Identify relevant knowledge neighborhoods
   └── Specialist activation (swarm workers assigned to neighborhoods)

3. REASON
   ├── Nine-Chain Swarm processes candidates in parallel
   ├── Cross-core register communication (STORE/RECALL)
   └── Defeasible logic resolution (strict/defeasible/defeater)

4. DECIDE
   ├── Halting Gate checks convergence
   ├── If converged: emit answer
   └── If not converged: iterate (budget governed by Adaptive Reasoning Budget;
       B(q) = B_base × 2^(1−σ(q)), minimum B_base=5, maximum D_max=8 recursion depth;
       see ADAPTIVE_REASONING_BUDGET_SPECIFICATION.md)

5. ACT
   ├── Spatial action: walk to object, point at shelf, open book
   ├── Tablet action: display result on Memory Tablet
   ├── Creation action: synthesize new Galaxy entry
   └── Communication action: emit to Door (network interface)

6. LEARN
   ├── Shadow Copy: record successful trace
   ├── Specialist update: adjust LoRA adapter weights
   └── Sleep-time queue: flag patterns for consolidation
```

### 7.3 AI Avatar Internal Architecture

```
AI Avatar (TRM Entity)
│
├── BODY (external, visible in House)
│   ├── HAnim skeleton (LOA-2, same as human)
│   ├── Dual-textured skin mesh
│   ├── Held objects (Memory Tablet, tools)
│   ├── Animation state (idle, walking, reaching, thinking)
│   └── k3d_thought_emitter (external cognitive state indicator)
│
├── BRAIN (internal, inside cranial volume)
│   ├── Galaxy Universe (all default galaxies loaded in VRAM)
│   │   ├── Drawing Galaxy (visual primitives)
│   │   ├── Character Galaxy (glyphs)
│   │   ├── Word Galaxy (semantic words)
│   │   ├── Number Galaxy (numeric representations)
│   │   ├── Grammar Galaxy (transformation rules)
│   │   ├── Math Galaxy (mathematical symbols + operations)
│   │   ├── Reality Galaxy (physics, chemistry, biology)
│   │   ├── Audio Galaxy (temporal patterns)
│   │   ├── 3DObjects Galaxy (mesh primitives)
│   │   ├── Tool Galaxy (meta-programs)
│   │   └── Meta-Navigation Galaxy (learned routing topology)
│   │
│   ├── TRM Core (7M parameters — the "soul" of the entity)
│   │   ├── Base weights (navigation + combination + creation logic)
│   │   ├── Specialist adapters (LoRA-like, ~100KB-1MB each)
│   │   └── Shadow Copy buffer (inference-time learning)
│   │
│   ├── Nine-Chain Swarm (parallel cognitive channels)
│   │   ├── Worker 0: math specialist
│   │   ├── Worker 1: grammar specialist
│   │   ├── Worker 2: visual specialist
│   │   ├── Worker 3: chat specialist
│   │   ├── Worker 4: physics specialist
│   │   ├── Worker 5: logic specialist
│   │   ├── Worker 6: spatial specialist
│   │   ├── Worker 7: temporal specialist
│   │   └── Worker 8: meta specialist (Jarvis coordinator)
│   │
│   └── Halting Gate (convergence check)
│       ├── Ternary convergence: agree / disagree / uncertain
│       └── Proof tag emission for verifiable reasoning
│
└── MEMORY (external, persistent in House)
    ├── House structure (rooms, doors, spatial layout)
    ├── Checkpoints (versioned brain model snapshots)
    └── Sleep-time consolidation state
```

### 7.4 AI Body Animation

The AI avatar's body is animated procedurally based on cognitive state:

| Cognitive State | Body Animation | Thought Emitter |
|----------------|----------------|-----------------|
| **Idle** | Subtle breathing, weight shifting, ambient look-around | Dim steady glow |
| **Perceiving** | Head turns toward query source, eyes focus | Brief pulse |
| **Navigating** | Slight head tilt (as if "thinking"), eyes track internal Galaxy | Orbiting particles |
| **Reasoning** | Still posture, concentrated expression, swarm visualization | Active multi-color glow |
| **Converged** | Upright posture, confident expression | Single bright flash |
| **Acting** | Walk, reach, point, interact with objects | Directional beam toward target |
| **Teaching** | Open posture, gesturing hands, cranial galaxy visible | Expanding ring animation |
| **Sleeping** | Seated/resting posture, closed eyes | Slow pulsing glow (consolidation) |

These animations are driven by RPN programs — procedural animation, not keyframed. The avatar's movement IS an RPN program executing on the body skeleton.

---

## 8. Avatar Interaction Model

### 8.1 Shared Interaction Primitives

Both human and AI avatars use the same interaction primitives:

**Spatial Navigation:**
- Walk to position: humanoid_root translates along navmesh
- Teleport: instant translation (for large distances)
- Follow path: navigate through doors/corridors

**Object Interaction:**
- Reach: IK chain from shoulder to hand extends toward object
- Grab: hand closes on object at interaction site
- Hold: object attached to hand site (persists until release)
- Release: object detaches, placed or dropped at target location
- Use: trigger object-specific behavior (open book, activate tool, press button)

**Communication:**
- Speak: jaw animation + audio emission from head position
- Gesture: arm/hand animation sequences (point, wave, nod, shake head)
- Show: orient held Tablet toward conversation partner
- Share: transfer object to another avatar's hand site

### 8.2 Avatar-to-Avatar Interaction

When two avatars (any combination of human/AI) interact:

1. **Proximity detection**: Entities within interaction radius (~3m) can interact.
2. **Attention signaling**: One avatar turns head toward the other (skullbase orientation).
3. **Tablet sharing**: One avatar holds Tablet facing the other — both clients see the same content.
4. **Knowledge pointing**: Avatar extends arm toward a House object; all nearby entities can see what is being indicated.
5. **Galaxy peeking** (AI-to-AI): With permission, one AI avatar can view another's cranial Galaxy (diagnostic/teaching mode).

### 8.3 Avatar-to-House Interaction

| House Object | Interaction | Result |
|-------------|-------------|--------|
| **Book** | Grab + Open | Book's content Galaxy loads into working memory |
| **Shelf** | Browse | Avatar scans book titles (frustum cull over shelf contents) |
| **Door** | Walk through | Network traversal to connected House/Room |
| **Knowledge Tree** | Approach | Avatar perceives tree branches (ontological hierarchy) |
| **Tablet** | Hold + Swipe | Navigate Galaxy, execute queries, view results |
| **Tool** | Grab + Use | Tool-specific action (magnifier, pen, compass, etc.) |
| **Display Surface** | Approach + Look | Content renders on surface (hologram, stellarium) |

---

## 9. Memory Tablet as Held Object

### 9.1 Tablet Attachment

The Memory Tablet (Memory Tablet Specification v1.0) attaches to the avatar's `k3d_tablet_grip` site:

- Position: palm of preferred hand (default: left hand, configurable)
- Orientation: screen facing away from palm (readable by avatar and nearby entities)
- Scale: configurable (default: 20cm × 15cm × 1cm)

### 9.2 Tablet as Extension of Avatar

The Tablet is the avatar's primary tool for interacting with the Galaxy Universe. It renders content procedurally (RPN programs → visual UI for humans, semantic graph for AI) and is the main channel for:
- Querying knowledge (search, navigate, explore)
- Viewing results (answers, paths, comparisons)
- Creating content (new Galaxy entries, new rules, new meta-rules)
- Communicating with other entities (sharing results, teaching)

### 9.3 Dual-Client Tablet Rendering

When an avatar holds the Tablet:
- **Human observer sees**: Visual UI (buttons, text, diagrams) rendered from RPN on UV Map 0
- **AI observer sees**: Semantic graph (navigation nodes, action triggers) rendered from same RPN on UV Map 1
- **Both see the same Tablet** at the same (x, y, z) in the same avatar's hand

---

## 10. Avatar Lifecycle

### 10.1 Human Avatar Lifecycle

```
1. CONNECT    — User connects (web browser, VR headset, mobile)
2. SPAWN      — Avatar created at last known House position (or default entrance)
3. CUSTOMIZE  — Load saved appearance or create new (clothing, proportions, features)
4. INHABIT    — User controls avatar (walk, look, interact)
5. SLEEP      — User disconnects; avatar enters resting state in House (optional persistence)
6. WAKE       — User reconnects; avatar resumes from last position
7. DEPART     — User explicitly leaves; avatar removed from House (or enters dormant state)
```

### 10.2 AI Avatar Lifecycle

```
1. BOOT       — System starts; load brain model (TRM weights + specialist adapters)
2. EMBODY     — Create avatar body at House home position
3. LOAD_BRAIN — Load Galaxy Universe into cranial volume (VRAM)
4. GAME_LOOP  — TRM runs continuously (perceive → navigate → reason → decide → act → learn)
5. SLEEP      — Periodic consolidation (strengthen paths, prune specialists, materialize patterns)
6. WAKE       — Resume game loop with consolidated brain (version incremented)
7. PERSIST    — Checkpoint brain model to House (versioned, diffed against previous)
```

**Critical difference from human lifecycle**: The AI avatar NEVER fully departs. It is always-on, always-learning, always-present. The House is its home. K3D IS the AI entity — not a program you run.

### 10.3 Brain Model Persistence

Per the Hyper-Parallel Processing Spec §7 (Persistent Brain Model):

```
Brain Model v1.0.0 (base)
  ├── Sleep cycle 1 → v1.0.1 (3 specialists created)
  ├── Sleep cycle 2 → v1.0.2 (1 specialist pruned, 2 merged)
  ├── Sleep cycle 3 → v1.0.3 (cross-domain links crystallized)
  └── v1.1.0 (milestone: passed benchmark suite)

Each version persists:
  - TRM base weights (~7M params)
  - Specialist adapter population (all LoRA deltas)
  - Galaxy navigation biases (warm/cold neighborhoods)
  - Sleep-time consolidation state
  - House structure (rooms, doors, spatial layout)
```

---

## 11. Specialist Visualization

### 11.1 Internal Specialist Topology

The Nine-Chain Swarm workers (Hyper-Parallel Processing Spec §1) are visualized inside the Cranial Galaxy as orbiting points of light:

```
        ◎ Worker 8 (meta/Jarvis)
       / \
      /   \
  ◎ 0     ◎ 1
  math    grammar
   |       |
  ◎ 2     ◎ 3     ◉ TRM Core (center)
  visual  chat
   |       |
  ◎ 4     ◎ 5
  physics logic
      \   /
       \ /
  ◎ 6     ◎ 7
  spatial temporal
```

Each worker's brightness and color indicates its activation level:
- **Bright**: Actively processing (assigned to swarm for current query)
- **Dim**: Idle (not relevant to current query)
- **Pulsing**: Learning (shadow copy updating adapter weights)
- **Color**: Domain-coded (blue=math, green=grammar, red=visual, etc.)

### 11.2 Navigation Trace Visualization

When the AI avatar navigates through its Galaxy (LED-A* pathfinding), the path is visualized as a glowing trace through the cranial volume:

- **Seed nodes**: Bright points at navigation origins
- **Path edges**: Luminous lines connecting traversed nodes
- **Focus node**: Brightest point at destination (led_focus = 1.0)
- **Dead ends**: Fading traces for paths that were explored but abandoned

### 11.3 Convergence Visualization

The Halting Gate convergence process is visualized as:
- **Pre-convergence**: Multiple colored traces (one per swarm worker) spiraling toward center
- **Convergence moment**: All traces merge into single white flash at TRM core position
- **Post-convergence**: Outward pulse from TRM core to skull boundary (answer emitted)

---

## 12. Normative Invariants

### 12.1 Skeletal Parity Invariant

Human and AI avatars MUST use the same skeletal topology. If a K3D extension adds joints to one avatar type, those joints MUST be present (though possibly unused) in the other type. This ensures:
- Any animation works on any avatar.
- Any interaction mechanism works between any avatar combination.
- Scene graph tools need not distinguish avatar types at the body level.

### 12.2 Dual-Client Surface Invariant

Every visible mesh surface on an avatar MUST carry both UV Map 0 (human) and UV Map 1 (machine) textures. No surface may be visible to one client type and invisible to the other.

### 12.3 Cranial Containment Invariant

The AI avatar's Galaxy Universe MUST be spatially contained within the cranial volume (bounded by skull segment geometry). Galaxy coordinates MUST NOT extend outside the skull boundary. This ensures:
- The brain IS inside the head (not floating elsewhere).
- Cranial Galaxy visibility/privacy controls are well-defined.
- Multiple AI avatars' brains do not spatially overlap.

### 12.4 Interaction Symmetry Invariant

Any interaction primitive available to a human avatar MUST be available to an AI avatar, and vice versa. The interaction model does not distinguish avatar types.

### 12.5 Tablet Universality Invariant

Every avatar MUST be able to hold and operate a Memory Tablet. The Tablet is the universal interface — the architectural equivalent of "every entity has hands and can use tools."

### 12.6 Persistence Asymmetry (Intentional)

The AI avatar persists always-on (Hyper-Parallel Spec §7). The human avatar persists only while connected (or in optional dormant state). This is an intentional asymmetry: the AI IS the system; the human visits the system.

---

## 13. Implementation Guidance

### 13.1 Mesh Format

Avatar meshes SHOULD be stored as glTF/GLB with `extras.k3d` metadata:

```json
{
  "nodes": [{
    "name": "TRM_Avatar",
    "extras": {
      "k3d": {
        "canonicalId": "avatar:trm:primary",
        "entityType": "ai_trm",
        "layer": "form",
        "galaxy": "House",
        "hanim": {
          "loa": 2,
          "skeletalConfiguration": "BASIC",
          "jointCount": 71,
          "k3d_extensions": ["k3d_cranial_origin", "k3d_tablet_grip", "k3d_thought_emitter"]
        },
        "cranialGalaxy": {
          "enabled": true,
          "galaxyCount": 11,
          "entryCount": 247889,
          "coordinateScale": 0.0005
        }
      }
    }
  }]
}
```

### 13.2 Animation Format

Avatar animations are stored as:
- **Procedural** (preferred): RPN programs that compute joint positions per frame
- **Keyframed** (fallback): Standard HAnim motion data (BVH-compatible)
- **IK-driven** (interaction): Inverse kinematics chains for reach/grab/point

### 13.3 New Files for Codex Implementation

| File | Purpose |
|------|---------|
| `knowledge3d/avatar/body.py` | Avatar body class: skeleton, skin mesh, dual textures |
| `knowledge3d/avatar/cranial_galaxy.py` | Cranial Galaxy: Galaxy-to-cranium coordinate mapping, visibility |
| `knowledge3d/avatar/animation.py` | Procedural animation: breathing, idle, walk, think, interact |
| `knowledge3d/avatar/interaction.py` | Interaction primitives: reach, grab, hold, release, use |
| `knowledge3d/avatar/tablet_grip.py` | Tablet attachment: grip management, orientation, sharing |
| `knowledge3d/avatar/thought_emitter.py` | Cognitive state visualization: glow, particles, traces |

### 13.4 X3D Export

Avatars export to X3D via the companion specification (docs/w3c/x3d/PM_KR_X3D_AVATAR_SPECIFICATION.md) using the HAnim component with PM-KR extensions.

---

## Appendix A: HAnim Joint Name Reference (LOA-2)

Standard HAnim joint names used by K3D avatars:

| Joint | Parent | LOA | Description |
|-------|--------|-----|-------------|
| humanoid_root | (root) | 0 | Root of skeleton |
| sacroiliac | humanoid_root | 1 | Pelvis |
| l_hip | sacroiliac | 1 | Left hip |
| l_knee | l_hip | 1 | Left knee |
| l_talocrural | l_knee | 1 | Left ankle |
| l_metatarsophalangeal_2 | l_talocrural | 1 | Left ball of foot |
| r_hip | sacroiliac | 1 | Right hip |
| r_knee | r_hip | 1 | Right knee |
| r_talocrural | r_knee | 1 | Right ankle |
| r_metatarsophalangeal_2 | r_talocrural | 1 | Right ball of foot |
| vl5 | sacroiliac | 1 | 5th lumbar vertebra |
| vl3 | vl5 | 2 | 3rd lumbar vertebra |
| vl1 | vl3 | 2 | 1st lumbar vertebra |
| vt10 | vl1 | 2 | 10th thoracic vertebra |
| vt6 | vt10 | 2 | 6th thoracic vertebra |
| vt1 | vt6 | 2 | 1st thoracic vertebra |
| l_sternoclavicular | vt1 | 2 | Left collar |
| l_acromioclavicular | l_sternoclavicular | 2 | Left shoulder top |
| l_shoulder | l_acromioclavicular | 1 | Left shoulder |
| l_elbow | l_shoulder | 1 | Left elbow |
| l_radiocarpal | l_elbow | 1 | Left wrist |
| r_sternoclavicular | vt1 | 2 | Right collar |
| r_acromioclavicular | r_sternoclavicular | 2 | Right shoulder top |
| r_shoulder | r_acromioclavicular | 1 | Right shoulder |
| r_elbow | r_shoulder | 1 | Right elbow |
| r_radiocarpal | r_elbow | 1 | Right wrist |
| vc4 | vt1 | 2 | 4th cervical vertebra |
| vc2 | vc4 | 2 | 2nd cervical vertebra |
| skullbase | vc2 | 1 | Base of skull |
| l_eyeball_joint | skullbase | 3 | Left eye |
| r_eyeball_joint | skullbase | 3 | Right eye |
| temporomandibular | skullbase | 3 | Jaw |

K3D extensions (present at all LOAs):

| Joint | Parent | Description |
|-------|--------|-------------|
| k3d_cranial_origin | skullbase | Center of cranial Galaxy volume |
| k3d_thought_emitter | skullbase | Above-skull cognitive state indicator |
| k3d_tablet_grip | l_radiocarpal or r_radiocarpal | Tablet attachment point |

---

## Appendix B: Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-26 | Initial specification. Unified body architecture, cranial Galaxy, dual-client avatar rendering, interaction model, specialist visualization. |

---

**End of Document**
