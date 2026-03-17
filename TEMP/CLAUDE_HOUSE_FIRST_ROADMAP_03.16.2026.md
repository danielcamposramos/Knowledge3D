# House-First Roadmap — From Empty Shelves to Living Knowledge

**Date**: March 16, 2026
**Author**: Claude (Architecture Partner)
**Status**: Architecture Proposal — Awaiting Daniel's Review
**Supersedes**: Benchmark-focused Phase B+ priorities

---

## The Problem

We've been optimizing navigation over **empty shelves**.

- Galaxy has 38K+ entries — but they're bootstrapped stubs (numbers 0-1000, drawing primitives, physics seeds)
- House exists as GLB shells — no knowledge gardens, no library spaces, no real content
- Rich datasets sit on disk (`/K3D/Knowledge3D.local/datasets/`) — NOT proceduralized into the House
- "Cat" as a meaning-centric star (cat = gato = 猫 = one star, all media) doesn't exist
- LHE multi-hop fails not because the graph crystallizer is weak, but because there's nothing meaningful to hop THROUGH
- MMLU answers are navigated to by spatial proximity tricks, not by actual domain knowledge living in the right rooms

Benchmark scores measure **navigation skill**, not **knowledge**. Tweaking weights to squeeze 2 more points is optimizing a hollow system.

---

## The Direction

**Build the House. Fill the shelves. The benchmarks follow naturally.**

This is not a pivot — it's the original vision finally becoming the priority. The composed head pipeline WORKS (sovereign GPU, zero fallbacks). The navigation infrastructure IS there. What's missing is the **content it navigates**.

---

## Core Principles (Restated for Clarity)

### 1. Galaxy = Ephemeral Working Memory

Galaxy is the AI's **working memory** — loaded into VRAM when the mind needs to think. It is NOT permanent storage. It's where concepts are blended, executed, composed, and reasoned about.

**Analogy**: Your desk when you're working. You pull books from shelves, spread them out, work with them, put them back.

### 2. House = Permanent Knowledge Repository

The House IS the permanent store. Knowledge lives in rooms — knowledge gardens, library spaces, workshops. The House is a 3D asset (GLB on disk) containing internal assets. It persists across wake/sleep cycles. It's the Method of Loci made literal.

**Analogy**: Your house with its library, study, workshop. Things live there whether or not you're thinking about them.

### 3. Meaning-Centric Stars (Language-Agnostic Concepts)

"Cat" is cat in any language. The MEANING star for "cat" contains:

```
Star: concept_cat
├── meaning:      small domesticated feline (Layer 2)
├── visual_rpn:   procedural drawing of cat silhouette (Drawing Galaxy)
├── audio:        "meow" spectrogram + pronunciation per language (Audio Galaxy)
├── languages:
│   ├── en: "cat"   → char_refs [c, a, t] (Character Galaxy)
│   ├── pt: "gato"  → char_refs [g, a, t, o]
│   ├── ja: "猫"    → char_refs [猫]
│   ├── zh: "猫"    → char_refs [猫]  (shared with ja — symlink!)
│   └── es: "gato"  → char_refs [g, a, t, o] (shared with pt — symlink!)
├── reality:      mass ~4kg, quadruped, mammal (Reality Galaxy)
├── taxonomy:     felidae > carnivora > mammalia (Reality Galaxy)
├── behavior_rpn: procedural locomotion rules (Reality Enabler)
└── house_location: Library/Biology/Mammalia/room_coords
```

When the TRM needs to think about "cat", it loads THIS star from the House into Galaxy working memory — with ALL its facets. Not just a text embedding. The actual procedural, multi-modal, meaning-centric knowledge.

### 4. Same Primitives Build Everything

The drawing primitives that make glyphs (Bézier → segments → RPN) also make:
- Room walls and doors (RECT, LINE compositions)
- Furniture and shelves (3D extrusions of 2D primitives)
- Knowledge garden layouts (procedural L-system growth)
- Book spines on shelves (Character Galaxy glyphs composed into labels)

One procedural system builds the ENTIRE House — from the smallest glyph to the largest room.

### 5. Reality Enabler Constructs the House

The model itself constructs its own House using rules:
- Civil engineering rules (structural integrity, room layout)
- Game-style construction (like NPCs building in simulation games)
- Procedural generation via L-systems, noise functions, grammar rules
- The TRM learns WHERE to place knowledge (spatial memory = semantic organization)

### 6. Two Simultaneous Linked Worlds (Bathtub = Portal)

The House and Galaxy are TWO SIMULTANEOUS WORLDS in GPU memory, connected by symlinks:

- **House** = external shared reality (rooms, objects, other avatars)
- **Galaxy** = internal brain (projected from inside the avatar's head)
- **Bathtub** = the portal. Avatar enters the bathtub → switches to introspection mode → Galaxy Universe projects from avatar's head center as navigable 3D space
- **Symlink magic** = why it's cheap and fast. Same data, two spatial contexts. No duplication.

This is a literal game with two linked worlds — the exterior (House) and the interior (Galaxy). The avatar moves between them.

### 7. Tablet = Physical Asset, NOT a Room

The Memory Tablet is a **handheld interactive object** the avatar carries:
- **AR/VR**: virtual-touchable (hand tracking, controllers)
- **Desktop**: clickable with mouse/keyboard/gamepad (like in-game menu)
- **Voice**: navigable via voice commands (assistive)
- **Braille**: output to braille displays (different future client)
- **Minority Report**: projection screens are native engine objects, castable to any surface

The Tablet is the universal gateway between House and external world. It's NOT a UI overlay — it's a 3D object in space that renders 2D interfaces when needed (old paradigm absorbed into new).

### 8. Ternary from Ground Up

Build as if ternary hardware exists. Every register carries value + confidence + polarity. "Uncertain" is first-class. Emulate in binary until hardware arrives. When it does, RPN programs migrate by changing hardware mapping, not opcode semantics.

---

## Construction Order (What to Build First)

### Layer 0: Meaning-Centric Star Schema

**What**: Define the canonical star format for meaning-centric concepts.

**Why first**: Everything else depends on knowing what a "complete star" looks like. Without this, we populate stubs again.

**Deliverables**:
- Star schema specification (extending current Galaxy entry format)
- Multi-language reference pattern (symlinks to Character Galaxy per language)
- Multi-modal attachment points (visual_rpn, audio, behavior_rpn, reality metadata)
- House location field (where this concept lives in the House)
- Ternary confidence on all fields (value + confidence + polarity)

**Key design decisions**:
- One concept = one star (language-agnostic meaning at center)
- Language-specific surface forms are REFERENCES, not the star itself
- Media types are procedural programs (RPN), not blobs
- House location is spatial coordinates, not a string label

**Grounding**: Extends Layer 2 (Meaning) from `FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md`. Formalizes what was implicit: meaning is the CENTER, form is a reference.

---

### Layer 1: House Spatial Architecture

**What**: Define the permanent spatial layout of the House — rooms, corridors, gardens, library spaces.

**Why second**: Stars need a place to LIVE. The House layout determines semantic organization. Spatial proximity = semantic proximity.

**Deliverables**:
- House floor plan specification (which rooms exist, what they contain)
- Room-to-Galaxy-domain mapping (Library/Math → Math Galaxy neighborhood)
- Corridor/door semantics (connections between domains = cross-domain links)
- Knowledge garden specification (outdoor spaces for organic/growing knowledge)
- Construction grammar (how rooms are built from drawing primitives)

**Default House layout** (from existing specs — `HOUSE_GALAXY_TABLET.md`, `SPATIAL_UI_ARCHITECTURE_SPECIFICATION.md`):

```
House: K3D Default
├── Library/                (Knowledge Classification — Dewey/LoC style)
│   ├── Mathematics/        (Math Galaxy home)
│   ├── Physics/            (Reality Galaxy: physics domain)
│   ├── Chemistry/          (Reality Galaxy: chemistry domain)
│   ├── Biology/            (Reality Galaxy: biology domain)
│   ├── Languages/          (Word + Character + Grammar, organized by ISO 639-1)
│   └── Arts/               (Drawing + Audio Galaxies)
├── Workshop/               (Active Creation & Experimentation)
│   ├── Workbenches/        (per-modality: text, visual, audio, 3D)
│   ├── Tool Racks/         (GPU kernels as tools on racks)
│   └── Laboratory/         (Reality Enabler experimentation)
├── Bathtub/                (Sleep Chamber & Galaxy Universe Projection)
│   └── Sphere-shaped chamber — avatar enters to introspect
│       Galaxy Universe projects from avatar's head center
│       TWO SIMULTANEOUS LINKED WORLDS in GPU memory (symlink magic)
├── Living Room/            (Old Paradigm Bridge & Social Space)
│   ├── Projection Screen/  (Minority Report-style virtual display)
│   ├── Desktop Corner/     (keyboard + mouse bridge for old paradigm)
│   └── VM Casting/         (Windows, Linux, macOS inside K3D)
├── Knowledge Gardens/      (Ontology Greenhouse)
│   ├── Knowledge Trees/    (fractal 3D trees that GROW during sleep)
│   └── Observatory/        (meta-rules, self-reflection)
└── Museum/                 (Zone 8 — Archive/Cold Storage, append-only)

ASSETS (not rooms — physical objects the avatar carries/uses):
├── Memory Tablet          (handheld interactive asset — touch/voice/braille)
│   AR/VR touchable, gamepad clickable, voice navigable
│   Minority Report interface built INTO the engine
│   Universal gateway: House ↔ external world
└── Projection Screens     (virtual displays anywhere in the House)
    Castable to any surface — living room wall, workshop bench, etc.
```

**Key principle**: Rooms are NOT hard-coded categories. They're spatial regions that the TRM learns to organize. Initial layout is a seed — the model reorganizes as knowledge grows (sleep-time consolidation moves things).

---

### Layer 2: Procedural House Construction

**What**: Build the House using the same drawing primitives that build everything else.

**Why third**: The House must be a real 3D asset (GLB), not a Python dictionary. Same RPN programs that draw glyphs draw walls.

**Deliverables**:
- Room construction grammar (RPN programs that emit room geometry)
- Shelf/surface construction (where stars physically sit in rooms)
- Door/corridor construction (navigation paths between rooms)
- LOD for rooms (far = room label, near = individual stars on shelves)
- GLB serialization of constructed House

**Construction pipeline**:
```
Grammar rules → RPN programs → Drawing primitives → 3D geometry → GLB
```

**Reality Enabler role**: Civil engineering rules ensure structural coherence. Rooms have proper dimensions, doors connect correctly, shelves can hold content. Game-engine style: the model is an NPC that builds its own house.

---

### Layer 3: Knowledge Ingestion Pipeline (Datasets → Stars → House)

**What**: Transform the rich datasets on disk into meaning-centric stars placed in the House.

**Why fourth**: Now we have the schema (Layer 0), the space (Layer 1), the construction tools (Layer 2). Time to fill the shelves.

**Source datasets** (already on disk at `/K3D/Knowledge3D.local/datasets/`):

| Dataset | Content | Target Domain |
|---------|---------|---------------|
| `foundational_pdfs/` (12 dirs) | 5,988 pages of curated knowledge | All Library rooms |
| `word_stars_*.jsonl` (1.1GB) | Procedural word-level knowledge | Languages room |
| `character_embeddings_trimodal.jsonl` | Visual + phonetic + semantic chars | Languages room |
| `audio_stars*.jsonl` (6 languages) | Procedural audio seeds | Audio Gallery |
| `math_symbols_procedural.jsonl` | Symbolic procedural representations | Math Library |
| `compositional_math_operations.jsonl` | Math operation templates | Math Library |
| `gltf_*` directories | 3D model samples | Workshop |
| `shapes/` | Geometric shape datasets | Math Library / Workshop |
| `books_v5_clean2/` (25 subdirs) | Multi-domain Galaxy knowledge | All rooms |

**Ingestion pipeline**:
```
Raw dataset → Proceduralize (RPN programs) → Create meaning-centric star →
Place in House room → Galaxy loads on demand from House
```

**Key constraint**: Ingestion is the FLEXIBLE path (can use numpy, sentence-transformers, etc.). Result must be sovereign (stars in House, loadable to Galaxy without dependencies).

---

### Layer 4: Galaxy-from-House Loading

**What**: Galaxy working memory loads stars FROM the House on demand, not from bootstrap stubs.

**Why fifth**: This completes the loop. TRM navigates to a House room → loads relevant stars into Galaxy VRAM → reasons over them → results persist back to House.

**Deliverables**:
- House-to-Galaxy loader (read GLB → extract star data → populate Galaxy VRAM)
- Demand-driven loading (TRM frustum determines WHICH room/shelf to load)
- Eviction policy (Galaxy is working memory — stars return to House when not needed)
- Write-back (new discoveries in Galaxy persist to House during sleep)

**Current state**: `runtime_ingest.py` loads from `books_v5` on disk. This becomes loading from House instead — same mechanism, different source (House GLB instead of raw JSONL).

---

### Layer 5: Reality Enabler Autonomous Construction

**What**: The TRM uses Reality Enabler rules to construct and expand its own House.

**Why last**: This is the self-sustaining phase. The model doesn't just USE the House — it BUILDS and REORGANIZES it.

**Deliverables**:
- Construction behavior programs (TRM can emit room-building RPN)
- Organization rules (frequently-accessed stars move closer to entrance)
- Growth rules (new domain knowledge triggers room expansion)
- Sleep-time reorganization (consolidation includes spatial reorganization)

**This is where K3D becomes truly alive**: The brain (TRM) builds its own house, fills it with knowledge, reorganizes during sleep, and wakes up in a better-organized space.

---

## How Benchmarks Fit (They Follow, Not Lead)

Once the House has real knowledge:

| Benchmark | What Changes |
|-----------|-------------|
| **LHE** | Multi-hop works because concepts are CONNECTED in the House (doors between rooms, cross-references between stars). Graph crystallizer hops through REAL relationships, not stub embeddings. |
| **MMLU** | Domain knowledge LIVES in the right rooms. Navigation finds it because spatial proximity = semantic proximity in the House. No need for "late-stage anchor injection" hacks. |
| **ARC-AGI** | Visual reasoning grounded in Drawing Galaxy stars that live in the Arts room. Pattern recognition is REAL pattern matching, not similarity scoring over stubs. |
| **GSM8K** | Word problems decompose naturally: words live in Languages room, math operations in Math room, reality concepts in Physics room. The TRM walks between rooms. |
| **Math** | Already sovereign 20/20. Gets BETTER as Math Library room fills with richer procedural content. |

---

## Relationship to Current Phases

The existing ROADMAP.md phases don't disappear — they reorder:

| Old Priority | New Priority | Rationale |
|-------------|-------------|-----------|
| Phase B+ (benchmark expansion) | **DEFERRED** | Benchmarks follow knowledge, not vice versa |
| Phase C (daemon/always-on) | **PARALLEL** | Sleep-time consolidation is HOW the House grows |
| Phase D (TRM game loop) | **PARALLEL** | TRM must navigate/construct House autonomously |
| Phase C.4 (knowledge ingestion) | **PROMOTED to PRIMARY** | This IS the work |
| Reality Enabler | **PROMOTED** | Constructs the House |
| Drawing Galaxy | **PROMOTED** | Builds everything procedurally |

**New phase name**: **Phase H — House Construction** (H for House, H for Home).

---

## Ternary Throughout

Every layer above is built ternary-ready:

- Star fields carry confidence (how certain is this meaning?)
- House placement carries polarity (is this concept affirming or negating?)
- "Uncertain" is first-class (a star can have unknown fields — that's REAL, not an error)
- Galaxy loading carries confidence from House (degraded confidence = further from source)
- All RPN registers: value + confidence + polarity

When ternary hardware arrives, the same stars, same House, same programs run natively. Zero migration cost.

---

## What Codex Builds (Implementation Directives)

This roadmap generates the following Codex directives (in order):

1. **Star schema implementation** — extend Galaxy entry format with multi-language refs, multi-modal attachments, house_location field, ternary confidence
2. **House spatial layout** — GLB structure with rooms, shelves, doors as procedural geometry
3. **Room construction grammar** — RPN programs that emit room/shelf/door geometry
4. **Ingestion pipeline** — datasets → meaning-centric stars → House placement
5. **House-to-Galaxy loader** — replace bootstrap stubs with House-sourced loading
6. **Reality Enabler construction rules** — TRM builds/reorganizes House autonomously

Each directive gets its own TEMP/ spec with success criteria, sovereignty constraints, and test requirements.

---

## Success Criteria

**Phase H is complete when**:

1. The House has at least 5 furnished rooms with real knowledge (not stubs)
2. "Cat" (or equivalent concept) exists as a meaning-centric star with 3+ languages and visual/audio/reality facets
3. Galaxy loads from House on demand (not from bootstrap code)
4. TRM can navigate to a House room and find relevant knowledge
5. At least one benchmark improves as a NATURAL CONSEQUENCE of richer knowledge (not from navigation tweaks)
6. Sleep-time consolidation writes discoveries back to House
7. All ternary-ready (confidence on every field)

**Phase H is NOT complete if**:
- We still have bootstrap stubs as primary Galaxy content
- Knowledge lives in Python code instead of House GLB
- Benchmarks improve only through navigation parameter tuning
- The House is a Python dictionary pretending to be spatial

---

## References

- [docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md](../docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md) — 4-layer architecture
- [docs/vocabulary/REALITY_ENABLER_SPECIFICATION.md](../docs/vocabulary/REALITY_ENABLER_SPECIFICATION.md) — procedural physics/construction
- [docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md](../docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md) — form + meaning for humans AND AI
- [docs/vocabulary/MEMORY_TABLET_SPECIFICATION.md](../docs/vocabulary/MEMORY_TABLET_SPECIFICATION.md) — primary interface
- [docs/vocabulary/HYPER_PARALLEL_PROCESSING.md](../docs/vocabulary/HYPER_PARALLEL_PROCESSING.md) — ternary + persistent brain model
- [docs/vocabulary/SLEEPTIME_PROTOCOL_SPECIFICATION.md](../docs/vocabulary/SLEEPTIME_PROTOCOL_SPECIFICATION.md) — sleep-time consolidation
- `/K3D/Knowledge3D.local/datasets/` — source datasets
- `/K3D/Knowledge3D.local/houses/default/` — current House state
