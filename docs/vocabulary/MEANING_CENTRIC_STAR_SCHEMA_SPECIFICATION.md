# Meaning-Centric Star Schema Specification — Semantic Gravity & Ternary Force

**Version**: 1.0
**Status**: Architecture Specification
**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Implementation)
**Date**: March 2026

---

## Abstract

This specification defines the **Meaning-Centric Star** — the atomic unit of knowledge in K3D. A star represents a **concept**, not a word. "Cat" is cat in every language; the meaning is the center, language-specific surface forms are references. Stars carry all modalities (visual, audio, behavioral, linguistic) as procedural RPN programs.

Stars live in the **House** (permanent 3D knowledge repository) and are loaded into the **Galaxy** (ephemeral VRAM working memory) when the AI needs to reason about them. In the Galaxy, **semantic gravity cohered by meaning** (Christoph Dorn) organizes loaded concepts: a ternary force operator borrowed from physics where meaning replaces mass and the ternary operator (+1/0/−1) replaces the gravitational constant. Stars attract (semantic affinity), repel (semantic contradiction), or float neutrally (no evidence of relationship). The House has INTENTIONAL physical organization — the TRM places stars deliberately, like a librarian shelving books. Semantic gravity operates in the Galaxy during reasoning, not in the House.

**Key relationship to existing specs:**
- Extends `REALITY_ENABLER_SPECIFICATION.md` §3 (dual-program stars) to ALL knowledge, not just physics/chemistry
- Implements `FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md` Layer 2 (Meaning) as the canonical center
- Realizes `DUAL_CLIENT_CONTRACT_SPECIFICATION.md` (same star serves humans AND AI)
- Integrates `HYPER_PARALLEL_PROCESSING.md` §6 (ternary logic as force operator)

---

## 1. Philosophy: Meaning at the Center

### 1.1 The Problem with Language-Centric Knowledge

Traditional AI knowledge stores organize by language surface: "cat" is an English entry, "gato" is a Portuguese entry, "猫" is a Japanese entry. Three entries for ONE concept. This causes:

- **Duplication**: same meaning stored N times per language
- **Translation gap**: cross-language reasoning requires alignment models
- **Modality blindness**: the sound of a cat, the visual shape, the behavioral patterns are disconnected from the word
- **No spatial grounding**: concepts don't LIVE anywhere

### 1.2 The K3D Solution: Meaning IS the Star

A Meaning-Centric Star has ONE center: the **concept** (Layer 2: Meaning). Everything else — words, glyphs, sounds, visuals, behaviors — are **references** attached to that center:

```
                        ┌─────────────────┐
                        │   MEANING        │
                        │   (concept_cat)  │
                        │                  │
                        │  "small domest.  │
                        │   feline"        │
                        └────────┬─────────┘
                                 │
          ┌──────────┬───────────┼───────────┬──────────┐
          │          │           │           │          │
     ┌────▼───┐ ┌───▼────┐ ┌───▼────┐ ┌───▼────┐ ┌───▼────┐
     │ FORM   │ │ FORM   │ │ VISUAL │ │ AUDIO  │ │BEHAVIOR│
     │ en:cat │ │ pt:gato│ │ RPN    │ │ RPN    │ │ RPN    │
     │ →[c,a,t│ │→[g,a,t,│ │(draw   │ │(meow   │ │(quadru-│
     │  ]     │ │  o]    │ │ silh.) │ │ spect.)│ │ ped    │
     └────────┘ └────────┘ └────────┘ └────────┘ │locomot)│
                                                  └────────┘
```

The meaning star IS the concept. Language surface forms, visuals, sounds, and behaviors are all references — symlinks to existing Galaxy entries (Character Galaxy, Drawing Galaxy, Audio Galaxy, Reality Galaxy).

### 1.3 "Cat is Cat Any Language"

This is not metaphor — it's architecture. The star `concept_cat` exists once. When the TRM loads it into Galaxy working memory, ALL facets come with it. Reasoning about cats uses the MEANING, not the English word "cat". This makes K3D reasoning **language-agnostic by construction**.

---

## 2. Star Schema

### 2.1 Core Fields

Every Meaning-Centric Star MUST contain:

```
MeaningCentricStar {
    // === IDENTITY ===
    star_id:        ContentHash     // Deterministic from meaning (no UUID randomness)
    meaning_class:  Enum            // concept | relation | action | property | meta

    // === MEANING (Layer 2 — THE CENTER) ===
    meaning_rpn:    RPN_Program     // Procedural definition of WHAT this concept IS
                                    // NOT a text string — a program that computes meaning
    domain:         DomainPath      // Library/Biology/Mammalia (House location hint)
    taxonomy_refs:  [StarRef]       // Parent/child in ontology (mammal → animal → living_thing)

    // === FORM (Layer 1 — Language References) ===
    surface_forms: {
        "en": { word_ref: StarRef, char_refs: [StarRef] },  // → Word Galaxy, Character Galaxy
        "pt": { word_ref: StarRef, char_refs: [StarRef] },
        "ja": { word_ref: StarRef, char_refs: [StarRef] },
        // ... any ISO 639-1 language
    }

    // === VISUAL (Drawing Galaxy Reference) ===
    visual_rpn:     RPN_Program     // Procedural drawing (how it LOOKS)
    visual_refs:    [StarRef]       // Symlinks to Drawing Galaxy entries used

    // === AUDIO (Audio Galaxy Reference) ===
    audio_rpn:      RPN_Program     // Procedural sound (how it SOUNDS)
    audio_refs:     [StarRef]       // Symlinks to Audio Galaxy entries
    pronunciations: {
        "en": AudioRef,             // "kæt" as spectral RPN
        "pt": AudioRef,             // "ga.tu"
        // ... per language
    }

    // === BEHAVIOR (Reality Enabler Reference) ===
    behavior_rpn:   RPN_Program     // How it BEHAVES (physics, biology, interaction rules)
    reality_refs:   [StarRef]       // Symlinks to Reality Galaxy entries

    // === RULES (Layer 3 — Grammar References) ===
    grammar_refs:   [StarRef]       // Transformation rules that apply to this concept

    // === META-RULES (Layer 4) ===
    meta_refs:      [StarRef]       // Meta-rules (pedagogy, eloquence, self-reflection)

    // === SPATIAL (House Location) ===
    house_position: Vec3            // Where this star LIVES in the House (3D coordinates)
    house_room:     RoomRef         // Which room (derived from gravitational clustering)

    // === TERNARY STATE ===
    confidence:     Trit            // +1 = established, 0 = uncertain, −1 = contested
    polarity:       Trit            // +1 = affirming, 0 = neutral, −1 = negating

    // === EMBEDDINGS (Matryoshka LOD) ===
    embeddings: {
        tier_64:    Vec64           // Coarse search key
        tier_128:   Vec128          // Structural similarity
        tier_512:   Vec512          // Fine-grained matching
        tier_2048:  Vec2048         // Full fidelity (regenerable from meaning_rpn)
    }

    // === COMPOSITION ===
    component_refs: [StarRef]       // What this star is MADE OF (atoms → molecules pattern)
    composite_of:   [StarRef]       // What higher-level stars include this one
}
```

### 2.2 Field Semantics

**`meaning_rpn` (THE essential field)**:
This is NOT a text definition like "a small domesticated feline". It's an RPN program that COMPUTES the concept's meaning — its properties, relationships, constraints. The text description is for humans; the RPN program is for the TRM.

Example for `concept_cat`:
```rpn
# Properties: mass range, leg count, class, diet type
4.0 STORE_mass_kg
4 STORE_leg_count
GALAXY_LOOKUP mammalia STORE_class
GALAXY_LOOKUP carnivore STORE_diet
# Relationships: domesticated, companion to humans
GALAXY_LOOKUP domesticated TQUANT  # ternary: +1 = yes
GALAXY_LOOKUP companion TQUANT     # ternary: +1 = yes
```

**`star_id` (Content-addressed)**:
Derived from `meaning_rpn` content hash, not random UUID. Two systems that independently define the same concept produce the SAME star_id. This enables deduplication and federation.

**`surface_forms` (Symlinks, NOT copies)**:
Each language entry points to existing Word Galaxy and Character Galaxy stars. Adding a new language to a concept is ONE new symlink, not a new star.

**`house_position` (Intentional, physically organized)**:
Position in the House is DELIBERATE — the TRM (or construction rules) PLACES knowledge where it belongs, like a librarian shelving a book. The House is a literal 3D virtual space with rooms, shelves, and objects. Semantic gravity (§3) operates in the GALAXY (working memory), not in the House. The House has physical, ontological, intentional organization.

### 2.3 Meaning Classes

| Class | Description | Example |
|-------|-------------|---------|
| `concept` | A thing, entity, category | cat, hydrogen, democracy |
| `relation` | A relationship between concepts | "is-a", "causes", "part-of" |
| `action` | A process or transformation | "run", "oxidize", "divide" |
| `property` | An attribute of concepts | "red", "heavy", "soluble" |
| `meta` | A rule about rules | "prefer specific over general", "check conservation" |

---

## 3. Two Worlds, Two Organizations

### 3.1 Critical Distinction: House ≠ Galaxy

The House and Galaxy organize knowledge in fundamentally DIFFERENT ways:

| Aspect | House (External Reality) | Galaxy (Internal Brain) |
|--------|------------------------|------------------------|
| **Organization** | Intentional, physical, ontological | Fluid, neural, gravitational |
| **Analogy** | Library with shelves, books, rooms | Mind's eye, thought-space |
| **Placement** | TRM PLACES knowledge deliberately | Concepts attract/repel by meaning |
| **Structure** | Rooms, corridors, shelves, objects | Neighborhoods, clusters, fields |
| **Persistence** | Permanent (GLB on disk) | Ephemeral (VRAM working memory) |
| **Who organizes** | Librarian (TRM + construction rules) | Physics (semantic gravity) |

**The House is a literal 3D virtual world** — software as a space. The IT industry borrowed spatial metaphors (windows, doors, folders, desktop). K3D reverses this: makes them ACTUAL 3D objects in a navigable virtual world. Every technology the game industry has achieved powers this.

**The Galaxy is working memory** — a multi-modal modular memory space designed for next-generation AI, enabling Explainable AI (ExAI) in a way humans and AI can share.

### 3.2 House Organization: Intentional & Physical

The House has rooms designed like real-world spaces. A library HAS bookshelves with books. A garden HAS trees whose branches and leaves carry knowledge details. You WALK to the shelf, OPEN the book, READ it.

**Example — "Cat" in the House:**
- **Lives on a shelf** in Library/Biology/Mammalia
- The shelf is a physical 3D object (procedural geometry)
- Next to it: "Dog", "Horse", "Whale" — other mammals
- The BOOK "Felidae" on the shelf contains detailed feline taxonomy
- The knowledge TREE in the Garden has a "Mammalia" branch where "Cat" is a leaf
- BOTH representations exist: the tree shows the ontological structure (visual, navigable), the book provides detailed reference content
- A researcher AI can walk to the Library, find the book, and reference it in a scientific project

**The same underlying knowledge, two physical representations:**
1. **Knowledge Trees** (Garden) — ontological, hierarchical, growing, visual
2. **Library Books** (Library) — reference, detailed, consultable, citable

This is NOT metaphor. These are actual 3D objects the avatar interacts with.

### 3.3 House Room Design (From Existing Specs)

Rooms are architecturally designed spaces, not emergent clusters:

- **Library**: Dewey/LoC organized shelves. Books = consolidated knowledge. Reference works = dictionaries, lexicons. Stars sit on physical shelves in their ontological location.
- **Workshop**: Workbenches per modality. Tool racks = GPU kernels as pickable tools. Active experimentation space.
- **Bathtub**: Sleep chamber. Sphere-shaped introspection portal. Avatar enters → Galaxy Universe projects from head center. The portal between House (external) and Galaxy (internal).
- **Living Room**: Projection screens (Minority Report-style). Desktop corner for old-paradigm bridge. VM casting.
- **Knowledge Gardens**: Actual 3D trees. Roots = foundational concepts. Trunk = core domains. Branches = specializations. Leaves = specific knowledge. Trees GROW during sleep as knowledge consolidates.
- **Museum**: Archive. Deprecated/superseded artifacts. Historical audit trail.

### 3.4 Galaxy Organization: "Semantic Gravity Cohered by Meaning" (Ternary Force)

> **"Semantic gravity cohered by meaning"**
> — Coined by Christoph Dorn (March 16, 2026), from an idea by Daniel Ramos, developed in collaboration with Claude

**Inside the Galaxy** (the AI's working memory), semantic gravity cohered by meaning governs how loaded concepts organize for active reasoning:

Classical gravity:
```
F = G × (m₁ × m₂) / r²
```

Semantic gravity (Galaxy working memory):
```
F = T(s₁, s₂) × M(s₁) × M(s₂) / d²
```

Where:
- `T(s₁, s₂)` = **ternary semantic operator** = {+1, 0, −1}
  - +1 = semantic affinity (attract) — "cat" and "mammal" pull together in thought
  - 0 = no semantic evidence (neutral) — "cat" and "integer" don't interact
  - −1 = semantic contradiction (repel) — "alive" and "dead" push apart
- `M(s)` = **meaning mass** = richness/connectedness of the star
- `d` = **spatial distance** in Galaxy coordinates (VRAM working memory, NOT House)
- `F` = resulting force vector

### 3.5 Ternary Operator: The Heart of the Force

The ternary operator `T(s₁, s₂)` is computed from the stars' `meaning_rpn` programs via existing ternary opcodes:

```rpn
# Compute ternary semantic force between two stars
s1.meaning_rpn RECALL
s2.meaning_rpn RECALL
TCOMP                    # Ternary compare: +1 (compatible), 0 (unknown), −1 (contradictory)
```

This uses the existing `OP_TCOMP` (0x73) opcode — already implemented in PTX:

| T value | Meaning | Force | Example |
|---------|---------|-------|---------|
| +1 | Semantic affinity | **Attract** | cat ↔ mammal, derivative ↔ calculus |
| 0 | No evidence | **Neutral** | cat ↔ integer (no relationship known) |
| −1 | Semantic contradiction | **Repel** | alive ↔ dead, true ↔ false |

### 3.6 Meaning Mass

A star's "mass" in the Galaxy gravity field is its **richness**:

```
M(s) = |surface_forms| + |visual_refs| + |audio_refs| + |reality_refs|
      + |component_refs| + |composite_of| + |grammar_refs|
```

Rich concepts are heavy — they attract more strongly and become thought-centers during active reasoning.

### 3.7 Galaxy Dynamics (During Reasoning)

When the TRM loads stars from the House into Galaxy for reasoning:

1. Stars enter Galaxy at initial positions (derived from House room structure)
2. Semantic gravity organizes them in working memory — related concepts cluster for efficient access
3. The Nine-Chain Swarm processes clusters in parallel
4. New provisional stars form at positions influenced by their semantic neighborhood
5. After reasoning completes, stars return to their House positions (unmodified)

**Key difference from House**: Galaxy organization is TEMPORARY and FLUID. Stars don't permanently move. The Galaxy reorganizes every time new stars are loaded. The House stays stable.

### 3.8 Sleep-Time: Galaxy Informs House

During sleep-time consolidation, semantic gravity results INFORM (but don't dictate) House organization:

1. Galaxy gravity reveals which concepts cluster naturally during reasoning
2. The TRM uses this information to SUGGEST reorganization of the House
3. Frequently co-loaded stars might be placed on adjacent shelves
4. New discoveries get placed by the TRM (like a librarian shelving a new book)
5. Knowledge trees in the Garden grow new branches/leaves based on consolidated insights

The TRM acts as **librarian**: it uses Galaxy reasoning patterns to make deliberate organizational decisions about the House. Semantic gravity is a tool for the librarian, not an autonomous force reshaping the shelves.

### 3.9 The Vision: Houses for Every Context

The House paradigm extends beyond a single default:

- **Personal AI House**: Digital copy of the user's home. Their AI lives here. Personal knowledge on personal shelves.
- **Classroom AI House**: The classroom IS the House. School AI creates 3D demonstrations with AR. No more "imagine the pulley" — the AI builds it in 3D, right there in the room.
- **Service AI**: Works WITH the user's personal AI through the Tablet. Service AI enters the user's House to collaborate.
- **Specialist AI**: Has a Workshop-heavy House. Tools and domain expertise on its workbenches.

**For v1.0**: One standard default House. The K3D brain model crafts it. This is the product that labs build on.

The Tablet eliminates the need for any other interface. Service AI collaborates with personal AI through it — the Tablet IS the universal gateway. In a classroom, the teacher's AI and students' AIs share a House view, with the teacher's AI demonstrating concepts as literal 3D constructions.

---

## 4. Relationship to Galaxy (Working Memory)

### 4.1 House vs Galaxy

| Aspect | House (Software as Space) | Galaxy (AI Working Memory) |
|--------|--------------------------|---------------------------|
| Persistence | Permanent (GLB on disk) | Ephemeral (VRAM) |
| Role | Where stars LIVE as 3D objects | Where stars are THOUGHT ABOUT |
| Organization | Intentional, physical (rooms, shelves) | Fluid, gravitational (semantic gravity) |
| Capacity | Unlimited (disk) | Limited (VRAM budget) |
| Access | On-demand loading | Always-loaded subset |
| Modification | TRM deliberate placement (librarian) | Read-write during inference |
| Representation | Literal 3D objects (books, trees, tools) | Abstract multi-modal working memory |
| Human access | Walk through rooms, open books, touch objects | View via Bathtub introspection projection |
| Analogy | Your actual house/library/workshop | Your mind's eye while thinking |

### 4.2 Loading: House → Galaxy

When the TRM navigates to a House room (frustum culling + LOD):

1. Frustum determines which room/shelf is in view
2. LOD determines detail level (room label → shelf → individual stars)
3. Stars in view are loaded from House (GLB) into Galaxy (VRAM)
4. Galaxy entries are the SAME stars — same star_id, same fields
5. In Galaxy, semantic gravity organizes them for efficient reasoning
6. TRM reasons over Galaxy entries (meaning_rpn execution, composition, creation)
7. When reasoning completes, stars remain in their House positions (House is stable)

### 4.3 Write-Back: Galaxy → House

When inference creates new knowledge:

1. New star created in Galaxy (working memory) during reasoning
2. Marked as `provisional` (confidence = 0, uncertain)
3. During sleep-time consolidation:
   a. Provisional stars are evaluated (does this new knowledge hold up?)
   b. Confirmed stars get `confidence = +1` and are written to House
   c. Rejected stars get `confidence = −1` and are discarded or sent to Museum
   d. Uncertain stars remain provisional for next sleep cycle
4. TRM places written-to-House stars in their appropriate room/shelf (informed by Galaxy gravity patterns, but placed deliberately)

### 4.4 Two Simultaneous Worlds (Symlink Architecture)

The House and Galaxy coexist in GPU memory via symlinks:

```
House (external reality)     Galaxy (internal brain)
┌──────────────────┐        ┌──────────────────┐
│ Library room     │        │ Math neighborhood │
│  ├── shelf_calc  │◄──────►│  ├── derivative   │ ← SAME star_id
│  │   ├── deriv.  │symlink │  ├── integral     │ ← SAME star_id
│  │   ├── integ.  │        │  ├── [working]    │ ← provisional
│  └── shelf_alg   │        │  └── limit        │
└──────────────────┘        └──────────────────┘
```

The Bathtub is where these worlds overlap visually — the avatar enters introspection mode and SEES the Galaxy projected from their head, overlaid on the House structure. Two worlds, one GPU, symlink pointers between them.

---

## 5. Star Lifecycle

### 5.1 Creation (Ingestion Path)

```
External data → Proceduralize → Create meaning_rpn → Compute star_id →
Attach surface_forms (per language) → Attach visual_rpn → Attach audio_rpn →
Attach behavior_rpn → Compute embeddings → TRM places in House
(like a librarian shelving a new book in the right section)
```

**Sovereignty boundary**: Ingestion can use any tools (numpy, sentence-transformers, etc.). The RESULT must be a sovereign star (RPN programs, Galaxy references, no external dependencies at runtime).

### 5.2 Placement (Deliberate, NOT Gravitational)

```
New star created → TRM determines appropriate House location:
  - Domain hint ("Library/Biology/Mammalia") → find the right shelf
  - Taxonomy refs → place near parent/sibling concepts
  - Physical organization rules → books next to related books
Star is PLACED on a shelf, in a book, as a leaf on a knowledge tree
```

The House is organized like a REAL library — with intention, ontology, and physical logic. Not by gravitational drift. A concept about cats goes on the biology shelf because a librarian (the TRM) put it there.

**Dual representation**: The same star may appear in multiple House locations:
- As a **leaf** on a knowledge tree in the Garden (ontological view)
- As an **entry** in a book on a Library shelf (reference view)
- As a **tool** on a Workshop bench (if it's actionable)
These are symlink references — one star, multiple physical representations.

### 5.3 Recall (Inference)

```
TRM navigates House → Frustum culls to relevant room → LOD selects detail →
Star loaded from House shelf into Galaxy VRAM →
In Galaxy, semantic gravity organizes stars for reasoning →
TRM executes meaning_rpn → Reasoning proceeds →
Galaxy clears after reasoning (ephemeral working memory)
House remains unchanged (stable, persistent)
```

### 5.4 Evolution (Discovery)

```
TRM reasoning creates new insight → New provisional star in Galaxy →
Sleep-time evaluation → Confirmed? →
  Yes: TRM places confirmed star in House (deliberate shelf location)
       Knowledge tree in Garden grows a new leaf
  No:  Star discarded or sent to Museum with "unconfirmed" tag
```

### 5.5 Deprecation

```
Star contradicted by new evidence → confidence = −1 →
Sleep-time review → TRM moves star to Museum (cold storage)
Like a librarian moving an outdated book to the archive
Still accessible if specifically sought, but not on active shelves
```

---

## 6. Ternary Throughout

Every field in the star schema carries ternary semantics:

| Field | Value | Confidence | Polarity |
|-------|-------|-----------|----------|
| meaning_rpn | The program | How established (+1/0/−1) | Affirming/neutral/negating |
| surface_forms | Language refs | Source quality | — |
| visual_rpn | Drawing program | Visual accuracy | — |
| behavior_rpn | Behavior program | Empirical validation | — |
| house_position | Coordinates | Settlement stability | — |
| taxonomy_refs | Ontology links | Classification certainty | Is-a (+1) / is-not (−1) |

**Ternary registers**: When a star is loaded into Galaxy, its fields populate STORE/RECALL registers as ternary triples (value + confidence + polarity), per `RPN_DOMAIN_OPCODE_REGISTRY.md` §7.

**Halting gate integration**: The Nine-Chain Swarm's halting gate uses star confidence to weight votes. High-confidence stars contribute more to convergence; uncertain stars are noted but don't force decisions.

---

## 7. Matryoshka Embeddings (Search LOD)

Stars carry Matryoshka embeddings at 4 tiers, regenerable from `meaning_rpn`:

| Tier | Dimensions | Purpose | Example Use |
|------|-----------|---------|-------------|
| 64D | Coarse | "Is this roughly relevant?" | Room-level navigation |
| 128D | Structural | "Is this the right category?" | Shelf-level filtering |
| 512D | Fine | "Is this the specific concept?" | Star-level matching |
| 2048D | Full | "Complete semantic fingerprint" | Deduplication, federation |

Embeddings are NOT the knowledge — they're search keys. The knowledge is in `meaning_rpn`. If embeddings are lost, they're regenerated from the procedural program.

---

## 8. Content-Addressed Identity

### 8.1 star_id Computation

```
star_id = ContentHash(meaning_rpn || meaning_class || domain)
```

Two independent systems that define the same concept (same meaning_rpn program) produce the SAME star_id. This enables:

- **Deduplication**: Same concept from different sources merges automatically
- **Federation**: Houses can share stars by star_id without centralized coordination
- **Integrity**: Changing meaning_rpn changes star_id (forces explicit versioning)

### 8.2 Versioning

When a star's meaning evolves:
1. Old star_id preserved in Museum (historical record)
2. New star created with updated meaning_rpn → new star_id
3. All references updated (composite_of stars recompute their own embeddings)
4. TRM places updated star (may keep same shelf or relocate based on changed meaning)

---

## 9. Integration with Existing Specifications

| Specification | Integration Point |
|--------------|-------------------|
| **REALITY_ENABLER** | Meaning-centric stars ARE reality nodes. `behavior_rpn` = Reality Enabler programs. Semantic gravity = Reality Enabler physics. |
| **FOUNDATIONAL_KNOWLEDGE** | Layer 1 (Form) = surface_forms. Layer 2 (Meaning) = meaning_rpn (THE CENTER). Layer 3 (Rules) = grammar_refs. Layer 4 (Meta) = meta_refs. |
| **DUAL_CLIENT_CONTRACT** | Same star serves humans (visual_rpn rendering) AND AI (meaning_rpn execution). |
| **THREE_BRAIN_SYSTEM** | House stores stars. Galaxy loads stars for reasoning. Cranium executes star programs. |
| **HYPER_PARALLEL_PROCESSING** | Ternary force operator. Specialist swarm processes multiple stars in parallel. |
| **SLEEPTIME_PROTOCOL** | Semantic gravity runs during sleep. Stars settle. New discoveries confirmed or rejected. |
| **TRM_SPECIALIST_MATRYOSHKA** | Specialists carry domain-specific LoRA adapters. Stars in their domain get bonus meaning mass from specialist expertise. |
| **KNOWLEDGEVERSE** | Stars populate the 7-region VRAM substrate. Galaxy entries ARE meaning-centric stars loaded from House. |
| **RPN_DOMAIN_OPCODE_REGISTRY** | All star programs use existing RPN opcodes. Ternary ops (TADD, TMUL, TNOT, TCOMP, TQUANT) compute semantic force. |

---

## 10. Implementation Priorities

### 10.1 Phase H.0: Star Schema (THIS SPEC)
- Define canonical star format (this document)
- Extend Galaxy entry dataclass with meaning-centric fields
- Implement content-addressed star_id
- Add ternary confidence/polarity fields

### 10.2 Phase H.1: Seed Stars + House Room Construction
- Create 10-20 anchor stars per domain (mathematics, physics, biology, language, tool, etc.)
- Build initial House room geometry using drawing primitives (Library shelves, Garden trees, Workshop benches)
- Place anchor stars DELIBERATELY in rooms (librarian placement, not gravity)
- Populate from existing `foundational_pdfs/` and `books_v5` datasets
- Dual representation: same star as tree-leaf (Garden) AND book-entry (Library)

### 10.3 Phase H.2: Semantic Gravity Engine (Galaxy Working Memory)
- Implement ternary force computation (TCOMP-based) for GALAXY organization
- Implement meaning mass calculation
- Galaxy uses semantic gravity to organize loaded stars during active reasoning
- Test: stars loaded from same House room cluster in Galaxy, cross-domain stars separate

### 10.4 Phase H.3: Knowledge Ingestion Pipeline
- Datasets on disk → proceduralized meaning-centric stars
- TRM placement rules: domain → room → shelf → position
- Knowledge tree growth: new stars = new leaves/branches in Garden
- Book consolidation: related stars → book entries in Library
- GLB serialization of populated House

### 10.5 Phase H.4: Galaxy-from-House Loading
- Replace bootstrap stubs with House-sourced loading
- Frustum-driven demand loading (TRM looks at shelf → stars load)
- Eviction policy for VRAM management (Galaxy is working memory)
- Write-back during sleep (new discoveries → TRM places in House)

### 10.6 Phase H.5: Multi-House Vision (Future)
- Personal AI houses (digital copy of user's home)
- Classroom AI houses (the classroom IS the House)
- Service AI collaboration via Tablet (enter user's House to help)
- Standard House template for new K3D instances

---

## 11. References

- `docs/vocabulary/REALITY_ENABLER_SPECIFICATION.md` — dual-program stars, transmutation
- `docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md` — 4-layer architecture
- `docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md` — form + meaning for humans AND AI
- `docs/vocabulary/HYPER_PARALLEL_PROCESSING.md` — ternary logic, persistent brain model
- `docs/vocabulary/SLEEPTIME_PROTOCOL_SPECIFICATION.md` — consolidation, write-back
- `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` — ternary opcodes, register semantics
- `docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md` — Cranium + Galaxy + House
- `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md` — 7-region VRAM substrate
- `docs/vocabulary/TRM_SPECIALIST_MATRYOSHKA_ARCHITECTURE.md` — specialist swarm
- `docs/HOUSE_GALAXY_TABLET.md` — House rooms, doors, portals
- `docs/vocabulary/SPATIAL_UI_ARCHITECTURE_SPECIFICATION.md` — room taxonomy
