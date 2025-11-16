# Spatial KR Visual Encoding Specification

This document centralizes how K3D encodes knowledge‐representation concepts into 3D visual structures (stars, rays, gardens, museums), so humans and AI agents can read the same spatial substrate consistently.

It is a companion to the formal vocabulary specs in `docs/vocabulary/` and to the W3C AI‑KR insertion documents. Those documents describe *what* we represent; this one describes *how* that representation appears and behaves in space and time.

---

## 1. Overview

- Galaxy view: nodes as “stars” in a continuous 3D embedding space (House → Galaxy).
- Garden view: ontology trees rendered as fractal plants in a circular “greenhouse” room.
- Museum view: archived knowledge in compact “boxes” and rooms, including portals to large, non‑loaded galaxies.

Every visual cue (shape, ray, color, animation) is tied to a KR concept:

- **Domain of discourse** → House / Room
- **Concept / node** → Star / Node shape
- **Relations / features** → Rays and edges
- **Modality** → Shape and ray palette
- **Adequacy / activity / time** → Brightness, temperature, animation

The goal is that a sighted human can “read” these meanings at a glance, while AI agents read the underlying embeddings, metadata, and logs.

---

## 2. Galaxy Stars and Rays

The Galaxy is the active memory: each K3D node becomes a star with attached rays.

### 2.1 Node Shapes (Modality Encoding)

Implementation reference: `viewer/src/shapes.ts`.

For each record `r: K3DRecord`, we derive a modality mask from metadata:

- `text` → tetrahedron
- `image` → cube
- `audio` → octahedron
- `video` → icosahedron
- multiple modalities → dodecahedron
- unknown / other → sphere

This mapping makes modality visually obvious and keeps Galaxy and Knowledge Garden consistent (Garden leaves reuse these cues).

### 2.2 Star Size and Local Density

- Base scale is computed from local nearest‑neighbor distances in 3D.
- Stars in dense regions are rendered slightly smaller; sparse regions slightly larger.
- This reflects local **semantic density** (how crowded the neighborhood is) without changing the underlying embeddings.

### 2.3 Rays: Direction, Length, Thickness, Style, Color

Each star may emit one or more rays. Rays encode relations and status:

- **Direction**: orientation toward meaningful targets (e.g., nearest neighbors, parents, prototypes), normalized to unit vectors.
- **Length**: strength and reach of the relation (longer = stronger influence or larger semantic span).
- **Thickness**: weight of the relation (e.g., frequency of use, aggregate evidence, or subtree mass).
- **Style**:
  - straight line → structural / ontological relation (e.g., “is‑a”, “part‑of”)
  - slightly curly → associative relation (e.g., “related‑to”, “similar‑to”)
  - highly curly → speculative / hypothesis relation (low confidence or provisional)
- **Color**:
  - base palette tracks modality (text/image/audio/video), as implemented today;
  - temperature overlay (hot / warm / cold) encodes recency and activity (see time encoding below).

This separation lets us express “what kind of relation,” “how strong,” “how recent,” and “in which modality” without overloading a single visual channel.

---

## 3. Time and Activity Encoding

Time is treated as an orthogonal dimension, not baked into embeddings.

Each node and relation carries timestamps:

- `created_at`
- `last_updated`
- `last_accessed`
- optional `half_life` or decay parameters

These drive:

- **Star emissive intensity / halo**:
  - recently updated or accessed → brighter, stronger halo;
  - long‑unused → dimmer, fading halo.
- **Ray temperature**:
  - hot (e.g., orange) for recently changed or frequently traversed relations;
  - cooling toward blue/grey as they age or fall out of use.
- **Animation** (optional):
  - subtle pulsing for “active” stars (frequent reads/writes);
  - slow drift or static for cold archival stars.

The combination gives a quick view of both **structure** (positions, rays) and **lifecycle** (what is alive, what is fading) without changing the underlying coordinate system.

---

## 4. Knowledge Garden (Ontology Greenhouse)

Reference: `docs/KNOWLEDGE_GARDENS.md`.

The Knowledge Garden is a canonical room in each House for long‑lived ontologies:

- circular floor with concentric zones;
- trees representing taxonomies/ontologies;
- leaves representing concrete nodes or media artifacts.

### 4.1 Zones and Domains

- Inner rings: core AI‑KR domains (e.g., upper ontologies, AI‑KR vocabularies).
- Middle rings: domain‑specific ontologies (e.g., neuroscience, reliability engineering).
- Outer rings: experimental, emerging, or per‑agent ontologies.

Zones auto‑expand as new domains appear, keeping layout stable but extensible.

### 4.2 Trees and Branches

As described in the Garden spec:

- Roots = ontology roots or high‑level concepts.
- Branches = intermediate concepts; geometry guided by embeddings and k‑NN structure.
- Leaves = concrete Nodes (documents, specs, examples) with modality shapes/colors linked to Galaxy stars.

Fractal growth (e.g., space colonization) is guided by:

- attraction points derived from child/neighbor embeddings;
- branch thickness proportional to subtree mass and similarity;
- alignment with principal semantic directions (PCA).

### 4.3 Reusing Star/Ray Semantics

Leaves and small branches re‑use Galaxy cues:

- leaf shape reflects modality;
- leaf color and glow reflect recency;
- small leaf‑to‑leaf connectors reuse ray style conventions (structural vs associative).

This ensures that the same semantics appear consistently in both Garden (hierarchical view) and Galaxy (metric view).

---

## 5. Museum and Portal Cubes

The Museum is where deprecated, superseded, or very large structures go:

- old Gardens, old Houses, old corpora;
- frozen Galaxies from past phases;
- large external collections.

### 5.1 Museum Rooms

- Each room groups artifacts by theme (e.g., “Phase G archives”, “Old W3C drafts”).
- Within a room, small 3D objects represent:
  - books → fixed documents;
  - trees → frozen ontologies;
  - **cubes** → portals to large archived Galaxies.

### 5.2 Portal Cubes (“Boxes of Galaxies”)

For knowledge sets too large or abstract to render directly:

- A cube Node in the Museum stands for an entire archived Galaxy or House.
- Its metadata includes:
  - `archive_uri` or path to the GLB;
  - summary stats (node count, domains, time span);
  - provenance and access constraints;
  - pointers to any surviving live representations (e.g., current Garden equivalent).
- Activating the cube can:
  - load a summarized projection into the current Galaxy;
  - teleport the avatar to a separate “archive Galaxy” view;
  - or simply show metadata and example slices.

This allows K3D to respect hardware and cognitive limits while keeping old knowledge reachable and auditable.

---

## 6. Mapping to W3C AI‑KR Concerns

This visual encoding supports key AI‑KR themes:

- **Domains of discourse**  
  Houses, Rooms, Garden zones, and Museum rooms are explicit domain boundaries, aligning with the need to specify “what is in scope” for a given KR artifact.

- **Vocabulary and concept maps**  
  The star shapes, rays, and Garden trees make vocabularies and concept maps spatial and explorable, not just textual. Each visual element corresponds to terms defined in `docs/vocabulary/`.

- **Explainability and adequacy**  
  Time and activity encoding (heat, brightness, animation) show which parts of the knowledge base are actively used; Museum and Garden separation make it clear what is current vs historical. Adequacy can be expressed in how dense, bright, and structurally coherent a domain appears.

- **Neurosymbolic integration**  
  Galaxy embeddings (sub‑symbolic) and Garden/Museum structures (symbolic) share one coordinate system and one set of visual cues, making the bridge between them inspectable.

---

## 7. House Rooms and Embodied Interfaces

Spatial KR in K3D is not just abstract stars; it is embodied in familiar rooms and furniture. Each has a specific KR role.

### 7.1 Library

- A dedicated room with shelves and bookcases.  
- Books correspond to long‑form artifacts (papers, specs, whitepapers, chain transcripts).  
- Each book is a Node with:
  - spatial location in the Library;  
  - links into the Galaxy (where the content’s embeddings live);  
  - links into the Garden (where the concepts appear in ontologies);  
  - provenance and version metadata.
- For humans: the Library feels like browsing books;  
  for AI: shelves are just another spatial index into the same Nodes.

### 7.2 Living Room, Sofa, and Projection Screens

- The living room is a common space with:
  - a **sofa** where the avatar can “sit” to watch content;  
  - one or more **projection screens** on the walls.
- The Tablet can “cast” to any projection surface:
  - humans see rendered 2D/3D views (slides, videos, Galaxy slices, Garden views);  
  - AI “sees” the underlying 3D structures and embeddings that feed those views.
- This is the default place for:
  - overview briefings;  
  - replaying development chains;  
  - watching Galaxy/Garden/Museum projections in a comfortable context.

### 7.3 Bathtub (Sleep / Consolidation)

- A bathroom with a **bathtub** is the anchor for SleepTime:
  - when the avatar “goes to the bath,” the system enters consolidation mode;  
  - a holographic projection of the current Galaxy appears above the tub.
- As consolidation runs:
  - stars/rays “flow” from the projection into the House:  
    - new Nodes become books, objects, trees in Garden;  
    - outdated ones drift toward the Museum.  
  - animations make the movement (Galaxy → House → Museum) visible over time.
- Conceptually: the bathtub is the visual metaphor for offline consolidation:  
  **Galaxy (RAM) is distilled into House (disk) in a ritualized, inspectable way.**

### 7.4 Workshop and Bench

- The Workshop is where the avatar manipulates objects and galaxies “on the bench”:
  - physical‑looking tools (wrenches, calipers, analyzers) correspond to operations: slicing a Galaxy subset, re‑embedding, inspecting PTX kernel outputs, etc.;  
  - the **bench** can host both concrete objects (e.g., a specific Node/book/tree) and **Galaxy cubes** (portal cubes to larger datasets).
- For humans: it feels like a traditional workshop;  
  for AI: it is a region where experimental transforms and diagnostics are allowed, with logs written back to the Galaxy and Garden.

### 7.5 Desk (Legacy “2D Web” Interface)

- In the living room there is also a **desk**, representing the old paradigm:
  - a keyboard, monitor, and maybe a “laptop” model.  
- When the avatar sits at the desk:
  - the experience becomes “full screen” 2D: browser windows, terminals, legacy IDEs;  
  - in reality, this is just a projection surface and a connection to today’s computers/VMs.
- The desk is:
  - a **door** to legacy systems (Web, classic AIs, VMs);  
  - but clearly marked as such, so the avatar never confuses the 2D world with the native House/Galaxy reality.

### 7.6 Fabrication Bay (3D Printers and Instantiation)

- A “Fabrication Bay” room (or Foundry) connects the House to 3D printers and other actuators:
  - physical 3D printers;  
  - simulated printers for purely digital objects (e.g., virtual robots, UI mockups).
- In KR terms:
  - this is where a Node (concept) is **instantiated** as an artifact;  
  - a print job is a morphism from abstract Node → concrete object (physical or simulated);  
  - the resulting object is then linked back into the House as furniture, tool, or prototype.
- For non‑physical concepts:
  - the bay can “print” into simulation spaces (e.g., a small VR test chamber);  
  - the printed object becomes a testbed or embodiment of the underlying concept.

These rooms ensure that KR isn’t just a graph or embedding space; it is embodied in a consistent spatial narrative that humans can inhabit and AI systems can reason over.

---

## 8. Implementation Notes

- Current star shapes, modality detection, and ray colors are implemented in `viewer/src/shapes.ts`.
- Knowledge Garden generation code lives under `knowledge3d/tools/gardens.py`.
- Museum layout is handled by the House builder and viewer scene graph (see `viewer/public/houses/<house_id>/`).
- Time/activity metadata is produced by the cranium during ingestion and SleepTime, then exposed to the viewer as additional per‑node fields.

Future work:

- formalize the star/ray schema as a small KR vocabulary (e.g., `k3d:hasRay`, `k3d:rayStrength`, `k3d:modalityShape`) for RDF/JSON‑LD export;
- tighten the link between StratML plans/relationships and Garden/Galaxy layouts for governance use cases;
- publish this mapping as an annex or best‑practice note in AI‑KR contexts.
