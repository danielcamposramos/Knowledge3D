# Memory Tablet & Spatial UI Architecture

This document defines the Knowledge3D memory workflow around three persistent structures—**Galaxy**, **House**, and **Museum**—and the **Memory Tablet** interface that bridges them all. The avatar is always embodied in the House: the Galaxy is an introspection layer (akin to a brain scan) that the avatar consults during thinking time, not a place where it "lives".

**NEW (November 2025)**: Complete spatial UI architecture with detailed room taxonomy, portal federation, and dual-client game interface.

---

## Memory Layers at a Glance

| Layer | Analogy | Purpose | Lifespan | Access Pattern |
|-------|---------|---------|----------|----------------|
| **Galaxy** | Active RAM | High-frequency reasoning buffer. Holds embeddings the fused head, PTX operators, and neural layers need right now. | Volatile; repopulated per session or per query. | PTX cosine, on-demand streaming, high-frequency updates. |
| **House** | Persistent SSD/HDD | Consolidated knowledge, crystallised into explicit artifacts (books, diaries, fractal trees, learning insights). | Long-term; evolves during sleep cycles. | Tablet search, SleepTime export, selective PTX load. |
| **Museum (Zone 8)** | Archive / Cold Storage | Deprecated or superseded artifacts kept for retrospection, audit trails, and error-pattern training. | Long-term; mostly append-only. | Loaded only when the user explicitly opens the museum. |

The sleep-time compute pipeline moves validated memories from **Galaxy → House**, while relocation utilities send obsolete items **House → Museum**.

---

## The House as Spatial UI: "Software as Space"

The House is not just a memory store—it is the **primary user interface** for both human and AI users. Every room serves a specific cognitive function, combining memory architecture with game-like navigation and interaction.

**Core Principle**: "The House is a game. Rooms are game modes. Knowledge is the terrain."

### Dual-Client Reality

**Humans see**: 3D geometry, textures, lighting, avatars, spatial audio
**AI sees**: Semantic embeddings, graph topology, 288-byte action buffers, Galaxy projections

**Shared Reality**: Same glTF files, different perceptual layers—procedural drawing engine constructs dual-view from atomic level.

---

## Primary Rooms: The Semantic Zones

### 1. Library (Knowledge Classification)

**Purpose**: Systematic knowledge organization following real-world library standards

**Architecture**:
- Classification system adapted from Dewey Decimal / Library of Congress
- Language grammars organized systematically (each language gets dedicated sections)
- Atomic procedural knowledge: characters → words → phrases → texts
- Star-based consolidation during sleep (procedural compression)

**Contents**:
- **Books**: Consolidated documents (PDF ingestion → sleep → book artifacts)
- **Language Grammars**: ISO 639-1 organized (en/, pt/, es/, ru/, ar/, zh/, ja/, etc.)
- **Atomic Foundations**: Character stars (multi-glyph + multilingual metadata)
- **Reference Works**: Dictionaries, lexicons (WordNet synsets, multilingual dictionaries)

**Access Patterns**:
- Direct shelf browsing (human: walk to shelf, AI: spatial query)
- Tablet search (semantic query → location)
- LOD loading (browse spines → open book → full text)

**Future Expansion**:
- Word-level stars (post-character foundation)
- Phrase templates (compositional procedural)
- Full text procedural synthesis

### 2. Workshop (Active Creation & Experimentation)

**Purpose**: Cross-disciplinary workspace for active knowledge manipulation

**Architecture**:
- Workbenches for different modalities (text, visual, audio, 3D)
- Tool racks with GPU kernels (procedural drawing, RPN executor, compression codecs)
- Galaxy boxes from Museum (on-demand loading of archived knowledge)

**Contents**:
- **Active Projects**: Work-in-progress knowledge artifacts
- **Experimental Tools**: New specialists, adapter prototypes, kernel tests
- **Museum Boxes**: Deprecated knowledge loaded for analysis/retraining
- **Collaboration Space**: Multi-user editing (future: real-time co-creation)

**Access Patterns**:
- Create new artifacts (text, diagrams, procedural drawings)
- Modify existing knowledge (edit books, refine ontologies)
- Load Museum artifacts (explicitly request Zone 8 items)
- Export to Galaxy (stream working sets to active memory)

**Cross-Discipline Integration**:
- Text ↔ Visual (character drawing, OCR training)
- Audio ↔ Text (speech synthesis, transcription)
- 3D ↔ Semantics (spatial knowledge graphs)

### 3. Bathtub (Sleep Chamber & Galaxy Universe Projection)

**Purpose**: Sleep-time consolidation and Galaxy Universe introspection

**Architecture**:
- **Sphere-shaped imaginary space** (carved into floor like ball pit or sofa)
- Avatar center point for sleep cycles
- **Galaxy Universe projection from avatar's head center**
  - Universe = addressable 3D RAM space (like physical memory address space)
  - Multiple galaxies loaded simultaneously (text, visual, audio, reasoning, etc.)
  - Stars transform: **light → 3D shapes/textures** (procedural dual-view construction)

**Galaxy Universe Analogy** (Backwards RAM):
```
Computer RAM:           Galaxy Universe:
├─ Address Space        ├─ 3D Spatial Universe
│  (linear 0x0-0xFFFF)  │  (x,y,z coordinates)
├─ Memory Regions       ├─ Individual Galaxies
│  (heap, stack, etc.)  │  (text, visual, audio, etc.)
└─ Data Bytes           └─ Knowledge Stars
   (values at addresses)   (embeddings at positions)
```

**Loaded Galaxies** (Simultaneous in Universe):
- **Text Galaxy**: Language embeddings, RPN vocabulary (33K+ trigrams)
- **Visual Galaxy**: Font glyphs, procedural drawings (168K+ programs)
- **Audio Galaxy**: Speech patterns, acoustic features (4K+ audio files)
- **Reasoning Galaxy**: ARC-AGI patterns, logic structures
- **Domain Galaxies**: Math, physics, chemistry (future specialists)

**Visualization**:
```
   🌌 Galaxy Universe (addressable 3D space)
        ↓ Multiple Galaxies Loaded
   📚 Text    🎨 Visual    🎵 Audio
        ↓ Stars within each Galaxy
        🌟 Knowledge Stars (light particles)
          ↓ Procedural Drawing
     🔷 3D Shapes + Textures
          ↓ Dual-View Rendering
  👁️ Human View      🤖 AI View
  (visual 3D)       (embeddings + graph)
```

**Functions**:
- **Sleep Consolidation**: All galaxies → House memory crystallization
- **Galaxy Introspection**: View entire Universe as navigable 3D space
- **Multi-Galaxy Queries**: Query across all loaded galaxies simultaneously
- **Visual + Data Querying**: Both clients can pick and query stars
  - Human: Point and click on floating 3D objects
  - AI: Spatial queries on embedding positions (across all galaxies)
- **Procedural Dual-View**: Drawing engine constructs shared reality from atomic stars
- **Galaxy Loading/Unloading**: Explicitly load or unload specific galaxies (memory management)

**Sleep Cycle Process**:
1. Avatar enters Bathtub (triggering sleep mode)
2. Galaxy Universe projects from avatar's head center (all loaded galaxies visible)
3. Stars consolidate: EMA updates, redundancy pruning (across all galaxies)
4. Cross-galaxy patterns detected (transitive learning)
5. Validated knowledge crystallizes into House artifacts
6. Avatar awakens, House updated with new books/insights

**Introspection Mode** (Awake):
- Avatar can enter Bathtub while awake for Galaxy Universe inspection
- All loaded galaxies visible in 3D space (color-coded by modality)
- Stars visible as live 3D visualization
- Confidence scores shown as brightness/size
- Semantic clusters visible as spatial groupings
- Navigate between galaxies (fly/teleport in Universe space)

### 4. Living Room (Old Paradigm Bridge & Social Space)

**Purpose**: Interface to conventional computing and social interaction

**Architecture**:
- **Sofa/Furniture**: Customizable (Minecraft/Sims-like building)
  - Each AI/human constructs their house with assistance
  - Furniture styles reflect personality/preferences
- **Large Projection Screen**: Main display for 2D interfaces
  - Full-screen mode when in 2D interfaces (monitor + keyboard paradigm)
  - Works for both human and AI clients (same projection screen, different rendering)
- **Desktop Corner**: 3D representation of old paradigm
  - Physical keyboard + mouse (functional in AR/VR)
  - Mapped to UI controls when in full-screen mode
  - **"Casting" VMs**: Access any OS inside K3D
    - Web browsers → any webpage
    - Operating systems → Windows, Linux, macOS VMs
    - Legacy apps → zero code rewrite, full backwards compatibility

**Projection Screen Capabilities**:
- **Virtual KVM**: Multiple instances, multiple screens
- **Move-Along Mode**: Picture-in-picture in 3D space (AR/VR concept)
- **Any Output**: Cast any VM/browser/app output to any projection screen
- **Tablet Integration**: Tablet is also a projection screen (portable displays)

**Social Features** (Future):
- Multi-user seating (collaborative viewing)
- Shared screens (co-browsing, pair programming)
- Voice chat via spatial audio
- Avatar presence indicators

**Why This Matters**:
- **Zero Code Rewrite**: All old paradigm software works inside K3D
- **Full Backwards Compatibility**: Historical displays supported via casting
- **Unified Interface**: Switch between 3D spatial and 2D conventional seamlessly
- **Synthetic User Rights**: AI accounts = user accounts (web, APIs, legacy systems)

### 5. Knowledge Gardens (Ontology Greenhouse)

**Purpose**: Non-linear knowledge visualization and ontology exploration

**Architecture**:
- **Circular indoor greenhouse** (glass walls, natural lighting simulation)
- **Knowledge Trees**: Hierarchical ontologies as actual 3D trees
  - Roots = foundational concepts
  - Trunk = core principles
  - Branches = derived knowledge
  - Leaves = specific facts/examples
- **Fractal Trees**: Self-similar knowledge structures
- **Growth Patterns**: Trees grow during sleep as knowledge consolidates

**Contents**:
- **Ontologies**: Taxonomies, concept hierarchies
- **Semantic Networks**: Graph-based knowledge (not linear like books)
- **Cross-References**: Connections between library books and garden trees
- **Evolving Knowledge**: Trees change shape as understanding deepens

**Access Patterns**:
- Walk through garden (spatial exploration)
- Climb trees (drill down into concepts)
- Tablet query (find specific branches/leaves)
- Prune dead branches (deprecated concepts → Museum)

**Why Gardens vs Library**:
- Library = linear, structured, classification-based
- Gardens = organic, networked, growth-based
- Both needed for complete knowledge representation

---

## Doors and Portals: Scene Management & Federation

### Inner Doors (Scene Separation)

**Purpose**: Optimize data loading and separate concerns

**Architecture**:
- **Game-Engine Style**: Load/unload scenes like GTA, Unreal Engine
- **FOV/LOD Optimization**: Only load visible/active rooms
- **Separation of Concerns**: Group related knowledge by room

**Technical Implementation**:
- glTF scene nodes per room
- Lazy loading via bufferView streaming
- GPU memory management (keep <200MB active)
- Frustum culling (only render visible geometry)

**Examples**:
- Door between Library and Workshop (different active datasets)
- Door between Living Room and Knowledge Gardens (different rendering modes)
- Door to Bathtub (switch to Galaxy projection mode)

### Portals (Federated Spaces)

**Purpose**: Connect local and remote houses, enable networked knowledge

**Architecture**:
- **Standard Endpoints**: Use existing web standards (WebRTC, WebSocket, HTTPS)
- **Local Federation**: User House ↔ AI House (same machine or LAN)
- **Wide Area**: Connect houses across internet (VPN, direct connect, relay)
- **Attribution Preserved**: Origin metadata tracked for all knowledge
- **Locability**: Always know where knowledge came from

**Portal Types**:

**1. Local Portal** (same machine):
```
User House               AI House
    |                        |
    +-- Portal Door --------+
    |   (localhost)         |
    +------------------------+
```

**2. Remote Portal** (internet):
```
User House                 Remote AI House
    |                             |
    +-- Portal Door -------------+
        (wss://remote.k3d.io)
```

**3. Museum Portal** (Zone 8 archive):
```
House                    Museum (Zone 8)
    |                          |
    +-- Special Portal -------+
        (local, read-mostly)
```

**Tablet as Cross-Space Interface**:
- **For AI**: Connection to home House, load knowledge to Galaxy
- **For Humans**: Query remote knowledge, browse federated houses
- **Portable Displays**: Tablet acts as projection screen in any space

**Selling Model: "Software as Space"**:
- Same web paradigm transferred to 3D
- Houses = websites (local or hosted)
- Portals = hyperlinks (navigate between spaces)
- K3D = abstraction layer (like HTTP for spatial web)

**Examples**:
- Personal house + work house (same user, different contexts)
- Collaborative house (multiple users, shared knowledge)
- Public library house (read-only knowledge repository)
- AI assistant house (service provider, API-like access)

---

## The Memory Tablet: Universal Interface

The Memory Tablet is a persistent 3D object available to the avatar at all times. While the avatar can grab items directly in-room, the tablet offers a galaxy-standard view when the avatar wants to align house artifacts with the active thinking memory.

### Core Functions

**1. Inventory Browser**:
- Zero-latency search across House inventory ("disk")
- Filtered views: books, trees, learning insights, diaries, dream artifacts
- Quick teleport links to rooms/shelves

**2. Galaxy Bridge**:
- Surfaces active Galaxy content (RAM)
- Confidence scores, PTX task queues, teacher tags
- Request explicit loads: House → Galaxy (on-demand)

**3. Old-World Connectors**:
- Embedded browser (Firefox container, lightweight)
- Interact with conventional web content, docs, legacy chat
- Captured context stored as tablet notes → SleepTime consolidation

**4. Context Mixer (LOD Controls)**:
- **Coarse**: Summaries, centroids, low-resolution
- **Medium**: Subset of embeddings, partial geometry
- **Full**: Complete GLBs, high-fidelity textures

**5. Projection Screen (NEW)**:
- **Cast ANY OS app** to tablet display
- Works as portable display in 3D space
- **Zero code rewrite**: Full backwards compatibility
- **Historical displays**: Support legacy interfaces

### Implementation Expectations

**1. Always-On Link to House**:
- Queries house-memory index (GLB + manifest)
- Generated every sleep cycle
- Highest-priority retrieval source for fused head

**2. On-Demand Streaming**:
- Stream artifacts into Galaxy (respecting LOD/memory budgets)
- Fused head receives callbacks for working set expansion
- PTX cache updates automatically

**3. Browser Integration**:
- Prefer open-source containers (Firefox-based, Docker)
- Authenticate through doors (`k3d://` URIs)
- Capture fetched context as structured entries

**4. Mutation Hooks**:
- Edit artifact → SleepTime reconciliation
- Materialize new House asset
- Relocate previous version → Museum

**5. VM Casting**:
- Virtual KVM functionality
- Multiple VM instances → multiple screens
- Move-along mode (3D picture-in-picture)
- Synthetic user accounts (AI = user for legacy systems)

---

## The House as Game UI

**Core Concept**: "The House is a game. Rooms are game modes. Knowledge is the terrain."

### Game-Like Features

**1. Room-Based Navigation**:
- Rooms = game modes/level selection
- Doors = loading screens (optimized via FOV/LOD)
- Portals = warp points to other worlds (federated houses)

**2. 3D Embodied Actions**:
- Pick up books (load knowledge)
- Place objects on shelves (organize memory)
- Climb trees (explore ontologies)
- Enter Bathtub (sleep/introspection)
- Use tablet (universal tool/HUD)

**3. Customizable Spaces**:
- Minecraft/Sims-like building (with AI assistance)
- Furniture placement, room layouts
- Aesthetic choices (styles, colors, lighting)

**4. Multiplayer Support**:
- **Human vs Human**: Collaborative knowledge building
- **AI vs AI**: Swarm reasoning, debate
- **Mixed Matches**: Human-AI co-creation

**5. Spatial Audio**:
- Conversations localized to position
- Sound sources (fountain in garden, projector in living room)
- Accessibility feature (blind navigation via spatial cues)

### Why Game Architecture Works

**Performance**:
- Game engines solve 3D rendering, physics, scene management
- Proven scalability (millions of players in MMOs)
- LOD/culling techniques directly applicable

**Accessibility**:
- VR/AR native (Oculus, HoloLens, Vision Pro)
- Desktop mode (monitor + keyboard + mouse)
- Mobile mode (touchscreen navigation)

**Familiarity**:
- Users already understand game UIs
- Lower learning curve than custom 3D interfaces
- Leverages decades of game UX research

**Future Vision**:
- Networked knowledge universes (MMO-like)
- Marketplace for houses, rooms, knowledge artifacts
- User-generated content (custom rooms, tools, visualizations)

---

## Sleep-Time & LOD Interactions

**Consolidation**:
- SleepTime Compute rebuilds `learning_memory.glb`
- Regenerates house index for tablet
- Records stable 1.0 honesty prompts → retire from active drills

**Dynamic Loading**:
- **Coarse**: Centroids for quick scans
- **Medium**: Embeddings + metadata
- **Full**: Complete GLBs (geometry, textures)

**Museum Handling**:
- Relocation utilities tag: `previous_zone`, `relocated_at`
- Tablet "Museum mode" for post-mortem analysis
- No automatic Galaxy reload (explicit promotion only)

---

## Agent & Tooling Requirements

**House Memory Builder**:
- Extend SleepTime to emit PTX-ready `house_memory.glb`
- Fused head loads before querying modality galaxies

**Prompt Pruning**:
- Retire mastered prompts (fused head + tablet confirmation)
- Move to verification list (not main drill sets)

**Deprecation Workflow**:
- Call `relocate_to_museum` when superseding artifacts
- Tag previous version, move to Zone 8

**Tablet UI**:
- Viewer-side interface (search, LOD controls, browser integration)
- Indicators showing source: Galaxy, House, or Museum

**Room Builders**:
- GLB generators for each room type
- Standard layouts + customization hooks
- Door/portal placement tools

**Projection Screen System**:
- VM casting infrastructure (Docker, VNC, RDP)
- Tablet display rendering
- Input mapping (keyboard/mouse → spatial controls)

---

## Technical Specifications

### Room GLB Format

```json
{
  "scenes": [
    {
      "name": "Library",
      "nodes": [0, 1, 2],  // shelves, books, doors
      "extras": {
        "k3d": {
          "room_type": "library",
          "classification_system": "dewey_decimal",
          "language_sections": ["en", "pt", "es", "ru", "ar", "zh", "ja"],
          "lod_levels": ["coarse", "medium", "full"],
          "memory_budget_mb": 50
        }
      }
    }
  ],
  "nodes": [
    {
      "name": "Shelf_A",
      "mesh": 0,
      "extras": {
        "k3d": {
          "category": "000_computer_science",
          "books": [
            {"title": "SICP", "artifact_path": "house_zone7/books/sicp.glb"}
          ]
        }
      }
    }
  ]
}
```

### Portal Endpoint Format

```json
{
  "portal": {
    "type": "federated",  // "local", "remote", "museum"
    "endpoint": "wss://remote.k3d.io/house/alice",
    "protocol": "k3d-portal-v1",
    "auth": {
      "method": "oauth2",
      "provider": "github"
    },
    "capabilities": ["read", "write", "collaborate"],
    "bandwidth_limit_mbps": 10
  }
}
```

### Tablet Display Protocol

```json
{
  "display": {
    "type": "projection_screen",  // "tablet", "wall_screen", "desktop"
    "resolution": [1920, 1080],
    "casting_source": {
      "type": "vm",  // "browser", "app", "galaxy", "house"
      "vm_id": "ubuntu-dev-01",
      "protocol": "vnc",
      "endpoint": "localhost:5901"
    },
    "mode": "fullscreen",  // "pip", "move-along"
    "controls": {
      "keyboard": "mapped",  // "native", "mapped", "disabled"
      "mouse": "3d_pointer"  // "native", "3d_pointer", "disabled"
    }
  }
}
```

---

## Migration Path (Old → New Paradigm)

**Phase 1: Room Foundation** (Current)
- ✅ Basic house structure (single scene)
- ✅ Memory Tablet interface (inventory browser)
- ⏳ Room separation (Library, Workshop, Gardens)

**Phase 2: Spatial UI** (Next)
- 🔲 Bathtub sleep chamber (Galaxy projection)
- 🔲 Living Room projection screens (VM casting)
- 🔲 Door system (scene management)

**Phase 3: Federation** (Future)
- 🔲 Portal architecture (local/remote)
- 🔲 Tablet as projection screen
- 🔲 Synthetic user authentication

**Phase 4: Game UI** (Future)
- 🔲 Multiplayer support
- 🔲 Customizable houses (Minecraft/Sims-like)
- 🔲 Marketplace (user-generated content)

---

## References

- Sleep pipeline: `docs/SLEEP_COMPUTE.md`, `knowledge3d/cranium/phase10/sleep_time_compute.py`
- Museum relocation: `knowledge3d/tools/phase18/meaning_cluster_trainer.py::relocate_to_museum`
- Fused head routing: `knowledge3d/cranium/fused_head.py`
- PTX learning memory builder: `knowledge3d/tools/learning_memory_builder.py`
- Viewer (Three.js): `viewer/src/` (room rendering, door navigation, portal system)
- Projection screens: `viewer/src/components/ProjectionScreen.tsx` (VM casting, tablet display)

---

**Last Updated**: November 19, 2025
**Status**: Architectural vision complete, implementation in progress
**Next**: Bathtub sleep chamber + Living Room projection screens

Keep this document in sync with roadmap changes and spatial UI implementation progress.
