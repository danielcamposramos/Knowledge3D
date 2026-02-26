# Memory Tablet Specification

**Version:** 1.0
**Status:** Architectural Design (Post-MVP Implementation)
**Date:** February 26, 2026
**Author:** Daniel Ramos (K3D Architect)

---

## Abstract

The **Memory Tablet** is K3D's primary interface object — the common interactive surface through which both humans and AI navigate, manipulate, and create knowledge in the spatial environment. Unlike traditional GUI elements (buttons, windows, menus), the Memory Tablet is a **3D object in K3D space** whose content is rendered procedurally, enabling dual-client perception: humans see visual UI, AI sees semantic graphs.

**Core principle:** *"Software was always meant to be a place, not a window."*

The Memory Tablet transforms this principle into concrete interface design — it's a **physical object** you carry through K3D's spatial environments (Houses, Galaxy Universe), not a 2D overlay on top of reality.

---

## 1. Vision Statement

### The Problem with Traditional UIs

**Traditional GUI paradigm:**
```
User → 2D Screen (flat window) → Application
       ↑
  Fixed interface, modal dialogs, nested menus
```

**Problems:**
- **Flat hierarchy:** Everything squeezed into 2D
- **Modal interruptions:** Dialog boxes block workflow
- **Context loss:** Switching apps = losing spatial position
- **Accessibility barriers:** Visual-only (screen readers bolt-on)
- **AI-opaque:** AI cannot inspect UI structure

---

### K3D's Spatial UI Paradigm

**K3D paradigm:**
```
User → 3D Space (House/Galaxy) → Memory Tablet (3D object) → Knowledge
       ↑                           ↑
   Navigate freely          Persistent canvas, spatial context
```

**Benefits:**
- **3D navigation:** Move through knowledge landscapes
- **Non-modal:** Tablet = persistent object, no popups blocking view
- **Spatial memory:** Return to places, remember locations
- **Accessibility-native:** Multi-modal by architecture (visual, audio, Braille, haptics)
- **AI-transparent:** Tablet content = procedural programs AI can read

---

## 2. Architectural Position

### K3D Stack Layers

```
┌─────────────────────────────────────────┐
│  Memory Tablet (User Interface Layer)  │ ← This Spec
├─────────────────────────────────────────┤
│  Spatial UI (Houses, Rooms, Portals)   │
├─────────────────────────────────────────┤
│  Galaxy Universe (Knowledge Layer)     │
├─────────────────────────────────────────┤
│  Knowledgeverse (7-Region VRAM)        │
├─────────────────────────────────────────┤
│  Cranium (PTX Execution)               │
└─────────────────────────────────────────┘
```

**Memory Tablet sits at the TOP** — it's the **interface** through which users (human and AI) access all lower layers.

---

## 3. Core Concepts

### 3.1. Memory Tablet = Physical 3D Object

**Not a 2D UI overlay. It's a THING you hold in 3D space.**

**Analogy:** Like holding an iPad in real life, but:
- It's **persistent** (doesn't disappear when you switch "apps")
- It's **spatial** (exists in House/Galaxy 3D coordinates)
- It's **procedural** (content rendered from RPN programs)
- It's **dual-client** (humans see visuals, AI sees semantic structure)

**Physical properties:**
- Position: (x, y, z) in House or Galaxy space
- Orientation: Euler angles or quaternion
- Size: Configurable (e.g., 8" tablet, 12" tablet, wall-sized projection)
- Surface: Procedural canvas (VectorDotMap rendering)

---

### 3.2. Dual-Client Perception

**Critical principle:** The **same tablet object** is perceived differently by humans vs. AI.

#### Human Perception (Visual Client)

**When human looks at tablet:**
```
Tablet Surface
  ↓
Procedural Content (RPN programs)
  ↓
Visual Renderer (VectorDotMap, Character Galaxy)
  ↓
Human sees: Glyphs, diagrams, buttons, text
```

**Example:** Tablet displays "Home" button
- **Stored as:** `["CIRCLE", "r=50", "color=#4A90E2", "TEXT", "Home", "x=50", "y=50"]` (VectorDotMap)
- **Human sees:** Blue circle with "Home" text (visual rendering)

---

#### AI Perception (Semantic Client)

**When AI looks at tablet:**
```
Tablet Surface
  ↓
Procedural Content (RPN programs)
  ↓
Semantic Parser (TRM Navigation)
  ↓
AI sees: Navigation graph, references to Galaxy nodes
```

**Same example:** Tablet displays "Home" button
- **Stored as:** `["CIRCLE", "r=50", "color=#4A90E2", "TEXT", "Home", "x=50", "y=50"]`
- **AI sees:** `{@type: "NavigationAction", target: "house:home", trigger: "button_press"}`

**Result:** AI understands the button is a navigation trigger to "house:home" location.

---

### 3.3. Procedural UI Rendering

**All tablet UI elements = procedural programs (symlink-style composition).**

#### Example: "Start" Button

**Traditional UI (static duplication):**
```html
<button>Start</button>  <!-- Stores "Start" as static text -->
```

**K3D Tablet (procedural references):**
```json
{
  "@type": "UIButton",
  "label": ["@char_S", "@char_t", "@char_a", "@char_r", "@char_t"],  // References to Character Galaxy!
  "shape": "@drawing:circle",  // Reference to Drawing Galaxy circle primitive
  "action": "@galaxy:navigate:home"
}
```

**Benefits:**
- **Zero duplication:** "S" stored once in Character Galaxy, referenced infinitely
- **Multi-font support:** Same button, different fonts = just change Character Galaxy reference
- **Accessibility:** Screen reader reads procedural text natively (no ARIA needed)
- **AI-transparent:** AI sees button = navigation action (semantic meaning)

---

## 4. Tablet Capabilities

### 4.1. Browse Galaxy Universe

**Navigate public canonical knowledge:**
- View galaxies (Drawing, Character, Word, Grammar, Math, Reality, Audio)
- Query knowledge nodes (search, filter, traverse)
- Inspect node details (procedural programs, metadata)
- Bookmark nodes (save references to House)

**Example interaction:**
```
Human: Taps tablet, "Math Galaxy" icon
  ↓
Tablet displays: Math symbols grid (procedural rendering)
  ↓
Human: Taps "∫" (integral symbol)
  ↓
Tablet shows: RPN program for integral, example uses, related symbols
```

**AI equivalent:**
```python
ai.tablet.navigate("galaxy:math")
ai.tablet.query("symbol:integral")
result = ai.tablet.inspect("galaxy:math:integral")
# Returns: {program: [...], metadata: {...}}
```

---

### 4.2. Browse House Universe

**Navigate private workspace:**
- View rooms (bedroom, office, lab)
- Access local knowledge (notes, discoveries, drafts)
- Manage personal Galaxy references (bookmarks, custom collections)
- Create new knowledge nodes (private until published to Galaxy)

**Example interaction:**
```
Human: In House environment, taps tablet "My Notes"
  ↓
Tablet displays: List of private notes (procedural text)
  ↓
Human: Taps "Physics Research Draft"
  ↓
Tablet shows: Note content (editable procedural canvas)
```

---

### 4.3. Control Procedural Projection Screens

**Projection screens = virtual monitors inside K3D space.**

**What they are:**
- 3D objects in House or Galaxy environment
- Display procedural content (like tablet, but larger/fixed position)
- Controlled via tablet (select content, adjust size, position)

**Use cases:**
- **Presentations:** Display Galaxy content on wall-sized screen
- **Multi-monitor setup:** Multiple screens showing different knowledge domains
- **Collaboration:** AI and human view same screen (dual-client rendering)
- **Legacy content:** Display VM outputs (external apps) inside K3D space

**Example interaction:**
```
Human: Taps tablet "Projection Control"
  ↓
Tablet displays: List of available screens + "Create New Screen" button
  ↓
Human: Taps "Create New Screen"
  ↓
Screen appears in House (default position: wall opposite user)
  ↓
Human: Drags content from tablet to screen (e.g., Math Galaxy diagram)
  ↓
Screen displays: Diagram at wall scale (procedural rendering, infinite LOD)
```

---

### 4.4. Access Legacy Content

**Bridge to non-K3D data:**
- File system access (read local files, not yet in Galaxy)
- VM outputs (external apps like browsers, IDEs)
- Import tools (convert PDF → Galaxy nodes, images → VectorDotMap)

**Why needed:**
Not all knowledge is procedural yet. Tablet provides seamless access to:
- Local files (PDFs, images, videos)
- Running applications (web browsers, code editors)
- External APIs (Wikipedia, Wolfram Alpha)

**Example interaction:**
```
Human: Taps tablet "Files"
  ↓
Tablet displays: Local filesystem browser (procedural tree view)
  ↓
Human: Taps "research_paper.pdf"
  ↓
Tablet offers: "View in K3D" (convert to procedural) or "Open External" (VM)
  ↓
If "View in K3D": PDF ingested, pages → VectorDotMap, text → Character Galaxy
If "Open External": PDF opens in VM, output displayed on projection screen
```

---

### 4.5. Create New Knowledge

**Tablet as authoring tool:**
- Sketch diagrams (Drawing Galaxy primitives)
- Write text (Character + Word Galaxy references)
- Define new procedures (RPN program editor)
- Annotate existing knowledge (attach metadata to Galaxy nodes)

**Example interaction:**
```
Human: Taps tablet "Create" → "New Diagram"
  ↓
Tablet displays: Drawing canvas (VectorDotMap editor)
  ↓
Human: Draws circle, line, arrow (gesture or stylus)
  ↓
Tablet generates: RPN program ["CIRCLE", "x=100", "y=100", "r=50", ...]
  ↓
Human: Saves → Stored in House:my_room:diagrams (private)
  ↓
Optional: Publish → Copied to Galaxy:drawing (public, canonical)
```

---

## 5. Game Menu System

### 5.1. Entry Sequence

**K3D launches with game-style menu (not traditional desktop app).**

**Boot sequence:**
```
1. K3D Logo (procedural animation)
   ↓
2. Main Menu (procedural UI)
   - [Enter House] (your private workspace)
   - [Enter Galaxy] (public knowledge universe)
   - [Settings]
   - [Exit]
   ↓
3. User selects environment
   ↓
4. 3D space loads (House or Galaxy)
   ↓
5. Memory Tablet spawns in hand (default position)
```

**Why game-style menu?**
- **Spatial metaphor:** Software = place, not window
- **No desktop clutter:** No taskbar, no minimize/maximize
- **Intentional navigation:** Consciously choose where you go (House vs. Galaxy)
- **Immersive:** VR-ready (menu = 3D objects, not flat overlay)

---

### 5.2. Environment Selection

**Two primary environments (both can run simultaneously):**

#### House (Private Workspace)
- **Purpose:** Personal knowledge, work-in-progress, private notes
- **Structure:** Rooms (bedroom, office, lab, etc.)
- **Ownership:** User-controlled, bounded domain
- **Visibility:** Private by default (can share specific rooms)

#### Galaxy (Public Knowledge Universe)
- **Purpose:** Canonical shared knowledge, community contributions
- **Structure:** Galaxies (Drawing, Math, Reality, etc.)
- **Ownership:** Community-curated, canonical forms
- **Visibility:** Public, discoverable, read-only by default

**Key insight:** House and Galaxy **run concurrently** (not exclusive). You can:
- View Galaxy content from inside House (via projection screens)
- Reference Galaxy nodes in House notes (symlink-style)
- Switch between House and Galaxy via portals (doorways in 3D space)

---

### 5.3. In-Game Controls

**Tablet provides in-game menu (not separate pause menu):**

**Core controls (always accessible on tablet):**
- **Home:** Return to House entrance
- **Galaxy:** Open portal to Galaxy
- **Settings:** Adjust preferences (graphics, audio, accessibility)
- **Help:** Context-sensitive guidance (AI assistant)
- **Exit:** Leave K3D (with save prompt if unsaved work)

**Navigation controls:**
- **Teleport:** Jump to bookmarked locations (House or Galaxy)
- **Portal:** Create temporary doorway (House ↔ Galaxy)
- **Fly/Walk toggle:** Change locomotion mode (VR, keyboard, gamepad)

**Content controls:**
- **Search:** Query Galaxy Universe (semantic search)
- **Filter:** Show/hide knowledge domains (e.g., hide Audio Galaxy)
- **Create:** New diagram, note, procedure
- **Share:** Publish House content to Galaxy (with provenance)

---

## 6. Procedural Projection Screens

### 6.1. What Are Projection Screens?

**Projection screens = virtual monitors/displays inside K3D 3D space.**

**Not external monitors.** They're **3D objects** in House/Galaxy environment, like:
- Wall-mounted TV (fixed position)
- Floating hologram (movable)
- Desk monitor (on virtual desk object)

**Content:** Anything displayable on tablet can be projected:
- Galaxy knowledge (diagrams, text, formulas)
- House notes (private content)
- Legacy content (VM outputs, PDFs, web pages)
- Live data (API feeds, AI reasoning traces)

---

### 6.2. Dual-Client Rendering (Again!)

**Same dual-client principle as tablet:**

**Human looking at projection screen:**
- Sees visual rendering (glyphs, images, animations)
- Procedural content rendered at screen resolution
- Infinite LOD (zoom in = more detail, no pixelation)

**AI looking at projection screen:**
- Sees procedural program (semantic structure)
- Can query screen content (extract knowledge nodes)
- Can annotate screen (attach metadata to displayed content)

**Use case:** Human and AI collaborate on same screen
- Human: "AI, explain this diagram" (points at screen)
- AI: Queries screen content, identifies procedural program, generates explanation
- AI explanation appears on tablet (human's device)

---

### 6.3. Screen Control via Tablet

**Tablet as remote control:**

**Create screen:**
```
Tablet: "Projection Control" → "Create Screen"
  ↓
User places screen in 3D space (gesture, drag, voice command)
  ↓
Screen spawns (default size: 1920×1080 equivalent, adjustable)
```

**Display content:**
```
Tablet: Select content (e.g., Math Galaxy diagram)
  ↓
Drag from tablet to screen (gesture)
  ↓
Screen displays content (procedural rendering)
```

**Adjust screen:**
```
Tablet: "Screen Settings"
  ↓
- Resize (pinch gesture, slider)
- Reposition (drag screen in 3D space)
- Rotate (twist gesture, orientation controls)
- Delete (swipe to dismiss)
```

---

### 6.4. Legacy Content Integration

**Display VM outputs inside K3D:**

**Problem:** External apps (browsers, IDEs) output to OS windows (flat 2D).

**Solution:** Capture VM output, render on K3D projection screen.

**Example:**
```
User runs: Web browser (Firefox) in VM
  ↓
K3D captures: Browser window framebuffer
  ↓
Tablet offers: "Display Browser on Screen 1"
  ↓
Screen 1 shows: Live browser output (flat 2D texture on 3D screen object)
  ↓
User interacts: Click screen → mouse events forwarded to VM browser
```

**Why this matters:**
- Seamless transition (legacy apps inside K3D environment)
- No context switching (no Alt+Tab to external windows)
- Spatial memory (browser always on Screen 1 in House:office)
- Collaboration (AI can see browser content, assist with research)

---

## 7. Simultaneous Environments

### 7.1. House + Galaxy Run Concurrently

**Unlike traditional apps (single active window), K3D runs TWO 3D spaces at once:**

```
┌─────────────────────┐     ┌─────────────────────┐
│   House Universe    │ ⟷  │  Galaxy Universe    │
│   (Private)         │     │   (Public)          │
│                     │     │                     │
│  - Rooms            │     │  - Galaxies         │
│  - Local knowledge  │     │  - Canonical nodes  │
│  - Work-in-progress │     │  - Community refs   │
└─────────────────────┘     └─────────────────────┘
         ↑                           ↑
         └──── User + Memory Tablet ─┘
```

**Why simultaneous?**
- **Seamless reference:** House notes reference Galaxy nodes (no loading delay)
- **Live sync:** Galaxy updates appear in House views instantly
- **Portals:** Doorways between House and Galaxy (step through = instant transition)
- **Dual workspaces:** Research in Galaxy, write notes in House (switch back and forth)

---

### 7.2. Portal System

**Portals = doorways connecting House ↔ Galaxy.**

**Visual metaphor:** Literal door you walk through (VR) or step through (keyboard).

**Example:**
```
User in House:office (private workspace)
  ↓
Looks at wall: Sees portal (glowing doorway)
  ↓
Tablet shows: "Portal to Galaxy:math" (destination labeled)
  ↓
User walks through portal
  ↓
Instantly in Galaxy:math (star field with math symbols)
  ↓
Can walk back through portal → returns to House:office
```

**Portal types:**
- **Fixed:** Permanent doorways (e.g., House:entrance → Galaxy:home)
- **Temporary:** Created on-demand (tablet "Create Portal" → select destination)
- **Bookmarked:** Saved favorite locations (quick travel)

---

### 7.3. Memory Tablet Persistence

**Tablet stays with you across environments:**

**Traditional apps:** Lose state when switching (clipboard lost, tabs closed).

**K3D Tablet:** Persistent state (like holding a physical object).

**Example:**
```
User in House:office, writes note on tablet
  ↓
User walks through portal to Galaxy:math
  ↓
Tablet still shows note (not cleared)
  ↓
User references Galaxy:math symbol, adds to note
  ↓
User returns to House:office
  ↓
Note still there (persistent across environments)
```

**Why this matters:**
- **No context loss:** Tablet = your "working memory" (always available)
- **Cross-environment workflow:** Research in Galaxy, document in House (single tool)
- **Spatial anchoring:** Tablet position relative to body (always in hand or on desk)

---

## 8. Procedural UI Architecture

### 8.1. UI Elements = Galaxy References

**Core principle:** UI is NOT separate from knowledge — UI **IS** knowledge.

**Every UI element = procedural program composed from Galaxy.**

| UI Element | Procedural Composition |
|------------|------------------------|
| **Text label** | Character Galaxy references (symlink-style) |
| **Button** | Drawing Galaxy circle + Character Galaxy text |
| **Diagram** | VectorDotMap program (CIRCLE, LINE, etc.) |
| **Menu** | Word Galaxy references + Grammar Galaxy rules |
| **Icon** | Drawing Galaxy primitives (composed shapes) |
| **Animation** | Procedural tweening (RPN program with time parameter) |

**Example: "Settings" button**
```json
{
  "@type": "UIButton",
  "visual": {
    "background": "@drawing:rounded_rect",  // Reference to Drawing Galaxy
    "label": ["@char_S", "@char_e", "@char_t", "@char_t", "@char_i", "@char_n", "@char_g", "@char_s"],  // Character Galaxy
    "icon": "@drawing:gear"  // Reference to gear icon (Drawing Galaxy primitive)
  },
  "semantic": {
    "@type": "NavigationAction",
    "target": "tablet:settings_menu"
  }
}
```

**Benefits:**
1. **Zero duplication:** "Settings" characters stored once, referenced infinitely
2. **Multi-language:** Change Character Galaxy reference → instant translation
3. **Accessibility:** Screen reader reads Character Galaxy metadata (pronunciation, meaning)
4. **AI-transparent:** AI sees button = navigation action (understands purpose)
5. **Themeable:** Change Drawing Galaxy primitives → entire UI theme changes

---

### 8.2. Codec-Driven Rendering

**All tablet rendering uses procedural codecs (PD04, VectorDotMap, Procedural Fonts).**

**Rendering pipeline:**
```
Tablet Content (RPN programs)
  ↓
Select codec based on content type:
  - Text → Procedural Fonts codec (Character Galaxy)
  - Diagrams → VectorDotMap codec (Drawing Galaxy)
  - Embeddings → PD04 codec (if displaying semantic similarity)
  ↓
Execute codec (PTX kernels in Cranium)
  ↓
Render output:
  - Human client: Visual glyphs, shapes, colors
  - AI client: Semantic graph, navigation structure
```

**Why codec-driven?**
- **Compression:** Tablet content = RPN programs (small), not rasterized images (large)
- **Infinite LOD:** Zoom tablet → re-execute codecs at higher resolution (no pixelation)
- **Dynamic:** Tablet content changes → re-execute codecs (real-time updates)
- **Sovereign:** All rendering via PTX kernels (zero external dependencies)

---

### 8.3. Symlink-Style Composition

**Tablet UI = graph of references (like a website with hyperlinks).**

**Anti-pattern (static duplication):**
```json
{
  "button1": {"label": "Home", "chars": ["H", "o", "m", "e"]},
  "button2": {"label": "Settings", "chars": ["S", "e", "t", "t", "i", "n", "g", "s"]},
  "button3": {"label": "Help", "chars": ["H", "e", "l", "p"]}
}
// Duplicates "H", "e" across buttons!
```

**Procedural pattern (symlink references):**
```json
{
  "button1": {"label": ["@char_H", "@char_o", "@char_m", "@char_e"]},
  "button2": {"label": ["@char_S", "@char_e", "@char_t", "@char_t", "@char_i", "@char_n", "@char_g", "@char_s"]},
  "button3": {"label": ["@char_H", "@char_e", "@char_l", "@char_p"]}
}
// "H" and "e" stored once in Character Galaxy, referenced 3 times!
```

**Result:** 70%+ compression across entire UI (same as Galaxy Universe).

---

## 9. Interaction Patterns

### 9.1. Human Interaction

**Multi-modal by design (not accessibility bolt-on):**

#### Visual Mode (Default)
- **Input:** Touchscreen, mouse, stylus
- **Gestures:** Tap (select), drag (move), pinch (zoom), swipe (navigate)
- **Output:** Visual rendering (glyphs, shapes, colors)

#### Audio Mode
- **Input:** Voice commands ("Show Math Galaxy", "Create note")
- **Output:** Audio descriptions (text-to-speech from Character Galaxy metadata)
- **Gestures:** Hands-free (useful in VR, or for accessibility)

#### Haptic Mode
- **Input:** Haptic feedback (vibration patterns for buttons, edges)
- **Output:** Tactile rendering (Braille Galaxy for text, texture for shapes)
- **Use case:** Blind users navigate tablet via haptic feedback + audio

#### Keyboard/Gamepad Mode
- **Input:** Arrow keys (navigate), Enter (select), Tab (cycle)
- **Output:** Visual + audio (screen reader announces focused element)
- **Use case:** Desktop users without touchscreen, gamers

---

### 9.2. AI Interaction

**AI interacts with tablet programmatically (not via visual parsing).**

**API-style access:**
```python
# AI queries tablet state
current_view = ai.tablet.get_current_view()
# Returns: {view: "galaxy:math", focus: "symbol:integral"}

# AI navigates tablet
ai.tablet.navigate("house:my_notes")

# AI creates content
note = ai.tablet.create("note", content=["@char_H", "@char_e", "@char_l", "@char_l", "@char_o"])

# AI queries Galaxy via tablet
results = ai.tablet.query_galaxy("symbol:integral", domain="math")
# Returns: [node_id_1, node_id_2, ...]

# AI inspects screen (projection screen, not tablet)
screen_content = ai.screen.get_content(screen_id="screen_1")
# Returns: {type: "diagram", program: [...], metadata: {...}}
```

**Why programmatic?**
- **No vision model needed:** AI doesn't "see" tablet, it queries procedural content directly
- **Precise:** No ambiguity (visual parsing = error-prone)
- **Fast:** Direct memory access (no OCR, no image processing)
- **Semantic:** AI understands meaning (not just recognizing shapes)

---

### 9.3. Collaborative Interaction (Human + AI)

**Human and AI share same tablet (dual perception, same object).**

**Example scenario:**
```
Human: Draws diagram on tablet (circle + arrow)
  ↓
Tablet stores: VectorDotMap program ["CIRCLE", "x=100", "y=100", ...]
  ↓
AI queries tablet: "What did the human just draw?"
  ↓
AI sees: Procedural program (CIRCLE + arrow)
  ↓
AI interprets: "This is a flow diagram (single node)"
  ↓
AI suggests: "Would you like to add more nodes?" (text on tablet)
  ↓
Human sees: AI's suggestion (rendered text)
  ↓
Human: Taps "Yes"
  ↓
AI generates: Additional nodes (procedural programs)
  ↓
Tablet displays: Extended diagram (human + AI co-creation)
```

**Key insight:** Human and AI **collaborate** on same object (not separate tools).

---

## 10. Legacy Content Bridge

### 10.1. File System Access

**Tablet provides file browser (local files not yet in Galaxy):**

**Interface:**
```
Tablet: "Files" menu
  ↓
Displays: Tree view of local filesystem (procedural rendering)
  - /home/user/documents/
    - research_paper.pdf
    - notes.txt
  - /home/user/pictures/
    - diagram.png
```

**Actions:**
- **View:** Open file in K3D (convert to procedural)
  - PDF → Pages ingested, text → Character Galaxy, images → VectorDotMap
  - PNG → VectorDotMap conversion (if vectorizable)
  - TXT → Character Galaxy references (instant)
- **Import:** Copy file to House Universe (private storage)
- **Open External:** Launch file in VM (legacy app), display on projection screen

---

### 10.2. VM Output Integration

**Display external apps inside K3D (seamless experience):**

**How it works:**
```
1. User launches external app (e.g., web browser) in VM
   ↓
2. K3D captures VM framebuffer (screen content)
   ↓
3. Tablet offers: "Display on Projection Screen"
   ↓
4. User selects screen (or creates new screen)
   ↓
5. Screen shows: Live VM output (flat 2D texture on 3D screen object)
   ↓
6. User interacts with screen: Clicks → mouse events forwarded to VM
```

**Benefits:**
- **No context switching:** External apps inside K3D environment
- **Spatial persistence:** Browser always on Screen 1 in House:office (remember position)
- **Collaboration:** AI can "see" VM output (framebuffer analysis), assist user

**Limitations:**
- **Flat content:** VM output = rasterized pixels (not procedural)
- **No semantic access:** AI sees pixels (must use vision model), not procedural structure
- **One-way bridge:** K3D → VM (VM doesn't know about K3D)

**Future enhancement (post-MVP):**
- **Procedural conversion:** Convert VM output to procedural (e.g., web page → Character + Drawing Galaxy)
- **Bidirectional bridge:** VM apps can query K3D Galaxy (API access)

---

## 11. Technical Implementation (High-Level)

### 11.1. Tablet Object Structure

**K3D Node representation:**
```json
{
  "@type": "MemoryTablet",
  "@id": "tablet:user_primary",
  "position": [x, y, z],  // 3D coordinates in current environment
  "orientation": [qx, qy, qz, qw],  // Quaternion rotation
  "size": [width, height],  // Virtual screen dimensions (e.g., 1024x768)
  "surface": {
    "@type": "ProceduralCanvas",
    "content": [...],  // Current view (RPN programs for UI elements)
    "render_mode": "dual_client"  // Enable human + AI rendering
  },
  "state": {
    "current_view": "house:my_notes",  // Where tablet is currently showing
    "navigation_stack": ["house:home", "house:my_notes"],  // History (back button)
    "clipboard": [...],  // Copied content (procedural programs)
    "bookmarks": [...]  // Saved locations (Galaxy + House references)
  }
}
```

---

### 11.2. Rendering Pipeline

**Per-frame update:**
```python
def render_tablet(tablet, user_type):
    """
    Render tablet for human or AI user.

    user_type: "human" or "ai"
    """
    # 1. Get current view content (RPN programs)
    content = tablet.surface.content

    # 2. Select rendering mode
    if user_type == "human":
        # Visual rendering (VectorDotMap + Procedural Fonts)
        rendered = visual_renderer.render(content)
        # Returns: Pixel buffer (RGBA) for screen
    elif user_type == "ai":
        # Semantic rendering (navigation graph)
        rendered = semantic_renderer.render(content)
        # Returns: Graph structure (nodes + edges)

    # 3. Display on tablet surface (3D texture)
    tablet.surface.update_texture(rendered)

    # 4. Handle interactions (touch, voice, queries)
    process_interactions(tablet, user_type)
```

---

### 11.3. Projection Screen Structure

**K3D Node representation:**
```json
{
  "@type": "ProjectionScreen",
  "@id": "screen:house_office_1",
  "position": [x, y, z],  // Fixed or movable
  "orientation": [qx, qy, qz, qw],
  "size": [width, height],  // Physical dimensions in 3D space
  "content_source": {
    "@type": "ProceduralContent",  // or "VMOutput" for legacy apps
    "source_id": "galaxy:math:diagram_123"  // What's being displayed
  },
  "controller": "tablet:user_primary"  // Which tablet controls this screen
}
```

---

### 11.4. Environment Management

**House + Galaxy simultaneous loading:**
```python
class K3DRuntime:
    def __init__(self):
        # Load both environments at startup
        self.house = HouseUniverse.load(user_id)
        self.galaxy = GalaxyUniverse.load()

        # Spawn user + tablet in House (default)
        self.user = User.spawn(position=self.house.entrance)
        self.tablet = MemoryTablet.spawn(owner=self.user)

        # Create portals (House ↔ Galaxy)
        self.portals = PortalSystem.create_default_portals(self.house, self.galaxy)

    def update(self, delta_time):
        # Update both environments every frame
        self.house.update(delta_time)
        self.galaxy.update(delta_time)

        # Update tablet (always with user)
        self.tablet.update(delta_time, self.user.position)

        # Check portal transitions
        if self.user.collides_with_portal():
            self.user.teleport(portal.destination)
```

---

## 12. Accessibility by Architecture

### 12.1. Multi-Modal Rendering

**Tablet supports ALL modalities natively (not bolt-on):**

| Modality | Input | Output | Use Case |
|----------|-------|--------|----------|
| **Visual** | Touch, mouse | Glyphs, shapes, colors | Sighted users |
| **Audio** | Voice | Text-to-speech | Blind users, hands-free |
| **Braille** | Braille keyboard | Braille display (Braille Galaxy) | Blind users (tactile) |
| **Haptic** | Haptic controller | Vibration patterns | VR users, blind users |
| **Keyboard** | Arrow keys, Tab | Visual + audio | Desktop users |

**No separate "accessibility mode"** — all modes available simultaneously.

---

### 12.2. Screen Reader Native Support

**Screen reader = direct access to procedural content (not OCR):**

**Traditional apps:**
```
Screen → Pixels → OCR (guess what text says) → TTS (read aloud)
               ↑
         Error-prone, slow
```

**K3D Tablet:**
```
Tablet Content → Character Galaxy (procedural text) → TTS metadata (pronunciation) → Read aloud
                           ↑
                   Always accurate (no guessing)
```

**Example:**
```
Tablet displays "Home" button
  ↓
Screen reader queries: tablet.get_focused_element()
  ↓
Returns: {type: "button", label: ["@char_H", "@char_o", "@char_m", "@char_e"]}
  ↓
Screen reader looks up Character Galaxy:
  - "H" → pronunciation: "/eɪtʃ/", meaning: "eighth letter"
  - "o" → pronunciation: "/oʊ/", meaning: "fifteenth letter"
  - ...
  ↓
TTS: "Button. Home. H-O-M-E."
```

**Result:** Perfect accuracy (no OCR guessing), rich metadata (pronunciation, meaning).

---

### 12.3. Braille Galaxy Integration

**Braille output = procedural (not rasterized dots):**

**Character Galaxy → Braille Galaxy mapping:**
```
Character Galaxy: "A" → glyph_program (visual form)
                     ↓
            Braille Galaxy: "⠁" (Braille cell pattern 1)
```

**Tablet with Braille display:**
```
Tablet displays "Home" button (visual)
  ↓
User has Braille display attached
  ↓
Tablet queries Braille Galaxy: Convert ["@char_H", "@char_o", "@char_m", "@char_e"]
  ↓
Returns: ["⠓", "⠕", "⠍", "⠑"] (Braille cells)
  ↓
Braille display shows: Tactile pins (user feels "Home")
```

**Specification:** [UNIVERSAL_ACCESSIBILITY_SPECIFICATION.md](UNIVERSAL_ACCESSIBILITY_SPECIFICATION.md) (Section 3: Braille Galaxy)

---

## 13. Future Enhancements (Post-MVP)

### 13.1. Multi-User Tablets

**Shared environments (collaborative workspaces):**
- Multiple users in same House (co-located)
- Each user has their own tablet (personal view)
- Tablets can "sync" (share content, co-editing)

**Example:**
```
User A and User B in House:conference_room
  ↓
User A displays diagram on projection screen
  ↓
User B's tablet shows: "User A shared diagram - View on your tablet?"
  ↓
User B accepts → diagram appears on User B's tablet (procedural copy)
  ↓
User B annotates diagram → annotations visible to User A in real-time
```

---

### 13.2. AI Avatar Tablets

**AI has its own tablet (visible to humans):**

**Use case:** Human and AI work side-by-side in shared space.

**Example:**
```
Human: "AI, research this topic"
  ↓
AI: Spawns its own tablet (3D object visible to human)
  ↓
AI tablet displays: Research notes (procedural text)
  ↓
Human looks at AI's tablet → sees AI's "thought process" (notes, queries, results)
  ↓
Human: Taps AI's tablet → "Explain this step"
  ↓
AI: Highlights relevant section, adds explanation to tablet
```

**Why visible tablets?**
- **Transparency:** Human sees what AI is working on (builds trust)
- **Collaboration:** Human can inspect AI's intermediate results
- **Shared workspace:** AI and human = peers (not command-response)

---

### 13.3. Procedural Animation

**Tablet UI = animated (not static):**

**Examples:**
- **Button press:** Circle scales down (procedural tween), color shifts
- **Page transition:** Fade out old content, fade in new (alpha procedural)
- **Loading indicator:** Spinning circle (rotation RPN program with time parameter)

**Why procedural animation?**
- **Smooth:** No frame drops (calculated per-frame via PTX)
- **Scalable:** Animation speed adjusts to framerate (adaptive)
- **Accessible:** Animation can be disabled (user preference) without breaking UI

---

### 13.4. Cross-Device Sync

**Tablet state syncs across devices:**

**Example:**
```
User on Desktop (K3D running)
  ↓
User's tablet shows: House:my_notes, editing note
  ↓
User switches to VR headset (K3D VR mode)
  ↓
Tablet state syncs: Same note, same cursor position (seamless transition)
```

**Implementation:**
- Tablet state stored in House Universe (persistent)
- K3D runtime loads tablet state on startup (any device)
- Changes synced in real-time (WebSocket or similar)

---

## 14. Comparison to Traditional UIs

| Feature | Traditional GUI | K3D Memory Tablet |
|---------|----------------|-------------------|
| **Paradigm** | 2D windows, flat hierarchy | 3D object, spatial navigation |
| **Persistence** | Lost on app close | Persistent across sessions |
| **Accessibility** | Bolt-on (ARIA, screen readers) | Native (multi-modal architecture) |
| **AI access** | Vision model (OCR, pixel parsing) | Direct procedural content access |
| **Duplication** | Static UI elements (duplicated text) | Symlink-style references (70% compression) |
| **Rendering** | Rasterized (fixed resolution) | Procedural (infinite LOD) |
| **Interaction** | Mouse/touch only | Multi-modal (touch, voice, haptic, keyboard) |
| **Collaboration** | Screen sharing (separate views) | Dual-client (same object, different perceptions) |

---

## 15. Summary

### What Is the Memory Tablet?

**The Memory Tablet is K3D's interface layer** — the physical 3D object through which humans and AI navigate, manipulate, and create knowledge in spatial environments (House and Galaxy).

**Core properties:**
1. **Persistent:** Tablet state survives across sessions (like a physical notebook)
2. **Spatial:** Exists in 3D coordinates (not 2D overlay)
3. **Procedural:** Content = RPN programs (not rasterized pixels)
4. **Dual-client:** Humans see visuals, AI sees semantic structure
5. **Multi-modal:** Visual, audio, Braille, haptic (native accessibility)
6. **Collaborative:** Shared between human and AI (same object, different perceptions)

**Capabilities:**
- Browse Galaxy Universe (public knowledge)
- Browse House Universe (private workspace)
- Control projection screens (virtual monitors in 3D space)
- Access legacy content (files, VMs)
- Create new knowledge (diagrams, notes, procedures)

**Game-like UI:**
- Entry menu (choose House or Galaxy)
- Simultaneous environments (House + Galaxy run concurrently)
- Portal system (doorways between environments)
- In-game controls (no external UI needed)

**Procedural UI:**
- All UI elements = Galaxy references (symlink-style)
- Rendering via codecs (VectorDotMap, Procedural Fonts)
- 70%+ compression (zero duplication)
- AI-transparent (procedural content, not visual parsing)

---

## 16. Hardware Integration: HDMI as Procedural Protocol

### 16.1. The Insight: Displays Are Already Procedural

**Critical realization:** We don't need to invent a new way to drive displays — **HDMI is already a procedural protocol**.

**Traditional analog video (VGA, composite):**
```
Continuous voltage signals → Electron beam scanning → Phosphors glow
```

**Digital video (HDMI, DisplayPort):**
```
Digital packets → Pixel instructions → LCD/LED controller → Pixels update
```

**HDMI sends procedural commands:**
- Frame metadata (resolution: 1920×1080, refresh: 60Hz, color space: RGB)
- Pixel data packets: "Set pixel at (x,y) to RGB(r,g,b)"
- Control signals (blanking intervals, sync pulses)

**This is procedural!** Each frame = set of instructions to display controller.

---

### 16.2. K3D → HDMI Pipeline

**K3D renders procedurally all the way to the display:**

```
K3D Tablet Content (RPN programs)
  ↓
Procedural Codecs (VectorDotMap, Procedural Fonts)
  ↓
PTX Rendering Kernels (pixel_genesis, rasterization)
  ↓
Framebuffer (RGBA pixel array in VRAM)
  ↓
GPU Display Engine (scanout)
  ↓
HDMI Signal (digital pixel packets)
  ↓
Display Controller (decode packets)
  ↓
Physical Pixels (photons emitted)
```

**Every layer is procedural** — K3D extends the procedural paradigm from high-level (RPN) down to low-level (HDMI packets).

---

### 16.3. Abstraction Layers

**Traditional graphics stack (multiple translation layers):**
```
Application (OpenGL/Vulkan calls)
  → Driver (translate to GPU commands)
  → GPU (execute draw commands)
  → Display Engine (convert framebuffer to HDMI)
  → Monitor (convert HDMI to pixels)
```

**K3D's unified procedural stack:**
```
Tablet Content (RPN programs)
  → PTX Kernels (direct GPU execution, no driver translation)
  → Framebuffer (VRAM)
  → HDMI (native digital output)
  → Monitor (display pixels)
```

**Fewer translation layers = lower latency, deterministic output.**

---

### 16.4. Why This Matters for Projection Screens

**When K3D displays content on projection screens inside the 3D environment:**

**Option 1: Virtual screens (no hardware):**
- Screen = 3D object with procedural texture (VectorDotMap)
- Rendered entirely in K3D spatial environment
- No HDMI involved (purely virtual)

**Option 2: Physical screens (connected via HDMI):**
- K3D generates framebuffer via PTX kernels
- GPU display engine outputs to HDMI port
- Physical monitor displays content
- **Seamless integration** — same procedural source, different endpoints

**Example: Dual output**
```
K3D Tablet Content (RPN program: Math diagram)
  ↓
Render to TWO destinations:
  1. Virtual screen in K3D House (3D object texture)
  2. Physical monitor via HDMI (external display)
  ↓
Both show SAME content (procedural source), different physical/virtual targets
```

---

### 16.5. Procedural Display Protocol

**HDMI already provides the procedural abstraction K3D needs:**

| HDMI Feature | K3D Use |
|--------------|---------|
| **Resolution metadata** | Matches tablet/screen virtual resolution (e.g., 1024×768) |
| **Refresh rate** | Syncs with K3D frame rate (30/60/120 Hz) |
| **Pixel packets** | Direct output from PTX kernel framebuffer |
| **Color space** | RGB (matches procedural codec output) |
| **Display timing** | GPU handles automatically (scanout from VRAM) |

**No custom protocol needed** — HDMI's digital packets = procedural instructions K3D already generates.

---

### 16.6. Future: Direct Procedural Output

**Current (MVP):** K3D → framebuffer → HDMI → display
- Pro: Works with all HDMI monitors (standard protocol)
- Con: Framebuffer = rasterized (loses procedural representation)

**Future (Post-MVP):** K3D → procedural packets → HDMI-like protocol → procedural display
- Pro: Display decodes procedural commands directly (infinite LOD, no rasterization)
- Con: Requires custom display hardware (doesn't exist yet)

**Example procedural HDMI extension:**
```
Instead of: "Pixel (100, 100) = RGB(255, 0, 0)"  (rasterized)
Send: "CIRCLE x=100 y=100 r=50 color=#FF0000"  (procedural)
```

**Display with procedural decoder:**
- Receives VectorDotMap commands via HDMI
- Decodes procedurally (hardware VectorDotMap renderer)
- Infinite LOD (zoom → display re-executes procedural commands at higher resolution)

**This is speculative** — current HDMI is pixel-based. But K3D's architecture is **ready** for procedural displays when hardware catches up.

---

### 16.7. Reference: Display as Procedural Pipeline

**From ATTRIBUTIONS.md:**
> Frame "monitor reality" as the final stage of a **procedural pipeline**: RPN programs → GPU commands → pixels → photons

**K3D's contribution:**
- Extends procedural paradigm from application layer (RPN programs) down to pixel layer (HDMI packets)
- Treats display as **endpoint of procedural execution** (not separate rendering concern)
- Positions K3D for future **procedural display hardware** (no changes to K3D architecture needed)

---

## 17. References

**Related Specifications:**
- [SPATIAL_UI_ARCHITECTURE_SPECIFICATION.md](SPATIAL_UI_ARCHITECTURE_SPECIFICATION.md) — Houses, Rooms, Portals
- [DUAL_CLIENT_CONTRACT_SPECIFICATION.md](DUAL_CLIENT_CONTRACT_SPECIFICATION.md) — Dual-client paradigm
- [KNOWLEDGEVERSE_SPECIFICATION.md](KNOWLEDGEVERSE_SPECIFICATION.md) — 7-region VRAM architecture
- [UNIVERSAL_ACCESSIBILITY_SPECIFICATION.md](UNIVERSAL_ACCESSIBILITY_SPECIFICATION.md) — Multi-modal accessibility
- [PROCEDURAL_VISUAL_SPECIFICATION.md](PROCEDURAL_VISUAL_SPECIFICATION.md) — VectorDotMap codec
- [ADAPTIVE_PROCEDURAL_COMPRESSION_SPECIFICATION.md](ADAPTIVE_PROCEDURAL_COMPRESSION_SPECIFICATION.md) — PD04 codec

**PM-KR Context:**
- [docs/W3C/PM_KR_STRATEGIC_STEERING.md](../W3C/PM_KR_STRATEGIC_STEERING.md) — Developer Familiarity imperative (Section 5)
- [docs/W3C/PM_KR_PROBLEM_STATEMENT.md](../W3C/PM_KR_PROBLEM_STATEMENT.md) — UI duplication crisis

**Historical Inspiration:**
- **Tony Parisi (WebXR pioneer):** "Software as space, not window"
- **Ivan Sutherland (Sketchpad, 1963):** Direct manipulation interfaces
- **Doug Engelbart (NLS, 1968):** Hypertext, collaborative computing
- **Apple iPad (2010):** Touch-first interface paradigm

---

**Version:** 1.0
**Status:** Architectural Design (Post-MVP Implementation)
**Next Milestone:** Phase P (Q1 2027) — Tablet + Projection Screen Prototyping
**Maintained by:** Daniel Ramos (K3D Architect)

---

**END OF MEMORY TABLET SPECIFICATION**
