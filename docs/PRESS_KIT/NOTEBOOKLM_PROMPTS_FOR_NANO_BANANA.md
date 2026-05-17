# NotebookLM Prompts for Nano Banana (Visual Assets Generation)

**Date:** March 4, 2026
**Purpose:** Generate infographics, slides, and video scripts from PM-KR press kit
**Target:** Nano Banana (NotebookLM Graphics Generation)
**Human modem:** Daniel Ramos

---

## Setup Instructions (For Daniel)

1. **Upload entire `docs/PRESS_KIT/` folder to NotebookLM**
2. **Copy-paste prompts below** (one at a time)
3. **Download generated assets** (PNG, PDF, or script text)
4. **Save to** `docs/PRESS_KIT/08_MEDIA_RESOURCES/generated_assets/`

---

## Prompt 1: Carbon Impact Timeline Infographic

**Copy-paste this to NotebookLM:**

```
Generate a professional infographic showing PM-KR's carbon impact projection over 10 years (2026-2035).

Data source: docs/PRESS_KIT/05_CARBON_IMPACT/infographic_data.json

Visual requirements:
- Title: "PM-KR Carbon Impact Projection"
- Subtitle: "10-Year Cumulative CO₂ Savings"
- X-axis: Years (2026 to 2035)
- Y-axis: Gigatons CO₂ saved (0 to 5.5 Gt)
- Data points:
  - 2026: 0.05 Gt
  - 2027: 0.15 Gt
  - 2028: 0.35 Gt
  - 2029: 0.70 Gt
  - 2030: 1.20 Gt
  - 2031: 1.80 Gt
  - 2032: 2.50 Gt
  - 2033: 3.30 Gt
  - 2034: 4.20 Gt
  - 2035: 5.20 Gt
- Cumulative total callout: "12 Gt CO₂ cumulative savings (2026-2035)"
- Comparison callout: "Equivalent to global aviation emissions × 2"
- Footer disclaimer: "Scenario projection based on procedural compression adoption modeling"

Style guide:
- Clean, professional, conference-ready
- Color scheme: Green (carbon savings), blue (technology), earth tones
- Readable font (sans-serif, 14pt+ for labels)
- Include PM-KR logo placement area
- Export as PNG (1920x1080) and PDF (print-ready)

Sources cited:
- docs/CARBON_BLUEPRINT_10_YEAR_PROJECTION.md
- docs/PRESS_KIT/05_CARBON_IMPACT/carbon_blueprint_summary.md
```

**Expected output:** PNG infographic + PDF version

---

## Prompt 2: Seven-Region Knowledgeverse Architecture Diagram

**Copy-paste this to NotebookLM:**

```
Generate a technical architecture diagram showing PM-KR's seven-region Knowledgeverse unified memory substrate.

Data source: docs/PRESS_KIT/03_TECHNICAL_OVERVIEW/architecture_diagram.md

Visual layout (top-to-bottom flow):

1. **Top layer: External Sources**
   - Box: "External Sources"
   - Labels: PDFs, Benchmarks, Documents
   - Arrow down to Ingestion

2. **Ingestion layer:**
   - Box: "Ingestion + Augmentation"
   - Sub-labels: classify → enrich → map
   - Arrow down to Knowledgeverse

3. **Main container: Knowledgeverse (7 regions)**
   - Large container box labeled "KNOWLEDGEVERSE (Unified Arena)"
   - Internal layout (3x2 grid + 1):
     - Region 1: Kernels (PTX modules)
     - Region 2: Galaxy Universe (active reasoning memory)
     - Region 3: House (persistent memory)
     - Region 4: World View (network/collaboration)
     - Region 5: TRM Weights (routing/specialists)
     - Region 6: Audit (trace/provenance)
     - Region 7: Ingestion/Staging (external, feeds into Knowledgeverse)
   - Arrow down to Runtime

4. **Runtime layer:**
   - Box: "Cranium Runtime"
   - Sub-labels: route → retrieve → compose → execute → return → log
   - Arrow down to Clients

5. **Bottom layer: Dual-Client Reality**
   - Two boxes side-by-side:
     - Left: "Human Client (visual)"
     - Right: "AI Client (semantic)"
   - Center: "Shared Data" (connecting both)
   - Bidirectional arrows showing symmetry

Style guide:
- Clean technical infographic (no decorative effects)
- Color coding:
  - Blue: Memory regions (Galaxy, House)
  - Green: Execution (Cranium, TRM)
  - Orange: Ingestion/Staging
  - Purple: Audit/Provenance
- Low-noise labels (readable at conference size)
- Arrows show data flow (top-down + bidirectional at bottom)
- Export as PNG (1920x1080) and SVG (vector, scalable)

Key message: "One shared procedural memory substrate supports ingestion, execution, and audit with dual-client consistency"

Sources cited:
- docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md
- docs/vocabulary/PROCEDURAL_MEMORY_KR_STANDARD_SPECIFICATION.md
- docs/W3C/PM_KR_NORMATIVE_MODEL.md
```

**Expected output:** PNG diagram + SVG vector version

---

## Prompt 3: Comparison Chart (PM-KR vs LLMs vs RAG)

**Copy-paste this to NotebookLM:**

```
Generate a comparison table infographic showing PM-KR vs Traditional LLMs vs RAG Pipelines across 10 dimensions.

Data source: docs/PRESS_KIT/03_TECHNICAL_OVERVIEW/comparison_chart.md

Table structure:

Columns:
1. Dimension (left-aligned)
2. Traditional LLMs
3. RAG Pipelines
4. PM-KR Procedural Memory (highlighted)

Rows (10 dimensions):
1. Core storage model
2. Typical model size reference
3. Latency transparency
4. Explainability mode
5. Composability
6. Duplication pressure
7. Human and AI source parity
8. Governance controls
9. Carbon profile tendency
10. Standards posture

Key differentiators to highlight (PM-KR column):
- 7M parameters (vs 100B-1T+ for LLMs)
- Deterministic operation traces (vs opaque for LLMs)
- Hard traceability (vs post-hoc narrative)
- Reference-first model (vs high duplication)
- Explicit dual-client contract (vs indirect for LLMs)
- W3C specification path (vs vendor-defined)

Style guide:
- Table format with alternating row colors (light gray/white)
- PM-KR column highlighted (light green background)
- Icons for key differentiators (checkmark ✓ for advantages)
- Footer: "PM-KR values based on K3D published reports and W3C drafts. Carbon figures are scenario projections."
- Export as PNG (1920x1080) and PDF (print-ready)

Sources cited:
- docs/W3C/PM_KR_NORMATIVE_MODEL.md
- docs/W3C/PM_KR_EVIDENCE_VALIDATION_MATRIX.md
- docs/CARBON_BLUEPRINT_10_YEAR_PROJECTION.md
```

**Expected output:** PNG table infographic + PDF version

---

## Prompt 4: 10-Slide Presentation Deck

**Copy-paste this to NotebookLM:**

```
Generate a 10-slide presentation deck outline for PM-KR / K3D press kit.

Data sources: Entire docs/PRESS_KIT/ folder

Slide structure:

**Slide 1: Title Slide**
- Title: "Procedural Memory Knowledge Representation (PM-KR)"
- Subtitle: "W3C Community Group | Brazil-Netherlands MERCOSUR-EU Collaboration"
- Logos: W3C, EchoSystems AI Studios, Rainbow Warriors Core Foundation
- Footer: daniel@echosystems.ai | rwiciamsd@gmail.com

**Slide 2: The Problem**
- Title: "Knowledge Duplication Crisis"
- Visual: 6 duplicate representations of letter 'A' (fonts, embeddings, accessibility, etc.)
- Key stat: "70%+ of knowledge duplicated across systems"
- Tagline: "Same knowledge stored 6+ times in incompatible formats"

**Slide 3: The Solution**
- Title: "Procedural Memory: Store Once, Reference Forever"
- Visual: One canonical source → multiple consumers (humans, AI, accessibility)
- Key concept: "Dual-client contract: Humans and AI consume identical procedural data"

**Slide 4: Key Personnel**
- Title: "MERCOSUR-EU Partnership"
- Photos: Daniel Ramos (Brazil) + Milton Ponson (Netherlands) + Christoph Dorn (Canada)
- Quote: "First groundbreaking MERCOSUR-EU joint effort in frontier technologies"

**Slide 5: Architecture**
- Title: "Seven-Region Knowledgeverse"
- Visual: Architecture diagram (from Prompt 2 above)
- Key message: "Unified procedural memory substrate"

**Slide 6: Carbon Impact**
- Title: "12 Gt CO₂ Projected Savings (2026-2035)"
- Visual: Carbon timeline infographic (from Prompt 1 above)
- Comparison: "Equivalent to global aviation emissions × 2"

**Slide 7: W3C Collaboration**
- Title: "Cross-CG Synergy"
- Visual: Network diagram showing PM-KR connections:
  - WebML CG (Intel/Anssi)
  - GPU for the Web WG (Mozilla/Google)
  - Sustainable Web IG (Tzviya Siegman)
  - Web Fonts WG (Chris Lilley)
  - Immersive Web WG (Ada Rose Cannon/Samsung)

**Slide 8: Use Cases**
- Title: "Procedural Displays Revolution"
- Four quadrants:
  1. E-readers without font files (E Ink)
  2. Infinite zoom accessibility
  3. Power-efficient OLED (Samsung)
  4. Multi-language zero-cost

**Slide 9: Comparison**
- Title: "PM-KR vs Traditional AI"
- Visual: Comparison chart (from Prompt 3 above)
- Key differentiator: "7M params vs 175B+ params = 25,000× smaller"

**Slide 10: Call to Action**
- Title: "Join PM-KR Community Group"
- QR code: https://www.w3.org/community/pm-kr/
- Resources:
  - GitHub: github.com/danielcamposramos/Knowledge3D
  - Carbon Blueprint: [link]
  - WebML Issue #17: [link]
- Contact: daniel@echosystems.ai

Style guide:
- Clean, professional, W3C-appropriate
- Color scheme: Blue (technology), green (carbon), purple (collaboration)
- Minimal text per slide (conference presentation style)
- Large fonts (24pt+ body, 36pt+ titles)
- Export as PDF (slides) and individual PNGs

Sources: Entire docs/PRESS_KIT/ folder
```

**Expected output:** PDF presentation deck + individual PNG slides

---

## Prompt 5: Explainer Video Script (8-10 minutes)

**Copy-paste this to NotebookLM:**

```
Generate an 8-10 minute explainer video script for NotebookLM Audio Overview (narration).

Data sources:
- docs/PRESS_KIT/01_EXECUTIVE_SUMMARY/elevator_pitch.md (5-minute section)
- docs/PRESS_KIT/05_CARBON_IMPACT/carbon_blueprint_summary.md
- docs/PRESS_KIT/03_TECHNICAL_OVERVIEW/what_is_procedural_memory.md

Script structure:

**[INTRO - 30 seconds]**
Hook: "What if AI memory wasn't locked inside billions of parameters, but lived outside — as explorable universes we can inspect together?"

Introduction: "Welcome to Procedural Memory Knowledge Representation, a W3C Community Group redefining how humans and AI share knowledge."

**[PROBLEM - 90 seconds]**
The duplication crisis:
- Same knowledge duplicated 6+ times (fonts, embeddings, accessibility, documentation)
- 70%+ storage waste
- Maintenance nightmare (updates require manual synchronization)
- AI carbon footprint: training duplicates Wikipedia, textbooks, documentation across every company

Visual: Show letter 'A' duplicated across 6 systems

**[SOLUTION - 2 minutes]**
Procedural Memory approach:
- Store knowledge ONCE as executable programs + metadata
- All consumers reference same canonical source
- Dual-client contract: Humans see visual rendering, AI reads semantic metadata — SAME source

Example: Character 'A'
- Visual rendering: Bézier curves → pixels
- Audio rendering: Screen reader pronunciation
- Tactile rendering: Braille dots
- AI semantic: Unicode properties, language metadata

ONE source, four modalities. Zero duplication.

**[ARCHITECTURE - 90 seconds]**
Seven-Region Knowledgeverse:
- Galaxy Universe: Active reasoning memory (Drawing, Character, Word, Grammar, Math, Reality, Audio galaxies)
- House Universe: Persistent memory (long-term knowledge storage)
- Cranium Runtime: PTX-native execution (sovereign, zero external dependencies)
- TRM: Tiny Reasoning Module (7M parameters, lightweight inference)

Visual: Architecture diagram animation (top-down flow)

**[CARBON IMPACT - 2 minutes]**
The sustainability case:
- 12 Gigatons CO₂ cumulative savings projected (2026-2035)
- 2.5 Gt/year by 2035 = 6.9% of global emissions
- Mechanism: Compression (70%+ reduction) + lightweight reasoning (procedural vs LLM inference)
- Comparison: Equivalent to global aviation emissions × 2

Visual: Carbon timeline infographic animation (year-by-year growth)

**[W3C COLLABORATION - 90 seconds]**
Cross-CG momentum:
- WebML CG: Intel/Anssi collaboration (Issue #17)
- GPU for the Web WG: Mozilla/Google procedural rendering integration
- Sustainable Web IG: Web Sustainability Guidelines alignment
- Web Fonts WG: Chris Lilley procedural fonts evolution
- Samsung: Immersive Web WG synergy (Ada Rose Cannon)

Visual: Network diagram showing W3C ecosystem

**[MERCOSUR-EU PARTNERSHIP - 60 seconds]**
Historic collaboration:
- EchoSystems AI Studios (Brazil) + Rainbow Warriors Core Foundation (Netherlands)
- Daniel Ramos (electrical engineer) + Milton Ponson (mathematician, 30 years environmental knowledge)
- First groundbreaking MERCOSUR-EU joint effort in frontier technologies
- PM-KR Community Group co-chairs

Visual: Photos of Daniel + Milton, Brazil/Netherlands flags

**[USE CASES - 90 seconds]**
Procedural displays revolution:
- E-readers without font files (100,000+ Unicode characters, zero storage)
- Infinite zoom accessibility (visually impaired users, no pixelation)
- Power-efficient OLED rendering (Samsung, procedural composition reduces GPU load)
- Multi-language zero-cost (Chinese, Arabic, Cyrillic = procedural glyphs)

Visual: E-reader mockup showing infinite zoom

**[CALL TO ACTION - 30 seconds]**
Join the movement:
- PM-KR Community Group: w3.org/community/pm-kr/
- GitHub: github.com/danielcamposramos/Knowledge3D
- Contact: daniel@echosystems.ai, rwiciamsd@gmail.com

Closing: "Software was always meant to be a place, not a window. Welcome home."

**[TOTAL: 8.5 minutes]**

Narration style:
- Professional, inspirational, accessible
- NotebookLM Audio Overview tone (conversational but authoritative)
- Pause for visual transitions (infographics, diagrams)
- Emphasize key stats (12 Gt CO₂, 7M params, 70%+ compression)

Visual assets needed:
- Architecture diagram animation
- Carbon timeline infographic animation
- Comparison chart
- E-reader mockup
- Team photos

Export as:
- Audio MP3 (NotebookLM narration)
- Script TXT (for video editing)
- Timestamp markers (for syncing visuals)
```

**Expected output:** Audio MP3 + script TXT + timestamp markers

---

## Prompt 6: Team Photo Mockup (Virtual Composition)

**Copy-paste this to NotebookLM:**

```
Generate instructions for virtual team photo composition (for graphic designer or AI image generator).

Data source: docs/PRESS_KIT/02_KEY_PERSONNEL/team_photo.md

Composition requirements:

**Layout:** Horizontal trio (left-center-right)

**Subjects:**
- Left: Daniel Ramos (Brazil) - Electrical engineer, PM-KR Co-Chair
- Center: Milton Ponson (Netherlands) - Mathematician, PM-KR Co-Chair
- Right: Christoph Dorn (Canada) - Sovereignty architect

**Background elements:**
- Subtle W3C logo (watermark, center top)
- MERCOSUR flag (left side, faded)
- EU flag (center-left, faded)
- Canadian maple leaf (right side, subtle)

**Style:**
- Professional, formal but approachable
- Conference-ready (suitable for press releases)
- Color: Business professional attire
- Lighting: Studio-quality, even lighting
- Resolution: 1920x1080 (landscape) or 1080x1080 (square for social media)

**Text overlay (optional):**
- Top: "PM-KR Community Group Co-Chairs + Senior Contributors"
- Bottom: Names with country flags
  - 🇧🇷 Daniel Ramos (Brazil)
  - 🇳🇱 Milton Ponson (Netherlands)
  - 🇨🇦 Christoph Dorn (Canada)

**Export formats:**
- PNG with transparent background (for press materials)
- JPG with white background (for web use)
- Square crop (1080x1080) for social media

Note: Since we don't have actual photos yet, generate as:
1. Placeholder with silhouettes + text overlay (immediate use)
2. Instructions for photo shoot (when actual photos available)
```

**Expected output:** Placeholder mockup + photo shoot instructions

---

## Prompt 7: Social Media Graphics (Twitter/LinkedIn)

**Copy-paste this to NotebookLM:**

```
Generate social media graphics for Twitter/LinkedIn posts.

Data source: docs/PRESS_KIT/08_MEDIA_RESOURCES/social_media_copy.md

**Graphic 1: Carbon Impact**
- Dimensions: 1200x675 (Twitter/LinkedIn optimized)
- Main stat: "12 Gt CO₂"
- Subtitle: "Projected savings by 2035"
- Visual: Carbon timeline mini-chart (simplified from infographic)
- Footer: w3.org/community/pm-kr | #ProceduralMemory #SustainableAI
- Export: PNG

**Graphic 2: MERCOSUR-EU Collaboration**
- Dimensions: 1080x1080 (Instagram/LinkedIn square)
- Main text: "First MERCOSUR-EU Collaboration in Frontier Tech"
- Flags: Brazil 🇧🇷 + Netherlands 🇳🇱
- Subtitle: "PM-KR Community Group"
- Footer: daniel@echosystems.ai | rwiciamsd@gmail.com
- Export: PNG

**Graphic 3: W3C Ecosystem**
- Dimensions: 1200x675
- Main text: "Cross-CG Synergy"
- Visual: Network nodes showing:
  - PM-KR (center, highlighted)
  - WebML (Intel)
  - GPU for the Web (Mozilla/Google)
  - Sustainable Web IG
  - Web Fonts WG
  - Immersive Web WG (Samsung)
- Footer: "Building the procedural web" | #W3C #WebStandards
- Export: PNG

**Graphic 4: Procedural Display Revolution**
- Dimensions: 1080x1080
- Main text: "E-readers without font files"
- Visual: E-reader mockup with "∞" symbol (infinite zoom)
- Subtitle: "100,000+ Unicode characters, zero storage"
- Footer: "Join PM-KR CG" | w3.org/community/pm-kr
- Export: PNG

Style guide (all graphics):
- Color scheme: Blue (technology), green (carbon), purple (collaboration)
- Professional fonts (sans-serif, bold for stats)
- PM-KR branding consistent
- Readable on mobile (large text, high contrast)
```

**Expected output:** 4 social media graphics (PNG)

---

## Summary: Assets to Generate

| # | Asset | Prompt | Output Format | Priority |
|---|-------|--------|---------------|----------|
| 1 | Carbon Impact Infographic | Prompt 1 | PNG + PDF | HIGH ⭐⭐⭐ |
| 2 | Architecture Diagram | Prompt 2 | PNG + SVG | HIGH ⭐⭐⭐ |
| 3 | Comparison Chart | Prompt 3 | PNG + PDF | HIGH ⭐⭐⭐ |
| 4 | Presentation Deck | Prompt 4 | PDF + PNG slides | MEDIUM ⭐⭐ |
| 5 | Explainer Video Script | Prompt 5 | Audio MP3 + TXT | MEDIUM ⭐⭐ |
| 6 | Team Photo Mockup | Prompt 6 | PNG + JPG | LOW ⭐ |
| 7 | Social Media Graphics | Prompt 7 | 4 PNGs | MEDIUM ⭐⭐ |

---

## Workflow (For Daniel)

### Step 1: Upload Press Kit to NotebookLM
1. Open NotebookLM: https://notebooklm.google.com/
2. Create new notebook: "PM-KR Press Kit Assets"
3. Upload entire `docs/PRESS_KIT/` folder (drag-and-drop)
4. Wait for indexing (~2-3 minutes)

### Step 2: Generate Assets (Priority Order)
1. **Carbon Impact Infographic** (Prompt 1) - needed for all media outreach
2. **Architecture Diagram** (Prompt 2) - needed for technical audiences
3. **Comparison Chart** (Prompt 3) - needed for differentiation messaging
4. **Social Media Graphics** (Prompt 7) - immediate use for announcing display manufacturer invitations
5. **Presentation Deck** (Prompt 4) - needed for conference submissions
6. **Explainer Video Script** (Prompt 5) - next YouTube video
7. **Team Photo Mockup** (Prompt 6) - low priority until actual photos available

### Step 3: Save Generated Assets
- Create folder: `docs/PRESS_KIT/08_MEDIA_RESOURCES/generated_assets/`
- Save all generated files with descriptive names:
  - `carbon_impact_infographic_2026-03-04.png`
  - `architecture_diagram_knowledgeverse_2026-03-04.svg`
  - `comparison_chart_pmkr_vs_llms_2026-03-04.pdf`
  - etc.

### Step 4: Update Press Kit README
- Add "Generated Assets" section to `docs/PRESS_KIT/README.md`
- Link to all generated files
- Include generation date and NotebookLM notebook link

---

## Notes for Daniel

**NotebookLM limitations:**
- May not generate visual assets directly (depends on NotebookLM capabilities as of March 2026)
- If NotebookLM generates text descriptions instead of images → use descriptions as input for:
  - Canva (drag-and-drop design tool)
  - Figma (professional design)
  - DALL-E / Midjourney (AI image generation)
  - Nano Banana (if separate graphics tool)

**Human modem task:**
- Copy prompt → Paste to NotebookLM → Wait for output → Copy output → Paste to Canva/Figma/etc.
- Iterative refinement: If output doesn't match expectations, adjust prompt and regenerate

**Quality check:**
- All stats match source documents (carbon: 12 Gt, TRM: 7M params, compression: 70%+)
- MERCOSUR-EU narrative verbatim (Milton's canonical text)
- W3C branding consistent (professional, standards-appropriate)
- Export formats suitable for press kit distribution (PNG, PDF, SVG)

---

**Last Updated:** March 4, 2026
**Created by:** Claude (Architecture Partner)
**For:** Nano Banana graphics generation via NotebookLM
**Human modem:** Daniel Ramos
