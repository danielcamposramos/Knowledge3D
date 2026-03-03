# NotebookLM Audio Overview & Data Table Prompts

**Date:** March 4, 2026
**Purpose:** Generate podcast-style audio + data visualizations from PM-KR press kit
**Target:** NotebookLM Audio Overview + Data Table features
**Human modem:** Daniel Ramos

---

## Setup Instructions (For Daniel)

1. **Upload entire `docs/PRESS_KIT/` folder to NotebookLM**
2. **For Audio Overview:** Click "Audio Overview" button (podcast-style generation)
3. **For Data Table:** Click "Generate data table" button (new feature)
4. **Download outputs:** Audio MP3 + Data CSV/JSON

---

## 🎙️ Prompt 1: Audio Overview (Podcast-Style Explainer)

**NotebookLM Feature:** Audio Overview (generates ~10 minute podcast-style conversation)

**Copy-paste this to NotebookLM chat (before clicking "Audio Overview"):**

```
Generate an Audio Overview podcast explaining PM-KR (Procedural Memory Knowledge Representation) for a general tech audience.

Focus on these key topics:

1. THE PROBLEM (2 minutes)
   - Knowledge duplication crisis: Same knowledge stored 6+ times across systems
   - Example: Letter 'A' duplicated in fonts, embeddings, accessibility, AI training data
   - Carbon impact: Every company duplicates Wikipedia, textbooks, documentation
   - Source: docs/PRESS_KIT/03_TECHNICAL_OVERVIEW/what_is_procedural_memory.md

2. THE SOLUTION (3 minutes)
   - Procedural Memory: Store knowledge ONCE as executable programs + metadata
   - Dual-client contract: Humans and AI consume identical procedural data
   - Example: Character 'A' renders visually (Bézier curves), audibly (pronunciation),
     tactilely (Braille), and semantically (AI metadata) from ONE source
   - Source: docs/PRESS_KIT/01_EXECUTIVE_SUMMARY/elevator_pitch.md

3. CARBON IMPACT (2 minutes)
   - 12 Gigatons CO₂ cumulative savings projected (2026-2035)
   - 2.5 Gt/year by 2035 = 6.9% of global emissions
   - Mechanism: 70%+ compression + lightweight reasoning (7M params vs 175B+ LLMs)
   - Comparison: Equivalent to global aviation emissions × 2
   - Source: docs/PRESS_KIT/05_CARBON_IMPACT/carbon_blueprint_summary.md

4. W3C COLLABORATION (2 minutes)
   - Cross-CG momentum: WebML (Intel), GPU for the Web (Mozilla/Google),
     Sustainable Web IG, Web Fonts WG (Chris Lilley), Samsung (Ada Rose Cannon)
   - Display manufacturers: Procedural displays (E-readers without font files, infinite zoom)
   - Source: docs/PRESS_KIT/06_W3C_COLLABORATION/cross_cg_synergy.md

5. MERCOSUR-EU PARTNERSHIP (1 minute)
   - Historic collaboration: Brazil (Daniel Ramos, electrical engineer) +
     Netherlands (Milton Ponson, mathematician)
   - First groundbreaking MERCOSUR-EU joint effort in frontier technologies
   - W3C PM-KR Community Group co-chairs
   - Source: docs/PRESS_KIT/07_MERCOSUR_EU_PARTNERSHIP/historic_significance.md

TONE: Conversational, accessible, enthusiastic
AUDIENCE: Tech journalists, display manufacturers, W3C members
KEY STATS: Emphasize 12 Gt CO₂, 70%+ compression, 7M params, MERCOSUR-EU "first"

After generating, use the "Audio Overview" button to create podcast-style narration.
```

**Expected output:**
- Audio MP3 (~10 minutes, podcast-style conversation)
- Downloadable for YouTube, press materials, conference submissions

**Usage:**
- YouTube explainer video (add visuals from infographics)
- Press kit audio companion
- Conference booth audio loop
- Social media snippets (extract 30-60 second clips)

---

## 📊 Prompt 2A: Data Table - Carbon Impact Timeline

**NotebookLM Feature:** Generate data table (new feature, creates structured CSV/JSON)

**Copy-paste this to NotebookLM chat (before clicking "Generate data table"):**

```
Generate a data table showing PM-KR carbon impact projection timeline (2026-2035).

Table structure:

Column 1: Year (2026 to 2035)
Column 2: Annual CO₂ Savings (Gigatons)
Column 3: Cumulative CO₂ Savings (Gigatons)
Column 4: % of Global Emissions (annual)
Column 5: Comparison Metric

Data source: docs/PRESS_KIT/05_CARBON_IMPACT/infographic_data.json

Row data:
- 2026: 0.05 Gt annual, 0.05 Gt cumulative, 0.14% global, "Equivalent to 50M cars removed"
- 2027: 0.10 Gt annual, 0.15 Gt cumulative, 0.28% global, "Equivalent to 100M cars removed"
- 2028: 0.20 Gt annual, 0.35 Gt cumulative, 0.56% global, "Equivalent to Belgium's annual emissions"
- 2029: 0.35 Gt annual, 0.70 Gt cumulative, 0.97% global, "Equivalent to Netherlands' annual emissions"
- 2030: 0.50 Gt annual, 1.20 Gt cumulative, 1.39% global, "Equivalent to Spain's annual emissions"
- 2031: 0.60 Gt annual, 1.80 Gt cumulative, 1.67% global, "Equivalent to Germany's annual emissions"
- 2032: 0.70 Gt annual, 2.50 Gt cumulative, 1.94% global, "Equivalent to Japan's annual emissions"
- 2033: 0.80 Gt annual, 3.30 Gt cumulative, 2.22% global, "Equivalent to Russia's annual emissions"
- 2034: 0.90 Gt annual, 4.20 Gt cumulative, 2.50% global, "Equivalent to global aviation industry"
- 2035: 1.00 Gt annual, 5.20 Gt cumulative, 2.78% global, "Equivalent to global shipping industry"

TOTAL CUMULATIVE (2026-2035): 12.0 Gt CO₂ savings
COMPARISON: Equivalent to global aviation emissions × 2

METHODOLOGY NOTE: Scenario projection based on procedural compression adoption modeling
(70%+ compression + lightweight reasoning vs LLM inference). Full methodology in
docs/CARBON_BLUEPRINT_10_YEAR_PROJECTION.md

After generating, use the "Generate data table" button to create structured output.
```

**Expected output:**
- CSV file (Excel-compatible)
- JSON file (programmatic access)
- Markdown table (for documentation)

**Usage:**
- Press releases (copy-paste table)
- Grant applications (structured data)
- Conference presentations (import to slides)
- Academic papers (citation-ready data)

---

## 📊 Prompt 2B: Data Table - PM-KR vs LLMs vs RAG Comparison

**Copy-paste this to NotebookLM chat (before clicking "Generate data table"):**

```
Generate a comparison data table: PM-KR vs Traditional LLMs vs RAG Pipelines.

Table structure:

Column 1: Dimension (10 rows)
Column 2: Traditional LLMs
Column 3: RAG Pipelines
Column 4: PM-KR Procedural Memory

Data source: docs/PRESS_KIT/03_TECHNICAL_OVERVIEW/comparison_chart.md

Row data:

1. Core storage model
   - LLMs: Knowledge internal to model weights
   - RAG: External retrieval + model synthesis
   - PM-KR: Canonical procedural nodes + references

2. Typical model size
   - LLMs: 100B to 1T+ parameters (700GB+)
   - RAG: Depends on downstream model (50B-175B typical)
   - PM-KR: 7M parameters (28MB)

3. Latency
   - LLMs: 142ms (transformer inference, GPT-3 baseline)
   - RAG: Depends on retrieval + LLM (200-500ms typical)
   - PM-KR: <1ms (RPN stack-based execution)

4. Explainability
   - LLMs: Opaque (post-hoc narrative only)
   - RAG: Partial (retrieval context visible, synthesis opaque)
   - PM-KR: Hard traceability (deterministic operation traces)

5. Composability
   - LLMs: Monolithic (weight updates only)
   - RAG: Prompt chains + retriever orchestration
   - PM-KR: Layered form→meaning→rules→meta-rules composition

6. Duplication pressure
   - LLMs: High (data replication across training/serving)
   - RAG: Moderate to high (indexes + snapshots)
   - PM-KR: Reference-first (70%+ compression, canonical forms)

7. Human and AI source parity
   - LLMs: Indirect (humans can't inspect weights)
   - RAG: Partial (retrieval visible, synthesis hidden)
   - PM-KR: Explicit dual-client contract (same source for both)

8. Governance controls
   - LLMs: Policy wrappers around model behavior
   - RAG: Policy + retrieval guardrails
   - PM-KR: Boundary contracts + auditable procedural lineage

9. Carbon footprint
   - LLMs: High (175B params = 700GB, GPU clusters for training/inference)
   - RAG: Moderate to high (depends on model size + retrieval infrastructure)
   - PM-KR: Low (7M params = 28MB, procedural compression reduces compute)

10. Standards posture
    - LLMs: Vendor-defined interfaces (OpenAI API, Anthropic API, etc.)
    - RAG: Framework-specific (LangChain, LlamaIndex, proprietary)
    - PM-KR: W3C Community Group standardization (open standards path)

KEY DIFFERENTIATOR: PM-KR achieves 25,000× smaller model size (7M vs 175B params)
with hard explainability and dual-client transparency.

SOURCES:
- docs/W3C/PM_KR_NORMATIVE_MODEL.md
- docs/W3C/PM_KR_EVIDENCE_VALIDATION_MATRIX.md
- docs/CARBON_BLUEPRINT_10_YEAR_PROJECTION.md

After generating, use the "Generate data table" button to create structured output.
```

**Expected output:**
- CSV file (Excel-compatible)
- JSON file (programmatic access)
- Markdown table (for documentation)

**Usage:**
- Press releases (competitive differentiation)
- Display manufacturer pitches (technical comparison)
- W3C presentations (standards positioning)
- Grant applications (innovation justification)

---

## 📊 Prompt 2C: Data Table - W3C Cross-CG Collaboration Matrix

**Copy-paste this to NotebookLM chat (before clicking "Generate data table"):**

```
Generate a W3C cross-CG collaboration matrix for PM-KR Community Group.

Table structure:

Column 1: W3C Group
Column 2: Key Contact
Column 3: Collaboration Focus
Column 4: Status
Column 5: Strategic Value

Data source: docs/PRESS_KIT/06_W3C_COLLABORATION/cross_cg_synergy.md

Row data:

1. WebML CG
   - Contact: Anssi Kostiainen (Intel)
   - Focus: Procedural reasoning substrates for WebNN (Issue #17)
   - Status: Proposal submitted, awaiting maintainer review
   - Value: Intel NPU integration, hardware manufacturer credibility

2. GPU for the Web WG
   - Contact: Jim Blandy (Mozilla), Corentin Wallez (Google)
   - Focus: Procedural rendering integration with WebGPU
   - Status: Invitation sent, collaboration discussion initiated
   - Value: Browser vendor support (Chrome, Firefox), GPU-accelerated procedural rendering

3. Web Fonts WG
   - Contact: Chris Lilley (W3C Technical Director)
   - Focus: Procedural fonts evolution (building on WOFF legacy)
   - Status: Invitation sent, bridge to Adobe/Google Fonts teams
   - Value: Font industry standardization, multi-modal accessibility

4. Sustainable Web IG
   - Contact: Tzviya Siegman (W3C Director of Sustainability)
   - Focus: Carbon impact alignment (12 Gt CO₂ projection)
   - Status: Email sent, positioning as Web Sustainability Guidelines case study
   - Value: Environmental credibility, policy maker visibility

5. Immersive Web WG
   - Contact: Ada Rose Cannon (Samsung, Co-chair)
   - Focus: Spatial procedural content for WebXR
   - Status: Invitation sent, Samsung Display connection
   - Value: Display manufacturer partnership (Samsung OLED), Immersive Web synergy

6. CogAI CG
   - Contact: Dave Raggett (W3C)
   - Focus: Multimodal reasoning, cognitive architecture integration
   - Status: Early discussion, PM-KR Galaxy Universe alignment
   - Value: Cognitive architecture validation, academic credibility

7. AIKR CG
   - Contact: Paola Di Maio
   - Focus: Trust/safety/interoperability, boundary contracts
   - Status: Monitoring, potential future collaboration
   - Value: Governance frameworks, AI safety positioning

STRATEGIC SUMMARY: PM-KR positioned as cross-CG enabler (not isolated proposal).
7 W3C groups engaged = comprehensive standards ecosystem alignment.

SOURCES:
- docs/PRESS_KIT/06_W3C_COLLABORATION/pm_kr_cg_overview.md
- TEMP/EMAIL_ADA_ROSE_CANNON_SAMSUNG_2026-03-04.md
- TEMP/EMAIL_CHRIS_LILLEY_W3C_FONTS_2026-03-04.md

After generating, use the "Generate data table" button to create structured output.
```

**Expected output:**
- CSV file (Excel-compatible)
- JSON file (programmatic access)
- Markdown table (for documentation)

**Usage:**
- W3C member onboarding (show collaboration landscape)
- Display manufacturer pitches (demonstrate standards momentum)
- Grant applications (show ecosystem buy-in)
- Conference presentations (cross-CG synergy visualization)

---

## 🎯 Priority Workflow (Recommended Order)

### Step 1: Audio Overview (HIGHEST PRIORITY)
1. Upload `docs/PRESS_KIT/` to NotebookLM
2. Paste Prompt 1 (Audio Overview) to chat
3. Click "Audio Overview" button
4. Download MP3 (~10 minutes)
5. **Use for:** YouTube video, press kit audio, conference booth

**Why first:** Audio Overview = immediate press kit enhancement, shareable on all platforms

---

### Step 2: Carbon Impact Data Table (HIGH PRIORITY)
1. Paste Prompt 2A (Carbon Timeline) to chat
2. Click "Generate data table" button
3. Download CSV + JSON
4. **Use for:** Press releases, grant applications, conference slides

**Why second:** Carbon data = key differentiator, journalists need structured data

---

### Step 3: Comparison Data Table (MEDIUM PRIORITY)
1. Paste Prompt 2B (PM-KR vs LLMs vs RAG) to chat
2. Click "Generate data table" button
3. Download CSV + JSON
4. **Use for:** Display manufacturer pitches, W3C presentations

**Why third:** Competitive differentiation = technical audiences (display manufacturers, W3C members)

---

### Step 4: W3C Collaboration Matrix (MEDIUM PRIORITY)
1. Paste Prompt 2C (Cross-CG Matrix) to chat
2. Click "Generate data table" button
3. Download CSV + JSON
4. **Use for:** W3C member onboarding, grant applications

**Why fourth:** Collaboration landscape = strategic positioning (shows ecosystem buy-in)

---

## 📁 File Organization (After Generation)

**Create folder:** `docs/PRESS_KIT/08_MEDIA_RESOURCES/generated_assets/`

**Save files with descriptive names:**
- `audio_overview_podcast_2026-03-04.mp3` (Audio Overview)
- `carbon_impact_timeline_2026-03-04.csv` (Data Table 2A)
- `carbon_impact_timeline_2026-03-04.json` (Data Table 2A, JSON format)
- `pmkr_vs_llms_comparison_2026-03-04.csv` (Data Table 2B)
- `pmkr_vs_llms_comparison_2026-03-04.json` (Data Table 2B, JSON format)
- `w3c_collaboration_matrix_2026-03-04.csv` (Data Table 2C)
- `w3c_collaboration_matrix_2026-03-04.json` (Data Table 2C, JSON format)

---

## 🎬 Audio Overview: Next Steps (After Generation)

**Upload to YouTube:**
1. Create video: Combine audio MP3 + static visuals (carbon infographic, architecture diagram)
2. Title: "PM-KR: Procedural Memory for Sustainable AI | 12 Gt CO₂ Savings Explained"
3. Description: Include MERCOSUR-EU text + W3C links + contact info
4. Tags: #ProceduralMemory #SustainableAI #W3C #MERCOSUR #EU #CarbonReduction

**Extract snippets:**
- 30-second carbon impact hook (for Twitter/LinkedIn)
- 60-second MERCOSUR-EU partnership (for press releases)
- 90-second W3C collaboration (for conference teasers)

**Distribute:**
- YouTube (main explainer video)
- Press kit download (audio MP3)
- Conference booth (loop audio)
- Podcast platforms (submit as episode)

---

## 📊 Data Tables: Next Steps (After Generation)

**Import to presentations:**
1. Open CSV in Excel/Google Sheets
2. Create charts (carbon timeline = line chart, comparison = horizontal bar chart)
3. Export as images (PNG) for slides
4. Include in `docs/PRESS_KIT/08_MEDIA_RESOURCES/generated_assets/`

**Use in press releases:**
1. Copy-paste markdown table directly into press release
2. Add footer: "Source: PM-KR Community Group, March 2026"
3. Include methodology disclaimer (scenario projection)

**Academic citations:**
1. JSON format = machine-readable (Zotero, EndNote compatible)
2. Include DOI (if publishing paper): "Data available at github.com/danielcamposramos/Knowledge3D"
3. Cite source documents (CARBON_BLUEPRINT_10_YEAR_PROJECTION.md)

---

## 🎯 Success Metrics

**Audio Overview generated when:**
- [ ] MP3 file downloaded (~10 minutes duration)
- [ ] Podcast-style conversation (2 voices, conversational tone)
- [ ] Key stats emphasized (12 Gt CO₂, 70%+ compression, 7M params)
- [ ] MERCOSUR-EU partnership mentioned

**Data Tables generated when:**
- [ ] CSV files downloadable (Excel-compatible)
- [ ] JSON files downloadable (programmatic access)
- [ ] All rows/columns populated with correct data
- [ ] Source citations included

**Press kit enhanced when:**
- [ ] Audio Overview uploaded to YouTube
- [ ] Carbon data table imported to presentations
- [ ] Comparison table sent to display manufacturers
- [ ] W3C collaboration matrix shared with members

---

## 💡 Pro Tips

**Audio Overview:**
- NotebookLM generates ~10 minute podcast automatically (no manual scripting needed)
- Two AI hosts discuss content conversationally (engaging for journalists)
- Download as MP3, use as-is or add visuals for YouTube

**Data Tables:**
- New NotebookLM feature (as of March 2026, still evolving)
- Generates structured CSV/JSON from unstructured text
- If table generation fails → manually create CSV from prompt data
- Validate data against source documents before using in press materials

**Iteration:**
- If audio doesn't emphasize key stats → regenerate with more specific prompt
- If data table misses columns → add explicit column names to prompt
- If tone isn't right → adjust prompt language ("conversational" vs "authoritative")

---

**Last Updated:** March 4, 2026
**Created by:** Claude (Architecture Partner)
**For:** NotebookLM Audio Overview + Data Table generation
**Human modem:** Daniel Ramos

**Focus:** Audio Overview (podcast) + Data Tables (structured data) = most valuable for press kit! 🎯
