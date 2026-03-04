# PM-KR / K3D Press Kit

## Purpose
This folder is a media-ready package for journalists, standards participants, display manufacturers, conference organizers, and grant reviewers. It is built from repository-grounded artifacts and organized for direct NotebookLM upload.

## Audience Modes
1. Journalists
- Start with `01_EXECUTIVE_SUMMARY/one_pager.md`
- Then use `10_FAQ/media_faq.md` and `08_MEDIA_RESOURCES/videos.md`

2. Display manufacturers
- Start with `04_USE_CASES/procedural_displays.md`
- Then use `03_TECHNICAL_OVERVIEW/architecture_diagram.md`

3. W3C members
- Start with `06_W3C_COLLABORATION/pm_kr_cg_overview.md`
- Then use `05_TECH_GIANTS_TRANSFORMATION/GLOBAL_ECOSYSTEM_SUMMARY.md`
- Then use `03_TECHNICAL_OVERVIEW/how_it_works.md`

4. Tech ecosystem leaders (Google, Amazon, Huawei, TSMC, SAP, etc.)
- Start with `05_TECH_GIANTS_TRANSFORMATION/GLOBAL_ECOSYSTEM_SUMMARY.md`
- Then use individual ecosystem document relevant to your organization

5. Conference decks
- Start with `01_EXECUTIVE_SUMMARY/elevator_pitch.md`
- Then use carbon and architecture assets from sections 03, 05, and 05_TECH_GIANTS_TRANSFORMATION

## Package Map
- `01_EXECUTIVE_SUMMARY/`: one-page overview, release template, pitch set
- `02_KEY_PERSONNEL/`: bios and team-photo direction
- `03_TECHNICAL_OVERVIEW/`: non-technical + technical explainers, comparison chart, architecture diagram
- `04_USE_CASES/`: displays, sustainability, web standards, enterprise knowledge
- `05_CARBON_IMPACT/`: summary, method, comparisons, infographic JSON
- `05_TECH_GIANTS_TRANSFORMATION/`: **15 global ecosystem analyses** (Google, Amazon, Huawei/Alibaba, Apple, Taiwan/TSMC, Russia, Japan, EU, South Korea, India, Latin America, Africa, Australia, Southeast Asia, Caribbean) — **30-52 Gt CO₂ impact projection**
- `06_W3C_COLLABORATION/`: PM-KR CG context, cross-CG map, roadmap, joining guide
- `07_MERCOSUR_EU_PARTNERSHIP/`: collaboration narrative and canonical Milton text
- `08_MEDIA_RESOURCES/`: videos, screenshot/logo placeholders, social copy
- `09_QUOTES/`: quote banks and testimonial status
- `10_FAQ/`: public, technical, and media FAQs
- `11_CONTACT_INFO/`: press contacts and interview preferences

## NotebookLM / Nano Banana Workflow

**🎯 RECOMMENDED (Most Useful):** [NOTEBOOKLM_AUDIO_AND_DATA_PROMPTS.md](NOTEBOOKLM_AUDIO_AND_DATA_PROMPTS.md)
- **Audio Overview** (podcast-style, ~10 min) → YouTube explainer, press kit audio
- **Data Tables** (structured CSV/JSON) → Press releases, grant applications, presentations
  - Carbon Impact Timeline (2026-2035)
  - PM-KR vs LLMs vs RAG Comparison
  - W3C Cross-CG Collaboration Matrix

**Alternative (Visual Assets):** [NOTEBOOKLM_PROMPTS_FOR_NANO_BANANA.md](NOTEBOOKLM_PROMPTS_FOR_NANO_BANANA.md)
- 7 prompts for infographics, architecture diagrams, slides, team photos, social graphics
- Useful for conferences, display manufacturer pitches, social media

Quick workflow (Audio + Data - RECOMMENDED):
1. Upload entire `docs/PRESS_KIT/` folder to NotebookLM
2. Click **"Audio Overview"** button (generates podcast automatically)
3. Click **"Generate data table"** button (new feature, creates CSV/JSON)
4. Download outputs: Audio MP3 + Data CSV/JSON
5. Save to `08_MEDIA_RESOURCES/generated_assets/`
6. Use for press outreach, YouTube, conference submissions

## Grounding Sources Used
- `TEMP/PROJECT_OVERVIEW_KEY_PERSONNEL_2026-03-04.md`
- `docs/CARBON_BLUEPRINT_10_YEAR_PROJECTION.md`
- `docs/PRESS_KIT/05_TECH_GIANTS_TRANSFORMATION/GLOBAL_ECOSYSTEM_SUMMARY.md` **(NEW: March 5, 2026)**
- `docs/W3C/PM_KR_NORMATIVE_MODEL.md`
- `docs/W3C/PM_KR_EVIDENCE_VALIDATION_MATRIX.md`
- `docs/W3C_PM_KR_COMMUNITY_GROUP_MISSION.md`
- `docs/W3C_PM_KR_OBJECTIVES_v1.2.md`
- `docs/vocabulary/PROCEDURAL_MEMORY_KR_STANDARD_SPECIFICATION.md`
- `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md`
- `docs/vocabulary/SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md` **(NEW: March 3, 2026)**
- `docs/vocabulary/SUPERHUMAN_GENERAL_INTELLIGENCE_SPECIFICATION.md` **(NEW: March 5, 2026)**
- `docs/Sovereign_Systems_Charter/FINAL_REPORT_FOR_CHRISTOPH.md`
- `README.md`

## Canonical MERCOSUR-EU Narrative Placement
Milton's verbatim canonical text is included in:
- `01_EXECUTIVE_SUMMARY/one_pager.md`
- `07_MERCOSUR_EU_PARTNERSHIP/historic_significance.md`
- `07_MERCOSUR_EU_PARTNERSHIP/milton_text.md`

## Quality and Usage Notes
- Carbon numbers are presented as scenario projections with explicit methodology caveat.
- Quotes are formatted for press use and should be speaker-confirmed before external publication.
- Placeholder media paths should be replaced with final assets before sending to external press lists.
- Keep all external claims tied to source paths listed above.
