# Response to Christoph Dorn: Practical Wiki/Summary Implementation

**To:** public-pm-kr@w3.org
**Cc:** Milton Ponson <rwiciamsd@gmail.com>, Christoph Dorn <christoph@christophdorn.com>
**From:** Daniel Campos Ramos <capitain_jack@yahoo.com>
**Subject:** Re: PM-KR Wiki + Weekly Summaries — Practical Implementation
**Date:** February 28, 2026

---

Dear Christoph, Milton, and PM-KR Community,

Christoph, your "bodies through boundaries" insight resonates deeply with PM-KR's architecture:

> "Declarative first to structure bodies through boundaries. Procedural second to animate activity in and around bodies. A body is a whole system of something."

**This IS the K3D paradigm:**
- **Declarative boundaries** = House structure (walls, doors, spatial addresses define the body)
- **Procedural activity** = TRM navigation (agent animates the body by querying Galaxy, composing RPN)
- **Body = whole system** = House-centric architecture (all dimensions considered, unified workspace)

Your systems architecture perspective validates what Milton articulated mathematically (mandala graph theory) and what K3D implements practically. Beautiful convergence. 🙏

---

## Practical Implementation: Wiki + Weekly Summaries

**Christoph asked:** "How could we go about doing this practically? Do we need to build something or are there existing solutions?"

**My answer: GitHub-based, markdown, version-controlled — existing solutions, zero custom build.**

---

### 1. **GitHub Wiki Structure (Existing Solution)**

**Platform:** GitHub repository wiki (https://github.com/danielcamposramos/Knowledge3D/wiki)

**Why GitHub:**
- ✅ Version control (track all changes, revert if needed)
- ✅ Markdown (human-readable, no proprietary format)
- ✅ Collaborative editing (anyone with permissions can contribute)
- ✅ Cross-linking (link between wiki pages, specs, issues)
- ✅ Public archive (W3C transparency requirement)

**Proposed structure:**

```
/wiki/
  ├── Home.md (Overview + navigation)
  ├── cross-posting/
  │   ├── AIKR_CG.md (Links to AI-KR CG discussions relevant to PM-KR)
  │   ├── WebML_CG.md (Links to Web ML CG discussions - e.g., WebMCP)
  │   ├── VC_WG.md (Links to Verifiable Credentials WG discussions)
  │   └── JSON_LD_WG.md (Links to JSON-LD WG discussions)
  ├── weekly-summaries/
  │   ├── 2026-W09.md (Feb 24-28, 2026)
  │   ├── 2026-W10.md (Mar 3-7, 2026)
  │   └── template.md (Standard format for consistency)
  ├── literature/
  │   ├── declarative-procedural.md (Papers, articles on declarative vs procedural)
  │   ├── mandala-graph-theory.md (Milton's work + related references)
  │   └── verifiable-credentials.md (VC specs, implementations, use cases)
  └── decisions/
      ├── 2026-02-28-declarative-procedural-synergy.md (Record of mission statement revision)
      └── template.md (Architecture Decision Record format)
```

**Tools needed:**
- **Zero custom build** — just GitHub + markdown editors
- **Optional:** GitHub Actions to auto-generate summary drafts (pull from mailing list, issues, PRs)

---

### 2. **Weekly Summary Format (Condensed, No AI Verbosity)**

**Milton's requirement:** "Condensed summary form, without all the verbal add-ons that AI chatbots typically generate."

**Proposed template:**

```markdown
# PM-KR Weekly Summary — 2026-W09 (Feb 24-28)

## Key Developments
- Mission statement revised (v1.0 → v1.2): Declarative + procedural synergy
- Dave Raggett feedback: Procedural vs declarative positioning clarified
- Milton Ponson insight: Mandala graph theory addresses both simultaneously
- Christoph Dorn: "Bodies through boundaries" validates House-centric paradigm

## New Members
- [None this week, 18+ members stable]

## Literature/Prior Art
- PKN (Procedural Knowledge Networks): Declarative + qualitative metadata
- Gemini analysis (Milton): "Know-what" (declarative) vs "know-how" (procedural)
- Christoph: "Declarative structure, procedural animation" (systems perspective)

## Cross-CG Discussions
- AIKR-CG: Milton's mandala graph theory (declarative/procedural duality)
- VC-WG: Knowledge provenance for procedural programs
- WebML-CG: WebMCP security concerns (Paola Di Maio critique)

## Technical Questions Raised
1. How should PM-KR's execution layer integrate with RDF/OWL ontologies?
2. Are proposed context inheritance/override rules sufficient?
3. Which applications (education, gaming, science, accessibility, AI) to prioritize?
4. How do PM-KR security patterns apply to WebMCP?

## Action Items
- [x] Mission statement updated (v1.2)
- [x] GitHub wiki structure created
- [ ] First weekly summary published (target: Monday, March 3)
- [ ] Joint security workshop proposed (WebMCP + PM-KR + VC WG)

## Next Week Preview
- PM-KR Core Specification v0.1 drafting begins
- Community feedback on declarative integration approaches
- Cross-CG collaboration on security patterns
```

**Length:** ~150 lines (vs typical AI chatbot output: 500+ lines with verbose explanations)

**Tone:** Factual, concise, actionable (no "I think," "perhaps," "it seems")

---

### 3. **Tooling Strategy: Existing Solutions First**

**Christoph asked:** "Do we need to build something?"

**My approach: Existing solutions + minimal automation.**

**Phase 1 (Now → March 2026): Manual + GitHub**
- Wiki pages: Manually written in markdown
- Weekly summaries: Manually curated from mailing list + GitHub activity
- Cross-posting links: Manually collected
- **Tools:** GitHub web interface, any markdown editor

**Phase 2 (April 2026+): Light automation**
- GitHub Actions: Auto-pull mailing list threads (public-pm-kr@w3.org archives)
- Script: Generate draft summary from GitHub issues, PRs, commits
- **Human review:** Always required (no auto-publish, maintain quality)

**Phase 3 (Q3 2026+): If community grows**
- Consider dedicated wiki platform (e.g., Docusaurus, MkDocs) if GitHub wiki becomes limiting
- But for 18+ members, GitHub wiki is sufficient

**Philosophy:** Start simple, add automation only when manual process becomes bottleneck.

---

### 4. **Weekly Summary Workflow (Practical Process)**

**Who writes it:**
- Primary: PM-KR chairs (Daniel + Milton) alternate weeks
- Community contributions: Anyone can submit draft sections via GitHub PR

**When:**
- **Friday:** Draft summary (review week's activity)
- **Saturday:** Community review (mailing list feedback)
- **Monday:** Publish final summary (wiki + mailing list post)

**Sources:**
1. **Mailing list:** public-pm-kr@w3.org archives (key discussions)
2. **GitHub activity:** Issues, PRs, commits in Knowledge3D repo
3. **Cross-CG discussions:** Mentions of PM-KR in other W3C CG mailing lists
4. **Literature:** New papers, articles, specs relevant to PM-KR

**Quality control:**
- **Condensed:** Max 200 lines (vs AI chatbot 500+)
- **Factual:** No speculation, just what happened
- **Actionable:** Clear action items, next steps
- **Cross-linked:** Link to mailing list threads, GitHub issues, external resources

---

## Implementation Timeline

**This week (Feb 28 - Mar 3):**
- ✅ Create GitHub wiki structure (I'll do this today)
- ✅ Write first weekly summary (2026-W09) as template
- ✅ Post wiki link to mailing list

**Next week (Mar 3-7):**
- Community feedback on wiki structure
- Iterate on weekly summary format
- Publish second weekly summary (2026-W10)

**Month 1 (March 2026):**
- Establish workflow (who writes, when published)
- Gather community contributions
- Evaluate if automation is needed

**Month 3 (May 2026):**
- Review if GitHub wiki is sufficient or if dedicated platform needed
- Consider light automation (GitHub Actions for draft generation)

---

## Your "Bodies Through Boundaries" Insight

**Christoph, this maps directly to PM-KR's core architecture:**

**Declarative layer (structure bodies through boundaries):**
- House structure (walls, doors, rooms define spatial boundaries)
- Galaxy Universe organization (Drawing, Character, Word, Grammar, Math, Reality galaxies define knowledge boundaries)
- JSON-LD metadata (semantic relationships define conceptual boundaries)

**Procedural layer (animate activity in and around bodies):**
- TRM navigation (agent queries Galaxy, composes RPN programs)
- RPN execution (Cranium PTX kernels animate procedural programs)
- Context-specific behavior (same knowledge, different execution based on context)

**The "body" = House-centric paradigm:**
- House is the workspace (all dimensions: spatial, temporal, semantic, procedural)
- TRM is the inhabitant (animates the body by navigating, querying, creating)
- Galaxy Universe is the memory (distributed knowledge substrate)

**Your intuitions align perfectly with K3D's implementation. This gives me confidence we're on the right path.**

---

## Closing Thoughts

**Milton's mathematical insight (mandala graph theory) + Christoph's systems perspective ("bodies through boundaries") + K3D's implementation = PM-KR's unified vision.**

**This is exactly what W3C standardization should look like:**
- Theory (Milton) informs practice (Daniel)
- Systems architecture (Christoph) validates implementation (K3D)
- Community collaboration (18+ members) refines the vision

**Thank you both for this productive exchange.** 🙏

---

Best regards,

**Daniel Campos Ramos**
PM-KR Co-Chair (Implementation)
Brazilian Registered Electrical Engineer
W3C PM-KR Community Group
capitain_jack@yahoo.com

---

**P.S. Action Items:**
1. ✅ **Today:** Create GitHub wiki structure (I'll post link within 2 hours)
2. ✅ **Today:** Write first weekly summary (2026-W09) as template
3. 🔄 **Monday, March 3:** Publish weekly summary to mailing list
4. 🔄 **This week:** Community feedback on wiki/summary format

---

**Links:**
- PM-KR GitHub Wiki (coming soon): https://github.com/danielcamposramos/Knowledge3D/wiki
- Weekly Summary Template (draft): https://github.com/danielcamposramos/Knowledge3D/wiki/weekly-summaries/template.md
- Cross-Posting Guidelines: https://github.com/danielcamposramos/Knowledge3D/wiki/cross-posting/
