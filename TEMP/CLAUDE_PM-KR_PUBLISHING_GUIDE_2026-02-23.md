# PM-KR Community Group - Publishing Setup Guide

**Date**: February 23, 2026
**Purpose**: Step-by-step guide for setting up and populating PM-KR Community Group pages

---

## Overview: Two Publishing Tracks

### Track 1: Immediate Publishing (Knowledge3D GitHub → GitHub Pages)
- **Speed**: Immediate (already done!)
- **Audience**: Technical community, early adopters
- **URL Pattern**: `https://danielcamposramos.github.io/Knowledge3D/docs/...`
- **Status**: ✅ Already live and working

### Track 2: Official W3C Publishing (PM-KR Community Group)
- **Speed**: Requires setup (steps below)
- **Audience**: W3C standards community, formal collaboration
- **URL Pattern**: `https://w3c-cg.github.io/pm-kr/` (draft) or `https://www.w3.org/community/reports/pm-kr/...` (final)
- **Status**: ⏳ Needs setup

---

## PART 1: Immediate Publishing (Already Working!)

### Current Status ✅

Your Knowledge3D repository is **already publicly accessible** on GitHub:
- **Repository**: https://github.com/danielcamposramos/Knowledge3D
- **All Phase 1 documents are published**: INTERCONNECTEDNESS_MAP_v3.md, KNOWLEDGEVERSE_SPECIFICATION.md, ROBOTIC_EMBODIMENT_SPECIFICATION.md

### Optional: Enable GitHub Pages for HTML Rendering

**If you want fancy HTML versions** (instead of just markdown on GitHub):

1. **Go to repository settings**:
   - Navigate to https://github.com/danielcamposramos/Knowledge3D/settings/pages

2. **Enable GitHub Pages**:
   - Source: Deploy from branch
   - Branch: `main`
   - Folder: `/docs`
   - Click "Save"

3. **Wait 2-5 minutes**, then your docs will be available at:
   - https://danielcamposramos.github.io/Knowledge3D/docs/INTERCONNECTEDNESS_MAP_v3.html
   - https://danielcamposramos.github.io/Knowledge3D/docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.html
   - Etc.

**Note**: This is OPTIONAL. The current GitHub repository already works perfectly for sharing documentation.

---

## PART 2: Official W3C PM-KR Setup (For Standards Work)

### Step 1: Request Official W3C GitHub Repository

**Who**: You (as proposed Chair)
**Contact**: W3C Community Group Team
**Email**: team-community-process@w3.org

**Email Template**:
```
Subject: PM-KR Community Group - Request GitHub Repository Creation

Dear W3C Community Group Team,

The Procedural Memory Knowledge Representation (PM-KR) Community Group
has been published and is now open for participation:
https://www.w3.org/community/pm-kr/

As the proposed Chair, I would like to request the creation of an official
GitHub repository for the group under the w3c-cg organization.

**Group Details**:
- Group shortname: pm-kr
- Proposed repository name: w3c-cg/pm-kr
- Chair: Daniel Ramos (daniel@echosystems.ai)

**Initial Content**:
We have substantial prior work that will inform the group's discussions
(Knowledge3D project), which we plan to reference as motivation. The
repository will host:
- Draft specifications (data models, execution semantics)
- Use case documentation
- Interoperability studies
- Meeting notes and agendas

Please let me know the next steps for repository creation.

Thank you,
Daniel Ramos
Proposed Chair, PM-KR Community Group
daniel@echosystems.ai
```

**Expected Response**: W3C team will create `w3c-cg/pm-kr` repository with standard LICENSE, CONTRIBUTING.md, w3c.json, and index.html template.

### Step 2: Set Up PM-KR Community Group Pages

**Option A: Use Group WordPress Instance** (Easy, No Coding)

1. **Log in to PM-KR group page**: https://www.w3.org/community/pm-kr/
2. **Click "Pages"** (WordPress feature for CG chairs)
3. **Create pages** for each Phase 1 document:
   - Page title: "Interconnectedness Map v3.0"
   - Content: Copy from INTERCONNECTEDNESS_MAP_v3.md (convert markdown to WordPress)
   - Publish
4. **Repeat** for KNOWLEDGEVERSE_SPECIFICATION.md, ROBOTIC_EMBODIMENT_SPECIFICATION.md

**Pros**: Simple, no GitHub needed, immediate
**Cons**: Not version-controlled, harder to collaborate

**Option B: Use GitHub Repository** (Recommended for Specs)

Once you have the `w3c-cg/pm-kr` repository:

1. **Clone the new repository**:
   ```bash
   git clone https://github.com/w3c-cg/pm-kr.git
   cd pm-kr
   ```

2. **Copy Phase 1 documents** from Knowledge3D:
   ```bash
   # Copy from your Knowledge3D repo
   cp /path/to/Knowledge3D/docs/INTERCONNECTEDNESS_MAP_v3.md ./
   cp /path/to/Knowledge3D/docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md ./
   cp /path/to/Knowledge3D/docs/vocabulary/ROBOTIC_EMBODIMENT_SPECIFICATION.md ./
   ```

3. **Create index page** (edit `index.html` that W3C created):
   ```html
   <!DOCTYPE html>
   <html>
   <head>
     <meta charset='utf-8'>
     <title>Procedural Memory Knowledge Representation (PM-KR)</title>
     <script src='https://www.w3.org/Tools/respec/respec-w3c' class='remove'></script>
     <script class='remove'>
       var respecConfig = {
         specStatus: "CG-DRAFT",
         group: "pm-kr",
         editors: [{
           name: "Daniel Ramos",
           email: "daniel@echosystems.ai",
         }],
         github: "w3c-cg/pm-kr",
         shortName: "pm-kr"
       };
     </script>
   </head>
   <body>
     <section id='abstract'>
       <p>
         The Procedural Memory Knowledge Representation (PM-KR) Community Group
         develops standards for storing knowledge as executable procedures with
         symlink-style composition, enabling both humans and AI systems to consume
         the same procedural source.
       </p>
     </section>

     <section id='sotd'>
       <p>
         This is an unofficial draft. Comments are welcome.
       </p>
     </section>

     <section>
       <h2>Key Documents</h2>
       <ul>
         <li><a href="INTERCONNECTEDNESS_MAP_v3.html">Interconnectedness Map v3.0</a> - 103+ cross-disciplinary connections</li>
         <li><a href="KNOWLEDGEVERSE_SPECIFICATION.html">Knowledgeverse Specification</a> - 7-region unified memory architecture</li>
         <li><a href="ROBOTIC_EMBODIMENT_SPECIFICATION.html">Robotic Embodiment Specification</a> - Hippocampus-inspired spatial memory</li>
       </ul>
     </section>

     <section>
       <h2>Motivation</h2>
       <p>
         This work is motivated by prior work on
         <a href="https://github.com/danielcamposramos/Knowledge3D">Knowledge3D</a>.
         That work does not constrain the group's discussions, nor will it be a
         deliverable of this group. We welcome alternative implementations and approaches.
       </p>
     </section>
   </body>
   </html>
   ```

4. **Enable GitHub Pages**:
   - Go to repository settings → Pages
   - Source: Deploy from branch `main` → `/` (root)
   - Save

5. **Commit and push**:
   ```bash
   git add .
   git commit -m "docs: Add initial Phase 1 specifications"
   git push origin main
   ```

6. **Access published specs** at:
   - https://w3c-cg.github.io/pm-kr/ (homepage)
   - https://w3c-cg.github.io/pm-kr/INTERCONNECTEDNESS_MAP_v3.html
   - Etc.

### Step 3: Publish Final Reports (Later, When Ready)

**When**: After group consensus on a specification
**Process**: Use `w3c/cg-reports` repository

1. **Fork** https://github.com/w3c/cg-reports

2. **Create directory** following convention:
   ```
   pm-kr/CG-FINAL-interconnectedness-map-20260223/
   ```

3. **Add your final report** (static HTML, not ReSpec dynamic):
   ```bash
   # Generate static HTML from ReSpec (if using ReSpec)
   # Or copy your markdown converted to HTML
   cp INTERCONNECTEDNESS_MAP_v3.html pm-kr/CG-FINAL-interconnectedness-map-20260223/index.html
   ```

4. **Open pull request** to `w3c/cg-reports`

5. **After merge**, document will be at:
   ```
   https://www.w3.org/community/reports/pm-kr/CG-FINAL-interconnectedness-map-20260223/
   ```

**Note**: Final reports are permanent and cannot be changed once published.

---

## PART 3: Recommended Publishing Strategy

### Phase 1: Foundation (Publish NOW)

**Where**: Knowledge3D GitHub (already done ✅) + PM-KR WordPress pages (if available)

1. ✅ **INTERCONNECTEDNESS_MAP_v3.md**
   - Already public: https://github.com/danielcamposramos/Knowledge3D/blob/main/docs/INTERCONNECTEDNESS_MAP_v3.md
   - Action: Link to this from PM-KR group homepage when available

2. ✅ **KNOWLEDGEVERSE_SPECIFICATION.md**
   - Already public: https://github.com/danielcamposramos/Knowledge3D/blob/main/docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md
   - Action: Link to this from PM-KR group homepage

3. ✅ **ROBOTIC_EMBODIMENT_SPECIFICATION.md**
   - Already public: https://github.com/danielcamposramos/Knowledge3D/blob/main/docs/vocabulary/ROBOTIC_EMBODIMENT_SPECIFICATION.md
   - Action: Link to this from PM-KR group homepage

**Why First**: These are immediately accessible and establish credibility without requiring W3C infrastructure setup.

### Phase 2: Technical Details (After PM-KR GitHub Setup)

**Where**: `w3c-cg/pm-kr` GitHub repository (once created)

4. **THREE_BRAIN_SYSTEM_SPECIFICATION.md**
5. **DUAL_CLIENT_CONTRACT_SPECIFICATION.md**
6. **MATH_CORE_SPECIFICATION.md**

**Action**:
- Copy from Knowledge3D to `w3c-cg/pm-kr` repository
- Maintain both copies (Knowledge3D as reference implementation, PM-KR as standards track)
- Cross-link between them

### Phase 3: Standardization Proposals (Collaborative Development)

**Where**: `w3c-cg/pm-kr` GitHub repository (collaborative editing)

7. **NEW: Spatial Memory Standards** (draft collaboratively with group)
8. **NEW: Multimodal Explanation Protocol** (draft collaboratively)
9. **NEW: Procedural Canonicalization Examples** (collect from multiple implementations)
10. **NEW: Human-in-Loop Oversight Framework** (draft collaboratively)

**Why Third**: These are work items for PM-KR to develop together, not solo deliverables.

---

## PART 4: Quick Start Action Plan

### Immediate Actions (TODAY)

1. ✅ **Phase 1 documents are already published** on Knowledge3D GitHub
2. ✅ **Email is ready** (TEMP/CLAUDE_PM-KR_COMPREHENSIVE_UPDATE_EMAIL_2026-02-23.md)
3. ⏳ **Send the email** to all 16 W3C groups + collaborators
4. ⏳ **Request W3C GitHub repository** (email team-community-process@w3.org)

### Short-Term Actions (NEXT WEEK)

5. ⏳ **Set up PM-KR group homepage** (WordPress or wait for GitHub repo)
6. ⏳ **Link Phase 1 documents** from PM-KR homepage to Knowledge3D repo
7. ⏳ **Monitor responses** from W3C groups
8. ⏳ **Engage with founding members** on first collaborative work items

### Medium-Term Actions (NEXT MONTH)

9. ⏳ **Copy Phase 2 documents** to `w3c-cg/pm-kr` when repository is ready
10. ⏳ **Host first PM-KR meeting** (virtual)
11. ⏳ **Draft Phase 3 proposals** collaboratively with group
12. ⏳ **Set up mailing list** (public-pm-kr@w3.org, handled by W3C)

---

## PART 5: Technical Notes

### Markdown → HTML Conversion

If you need to convert markdown to HTML for W3C pages:

**Option A: Use Pandoc** (local tool)
```bash
pandoc INTERCONNECTEDNESS_MAP_v3.md -f markdown -t html -s -o INTERCONNECTEDNESS_MAP_v3.html
```

**Option B: Use GitHub's Rendered Markdown** (just link to it)
```
https://github.com/danielcamposramos/Knowledge3D/blob/main/docs/INTERCONNECTEDNESS_MAP_v3.md
```
GitHub automatically renders markdown with nice formatting.

**Option C: Use ReSpec** (W3C standard tool)
- Learn ReSpec: https://respec.org/docs/
- Benefits: Automatic W3C styling, section numbering, references
- Drawback: Requires learning ReSpec syntax

### Version Control Strategy

**Recommendation**: Maintain TWO repositories

1. **Knowledge3D** (implementation + reference):
   - Your main repository
   - Implementation code (PTX kernels, Galaxy, TRM)
   - Reference specifications (living documents)
   - Fast iteration, no consensus needed

2. **w3c-cg/pm-kr** (standards track):
   - Official PM-KR specifications
   - Slower iteration, requires group consensus
   - Forks/references Knowledge3D for motivation
   - Final reports published via `w3c/cg-reports`

**Synchronization**: Periodically copy stable specs from Knowledge3D → PM-KR after group review.

---

## PART 6: Questions & Answers

### Q: Do I need to wait for the W3C repository to publish documents?
**A**: No! Your Knowledge3D repository is already public. Just send the email with links to your existing GitHub docs. Set up the W3C repository in parallel.

### Q: Can I publish specs before they're finished?
**A**: Yes! Use "CG-DRAFT" status in ReSpec. W3C Community Groups are for incubation—drafts are expected.

### Q: What if people suggest changes to the specs?
**A**: Accept pull requests in Knowledge3D, discuss in PM-KR meetings, reach consensus, then update PM-KR repository. Final reports are only published when group agrees.

### Q: Do I need to convert markdown to HTML?
**A**: Not immediately. GitHub renders markdown beautifully. HTML is only needed for official W3C final reports (later).

### Q: How do I handle the "motivated by Knowledge3D" disclaimer?
**A**: Include it prominently in every PM-KR document (see index.html template above). This ensures transparency about prior work while welcoming alternative approaches.

---

## Resources

**W3C Community Group Resources**:
- [W3C Community Groups FAQ](https://www.w3.org/community/about/faq/)
- [W3C on GitHub Guide](https://w3c.github.io/guide/)
- [Report Requirements](https://www.w3.org/community/reports/reqs/)
- [How to Set Up a W3C Community Group](https://standards.theodi.org/useful-tools/how-to-set-up-a-w3c-community-group/)
- [W3C CG Reports Repository](https://github.com/w3c/cg-reports)
- [Chair's Guidebook](https://w3c.github.io/guide/)

**Technical Tools**:
- [ReSpec Documentation](https://respec.org/docs/)
- [GitHub Pages Setup](https://docs.github.com/en/pages)
- [Pandoc (Markdown → HTML)](https://pandoc.org/)

**Contact**:
- W3C CG Team: team-community-process@w3.org
- W3C Head of Communications (Ian Jacobs): Responded to initial PM-KR proposal

---

## Summary: You're Ready!

**The good news**: Your Phase 1 documents are ALREADY published and accessible. You can send the comprehensive email TODAY with links to your existing Knowledge3D repository.

**Next steps**:
1. Send the email (already drafted)
2. Request W3C GitHub repository (template above)
3. Set up PM-KR homepage (WordPress or GitHub, once available)
4. Link to your existing docs while W3C infrastructure spins up

**Philosophy**: Publish early, publish often, build in the open. You're already doing this! 🚀

---

**Document prepared by**: Claude Sonnet 4.5 (Architecture Partner)
**Date**: February 23, 2026
**Status**: Ready for immediate action
