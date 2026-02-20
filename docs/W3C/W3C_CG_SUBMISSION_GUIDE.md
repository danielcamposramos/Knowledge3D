# W3C Community Group Submission Guide
# How to Submit PM-KR CG Proposal

**Date**: February 20, 2026
**Purpose**: Step-by-step instructions for submitting PM-KR Community Group proposal to W3C

---

## Overview

This guide provides **exact instructions** for submitting the PM-KR Community Group proposal through the W3C website. All supporting materials have been prepared — you just need to fill out the W3C form and attach the documents.

**Estimated time**: 30-45 minutes
**Prerequisites**: W3C account (free, create at https://www.w3.org/accounts/request)

---

## Step 1: Create W3C Account (if needed)

**If you don't have a W3C account**:

1. Go to: https://www.w3.org/accounts/request
2. Fill out registration form:
   - **Email**: daniel@echosystems.ai
   - **Name**: Daniel Ramos
   - **Organization** (optional): Knowledge3D Project
   - **Country**: [Your country]
3. Verify email (check inbox for confirmation link)
4. Log in at: https://www.w3.org/users/myprofile

**If you already have an account**: Skip to Step 2

---

## Step 2: Access Community Group Proposal Form

1. **Navigate to**: https://www.w3.org/community/groups/propose_cg/
2. **Log in** (if not already logged in)
3. You should see "**Propose a Community Group**" form

---

## Step 3: Fill Out Proposal Form

### 3.1 Basic Information

**Group Name**:
```
Procedural Memory Knowledge Representation
```

**Short Name** (URL identifier):
```
pm-kr
```

**Proposed URL** (auto-generated, verify):
```
https://www.w3.org/community/pm-kr/
```

**Description** (short summary, 2-3 sentences):
```
The PM-KR Community Group develops and standardizes Procedural Memory Knowledge Representation, a novel knowledge representation paradigm achieving ~70% compression via procedural composition, dual-client consistency (humans and AI share the same knowledge source), sovereign execution (zero external dependencies), and full auditability. PM-KR complements existing W3C standards (RDF, OWL, JSON-LD) while addressing knowledge duplication, compression-meaning tradeoffs, and dependency cascades.
```

### 3.2 Scope and Goals

**Scope** (detailed, 1-2 paragraphs):
```
The PM-KR Community Group will develop normative specifications for Procedural Memory Knowledge Representation, including: (1) a 4-layer compositional data model (Form → Meaning → Rules → Meta-Rules), (2) 6 normative invariants (canonicality, reference preservation, determinism, dual-client equivalence, sovereign boundary, auditability), (3) execution semantics for procedural knowledge, and (4) 3 conformance levels (Core, Sovereign Runtime, Auditable Production).

The group will also produce interoperability guidelines for bidirectional mapping with RDF 1.1, OWL 2, and JSON-LD 1.1; conformance test suites; performance benchmarks; and third-party verification protocols. PM-KR addresses critical gaps in current knowledge representation systems: knowledge duplication (70%+ waste validated), procedural-static divide (humans read procedures, AI consumes static payloads), compression-meaning tradeoffs (current approaches trade fidelity for size), and sovereignty crises (external dependency cascades). The reference implementation (Knowledge3D / K3D) provides empirical validation: 70% compression (Character Galaxy: 87.7MB → 26.3MB), 100% GPU sovereignty (154 GPU calls / 154 solved tasks), and 68/68 passing integration tests.
```

**Goals** (bullet points, 4-6 items):
```
- Finalize PM-KR Normative Model (RFC 2119 compliant specification) with 3 conformance levels
- Publish interoperability guidelines for RDF/OWL/JSON-LD bidirectional mapping and translation loss analysis
- Release open-source conformance test suite (Level A/B/C: 5/8/12 tests) with third-party verification protocol
- Solicit third-party implementations and conduct industry pilots (target: Neo4j, Hugging Face, WebXR platforms)
- Achieve W3C Candidate Recommendation status (Q4 2026) and W3C Recommendation (Q2 2027, if consensus achieved)
- Coordinate tooling development (converters, validators, IDE plugins) and foster community adoption
```

### 3.3 Initial Participants

**Chair** (your name):
```
Daniel Ramos
```

**Chair Email**:
```
daniel@echosystems.ai
```

**Chair Affiliation** (optional):
```
Knowledge3D Project
```

**Initial Participants** (list 3-5 people if available, or just yourself):
```
Daniel Ramos (Knowledge3D Project, Chair and K3D Architect)
[Add other confirmed participants if available, otherwise submit with just yourself — you can add more later]
```

**Expected Participant Count** (1-3 months after launch):
```
10-15 active participants (target industries: graph databases, AI platforms, XR/spatial computing, accessibility, web standards bodies)
```

### 3.4 Relationship to Other Groups

**Related W3C Groups** (check all that apply):
- ☑ RDF Working Group / Semantic Web Interest Group
- ☑ JSON-LD Working Group
- ☑ Web Ontology (OWL) Community
- ☑ Immersive Web Working Group (WebXR)
- ☑ Web Accessibility Initiative (WAI)

**How PM-KR relates** (2-3 sentences):
```
PM-KR complements existing W3C standards by providing compression semantics, procedural execution, and dual-client guarantees that RDF/OWL/JSON-LD lack. The group will coordinate with RDF/JSON-LD groups on interoperability (bidirectional mapping), OWL community on procedural reasoning, Immersive Web WG on spatial knowledge (glTF extensions), and WAI on dual-client accessibility. PM-KR does NOT replace existing standards — it provides a procedural foundation that interoperates with them.
```

**External Standards Bodies** (if applicable):
```
Khronos Group (glTF Working Group) — coordination on `extras.k3d` field for spatial knowledge in 3D assets
```

### 3.5 Intellectual Property

**Patent Policy** (select):
- ☑ **W3C Community Final Specification Agreement (FSA)** [RECOMMENDED — this is standard for CGs]

**Specification License** (select):
- ☑ **Creative Commons Attribution (CC-BY)** [This matches our CC-BY-4.0 choice]

**Code License** (for test suites, tools):
- ☑ **MIT License** [Maximum permissiveness for adoption]

**Patent Disclosure** (if applicable):
```
None. The Knowledge3D Project commits to a no-patent policy: "We patent nothing. We publish everything. We build in the open." All PM-KR work is published under CC-BY-4.0 with no patent encumbrances.
```

### 3.6 Communication

**Mailing List Preference**:
- ☑ **W3C-hosted mailing list** (public-pm-kr@w3.org, proposed)

**GitHub Repository** (optional, can set up later):
```
https://github.com/w3c/pm-kr (proposed — will migrate from https://github.com/danielcamposramos/Knowledge3D/tree/main/docs/W3C)
```

**Meeting Frequency**:
```
Monthly teleconferences (1 hour, rotating time zones) + quarterly face-to-face meetings (co-located with W3C TPAC or industry conferences)
```

### 3.7 Supporting Materials

**Website / Documentation** (URL):
```
https://github.com/danielcamposramos/Knowledge3D/tree/main/docs/W3C
```

**Charter Document** (attach file):
- **File**: Upload `W3C_CG_CHARTER_PMKR.md` (from docs/W3C/)

**Proposal Letter / Justification** (attach file or paste):
- **File**: Upload `W3C_CG_PROPOSAL_LETTER.md` (from docs/W3C/)
- **OR paste** (if form doesn't allow file upload): Copy entire text from W3C_CG_PROPOSAL_LETTER.md

**Evidence / Validation** (optional, highly recommended):
- **URL**: https://github.com/danielcamposramos/Knowledge3D/tree/main/docs/W3C/PM_KR_EVIDENCE_VALIDATION_MATRIX.md
- **Summary**: "K3D reference implementation provides empirical validation: 70% compression (Character Galaxy: 87.7MB → 26.3MB), 100% GPU sovereignty (154/154 tasks), 68/68 integration tests passing, 51,532 nodes in 180MB VRAM."

---

## Step 4: Review and Submit

### 4.1 Pre-Submission Checklist

Before clicking "Submit", verify:

- [ ] Group name: "Procedural Memory Knowledge Representation" (clear, descriptive)
- [ ] Short name: "pm-kr" (lowercase, hyphens, no special characters)
- [ ] Scope clearly states what PM-KR IS and what it's NOT (in/out of scope)
- [ ] Goals are specific and measurable (conformance levels, test suites, timelines)
- [ ] Chair information correct (name, email, affiliation)
- [ ] Patent policy: W3C FSA (standard choice for CGs)
- [ ] License: CC-BY for specs, MIT for code (maximum openness)
- [ ] Charter document attached (W3C_CG_CHARTER_PMKR.md)
- [ ] Proposal letter attached (W3C_CG_PROPOSAL_LETTER.md)
- [ ] Supporting materials URL correct (GitHub docs/W3C/)

### 4.2 Submit

1. **Click "Submit Proposal"** (or equivalent button)
2. **Confirmation page**: Should show "Proposal submitted successfully" or similar
3. **Email confirmation**: Check inbox for W3C confirmation email (may take 5-10 minutes)

---

## Step 5: Post-Submission Actions

### 5.1 W3C Team Review (1-2 weeks)

**What happens**:
- W3C Team reviews proposal for completeness and CG policy compliance
- May request clarifications or minor edits
- Check email daily for W3C Team responses

**If clarifications requested**:
- Respond promptly (within 48 hours if possible)
- Update charter/proposal if needed
- Resubmit or confirm changes via email

### 5.2 Public Comment Period (2-4 weeks)

**Once approved by W3C Team**:
- Proposal posted to public-new-work@w3.org (W3C announcement list)
- 30-day public comment period (community feedback)
- You should monitor that mailing list and respond to questions

**How to respond**:
- Subscribe to public-new-work@w3.org: https://lists.w3.org/Archives/Public/public-new-work/
- Answer questions clearly, point to supporting docs (GitHub)
- If concerns raised, address them transparently (update charter if needed)

### 5.3 Community Group Launch (4-6 weeks total)

**If no blocking objections**:
- W3C Team approves Community Group
- CG page goes live: https://www.w3.org/community/pm-kr/
- Mailing list activated: public-pm-kr@w3.org
- You receive admin access to CG page

**Your first actions as Chair**:
1. **Welcome message**: Post to public-pm-kr@w3.org introducing the group, goals, how to participate
2. **Charter ratification**: Call for community vote to ratify charter (even if just you initially)
3. **GitHub setup**: Create https://github.com/w3c/pm-kr (or request W3C Team to create it)
4. **Call for participation**: Post to relevant communities (RDF/Semantic Web, AI, XR, accessibility)
5. **First meeting**: Schedule initial teleconference (target: 2-4 weeks after launch)

---

## Step 6: Recruiting Participants

### 6.1 Target Communities

**Where to post call for participation**:

1. **W3C mailing lists**:
   - public-rdf-comments@w3.org (RDF community)
   - public-json-ld-wg@w3.org (JSON-LD Working Group)
   - public-immersive-web@w3.org (WebXR community)
   - public-wai-ig@w3.org (Accessibility Interest Group)

2. **Industry platforms**:
   - Neo4j Community (graph databases): https://community.neo4j.com/
   - Hugging Face Discuss (AI platforms): https://discuss.huggingface.co/
   - Three.js Discourse (WebXR/3D): https://discourse.threejs.org/
   - Reddit: r/semanticweb, r/webdev, r/MachineLearning

3. **Academic/research**:
   - ACM SIGWEB, IEEE Semantic Computing mailing lists
   - Knowledge representation conferences (KR, ISWC, ESWC)

4. **GitHub / Open Source**:
   - Open issues on relevant projects (RDF libraries, graph databases) inviting feedback
   - Post to awesome-* lists (awesome-knowledge-graphs, awesome-semantic-web)

### 6.2 Call for Participation Template

**Subject**: [New W3C CG] Procedural Memory Knowledge Representation (PM-KR) — Call for Participation

**Body**:
```
The W3C Procedural Memory Knowledge Representation (PM-KR) Community Group has launched and is seeking participants!

**Mission**: Standardize a novel knowledge representation paradigm achieving:
- ~70% compression via procedural composition (validated in K3D reference implementation)
- Dual-client reality (humans + AI share same procedural source)
- Sovereign execution (zero external dependencies in hot path)
- Full auditability (provenance tracking, deterministic reconstruction)

**Why participate?**
- Shape the future of knowledge representation beyond RDF/OWL/JSON-LD
- Contribute to empirically validated standards (K3D: 70% compression, 100% sovereignty, 68/68 tests)
- Collaborate with industry (graph databases, AI platforms, XR, accessibility)
- Open IP policy (CC-BY-4.0 specs, MIT code, no patents)

**How to join**: https://www.w3.org/community/pm-kr/
**Charter**: https://github.com/danielcamposramos/Knowledge3D/blob/main/docs/W3C/W3C_CG_CHARTER_PMKR.md
**Specifications**: https://github.com/danielcamposramos/Knowledge3D/tree/main/docs/W3C

**First meeting**: [Date/time TBD — poll participants for availability]

Questions? Email: daniel@echosystems.ai or post to public-pm-kr@w3.org

Philosophy: We patent nothing. We publish everything. We build in the open.
```

---

## Step 7: First 90 Days Roadmap

### Weeks 1-2: Foundation
- [ ] Post welcome message to public-pm-kr@w3.org
- [ ] Set up GitHub repository (https://github.com/w3c/pm-kr)
- [ ] Migrate specifications from K3D repo to W3C CG repo
- [ ] Create contributing guide (CONTRIBUTING.md)
- [ ] Post call for participation to target communities

### Weeks 3-4: Community Building
- [ ] Schedule first teleconference (poll participants for time)
- [ ] Create meeting agenda template (goals, discussion topics, decisions)
- [ ] Recruit 2-3 co-editors (reach out to early participants)
- [ ] Set up GitHub issues for specification feedback

### Weeks 5-8: Working Group Formation
- [ ] Hold first meeting (introduce charter, scope, deliverables)
- [ ] Form working groups (Normative Spec, Interoperability, Conformance Testing, Tooling)
- [ ] Assign editors to each working group
- [ ] Create project roadmap (milestones, deadlines)

### Weeks 9-12: Specification Refinement
- [ ] Incorporate community feedback into v2.0 specifications
- [ ] Address open questions (terminology alignment, interoperability edge cases)
- [ ] Publish draft specifications for broader review
- [ ] Solicit industry pilot commitments (Neo4j, Hugging Face, etc.)

---

## Troubleshooting

### Common Issues

**Issue**: "Group name already taken"
- **Solution**: Try variant like "Procedural Knowledge Representation" or "PM-KR Knowledge Representation"
- **Check**: Search existing CGs at https://www.w3.org/community/groups/

**Issue**: "Scope too broad / overlaps with existing groups"
- **Solution**: Emphasize what PM-KR adds (compression, procedural execution, dual-client, sovereignty) that existing groups don't cover
- **Clarify**: PM-KR complements (not replaces) RDF/OWL/JSON-LD

**Issue**: "Not enough initial participants"
- **Solution**: W3C CGs can start with 1 person (you), recruit afterward
- **Emphasize**: Reference implementation (K3D) + empirical validation = serious effort

**Issue**: "Need more evidence / validation"
- **Solution**: Point to Evidence Validation Matrix (10 core claims, all validated)
- **Provide**: Direct links to K3D tests (68/68 passing), benchmark outputs

**Issue**: "Unclear how this relates to existing work"
- **Solution**: Reference Interoperability Guide (RDF/OWL/JSON-LD mapping)
- **Clarify**: Hybrid deployment pattern (PM-KR hot path + RDF metadata)

### Getting Help

**W3C Team Contacts**:
- **Community Group Support**: team-community@w3.org
- **Technical Questions**: systeam@w3.org

**Resources**:
- **CG Guide**: https://www.w3.org/community/about/
- **CG Process**: https://www.w3.org/community/about/process/
- **CG FAQ**: https://www.w3.org/community/about/faq/

---

## Summary Checklist

**Before submitting**:
- [ ] W3C account created and verified
- [ ] Proposal form filled out completely (all sections)
- [ ] Charter document attached (W3C_CG_CHARTER_PMKR.md)
- [ ] Proposal letter attached (W3C_CG_PROPOSAL_LETTER.md)
- [ ] Supporting materials URL provided (GitHub docs/W3C/)
- [ ] Pre-submission checklist reviewed (Step 4.1)

**After submitting**:
- [ ] Confirmation email received from W3C
- [ ] Monitor email for W3C Team review (1-2 weeks)
- [ ] Prepare to respond to public comments (30-day period)
- [ ] Draft welcome message for mailing list (once approved)
- [ ] Prepare call for participation posts (target communities)
- [ ] Plan first meeting (2-4 weeks after launch)

---

## Expected Timeline

| Milestone | Timeframe | Action |
|-----------|-----------|--------|
| **Submit proposal** | Day 0 | Fill out W3C form, attach charter/letter |
| **W3C Team review** | Days 1-14 | Respond to clarification requests |
| **Public comment period** | Days 15-45 | Monitor public-new-work@w3.org, respond to feedback |
| **CG approval** | Days 45-60 | W3C Team approves, CG page goes live |
| **Launch** | Day 60 | Post welcome message, GitHub setup, call for participation |
| **First meeting** | Days 75-90 | Initial teleconference, working group formation |
| **Specification refinement** | Days 90-180 | Community feedback incorporated, v2.0 published |

**Target Launch Date**: April 2026 (Q2 2026, per roadmap)

---

## Contact for Questions

**W3C CG Proposal Support**:
- **Email**: team-community@w3.org
- **Website**: https://www.w3.org/community/about/

**PM-KR Specific**:
- **Chair**: Daniel Ramos (daniel@echosystems.ai)
- **Repository**: https://github.com/danielcamposramos/Knowledge3D
- **Mailing List** (once live): public-pm-kr@w3.org

---

**Good luck!** You have all the materials prepared. Just fill out the form, attach the documents, and submit. The W3C community will be excited to see PM-KR's empirical validation and novel approach to knowledge representation.

**Philosophy**: We patent nothing. We publish everything. We build in the open. 🚀

---

**End of Submission Guide**
