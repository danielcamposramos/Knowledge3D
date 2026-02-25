# PM-KR Community Group Announcement Email
# Cross-Group Invitation to Participate

**Date**: February 20, 2026
**Purpose**: Announce PM-KR CG proposal to related W3C groups

---

## Email Recipients

**To**: public-new-work@w3.org (W3C new group announcements)

**CC** (Related W3C Groups):
- public-json-ld-wg@w3.org (JSON-LD Working Group)
- public-immersive-web@w3.org (Immersive Web WG)
- w3c-wai-ig@w3.org (WAI Interest Group - Accessibility)
- semantic-web@w3.org (RDF/Semantic Web discussions)
- public-aikr@lists.w3.org (AI KR Community Group)
- public-webmachinelearning@lists.w3.org (Web ML CG)
- public-cogai@w3.org (Cognitive AI CG)
- public-kg-construct@w3.org (Knowledge Graph Construction CG)

---

## Email Content

**Subject**: [Announcement] Procedural Memory Knowledge Representation (PM-KR) Community Group — Call for Participation

```
Dear W3C Community,

I'm excited to announce that the **Procedural Memory Knowledge Representation (PM-KR) Community Group** proposal has been published and is now open for public review and participation:

📢 **Official Announcement**: https://www.w3.org/community/blog/2026/02/20/proposed-group-procedural-memory-knowledge-representation-community-group/

🔗 **Join the Group**: https://www.w3.org/community/pm-kr/

---

## What is PM-KR?

Current knowledge representation systems suffer from massive duplication and fragmentation: the same knowledge (e.g., a Unicode character, mathematical symbol, or spatial concept) is duplicated across fonts, embeddings, accessibility metadata, and visual renderings. Redundant representations can create maintenance, performance, security, and licensing issues.

The PM-KR Community Group will develop a knowledge representation paradigm where knowledge is stored once as executable procedures (like font programs or mathematical formula definitions) and referenced via symlink-style composition, enabling both humans and AI systems to consume the same procedural source.

The group will study data models, execution semantics, conformance levels, and relationships with other W3C technologies (e.g., RDF, OWL, JSON-LD).

---

## Why This Matters to Your Group

### 🔗 For JSON-LD & RDF/Semantic Web Communities
**Interoperability and Compression**: PM-KR complements RDF/JSON-LD by providing procedural execution semantics and compression through symlink-style composition. We'll study bidirectional mapping strategies, enabling hybrid deployments where PM-KR handles hot-path execution while RDF/JSON-LD provides discovery and metadata layers. Translation loss analysis and compatibility guidelines will be key deliverables.

### 🎨 For Immersive Web (WebXR) Community
**Spatial Knowledge Navigation**: PM-KR extends glTF's `extras.k3d` field to encode spatial knowledge in 3D assets, enabling dual-client reality where humans and AI agents navigate the same 3D knowledge workspace. This supports semantic proximity = spatial proximity patterns in XR environments.

### ♿ For Web Accessibility (WAI) Community
**Dual-Client Accessibility**: PM-KR's procedural sources (like font programs and formula definitions) can be rendered across multiple modalities—visual glyphs for sighted users, semantic descriptions for screen readers, tactile patterns for braille displays—all from the same canonical source. This eliminates dual-maintenance overhead and ensures consistency across accessibility tools.

### 🤖 For AI & Machine Learning Communities
**AI Knowledge Representation Integration**: The PM-KR paradigm directly addresses AI knowledge representation challenges. Notably, **AI KR (Artificial Intelligence Knowledge Representation) work naturally lives inside the PM-KR umbrella**—where knowledge is stored as executable procedures that both humans and AI systems can consume. This enables:
- **Compression without information loss**: Symlink-style composition reduces duplication while preserving semantic fidelity
- **Sovereign execution**: Zero external dependencies in runtime hot paths (critical for AI inference)
- **Explainability**: Procedural sources are inherently auditable and traceable (vs. black-box embeddings)
- **Multi-modal reasoning**: Unified workspace for visual, spatial, symbolic, and procedural knowledge

### 🕸️ For Knowledge Graph Communities
**Graph Compression and Execution**: PM-KR complements property graph and RDF graph approaches by introducing procedural canonicalization—storing graph patterns as reusable procedures rather than duplicating structure. This supports knowledge graph construction with reduced redundancy and executable semantics.

---

## How to Participate

**Join the Community Group**: https://www.w3.org/community/pm-kr/
- Free W3C account required (create at https://www.w3.org/accounts/request)
- No membership fees, open participation
- Public mailing list, GitHub repositories, monthly meetings

**Contribute to the Discussion**:
- Review the proposal and share feedback
- Propose use cases from your domain (accessibility, XR, knowledge graphs, AI)
- Participate in interoperability studies (RDF/OWL/JSON-LD mapping)
- Help shape conformance levels and test suites

**Motivation and Prior Work**:
This group is motivated by prior work on Knowledge3D (https://github.com/danielcamposramos/Knowledge3D), and that work may inform the group's discussions. That work does not constrain the group's discussions, nor will it be a deliverable of this group. We welcome alternative implementations and approaches!

---

## Timeline and Deliverables

**Public Comment Period**: Now through March 2026
**Expected CG Launch**: April 2026
**First Deliverables**: Study data models, execution semantics, relationships with W3C technologies (RDF, OWL, JSON-LD)

**Potential Outputs** (subject to group consensus):
- Data model specification (procedural composition, symlink references)
- Execution semantics (dual-client rendering, canonical procedures)
- Conformance levels (core, sovereign runtime, auditable production)
- Interoperability guidelines (RDF/OWL/JSON-LD bidirectional mapping)
- Use case documentation (accessibility, XR, knowledge graphs, AI systems)

---

## Contact and Resources

**Mailing List** (once CG launches): public-pm-kr@w3.org
**Proposal**: https://www.w3.org/community/blog/2026/02/20/proposed-group-procedural-memory-knowledge-representation-community-group/
**Reference Work**: https://github.com/danielcamposramos/Knowledge3D

**Questions?** Reply to this email or contact me directly at daniel@echosystems.ai

---

## Why Join?

- **Cross-domain impact**: Work that benefits accessibility, XR, knowledge graphs, and AI systems simultaneously
- **Empirically grounded**: Motivated by real-world validation and production use cases
- **Open collaboration**: No patents, public mailing lists, transparent decision-making
- **Interoperability focus**: Designed to complement existing W3C standards (RDF, OWL, JSON-LD, glTF)

**Philosophy**: We patent nothing. We publish everything. We build in the open.

---

I look forward to collaborating with all of you on this exciting new direction for knowledge representation!

Best regards,
Daniel Ramos

---
Founder, Knowledge3D Project
Proposed Chair, PM-KR Community Group
daniel@echosystems.ai
https://github.com/danielcamposramos/Knowledge3D
```

---

## Mailing List Addresses (Complete)

Copy these for the CC field:

```
public-json-ld-wg@w3.org, public-immersive-web@w3.org, w3c-wai-ig@w3.org, semantic-web@w3.org, public-aikr@lists.w3.org, public-webmachinelearning@lists.w3.org, public-cogai@w3.org, public-kg-construct@w3.org
```

---

## Key Points Addressed

### 1. AI KR Inside PM-KR Umbrella ✅
**Explicitly stated in AI & ML Communities section**:
> "Notably, **AI KR (Artificial Intelligence Knowledge Representation) work naturally lives inside the PM-KR umbrella**—where knowledge is stored as executable procedures that both humans and AI systems can consume."

### 2. Small Topics for Clarity ✅
Organized by W3C group with clear headings:
- 🔗 JSON-LD & RDF/Semantic Web
- 🎨 Immersive Web (WebXR)
- ♿ Web Accessibility (WAI)
- 🤖 AI & Machine Learning
- 🕸️ Knowledge Graph Communities

### 3. Single Email to All Groups ✅
One comprehensive message with relevance statements for each community

### 4. Official Blog Post Link ✅
Prominent placement at the top of email

### 5. Call for Participation ✅
Clear instructions on how to join and contribute

---

## Source References

**Mailing list addresses verified from**:
- [JSON-LD Working Group](https://www.w3.org/groups/wg/json-ld/)
- [Immersive Web WG](https://www.w3.org/groups/wg/immersive-web/)
- [WAI Interest Group](https://www.w3.org/WAI/about/groups/waiig/)
- [Semantic Web Archives](https://lists.w3.org/Archives/Public/semantic-web/)
- [AI KR Community Group](https://www.w3.org/groups/cg/aikr/)
- [Web Machine Learning CG](https://www.w3.org/groups/cg/webmachinelearning/)
- [Cognitive AI CG](https://www.w3.org/community/cogai/)
- [Knowledge Graph Construction CG](https://www.w3.org/community/kg-construct/)

---

**Ready to send!** Just copy the email content and paste into your email client with the CC addresses listed above.

**End of Announcement Email**
