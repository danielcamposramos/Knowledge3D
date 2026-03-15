# PM-KR Community Group — Final Proposal Text
# (Incorporating Ian Jacobs' Feedback)

**Date**: February 20, 2026
**Version**: Final (v3)
**For**: W3C Community Group Proposal Form

---

## Short Description (For Proposal Form)

Current knowledge representation systems suffer from massive duplication and fragmentation: the same knowledge (e.g., a Unicode character, mathematical symbol, or spatial concept) is duplicated across fonts, embeddings, accessibility metadata, and visual renderings. Redundant representations can create maintenance, performance, security, and licensing issues.

The PM-KR Community Group will develop a knowledge representation paradigm where knowledge is stored once as executable procedures (like font programs or mathematical formula definitions) and referenced via symlink-style composition, enabling both humans and AI systems to consume the same procedural source.

The group will study data models, execution semantics, conformance levels, and relationships with other W3C technologies (e.g., RDF, OWL, JSON-LD).

---

## Regarding Knowledge3D Reference Implementation

This group is motivated by prior work on Knowledge3D, and that work may inform the group's discussions. That work does not constrain the group's discussions, nor will it be a deliverable of this group.

---

## Email Response to Ian (Final Version)

```
Dear Ian,

Thank you for the clear guidance! I really appreciate the simplifications—they make the proposal much more accessible.

Here is the revised short description incorporating your feedback:

---

**Short Description:**

Current knowledge representation systems suffer from massive duplication and fragmentation: the same knowledge (e.g., a Unicode character, mathematical symbol, or spatial concept) is duplicated across fonts, embeddings, accessibility metadata, and visual renderings. Redundant representations can create maintenance, performance, security, and licensing issues.

The PM-KR Community Group will develop a knowledge representation paradigm where knowledge is stored once as executable procedures (like font programs or mathematical formula definitions) and referenced via symlink-style composition, enabling both humans and AI systems to consume the same procedural source.

The group will study data models, execution semantics, conformance levels, and relationships with other W3C technologies (e.g., RDF, OWL, JSON-LD).

---

**Regarding Knowledge3D:**

This group is motivated by prior work on Knowledge3D, and that work may inform the group's discussions. That work does not constrain the group's discussions, nor will it be a deliverable of this group.

---

I've removed the specific "~70% waste" figure (as you suggested, too detailed for the short intro—it can be discussed in the group's work if relevant). I've also simplified the problem statement to focus on the high-level issue ("Redundant representations can create maintenance, performance, security, and licensing issues") and changed "standardize" to "study" to properly reflect CG's role in incubating technologies.

Regarding the charter draft: understood! I'll prepare a simplified version aligned with CG policies and send it to the list when ready.

Thank you again for the patient guidance and for helping shape this into something clear and aligned with W3C practices. Have a great weekend!

Best regards,
Daniel Ramos

---
daniel@echosystems.ai (personal)
capitain_jack@yahoo.com (current thread)
https://github.com/danielcamposramos/Knowledge3D
```

---

## Changes Made (v2 → v3)

### 1. Removed Specific Metrics
- ❌ "~70% waste" (too detailed for short intro)
- ❌ "validated ~70% reduction without information loss" (moved to later group discussions)

### 2. Simplified Problem Statement
**Before (v2)**:
> "This duplication creates three critical problems: (1) compression-meaning tradeoffs force systems to choose between lossless-but-large (zip, gzip) or lossy-but-small (embeddings, quantization), losing semantic fidelity; (2) procedural-static divides mean humans consume procedural sources (fonts, SVG, LaTeX) while AI systems consume separate static payloads (embeddings, pixels), requiring dual maintenance; (3) sovereignty crises arise when knowledge systems depend on external frameworks (numpy, CUDA, ML libraries), creating cascading dependencies that introduce security, licensing, and maintenance risks."

**After (v3)**:
> "Redundant representations can create maintenance, performance, security, and licensing issues."

✅ **Much clearer and more accessible!**

### 3. Added Paragraph Break
- New paragraph after describing the approach (before group activities)
- Improves readability and flow

### 4. Changed "standardize" to "study"
**Before (v2)**:
> "The group will standardize the data model, execution semantics, conformance levels, and interoperability with existing W3C standards (RDF, OWL, JSON-LD)."

**After (v3)**:
> "The group will study data models, execution semantics, conformance levels, and relationships with other W3C technologies (e.g., RDF, OWL, JSON-LD)."

✅ **Properly reflects CG role (incubate, not standardize)**

### 5. Simplified K3D Role Statement
**Before (v2)**:
> "K3D is input for conversation and empirical validation, NOT a requirement for participation. The CG will develop vendor-neutral specifications that anyone can implement independently. K3D provides proof-of-concept evidence (compression metrics, test results, benchmark data) to ground the standardization work, but participants are welcome to contribute alternative implementations, critique the approach, or propose different solutions to the same problems. The goal is a standard that works for graph databases (Neo4j), AI platforms (Hugging Face), XR systems (Three.js), and accessibility tools—not to promote a single implementation."

**After (v3)**:
> "This group is motivated by prior work on Knowledge3D, and that work may inform the group's discussions. That work does not constrain the group's discussions, nor will it be a deliverable of this group."

✅ **Single clear sentence, removes promotional tone**

### 6. Charter Process Clarification
- Will send charter draft to the list (not directly to Ian)
- Allows community review and feedback

---

## What to Send

**Copy the email from "Email Response to Ian (Final Version)" section** and send to:

**To**: public-new-work@w3.org (W3C mailing list, as Ian indicated "this list")
**CC**: ij@w3.org (keep Ian in the loop)
**Subject**: RE: Procedural Memory Knowledge Representation Community Group Proposal

**Body**: Copy the email text from above

---

## Next Steps

**After sending this email**:

1. **Ian updates proposal** with the final short description
2. **Prepare simplified charter** (aligned with W3C CG template, focused on incubation not standardization)
3. **Send charter draft to list** (public-new-work@w3.org) for community review
4. **W3C approves CG** (estimated 1-2 weeks after charter finalized)
5. **CG launches** at w3.org/community/pm-kr/

**Key Learnings**:
- ✅ Keep it simple (don't overwhelm with technical details)
- ✅ Focus on problem, not solution metrics
- ✅ CGs incubate (not standardize)
- ✅ Reference work informs but doesn't constrain
- ✅ Trust W3C processes (don't override)

---

**Ian's feedback was excellent** — this version is much more accessible and properly positioned for a Community Group! 🎉

**End of Final Proposal Text**
