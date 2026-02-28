# Re-invitation Email to Dave Raggett and Tyson Clark

**To:** Dave Raggett <dsr@w3.org>, Tyson Clark <tyson@slashlife.ai>
**From:** Daniel Campos Ramos <capitain_jack@yahoo.com>
**CC:** public-pm-kr@w3.org
**Subject:** PM-KR Mission Updated — Addressing Procedural vs Declarative & Context Handling
**Date:** February 28, 2026

---

Dear Dave and Tyson,

Thank you, Dave, for your thoughtful feedback on PM-KR's mission statement. Your questions about **why procedural over declarative** and **how we handle context-dependent meanings** were exactly the rigor we needed to strengthen our positioning.

I've revised the PM-KR mission statement to directly address your concerns:

**Updated Mission:** https://www.w3.org/community/pm-kr/

---

## Key Additions Based on Your Feedback

### 1. Why Procedural Over Declarative?

**Your question:** "What's wrong with a declarative approach?"

**Our answer (new section):**

Declarative approaches (RDF, OWL, JSON-LD) excel at **describing relationships** but face limitations:
- **Static descriptions don't compose**: Declaring "a chair is furniture" doesn't tell you HOW to render a chair at different scales, orientations, or styles
- **Context-dependent meanings require execution**: The symbol "∫" means different things in calculus, physics, and probability — declaring these relationships doesn't provide the EXECUTION logic needed to compute results
- **Duplication across modalities**: Declarative systems require separate descriptions for visual rendering, audio pronunciation, tactile representation, and computational execution

**PM-KR's contribution:** We combine **transparency of declarative systems** (inspectable rules) with **executability of neural networks** (procedural computation) plus **composability** (which neither provides).

See comparison table in revised mission: Declarative vs Neural Networks vs PM-KR.

---

### 2. Handling Context-Dependent Meanings

**Your question:** "What about the likelihood that different terms have subtly different meanings in their respective context of use?"

**Our answer (new section with concrete example):**

PM-KR uses **explicit context rules** (procedural programs) that adapt execution based on context:

**Example: "Chair" in Different Contexts**
- **Furniture catalog:** Photorealistic rendering + price metadata
- **Architectural BIM:** Collision mesh + load-bearing physics
- **Game environment:** Low-poly mesh + interaction rules (sittable, throwable)
- **Accessibility:** Audio pronunciation + 3D-printable tactile mesh

**Result:**
- Same base knowledge (chair = seat + back + legs)
- Context-specific execution (catalog vs BIM vs game vs accessibility)
- Composable (contexts can inherit/override base procedures)
- Transparent (every context rule is inspectable RPN program)

Unlike neural networks (context hidden in weights) or declarative systems (manual context modeling), PM-KR provides **inspectable, composable context rules**.

---

### 3. Concrete Applications (Addressing "Why Is It Needed?")

**Your recommendation:** "Explain PM-KR in respect to why it is needed and what applications it targets."

**Our answer (new section with 5 detailed examples):**

1. **Education**: MIT OpenCourseWare publishes procedural calculus textbooks → AI tutors AND students consume same source
2. **Gaming**: D&D SRD as procedural rules → Human DMs read, AI DMs execute (zero divergence)
3. **Science**: Nature Protocols publishes procedural biochemistry experiments → Scientists read, lab robots execute
4. **Accessibility**: Mathematical textbooks with visual equations, spoken descriptions, tactile 3D-printed graphs (all from ONE source)
5. **AI Training**: Wikipedia as PM-KR procedural knowledge base → AI systems query during inference (no training duplication)

Each example shows **why procedural execution is needed** (not just declarative description) and **what problem it solves**.

---

## Comparison to PKN (Dave's Reference)

You mentioned PKN (Procedural Knowledge Networks) as demonstrating "the potential for combining declarative approaches with a qualitative treatment of metadata."

**PM-KR's position:**
- We agree metadata is critical (that's why we use JSON-LD + Verifiable Credentials for provenance)
- We add **executable procedures** on top of declarative metadata (complementary, not replacing)
- Our focus: Knowledge that BOTH humans and AI systems can consume from the SAME source

**Positioning:**
- RDF/OWL: Declarative relationships (static)
- Neural Networks: Procedural execution (opaque)
- PKN: Declarative + metadata (qualitative)
- **PM-KR: Procedural + metadata + composability + transparency**

We see PM-KR as the **execution layer** that complements declarative standards (RDF/OWL/JSON-LD), not replacing them.

---

## Re-Invitation to Join PM-KR

Dave, Tyson — your expertise would be invaluable:

**For Dave:**
- W3C veteran perspective on semantic web integration
- Critical assessment of PM-KR's relationship to RDF/OWL
- Guidance on standards maturity and adoption pathways

**For Tyson:**
- Verifiable reasoning traces across institutional boundaries (your original question)
- PM-KR's Verifiable Credentials integration for knowledge provenance
- Real-world validation in distributed knowledge systems

**No obligation to implement** — even critical feedback and use case validation would help us refine PM-KR's scope and positioning.

**Join here:** https://www.w3.org/community/pm-kr/

---

## Current Momentum (Since Feb 20, 2026)

- ✅ 18+ members (MIT, Huawei, JSON-LD co-creator Manu Sporny)
- ✅ Initial specification drafts in GitHub
- ✅ Reference implementation (Knowledge3D: Python/CUDA)
- ✅ Empirical validation (50-90% compression, dual-client contract working)

**Next milestones:**
- Q2 2026: PM-KR Core Specification v0.5 (draft for community review)
- Q3 2026: W3C TPAC breakout session
- Q4 2026: PM-KR Core Specification v1.0

---

## Thank You

Dave, your feedback transformed our mission statement from "here's what we're doing" to "here's WHY procedural over declarative, HOW we handle context, and WHAT applications need this."

This is exactly the rigor PM-KR needs to mature into a viable W3C standard.

**Questions? Concerns? Critical feedback?** Please share — we want to get this right.

Best regards,

**Daniel Campos Ramos**
PM-KR Co-Chair
Brazilian Registered Electrical Engineer
W3C PM-KR Community Group
capitain_jack@yahoo.com

---

**P.S.** Tyson, your original question about "verifiable reasoning traces across institutional boundaries" maps directly to PM-KR's architecture:
- **Procedural programs** = reasoning traces (inspectable RPN execution)
- **Verifiable Credentials** = provenance (who validated procedures, delegation chains)
- **Compositional architecture** = institutional boundaries (federated Galaxy knowledge bases)

Would love to discuss how this applies to your work at SlashLife AI.

---

**Links:**
- Updated Mission: https://www.w3.org/community/pm-kr/
- GitHub Specs: https://github.com/danielcamposramos/Knowledge3D/tree/main/docs/vocabulary
- Mailing List: public-pm-kr@w3.org
