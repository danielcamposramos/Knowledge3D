# Public PM-KR Email: Mission Statement Update Based on Community Feedback

**To:** public-pm-kr@w3.org
**From:** Daniel Campos Ramos <capitain_jack@yahoo.com>
**Subject:** PM-KR Mission Updated — Procedural vs Declarative, Context Handling, Concrete Applications
**Date:** February 28, 2026

---

Dear PM-KR Community,

Following feedback from W3C veteran **Dave Raggett** (dsr@w3.org), I've significantly revised the PM-KR Community Group mission statement to address critical questions about our approach.

**Updated Mission:** https://www.w3.org/community/pm-kr/

---

## What Changed (Summary)

### 1. **Why Procedural Over Declarative?** (New Section)

**Dave's question:** "What's wrong with a declarative approach?"

We've added a detailed comparison showing:

| **Approach** | **Execution** | **Transparency** | **Composability** | **Context Handling** |
|--------------|---------------|------------------|-------------------|----------------------|
| **Declarative (RDF/OWL)** | ❌ No (describes, doesn't execute) | ✅ Transparent | ⚠️ Limited (static relationships) | ⚠️ Manual modeling |
| **Neural Networks** | ✅ Yes (procedural) | ❌ Opaque (black box) | ❌ No (monolithic) | ✅ Learned (but unexplainable) |
| **PM-KR** | **✅ Yes (procedural)** | **✅ Transparent** | **✅ Yes (compositional)** | **✅ Explicit context rules** |

**Key insight:** PM-KR combines **transparency of declarative systems** with **executability of neural networks** while adding **composability** that neither provides.

**Example:** Mathematical symbol "∫" (integral)
- **Declarative (RDF):** `rdfs:label "integral" ; math:relatedTo :summation` ❌ Doesn't tell you HOW to compute or render
- **PM-KR:** Provides `visual_rpn` (glyph rendering), `execution_rpn` (integration algorithm), `audio_rpn` (pronunciation), context-specific rules ✅

**Positioning:** PM-KR is the **execution layer** that complements declarative standards (RDF/OWL/JSON-LD), not replacing them.

---

### 2. **Handling Context-Dependent Meanings** (New Section)

**Dave's question:** "What about subtly different meanings in different contexts?"

**PM-KR's answer:** Explicit context rules (procedural programs)

**Example: "Chair" in Different Contexts**
- **Furniture catalog:** Photorealistic rendering + price metadata
- **Architectural BIM:** Collision mesh + load-bearing physics
- **Game environment:** Low-poly mesh + interaction rules (sittable, throwable)
- **Accessibility:** Audio pronunciation + 3D-printable tactile mesh

**Result:**
- Same base knowledge (`chair = seat + back + legs`)
- Context-specific execution (inspectable procedural rules)
- Composable (contexts inherit/override base procedures)
- Transparent (not hidden in neural weights)

See revised mission for full JSON-LD example.

---

### 3. **Concrete Applications** (New Section)

**Dave's recommendation:** "Explain PM-KR in respect to why it is needed and what applications it targets."

**Added 5 detailed examples:**

1. **Education (MIT OpenCourseWare)**
   - ONE procedural calculus textbook → AI tutors AND students consume same source
   - Impact: Accessibility (visual, audio, tactile), AI integration, maintenance

2. **Gaming (D&D SRD)**
   - Game rules as procedural programs → Human DMs read, AI DMs execute (same source)
   - Impact: Zero divergence, AI DM uses canonical rules (no hallucination)

3. **Science (Nature Protocols)**
   - Experimental protocols as procedural programs → Scientists read, lab robots execute
   - Impact: Reproducibility crisis addressed (protocol description = execution)

4. **Accessibility (Multi-Modal Textbooks)**
   - Visual equations, spoken descriptions, tactile 3D-printed graphs (all from ONE source)
   - Impact: Blind students access same knowledge as sighted (not separate "accessible version")

5. **AI Training (Wikipedia Procedural KB)**
   - Wikipedia as PM-KR source → AI queries during inference (no training duplication)
   - Impact: Carbon reduction (train once, reference forever)

Each example shows **why procedural execution is needed** (not just declarative description) and **what problem it solves**.

---

## Community Discussion Invited

**Questions for the group:**

1. **Declarative integration:** How should PM-KR's execution layer integrate with existing RDF/OWL ontologies? Should we define mapping rules?

2. **Context rule semantics:** Are the proposed context inheritance/override rules sufficient? What edge cases are we missing?

3. **Application priorities:** Which of the 5 applications (education, gaming, science, accessibility, AI training) should we prioritize for early validation?

4. **Neural network comparison:** Is our positioning vs neural networks (procedural + transparent vs procedural + opaque) clear? How can we strengthen this distinction?

5. **PKN relationship:** Dave mentioned PKN (Procedural Knowledge Networks) as combining declarative approaches with qualitative metadata. Should we establish liaison with PKN researchers?

**Please share:**
- Critical feedback (what's unclear, what's wrong?)
- Additional use cases from your domain
- Alternative approaches we should consider
- Relevant prior art (papers, systems, standards)

---

## Acknowledgment

Thank you, **Dave Raggett**, for the rigorous feedback. Your questions transformed our mission statement from "here's what we're doing" to "here's WHY procedural over declarative, HOW we handle context, and WHAT applications need this."

This is exactly the rigor PM-KR needs to mature into a viable W3C standard.

**To all members:** Please continue challenging our assumptions. PM-KR's strength will come from community scrutiny and diverse perspectives.

---

## Current Status (Week 1 Complete)

**Momentum (Feb 20-28, 2026):**
- ✅ 18+ members (MIT, Huawei, JSON-LD co-creator Manu Sporny)
- ✅ Mission statement revised based on community feedback (v1.0 → v1.1)
- ✅ Initial specification drafts in GitHub
- ✅ Reference implementation (Knowledge3D: Python/CUDA)
- ✅ Empirical validation (50-90% compression, dual-client contract working)

**Next steps:**
- **Q2 2026:** PM-KR Core Specification v0.5 (draft for community review)
- **Q3 2026:** W3C TPAC breakout session
- **Q4 2026:** PM-KR Core Specification v1.0

---

## Resources

**Updated Mission Statement:**
- https://www.w3.org/community/pm-kr/

**GitHub Repository:**
- https://github.com/danielcamposramos/Knowledge3D
- Specifications: `/docs/vocabulary/`
- Reference implementation: `/knowledge3d/`

**Mailing List Archives:**
- https://lists.w3.org/Archives/Public/public-pm-kr/

**How to Contribute:**
- Join: https://www.w3.org/community/pm-kr/
- Participate: Review specs, propose use cases, implement prototypes
- Discuss: Email public-pm-kr@w3.org with feedback, questions, ideas

---

## Closing Thoughts

PM-KR is a **community effort**. Our mission statement will continue evolving based on your feedback, empirical validation, and real-world use cases.

**Week 1 lesson:** Rigorous critique (like Dave's) makes PM-KR stronger. Please keep challenging us.

**What PM-KR needs from you:**
- Critical feedback (technical, conceptual, strategic)
- Domain expertise (education, gaming, science, accessibility, AI)
- Implementation experience (prototypes, benchmarks, validations)
- Prior art pointers (papers, systems, standards we should know)

Let's build the future of knowledge representation together — **transparently, collaboratively, rigorously**. 🚀

---

Best regards,

**Daniel Campos Ramos**
PM-KR Co-Chair
Brazilian Registered Electrical Engineer
W3C PM-KR Community Group
capitain_jack@yahoo.com

**Milton Ponson**
PM-KR Co-Chair (Mathematical Foundations)

---

**P.S.** If you haven't joined yet: https://www.w3.org/community/pm-kr/

No W3C membership required — Community Groups are open to all.

---

**Links:**
- Updated Mission (v1.1): https://www.w3.org/community/pm-kr/
- GitHub Specs: https://github.com/danielcamposramos/Knowledge3D/tree/main/docs/vocabulary
- Mailing List: public-pm-kr@w3.org
- NotebookLM Research Space: https://notebooklm.google.com/notebook/1bd10bda-8900-4c41-931e-c9ec67ac865f
