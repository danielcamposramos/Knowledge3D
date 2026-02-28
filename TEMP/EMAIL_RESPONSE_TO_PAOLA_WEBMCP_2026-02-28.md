# Response to Paola Di Maio: WebMCP Security Concerns + PM-KR Patterns

**To:** public-webagents@w3.org, public-aikr@w3.org
**Cc:** public-pm-kr@w3.org, public-webmachinelearning@w3.org
**From:** Daniel Campos Ramos <capitain_jack@yahoo.com>
**Subject:** Re: WebMCP Security Concerns — PM-KR's Security-First Approach as Potential Model
**Date:** February 28, 2026

---

Dear Paola and W3C AI/ML Community,

Thank you, Paola, for raising these critical concerns about WebMCP's security, privacy, and accessibility gaps. Your rigorous scrutiny is exactly what W3C standardization needs — and it's what led to PM-KR's creation when you challenged our initial AI-KR proposals.

**Your core concern resonates deeply:**
> "The security implications of exposing website functionality to autonomous agents through a browser API without a defined consent model, permission framework, or threat analysis have not been resolved by the standards body. Selling security tooling for a threat model that does not yet exist is not responsible engineering."

This is the right standard to hold W3C work to. **Define security/privacy BEFORE deployment, not after commercialization.**

---

## PM-KR's Security-First Approach (Potential Lessons for WebMCP)

The **PM-KR Community Group** (launched Feb 20, 2026) designed security, privacy, and provenance **from day 1** — before any implementation or commercialization. I'd like to share our approach as potential patterns for WebMCP's missing sections.

**PM-KR Mission:** Procedural knowledge representation that AI agents can consume with **transparent provenance, inspectable execution, and user sovereignty**.

### 1. **Security: Sovereignty Firewall**

**PM-KR's approach:**
- **Sovereign execution:** All knowledge execution happens in user-controlled environments (local GPU, no external API calls in hot path)
- **Inspectable procedures:** Every RPN program is human-readable (users see EXACTLY what AI agents execute)
- **Explicit permissions:** Users approve which procedural knowledge sources AI agents can access

**Parallel to WebMCP:**
- WebMCP exposes website functions to AI agents → **Who verifies the function is safe?**
- PM-KR exposes procedural knowledge to AI agents → **User inspects RPN programs before agent executes**

**Potential lesson:** Define **user consent model** before agents invoke website tools. What does "permission" look like for WebMCP tool registration?

---

### 2. **Privacy: Verifiable Credentials for Provenance**

**PM-KR's approach:**
- **Verifiable Credentials integration:** Every procedural knowledge source has cryptographic provenance (who created it, who validated it, delegation chains)
- **Audit trails:** AI agents record which knowledge sources they queried (inspectable, verifiable)
- **Public/private separation:** Users control what knowledge AI agents can access

**Parallel to WebMCP:**
- WebMCP tools execute on user's behalf → **Who audits what the tool did?**
- PM-KR procedures execute on user's behalf → **Verifiable Credentials provide audit trail**

**Potential lesson:** Define **provenance model** for WebMCP tool invocations. How do users verify that an agent invoked the right tool with the right parameters?

---

### 3. **Accessibility: Multi-Modal Rendering (Not an Afterthought)**

**PM-KR's approach:**
- **Design principle:** Procedural knowledge MUST render in multiple modalities (visual, audio, tactile)
- **Same source, multiple modalities:** ONE RPN program → visual rendering, audio pronunciation, tactile 3D print
- **Accessibility = core feature, not separate section**

**Parallel to WebMCP:**
- WebMCP tools described as JSON schemas → **How do screen readers interpret tool descriptions?**
- PM-KR knowledge described as RPN + metadata → **Audio/tactile rendering rules included**

**Potential lesson:** Define **accessibility contract** for WebMCP. How do assistive technologies interpret tool schemas? What's the screen reader experience when an agent registers a tool?

---

### 4. **Premature Commercialization Prevention**

**PM-KR's approach:**
- **Apache 2.0 license, no patents:** Public prior art prevents patent trolling
- **Specifications BEFORE products:** PM-KR Core Spec v1.0 target = Q4 2026 (9 months of community review BEFORE declaring "ready")
- **No fear-based marketing:** PM-KR complements declarative standards (RDF/OWL/JSON-LD), doesn't claim "implement or die"

**Parallel to WebMCP:**
- Commercial products selling "WebMCP readiness audits" before spec defines security → **Classic premature commercialization**
- PM-KR: Reference implementation exists (Knowledge3D), but NO commercial products until specs finalize

**Potential lesson:** W3C should clarify that **Draft ≠ Production-Ready**. Chrome's feature flag preview is for experimentation, not deployment.

---

## Invitation: Cross-CG Collaboration on Security Patterns

**WebMCP and PM-KR address different layers:**
- **WebMCP:** Browser API for AI agents to invoke website functions
- **PM-KR:** Knowledge representation for AI agents to consume procedural knowledge

**But both need:**
- ✅ Security model (who can do what?)
- ✅ Privacy/provenance (audit trail for agent actions)
- ✅ Accessibility (how do assistive technologies interact?)
- ✅ User consent (what permissions are needed?)

**Proposal:** Could PM-KR and WebMCP CGs collaborate on shared security patterns?

**Example collaboration:**
1. PM-KR's Verifiable Credentials approach → Adapted for WebMCP tool provenance
2. PM-KR's Sovereignty Firewall → Adapted for WebMCP cross-origin tool isolation
3. PM-KR's multi-modal accessibility → Adapted for WebMCP tool description rendering

**Benefit:** Both CGs solve security/privacy/accessibility together, faster than separately.

---

## Supporting Paola's Recommendations

**I endorse Paola's guidance:**

1. ✅ **Developers:** Experiment with Chrome preview, but **DO NOT deploy WebMCP tools on production sites** until security model is defined
2. ✅ **Buyers:** No commercial product can deliver "WebMCP security compliance" because **the spec has not defined what compliance means**
3. ✅ **Standards participants:** The window for meaningful input is NOW (next W3C Web ML CG meeting: March 5, 2026)

**Additional recommendation:**
4. ✅ **W3C leadership:** Consider holding a joint security workshop (WebMCP + PM-KR + Verifiable Credentials WG) to define shared patterns for AI agent security/privacy/provenance

---

## PM-KR as Complementary (Not Competitive)

**PM-KR does NOT compete with WebMCP:**
- WebMCP: "How do AI agents invoke website functions?"
- PM-KR: "How do AI agents consume procedural knowledge?"

**But PM-KR's security-first design could inform WebMCP:**
- Sovereignty Firewall → User-controlled execution environments
- Verifiable Credentials → Cryptographic provenance for agent actions
- Inspectable procedures → Transparency (users see what agents execute)
- Multi-modal accessibility → Assistive technology integration from day 1

**We're happy to share PM-KR's security patterns with WebMCP CG.**

---

## Closing Thoughts

**Paola's critique is a gift to W3C standardization.**

Premature commercialization undermines trust in W3C standards. When businesses sell "compliance" for specs that haven't defined compliance requirements, it creates:
- ❌ False sense of security (customers think they're protected)
- ❌ Vendor lock-in (proprietary "compliance tools" before open standards exist)
- ❌ Reputation damage (W3C associated with premature products)

**The right sequence:**
1. Define security/privacy/accessibility model (W3C CG community work)
2. Publish candidate spec with normative requirements (not TODOs)
3. Implement reference implementations (validate spec is implementable)
4. THEN commercialize (after compliance requirements are clear)

**PM-KR committed to this sequence.** We hope WebMCP CG does the same.

Thank you, Paola, for holding us to high standards. 🙏

---

Best regards,

**Daniel Campos Ramos**
PM-KR Co-Chair
Brazilian Registered Electrical Engineer
W3C PM-KR Community Group
capitain_jack@yahoo.com

**Milton Ponson**
PM-KR Co-Chair (Mathematical Foundations)
W3C PM-KR Community Group
rwiciamsd@gmail.com

---

**Links:**
- PM-KR Community Group: https://www.w3.org/community/pm-kr/
- PM-KR Security Approach: https://github.com/danielcamposramos/Knowledge3D/blob/main/docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md (Section 3: Sovereignty Firewall)
- PM-KR Provenance: Verifiable Credentials integration (collaboration with W3C VC WG)
- WebMCP Spec (referenced by Paola): https://webmachinelearning.github.io/webmcp/
- WebMCP Security Issues: https://github.com/webmachinelearning/webmcp/issues

---

**P.S. To WebMCP CG:**

This is NOT criticism — it's an offer to collaborate. PM-KR spent significant effort designing security/privacy/provenance from day 1. If our patterns can help WebMCP fill the empty Security/Privacy/Accessibility sections faster, we're happy to contribute.

**Invitation:** Join PM-KR's March 5 discussion on Verifiable Credentials for AI agent provenance (same day as WebMCP CG meeting). Perhaps we can find synergies.
