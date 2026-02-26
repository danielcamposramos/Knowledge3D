# PM-KR Email: Welcome Shaun Conway (ixo.world)

**To:** public-pm-kr@w3.org
**Subject:** Welcome ixo.world + Verifiable Procedural Knowledge
**Date:** February 26, 2026
**Status:** DRAFT

---

## Email Body

Subject: Welcome ixo.world + Verifiable Procedural Knowledge

Hi Shaun,

Welcome to the PM-KR Community Group! We're excited to have ixo.world join—your work on W3C DIDs, Verifiable Claims, and "Blockchain Internet of Impact" aligns beautifully with where our metadata discussions are heading.

### Context: The Metadata Evolution

Adam and Christoph have been exploring how JSON-LD metadata evolves with procedural knowledge. Adam recently shared the Solid Community Group's Web Access Control (WAC) and Access Control Policy (ACP) specifications, highlighting:

- **JSON-LD merging** across multiple contexts (security, provenance, permissions)
- **SHACL/SPARQL constraints** for validating metadata integrity
- **Runtime function-call contexts** where metadata guides AI agent behavior

Christoph emphasized the evolutionary path: "1) Attach concrete things first, 2) Allow for mapping concrete to more abstract later, 3) Abstract can evolve into new concrete things."

(In our previous email, we covered the JSON-LD fundamentals—how procedural programs carry metadata for preconditions, effects, and permissions. This builds on that foundation.)

### ixo.world + PM-KR: Verifiable Procedural Provenance

Your **InterNFTs** as "programmable, dynamic containers of verifiable data" map perfectly to PM-KR's vision:

**What ixo brings:**
- W3C DIDs for cryptographic identity and attribution
- Verifiable Claims/Credentials for blockchain-backed provenance
- "Proof of Impact" as immutable audit trails

**What PM-KR adds:**
- Procedural programs (RPN) as the *executable content* inside those verifiable containers
- JSON-LD metadata describing what procedures do (preconditions, effects, permissions)
- Compositional references enabling knowledge chains (procedure A references procedure B)

**Together:** Cryptographically verifiable knowledge chains where each procedural step carries its own DID-signed provenance.

### Example: Verifiable Procedural Chain

```json
{
  "@context": [
    "https://pm-kr.org/contexts/procedural.jsonld",
    "https://w3id.org/did/v1",
    "https://w3id.org/vc/v1"
  ],
  "@type": "VerifiableProceduralProgram",
  "id": "did:ixo:Ef3j2kL9sN7qP1mR",
  "issuer": "did:ixo:shaun-conway",
  "issuanceDate": "2026-02-26T12:00:00Z",
  "credentialSubject": {
    "@type": "ProceduralProgram",
    "name": "Carbon Credit Verification",
    "program": [
      "LOAD", "satellite_data",
      "CALL", "did:ixo:forest_analysis_v2",  // References another verified procedure
      "CALL", "did:ixo:carbon_calculation",  // Blockchain-backed provenance
      "VERIFY", "stakeholder_signatures",
      "ISSUE", "carbon_credit"
    ],
    "preconditions": [
      {"type": "DataAvailability", "source": "Sentinel-2"},
      {"type": "CalibrationCheck", "tolerance": 0.05}
    ],
    "effects": [
      {"type": "CreditIssued", "standard": "Verra-VCS"},
      {"type": "BlockchainRecord", "chain": "ixo"}
    ]
  },
  "proof": {
    "type": "Ed25519Signature2020",
    "created": "2026-02-26T12:00:00Z",
    "proofPurpose": "assertionMethod",
    "verificationMethod": "did:ixo:shaun-conway#key-1",
    "proofValue": "z3j4k5L6m7N8..."
  }
}
```

### Responding to Adam & Christoph's Thread

**On JSON-LD merging:**
ixo's InterNFTs already demonstrate this—your verifiable data containers merge contexts from DIDs, Verifiable Credentials, and domain-specific schemas. PM-KR extends this to *procedural* contexts where the metadata describes **what the program does**, not just **what the data represents**.

**On SHACL/SPARQL constraints:**
Perfect for validating procedural metadata! Example: "This carbon calculation procedure MUST reference a calibrated satellite data source (SHACL constraint), and its provenance chain MUST include stakeholder signatures (SPARQL query validation)."

**On Christoph's evolution path:**
1. **Concrete first:** Attach DIDs to specific procedures (e.g., "forest_analysis_v2")
2. **Abstract later:** Create categories (e.g., "EnvironmentalAnalysis" type)
3. **Evolve:** New concrete procedures inherit metadata schemas from abstract types

This matches ixo's approach—start with concrete impact claims, build ontologies over time.

### Where We Could Collaborate

**1. Procedural Impact Claims:**
Instead of static Verifiable Claims, PM-KR enables *executable* impact verification:
- "Here's the procedure I used to calculate carbon credits" (reproducible, auditable)
- "Here's the provenance chain of all referenced procedures" (blockchain-backed)

**2. Multi-Modal Accessibility:**
PM-KR's procedural representation enables rendering the *same knowledge* in multiple formats:
- Visual flowcharts for auditors
- Audio descriptions for accessibility
- Machine-executable for AI validation

All from one canonical procedural source with DID-signed provenance.

**3. Knowledge Evolution:**
As procedures get refined (e.g., "forest_analysis_v2" → "v3"), PM-KR's compositional references + ixo's DIDs = complete audit trails of knowledge evolution.

### Questions for ixo

1. **Current metadata schemas:**
   What JSON-LD contexts do you use for InterNFT metadata? (We'd love to see how you structure impact claims.)

2. **Provenance granularity:**
   Do you track provenance at the data level, or could it extend to the *procedures* that process/validate that data?

3. **Constraint validation:**
   Are you exploring SHACL/SPARQL for validating InterNFT metadata, or using custom validation logic?

4. **AI integration:**
   How do you envision AI agents interacting with verifiable impact data? (PM-KR's procedural programs could be the executable bridge.)

### Looking Forward

PM-KR aims to be the **procedural layer** for W3C's knowledge stack:
- **RDF/OWL:** Declarative facts
- **DIDs/VCs:** Verifiable identities and claims
- **PM-KR:** Executable procedures with metadata

ixo.world is already bridging DIDs + domain knowledge (impact verification). Adding PM-KR's procedural representation could enable **cryptographically verifiable knowledge chains** where every step is auditable, reproducible, and blockchain-backed.

We'd love to hear your thoughts—especially how "Proof of Impact" could evolve into "Proof of Procedural Impact" (not just verifying outcomes, but verifying the process itself).

Welcome aboard!

**Daniel Ramos**
Co-Chair, W3C PM-KR Community Group
AI Knowledge Architect

---

## Notes

**References:**
- ixo.world: https://www.ixo.world/
- W3C DIDs: https://www.w3.org/TR/did-core/
- W3C Verifiable Claims: https://www.w3.org/TR/vc-data-model/
- Solid CG WAC: https://github.com/solid/web-access-control-spec
- Solid CG ACP: https://github.com/solid/authorization-panel/blob/main/proposals/acp/index.md

**Previous email context:**
- Feb 25 email covered JSON-LD metadata fundamentals (preconditions, effects, permissions)
- Adam's intervention on procedural metadata (STRIPS, PDDL, security frameworks)
- Christoph's use case (metadata evolution, multiple schemas)

**Draft notes:**
- Connects ixo's W3C DIDs/VCs to PM-KR's procedural provenance
- References Adam & Christoph's ongoing discussion (JSON-LD merging, SHACL, evolution)
- Acknowledges previous email covered JSON-LD basics
- Provides concrete example (carbon credit verification as procedural program)
- Asks open-ended questions to invite ixo's perspective
- Positions PM-KR as "procedural layer" in W3C knowledge stack

**Last Updated:** February 26, 2026
