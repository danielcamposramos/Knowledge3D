# From form to meaning: shared knowledge for people and AI

**For researchers, implementers and curious readers:** this explains the problem, a proposed first deliverable, and what still needs to be demonstrated.

**Status:** Knowledge3D project draft, 29 October 2026, offered for discussion in the [Procedural Memory Knowledge Representation (PM-KR) Community Group](https://www.w3.org/community/pm-kr/).
It is not an adopted group report or a W3C standard.
The code is paused while work focuses on specifications and the group's deliverables, as Daniel stated [on 28 October](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0016.html).

## Authors

Daniel Campos Ramos, Knowledge3D, EchoSystems AI Studios.
AI partners assisted with drafting and review; project contributions are credited in [Attributions](../../ATTRIBUTIONS.md).

## Participate

Read the [public discussion](https://lists.w3.org/Archives/Public/public-pm-kr/) and bring a concrete use case or objection to the [PM-KR issue tracker](https://github.com/w3c-cg/pm-kr/issues).
Use [Knowledge3D issues](https://github.com/danielcamposramos/Knowledge3D/issues) for corrections to these project documents.
Anyone with a W3C account can [join the group](https://www.w3.org/community/pm-kr/), subject to the [participation terms](https://www.w3.org/community/about/faq/#how-do-i-join-a-group).
W3C membership is not required.

## Contents

- [Introduction](#introduction)
- [User needs and use cases](#user-needs-and-use-cases)
- [Goals](#goals) and [non-goals](#non-goals)
- [Proposed approach](#proposed-approach) and [alternatives considered](#alternatives-considered)
- [Considerations](#accessibility-internationalization-privacy-and-security), [stakeholder feedback](#stakeholder-feedback) and [references](#references-and-acknowledgements)

## Introduction

A teacher may maintain a diagram, its explanation and the information an AI assistant reads as separate records.
When a fact changes, those records can disagree.
Knowledge3D explores storing knowledge once as reusable procedures, with explicit links to meaning and origin, so people and AI can work with the same records in a shared place.
A procedure is a set of instructions for producing a form or carrying out an operation.
This is the question developed in the [problem statement](PM_KR_PROBLEM_STATEMENT.md).

The first proposed step is a small “font file” repository for knowledge.
Each entry would connect a procedure for form, a SKOS concept for meaning, and PROV information for origin and time.
SKOS is a vocabulary for concepts and their relationships; PROV describes how information was produced.
This would complement RDF, OWL and JSON-LD, which already describe and exchange linked knowledge.
The immediate task is to define and test those connections, as proposed in [Daniel's public reply](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0016.html).

## User needs and use cases

These are proposed evaluation scenarios, not reported deployments or user-study results.

| Person and task | Need | What an evaluation should check |
| --- | --- | --- |
| A teacher and learners reviewing a diagram | Keep the drawing, explanation and AI reference tied to the same version | Change a fact and check that each view identifies the same revised record, including a nonvisual view |
| A developer composing small software modules | State what each module accepts and produces, and why modules can be joined | Exchange a small composition between two tools and detect incompatible boundaries |
| A researcher reviewing an explanation | Inspect the source, time and transformations behind a claim | Follow references back to a versioned source and distinguish an assertion from supporting evidence |
| A person and an AI inspecting an object in a spatial scene | Connect the visible object to what it means and where that knowledge came from | Resolve the same object identifier in both clients and report missing or conflicting metadata |

The module scenario comes from [Christoph's public account of his own work](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0015.html); it is an invitation to investigate common requirements, not a claim that he uses Knowledge3D.
The spatial scenario connects to [point 8 of Daniel's TPAC spatial-browser input](https://github.com/w3c-cg/pm-kr/blob/main/talks/2026-10-28-tpac-spatial-browser/README.md#8-the-knowledge-layer-twins-robots-and-provenance): a scene graph describes placement, while a knowledge layer would describe identity, behavior and origin.
His [Web of Worlds input](https://github.com/w3c-cg/pm-kr/blob/main/talks/2026-10-30-tpac-web-of-worlds/README.md) explores related node and shared-client questions.
Those pages are individual contributions; the latter is prepared for 30 October, not evidence of a completed session or agreement.

### User research and open questions

The public discussion supplies objections and candidate use cases; it does not establish that this approach improves a user's work.
Evaluation still needs participants, representative tasks and comparison with existing tools.
How small can the first useful vocabulary be?
Who decides that two entries mean the same thing?
At what point does decomposing an entry into smaller procedures stop helping?
What information survives exchange, and what must be reported as lost?

## Goals

- Give people and AI stable references to the same versioned knowledge, while allowing different presentations.
- Reuse procedures and concepts through explicit references, with origin and time available for inspection.
- Define a small vocabulary and exchange example that can be tested independently of the full spatial environment.
- Document translation losses and compare the proposed benefit with existing linked-data and module systems.

## Non-goals

- Replacing RDF, OWL, JSON-LD, font formats or existing knowledge graphs.
- Encoding all knowledge or physical reality before a useful first example exists.
- Requiring a 3D interface, a headset or Knowledge3D's GPU runtime for the first vocabulary report.
- Claiming general intelligence, automatic accessibility, compression gains or independent interoperability from this draft.

## Proposed approach

### The font precedent

A font combines reusable glyph descriptions with rules for selecting and positioning glyphs.
[OpenType](https://learn.microsoft.com/en-us/typography/opentype/spec/overview) separates character mapping, outlines and layout tables; GSUB substitutes glyphs and GPOS positions them.
[Unicode](https://www.unicode.org/standard/principles.html) identifies characters, which can have different rendered shapes.
The example demonstrates reusable units and context-dependent composition.
Character identity does not supply the meaning of every word, and font engineering does not prove the proposed knowledge system will scale.

Daniel's [clarification](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0018.html) treats fonts as a precedent for the format, not as a new procedural-font product.
A domain would choose useful primitives and compose them with a graph and rules.
The stopping point depends on the domain and remains a design question to test.

### One entry, explicit references

The [normative-model draft](PM_KR_NORMATIVE_MODEL.md) proposes records with identifiers, a layer, procedures where applicable, references, and metadata.
Its layers distinguish form, meaning, rules and rules for choosing rules.
Its canonicality requirement is scoped to a versioned namespace, which avoids assuming one universal interpretation of a concept.
The [interoperability guide](PM_KR_INTEROPERABILITY_GUIDE.md) explores mappings to existing linked-data formats and the information those mappings can lose.

The following is an illustrative JSON-LD sketch of a square diagram's record.
It shows the intended connections, not an agreed vocabulary, executable procedure, or conformance fixture.
The `ex:` terms and addresses are examples; the SKOS and PROV terms come from their published vocabularies.

```json
{
  "@context": {
    "ex": "https://example.org/knowledge/",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "prov": "http://www.w3.org/ns/prov#",
    "xsd": "http://www.w3.org/2001/XMLSchema#"
  },
  "@id": "ex:square-diagram-v1",
  "@type": "prov:Entity",
  "ex:form": { "@id": "ex:draw-square-v1" },
  "ex:meaning": {
    "@id": "ex:square",
    "@type": "skos:Concept",
    "skos:prefLabel": { "@value": "square", "@language": "en" }
  },
  "prov:wasDerivedFrom": { "@id": "ex:geometry-lesson-v1" },
  "prov:generatedAtTime": {
    "@value": "2026-10-28T12:00:00Z",
    "@type": "xsd:dateTime"
  }
}
```

A person could inspect the drawing and its source while an AI follows the same record's meaning reference.
This example does not define how to execute `draw-square-v1`, establish that the source is correct, or express every property of a mathematical square.
Those are separate requirements for a usable fixture.
The proposed test would check identity and version across clients, reference resolution, and declared losses after export and import.

### Human-readable and machine-readable forms

The [MR/HR draft](PM_KR_MR_HR_PARADIGM.md) explores a documented human-readable source and a generated machine-readable form, tied to the same version.
Behavioral equivalence needs a test; smaller source text does not by itself establish lower runtime memory use.
The broader [dual-client contract](../vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md) asks human and AI clients to consume the same underlying records.
That is a proposed contract, not evidence that their experiences or interpretations are equivalent.

### Stepping stones and evidence

A candidate sequence, assembled here from the public discussion and existing drafts, is:

1. Choose a narrow use case and write a vocabulary report defining its entries, references, origin and time.
2. Publish a small fixture, including an invalid reference and a version change, with expected results.
3. Have independent tools exchange it, checking shared identity and documenting lost information.
4. Measure whether that helps the chosen task before extending the vocabulary or standardizing an execution format.

This sequence is proposed for review, not an adopted group schedule.
The [conformance profiles](PM_KR_CONFORMANCE_PROFILES.md) distinguish a core model, a runtime profile and an audit profile.
Their requirement words belong to a project draft; they do not certify an implementation.
The [evidence matrix](PM_KR_EVIDENCE_VALIDATION_MATRIX.md) explains what artifacts a public implementation claim needs.

### Dependencies and unresolved choices

The first vocabulary discussion can use existing web standards.
The mapping between SKOS concepts, procedures and versioned records still needs agreement, as do procedure languages, execution boundaries, namespace ownership and equivalence rules.
Knowledge3D's broader GPU and spatial designs are separate implementation choices.
No new browser API is proposed here.
The [early X3D inputs](README.md#early-x3d-inputs) record an earlier route and do not establish an accepted X3D extension.

## Alternatives considered

| Alternative | What it offers | Tradeoff and current position |
| --- | --- | --- |
| RDF, OWL and JSON-LD with existing application code | Established identifiers, relationships, constraints and data exchange | This is the baseline to compare; the proposal must show why a shared procedure contract adds value |
| Content-addressed storage with links | Reuse of identical content and versioned references | Useful as infrastructure; byte identity alone does not determine conceptual equivalence |
| A knowledge graph connecting ordinary software modules | Explicit interfaces and composition without making every unit procedural | Directly relevant to Christoph's use case; remains an option for the first experiment |
| A complete spatial environment first | Lets people inspect objects and relationships in a shared scene | Adds presentation and runtime work before the vocabulary can be evaluated; the small repository is the proposed starting point |
| Separate records for each client | Lets each tool optimize its own representation | Requires explicit synchronization when overlapping facts change; may still be appropriate where meanings differ |

These are design comparisons, not benchmark findings.
The proposal does not rule out a mixed system or assume the procedural approach wins every task.

## Accessibility, internationalization, privacy and security

A common identifier does not supply a screen-reader description, tactile presentation or usable keyboard navigation.
Those need their own designs and tests with users, including ways to inspect knowledge without a spatial view.
Language-tagged labels need review for ambiguity and culturally different concepts; translation need not imply identical meaning.

Origin and time records can reveal personal information, so an implementation needs controls for access, retention and disclosure.
Provenance identifies a recorded source; it does not prove that the source is true or trustworthy.
Downloaded procedures require an explicit execution policy, resource bounds and handling of untrusted references.
Signatures and repeatable outputs cannot substitute for that review.
These questions are also raised by the [normative model](PM_KR_NORMATIVE_MODEL.md#9-security-and-privacy) and [accessibility design](../vocabulary/UNIVERSAL_ACCESSIBILITY_SPECIFICATION.md).

## Stakeholder feedback

**Christoph, 28 October:** his [first message](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0015.html) asks for incremental value, a clearer shared goal and evidence that procedural primitives can scale.
He points out that people and AI need different summaries, that knowledge models need a time dimension to represent learning and change, and that a general model of physical reality is too large a starting point.
In his [follow-up](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0017.html), he asks why a font example helps module composition, where decomposition should stop, and how processing units differ from the context and purpose that organize them.
These are substantive objections to the proposed route, not an endorsement.

**Daniel's answers:** the [first reply](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0016.html) proposes the small repository, with SKOS and PROV, alongside existing semantic-web standards.
It states that the code is paused and invites Christoph to contribute his use case.
The [second reply](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0018.html) explains that OpenType combines units with composition rules and that each domain must choose a useful primitive boundary.
These replies clarify the intended starting point; they do not establish scalability or resolve the request for measured benefits.

This explainer cites no organizational endorsement, adoption count or independent conformance result.
Further feedback should be recorded with the speaker's actual position and a public link.

## References and acknowledgements

The [document index](README.md) covers the project drafts used here, including the strategic-steering input and superseded papers.
Thanks to Christoph for the public questions summarized above.
Earlier contributors and technical influences remain credited in [Attributions](../../ATTRIBUTIONS.md).

- [JSON-LD 1.1](https://www.w3.org/TR/json-ld11/), a serialization for linked data.
- [OpenType overview](https://learn.microsoft.com/en-us/typography/opentype/spec/overview), glyph outlines, character mapping and layout.
- [OWL 2 overview](https://www.w3.org/TR/owl2-overview/), the web ontology language.
- [PROV-O](https://www.w3.org/TR/prov-o/), provenance entities, activities and relationships.
- [RDF 1.1 concepts](https://www.w3.org/TR/rdf11-concepts/), graph-based data and identifiers.
- [SKOS reference](https://www.w3.org/TR/skos-reference/), concepts, labels and relationships.
- [TAG explainer template](https://github.com/w3ctag/explainer-explainer/blob/main/template.md), the shape followed by this document.
- [Unicode principles](https://www.unicode.org/standard/principles.html), characters and their representation.

**Licence:** CC BY 4.0, consistent with the project specification drafts.
