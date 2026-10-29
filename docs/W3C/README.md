# Knowledge3D specifications and discussion inputs

**For standards contributors, researchers and implementers:** start with the [explainer](EXPLAINER.md) to understand the problem and proposed first step.

**Status:** the code is paused; work now focuses on these specifications and the [PM-KR Community Group's](https://www.w3.org/community/pm-kr/) deliverables.
Daniel stated this direction in his [28 October reply](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0016.html).
These are Knowledge3D contributions, not adopted W3C standards or evidence of group consensus.
The [group repository](https://github.com/w3c-cg/pm-kr) carries its own publication and participation records.

**Draft** means a project proposal under discussion.
**Input** means background or an individual contribution, without adoption implied.
**Superseded** means an earlier text retained as a historical pointer; use the linked replacement for the current direction.
Words such as “final” and “normative” in filenames do not change those statuses.

## Read in this order

| Document | Status | What it covers |
| --- | --- | --- |
| [Explainer](EXPLAINER.md) | Draft | User needs, the font precedent, the small repository proposal and public objections |
| [Problem statement](PM_KR_PROBLEM_STATEMENT.md) | Draft | Why separate knowledge representations can drift and what a shared source should address |
| [Normative model](PM_KR_NORMATIVE_MODEL.md) | Draft | Proposed records, references, layers and invariants |
| [Conformance profiles](PM_KR_CONFORMANCE_PROFILES.md) | Draft | Candidate core, runtime and audit requirements for implementers |
| [Interoperability guide](PM_KR_INTEROPERABILITY_GUIDE.md) | Draft | Proposed exchanges with RDF, OWL and JSON-LD, including translation losses |
| [Human-readable and machine-readable forms](PM_KR_MR_HR_PARADIGM.md) | Input | Linking documented source to generated forms and testing their equivalence |
| [Evidence matrix](PM_KR_EVIDENCE_VALIDATION_MATRIX.md) | Draft | Evidence needed to support implementation and conformance claims |
| [Strategic steering](PM_KR_STRATEGIC_STEERING.md) | Superseded | Historical project roadmap retained as input to current planning |

## Other documents

| Document | Status | What it covers |
| --- | --- | --- |
| [Hyper-modular definition](HYPER_MODULAR_DEFINITION.md) | Input | Composition and reuse through references |
| [Standards alignment](K3D_W3C_STANDARDS_ALIGNMENT.md) | Input | Relationships to existing web standards |
| [Procedural representation proposal](PROCEDURAL_KNOWLEDGE_REPRESENTATION_STANDARD.md) | Input | An earlier procedural knowledge format direction |
| [W3C ecosystem mapping](pmkr-w3c-ecosystem-mapping.md) | Input | Relevant standards and venues, with no partnership implied |
| [Ecosystem mapping HTML](pmkr-w3c-ecosystem-mapping.html) | Input | Browser presentation of the same reference map |
| [Technical white paper](Knowledge3D%20%28K3D%29%3A%20A%20Technical%20White%20Paper%20on%20the%20Architecture%20of%20a%20Unified%20Knowledge%20Universe.md) | Superseded | Earlier architecture narrative, with its historical source and credits |
| [State-of-the-art comparison](K3D_VS_STATE_OF_THE_ART_2026.md) | Superseded | Historical comparison replaced by scoped evaluation questions |
| [Datacenter impact analysis](PM-KR_Datacenter_Impact_Analysis.md) | Superseded | Historical impact assumptions; no measured deployment benefit established |
| [Datacenter PDF notice](PM-KR_Datacenter_Impact_Analysis.pdf) | Superseded | Withdrawal notice replacing the stale quantitative PDF at its original address |
| [AI knowledge representation report](W3C_AI_KR_Report_Final.md) | Superseded | Earlier outreach and research framing |
| [Community Group charter proposal](W3C_CG_CHARTER_PMKR.md) | Superseded | Historical launch proposal, not the current group's governance record |
| [Community Group proposal text](W3C_CG_FINAL_PROPOSAL_TEXT.md) | Superseded | Historical formation text, not evidence of adoption |
| [MVCIC proposal](FINAL_MVCIC_W3C_PROPOSAL.md) | Superseded | Earlier AI-assisted standards-work proposal |
| [Hyper-modular encyclopedia draft](WIKIPEDIA_HYPER_MODULAR_ARCHITECTURE.md) | Superseded | Earlier advocacy draft, not an encyclopedia publication |

## Early X3D inputs

These documents retain their status as an early non-fruitful attempt and individual input.
They do not establish extensions accepted by W3C or the Web3D Consortium.

| Document | Status | What it covers |
| --- | --- | --- |
| [Ontology](x3d/PM_KR_X3D_ONTOLOGY.md) | Input | Proposed procedural knowledge additions to the X3D ontology |
| [Procedural memory component](x3d/PM_KR_X3D_PROCEDURAL_MEMORY_COMPONENT.md) | Input | Proposed scene nodes for procedures and references |
| [Adaptive reasoning component](x3d/PM_KR_X3D_ADAPTIVE_REASONING_COMPONENT.md) | Input | Proposed scene-level reasoning behavior |
| [Avatar specification](x3d/PM_KR_X3D_AVATAR_SPECIFICATION.md) | Input | Earlier avatar and embodiment design |
| [RPN scene emission](x3d/PM_KR_RPN_X3D_SCENE_EMISSION.md) | Input | Producing scenes from Reverse Polish Notation procedures |

## Knowledge, presentation and editorial tooling

Daniel's individual TPAC 2026 inputs discuss [the spatial browser's knowledge layer, point 8](https://github.com/w3c-cg/pm-kr/blob/main/talks/2026-10-28-tpac-spatial-browser/README.md#8-the-knowledge-layer-twins-robots-and-provenance) and [shared knowledge in a Web of Worlds](https://github.com/w3c-cg/pm-kr/blob/main/talks/2026-10-30-tpac-web-of-worlds/README.md).
They supply discussion context, not session outcomes or endorsements.
[Sparky Stereo OS](https://github.com/Sparky-OS/sparky-stereo-os) concerns desktop presentation; [respec-mcp](https://github.com/danielcamposramos/respec-mcp) concerns document tooling.
Neither establishes the proposed knowledge model's interoperability.

## Evidence for public claims

A result needs a public revision, command, input, environment, output artifact and stated limits.
Distinguish design requirements, illustrative examples, dated observations and independently reproduced results.
A local path or unpublished log is not a public source.
Participation in a discussion does not establish adoption, partnership or endorsement.
The [evidence matrix](PM_KR_EVIDENCE_VALIDATION_MATRIX.md) develops those checks.

## Writing and review

Follow the [TAG explainer guidance](https://github.com/w3ctag/explainer-explainer) for user needs and alternatives, and the [document skills](https://github.com/danielcamposramos/respec-mcp/blob/main/docs/W3C_SKILLS.md) for the kind of publication being prepared.
State the person's problem before the architecture, explain unfamiliar terms, and put detail behind a descriptive link.
Keep one sentence per line and distinguish what is proposed from what has been demonstrated.
Authors may explain ideas in their preferred language and use AI assistance for an English draft, preserving meaning, uncertainty and attribution.
Document checks do not establish runtime correctness, accessibility, semantic equivalence or consensus.
