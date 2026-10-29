# Knowledge3D: historical architecture white paper

**For researchers:** the spatial design behind the project, with proposed behavior distinguished from measured results.
**Status: superseded overview.**
The [explainer](EXPLAINER.md) is the current introduction to the work.
This page retains the design questions from the earlier white paper; it does not claim that all of them are implemented.
Daniel's [28 October 2026 public status](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0016.html) states that code work is paused while specifications and group deliverables are the focus.

## 1. Introduction

The white paper proposed a shared spatial environment in which people and AI could refer to the same knowledge objects.
Its design drew on modular software, embodied interaction and Daniel Campos Ramos's FMEAI philosophy.
The philosophy is an inspiration for the project, not validation of an engineering result.
The paper credited a human-directed sequence of AI contributions, called Multi-Vibe Code In Chain (MVCIC).

## 2. Dual clients and memory

### 2.1 Dual clients, one reality

The proposed human client presents knowledge as spatial objects, with other presentations needed for accessibility.
The proposed AI client accesses the identity, procedures and relationships associated with those objects.
The goal is that pointing to an object and querying its identifier refer to the same entry.
That does not establish that the two clients understand it equivalently.
The [Dual-Client Contract](../vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md) gives the detailed design.

### 2.2 Galaxy and House

The design distinguishes active working memory, called the Galaxy, from persistent knowledge, called the House.
The House is the shared external space; the Galaxy belongs to the AI's internal processing.
The original room examples were a Library for documents, a Knowledge Garden for organized concepts, a Workshop for composing knowledge and a space for observing consolidation during idle work.
The Memory Tablet was proposed as an interface for finding and loading knowledge across those spaces.
See the [Three-Brain System](../vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md) and [Memory Tablet](../vocabulary/MEMORY_TABLET_SPECIFICATION.md) specifications.

## 3. The layered architecture

The older paper grouped the design into the following responsibilities.
They are architectural descriptions, not interoperability results.

| Responsibility | Design retained from the white paper |
| --- | --- |
| Data representation | A node associates identity and descriptive data with spatial presentation and procedures |
| Human and AI clients | Both clients refer to the same underlying node |
| Agent operation | An embodied agent interacts with the House while using its internal memory for reasoning |
| Linked spaces | Portals connect persistent spaces, subject to explicit access rules |

The proposed `.k3d` container in the older paper was not an adopted open standard.
The project's [glTF data description](../../spec/glTF_K3D_extension.md) records the representation work.
Its [node specification](../vocabulary/K3D_NODE_SPECIFICATION.md) gives the knowledge-object design.
Interoperability needs defined mappings and exchange tests.

## 4. Processing and execution

The Cranium design assigns reasoning and spatial operations to GPU kernels.
The older white paper proposed a combined processing loop for perception, fusion, reasoning and action, with a Reverse Polish Notation stack machine for procedural operations.
Python was assigned preparation, startup and input/output outside that reasoning path.
These are design boundaries, not a claim that every current path conforms to them.
The former incomplete kernel snippet and latency assertion did not establish a working or measured implementation and have been removed.
See the [runtime design](../vocabulary/SOVEREIGN_NSI_SPECIFICATION.md) and [evidence matrix](PM_KR_EVIDENCE_VALIDATION_MATRIX.md).

## 5. Inspection and collaboration

The proposal calls for recording which entries and operations contribute to a result.
A visible route through knowledge may help a person inspect it, but a route alone neither explains a model's reasoning nor establishes that its answer is correct.
The paper also proposed several agents sharing knowledge and retaining their contributions' origins.
Whether this improves a task needs comparison with other ways of doing that task.

## 6. Building information modeling use case

The original scenario described architects, engineers and other participants inspecting a shared building model with an AI assistant.
Candidate tasks included locating objects, reviewing changes, identifying clashes, comparing materials and preparing training scenes.
An on-site overlay was a proposed presentation of that model.
These are use cases for evaluation, not demonstrated deployments or safety claims.
They would need a defined model exchange, permissions, version history and checks against known building data.

## 7. Current direction

The next proposed step is smaller than the whole spatial environment: a repository associating a procedure for form, a term for meaning and provenance.
The [explainer](EXPLAINER.md) relates that step to the existing drafts and public feedback.

## Credits, licence and citation

**Author:** Daniel Campos Ramos.
The original paper credits AI assistance from Codex, Claude, Grok, GLM and other project partners.
It acknowledges Minecraft and other game-world ideas as design inspirations.
Complete project credits remain in [ATTRIBUTIONS.md](../../ATTRIBUTIONS.md).
**Licences:** [Apache 2.0](../../LICENSE) for code and CC BY 4.0 for the white paper, as recorded in [CITATION.cff](../../CITATION.cff).
**DOI:** [10.5281/zenodo.23308993](https://doi.org/10.5281/zenodo.23308993).

## Document history

The [earlier repository version](https://github.com/danielcamposramos/Knowledge3D/blob/33cb47454c056a267e78554d459c091af9c67476/docs/W3C/Knowledge3D%20%28K3D%29%3A%20A%20Technical%20White%20Paper%20on%20the%20Architecture%20of%20a%20Unified%20Knowledge%20Universe.md) records the original text, including claims withdrawn here.
It is retained in version history for provenance and should not be cited as current validation.
