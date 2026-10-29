# PM-KR × W3C Ecosystem Mapping

**For standards readers:** find the existing specifications relevant to the proposed knowledge format and the questions an exchange test would need to answer.

**Status:** individual discussion input, revised 29 October 2026.
The earlier outreach and priority matrix is superseded by this reference map.
The links identify related work; they do not establish partnerships, adoption or agreement by the organizations named.
Knowledge3D's code is paused while the work focuses on specifications and PM-KR deliverables, as Daniel states in his [28 October public reply](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0016.html).
Start with the [explainer](EXPLAINER.md) for the proposal and its open questions.

**Original research credit:** EchoSystems AI Studios; Claude Sonnet 4.6, 11 March 2026.

## Table of Contents

- [T1 · Direct AI / KR Overlap](#t1--direct-ai--kr-overlap)
- [T2 · 3D, Spatial & Immersive Web](#t2--3d-spatial--immersive-web)
- [T3 · Data, Semantics & Linked Data](#t3--data-semantics--linked-data)
- [T4 · Web Architecture & Runtime](#t4--web-architecture--runtime)
- [T5 · Accessibility, i18n & Human Interface](#t5--accessibility-i18n--human-interface)
- [T6 · Industry Verticals](#t6--industry-verticals)
- [T7 · Governance, Process & Standards Infrastructure](#t7--governance-process--standards-infrastructure)
- [Σ · Strategic Priority Matrix](#σ--strategic-priority-matrix)

<a id="t1"></a>

## T1 · Direct AI / KR Overlap

The [PM-KR Community Group](https://www.w3.org/community/pm-kr/) is the discussion venue for procedural memory knowledge representation.
Knowledge3D's documents are individual project inputs; the [group repository](https://github.com/w3c-cg/pm-kr) carries the group's own work and status statements.
Daniel's proposed stepping stone is a small repository linking a procedure for form, a term for meaning and a record of origin and time.
His [font clarification](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0018.html) explains why composition rules matter alongside reusable units.

<a id="t2"></a>

## T2 · 3D, Spatial & Immersive Web

| Reference | Question for this proposal |
| --- | --- |
| [X3D 4.0 architecture](https://www.web3d.org/documents/specifications/19775-1/V4.0/Part01/Architecture.html) | Which knowledge fields could accompany a scene without being mistaken for registered X3D nodes? |
| [glTF 2.0 specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html) | Which references and provenance survive asset export and import? |
| [WebXR Device API](https://www.w3.org/TR/webxr/) | How could a spatial client expose the same knowledge entry as another client? |

The [early X3D inputs](README.md) record an approach that did not progress to a group specification.
Daniel's TPAC [spatial browser point 8](https://github.com/w3c-cg/pm-kr/blob/main/talks/2026-10-28-tpac-spatial-browser/README.md#8-the-knowledge-layer-twins-robots-and-provenance) distinguishes the knowledge layer from scene placement.
That participant input does not establish a browser integration or a joint specification.

<a id="t3"></a>

## T3 · Data, Semantics & Linked Data

| Reference | Role to explore |
| --- | --- |
| [RDF 1.1 Concepts](https://www.w3.org/TR/rdf11-concepts/) | Identify entries and express relationships as a graph. |
| [OWL 2 Overview](https://www.w3.org/TR/owl2-overview/) | State ontology relationships and constraints. |
| [JSON-LD 1.1](https://www.w3.org/TR/json-ld11/) | Exchange linked data using JSON. |
| [SKOS Reference](https://www.w3.org/TR/skos-reference/) | Name concepts and connect labels and related terms. |
| [PROV-O](https://www.w3.org/TR/prov-o/) | Describe sources, activities and time. |

The [interoperability guide](PM_KR_INTEROPERABILITY_GUIDE.md) proposes mappings to review.
A mapping needs round-trip examples and a description of information loss before it can support an interoperability claim.

<a id="t4"></a>

## T4 · Web Architecture & Runtime

[WebGPU](https://www.w3.org/TR/webgpu/) and [WebAssembly](https://www.w3.org/TR/wasm-core-2/) are references for evaluating possible browser execution paths.
This map supplies no evidence of a K3D port to either.
An execution proposal needs to define its language, resource limits, permissions and behavior when a client cannot execute a procedure.
The [TAG explainer guidance](https://w3ctag.github.io/explainer-explainer/) helps frame those choices around user needs and alternatives.

<a id="t5"></a>

## T5 · Accessibility, i18n & Human Interface

[XR Accessibility User Requirements](https://www.w3.org/TR/xaur/) describes needs to consider for spatial interfaces.
It is a Working Group Note, not certification of this design.
Sharing a knowledge identifier does not prove that visual, speech, Braille and other presentations communicate the same information.
Each presentation requires implementation and review with users.
Language and writing-system questions belong in the model from the start; the [W3C internationalization review checklist](https://www.w3.org/TR/international-specs/) is a review aid.

<a id="t6"></a>

## T6 · Industry Verticals

The [explainer's use cases](EXPLAINER.md#user-needs-and-use-cases) are scenarios to test.
An education, engineering or media example is not evidence of an adopter or a deployment.
A useful contribution describes a task, the existing workflow, a small exchange fixture and the result that would demonstrate value.

<a id="t7"></a>

## T7 · Governance, Process & Standards Infrastructure

Use W3C's own [Community Group process](https://www.w3.org/community/about/process/) and [policy summary](https://www.w3.org/community/about/process/summary/) for participation and publication requirements.
This map makes no prediction about a Working Group charter or a Recommendation.
Publication status, licensing and evidence of group agreement must be stated for each actual document.

<a id="ts"></a>

## Σ · Strategic Priority Matrix

The earlier schedule of outreach to named people is retired.
Daniel's [public next step](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0016.html) is the vocabulary report and a small, testable repository.
Contribute a use case or review question through the [PM-KR group](https://www.w3.org/community/pm-kr/).
