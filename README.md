# Knowledge3D

Keep a lesson, diagram or model in one place so people and AI can refer to the same knowledge.
Knowledge3D explores shared knowledge in 3D spaces for researchers, developers and curious readers, including people without a technical background.

**Status:** the code is paused.
Work now focuses on the [specifications](docs/W3C/README.md) and reports for [PM-KR](https://www.w3.org/community/pm-kr/), a W3C community group studying how people and AI share knowledge.
These project drafts are proposals for discussion, not adopted W3C standards.

[![Proposed design: a person and AI read the same lesson, with its drawing, meaning and source.](docs/images/shared-knowledge.svg)](docs/W3C/EXPLAINER.md)

**Start here:** read the [explainer](docs/W3C/EXPLAINER.md) for the problem, the proposed first step and the questions still open.

<a id="knowledge3d--spatial-ai-and-procedural-knowledge-research"></a>
<a id="participate"></a>
<a id="the-user-facing-problem"></a>
<a id="proposed-approach"></a>
<a id="conceptual-data-example"></a>
<a id="use-cases"></a>
<a id="1-inclusive-education"></a>
<a id="2-explainable-ai"></a>
<a id="3-knowledge-deduplication"></a>
<a id="4-multi-modal-accessibility"></a>
<a id="non-goals"></a>
<a id="architecture-overview"></a>
<a id="three-brain-system-neuroscience-inspired"></a>
<a id="sovereignty-zero-external-dependencies-in-hot-path"></a>
<a id="key-specifications"></a>
<a id="alternatives-considered"></a>
<a id="why-not-traditional-knowledge-graphs-rdfowl"></a>
<a id="why-not-llm-embeddings"></a>
<a id="why-not-existing-game-engines-unityunreal"></a>
<a id="why-not-framework-based-ai-pytorchtensorflow"></a>
<a id="why-build-the-house-as-literal-3d-space"></a>
<a id="benchmark-results"></a>
<a id="published-evidence-and-its-limits"></a>
<a id="historical-results"></a>
<a id="getting-started"></a>
<a id="prerequisites"></a>
<a id="install"></a>
<a id="runtime-workspace"></a>
<a id="launch"></a>
<a id="troubleshooting"></a>
<a id="scientific-reproduction"></a>
<a id="citation"></a>
<a id="accessibility-internationalization-privacy-and-security-considerations"></a>
<a id="accessibility"></a>
<a id="internationalization"></a>
<a id="privacy"></a>
<a id="security"></a>
<a id="video-presentations"></a>
<a id="stakeholder-feedback"></a>
<a id="contributing"></a>
<a id="references--acknowledgments"></a>
<a id="scientific-reproduction-week-219-results"></a>
<a id="-week-219-validation--architecture-breakthrough-february-10-2026"></a>
<a id="week-17-multi-curriculum-benchmarks-february-2026"></a>

## From form to meaning

The idea is to store knowledge once as procedures, instructions that describe how to make or do something, with people and AI reading it in the same place.

A font file is the working precedent.
Glyphs describe the shapes of letters; OpenType shaping rules combine and position them; Unicode identifies the characters those shapes represent.
The person sees a letter and software can identify it.
Character identity is a starting point for meaning; a font does not explain a whole concept.

The next proposed deliverable is a small “font file” repository for knowledge: a procedure for form, a SKOS term for meaning, and PROV records for origin and time.
SKOS names concepts and their relationships; PROV records where information came from.
This would sit alongside RDF, OWL and JSON-LD, the web's ways to describe and exchange linked knowledge.
Daniel's [28 October proposal](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0016.html) and [clarification](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0018.html) explain the font precedent and why each subject needs a useful stopping point for breaking knowledge into parts.

## Find your way

- **Standards:** the [document index](docs/W3C/README.md) labels drafts, discussion inputs and superseded texts.
- **Researchers:** the [use cases and open questions](docs/W3C/EXPLAINER.md#user-needs-and-use-cases) identify what needs testing.
- **Implementers:** the [proposed approach](docs/W3C/EXPLAINER.md#proposed-approach) connects the model, conformance drafts and exchange guide.
- **Take part:** the [PM-KR Community Group](https://www.w3.org/community/pm-kr/) is open to anyone with a W3C account, subject to its participation terms.

## Credits, licence and citation

Daniel Campos Ramos directs Knowledge3D, with AI assistance from Claude, Codex and the contributors credited in [Attributions](ATTRIBUTIONS.md).
The [project's influences](docs/POP_CULTURE_HERITAGE.md) are recorded separately.
Code is under [Apache 2.0](LICENSE); the white paper is under CC BY 4.0.
Cite the project using [CITATION.cff](CITATION.cff) and [DOI: 10.5281/zenodo.23308993](https://doi.org/10.5281/zenodo.23308993).
