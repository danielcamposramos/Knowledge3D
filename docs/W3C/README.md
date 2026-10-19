# Knowledge3D contributions to W3C-related discussion

Knowledge3D develops a spatial AI architecture and procedural knowledge implementation. This folder contains project proposals, discussion inputs and evidence inventories relevant to the [PM-KR Community Group](https://www.w3.org/community/pm-kr/). Hosting a document here does not make it a W3C publication or a group decision. The [PM-KR repository](https://github.com/w3c-cg/pm-kr) is the venue for collaborative incubation; alternative implementations are welcome.

## Read in this order

| Question | Starting point | How to read it |
| --- | --- | --- |
| What user problem motivates the work? | [Project overview](../../README.md), [problem statement](PM_KR_PROBLEM_STATEMENT.md) | Goals and candidate scenarios; evaluate the proposed benefit |
| What is proposed for exchange? | [Normative model](PM_KR_NORMATIVE_MODEL.md), [conformance profiles](PM_KR_CONFORMANCE_PROFILES.md) | Candidate requirements, not adopted W3C conformance rules |
| How might it connect to existing systems? | [Interoperability guide](PM_KR_INTEROPERABILITY_GUIDE.md), [ecosystem mapping](pmkr-w3c-ecosystem-mapping.md) | Proposed mappings and relationships; require actual exchange tests |
| What evidence exists? | [Evidence matrix](PM_KR_EVIDENCE_VALIDATION_MATRIX.md), [published implementation record](../../CODEX.md#current-benchmark-state) | Dated observations, known failures and targets; follow artifact references |
| What connects spatial knowledge to screens? | [PM-KR reports](https://github.com/w3c-cg/pm-kr/tree/main/reports), [individual X3D inputs](x3d/) | Presentation and interoperability proposals with their own status statements |
| Where do I discuss or challenge a claim? | [K3D issues](https://github.com/danielcamposramos/Knowledge3D/issues), [PM-KR issues](https://github.com/w3c-cg/pm-kr/issues) | Implementation questions in K3D; candidate shared requirements in PM-KR |

Other dated papers, charters, mission statements and objectives in this corpus retain their historical context. Words such as “final”, “standard” or “normative” in a filename are not evidence of group adoption. Cite a recorded decision when describing adopted group work.

## Knowledge, presentation and editorial tooling

Knowledge3D explores shared knowledge identity, procedures, provenance and human/AI interaction. [Sparky Stereo OS](https://github.com/danielcamposramos/sparky-stereo-os) supplies a desktop presentation path and device experience relevant to spatial interfaces. Stereo and HDR both require preserving descriptive information through the stack; stereo observations do not settle the HDR engineering choice. Two-dimensional and accessible interaction remain necessary alongside immersive views.

[respec-mcp](https://github.com/danielcamposramos/respec-mcp) provides a different part of the collaboration: human-defined document rules, ReSpec checks and W3C document guidance. Its CI results can make editorial review easier to inspect. They do not test Knowledge3D's GPU runtime, accessibility, semantic equivalence or standards conformance.

## Evidence for public claims

For an implementation claim, record the public code revision, command, dataset or fixture, environment, result artifact and known limits. Separate:

- Reproducible results with publicly available tests and inputs.
- Observations supported by retained run logs, with availability stated.
- Illustrative examples, architectural requirements and future targets.

Report failed and untested cases alongside successes. A small curated dataset is not a full benchmark score; one implementation is not independent interoperability evidence. An execution trace can show operations without proving an answer correct. Private logs and uncommitted development do not become public evidence through a summary.

## Writing and review

Use [respec-mcp's document skills](https://github.com/danielcamposramos/respec-mcp/blob/main/docs/W3C_SKILLS.md) for the artifact being prepared: explainer guidance for user problems and alternatives, CG/report guidance for publication status, implementation-report guidance for evidence, and review skills for accessibility, internationalization, privacy and security. Charters, meeting records, workshops and presentations have their own guidance. Preserve historical correspondence and minutes.

Authors may explain ideas in their preferred language and work with AI partners on an English draft. **Se for mais fácil para você, pode explicar em sua própria língua.** Preserve technical meaning, uncertainty, attribution and requirement strength. Language assistance does not transfer authorship decisions to a validator.

This repository's Markdown material receives editorial review; the ReSpec checker applies to ReSpec HTML. The [PM-KR report workflow](https://github.com/w3c-cg/pm-kr/tree/main/reports) provides practical automated examples. Claims of group consensus, review completion or organizational endorsement still need their own public records.
