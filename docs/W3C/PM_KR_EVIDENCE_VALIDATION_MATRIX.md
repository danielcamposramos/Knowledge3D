# PM-KR Evidence and Validation Matrix

**For reviewers:** this draft defines the evidence needed to test the proposed model.
It reports no current benchmark score, conformance certification or production deployment.
Knowledge3D code is paused while work focuses on specifications and PM-KR deliverables ([Daniel, 28 October 2026](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0016.html)).

**Document type:** Project evidence plan offered for PM-KR discussion.
**Authors:** Knowledge3D Project Contributors.
**Status:** Draft, revised 29 October 2026.
**Earlier version:** 1.2, dated 20 February 2026.

The earlier result tables were withdrawn because their claimed measurements were not tied to sufficient public input manifests, exact commands and output artifacts.
Repository source and dated development notes are useful leads, but they do not establish the results formerly listed here.
This revision reruns no benchmark.

## Abstract

This matrix connects proposed requirements to evaluation methods and publication requirements.
It distinguishes source inspection, execution results and independent reproduction.

## 1. Validation Methodology

Use the [normative model](PM_KR_NORMATIVE_MODEL.md) and [conformance profiles](PM_KR_CONFORMANCE_PROFILES.md) as draft requirements.
Record the profile revision because these documents remain proposals.

### 1.1 Evidence Standards and Maturity

**Source available** means a file exists at a cited public revision.
**Run observed** means the command, environment, inputs and raw output are retained.
**Independently reproduced** means another evaluator ran the published procedure and reported the result.
**Proposed** means the work has not been established by such evidence.
A test file existing in the repository is not a passing test result.

### 1.2 K3D Reference Implementation

The [source repository](https://github.com/danielcamposramos/Knowledge3D) contains implementation material.
Examples for source inspection include [procedural-font tests](../../tests/test_procedural_fonts.py), [hot-path checks](../../tests/test_hot_path_sovereignty.py) and [Knowledgeverse integration tests](../../tests/test_knowledgeverse_integration.py).
These links establish available source, not execution, coverage or conformance.

## 2. Core Claims Validation

The following are evaluation questions.
No row supplies a measured result.

<a id="21-claim-70-compression-via-symlink-composition"></a>

### 2.1 Reference Reuse and Compression

Compare identical content in the baseline and procedural representations.
Count procedures, references, metadata, indexes, caches and reconstruction data.
Record any lost information.
A reference-preservation rule alone implies no fixed compression percentage.

### 2.2 Claim: Deterministic Reconstruction

Fix input data, seeds, procedure versions, runtime versions and declared numerical tolerances.
Repeat the reconstruction and compare the resulting artifacts.
Include a changed-input control so a constant output cannot pass as meaningful reconstruction.

### 2.3 Claim: Dual-Client Equivalence

Compare identity, version, provenance and task-relevant meaning across human and machine views of the same entry.
A shared identifier alone does not prove equivalent interpretation or accessibility.
Visual, non-spatial and assistive-technology paths require separate evaluation.

### 2.4 Claim: Sovereign Hot Path (Zero External Dependencies)

Trace the complete declared execution boundary, including failures and preloaded dependencies.
An import scan cannot detect a dependency that was already loaded.
A solved answer does not prove a GPU kernel ran.
Include controls that intentionally introduce a forbidden operation and confirm detection.

## 3. Performance Benchmarks

Measure performance only after checking behavior against a reference.
Report workload, warm-up, repetitions, synchronization and measurement boundaries.

### 3.1 Execution Latency

Publish raw timings and summary statistics for each operation.
Separate host dispatch, transfers, GPU completion and end-to-end latency.

### 3.2 Throughput

Report completed work, elapsed time, concurrency and failure counts.
State whether execution is serialized or overlapped; a query-latency result is not automatically a throughput result.

### 3.3 Memory Efficiency

Measure allocated and used memory separately.
Include runtime state, indexes, temporary buffers and caches.
A source-file byte count is not a process-memory or GPU-memory measurement.

## 4. Production Validation

No production deployment or readiness is established here.
The following are candidate evaluations.

### 4.1 Math Benchmark Accuracy

Publish the exact task manifest, scoring rules, answers, failures and evaluation code.
Distinguish curated examples, training material and held-out evaluation.
Include all attempted tasks in the denominator.

### 4.2 ARC-AGI Visual Reasoning

Identify the dataset version, split, task identifiers, allowed attempts and scoring method.
Retain outputs and failure cases.
A selected level or subset must not be presented as a full-benchmark score.

### 4.3 Character Galaxy Multilingual Support

Publish font and character coverage, shaping cases, language context and expected output.
Compare glyph reconstruction and semantic identity separately.
Passing visual comparisons does not establish accessible equivalence or language coverage.

## 5. Conformance Level Validation

The A/B/C levels are project proposals.
This document certifies none of them.

### 5.1 Level A: PM-KR Core

Evaluate the data model, canonical references, reconstruction and declared composition rules against the selected draft revision.
Publish fixture-level outcomes, including rejected malformed inputs.

### 5.2 Level B: PM-KR Sovereign Runtime

Include the Level A artifacts and execution traces for the proposed runtime boundary.
Do not infer isolation from a success counter.

### 5.3 Level C: PM-KR Auditable Production

Include the lower-level artifacts, provenance export, audit trail, metrics and publisher-signature checks.
Checksums and signatures serve different purposes and require separate validation.

## 6. Integration Validation

Integration evidence must exercise the interacting components, not merely instantiate them.

### 6.1 Knowledgeverse Integration

The [integration test source](../../tests/test_knowledgeverse_integration.py) is available for inspection.
A report must state the revision, executed cases, results and gaps before claiming integration behavior.

### 6.2 Shadow Copy Learning

A learning claim needs before/after behavior on a declared evaluation set, retained state and an ablation or control.
An event in an audit log alone does not establish improved behavior.

## 7. Scalability Validation

Do not extrapolate a small fixture to web-scale deployment without measurements and a stated model.

### 7.1 Node Count Scaling

Vary graph size, fan-out, reference depth, cache conditions and query distribution.
Publish resource use and error behavior at each tested point.

### 7.2 Overnight PDF Ingestion

Publish a corpus manifest, licensing/access conditions, extraction outputs, elapsed time and failures.
A smaller derived graph is not lossless compression of the original PDFs unless reconstruction is demonstrated.

## 8. Security and Auditability

Inspectability and provenance support review; they do not establish safe execution.

### 8.1 Provenance Chain Integrity

Check that every claimed source and transformation resolves to the intended version.
Test altered, missing and unauthorized records.
A recorded source is an attribution, not proof that the source is correct.

### 8.2 Sovereignty Enforcement

Review [hot-path tests](../../tests/test_hot_path_sovereignty.py) against the exact execution boundary.
Instrument actual calls and verify that intentional violations fail.
This matrix supplies no enforcement verdict.

## 9. Evidence Summary

Public source files are available; the previous measurements and conformance verdicts are not retained as verified results.

### 9.1 Claims vs Evidence Matrix

| Topic | Available input | Evidence still required |
| --- | --- | --- |
| Reference reuse | Draft invariants and examples | Complete serialized inputs/outputs and reconstruction checks |
| Determinism | Proposed fixed-runtime requirement | Repeated runs and negative controls |
| Human/AI consistency | Dual-client design | Shared identity/version tests and user-facing validation |
| Runtime isolation | Source checks | Full execution traces and violation controls |
| Performance and scale | Candidate workloads | Reproducible measurements with failures and limits |

### 9.2 Conformance Summary

No independent A/B/C conformance assessment is established here.
Use the draft profiles to prepare an evaluation, not as a badge of implementation status.

## 10. Evidence Publication Plan

This is a proposed artifact contract for discussion.

### 10.1 Purpose and Scope

Make each result inspectable and repeatable without relying on private logs or the author's account.

### 10.2 Level A: PM-KR Core Artifacts

Publish the schema/profile revision, fixture manifest, test code, commands, expected behavior and observed outcomes.
Include malformed references and reconstruction failures.

### 10.3 Level B: PM-KR Sovereign Runtime Artifacts

Add runtime and driver versions, allowed/forbidden operation policy, telemetry, traces and failure controls.

### 10.4 Level C: PM-KR Auditable Production Artifacts

Add provenance and audit exports, measured representation sizes, signer/key information and independent verification instructions.
Use a reviewed signing scheme; a bare digest does not authenticate a publisher.

### 10.5 File Naming Conventions

Use names that identify implementation revision, profile revision, fixture and run.
A manifest must map each filename to its content digest and role.
Suggested filenames are not evidence that an artifact exists.

### 10.6 Minimal Rerun Instructions

The earlier shell examples referred to an unpublished conformance package and are withdrawn as runnable instructions.
A published package must supply exact commands, dependencies, fixtures, expected outputs and failure behavior.
Until those exist, this section is an artifact requirement, not an installation guide.

### 10.7 Reproducibility Requirements

Pin implementation and dependency revisions; record hardware, driver, runtime, seeds and configuration.
Publish licensed inputs or a reproducible way to obtain them.
Keep timestamps and other volatile metadata separate from the data being compared, with any normalization rule documented.

### 10.8 K3D Publication Roadmap

The earlier quarterly schedule is superseded by the code pause and specification focus.
The next deliverables must be agreed and tracked in the [PM-KR repository](https://github.com/w3c-cg/pm-kr).

## 11. Third-Party Verification Protocol

This proposed protocol describes independent review; it does not claim that review has occurred.

### 11.1 Purpose and Scope

Evaluate the stated requirement and artifact set, and report precisely which cases were reproduced.

### 11.2 Verifier Roles and Responsibilities

The author supplies accessible artifacts and expected results.
The verifier discloses relevant affiliations, records the environment and reports discrepancies and untested cases.

### 11.3 Verification Workflow

1. Obtain the published manifest, fixtures, test code and reference outputs.
2. Check integrity and publisher signatures where provided.
3. Run in the declared isolated environment, including failure controls.
4. Compare actual output to the expected behavior and publish the complete result.
5. Investigate discrepancies before issuing a scoped verdict.

### 11.4 Pass/Fail Criteria

Apply the cited draft requirements to the declared fixture set.
Missing artifacts or unexecuted checks must be reported as unverified, never passed.

#### Level A: PM-KR Core

Check all claimed data-model and reference invariants.
Record observed compression separately from the draft's proposed thresholds and their workload assumptions.

#### Level B: PM-KR Sovereign Runtime

Require Level A evidence and observed compliance with the declared runtime boundary.
A failed violation control invalidates the enforcement claim.

#### Level C: PM-KR Auditable Production

Require the lower-level evidence and complete provenance, audit and signature checks.
Report missing records and unverifiable signatures explicitly.

### 11.5 Verification Report Template

```text
Implementation and revision:
Profile and revision:
Verifier and relevant affiliations:
Date and environment:
Input manifest and public URLs:
Commands and raw output URLs:
Expected behavior:
Observed behavior, failures and negative controls:
Integrity and signature checks:
Untested cases and limitations:
Scoped verdict:
```

### 11.6 Required Logs and Signatures

Retain command output, per-case results, environment details and artifact digests.
Retain publisher signatures and verification keys separately from checksums.
An output log should not expose private data or credentials.

### 11.7 Dispute Resolution

Publish the minimal differing case and its environment.
Determine whether the cause is an implementation defect, an underspecified requirement or a measurement error.
Preserve the earlier report and link the correction.

### 11.8 K3D Verification Status

No current conformance or production-readiness claim is established by this matrix.
Available source and historical development records remain inputs for a future evaluation.

## 12. Limitations and Future Work

This editorial revision is a source and claims audit, not a runtime evaluation.

### 12.1 Known Limitations

The earlier result tables lacked sufficient public manifests, exact commands and output artifacts.
No runtime, benchmark, accessible presentation or cross-implementation exchange was tested for this revision.

### 10.2 Future Validation

Start with a bounded entry, shared meaning and provenance, then test a human view and a machine view.
Use the results to refine the vocabulary and identify the next useful exchange case.

## 11. Conclusion

A specification gains evidence through inspectable runs against explicit expectations.

### 11.1 Validation Verdict

No benchmark or conformance verdict is issued here.
The withdrawn measurements cannot support public performance claims.

### 11.2 Technology Development Implications

The next useful evidence is a small reproducible example and an independent review.
Broader scale, performance and adoption remain separate questions.

## 12. References

- [Normative model](PM_KR_NORMATIVE_MODEL.md).
- [Conformance profiles](PM_KR_CONFORMANCE_PROFILES.md).
- [Interoperability guide](PM_KR_INTEROPERABILITY_GUIDE.md).
- [Problem statement](PM_KR_PROBLEM_STATEMENT.md).
- [Project source](https://github.com/danielcamposramos/Knowledge3D).
- [W3C Community Groups FAQ](https://www.w3.org/community/about/faq/).

**License:** CC-BY-4.0.
**Publication status:** Project draft; not a Community Group certification or W3C approval.
