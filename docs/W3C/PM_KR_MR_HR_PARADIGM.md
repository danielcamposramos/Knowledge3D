# PM-KR: Machine-Readable (MR) vs. Human-Readable (HR) Dual Versioning Paradigm

> **Status, 29 October 2026: Input.** This is project input to the [PM-KR Community Group](https://www.w3.org/community/pm-kr/).
> Knowledge3D code is paused while work focuses on specifications and group deliverables, as [Daniel explained on 28 October](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0016.html).
> Requirements and examples below describe proposals, not implemented features or adopted W3C conformance rules.
<a id="a-procedural-knowledge-standard-for-ai-generated-code-in-multi-agent-systems"></a>

## A Proposal for Procedural Knowledge in Multi-Agent Systems

**Document Status**: Draft for PM-KR Community Group (Year 1)
**Authors**: Daniel Campos Ramos (Knowledge3D), GLM-4.5 (Zhipu AI)
**Date**: February 24, 2026
**Version**: 1.0

---

## Executive Summary

This draft proposes two linked presentations of procedural knowledge: a machine representation (MR) for execution and a human representation (HR) for explanation and review.
Both should derive from the same authored source and identify the same version.
The purpose is traceable generation and review.
No runtime-memory saving, deployment or safety result is established here.

The repository contains a [`codeopt` source transformer](../../codeopt/) and [its documentation](../DUAL_CODE.md).
Its presence is evidence of source code, not proof of semantic equivalence or measured memory savings.

---

## The Problem: Comments as Runtime Waste

### The Scenic Road Analogy (Cabot Trail, Canada)

A person following a route needs concise directions; the person reviewing the route may need a fuller explanation.
The analogy motivates different presentations from one source.
It does not describe how an interpreter handles comments.

---

### The Technical Problem

Source-file size, compiled-code size, process memory and GPU memory are different measurements.
Python comments are ignored by the syntax ([Python language reference](https://docs.python.org/3/reference/lexical_analysis.html#comments)).
Docstrings are different: Python exposes them through `__doc__`, and CPython's `-OO` option removes them ([command-line reference](https://docs.python.org/3/using/cmdline.html#cmdoption-OO)).
Removing docstrings can therefore change observable behavior.
No general memory percentage follows from deleting comments.

The proposal needs a defined behavior contract and tests for each source language, including introspection and tooling that consumes documentation.

---

## The Solution: MR-HR Dual Versioning

### Conceptual Model

**One Source, Two Representations:**

```
┌─────────────────────────────────────────┐
│  HR (Human-Readable)                    │
│  ================================        │
│  - Rich comments & docstrings           │
│  - Explanatory variable names           │
│  - Whitespace for readability           │
│  - Examples & usage documentation       │
│  - Type hints & annotations             │
│  ─────────────────────────────          │
│  SOURCE OF TRUTH (git committed)        │
└─────────────────────────────────────────┘
           │
           │ codeopt (MR compiler)
           ▼
┌─────────────────────────────────────────┐
│  MR (Machine-Readable)                  │
│  ================================        │
│  - Zero comments/docstrings             │
│  - Compressed whitespace                │
│  - Semantically identical               │
│  - Size and behavior require testing   │
│  ─────────────────────────────          │
│  BUILD ARTIFACT (NOT committed)         │
└─────────────────────────────────────────┘
```

**Key Principles:**
1. **HR is authoritative** — All edits happen in HR, MR is generated
2. **Semantic equivalence** — MR behaves identically to HR (no logic changes)
3. **Symlink-style storage** — Multiple HR versions can reference same MR canonical form
4. **Versioned together** — HR v2.3 generates MR v2.3 (deterministic compilation)

---

### K3D Implementation (Proof-of-Concept)

The [`codeopt` implementation](../../codeopt/) and [dual-code guide](../DUAL_CODE.md) are available for inspection.
They describe source transformation for Python and JavaScript/TypeScript.
Generated output must be tested against the authored source and its supported behavior contract.
This document provides no current implementation size, performance measurement or production-readiness claim.

---

## Benchmarks: Memory Savings in Production

### K3D Production Measurements

The previous size and timing figures are withdrawn from this draft because their input revision, commands and result artifacts were not supplied.
A future measurement should publish source and generated files, byte counts, interpreter version, warm/cold import conditions and process-memory measurements.
File-size reduction must not be reported as runtime or GPU-memory reduction.

### Multi-Instance Deployment Scenarios

Candidate deployments include multiple workers, devices with limited memory and shared workflow repositories.
Measure each configuration directly, including shared pages, caches, runtime state and reference overhead.
Multiplying source-file bytes by the number of workers does not establish process-memory consumption.

---

## PM-KR Application: Procedural Knowledge Dual Versioning

### Extending MR-HR to Procedural Workflows

**PM-KR workflows have BOTH representations:**

**MR (Machine-Executable):**
- RPN programs (canonical procedural forms)
- Stripped of explanatory metadata
- Optimized for execution (PTX kernels, GPU)
- Proposed signatures require a canonical encoding and a reviewed signing scheme; RDF Dataset Canonicalization is related work, not a procedural-program canonicalizer.

**HR (Human-Auditable):**
- Documented workflows with explanations
- Safety constraints annotated ("MUST check patient_id before transfer")
- Examples and usage documentation
- Ethical considerations ("MUST obtain consent before medication")

---

### Example: Healthcare Robot Workflow

**HR Version** (humans read, auditors review):

```yaml
# PM-KR Healthcare Workflow v2.3
# Patient Transfer Protocol
#
# PURPOSE: Safe transfer of patient from admission to ICU
# AUTHOR: Hospital Safety Committee
# REVIEWED: 2026-02-15
# ETHICAL REVIEW: Approved (consent protocol included)
#
# SAFETY CONSTRAINTS:
#   - MUST verify patient_id matches wristband RFID
#   - MUST confirm patient consent (verbal or documented)
#   - MUST NOT transfer if patient vitals unstable
#   - MUST log all transfer events to audit journal
#
# DEPENDENCIES:
#   - Galaxy_Healthcare/verify_patient_id_v1.2
#   - Galaxy_Healthcare/check_vitals_v3.0
#   - Galaxy_Healthcare/obtain_consent_v2.1

workflow: patient_transfer_v2.3
steps:
  - name: verify_identity
    description: "Verify patient identity via RFID wristband scan"
    procedure: Galaxy_Healthcare/verify_patient_id_v1.2
    safety_check: true
    failure_action: abort_transfer

  - name: check_vitals
    description: "Confirm patient vitals stable for transfer"
    procedure: Galaxy_Healthcare/check_vitals_v3.0
    parameters:
      thresholds:
        heart_rate: [60, 100]
        blood_pressure: [90, 140]
    safety_check: true
    failure_action: alert_nurse

  - name: obtain_consent
    description: "Obtain verbal or documented patient consent"
    procedure: Galaxy_Healthcare/obtain_consent_v2.1
    ethical_requirement: true
    failure_action: abort_transfer

  - name: execute_transfer
    description: "Physical transfer from admission to ICU"
    procedure: Galaxy_Healthcare/robot_transport_v4.0
    parameters:
      route: admission_to_icu_safe_path
      speed: cautious
    audit_log: true
```

**MR Version** (robots execute):

```rpn
# PM-KR MR: patient_transfer_v2.3
# Signature: SHA256(canonical_form) = a3f9c2...
# Signed: Hospital_Safety_Committee (2026-02-15)

LOAD_PROC Galaxy_Healthcare/verify_patient_id_v1.2
EXECUTE
STACK_TOP
ASSERT_EQ true
BRANCH_IF_FALSE :abort_transfer

LOAD_PROC Galaxy_Healthcare/check_vitals_v3.0
PUSH 60 100 90 140
EXECUTE
STACK_TOP
ASSERT_EQ "STABLE"
BRANCH_IF_FALSE :alert_nurse

LOAD_PROC Galaxy_Healthcare/obtain_consent_v2.1
EXECUTE
STACK_TOP
ASSERT_EQ "CONSENT_OBTAINED"
BRANCH_IF_FALSE :abort_transfer

LOAD_PROC Galaxy_Healthcare/robot_transport_v4.0
PUSH "admission_to_icu_safe_path" "cautious"
EXECUTE
AUDIT_LOG "patient_transfer_v2.3" "COMPLETED"
RETURN

:abort_transfer
AUDIT_LOG "patient_transfer_v2.3" "ABORTED"
ALERT_HUMAN "Transfer aborted: safety check failed"
RETURN

:alert_nurse
AUDIT_LOG "patient_transfer_v2.3" "VITALS_UNSTABLE"
CALL_NURSE
RETURN
```

**Storage:**

This illustrative workflow has not been measured or validated for clinical use.
Its source and generated representations require executable comparison tests.

---

### PM-KR Galaxy Distribution Model

The proposal compares a workflow repository to a package repository: machines retrieve executable procedures, while people can retrieve the corresponding source and explanation.
Versions, dependencies, permissions and provenance must remain linked.
A content checksum detects changes relative to that checksum; authenticating a publisher also requires a signature and a trusted key.
The analogy establishes no deployment, endorsement or distribution saving.

---

## PM-KR Specification: MR-HR Technology

### Conformance Levels

**Level A (Basic Procedural Representation)**:
- ✅ Procedural forms exist (RPN, BPMN, etc.)
- ⚠️ No distinction between MR/HR (monolithic)

**Level B (Dual Versioning)**:
- ✅ MR version exists (executable, stripped)
- ✅ HR version exists (documented, explanatory)
- Required semantic equivalence within the declared behavior contract (MR ≡ HR)
- ⚠️ No symlink compression (duplicate storage)

**Level C (Auditable Production with Symlink Compression)**:
- ✅ All Level B requirements
- ✅ Symlink-style storage (multiple HR → one canonical MR)
- ✅ Procedural c14n (deterministic canonical form for signatures)
- ✅ Audit Journal ("at point X during Y, observed Z")

**Level D (Robotics Safety + Ethics)**:
- ✅ All Level C requirements
- ✅ Safety constraints in HR version (MUST/MUST NOT predicates)
- ✅ Ethical review metadata (consent, privacy, harm prevention)
- Proposed formal verification of explicitly stated properties
- Proposed structured validation reports with explicit input, output and evidence fields

---

### Normative Requirements

**For PM-KR Conformance Level B+ (Dual Versioning):**

**MUST**:
1. **Maintain two representations**: MR (machine-executable) and HR (human-readable)
2. **Semantic equivalence**: MR ≡ HR (identical behavior, no logic changes)
3. **Deterministic generation**: Same HR input → same MR output (reproducible builds)
4. **Version synchronization**: HR v2.3 generates MR v2.3 (no version drift)

**SHOULD**:
1. **HR as authoritative source**: All edits happen in HR, MR is generated
2. **Symlink-style references**: Multiple HR versions can reference same canonical MR
3. **Signature verification**: Sign MR canonical form via procedural c14n

**MAY**:
1. **Multi-tier compilation**: Different MR optimization levels (core, trainers, full)
2. **Language-specific transforms**: Python docstring removal, JS comment stripping, etc.

---

### Compilation Guarantees

These are proposed preservation requirements, not demonstrated guarantees.
A transformation that changes an observable `__doc__` value fails full behavioral equivalence for consumers relying on it.
The supported contract must state such exclusions explicitly, and any safety check must remain executable code.

**Python MR Generation**:
- Remove `#` comments (single-line)
- Remove `"""` and `'''` docstrings (multi-line)
- Insert `pass` for docstring-only function bodies
- Compress blank lines (max 1 consecutive)
- **Preserve**: Indentation, token order, semantics

**JavaScript/TypeScript MR Generation**:
- Remove `//` comments (single-line)
- Remove `/* */` and `/** */` comments (multi-line)
- Respect string literals and template strings
- Compress whitespace
- **Preserve**: Token order, semantics

**RPN MR Generation** (PM-KR workflows):
- Remove YAML/JSON comments
- Remove explanatory metadata (`description`, `author`, `reviewed`, etc.)
- Extract pure RPN program (executable operations only)
- **Preserve**: Control flow, stack operations, procedure calls

---

## Implementation Guide

### For AI Code Generators

**When generating code for multi-agent systems:**

1. **Generate BOTH versions simultaneously**:
```python
# AI prompt
"Generate Python function for patient_transfer workflow with:
- HR version: Rich docstrings, comments, examples
- MR version: Stripped, executable-only
- Guarantee semantic equivalence"
```

2. **Validate semantic equivalence**:
```bash
# Test both versions produce identical output
pytest test_patient_transfer.py --hr-version
pytest test_patient_transfer.py --mr-version
diff hr_output.json mr_output.json  # Should be empty
```

3. **Sign canonical MR form**:
```bash
# Procedural c14n → signature
codeopt --canonicalize patient_transfer_v2.3.py > canonical.rpn
sha256sum canonical.rpn > signature.txt
gpg --sign signature.txt
```

---

### For Multi-Agent Orchestrators

**Deployment pattern**:

```python
# Agent configuration
agent_config = {
    'execution': {
        'source': 'MR',  # Agents execute MR version (memory-efficient)
        'path': '/galaxy/workflows/mr/patient_transfer_v2.3.rpn'
    },
    'audit': {
        'source': 'HR',  # Audit logs reference HR for explanation
        'path': '/galaxy/workflows/hr/patient_transfer_v2.3.yaml'
    },
    'verification': {
        'signature': '/galaxy/signatures/patient_transfer_v2.3.sig',
        'algorithm': 'procedural_c14n + SHA256 + GPG'
    }
}

# Agent executes MR
agent.load_workflow(agent_config['execution']['path'])
agent.execute()

# Audit references HR for human review
audit_entry = {
    'timestamp': '2026-02-24T10:42:33Z',
    'agent_id': 'robot_01',
    'workflow_mr': 'patient_transfer_v2.3.rpn',
    'workflow_hr': 'patient_transfer_v2.3.yaml',  # Human-readable explanation
    'position': 'step_3_obtain_consent',
    'observation': 'CONSENT_OBTAINED',
    'signature_valid': True
}
```

---

<a id="for-robotics-safety-adams-use-case"></a>

### For Robotics Safety

This is a hypothetical workflow example; no deployment, clinical safety assessment or endorsement is established here.

**Industrial robot downloading workflow from Galaxy**:

1. **Robot queries Galaxy** for task workflow:
```python
workflow = galaxy.query('Galaxy_Manufacturing/assembly_procedure_v5.2')
# Returns: { 'mr': <RPN_executable>, 'hr': <documented_yaml>, 'signature': <GPG_sig> }
```

2. **Robot verifies signature** (procedural c14n):
```python
canonical_form = procedural_c14n(workflow['mr'])
signature_valid = gpg.verify(canonical_form, workflow['signature'])
if not signature_valid:
    abort("Workflow signature invalid - potential tampering!")
```

3. **Robot loads MR version** (memory-efficient execution):
```python
robot.load_procedure(workflow['mr'])  # Proposed executable representation
# The corresponding authored source remains available for review.
```

4. **Robot executes + audits**:
```python
robot.execute()
# Audit entry references HR for human inspectors
audit_journal.log({
    'position': 'step_5_bolt_tightening',
    'procedure_mr': 'assembly_procedure_v5.2.rpn',
    'procedure_hr': 'assembly_procedure_v5.2.yaml',
    'observation': 'TORQUE_CORRECT',
    'safety_check_passed': True
})
```

5. **Human auditor reviews HR** (after incident):
```bash
# Auditor investigates: "Why did robot skip safety step?"
audit_query --workflow assembly_procedure_v5.2 --date 2026-02-24
# Returns: Audit entries with references to HR version (documented, explained)
cat /galaxy/workflows/hr/assembly_procedure_v5.2.yaml
# Human reads: "Step 5: MUST tighten bolt to 45Nm ± 5Nm (safety critical)"
```

---

## Performance Analysis

### Memory Efficiency

No memory saving is established here.
Measure resident process memory separately from source bytes and include sharing, caches and runtime objects.
GPU memory requires its own measurement.

### Import Speed

No import-speed result is established here.
Compare the same module and dependencies under a pinned interpreter, separating cold imports, bytecode-cache reuse and repeated imports.
Retain the measurement script and raw timings.

### Storage Efficiency (Galaxy Distribution)

No deployment scale or distribution saving is established here.
Compare the same workflows and versions, including metadata, references, transport compression, caches and the retained human-readable sources.
Reference reuse within a repository does not eliminate the cost of delivering executable content to each device.

---

## PM-KR Year 1 Deliverables

### Proposed Specification Documents

The following are topics from the original draft, not agreed group deliverables or a current schedule.

1. **PM-KR MR-HR Core Specification** (Q2 2026)
   - Normative requirements for dual versioning
   - Semantic equivalence guarantees
   - Compilation validation criteria

2. **PM-KR Procedural Canonicalization (c14n)** (Q3 2026)
   - Deterministic ordering of RPN programs (with RDF Dataset Canonicalization as related work)
   - Signature verification protocol
   - Reproducible builds standard

3. **PM-KR Galaxy Distribution Protocol** (Q3 2026)
   - Debian `apt` model for procedural workflows
   - Symlink-style compression
   - Regional mirrors, version pinning

4. **PM-KR Audit Schema** (Q2 2026)
   - A structured observation format identifying event, context and time
   - MR execution + HR explanation linking
   - Structured validation reports

5. **PM-KR Robotics Safety Profile** (Q4 2026)
   - Level D conformance requirements
   - Safety constraints in HR version
   - Ethical review metadata
   - Lean4 verification examples

---

### Reference Implementations

The [`codeopt` source](../../codeopt/) is a project implementation input.
The proposed workflow-distribution demonstration remains a candidate experiment.
No fleet deployment, memory saving or distribution percentage is claimed.

---

## Related Work

### Compiled vs. Interpreted Languages

Comment and documentation handling depends on the language and runtime.
See the Python references above for the distinction between ignored comments and observable docstrings.
The proposed MR/HR relationship must preserve the supported behavior contract and keep the authored source available for review.

### Procedural Compression

Related approaches include source minification, ordinary lossless compression and procedural asset generation.
Compare them on the same behavior and storage requirements.
Preserving source, distributing documentation and validating transformations are not exclusive to PM-KR.
The [Farbrausch tool sources](https://github.com/farbrausch/fr_public) are a procedural-generation precedent.

### AI Code Generation

A generator could produce authored source and a derived runtime representation, then publish the transformation and comparison tests.
An AI-generated test result is not itself evidence that the tests ran or that behavior is equivalent.
Human review must retain responsibility for source, licensing, uncertainty and accepted behavior changes.

---

## Conclusion

This draft proposes a traceable relationship between executable procedures and their human explanations.
The repository's source transformer is available for inspection.
Equivalence, memory savings, performance and suitability for any safety-critical use require separate evidence.

---

## Appendix A: Tool Documentation

### `codeopt` CLI Reference

**Installation**:
```bash
pip install codeopt  # (when released to PyPI)
# Or: Use K3D's bundled version
git clone https://github.com/danielcamposramos/Knowledge3D
cd Knowledge3D
python -m k3dgen.tools.codeopt --help
```

**Basic Usage**:
```bash
# Generate MR for entire repo
codeopt --in . --out ../Knowledge3D.local/mr --lang auto --stats

# Python only
codeopt --in knowledge3d --out ../mr --lang py --stats

# JavaScript/TypeScript only
codeopt --in viewer/src --out ../mr/viewer --lang js --stats
```

**Options**:
- `--in <path>`: Input directory (HR sources)
- `--out <path>`: Output directory (MR artifacts)
- `--lang <auto|py|js>`: Language filter (auto-detect if `auto`)
- `--stats`: Print savings report (bytes saved per file)
- `--verify`: Run import tests on MR outputs (Python only)

**K3D Makefile Targets**:
```bash
# Tier 1: Hot-path core only
make compile-mr-core

# Tier 2: Add trainers
make compile-mr-trainers

# Tier 3: Full repo
make compile-mr-all

# Clean MR artifacts
make clean-mr

# Savings report
make mr-report
```

---

## Appendix B: Semantic Equivalence Testing

### Test Harness Example

```python
# test_semantic_equivalence.py
import sys
import importlib

def test_equivalence(module_name_hr, module_name_mr, test_function):
    """
    Verify HR and MR versions produce identical output.

    Args:
        module_name_hr: HR module path (e.g., 'knowledge3d.cranium.fused_head')
        module_name_mr: MR module path (e.g., '../Knowledge3D.local/mr/knowledge3d/cranium/fused_head')
        test_function: Function to call in both modules (e.g., 'forward')
    """
    # Load HR module
    sys.path.insert(0, '.')
    module_hr = importlib.import_module(module_name_hr)

    # Load MR module
    sys.path.insert(0, '../Knowledge3D.local/mr')
    module_mr = importlib.import_module(module_name_mr)

    # Get test function from both
    func_hr = getattr(module_hr, test_function)
    func_mr = getattr(module_mr, test_function)

    # Test with sample inputs
    test_inputs = [
        {'text': 'Show me a red circle'},
        {'image': torch.randn(3, 224, 224)},
        {'audio': torch.randn(16000)}
    ]

    for input_data in test_inputs:
        output_hr = func_hr(input_data)
        output_mr = func_mr(input_data)

        # Verify outputs are identical
        assert torch.allclose(output_hr, output_mr, atol=1e-6), \
            f"Semantic equivalence violated! HR != MR for input {input_data}"

    print(f"✅ Semantic equivalence verified for {module_name_hr}::{test_function}")

# Run tests
test_equivalence('knowledge3d.cranium.fused_head', 'fused_head', 'forward')
test_equivalence('knowledge3d.skills.vision', 'vision', 'encode')
```

---

## Appendix C: References

**K3D Documentation**:
- [DUAL_CODE.md](../DUAL_CODE.md) — Technical implementation of `codeopt`
- [DUAL_CODE_STRATEGY.md](../DUAL_CODE_STRATEGY.md) — When/how to use MR-HR
- [HR_MR_STANDARD.md](../HR_MR_STANDARD.md) — GLM-4.5 standard specification

**PM-KR Specifications**:
- [PM-KR Problem Statement](PM_KR_PROBLEM_STATEMENT.md)
- [PM-KR Normative Model](PM_KR_NORMATIVE_MODEL.md)
- [PM-KR Conformance Profiles](PM_KR_CONFORMANCE_PROFILES.md)

**External References**:
- Manu Sporny, RDF Canonicalization (rdf-canon): https://www.w3.org/TR/rdf-canon/
- Adam Sobieski, [Web Context, WICG proposal #188](https://github.com/WICG/proposals/issues/188), related discussion; not a PM-KR implementation or endorsement.
- OpenFn Workflow Automation: https://www.openfn.org/
- [Farbrausch public tool sources](https://github.com/farbrausch/fr_public)

---

**Document History**:
- **v1.0** (2026-02-24): Initial draft for PM-KR CG Year 1
- **Authors**: Daniel Campos Ramos (K3D), GLM-4.5 (Zhipu AI)
- **Project credits**: See [ATTRIBUTIONS.md](../../ATTRIBUTIONS.md).
The earlier draft's role assignments are not used here as evidence of participation or endorsement.

**License**: CC BY 4.0 (Creative Commons Attribution)

**For PM-KR Community Group**: https://www.w3.org/community/pm-kr/

---

**END OF DOCUMENT**
