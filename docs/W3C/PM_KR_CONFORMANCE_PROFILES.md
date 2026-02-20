# PM-KR Conformance Profiles: Implementation Guide

**Document Type**: W3C Community Group Implementation Guide (Draft)
**Version**: 1.2
**Date**: February 20, 2026
**Authors**: Knowledge3D Project Contributors
**Status**: Draft Guide

---

## Abstract

This document provides **implementation guidance** for achieving Procedural Memory Knowledge Representation (PM-KR) conformance. It defines three conformance levels (A, B, C) with specific requirements, validation criteria, and migration paths. Reference implementation: Knowledge3D (K3D) project.

**Evidence Note**: This document defines normative conformance targets. Any implementation status statements are explicitly marked as either repo-verified, run-log verified, or target/projection.

---

## 1. Conformance Levels Overview

| Level | Name | Focus | Typical Use Case |
|-------|------|-------|------------------|
| **A** | PM-KR Core | Data model + composition | Static knowledge bases, archives |
| **B** | PM-KR Sovereign Runtime | + Zero external dependencies | Real-time AI inference systems |
| **C** | PM-KR Auditable Production | + Provenance + metrics | Mission-critical, regulated environments |

**Progressive Enhancement**: Level B includes all Level A requirements; Level C includes all Level A+B requirements.

---

## 2. Level A: PM-KR Core

### 2.1 Requirements

**MUST**:
1. Implement 4-layer compositional model (Form → Meaning → Rules → Meta-Rules)
2. Enforce Canonicality Invariant (one canonical source per concept)
3. Enforce Reference Preservation Invariant (symlink composition)
4. Support Deterministic Reconstruction Invariant (checksums pass)
5. Expose minimal node schema (id, layer, programs, refs, metadata)

**SHOULD**:
- Implement reference graph validation (detect cycles, broken refs)
- Provide compression metrics (reference graph size vs payload size)
- Document procedural language used (RPN, Lisp, Forth, etc.)

**MAY**:
- Add custom metadata fields
- Implement caching strategies
- Support multiple procedural languages

### 2.2 Validation Criteria

**Test Suite** (minimum 5 tests required):

1. **Canonicality Test**
   ```python
   def test_canonicality():
       """Verify no duplicate canonical procedures."""
       nodes = pm_kr_system.get_all_nodes()
       canonical_ids = [n.id for n in nodes if n.is_canonical]
       assert len(canonical_ids) == len(set(canonical_ids))  # No duplicates
   ```

2. **Reference Resolution Test**
   ```python
   def test_reference_resolution():
       """Verify all references resolve to valid nodes."""
       nodes = pm_kr_system.get_all_nodes()
       for node in nodes:
           for ref_id in node.get_all_refs():
               resolved = pm_kr_system.resolve_ref(ref_id)
               assert resolved is not None  # All refs must resolve
   ```

3. **Determinism Test**
   ```python
   def test_determinism():
       """Same seed → same output."""
       seed = 42
       output1 = pm_kr_system.reconstruct_node("char_a", seed=seed)
       output2 = pm_kr_system.reconstruct_node("char_a", seed=seed)
       assert output1 == output2  # Bit-identical
   ```

4. **Compression Test**
   ```python
   def test_compression():
       """Verify symlink compression achieves >50% reduction."""
       payload_size = calculate_payload_size(nodes)
       reference_size = calculate_reference_graph_size(nodes)
       compression_ratio = 1 - (reference_size / payload_size)
       assert compression_ratio > 0.5  # At least 50% reduction
   ```

5. **Layer Composition Test**
   ```python
   def test_layer_composition():
       """Verify higher layers reference lower layers."""
       meaning_nodes = [n for n in nodes if n.layer == "meaning"]
       for node in meaning_nodes:
           assert len(node.refs) > 0  # Meaning MUST reference Form
   ```

### 2.3 Implementation Checklist

- [ ] **Data Model**: Implement minimal node schema (see Normative Model §6.1)
- [ ] **Layer System**: Support Form, Meaning, Rules, Meta-Rules layers
- [ ] **Reference Types**: Implement char_refs, word_refs, symbol_refs, rule_refs, component_refs
- [ ] **Canonicalization**: Content-addressable IDs or registry enforcement
- [ ] **Reference Resolution**: Resolve refs to canonical nodes
- [ ] **Validation**: Pass 5 core conformance tests
- [ ] **Documentation**: Document procedural language and execution environment

### 2.4 Example: Level A Minimal Implementation

**Python Reference (Simplified)**:

```python
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class PMKRNode:
    """Minimal PM-KR Level A node."""
    id: str
    layer: str  # "form" | "meaning" | "rules" | "meta_rules"
    form_program: Optional[str] = None
    meaning_program: Optional[str] = None
    refs: Dict[str, List[str]] = None  # char_refs, word_refs, etc.
    metadata: Dict = None

    def get_all_refs(self) -> List[str]:
        """Extract all reference IDs."""
        if not self.refs:
            return []
        return [ref_id for ref_list in self.refs.values() for ref_id in ref_list]

class PMKRSystemLevelA:
    """Level A: PM-KR Core implementation."""

    def __init__(self):
        self.nodes: Dict[str, PMKRNode] = {}
        self.canonical_registry: Dict[str, str] = {}  # content_hash → node_id

    def add_node(self, node: PMKRNode):
        """Add node with canonicality check."""
        # Check if canonical source already exists
        content_hash = self._compute_content_hash(node)
        if content_hash in self.canonical_registry:
            raise ValueError(f"Canonical node already exists: {self.canonical_registry[content_hash]}")

        self.nodes[node.id] = node
        self.canonical_registry[content_hash] = node.id

    def resolve_ref(self, ref_id: str) -> Optional[PMKRNode]:
        """Resolve reference to canonical node."""
        return self.nodes.get(ref_id)

    def reconstruct_node(self, node_id: str, seed: int = 42) -> str:
        """Deterministically reconstruct node output."""
        node = self.nodes[node_id]

        # Resolve references
        if node.refs:
            resolved_refs = {
                ref_type: [self.resolve_ref(rid) for rid in ref_ids]
                for ref_type, ref_ids in node.refs.items()
            }
        else:
            resolved_refs = {}

        # Execute procedural program (deterministic)
        output = self._execute_program(
            node.form_program or node.meaning_program,
            resolved_refs,
            seed
        )

        return output

    def _compute_content_hash(self, node: PMKRNode) -> str:
        """Compute content-addressable hash."""
        import hashlib
        content = f"{node.layer}:{node.form_program}:{node.meaning_program}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _execute_program(self, program: str, refs: Dict, seed: int) -> str:
        """Execute procedural program (placeholder)."""
        # Implementation depends on procedural language (RPN, Lisp, etc.)
        return f"executed:{program}:seed{seed}"
```

**Usage**:
```python
# Create system
system = PMKRSystemLevelA()

# Add Form layer node (canonical character)
char_a = PMKRNode(
    id="char_latin_a",
    layer="form",
    form_program="BEZIER_CURVE [...] PROCEDURAL_FONT_LATIN_A",
    metadata={"domain": "language", "version": "1.0"}
)
system.add_node(char_a)

# Add Meaning layer node (word composed from char refs)
word_rotation = PMKRNode(
    id="word_rotation",
    layer="meaning",
    meaning_program="CONCEPT_ROTATION SPATIAL_TRANSFORMATION",
    refs={"char_refs": ["char_r", "char_o", "char_t", "char_a", "char_t", "char_i", "char_o", "char_n"]},
    metadata={"domain": "language", "version": "1.0"}
)
system.add_node(word_rotation)

# Validate
assert system.resolve_ref("char_latin_a") == char_a  # Reference resolution works
output = system.reconstruct_node("char_latin_a", seed=42)  # Deterministic reconstruction
```

---

## 3. Level B: PM-KR Sovereign Runtime

### 3.1 Requirements

**Includes all Level A requirements, plus:**

**MUST**:
1. Enforce Sovereign Boundary Invariant (hot path = zero external dependencies)
2. Implement fail-fast behavior for unavailable sovereign backends
3. Provide execution telemetry (latency, GPU call counts, fallback triggers)
4. Document execution environment (PTX, WebAssembly, LLVM IR, etc.)
5. Guarantee determinism in execution environment (not just reconstruction)

**SHOULD**:
- Implement hot-path sovereignty tests (grep for forbidden imports)
- Provide performance benchmarks (latency percentiles, throughput)
- Support multiple execution backends with consistent semantics

**MAY**:
- Implement GPU acceleration (PTX, CUDA, Metal, Vulkan)
- Provide sovereignty violation alerts/logging
- Support sandboxed execution for untrusted procedures

### 3.2 Validation Criteria

**Test Suite** (Level A tests + 3 additional tests):

6. **Sovereignty Test**
   ```python
   def test_sovereignty():
       """Hot path uses zero external dependencies."""
       import sys

       # Baseline: Capture current imports
       baseline_modules = set(sys.modules.keys())

       # Execute hot path
       result = pm_kr_system.execute_hot_path(query="solve x+2=5")

       # Check no new external modules loaded
       new_modules = set(sys.modules.keys()) - baseline_modules
       forbidden = ["numpy", "scipy", "sympy", "pandas"]
       violations = [m for m in new_modules if any(f in m for f in forbidden)]

       assert len(violations) == 0, f"Sovereignty violations: {violations}"
   ```

7. **Execution Determinism Test**
   ```python
   def test_execution_determinism():
       """Same procedural program → same execution result."""
       program = "2 3 ADD"
       result1 = pm_kr_system.execute_program(program, seed=42)
       result2 = pm_kr_system.execute_program(program, seed=42)
       assert result1 == result2  # Execution deterministic, not just reconstruction
   ```

8. **Telemetry Test**
   ```python
   def test_telemetry():
       """Execution telemetry captures GPU calls and latency."""
       result = pm_kr_system.execute_program("SOLVE x+2=5", collect_telemetry=True)

       assert "gpu_call_count" in result.telemetry
       assert "latency_ms" in result.telemetry
       assert "fallback_triggered" in result.telemetry
       assert result.telemetry["fallback_triggered"] == False  # Sovereign execution
   ```

### 3.3 Implementation Checklist

**Level A checklist, plus:**

- [ ] **Execution Environment**: Implement sovereign backend (PTX, WebAssembly, etc.)
- [ ] **Sovereignty Enforcement**: Static analysis + runtime checks for external deps
- [ ] **Telemetry**: GPU call counters, latency guards, fallback detection
- [ ] **Fail-Fast**: Explicit errors for unavailable sovereign backends (no silent fallbacks)
- [ ] **Documentation**: Describe execution semantics and determinism guarantees
- [ ] **Validation**: Pass 8 total tests (5 from Level A + 3 from Level B)

### 3.4 Example: Level B Sovereign Execution

**K3D Reference (PTX Backend)**:

```python
class PMKRSystemLevelB(PMKRSystemLevelA):
    """Level B: PM-KR Sovereign Runtime."""

    def __init__(self):
        super().__init__()
        self.ptx_engine = self._init_ptx_engine()
        self.telemetry = {"gpu_calls": 0, "fallbacks": 0}

    def execute_hot_path(self, query: str):
        """Execute query with sovereign PTX-only hot path."""
        # Ensure no external dependencies
        self._assert_sovereignty()

        # Parse query → RPN program
        rpn_program = self._parse_to_rpn(query)

        # Execute on GPU via PTX
        result = self._execute_ptx(rpn_program)

        # Record telemetry
        self.telemetry["gpu_calls"] += 1

        return result

    def _assert_sovereignty(self):
        """Fail-fast check for external dependencies."""
        import sys
        forbidden = ["numpy", "scipy", "sympy"]
        loaded = [m for m in sys.modules.keys() if any(f in m for f in forbidden)]

        if loaded:
            raise RuntimeError(
                f"Sovereignty violation: Hot path loaded forbidden modules: {loaded}"
            )

    def _execute_ptx(self, rpn_program: str):
        """Execute RPN via PTX kernels (sovereign)."""
        # Actual PTX execution via CUDA Driver API
        # See K3D: knowledge3d/cranium/sovereign/loader.py
        return self.ptx_engine.execute(rpn_program)

    def _init_ptx_engine(self):
        """Initialize PTX execution engine."""
        from knowledge3d.cranium.sovereign import loader
        return loader.SovereignRPNEngine()
```

**Sovereignty Test**:
```python
def test_k3d_sovereignty():
    """K3D week-22 snapshot: solved tasks map 1:1 to GPU calls."""
    system = PMKRSystemLevelB()

    # Solve 400 math problems
    results = []
    for problem in math_benchmark[:400]:
        result = system.execute_hot_path(problem)
        results.append(result)

    # Validate sovereignty
    assert system.telemetry["gpu_calls"] == 154  # Only solved tasks
    assert system.telemetry["fallbacks"] == 0    # Zero fallbacks
    # Sovereignty: 154 GPU calls = 154 solved tasks (100%)
```

---

## 4. Level C: PM-KR Auditable Production

### 4.1 Requirements

**Includes all Level A+B requirements, plus:**

**MUST**:
1. Enforce Auditability Invariant (provenance tracking, transformation chains)
2. Provide compression metrics reporting (reference graph stats)
3. Include conformance test artifacts (all test results published)
4. Implement provenance verification (cryptographic signatures)
5. Support audit trail export (for compliance/debugging)

**SHOULD**:
- Implement distributed provenance tracking (multi-agent lineage)
- Provide visual tools for reference graph inspection
- Support third-party audits (independent conformance validation)

**MAY**:
- Implement blockchain-style provenance chains
- Provide formal verification of critical procedures
- Support real-time compliance dashboards

### 4.2 Validation Criteria

**Test Suite** (Level A+B tests + 4 additional tests):

9. **Provenance Tracking Test**
   ```python
   def test_provenance_tracking():
       """Node provenance includes full transformation chain."""
       node = pm_kr_system.get_node("word_rotation")

       assert "provenance" in node.metadata
       assert "source" in node.metadata.provenance
       assert "transformation_chain" in node.metadata.provenance
       assert "timestamp" in node.metadata.provenance
       assert "agent" in node.metadata.provenance
   ```

10. **Compression Metrics Test**
    ```python
    def test_compression_metrics():
        """System reports compression statistics."""
        metrics = pm_kr_system.get_compression_metrics()

        assert "total_nodes" in metrics
        assert "reference_count" in metrics
        assert "payload_size_mb" in metrics
        assert "reference_graph_size_mb" in metrics
        assert "compression_ratio" in metrics
        assert metrics["compression_ratio"] > 0.5  # At least 50%
    ```

11. **Audit Trail Export Test**
    ```python
    def test_audit_trail_export():
        """System exports complete audit trail."""
        audit_trail = pm_kr_system.export_audit_trail(node_id="word_rotation")

        assert "creation_event" in audit_trail
        assert "modification_events" in audit_trail
        assert "access_events" in audit_trail

        # Verify cryptographic signatures
        for event in audit_trail["modification_events"]:
            assert pm_kr_system.verify_signature(event)
    ```

12. **Conformance Report Test**
    ```python
    def test_conformance_report():
        """System generates full conformance report."""
        report = pm_kr_system.generate_conformance_report()

        assert report["level"] in ["A", "B", "C"]
        assert "test_results" in report
        assert all(result["status"] == "PASS" for result in report["test_results"])
        assert "compression_metrics" in report
        assert "sovereignty_metrics" in report
        assert "provenance_coverage" in report
    ```

### 4.3 Implementation Checklist

**Level A+B checklists, plus:**

- [ ] **Provenance System**: Track creation, modification, access events
- [ ] **Cryptographic Signing**: Sign canonical procedures and transformations
- [ ] **Compression Metrics**: Real-time stats on reference graph efficiency
- [ ] **Audit Trail Export**: JSON/JSONL export of full event history
- [ ] **Conformance Reporting**: Automated report generation (all 12 tests)
- [ ] **Third-Party Audit**: External validation of conformance claims
- [ ] **Documentation**: Provenance schema, signature algorithms, audit procedures
- [ ] **Validation**: Pass 12 total tests (5+3+4)

### 4.4 Example: Level C Provenance Tracking

**K3D Reference (Shadow Copy Integration)**:

```python
class PMKRSystemLevelC(PMKRSystemLevelB):
    """Level C: PM-KR Auditable Production."""

    def __init__(self):
        super().__init__()
        self.audit_journal = []
        self.compression_metrics = {
            "total_nodes": 0,
            "reference_count": 0,
            "payload_size_bytes": 0,
            "reference_graph_size_bytes": 0
        }

    def add_node(self, node: PMKRNode, agent: str = "system"):
        """Add node with full provenance tracking."""
        # Canonical check (Level A)
        super().add_node(node)

        # Record creation event
        event = {
            "type": "node_created",
            "node_id": node.id,
            "agent": agent,
            "timestamp": self._current_timestamp(),
            "provenance": {
                "source": node.metadata.get("provenance", "unknown"),
                "transformation_chain": node.metadata.get("transformation_chain", []),
            },
            "signature": self._sign_event(node)
        }
        self.audit_journal.append(event)

        # Update compression metrics
        self._update_compression_metrics(node)

    def export_audit_trail(self, node_id: str):
        """Export full audit trail for node."""
        events = [e for e in self.audit_journal if e["node_id"] == node_id]
        return {
            "node_id": node_id,
            "creation_event": events[0] if events else None,
            "modification_events": [e for e in events if e["type"] == "node_modified"],
            "access_events": [e for e in events if e["type"] == "node_accessed"]
        }

    def generate_conformance_report(self):
        """Generate full conformance report."""
        return {
            "level": "C",
            "test_results": self._run_all_tests(),
            "compression_metrics": self.compression_metrics,
            "sovereignty_metrics": self.telemetry,
            "provenance_coverage": len(self.audit_journal) / self.compression_metrics["total_nodes"],
            "timestamp": self._current_timestamp(),
            "signature": self._sign_report()
        }

    def _update_compression_metrics(self, node: PMKRNode):
        """Update compression statistics."""
        self.compression_metrics["total_nodes"] += 1

        # Count references
        ref_count = sum(len(refs) for refs in (node.refs or {}).values())
        self.compression_metrics["reference_count"] += ref_count

        # Estimate sizes
        payload_size = len(node.form_program or "") + len(node.meaning_program or "")
        self.compression_metrics["payload_size_bytes"] += payload_size
        self.compression_metrics["reference_graph_size_bytes"] += (ref_count * 32)  # 32 bytes per ref

    def _sign_event(self, node: PMKRNode):
        """Cryptographically sign event."""
        import hashlib
        content = f"{node.id}:{node.layer}:{self._current_timestamp()}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _sign_report(self):
        """Sign conformance report."""
        import hashlib
        import json
        content = json.dumps(self.compression_metrics, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
```

---

## 5. Migration Paths

### 5.1 From Traditional Knowledge Graphs

**Challenge**: Migrate from RDF/OWL/JSON-LD to PM-KR.

**Strategy**:
1. **Identify Canonical Entities**: Group duplicate triples by subject URI
2. **Extract Procedural Semantics**: Convert static properties to executable programs
3. **Build Reference Graph**: Replace repeated values with references
4. **Validate Compression**: Measure reduction (target >50%)

**Example**:
```turtle
# Original RDF (duplicated)
:person1 :name "Alice" .
:person2 :name "Bob" .
:person3 :name "Alice" .  # Duplicate!

# PM-KR (canonical + refs)
:name_alice rdfs:label "Alice" ;
            pm:form_program "RENDER_NAME 'Alice'" .

:person1 pm:name_ref :name_alice .
:person2 pm:name_ref :name_bob .
:person3 pm:name_ref :name_alice .  # Reference, not duplicate
```

### 5.2 From Static Embeddings

**Challenge**: Migrate from embedding-only systems (no procedural source).

**Strategy**:
1. **Reverse-Engineer Procedures**: Use code generation to create procedural approximations
2. **Canonical Clustering**: Group similar embeddings, designate canonical representatives
3. **Reference Linking**: Lower-similarity nodes reference canonical clusters
4. **Validate Reconstruction**: Ensure procedural execution ≈ original embeddings

**K3D Validation**: Character Galaxy (21,915 chars) migrated from font files → procedural fonts (70% reduction).

### 5.3 From Monolithic Systems

**Challenge**: Migrate from single large model (LLM) to PM-KR compositional knowledge.

**Strategy**:
1. **Knowledge Extraction**: Use LLM to generate procedural knowledge base
2. **Canonical Deduplication**: Content-addressable IDs for procedures
3. **Reference Graph Construction**: Build symlink composition layers
4. **Sovereignty Migration**: Replace LLM inference calls with PTX execution

**K3D Example**: Math solver (originally sympy-based) → fully sovereign PTX (38.5% accuracy, 100% GPU).

---

## 6. Third-Party Verification Guide

### 6.1 Overview

This section provides **step-by-step verification instructions** for independent auditors validating PM-KR conformance claims. It complements the full Third-Party Verification Protocol in the Evidence Validation Matrix document.

**Target Audience**: Certification bodies, academic reviewers, peer implementers.

---

### 6.2 Quick Verification Checklist

**Level A: PM-KR Core** (5 tests required):

- [ ] Clone repository and install dependencies
- [ ] Run `pytest tests/test_pmkr_level_a_*.py -v`
- [ ] Verify 5/5 tests passing
- [ ] Check canonicality (zero duplicate IDs)
- [ ] Validate compression (>50% reduction)
- [ ] Confirm determinism (checksums match)
- [ ] Issue verdict: PASS or FAIL

**Level B: PM-KR Sovereign Runtime** (8 tests required):

- [ ] Complete Level A verification (prerequisite)
- [ ] Run `grep -r "import numpy\|import scipy\|import sympy" {hot_path_dirs}/`
- [ ] Verify zero matches (sovereignty static check)
- [ ] Run benchmark with telemetry collection
- [ ] Validate GPU calls / solved tasks = 1.0
- [ ] Confirm execution determinism (same RPN → same output)
- [ ] Issue verdict: PASS or FAIL

**Level C: PM-KR Auditable Production** (12 tests required):

- [ ] Complete Level A+B verification (prerequisite)
- [ ] Request provenance audit export
- [ ] Validate 100% provenance coverage
- [ ] Verify cryptographic signatures (SHA-256)
- [ ] Check compression metrics reporting
- [ ] Validate conformance report completeness
- [ ] Issue verdict: PASS or FAIL

---

### 6.3 Environment Requirements

**Minimum Requirements**:

| Component | Level A | Level B | Level C |
|-----------|---------|---------|---------|
| **Python** | 3.10+ | 3.10+ | 3.10+ |
| **GPU** | Not required | NVIDIA 8GB+ VRAM | NVIDIA 8GB+ VRAM |
| **CUDA** | Not required | 11.8+ | 11.8+ |
| **Disk Space** | 2GB | 5GB | 10GB |
| **Network** | Git clone only | Git clone only | Git clone only |

**Recommended** (for K3D verification):
- NVIDIA RTX 3060 (12GB VRAM) or equivalent
- Ubuntu 22.04 LTS
- 16GB system RAM

---

### 6.4 Artifact Validation

**Required Artifacts by Level**:

**Level A**:
- `pmkr_level_a_test_report_YYYY-MM-DD.json` (test results)
- `pmkr_level_a_reference_dataset.jsonl` (sample nodes)
- `pmkr_level_a_signatures_YYYY-MM-DD.txt` (SHA-256 checksums)

**Level B** (includes Level A + additional):
- `pmkr_level_b_telemetry_YYYY-MM-DD.jsonl` (GPU call telemetry)
- `pmkr_level_b_benchmark_run_YYYY-MM-DD.log` (execution log)
- `pmkr_level_b_forbidden_imports.txt` (sovereignty blacklist)

**Level C** (includes Level A+B + additional):
- `pmkr_level_c_provenance_audit_YYYY-MM-DD.jsonl` (audit trail)
- `pmkr_level_c_compression_metrics_YYYY-MM-DD.json` (compression stats)
- `pmkr_level_c_conformance_report_YYYY-MM-DD.json` (full report)

**Validation Commands**:
```bash
# Verify checksums for all artifacts
sha256sum -c pmkr_level_{a|b|c}_signatures_YYYY-MM-DD.txt

# Expected output: All files PASS
```

---

### 6.5 Common Verification Issues

**Issue 1: Non-Deterministic Test Outputs**

**Symptom**: Test passes locally but checksums don't match published results.

**Diagnosis**:
```bash
# Check for timestamp differences
diff \
  <(jq 'del(.timestamp)' published_report.json) \
  <(jq 'del(.timestamp)' verifier_report.json)
```

**Resolution**: Exclude timestamp fields from checksum comparison (per spec §10.7).

---

**Issue 2: GPU Availability**

**Symptom**: Level B tests fail with "CUDA device not found."

**Diagnosis**:
```bash
# Verify GPU availability
nvidia-smi

# Check CUDA version
nvcc --version
```

**Resolution**: Level B/C require NVIDIA GPU. Skip GPU tests only if documentation explicitly allows CPU fallback (non-normative for K3D).

---

**Issue 3: Dependency Version Mismatches**

**Symptom**: Tests fail due to library version differences.

**Diagnosis**:
```bash
# Compare installed vs pinned requirements
diff requirements-pinned.txt <(pip freeze)
```

**Resolution**: Use `pip install -r requirements-pinned.txt` exactly (no upgrades).

---

### 6.6 Verification Report Template (Simplified)

For the full verification report template, see Evidence Validation Matrix §11.5. This simplified version is for quick peer reviews:

```markdown
# PM-KR Conformance Verification Report

**Implementation**: {Name} v{Version}
**Verifier**: {Name/Organization}
**Date**: {YYYY-MM-DD}
**Level Claimed**: {A|B|C}

## Verification Summary

- [ ] Artifacts obtained from: {URL}
- [ ] Checksums validated: {PASS|FAIL}
- [ ] Environment: Python {version}, CUDA {version}, GPU {model}
- [ ] Tests executed: {passed}/{total}

## Level A Results
- Canonicality: {PASS|FAIL}
- Reference Integrity: {PASS|FAIL}
- Determinism: {PASS|FAIL}
- Compression: {PASS|FAIL} ({ratio}%)
- Layer Composition: {PASS|FAIL}

## Level B Results (if applicable)
- Sovereignty Static: {PASS|FAIL}
- Sovereignty Runtime: {PASS|FAIL}
- Execution Determinism: {PASS|FAIL}
- Telemetry: {PASS|FAIL}

## Level C Results (if applicable)
- Provenance Coverage: {PASS|FAIL}
- Compression Metrics: {PASS|FAIL}
- Audit Trail: {PASS|FAIL}
- Conformance Report: {PASS|FAIL}

## Verdict
**Level A**: {PASS|FAIL}
**Level B**: {PASS|FAIL|NOT_VERIFIED}
**Level C**: {PASS|FAIL|NOT_VERIFIED}

## Notes
{Any discrepancies, environment issues, or clarifications}

**Verifier Signature**: {PGP fingerprint or name}
```

---

### 6.7 K3D Verification Readiness

**Current Status** (February 2026):

| Conformance Level | Artifacts Status | Verification Readiness |
|-------------------|------------------|------------------------|
| **Level A** | 🟠 Pending publication | Not ready (Q2 2026 target) |
| **Level B** | 🟡 Partial (repo tests + run logs) | Not ready (Q3 2026 target) |
| **Level C** | 🟠 In progress | Not ready (Q4 2026 target) |

**Blockers**:
- Level A: Test suite externalization (currently integrated in Knowledgeverse tests)
- Level B: Telemetry exporter (telemetry captured but needs dedicated export)
- Level C: Audit pack generator (provenance journal exists but needs consolidation)

**Timeline**:
- Q2 2026: Level A artifacts published
- Q3 2026: Level B artifacts published + first independent verification
- Q4 2026: Level C artifacts published

---

## 7. Conformance Certification

### 7.1 Self-Attestation (Current Practice)

Implementers MAY self-certify conformance by:
1. Publishing test results (all required tests passing)
2. Providing public API/endpoint for third-party validation
3. Documenting procedural language and execution environment

**K3D Self-Certification**:
- Level: **Provisional B+ (run-log verified), Level C target**
- Test Results:
  - Repo-verified: 28/28 Knowledgeverse tests
  - Run-log verified: hot-path sovereignty and benchmark snapshots
  - Pending: externalized PM-KR A/B/C conformance suite
- Public Repo: https://github.com/danielcamposramos/Knowledge3D
- Documentation: `docs/vocabulary/` (full spec suite)

### 7.2 Third-Party Certification (Future)

W3C Community Group MAY establish third-party certification by:
1. Maintaining reference test suite (independent of K3D)
2. Running conformance tests on submitted implementations
3. Publishing certification registry

**Proposed**: W3C PM-KR Conformance Registry (similar to HTML5 validator).

---

## 8. Performance Benchmarks

### 8.1 Compression Benchmarks

| Benchmark | Metric | Level A Target | Level B Target | Level C Target |
|-----------|--------|----------------|----------------|----------------|
| **Compression Ratio** | (1 - ref_size / payload_size) | >50% | >60% | >70% |
| **Reference Resolution** | avg latency (ms) | <10ms | <1ms | <0.1ms (GPU) |
| **Canonicalization** | duplicate detection rate | >95% | >99% | >99.9% |

**K3D Results** *(run-log verified unless otherwise stated)*:
- Compression: **70%** (Character Galaxy: 87.7MB → 26.3MB)
- Resolution: **<0.1ms** (PTX kernel, sub-100µs validated)
- Canonicalization: **100%** (content-addressable IDs)

### 8.2 Sovereignty Benchmarks

| Benchmark | Metric | Level B Target | Level C Target |
|-----------|--------|----------------|----------------|
| **GPU Sovereignty** | (GPU calls / solved tasks) | =1.0 | =1.0 |
| **Fallback Rate** | (fallbacks / total calls) | <1% | 0% |
| **External Deps** | count in hot path | 0 | 0 |

**K3D Results** (Math benchmark snapshot, 400 tasks):
- GPU Sovereignty: **1.0** (154 GPU calls = 154 solved tasks)
- Fallback Rate: **0%** (zero fallbacks)
- External Deps: **0** (PTX-only hot path)

---

## 9. Conclusion

**PM-KR Conformance Profiles** provide clear, testable paths for implementation:
- **Level A**: Data model + composition (5 tests)
- **Level B**: + Sovereign runtime (8 tests)
- **Level C**: + Auditability + metrics (12 tests)

**K3D Reference Snapshot**:
- Repo-verified Level B signals: Knowledgeverse integration tests + sovereign runtime checks
- Level C status: target profile defined; full third-party auditable certification pack pending

**Next Steps**: Implement conformance test suite, establish W3C certification registry, onboard early adopters.

---

## References

- PM-KR Normative Model (normative data model and invariants)
- PM-KR Problem Statement (motivation and broader impact)
- K3D Test Suites: `tests/test_knowledgeverse_*.py`, `tests/test_hot_path_sovereignty.py`, `tests/test_procedural_fonts.py`
- K3D Reference Implementation: https://github.com/danielcamposramos/Knowledge3D

---

**Document Status**: Draft Implementation Guide
**License**: CC-BY-4.0
**Version**: 1.1 (February 20, 2026)
