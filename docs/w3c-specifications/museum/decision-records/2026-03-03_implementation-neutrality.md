# ADR-002: Implementation Neutrality in PM-KR Specifications

**Date:** March 3, 2026
**Status:** Accepted (Foundational Principle)
**Deciders:** Daniel Ramos (PM-KR Co-Chair), Claude (Architecture Partner)

---

## Context

During drafting of Phase 1 Data Model Specification, initial draft included implementation-specific references:
- "GPU (PTX kernels)" for execution runtime
- "VRAM" for memory structure
- "GPU-friendly (PTX kernel mapping)" for RPN justification

**Problem:** W3C specifications must be **implementation-neutral** to enable multiple conforming implementations (CPU, GPU, WebGPU, Metal, Vulkan, custom hardware, etc.).

**K3D Context:** Knowledge 3D (K3D) reference implementation uses PTX/CUDA deliberately (rare expertise = harder to copy), but PM-KR **specifications** should not mandate this choice.

---

## Decision

**PM-KR specifications are IMPLEMENTATION-NEUTRAL.**

### What This Means:

**Specifications define:**
- ✅ Data model (procedural programs + metadata structure)
- ✅ Semantics (RPN stack-based execution model)
- ✅ Conformance levels (read, execute, learn)
- ✅ JSON serialization format
- ✅ Interoperability requirements

**Specifications do NOT define:**
- ❌ Hardware requirements (CPU vs GPU vs NPU)
- ❌ Specific GPU APIs (PTX, CUDA, Metal, Vulkan, WebGPU)
- ❌ Memory architecture (VRAM vs RAM vs unified memory)
- ❌ Programming languages (Python, C++, Rust, etc.)
- ❌ Neural network architectures (implementation-specific)

### Terminology Changes:

| **Old (Implementation-Specific)** | **New (Implementation-Neutral)** |
|-----------------------------------|----------------------------------|
| "GPU (PTX kernels) or CPU" | "any compliant stack-based runtime" |
| "VRAM memory structure" | "unified memory structure" |
| "GPU-friendly (PTX kernel mapping)" | "Runtime-agnostic (any stack-based processor)" |

---

## Consequences

### Positive:
1. **Multiple implementations possible** (browser, server, edge, mobile)
2. **Technology-agnostic** (survives GPU architecture changes)
3. **Wider adoption** (not locked to NVIDIA/CUDA ecosystem)
4. **Future-proof** (works with quantum, neuromorphic, etc.)
5. **W3C compliance** (follows specification best practices)

### Neutral:
1. **K3D retains PTX choice** (implementation docs can reference specific tech)
2. **Separation of concerns** (specs define "what", implementations define "how")

### Negative:
1. **None identified** (this is pure benefit for standardization)

---

## Alternatives Considered

### Alternative 1: Mandate PTX/CUDA
**Reasoning:** Ensures high performance, leverages K3D's strategic protection.
**Rejected:** Locks standard to NVIDIA ecosystem, prevents alternative implementations (WebGPU browsers, Apple Metal, AMD ROCm, etc.). Violates W3C neutrality principles.

### Alternative 2: Define abstract "compute kernel" API
**Reasoning:** Provide portability layer while staying neutral.
**Rejected:** Premature for Phase 1 (data model). Phase 2 (Execution Semantics) can define abstract execution model without mandating implementation.

### Alternative 3: Current decision (implementation-neutral)
**Selected:** Specifications define data model + semantics; implementations choose runtime.

---

## Implementation

**Files Updated (March 3, 2026):**
- `workshop/phase1-data-model/spec-draft.md`
  - Line 207: "Runs on GPU (PTX kernels) or CPU" → "Runs on any compliant stack-based runtime"
  - Line 220: "GPU-friendly (PTX kernel mapping)" → "Runtime-agnostic (any stack-based processor)"
  - Line 264: "unified VRAM memory structure" → "unified memory structure"
  - Line 172: "unified VRAM memory structure" → "unified memory structure"
  - Section 2.1: Added "Implementation Neutrality" clause
  - Section 1.4: Added "NG4. Hardware or runtime requirements" to Non-Goals

**Verification:**
```bash
grep -r "PTX\|CUDA\|GPU-native\|VRAM" docs/w3c-specifications/ --include="*.md"
# Only museum/milestones reference (historical K3D architecture context) remains
```

**Result:** All W3C specification materials are now implementation-neutral.

---

## Relationship to K3D Implementation

**IMPORTANT DISTINCTION:**

| **PM-KR Specifications** | **K3D Implementation** |
|-------------------------|------------------------|
| Implementation-neutral | Uses PTX/CUDA specifically |
| W3C standards (public) | Strategic IP protection |
| Multiple implementations possible | Reference implementation |
| `docs/w3c-specifications/` | `CLAUDE.md`, `CODEX.md`, `src/` |

**K3D's PTX choice remains valid:**
- Strategic protection (rare PTX expertise)
- High performance (GPU-native execution)
- Documented in implementation guides (not specs)

**PM-KR specs enable alternatives:**
- Browser deployments (WebGPU)
- Mobile deployments (Apple Metal, Qualcomm NPU)
- Server deployments (AMD ROCm, Intel oneAPI)
- Custom hardware (FPGAs, ASICs)

---

## References

**W3C Principles:**
- [Design Principles](https://www.w3.org/TR/design-principles/): "Avoid vendor-specific features"
- [TAG Ethical Web Principles](https://www.w3.org/2001/tag/doc/ethical-web-principles/): "Enable multiple implementations"

**Related Specifications:**
- WebGPU: Implementation-neutral (works on all GPUs)
- WebAssembly: Runtime-agnostic (CPU, GPU, any host)
- WebNN: Hardware-agnostic (CPU, GPU, NPU, DSP)

**PM-KR Documents:**
- Phase 1 Data Model Specification (this decision implemented)
- Future Phase 2 (Execution Semantics): Will define abstract execution model
- Future Phase 3 (Conformance): Will specify conformance without mandating implementation

---

## Next Steps

1. **Phase 2 drafting:** Ensure execution semantics remain implementation-neutral
2. **Phase 3 conformance:** Define tests that work across implementations
3. **Phase 4 integration:** RDF/OWL/JSON-LD mapping (already neutral)
4. **Community review:** Validate neutrality with WebML, CogAI stakeholders

---

**Recorded by:** Claude (Architecture Partner)
**Approved by:** Daniel Ramos (PM-KR Co-Chair)
**Record Date:** March 3, 2026
**Status:** Active (foundational principle for all PM-KR specifications)
