# WebML Gap Analysis: Where PM-KR Can Add Value

Date: 2026-03-03
Scope baseline: `webmachinelearning/proposals` README + issues #1-#16.

## 1. Current Scope Observed in Proposals Repo

From README and issue stream, current focus areas are:
1. API ideas for inference/runtime capabilities on web clients.
2. Hybrid client/cloud placement and model management.
3. Agent/tool interop (WebMCP direction).
4. Model graph/tooling portability (Graph DSL discussions).
5. Domain-specific app use cases (proofreading, fact-checking, filtering, time series).

Process-wise, proposals are incubated through issues, then often moved to dedicated repos.

## 2. Gaps PM-KR Can Fill (Evidence-Based)

### Gap A: Canonical procedural knowledge layer (beyond model files)
Observed:
- Current proposals discuss models, tools, APIs, and graph formats.
- No proposal defines a standardized procedural knowledge substrate with compositional references (form/meaning/rules/meta-rules).

PM-KR contribution:
- Procedural memory model with canonical references and compositional reuse.
- Aligns with local PM-KR docs:
  - `docs/vocabulary/PROCEDURAL_MEMORY_KR_STANDARD_SPECIFICATION.md`
  - `docs/W3C/PM_KR_NORMATIVE_MODEL.md`

### Gap B: Transparent, auditable reasoning artifacts
Observed:
- Explainability and governance concerns appear (e.g., Fact-checking API thread), but no common auditable program representation is proposed.

PM-KR contribution:
- Stack-executable procedural artifacts with explicit traceability.
- Potential bridge between agent tool calls and inspectable execution trails.
- Relevant local evidence:
  - `docs/W3C/PM_KR_EVIDENCE_VALIDATION_MATRIX.md`
  - `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md`

### Gap C: Compression-preserving knowledge reuse
Observed:
- Hybrid/model portability proposals emphasize distribution and caching.
- No dedicated proposal defines symlink-like semantic deduplication for knowledge/program elements.

PM-KR contribution:
- Reference-based composition to reduce duplication across representations.
- Local evidence includes observed and projected compression narratives:
  - `docs/W3C/PM_KR_EVIDENCE_VALIDATION_MATRIX.md`
  - `docs/CARBON_BLUEPRINT_10_YEAR_PROJECTION.md` (projection-oriented metrics)

### Gap D: Unified form-to-meaning interoperability contract
Observed:
- Proposals handle either model runtime behavior or API UX shape.
- No explicit standard contract for linking representational form to machine-usable meaning across modalities.

PM-KR contribution:
- Formalized Form -> Meaning -> Rule composition contract.
- Relevant sources:
  - `docs/vocabulary/PROCEDURAL_MEMORY_KR_STANDARD_SPECIFICATION.md`
  - `docs/W3C_PM_KR_COMMUNITY_GROUP_MISSION.md`

### Gap E: Explicit boundary model for privacy/transparency tradeoffs
Observed:
- Privacy concerns appear repeatedly (offloading, fact-checking), but with no unified boundary semantics.

PM-KR contribution:
- Boundary-aware policy framing at representation and execution levels.
- Aligns with PM-KR mission/objectives framing and social synthesis inputs.

## 3. Gaps PM-KR Should Not Claim to Fill Alone

To stay credible and complementary:
1. Hardware scheduling/offloading policy engines (primary scope in DAOP-like work).
2. Browser-native model execution APIs themselves (primary scope in WebNN work).
3. Complete agent protocol surface (primary scope in WebMCP work).

PM-KR is better positioned as:
- Knowledge representation and execution semantics layer that interoperates with those tracks.

## 4. Candidate PM-KR Alignment Targets in WebML

Most aligned current proposal threads:
1. #16 Graph DSL/portable graph format
2. #15 Dynamic offloading (weightless model description)
3. #12 WebMCP API
4. #5 Hybrid AI Exploration

Why these four:
- They already discuss portability, representation, orchestration, and implementation constraints where PM-KR can add structure.

## 5. Strategic Recommendation

Position PM-KR to WebML as:
- A complementary procedural representation profile for reusable, auditable, and compressible knowledge artifacts.
- Not a replacement for WebNN execution APIs, and not a competing agent protocol.

This reduces scope conflict risk while targeting clear unmet needs in the current proposal landscape.
