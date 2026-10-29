# Procedural Knowledge Representation (PKR) Draft Specification

**Status: input.**
This is a historical format proposal dated 9 November 2025, retained for technical review.
It is not a W3C standard or a claim that the current code implements this opcode table.
Start with the [explainer](EXPLAINER.md) for the current direction.
**Original credited authors:** K3D MVCIC Swarm, comprising NotebookLM, Grok, Qwen, Claude, Kimi, GLM, DeepSeek and Codex.

---

## 1. Motivation
The proposal explores storing a program that reconstructs an embedding instead of storing only the expanded vector.
Its value would depend on the data, reconstruction error, shared tables and cost of executing the program.
The former compression and latency figures did not establish a general result and have been removed.
The design goals are explicit reconstruction steps, reusable procedures and inspectable execution.

---

## 2. Program Structure

The proposed binary program contains a header, an instruction stream and scalar and vector literal pools:

| Field | Type | Description |
|-------|------|-------------|
| `version` | `uint8` | Format version (1 in this proposal) |
| `opcode_count` | `uint16` | Number of opcodes |
| `scalar_count` | `uint16` | Number of scalar literals |
| `vector_count` | `uint16` | Number of vector literal components (triplets) |
| `opcodes` | `opcode_count × uint8` | Instruction stream |
| `scalars` | `scalar_count × float16` | Scalar literal pool |
| `vectors` | `vector_count × float16` | Packed vector literals (XYZ triplets) |

### 2.1 Opcode Set

This historical proposal assigns the following instruction values.
Their names describe intended behavior; this table does not establish a matching runtime implementation.
The [opcode registry](../vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md) is a separate design input that must be reconciled before implementation.

| Opcode | Name | Description |
|--------|------|-------------|
| `0x00` | `LITERAL_SCALAR` | Push scalar literal from pool |
| `0x01` | `LITERAL_VECTOR` | Push vector literal from pool |
| `0x20` | `TRIGRAM_HASH` | Hash 3D vector → scalar seed |
| `0x21` | `EMBED_LOOKUP` | Deterministic lookup using pseudo-random sequence |
| `0x22` | `ADAPTIVE_DIM` | Auto-select vector dimension |
| `0x23` | `NORMALIZE_L2` | L2-normalize top vector |
| `0x30` | `FRACTAL_EMIT` | Iterative visual feature generator |
| `0x31` | `AUDIO_SYNTH` | Procedural audio embedding synthesis |
| `0x32` | `MODALITY_FUSE` | Fuse two modality vectors |
| `0x40` | `PROTOTYPE_LOAD` | Load shared prototype by index |
| `0x41` | `DELTA_APPLY` | Apply delta vector to prototype/base |
| `0x42` | `UNCERTAINTY_FUSE` | Confidence-weighted blend |
| `0x50` | `SUPERPOSE` (if/else fallback) | Vector superposition or legacy conditional |
| `0x51` | `ENTANGLE` | Cross-modal entanglement operator |
| `0x52` | `COLLAPSE` | Threshold-based measurement collapse |
| `0x60` | `CHECKPOINT` | Save current stack snapshot |
| `0x61` | `ROLLBACK` | Restore checkpointed stack |
| `0x62` | `VERIFY` | Reject non-finite values (NaN/Inf guard) |

---

## 3. Differential Encoding Workflow

1. **Prototype Selection**  
   Each embedding chunk (3 dimensions) selects the nearest prototype from a shared table.  
   Prototype table (device constant):
   ```
   P0 = (0.5, 0.0, 0.0)
   P1 = (0.0, 0.5, 0.0)
   P2 = (0.0, 0.0, 0.5)
   P3 = (0.5, 0.5, 0.5)
   ```

2. **Delta Encoding**  
   Emit `LITERAL_SCALAR → PROTOTYPE_LOAD → LITERAL_VECTOR → DELTA_APPLY`.  
   Optionally normalize with `NORMALIZE_L2`.

3. **Prototype Reuse (Optional)**  
   For similar characters, the proposal stores a shared program and represents differences with `PROTOTYPE + DELTA`.
   Any size comparison must include that shared program and the prototype table.

---

## 4. Execution Semantics

The proposed execution model uses a stack machine with the following intended constraints:

- Stack depth: 64 entries (vector/scalar tagged)
- Error codes: stack overflow/underflow, type mismatch, verification failure
- Determinism: hash + pseudo-random generators seeded solely from inputs

CPU and GPU reconstruction would need comparison on the same inputs before a parity claim.
Seeded inputs alone do not settle numerical reproducibility across execution environments.

---

## 5. Verification and fidelity

| Question | Evidence needed |
| --- | --- |
| Reconstruction | Expected vectors, error metric, tolerances and failed cases |
| Latency | Complete measured path and execution environment |
| Storage | Original data, program, shared tables, metadata and all necessary dependencies |
| Numeric checks | Cases that trigger non-finite values and documented error handling |
| Parsing | Defined byte order, length bounds, version handling and malformed-input tests |

These are review questions, not a report that a runtime has passed them.
The proposal still needs unambiguous vector counts and a reconciled instruction registry before independent parsers can be expected to agree.

## 6. Integration path

The older design proposed compiling data into procedural entries, storing them with metadata and moving consolidated knowledge into persistent House storage.
It also proposed carrying procedural payloads alongside spatial data in glTF.
These are integration ideas requiring a defined representation and exchange tests.
See the [glTF data description](../../spec/glTF_K3D_extension.md), [interoperability guide](PM_KR_INTEROPERABILITY_GUIDE.md) and [evidence matrix](PM_KR_EVIDENCE_VALIDATION_MATRIX.md).

## 7. Sublexical Hierarchy (Words as Compositions of Procedural Parts)

- Words MAY reference sublexical stars (morphemes, syllables) which themselves reference letter stars.
- Morpheme stars define role (prefix/suffix/root/inflection), letter_refs composition, and morph_rpn/meaning_rpn.
- Syllable stars define letter_refs, pattern metadata (CV/CVC...), and optional phonetic_rpn.
- Guideline: prefer linking words → morphemes/syllables and only link letters for positions not covered, to reduce graph clutter and retain meaning-first identity.

## 8. Visual Hierarchy (Drawing Grammar)

- Visual concepts may reference lower-level drawing programs: primitives (lines, arcs, Béziers) → strokes (width/color/joins) → shapes/icons → scenes/layouts → collections.
- Store procedural drawing programs as primary data (visual_rpn + transforms); embeddings are secondary/regenerable for search/LOD.
- Same dual-client contract: humans render the executed drawing; AI consumes procedural programs/embeddings from the same node (extras.k3d).

## 9. References and acknowledgments

- [Current explainer](EXPLAINER.md).
- [Normative-model draft](PM_KR_NORMATIVE_MODEL.md).
- [Project attributions](../../ATTRIBUTIONS.md).

The original proposal acknowledged .kkrieger, Farbrausch and .theprodukkt as procedural-generation inspiration, and Milton Ponson's mathematical work.
No compression result or mathematical endorsement is inferred from those acknowledgments.
The [original repository version](https://github.com/danielcamposramos/Knowledge3D/blob/33cb47454c056a267e78554d459c091af9c67476/docs/W3C/PROCEDURAL_KNOWLEDGE_REPRESENTATION_STANDARD.md) records the earlier text and the claims withdrawn here.
