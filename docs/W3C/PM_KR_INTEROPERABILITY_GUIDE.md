# PM-KR Interoperability Guide: Migration and Integration Strategies

> **Status, 29 October 2026: Draft.** This is project input to the [PM-KR Community Group](https://www.w3.org/community/pm-kr/).
> Knowledge3D code is paused while work focuses on specifications and group deliverables, as [Daniel explained on 28 October](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0016.html).
> Requirements and examples below describe proposals, not implemented features or adopted W3C conformance rules.

> **Publication status, clarified 19 October 2026:** This is a Knowledge3D project proposal offered for PM-KR discussion. It is not an adopted Community Group report or a W3C standard. Requirement words describe the proposed model, not an approved W3C conformance regime. See the [document and evidence index](README.md).


**Document Type**: Knowledge3D candidate interoperability guide for PM-KR discussion
**Version**: 1.0
**Date**: February 20, 2026
**Authors**: Knowledge3D Project Contributors
**Status**: Draft Guide

---

## Abstract

This document provides **migration and integration strategies** for adopting Procedural Memory Knowledge Representation (PM-KR) in existing systems. It covers interoperability with traditional knowledge representation formats (RDF, OWL, JSON-LD), migration paths from embedding-only systems, and hybrid deployment strategies. Target audience: system architects, data engineers, and standards bodies.

---

## Table of Contents

1. [Interoperability Principles](#1-interoperability-principles)
2. [RDF/OWL Integration](#2-rdfowl-integration)
3. [JSON-LD Mapping](#3-json-ld-mapping)
4. [Embedding System Migration](#4-embedding-system-migration)
5. [LLM Knowledge Extraction](#5-llm-knowledge-extraction)
6. [Hybrid Deployment Patterns](#6-hybrid-deployment-patterns)
7. [Translation Loss Analysis](#7-translation-loss-analysis)
8. [Tooling and Automation](#8-tooling-and-automation)

---

## 1. Interoperability Principles

### 1.1 PM-KR Position in KR Landscape

```
┌──────────────────────────────────────────────────────────────┐
│ Knowledge Representation Standards Landscape                 │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│ Static Payload-Based:                                       │
│   ├─ RDF/RDFS (W3C)          → Triple stores, SPARQL       │
│   ├─ OWL (W3C)               → Ontologies, reasoners       │
│   ├─ JSON-LD (W3C)           → Linked data, web APIs       │
│   └─ Property Graphs (Neo4j) → Graph databases             │
│                                                              │
│ Procedural/Executable:                                      │
│   ├─ PM-KR (K3D)              → Procedural memory, GPU exec │
│   ├─ Gremlin (Apache)         → Graph traversal language   │
│   └─ Prolog                   → Logic programming          │
│                                                              │
│ Hybrid/Bridge:                                              │
│   └─ SHACL (W3C)              → Shape constraints (validation) │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**PM-KR fills a gap**: Procedural knowledge representation with compression-preserving symlink composition and dual-client consistency.

### 1.2 Interoperability Goals

1. **Bidirectional Translation**: PM-KR ↔ RDF/OWL/JSON-LD (with documented losses)
2. **Incremental Migration**: Gradual adoption without full system rewrite
3. **Hybrid Deployment**: PM-KR hot path + traditional KR for metadata/discovery
4. **Standard Alignment**: Reuse existing vocabularies (schema.org, Dublin Core, etc.)

### 1.3 Non-Goals

PM-KR does NOT aim to:
- Replace RDF/OWL for all use cases (domain-specific trade-offs)
- Preserve 100% fidelity in all translations (documented loss analysis)
- Support all OWL reasoning features (focus on procedural execution)

---

## 2. RDF/OWL Integration

### 2.1 Mapping Strategy

**Core Principle**: RDF triples map to PM-KR reference graphs with procedural semantics.

**Mapping Rules**:
```
RDF Triple (subject, predicate, object)
    ↓
PM-KR Node
    ├─ id: subject URI
    ├─ layer: inferred from predicate type
    ├─ refs: object URI (if object is resource)
    └─ meaning_program: predicate semantics (if procedural)
```

### 2.2 Example: Person Entity

**RDF/Turtle**:
```turtle
@prefix ex: <http://example.org/> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .

ex:alice a foaf:Person ;
    foaf:name "Alice Smith" ;
    foaf:knows ex:bob ;
    foaf:age 30 .

ex:bob a foaf:Person ;
    foaf:name "Bob Jones" .
```

**PM-KR Conversion**:
```json
{
  "nodes": [
    {
      "id": "http://example.org/alice",
      "layer": "meaning",
      "metadata": {
        "rdf_type": "http://xmlns.com/foaf/0.1/Person"
      },
      "refs": {
        "name_ref": ["name_alice_smith"],
        "knows_refs": ["http://example.org/bob"]
      },
      "meaning_program": "PERSON age=30"
    },
    {
      "id": "name_alice_smith",
      "layer": "form",
      "char_refs": ["char_A", "char_l", "char_i", "char_c", "char_e", "char_SPACE", "char_S", "char_m", "char_i", "char_t", "char_h"],
      "form_program": "RENDER_NAME 'Alice Smith'"
    },
    {
      "id": "http://example.org/bob",
      "layer": "meaning",
      "metadata": {
        "rdf_type": "http://xmlns.com/foaf/0.1/Person"
      },
      "refs": {
        "name_ref": ["name_bob_jones"]
      }
    }
  ]
}
```

**Key Differences**:
- **Canonicalization**: "Alice Smith" stored once (Form layer), referenced by alice node
- **Procedural Semantics**: `meaning_program` makes age executable (e.g., age comparison)
- **Reference Graph**: `foaf:knows` becomes `knows_refs` array (symlink pattern)

### 2.3 RDF → PM-KR Conversion Algorithm

**Algorithm**:
```python
def rdf_to_pmkr(rdf_graph):
    """Convert RDF graph to PM-KR nodes."""
    pm_kr_nodes = []
    canonical_literals = {}  # Deduplicate literal values

    for (subject, predicate, obj) in rdf_graph:
        # Create Meaning-layer node for subject
        subject_node = {
            "id": str(subject),
            "layer": "meaning",
            "refs": {},
            "metadata": {"rdf_type": infer_type(subject, rdf_graph)}
        }

        # Handle object
        if is_literal(obj):
            # Canonicalize literal (Form layer)
            literal_id = canonical_literals.get(str(obj))
            if not literal_id:
                literal_id = f"literal_{hash(str(obj))}"
                canonical_literals[str(obj)] = literal_id
                pm_kr_nodes.append({
                    "id": literal_id,
                    "layer": "form",
                    "form_program": f"RENDER_LITERAL '{obj}'",
                    "char_refs": string_to_char_refs(str(obj))
                })

            # Add reference
            ref_key = f"{predicate}_ref"
            subject_node["refs"][ref_key] = [literal_id]

        elif is_resource(obj):
            # Add URI reference
            ref_key = f"{predicate}_refs"
            if ref_key not in subject_node["refs"]:
                subject_node["refs"][ref_key] = []
            subject_node["refs"][ref_key].append(str(obj))

        pm_kr_nodes.append(subject_node)

    return deduplicate_nodes(pm_kr_nodes)
```

**Evaluation needed**: Measure the complete translated graph and compare preserved meaning; canonicalization alone establishes no compression percentage.

### 2.4 PM-KR → RDF Conversion

**Reverse Mapping**:
```python
def pmkr_to_rdf(pm_kr_nodes):
    """Convert PM-KR nodes to RDF triples."""
    rdf_triples = []

    for node in pm_kr_nodes:
        subject = URIRef(node["id"])

        # Layer → rdf:type
        if node["layer"] == "meaning":
            rdf_triples.append((subject, RDF.type, node["metadata"].get("rdf_type")))

        # References → predicates
        for ref_key, ref_ids in node.get("refs", {}).items():
            predicate_name = ref_key.replace("_ref", "").replace("_refs", "")
            predicate = URIRef(f"http://example.org/{predicate_name}")

            for ref_id in ref_ids:
                if is_literal_id(ref_id):
                    # Resolve Form node → literal value
                    literal_node = resolve_node(ref_id)
                    literal_value = extract_literal(literal_node["form_program"])
                    rdf_triples.append((subject, predicate, Literal(literal_value)))
                else:
                    # URI reference
                    rdf_triples.append((subject, predicate, URIRef(ref_id)))

    return rdf_triples
```

**Translation Loss**: Procedural semantics (`meaning_program`) lost in static RDF (documented below).

---

## 3. JSON-LD Mapping

### 3.1 JSON-LD Context for PM-KR

**Define PM-KR vocabulary** in JSON-LD context:

```json
{
  "@context": {
    "pmkr": "http://knowledge3d.org/pmkr/",
    "id": "@id",
    "layer": "pmkr:layer",
    "form_program": "pmkr:formProgram",
    "meaning_program": "pmkr:meaningProgram",
    "char_refs": {"@id": "pmkr:charRefs", "@type": "@id"},
    "word_refs": {"@id": "pmkr:wordRefs", "@type": "@id"},
    "symbol_refs": {"@id": "pmkr:symbolRefs", "@type": "@id"}
  }
}
```

### 3.2 Example: PM-KR Node as JSON-LD

**PM-KR Node**:
```json
{
  "@context": "http://knowledge3d.org/pmkr/context.jsonld",
  "id": "http://example.org/word_rotation",
  "layer": "meaning",
  "char_refs": [
    "http://example.org/char_r",
    "http://example.org/char_o",
    "http://example.org/char_t"
  ],
  "meaning_program": "CONCEPT_ROTATION SPATIAL_TRANSFORMATION"
}
```

**JSON-LD Expansion** (automatic):
```json
{
  "@id": "http://example.org/word_rotation",
  "http://knowledge3d.org/pmkr/layer": "meaning",
  "http://knowledge3d.org/pmkr/charRefs": [
    {"@id": "http://example.org/char_r"},
    {"@id": "http://example.org/char_o"},
    {"@id": "http://example.org/char_t"}
  ],
  "http://knowledge3d.org/pmkr/meaningProgram": "CONCEPT_ROTATION SPATIAL_TRANSFORMATION"
}
```

**Benefit**: PM-KR nodes are valid JSON-LD → interoperable with existing Semantic Web tools (SPARQL, triple stores).

### 3.3 Publishing PM-KR Knowledge Bases

**Strategy**: Publish PM-KR nodes as JSON-LD with schema.org vocabulary alignment.

**Example** (Character node with schema.org):
```json
{
  "@context": [
    "http://knowledge3d.org/pmkr/context.jsonld",
    "https://schema.org/"
  ],
  "@type": "Character",
  "id": "http://example.org/char_latin_a",
  "name": "Latin Letter A",
  "layer": "form",
  "form_program": "BEZIER_CURVE [...] PROCEDURAL_FONT_LATIN_A",
  "metadata": {
    "description": "Uppercase Latin letter A with procedural font",
    "inLanguage": "en",
    "version": "1.0"
  }
}
```

**Result**: Discoverable via schema.org search, executable via PM-KR procedural semantics.

---

## 4. Embedding System Migration

### 4.1 Challenge: Static Embeddings → Procedural Memory

**Problem**: Existing systems store embeddings (vectors) without procedural source.

**Example**:
```python
# Traditional embedding system
embeddings_db = {
    "char_a": np.array([0.1, 0.5, ...]),  # 1024-dim vector
    "char_b": np.array([0.2, 0.4, ...]),
    # No procedural source!
}
```

**PM-KR Requirement**: Embeddings must be regenerable from procedural source.

### 4.2 Migration Strategy: Reverse-Engineering Procedures

**Approach 1: Code Generation from Embeddings**

Use embedding analysis to generate approximate procedural source:

```python
def reverse_engineer_procedure(embedding, domain="char"):
    """Generate procedural source from embedding."""
    if domain == "char":
        # Cluster similar embeddings → identify canonical glyph
        canonical_glyph = find_nearest_canonical_glyph(embedding)

        # Generate procedural font reference
        procedure = f"PROCEDURAL_FONT_{canonical_glyph.script} GLYPH_{canonical_glyph.name}"

        return procedure

    elif domain == "word":
        # Decompose word embedding → char embeddings
        char_embeddings = decompose_word_embedding(embedding)

        # Map chars → canonical IDs
        char_refs = [embedding_to_char_id(ce) for ce in char_embeddings]

        return {"char_refs": char_refs}
```

**Evaluation needed**: Compare procedure reconstruction against the original glyphs and shaping behavior.
No migration result is established here.

**Approach 2: LLM-Assisted Procedure Generation**

Use LLM to generate procedural semantics:

```python
def llm_generate_procedure(concept, embedding):
    """Use LLM to generate procedural semantics."""
    prompt = f"""
    Concept: {concept}
    Embedding: {embedding[:10]}... (1024-dim)

    Generate a procedural program (RPN) that represents this concept's semantics.
    Focus on: What does this concept DO? (not just what it IS)
    """

    procedure = llm_query(prompt, model="claude-sonnet-4")

    # Validate procedure executability
    if not validate_rpn(procedure):
        raise ValueError("Generated procedure not executable")

    return procedure
```

**Example** (Word "rotation"):
```python
# Input
concept = "rotation"
embedding = word_embeddings["rotation"]  # 1024-dim vector

# LLM generates
procedure = "SPATIAL_TRANSFORMATION ANGULAR AXIS_PRESERVE ANGLE_VARIABLE"

# PM-KR node
node = {
    "id": "word_rotation",
    "layer": "meaning",
    "char_refs": ["char_r", "char_o", "char_t", ...],
    "meaning_program": procedure,
    "embeddings": {
        "original_1024d": embedding,  # Keep for validation
        "regenerated_1024d": execute_procedure_to_embedding(procedure)
    }
}

# Validate reconstruction
assert cosine_similarity(embedding, node["embeddings"]["regenerated_1024d"]) > 0.95
```

### 4.3 Hybrid Approach: Procedural + Legacy Embeddings

**Strategy**: Maintain both procedural source and legacy embeddings during migration.

```python
class HybridPMKRNode:
    """PM-KR node with legacy embedding fallback."""

    def __init__(self, id, layer, form_program=None, meaning_program=None):
        self.id = id
        self.layer = layer
        self.form_program = form_program
        self.meaning_program = meaning_program
        self.legacy_embedding = None  # Fallback

    def get_embedding(self, tier="512d", mode="procedural"):
        """Get embedding with procedural-first, fallback to legacy."""
        if mode == "procedural" and self.meaning_program:
            # Execute procedural program → embedding
            return execute_procedure_to_embedding(self.meaning_program, tier)

        elif mode == "legacy" and self.legacy_embedding is not None:
            # Return legacy embedding
            return self.legacy_embedding

        else:
            raise ValueError("No embedding source available")
```

**Migration Path**:
1. **Phase 1**: Add `meaning_program` to existing nodes, keep `legacy_embedding`
2. **Phase 2**: Validate procedural embeddings ≈ legacy embeddings (>95% similarity)
3. **Phase 3**: Remove `legacy_embedding` once procedural validated
4. **Phase 4**: Full PM-KR compliance (procedural-only)

---

## 5. LLM Knowledge Extraction

### 5.1 Challenge: Monolithic LLM → Compositional PM-KR

**Problem**: Knowledge embedded in LLM weights (175B parameters), not accessible as procedural memory.

**Solution**: Use LLM to generate PM-KR knowledge base.

### 5.2 Extraction Strategy

**Approach**: Prompt LLM to generate procedural semantics for concepts.

**Example** (Math concept extraction):

```python
def extract_math_concept(concept_name):
    """Extract math concept as PM-KR node."""
    prompt = f"""
    Concept: {concept_name} (mathematics)

    Generate:
    1. Visual representation (RPN drawing program)
    2. Semantic meaning (RPN mathematical operation)
    3. Transformation rules (when/how to use)

    Format: JSON with form_program, meaning_program, metadata.
    """

    response = llm_query(prompt, model="gpt-4")
    node_json = parse_json(response)

    # Validate executability
    validate_rpn(node_json["form_program"])
    validate_rpn(node_json["meaning_program"])

    return PMKRNode(**node_json)
```

**Evaluation needed**: Retain the generated knowledge, its source and validation results.
Executability alone does not establish factual accuracy.
The proposed K3D reasoning boundary excludes LLM calls; verify it with execution traces.

### 5.3 Distillation: LLM → Sovereign PM-KR

**Goal**: Extract LLM knowledge, then eliminate LLM dependency.

**Strategy**:
1. **Extraction Phase** (use LLM):
   - Generate procedural knowledge base (RPN programs)
   - Validate executability (PTX kernels)
   - Deduplicate canonicals (content-addressable)

2. **Crystallization Phase** (sovereign):
   - Store procedural programs in Galaxy
   - Build reference graphs (symlink composition)
   - Validate deterministic execution (no LLM needed)

3. **Sovereignty Phase** (zero LLM):
   - All inference via PTX kernels
   - LLM only used for periodic knowledge updates (offline)

**Evaluation needed**: Publish the input corpus, generated procedures and complete traces before claiming successful distillation or execution isolation.

---

## 6. Hybrid Deployment Patterns

### 6.1 Pattern 1: PM-KR Hot Path + RDF Metadata

**Architecture**:
```
┌──────────────────────────────────────────┐
│ User Query                               │
└──────────────┬───────────────────────────┘
               ↓
     ┌─────────────────────┐
     │ RDF Triple Store    │  Discovery, metadata, indexing
     │ (SPARQL queries)    │
     └─────────┬───────────┘
               ↓ (retrieve PM-KR node IDs)
     ┌─────────────────────┐
     │ PM-KR Hot Path      │  Sovereign execution (PTX kernels)
     │ (procedural exec)   │
     └─────────┬───────────┘
               ↓
     ┌─────────────────────┐
     │ Result              │
     └─────────────────────┘
```

**Benefits**:
- **RDF**: Fast SPARQL queries for discovery (mature tooling)
- **PM-KR**: Sovereign execution for reasoning (zero external deps)

**Example Use Case**: Semantic search (RDF) → procedural reasoning (PM-KR).

### 6.2 Pattern 2: PM-KR Core + JSON-LD Publishing

**Architecture**:
```
┌──────────────────────────────────────────┐
│ PM-KR Knowledge Base                     │  Canonical source (procedural)
└──────────────┬───────────────────────────┘
               ↓ (export)
     ┌─────────────────────┐
     │ JSON-LD Published   │  Web-accessible (schema.org)
     │ (with @context)     │
     └─────────┬───────────┘
               ↓ (discovered by)
     ┌─────────────────────┐
     │ External Systems    │  RDF parsers, SPARQL endpoints
     └─────────────────────┘
```

**Benefits**:
- **PM-KR**: Compression + procedural semantics (internal)
- **JSON-LD**: Interoperability + discovery (external)

**Example Use Case**: Publish PM-KR knowledge bases to Semantic Web.

### 6.3 Pattern 3: Gradual Migration (Dual-Format Storage)

**Architecture**:
```
┌──────────────────────────────────────────┐
│ Legacy RDF Data                          │
└──────────────┬───────────────────────────┘
               ↓ (migrate incrementally)
     ┌─────────────────────┐
     │ Hybrid Storage      │
     │ ├─ RDF (legacy)     │  70% of data (not yet migrated)
     │ └─ PM-KR (new)      │  30% of data (high-value, procedural)
     └─────────┬───────────┘
               ↓ (unified API)
     ┌─────────────────────┐
     │ Application Layer   │  Abstract over both formats
     └─────────────────────┘
```

**Benefits**:
- **No big-bang migration**: Gradual adoption
- **Risk mitigation**: Keep legacy RDF as fallback
- **Validation**: Compare PM-KR vs RDF results

**Migration Priority**: High-frequency queries → PM-KR first (maximize ROI).

---

## 7. Translation Loss Analysis

### 7.1 PM-KR → RDF Losses

**What's Lost**:
1. **Procedural Semantics**: `meaning_program` → static literals (no execution)
2. **Representation size**: Expanding references may change size; RDF can also use shared identifiers.
3. **Dual-Client Consistency**: Procedural source → separate representations

**Example**:
```python
# PM-KR (procedural)
node = {
    "id": "word_rotation",
    "char_refs": ["char_r", "char_o", "char_t", ...],  # Symlink refs
    "meaning_program": "SPATIAL_TRANSFORMATION ANGULAR"  # Executable
}

# RDF (static)
triples = [
    (word_rotation, rdfs:label, "rotation"),  # Duplicate string!
    (word_rotation, rdf:type, Concept),       # No execution semantics
]
```

**Impact to evaluate**: The proposed export does not preserve procedural execution; storage changes depend on the chosen RDF encoding.

### 7.2 RDF → PM-KR Losses

**What's Lost**:
1. **OWL Reasoning**: Transitive closures, class hierarchies (static inference)
2. **SPARQL Federation**: Distributed queries across endpoints
3. **SHACL Validation**: Shape constraints (static schema validation)

**Mitigation**:
- **OWL Reasoning**: Materialize inferences as PM-KR rules (procedural equivalents)
- **SPARQL Federation**: Implement portal federation (distributed PM-KR knowledge bases)
- **SHACL**: Implement validation as PM-KR meta-rules

**Example** (OWL Transitivity → PM-KR Rule):
```turtle
# OWL (static)
:knows a owl:TransitiveProperty .

# PM-KR (procedural)
{
  "id": "rule_transitive_knows",
  "layer": "meta_rules",
  "transformation_rpn": "IF knows(A, B) AND knows(B, C) THEN knows(A, C)",
  "metadata": {"derived_from": "owl:TransitiveProperty"}
}
```

### 7.3 Embedding Systems → PM-KR Losses

**What's Lost**:
1. **Exact embeddings**: A procedural reconstruction may differ from its source vector; measure the error for the actual use case.
2. **Black-Box Models**: Internal LLM states not reproducible

**Mitigation**:
- Keep legacy embeddings as validation baseline (hybrid approach)
- Define a use-case-specific error criterion; vector similarity alone does not prove semantic equivalence.
- Document reconstruction error metrics

---

## 8. Tooling and Automation

These are proposed interfaces and pseudocode, not installed commands or measured output.

### 8.1 Conversion Tools

**Tool 1: RDF → PM-KR Converter**

```python
# CLI: rdf2pmkr
$ rdf2pmkr input.ttl --output pmkr_nodes.jsonl --canonicalize

# Python API
from pm_kr.converters import RDFtoPMKR

converter = RDFtoPMKR()
pm_kr_nodes = converter.convert("input.ttl", canonicalize=True)
pm_kr_nodes.save("pmkr_nodes.jsonl")

# Report compression
print(f"Compression: {converter.compression_ratio:.1%}")
# Report measured sizes for the supplied input.
```

**Tool 2: PM-KR → JSON-LD Publisher**

```python
# CLI: pmkr2jsonld
$ pmkr2jsonld pmkr_nodes.jsonl --context http://knowledge3d.org/pmkr/context.jsonld

# Python API
from pm_kr.converters import PMKRtoJSONLD

publisher = PMKRtoJSONLD()
jsonld_docs = publisher.convert("pmkr_nodes.jsonl", context_url="http://knowledge3d.org/pmkr/context.jsonld")
jsonld_docs.save("output.jsonld")
```

### 8.2 Validation Tools

**Tool 3: Conformance Validator**

```python
# CLI: pmkr-validate
$ pmkr-validate pmkr_nodes.jsonl --level A

# Output:
# ✅ Canonicality:

# Report each tested invariant, failures and untested cases.
# No conformance result is supplied in this example.

```

**Tool 4: Translation Loss Analyzer**

```python
# CLI: pmkr-compare
$ pmkr-compare original.rdf translated.pmkr --metrics all

# Output:
# Size:

# Report measured size, preserved semantics and all known translation losses.
# No result is supplied in this example.

```

### 8.3 Migration Automation

**Tool 5: Incremental Migration Planner**

```python
from pm_kr.migration import MigrationPlanner

planner = MigrationPlanner()
planner.load_rdf("legacy_data.rdf")

# Analyze migration candidates
candidates = planner.analyze(priority="high_frequency")

# Generate migration plan
plan = planner.create_plan(
    target_coverage=0.3,  # Migrate 30% of data (high-value)
    validation_threshold=0.95  # >95% similarity required
)

# Execute migration (phase-by-phase)
for phase in plan.phases:
    phase.execute(dry_run=False)
    phase.validate()
    phase.report()

# Final report
print(plan.summary())
# Output:
# Phase 1:

# Report the actual migrated entries and validation failures.
# No migration result is supplied in this example.

```

---

## 9. Case Studies

### 9.1 K3D Migration: TTF Fonts → Procedural Characters

This is a candidate study, not a reported completed migration.
Existing fonts already carry glyph outlines and may carry instructions; they are the precedent for the proposal, as [Daniel explained](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0018.html).
An evaluation would extract glyph outlines, encode the selected procedures, retain character identities and shaping information, and compare reconstruction against the original font.
It must publish source assets, output artifacts, runtime versions and differences.

### 9.2 K3D Migration: Math Benchmark Augmentation

This is a candidate study, not a reported completed migration.
Extract distinct tags, assign explicit identities, replace reuse with references and check meaning in context.
The report must retain original inputs and compare both size and task behavior.
Embedding similarity alone does not prove equivalent meaning.

---

## 10. Community Group Pathways

### 10.1 W3C Community Group Path

The [PM-KR Community Group](https://www.w3.org/community/pm-kr/) is the venue for discussion and shared deliverables.
This project proposes a model, interoperability examples and a test suite for consideration.
A Community Group does not publish W3C Recommendations; any Working Group standardization would require its own charter and process ([W3C FAQ](https://www.w3.org/community/about/faq/)).
There is no Recommendation schedule established here.

### 10.2 Industry Adoption Strategy

Potential reviewers include knowledge-graph developers, AI researchers and spatial-interface implementers.
Candidate experiments include a graph converter, a linked-data publisher and a shared human/AI view.
No vendor partnership, deployment or pilot participation is claimed.

---

## 11. Conclusion

The guide proposes mappings alongside RDF, OWL and JSON-LD.
The mapping examples need executable tools, explicit loss analysis and round-trip tests before compatibility can be claimed.
The next step is one bounded exchange example with published inputs and independently checked outputs.

---

## References

- PM-KR Normative Model (data model and invariants)
- PM-KR Conformance Profiles (implementation levels)
- RDF 1.1 Specification (W3C)
- OWL 2 Web Ontology Language (W3C)
- JSON-LD 1.1 Specification (W3C)
- [K3D implementation repository](https://github.com/danielcamposramos/Knowledge3D), for source inspection rather than a conformance verdict

---

**Document Status**: Draft Interoperability Guide
**License**: CC-BY-4.0
**Version**: 1.0 (February 20, 2026)
