# Paper D — Form → Meaning: Four-Layer Architecture — Skeleton Draft

**Date**: 2026-04-19
**Authors**: Daniel Campos Ramos (first — architectural origination), PM-KR co-authors TBD
**Target venue**: companion preprint to Paper A (arXiv cs.AI or cs.CL; venue TBD)
**Status**: Skeleton — section-by-section targets, contrast-with-prior-art tables.
**External validation**: Acknowledged as novel by NLP chief professor at [institution redacted pending consent], early PM-KR member, shared the PM-KR group LinkedIn invitation. See [`feedback_form_meaning_externally_validated.md`](../../home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/feedback_form_meaning_externally_validated.md).
**Related**: ATTRIBUTIONS.md §6.4 (post-renumber), [`FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md`](../docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md), [`DUAL_CLIENT_CONTRACT_SPECIFICATION.md`](../docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md).

---

## Working Title

**Form → Meaning: A Four-Layer Architecture for Language-Agnostic Knowledge Representation**

Alternates:
- *Meaning First: Inverting the RDF/OWL Stack for Sovereign Cognition*
- *The Layer the Web Ontology Is Missing: Meaning as the Canonical Axis, Surface Form as the Symlink*

---

## Abstract (≤ 175 words)

**Target 4-sentence arc:**

1. **Problem.** Web-ontology frameworks (RDF, OWL, W3C Framework Ontology) organise knowledge by *surface form* — the same concept in English and Portuguese is two different resources, related by annotation. This flattens meaning into a property of strings.
2. **Proposal.** K3D inverts the stack with a four-layer architecture: **(Layer 1) Form** — surface encodings (glyphs, Bézier paths, audio spectrograms); **(Layer 2) Meaning** — canonical language-agnostic stars to which all surface forms symlink; **(Layer 3) Rules** — transformation grammars (RPN) expressed between meaning stars; **(Layer 4) Meta-Rules** — rules about rules (defeasibility, priority, superior_to relations). Reasoning operates at Layers 2-4; Layer 1 is the interface to humans and audio devices.
3. **External acknowledgement.** An NLP chief professor at [institution redacted] has independently flagged the Form → Meaning inversion as a genuine novel organising principle (PM-KR early member, shared group LinkedIn invitation; see ATTRIBUTIONS.md §6.4).
4. **Claim.** Organising by meaning rather than by form is not a re-labelling of existing ontologies — it is the architectural pre-condition for the dual-client contract (humans and AI share the same canonical referent across all languages and modalities).

Word budget: ~185 words; trim if venue enforces 150.

---

## §1 Introduction (~0.75 page)

### §1.1 Hook

*The RDF triple `(ex:dog, rdf:label, "chien"@fr)` treats the French word as an annotation on the English URI `ex:dog`. This reading is convenient but backwards: the dog-meaning preceded both words. K3D writes the triple the other way around.*

### §1.2 Motivation

Four problems with surface-first organisation:

1. **Language bias.** Picking English as the primary URI axis silently centres anglophone concept structure. Non-anglophone concept-meanings get coerced through English.
2. **Dual-client failure.** A human sees `"chien"` as a legitimate lemma; an AI reasoner sees it as an annotation on `ex:dog`. The two clients don't see the same object.
3. **Symbolic vs statistical gap.** Symbolic KR (RDF/OWL) and distributional semantics (embeddings) sit in parallel stacks that don't compose cleanly because one organises by *form* and the other organises by *latent geometry* with no canonical identity axis.
4. **Multi-modal sprawl.** Images, audio, procedural primitives all get mapped to text strings (captions, descriptions). The text string becomes the concept identifier; the multi-modal signal becomes a secondary property.

### §1.3 Contributions

> **D.1** — Formal statement of the four-layer architecture (Form, Meaning, Rules, Meta-Rules) with interface contracts.
>
> **D.2** — The bidirectional-symlink normalisation (`feedback_bidirectional_symlinks_norm.md`): every multilingual/multi-modal surface form points at, and is pointed at by, a single canonical meaning star.
>
> **D.3** — A worked dual-client example showing humans and the AI both resolve to the same canonical referent while seeing language/modality-appropriate surface.
>
> **D.4** — Independent external acknowledgement (PM-KR NLP chief professor) that the inversion is architecturally novel, not cosmetic.

### §1.4 Companion positioning

Paper A §2.2 (TRM-as-Avatar) mentions "meaning-centric stars"; Paper B (semantic gravity) uses meaning-distance `d` in the Form → Meaning Layer 2 axis; Paper D formalises the axis itself. Paper C §2.1 Level 4 (Nodes) is instantiated *as* Form → Meaning stars.

---

## §2 Background and Prior Art (~0.75 page)

### §2.1 The RDF/OWL surface-first convention

RDF (Lassila & Swick 1999), OWL (Bechhofer et al. 2004), SKOS. In these frameworks, the URI *is* the identifier — and in practice URIs are minted in a source language (typically English). Annotation properties (`rdfs:label`, `skos:prefLabel`) carry translations. **Consequence:** translations are annotations on a (usually English) anchor.

### §2.2 Cyc and its compromise

Cyc (Lenat 1995) attempted language-neutral micro-theories with `CycL` predicates. **Why it didn't scale:** editorial effort required to maintain language neutrality; in practice `#$Dog` and `#$Cão` end up as synonyms with English primacy.

### §2.3 Distributional semantics

Word2Vec (Mikolov et al. 2013), BERT (Devlin et al. 2018), modern LLM embeddings. Latent-geometry organisation without canonical identity. **Contrast:** these systems *can* compute cross-lingual similarity but cannot state "this word in French and this word in English refer to the exact same canonical concept" without an auxiliary ontology. K3D's canonical stars fill this gap.

### §2.4 Framework ontology efforts

W3C Framework Ontology, BFO (Basic Formal Ontology), IFF (Information Flow Framework). **Contrast:** these provide upper-level categories in a form-first framing. K3D's four layers are orthogonal to upper-ontology categorisation.

### §2.5 The gap this paper fills

A substrate-level organising principle that treats meaning as the canonical axis and surface form as bidirectional symlink — enabling dual-client computation (human avatar, AI avatar) over the same persistent objects. Not a new ontology; a new axis choice.

---

## §3 The Four-Layer Architecture (~1.5 pages — technical core)

### Figure 1 — Four-layer stack (half-page)

Four horizontal layers, arrows showing:
- Layer 1 Form → Layer 2 Meaning: "resolves to"
- Layer 2 Meaning → Layer 3 Rules: "participates in"
- Layer 3 Rules → Layer 4 Meta-Rules: "governed by"
- And symmetric bidirectional symlinks at each interface.

### §3.1 Layer 1 — Form

**What it contains.** Surface encodings: glyph Bézier paths, character IDs with language/font/pronunciation metadata, word-level character sequences (references, not duplicates), audio spectrograms, raster images, procedural drawing primitives (LINE, CIRCLE, RECT as RPN programs).

**Why Layer 1 is Form, not Surface.** "Surface" implies shallowness; "Form" is the Aristotelian/Scholastic term preserved in the architecture. Form is what is *perceived*; Meaning is what is *referred to*. The classical distinction is load-bearing here.

**Procedural vs static.** Glyphs are Bézier programs, not images. Words are character sequences, not strings. Images are procedural compositions when possible. Per `DUAL_CLIENT_CONTRACT_SPECIFICATION.md` §1.6: *save information* — reference, don't duplicate.

### §3.2 Layer 2 — Meaning

**What it contains.** Canonical *stars* (meaning-centric concepts), each with:
- Canonical ID (content-hash or authority-assigned).
- Meaning-mass `M` (Paper B §3.3 definition).
- Bidirectional symlinks to all Layer 1 surface forms (multilingual, multi-modal).
- Optional RPN program if the meaning is evaluable.

**Load-bearing property.** One star, all languages, all modalities. Daniel's corrected vision: *"meaning-first multilingualism, all writing/numeral systems"* (`project_universal_knowledge_vision.md`). The English word and the Portuguese word for *dog* are two Layer 1 surface symlinks into one Layer 2 star.

### §3.3 Layer 3 — Rules

**What it contains.** Transformation grammars expressed as RPN programs between Meaning stars. Grammar Galaxy entries (Paper A §2.1 context) live at Layer 3.

**Example.** "Number-candidate" rule from `RETE_AT_OPCODE_LEVEL.md`:

```
R1: IF token(t, kind=DIGIT) AND next(t, t') AND token(t', kind=DIGIT)
    THEN emit(number_candidate, span=[t, t'])
```

This is a Layer 3 rule referencing Layer 2 concepts (`DIGIT`, `next`, `number_candidate`), compiled to Layer 7 opcodes (0xE0 RETE_ALPHA_TEST, 0xE1 RETE_BETA_JOIN, 0xE2 AGENDA_INSERT).

### §3.4 Layer 4 — Meta-Rules

**What it contains.** Rules about rules: defeasibility, priority, `superior_to` relations, `rule_strength`, `trust_weight`. Extends `GrammarRule` per `feedback_exploratory_grammar_deferred.md`. Provides the conflict-resolution machinery when multiple Layer 3 rules fire on the same input.

**Why separate from Layer 3.** Mixing rule logic with rule-priority logic creates a hierarchy-collapse problem: whoever can write rules can also override all prior rules. Separating meta-rules gives a clean audit surface (who approved this priority? when?).

### §3.5 Interface contracts between layers

Each interface is specified by a symlink convention:
- **Layer 1 ↔ Layer 2:** every Form element has a `meaning_id` symlink to a Meaning star; every Meaning star maintains a set of Form symlinks.
- **Layer 2 ↔ Layer 3:** Rules reference Meaning star IDs; Meaning stars maintain an in-degree set of rules that mention them.
- **Layer 3 ↔ Layer 4:** Meta-Rules reference Rule IDs; Rules carry a `governed_by` meta-rule reference.

Bidirectionality is an invariant (`feedback_bidirectional_symlinks_norm.md`). A unidirectional link violates the architecture.

### §3.6 What lives where — worked example

Table for the dog-concept:

| Layer | What it holds |
|-------|---------------|
| Layer 1 | `"dog"@en` (string), `"cão"@pt` (string), `/k9`/ (phoneme sequence), 🐕 (glyph), `dog_bark.wav` (audio spectrogram) |
| Layer 2 | Canonical star `S_DOG` — `meaning_mass`, quality-dim vector, symlink set referencing all Layer 1 forms above |
| Layer 3 | Rules: "dog is-a mammal", "dog has-property bark", etc. — each an RPN program over Meaning stars |
| Layer 4 | Meta-rules: "biological-kind rules outrank cultural-kind rules" (defeasible priority) |

Reasoning (Paper A, Paper B) operates on Layers 2-4. Display (to humans, via the Memory Tablet or House avatar) renders Layer 1 forms appropriate to the viewer's language/modality.

---

## §4 Why the Inversion Matters (~0.5 page)

### §4.1 Dual-client contract enabled

Paper A's dual-client property (humans and AI share the House) is *only* possible because both clients resolve to the same Layer 2 star. If the canonical identifier were a surface form in a specific language, one client would always be second-class.

### §4.2 Sovereignty enabled

Paper A's C1 (zero NumPy in hot path) requires reasoning over identity-stable objects. A substrate where identity is language-dependent forces every reasoning step to do surface-form normalisation — which is exactly the Python/regex orchestration K3D has sworn off.

### §4.3 Semantic gravity enabled

Paper B's `F = T · M · M / d²` is computed between stars (Layer 2), not between surface forms. Without the Layer 2 canonical axis, meaning-distance is an embedding-distance in a specific language's vector space — which is a different quantity.

### §4.4 Cross-modal reasoning enabled

Visual (Drawing Galaxy), audio (Audio Galaxy), and textual knowledge all resolve to Layer 2 stars. A reasoning step can move across modalities by staying at Layer 2 — no format translation required.

---

## §5 External Acknowledgement and Independent Evidence (~0.5 page)

### §5.1 External validation

An NLP chief professor at [institution redacted pending consent], an early PM-KR member, shared the PM-KR group's LinkedIn invitation and flagged the Form → Meaning inversion as a genuine novel organising principle in the NLP ontology space. This is an *independent third-party architectural novelty acknowledgement*, noted neutrally.

**Paper framing instruction.** Until the professor's written consent is obtained, the institution and name remain redacted. If consent is granted, a one-sentence acknowledgement appears here with the professor's name and affiliation.

See ATTRIBUTIONS.md §6.4 (post-renumber) and `feedback_form_meaning_externally_validated.md` for the audit trail.

### §5.2 Runtime evidence

The four-layer architecture is not a proposal — it is the shipping structure of K3D Galaxy Universe entries. Every Galaxy star in the Knowledgeverse (38,144+ entries at April 2026) carries the Layer 1 ↔ Layer 2 bidirectional symlink pattern. Grammar Galaxy entries carry Layer 3 ↔ Layer 2 references. The Defeasible Logic module (SPINdle-derived, per `feedback_exploratory_grammar_deferred.md`) instantiates Layer 4.

### §5.3 Honest limitations

- Layer 4 is least-developed at evaluation time; conflict-resolution cases are still being hand-audited.
- Bidirectional-symlink integrity has been the source of multiple drift incidents (pushed into sleep-time consolidation per `feedback_no_fallbacks_ever_including_sleeptime.md`).
- Canonical-ID assignment policy for new stars (content-hash vs editorial) is an open design question.

---

## §6 Discussion (~0.5 page)

### §6.1 What Form → Meaning is *not*

It is not:
- A new ontology (RDF/OWL/BFO are ontologies; Form → Meaning is an axis-choice).
- A replacement for embeddings (embeddings live inside Layer 1 or approximate Layer 2 `d`).
- A multilingual resource (WordNet, BabelNet do resources; K3D does runtime identity).

### §6.2 Limits of the inversion

For data where surface form *is* the meaning (legal citations, historical orthographies), Layer 1 ≈ Layer 2. The four-layer architecture does not force extra indirection where it's not warranted — Layer 2 stars can be thin wrappers when appropriate.

### §6.3 Relationship to House vs Galaxy

Per `HOUSE_VS_KNOWLEDGEVERSE_DISTINCTION.md`: the House is intentional placement of Layer 1 forms (a physical shelf, a 3D asset). The Galaxy computes semantic gravity over Layer 2 stars. The two substrates connect via the Layer 1 ↔ Layer 2 symlink.

---

## §7 Conclusion (~0.25 page)

Three sentences:

1. K3D organises knowledge by meaning (Layer 2) rather than by surface form (Layer 1), with rules (Layer 3) and meta-rules (Layer 4) composing above.
2. The inversion is independently acknowledged as architecturally novel and is the substrate-level precondition for dual-client (human + AI) shared cognition.
3. Form → Meaning is not a re-labelling of RDF/OWL; it is a different axis choice, and it enables every contribution in Papers A, B, C, E, and F.

---

## Page Budget Check

| Section | Words | Pages (approx) |
|---------|-------|----------------|
| Abstract | 185 | 0.25 |
| §1 Introduction | 500 | 0.75 |
| §2 Background | 500 | 0.75 |
| §3 Four-layer architecture | 1000 + Fig 1 | 1.5 |
| §4 Why inversion matters | 350 | 0.5 |
| §5 External acknowledgement + runtime evidence | 350 | 0.5 |
| §6 Discussion | 325 | 0.5 |
| §7 Conclusion | 175 | 0.25 |
| References | — | ~0.5 |
| **Total** | **~3385 words + 1 fig + refs** | **~5.5 pages** |

Fits 6-page venue; could expand §3 or §5 if budget allows.

---

## Writing-phase todos

- [ ] Obtain NLP chief professor's written consent for §5.1 naming OR finalise neutral-redacted language.
- [ ] Render Figure 1 (four-layer stack with bidirectional symlink arrows).
- [ ] Expand §3.6 worked example with additional concrete stars if space permits.
- [ ] Verify `DUAL_CLIENT_CONTRACT_SPECIFICATION.md` §1.6 phrasing for the "save information" quote.
- [ ] Check ATTRIBUTIONS.md §6.4 language matches §5.1 of this paper.

---

**Location**: `TEMP/CLAUDE_PAPER_D_SKELETON_DRAFT_04.19.2026.md`
**Parallel to**: Papers A, B, C skeletons.
**Next in series**: Paper E (Ternary-First Computation).
