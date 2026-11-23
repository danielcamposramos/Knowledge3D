# Knowledge3D: the problem and the proposed approach

Part of [Knowledge3D](../../README.md), moved here whole from its front page.

## The User-Facing Problem

A lesson, character or scientific concept may be represented by geometry, text, semantics, accessible descriptions and AI-facing data. These representations serve different purposes, but overlapping facts can drift when maintained independently. K3D explores a common procedural source and explicit references so clients can share identity and provenance while retaining the presentation each user needs.

A concrete goal is a spatial lesson that a sighted student, a screen-reader user and an AI partner can inspect consistently. That requires both a knowledge model and tested presentation/accessibility paths; a 3D scene alone does not supply either semantic equivalence or accessibility.

## Proposed Approach

Knowledge3D develops executable Reverse Polish Notation (RPN) programs with symlink-style composition as a way to reuse knowledge. The design links semantic identity, visual programs and language-specific forms. Speech, Braille and other accessible presentations need modality-specific rules and validation; sharing an entry is the architectural aim, not proof that all outputs are implemented.

**The architecture in one sentence:** A 3D spatial reality (the House) where knowledge lives as permanent objects, processed by an AI brain (the Galaxy) that loads concepts on demand and reasons over them on GPU via sovereign PTX kernels — with zero external dependencies in the hot path.

### Conceptual data example

This illustrates the proposed relationship between fields; it is not a complete executable example or an accessibility test.

A character like "A" is stored once as a procedural star:

```python
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar

# One star = one concept = all languages, all modalities
star = MeaningCentricStar(
    meaning_rpn="CONCEPT LETTER LATIN UPPERCASE PUSH",  # The meaning (executable)
    meaning_class="concept",
    domain="character",
    surface_forms={
        "en": SurfaceForm(word_ref="letter_A", char_refs=["char_u0041"]),
        "pt": SurfaceForm(word_ref="letra_A", char_refs=["char_u0041"]),
        "ja": SurfaceForm(word_ref="エー", char_refs=["char_u30A8", "char_u30FC"]),
    },
    visual_rpn="SET_COLOR 0 0 0 1 STROKE_WIDTH 0.05 MOVE -0.3 -0.5 LINE 0.0 0.5 LINE 0.3 -0.5 STROKE",
    confidence=1,   # Ternary: +1 confirmed, 0 uncertain, -1 contradicted
    polarity=1,
)

# Intended clients share the entry; each presentation path needs its own implementation.
```

The visual_rpn draws the glyph. The meaning_rpn carries the semantic identity. The surface_forms link to language-specific words without duplicating them. **One source, every client.**

## Use Cases

These are target scenarios for development and evaluation, not end-to-end test results.

### 1. Inclusive Education
A physics teacher says "demonstrate a pulley system." The classroom AI builds a working 3D pulley — visible on screen, navigable by screen reader, explorable by touch. The blind student and sighted student share the **same** spatial lesson, not parallel approximations.

### 2. Explainable AI
When K3D's AI reasons about "Is water an element?", you can watch the reasoning path: the avatar walks to the Library, opens the Chemistry book, navigates from "water" to "compound" to "hydrogen + oxygen." Every step is spatial, inspectable, auditable — not hidden in matrix multiplications.

### 3. Knowledge Deduplication
A university's knowledge base stores "photosynthesis" once — as a procedural star with RPN programs for the biochemical process, visual diagrams, audio explanations, and multi-language surface forms. Every course, every modality, every AI assistant references the same canonical entry. Zero duplication.

### 4. Multi-Modal Accessibility
The same procedural font program that renders "A" on screen also drives a Braille cell, generates an audio description ("uppercase Latin letter A"), and provides the AI with semantic identity — with a shared semantic reference. A glyph program alone does not encode every accessible representation; end-to-end size and usability must be measured.

## Non-Goals

- **Replacing LLMs** — K3D is not a chatbot or language model. It's a knowledge system that AI agents (including LLMs) can inhabit and use.
- **Cloud dependency** — K3D runs on consumer GPUs (RTX 3060 12GB). No cloud required for core reasoning.
- **Replacing RDF/OWL** — K3D explores a procedural layer alongside declarative semantics. Proposed mappings require explicit equivalence and round-trip tests.
- **Game engine** — K3D uses game industry technology (3D rendering, spatial indexing, LOD) but is a knowledge system, not an entertainment platform.
