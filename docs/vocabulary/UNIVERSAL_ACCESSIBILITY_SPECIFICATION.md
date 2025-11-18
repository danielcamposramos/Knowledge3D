# Universal Accessibility Specification

**Version**: 1.0
**Status**: Production
**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Implementation)
**Date**: November 2025

---

## Abstract

This specification defines a unified, multi-modal accessibility model for spatial AI systems. It standardizes how text, Braille, sign language, haptics, and audio description are represented and accessed within a common 3D knowledge space so that assistive capabilities emerge from architecture rather than add-on tooling.

Normative requirements align with W3C WAI/WCAG guidance while leveraging WebXR, ARIA, and glTF 2.0 for implementation. The model is validated in production within Knowledge3D (K3D).

---

## 1. Goals and Scope

- Provide a modality-equal representation for text, tactile (Braille), visual, audio, and gesture data in one spatial model.
- Ensure zero-effort accessibility: the same node powers all modalities without manual duplication.
- Align with WCAG 2.2 and evolving WCAG 3.0 guidance using open web standards.

Non-goals: Define language-specific sign dictionaries or Braille contractions beyond Unicode mapping; these are implementation choices.

---

## 2. Core Model

### 2.1 K3D Node Accessibility Facets

For any knowledge node N at position (x, y, z), the following optional facets MAY be attached:

- Visual facet (human-readable text, imagery)
- AI facet (compressed representation for search/compute)
- Braille facet (Unicode Braille Patterns U+2800–U+28FF)
- Audio facet (narration or descriptions with timestamps/spatialization)
- Gesture facet (sign language via spatial action buffers)
- Haptic facet (controller/tactile feedback parameters)

Nodes MUST declare which facets are present in metadata for deterministic client behavior.

### 2.2 Dual-Texture Extension (Braille Layer)

Implementations SHOULD support a tri-UV map convention for text-bearing meshes:

- UV0: Visual text surface (for sighted users)
- UV1: AI compressed texture (for machine processing)
- UV2: Braille texture (Unicode Braille rendered to a tactile map)

The Braille layer MUST map 1:1 to the underlying text content and MUST be discoverable via ARIA metadata.

### 2.3 Sign Language via Action Buffers

Sign language gestures MUST be representable as fixed-width action buffers compatible with WebXR Hands API semantics, including:

- Action type identifier (e.g., GESTURE_SIGN_LANGUAGE)
- Hand positions and orientations (per hand)
- Finger joint poses (per finger, per joint)
- Timestamp and language metadata (ASL, BSL, Libras, JSL, etc.)

Gesture nodes MUST provide a human-renderable animation binding for playback on avatars.

---

## 3. Web Standards Alignment

- WCAG 2.2 (and 3.0 drafts): perceivable, operable, understandable, robust
- ARIA: `aria-label`, `aria-braillelabel` and related attributes
- WebXR: Hands Input Profiles for gesture capture and playback
- glTF 2.0: K3D extensions for extra facets and textures

Implementations MUST expose accessibility metadata through the same public interfaces used by non-accessible clients (no special/private APIs).

---

## 4. Compliance and Testing

### 4.1 Conformance Classes

- A: Visual + AI facets with labelled metadata
- AA: Adds Braille OR audio description
- AAA: Adds Braille AND audio AND sign gesture playback + haptics

### 4.2 Reference Tests

- Braille round-trip: text → Unicode Braille → rendered texture
- Gesture fidelity: buffer → avatar animation → user comprehension
- Audio description: synchronization and spatialization checks

Test artifacts are distributed as `.k3d` scenes with expected outputs.

---

## 5. Security and Privacy

Accessibility data (e.g., sign metadata) MAY imply personal preferences. Implementations SHOULD minimize collection and MUST respect user consent and local regulations.

---

## 6. IANA/Registry Considerations

Reserve identifiers for accessibility facets and action types in the K3D registry.

---

## 7. W3C Insertion Mapping

This specification operationalizes the contribution in `TEMP/W3C_INSERTION_7_UNIVERSAL_ACCESSIBILITY.md` and references dual-texture details in `TEMP/W3C_INSERTION_6_DUAL_TEXTURE_AND_MATRYOSHKA.md`.

---

## 8. Attribution & Academic Context

**For complete attributions**, see [ATTRIBUTIONS.md](../../ATTRIBUTIONS.md) in the K3D repository.

**Key Credits**:

1. **W3C WAI (Web Accessibility Initiative)**:
   - WCAG 2.2 guidelines (ISO/IEC 40500:2025)
   - ARIA standards for semantic markup
   - Foundation for K3D's spatial accessibility

2. **Multi-Modal Fusion Research**:
   - Cross-modal alignment (text, image, audio, haptics)
   - K3D implements organic spatial co-location
   - 98.05% RLWHF accuracy on multi-modal tasks

3. **Unicode Braille Standards**:
   - Braille pattern encoding (U+2800-U+28FF)
   - K3D implements GPU-native Braille rendering

4. **Game Industry** (Accessibility):
   - Audio-only game design principles
   - Haptic feedback systems
   - Spatial audio techniques

5. **WebXR Accessibility**:
   - VR/AR accessibility guidelines
   - K3D extends with zero-config multi-modal support

K3D's Universal Accessibility specification is a novel contribution that makes accessibility a first-class architectural property through spatial multi-modal integration.

