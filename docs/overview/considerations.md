# Knowledge3D: accessibility, internationalization, privacy and security

Part of [Knowledge3D](../../README.md), moved here whole from its front page.

## Accessibility, Internationalization, Privacy, and Security Considerations

### Accessibility
The Dual-Client Contract is the design goal that clients share underlying identity and meaning. Visual, speech, Braille, haptic and non-spatial interaction paths need separate implementation, assistive-technology tests and review with users. A shared source does not by itself establish accessible equivalence.

**Spec**: [UNIVERSAL_ACCESSIBILITY_SPECIFICATION.md](../vocabulary/UNIVERSAL_ACCESSIBILITY_SPECIFICATION.md)

### Internationalization
Knowledge is meaning-centric, not language-centric. "Cat" is one star containing surface forms for every language (English "cat", Portuguese "gato", Japanese "猫"). Reasoning operates on meaning (Layer 2-4), not language surface (Layer 1). Language-specific forms can reference shared entries, while context, ambiguity and culturally distinct concepts may require separate modeling and review.

### Privacy
K3D runs on local consumer hardware. No cloud dependencies for core reasoning. The House (personal knowledge) stays on the user's device. Network interfaces (portals between Houses) are explicit and permissioned.

### Security
The hot-path design avoids external ML frameworks and includes sovereignty checks. Drivers, build tools, runtimes, input procedures and surrounding services still require security review. Inspectable code and provenance help review; they do not establish that execution is safe.
