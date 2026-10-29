# Knowledge3D: proposed relationships with web standards

**For implementers:** see where the project proposes to reuse existing formats and what would need testing.
**Status: input.**
This project note updates the December 2025 alignment draft.
It does not claim implemented compatibility, a registered extension, a W3C namespace or standards-body approval.
Start with the [explainer](EXPLAINER.md) for the current scope.

## 1. Standards alignment overview

| Technology | Existing role | Proposed Knowledge3D relationship |
| --- | --- | --- |
| [RDF](https://www.w3.org/TR/rdf11-concepts/) | Descriptions expressed as graphs of statements | Identify knowledge entries and link their relationships |
| [OWL](https://www.w3.org/TR/owl2-overview/) | Ontology language | Compare declared concepts and constraints with procedural behavior |
| [JSON-LD](https://www.w3.org/TR/json-ld11/) | Linked data expressed in JSON | Exchange descriptive metadata with explicit contexts |
| [SKOS](https://www.w3.org/TR/skos-reference/) | Concepts, labels and relations in knowledge organization systems | Associate a procedure's form with a term for its meaning |
| [PROV](https://www.w3.org/TR/prov-o/) | Descriptions of entities, activities, agents and provenance | Record origin, derivation and time |
| [glTF](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html) | Khronos format for delivering scenes and models | Explore carrying spatial presentation data alongside references to knowledge |

The SKOS and PROV direction follows Daniel's [28 October 2026 proposal](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0016.html).
These relationships are proposed mappings; neither a shared field name nor a similar concept establishes equivalent behavior.
The [interoperability guide](PM_KR_INTEROPERABILITY_GUIDE.md) discusses translation and losses.

## 2. Layer-by-layer alignment

### Data and references

An entry needs an identifier, a procedure describing form, a semantic term and provenance.
The project must define which version of an entry a reference identifies and how a reader handles missing or changed references.
The [node specification](../vocabulary/K3D_NODE_SPECIFICATION.md) and [normative-model draft](PM_KR_NORMATIVE_MODEL.md) contain the design inputs.
The `.k3d` file type, extension names, MIME types and W3C-hosted vocabulary identifiers in the old draft were proposals.
This page makes no registration claim for them.

### Human and AI clients

The [Dual-Client Contract](../vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md) proposes that both clients use the same knowledge source.
This does not by itself provide keyboard access, screen-reader support, text alternatives or Braille output.
[WCAG 2.2](https://www.w3.org/TR/WCAG22/) supplies accessibility requirements against which a user interface can be evaluated.
The project's [accessibility specification](../vocabulary/UNIVERSAL_ACCESSIBILITY_SPECIFICATION.md) is design input requiring implementation and user testing.

### Execution and presentation

The project's GPU execution design does not establish that its programs run in a web browser.
WebGPU, WebNN and WebXR integration would each need an explicit implementation and tests.
The earlier `builder.rpn` example was a proposed API, not an available WebNN operation.
Spatial presentation also needs a usable non-spatial route and clear permissions for access to knowledge and devices.

### Linked spaces

The earlier House and Portal examples suggested links between persistent knowledge spaces.
They did not demonstrate Linked Data Platform conformance, SPARQL support or cross-system exchange.
The concrete question is which identity, permissions, provenance and behavior survive a transfer.

## 3. Validation needed

A mapping needs example inputs and expected outputs, a statement of what is lost, and tests in independent implementations.
Accessibility claims need evaluation of the actual interface with its intended users.
Security and privacy need explicit execution limits, access rules and treatment of sensitive provenance.
A trace or a content hash alone does not establish safe execution or a true claim.
See the [conformance profiles](PM_KR_CONFORMANCE_PROFILES.md) and [evidence matrix](PM_KR_EVIDENCE_VALIDATION_MATRIX.md).

## Credits

The original draft credits the K3D Architecture Team and acknowledges W3C's immersive-web, spatial-data and incubation communities, the Khronos glTF authors and the IEEE Spatial Web work.
Those acknowledgments record prior work, not partnerships or approval.
See the [project attributions](../../ATTRIBUTIONS.md).
**Licence:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), retained from the original draft.

## Document history

The [earlier repository version](https://github.com/danielcamposramos/Knowledge3D/blob/33cb47454c056a267e78554d459c091af9c67476/docs/W3C/K3D_W3C_STANDARDS_ALIGNMENT.md) records the original text, including claims withdrawn here.
It is retained in version history for provenance and should not be cited as current validation.
