# Hyper-Modular Architecture: project definition

**For researchers and implementers:** a proposal for composing knowledge through references at several levels.
**Status: input.**
Daniel Ramos used this term in the Knowledge3D design dated 20 February 2026.
This is a project definition offered for discussion, not an adopted W3C definition or a claim of priority over other modular systems.
The [explainer](EXPLAINER.md) provides the current introduction.

## Definition

The proposed architecture organizes knowledge at several levels and composes procedures by referring to shared entries.
A reference identifies an entry without embedding another copy of its content.
This is the intended data model; it does not establish universal deduplication, smaller storage or equivalent meaning across clients.

## Core principles

| Level in the original proposal | Proposed responsibility |
| --- | --- |
| Domains, called Galaxies | Organize procedural vocabularies such as characters, drawing and mathematics |
| Contexts, called Houses | Hold persistent knowledge and compositions for a bounded context |
| Rooms | Organize related knowledge within a House |
| Nodes | Identify entries containing procedures, data and references |
| Procedures | Compose executable operations and references |
| Operations | Supply primitive instructions such as stack and arithmetic operations |
| Runtime kernels | Execute the operations in a chosen implementation |

The [architecture specification](../vocabulary/HYPER_MODULAR_ARCHITECTURE.md) develops these relationships.
The [Three-Brain System](../vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md) distinguishes active memory, processing and persistent storage.
These levels describe project organization; other modular approaches are not restricted to a fixed number of levels.

### Symlink-style composition

“Symlink-style” means that a node refers to a canonical entry, by analogy with a file link.
It does not require a filesystem symlink.
A composition must specify reference resolution, versions, permitted changes and the response to a broken reference.
Sharing an entry can avoid repeated payloads, while references, caches and generated outputs still have storage costs.

### Procedural canonicalization

The proposal stores a reusable procedure and its parameters rather than separately authored copies of its result.
Canonicalization needs a defined scope and equivalence rule.
Identical bytes, equivalent program behavior and equivalent meaning are different questions.
The original goal of mapping all equivalent knowledge to one form remains an open research question.

### Dual-client composability

Human and AI clients would access the same entry while presenting or processing it differently.
For example, a person could view a rendered glyph while software follows its identifier and semantic links.
A glyph's shape alone does not determine its meaning or supply every accessible representation.
The [Dual-Client Contract](../vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md) describes this design goal.

### Sovereign execution

The project's runtime design places reasoning operations in its own GPU kernels, with external preparation outside that path.
This is an implementation choice and proposed boundary, not evidence that execution is safe, portable or free of driver and runtime dependencies.
See the [conformance profiles](PM_KR_CONFORMANCE_PROFILES.md) for the candidate tiers.

## Use cases

The original examples considered a shared curriculum, department-specific workflows and agents sharing procedures.
A curriculum could reference a common mathematical procedure while retaining its own explanations and exercises.
A workflow could reuse public definitions while keeping private records in its own context.
Both scenarios require access-control and versioning tests; composition alone does not prove those protections.

## Formal definition for discussion

The original candidate definition proposed the following conditions:

1. Organize knowledge into at least three levels covering domains, contexts and individual entries.
2. Give modules executable procedures rather than only static payloads.
3. Compose modules through references to canonical procedures.
4. Define deterministic canonicalization within a stated equivalence relation.
5. Allow modules at one level to compose modules from another level through references.
6. Provide presentations or processing paths for at least two client types.
7. Define the execution boundary and its permitted dependencies for a runtime profile.

The original core tier covered hierarchy, procedures, references and client access.
Its runtime tier added execution boundaries and deterministic canonicalization.
Its audit tier added provenance, storage measurements and verification of compositions.
These are proposed requirements, not a conformance certificate for Knowledge3D.
The [normative-model draft](PM_KR_NORMATIVE_MODEL.md) and [profiles](PM_KR_CONFORMANCE_PROFILES.md) carry the current candidate requirements.
Their relationship to this older definition remains a specification review task.

## Academic research directions

- Define equivalence and canonicalization within a bounded domain.
- Test whether composed procedures preserve the intended meaning.
- Compare total storage and execution costs on the same workload.
- Determine where decomposition should stop for a useful primitive.
- Test the presentations and access controls with their intended users.

Daniel's [28 October clarification](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0018.html) argues for domain-specific stopping points and graph-based composition alongside RDF and OWL.

## Credits

**Project definition:** Daniel Ramos, Knowledge3D.
Prior work and other contributions are recorded in [ATTRIBUTIONS.md](../../ATTRIBUTIONS.md).

## Document history

The [earlier repository version](https://github.com/danielcamposramos/Knowledge3D/blob/33cb47454c056a267e78554d459c091af9c67476/docs/W3C/HYPER_MODULAR_DEFINITION.md) records the original text, including claims withdrawn here.
It is retained in version history for provenance and should not be cited as current validation.
