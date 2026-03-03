# Procedural Reasoning Profile for WebML

## Proposal name

Procedural Reasoning Profile for WebML

## Short description

Current WebML proposals focus on model execution (WebNN), agent protocols (WebMCP), and hybrid placement (#5, #15). What's missing is a standard way to represent and exchange reusable reasoning artifacts that can be inspected, composed, and cached across these systems.

This proposal introduces a lightweight profile for procedural knowledge representation — think of it as a companion to WebNN graph portability (#16). Instead of just exchanging model topology, we'd also have a way to exchange executable reasoning units with clear semantics, provenance, and boundary constraints. The profile is explicitly complementary to existing work, not a replacement.

## Example use cases

**Use case 1: Transparent agent tools**

Problem: When an agent proposes using a tool (via WebMCP or similar), users can't inspect what that tool actually does beyond a text description.

With procedural profile: Each tool references a procedural artifact that shows both human-readable semantics and machine-executable logic. If a math solving tool references artifact `pmkr:algebra:solve_linear`, you can inspect its step-by-step procedure and verify it's doing what you expect before approving execution.

**Use case 2: Reducing redundant transfers in hybrid systems**

Problem: Hybrid client/cloud workflows (#5) often transfer similar knowledge structures repeatedly — the same math operations, text patterns, or reasoning steps get duplicated across requests.

With procedural profile: Artifacts have canonical IDs and support reference-based composition. Instead of sending the full procedure each time, you reference `pmkr:op:solve.v1` and only transfer it once. Subsequent uses are by-reference, reducing bandwidth and enabling smarter caching strategies.

**Use case 3: Portable semantics for graph nodes**

Problem: Graph portability work (#16) focuses on topology exchange, but graph nodes often need richer semantic metadata to be truly portable across toolchains.

With procedural profile: Attach optional procedural metadata to graph nodes. This doesn't change graph execution but lets toolchains understand and document what each node represents beyond its mathematical operation. You get both graph structure portability and semantic portability in one package.

## Implementation approach

Start with an explainer document defining a minimal JSON profile with these core fields:
- `id`: canonical identifier
- `form_program`: optional structural representation
- `meaning_program`: optional semantic payload (stack-based execution format like RPN)
- `refs`: references to other artifacts
- `provenance`: source and transformation metadata
- Optional: `boundary_contract` for governance/privacy policies

Build two reference adapters:
1. Graph-to-procedural: Maps WebNN graph nodes to procedural artifacts
2. Procedural-to-tool-annotation: Maps procedural artifacts to agent tool metadata

Validate with a two-toolchain interop demo measuring:
- Artifact size savings vs naive duplication
- Serialization/deserialization overhead
- Interop success rate

Keep the initial scope narrow and explainer-focused. Only pursue formal specification if the prototype demonstrates clear value to WebML implementers.

---

**Background**: This builds on work from the PM-KR Community Group (procedural memory knowledge representation) at W3C, which has been exploring procedural knowledge composition patterns since February 2026. The profile is designed to integrate with existing WebML directions rather than compete with them.

**Related issues**: #5 (Hybrid AI), #12 (WebMCP), #15 (Dynamic offloading), #16 (Graph portability)
