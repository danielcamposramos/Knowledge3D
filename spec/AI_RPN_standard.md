AI Runtime: RPN Logic Standard

Overview
- The internal AI control logic uses a small Reverse Polish Notation (RPN) core for deterministic, data‑parsimonius computation across clients.
- RPN covers vector ops (dot, norm, cosine), scalars (+, -, *, /, abs, min, max), and simple stack machine flow.

Rationale
- Determinism and auditability: tiny, testable operators make reasoning traces reliable.
- Transport efficiency: instructions are compact and loggable.
- Cross‑client parity: the same minimal core works in HR viewer and MR runtimes.

Reference Implementation
- HR: `viewer/src/rpn.ts` provides the canonical stack machine used by `K3DAgent` for cosine similarity and mic processing.
- MR: mirror this module when generating production builds via `codeopt`.

Integration Points
- Agent navigation: cosine similarity between embeddings powers “per‑hop” explain traces.
- Signal handling: simple DSP readouts (e.g., mic peak) rely on RPN ops.
- Future: server‑side MR components can adopt the same operator set for consistency.

Spec Notes
- RPN is a logic substrate, not a replacement for model inference. It standardizes local computations close to the data plane.
- Operators MUST be pure and side‑effect free.

