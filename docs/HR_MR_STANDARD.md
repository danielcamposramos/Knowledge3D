# HR/MR Standard for Knowledge3D

Human‑Readable (HR) and Machine‑Runtime (MR) are dual facets of the same system:

- HR: Clear, auditable, developer‑first code and documentation in this repo (TypeScript/Three.js viewer, Python generators/servers, specs).
- MR: Optimized, deployment‑ready sources generated outside the repo via `codeopt`, mirroring HR behavior for production runtimes.

Principles
- One truth, two builds: HR defines behavior; MR mirrors it without changing semantics.
- Explicit data contracts: `primitive.extras.k3d` is the AI‑native substrate (ids, vectors, embeddings, metadata, neighbors, AI flags).
- Traceable reasoning: Agents emit concise, structured “explain‑as‑you‑move” logs.

Artifacts
- Schema: `spec/k3d_node_schema.json` (node contract) and `spec/glTF_K3D_extension.md` (embedding in glTF).
- AI runtime: `knowledge3d/spatial/address.py` (spatial addresses), `knowledge3d/spatial/osi.py` (OSI layers), RPN logic standard (below).
- Viewer parity: `viewer/src/` implements human affordances over the same AI substrate.

MR Generation
- Use `codeopt` to generate MR sources to `../Knowledge3D.local/mr` (see `docs/DUAL_CODE.md`).
- Do not commit MR outputs; keep HR as the authored source of truth.

