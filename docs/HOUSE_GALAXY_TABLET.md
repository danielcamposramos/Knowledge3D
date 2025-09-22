# Memory Tablet & Dual-Space Memory Architecture

This document refines the Knowledge3D memory workflow around three persistent structures—**Galaxy**, **House**, and **Museum**—and the new **Memory Tablet** interface that keeps them reachable from the avatar’s point of view. The avatar is always embodied in the House: the Galaxy is an introspection layer (akin to a brain scan) that the avatar consults during thinking time, not a place where it “lives”.

## Memory Layers at a Glance

| Layer | Analogy | Purpose | Lifespan | Access Pattern |
|-------|---------|---------|----------|----------------|
| **Galaxy** | Active RAM | High-frequency reasoning buffer. Holds embeddings the fused head, PTX operators, and neural layers need right now. | Volatile; repopulated per session or per query. | PTX cosine, on-demand streaming, high-frequency updates. |
| **House** | Persistent SSD/HDD | Consolidated knowledge, crystallised into explicit artifacts (books, diaries, fractal trees, learning insights). | Long-term; evolves during sleep cycles. | Tablet search, SleepTime export, selective PTX load. |
| **Museum (Zone 8)** | Archive / Cold Storage | Deprecated or superseded artifacts kept for retrospection, audit trails, and error-pattern training. | Long-term; mostly append-only. | Loaded only when the user explicitly opens the museum. |

The sleep-time compute pipeline moves validated memories from **Galaxy → House**, while relocation utilities send obsolete items **House → Museum**.

## Memory Tablet

The Memory Tablet is a persistent 3D object available to the avatar at all times. While the avatar can grab items directly in-room, the tablet offers a galaxy-standard view when the avatar wants to align house artifacts with the active thinking memory. Most day-to-day cognition stays embodied in the House; the tablet is a support tool and an introspection bridge.

### Functions

- **Inventory Browser**: zero-latency search across the House inventory (“disk”) without forcing the user to traverse every room. Provides filtered views (books, trees, learning insights, diaries, dream artifacts) and quick teleport links.
- **Galaxy Bridge**: surfaces what is currently active in the Galaxy (RAM), including confidence scores, PTX task queues, and recent teacher tags. Allows the agent to request explicit loads from House → Galaxy if more detail is required.
- **Old-World Connectors**: embeds a lightweight browser (Firefox or an existing containerised browser) so the avatar can interact with conventional web content, documentation, or legacy chat interfaces. Captured context is stored as tablet notes that can later be consolidated via SleepTime.
- **Context Mixer**: exposes toggles for level-of-detail (LOD) tiering. The avatar can request coarse summaries, medium fidelity (subset of embeddings), or full-resolution GLBs to match the current reasoning task.

### Implementation Expectations

1. **Always-On Link to House**: the tablet queries a house-memory index (GLB + manifest) generated every sleep cycle. The fused head should treat this tablet index as the highest-priority retrieval source before falling back to language galaxies. In most cases, in-room inspection suffices; the tablet is used when the avatar wants galaxy-normalised context (e.g., to prepare items for temporary thinking memory).
2. **On-Demand Streaming**: tablet requests can stream artifacts into the Galaxy, respecting LOD/memory budgets. The fused head receives callbacks so it can expand its working set and update PTX caches.
3. **Browser Integration**: prefer existing open-source browser containers (e.g., the Firefox-based container already present in local Docker inventory). Tablet sessions authenticate through doors (`k3d://` URIs) and capture the fetched context as structured entries.
4. **Mutation Hooks**: whenever an artifact on the tablet is edited or a new insight is recorded, SleepTime must reconcile the change, materialise a new House asset, and relocate prior versions to the Museum.

## Sleep-Time & LOD Interactions

- **Consolidation**: SleepTime Compute rebuilds `learning_memory.glb`, regenerates the house index for the tablet, and records which galaxy prompts achieved stable 1.0 honesty scores. Those prompts can be removed from active drills after at least one consolidated confirmation.
- **Dynamic Loading**: LOD strategies (borrowed from game streaming) keep GPU memory bounded. The tablet can:
  - Load coarse vectors (centroids) for quick scans.
  - Request medium-detail patches (e.g., only embeddings + metadata) when reasoning needs more cues.
  - Pull full GLBs (geometry, textures) when a shape must be inspected.
- **Museum Handling**: relocation utilities tag deprecated entries with `previous_zone` and `relocated_at`; the tablet exposes these in a “Museum mode” for post-mortem analysis. Nothing should automatically flow back into the Galaxy unless the avatar explicitly promotes it.

## Agent & Tooling Requirements

- **House Memory Builder**: extend SleepTime to emit a PTX-ready `house_memory.glb` (and manifest) summarising consolidated artifacts. The fused head must load this before querying modality galaxies.
- **Prompt Pruning**: training loops (Phase18/Phase25) should retire prompts that the fused head + tablet confirmation mark as mastered. Prompts move to a verification list rather than main drill sets.
- **Deprecation Workflow**: any time sleep-time reasoning supersedes an artifact, call `relocate_to_museum` (or an equivalent) so the previous version is tagged and moved to Zone 8.
- **Tablet UI**: design the viewer-side interface for the tablet, including search, LOD controls, and browser integration. Surface explicit indicators showing whether a fact came from Galaxy, House, or Museum.

## References

- Sleep pipeline: `docs/SLEEP_COMPUTE.md`, `knowledge3d/cranium/phase10/sleep_time_compute.py`
- Museum relocation utility: `knowledge3d/tools/phase18/meaning_cluster_trainer.py::relocate_to_museum`
- Fused head routing: `knowledge3d/cranium/fused_head.py`
- PTX learning memory builder: `knowledge3d/tools/learning_memory_builder.py`

Keep this document in sync with roadmap changes and the tablet implementation plan.
