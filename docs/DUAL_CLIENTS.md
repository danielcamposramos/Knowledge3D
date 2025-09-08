# Dual Clients (Human vs AI) — No Toggle

Principle
- Two tailored clients consume the same House memory (GLB + `extras.k3d`). No runtime toggle. Humans use the human client; AIs use the AI client. Both inhabit the same world and exchange the same objects.

Shared Assets
- All geometry, embeddings, neighbors, and metadata live in `primitive.extras.k3d` in the GLB. Each client renders the same nodes/rooms/doors but with different materials/overlays.

Human Client (Game‑like)
- First‑person controls and HUD (menus, prompts, multi‑language text).
- Materials/textures: bark/leaf/stone/metal; lights/shadows; media screens show videos/images.
- Tablet is a 3D object with readable UI; casting to in‑world projection screens is visible.

AI Client (Embedding‑first)
- First‑person navigation in the same space with “AI vision” materials:
  - Shapes carry modality cues (text/image/audio/video) and show “rays” for active semantic channels.
  - Embeddings are the texture: vectors, masks, and temporal alpha modulate appearance.
  - Optional overlays: neighbor rays, cluster halos, similarity readouts.
- Tablet UI shows vector forms: embeddings and structured payloads instead of text.

Galaxy vs House Views
- Galaxy (space game/Descent‑like): Stars = meanings; vectors/rays visible; sparse fog/grid to aid orientation.
- House (rooms/objects): Knowledge Garden (trees), books, shelves, doors; same objects carry embeddings.

Runtime
- Build/launch separate clients or run two viewer instances with different defaults (env/URL parameter) — but no in‑session toggle. The choice is made at start.

Agentic Tools
- Both clients drive the same Cranium skills: chat, retrieval, media playback, embedding edits, and tool invocations. Tools change the world by updating K3D payloads + re‑exporting GLBs.

