# Chat Memory — Books, Messages, and Consolidation

Goal
- Keep the full chat history (human and agent turns) inside the House memory so the model can retain and revisit context across sessions, then consolidate older turns into a persistent “Chat Book”.

Representation (embedded in GLB)
- Objects live in the House GLB as `primitive.extras.k3d` entries:
  - `chat_book` (furniture-like object): named per channel/session (e.g., `Chat #general`).
  - `chat_message` (object): one node per turn with metadata:
    - `parent`: id of the `chat_book`
    - `nick`: speaker name (e.g., `agent`, `user1234`)
    - `role`: `agent` | `human`
    - `ts`: ISO8601 UTC timestamp
    - `prev`: previous message id (for turn-by-turn linking)
    - `embedding32`: deterministic 32‑d hash of `nick|text` (upgraded later by sleep-time embedding if desired)

Runtime
- The live server writes each chat turn into the active House memory (`data/houses/<K3D_HOUSE_ID>/memory_house.json`), under a `chat_book` per channel.
- Export happens during `/sleep consolidate` (or `/mem export`), producing `viewer/public/houses/<id>/memory_house.glb`.

APIs
- Creation and updates are handled by `knowledge3d.tools.house_memory.MemoryHouse`:
  - `ensure_chat_book(label, room="Diary")`
  - `add_chat_message(book_label, nick, text, role="human", prev_id=None, room="Diary")`

Viewer
- Messages appear in the House memory GLB and can be surfaced in the Tablet (e.g., a Chat app) and searched by embeddings. Older sessions can be kept as separate books.

Notes
- Embeddings: the 32‑d `embedding32` hash keeps memory cheap and deterministic at runtime; sleep-time compute may re-embed turns with a larger encoder when available.
- Privacy: journals and chat messages are persistent; redact/export guidelines should be considered for multi-user deployments.

Inspiration & References
- IRC (protocol and UX ideas):
  - ircd-hybrid — https://github.com/ircd-hybrid/ircd-hybrid
  - charybdis — https://github.com/charybdis-ircd/charybdis
  - We adopt channel semantics (/join, /nick, /me, /msg), topics, names/who, flood control, and 512‑char caps inspired by IRC.
- Modern multimodal chat boxes:
  - Open WebUI — https://github.com/open-webui/open-webui
  - We treat the chat composer as a multimodal box (text + attachments) for both human and AI clients.
