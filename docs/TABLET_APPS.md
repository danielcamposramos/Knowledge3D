# Tablet Apps (In-World)

Architecture
- Apps implement `TabletApp` (id, title, renderCanvas, openOverlay). The tablet draws a compact view on the 3D screen and opens a full UI in Focus mode.
- Registry: Console, Notes, RPN Calc, Web (text fetch). Extensible for Calendar/Email/OAuth connectors.

Current apps
- Console: shows explain-as-you-move messages; clears in Focus.
- Notes: local IndexedDB store; offline by default. Future: optional OAuth sync for human (e.g., to a Keep-like OSS service).
- RPN Calc: HP‑style stack operations using the shared `RPN` engine. Works offline.
- Web: text-only fetch (CORS permitting). Offline history in IndexedDB. Future: iframe embed and agentic browsing adapter.

Offline & Sync
- All apps default to offline storage; human can authorize OAuth connectors (documented per app when added).
- Tablet keeps an outbox and snapshots for house graph/doors.

Extending
- Add a new app by implementing `TabletApp` in `viewer/src/apps.ts` and registering in `viewer/src/tablet.ts`.
- For OAuth: add a connector that saves tokens client-side for human mode; AI mode remains offline unless explicitly permitted.

Planned
- Calendar: offline ICS store; optional OAuth sync.
- Email: local MBOX view; optional OAuth/IMAP sync.
- Embeddings Peek: show nearest neighbors for a selected label.
- Agentic Browser: persistent browsing sessions rendered into the tablet.

