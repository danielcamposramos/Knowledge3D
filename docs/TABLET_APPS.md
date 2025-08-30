# Tablet Apps (In-World)

Architecture
- Apps implement `TabletApp` (id, title, renderCanvas, openOverlay). The tablet draws a compact view on the 3D screen and opens a full UI in Focus mode.
- Registry: Console, Notes, RPN Calc, Web (text fetch). Extensible for Calendar/Email/OAuth connectors.

TypeScript Interface
```
export interface TabletApp {
  id: string;
  title: string;
  renderCanvas(ctx: CanvasRenderingContext2D, rect: { x: number; y: number; w: number; h: number }): void;
  openOverlay(el: HTMLDivElement): void;
  onEvent?(ev: { type: string; payload?: any }): void; // fan‑in events from tablet.publish()
  setContext?(ctx: { records: ReadonlyArray<K3DRecord>; publish?: (ev: { type: string; payload?: any }) => void }): void;
}
```

Event Bus
- The tablet forwards app events to the live server as `event` envelopes via `ChatClient.sendEvent`.
- Canonical events used by built‑in apps:
  - `browser_search`: `{ engine, query, count }`
  - `browser_visit`: `{ engine, url, title, len }`
  - `browser_iframe`: `{ url }`
  - `goto_resolution`: `{ target, query?, sim?, model_confidence? }`
  - `applyLayers`: `{ enabled: string[] }`
  - `diary_entry`: `{ text, tz }`
  - `sleep` / `wake`: `{ mode?: 'consolidate' }`

Current apps
- Console: shows explain-as-you-move messages; mirrors to live logs; clears in Focus.
- Notes: local IndexedDB store; offline by default. Future: optional OAuth sync for human (e.g., to a Keep-like OSS service).
- RPN Calc: HP‑style stack operations using the shared `RPN` engine. Works offline.
- Agentic Browser: Wikipedia search via MediaWiki API (CORS-safe) + summary view; attempts direct fetch with text fallback and iframe fallback. Emits session logs to the live server (`browser_search`, `browser_visit`, `browser_iframe`).
- Live Stats: aggregates live `goto` resolutions parsed from agent messages (direct vs. resolved; model-assisted count).

Offline & Sync
- All apps default to offline storage; human can authorize OAuth connectors (documented per app when added).
- Tablet keeps an outbox and snapshots for house graph/doors.

UI Details
- Focus header shows: mode, ws status, queue, house URI, nodes, info.
- No placeholder actions: the "Install App…" control has been removed; focus panel strictly hosts real app UIs.

Extending
- Add a new app by implementing `TabletApp` in `viewer/src/apps.ts` and registering in `viewer/src/tablet.ts`.
- For OAuth: add a connector that saves tokens client-side for human mode; AI mode remains offline unless explicitly permitted.

Planned
- Calendar: offline ICS store; optional OAuth sync.
- Email: local MBOX view; optional OAuth/IMAP sync.
- Embeddings Peek: show nearest neighbors for a selected label.
- Galaxy: added Expand φ and Freeze controls; lays out rings by neighbor-of-neighbor similarity.
