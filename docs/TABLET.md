# K3D Tablet (Wallet + Offline Sync)

Concept
- The Tablet is a local wallet that stays connected to your House (K3D). When the network drops, it buffers your intent and syncs back on reconnect so the agent isn’t lost.

What it stores
- Outbox: unsent chat, commands, and events queued in IndexedDB (`k3d-tablet/outbox`).
- Snapshot: last dataset graph and doors per house URI in IndexedDB (`k3d-tablet/tablet`).
- Tablet ID: stored in `localStorage` (future: signatures, auth, multi‑homes).

Where
- Viewer (browser) via IndexedDB — lightweight, persistent across reloads.

How it works
- The viewer’s ChatClient writes to an outbox when offline and flushes when reconnected (button status shows queue length).
- The viewer saves `dataset_graph` and `doors` snapshots for the current house. On reconnect, the agent can reconcile via normal events.

UI
- Tablet status shows online/offline and queue count.
- Works with Pause/Resume and all live commands.

Dev notes
- Stores: `openStore('k3d-tablet','outbox'|'tablet')` in `viewer/src/chat.ts` and `viewer/src/main.ts`.
- All messages remain standard JSON envelopes (`chat`/`command`/`event`), so the live server logs them uniformly after flush.

Message Envelope
- Fields appended by the tablet client for continuity and audit:
  - `tabletId: string` — stable per-browser ID stored at `localStorage['k3d-tablet/id']`.
  - `house?: string` — current GLB URI (set by viewer on house load).
  - `mode?: 'ai'|'human'` — current operating mode.
- Example (chat):
  `{ "type":"chat", "from":"human", "text":"/open gravity", "tabletId":"abcd…", "house":"/ai_compendium.80k.pca.doors.glb", "mode":"ai" }`

Outbox Schema (IndexedDB)
- DB: `k3d-tablet`, Store: `outbox`
- Key: `'queue'`
- Value: `Array<Envelope & { ts: number }>` where `Envelope` is one of `chat|command|event` with `tabletId,house,mode` appended.
- Flush: on `ws.onopen`, attempt send in order, re-queue unsent.

Tablet Snapshot Schema (IndexedDB)
- DB: `k3d-tablet`, Store: `tablet`
- Keys per house URI:
  - `${house}:dataset_graph` — minimal record list for UI (labels/ids, optional doors mask).
  - `${house}:doors` — boolean[] mask for `has_new_information`.
- Purpose: provide continuity hints and local app views while offline.

Roadmap
- Add Tablet ID in messages for continuity.
- Optional CRDT merge for multi‑device note streams.
- Wallet export/import for long‑running sessions.
