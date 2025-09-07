# Doors & Network — OSI in Spatial Form

A Door in K3D is both a 3D object and a networking interface. It has a visible frame, an address bar, and optional UI controls to open, close, or bookmark routes. The concept mirrors real‑world doors while mapping to OSI/network primitives.

Key ideas
- Address Bar: Users (or agents) type an address into the door, e.g. `k3d://rx,ry,rz:port@x,y,z?label=Knowledge Garden`.
- OSI Analogy: Doors resolve to routes across the current House graph and can connect to other Houses (LAN), datasets, or external services.
- Dual Client: Human clients see the door and type; AI clients can resolve addresses directly against embeddings/labels.

Address format
- URI form: `k3d://rx,ry,rz:port@x,y,z[?label=...]`
  - `rx,ry,rz`: routing vector hint (optional), typically 0,0,0.
  - `port`: abstract port number for service multiplexing.
  - `x,y,z`: spatial destination hint or anchor.
  - `label`: human‑friendly caption.
- The live server also supports free‑text labels; it resolves them via gazetteer + TF‑IDF.

Door metadata
- A door is a node with `metadata.type = "door"` and `metadata.address: string`.
- Doors are included in `extras.k3d` alongside all nodes and can be followed/traversed like any other node.

Runtime behavior
- `/open <label|k3d://...>`: resolve and emit a `goto` route for the viewer.
- The bridge composes a concise route trace with per‑hop similarity and distance.
- Pausing: when paused, navigation is suppressed until `/resume`.

Multi‑house (LAN)
- Each avatar has a House (`K3D_HOUSE_ID`). Linking Houses is modeled by creating doors whose `metadata.address` points to the neighbor House asset (e.g., `/houses/<id>/memory_house.gltf`).
- Agents remain grounded in their own House; doors provide the bridge to others.

Viewer UI (suggested)
- Show a floating input field on a door panel (address bar).
- Support autocomplete from labels and known doors.
- Display a small route preview when an address is valid.

See also
- `knowledge3d/spatial/address.py`
- `spec/k3d_agent_protocol.md` (door metadata)
- `docs/CRANIUM.md` (agent policy and autonomy)
