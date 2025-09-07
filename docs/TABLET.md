# The K3D Tablet — Bridge UI and Casting

The Tablet is a 3D object and the central UI surface for agents and humans. It connects the “old” (web/apps) with the “new” (spatial House) and can cast to projection surfaces (e.g., a large wall screen) inside the House.

UI Modes (three levels)
- In‑World Use: Interact directly with the Tablet mesh (touch/buttons). Quick actions: open doors, chat, highlight, cast.
- Close‑up View: The camera moves closer to the Tablet while staying in the 3D scene; richer controls and typing.
- Screen‑First Mode: The Tablet screen takes over as the primary UI (like in games); full apps, multi‑pane views.

Casting
- The Tablet can “chromecast” views to larger projection screens or surfaces in the House — useful for presentations or shared reading.
- The agent can choose to cast a panel (e.g., route planner, diary page, or portal window) to a target screen.

Network & Doors
- The Tablet serves as the anchor for network actions: opening doors via an address bar, maintaining connection overlays, and managing routes.
- For network‑intensive links (other Houses, remote datasets), doors behave as Portals using Quake‑3 style portal rendering: the door plays a portal animation and shows a rendered view of the destination as a texture when open.

Apps
- Small, composable apps (chat, door control, garden view, diary reader) run within the Tablet, keeping logic small and memory stable.

Implementation Notes
- Viewer should expose Tablet interactions in all three modes; a “cast to screen” action should target any projection surfaces present in the House.
- Door address bars can live on the Tablet as well as on door frames; both paths call into the same `/open` flow.
- See also `docs/DOORS_AND_NETWORK.md` and `docs/CRANIUM.md`.
