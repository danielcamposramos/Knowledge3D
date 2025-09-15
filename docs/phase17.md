PHASE 17: ETERNAL COGNITION — FULL STATE SERIALIZATION + BOOT CONTINUITY

GOAL
Serialize full Galaxy + House state; load on boot for exact continuity; auto‑save after sleep.

COMPONENTS
- knowledge3d/cranium/phase17/galaxy_state_serializer.py — saves/loads Galaxy stars/rays/embeddings/honesty to viewer/public/galaxy/galaxy_state.json
- knowledge3d/bridge/live_server.py — loads Galaxy state at boot (logs), saves House manifest to viewer/public/house/house_manifest.json after material load
- knowledge3d/cranium/phase10/sleep_time_compute.py — auto‑saves Galaxy state at end of sleep

OUTPUT
- viewer/public/galaxy/galaxy_state.json — full Galaxy state
- viewer/public/house/house_manifest.json — full House manifest

NEXT
Phase 18: Cognitive Export — bundle full state as shareable mind package (.zip/.k3d).

