# Maintenance Log

## Cleanup — Local Workspace Reset

- Date (UTC): 2025-09-07T07:58:36Z
- Local path: ../Knowledge3D.local
- Removed: models/* (except root-owned entries), logs/, mr/
- Kept: datasets/ (downloaded data), repos/ (source checkouts), conda_pkgs/ (caches)

Notes
- One or more root-owned artifacts could not be deleted without elevated privileges (e.g., models/world_rssm/rssm.pt). If desired, remove manually via:
  - sudo rm -rf ../Knowledge3D.local/models/world_rssm
- The goal is to start fresh while preserving downloaded datasets to save bandwidth/time.

Next
- Build fresh 50k multimodal GLBs using `docs/DATASETS_50K.md`.
- Export per-house memory to `viewer/public/houses/<K3D_HOUSE_ID>/` going forward.
