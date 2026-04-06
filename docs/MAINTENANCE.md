# Maintenance Log

## Sovereign Artifact Rebuild

Use the managed GPU environment and the canonical maintenance entrypoint:

```bash
env PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/rebuild_sovereign_artifact.py \
  --refresh-feed-source --refresh-build-feed --force-rebuild --verbose
```

This rebuild is mandatory after:
- foundational star schema changes
- layer, role, or ref wiring changes
- default knowledge signature changes

Operational rules:
- let the rebuild finish; do not interrupt it mid-write
- the rebuild loads the full resident Knowledgeverse, not a selective galaxy subset
- normal warm boot is not the artifact generation mechanism
- after rebuild, normal boot should load in `artifact` or `resident` mode
- `--refresh-feed-source` is the only maintenance step allowed to rehydrate source rows
- the build feed is the authoritative production rebuild input; if it is stale or missing, rebuild fails fast instead of translating source rows on boot

Storage policy:
- `gpu_cache/` keeps one active `flat_<signature>.npy` plus one paired `catalog_<signature>.pkl`
- heavy timestamped checkpoint families are rotated automatically:
  - keep `2` `galaxy_consolidated_*`
  - keep `2` `sovereign_runtime_bundle_*`
  - keep `3` `trm_weights_*`
  - keep `3` `shadow_patterns_*`

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
