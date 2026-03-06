# Temporal Route Promotion Benchmark

Date: 2026-03-06
Route family: `surface_material_to_timeline_preset`

## Result

- `ui_idle`: `80.227 ms` -> `75.899 ms`
- `world_breathe`: `129.709 ms` -> `120.361 ms`

## Interpretation

The temporal preset PTX path was correct but not sufficient alone. The meaningful reduction came after promoting procedural frame synthesis inside the temporal bridge. The route is improved, but not yet closed; the remaining heavier work is still in broader timeline orchestration and encode-time bookkeeping.
