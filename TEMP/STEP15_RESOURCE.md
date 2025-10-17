# Step 15 – Resource Controller Validation

**Date**: 2025-10-16  
**Agent**: Codex  
**Environment**: `/K3D/Knowledge3D.local/envs/k3d-cranium` (RTX 3060)

## Objective
Validate the `ResourceSafeIngestionController` with VRAM monitoring and spill-to-House behaviour under an intentionally tiny budget (0.5 MB) to force overflow logic.

## Test Script
```python
from knowledge3d.ingestion.language.resource_controller import ResourceSafeIngestionController

controller = ResourceSafeIngestionController(vram_budget_gb=0.0005)

def handler(batch):
    return [item.upper() for item in batch]

text_items = [f"sentence_{i}" for i in range(300)]
results = controller.batch_ingest_linear(
    text_items=text_items,
    text_handler=handler,
    audio_items=None,
    audio_handler=None,
    visual_items=None,
    visual_handler=None,
    batch_size=64,
)
```

## Observations
- VRAM check reports (`get_vram_usage`) were taken before each batch.
- Every batch exceeded the 0.5 MB limit, triggering `OOMSpillManager` planning and House spill.
- LatencyGuard stayed below the 95 µs threshold (34–48 µs per batch).
- Spill artefact written to: `../Knowledge3D.local/house_spill/text-spill-20251017-014258.json`.

### Console Output
```
Starting resource-safe ingestion (linear sequence).
Ingesting text modality (300 items)…
VRAM budget exceeded for text: used=115.88 MB, estimate=128.00 KB (budget=488.28 KB). Spilling to House.
[text] batch 0 size=64 latency=34.82µs
...
```

## Conclusion
- `ResourceSafeIngestionController` successfully:
  - Monitors live VRAM usage via the new `get_vram_usage()`.
  - Spills overflow batches to House storage.
  - Maintains latency guardrails (<95 µs).
- Ready for integration with sovereign text/audio/visual pipelines using a realistic 8 GB budget.
