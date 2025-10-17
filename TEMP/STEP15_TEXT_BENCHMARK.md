# Step 15 – Sovereign Text Pipeline Benchmark

**Date**: 2025-10-16  
**Agent**: Codex  
**Environment**: `/K3D/Knowledge3D.local/envs/k3d-cranium` (RTX 3060, gensim not installed → fallback to zero bootstrap vectors)

## Benchmark Setup

```python
from knowledge3d.ingestion.language.sovereign_text_pipeline import SovereignTextIngestor

words = [f"token_{i}" for i in range(1000)]
ingestor = SovereignTextIngestor(languages=['en'])
positions = ingestor.ingest_vocabulary('en', words)
```

## Results

| Metric                     | Value            |
|---------------------------|------------------|
| Tokens processed          | 1,000            |
| Wall-clock latency        | **0.42 s**       |
| Output shape              | (1000, 3)        |
| Peak VRAM (`nvidia-smi`)  | < 400 MB         |

## Notes
- gensim is not currently installed in the environment; the pipeline fell back to zero bootstrap vectors (warning emitted). Once gensim is available, the same pipeline will automatically leverage the 50-d GloVe seeds for richer initial embeddings.
- Even with fallback, the sovereign pipeline meets the sub-second target for 1k tokens and stays well within the 12 GB VRAM budget.
