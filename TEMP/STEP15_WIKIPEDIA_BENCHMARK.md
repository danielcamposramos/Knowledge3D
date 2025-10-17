# Step 15 – Wikipedia Sovereign Ingestion Benchmark

**Date**: 2025-10-16  
**Agent**: Codex  
**Environment**: `/K3D/Knowledge3D.local/envs/k3d-cranium` (RTX 3060)

## Test
```bash
pytest tests/test_step15_wikipedia_benchmark.py -s
```

## Articles
Processed 10 articles (en/pt/es) with `max_sentences=30`.

| # | Language | Title | Sentences | Total Latency (s) | Per Sentence (ms) |
|---|----------|-------|-----------|-------------------|-------------------|
| 1 | en | Artificial_intelligence | 30 | 0.10 | 3.41 |
| 2 | en | Machine_learning | 30 | 0.13 | 4.42 |
| 3 | en | Deep_learning | 30 | 0.11 | 3.74 |
| 4 | en | Natural_language_processing | 30 | 0.10 | 3.19 |
| 5 | en | Computer_vision | 30 | 0.11 | 3.74 |
| 6 | pt | Inteligência_artificial | 30 | 0.16 | 5.17 |
| 7 | pt | Aprendizado_de_máquina | 30 | 0.15 | 4.95 |
| 8 | es | Inteligencia_artificial | 30 | 0.21 | 6.99 |
| 9 | es | Aprendizaje_automático | 30 | 0.12 | 4.09 |
|10 | en | Neural_network | 30 | 0.11 | 3.73 |

## Summary
- ✅ Articles ingested: **10 / 10**
- ✅ Peak VRAM: **0.12 GB** (budget < 8 GB)
- ✅ Average latency per article: **0.14 s** (< 5 s target)
- ✅ Average per sentence: **4.64 ms**

The sovereign Wikipedia pipeline is ready for Phase A demonstrations.
