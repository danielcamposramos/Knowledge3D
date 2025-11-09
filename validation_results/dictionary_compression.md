# Prototype-Delta Compression Validation

**Tokens evaluated:** 1000
**Embedding dim:** 2048
**Prototype table:** `validation_cache/prototype_table_2048d_512.npz`
**Top-K range:** min 0 / max 0

## Aggregate Metrics
- Average compression ratio: **11.91:1**
- Min / Max compression ratio: 3.97 / 18.79
- Average cosine similarity: **0.99996**
- Valid samples (≥ threshold): 100.00%

## Additional Statistics
- Average nnz (top-k corrections): 0.0
- Prototype distance (avg/max): 0.3770 / 0.9156

## JSON Metrics
```json
{
  "count": 1000,
  "average_compression": 11.91429772466386,
  "average_similarity": 0.9999571704268455,
  "min_similarity": 0.9998757839202881,
  "max_similarity": 0.999970018863678,
  "valid_ratio": 1.0,
  "tokens_evaluated": 1000,
  "embedding_dim": 2048,
  "prototype_table": "validation_cache/prototype_table_2048d_512.npz",
  "compression_ratio_min": 3.9728419010669254,
  "compression_ratio_max": 18.788990825688074,
  "average_topk": 0.0,
  "min_topk": 0,
  "max_topk": 0,
  "proto_distance_avg": 0.3770150476694107,
  "proto_distance_max": 0.9156246185302734,
  "threshold": 0.99,
  "topk": 16,
  "topk_step": 8,
  "topk_cap": 128,
  "tokens_file": [
    "data/ai_compendium.txt"
  ],
  "use_basis": false,
  "codec": "dict"
}
```