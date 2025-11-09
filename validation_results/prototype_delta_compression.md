# Prototype-Delta Compression Validation

**Tokens evaluated:** 1000
**Embedding dim:** 2048
**Prototype table:** `validation_cache/prototype_table_2048d_512.npz`
**Top-K range:** min 0 / max 0

## Aggregate Metrics
- Average compression ratio: **3.97:1**
- Min / Max compression ratio: 3.97 / 3.97
- Average cosine similarity: **0.99995**
- Valid samples (≥ threshold): 100.00%

## Additional Statistics
- Average nnz (top-k corrections): 0.0
- Prototype distance (avg/max): 0.7920 / 0.9156

## JSON Metrics
```json
{
  "count": 1000,
  "average_compression": 3.9728419010669267,
  "average_similarity": 0.9999466901421546,
  "min_similarity": 0.9998757839202881,
  "max_similarity": 0.9999728202819824,
  "valid_ratio": 1.0,
  "tokens_evaluated": 1000,
  "embedding_dim": 2048,
  "prototype_table": "validation_cache/prototype_table_2048d_512.npz",
  "compression_ratio_min": 3.9728419010669254,
  "compression_ratio_max": 3.9728419010669254,
  "average_topk": 0.0,
  "min_topk": 0,
  "max_topk": 0,
  "proto_distance_avg": 0.7920002338290214,
  "proto_distance_max": 0.9156246185302734,
  "threshold": 0.99,
  "topk": 16,
  "topk_step": 8,
  "topk_cap": 128,
  "tokens_file": [
    "data/ai_compendium.txt"
  ],
  "use_basis": false,
  "codec": "dense"
}
```