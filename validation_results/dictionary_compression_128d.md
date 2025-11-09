# Prototype-Delta Compression Validation

**Tokens evaluated:** 1000
**Embedding dim:** 2048
**Target dim:** 128
**Prototype table:** `validation_cache/prototype_table_2048d_512.npz`
**Top-K range:** min 0 / max 0

## Aggregate Metrics
- Average compression ratio: **4.34:1**
- Min / Max compression ratio: 2.84 / 6.10
- Average cosine similarity: **0.99998**
- Valid samples (≥ threshold): 100.00%

## Additional Statistics
- Average nnz (top-k corrections): 0.0
- Prototype distance (avg/max): 0.0000 / 0.0000

## JSON Metrics
```json
{
  "count": 1000,
  "average_compression": 4.342444809928383,
  "average_similarity": 0.9999795354604721,
  "min_similarity": 0.9999562501907349,
  "max_similarity": 0.9999895691871643,
  "valid_ratio": 1.0,
  "tokens_evaluated": 1000,
  "embedding_dim": 2048,
  "target_dimension": 128,
  "prototype_table": "validation_cache/prototype_table_2048d_512.npz",
  "compression_ratio_min": 2.8444444444444446,
  "compression_ratio_max": 6.095238095238095,
  "average_topk": 0.0,
  "min_topk": 0,
  "max_topk": 0,
  "proto_distance_avg": 0.0,
  "proto_distance_max": 0.0,
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