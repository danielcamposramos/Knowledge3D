# Post-Fullbench Fixes Status — 2026-03-19

## Scope

Applied the two confirmed post-full-benchmark fixes:

1. Removed the H19 ingestion cap from the enriched Galaxy pipeline.
2. Switched Math benchmark loading to the real dataset when present.

## Files Changed

- `scripts/ingest_meaning_layer.py`
- `scripts/run_enriched_benchmarks.py`
- `benchmarks/math_competitions.py`

## Validation

### H19 uncapped selection

Command result:

```python
{
  'selected': 42749,
  'stats': {
    'available': 117497,
    'min_languages': 5,
    'after_min_languages': 75970,
    'removed_for_min_languages': 41527,
    'after_stopwords': 75822,
    'stopwords_removed': 148,
    'after_foundation_dedup': 72413,
    'foundation_duplicates_removed': 3409,
    'after_keyword_filter': 42749,
    'keyword_filter_removed': 29664,
    'after_dedup': 42749,
    'dedup_removed': 0,
    'after_selection': 42749
  }
}
```

This confirms the Python-side quantity cap is gone. All quality filters remain active.

### Math dataset mode

Command result:

```python
{
  'dataset_mode': 'present',
  'dataset_path': '/K3D/K3D_llama_cpp/datasets',
  'problem_count': 500,
  'first_id': 'math_0'
}
```

This confirms Math now uses the real dataset instead of the 20-problem synthetic guard path.

## Next Step

Start a single clean rerun:

```bash
export CUDA_VISIBLE_DEVICES=0
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -u scripts/run_enriched_benchmarks.py --full
```

The run should append to the existing accumulative health log and trigger sleep-time consolidation at the end.

## Active Run

The fixed rerun is now active.

- Command: `CUDA_VISIBLE_DEVICES=0 /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -u scripts/run_enriched_benchmarks.py --full`
- Live log: `/tmp/k3d_full_benchmark_postfix_20260319.log`
- Resume state: `/K3D/Knowledge3D.local/logs/health_log.full.run_state.json`
