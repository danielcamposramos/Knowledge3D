# Adaptive Procedural Compression Guide

This guide describes how to use the Phase 2.6 adaptive compression stack that
combines Matryoshka dimensions with dictionary codecs (PD04).

## Quality Levels

| Quality | Dimension | Compression (vs 2048D) | Fidelity (avg) | Use Case |
| ------- | --------- | --------------------- | -------------- | -------- |
| ultrafast | 64D  | ~80× | 0.996 | Semantic search / routing |
| fast      | 128D | ~69× | 0.99998 | Default inference tier |
| balanced  | 512D | ~24× | 0.99998 | Complex reasoning |
| maximum   | 2048D| ~12× | 0.99996 | Highest fidelity |

## Getting Started

1. Train the dimension-specific dictionaries:
   ```
   PYTHONPATH=. python3 scripts/train_dictionary.py \
       --tokens-file data/ai_compendium.txt \
       --dimensions 64,128,512,2048 \
       --num-samples 5000 \
       --components 512 \
       --output-dir validation_cache \
       --report validation_results/dictionary_training.md
   ```

2. Instantiate the compressor:
   ```python
   from knowledge3d.cranium import AdaptiveDimensionCompressor
   compressor = AdaptiveDimensionCompressor()
   ```

3. Compress embeddings:
   ```python
   embedding = matryoshka_vector  # np.ndarray (2048D)
   program, metadata = compressor.compress(embedding, quality="fast", return_metadata=True)
   ```

4. Decompress on-demand:
   ```python
   recovered = compressor.decompress(program, metadata["target_dim"])
   ```

## Integration with Phase H

Use `PhaseHProceduralIntegration` to wire the compressor into existing Matryoshka pipelines:

```python
from knowledge3d.cranium import AdaptiveDimensionCompressor, PhaseHProceduralIntegration

compressor = AdaptiveDimensionCompressor()
integration = PhaseHProceduralIntegration(compressor)
program = integration.compress_embedding(matryoshka_vector, quality="balanced")
recovered = integration.decompress_embedding(program)
```

## Backward Compatibility

- Set `enable_compression=False` to return raw embeddings (legacy behaviour).
- Fallback to PD02 dense codec is automatic whenever dictionary compression cannot meet the configured fidelity threshold for a dimension.

## Validation Commands

```
python3 -m pytest knowledge3d/cranium/tests/test_procedural_compression.py \
                   knowledge3d/cranium/tests/test_prototype_delta.py \
                   knowledge3d/cranium/tests/test_adaptive_compression.py -v

PYTHONPATH=. python3 examples/adaptive_compression_demo.py
```

All validation reports live under `validation_results/dictionary_compression_<dim>d.*`. Use these when reporting progress to Milton / Claude.
