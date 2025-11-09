# Character Embedding Compression Validation

**Dataset**: `/K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars/galaxy_character_embeddings.npz`
**Embeddings evaluated**: 4960
**Embedding dimension**: 128D
**Compression quality**: fast
**Dictionary**: `validation_cache/dictionary_128d_128.npz`

## Aggregate Metrics
- Average compression ratio (vs. 2048D baseline): **57.69:1**
- Per-dimension compression (128D): 3.61:1
- Min / Max compression ratio: 57.69 / 57.69
- Average cosine similarity: **0.999992**
- Min / Max similarity: 0.999988 / 0.999995
- Valid samples (≥ 0.99 threshold): 100.00%

## Codec Usage
- PD04 (dictionary): 0.00% (0 embeddings)
- PD02 (dense fallback): 100.00% (4960 embeddings)
- Simple fallback: 0.00% (0 embeddings)

## Comparison to Text Corpus

| Dataset | Dimension | Compression | Fidelity | Notes |
|---------|-----------|-------------|----------|-------|
| ai_compendium.txt | 128D | 69.4:1 | 0.99998 | Text embeddings |
| Character glyphs | 128D | 57.69:1 | 0.999992 | Visual embeddings |

## Character Samples

| Character | Embeddings | Avg Compression | Avg Fidelity | PD04 Usage |
|-----------|------------|-----------------|--------------|-----------|
| A | 80 | 57.69:1 | 0.999992 | 0.0% |
| B | 80 | 57.69:1 | 0.999993 | 0.0% |
| C | 80 | 57.69:1 | 0.999993 | 0.0% |
| D | 80 | 57.69:1 | 0.999993 | 0.0% |
| E | 80 | 57.69:1 | 0.999993 | 0.0% |
| F | 80 | 57.69:1 | 0.999993 | 0.0% |
| G | 80 | 57.69:1 | 0.999993 | 0.0% |
| H | 80 | 57.69:1 | 0.999993 | 0.0% |
| I | 80 | 57.69:1 | 0.999992 | 0.0% |
| J | 80 | 57.69:1 | 0.999992 | 0.0% |