# Procedural Compression Summary (Phase 2.6)

| Phase | Technique | Average Compression | Fidelity (avg) | Notes |
| --- | --- | --- | --- | --- |
| 1.0 | Simple quantised residual (2048D) | 3.88 : 1 | 0.9999 | `validation_results/procedural_compression_proof.md` |
| 2.2 | Prototype dense PD02 (2048D) | 3.97 : 1 | 0.99995 | `validation_results/prototype_delta_compression.md` |
| 2.5 | Dictionary PD04 (2048D) | 12.0 : 1 | 0.99996 | `validation_results/dictionary_compression_2048d.md` |
| 2.6 | Adaptive PD04 (128D) | 69.4 : 1 | 0.99998 | `validation_results/dictionary_compression_128d.md` |
| 2.6 | Adaptive PD04 (64D) | 80.6 : 1 | 0.9963 | `validation_results/dictionary_compression_64d.md` |
| 2.6 | Adaptive PD04 (512D) | 24.2 : 1 | 0.99998 | `validation_results/dictionary_compression_512d.md` |

All metrics measured on 1 000 samples from `data/ai_compendium.txt`. Dictionaries generated via `scripts/train_dictionary.py`.
