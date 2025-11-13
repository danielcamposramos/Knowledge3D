# Adaptive Procedural Compression Specification

**Version**: 1.0
**Status**: Production
**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Implementation)
**Date**: November 2025

---

## Abstract

This specification defines a procedural codec for spatial knowledge embeddings. Instead of storing dense vectors, systems store compact programs that reconstruct embeddings on-demand at target dimensions. The approach achieves 12–80× compression with near-lossless fidelity and deterministic GPU execution.

---

## 1. Model Overview

### 1.1 Terms

- Target Dimension (D): one of {64, 128, 512, 2048}
- Prototype Dictionary: set of K centroid vectors for dimension D
- Delta: sparse correction applied to a prototype
- Program: fixed-width sequence of opcodes and parameters

### 1.2 Program Structure

```
program := [LOAD_PROTOTYPE i] [ADD_DELTA δ_sparse] [NORMALIZE]
```

Where `i` is a dictionary index and `δ_sparse` is an index/value list with thresholding.

---

## 2. Codec Definition (PD04)

### 2.1 Offline Dictionary Training (Non-normative)

Implementations typically compute K=512 prototypes per dimension D from N≈5000 samples using K-means. Prototypes are stored in constant memory for fast access.

### 2.2 Compression (Normative)

Given an input embedding E of dimension 2048:

1. Select target D ∈ {64, 128, 512, 2048}.
2. Truncate E to D (Matryoshka ordering preserved).
3. Select nearest prototype `P_i` from dictionary for D.
4. Compute delta `δ = E[:D] - P_i`; sparsify via threshold τ.
5. Emit program with `[LOAD_PROTOTYPE i] [ADD_DELTA δ_sparse] [NORMALIZE]`.

### 2.3 Decompression (Normative)

Execute program on GPU using PTX kernels:

- LOAD_PROTOTYPE: fetch `P_i` from constant memory
- ADD_DELTA: sparse add with SIMD
- NORMALIZE: L2 normalization via warp reduction

Output: reconstructed embedding `E_recon` with dimension D.

---

## 3. Quality Levels

| Level | D   | Typical Size | Compression vs 2048D | Fidelity (cosine) |
|-------|-----|--------------|----------------------|-------------------|
| ultrafast | 64  | ~100 B      | ~80×                 | ≥ 0.996           |
| fast      | 128 | ~115 B      | ~69×                 | ≥ 0.99998         |
| balanced  | 512 | ~340 B      | ~24×                 | ≥ 0.99998         |
| maximum   | 2048| ~700 B      | ~12×                 | ≥ 0.99996         |

Fidelity thresholds are measured on validation sets and MUST be verified per deployment.

---

## 4. Interoperability

- Programs are byte-addressable and MAY be embedded in glTF via K3D extensions.
- Matryoshka ordering ensures monotonic refinement across dimensions.
- Deterministic PTX kernel execution ensures reproducibility on consumer GPUs.

---

## 5. Security and Safety

- Program size caps MUST be enforced to avoid OOM or DOS vectors.
- Dictionaries SHOULD be versioned and integrity-checked (hashes).

---

## 6. Conformance

An implementation conforms if it:

- Emits valid PD04 programs for D ∈ {64, 128, 512, 2048}
- Reconstructs embeddings within declared fidelity bounds
- Executes programs deterministically on GPU

---

## 7. W3C Insertion Mapping

This specification operationalizes `TEMP/W3C_INSERTION_9_PROCEDURAL_COMPRESSION.md` and is referenced by `docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md` for memory efficiency guarantees.

