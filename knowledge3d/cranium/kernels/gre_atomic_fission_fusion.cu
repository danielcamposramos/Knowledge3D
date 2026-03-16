// Atomic Fission/Fusion - compositional verification over small embedding sets.
//
// Mode 0 (fission):
//   Decompose a compound embedding against a small set of atom directions and
//   report how much of the compound is explained by those atoms.
//
// Mode 1 (fusion):
//   Build a weighted centroid over the atoms using atom-centroid agreement.

#include <math.h>

#define MAX_D 128
#define MAX_K 16

extern "C" __global__ void gre_atomic_fission_fusion(
    const float* __restrict__ compound,
    const float* __restrict__ atoms,
    float* __restrict__ result,
    float* __restrict__ consistency,
    int K,
    int D,
    int mode
)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (K <= 0 || D <= 0) {
        if (consistency != nullptr) {
            *consistency = 0.0f;
        }
        return;
    }

    if (K > MAX_K) {
        K = MAX_K;
    }
    if (D > MAX_D) {
        D = MAX_D;
    }

    const float eps = 1e-12f;

    if (mode == 0) {
        float residual[MAX_D];
        for (int d = 0; d < D; ++d) {
            residual[d] = compound[d];
        }

        for (int k = 0; k < K; ++k) {
            float dot = 0.0f;
            float norm = 0.0f;
            const int base = k * D;
            for (int d = 0; d < D; ++d) {
                const float atom = atoms[base + d];
                dot += residual[d] * atom;
                norm += atom * atom;
            }
            const float coeff = dot / (norm + eps);
            for (int d = 0; d < D; ++d) {
                residual[d] -= coeff * atoms[base + d];
            }
        }

        float residual_norm = 0.0f;
        float compound_norm = 0.0f;
        for (int d = 0; d < D; ++d) {
            result[d] = compound[d] - residual[d];
            residual_norm += residual[d] * residual[d];
            compound_norm += compound[d] * compound[d];
        }
        const float explained = 1.0f - (sqrtf(residual_norm) / (sqrtf(compound_norm) + eps));
        *consistency = fminf(1.0f, fmaxf(0.0f, explained));
        return;
    }

    float centroid[MAX_D];
    for (int d = 0; d < D; ++d) {
        float total = 0.0f;
        for (int k = 0; k < K; ++k) {
            total += atoms[k * D + d];
        }
        centroid[d] = total / (float)K;
    }

    float centroid_norm = 0.0f;
    for (int d = 0; d < D; ++d) {
        centroid_norm += centroid[d] * centroid[d];
    }
    centroid_norm = sqrtf(centroid_norm + eps);

    float weights[MAX_K];
    float weight_sum = 0.0f;
    float agreement_sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        float dot = 0.0f;
        float atom_norm = 0.0f;
        const int base = k * D;
        for (int d = 0; d < D; ++d) {
            const float atom = atoms[base + d];
            dot += atom * centroid[d];
            atom_norm += atom * atom;
        }
        float agreement = dot / (sqrtf(atom_norm + eps) * centroid_norm + eps);
        agreement = fmaxf(agreement, 0.0f);
        weights[k] = agreement;
        weight_sum += agreement;
        agreement_sum += agreement;
    }

    if (weight_sum <= eps) {
        weight_sum = (float)K;
        agreement_sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            weights[k] = 1.0f;
        }
    }

    for (int d = 0; d < D; ++d) {
        float blended = 0.0f;
        for (int k = 0; k < K; ++k) {
            blended += (weights[k] / weight_sum) * atoms[k * D + d];
        }
        result[d] = blended;
    }

    *consistency = fminf(1.0f, fmaxf(0.0f, agreement_sum / (float)K));
}
