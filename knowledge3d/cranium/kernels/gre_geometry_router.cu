// Geometry Router - pairwise spatial relationship features between embeddings.

#include <math.h>

extern "C" __global__ void gre_geometry_router(
    const float* __restrict__ embedding_a,
    const float* __restrict__ embedding_b,
    float* __restrict__ relations,
    int N,
    int D,
    int R
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const float eps = 1e-12f;

    for (; i < N; i += stride) {
        const float* a = embedding_a + (i * D);
        const float* b = embedding_b + (i * D);
        float* out = relations + (i * R);

        float dot_ab = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot_ab += a[d] * b[d];
            norm_a += a[d] * a[d];
            norm_b += b[d] * b[d];
        }
        float inv_norms = rsqrtf(norm_a + eps) * rsqrtf(norm_b + eps);
        out[0] = dot_ab * inv_norms;

        float l2 = 0.0f;
        for (int d = 0; d < D; ++d) {
            float diff = a[d] - b[d];
            l2 += diff * diff;
        }
        out[1] = sqrtf(l2) / (sqrtf((float)D) + eps);

        int q_size = max(1, D / 4);
        for (int q = 0; q < 4; ++q) {
            float qd = 0.0f;
            float qa = 0.0f;
            float qb = 0.0f;
            int q_start = q * q_size;
            int q_end = (q == 3) ? D : min(D, (q + 1) * q_size);
            for (int d = q_start; d < q_end; ++d) {
                qd += a[d] * b[d];
                qa += a[d] * a[d];
                qb += b[d] * b[d];
            }
            out[2 + q] = qd * rsqrtf((qa + eps) * (qb + eps));
        }

        float ratio_sum = 0.0f;
        float ratio_sq = 0.0f;
        float ratio_max = -1e30f;
        float ratio_min = 1e30f;
        for (int d = 0; d < D; ++d) {
            float denom = b[d];
            if (fabsf(denom) < 1e-8f) {
                denom = copysignf(1e-8f, denom == 0.0f ? 1.0f : denom);
            }
            float ratio = a[d] / denom;
            ratio_sum += ratio;
            ratio_sq += ratio * ratio;
            ratio_max = fmaxf(ratio_max, ratio);
            ratio_min = fminf(ratio_min, ratio);
        }
        float ratio_mean = ratio_sum / (float)D;
        out[6] = ratio_mean;
        out[7] = sqrtf(fmaxf((ratio_sq / (float)D) - (ratio_mean * ratio_mean), 0.0f));
        out[8] = ratio_max;
        out[9] = ratio_min;

        float best_corr = -1.0f;
        int best_offset = 0;
        for (int shift = 0; shift < D; ++shift) {
            float corr = 0.0f;
            for (int d = 0; d < D; ++d) {
                corr += a[d] * b[(d + shift) % D];
            }
            corr *= inv_norms;
            if (corr > best_corr) {
                best_corr = corr;
                best_offset = shift;
            }
        }
        out[10] = best_corr;
        out[11] = (float)best_offset / (float)max(D, 1);

        int sign_agree = 0;
        int a_dominates = 0;
        for (int d = 0; d < D; ++d) {
            if ((a[d] >= 0.0f) == (b[d] >= 0.0f)) {
                sign_agree += 1;
            }
            if (fabsf(a[d]) > fabsf(b[d])) {
                a_dominates += 1;
            }
        }
        out[12] = (float)sign_agree / (float)D;
        out[13] = (float)a_dominates / (float)D;

        float proj_coeff = dot_ab / (norm_b + eps);
        float residual_sq = 0.0f;
        for (int d = 0; d < D; ++d) {
            float residual = a[d] - (proj_coeff * b[d]);
            residual_sq += residual * residual;
        }
        out[14] = sqrtf(residual_sq) / (sqrtf(norm_a) + eps);
        out[15] = residual_sq / (norm_a + eps);
    }
}
