/**
 * VectorDotMap encoder — compress images to compact field coefficients.
 */

extern "C" {

// Reduce DCT coefficients to field coefficients using importance weights.
__global__ void field_coefficient_fit(
    const float* dct_coeffs,     // (num_blocks, 64)
    float* field_coeffs,         // (n_coeffs)
    const float* importance,     // (64)
    int num_blocks,
    int n_coeffs
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_coeffs) return;

    int freq_idx = k % 64;
    int stride = max(1, num_blocks / max(1, n_coeffs / 64));
    float sum = 0.0f;
    float wsum = 0.0f;
    for (int b = 0; b < num_blocks; b += stride) {
        float w = importance[freq_idx];
        sum += w * dct_coeffs[b * 64 + freq_idx];
        wsum += w;
    }
    field_coeffs[k] = (wsum > 0.0f) ? sum / wsum : 0.0f;
}

// Expand field coefficients back to DCT coefficients (simple replication).
__global__ void field_coefficient_expand(
    const float* field_coeffs,
    float* dct_coeffs,
    const float* importance,
    int num_blocks,
    int n_coeffs
) {
    int b = blockIdx.x;
    int f = threadIdx.x;  // 0..63
    if (b >= num_blocks || f >= 64) return;

    int field_idx = f + (b * 64 / max(1, num_blocks)) * 1;
    field_idx = min(field_idx, n_coeffs - 1);
    dct_coeffs[b * 64 + f] = field_coeffs[field_idx] * importance[f];
}

__device__ __constant__ float FREQ_IMPORTANCE[64] = {
    1.00f, 0.98f, 0.95f, 0.90f, 0.83f, 0.75f, 0.65f, 0.55f,
    0.98f, 0.95f, 0.90f, 0.85f, 0.78f, 0.70f, 0.60f, 0.50f,
    0.95f, 0.90f, 0.85f, 0.80f, 0.73f, 0.65f, 0.55f, 0.45f,
    0.90f, 0.85f, 0.80f, 0.75f, 0.68f, 0.60f, 0.50f, 0.40f,
    0.83f, 0.78f, 0.73f, 0.68f, 0.63f, 0.55f, 0.45f, 0.35f,
    0.75f, 0.70f, 0.65f, 0.60f, 0.55f, 0.48f, 0.40f, 0.30f,
    0.65f, 0.60f, 0.55f, 0.50f, 0.45f, 0.38f, 0.30f, 0.22f,
    0.55f, 0.50f, 0.45f, 0.40f, 0.35f, 0.28f, 0.20f, 0.15f,
};

}  // extern "C"
