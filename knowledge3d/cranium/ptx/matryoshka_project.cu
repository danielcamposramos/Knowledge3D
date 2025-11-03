/*
 * matryoshka_project.cu - GPU projection for Matryoshka TRM
 *
 * Computes y = W[:dim, :dim] * x where W is stored row-major with stride equal
 * to the full Matryoshka capacity (max_dim). Follows sovereign safety rules:
 * NaN/Inf guards on inputs and outputs plus relaxed ±10 clipping.
 */

extern "C" __global__ void matryoshka_project(
    const float* __restrict__ weights,  // Full matrix [max_dim, max_dim]
    const float* __restrict__ vector,   // Input vector [target_dim]
    float* __restrict__ output,         // Output vector [target_dim]
    int target_dim,
    int stride                          // Row stride (max_dim)
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= target_dim) {
        return;
    }

    float acc = 0.0f;
    for (int col = 0; col < target_dim; ++col) {
        float w = weights[row * stride + col];
        float v = vector[col];

        if (isnan(w) || isinf(w)) {
            w = 0.0f;
        }
        if (isnan(v) || isinf(v)) {
            v = 0.0f;
        }

        acc += w * v;
    }

    if (isnan(acc) || isinf(acc)) {
        acc = 0.0f;
    }

    // Optional clipping for stability
    acc = fmaxf(fminf(acc, 10.0f), -10.0f);

    output[row] = acc;
}
