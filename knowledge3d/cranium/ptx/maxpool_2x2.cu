/*
 * maxpool_2x2.cu - 2×2 Max Pooling Kernel
 *
 * Implements spatial downsampling via 2×2 max pooling.
 * Used in OCR pipeline to reduce spatial dimensions while
 * preserving important features.
 *
 * Architecture:
 *   - Kernel size: 2×2
 *   - Stride: 2 (non-overlapping)
 *   - Output size: H/2 × W/2
 *
 * Performance target: <100µs for 256×256 input on RTX 3060
 */

#include <cuda_runtime.h>

#define TILE_SIZE 16  // Process 16×16 output tiles

/*
 * maxpool_2x2 - 2×2 max pooling with stride 2
 *
 * Grid: ((W/2 + TILE_SIZE - 1) / TILE_SIZE, (H/2 + TILE_SIZE - 1) / TILE_SIZE, C)
 * Block: (TILE_SIZE, TILE_SIZE, 1)
 *
 * Args:
 *   input: Input feature map [H, W, C]
 *   output: Output feature map [H/2, W/2, C]
 *   H, W: Input spatial dimensions
 *   C: Number of channels
 */
extern "C" __global__ void maxpool_2x2(
    const float* __restrict__ input,
    float* __restrict__ output,
    int H,
    int W,
    int C
) {
    const int out_col = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int out_row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int c = blockIdx.z;

    const int H_out = H / 2;
    const int W_out = W / 2;

    if (out_row >= H_out || out_col >= W_out) return;

    // Input coordinates (2× for stride 2)
    const int in_row = out_row * 2;
    const int in_col = out_col * 2;

    // Load 2×2 window
    float val00 = input[(in_row * W + in_col) * C + c];
    float val01 = input[(in_row * W + in_col + 1) * C + c];
    float val10 = input[((in_row + 1) * W + in_col) * C + c];
    float val11 = input[((in_row + 1) * W + in_col + 1) * C + c];

    // Compute max
    float max_val = fmaxf(fmaxf(val00, val01), fmaxf(val10, val11));

    // Write output
    output[(out_row * W_out + out_col) * C + c] = max_val;
}

/*
 * maxpool_2x2_indices - 2×2 max pooling with index tracking
 *
 * Tracks which input pixel was selected as max (useful for unpooling).
 *
 * Grid: ((W/2 + TILE_SIZE - 1) / TILE_SIZE, (H/2 + TILE_SIZE - 1) / TILE_SIZE, C)
 * Block: (TILE_SIZE, TILE_SIZE, 1)
 *
 * Args:
 *   input: Input feature map [H, W, C]
 *   output: Output feature map [H/2, W/2, C]
 *   indices: Max indices [H/2, W/2, C] (0-3 for 2×2 window)
 *   H, W: Input spatial dimensions
 *   C: Number of channels
 */
extern "C" __global__ void maxpool_2x2_indices(
    const float* __restrict__ input,
    float* __restrict__ output,
    int* __restrict__ indices,
    int H,
    int W,
    int C
) {
    const int out_col = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int out_row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int c = blockIdx.z;

    const int H_out = H / 2;
    const int W_out = W / 2;

    if (out_row >= H_out || out_col >= W_out) return;

    const int in_row = out_row * 2;
    const int in_col = out_col * 2;

    // Load 2×2 window
    float vals[4];
    vals[0] = input[(in_row * W + in_col) * C + c];
    vals[1] = input[(in_row * W + in_col + 1) * C + c];
    vals[2] = input[((in_row + 1) * W + in_col) * C + c];
    vals[3] = input[((in_row + 1) * W + in_col + 1) * C + c];

    // Find max and its index
    float max_val = vals[0];
    int max_idx = 0;

    #pragma unroll
    for (int i = 1; i < 4; ++i) {
        if (vals[i] > max_val) {
            max_val = vals[i];
            max_idx = i;
        }
    }

    // Write outputs
    int out_idx = (out_row * W_out + out_col) * C + c;
    output[out_idx] = max_val;
    indices[out_idx] = max_idx;
}

/*
 * avgpool_2x2 - 2×2 average pooling with stride 2
 *
 * Alternative to max pooling that computes average of 2×2 window.
 *
 * Grid: ((W/2 + TILE_SIZE - 1) / TILE_SIZE, (H/2 + TILE_SIZE - 1) / TILE_SIZE, C)
 * Block: (TILE_SIZE, TILE_SIZE, 1)
 */
extern "C" __global__ void avgpool_2x2(
    const float* __restrict__ input,
    float* __restrict__ output,
    int H,
    int W,
    int C
) {
    const int out_col = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int out_row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int c = blockIdx.z;

    const int H_out = H / 2;
    const int W_out = W / 2;

    if (out_row >= H_out || out_col >= W_out) return;

    const int in_row = out_row * 2;
    const int in_col = out_col * 2;

    // Load 2×2 window
    float sum = 0.0f;
    sum += input[(in_row * W + in_col) * C + c];
    sum += input[(in_row * W + in_col + 1) * C + c];
    sum += input[((in_row + 1) * W + in_col) * C + c];
    sum += input[((in_row + 1) * W + in_col + 1) * C + c];

    // Compute average
    float avg_val = sum * 0.25f;

    // Write output
    output[(out_row * W_out + out_col) * C + c] = avg_val;
}
