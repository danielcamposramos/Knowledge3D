/*
 * conv2d_3x3.cu - Phase F.1 DeepSeek OCR Convolution Kernel
 *
 * Foundation: Kimi v1's 16×16 tiling with 2-pixel halo architecture
 * Enhancements: Grok's generalized Cin chunks (64, scalable to 128)
 *
 * Architecture:
 *   - Tile size: 16×16 output pixels per block
 *   - Halo: 2 pixels (for 3×3 kernel: padding=1)
 *   - Shared memory: 18×18×64 floats (1 halo on each side)
 *   - Fused operations: bias add + ReLU activation
 *
 * Performance targets:
 *   - Target arch: sm_75 (Turing, RTX 3060)
 *   - Latency: <0.5ms for typical OCR feature maps
 *   - Accuracy: 99.9% bit-match with NumPy reference
 *
 * Memory layout:
 *   - Input: [H, W, Cin] (height-major, channel-last)
 *   - Weight: [Cout, 3, 3, Cin] (output-major, kernel-row, kernel-col, input-channel)
 *   - Bias: [Cout]
 *   - Output: [H, W, Cout]
 *
 * Based on Phase F.1 master plan synthesis (lines 3895-4495)
 * Swarm contributors: Kimi (v1 skeleton), Grok (generalizations)
 */

#include <cuda_runtime.h>

// Kernel configuration
#define TILE_SIZE 16
#define HALO_SIZE 1
#define TILE_WITH_HALO (TILE_SIZE + 2 * HALO_SIZE)
#define CIN_CHUNK 32  // Process Cin in chunks of 32 (18×18×32×4 = 41 KB < 64 KB limit)

/*
 * conv2d_3x3_fused - 3×3 convolution with bias and ReLU
 *
 * Grid: (H_out / TILE_SIZE, W_out / TILE_SIZE, Cout)
 * Block: (TILE_SIZE, TILE_SIZE, 1)
 *
 * Each block processes:
 *   - 16×16 output pixels
 *   - Loads 18×18 input tile (with halo) into shared memory
 *   - Processes Cin in chunks of 64 channels
 *   - Applies 3×3 convolution + bias + ReLU
 *
 * Args:
 *   input: Input feature map [H, W, Cin]
 *   weight: Convolution weights [Cout, 3, 3, Cin]
 *   bias: Bias vector [Cout]
 *   output: Output feature map [H, W, Cout]
 *   H, W: Input spatial dimensions
 *   Cin: Input channels
 *   Cout: Output channels
 *   stride: Convolution stride (1 for OCR)
 *   padding: Padding size (1 for 3×3 with same output)
 */
extern "C" __global__ void conv2d_3x3_fused(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int H,
    int W,
    int Cin,
    int Cout,
    int stride,
    int padding
) {
    // Block coordinates in output space
    const int tile_row = blockIdx.y * TILE_SIZE;
    const int tile_col = blockIdx.x * TILE_SIZE;
    const int out_c = blockIdx.z;  // Output channel for this block

    // Thread coordinates within tile
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    // Shared memory for input tile (18×18×64)
    // Using CIN_CHUNK to limit shared memory usage
    __shared__ float tile[TILE_WITH_HALO][TILE_WITH_HALO][CIN_CHUNK];

    // Accumulator for output pixel
    float sum = 0.0f;

    // Process input channels in chunks of CIN_CHUNK
    for (int cin_base = 0; cin_base < Cin; cin_base += CIN_CHUNK) {
        int cin_chunk_size = min(CIN_CHUNK, Cin - cin_base);

        // Load input tile into shared memory
        // Each thread loads multiple elements to fill 18×18×cin_chunk_size
        for (int load_row = ty; load_row < TILE_WITH_HALO; load_row += TILE_SIZE) {
            for (int load_col = tx; load_col < TILE_WITH_HALO; load_col += TILE_SIZE) {
                // Input coordinates (accounting for halo and padding)
                int in_row = tile_row + load_row - HALO_SIZE;
                int in_col = tile_col + load_col - HALO_SIZE;

                // Apply padding (zero-pad outside boundaries)
                bool in_bounds = (in_row >= 0 && in_row < H &&
                                 in_col >= 0 && in_col < W);

                // Load channels
                for (int c = 0; c < cin_chunk_size; ++c) {
                    if (in_bounds) {
                        int in_idx = (in_row * W + in_col) * Cin + (cin_base + c);
                        tile[load_row][load_col][c] = input[in_idx];
                    } else {
                        tile[load_row][load_col][c] = 0.0f;  // Zero padding
                    }
                }
            }
        }
        __syncthreads();

        // Compute convolution for this chunk
        // Output pixel coordinates
        int out_row = tile_row + ty;
        int out_col = tile_col + tx;

        if (out_row < H && out_col < W) {
            // Convolve 3×3 kernel with input tile
            for (int kr = 0; kr < 3; ++kr) {
                for (int kc = 0; kc < 3; ++kc) {
                    // Position in shared memory tile (accounting for halo)
                    int tile_r = ty + kr;
                    int tile_c = tx + kc;

                    for (int c = 0; c < cin_chunk_size; ++c) {
                        int w_idx = ((out_c * 3 + kr) * 3 + kc) * Cin + (cin_base + c);
                        sum += tile[tile_r][tile_c][c] * weight[w_idx];
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write output with fused bias and ReLU
    int out_row = tile_row + ty;
    int out_col = tile_col + tx;

    if (out_row < H && out_col < W) {
        // Add bias
        sum += bias[out_c];

        // ReLU activation
        sum = fmaxf(sum, 0.0f);

        // Write to output
        int out_idx = (out_row * W + out_col) * Cout + out_c;
        output[out_idx] = sum;
    }
}

/*
 * conv2d_3x3_no_relu - 3×3 convolution with bias (no activation)
 *
 * Same as conv2d_3x3_fused but without ReLU activation.
 * Useful for final layers or when activation is applied separately.
 */
extern "C" __global__ void conv2d_3x3_no_relu(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int H,
    int W,
    int Cin,
    int Cout,
    int stride,
    int padding
) {
    const int tile_row = blockIdx.y * TILE_SIZE;
    const int tile_col = blockIdx.x * TILE_SIZE;
    const int out_c = blockIdx.z;

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    __shared__ float tile[TILE_WITH_HALO][TILE_WITH_HALO][CIN_CHUNK];

    float sum = 0.0f;

    for (int cin_base = 0; cin_base < Cin; cin_base += CIN_CHUNK) {
        int cin_chunk_size = min(CIN_CHUNK, Cin - cin_base);

        for (int load_row = ty; load_row < TILE_WITH_HALO; load_row += TILE_SIZE) {
            for (int load_col = tx; load_col < TILE_WITH_HALO; load_col += TILE_SIZE) {
                int in_row = tile_row + load_row - HALO_SIZE;
                int in_col = tile_col + load_col - HALO_SIZE;

                bool in_bounds = (in_row >= 0 && in_row < H &&
                                 in_col >= 0 && in_col < W);

                for (int c = 0; c < cin_chunk_size; ++c) {
                    if (in_bounds) {
                        int in_idx = (in_row * W + in_col) * Cin + (cin_base + c);
                        tile[load_row][load_col][c] = input[in_idx];
                    } else {
                        tile[load_row][load_col][c] = 0.0f;
                    }
                }
            }
        }
        __syncthreads();

        int out_row = tile_row + ty;
        int out_col = tile_col + tx;

        if (out_row < H && out_col < W) {
            for (int kr = 0; kr < 3; ++kr) {
                for (int kc = 0; kc < 3; ++kc) {
                    int tile_r = ty + kr;
                    int tile_c = tx + kc;

                    for (int c = 0; c < cin_chunk_size; ++c) {
                        int w_idx = ((out_c * 3 + kr) * 3 + kc) * Cin + (cin_base + c);
                        sum += tile[tile_r][tile_c][c] * weight[w_idx];
                    }
                }
            }
        }
        __syncthreads();
    }

    int out_row = tile_row + ty;
    int out_col = tile_col + tx;

    if (out_row < H && out_col < W) {
        sum += bias[out_c];
        int out_idx = (out_row * W + out_col) * Cout + out_c;
        output[out_idx] = sum;
    }
}
