/*
 * conv2d_3x3_backward.cu - Conv2D 3×3 Backward Pass
 *
 * Computes gradients for 3×3 convolution:
 *   - d_weight: gradient w.r.t. filter weights [Cout, 3, 3, Cin]
 *   - d_bias: gradient w.r.t. biases [Cout]
 *   - d_input: gradient w.r.t. input [H_in, W_in, Cin]
 *
 * Uses atomic operations for gradient accumulation.
 * Multiple kernels for efficiency:
 *   - conv2d_backward_weight: compute d_weight and d_bias
 *   - conv2d_backward_input: compute d_input
 */

// ============================================================================
// Kernel 1: Backward w.r.t. Weights and Biases
// ============================================================================
extern "C" __global__ void conv2d_backward_weight(
    const float* __restrict__ d_out,      // Gradient from next layer [H_out, W_out, Cout]
    const float* __restrict__ x_in,       // Input (padded) [H_in+2, W_in+2, Cin]
    float* __restrict__ d_weight,         // Output: d_weight [Cout, 3, 3, Cin]
    float* __restrict__ d_bias,           // Output: d_bias [Cout]
    int H_out,
    int W_out,
    int H_in,
    int W_in,
    int Cin,
    int Cout
) {
    // Each block handles one output channel
    int cout = blockIdx.x;
    if (cout >= Cout) return;

    int tid = threadIdx.x;
    int threads_per_block = blockDim.x;

    // Shared memory for bias reduction
    __shared__ float s_bias[256];
    float local_bias = 0.0f;

    // Compute d_weight and accumulate d_bias
    // Each thread processes subset of output spatial positions
    for (int out_h = 0; out_h < H_out; out_h++) {
        for (int out_w = tid; out_w < W_out; out_w += threads_per_block) {
            int out_idx = (out_h * W_out + out_w) * Cout + cout;
            float grad = d_out[out_idx];

            // Accumulate bias gradient (no clipping - rely on small LR)
            local_bias += grad;

            // Input window starts (accounting for padding)
            int in_h_start = out_h;  // stride=1
            int in_w_start = out_w;

            // Accumulate weight gradients
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    for (int cin = 0; cin < Cin; cin++) {
                        int in_h = in_h_start + kh;
                        int in_w = in_w_start + kw;

                        // Input index (padded)
                        int in_idx = (in_h * (W_in + 2) + in_w) * Cin + cin;
                        float in_val = x_in[in_idx];

                        // Weight gradient index [Cout, 3, 3, Cin]
                        int w_idx = ((cout * 3 + kh) * 3 + kw) * Cin + cin;

                        // Accumulate: d_weight = grad * input
                        float weight_grad = grad * in_val;
                        // RPN-style NaN guard: only accumulate valid gradients
                        if (!isnan(weight_grad) && !isinf(weight_grad)) {
                            // No clipping - rely on small learning rate for stability
                            atomicAdd(&d_weight[w_idx], weight_grad);
                        }
                    }
                }
            }
        }
    }

    // Reduce bias gradient across threads
    s_bias[tid] = local_bias;
    __syncthreads();

    for (int stride = threads_per_block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_bias[tid] += s_bias[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&d_bias[cout], s_bias[0]);
    }
}


// ============================================================================
// Kernel 2: Backward w.r.t. Input
// ============================================================================
extern "C" __global__ void conv2d_backward_input(
    const float* __restrict__ d_out,      // Gradient from next layer [H_out, W_out, Cout]
    const float* __restrict__ weight,     // Filter weights [Cout, 3, 3, Cin]
    float* __restrict__ d_input,          // Output: d_input [H_in+2, W_in+2, Cin]
    int H_out,
    int W_out,
    int H_in,
    int W_in,
    int Cin,
    int Cout
) {
    // Each thread handles one input position
    int in_h = blockIdx.y * blockDim.y + threadIdx.y;
    int in_w = blockIdx.x * blockDim.x + threadIdx.x;
    int cin = blockIdx.z;

    if (in_h >= H_in + 2 || in_w >= W_in + 2 || cin >= Cin) return;

    float grad_sum = 0.0f;

    // For each output position that used this input
    // Output position (oh, ow) uses input window [oh:oh+3, ow:ow+3]
    // So input (in_h, in_w) contributes to outputs where:
    //   oh <= in_h < oh+3  =>  in_h-2 <= oh <= in_h
    //   ow <= in_w < ow+3  =>  in_w-2 <= ow <= in_w

    int oh_start = max(0, in_h - 2);
    int oh_end = min(H_out, in_h + 1);
    int ow_start = max(0, in_w - 2);
    int ow_end = min(W_out, in_w + 1);

    for (int oh = oh_start; oh < oh_end; oh++) {
        for (int ow = ow_start; ow < ow_end; ow++) {
            // Kernel position that used this input
            int kh = in_h - oh;
            int kw = in_w - ow;

            // Accumulate from all output channels
            for (int cout = 0; cout < Cout; cout++) {
                int out_idx = (oh * W_out + ow) * Cout + cout;
                float d_out_val = d_out[out_idx];

                // Weight index [Cout, 3, 3, Cin]
                int w_idx = ((cout * 3 + kh) * 3 + kw) * Cin + cin;
                float w_val = weight[w_idx];

                grad_sum += d_out_val * w_val;
            }
        }
    }

    // Write gradient
    int in_idx = (in_h * (W_in + 2) + in_w) * Cin + cin;
    d_input[in_idx] = grad_sum;
}


// ============================================================================
// Kernel 3: ReLU Backward (simple element-wise)
// ============================================================================
extern "C" __global__ void relu_backward(
    const float* __restrict__ d_out,      // Gradient from next layer
    const float* __restrict__ x_in,       // Input to ReLU
    float* __restrict__ d_input,          // Output: gradient w.r.t. input
    int N                                  // Total elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // d_input = d_out * (x_in > 0)
    float grad = d_out[idx];
    // RPN-style NaN guard: zero out invalid gradients before ReLU masking
    if (isnan(grad) || isinf(grad)) {
        grad = 0.0f;
    }
    d_input[idx] = grad * (x_in[idx] > 0.0f ? 1.0f : 0.0f);
}
