/*
 * maxpool_2x2_backward.cu - MaxPool 2x2 Backward Pass
 *
 * Routes gradients from output back to input through max positions.
 * Each thread handles one output position and finds the max location
 * in the corresponding 2x2 input window.
 *
 * Architecture:
 *   - Block: 16×16 threads (256 threads/block)
 *   - Each thread processes one (out_h, out_w, c) position
 *   - Atomic adds for gradient accumulation
 */

extern "C" __global__ void maxpool_2x2_backward(
    const float* __restrict__ d_out,      // Gradient from next layer [H_out, W_out, C]
    const float* __restrict__ x_in,       // Input that was max-pooled [H_in, W_in, C]
    float* __restrict__ d_in,             // Output: gradient w.r.t. input [H_in, W_in, C]
    int H_in,
    int W_in,
    int H_out,
    int W_out,
    int C
) {
    // Thread coordinates in output space
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int out_w = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.z;

    if (out_h >= H_out || out_w >= W_out || c >= C) return;

    // Corresponding input window (stride=2, pool_size=2)
    int in_h_start = out_h * 2;
    int in_w_start = out_w * 2;
    int in_h_end = min(in_h_start + 2, H_in);
    int in_w_end = min(in_w_start + 2, W_in);

    // Find max value and its location in the window
    float max_val = -1e38f;
    int max_h = in_h_start;
    int max_w = in_w_start;

    for (int h = in_h_start; h < in_h_end; h++) {
        for (int w = in_w_start; w < in_w_end; w++) {
            int in_idx = (h * W_in + w) * C + c;
            float val = x_in[in_idx];
            if (val > max_val) {
                max_val = val;
                max_h = h;
                max_w = w;
            }
        }
    }

    // Route gradient to max location
    int out_idx = (out_h * W_out + out_w) * C + c;
    int max_in_idx = (max_h * W_in + max_w) * C + c;

    float grad = d_out[out_idx];
    atomicAdd(&d_in[max_in_idx], grad);
}
