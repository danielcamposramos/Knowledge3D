/*
 * classification_loss.cu - Classification Loss and Softmax Backward
 *
 * Computes cross-entropy loss and gradients for CNN classification.
 * Includes softmax forward and backward passes.
 */

// ============================================================================
// Softmax Forward Pass
// ============================================================================
extern "C" __global__ void softmax_forward(
    const float* __restrict__ logits,     // Input logits [N, num_classes]
    float* __restrict__ probs,            // Output probabilities [N, num_classes]
    int N,
    int num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float* logit_row = logits + idx * num_classes;
    float* prob_row = probs + idx * num_classes;

    // Find max for numerical stability
    float max_logit = logit_row[0];
    for (int c = 1; c < num_classes; c++) {
        if (logit_row[c] > max_logit) {
            max_logit = logit_row[c];
        }
    }

    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int c = 0; c < num_classes; c++) {
        float exp_val = expf(logit_row[c] - max_logit);
        prob_row[c] = exp_val;
        sum_exp += exp_val;
    }

    // Normalize
    for (int c = 0; c < num_classes; c++) {
        prob_row[c] /= sum_exp;
    }
}


// ============================================================================
// Cross-Entropy Loss Forward
// ============================================================================
extern "C" __global__ void cross_entropy_forward(
    const float* __restrict__ probs,      // Softmax probabilities [N, num_classes]
    const int* __restrict__ targets,      // True class labels [N]
    float* __restrict__ losses,           // Output losses [N]
    int N,
    int num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int target = targets[idx];
    float prob = probs[idx * num_classes + target];

    // Clip for numerical stability
    prob = fmaxf(prob, 1e-7f);

    // Loss = -log(prob)
    losses[idx] = -logf(prob);
}


// ============================================================================
// Cross-Entropy + Softmax Backward (Combined)
// ============================================================================
extern "C" __global__ void cross_entropy_softmax_backward(
    const float* __restrict__ probs,      // Softmax probabilities [N, num_classes]
    const int* __restrict__ targets,      // True class labels [N]
    float* __restrict__ d_logits,         // Output: gradient w.r.t. logits [N, num_classes]
    int N,
    int num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int target = targets[idx];
    const float* prob_row = probs + idx * num_classes;
    float* d_logit_row = d_logits + idx * num_classes;

    // Gradient of cross-entropy + softmax: probs - one_hot
    for (int c = 0; c < num_classes; c++) {
        if (c == target) {
            d_logit_row[c] = prob_row[c] - 1.0f;
        } else {
            d_logit_row[c] = prob_row[c];
        }
    }
}


// ============================================================================
// Global Average Pooling Forward
// ============================================================================
extern "C" __global__ void global_avgpool_forward(
    const float* __restrict__ feature_map,  // Input [H, W, C]
    float* __restrict__ output,             // Output [C]
    int H,
    int W,
    int C
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    float sum = 0.0f;
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            sum += feature_map[(h * W + w) * C + c];
        }
    }

    output[c] = sum / (H * W);
}


// ============================================================================
// Global Average Pooling Backward
// ============================================================================
extern "C" __global__ void global_avgpool_backward(
    const float* __restrict__ d_out,        // Gradient from next layer [C]
    float* __restrict__ d_feature_map,      // Output: gradient w.r.t. feature map [H, W, C]
    int H,
    int W,
    int C
) {
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.z;

    if (h >= H || w >= W || c >= C) return;

    // Gradient is distributed uniformly across spatial positions
    float grad = d_out[c] / (H * W);

    d_feature_map[(h * W + w) * C + c] = grad;
}


// ============================================================================
// Fully Connected Layer Forward
// ============================================================================
extern "C" __global__ void fc_forward(
    const float* __restrict__ input,        // Input [N, D_in]
    const float* __restrict__ weight,       // Weights [D_out, D_in]
    const float* __restrict__ bias,         // Bias [D_out]
    float* __restrict__ output,             // Output [N, D_out]
    int N,
    int D_in,
    int D_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Sample index
    int out_dim = blockIdx.y;  // Output dimension

    if (idx >= N || out_dim >= D_out) return;

    float sum = bias[out_dim];
    const float* w_row = weight + out_dim * D_in;
    const float* x_row = input + idx * D_in;

    for (int d = 0; d < D_in; d++) {
        sum += w_row[d] * x_row[d];
    }

    output[idx * D_out + out_dim] = sum;
}


// ============================================================================
// Fully Connected Layer Backward
// ============================================================================
extern "C" __global__ void fc_backward(
    const float* __restrict__ d_out,        // Gradient from next layer [N, D_out]
    const float* __restrict__ input,        // Input [N, D_in]
    const float* __restrict__ weight,       // Weights [D_out, D_in]
    float* __restrict__ d_input,            // Output: gradient w.r.t. input [N, D_in]
    float* __restrict__ d_weight,           // Output: gradient w.r.t. weight [D_out, D_in]
    float* __restrict__ d_bias,             // Output: gradient w.r.t. bias [D_out]
    int N,
    int D_in,
    int D_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Sample index
    int in_dim = blockIdx.y;  // Input dimension

    if (idx >= N || in_dim >= D_in) return;

    // Compute d_input
    float grad_sum = 0.0f;
    const float* d_out_row = d_out + idx * D_out;

    for (int out_dim = 0; out_dim < D_out; out_dim++) {
        const float* w_row = weight + out_dim * D_in;
        grad_sum += d_out_row[out_dim] * w_row[in_dim];

        // Compute d_weight (atomic since multiple samples contribute)
        float x_val = input[idx * D_in + in_dim];
        atomicAdd(&d_weight[out_dim * D_in + in_dim], d_out_row[out_dim] * x_val);

        // Compute d_bias (once per output dimension)
        if (in_dim == 0) {
            atomicAdd(&d_bias[out_dim], d_out_row[out_dim]);
        }
    }

    d_input[idx * D_in + in_dim] = grad_sum;
}
