/*
 * sgd_optimizer.cu - SGD with Momentum Weight Updates
 *
 * Updates model parameters using stochastic gradient descent
 * with momentum on GPU.
 */

// ============================================================================
// SGD Update (No Momentum)
// ============================================================================
extern "C" __global__ void sgd_update(
    float* __restrict__ param,            // Parameters to update
    const float* __restrict__ grad,       // Gradients
    float learning_rate,
    int N                                  // Number of parameters
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    param[idx] -= learning_rate * grad[idx];
}


// ============================================================================
// SGD Update with Momentum (CRITICAL FIX: Added gradient/velocity clipping)
// ============================================================================
extern "C" __global__ void sgd_momentum_update(
    float* __restrict__ param,            // Parameters to update
    const float* __restrict__ grad,       // Gradients
    float* __restrict__ velocity,         // Momentum velocity
    float learning_rate,
    float momentum,
    int N                                  // Number of parameters
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float g = grad[idx];
    float v_old = velocity[idx];

    // CRITICAL FIX 1: Clip gradients to [-10, 10] before velocity update
    if (isnan(g) || isinf(g)) {
        g = 0.0f;
    } else {
        g = fmaxf(fminf(g, 10.0f), -10.0f);
    }

    // Update velocity: v = momentum * v - lr * grad
    float v = momentum * v_old - learning_rate * g;

    // CRITICAL FIX 2: Clip velocity to [-1, 1]
    if (isnan(v) || isinf(v)) {
        v = 0.0f;
    } else {
        v = fmaxf(fminf(v, 1.0f), -1.0f);
    }

    velocity[idx] = v;

    // CRITICAL FIX 3: NaN/inf check before parameter update
    float new_param = param[idx] + v;
    if (!isnan(new_param) && !isinf(new_param)) {
        param[idx] = new_param;
    }
    // If new_param is NaN/inf, keep old parameter value
}


// ============================================================================
// Adam Optimizer Update
// ============================================================================
extern "C" __global__ void adam_update(
    float* __restrict__ param,            // Parameters to update
    const float* __restrict__ grad,       // Gradients
    float* __restrict__ m,                // First moment estimate
    float* __restrict__ v,                // Second moment estimate
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    int t,                                 // Time step
    int N                                  // Number of parameters
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float g = grad[idx];

    // Update biased first moment estimate
    float m_new = beta1 * m[idx] + (1.0f - beta1) * g;
    m[idx] = m_new;

    // Update biased second moment estimate
    float v_new = beta2 * v[idx] + (1.0f - beta2) * g * g;
    v[idx] = v_new;

    // Bias correction
    float m_hat = m_new / (1.0f - powf(beta1, t));
    float v_hat = v_new / (1.0f - powf(beta2, t));

    // Update parameter
    param[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
}


// ============================================================================
// Gradient Clipping (by norm)
// ============================================================================
extern "C" __global__ void clip_grad_norm(
    float* __restrict__ grad,             // Gradients to clip
    float max_norm,
    int N                                  // Number of parameters
) {
    // First pass: compute global norm (requires reduction)
    __shared__ float s_norm_sq[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float local_sum = 0.0f;
    if (idx < N) {
        float g = grad[idx];
        local_sum = g * g;
    }

    s_norm_sq[tid] = local_sum;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_norm_sq[tid] += s_norm_sq[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float global_norm;
    if (tid == 0) {
        global_norm = sqrtf(s_norm_sq[0]);
    }
    __syncthreads();

    // Second pass: clip if needed
    if (idx < N && global_norm > max_norm) {
        float scale = max_norm / (global_norm + 1e-6f);
        grad[idx] *= scale;
    }
}


// ============================================================================
// Zero Gradients
// ============================================================================
extern "C" __global__ void zero_grad(
    float* __restrict__ grad,             // Gradients to zero
    int N                                  // Number of parameters
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    grad[idx] = 0.0f;
}
