// Sovereign LoRA GPU Kernels
// Implements basic linear algebra primitives for low-rank specialist training.
//
// Kernels:
//  - matvec_general:        y = W @ x
//  - matvec_transpose:      y = W^T @ x
//  - vec_add_scaled:        out = a + scale * b
//  - vec_sub_square:        error = pred - target, accumulate squared error
//  - outer_product_scale:   out = scale * (a ⊗ b)
//  - matrix_axpy:           dest += scale * src (flattened matrices)
//
// Compile:
//   nvcc -ptx -arch=sm_86 knowledge3d/cranium/kernels/lora_gpu.cu \
//        -o knowledge3d/cranium/ptx/lora_gpu.ptx

#include <cuda_runtime.h>

extern "C" __global__ void matvec_general(
    const float* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const float* row_ptr = W + row * cols;
    float acc = 0.0f;
    #pragma unroll 8
    for (int c = 0; c < cols; ++c) {
        acc += row_ptr[c] * x[c];
    }
    y[row] = acc;
}

extern "C" __global__ void matvec_transpose(
    const float* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols
) {
    // Computes y = W^T @ x
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) {
        return;
    }
    float acc = 0.0f;
    #pragma unroll 8
    for (int r = 0; r < rows; ++r) {
        acc += W[r * cols + col] * x[r];
    }
    y[col] = acc;
}

extern "C" __global__ void vec_add_scaled(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float scale,
    float* __restrict__ out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    out[idx] = a[idx] + scale * b[idx];
}

extern "C" __global__ void vec_sub_square(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__ error,
    float* __restrict__ loss,
    int n
) {
    extern __shared__ float smem[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float diff = 0.0f;
    if (idx < n) {
        diff = pred[idx] - target[idx];
        error[idx] = diff;
    }
    smem[tid] = diff * diff;
    __syncthreads();

    // Reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(loss, smem[0]);
    }
}

extern "C" __global__ void outer_product_scale(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int rows,
    int cols,
    float scale
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || col >= cols) {
        return;
    }
    out[row * cols + col] = scale * a[row] * b[col];
}

extern "C" __global__ void matrix_axpy(
    float* __restrict__ dest,
    const float* __restrict__ src,
    float scale,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) {
        return;
    }
    dest[idx] += scale * src[idx];
}

extern "C" __global__ void matvec_batch(
    const float* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int rows,
    int cols,
    int batch
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * batch;
    if (idx >= total) {
        return;
    }
    int row = idx % rows;
    int b = idx / rows;
    const float* x = X + b * cols;
    float* y = Y + b * rows;
    const float* row_ptr = W + row * cols;
    float acc = 0.0f;
    #pragma unroll 8
    for (int c = 0; c < cols; ++c) {
        acc += row_ptr[c] * x[c];
    }
    y[row] = acc;
}

extern "C" __global__ void matvec_transpose_batch(
    const float* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int rows,
    int cols,
    int batch
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = cols * batch;
    if (idx >= total) {
        return;
    }
    int col = idx % cols;
    int b = idx / cols;
    const float* x = X + b * rows;
    float* y = Y + b * cols;
    float acc = 0.0f;
    #pragma unroll 8
    for (int r = 0; r < rows; ++r) {
        acc += W[r * cols + col] * x[r];
    }
    y[col] = acc;
}

extern "C" __global__ void vec_add_scaled_batch(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float scale,
    float* __restrict__ out,
    int n,
    int batch
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * batch;
    if (idx >= total) {
        return;
    }
    out[idx] = a[idx] + scale * b[idx];
}

extern "C" __global__ void vec_sub_square_batch(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__ error,
    float* __restrict__ losses,
    int dims,
    int batch
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = dims * batch;
    if (idx >= total) {
        return;
    }
    int sample = idx / dims;
    float diff = pred[idx] - target[idx];
    error[idx] = diff;
    atomicAdd(&losses[sample], diff * diff);
}

extern "C" __global__ void outer_product_accumulate_batch(
    const float* __restrict__ A_batch,
    const float* __restrict__ B_batch,
    float* __restrict__ out,
    int rows,
    int cols,
    int batch,
    float scale
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || col >= cols) {
        return;
    }
    float acc = 0.0f;
    for (int b = 0; b < batch; ++b) {
        float a_val = A_batch[b * rows + row];
        float b_val = B_batch[b * cols + col];
        acc += a_val * b_val;
    }
    out[row * cols + col] += scale * acc;
}
