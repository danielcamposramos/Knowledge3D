/**
 * Narrow GPU-backed CAS matrix helpers.
 *
 * These kernels support the live matrix surface used by the sovereign CAS
 * bridge for literal and scaled matrices without pretending full symbolic
 * matrix algebra is already available in PTX.
 */

#include <cuda_runtime.h>

extern "C" __global__ void matrix_literal_copy_kernel(
    const float* input_matrix,
    float* output_matrix,
    int element_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= element_count) return;
    output_matrix[idx] = input_matrix[idx];
}

extern "C" __global__ void matrix_scale_kernel(
    const float* input_matrix,
    float* output_matrix,
    float scalar,
    int element_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= element_count) return;
    output_matrix[idx] = input_matrix[idx] * scalar;
}
