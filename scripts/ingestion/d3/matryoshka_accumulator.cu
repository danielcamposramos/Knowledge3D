/*
 * D3 standalone Matryoshka accumulator.
 *
 * Input basis rows are packed as 5 ternary digits per byte in base-3 form,
 * with trit values mapped as 0 -> -1, 1 -> 0, 2 -> +1.
 */

#include <stdint.h>
#include <math.h>

static __device__ __forceinline__ int unpack_trit(const uint8_t* packed, int dim) {
    int byte_index = dim / 5;
    int offset = dim % 5;
    uint8_t value = packed[byte_index];
    #pragma unroll
    for (int i = 0; i < offset; ++i) {
        value /= 3;
    }
    return int(value % 3) - 1;
}

static __device__ __forceinline__ int ternary_affinity(uint32_t lhs, uint32_t rhs) {
    uint32_t lhs_block = (lhs >> 8) & 0xFFu;
    uint32_t rhs_block = (rhs >> 8) & 0xFFu;
    if (lhs_block == rhs_block) {
        return 1;
    }
    if (((lhs_block ^ rhs_block) & 0x1u) != 0u) {
        return 0;
    }
    return -1;
}

extern "C" __global__ void matryoshka_accumulator(
    const uint8_t* __restrict__ basis,
    int basis_stride,
    const uint32_t* __restrict__ token_basis_indices,
    const uint32_t* __restrict__ token_ids,
    const uint8_t* __restrict__ masses,
    int token_count,
    int8_t* __restrict__ output
) {
    __shared__ int accumulators[2048];
    __shared__ float partial[256];

    const int tid = int(threadIdx.x);

    for (int dim = tid; dim < 2048; dim += int(blockDim.x)) {
        int acc = 0;
        for (int i = 0; i < token_count; ++i) {
            const uint8_t* token_basis = basis + (size_t(token_basis_indices[i]) * size_t(basis_stride));
            int contribution = unpack_trit(token_basis, dim);
            int mass_t = int(masses[i]);
            int weight = mass_t * 64;
            for (int p = 0; p < i; ++p) {
                int distance = i - p;
                if (distance < 1) {
                    distance = 1;
                }
                int affinity = ternary_affinity(token_ids[p], token_ids[i]);
                int mass_p = int(masses[p]);
                weight += (affinity * mass_p * mass_t) / (distance * distance);
            }
            acc += contribution * weight;
        }
        accumulators[dim] = acc;
    }

    __syncthreads();

    float sumsq = 0.0f;
    for (int dim = tid; dim < 2048; dim += int(blockDim.x)) {
        float value = float(accumulators[dim]);
        sumsq += value * value;
    }
    partial[tid] = sumsq;
    __syncthreads();

    for (int stride = int(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    float scale = 0.0f;
    if (partial[0] > 0.0f) {
        scale = 64.0f / sqrtf(partial[0]);
    }
    __syncthreads();

    for (int dim = tid; dim < 2048; dim += int(blockDim.x)) {
        float value = float(accumulators[dim]) * scale;
        int rounded = int(roundf(value));
        if (rounded > 127) {
            rounded = 127;
        }
        if (rounded < -128) {
            rounded = -128;
        }
        output[dim] = int8_t(rounded);
    }
}
