/**
 * Nine-Chain Swarm Prototype Kernel
 *
 * Bio-inspired collective intelligence proof-of-concept.
 * Each CUDA block represents a reasoning chain. Chains adapt
 * based on resonance with the swarm and chain 9 synthesises
 * the final output.
 *
 * Architecture
 *  - Chain 0: Ingest
 *  - Chain 1-2: Fuse variants
 *  - Chain 3-5: Spatial reasoning
 *  - Chain 6: Reductionist reasoning
 *  - Chain 7: Creative reasoning
 *  - Chain 8: Synthesis
 *
 * This prototype focuses on demonstrating the swarm pattern;
 * chain-specialised logic and richer messaging are deferred to
 * the full Step 14 implementation.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdint.h>

#define NUM_CHAINS 9
#define CHAIN_STATE_DIM 64
#define SWARM_BLOCK_SIZE 256

#define CHAIN_INGEST 0
#define CHAIN_FUSE_A 1
#define CHAIN_FUSE_B 2
#define CHAIN_SPATIAL_A 3
#define CHAIN_SPATIAL_B 4
#define CHAIN_SPATIAL_C 5
#define CHAIN_REASON_REDUCTIONIST 6
#define CHAIN_REASON_CREATIVE 7
#define CHAIN_SYNTHESIS 8

__device__ inline float pseudo_random(int chain_id, int dim) {
    // Hash-based deterministic pseudo random in [-0.5, 0.5]
    unsigned int seed = static_cast<unsigned int>(chain_id * 73856093 ^ dim * 19349663);
    seed ^= seed >> 13;
    seed *= 1274126177u;
    seed ^= seed >> 16;
    return (static_cast<float>(seed & 0xFFFF) / 65535.0f) - 0.5f;
}

__device__ void compute_resonance(
    const float* __restrict__ chain_states,
    float* __restrict__ resonance_scores,
    int my_chain_id
) {
    __shared__ float partial_sums[NUM_CHAINS];

    for (int other_chain = threadIdx.x; other_chain < NUM_CHAINS; other_chain += blockDim.x) {
        if (other_chain == my_chain_id) {
            partial_sums[other_chain] = 0.0f;
            continue;
        }

        const float* my_state = &chain_states[my_chain_id * CHAIN_STATE_DIM];
        const float* other_state = &chain_states[other_chain * CHAIN_STATE_DIM];

        float dot = 0.0f;
        for (int d = 0; d < CHAIN_STATE_DIM; ++d) {
            dot += my_state[d] * other_state[d];
        }
        partial_sums[other_chain] = dot;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < NUM_CHAINS; ++i) {
            total += partial_sums[i];
        }
        resonance_scores[my_chain_id] = total / (NUM_CHAINS - 1);
    }
    __syncthreads();
}

__device__ void adapt_chain_state(
    float* __restrict__ my_state,
    const float* __restrict__ all_states,
    float my_resonance
) {
    __shared__ float consensus[CHAIN_STATE_DIM];

    for (int d = threadIdx.x; d < CHAIN_STATE_DIM; d += blockDim.x) {
        float sum = 0.0f;
        for (int c = 0; c < NUM_CHAINS; ++c) {
            sum += all_states[c * CHAIN_STATE_DIM + d];
        }
        consensus[d] = sum / static_cast<float>(NUM_CHAINS);
    }
    __syncthreads();

    const float ADAPTATION_RATE = 0.1f;
    float blend = (my_resonance > 0.8f) ? 0.0f : ADAPTATION_RATE;

    for (int d = threadIdx.x; d < CHAIN_STATE_DIM; d += blockDim.x) {
        float current = my_state[d];
        float target = consensus[d];
        my_state[d] = (1.0f - blend) * current + blend * target;
    }
    __syncthreads();
}

extern "C" __global__ void nine_chain_swarm_kernel(
    const float* __restrict__ input_embedding,
    float* __restrict__ chain_states,
    float* __restrict__ output_embedding,
    float* __restrict__ resonance_scores,
    int num_iterations
) {
    const int chain_id = blockIdx.x;
    if (chain_id >= NUM_CHAINS) {
        return;
    }

    float* my_state = &chain_states[chain_id * CHAIN_STATE_DIM];

    if (chain_id == CHAIN_INGEST) {
        for (int d = threadIdx.x; d < CHAIN_STATE_DIM; d += blockDim.x) {
            my_state[d] = input_embedding[d];
        }
    } else {
        for (int d = threadIdx.x; d < CHAIN_STATE_DIM; d += blockDim.x) {
            my_state[d] = pseudo_random(chain_id, d) * 0.1f;
        }
    }
    __syncthreads();

    for (int iter = 0; iter < num_iterations; ++iter) {
        for (int d = threadIdx.x; d < CHAIN_STATE_DIM; d += blockDim.x) {
            my_state[d] = tanhf(my_state[d]);
        }
        __syncthreads();

        compute_resonance(chain_states, resonance_scores, chain_id);

        float my_resonance = resonance_scores[chain_id];
        adapt_chain_state(my_state, chain_states, my_resonance);
        __syncthreads();
    }

    if (chain_id == CHAIN_SYNTHESIS) {
        __shared__ float weights[NUM_CHAINS];

        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (int c = 0; c < NUM_CHAINS; ++c) {
                float score = resonance_scores[c];
                weights[c] = score;
                sum += fabsf(score);
            }
            if (sum < 1e-6f) {
                sum = 1e-6f;
            }
            for (int c = 0; c < NUM_CHAINS; ++c) {
                weights[c] /= sum;
            }
        }
        __syncthreads();

        for (int d = threadIdx.x; d < CHAIN_STATE_DIM; d += blockDim.x) {
            float accumulator = 0.0f;
            for (int c = 0; c < NUM_CHAINS; ++c) {
                accumulator += weights[c] * chain_states[c * CHAIN_STATE_DIM + d];
            }
            output_embedding[d] = accumulator;
        }
    }
}
