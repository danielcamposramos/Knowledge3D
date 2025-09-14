#include <cuda_runtime.h>

extern "C" __global__ void morph_embedding(float* embedding_ptr, unsigned int dim, float value, unsigned int embedding_size) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (dim >= embedding_size) return;
    embedding_ptr[dim] = value;
}

