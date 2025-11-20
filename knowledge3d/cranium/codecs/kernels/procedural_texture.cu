// Simple procedural checkerboard texture generator.
#include <cuda_runtime.h>

extern "C" __global__ void checker_texture(int width, int height, int tile, unsigned char* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = (y * width + x) * 3;
    int checker = ((x / tile) + (y / tile)) & 1;
    unsigned char value = checker ? 255 : 0;
    output[idx + 0] = value;
    output[idx + 1] = value;
    output[idx + 2] = value;
}
