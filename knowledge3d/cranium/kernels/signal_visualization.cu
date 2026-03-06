/**
 * GPU signal visualization kernels.
 */

extern "C" {

// Convert ternary/int-valued spectrogram cells into float32 RGBA preview colors.
__global__ void spectrogram_to_rgba_kernel(
    const int* spectrogram,
    float* rgba,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int out = idx * 4;
    int value = spectrogram[idx];

    if (value > 0) {
        rgba[out + 0] = 1.0f;
        rgba[out + 1] = 0.72f;
        rgba[out + 2] = 0.18f;
        rgba[out + 3] = 1.0f;
    } else if (value < 0) {
        rgba[out + 0] = 0.14f;
        rgba[out + 1] = 0.46f;
        rgba[out + 2] = 0.88f;
        rgba[out + 3] = 1.0f;
    } else {
        rgba[out + 0] = 0.08f;
        rgba[out + 1] = 0.08f;
        rgba[out + 2] = 0.10f;
        rgba[out + 3] = 1.0f;
    }
}

}  // extern "C"
