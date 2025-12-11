/**
 * Drawing Galaxy transformation kernels - GPU-native visual transforms.
 * Used by RPN executor for ROT90, FLIP, SCALE, TILE operations.
 */

extern "C" {

// Rotate grid 90 degrees clockwise
__global__ void rot90_cw_kernel(
    const int* input,    // (H, W)
    int* output,         // (W, H) - dimensions swap
    int in_height, int in_width
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= in_height || out_y >= in_width) return;

    // rot90_cw: out[x][y] = in[H-1-y][x]
    int in_x = out_y;
    int in_y = in_height - 1 - out_x;

    output[out_y * in_height + out_x] = input[in_y * in_width + in_x];
}

// Rotate grid 90 degrees counter-clockwise
__global__ void rot90_ccw_kernel(
    const int* input,
    int* output,
    int in_height, int in_width
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= in_height || out_y >= in_width) return;

    // rot90_ccw: out[x][y] = in[y][W-1-x]
    int in_x = in_width - 1 - out_y;
    int in_y = out_x;

    output[out_y * in_height + out_x] = input[in_y * in_width + in_x];
}

// Flip horizontally
__global__ void flip_h_kernel(
    const int* input,
    int* output,
    int height, int width
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    output[y * width + x] = input[y * width + (width - 1 - x)];
}

// Flip vertically
__global__ void flip_v_kernel(
    const int* input,
    int* output,
    int height, int width
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    output[y * width + x] = input[(height - 1 - y) * width + x];
}

// Transpose (flip diagonal)
__global__ void transpose_kernel(
    const int* input,
    int* output,
    int in_height, int in_width
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= in_height || out_y >= in_width) return;

    output[out_y * in_height + out_x] = input[out_x * in_width + out_y];
}

// Scale 2x (nearest neighbor upsampling)
__global__ void scale_2x_kernel(
    const int* input,
    int* output,
    int in_height, int in_width
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    int out_width = in_width * 2;
    int out_height = in_height * 2;

    if (out_x >= out_width || out_y >= out_height) return;

    int in_x = out_x / 2;
    int in_y = out_y / 2;

    output[out_y * out_width + out_x] = input[in_y * in_width + in_x];
}

// Recolor: map old_color to new_color
__global__ void recolor_kernel(
    int* grid,
    int old_color, int new_color,
    int height, int width
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    if (grid[idx] == old_color) {
        grid[idx] = new_color;
    }
}

// Tile 2x2: replicate grid into 2x2 pattern
__global__ void tile_2x2_kernel(
    const int* input,
    int* output,
    int in_height, int in_width
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    int out_width = in_width * 2;
    int out_height = in_height * 2;

    if (out_x >= out_width || out_y >= out_height) return;

    int in_x = out_x % in_width;
    int in_y = out_y % in_height;

    output[out_y * out_width + out_x] = input[in_y * in_width + in_x];
}

// Overlay: grid_a over grid_b (non-zero from a wins)
__global__ void overlay_kernel(
    const int* grid_a,
    const int* grid_b,
    int* output,
    int height, int width
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int val_a = grid_a[idx];
    output[idx] = (val_a != 0) ? val_a : grid_b[idx];
}

}  // extern "C"
