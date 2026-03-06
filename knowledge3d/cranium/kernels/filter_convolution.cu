/**
 * GPU convolution filters — blur, sharpen, edge detection.
 */

extern "C" {

// Separable Gaussian blur (horizontal pass)
__global__ void blur_horizontal_kernel(
    const float* input,
    float* output,
    const float* kernel,
    int kernel_radius,
    int width, int height, int channels
) {
    extern __shared__ float s_row[];
    int y = blockIdx.y;
    int c = blockIdx.z;
    if (y >= height || c >= channels) return;

    int tid = threadIdx.x;
    int kernel_size = 2 * kernel_radius + 1;
    for (int i = tid; i < width + 2 * kernel_radius; i += blockDim.x) {
        int x = i - kernel_radius;
        x = max(0, min(width - 1, x));
        s_row[i] = input[(y * width + x) * channels + c];
    }
    __syncthreads();

    for (int px = tid; px < width; px += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            sum += s_row[px + k] * kernel[k];
        }
        output[(y * width + px) * channels + c] = sum;
    }
}

// Separable Gaussian blur (vertical pass)
__global__ void blur_vertical_kernel(
    const float* input,
    float* output,
    const float* kernel,
    int kernel_radius,
    int width, int height, int channels
) {
    extern __shared__ float s_col[];
    int x = blockIdx.x;
    int c = blockIdx.z;
    if (x >= width || c >= channels) return;

    int tid = threadIdx.y;
    int kernel_size = 2 * kernel_radius + 1;
    for (int i = tid; i < height + 2 * kernel_radius; i += blockDim.y) {
        int y = i - kernel_radius;
        y = max(0, min(height - 1, y));
        s_col[i] = input[(y * width + x) * channels + c];
    }
    __syncthreads();

    for (int py = tid; py < height; py += blockDim.y) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            sum += s_col[py + k] * kernel[k];
        }
        output[(py * width + x) * channels + c] = sum;
    }
}

// Sobel edge detection on grayscale input
__global__ void sobel_edge_kernel(const float* input, float* output, int width, int height) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px < 1 || px >= width - 1 || py < 1 || py >= height - 1) return;

    float gx = -1.0f * input[(py - 1) * width + (px - 1)] + 1.0f * input[(py - 1) * width + (px + 1)]
             + -2.0f * input[(py) * width + (px - 1)]     + 2.0f * input[(py) * width + (px + 1)]
             + -1.0f * input[(py + 1) * width + (px - 1)] + 1.0f * input[(py + 1) * width + (px + 1)];

    float gy = -1.0f * input[(py - 1) * width + (px - 1)] + -2.0f * input[(py - 1) * width + px] + -1.0f * input[(py - 1) * width + (px + 1)]
             +  1.0f * input[(py + 1) * width + (px - 1)] +  2.0f * input[(py + 1) * width + px] +  1.0f * input[(py + 1) * width + (px + 1)];

    output[py * width + px] = sqrtf(gx * gx + gy * gy);
}

// Unsharp mask sharpen
__global__ void sharpen_kernel(
    const float* input,
    const float* blurred,
    float* output,
    float amount,
    int width, int height, int channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height * channels;
    if (idx >= total) return;

    float orig = input[idx];
    float blur = blurred[idx];
    output[idx] = orig + amount * (orig - blur);
}

// Convert RGB/RGBA canvas to grayscale luminance field
__global__ void rgba_to_luma_kernel(
    const float* input,
    float* output,
    int width,
    int height,
    int channels
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    int base = (py * width + px) * channels;
    float r = input[base + 0];
    float g = channels > 1 ? input[base + 1] : r;
    float b = channels > 2 ? input[base + 2] : g;
    output[py * width + px] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// Straight-alpha compositing: foreground over background
__global__ void alpha_over_rgba_kernel(
    const float* background,
    const float* foreground,
    float* output,
    int width,
    int height
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    int idx = (py * width + px) * 4;

    float br = background[idx + 0];
    float bg = background[idx + 1];
    float bb = background[idx + 2];
    float ba = fminf(fmaxf(background[idx + 3], 0.0f), 1.0f);

    float fr = foreground[idx + 0];
    float fg = foreground[idx + 1];
    float fb = foreground[idx + 2];
    float fa = fminf(fmaxf(foreground[idx + 3], 0.0f), 1.0f);

    float out_a = fa + ba * (1.0f - fa);
    float premul_r = fr * fa + br * ba * (1.0f - fa);
    float premul_g = fg * fa + bg * ba * (1.0f - fa);
    float premul_b = fb * fa + bb * ba * (1.0f - fa);

    if (out_a > 1e-6f) {
        output[idx + 0] = premul_r / out_a;
        output[idx + 1] = premul_g / out_a;
        output[idx + 2] = premul_b / out_a;
    } else {
        output[idx + 0] = 0.0f;
        output[idx + 1] = 0.0f;
        output[idx + 2] = 0.0f;
    }
    output[idx + 3] = out_a;
}

// Simple color invert for RGB/RGBA canvases
__global__ void invert_rgba_kernel(
    const float* input,
    float* output,
    int total_values,
    int channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_values) return;

    int channel = idx % channels;
    float value = input[idx];
    output[idx] = channel == 3 ? value : 1.0f - value;
}

}  // extern "C"
