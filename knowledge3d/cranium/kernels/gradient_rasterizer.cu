/**
 * Procedural gradient rasterization — GPU-native linear/radial/conic gradients.
 */

extern "C" {

__global__ void gradient_linear_kernel(
    float* output,           // (H, W, 4) RGBA
    float x1, float y1,      // Start point (0..1 normalized)
    float x2, float y2,      // End point (0..1 normalized)
    const float* stops,      // [pos, r, g, b, a] × n_stops
    int n_stops,
    int width, int height
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    float dx = x2 - x1;
    float dy = y2 - y1;
    float len_sq = dx * dx + dy * dy;
    if (len_sq < 1e-6f) len_sq = 1e-6f;

    float fx = (float)px / (float)(width - 1);
    float fy = (float)py / (float)(height - 1);
    float t = ((fx - x1) * dx + (fy - y1) * dy) / len_sq;
    t = fmaxf(0.0f, fminf(1.0f, t));

    int idx0 = 0, idx1 = n_stops - 1;
    for (int i = 0; i < n_stops - 1; i++) {
        float pos0 = stops[i * 5];
        float pos1 = stops[(i + 1) * 5];
        if (t >= pos0 && t <= pos1) {
            idx0 = i;
            idx1 = i + 1;
            break;
        }
    }

    float pos0 = stops[idx0 * 5];
    float pos1 = stops[idx1 * 5];
    float local_t = (pos1 > pos0) ? (t - pos0) / (pos1 - pos0) : 0.0f;

    float r = stops[idx0 * 5 + 1] + local_t * (stops[idx1 * 5 + 1] - stops[idx0 * 5 + 1]);
    float g = stops[idx0 * 5 + 2] + local_t * (stops[idx1 * 5 + 2] - stops[idx0 * 5 + 2]);
    float b = stops[idx0 * 5 + 3] + local_t * (stops[idx1 * 5 + 3] - stops[idx0 * 5 + 3]);
    float a = stops[idx0 * 5 + 4] + local_t * (stops[idx1 * 5 + 4] - stops[idx0 * 5 + 4]);

    int idx = (py * width + px) * 4;
    output[idx + 0] = r;
    output[idx + 1] = g;
    output[idx + 2] = b;
    output[idx + 3] = a;
}

__global__ void gradient_radial_kernel(
    float* output,
    float cx, float cy, float radius,
    const float* stops, int n_stops,
    int width, int height
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    float fx = (float)px / (float)(width - 1);
    float fy = (float)py / (float)(height - 1);
    float dist = sqrtf((fx - cx) * (fx - cx) + (fy - cy) * (fy - cy));
    float t = fminf(dist / fmaxf(radius, 1e-6f), 1.0f);

    int idx0 = 0, idx1 = n_stops - 1;
    for (int i = 0; i < n_stops - 1; i++) {
        if (t >= stops[i * 5] && t <= stops[(i + 1) * 5]) {
            idx0 = i;
            idx1 = i + 1;
            break;
        }
    }

    float pos0 = stops[idx0 * 5];
    float pos1 = stops[idx1 * 5];
    float local_t = (pos1 > pos0) ? (t - pos0) / (pos1 - pos0) : 0.0f;

    float r = stops[idx0 * 5 + 1] + local_t * (stops[idx1 * 5 + 1] - stops[idx0 * 5 + 1]);
    float g = stops[idx0 * 5 + 2] + local_t * (stops[idx1 * 5 + 2] - stops[idx0 * 5 + 2]);
    float b = stops[idx0 * 5 + 3] + local_t * (stops[idx1 * 5 + 3] - stops[idx0 * 5 + 3]);
    float a = stops[idx0 * 5 + 4] + local_t * (stops[idx1 * 5 + 4] - stops[idx0 * 5 + 4]);

    int idx = (py * width + px) * 4;
    output[idx + 0] = r;
    output[idx + 1] = g;
    output[idx + 2] = b;
    output[idx + 3] = a;
}

__global__ void gradient_conic_kernel(
    float* output,
    float cx, float cy, float start_angle,
    const float* stops, int n_stops,
    int width, int height
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    float fx = (float)px / (float)(width - 1);
    float fy = (float)py / (float)(height - 1);

    float angle = atan2f(fy - cy, fx - cx) - start_angle;
    if (angle < 0) angle += 2.0f * 3.14159265358979f;
    float t = angle / (2.0f * 3.14159265358979f);

    int idx0 = 0, idx1 = n_stops - 1;
    for (int i = 0; i < n_stops - 1; i++) {
        if (t >= stops[i * 5] && t <= stops[(i + 1) * 5]) {
            idx0 = i;
            idx1 = i + 1;
            break;
        }
    }

    float pos0 = stops[idx0 * 5];
    float pos1 = stops[idx1 * 5];
    float local_t = (pos1 > pos0) ? (t - pos0) / (pos1 - pos0) : 0.0f;

    float r = stops[idx0 * 5 + 1] + local_t * (stops[idx1 * 5 + 1] - stops[idx0 * 5 + 1]);
    float g = stops[idx0 * 5 + 2] + local_t * (stops[idx1 * 5 + 2] - stops[idx0 * 5 + 2]);
    float b = stops[idx0 * 5 + 3] + local_t * (stops[idx1 * 5 + 3] - stops[idx0 * 5 + 3]);
    float a = stops[idx0 * 5 + 4] + local_t * (stops[idx1 * 5 + 4] - stops[idx0 * 5 + 4]);

    int idx = (py * width + px) * 4;
    output[idx + 0] = r;
    output[idx + 1] = g;
    output[idx + 2] = b;
    output[idx + 3] = a;
}

}  // extern "C"
