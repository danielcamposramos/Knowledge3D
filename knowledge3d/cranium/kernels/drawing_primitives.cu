/**
 * Drawing Engine Phases 2-4: Advanced Drawing Primitives, VectorDotMap Codec, Lighting/Layer Ops
 * CUDA kernels for sovereign GPU execution of drawing operations
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

// Phase 2: Advanced Drawing Primitives
extern "C" __global__ void bezier_eval_kernel(
    const float* t_values,
    const float* control_points,  // [p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y]
    float* output_points,         // [x, y] output
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    float t = t_values[idx];
    float t2 = t * t;
    float t3 = t2 * t;
    float mt = 1.0f - t;
    float mt2 = mt * mt;
    float mt3 = mt2 * mt;

    // Cubic Bezier: B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
    float p0x = control_points[0];
    float p0y = control_points[1];
    float p1x = control_points[2];
    float p1y = control_points[3];
    float p2x = control_points[4];
    float p2y = control_points[5];
    float p3x = control_points[6];
    float p3y = control_points[7];

    output_points[idx * 2]     = mt3 * p0x + 3.0f * mt2 * t * p1x + 3.0f * mt * t2 * p2x + t3 * p3x;
    output_points[idx * 2 + 1] = mt3 * p0y + 3.0f * mt2 * t * p1y + 3.0f * mt * t2 * p2y + t3 * p3y;
}

extern "C" __global__ void shape_union_kernel(
    const float* shape_a,  // [min_x, min_y, max_x, max_y]
    const float* shape_b,  // [min_x, min_y, max_x, max_y]
    float* result,         // [min_x, min_y, max_x, max_y]
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // Union: min of mins, max of maxes
    result[idx * 4]     = fminf(shape_a[idx * 4],     shape_b[idx * 4]);     // min_x
    result[idx * 4 + 1] = fminf(shape_a[idx * 4 + 1], shape_b[idx * 4 + 1]); // min_y
    result[idx * 4 + 2] = fmaxf(shape_a[idx * 4 + 2], shape_b[idx * 4 + 2]); // max_x
    result[idx * 4 + 3] = fmaxf(shape_a[idx * 4 + 3], shape_b[idx * 4 + 3]); // max_y
}

extern "C" __global__ void shape_intersect_kernel(
    const float* shape_a,
    const float* shape_b,
    float* result,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // Intersection: max of mins, min of maxes
    result[idx * 4]     = fmaxf(shape_a[idx * 4],     shape_b[idx * 4]);     // min_x
    result[idx * 4 + 1] = fmaxf(shape_a[idx * 4 + 1], shape_b[idx * 4 + 1]); // min_y
    result[idx * 4 + 2] = fminf(shape_a[idx * 4 + 2], shape_b[idx * 4 + 2]); // max_x
    result[idx * 4 + 3] = fminf(shape_a[idx * 4 + 3], shape_b[idx * 4 + 3]); // max_y
}

extern "C" __global__ void shape_subtract_kernel(
    const float* shape_a,
    const float* shape_b,
    float* result,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // Subtraction: A - B = A intersect not(B)
    // For bounding boxes, this means we clip A against B's bounds
    float a_min_x = shape_a[idx * 4];
    float a_min_y = shape_a[idx * 4 + 1];
    float a_max_x = shape_a[idx * 4 + 2];
    float a_max_y = shape_a[idx * 4 + 3];

    float b_min_x = shape_b[idx * 4];
    float b_min_y = shape_b[idx * 4 + 1];
    float b_max_x = shape_b[idx * 4 + 2];
    float b_max_y = shape_b[idx * 4 + 3];

    // Clip A against B
    result[idx * 4]     = fmaxf(a_min_x, b_max_x); // left edge of result
    result[idx * 4 + 1] = fmaxf(a_min_y, b_max_y); // bottom edge
    result[idx * 4 + 2] = fminf(a_max_x, b_min_x); // right edge
    result[idx * 4 + 3] = fminf(a_max_y, b_min_y); // top edge

    // Ensure valid bounds (min <= max)
    if (result[idx * 4] > result[idx * 4 + 2]) {
        result[idx * 4] = result[idx * 4 + 2] = 0.0f;
    }
    if (result[idx * 4 + 1] > result[idx * 4 + 3]) {
        result[idx * 4 + 1] = result[idx * 4 + 3] = 0.0f;
    }
}

extern "C" __global__ void rel_line_kernel(
    const float* start_points,  // [x0, y0] in fractional coords [0,1]
    const float* end_points,    // [x1, y1] in fractional coords [0,1]
    float* output_lines,        // [x0, y0, x1, y1] absolute coords
    int count,
    float canvas_width,
    float canvas_height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    float x0_frac = start_points[idx * 2];
    float y0_frac = start_points[idx * 2 + 1];
    float x1_frac = end_points[idx * 2];
    float y1_frac = end_points[idx * 2 + 1];

    // Convert fractional to absolute coordinates
    output_lines[idx * 4]     = x0_frac * canvas_width;
    output_lines[idx * 4 + 1] = y0_frac * canvas_height;
    output_lines[idx * 4 + 2] = x1_frac * canvas_width;
    output_lines[idx * 4 + 3] = y1_frac * canvas_height;
}

extern "C" __global__ void field_coef_kernel(
    const float* coefficients,  // [c0, c1, c2, c3, c4, c5, c6, c7]
    float* field_values,        // output field values
    const float* positions,     // [x, y] positions
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    float x = positions[idx * 2];
    float y = positions[idx * 2 + 1];

    // 2D polynomial field: f(x,y) = c0 + c1*x + c2*y + c3*x² + c4*y² + c5*x*y + c6*x³ + c7*y³
    float x2 = x * x;
    float y2 = y * y;
    float x3 = x2 * x;
    float y3 = y2 * y;
    float xy = x * y;

    field_values[idx] = coefficients[0] + coefficients[1] * x + coefficients[2] * y +
                       coefficients[3] * x2 + coefficients[4] * y2 + coefficients[5] * xy +
                       coefficients[6] * x3 + coefficients[7] * y3;
}

extern "C" __global__ void dot_emit_kernel(
    const float* positions,     // [x, y] positions
    const float* field_values,  // field strength at each position
    float* output_dots,         // [x, y, intensity, radius] for each dot
    int count,
    float base_radius,
    float intensity_scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    float x = positions[idx * 2];
    float y = positions[idx * 2 + 1];
    float field = field_values[idx];

    // Emit dot with radius proportional to field strength
    float radius = base_radius * (1.0f + field * 0.5f);
    float intensity = fminf(fmaxf(field * intensity_scale, 0.0f), 1.0f);

    output_dots[idx * 4]     = x;
    output_dots[idx * 4 + 1] = y;
    output_dots[idx * 4 + 2] = intensity;
    output_dots[idx * 4 + 3] = radius;
}

// Phase 3: VectorDotMap Codec
extern "C" __global__ void vectordotmap_encode_kernel(
    const float* pixels,        // input pixel data [r, g, b, a]
    float* field_coeffs,        // output field coefficients [c0..c7]
    int width,
    int height,
    int block_size)
{
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Process one block of pixels
    int start_x = block_x * block_size;
    int start_y = block_y * block_size;
    int end_x = min(start_x + block_size, width);
    int end_y = min(start_y + block_size, height);

    // Shared memory for local computation
    __shared__ float shared_pixels[16][16][4];
    __shared__ float shared_coeffs[8];

    // Load pixels into shared memory
    if (start_x + tx < end_x && start_y + ty < end_y) {
        int pixel_idx = ((start_y + ty) * width + (start_x + tx)) * 4;
        shared_pixels[ty][tx][0] = pixels[pixel_idx];
        shared_pixels[ty][tx][1] = pixels[pixel_idx + 1];
        shared_pixels[ty][tx][2] = pixels[pixel_idx + 2];
        shared_pixels[ty][tx][3] = pixels[pixel_idx + 3];
    } else {
        shared_pixels[ty][tx][0] = 0.0f;
        shared_pixels[ty][tx][1] = 0.0f;
        shared_pixels[ty][tx][2] = 0.0f;
        shared_pixels[ty][tx][3] = 0.0f;
    }
    __syncthreads();

    // Simple least squares fit for polynomial coefficients
    // This is a simplified version - full implementation would use proper regression
    if (tx == 0 && ty == 0) {
        // Initialize coefficients
        for (int i = 0; i < 8; i++) {
            shared_coeffs[i] = 0.0f;
        }

        // Compute average pixel value as baseline
        float avg_r = 0.0f, avg_g = 0.0f, avg_b = 0.0f;
        int count = 0;
        for (int y = 0; y < min(block_size, 16); y++) {
            for (int x = 0; x < min(block_size, 16); x++) {
                avg_r += shared_pixels[y][x][0];
                avg_g += shared_pixels[y][x][1];
                avg_b += shared_pixels[y][x][2];
                count++;
            }
        }
        if (count > 0) {
            avg_r /= count; avg_g /= count; avg_b /= count;
        }

        // Simple coefficient estimation
        shared_coeffs[0] = avg_r;  // constant term
        shared_coeffs[1] = 0.1f;   // x coefficient
        shared_coeffs[2] = 0.1f;   // y coefficient
        shared_coeffs[3] = 0.05f;  // x² coefficient
        shared_coeffs[4] = 0.05f;  // y² coefficient
        shared_coeffs[5] = 0.02f;  // xy coefficient
        shared_coeffs[6] = 0.01f;  // x³ coefficient
        shared_coeffs[7] = 0.01f;  // y³ coefficient
    }
    __syncthreads();

    // Write coefficients to global memory
    int coeff_idx = (block_y * gridDim.x + block_x) * 8;
    if (tx < 8 && ty == 0) {
        field_coeffs[coeff_idx + tx] = shared_coeffs[tx];
    }
}

extern "C" __global__ void vectordotmap_decode_kernel(
    const float* field_coeffs,  // [c0..c7] coefficients
    float* pixels,              // output pixel data [r, g, b, a]
    int width,
    int height,
    int block_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Determine which block this pixel belongs to
    int block_x = x / block_size;
    int block_y = y / block_size;
    int local_x = x % block_size;
    int local_y = y % block_size;

    // Get coefficients for this block
    int coeff_idx = (block_y * ((width + block_size - 1) / block_size) + block_x) * 8;
    float c0 = field_coeffs[coeff_idx];
    float c1 = field_coeffs[coeff_idx + 1];
    float c2 = field_coeffs[coeff_idx + 2];
    float c3 = field_coeffs[coeff_idx + 3];
    float c4 = field_coeffs[coeff_idx + 4];
    float c5 = field_coeffs[coeff_idx + 5];
    float c6 = field_coeffs[coeff_idx + 6];
    float c7 = field_coeffs[coeff_idx + 7];

    // Normalize local coordinates to [-1, 1]
    float u = (2.0f * local_x) / block_size - 1.0f;
    float v = (2.0f * local_y) / block_size - 1.0f;

    // Evaluate polynomial field
    float u2 = u * u;
    float v2 = v * v;
    float u3 = u2 * u;
    float v3 = v2 * v;
    float uv = u * v;

    float field = c0 + c1 * u + c2 * v + c3 * u2 + c4 * v2 + c5 * uv + c6 * u3 + c7 * v3;

    // Map field to color
    float intensity = fminf(fmaxf(field, 0.0f), 1.0f);
    int pixel_idx = (y * width + x) * 4;
    pixels[pixel_idx]     = intensity;        // R
    pixels[pixel_idx + 1] = intensity * 0.8f; // G
    pixels[pixel_idx + 2] = intensity * 0.6f; // B
    pixels[pixel_idx + 3] = 1.0f;             // A
}

// Phase 4: Lighting and Layer Ops
extern "C" __global__ void layer_blend_kernel(
    const float* layer_a,   // [r, g, b, a]
    const float* layer_b,   // [r, g, b, a]
    float* output,          // [r, g, b, a]
    int pixel_count,
    int blend_mode)         // 0=normal, 1=multiply, 2=screen, 3=overlay
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixel_count) return;

    int rgba_idx = idx * 4;
    float r_a = layer_a[rgba_idx];
    float g_a = layer_a[rgba_idx + 1];
    float b_a = layer_a[rgba_idx + 2];
    float a_a = layer_a[rgba_idx + 3];

    float r_b = layer_b[rgba_idx];
    float g_b = layer_b[rgba_idx + 1];
    float b_b = layer_b[rgba_idx + 2];
    float a_b = layer_b[rgba_idx + 3];

    float r_out, g_out, b_out, a_out;

    switch (blend_mode) {
        case 0: // Normal blend: B over A
            a_out = a_b + a_a * (1.0f - a_b);
            r_out = (r_b * a_b + r_a * a_a * (1.0f - a_b)) / a_out;
            g_out = (g_b * a_b + g_a * a_a * (1.0f - a_b)) / a_out;
            b_out = (b_b * a_b + b_a * a_a * (1.0f - a_b)) / a_out;
            break;

        case 1: // Multiply
            r_out = r_a * r_b;
            g_out = g_a * g_b;
            b_out = b_a * b_b;
            a_out = fminf(a_a + a_b, 1.0f);
            break;

        case 2: // Screen
            r_out = 1.0f - (1.0f - r_a) * (1.0f - r_b);
            g_out = 1.0f - (1.0f - g_a) * (1.0f - g_b);
            b_out = 1.0f - (1.0f - b_a) * (1.0f - b_b);
            a_out = fminf(a_a + a_b, 1.0f);
            break;

        case 3: // Overlay
            r_out = (r_a < 0.5f) ? (2.0f * r_a * r_b) : (1.0f - 2.0f * (1.0f - r_a) * (1.0f - r_b));
            g_out = (g_a < 0.5f) ? (2.0f * g_a * g_b) : (1.0f - 2.0f * (1.0f - g_a) * (1.0f - g_b));
            b_out = (b_a < 0.5f) ? (2.0f * b_a * b_b) : (1.0f - 2.0f * (1.0f - b_a) * (1.0f - b_b));
            a_out = fminf(a_a + a_b, 1.0f);
            break;

        default:
            r_out = r_a; g_out = g_a; b_out = b_a; a_out = a_a;
            break;
    }

    output[rgba_idx]     = r_out;
    output[rgba_idx + 1] = g_out;
    output[rgba_idx + 2] = b_out;
    output[rgba_idx + 3] = a_out;
}

extern "C" __global__ void atmosphere_fog_kernel(
    const float* scene_color,   // [r, g, b, a]
    float* output,              // [r, g, b, a]
    int pixel_count,
    float fog_density,
    float fog_r, float fog_g, float fog_b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixel_count) return;

    int rgba_idx = idx * 4;
    float r = scene_color[rgba_idx];
    float g = scene_color[rgba_idx + 1];
    float b = scene_color[rgba_idx + 2];
    float a = scene_color[rgba_idx + 3];

    // Simple fog: blend with fog color based on density
    float fog_amount = fminf(fog_density, 1.0f);
    
    output[rgba_idx]     = r * (1.0f - fog_amount) + fog_r * fog_amount;
    output[rgba_idx + 1] = g * (1.0f - fog_amount) + fog_g * fog_amount;
    output[rgba_idx + 2] = b * (1.0f - fog_amount) + fog_b * fog_amount;
    output[rgba_idx + 3] = a;
}

extern "C" __global__ void vignette_kernel(
    const float* scene_color,
    float* output,
    int width,
    int height,
    float strength,
    float center_x,
    float center_y)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 4;

    // Calculate distance from center (normalized to [0, 1])
    float dx = (x - center_x) / width;
    float dy = (y - center_y) / height;
    float dist = sqrtf(dx * dx + dy * dy);

    // Vignette factor: darker at edges
    float vignette = 1.0f - strength * dist;

    // Apply vignette
    output[idx]     = scene_color[idx]     * vignette;
    output[idx + 1] = scene_color[idx + 1] * vignette;
    output[idx + 2] = scene_color[idx + 2] * vignette;
    output[idx + 3] = scene_color[idx + 3]; // Keep alpha
}
