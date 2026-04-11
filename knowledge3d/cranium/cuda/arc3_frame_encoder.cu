#include <math.h>

#include "device_functions.cuh"

#define ARC3_MAX_COLORS 16

extern "C" __global__ void arc3_encode_frame(
    const unsigned char* __restrict__ frame,
    float* __restrict__ embedding,
    unsigned int width,
    unsigned int height
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    for (int dim = 0; dim < GPU_TASK_EMBED_DIMS; ++dim) {
        embedding[dim] = 0.0f;
    }

    const unsigned int total = width * height;
    if (total == 0u || width == 0u || height == 0u) {
        return;
    }

    float color_mass[ARC3_MAX_COLORS];
    float color_row_moment[ARC3_MAX_COLORS];
    float color_col_moment[ARC3_MAX_COLORS];
    float row_energy[4];
    float col_energy[4];
    for (int index = 0; index < ARC3_MAX_COLORS; ++index) {
        color_mass[index] = 0.0f;
        color_row_moment[index] = 0.0f;
        color_col_moment[index] = 0.0f;
    }
    for (int index = 0; index < 4; ++index) {
        row_energy[index] = 0.0f;
        col_energy[index] = 0.0f;
    }

    float centroid_x = 0.0f;
    float centroid_y = 0.0f;
    float min_x = static_cast<float>(width);
    float min_y = static_cast<float>(height);
    float max_x = 0.0f;
    float max_y = 0.0f;
    unsigned int nonzero_count = 0u;

    for (unsigned int y = 0u; y < height; ++y) {
        for (unsigned int x = 0u; x < width; ++x) {
            const unsigned int index = (y * width) + x;
            const unsigned int color = static_cast<unsigned int>(frame[index]) & 0x0Fu;
            const float color_f = static_cast<float>(color);
            color_mass[color] += 1.0f;
            color_row_moment[color] += static_cast<float>(y);
            color_col_moment[color] += static_cast<float>(x);
            row_energy[y & 3u] += color_f;
            col_energy[x & 3u] += color_f;
            if (color > 0u) {
                centroid_x += static_cast<float>(x);
                centroid_y += static_cast<float>(y);
                min_x = device_minf(min_x, static_cast<float>(x));
                min_y = device_minf(min_y, static_cast<float>(y));
                max_x = device_maxf(max_x, static_cast<float>(x));
                max_y = device_maxf(max_y, static_cast<float>(y));
                nonzero_count += 1u;
            }
        }
    }

    const float total_f = static_cast<float>(total);
    const float occupancy = static_cast<float>(nonzero_count) / total_f;
    float normalized_col = 0.5f;
    float normalized_row = 0.5f;
    float spread_x = 0.0f;
    float spread_y = 0.0f;
    float bbox_width = 0.0f;
    float bbox_height = 0.0f;

    if (nonzero_count > 0u) {
        const float nonzero_f = static_cast<float>(nonzero_count);
        centroid_x /= nonzero_f;
        centroid_y /= nonzero_f;
        normalized_col = centroid_x / static_cast<float>(width);
        normalized_row = centroid_y / static_cast<float>(height);
        bbox_width = (max_x - min_x + 1.0f) / static_cast<float>(width);
        bbox_height = (max_y - min_y + 1.0f) / static_cast<float>(height);

        float variance_x = 0.0f;
        float variance_y = 0.0f;
        for (unsigned int y = 0u; y < height; ++y) {
            for (unsigned int x = 0u; x < width; ++x) {
                if ((static_cast<unsigned int>(frame[(y * width) + x]) & 0x0Fu) == 0u) {
                    continue;
                }
                const float dx = static_cast<float>(x) - centroid_x;
                const float dy = static_cast<float>(y) - centroid_y;
                variance_x += dx * dx;
                variance_y += dy * dy;
            }
        }
        spread_x = sqrtf(variance_x / nonzero_f) / static_cast<float>(width);
        spread_y = sqrtf(variance_y / nonzero_f) / static_cast<float>(height);
    }

    float h_trans = 0.0f;
    float v_trans = 0.0f;
    for (unsigned int y = 0u; y < height; ++y) {
        for (unsigned int x = 0u; x + 1u < width; ++x) {
            if (frame[(y * width) + x] != frame[(y * width) + (x + 1u)]) {
                h_trans += 1.0f;
            }
        }
    }
    for (unsigned int y = 0u; y + 1u < height; ++y) {
        for (unsigned int x = 0u; x < width; ++x) {
            if (frame[(y * width) + x] != frame[((y + 1u) * width) + x]) {
                v_trans += 1.0f;
            }
        }
    }

    const float max_h_trans = static_cast<float>(height * (width > 0u ? width - 1u : 0u));
    const float max_v_trans = static_cast<float>((height > 0u ? height - 1u : 0u) * width);
    const float transition_h = max_h_trans > 0.0f ? h_trans / max_h_trans : 0.0f;
    const float transition_v = max_v_trans > 0.0f ? v_trans / max_v_trans : 0.0f;
    const float boundary_density = device_clamp01(0.5f * (transition_h + transition_v));

    const float cx = normalized_col - 0.5f;
    const float cy = normalized_row - 0.5f;
    const float spread_mag = sqrtf((spread_x * spread_x) + (spread_y * spread_y));
    const float centeredness = 1.0f - device_clamp01((device_absf(cx) + device_absf(cy)) * 1.25f);
    const float movement_need =
        device_clamp01((device_absf(cx) + device_absf(cy)) * 1.6f + (0.35f * spread_mag));
    const float interaction_density =
        device_clamp01((0.50f * occupancy) + (0.25f * boundary_density) + (0.25f * centeredness));
    const float click_readiness =
        device_clamp01(interaction_density * (1.0f - spread_mag) * boundary_density);
    const float structural_density = device_clamp01((0.60f * occupancy) + (0.40f * boundary_density));
    const float north_open = device_clamp01(normalized_row);
    const float south_open = device_clamp01(1.0f - normalized_row);
    const float west_open = device_clamp01(normalized_col);
    const float east_open = device_clamp01(1.0f - normalized_col);

    unsigned int dominant_color = 0u;
    unsigned int secondary_color = 0u;
    float dominant_mass = 0.0f;
    float secondary_mass = 0.0f;
    float color_entropy = 0.0f;
    for (unsigned int color = 0u; color < ARC3_MAX_COLORS; ++color) {
        const float mass = color_mass[color] / total_f;
        if (mass > dominant_mass) {
            secondary_mass = dominant_mass;
            secondary_color = dominant_color;
            dominant_mass = mass;
            dominant_color = color;
        } else if (mass > secondary_mass) {
            secondary_mass = mass;
            secondary_color = color;
        }
        if (mass > 1.0e-6f) {
            color_entropy -= mass * log2f(mass);
        }
    }
    color_entropy = device_clamp01(color_entropy / 4.0f);

    float row_harmonic = 0.0f;
    float col_harmonic = 0.0f;
    float diag_trace = 0.0f;
    float anti_diag_trace = 0.0f;
    for (unsigned int y = 0u; y < height; ++y) {
        for (unsigned int x = 0u; x < width; ++x) {
            const float value = static_cast<float>(static_cast<unsigned int>(frame[(y * width) + x]) & 0x0Fu) / 15.0f;
            const float row_phase = 6.28318530718f * static_cast<float>(y & 3u) / 4.0f;
            const float col_phase = 6.28318530718f * static_cast<float>(x & 3u) / 4.0f;
            row_harmonic += value * cosf(row_phase);
            col_harmonic += value * cosf(col_phase);
            if (x == y) {
                diag_trace += value;
            }
            if ((x + y + 1u) == width) {
                anti_diag_trace += value;
            }
        }
    }
    row_harmonic = 0.5f + (0.5f * device_clamp_range(row_harmonic / total_f, -1.0f, 1.0f));
    col_harmonic = 0.5f + (0.5f * device_clamp_range(col_harmonic / total_f, -1.0f, 1.0f));
    diag_trace = device_clamp01(diag_trace / device_maxf(1.0f, static_cast<float>(width)));
    anti_diag_trace = device_clamp01(anti_diag_trace / device_maxf(1.0f, static_cast<float>(width)));

    embedding[0] = nonzero_count > 0u ? 1.0f : 0.0f;
    embedding[1] = centeredness;
    embedding[2] = movement_need;
    embedding[3] = interaction_density;
    embedding[4] = click_readiness;
    embedding[5] = device_clamp01(0.5f + (0.5f * device_clamp_range(cx * 2.0f, -1.0f, 1.0f)));
    embedding[6] = device_clamp01(0.5f + (0.5f * device_clamp_range(cy * 2.0f, -1.0f, 1.0f)));
    embedding[7] = color_entropy;
    embedding[8] = north_open;
    embedding[9] = south_open;
    embedding[10] = normalized_col;
    embedding[11] = normalized_row;
    embedding[12] = spread_x;
    embedding[13] = spread_y;
    embedding[14] = west_open;
    embedding[15] = east_open;
    embedding[16] = bbox_width;
    embedding[17] = bbox_height;
    embedding[18] = dominant_mass;
    embedding[19] = secondary_mass;
    embedding[20] = static_cast<float>(dominant_color) / 15.0f;
    embedding[21] = static_cast<float>(secondary_color) / 15.0f;
    embedding[22] = transition_h;
    embedding[23] = transition_v;
    embedding[24] = row_harmonic;
    embedding[25] = col_harmonic;
    embedding[26] = diag_trace;
    embedding[27] = anti_diag_trace;
    embedding[28] = occupancy;
    embedding[29] = boundary_density;
    embedding[30] = interaction_density;
    embedding[31] = structural_density;

    for (unsigned int color = 0u; color < ARC3_MAX_COLORS; ++color) {
        const float mass = color_mass[color] / total_f;
        const unsigned int lane = 32u + color;
        embedding[lane] = mass;
    }

    for (unsigned int color = 0u; color < 8u; ++color) {
        const float mass = color_mass[color] / total_f;
        const float row_center = color_mass[color] > 0.0f
            ? (color_row_moment[color] / color_mass[color]) / static_cast<float>(height)
            : 0.5f;
        const float col_center = color_mass[color] > 0.0f
            ? (color_col_moment[color] / color_mass[color]) / static_cast<float>(width)
            : 0.5f;
        embedding[48u + color] = device_clamp01((0.55f * mass) + (0.45f * row_center));
        embedding[56u + color] = device_clamp01((0.55f * mass) + (0.45f * col_center));
    }

    float norm = 0.0f;
    for (int dim = 0; dim < GPU_TASK_EMBED_DIMS; ++dim) {
        norm += embedding[dim] * embedding[dim];
    }
    norm = sqrtf(norm + 1.0e-12f);
    const float inv_norm = norm > 1.0e-6f ? (1.0f / norm) : 1.0f;
    for (int dim = 0; dim < GPU_TASK_EMBED_DIMS; ++dim) {
        embedding[dim] *= inv_norm;
    }

    // Keep the dispatch-consumed control lanes stable after normalization.
    embedding[10] = normalized_col;
    embedding[11] = normalized_row;
    embedding[12] = spread_x;
    embedding[13] = spread_y;
    embedding[28] = occupancy;
    embedding[29] = boundary_density;
    embedding[31] = structural_density;
}
