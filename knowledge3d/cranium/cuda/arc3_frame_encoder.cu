#include <math.h>

#include "device_functions.cuh"

#define ARC3_MAX_COLORS 16
#define ARC3_TOKEN_BUCKET_START 8

__device__ __forceinline__ unsigned int fnv1a32_arc3_device(const char* text, int len) {
    unsigned int value = 2166136261u;
    for (int index = 0; index < len; ++index) {
        value ^= static_cast<unsigned int>(static_cast<unsigned char>(text[index]));
        value *= 16777619u;
    }
    return value;
}

__device__ __forceinline__ void hash_token_into_embedding_arc3(
    float* embedding,
    const char* token,
    int token_len,
    float magnitude
) {
    const unsigned int token_hash = fnv1a32_arc3_device(token, token_len);
    const int bucket =
        ARC3_TOKEN_BUCKET_START +
        static_cast<int>(token_hash % static_cast<unsigned int>(GPU_TASK_EMBED_DIMS - ARC3_TOKEN_BUCKET_START));
    const float sign = ((token_hash >> 16) & 1u) ? 1.0f : -1.0f;
    const float hash_magnitude = 1.0f + (0.25f * (static_cast<float>((token_hash >> 8) & 0xFFu) / 255.0f));
    embedding[bucket] += sign * magnitude * hash_magnitude;
}

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

    float hist[ARC3_MAX_COLORS];
    for (int color = 0; color < ARC3_MAX_COLORS; ++color) {
        hist[color] = 0.0f;
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
            const unsigned int color = static_cast<unsigned int>(frame[index]);
            if (color < ARC3_MAX_COLORS) {
                hist[color] += 1.0f;
            }
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

    if (nonzero_count > 0u) {
        const float nonzero_f = static_cast<float>(nonzero_count);
        centroid_x /= nonzero_f;
        centroid_y /= nonzero_f;
        normalized_col = centroid_x / static_cast<float>(width);
        normalized_row = centroid_y / static_cast<float>(height);

        float variance_x = 0.0f;
        float variance_y = 0.0f;
        for (unsigned int y = 0u; y < height; ++y) {
            for (unsigned int x = 0u; x < width; ++x) {
                if (frame[(y * width) + x] == 0u) {
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
    const float interaction_readiness =
        device_clamp01(centeredness * occupancy * (0.45f + (0.55f * boundary_density)));
    const float click_readiness =
        device_clamp01(interaction_readiness * (1.0f - spread_mag) * boundary_density);
    const float structural_density = device_clamp01((0.60f * occupancy) + (0.40f * boundary_density));

    unsigned int dominant_color = 0u;
    float dominant_mass = 0.0f;
    for (unsigned int color = 1u; color < ARC3_MAX_COLORS; ++color) {
        if (hist[color] > dominant_mass) {
            dominant_color = color;
            dominant_mass = hist[color];
        }
    }

    float semantic[GPU_TASK_EMBED_DIMS];
    for (int dim = 0; dim < GPU_TASK_EMBED_DIMS; ++dim) {
        semantic[dim] = 0.0f;
    }

    semantic[0] = nonzero_count > 0u ? 0.35f : 0.15f;
    semantic[1] = device_maxf(-1.0f, device_minf(1.0f, cy * 2.0f));
    semantic[2] = 0.0f;
    semantic[3] = nonzero_count > 0u ? 0.65f : 0.0f;
    semantic[4] = 0.0f;
    semantic[5] = nonzero_count > 0u ? 0.35f : 0.0f;
    semantic[6] = 0.0f;
    semantic[7] = 0.0f;

    if (nonzero_count > 0u) {
        semantic[2] = device_clamp01(0.45f + (0.55f * movement_need));
        semantic[4] = device_clamp01(0.25f + (0.75f * movement_need));
        semantic[6] = device_maxf(-1.0f, device_minf(1.0f, cy >= 0.0f ? movement_need : -movement_need));
        semantic[7] = 1.0f;
    }

    hash_token_into_embedding_arc3(semantic, "spatial", 7, 0.30f);
    hash_token_into_embedding_arc3(semantic, "grid", 4, 0.40f);
    hash_token_into_embedding_arc3(semantic, "navigate", 8, 0.25f);
    hash_token_into_embedding_arc3(semantic, "translate", 9, 0.25f);

    if (nonzero_count > 0u) {
        hash_token_into_embedding_arc3(semantic, "object", 6, 0.30f);
        hash_token_into_embedding_arc3(semantic, "color", 5, 0.25f + (0.15f * occupancy));
        hash_token_into_embedding_arc3(semantic, "cell", 4, 0.25f + (0.15f * occupancy));
        hash_token_into_embedding_arc3(semantic, "grid_cell", 9, 0.55f + (0.35f * movement_need));
        hash_token_into_embedding_arc3(semantic, "translate_2d", 12, 0.85f + (0.85f * movement_need));
        hash_token_into_embedding_arc3(semantic, "translation_concept", 19, 0.75f + (0.85f * movement_need));
        hash_token_into_embedding_arc3(semantic, "vec2_add", 8, 0.70f + (0.70f * movement_need));
        hash_token_into_embedding_arc3(semantic, "occupied", 8, 0.20f + (0.20f * occupancy));
        if (nonzero_count == 1u) {
            hash_token_into_embedding_arc3(semantic, "single", 6, 0.55f);
        } else {
            hash_token_into_embedding_arc3(semantic, "cluster", 7, 0.55f + (0.25f * occupancy));
        }
        if (dominant_color > 0u) {
            hash_token_into_embedding_arc3(semantic, "filled", 6, 0.30f + (0.05f * static_cast<float>(dominant_color)));
        }
    } else {
        hash_token_into_embedding_arc3(semantic, "empty", 5, 0.60f);
        hash_token_into_embedding_arc3(semantic, "center", 6, 0.35f);
    }

    if (cy < -0.12f) {
        const float strength = 1.0f + (2.2f * (-cy));
        hash_token_into_embedding_arc3(semantic, "up", 2, strength);
        hash_token_into_embedding_arc3(semantic, "north", 5, 0.85f + (0.65f * movement_need));
        hash_token_into_embedding_arc3(semantic, "move_up", 7, 0.75f + (0.65f * movement_need));
        hash_token_into_embedding_arc3(semantic, "above", 5, 0.65f + (0.45f * movement_need));
    }
    if (cy > 0.12f) {
        const float strength = 1.0f + (2.2f * cy);
        hash_token_into_embedding_arc3(semantic, "down", 4, strength);
        hash_token_into_embedding_arc3(semantic, "south", 5, 0.85f + (0.65f * movement_need));
        hash_token_into_embedding_arc3(semantic, "move_down", 9, 0.75f + (0.65f * movement_need));
        hash_token_into_embedding_arc3(semantic, "below", 5, 0.65f + (0.45f * movement_need));
    }
    if (cx < -0.12f) {
        const float strength = 1.0f + (2.2f * (-cx));
        hash_token_into_embedding_arc3(semantic, "left", 4, strength);
        hash_token_into_embedding_arc3(semantic, "west", 4, 0.85f + (0.55f * movement_need));
        hash_token_into_embedding_arc3(semantic, "move_left", 9, 0.75f + (0.55f * movement_need));
    }
    if (cx > 0.12f) {
        const float strength = 1.0f + (2.2f * cx);
        hash_token_into_embedding_arc3(semantic, "right", 5, strength);
        hash_token_into_embedding_arc3(semantic, "east", 4, 0.85f + (0.55f * movement_need));
        hash_token_into_embedding_arc3(semantic, "move_right", 10, 0.75f + (0.55f * movement_need));
    }
    if (device_absf(cx) < 0.15f && device_absf(cy) < 0.15f) {
        hash_token_into_embedding_arc3(semantic, "center", 6, 0.90f);
        hash_token_into_embedding_arc3(semantic, "centered", 8, 0.75f);
        hash_token_into_embedding_arc3(semantic, "balanced", 8, 0.65f);
    }
    if (interaction_readiness > 0.10f) {
        hash_token_into_embedding_arc3(semantic, "interact", 8, 0.55f + interaction_readiness);
        hash_token_into_embedding_arc3(semantic, "click", 5, 0.45f + click_readiness);
    }
    if (boundary_density > 0.12f) {
        hash_token_into_embedding_arc3(semantic, "delta", 5, 0.35f + boundary_density);
        hash_token_into_embedding_arc3(semantic, "changed", 7, 0.30f + boundary_density);
        hash_token_into_embedding_arc3(semantic, "moved", 5, 0.25f + movement_need);
        hash_token_into_embedding_arc3(semantic, "boundary", 8, 0.30f + boundary_density);
    }
    if (spread_x > (spread_y * 1.25f) && spread_x > 1.0e-4f) {
        hash_token_into_embedding_arc3(semantic, "horizontal", 10, 0.45f + spread_x);
        hash_token_into_embedding_arc3(semantic, "wide", 4, 0.35f + spread_x);
    }
    if (spread_y > (spread_x * 1.25f) && spread_y > 1.0e-4f) {
        hash_token_into_embedding_arc3(semantic, "vertical", 8, 0.45f + spread_y);
        hash_token_into_embedding_arc3(semantic, "tall", 4, 0.35f + spread_y);
    }

    float norm = 0.0f;
    for (int dim = 0; dim < GPU_TASK_EMBED_DIMS; ++dim) {
        norm += semantic[dim] * semantic[dim];
    }
    norm = sqrtf(norm + 1.0e-12f);
    const float inv_norm = norm > 1.0e-6f ? (1.0f / norm) : 1.0f;
    for (int dim = 0; dim < GPU_TASK_EMBED_DIMS; ++dim) {
        embedding[dim] = semantic[dim] * inv_norm;
    }

    // Preserve the direct action-control lanes consumed by arc3_action_select_device.
    embedding[10] = normalized_col;
    embedding[11] = normalized_row;
    embedding[12] = spread_x;
    embedding[13] = spread_y;
    embedding[28] = occupancy;
    embedding[29] = boundary_density;
    embedding[31] = structural_density;
}
