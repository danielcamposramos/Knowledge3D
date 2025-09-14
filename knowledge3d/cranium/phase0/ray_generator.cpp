#include <vector>
#include <cmath>
#include <cstring>

struct DynamicRay {
    float start[3];
    float end[3];
    float thickness;
    float color[3];
    int type;      // 0=straight, 1=curved, 2=curly
    float honesty; // -1 to 1
};

static inline float sigmoidf(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

std::vector<DynamicRay> generateRaysFromEmbedding(
    const std::vector<float>& embedding,
    const std::vector<float>& vertex_positions
) {
    std::vector<DynamicRay> rays;
    if (vertex_positions.size() < 3) return rays;

    if (embedding.size() >= 73) {
        DynamicRay ray{};
        std::memcpy(ray.start, &vertex_positions[0], 3 * sizeof(float));
        ray.end[0] = ray.start[0] + embedding[64] * 2.0f;
        ray.end[1] = ray.start[1] + embedding[65] * 2.0f;
        ray.end[2] = ray.start[2] + embedding[66] * 2.0f;
        ray.thickness = std::fabs(embedding[67]) * 0.1f;
        ray.color[0] = sigmoidf(embedding[68]);
        ray.color[1] = sigmoidf(embedding[69]);
        ray.color[2] = sigmoidf(embedding[70]);
        float tv = embedding[71];
        if (tv > 0.7f) ray.type = 0; else if (tv > 0.3f) ray.type = 1; else ray.type = 2;
        ray.honesty = embedding[72];
        rays.push_back(ray);
    }
    return rays;
}

// Note: This file provides the core logic. A separate executable or binding
// will glue I/O (np.load, JSON emit) according to the repo constraints.

