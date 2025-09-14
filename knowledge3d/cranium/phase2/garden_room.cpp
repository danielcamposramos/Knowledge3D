// Circular greenhouse room geometry (floor + cylindrical walls)
// Geometry-only stub; export is handled in Python via pygltflib.

#include <vector>
#include <cmath>

namespace k3d::phase2 {

struct MeshData {
    std::vector<float> vertices;   // xyz triplets
    std::vector<uint32_t> indices; // triangles
};

inline MeshData generateCircularGreenhouse(float radius, float height) {
    const int segments = 64;
    std::vector<float> v;
    std::vector<uint32_t> idx;

    // Floor (triangle fan)
    v.push_back(0.0f); v.push_back(0.0f); v.push_back(0.0f);
    for (int i = 0; i <= segments; ++i) {
        float a = 2.0f * float(M_PI) * float(i) / float(segments);
        float x = radius * std::cos(a);
        float z = radius * std::sin(a);
        v.push_back(x); v.push_back(0.0f); v.push_back(z);
    }
    for (int i = 1; i <= segments; ++i) {
        idx.push_back(0u);
        idx.push_back(static_cast<uint32_t>(i));
        idx.push_back(static_cast<uint32_t>((i % segments) + 1));
    }

    // Walls (cylinder quads -> triangles)
    const int wall_segments = 32;
    const uint32_t base = static_cast<uint32_t>(v.size() / 3);
    for (int i = 0; i <= wall_segments; ++i) {
        float a = 2.0f * float(M_PI) * float(i) / float(wall_segments);
        float x = radius * std::cos(a);
        float z = radius * std::sin(a);
        // bottom
        v.push_back(x); v.push_back(0.0f); v.push_back(z);
        // top
        v.push_back(x); v.push_back(height); v.push_back(z);
    }
    for (int i = 0; i < wall_segments; ++i) {
        uint32_t i0 = base + i * 2;
        uint32_t i1 = base + ((i + 1) % wall_segments) * 2;
        // two triangles per quad: (i0,i0+1,i1+1) and (i0,i1+1,i1)
        idx.push_back(i0); idx.push_back(i0 + 1); idx.push_back(i1 + 1);
        idx.push_back(i0); idx.push_back(i1 + 1); idx.push_back(i1);
    }

    return MeshData{std::move(v), std::move(idx)};
}

} // namespace k3d::phase2

