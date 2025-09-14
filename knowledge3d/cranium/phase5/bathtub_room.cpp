// Bathtub Room geometry: undulating floor, oval tub, dream particles (positions only)

#include <vector>
#include <cmath>

namespace k3d::phase5 {

struct MeshData {
    std::vector<float> vertices;   // xyz triplets
    std::vector<uint32_t> indices; // triangles
};

inline MeshData generateBathtubRoom(float width, float height, float depth) {
    std::vector<float> v;
    std::vector<uint32_t> idx;

    // Undulating floor grid
    const int grid = 32;
    for (int i = 0; i <= grid; ++i) {
        for (int j = 0; j <= grid; ++j) {
            float x = -width*0.5f + (width * i) / grid;
            float z = -depth*0.5f + (depth * j) / grid;
            float y = 0.1f * std::sin(x * 0.5f) * std::cos(z * 0.5f);
            v.push_back(x); v.push_back(y); v.push_back(z);
        }
    }
    for (int i = 0; i < grid; ++i) {
        for (int j = 0; j < grid; ++j) {
            int idx0 = i * (grid + 1) + j;
            int idx1 = idx0 + 1;
            int idx2 = idx0 + (grid + 1);
            int idx3 = idx2 + 1;
            idx.push_back(idx0); idx.push_back(idx2); idx.push_back(idx1);
            idx.push_back(idx1); idx.push_back(idx2); idx.push_back(idx3);
        }
    }

    // Oval tub side walls (ring bottom+top)
    const int seg = 32; float rx = 1.5f, rz = 2.0f, th = 0.8f;
    int base = static_cast<int>(v.size()/3);
    for (int i = 0; i <= seg; ++i) {
        float a = 2.0f * float(M_PI) * float(i) / float(seg);
        float x = rx * std::cos(a), z = rz * std::sin(a);
        v.push_back(x); v.push_back(0.0f); v.push_back(z);
    }
    for (int i = 0; i <= seg; ++i) {
        float a = 2.0f * float(M_PI) * float(i) / float(seg);
        float x = rx * std::cos(a), z = rz * std::sin(a);
        v.push_back(x); v.push_back(th); v.push_back(z);
    }
    for (int i = 0; i < seg; ++i) {
        int b0 = base + i;
        int b1 = base + (i + 1) % seg;
        int t0 = base + (seg + 1) + i;
        int t1 = base + (seg + 1) + (i + 1) % seg;
        idx.push_back(b0); idx.push_back(t0); idx.push_back(b1);
        idx.push_back(b1); idx.push_back(t0); idx.push_back(t1);
    }
    // Particles are left to exporter as POINTS (no triangles here)

    return MeshData{std::move(v), std::move(idx)};
}

} // namespace k3d::phase5

