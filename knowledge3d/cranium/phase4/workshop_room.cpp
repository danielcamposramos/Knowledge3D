// Workshop Room geometry: Matrix-style grid floor, walls, screens, and bench

#include <vector>
#include <cmath>

namespace k3d::phase4 {

struct MeshData {
    std::vector<float> vertices;   // xyz triplets
    std::vector<uint32_t> indices; // triangles
};

inline void addBox(std::vector<float>& vertices, std::vector<uint32_t>& indices,
                   float cx, float cy, float cz, float w, float h, float d) {
    float x0 = cx - w * 0.5f, x1 = cx + w * 0.5f;
    float y0 = cy - h * 0.5f, y1 = cy + h * 0.5f;
    float z0 = cz - d * 0.5f, z1 = cz + d * 0.5f;
    uint32_t base = static_cast<uint32_t>(vertices.size() / 3);
    const float box_verts[] = {
        x0,y0,z0, x1,y0,z0, x1,y1,z0, x0,y1,z0,
        x0,y0,z1, x1,y0,z1, x1,y1,z1, x0,y1,z1
    };
    vertices.insert(vertices.end(), std::begin(box_verts), std::end(box_verts));
    const uint32_t box_indices[] = {
        0,1,2, 2,3,0, 4,7,6, 6,5,4,
        0,4,5, 5,1,0, 2,6,7, 7,3,2,
        0,3,7, 7,4,0, 1,5,6, 6,2,1
    };
    for (uint32_t i : box_indices) indices.push_back(base + i);
}

inline MeshData generateWorkshopRoom(float width, float height, float depth) {
    std::vector<float> v;
    std::vector<uint32_t> idx;

    // Floor grid lines are suggested but triangles required for GLB PBR; we keep boxes for walls/screens/bench.
    // Walls (dark) with screens
    addBox(v, idx, 0.0f, height*0.5f, -depth*0.5f + 1.0f, width*0.8f, height*0.8f, 0.1f); // back wall
    addBox(v, idx, -width*0.5f + 1.0f, height*0.5f, 0.0f, 0.1f, height*0.8f, depth*0.8f); // left wall
    addBox(v, idx,  width*0.5f - 1.0f, height*0.5f, 0.0f, 0.1f, height*0.8f, depth*0.8f); // right wall
    // Screens
    addBox(v, idx, 0.0f, height*0.7f, depth*0.5f - 2.0f, width*0.6f, height*0.4f, 0.05f);
    addBox(v, idx, -width*0.5f + 3.0f, height*0.5f, 0.0f, 2.0f, 1.5f, 0.05f);
    addBox(v, idx,  width*0.5f - 3.0f, height*0.5f, 0.0f, 2.0f, 1.5f, 0.05f);
    // Workbench
    addBox(v, idx, 0.0f, 0.5f, 0.0f, width*0.4f, 1.0f, depth*0.4f);

    return MeshData{std::move(v), std::move(idx)};
}

} // namespace k3d::phase4

