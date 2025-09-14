// Book mesh template generator (Phase 1: Library Room)
// Generates a cuboid mesh representing a book (cover, back, spine, pages)
// Geometry only; export handled in Python via pygltflib.

#include <vector>
#include <cstdint>

namespace k3d::phase1 {

struct BookMesh {
    std::vector<float> positions;  // xyz triplets
    std::vector<uint16_t> indices; // triangles
};

// Create a simple axis-aligned cuboid centered at origin with given dims.
// width (X), height (Y), depth (Z). For a book, width ~ thickness.
inline BookMesh generateBookMesh(float width, float height, float depth) {
    const float hx = width * 0.5f;
    const float hy = height * 0.5f;
    const float hz = depth * 0.5f;

    // 8 vertices of a cuboid
    std::vector<float> v = {
        -hx, -hy, -hz,  // 0
         hx, -hy, -hz,  // 1
         hx,  hy, -hz,  // 2
        -hx,  hy, -hz,  // 3
        -hx, -hy,  hz,  // 4
         hx, -hy,  hz,  // 5
         hx,  hy,  hz,  // 6
        -hx,  hy,  hz   // 7
    };

    // 12 triangles (two per face)
    std::vector<uint16_t> idx = {
        0,1,2,  2,3,0,  // back (-Z)
        4,6,5,  6,4,7,  // front (+Z)
        0,4,5,  5,1,0,  // bottom (-Y)
        3,2,6,  6,7,3,  // top (+Y)
        1,5,6,  6,2,1,  // right (+X)
        0,3,7,  7,4,0   // left (-X) "spine"
    };

    return BookMesh{std::move(v), std::move(idx)};
}

} // namespace k3d::phase1

