// Phase 6 — Living Room geometry (MVP)
// Simple parametric room with floor, sofa, screen, and avatar tablet.
// Reference only; the Python exporter builds GLB used by the viewer.

#include <vector>
#include <cstdint>

enum class GeometryType { CUBE, CUSTOM };

struct MeshData {
  std::vector<float> vertices;
  std::vector<uint32_t> indices;
  GeometryType type;
};

static void addPlane(std::vector<float>& V, std::vector<uint32_t>& I,
                     float cx, float cy, float cz, float w, float d) {
  const float x0 = cx - w * 0.5f, x1 = cx + w * 0.5f;
  const float z0 = cz - d * 0.5f, z1 = cz + d * 0.5f;
  const uint32_t base = static_cast<uint32_t>(V.size() / 3);
  float verts[] = { x0,cy,z0,  x1,cy,z0,  x1,cy,z1,  x0,cy,z1 };
  V.insert(V.end(), std::begin(verts), std::end(verts));
  uint32_t tri[] = { base, base+1, base+2, base+2, base+3, base };
  I.insert(I.end(), std::begin(tri), std::end(tri));
}

static void addBox(std::vector<float>& V, std::vector<uint32_t>& I,
                   float cx, float cy, float cz, float w, float h, float d) {
  const float x = w * 0.5f, y = h * 0.5f, z = d * 0.5f;
  const uint32_t base = static_cast<uint32_t>(V.size() / 3);
  float verts[] = {
    -x+cx, -y+cy, -z+cz,   x+cx, -y+cy, -z+cz,   x+cx,  y+cy, -z+cz,  -x+cx,  y+cy, -z+cz,
    -x+cx, -y+cy,  z+cz,   x+cx, -y+cy,  z+cz,   x+cx,  y+cy,  z+cz,  -x+cx,  y+cy,  z+cz,
  };
  V.insert(V.end(), std::begin(verts), std::end(verts));
  uint32_t idx[] = {
    0,1,2, 2,3,0,  4,7,6, 6,5,4,
    0,4,5, 5,1,0,  2,6,7, 7,3,2,
    0,3,7, 7,4,0,  1,5,6, 6,2,1
  };
  for (uint32_t k = 0; k < sizeof(idx)/sizeof(idx[0]); ++k) I.push_back(base + idx[k]);
}

MeshData generateLivingRoom(float width, float height, float depth) {
  std::vector<float> V; std::vector<uint32_t> I;
  // Floor
  addPlane(V, I, 0.0f, 0.0f, 0.0f, width, depth);
  // Sofa: centered, near back wall
  addBox(V, I, 0.0f, 0.5f, -depth/2.0f + 2.0f, 4.0f, 1.0f, 2.0f);
  // Screen: wall mounted at front
  addBox(V, I, 0.0f, height*0.7f, depth/2.0f - 0.1f, width*0.8f, height*0.4f, 0.05f);
  // Avatar tablet: floating near center-left
  addBox(V, I, -1.0f, 1.2f, 0.0f, 0.3f, 0.5f, 0.02f);
  return MeshData{V, I, GeometryType::CUBE};
}

