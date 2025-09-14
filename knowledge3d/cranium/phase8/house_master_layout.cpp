// Phase 8 — House Master Layout (MVP reference)
// Single open-plan space with invisible zone markers for AI navigation.

#include <vector>
#include <cstdint>

enum class GeometryType { HOUSE_MASTER };

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

// Zone markers are AI-only: the exporter encodes them in GLB extras
static void addZoneMarker(std::vector<float>& V, float x, float y, float z) {
  V.push_back(x); V.push_back(y); V.push_back(z);
}

MeshData generateHouseMasterLayout(float scale) {
  std::vector<float> V; std::vector<uint32_t> I;
  addPlane(V, I, 0.0f, 0.0f, 0.0f, 100.0f * scale, 100.0f * scale);
  addZoneMarker(V, -30.0f * scale, 0.0f, 0.0f); // library
  addZoneMarker(V,  30.0f * scale, 0.0f, 0.0f); // garden
  addZoneMarker(V,   0.0f * scale, 0.0f, 30.0f * scale); // workshop
  addZoneMarker(V,   0.0f * scale, 0.0f,-30.0f * scale); // bathtub
  addZoneMarker(V,   0.0f * scale, 0.0f,  0.0f); // living_room
  return MeshData{V, I, GeometryType::HOUSE_MASTER};
}

