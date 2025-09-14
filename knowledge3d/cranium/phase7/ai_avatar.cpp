// Phase 7 — AI Avatar (MVP reference)
// Robotic avatar with a head and an internal Cranium Core geometry.
// This file documents the geometry layout; Python exporters/renderers build GLBs.

#include <vector>
#include <cstdint>
#include <cmath>

enum class GeometryType { AI_AVATAR, CRANIUM_CORE, CUBE, CYLINDER, TETRA, ICOSA }; 

struct MeshData {
  std::vector<float> vertices;
  std::vector<uint32_t> indices;
  GeometryType type;
  bool is_internal = false; // true for Cranium Core
};

// Helpers (signatures only for reference)
static void addCube(std::vector<float>& V, std::vector<uint32_t>& I,
                    float cx, float cy, float cz, float w, float h, float d);
static void addCylinder(std::vector<float>& V, std::vector<uint32_t>& I,
                        float cx, float cy, float cz, float r, float h, int seg = 24);
static void mergeMesh(std::vector<float>& V, std::vector<uint32_t>& I, const MeshData& m);
static void offsetMesh(MeshData& m, float ox, float oy, float oz);

struct AICustomization {
  GeometryType head_shape = GeometryType::CUBE;
  int working_memory_size = 64;
  int diary_size = 24;
};

static MeshData generateTetra(float s) {
  MeshData m; m.type = GeometryType::TETRA;
  const float a = s;
  float v[] = { 0,a,0,  -a,-a,0,  a,-a,0,  0,0,1.633f*a };
  uint32_t idx[] = {0,1,2, 0,2,3, 0,3,1, 1,3,2};
  m.vertices.assign(v, v+12);
  m.indices.assign(idx, idx+12);
  return m;
}

static MeshData generateIcosa(float s);

static MeshData generateCraniumCore(const AICustomization& custom, float scale) {
  MeshData core; core.type = GeometryType::CRANIUM_CORE; core.is_internal = true;
  std::vector<float> V; std::vector<uint32_t> I;
  // Working memory stars
  for (int i = 0; i < custom.working_memory_size; ++i) {
    MeshData star = generateTetra(0.05f * scale);
    float x = std::fmod(float(i) * 0.37f, 1.6f) - 0.8f;
    float y = std::fmod(float(i) * 0.53f, 1.6f) - 0.8f;
    float z = std::fmod(float(i) * 0.61f, 1.6f) - 0.8f;
    offsetMesh(star, x, y, z);
    mergeMesh(V, I, star);
  }
  // Diary pages (cubes) stacked inside head
  for (int i = 0; i < custom.diary_size; ++i) {
    MeshData page; page.type = GeometryType::CUBE; page.vertices = {}; page.indices = {};
    // Reuse addCube via a temporary mesh
    std::vector<float> Pv; std::vector<uint32_t> Pi;
    addCube(Pv, Pi, 0.0f, -0.3f + i * -0.02f * scale, 0.0f, 0.06f*scale, 0.04f*scale, 0.01f*scale);
    page.vertices.swap(Pv); page.indices.swap(Pi);
    mergeMesh(V, I, page);
  }
  // Logic engine at center
  MeshData logic = generateIcosa(0.12f * scale);
  mergeMesh(V, I, logic);
  core.vertices.swap(V); core.indices.swap(I);
  return core;
}

static MeshData generateAIAvatar(float scale, const AICustomization& custom) {
  std::vector<float> V; std::vector<uint32_t> I;
  // Head (cube proxy) sitting at y=1.7
  addCube(V, I, 0.0f, 1.7f, 0.0f, 0.3f*scale, 0.3f*scale, 0.3f*scale);
  // Neck + chest
  addCylinder(V, I, 0.0f, 1.3f, 0.0f, 0.07f*scale, 0.3f*scale);
  addCube(V, I, 0.0f, 0.9f, 0.0f, 0.4f*scale, 0.6f*scale, 0.25f*scale);
  // Arms
  addCylinder(V, I, -0.3f, 1.2f, 0.0f, 0.05f*scale, 0.5f*scale);
  addCylinder(V, I,  0.3f, 1.2f, 0.0f, 0.05f*scale, 0.5f*scale);
  // Legs
  addCylinder(V, I, -0.15f, 0.5f, 0.0f, 0.07f*scale, 0.7f*scale);
  addCylinder(V, I,  0.15f, 0.5f, 0.0f, 0.07f*scale, 0.7f*scale);

  // Internal cranium core (AI‑only)
  MeshData core = generateCraniumCore(custom, scale);
  mergeMesh(V, I, core);

  MeshData out; out.vertices.swap(V); out.indices.swap(I); out.type = GeometryType::AI_AVATAR; return out;
}

