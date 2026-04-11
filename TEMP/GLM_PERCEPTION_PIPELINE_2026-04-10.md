# K3D Perception Pipeline: House → Galaxy

## The Missing Piece

You have four孤立kernels and an O(N²) brute-force scanner. What's missing is the **glue**: query builders that translate entity state into kernel parameters, stream compaction between stages, a saliency scoring system that doesn't exist yet, and a binding mechanism that writes House objects into Galaxy working memory. Here's the complete pipeline.

---

## Architecture

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                         HOUSE (VRAM)                           │
  │   entity_pool[]   morton_octree   transform_history[]          │
  └────────┬───────────────────────────────────────────────────────┘
           │
     ┌─────▼──────┐
     │ STAGE 0:   │  EntityHotPath → Morton AABB query
     │ QUERY BUILD│  cranial_origin + perception_radius → search volume
     └─────┬──────┘
           │
     ┌─────▼──────┐
     │ STAGE 1:   │  morton_octree_range_query()
     │ SPATIAL    │  O(K log N) candidate extraction
     │ QUERY      │  → candidate_ids[] (compact)
     └─────┬──────┘
           │
     ┌─────▼──────┐
     │ STAGE 2:   │  frustum_cull()
     │ FRUSTUM    │  cranial_origin + gaze → 6 planes
     │ FILTER     │  → visible_ids[] (compact)
     └─────┬──────┘
           │
     ┌─────▼──────┐
     │ STAGE 3:   │  dynamic_lod_gate()
     │ LOD GATING │  distance / perception_radius → LOD level
     │            │  → perceived_ids[] + lod_level[]
     └─────┬──────┘
           │
     ┌─────▼──────┐
     │ STAGE 4:   │  saliency_score()  ← NEW KERNEL
     │ SALIENCY   │  motion + novelty + threat + goal + proximity
     │ RANKING    │  → ranked_ids[] (sorted by saliency)
     └─────┬──────┘
           │
     ┌─────▼──────┐
     │ STAGE 5:   │  galaxy_bind()  ← NEW KERNEL
     │ GALAXY     │  House entity → Galaxy working memory star
     │ BINDING    │  → galaxy_stars[] (top-K by saliency)
     └─────┬──────┘
           │
  ┌────────▼───────────────────────────────────────────────────────┐
  │                        GALAXY (VRAM)                           │
  │   working_memory[]   attention_focus   associative_links[]   │
  └───────────────────────────────────────────────────────────────┘
```

---

## Core Data Structures

```cuda
// ─── perception_flags bitmask ───
#define PERCEIVE_SPHERICAL   0x01  // 360° scan, bypass frustum
#define PERCEIVE_MOTION      0x02  // motion-saliency channel
#define PERCEIVE_NOVELTY     0x04  // novelty-saliency channel
#define PERCEIVE_THREAT      0x08  // threat-saliency channel
#define PERCEIVE_GOAL        0x10  // goal-saliency channel
#define PERCEIVE_AUDITORY    0x20  // extended range, reduced LOD
#define PERCEIVE_FORCE_HI    0x40  // override: force max LOD
#define PERCEIVE_TUNNEL      0x80  // narrow frustum, long range

// ─── EntityHotPath (already exists, extended) ───
struct EntityHotPath {
    float house_x, house_y, house_z;     // world position
    float cranial_origin[3];             // sensor offset from origin
    float perception_radius;             // max perception distance
    uint8_t perception_flags;            // bitmask above

    // ─── NEW: gaze state (was missing) ───
    float gaze_yaw;                      // radians, 0 = +X
    float gaze_pitch;                    // radians, 0 = horizon
    float gaze_fov;                       // horizontal FOV in radians

    // ─── NEW: attention target (feedback from Galaxy) ───
    uint32_t attention_entity_id;         // 0 = none, overrides gaze
    float     attention_weight;           // 0..1, blends into saliency
};

// ─── Pipeline intermediate buffers (all VRAM) ───
struct PerceptionScratch {
    // Stage 1 output
    uint32_t* candidate_ids;     // from Morton query
    uint32_t  candidate_count;

    // Stage 2 output
    uint32_t* visible_ids;       // from frustum cull
    uint32_t  visible_count;

    // Stage 3 output
    uint32_t* perceived_ids;     // from LOD gate
    uint8_t*  lod_levels;        // per-entity LOD
    uint32_t  perceived_count;

    // Stage 4 output
    uint32_t* ranked_ids;        // sorted by saliency desc
    float*    saliency_scores;   // per-entity score
    uint32_t  ranked_count;

    // Stage 5 output
    uint32_t  bound_count;       // number of Galaxy stars written
};

// ─── Galaxy Star (working memory entry) ───
struct GalaxyStar {
    uint32_t house_entity_id;    // source in House
    float    relative_pos[3];    // ego-centric position
    float    velocity[3];        // estimated velocity
    float    saliency;           // current attention score
    uint8_t  lod_level;          // detail level committed
    uint8_t  star_flags;         // IS_NOVEL, IS_THREAT, IS_GOAL, etc.
    uint32_t last_perceived_tick;
    uint32_t perceive_count;     // times perceived (novelty decay)
    uint32_t associative_link;   // linked star index (0 = none)
};

#define STAR_NOVEL    0x01
#define STAR_THREAT   0x02
#define STAR_GOAL     0x04
#define STAR_MOVING   0x08
#define STAR_VISIBLE  0x10
```

---

## STAGE 0: Query Builder

Translates `EntityHotPath` into concrete search parameters for downstream kernels. Runs as a **single-thread GPU task or host-side constant-mem write** — it's just parameter computation.

```cuda
struct PerceptionQuery {
    // Spatial query params (Stage 1)
    float3   sensor_origin;       // house_pos + cranial_origin
    float    search_radius;       // perception_radius (×1.5 if AUDITORY)
    uint32_t morton_min;          // AABB lower corner → Morton
    uint32_t morton_max;          // AABB upper corner → Morton

    // Frustum params (Stage 2)
    float4   frustum_planes[6];   // computed from sensor_origin + gaze
    bool     bypass_frustum;      // SPHERICAL flag

    // LOD params (Stage 3)
    float    lod_band_0;          // perception_radius * 0.25  (HI)
    float    lod_band_1;          // perception_radius * 0.50  (MED)
    float    lod_band_2;          // perception_radius * 0.75  (LO)
    float    lod_band_3;          // perception_radius * 1.00  (ULTRA_LO)

    // Saliency weights (Stage 4)
    float    w_proximity;
    float    w_motion;
    float    w_novelty;
    float    w_threat;
    float    w_goal;
};

// ─── Host or single-thread device launch ───
__global__ void k_perception_build_query(
    const EntityHotPath* entity,
    /* out */ PerceptionQuery* query,
    uint32_t current_tick)
{
    // ─── Sensor origin = entity position + cranial offset ───
    float3 origin = make_float3(
        entity->house_x + entity->cranial_origin[0],
        entity->house_y + entity->cranial_origin[1],
        entity->house_z + entity->cranial_origin[2]
    );
    query->sensor_origin = origin;

    // ─── Search radius with flag modifiers ───
    float radius = entity->perception_radius;
    if (entity->perception_flags & PERCEIVE_AUDITORY)  radius *= 1.5f;
    if (entity->perception_flags & PERCEIVE_TUNNEL)     radius *= 2.0f;
    query->search_radius = radius;

    // ─── AABB → Morton range ───
    float3 aabb_min = make_float3(origin.x - radius, origin.y - radius, origin.z - radius);
    float3 aabb_max = make_float3(origin.x + radius, origin.y + radius, origin.z + radius);
    query->morton_min = morton_encode(aabb_min);
    query->morton_max = morton_encode(aabb_max);

    // ─── Frustum planes from gaze ───
    query->bypass_frustum = (entity->perception_flags & PERCEIVE_SPHERICAL);
    if (!query->bypass_frustum) {
        float fov = entity->gaze_fov;
        if (entity->perception_flags & PERCEIVE_TUNNEL) fov *= 0.5f;

        // Build view matrix from sensor_origin + yaw/pitch
        float3 forward = make_float3(
            cosf(entity->gaze_pitch) * cosf(entity->gaze_yaw),
            sinf(entity->gaze_pitch),
            cosf(entity->gaze_pitch) * sinf(entity->gaze_yaw)
        );
        float3 up    = make_float3(0, 1, 0);
        float3 right = normalize(cross(forward, up));

        // Extract 6 frustum planes from view-projection
        // (standard near/far/left/right/top/bottom extraction)
        build_frustum_planes(origin, forward, right, up,
                             fov, radius, query->frustum_planes);
    }

    // ─── LOD bands driven by perception_radius ───
    query->lod_band_0 = radius * 0.25f;
    query->lod_band_1 = radius * 0.50f;
    query->lod_band_2 = radius * 0.75f;
    query->lod_band_3 = radius * 1.00f;

    // ─── Saliency weights from flags ───
    query->w_proximity = 1.0f;
    query->w_motion    = (entity->perception_flags & PERCEIVE_MOTION)  ? 1.0f : 0.0f;
    query->w_novelty   = (entity->perception_flags & PERCEIVE_NOVELTY) ? 0.8f : 0.0f;
    query->w_threat    = (entity->perception_flags & PERCEIVE_THREAT)  ? 1.5f : 0.0f;
    query->w_goal      = (entity->perception_flags & PERCEIVE_GOAL)    ? 1.2f : 0.0f;
}
```

---

## STAGE 1: Spatial Query (Morton Octree)

**Replaces `bh_perceive_count()` O(N²) scan with O(K log N) range query.**

Your existing Morton Octree kernel does spatial indexing. The missing piece is the **range query** that extracts all entity IDs within a Morton code range.

```cuda
// ─── Morton Octree Node (existing) ───
struct MortonNode {
    uint32_t morton_code;
    uint32_t entity_id;       // ENTITY_NULL if internal node
    uint32_t child[8];        // child node indices
    uint32_t pad;
};

// ─── Stack-based Morton range query (GPU kernel) ───
// Each thread processes one query avatar (or one thread block for one avatar)
// For a single avatar, this is a single-thread launch that walks the tree

__global__ void k_morton_range_query(
    const MortonNode* __restrict__ octree,
    uint32_t                root_index,
    const PerceptionQuery*  query,
    /* out */ uint32_t*     candidate_ids,
    /* out */ uint32_t*     candidate_count)
{
    // Stack for DFS traversal (max depth = 21 for 64-bit Morton codes)
    uint32_t stack[32];
    int      stack_ptr = 0;
    uint32_t out_idx   = 0;

    stack[stack_ptr++] = root_index;

    while (stack_ptr > 0) {
        uint32_t node_idx = stack[--stack_ptr];
        const MortonNode& node = octree[node_idx];

        // Prune: if node's Morton range doesn't overlap query range, skip
        if (node.morton_code < query->morton_min ||
            node.morton_code > query->morton_max) {
            // Could still have children in range if this is a parent
            // For leaf nodes, prune entirely
            if (node.entity_id != ENTITY_NULL) continue;
        }

        // Leaf: emit if in range
        if (node.entity_id != ENTITY_NULL) {
            if (node.morton_code >= query->morton_min &&
                node.morton_code <= query->morton_max) {
                candidate_ids[out_idx++] = node.entity_id;
            }
            continue;
        }

        // Internal: push children
        for (int i = 7; i >= 0; --i) {
            if (node.child[i] != ENTITY_NULL) {
                stack[stack_ptr++] = node.child[i];
            }
        }
    }

    *candidate_count = out_idx;
}
```

> **Why this replaces O(N²):** Instead of every entity checking every other entity, the avatar walks the octree once. For N=100K entities and K=200 in range, this is 200 comparisons vs 10 billion.

---

## STAGE 2: Frustum Filtering

Connects `cranial_origin` + `gaze_yaw/pitch` → `frustum_cull.cu`.

```cuda
__global__ void k_perception_frustum_cull(
    const uint32_t*  candidate_ids,     // from Stage 1
    uint32_t         candidate_count,
    const float4*    frustum_planes,    // from Stage 0 (6 planes)
    bool             bypass_frustum,    // SPHERICAL flag
    const float3     sensor_origin,     // for distance check
    const float*     __restrict__ entity_x,
    const float*     __restrict__ entity_y,
    const float*     __restrict__ entity_z,
    /* out */ uint32_t* visible_ids,
    /* out */ uint32_t* visible_count)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= candidate_count) return;

    uint32_t eid = candidate_ids[tid];

    float3 pos = make_float3(entity_x[eid], entity_y[eid], entity_z[eid]);

    // ─── Spherical bypass ───
    if (bypass_frustum) {
        //