/**
 * Morton Octree - GPU-Native Spatial Indexing for K3D
 *
 * This kernel implements Z-order curve (Morton code) spatial indexing for
 * sub-100ms queries in the K3D House memory. Replaces CPU-bound k-NN.
 *
 * Design Philosophy:
 * - Morton codes enable GPU-friendly binary search (sorted on GPU via CUB)
 * - Spatial proximity preserved via bit interleaving (Z-order curve)
 * - No external dependencies (no FAISS, pure CUDA)
 *
 * Author: Claude (K3D Core Team)
 * Date: 2025-10-04
 * License: Apache-2.0
 */

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

__device__ inline float clamp_unit(float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

__device__ inline uint32_t atomicAdd_u32(uint32_t* address, uint32_t val) {
    uint32_t old;
    asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(address), "r"(val));
    return old;
}

/**
 * Interleave bits of 3 integers to create a 30-bit Morton code.
 * Encodes (x,y,z) spatial position as a single scalar for Z-order sorting.
 *
 * Algorithm: For 10-bit inputs, interleave as:
 *   morton[0] = x[0], morton[1] = y[0], morton[2] = z[0],
 *   morton[3] = x[1], morton[4] = y[1], morton[5] = z[2], ...
 *
 * @param x X-coordinate (10-bit integer, range 0-1023)
 * @param y Y-coordinate (10-bit integer, range 0-1023)
 * @param z Z-coordinate (10-bit integer, range 0-1023)
 * @return 30-bit Morton code
 */
__device__ uint32_t morton_encode_3d(uint32_t x, uint32_t y, uint32_t z) {
    // Part1By2: x -> x000x000x000... (spread bits with 2 zeros between)
    auto part1by2 = [](uint32_t n) -> uint32_t {
        n &= 0x000003ff;                  // Keep only 10 bits
        n = (n ^ (n << 16)) & 0xff0000ff; // x--------|--------|xxxxxxxx
        n = (n ^ (n <<  8)) & 0x0300f00f; // x-------|xx-------|xxxx
        n = (n ^ (n <<  4)) & 0x030c30c3; // x---|xx|xx|xx|xxxx
        n = (n ^ (n <<  2)) & 0x09249249; // x-|x|x|x|x|x|x|x|x|x
        return n;
    };

    uint32_t mx = part1by2(x);
    uint32_t my = part1by2(y);
    uint32_t mz = part1by2(z);

    // Interleave: Z at bit 0, Y at bit 1, X at bit 2
    return (mx << 2) | (my << 1) | mz;
}

/**
 * Compute Morton codes for all nodes in the dataset.
 *
 * Each thread processes one node:
 * 1. Load (x,y,z) position
 * 2. Normalize to [0,1] using bounding box
 * 3. Quantize to 10-bit integers (1024 subdivisions per axis)
 * 4. Encode as Morton code
 * 5. Store for later sorting
 *
 * @param positions Input positions (Nx3 float array)
 * @param node_count Number of nodes
 * @param morton_codes Output Morton codes (N uint32 array)
 * @param bbox_min Minimum corner of bounding box (3 floats)
 * @param bbox_size Size of bounding box (scalar, assumes cubic)
 */
extern "C" __global__ void compute_morton_codes(
    const float* __restrict__ positions,    // [N, 3]
    uint32_t node_count,
    uint32_t* __restrict__ morton_codes,    // [N]
    float bbox_min_x,
    float bbox_min_y,
    float bbox_min_z,
    float bbox_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= node_count) return;

    // Load position (SoA layout assumed: [x0,x1,...,xN,y0,y1,...,yN,z0,z1,...,zN])
    // If AoS layout (interleaved xyz), adjust indexing
    float x = positions[idx * 3 + 0];
    float y = positions[idx * 3 + 1];
    float z = positions[idx * 3 + 2];

    // Normalize to [0, 1]
    float nx = (x - bbox_min_x) / bbox_size;
    float ny = (y - bbox_min_y) / bbox_size;
    float nz = (z - bbox_min_z) / bbox_size;

    // Clamp to [0, 1] (handle floating point edge cases)
    nx = clamp_unit(nx);
    ny = clamp_unit(ny);
    nz = clamp_unit(nz);

    // Quantize to 10-bit integers (0-1023)
    uint32_t ix = static_cast<uint32_t>(nx * 1023.0f);
    uint32_t iy = static_cast<uint32_t>(ny * 1023.0f);
    uint32_t iz = static_cast<uint32_t>(nz * 1023.0f);

    // Encode as Morton code
    uint32_t morton = morton_encode_3d(ix, iy, iz);

    // Store
    morton_codes[idx] = morton;
}

/**
 * Binary search for the first index where morton_codes[i] >= target.
 *
 * Standard binary search on sorted array. Returns lower_bound index.
 *
 * @param morton_codes Sorted Morton codes
 * @param count Array length
 * @param target Search key
 * @return Index of first element >= target (or count if not found)
 */
__device__ uint32_t binary_search_lower_bound(
    const uint32_t* morton_codes,
    uint32_t count,
    uint32_t target
) {
    uint32_t left = 0;
    uint32_t right = count;

    while (left < right) {
        uint32_t mid = (left + right) >> 1;

        if (morton_codes[mid] < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return left;
}

/**
 * Query octree using Morton code range.
 *
 * Algorithm:
 * 1. Compute query Morton code from center position
 * 2. Compute Morton code radius (approximate spatial radius in Morton space)
 * 3. Binary search for [min_morton, max_morton] range
 * 4. Collect all node IDs in range
 *
 * Note: Morton code range query is APPROXIMATE. Post-filtering by Euclidean
 * distance may be needed for exact radius queries. This kernel returns all
 * candidates within the bounding cube.
 *
 * @param morton_sorted Sorted Morton codes
 * @param node_ids_sorted Node IDs corresponding to sorted Morton codes
 * @param total_nodes Array length
 * @param query_morton Query position as Morton code
 * @param query_radius Radius in Morton code space (NOT Euclidean distance!)
 * @param result_buffer Output: node IDs within range
 * @param result_count Output: number of results
 * @param max_results Maximum results to return (buffer size)
 */
extern "C" __global__ void octree_query_morton(
    const uint32_t* __restrict__ morton_sorted,
    const uint32_t* __restrict__ node_ids_sorted,
    uint32_t total_nodes,
    uint32_t query_morton,
    uint32_t query_radius,
    uint32_t* __restrict__ result_buffer,
    uint32_t* __restrict__ result_count,
    uint32_t max_results
) {
    // Single-threaded for MVP (binary search is inherently serial)
    // Future optimization: use warp-level parallelism for range collection

    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Compute Morton code range
    uint32_t min_morton = (query_morton > query_radius) ? (query_morton - query_radius) : 0;
    uint32_t max_morton = query_morton + query_radius;

    // Binary search for lower bound
    uint32_t start_idx = binary_search_lower_bound(morton_sorted, total_nodes, min_morton);

    // Collect all nodes in range [min_morton, max_morton]
    uint32_t output_count = 0;

    for (uint32_t i = start_idx; i < total_nodes && output_count < max_results; ++i) {
        uint32_t morton = morton_sorted[i];

        if (morton > max_morton) break;  // Exceeded range

        // In range - store node ID
        result_buffer[output_count] = node_ids_sorted[i];
        output_count++;
    }

    // Write result count
    *result_count = output_count;
}

/**
 * Euclidean distance post-filter (optional refinement).
 *
 * Morton code queries return a bounding cube. This kernel refines the result
 * to an exact sphere by computing Euclidean distances.
 *
 * @param positions Node positions
 * @param candidate_ids Candidate node IDs from Morton query
 * @param candidate_count Number of candidates
 * @param query_center Query center (x,y,z)
 * @param query_radius Euclidean radius
 * @param refined_buffer Output: refined node IDs
 * @param refined_count Output: refined count
 */
extern "C" __global__ void refine_query_euclidean(
    const float* __restrict__ positions,
    const uint32_t* __restrict__ candidate_ids,
    uint32_t candidate_count,
    float query_x,
    float query_y,
    float query_z,
    float query_radius,
    uint32_t* __restrict__ refined_buffer,
    uint32_t* __restrict__ refined_count,
    uint32_t max_results
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= candidate_count) return;

    uint32_t node_id = candidate_ids[idx];

    // Load node position
    float x = positions[node_id * 3 + 0];
    float y = positions[node_id * 3 + 1];
    float z = positions[node_id * 3 + 2];

    // Compute distance
    float dx = x - query_x;
    float dy = y - query_y;
    float dz = z - query_z;
    float dist_sq = dx*dx + dy*dy + dz*dz;
    float radius_sq = query_radius * query_radius;

    // Check if within radius
    if (dist_sq <= radius_sq) {
        // Atomic append to output
        uint32_t pos = atomicAdd_u32(refined_count, 1);

        if (pos < max_results) {
            refined_buffer[pos] = node_id;
        }
    }
}
#include <math.h>
