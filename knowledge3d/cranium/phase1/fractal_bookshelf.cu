#include <cuda_runtime.h>

// Generate book center positions arranged along shelves using a simple
// low-discrepancy sequence (golden ratio) to produce visually balanced placement.
//
// shelves: number of shelf rows (Y levels)
// per_shelf: number of books per shelf
// spacing: nominal spacing between books along X
// shelf_pitch: vertical spacing between shelves along Y
// depth: Z offset for shelf plane
// out_positions: float3 array (size shelves*per_shelf)

extern "C" __global__ void generate_fractal_shelf(
    int shelves,
    int per_shelf,
    float spacing,
    float shelf_pitch,
    float depth,
    float jitter,
    float* __restrict__ out_positions
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = shelves * per_shelf;
    if (idx >= total) return;

    // Map 1D idx to shelf,row
    int s = idx / per_shelf;   // shelf index
    int i = idx % per_shelf;   // position within shelf

    // Golden ratio for low-discrepancy scattering
    const float phi = 1.61803398875f;
    float t = (i + 0.5f) / (float)per_shelf;
    float ld = fmodf(t * phi, 1.0f); // in [0,1)

    // Center shelves around X=0
    float x = (i - (per_shelf - 1) * 0.5f) * spacing;
    // Apply small low-discrepancy jitter to avoid perfect grid
    x += (ld - 0.5f) * jitter * spacing;

    float y = (s - (shelves - 1) * 0.5f) * shelf_pitch;
    float z = depth;

    out_positions[3 * idx + 0] = x;
    out_positions[3 * idx + 1] = y;
    out_positions[3 * idx + 2] = z;
}

