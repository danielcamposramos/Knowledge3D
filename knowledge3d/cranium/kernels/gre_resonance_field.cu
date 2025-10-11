// Resonance Field - GLM's Energetic Field Management
// Computes resonance strengths from positions and density
// Leverages RPN-style vector magnitude computation
//
// Based on: Step8 Resonance Field concept
// Integration: Uses RPN geometric operations (dot product, magnitude)

extern "C" __global__ void gre_resonance_field(
    const float* __restrict__ positions_ptr,  // [x,y,z] * count
    const float* __restrict__ density_ptr,    // density values
    float* __restrict__ output_ptr,           // resonance strengths
    unsigned int count
)
{
    // Get global thread ID
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Each thread processes multiple nodes via striding
    for (unsigned int i = idx; i < count; i += stride) {
        // Load position components
        unsigned int pos_idx = i * 3;
        float x = positions_ptr[pos_idx + 0];
        float y = positions_ptr[pos_idx + 1];
        float z = positions_ptr[pos_idx + 2];

        // Load density
        float density = density_ptr[i];

        // Compute resonance strength using RPN-style magnitude
        // RPN equivalent: x DUP mul y DUP mul add z DUP mul add sqrt density mul
        float mag_sq = x * x + y * y + z * z;
        float magnitude = sqrtf(mag_sq);
        float strength = magnitude * density;

        // Store result
        output_ptr[i] = strength;
    }
}
