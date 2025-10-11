// Temporal Reasoning - GLM's Sequential Reasoning
// Computes frame-to-frame deltas for temporal sequences
// Leverages RPN-style sequential operations
//
// Based on: Step8 Temporal Reasoning concept
// Integration: Uses RPN subtraction for delta computation

extern "C" __global__ void gre_temporal_reasoning(
    const float* __restrict__ sequence_ptr,  // [sequence_length * feature_dim]
    float* __restrict__ output_ptr,          // [sequence_length * feature_dim]
    unsigned int sequence_length,
    unsigned int feature_dim
)
{
    // Each thread handles one feature across all sequence steps
    unsigned int feat_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (feat_idx >= feature_dim) return;

    // Process each timestep
    for (unsigned int t = 0; t < sequence_length; t++) {
        unsigned int curr_idx = t * feature_dim + feat_idx;
        float curr = sequence_ptr[curr_idx];

        // Compute delta (current - previous)
        // RPN equivalent: curr next sub
        float delta;
        if (t + 1 < sequence_length) {
            unsigned int next_idx = (t + 1) * feature_dim + feat_idx;
            float next = sequence_ptr[next_idx];
            delta = next - curr;
        } else {
            // Last frame: delta is zero
            delta = 0.0f;
        }

        // Store delta
        output_ptr[curr_idx] = delta;
    }
}
