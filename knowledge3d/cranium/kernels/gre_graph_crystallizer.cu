// Graph Crystallizer - Grok's Recursive GNN
// Aggregates neighbor contributions with EMA for stability
// Leverages RPN-style EMA computation
//
// Based on: Step8 Graph Crystallizer concept
// Integration: Uses RPN EMA pattern from TRM (node * (1-rate) + neighbor * rate)

extern "C" __global__ void gre_graph_crystallizer(
    const float* __restrict__ node_ptr,      // Current node values
    const float* __restrict__ neighbor_ptr,  // Aggregated neighbor values
    float* __restrict__ output_ptr,          // Output node values
    unsigned int node_count,
    float ema_rate                           // EMA rate (0.999 for TRM stability)
)
{
    // Get global thread ID
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Pre-compute EMA factors (RPN-style constant folding)
    float inv_rate = 1.0f - ema_rate;

    // Each thread processes multiple nodes via striding
    // RPN equivalent (EMA update): node inv_rate mul neighbor rate mul add
    for (unsigned int i = idx; i < node_count; i += stride) {
        float node = node_ptr[i];
        float neighbor = neighbor_ptr[i];
        output_ptr[i] = node * inv_rate + neighbor * ema_rate;
    }
}
