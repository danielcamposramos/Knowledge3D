// Atomic Fission/Fusion - GLM's Atom Operations
// Applies fusion (compress) or fission (expand) scaling to atoms
// Leverages RPN-style conditional arithmetic
//
// Based on: Step8 Atomic Fission/Fusion concept
// Integration: Uses RPN conditional operations and scalar arithmetic

extern "C" __global__ void gre_atomic_fission_fusion(
    const float* __restrict__ input_ptr,
    float* __restrict__ output_ptr,
    unsigned int count,
    unsigned int mode,    // 0=fusion, 1=fission
    float ratio
)
{
    // Get global thread ID
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Each thread processes multiple elements via striding
    for (unsigned int i = idx; i < count; i += stride) {
        float value = input_ptr[i];
        float result;

        // RPN equivalent: mode 0 EQ IF value ratio mul ELSE value ratio div ENDIF
        if (mode == 0) {
            // Fusion: compress values by ratio
            result = value * ratio;
        } else {
            // Fission: expand values (distribute mass)
            result = value / ratio;
        }

        output_ptr[i] = result;
    }
}
