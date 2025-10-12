// Extended Modular RPN Kernel with sparse operations
#include <cstdint>

#define OP_SPARSE_LOAD 0x28
#define OP_SMAV 0x29
#define OP_ENTROPY_SUM 0x2A

extern "C" __global__ void modular_rpn_kernel(
    float* stack, 
    uint32_t* program, 
    uint32_t program_size,
    float** inputs,
    uint32_t num_inputs,
    float* output,
    uint32_t* sparse_indices,
    float* sparse_values,
    uint32_t* sparse_nnz
) {
    extern __shared__ float shared_mem[];
    uint32_t* shared_indices = (uint32_t*)shared_mem;
    float* shared_values = (float*)(shared_mem + SPARSE_MAX_NNZ);
    
    uint32_t pc = 0;
    uint32_t sp = 0;
    float entropy_acc = 0.0f;
    
    while (pc < program_size) {
        uint32_t opcode = program[pc++];
        
        switch (opcode) {
            case OP_SPARSE_LOAD: {
                // Load sparse matrix data to shared memory
                uint32_t matrix_id = program[pc++];
                uint32_t nnz = sparse_nnz[matrix_id];
                
                // Cooperative load of indices and values
                for (int i = threadIdx.x; i < nnz; i += blockDim.x) {
                    shared_indices[i] = sparse_indices[matrix_id * SPARSE_MAX_NNZ + i];
                    shared_values[i] = sparse_values[matrix_id * SPARSE_MAX_NNZ + i];
                }
                __syncthreads();
                break;
            }
            
            case OP_SMAV: {
                // Sparse matrix-vector multiplication
                float* input_vec = inputs[0]; // Assuming first input is vector
                float partial_sum = 0.0f;
                uint32_t nnz = sparse_nnz[0]; // Assuming current matrix
                
                // Warp-parallel sparse matvec
                for (int i = threadIdx.x; i < nnz; i += blockDim.x) {
                    uint32_t col = shared_indices[i];
                    float val = shared_values[i] * input_vec[col];
                    partial_sum += val;
                }
                
                // Warp reduction
                for (int offset = 16; offset > 0; offset /= 2) {
                    partial_sum += __shfl_down_sync(0xFFFFFFFF, partial_sum, offset);
                }
                
                if (threadIdx.x % 32 == 0) {
                    stack[sp++] = partial_sum;
                }
                __syncthreads();
                break;
            }
            
            case OP_ENTROPY_SUM: {
                // p * log(p) entropy calculation
                float p = stack[--sp];
                float term = p * logf(p + 1e-6f);
                
                // Atomic add to global entropy accumulator
                atomicAdd(&entropy_acc, term);
                break;
            }
            
            // Existing opcodes remain unchanged
            case 0x0A: { // MAX
                float b = stack[--sp];
                float a = stack[--sp];
                stack[sp++] = fmaxf(a, b);
                break;
            }
            
            case 0x0B: { // SIGMOID_APPROX (tanh-based)
                float x = stack[--sp];
                stack[sp++] = 0.5f * (1.0f + tanhf(0.5f * x));
                break;
            }
            
            case 0x12: { // MUL
                float b = stack[--sp];
                float a = stack[--sp];
                stack[sp++] = a * b;
                break;
            }
            
            case 0x06: { // DUP
                float a = stack[sp - 1];
                stack[sp++] = a;
                break;
            }
            
            // ... other existing opcodes
        }
    }
    
    // Store final entropy if calculated
    if (entropy_acc != 0.0f && threadIdx.x == 0) {
        output[0] = -entropy_acc; // Negative because we summed p*log(p)
    }
}
