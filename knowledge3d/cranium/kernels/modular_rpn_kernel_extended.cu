#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

namespace {
constexpr int kStackCapacity = 64;
constexpr int kMatrixMaxDim = 3;

enum class ItemType : uint8_t {
    kScalar = 0,
    kVector = 1,
    kMatrixRow = 2,
    kTensor = 3,
};

constexpr uint32_t kErrorNone = 0;
constexpr uint32_t kErrorUnknownOpcode = 9001;
constexpr uint32_t kErrorStackUnderflow = 9002;
constexpr uint32_t kErrorStackOverflow = 9003;
constexpr uint32_t kErrorInvalidMatrixDims = 9013;
constexpr uint32_t kErrorSingularMatrix = 9014;
constexpr uint16_t kOpMemcpyF32 = 0x90;
constexpr uint16_t kOpFillF32 = 0x91;
constexpr uint16_t kOpReduceSumF32 = 0x92;
constexpr uint16_t kOpReduceMaxF32 = 0x93;
constexpr uint16_t kOpReduceMinF32 = 0x94;
constexpr uint16_t kOpMatVecF32 = 0xA0;
constexpr uint16_t kOpVectorRelu = 0xA1;
constexpr uint16_t kOpVectorMulF32 = 0xA2;
constexpr uint16_t kOpVectorSigmoid = 0xA3;
constexpr uint16_t kOpEntropySum = 0x42;
constexpr uint16_t kOpTemporalCoherence = 0xF0;
constexpr uint16_t kOpTemporalMask = 0xF1;
constexpr uint16_t kOpTemporalAggregate = 0xF2;

struct StackItem {
    float value[4];
    ItemType type;
    int rows;
    int cols;
    int row_index;
};

struct alignas(16) InstanceState {
    uint32_t head;
    uint32_t size;
    uint32_t error;
    uint32_t reserved;
    float stack[kStackCapacity][4];
};

static_assert(sizeof(InstanceState) == 1040, "InstanceState layout mismatch");

__device__ inline uint8_t clamp_meta_dim(int value) {
    if (value < 0) {
        return 0;
    }
    if (value > 255) {
        return 255;
    }
    return static_cast<uint8_t>(value);
}

__device__ inline float pack_meta(ItemType type, int rows, int cols, int row_index) {
    uint32_t bits = static_cast<uint32_t>(type) |
                    (static_cast<uint32_t>(clamp_meta_dim(rows)) << 8) |
                    (static_cast<uint32_t>(clamp_meta_dim(cols)) << 16) |
                    (static_cast<uint32_t>(clamp_meta_dim(row_index)) << 24);
    return __uint_as_float(bits);
}

__device__ inline bool push_item(StackItem* stack, uint32_t& size, const StackItem& item, uint32_t& error) {
    if (size >= kStackCapacity) {
        error = kErrorStackOverflow;
        return false;
    }
    stack[size] = item;
    size += 1;
    return true;
}

__device__ inline bool pop_item(StackItem* stack, uint32_t& size, StackItem& out, uint32_t& error) {
    if (size == 0) {
        error = kErrorStackUnderflow;
        return false;
    }
    size -= 1;
    out = stack[size];
    return true;
}

__device__ inline bool pop_scalar(StackItem* stack, uint32_t& size, float& value, uint32_t& error) {
    StackItem item{};
    if (!pop_item(stack, size, item, error)) {
        return false;
    }
    if (item.type != ItemType::kScalar) {
        error = kErrorUnknownOpcode;
        return false;
    }
    value = item.value[0];
    return true;
}

__device__ inline void encode_pointer(StackItem& item, float* ptr) {
    uint64_t raw = reinterpret_cast<uint64_t>(ptr);
    uint32_t lo = static_cast<uint32_t>(raw & 0xFFFFFFFFull);
    uint32_t hi = static_cast<uint32_t>(raw >> 32);
    item.value[0] = __uint_as_float(lo);
    item.value[1] = __uint_as_float(hi);
}

__device__ inline float* decode_pointer(const StackItem& item) {
    uint32_t lo = __float_as_uint(item.value[0]);
    uint32_t hi = __float_as_uint(item.value[1]);
    uint64_t raw = (static_cast<uint64_t>(hi) << 32) | lo;
    return reinterpret_cast<float*>(raw);
}

struct TensorRef {
    float* ptr;
    int rows;
    int cols;
};

__device__ inline bool push_tensor(StackItem* stack, uint32_t& size, float* ptr, int rows, int cols, uint32_t& error) {
    if (ptr == nullptr) {
        error = kErrorUnknownOpcode;
        return false;
    }
    StackItem item{};
    encode_pointer(item, ptr);
    item.value[2] = 0.0f;
    item.value[3] = 0.0f;
    item.type = ItemType::kTensor;
    item.rows = rows;
    item.cols = cols;
    item.row_index = 0;
    return push_item(stack, size, item, error);
}

__device__ inline bool pop_tensor(StackItem* stack, uint32_t& size, TensorRef& tensor, uint32_t& error) {
    StackItem item{};
    if (!pop_item(stack, size, item, error)) {
        return false;
    }
    if (item.type != ItemType::kTensor) {
        error = kErrorUnknownOpcode;
        return false;
    }
    tensor.ptr = decode_pointer(item);
    tensor.rows = item.rows;
    tensor.cols = item.cols;
    return true;
}

struct Matrix {
    int rows;
    int cols;
    float data[kMatrixMaxDim * kMatrixMaxDim];
};

__device__ inline bool pop_matrix(StackItem* stack, uint32_t& size, Matrix& matrix, uint32_t& error) {
    if (size == 0) {
        error = kErrorStackUnderflow;
        return false;
    }

    StackItem top = stack[size - 1];
    if (top.type != ItemType::kMatrixRow) {
        error = kErrorUnknownOpcode;
        return false;
    }

    const int rows = static_cast<int>(top.rows);
    const int cols = static_cast<int>(top.cols);

    if (rows <= 0 || cols <= 0 || rows > kMatrixMaxDim || cols > kMatrixMaxDim) {
        error = kErrorInvalidMatrixDims;
        return false;
    }
    if (size < static_cast<uint32_t>(rows)) {
        error = kErrorStackUnderflow;
        return false;
    }

    matrix.rows = rows;
    matrix.cols = cols;

    for (int r = rows - 1; r >= 0; --r) {
        StackItem row_item{};
        if (!pop_item(stack, size, row_item, error)) {
            return false;
        }
        if (row_item.type != ItemType::kMatrixRow ||
            row_item.rows != top.rows ||
            row_item.cols != top.cols) {
            error = kErrorUnknownOpcode;
            return false;
        }
        int base = r * cols;
        matrix.data[base + 0] = row_item.value[0];
        if (cols > 1) {
            matrix.data[base + 1] = row_item.value[1];
        }
        if (cols > 2) {
            matrix.data[base + 2] = row_item.value[2];
        }
    }
    return true;
}

__device__ inline bool push_matrix(StackItem* stack, uint32_t& size, const Matrix& matrix, uint32_t& error) {
    if (matrix.rows <= 0 || matrix.cols <= 0 ||
        matrix.rows > kMatrixMaxDim || matrix.cols > kMatrixMaxDim) {
        error = kErrorInvalidMatrixDims;
        return false;
    }
    if (size + static_cast<uint32_t>(matrix.rows) > kStackCapacity) {
        error = kErrorStackOverflow;
        return false;
    }

    for (int r = 0; r < matrix.rows; ++r) {
        StackItem item{};
        int base = r * matrix.cols;
        item.value[0] = matrix.data[base + 0];
        item.value[1] = (matrix.cols > 1) ? matrix.data[base + 1] : 0.0f;
        item.value[2] = (matrix.cols > 2) ? matrix.data[base + 2] : 0.0f;
        item.value[3] = 0.0f;
        item.type = ItemType::kMatrixRow;
        item.rows = static_cast<uint8_t>(matrix.rows);
        item.cols = static_cast<uint8_t>(matrix.cols);
        item.row_index = static_cast<uint8_t>(r);
        if (!push_item(stack, size, item, error)) {
            return false;
        }
    }
    return true;
}

__device__ inline bool push_scalar(StackItem* stack, uint32_t& size, float value, uint32_t& error) {
    StackItem item{};
    item.value[0] = value;
    item.value[1] = 0.0f;
    item.value[2] = 0.0f;
    item.value[3] = 0.0f;
    item.type = ItemType::kScalar;
    item.rows = 1;
    item.cols = 1;
    item.row_index = 0;
    return push_item(stack, size, item, error);
}

__device__ inline bool push_vector(StackItem* stack, uint32_t& size, const float* vectors, uint32_t index, uint32_t& error) {
    StackItem item{};
    if (vectors) {
        item.value[0] = vectors[index * 3 + 0];
        item.value[1] = vectors[index * 3 + 1];
        item.value[2] = vectors[index * 3 + 2];
    } else {
        item.value[0] = 0.0f;
        item.value[1] = 0.0f;
        item.value[2] = 0.0f;
    }
    item.value[3] = 0.0f;
    item.type = ItemType::kVector;
    item.rows = 1;
    item.cols = 3;
    item.row_index = 0;
    return push_item(stack, size, item, error);
}

__device__ inline float determinant_2x2(const Matrix& m) {
    return m.data[0] * m.data[3] - m.data[1] * m.data[2];
}

__device__ inline float determinant_3x3(const Matrix& m) {
    const float a = m.data[0];
    const float b = m.data[1];
    const float c = m.data[2];
    const float d = m.data[3];
    const float e = m.data[4];
    const float f = m.data[5];
    const float g = m.data[6];
    const float h = m.data[7];
    const float i = m.data[8];
    return a * (e * i - f * h) -
           b * (d * i - f * g) +
           c * (d * h - e * g);
}

__device__ inline bool inverse_2x2(const Matrix& input, Matrix& output) {
    const float det = determinant_2x2(input);
    if (fabsf(det) < 1e-8f) {
        return false;
    }
    const float inv_det = 1.0f / det;
    output.rows = 2;
    output.cols = 2;
    output.data[0] = input.data[3] * inv_det;
    output.data[1] = -input.data[1] * inv_det;
    output.data[2] = -input.data[2] * inv_det;
    output.data[3] = input.data[0] * inv_det;
    return true;
}

__device__ inline bool inverse_3x3(const Matrix& input, Matrix& output) {
    const float det = determinant_3x3(input);
    if (fabsf(det) < 1e-8f) {
        return false;
    }
    output.rows = 3;
    output.cols = 3;

    const float inv_det = 1.0f / det;

    output.data[0] = (input.data[4] * input.data[8] - input.data[5] * input.data[7]) * inv_det;
    output.data[1] = (input.data[2] * input.data[7] - input.data[1] * input.data[8]) * inv_det;
    output.data[2] = (input.data[1] * input.data[5] - input.data[2] * input.data[4]) * inv_det;

    output.data[3] = (input.data[5] * input.data[6] - input.data[3] * input.data[8]) * inv_det;
    output.data[4] = (input.data[0] * input.data[8] - input.data[2] * input.data[6]) * inv_det;
    output.data[5] = (input.data[2] * input.data[3] - input.data[0] * input.data[5]) * inv_det;

    output.data[6] = (input.data[3] * input.data[7] - input.data[4] * input.data[6]) * inv_det;
    output.data[7] = (input.data[1] * input.data[6] - input.data[0] * input.data[7]) * inv_det;
    output.data[8] = (input.data[0] * input.data[4] - input.data[1] * input.data[3]) * inv_det;

    return true;
}

__device__ inline void matmul(const Matrix& a, const Matrix& b, Matrix& out) {
    out.rows = a.rows;
    out.cols = b.cols;
    for (int r = 0; r < out.rows; ++r) {
        for (int c = 0; c < out.cols; ++c) {
            float acc = 0.0f;
            for (int k = 0; k < a.cols; ++k) {
                acc += a.data[r * a.cols + k] * b.data[k * b.cols + c];
            }
            out.data[r * out.cols + c] = acc;
        }
    }
}

__device__ inline void transpose(const Matrix& m, Matrix& out) {
    out.rows = m.cols;
    out.cols = m.rows;
    for (int r = 0; r < out.rows; ++r) {
        for (int c = 0; c < out.cols; ++c) {
            out.data[r * out.cols + c] = m.data[c * m.cols + r];
        }
    }
}
}  // namespace

extern "C" __global__ void modular_rpn_kernel_extended(
    uint32_t instance_id,
    const uint16_t* __restrict__ op_codes,
    const float* __restrict__ scalars,
    const float* __restrict__ vectors,
    const float* __restrict__ matrices,
    InstanceState* __restrict__ states,
    uint32_t token_count) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    InstanceState* state = reinterpret_cast<InstanceState*>(
        reinterpret_cast<uint8_t*>(states) + instance_id * sizeof(InstanceState));

    __shared__ StackItem stack[kStackCapacity];
    __shared__ uint32_t stack_size;
    __shared__ uint32_t error_code;

    __shared__ uint32_t scalar_index;
    __shared__ uint32_t vector_index;
    __shared__ uint32_t matrix_index;

    __shared__ TensorRef tensor_a;
    __shared__ TensorRef tensor_b;
    __shared__ TensorRef tensor_c;
    __shared__ TensorRef tensor_d;
    __shared__ int shared_rows;
    __shared__ int shared_length;
    __shared__ int shared_inner;
    __shared__ float shared_scalar;
    __shared__ float reduction_buffer[256];
    __shared__ float shared_vector_cache[1024];

    if (tid == 0) {
        stack_size = 0;
        error_code = kErrorNone;
        scalar_index = 0;
        vector_index = 0;
        matrix_index = 0;
    }
    __syncthreads();

    for (uint32_t i = 0; i < token_count; ++i) {
        __syncthreads();
        if (error_code != kErrorNone) {
            break;
        }

        const uint16_t opcode = op_codes[i];

        if (opcode == kOpMemcpyF32) {
            bool ok = true;
            if (tid == 0) {
                ok = pop_tensor(stack, stack_size, tensor_b, error_code);  // src
                if (ok) ok = pop_tensor(stack, stack_size, tensor_a, error_code);  // dest
                if (ok) {
                    const int len_dst = tensor_a.rows * tensor_a.cols;
                    const int len_src = tensor_b.rows * tensor_b.cols;
                    if (len_dst != len_src || len_dst <= 0) {
                        error_code = kErrorInvalidMatrixDims;
                    } else {
                        shared_length = len_dst;
                    }
                }
            }
            __syncthreads();
            if (error_code != kErrorNone) {
                continue;
            }
            float* dst_ptr = tensor_a.ptr;
            const float* src_ptr = tensor_b.ptr;
            for (int idx = tid; idx < shared_length; idx += stride) {
                dst_ptr[idx] = src_ptr[idx];
            }
            __syncthreads();
            if (tid == 0) {
                push_tensor(stack, stack_size, dst_ptr, tensor_a.rows, tensor_a.cols, error_code);
            }
            continue;
        }

        if (opcode == kOpFillF32) {
            bool ok = true;
            if (tid == 0) {
                ok = pop_scalar(stack, stack_size, shared_scalar, error_code);
                if (ok) ok = pop_tensor(stack, stack_size, tensor_a, error_code);
                if (ok) {
                    shared_length = tensor_a.rows * tensor_a.cols;
                    if (shared_length <= 0) {
                        error_code = kErrorInvalidMatrixDims;
                    }
                }
            }
            __syncthreads();
            if (error_code != kErrorNone) {
                continue;
            }
            float* dst_ptr = tensor_a.ptr;
            for (int idx = tid; idx < shared_length; idx += stride) {
                dst_ptr[idx] = shared_scalar;
            }
            __syncthreads();
            if (tid == 0) {
                push_tensor(stack, stack_size, dst_ptr, tensor_a.rows, tensor_a.cols, error_code);
            }
            continue;
        }

        if (opcode == kOpReduceSumF32 || opcode == kOpReduceMaxF32 || opcode == kOpReduceMinF32) {
            bool ok = true;
            if (tid == 0) {
                ok = pop_tensor(stack, stack_size, tensor_a, error_code);
                if (ok) {
                    shared_length = tensor_a.rows * tensor_a.cols;
                    if (shared_length <= 0) {
                        error_code = kErrorInvalidMatrixDims;
                    }
                }
            }
            __syncthreads();
            if (error_code != kErrorNone) {
                continue;
            }
            const float* src_ptr = tensor_a.ptr;
            float thread_value = 0.0f;
            bool has_value = false;

            if (opcode == kOpReduceMaxF32) {
                thread_value = -FLT_MAX;
            } else if (opcode == kOpReduceMinF32) {
                thread_value = FLT_MAX;
            }

            for (int idx = tid; idx < shared_length; idx += stride) {
                const float v = src_ptr[idx];
                if (opcode == kOpReduceSumF32) {
                    thread_value += v;
                } else if (opcode == kOpReduceMaxF32) {
                    thread_value = has_value ? fmaxf(thread_value, v) : v;
                } else {
                    thread_value = has_value ? fminf(thread_value, v) : v;
                }
                has_value = true;
            }

            if (!has_value) {
                if (opcode == kOpReduceMaxF32) {
                    thread_value = -FLT_MAX;
                } else if (opcode == kOpReduceMinF32) {
                    thread_value = FLT_MAX;
                } else {
                    thread_value = 0.0f;
                }
            }

            reduction_buffer[tid] = thread_value;
            __syncthreads();

            if (tid == 0) {
                float result = 0.0f;
                if (opcode == kOpReduceMaxF32) {
                    result = -FLT_MAX;
                    for (int lane = 0; lane < blockDim.x; ++lane) {
                        result = fmaxf(result, reduction_buffer[lane]);
                    }
                } else if (opcode == kOpReduceMinF32) {
                    result = FLT_MAX;
                    for (int lane = 0; lane < blockDim.x; ++lane) {
                        result = fminf(result, reduction_buffer[lane]);
                    }
                } else {
                    result = 0.0f;
                    for (int lane = 0; lane < blockDim.x; ++lane) {
                        result += reduction_buffer[lane];
                    }
                }
                push_scalar(stack, stack_size, result, error_code);
            }
            __syncthreads();
            continue;
        }

        if (opcode == kOpMatVecF32) {
            bool ok = true;
            if (tid == 0) {
                ok = pop_tensor(stack, stack_size, tensor_c, error_code);  // vector
                if (ok) ok = pop_tensor(stack, stack_size, tensor_b, error_code);  // matrix
                if (ok) ok = pop_tensor(stack, stack_size, tensor_a, error_code);  // dest
                if (ok) {
                    shared_rows = tensor_b.rows;
                    shared_inner = tensor_b.cols;
                    if (tensor_c.rows != shared_inner || tensor_a.rows != shared_rows ||
                        tensor_a.cols != 1 || tensor_c.cols != 1) {
                        error_code = kErrorInvalidMatrixDims;
                    }
                }
            }
            __syncthreads();
            if (error_code != kErrorNone) {
                continue;
            }
            float* dst_ptr = tensor_a.ptr;
            const float* matrix_ptr = tensor_b.ptr;
            const float* vec_ptr = tensor_c.ptr;
            const int cols = shared_inner;

            if (cols <= 1024) {
                for (int col = tid; col < cols; col += stride) {
                    shared_vector_cache[col] = vec_ptr[col];
                }
                __syncthreads();
            }

            const int warp_size = 32;
            const int lane = tid & (warp_size - 1);
            const int warp_id = tid / warp_size;
            const int warp_count = blockDim.x / warp_size;

            for (int row = warp_id; row < shared_rows; row += warp_count) {
                const float* row_ptr = matrix_ptr + row * cols;
                float partial = 0.0f;
                if (cols <= 1024) {
                    for (int k = lane; k < cols; k += warp_size) {
                        partial += row_ptr[k] * shared_vector_cache[k];
                    }
                } else {
                    for (int k = lane; k < cols; k += warp_size) {
                        partial += row_ptr[k] * vec_ptr[k];
                    }
                }
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    partial += __shfl_down_sync(0xffffffff, partial, offset);
                }
                if (lane == 0) {
                    dst_ptr[row] = partial;
                }
            }
            __syncthreads();
            if (tid == 0) {
                push_tensor(stack, stack_size, dst_ptr, tensor_a.rows, tensor_a.cols, error_code);
            }
            continue;
        }

        if (opcode == kOpVectorRelu) {
            bool ok = true;
            if (tid == 0) {
                ok = pop_tensor(stack, stack_size, tensor_a, error_code);
                if (ok) {
                    shared_length = tensor_a.rows * tensor_a.cols;
                    if (shared_length <= 0) {
                        error_code = kErrorInvalidMatrixDims;
                    }
                }
            }
            __syncthreads();
            if (error_code != kErrorNone) {
                continue;
            }
            float* data = tensor_a.ptr;
            for (int idx = tid; idx < shared_length; idx += stride) {
                float v = data[idx];
                data[idx] = v > 0.0f ? v : 0.0f;
            }
            __syncthreads();
            if (tid == 0) {
                push_tensor(stack, stack_size, data, tensor_a.rows, tensor_a.cols, error_code);
            }
            continue;
        }

        if (opcode == kOpVectorMulF32) {
            bool ok = true;
            if (tid == 0) {
                ok = pop_tensor(stack, stack_size, tensor_b, error_code);  // multiplier
                if (ok) ok = pop_tensor(stack, stack_size, tensor_a, error_code);  // dest
                if (ok) {
                    const int len_a = tensor_a.rows * tensor_a.cols;
                    const int len_b = tensor_b.rows * tensor_b.cols;
                    if (len_a != len_b || len_a <= 0) {
                        error_code = kErrorInvalidMatrixDims;
                    } else {
                        shared_length = len_a;
                    }
                }
            }
            __syncthreads();
            if (error_code != kErrorNone) {
                continue;
            }
            float* dst_ptr = tensor_a.ptr;
            const float* mul_ptr = tensor_b.ptr;
            for (int idx = tid; idx < shared_length; idx += stride) {
                dst_ptr[idx] *= mul_ptr[idx];
            }
            __syncthreads();
            if (tid == 0) {
                push_tensor(stack, stack_size, dst_ptr, tensor_a.rows, tensor_a.cols, error_code);
            }
            continue;
        }

        if (opcode == kOpVectorSigmoid) {
            bool ok = true;
            if (tid == 0) {
                ok = pop_tensor(stack, stack_size, tensor_a, error_code);
                if (ok) {
                    shared_length = tensor_a.rows * tensor_a.cols;
                    if (shared_length <= 0) {
                        error_code = kErrorInvalidMatrixDims;
                    }
                }
            }
            __syncthreads();
            if (error_code != kErrorNone) {
                continue;
            }
            float* data = tensor_a.ptr;
            for (int idx = tid; idx < shared_length; idx += stride) {
                float x = data[idx];
                float sig = 1.0f / (1.0f + expf(-x));
                data[idx] = sig;
            }
            __syncthreads();
            if (tid == 0) {
                push_tensor(stack, stack_size, data, tensor_a.rows, tensor_a.cols, error_code);
            }
            continue;
        }

        if (opcode == kOpEntropySum) {
            bool ok = true;
            if (tid == 0) {
                ok = pop_tensor(stack, stack_size, tensor_a, error_code);
                if (ok) {
                    shared_length = tensor_a.rows * tensor_a.cols;
                    if (shared_length <= 0) {
                        error_code = kErrorInvalidMatrixDims;
                    }
                }
            }
            __syncthreads();
            if (error_code != kErrorNone) {
                continue;
            }
            const float* data = tensor_a.ptr;
            float thread_sum = 0.0f;
            for (int idx = tid; idx < shared_length; idx += stride) {
                float p = data[idx];
                p = fmaxf(p, 1e-6f);
                thread_sum += -p * logf(p);
            }
            reduction_buffer[tid] = thread_sum;
            __syncthreads();
            if (tid == 0) {
                float total = 0.0f;
                for (int lane = 0; lane < blockDim.x; ++lane) {
                    total += reduction_buffer[lane];
                }
                push_scalar(stack, stack_size, total, error_code);
            }
            __syncthreads();
            continue;
        }

        if (opcode == kOpTemporalCoherence) {
            bool ok = true;
            if (tid == 0) {
                ok = pop_tensor(stack, stack_size, tensor_b, error_code);  // context (T, D)
                if (ok) ok = pop_tensor(stack, stack_size, tensor_a, error_code);  // dest (D, 1)
                if (ok) {
                    shared_rows = tensor_b.rows;   // time steps
                    shared_length = tensor_b.cols; // feature dim
                    if (shared_rows <= 0 || shared_length <= 0 ||
                        tensor_a.rows != shared_length || tensor_a.cols != 1) {
                        error_code = kErrorInvalidMatrixDims;
                    }
                }
            }
            __syncthreads();
            if (error_code != kErrorNone) {
                continue;
            }
            const float* context_ptr = tensor_b.ptr;
            float* coherence_ptr = tensor_a.ptr;
            const int time_steps = shared_rows;
            const int feature_dim = shared_length;
            const float inv_time = time_steps > 0 ? 1.0f / static_cast<float>(time_steps) : 0.0f;
            for (int feature = tid; feature < feature_dim; feature += stride) {
                float sum = 0.0f;
                float sq_sum = 0.0f;
                for (int t = 0; t < time_steps; ++t) {
                    const float value = context_ptr[t * feature_dim + feature];
                    sum += value;
                    sq_sum += value * value;
                }
                const float mean = sum * inv_time;
                float variance = sq_sum * inv_time - mean * mean;
                if (variance < 0.0f) {
                    variance = 0.0f;
                }
                const float coherence = 1.0f / (1.0f + sqrtf(variance + 1e-8f));
                coherence_ptr[feature] = coherence;
            }
            __syncthreads();
            if (tid == 0) {
                push_tensor(stack, stack_size, coherence_ptr, tensor_a.rows, tensor_a.cols, error_code);
            }
            continue;
        }

        if (opcode == kOpTemporalMask) {
            bool ok = true;
            if (tid == 0) {
                ok = pop_scalar(stack, stack_size, shared_scalar, error_code);  // threshold
                if (ok) ok = pop_tensor(stack, stack_size, tensor_b, error_code);  // coherence scores
                if (ok) ok = pop_tensor(stack, stack_size, tensor_a, error_code);  // dest mask
                if (ok) {
                    shared_length = tensor_b.rows * tensor_b.cols;
                    if (tensor_a.rows * tensor_a.cols != shared_length || shared_length <= 0) {
                        error_code = kErrorInvalidMatrixDims;
                    }
                }
            }
            __syncthreads();
            if (error_code != kErrorNone) {
                continue;
            }
            const float* coherence_ptr = tensor_b.ptr;
            float* mask_ptr = tensor_a.ptr;
            const float threshold = shared_scalar;
            for (int idx = tid; idx < shared_length; idx += stride) {
                const float score = coherence_ptr[idx];
                const float shifted = (score - threshold) * 4.0f;  // temperature scaling
                const float mask = 1.0f / (1.0f + expf(-shifted));
                mask_ptr[idx] = fminf(fmaxf(mask, 0.0f), 1.0f);
            }
            __syncthreads();
            if (tid == 0) {
                push_tensor(stack, stack_size, mask_ptr, tensor_a.rows, tensor_a.cols, error_code);
            }
            continue;
        }

        if (opcode == kOpTemporalAggregate) {
            bool ok = true;
            if (tid == 0) {
                ok = pop_tensor(stack, stack_size, tensor_b, error_code);  // context (T, D)
                if (ok) ok = pop_tensor(stack, stack_size, tensor_a, error_code);  // dest (D, 1)
                if (ok) {
                    shared_rows = tensor_b.rows;
                    shared_length = tensor_b.cols;
                    if (shared_rows <= 0 || shared_length <= 0 ||
                        tensor_a.rows != shared_length || tensor_a.cols != 1) {
                        error_code = kErrorInvalidMatrixDims;
                    }
                }
            }
            __syncthreads();
            if (error_code != kErrorNone) {
                continue;
            }
            const float* context_ptr = tensor_b.ptr;
            float* dest_ptr = tensor_a.ptr;
            const int time_steps = shared_rows;
            const int feature_dim = shared_length;
            const float inv_time = time_steps > 0 ? 1.0f / static_cast<float>(time_steps) : 0.0f;
            for (int feature = tid; feature < feature_dim; feature += stride) {
                float accum = 0.0f;
                for (int t = 0; t < time_steps; ++t) {
                    const float value = context_ptr[t * feature_dim + feature];
                    accum += fabsf(value);
                }
                dest_ptr[feature] = accum * inv_time;
            }
            __syncthreads();
            if (tid == 0) {
                push_tensor(stack, stack_size, dest_ptr, tensor_a.rows, tensor_a.cols, error_code);
            }
            continue;
        }

        if (opcode == 0x60) {  // MATVEC_512x1024
            bool ok = true;
            if (tid == 0) {
                ok = pop_tensor(stack, stack_size, tensor_a, error_code);
                if (ok) ok = pop_tensor(stack, stack_size, tensor_b, error_code);
                if (ok) ok = pop_tensor(stack, stack_size, tensor_c, error_code);
                if (ok) {
                    if (tensor_b.rows != 1024 || tensor_b.cols != 512 ||
                        tensor_c.rows != 512 || tensor_c.cols != 1 ||
                        tensor_a.rows != 1024 || tensor_a.cols != 1) {
                        error_code = kErrorInvalidMatrixDims;
                    }
                }
            }
            __syncthreads();
            if (error_code != kErrorNone) {
                continue;
            }
            float* out = tensor_a.ptr;
            const float* weights = tensor_b.ptr;
            const float* input_vec = tensor_c.ptr;
            for (int r = tid; r < 1024; r += stride) {
                const float* row_ptr = weights + r * 512;
                float sum = 0.0f;
                #pragma unroll 8
                for (int c = 0; c < 512; ++c) {
                    sum += row_ptr[c] * input_vec[c];
                }
                out[r] = sum;
            }
            __syncthreads();
            if (tid == 0) {
                push_tensor(stack, stack_size, out, 1024, 1, error_code);
            }
            continue;
        }

        if (opcode == 0x61) {  // MATVEC_1024x512
            bool ok = true;
            if (tid == 0) {
                ok = pop_tensor(stack, stack_size, tensor_a, error_code);
                if (ok) ok = pop_tensor(stack, stack_size, tensor_b, error_code);
                if (ok) ok = pop_tensor(stack, stack_size, tensor_c, error_code);
                if (ok) {
                    if (tensor_b.rows != 512 || tensor_b.cols != 1024 ||
                        tensor_c.rows != 1024 || tensor_c.cols != 1 ||
                        tensor_a.rows != 512 || tensor_a.cols != 1) {
                        error_code = kErrorInvalidMatrixDims;
                    }
                }
            }
            __syncthreads();
            if (error_code != kErrorNone) {
                continue;
            }
            float* out = tensor_a.ptr;
            const float* weights = tensor_b.ptr;
            const float* input_vec = tensor_c.ptr;
            for (int r = tid; r < 512; r += stride) {
                const float* row_ptr = weights + r * 1024;
                float sum = 0.0f;
                #pragma unroll 8
                for (int c = 0; c < 1024; ++c) {
                    sum += row_ptr[c] * input_vec[c];
                }
                out[r] = sum;
            }
            __syncthreads();
            if (tid == 0) {
                push_tensor(stack, stack_size, out, 512, 1, error_code);
            }
            continue;
        }

        if (opcode == 0x62) {  // VEC_ADD3 (dest, a, b, c)
            bool ok = true;
            if (tid == 0) {
                ok = pop_tensor(stack, stack_size, tensor_a, error_code);  // dest
                if (ok) ok = pop_tensor(stack, stack_size, tensor_b, error_code);  // c
                if (ok) ok = pop_tensor(stack, stack_size, tensor_c, error_code);  // b
                if (ok) ok = pop_tensor(stack, stack_size, tensor_d, error_code);  // a
                if (ok) {
                    if (tensor_d.rows != tensor_c.rows || tensor_d.rows != tensor_b.rows ||
                        tensor_d.cols != 1 || tensor_c.cols != 1 || tensor_b.cols != 1 ||
                        tensor_a.rows != tensor_d.rows || tensor_a.cols != 1) {
                        error_code = kErrorInvalidMatrixDims;
                    }
                }
            }
            __syncthreads();
            if (error_code != kErrorNone) {
                continue;
            }
            float* out = tensor_a.ptr;
            const float* a_ptr = tensor_d.ptr;
            const float* b_ptr = tensor_c.ptr;
            const float* c_ptr = tensor_b.ptr;
            for (int i = tid; i < tensor_a.rows; i += stride) {
                out[i] = a_ptr[i] + b_ptr[i] + c_ptr[i];
            }
            __syncthreads();
            if (tid == 0) {
                push_tensor(stack, stack_size, out, tensor_a.rows, 1, error_code);
            }
            continue;
        }

        if (opcode == 0x63 || opcode == 0x64) {  // SWIGLU_512 / SWIGLU_1024
            bool ok = true;
            if (tid == 0) {
                ok = pop_tensor(stack, stack_size, tensor_a, error_code);  // dest
                if (ok) ok = pop_tensor(stack, stack_size, tensor_b, error_code);  // input
                if (ok) {
                    int expected = (opcode == 0x63) ? 512 : 1024;
                    if (tensor_b.rows != expected || tensor_b.cols != 1 ||
                        tensor_a.rows != expected || tensor_a.cols != 1) {
                        error_code = kErrorInvalidMatrixDims;
                    }
                }
            }
            __syncthreads();
            if (error_code != kErrorNone) {
                continue;
            }
            float* out = tensor_a.ptr;
            const float* in = tensor_b.ptr;
            int limit = tensor_a.rows;
            for (int idx = tid; idx < limit; idx += stride) {
                float x = in[idx];
                float sig = 1.0f / (1.0f + expf(-x));
                out[idx] = x * sig;
            }
            __syncthreads();
            if (tid == 0) {
                push_tensor(stack, stack_size, out, tensor_a.rows, 1, error_code);
            }
            continue;
        }

        if (tid == 0) {
            switch (opcode) {
            case 0x00: {  // literal scalar
                float value = scalars ? scalars[scalar_index] : 0.0f;
                scalar_index += 1;
                push_scalar(stack, stack_size, value, error_code);
                break;
            }
            case 0x01: {  // literal vector
                push_vector(stack, stack_size, vectors, vector_index, error_code);
                vector_index += 1;
                break;
            }
            case 0x03: {  // pointer literal (rows, cols, ptr_lo, ptr_hi)
                if (!scalars) {
                    error_code = kErrorUnknownOpcode;
                    break;
                }
                float rows_f = scalars[scalar_index++];
                float cols_f = scalars[scalar_index++];
                float lo_f = scalars[scalar_index++];
                float hi_f = scalars[scalar_index++];
                int rows = static_cast<int>(roundf(rows_f));
                int cols = static_cast<int>(roundf(cols_f));
                uint32_t lo_bits = __float_as_uint(lo_f);
                uint32_t hi_bits = __float_as_uint(hi_f);
                uint64_t raw_ptr = (static_cast<uint64_t>(hi_bits) << 32) | lo_bits;
                float* ptr = reinterpret_cast<float*>(raw_ptr);
                push_tensor(stack, stack_size, ptr, rows, cols, error_code);
                break;
            }
            case 0x02: {  // literal matrix
                float rows_f = scalars ? scalars[scalar_index] : 0.0f;
                float cols_f = scalars ? scalars[scalar_index + 1] : 0.0f;
                scalar_index += 2;
                int rows = static_cast<int>(roundf(rows_f));
                int cols = static_cast<int>(roundf(cols_f));
                if (rows < 1) rows = 1;
                if (rows > kMatrixMaxDim) rows = kMatrixMaxDim;
                if (cols < 1) cols = 1;
                if (cols > kMatrixMaxDim) cols = kMatrixMaxDim;

                Matrix mat{};
                mat.rows = rows;
                mat.cols = cols;
                for (int r = 0; r < rows; ++r) {
                    for (int c = 0; c < cols; ++c) {
                        float value = matrices ? matrices[matrix_index] : 0.0f;
                        matrix_index += 1;
                        mat.data[r * cols + c] = value;
                    }
                }
                push_matrix(stack, stack_size, mat, error_code);
                break;
            }
            case 0x0A:  // add
            case 0x0B:  // sub
            case 0x0C:  // mul
            case 0x0D: {  // div
                float rhs = 0.0f;
                float lhs = 0.0f;
                if (!pop_scalar(stack, stack_size, rhs, error_code)) break;
                if (!pop_scalar(stack, stack_size, lhs, error_code)) break;
                float result = 0.0f;
                if (opcode == 0x0A) {
                    result = lhs + rhs;
                } else if (opcode == 0x0B) {
                    result = lhs - rhs;
                } else if (opcode == 0x0C) {
                    result = lhs * rhs;
                } else {
                    result = lhs / rhs;
                }
                push_scalar(stack, stack_size, result, error_code);
                break;
            }
            case 0x14:  // sqrt
            case 0x15:  // exp
            case 0x16:  // log
            case 0x18:  // sin
            case 0x19:  // cos
            case 0x1A: {  // tan
                float value = 0.0f;
                if (!pop_scalar(stack, stack_size, value, error_code)) break;
                float result = 0.0f;
                if (opcode == 0x14) {
                    result = sqrtf(value);
                } else if (opcode == 0x15) {
                    result = expf(value);
                } else if (opcode == 0x16) {
                    result = logf(value);
                } else if (opcode == 0x18) {
                    result = sinf(value);
                } else if (opcode == 0x19) {
                    result = cosf(value);
                } else {
                    result = tanf(value);
                }
                push_scalar(stack, stack_size, result, error_code);
                break;
            }
            case 0x28:  // gt
            case 0x2A:  // lt
            case 0x2C:  // eq
            case 0x2E:  // max
            case 0x2F: {  // min
                float rhs = 0.0f;
                float lhs = 0.0f;
                if (!pop_scalar(stack, stack_size, rhs, error_code)) break;
                if (!pop_scalar(stack, stack_size, lhs, error_code)) break;
                float result = 0.0f;
                if (opcode == 0x28) {
                    result = lhs > rhs ? 1.0f : 0.0f;
                } else if (opcode == 0x2A) {
                    result = lhs < rhs ? 1.0f : 0.0f;
                } else if (opcode == 0x2C) {
                    result = fabsf(lhs - rhs) < 1e-6f ? 1.0f : 0.0f;
                } else if (opcode == 0x2E) {
                    result = fmaxf(lhs, rhs);
                } else {
                    result = fminf(lhs, rhs);
                }
                push_scalar(stack, stack_size, result, error_code);
                break;
            }
            case 0x32: {  // dup
                if (stack_size == 0) {
                    error_code = kErrorStackUnderflow;
                    break;
                }
                StackItem top = stack[stack_size - 1];
                push_item(stack, stack_size, top, error_code);
                break;
            }
            case 0x33: {  // swap
                if (stack_size < 2) {
                    error_code = kErrorStackUnderflow;
                    break;
                }
                StackItem tmp = stack[stack_size - 1];
                stack[stack_size - 1] = stack[stack_size - 2];
                stack[stack_size - 2] = tmp;
                break;
            }
            case 0x34: {  // drop
                StackItem discarded{};
                pop_item(stack, stack_size, discarded, error_code);
                break;
            }
            case 0x5E: {  // trace (94 decimal)
                Matrix mat{};
                if (!pop_matrix(stack, stack_size, mat, error_code)) break;
                if (mat.rows != mat.cols) {
                    error_code = kErrorInvalidMatrixDims;
                    break;
                }
                float trace = 0.0f;
                for (int d = 0; d < mat.rows; ++d) {
                    trace += mat.data[d * mat.cols + d];
                }
                push_scalar(stack, stack_size, trace, error_code);
                break;
            }
            case 0x5A: {  // matmul (90 decimal)
                Matrix b{}, a{}, result{};
                if (!pop_matrix(stack, stack_size, b, error_code)) break;
                if (!pop_matrix(stack, stack_size, a, error_code)) break;
                if (a.cols != b.rows) {
                    error_code = kErrorInvalidMatrixDims;
                    break;
                }
                matmul(a, b, result);
                push_matrix(stack, stack_size, result, error_code);
                break;
            }
            case 0x5B: {  // transpose (91 decimal)
                Matrix input{}, result{};
                if (!pop_matrix(stack, stack_size, input, error_code)) break;
                transpose(input, result);
                push_matrix(stack, stack_size, result, error_code);
                break;
            }
            case 0x5C: {  // determinant (92 decimal)
                Matrix input{};
                if (!pop_matrix(stack, stack_size, input, error_code)) break;
                if (input.rows != input.cols) {
                    error_code = kErrorInvalidMatrixDims;
                    break;
                }
                float det = 0.0f;
                if (input.rows == 1) {
                    det = input.data[0];
                } else if (input.rows == 2) {
                    det = determinant_2x2(input);
                } else if (input.rows == 3) {
                    det = determinant_3x3(input);
                } else {
                    error_code = kErrorInvalidMatrixDims;
                    break;
                }
                push_scalar(stack, stack_size, det, error_code);
                break;
            }
            case 0x5D: {  // inverse (93 decimal)
                Matrix input{}, result{};
                if (!pop_matrix(stack, stack_size, input, error_code)) break;
                if (input.rows != input.cols) {
                    error_code = kErrorInvalidMatrixDims;
                    break;
                }
                bool ok = false;
                if (input.rows == 1) {
                    if (fabsf(input.data[0]) < 1e-8f) {
                        error_code = kErrorSingularMatrix;
                        break;
                    }
                    result.rows = 1;
                    result.cols = 1;
                    result.data[0] = 1.0f / input.data[0];
                    ok = true;
                } else if (input.rows == 2) {
                    ok = inverse_2x2(input, result);
                } else if (input.rows == 3) {
                    ok = inverse_3x3(input, result);
                }
                if (!ok) {
                    error_code = kErrorSingularMatrix;
                    break;
                }
                push_matrix(stack, stack_size, result, error_code);
                break;
            }
            case 0x60: {  // MATVEC_512x1024 (dest, matrix, vector)
                TensorRef dest{}, matrix{}, vec{};
                if (!pop_tensor(stack, stack_size, dest, error_code)) break;
                if (!pop_tensor(stack, stack_size, matrix, error_code)) break;
                if (!pop_tensor(stack, stack_size, vec, error_code)) break;
                if (matrix.rows != 1024 || matrix.cols != 512 || vec.rows != 512 || vec.cols != 1 || dest.rows != 1024 || dest.cols != 1) {
                    error_code = kErrorInvalidMatrixDims;
                    break;
                }
                float* out = dest.ptr;
                const float* weights = matrix.ptr;
                const float* input_vec = vec.ptr;
                for (int r = 0; r < 1024; ++r) {
                    const float* row_ptr = weights + r * 512;
                    float sum = 0.0f;
                    #pragma unroll 8
                    for (int c = 0; c < 512; ++c) {
                        sum += row_ptr[c] * input_vec[c];
                    }
                    out[r] = sum;
                }
                push_tensor(stack, stack_size, out, 1024, 1, error_code);
                break;
            }
            case 0x61: {  // MATVEC_1024x512 (dest, matrix, vector)
                TensorRef dest{}, matrix{}, vec{};
                if (!pop_tensor(stack, stack_size, dest, error_code)) break;
                if (!pop_tensor(stack, stack_size, matrix, error_code)) break;
                if (!pop_tensor(stack, stack_size, vec, error_code)) break;
                if (matrix.rows != 512 || matrix.cols != 1024 || vec.rows != 1024 || vec.cols != 1 || dest.rows != 512 || dest.cols != 1) {
                    error_code = kErrorInvalidMatrixDims;
                    break;
                }
                float* out = dest.ptr;
                const float* weights = matrix.ptr;
                const float* input_vec = vec.ptr;
                for (int r = 0; r < 512; ++r) {
                    const float* row_ptr = weights + r * 1024;
                    float sum = 0.0f;
                    #pragma unroll 8
                    for (int c = 0; c < 1024; ++c) {
                        sum += row_ptr[c] * input_vec[c];
                    }
                    out[r] = sum;
                }
                push_tensor(stack, stack_size, out, 512, 1, error_code);
                break;
            }
            case 0x62: {  // VEC_ADD3 (dest, a, b, c)
                TensorRef dest{}, c{}, b{}, a{};
                if (!pop_tensor(stack, stack_size, dest, error_code)) break;
                if (!pop_tensor(stack, stack_size, c, error_code)) break;
                if (!pop_tensor(stack, stack_size, b, error_code)) break;
                if (!pop_tensor(stack, stack_size, a, error_code)) break;
                if (a.rows != b.rows || a.rows != c.rows || a.cols != 1 || b.cols != 1 || c.cols != 1 || dest.rows != a.rows || dest.cols != 1) {
                    error_code = kErrorInvalidMatrixDims;
                    break;
                }
                float* out = dest.ptr;
                const float* a_ptr = a.ptr;
                const float* b_ptr = b.ptr;
                const float* c_ptr = c.ptr;
                for (int i = 0; i < a.rows; ++i) {
                    out[i] = a_ptr[i] + b_ptr[i] + c_ptr[i];
                }
                push_tensor(stack, stack_size, out, a.rows, 1, error_code);
                break;
            }
            case 0x63: {  // SWIGLU_512 (dest, input)
                TensorRef dest{}, input{};
                if (!pop_tensor(stack, stack_size, dest, error_code)) break;
                if (!pop_tensor(stack, stack_size, input, error_code)) break;
                if (input.rows != 512 || dest.rows != 512 || input.cols != 1 || dest.cols != 1) {
                    error_code = kErrorInvalidMatrixDims;
                    break;
                }
                float* out = dest.ptr;
                const float* in = input.ptr;
                for (int i = 0; i < 512; ++i) {
                    float x = in[i];
                    float sig = 1.0f / (1.0f + expf(-x));
                    out[i] = x * sig;
                }
                push_tensor(stack, stack_size, out, 512, 1, error_code);
                break;
            }
            case 0x64: {  // SWIGLU_1024 (dest, input)
                TensorRef dest{}, input{};
                if (!pop_tensor(stack, stack_size, dest, error_code)) break;
                if (!pop_tensor(stack, stack_size, input, error_code)) break;
                if (input.rows != 1024 || dest.rows != 1024 || input.cols != 1 || dest.cols != 1) {
                    error_code = kErrorInvalidMatrixDims;
                    break;
                }
                float* out = dest.ptr;
                const float* in = input.ptr;
                for (int i = 0; i < 1024; ++i) {
                    float x = in[i];
                    float sig = 1.0f / (1.0f + expf(-x));
                    out[i] = x * sig;
                }
                push_tensor(stack, stack_size, out, 1024, 1, error_code);
                break;
            }
            default:
                error_code = kErrorUnknownOpcode;
                break;
        }
        }

        __syncthreads();
        if (error_code != kErrorNone) {
            break;
        }
    }

    if (tid == 0) {
        state->head = 0;
        state->size = stack_size;
        state->error = error_code;
        state->reserved = 0;

        const uint32_t count = (stack_size < kStackCapacity) ? stack_size : kStackCapacity;
        for (uint32_t idx = 0; idx < count; ++idx) {
            const StackItem& item = stack[idx];
            state->stack[idx][0] = item.value[0];
            state->stack[idx][1] = item.value[1];
            state->stack[idx][2] = item.value[2];
            state->stack[idx][3] = pack_meta(item.type, item.rows, item.cols, item.row_index);
        }
    }
}
