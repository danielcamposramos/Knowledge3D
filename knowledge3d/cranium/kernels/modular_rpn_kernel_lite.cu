#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {
constexpr int kStackCapacity = 64;
constexpr uint32_t kErrorNone = 0;
constexpr uint32_t kErrorUnknownOpcode = 9001;
constexpr uint32_t kErrorStackUnderflow = 9002;
constexpr uint32_t kErrorStackOverflow = 9003;

struct alignas(16) StackValue {
    float x;
    float y;
    float z;
    float w;
};

struct alignas(16) InstanceState {
    uint32_t head;
    uint32_t size;
    uint32_t error;
    uint32_t reserved;
    StackValue stack[kStackCapacity];
};

static_assert(sizeof(InstanceState) == 1040, "InstanceState layout mismatch");

__device__ inline bool push(StackValue* stack, uint32_t& size, const StackValue& value, uint32_t& error) {
    if (size >= kStackCapacity) {
        error = kErrorStackOverflow;
        return false;
    }
    stack[size] = value;
    size += 1;
    return true;
}

__device__ inline bool pop(StackValue* stack, uint32_t& size, StackValue& out, uint32_t& error) {
    if (size == 0) {
        error = kErrorStackUnderflow;
        return false;
    }
    size -= 1;
    out = stack[size];
    return true;
}

__device__ inline float to_scalar(const StackValue& value) {
    return value.x;
}

__device__ inline StackValue make_scalar(float v) {
    StackValue out{};
    out.x = v;
    out.y = 0.0f;
    out.z = 0.0f;
    out.w = 0.0f;
    return out;
}

__device__ inline StackValue make_vector(const float* vectors, uint32_t index) {
    StackValue out{};
    if (vectors) {
        out.x = vectors[index * 3 + 0];
        out.y = vectors[index * 3 + 1];
        out.z = vectors[index * 3 + 2];
    } else {
        out.x = 0.0f;
        out.y = 0.0f;
        out.z = 0.0f;
    }
    out.w = 0.0f;
    return out;
}
}  // namespace

extern "C" __global__ void modular_rpn_geometric_kernel(
    uint32_t instance_id,
    const uint16_t* __restrict__ op_codes,
    const float* __restrict__ scalars,
    const float* __restrict__ vectors,
    InstanceState* __restrict__ states,
    uint32_t token_count) {
    InstanceState* state = reinterpret_cast<InstanceState*>(
        reinterpret_cast<uint8_t*>(states) + instance_id * sizeof(InstanceState));

    const int tid = threadIdx.x;

    __shared__ StackValue stack[kStackCapacity];
    __shared__ uint32_t stack_size;
    __shared__ uint32_t error_code;
    __shared__ uint32_t scalar_index;
    __shared__ uint32_t vector_index;
    if (tid == 0) {
        stack_size = 0;
        error_code = kErrorNone;
        scalar_index = 0;
        vector_index = 0;
    }
    __syncthreads();

    for (uint32_t i = 0; i < token_count; ++i) {
        __syncthreads();
        if (error_code != kErrorNone) {
            break;
        }

        if (tid == 0) {
            const uint16_t opcode = op_codes[i];

            switch (opcode) {
                case 0x00: {  // literal scalar
                    float value = 0.0f;
                    if (scalars) {
                        value = scalars[scalar_index];
                    }
                    scalar_index += 1;
                    StackValue literal = make_scalar(value);
                    push(stack, stack_size, literal, error_code);
                    break;
                }
                case 0x01: {  // literal vector
                    StackValue literal = make_vector(
                        vectors ? vectors : nullptr,
                        vectors ? vector_index : 0);
                    vector_index += 1;
                    push(stack, stack_size, literal, error_code);
                    break;
                }
                case 0x0A:  // add
                case 0x0B:  // sub
                case 0x0C:  // mul
                case 0x0D:  // div
                case 0x0F: {  // neg
                    StackValue a{}, b{};
                    if (opcode == 0x0F) {
                        if (!pop(stack, stack_size, a, error_code)) break;
                        float v = -to_scalar(a);
                        push(stack, stack_size, make_scalar(v), error_code);
                    } else {
                        if (!pop(stack, stack_size, b, error_code)) break;
                        if (!pop(stack, stack_size, a, error_code)) break;

                        const float lhs = to_scalar(a);
                        const float rhs = to_scalar(b);
                        float result = 0.0f;
                        if (opcode == 0x0A) {
                            result = lhs + rhs;
                        } else if (opcode == 0x0B) {
                            result = lhs - rhs;
                        } else if (opcode == 0x0C) {
                            result = lhs * rhs;
                        } else {  // div
                            result = lhs / rhs;
                        }
                        push(stack, stack_size, make_scalar(result), error_code);
                    }
                    break;
                }
                case 0x14:  // sqrt
                case 0x15:  // exp
                case 0x16:  // log
                case 0x18:  // sin
                case 0x19:  // cos
                case 0x1A: {  // tan
                    StackValue value{};
                    if (!pop(stack, stack_size, value, error_code)) break;
                    float arg = to_scalar(value);
                    float result = 0.0f;
                    if (opcode == 0x14) {
                        result = sqrtf(arg);
                    } else if (opcode == 0x15) {
                        result = expf(arg);
                    } else if (opcode == 0x16) {
                        result = logf(arg);
                    } else if (opcode == 0x18) {
                        result = sinf(arg);
                    } else if (opcode == 0x19) {
                        result = cosf(arg);
                    } else {
                        result = tanf(arg);
                    }
                    push(stack, stack_size, make_scalar(result), error_code);
                    break;
                }
                case 0x28:  // gt
                case 0x2A:  // lt
                case 0x2C:  // eq
                case 0x2E:  // max
                case 0x2F: {  // min
                    StackValue a{}, b{};
                    if (!pop(stack, stack_size, b, error_code)) break;
                    if (!pop(stack, stack_size, a, error_code)) break;

                    float lhs = to_scalar(a);
                    float rhs = to_scalar(b);
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

                    push(stack, stack_size, make_scalar(result), error_code);
                    break;
                }
                case 0x32: {  // dup
                    if (stack_size == 0) {
                        error_code = kErrorStackUnderflow;
                        break;
                    }
                    StackValue top = stack[stack_size - 1];
                    push(stack, stack_size, top, error_code);
                    break;
                }
                case 0x33: {  // swap
                    if (stack_size < 2) {
                        error_code = kErrorStackUnderflow;
                        break;
                    }
                    StackValue tmp = stack[stack_size - 1];
                    stack[stack_size - 1] = stack[stack_size - 2];
                    stack[stack_size - 2] = tmp;
                    break;
                }
                case 0x34: {  // drop
                    StackValue discarded{};
                    pop(stack, stack_size, discarded, error_code);
                    break;
                }
                default:
                    error_code = kErrorUnknownOpcode;
                    break;
            }
        }
    }

    __syncthreads();
    if (tid == 0) {
        state->head = 0;
        state->size = stack_size;
        state->error = error_code;
        state->reserved = 0;

        for (uint32_t i = 0; i < stack_size && i < kStackCapacity; ++i) {
            state->stack[i] = stack[i];
        }
    }
}
