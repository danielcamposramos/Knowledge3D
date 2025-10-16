#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {
constexpr int kStackCapacity = 64;

enum class ValueType : uint32_t {
    kScalar = 0,
    kVector = 1,
};

constexpr uint32_t kErrorNone = 0;
constexpr uint32_t kErrorUnknownOpcode = 9001;
constexpr uint32_t kErrorStackUnderflow = 9002;
constexpr uint32_t kErrorStackOverflow = 9003;
constexpr uint32_t kErrorTypeMismatch = 9004;

struct alignas(16) StackValue {
    float x;
    float y;
    float z;
    float w;  // tag lane – 0.0 = scalar, 1.0 = vector
};

struct alignas(16) InstanceState {
    uint32_t head;
    uint32_t size;
    uint32_t error;
    uint32_t reserved;
    StackValue stack[kStackCapacity];
};

static_assert(sizeof(InstanceState) == 1040, "InstanceState layout mismatch");

__device__ inline StackValue make_scalar(float v) {
    StackValue out{};
    out.x = v;
    out.y = 0.0f;
    out.z = 0.0f;
    out.w = static_cast<float>(ValueType::kScalar);
    return out;
}

__device__ inline StackValue make_vector(float x, float y, float z) {
    StackValue out{};
    out.x = x;
    out.y = y;
    out.z = z;
    out.w = static_cast<float>(ValueType::kVector);
    return out;
}

__device__ inline bool is_vector(const StackValue& value) {
    return fabsf(value.w - static_cast<float>(ValueType::kVector)) < 1e-6f;
}

__device__ inline bool push(StackValue* stack, uint32_t& size, const StackValue& value, uint32_t& error) {
    if (size >= kStackCapacity) {
        error = kErrorStackOverflow;
        return false;
    }
    stack[size] = value;
    size += 1;
    return true;
}

__device__ inline bool pop(StackValue* stack, uint32_t& size, StackValue& value, uint32_t& error) {
    if (size == 0) {
        error = kErrorStackUnderflow;
        return false;
    }
    size -= 1;
    value = stack[size];
    return true;
}

__device__ inline bool pop_scalar(StackValue* stack, uint32_t& size, float& scalar, uint32_t& error) {
    StackValue tmp{};
    if (!pop(stack, size, tmp, error)) {
        return false;
    }
    if (is_vector(tmp)) {
        error = kErrorTypeMismatch;
        return false;
    }
    scalar = tmp.x;
    return true;
}

__device__ inline bool pop_vector(StackValue* stack, uint32_t& size, float3& vec, uint32_t& error) {
    StackValue tmp{};
    if (!pop(stack, size, tmp, error)) {
        return false;
    }
    if (!is_vector(tmp)) {
        error = kErrorTypeMismatch;
        return false;
    }
    vec.x = tmp.x;
    vec.y = tmp.y;
    vec.z = tmp.z;
    return true;
}

__device__ inline float dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float3 cross3(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ inline float3 normalize3(const float3& v) {
    float mag = sqrtf(dot3(v, v));
    if (mag < 1e-6f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    float inv = 1.0f / mag;
    return make_float3(v.x * inv, v.y * inv, v.z * inv);
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
                    float value = scalars ? scalars[scalar_index] : 0.0f;
                    scalar_index += 1;
                    push(stack, stack_size, make_scalar(value), error_code);
                    break;
                }
                case 0x01: {  // literal vector
                    float vx = 0.0f;
                    float vy = 0.0f;
                    float vz = 0.0f;
                    if (vectors) {
                        vx = vectors[vector_index * 3 + 0];
                        vy = vectors[vector_index * 3 + 1];
                        vz = vectors[vector_index * 3 + 2];
                    }
                    vector_index += 1;
                    push(stack, stack_size, make_vector(vx, vy, vz), error_code);
                    break;
                }
                case 0x0A:  // add
                case 0x0B:  // sub
                case 0x0C:  // mul
                case 0x0D:  // div
                case 0x0E:  // pow
                case 0x0F: {  // neg
                    float lhs = 0.0f;
                    float rhs = 0.0f;
                    if (opcode == 0x0F) {
                        if (!pop_scalar(stack, stack_size, lhs, error_code)) break;
                        push(stack, stack_size, make_scalar(-lhs), error_code);
                    } else if (opcode == 0x0E) {
                        if (!pop_scalar(stack, stack_size, rhs, error_code)) break;
                        if (!pop_scalar(stack, stack_size, lhs, error_code)) break;
                        push(stack, stack_size, make_scalar(powf(lhs, rhs)), error_code);
                    } else {
                        if (!pop_scalar(stack, stack_size, rhs, error_code)) break;
                        if (!pop_scalar(stack, stack_size, lhs, error_code)) break;
                        float result = 0.0f;
                        if (opcode == 0x0A) result = lhs + rhs;
                        else if (opcode == 0x0B) result = lhs - rhs;
                        else if (opcode == 0x0C) result = lhs * rhs;
                        else result = lhs / rhs;
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
                    float value = 0.0f;
                    if (!pop_scalar(stack, stack_size, value, error_code)) break;
                    float result = 0.0f;
                    if (opcode == 0x14) result = sqrtf(value);
                    else if (opcode == 0x15) result = expf(value);
                    else if (opcode == 0x16) result = logf(value);
                    else if (opcode == 0x18) result = sinf(value);
                    else if (opcode == 0x19) result = cosf(value);
                    else result = tanf(value);
                    push(stack, stack_size, make_scalar(result), error_code);
                    break;
                }
                case 0x28:  // gt
                case 0x2A:  // lt
                case 0x2C:  // eq
                case 0x2E:  // max
                case 0x2F: {  // min
                    float lhs = 0.0f;
                    float rhs = 0.0f;
                    if (!pop_scalar(stack, stack_size, rhs, error_code)) break;
                    if (!pop_scalar(stack, stack_size, lhs, error_code)) break;
                    float result = 0.0f;
                    if (opcode == 0x28) result = lhs > rhs ? 1.0f : 0.0f;
                    else if (opcode == 0x2A) result = lhs < rhs ? 1.0f : 0.0f;
                    else if (opcode == 0x2C) result = fabsf(lhs - rhs) < 1e-6f ? 1.0f : 0.0f;
                    else if (opcode == 0x2E) result = fmaxf(lhs, rhs);
                    else result = fminf(lhs, rhs);
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
                case 0x35: {  // over (duplicate next-to-top)
                    if (stack_size < 2) {
                        error_code = kErrorStackUnderflow;
                        break;
                    }
                    StackValue second = stack[stack_size - 2];
                    push(stack, stack_size, second, error_code);
                    break;
                }
                case 0x36: {  // rot (a b c -> b c a)
                    if (stack_size < 3) {
                        error_code = kErrorStackUnderflow;
                        break;
                    }
                    StackValue c = stack[stack_size - 1];
                    StackValue b = stack[stack_size - 2];
                    StackValue a = stack[stack_size - 3];
                    stack[stack_size - 3] = b;
                    stack[stack_size - 2] = c;
                    stack[stack_size - 1] = a;
                    break;
                }
                case 0x37: {  // clear
                    stack_size = 0;
                    break;
                }
                case 0x3C: {  // dot
                    float3 a{}, b{};
                    if (!pop_vector(stack, stack_size, b, error_code)) break;
                    if (!pop_vector(stack, stack_size, a, error_code)) break;
                    push(stack, stack_size, make_scalar(dot3(a, b)), error_code);
                    break;
                }
                case 0x3D: {  // cross
                    float3 a{}, b{};
                    if (!pop_vector(stack, stack_size, b, error_code)) break;
                    if (!pop_vector(stack, stack_size, a, error_code)) break;
                    float3 result = cross3(a, b);
                    push(stack, stack_size, make_vector(result.x, result.y, result.z), error_code);
                    break;
                }
                case 0x3E: {  // magnitude
                    float3 v{};
                    if (!pop_vector(stack, stack_size, v, error_code)) break;
                    push(stack, stack_size, make_scalar(sqrtf(dot3(v, v))), error_code);
                    break;
                }
                case 0x3F: {  // normalize
                    float3 v{};
                    if (!pop_vector(stack, stack_size, v, error_code)) break;
                    float3 norm = normalize3(v);
                    push(stack, stack_size, make_vector(norm.x, norm.y, norm.z), error_code);
                    break;
                }
                case 0x43: {  // sigmoid approximation
                    float value = 0.0f;
                    if (!pop_scalar(stack, stack_size, value, error_code)) break;
                    float sig = 0.5f * (1.0f + tanhf(0.5f * value));
                    push(stack, stack_size, make_scalar(sig), error_code);
                    break;
                }
                case 0x46: {  // rotate (around Z axis)
                    float angle = 0.0f;
                    float3 vec{};
                    if (!pop_scalar(stack, stack_size, angle, error_code)) break;
                    if (!pop_vector(stack, stack_size, vec, error_code)) break;
                    float s = sinf(angle);
                    float c = cosf(angle);
                    float x = vec.x * c - vec.y * s;
                    float y = vec.x * s + vec.y * c;
                    push(stack, stack_size, make_vector(x, y, vec.z), error_code);
                    break;
                }
                case 0x47: {  // scale
                    float factor = 0.0f;
                    float3 v{};
                    if (!pop_scalar(stack, stack_size, factor, error_code)) break;
                    if (!pop_vector(stack, stack_size, v, error_code)) break;
                    v.x *= factor;
                    v.y *= factor;
                    v.z *= factor;
                    push(stack, stack_size, make_vector(v.x, v.y, v.z), error_code);
                    break;
                }
                case 0x48: {  // translate
                    float3 delta{};
                    float3 base{};
                    if (!pop_vector(stack, stack_size, delta, error_code)) break;
                    if (!pop_vector(stack, stack_size, base, error_code)) break;
                    float3 result = make_float3(base.x + delta.x, base.y + delta.y, base.z + delta.z);
                    push(stack, stack_size, make_vector(result.x, result.y, result.z), error_code);
                    break;
                }
                case 0x50: {  // ifelse
                    StackValue false_branch{};
                    StackValue true_branch{};
                    float predicate = 0.0f;
                    if (!pop(stack, stack_size, false_branch, error_code)) break;
                    if (!pop(stack, stack_size, true_branch, error_code)) break;
                    if (!pop_scalar(stack, stack_size, predicate, error_code)) break;
                    const bool take_true = fabsf(predicate) > 1e-6f;
                    push(stack, stack_size, take_true ? true_branch : false_branch, error_code);
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
