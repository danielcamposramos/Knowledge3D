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
constexpr uint32_t kErrorVerificationFailed = 9005;

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

__device__ __constant__ float kProceduralPrototypeTable[4][3] = {
    {0.5f, 0.0f, 0.0f},
    {0.0f, 0.5f, 0.0f},
    {0.0f, 0.0f, 0.5f},
    {0.5f, 0.5f, 0.5f}
};

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

__device__ inline uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

__device__ inline uint32_t trigram_hash(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t hash = 0x811C9DC5u;
    hash ^= a + 0x9e3779b9u + (hash << 6) + (hash >> 2);
    hash ^= b + 0x9e3779b9u + (hash << 6) + (hash >> 2);
    hash ^= c + 0x9e3779b9u + (hash << 6) + (hash >> 2);
    return mix32(hash);
}

__device__ inline float3 pseudo_random_vec(uint32_t seed) {
    seed = mix32(seed);
    float x = (seed & 0x3FFu) / 1024.0f;
    seed = mix32(seed >> 10);
    float y = (seed & 0x3FFu) / 1024.0f;
    seed = mix32(seed >> 10);
    float z = (seed & 0x3FFu) / 1024.0f;
    float3 vec = make_float3(x * 2.0f - 1.0f, y * 2.0f - 1.0f, z * 2.0f - 1.0f);
    return normalize3(vec);
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
    __shared__ StackValue checkpoint_stack[kStackCapacity];
    __shared__ uint32_t checkpoint_size;
    __shared__ uint32_t checkpoint_valid;

    if (tid == 0) {
        stack_size = 0;
        error_code = kErrorNone;
        scalar_index = 0;
        vector_index = 0;
        checkpoint_size = 0;
        checkpoint_valid = 0;
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
                case 0x20: {  // OP_TRIGRAM_HASH
                    float3 tri{};
                    if (!pop_vector(stack, stack_size, tri, error_code)) break;
                    uint32_t h = trigram_hash(
                        static_cast<uint32_t>(fabsf(tri.x) * 255.0f) & 0xFFu,
                        static_cast<uint32_t>(fabsf(tri.y) * 255.0f) & 0xFFu,
                        static_cast<uint32_t>(fabsf(tri.z) * 255.0f) & 0xFFu);
                    float hash_norm = (h & 0xFFFFFFu) / static_cast<float>(0xFFFFFFu);
                    push(stack, stack_size, make_scalar(hash_norm), error_code);
                    break;
                }
                case 0x21: {  // OP_EMBED_LOOKUP
                    float hash_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, hash_scalar, error_code)) break;
                    uint32_t seed = static_cast<uint32_t>(fabsf(hash_scalar) * 4294967295.0f);
                    float3 vec = pseudo_random_vec(seed);
                    push(stack, stack_size, make_vector(vec.x, vec.y, vec.z), error_code);
                    break;
                }
                case 0x22: {  // OP_ADAPTIVE_DIM
                    float dim_scalar = 0.0f;
                    float3 vec{};
                    if (!pop_scalar(stack, stack_size, dim_scalar, error_code)) break;
                    if (!pop_vector(stack, stack_size, vec, error_code)) break;
                    int dims = max(1, min(3, static_cast<int>(dim_scalar + 0.5f)));
                    if (dims < 3) vec.z = 0.0f;
                    if (dims < 2) vec.y = 0.0f;
                    push(stack, stack_size, vec, error_code);
                    break;
                }
                case 0x23: {  // OP_NORMALIZE_L2
                    float3 vec{};
                    if (!pop_vector(stack, stack_size, vec, error_code)) break;
                    float3 norm = normalize3(vec);
                    push(stack, stack_size, make_vector(norm.x, norm.y, norm.z), error_code);
                    break;
                }
                case 0x30: {  // OP_FRACTAL_EMIT
                    float iterations = 0.0f;
                    float3 seed_vec{};
                    if (!pop_scalar(stack, stack_size, iterations, error_code)) break;
                    if (!pop_vector(stack, stack_size, seed_vec, error_code)) break;
                    float3 z = make_float3(0.0f, 0.0f, 0.0f);
                    int iters = max(1, min(64, static_cast<int>(iterations)));
                    for (int iter = 0; iter < iters; ++iter) {
                        float x = z.x * z.x - z.y * z.y + seed_vec.x;
                        float y = 2.0f * z.x * z.y + seed_vec.y;
                        z.x = x;
                        z.y = y;
                        z.z = seed_vec.z;
                        if (dot3(z, z) > 16.0f) break;
                    }
                    push(stack, stack_size, make_vector(z.x, z.y, z.z), error_code);
                    break;
                }
                case 0x31: {  // OP_AUDIO_SYNTH
                    float time_scalar = 0.0f;
                    float freq_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, time_scalar, error_code)) break;
                    if (!pop_scalar(stack, stack_size, freq_scalar, error_code)) break;
                    float w = 2.0f * 3.1415926535f * freq_scalar * time_scalar;
                    float3 audio = make_float3(sinf(w), cosf(w), sinf(w * 0.5f));
                    push(stack, stack_size, audio, error_code);
                    break;
                }
                case 0x32: {  // OP_MODALITY_FUSE
                    float3 b{};
                    float3 a{};
                    if (!pop_vector(stack, stack_size, b, error_code)) break;
                    if (!pop_vector(stack, stack_size, a, error_code)) break;
                    float3 fused = make_float3(
                        0.5f * (a.x + b.x),
                        0.5f * (a.y + b.y),
                        0.5f * (a.z + b.z));
                    push(stack, stack_size, fused, error_code);
                    break;
                }
                case 0x40: {  // OP_PROTOTYPE_LOAD
                    float proto_idx_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, proto_idx_scalar, error_code)) break;
                    int idx = max(0, min(3, static_cast<int>(proto_idx_scalar + 0.5f)));
                    float3 proto = make_float3(
                        kProceduralPrototypeTable[idx][0],
                        kProceduralPrototypeTable[idx][1],
                        kProceduralPrototypeTable[idx][2]);
                    push(stack, stack_size, proto, error_code);
                    break;
                }
                case 0x41: {  // OP_DELTA_APPLY
                    float3 delta{};
                    float3 base{};
                    if (!pop_vector(stack, stack_size, delta, error_code)) break;
                    if (!pop_vector(stack, stack_size, base, error_code)) break;
                    float3 result = make_float3(base.x + delta.x, base.y + delta.y, base.z + delta.z);
                    push(stack, stack_size, result, error_code);
                    break;
                }
                case 0x42: {  // OP_UNCERTAINTY_FUSE
                    float confidence = 0.0f;
                    float3 proposal{};
                    float3 reference{};
                    if (!pop_scalar(stack, stack_size, confidence, error_code)) break;
                    if (!pop_vector(stack, stack_size, proposal, error_code)) break;
                    if (!pop_vector(stack, stack_size, reference, error_code)) break;
                    float alpha = max(0.0f, min(1.0f, confidence));
                    float3 fused = make_float3(
                        reference.x * (1.0f - alpha) + proposal.x * alpha,
                        reference.y * (1.0f - alpha) + proposal.y * alpha,
                        reference.z * (1.0f - alpha) + proposal.z * alpha);
                    push(stack, stack_size, fused, error_code);
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
                case 0x50: {  // OP_SUPERPOSE (legacy ifelse fallback)
                    bool handled_ifelse = false;
                    if (stack_size >= 3) {
                        const StackValue& predicate_candidate = stack[stack_size - 3];
                        if (!is_vector(predicate_candidate)) {
                            StackValue false_branch{};
                            StackValue true_branch{};
                            float predicate = 0.0f;
                            if (!pop(stack, stack_size, false_branch, error_code)) break;
                            if (!pop(stack, stack_size, true_branch, error_code)) break;
                            if (!pop_scalar(stack, stack_size, predicate, error_code)) break;
                            const bool take_true = fabsf(predicate) > 1e-6f;
                            push(stack, stack_size, take_true ? true_branch : false_branch, error_code);
                            handled_ifelse = true;
                        }
                    }
                    if (!handled_ifelse) {
                        float3 b{};
                        float3 a{};
                        if (!pop_vector(stack, stack_size, b, error_code)) break;
                        if (!pop_vector(stack, stack_size, a, error_code)) break;
                        float3 sum = make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
                        sum = normalize3(sum);
                        push(stack, stack_size, sum, error_code);
                    }
                    break;
                }
                case 0x51: {  // OP_ENTANGLE
                    float3 b{};
                    float3 a{};
                    if (!pop_vector(stack, stack_size, b, error_code)) break;
                    if (!pop_vector(stack, stack_size, a, error_code)) break;
                    float3 entangled = make_float3(
                        a.x * b.y - a.y * b.x,
                        a.y * b.z - a.z * b.y,
                        a.z * b.x - a.x * b.z);
                    push(stack, stack_size, entangled, error_code);
                    break;
                }
                case 0x52: {  // OP_COLLAPSE
                    float threshold = 0.0f;
                    float3 state{};
                    if (!pop_scalar(stack, stack_size, threshold, error_code)) break;
                    if (!pop_vector(stack, stack_size, state, error_code)) break;
                    float clamp = fabsf(threshold);
                    if (fabsf(state.x) < clamp) state.x = 0.0f;
                    if (fabsf(state.y) < clamp) state.y = 0.0f;
                    if (fabsf(state.z) < clamp) state.z = 0.0f;
                    push(stack, stack_size, state, error_code);
                    break;
                }
                case 0x60: {  // OP_CHECKPOINT
                    checkpoint_size = stack_size;
                    for (uint32_t idx = 0; idx < stack_size && idx < kStackCapacity; ++idx) {
                        checkpoint_stack[idx] = stack[idx];
                    }
                    checkpoint_valid = 1;
                    break;
                }
                case 0x61: {  // OP_ROLLBACK
                    if (!checkpoint_valid) {
                        error_code = kErrorStackUnderflow;
                        break;
                    }
                    stack_size = checkpoint_size;
                    for (uint32_t idx = 0; idx < checkpoint_size && idx < kStackCapacity; ++idx) {
                        stack[idx] = checkpoint_stack[idx];
                    }
                    break;
                }
                case 0x62: {  // OP_VERIFY
                    bool ok = true;
                    for (uint32_t idx = 0; idx < stack_size; ++idx) {
                        const StackValue& val = stack[idx];
                        if (!isfinite(val.x) || !isfinite(val.y) || !isfinite(val.z) || !isfinite(val.w)) {
                            ok = false;
                            break;
                        }
                    }
                    if (!ok) {
                        error_code = kErrorVerificationFailed;
                        break;
                    }
                    push(stack, stack_size, make_scalar(1.0f), error_code);
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
