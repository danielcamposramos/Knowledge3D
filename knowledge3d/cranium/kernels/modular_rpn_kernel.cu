#include "physics_body_soa.h"
#include "entity_hot_path.h"
#include "cas_star_node.h"
#include "sas_hashcons.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

extern "C" {
__device__ __constant__ unsigned long long g_galaxy_entries_ptr = 0ULL;
__device__ __constant__ unsigned int g_galaxy_entry_count = 0u;
__device__ __constant__ unsigned int g_galaxy_entry_stride = 19u;
__device__ __constant__ unsigned int g_galaxy_embedding_dim = 16u;
__device__ __constant__ unsigned int g_galaxy_embedding_offset = 3u;
__device__ __constant__ unsigned long long g_query_embedding_ptr = 0ULL;
__device__ __constant__ unsigned int g_query_embedding_stride = 16u;
__device__ __constant__ unsigned long long g_physics_body_soa_ptr = 0ULL;
__device__ __constant__ unsigned long long g_physics_contact_soa_ptr = 0ULL;
__device__ __constant__ unsigned long long g_physics_event_queue_ptr = 0ULL;
__device__ __constant__ unsigned long long g_physics_predicted_soa_ptr = 0ULL;
struct PhysicsMaterialEntry {
    uint32_t star_id;
    float friction;
    float restitution;
    float density;
    uint32_t texture_id;
};
__device__ __constant__ unsigned long long g_physics_material_table_ptr = 0ULL;
__device__ __constant__ unsigned int g_physics_material_table_count = 0u;
__device__ __constant__ unsigned long long g_entity_hot_path_ptr = 0ULL;
__device__ __constant__ unsigned int g_entity_count = 0u;
struct TextureHandlePool {
    unsigned long long slot_ptr[256];
    unsigned int width[256];
    unsigned int height[256];
    unsigned char in_use[256];
    unsigned int baked_source_slot[64];
    unsigned char baked_in_use[64];
};
__device__ __constant__ unsigned long long g_texture_pool_ptr = 0ULL;
__device__ __constant__ unsigned long long g_texture_permutation_table_ptr = 0ULL;
__device__ StarNode g_cas_pool[CAS_POOL_SIZE];
__device__ float g_cas_coeffs[CAS_COEFF_SIZE];
__device__ uint32_t g_cas_pool_top = 0u;
__device__ uint32_t g_cas_coeff_top = 0u;
__device__ HashconsSlot g_sas_hashcons[SAS_HASHCONS_SIZE];
__device__ __constant__ float g_sas_symbol_values[256];
__device__ __constant__ uint32_t g_sas_symbol_star_ids[256];
}

namespace {
constexpr int kStackCapacity = 64;
constexpr uint32_t kCasPolyCoeffScratchMax = 64u;
constexpr uint32_t kSasTraversalCap = 128u;
constexpr uint32_t kSasBindingCap = 16u;
constexpr uint32_t kSasEmptyBinding = 0xFFFFFFFFu;
constexpr int kGalaxyEmbeddingDim = 16;
constexpr int kGalaxyEntryStrideDefault = 19;
constexpr int kGalaxyEmbeddingOffsetDefault = 3;
constexpr int kGalaxyTopKMax = 16;

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
constexpr uint32_t kErrorInvalidArgument = 9006;

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

__device__ inline int8_t clamp_trit(float v, float thresh = 0.333333f) {
    if (v > thresh) return 1;
    if (v < -thresh) return -1;
    return 0;
}

__device__ inline float trit_to_scalar(int8_t t) {
    return (t > 0) ? 1.0f : ((t < 0) ? -1.0f : 0.0f);
}

__device__ inline uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

__device__ inline uint32_t cas_pack_poly_meta(uint32_t symbol_id, uint32_t degree) {
    return ((degree & 0xFFFFu) << 16) | (symbol_id & 0xFFFFu);
}

__device__ inline uint32_t cas_poly_symbol(const StarNode& node) {
    return node.next & 0xFFFFu;
}

__device__ inline uint32_t cas_poly_degree(const StarNode& node) {
    return (node.next >> 16) & 0xFFFFu;
}

__device__ inline uint32_t cas_index_from_scalar(float value) {
    if (value < 0.0f) {
        return CAS_NULL_IDX;
    }
    return static_cast<uint32_t>(floorf(value + 0.5f));
}

__device__ inline uint32_t cas_alloc_node(uint32_t& error) {
    const uint32_t idx = atomicAdd(&g_cas_pool_top, 1u);
    if (idx >= CAS_POOL_SIZE) {
        error = kErrorInvalidArgument;
        return CAS_NULL_IDX;
    }
    return idx;
}

__device__ inline uint32_t cas_alloc_coeffs(uint32_t count, uint32_t& error) {
    const uint32_t offset = atomicAdd(&g_cas_coeff_top, count);
    if (offset >= CAS_COEFF_SIZE || count > (CAS_COEFF_SIZE - offset)) {
        error = kErrorInvalidArgument;
        return CAS_NULL_IDX;
    }
    return offset;
}

__device__ inline uint32_t cas_make_const(float value, uint32_t& error) {
    const uint32_t idx = cas_alloc_node(error);
    if (idx == CAS_NULL_IDX) return idx;
    g_cas_pool[idx].opcode = 0x235u;
    g_cas_pool[idx].flags = STAR_FLAGS(0, TAG_FLOAT, 0);
    g_cas_pool[idx].data.immf32 = value;
    g_cas_pool[idx].next = CAS_NULL_IDX;
    return idx;
}

__device__ inline uint32_t cas_make_symbol(uint32_t symbol_id, uint32_t& error) {
    const uint32_t idx = cas_alloc_node(error);
    if (idx == CAS_NULL_IDX) return idx;
    g_cas_pool[idx].opcode = 0x234u;
    g_cas_pool[idx].flags = STAR_FLAGS(0, TAG_SYMBOL, 0);
    g_cas_pool[idx].data.payload = symbol_id;
    g_cas_pool[idx].next = CAS_NULL_IDX;
    return idx;
}

__device__ inline uint32_t cas_make_unary(uint32_t opcode, uint32_t child_idx, uint32_t& error) {
    const uint32_t idx = cas_alloc_node(error);
    if (idx == CAS_NULL_IDX) return idx;
    g_cas_pool[idx].opcode = opcode;
    g_cas_pool[idx].flags = STAR_FLAGS(1, TAG_FLOAT, 0);
    STAR_CHILD0(g_cas_pool[idx]) = child_idx;
    STAR_CHILD1(g_cas_pool[idx]) = CAS_NULL_IDX;
    g_cas_pool[idx].next = CAS_NULL_IDX;
    return idx;
}

__device__ inline uint32_t cas_make_binary(uint32_t opcode, uint32_t left_idx, uint32_t right_idx, uint32_t& error) {
    const uint32_t idx = cas_alloc_node(error);
    if (idx == CAS_NULL_IDX) return idx;
    g_cas_pool[idx].opcode = opcode;
    g_cas_pool[idx].flags = STAR_FLAGS(2, TAG_FLOAT, 0);
    STAR_CHILD0(g_cas_pool[idx]) = left_idx;
    STAR_CHILD1(g_cas_pool[idx]) = right_idx;
    return idx;
}

__device__ inline uint32_t cas_make_poly(uint32_t symbol_id, const float* coeffs, uint32_t coeff_count, uint32_t& error) {
    if (coeff_count == 0u) {
        return cas_make_const(0.0f, error);
    }
    const uint32_t coeff_offset = cas_alloc_coeffs(coeff_count, error);
    if (coeff_offset == CAS_NULL_IDX) return coeff_offset;
    for (uint32_t idx = 0; idx < coeff_count; ++idx) {
        g_cas_coeffs[coeff_offset + idx] = coeffs[idx];
    }
    const uint32_t node_idx = cas_alloc_node(error);
    if (node_idx == CAS_NULL_IDX) return node_idx;
    g_cas_pool[node_idx].opcode = 0x221u;
    g_cas_pool[node_idx].flags = STAR_FLAGS(1, TAG_POLY, 0);
    g_cas_pool[node_idx].data.payload = coeff_offset;
    g_cas_pool[node_idx].next = cas_pack_poly_meta(symbol_id, coeff_count - 1u);
    return node_idx;
}

__device__ inline bool cas_node_is_const(uint32_t idx, float* value_out = nullptr) {
    if (idx == CAS_NULL_IDX) return false;
    const StarNode& node = g_cas_pool[idx];
    if (node.opcode != 0x235u || STAR_TAG(node.flags) != TAG_FLOAT || STAR_ARITY(node.flags) != 0u) {
        return false;
    }
    if (value_out != nullptr) *value_out = node.data.immf32;
    return true;
}

__device__ inline bool cas_node_is_symbol(uint32_t idx, uint32_t* symbol_out = nullptr) {
    if (idx == CAS_NULL_IDX) return false;
    const StarNode& node = g_cas_pool[idx];
    if (node.opcode != 0x234u || STAR_TAG(node.flags) != TAG_SYMBOL || STAR_ARITY(node.flags) != 0u) {
        return false;
    }
    if (symbol_out != nullptr) *symbol_out = node.data.payload;
    return true;
}

__device__ inline bool cas_same_expr(uint32_t lhs, uint32_t rhs) {
    if (lhs == rhs) return true;
    if (lhs == CAS_NULL_IDX || rhs == CAS_NULL_IDX) return false;
    const StarNode& a = g_cas_pool[lhs];
    const StarNode& b = g_cas_pool[rhs];
    if (a.opcode != b.opcode || a.flags != b.flags || a.next != b.next) return false;
    const uint32_t arity = STAR_ARITY(a.flags);
    if (arity == 0u) {
        if (STAR_TAG(a.flags) == TAG_FLOAT) {
            return fabsf(a.data.immf32 - b.data.immf32) < 1.0e-6f;
        }
        return a.data.payload == b.data.payload;
    }
    if (arity == 1u) return cas_same_expr(STAR_CHILD0(a), STAR_CHILD0(b));
    return cas_same_expr(STAR_CHILD0(a), STAR_CHILD0(b)) &&
           cas_same_expr(STAR_CHILD1(a), STAR_CHILD1(b));
}

__device__ inline bool cas_match_power_of_trig(uint32_t idx, uint32_t trig_opcode, uint32_t* arg_out) {
    if (idx == CAS_NULL_IDX) return false;
    const StarNode& power = g_cas_pool[idx];
    if (power.opcode != 0x0Eu || STAR_ARITY(power.flags) != 2u) return false;
    float exponent = 0.0f;
    if (!cas_node_is_const(STAR_CHILD1(power), &exponent) || fabsf(exponent - 2.0f) > 1.0e-6f) return false;
    const StarNode& trig = g_cas_pool[STAR_CHILD0(power)];
    if (trig.opcode != trig_opcode || STAR_ARITY(trig.flags) != 1u) return false;
    if (arg_out != nullptr) *arg_out = STAR_CHILD0(trig);
    return true;
}

__device__ inline uint32_t cas_simple_simplify(uint32_t root_idx, uint32_t& error) {
    if (root_idx == CAS_NULL_IDX) return root_idx;
    const StarNode& root = g_cas_pool[root_idx];
    const uint32_t arity = STAR_ARITY(root.flags);
    if (arity == 0u || STAR_TAG(root.flags) == TAG_POLY) return root_idx;
    if (arity == 1u && root.opcode == 0xDBu) {
        const uint32_t child_idx = STAR_CHILD0(root);
        if (child_idx != CAS_NULL_IDX && g_cas_pool[child_idx].opcode == 0xDBu) {
            return STAR_CHILD0(g_cas_pool[child_idx]);
        }
        float child_value = 0.0f;
        if (cas_node_is_const(child_idx, &child_value)) {
            return cas_make_const(-child_value, error);
        }
        return root_idx;
    }
    if (arity != 2u) return root_idx;

    const uint32_t lhs_idx = STAR_CHILD0(root);
    const uint32_t rhs_idx = STAR_CHILD1(root);
    float lhs_value = 0.0f;
    float rhs_value = 0.0f;
    const bool lhs_is_const = cas_node_is_const(lhs_idx, &lhs_value);
    const bool rhs_is_const = cas_node_is_const(rhs_idx, &rhs_value);

    if (lhs_is_const && rhs_is_const) {
        switch (root.opcode) {
            case 0x0Au: return cas_make_const(lhs_value + rhs_value, error);
            case 0x0Bu: return cas_make_const(lhs_value - rhs_value, error);
            case 0x0Cu: return cas_make_const(lhs_value * rhs_value, error);
            case 0x0Du: return cas_make_const(fabsf(rhs_value) > 1.0e-8f ? lhs_value / rhs_value : 0.0f, error);
            case 0x0Eu: return cas_make_const(powf(lhs_value, rhs_value), error);
            default: break;
        }
    }

    switch (root.opcode) {
        case 0x0Au: {
            if (lhs_is_const && fabsf(lhs_value) < 1.0e-6f) return rhs_idx;
            if (rhs_is_const && fabsf(rhs_value) < 1.0e-6f) return lhs_idx;
            uint32_t sin_arg = CAS_NULL_IDX;
            uint32_t cos_arg = CAS_NULL_IDX;
            if (cas_match_power_of_trig(lhs_idx, 0x18u, &sin_arg) &&
                cas_match_power_of_trig(rhs_idx, 0x19u, &cos_arg) &&
                cas_same_expr(sin_arg, cos_arg)) {
                return cas_make_const(1.0f, error);
            }
            if (cas_match_power_of_trig(lhs_idx, 0x19u, &cos_arg) &&
                cas_match_power_of_trig(rhs_idx, 0x18u, &sin_arg) &&
                cas_same_expr(sin_arg, cos_arg)) {
                return cas_make_const(1.0f, error);
            }
            break;
        }
        case 0x0Bu:
            if (rhs_is_const && fabsf(rhs_value) < 1.0e-6f) return lhs_idx;
            break;
        case 0x0Cu:
            if ((lhs_is_const && fabsf(lhs_value) < 1.0e-6f) || (rhs_is_const && fabsf(rhs_value) < 1.0e-6f)) {
                return cas_make_const(0.0f, error);
            }
            if (lhs_is_const && fabsf(lhs_value - 1.0f) < 1.0e-6f) return rhs_idx;
            if (rhs_is_const && fabsf(rhs_value - 1.0f) < 1.0e-6f) return lhs_idx;
            break;
        case 0x0Du:
            if (lhs_is_const && fabsf(lhs_value) < 1.0e-6f) return cas_make_const(0.0f, error);
            if (rhs_is_const && fabsf(rhs_value - 1.0f) < 1.0e-6f) return lhs_idx;
            break;
        case 0x0Eu:
            if (rhs_is_const && fabsf(rhs_value - 1.0f) < 1.0e-6f) return lhs_idx;
            if (rhs_is_const && fabsf(rhs_value) < 1.0e-6f) return cas_make_const(1.0f, error);
            break;
        default:
            break;
    }
    return root_idx;
}

__device__ inline void sas_hash_key_fields(const StarNode& node, uint32_t* child0, uint32_t* child1) {
    if (STAR_ARITY(node.flags) == 0u) {
        if (STAR_TAG(node.flags) == TAG_FLOAT) {
            *child0 = __float_as_uint(node.data.immf32);
            *child1 = 0u;
            return;
        }
        *child0 = node.data.payload;
        *child1 = 0u;
        return;
    }
    *child0 = STAR_CHILD0(node);
    *child1 = (STAR_ARITY(node.flags) >= 2u) ? STAR_CHILD1(node) : node.next;
}

__device__ inline uint32_t sas_hashcons_node(uint32_t idx) {
    if (idx == CAS_NULL_IDX) {
        return idx;
    }
    const StarNode& node = g_cas_pool[idx];
    uint32_t child0 = 0u;
    uint32_t child1 = 0u;
    sas_hash_key_fields(node, &child0, &child1);
    const uint32_t existing = hashcons_lookup(g_sas_hashcons, node.opcode, child0, child1, node.flags);
    if (existing != SAS_HASHCONS_EMPTY) {
        return existing;
    }
    return hashcons_insert(g_sas_hashcons, node.opcode, child0, child1, node.flags, idx);
}

__device__ inline int sas_operand_rank(uint32_t idx) {
    if (idx == CAS_NULL_IDX) {
        return 3;
    }
    const uint32_t tag = STAR_TAG(g_cas_pool[idx].flags);
    if (tag == TAG_FLOAT) return 0;
    if (tag == TAG_SYMBOL) return 1;
    return 2;
}

__device__ inline void sas_sort_operands(uint32_t* operands, uint32_t count) {
    for (uint32_t i = 0u; i < count; ++i) {
        for (uint32_t j = i + 1u; j < count; ++j) {
            const int rank_i = sas_operand_rank(operands[i]);
            const int rank_j = sas_operand_rank(operands[j]);
            if (rank_j < rank_i || (rank_i == rank_j && operands[j] < operands[i])) {
                const uint32_t tmp = operands[i];
                operands[i] = operands[j];
                operands[j] = tmp;
            }
        }
    }
}

__device__ inline uint32_t sas_rebuild_commutative_chain(uint32_t root_idx, uint32_t opcode, uint32_t& error) {
    uint32_t pending[kSasBindingCap];
    uint32_t operands[kSasBindingCap];
    uint32_t pending_top = 0u;
    uint32_t operand_count = 0u;
    pending[pending_top++] = root_idx;
    while (pending_top > 0u && operand_count < kSasBindingCap) {
        const uint32_t current = pending[--pending_top];
        if (current == CAS_NULL_IDX) {
            continue;
        }
        const StarNode& node = g_cas_pool[current];
        if (node.opcode == opcode && STAR_ARITY(node.flags) == 2u && pending_top + 2u <= kSasBindingCap) {
            pending[pending_top++] = STAR_CHILD1(node);
            pending[pending_top++] = STAR_CHILD0(node);
        } else {
            operands[operand_count++] = current;
        }
    }
    if (operand_count == 0u) {
        return root_idx;
    }
    sas_sort_operands(operands, operand_count);
    uint32_t current = operands[0];
    for (uint32_t idx = 1u; idx < operand_count; ++idx) {
        current = cas_make_binary(opcode, current, operands[idx], error);
        if (error != kErrorNone || current == CAS_NULL_IDX) {
            return CAS_NULL_IDX;
        }
        current = sas_hashcons_node(current);
    }
    return current;
}

__device__ inline uint32_t sas_find_result(uint32_t key, const uint32_t* keys, const uint32_t* values, uint32_t count) {
    for (uint32_t idx = 0u; idx < count; ++idx) {
        if (keys[idx] == key) {
            return values[idx];
        }
    }
    return CAS_NULL_IDX;
}

__device__ inline uint32_t sas_canonicalize_root(uint32_t root_idx, uint32_t& error) {
    if (root_idx == CAS_NULL_IDX) {
        return root_idx;
    }
    uint32_t stack[kSasTraversalCap];
    uint8_t state[kSasTraversalCap];
    uint32_t result_keys[kSasTraversalCap];
    uint32_t result_values[kSasTraversalCap];
    uint32_t top = 0u;
    uint32_t result_count = 0u;

    stack[top] = root_idx;
    state[top++] = 0u;
    while (top > 0u) {
        const uint32_t idx = stack[top - 1u];
        const uint8_t visit_state = state[top - 1u];
        top -= 1u;
        if (idx == CAS_NULL_IDX) {
            continue;
        }
        const StarNode& node = g_cas_pool[idx];
        const uint32_t arity = STAR_ARITY(node.flags);
        if (visit_state == 0u && arity > 0u && STAR_TAG(node.flags) != TAG_POLY) {
            stack[top] = idx;
            state[top++] = 1u;
            if (arity >= 2u && top < kSasTraversalCap) {
                stack[top] = STAR_CHILD1(node);
                state[top++] = 0u;
            }
            if (top < kSasTraversalCap) {
                stack[top] = STAR_CHILD0(node);
                state[top++] = 0u;
            }
            continue;
        }

        if (arity >= 1u) {
            const uint32_t mapped0 = sas_find_result(STAR_CHILD0(g_cas_pool[idx]), result_keys, result_values, result_count);
            if (mapped0 != CAS_NULL_IDX) {
                STAR_CHILD0(g_cas_pool[idx]) = mapped0;
            }
        }
        if (arity >= 2u) {
            const uint32_t mapped1 = sas_find_result(STAR_CHILD1(g_cas_pool[idx]), result_keys, result_values, result_count);
            if (mapped1 != CAS_NULL_IDX) {
                STAR_CHILD1(g_cas_pool[idx]) = mapped1;
            }
        }

        uint32_t canonical = cas_simple_simplify(idx, error);
        if (error != kErrorNone) {
            return CAS_NULL_IDX;
        }
        if (canonical != CAS_NULL_IDX) {
            const StarNode& canonical_node = g_cas_pool[canonical];
            if (STAR_ARITY(canonical_node.flags) == 2u &&
                (canonical_node.opcode == 0x0Au || canonical_node.opcode == 0x0Cu)) {
                canonical = sas_rebuild_commutative_chain(canonical, canonical_node.opcode, error);
                if (error != kErrorNone) {
                    return CAS_NULL_IDX;
                }
            }
            canonical = cas_simple_simplify(canonical, error);
            if (error != kErrorNone) {
                return CAS_NULL_IDX;
            }
            canonical = sas_hashcons_node(canonical);
        }

        if (result_count < kSasTraversalCap) {
            result_keys[result_count] = idx;
            result_values[result_count] = canonical;
            result_count += 1u;
        }
    }

    const uint32_t mapped_root = sas_find_result(root_idx, result_keys, result_values, result_count);
    return mapped_root == CAS_NULL_IDX ? root_idx : mapped_root;
}

__device__ inline int sas_binding_index(const uint32_t* binding_var_ids, uint32_t binding_count, uint32_t symbol_id) {
    for (uint32_t idx = 0u; idx < binding_count; ++idx) {
        if (binding_var_ids[idx] == symbol_id) {
            return static_cast<int>(idx);
        }
    }
    return -1;
}

__device__ inline bool sas_match_pattern(
    uint32_t pattern_root_idx,
    uint32_t subject_root_idx,
    uint32_t* binding_var_ids,
    uint32_t* binding_subj_idxs,
    uint32_t* binding_count)
{
    uint32_t pattern_stack[kSasTraversalCap];
    uint32_t subject_stack[kSasTraversalCap];
    uint32_t top = 0u;
    pattern_stack[top] = pattern_root_idx;
    subject_stack[top++] = subject_root_idx;
    while (top > 0u) {
        const uint32_t pattern_idx = pattern_stack[--top];
        const uint32_t subject_idx = subject_stack[top];
        if (pattern_idx == CAS_NULL_IDX || subject_idx == CAS_NULL_IDX) {
            return false;
        }

        uint32_t symbol_id = 0u;
        if (cas_node_is_symbol(pattern_idx, &symbol_id)) {
            const int bound_idx = sas_binding_index(binding_var_ids, *binding_count, symbol_id);
            if (bound_idx >= 0) {
                if (!cas_same_expr(binding_subj_idxs[bound_idx], subject_idx)) {
                    return false;
                }
                continue;
            }
            if (*binding_count >= kSasBindingCap) {
                return false;
            }
            binding_var_ids[*binding_count] = symbol_id;
            binding_subj_idxs[*binding_count] = subject_idx;
            *binding_count += 1u;
            continue;
        }

        const StarNode& pattern = g_cas_pool[pattern_idx];
        const StarNode& subject = g_cas_pool[subject_idx];
        if (pattern.opcode != subject.opcode || pattern.flags != subject.flags) {
            return false;
        }
        const uint32_t arity = STAR_ARITY(pattern.flags);
        if (arity == 0u) {
            if (STAR_TAG(pattern.flags) == TAG_FLOAT) {
                if (fabsf(pattern.data.immf32 - subject.data.immf32) > 1.0e-6f) {
                    return false;
                }
            } else if (STAR_TAG(pattern.flags) == TAG_POLY) {
                if (pattern.next != subject.next) {
                    return false;
                }
                const uint32_t coeff_count = (((pattern.next >> 16) & 0xFFFFu) + 1u);
                for (uint32_t idx = 0u; idx < coeff_count; ++idx) {
                    if (fabsf(g_cas_coeffs[pattern.data.payload + idx] - g_cas_coeffs[subject.data.payload + idx]) > 1.0e-6f) {
                        return false;
                    }
                }
            } else if (pattern.data.payload != subject.data.payload) {
                return false;
            }
            continue;
        }
        pattern_stack[top] = STAR_CHILD0(pattern);
        subject_stack[top++] = STAR_CHILD0(subject);
        if (arity >= 2u) {
            pattern_stack[top] = STAR_CHILD1(pattern);
            subject_stack[top++] = STAR_CHILD1(subject);
        }
    }
    return true;
}

__device__ inline uint32_t sas_lookup_binding(
    uint32_t symbol_id,
    const uint32_t* binding_var_ids,
    const uint32_t* binding_subj_idxs,
    uint32_t binding_count)
{
    for (uint32_t idx = 0u; idx < binding_count; ++idx) {
        if (binding_var_ids[idx] == symbol_id) {
            return binding_subj_idxs[idx];
        }
    }
    return CAS_NULL_IDX;
}

__device__ inline uint32_t sas_materialize_template(
    uint32_t root_idx,
    const uint32_t* binding_var_ids,
    const uint32_t* binding_subj_idxs,
    uint32_t binding_count,
    uint32_t& error)
{
    uint32_t stack[kSasTraversalCap];
    uint8_t state[kSasTraversalCap];
    uint32_t result_keys[kSasTraversalCap];
    uint32_t result_values[kSasTraversalCap];
    uint32_t top = 0u;
    uint32_t result_count = 0u;
    stack[top] = root_idx;
    state[top++] = 0u;

    while (top > 0u) {
        const uint32_t idx = stack[top - 1u];
        const uint8_t visit_state = state[top - 1u];
        top -= 1u;
        if (idx == CAS_NULL_IDX) {
            continue;
        }
        const StarNode& node = g_cas_pool[idx];
        const uint32_t arity = STAR_ARITY(node.flags);
        if (visit_state == 0u && arity > 0u && STAR_TAG(node.flags) != TAG_POLY) {
            stack[top] = idx;
            state[top++] = 1u;
            if (arity >= 2u && top < kSasTraversalCap) {
                stack[top] = STAR_CHILD1(node);
                state[top++] = 0u;
            }
            if (top < kSasTraversalCap) {
                stack[top] = STAR_CHILD0(node);
                state[top++] = 0u;
            }
            continue;
        }

        uint32_t materialized = idx;
        uint32_t symbol_id = 0u;
        if (cas_node_is_symbol(idx, &symbol_id)) {
            materialized = sas_lookup_binding(symbol_id, binding_var_ids, binding_subj_idxs, binding_count);
            if (materialized == CAS_NULL_IDX) {
                materialized = cas_make_symbol(symbol_id, error);
            }
        } else if (cas_node_is_const(idx)) {
            materialized = cas_make_const(node.data.immf32, error);
        } else if (arity == 1u) {
            const uint32_t child0 = sas_find_result(STAR_CHILD0(node), result_keys, result_values, result_count);
            materialized = cas_make_unary(node.opcode, child0 == CAS_NULL_IDX ? STAR_CHILD0(node) : child0, error);
        } else if (arity >= 2u) {
            const uint32_t child0 = sas_find_result(STAR_CHILD0(node), result_keys, result_values, result_count);
            const uint32_t child1 = sas_find_result(STAR_CHILD1(node), result_keys, result_values, result_count);
            materialized = cas_make_binary(
                node.opcode,
                child0 == CAS_NULL_IDX ? STAR_CHILD0(node) : child0,
                child1 == CAS_NULL_IDX ? STAR_CHILD1(node) : child1,
                error);
        }
        if (error != kErrorNone || materialized == CAS_NULL_IDX) {
            return CAS_NULL_IDX;
        }

        if (result_count < kSasTraversalCap) {
            result_keys[result_count] = idx;
            result_values[result_count] = materialized;
            result_count += 1u;
        }
    }

    const uint32_t mapped_root = sas_find_result(root_idx, result_keys, result_values, result_count);
    return mapped_root == CAS_NULL_IDX ? root_idx : mapped_root;
}

__device__ inline float cas_eval_node(uint32_t idx, uint32_t symbol_id, float symbol_value) {
    if (idx == CAS_NULL_IDX) return 0.0f;
    const StarNode& node = g_cas_pool[idx];
    if (STAR_ARITY(node.flags) == 0u) {
        if (STAR_TAG(node.flags) == TAG_FLOAT) return node.data.immf32;
        if (STAR_TAG(node.flags) == TAG_SYMBOL) {
            if (node.data.payload == symbol_id) return symbol_value;
            if (node.data.payload == SYM_PI) return 3.1415926535f;
            if (node.data.payload == SYM_E) return 2.7182818284f;
        }
        return 0.0f;
    }
    if (STAR_TAG(node.flags) == TAG_POLY) {
        const uint32_t degree = cas_poly_degree(node);
        float acc = 0.0f;
        for (uint32_t coeff_idx = 0; coeff_idx <= degree; ++coeff_idx) {
            acc = acc * symbol_value + g_cas_coeffs[node.data.payload + coeff_idx];
        }
        return acc;
    }
    const float lhs = cas_eval_node(STAR_CHILD0(node), symbol_id, symbol_value);
    if (STAR_ARITY(node.flags) == 1u) {
        switch (node.opcode) {
            case 0xDBu: return -lhs;
            case 0x18u: return sinf(lhs);
            case 0x19u: return cosf(lhs);
            case 0x15u: return expf(lhs);
            case 0x16u: return lhs > 0.0f ? logf(lhs) : 0.0f;
            default: return lhs;
        }
    }
    const float rhs = cas_eval_node(STAR_CHILD1(node), symbol_id, symbol_value);
    switch (node.opcode) {
        case 0x0Au: return lhs + rhs;
        case 0x0Bu: return lhs - rhs;
        case 0x0Cu: return lhs * rhs;
        case 0x0Du: return fabsf(rhs) > 1.0e-8f ? lhs / rhs : 0.0f;
        case 0x0Eu: return powf(lhs, rhs);
        default: return 0.0f;
    }
}

#include "tex_bake_kernel.cu"
#include "tex_noise_kernels.cu"
#include "tex_filter_kernels.cu"

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

__device__ inline const float* galaxy_entries() {
    return reinterpret_cast<const float*>(g_galaxy_entries_ptr);
}

__device__ inline const float* query_embedding_for_instance(uint32_t instance_id) {
    if (g_query_embedding_ptr == 0ULL) {
        return nullptr;
    }
    return reinterpret_cast<const float*>(g_query_embedding_ptr)
        + (instance_id * g_query_embedding_stride);
}

__device__ inline int rounded_index(float value) {
    return static_cast<int>(floorf(value + 0.5f));
}

__device__ inline bool galaxy_index_valid(int entry_index) {
    return entry_index >= 0 && static_cast<unsigned int>(entry_index) < g_galaxy_entry_count;
}

__device__ inline const float* galaxy_entry_base(int entry_index) {
    return galaxy_entries() + (entry_index * static_cast<int>(g_galaxy_entry_stride));
}

__device__ inline float galaxy_entry_confidence(int entry_index) {
    return galaxy_entry_base(entry_index)[0];
}

__device__ inline float galaxy_cosine_similarity(uint32_t instance_id, int entry_index) {
    const float* query = query_embedding_for_instance(instance_id);
    if (query == nullptr || g_galaxy_entries_ptr == 0ULL || !galaxy_index_valid(entry_index)) {
        return 0.0f;
    }
    const float* entry = galaxy_entry_base(entry_index) + static_cast<int>(g_galaxy_embedding_offset);
    const int dim = static_cast<int>(g_galaxy_embedding_dim);
    float dot = 0.0f;
    float norm_q = 0.0f;
    float norm_e = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float q = query[i];
        float e = entry[i];
        dot += q * e;
        norm_q += q * q;
        norm_e += e * e;
    }
    float denom = sqrtf(norm_q) * sqrtf(norm_e);
    if (denom <= 1e-8f) {
        return 0.0f;
    }
    return dot / denom;
}

__device__ inline PhysicsBodySOA* physics_bodies() {
    return reinterpret_cast<PhysicsBodySOA*>(g_physics_body_soa_ptr);
}

__device__ inline ContactManifoldSOA* physics_manifold() {
    return reinterpret_cast<ContactManifoldSOA*>(g_physics_contact_soa_ptr);
}

__device__ inline CollisionEventQueue* physics_event_queue() {
    return reinterpret_cast<CollisionEventQueue*>(g_physics_event_queue_ptr);
}

__device__ inline PhysicsPredictedSOA* physics_predicted() {
    return reinterpret_cast<PhysicsPredictedSOA*>(g_physics_predicted_soa_ptr);
}

__device__ inline EntityHotPath* entity_hot_paths() {
    return reinterpret_cast<EntityHotPath*>(g_entity_hot_path_ptr);
}

__device__ inline float3 physics_vec_add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 physics_vec_sub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 physics_vec_scale(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ inline float physics_vec_dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float physics_vec_len(float3 a) {
    return sqrtf(physics_vec_dot(a, a));
}

__device__ inline float3 physics_vec_normalize(float3 a) {
    float len = physics_vec_len(a);
    if (len <= 1e-8f) return make_float3(0.0f, 1.0f, 0.0f);
    return physics_vec_scale(a, 1.0f / len);
}

__device__ inline float4 physics_quat_mul(float4 a, float4 b) {
    return make_float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

__device__ inline float4 physics_quat_normalize(float4 q) {
    float norm = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (norm <= 1e-8f) return make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    float inv = 1.0f / norm;
    return make_float4(q.x * inv, q.y * inv, q.z * inv, q.w * inv);
}

#include "entity_behavior.cu"
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
    __shared__ float physics_gravity_y;
    __shared__ uint32_t entity_query_scratch[257];
    __shared__ float cas_poly_coeff_scratch[kCasPolyCoeffScratchMax];
    __shared__ uint32_t cas_poly_coeff_count;
    __shared__ uint32_t sas_binding_var_ids[kSasBindingCap];
    __shared__ uint32_t sas_binding_subj_idxs[kSasBindingCap];
    __shared__ uint32_t sas_binding_count;
    __shared__ uint32_t sas_last_subject_root;
    __shared__ uint32_t sas_last_rule_template;
    __shared__ uint32_t sas_last_rule_strength;
    __shared__ uint32_t sas_last_rule_matched;

    if (tid == 0) {
        stack_size = 0;
        error_code = kErrorNone;
        scalar_index = 0;
        vector_index = 0;
        checkpoint_size = 0;
        checkpoint_valid = 0;
        physics_gravity_y = 0.0f;
        cas_poly_coeff_count = 0u;
        for (uint32_t idx = 0; idx < 257u; ++idx) {
            entity_query_scratch[idx] = 0u;
        }
        for (uint32_t idx = 0; idx < kCasPolyCoeffScratchMax; ++idx) {
            cas_poly_coeff_scratch[idx] = 0.0f;
        }
        sas_binding_count = 0u;
        sas_last_subject_root = CAS_NULL_IDX;
        sas_last_rule_template = CAS_NULL_IDX;
        sas_last_rule_strength = 1u;
        sas_last_rule_matched = 0u;
        for (uint32_t idx = 0; idx < kSasBindingCap; ++idx) {
            sas_binding_var_ids[idx] = kSasEmptyBinding;
            sas_binding_subj_idxs[idx] = CAS_NULL_IDX;
        }
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
                case 0x1A:  // tan
                case 0x26:  // tanh
                case 0x27:  // abs
                case 0x29:  // ceil
                case 0x2B:  // floor
                case 0x2D:  // round
                case 0x39:  // log2
                case 0x3A: {  // log10
                    float value = 0.0f;
                    if (!pop_scalar(stack, stack_size, value, error_code)) break;
                    float result = 0.0f;
                    if (opcode == 0x14) result = sqrtf(value);
                    else if (opcode == 0x15) result = expf(value);
                    else if (opcode == 0x16) result = logf(value);
                    else if (opcode == 0x18) result = sinf(value);
                    else if (opcode == 0x19) result = cosf(value);
                    else if (opcode == 0x1A) result = tanf(value);
                    else if (opcode == 0x26) result = tanhf(value);
                    else if (opcode == 0x27) result = fabsf(value);
                    else if (opcode == 0x29) result = ceilf(value);
                    else if (opcode == 0x2B) result = floorf(value);
                    else if (opcode == 0x2D) result = roundf(value);
                    else if (opcode == 0x39) {
                        if (value <= 0.0f) { error_code = kErrorInvalidArgument; break; }
                        result = log2f(value);
                    } else if (opcode == 0x3A) {
                        if (value <= 0.0f) { error_code = kErrorInvalidArgument; break; }
                        result = log10f(value);
                    } else result = value;
                    push(stack, stack_size, make_scalar(result), error_code);
                    break;
                }
                case 0x38: {  // mod
                    float divisor = 0.0f;
                    float dividend = 0.0f;
                    if (!pop_scalar(stack, stack_size, divisor, error_code)) break;
                    if (!pop_scalar(stack, stack_size, dividend, error_code)) break;
                    if (divisor == 0.0f) { error_code = kErrorInvalidArgument; break; }
                    float result = fmodf(dividend, divisor);
                    push(stack, stack_size, make_scalar(result), error_code);
                    break;
                }
                case 0xAB: {  // gamma
                    float x = 0.0f;
                    if (!pop_scalar(stack, stack_size, x, error_code)) break;
                    if (x <= 0.0f) { error_code = kErrorInvalidArgument; break; }
                    const float g = 7.0f;
                    const float c[9] = {
                        0.99999999999980993f,
                        676.5203681218851f,
                        -1259.1392167224028f,
                        771.32342877765313f,
                        -176.61502916214059f,
                        12.507343278686905f,
                        -0.13857109526572012f,
                        9.9843695780195716e-6f,
                        1.5056327351493116e-7f};
                    float z = x - 1.0f;
                    float sum = c[0];
                    #pragma unroll
                    for (int i = 1; i < 9; ++i) {
                        sum += c[i] / (z + (float)i);
                    }
                    float t = z + g + 0.5f;
                    float result = sqrtf(2.0f * 3.14159265358979323846f) * powf(t, z + 0.5f) * expf(-t) * sum;
                    push(stack, stack_size, make_scalar(result), error_code);
                    break;
                }
                case 0xAC: {  // factorial
                    float n_f = 0.0f;
                    if (!pop_scalar(stack, stack_size, n_f, error_code)) break;
                    int n = (int)n_f;
                    if (n < 0) { error_code = kErrorInvalidArgument; break; }
                    float result = 1.0f;
                    if (n <= 12) {
                        for (int i = 2; i <= n; ++i) result *= (float)i;
                    } else {
                        float nf = (float)n;
                        result = sqrtf(2.0f * 3.14159265358979323846f * nf) * powf(nf / 2.718281828459045f, nf);
                    }
                    push(stack, stack_size, make_scalar(result), error_code);
                    break;
                }
                case 0xAD: {  // binomial
                    float k_f = 0.0f;
                    float n_f = 0.0f;
                    if (!pop_scalar(stack, stack_size, k_f, error_code)) break;
                    if (!pop_scalar(stack, stack_size, n_f, error_code)) break;
                    int n = (int)n_f;
                    int k = (int)k_f;
                    if (n < 0 || k < 0) { error_code = kErrorInvalidArgument; break; }
                    if (k > n) { push(stack, stack_size, make_scalar(0.0f), error_code); break; }
                    if (k > n - k) k = n - k;
                    float result = 1.0f;
                    for (int i = 0; i < k; ++i) {
                        result = result * (float)(n - i) / (float)(i + 1);
                    }
                    push(stack, stack_size, make_scalar(result), error_code);
                    break;
                }
                case 0xAE: {  // beta
                    float b = 0.0f;
                    float a = 0.0f;
                    if (!pop_scalar(stack, stack_size, b, error_code)) break;
                    if (!pop_scalar(stack, stack_size, a, error_code)) break;
                    if (a <= 0.0f || b <= 0.0f) { error_code = kErrorInvalidArgument; break; }
                    float result = expf(lgammaf(a) + lgammaf(b) - lgammaf(a + b));
                    push(stack, stack_size, make_scalar(result), error_code);
                    break;
                }
                case 0xD8: {  // gcd
                    float b_f = 0.0f;
                    float a_f = 0.0f;
                    if (!pop_scalar(stack, stack_size, b_f, error_code)) break;
                    if (!pop_scalar(stack, stack_size, a_f, error_code)) break;
                    int a = (int)fabsf(a_f);
                    int b = (int)fabsf(b_f);
                    while (b != 0) {
                        int t = b;
                        b = a % b;
                        a = t;
                    }
                    push(stack, stack_size, make_scalar((float)a), error_code);
                    break;
                }
                case 0xDB: {  // neg (alias)
                    float x = 0.0f;
                    if (!pop_scalar(stack, stack_size, x, error_code)) break;
                    push(stack, stack_size, make_scalar(-x), error_code);
                    break;
                }
                case 0xDC: {  // gte
                    float rhs = 0.0f;
                    float lhs = 0.0f;
                    if (!pop_scalar(stack, stack_size, rhs, error_code)) break;
                    if (!pop_scalar(stack, stack_size, lhs, error_code)) break;
                    push(stack, stack_size, make_scalar(lhs >= rhs ? 1.0f : 0.0f), error_code);
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
                    push(stack, stack_size, make_vector(vec.x, vec.y, vec.z), error_code);
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
                    push(stack, stack_size, make_vector(audio.x, audio.y, audio.z), error_code);
                    break;
                }
                case 0x53: {  // OP_MODALITY_FUSE
                    float3 b{};
                    float3 a{};
                    if (!pop_vector(stack, stack_size, b, error_code)) break;
                    if (!pop_vector(stack, stack_size, a, error_code)) break;
                    float3 fused = make_float3(
                        0.5f * (a.x + b.x),
                        0.5f * (a.y + b.y),
                        0.5f * (a.z + b.z));
                    push(stack, stack_size, make_vector(fused.x, fused.y, fused.z), error_code);
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
                    push(stack, stack_size, make_vector(proto.x, proto.y, proto.z), error_code);
                    break;
                }
                case 0x41: {  // OP_DELTA_APPLY
                    float3 delta{};
                    float3 base{};
                    if (!pop_vector(stack, stack_size, delta, error_code)) break;
                    if (!pop_vector(stack, stack_size, base, error_code)) break;
                    float3 result = make_float3(base.x + delta.x, base.y + delta.y, base.z + delta.z);
                    push(stack, stack_size, make_vector(result.x, result.y, result.z), error_code);
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
                    push(stack, stack_size, make_vector(fused.x, fused.y, fused.z), error_code);
                    break;
                }
                case 0x43: {  // sigmoid approximation
                    float value = 0.0f;
                    if (!pop_scalar(stack, stack_size, value, error_code)) break;
                    float sig = 0.5f * (1.0f + tanhf(0.5f * value));
                    push(stack, stack_size, make_scalar(sig), error_code);
                    break;
                }
                case 0xE0: {  // LOAD_GALAXY
                    float entry_index_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, entry_index_scalar, error_code)) break;
                    int entry_index = rounded_index(entry_index_scalar);
                    if (g_galaxy_entries_ptr == 0ULL || !galaxy_index_valid(entry_index)) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const float* entry = galaxy_entry_base(entry_index);
                    push(stack, stack_size, make_scalar(entry[0]), error_code);  // confidence
                    if (error_code != kErrorNone) break;
                    push(stack, stack_size, make_scalar(entry[1]), error_code);  // domain hash
                    if (error_code != kErrorNone) break;
                    push(stack, stack_size, make_scalar(entry[2]), error_code);  // subject hash
                    if (error_code != kErrorNone) break;
                    push(stack, stack_size, make_scalar(static_cast<float>(entry_index)), error_code);
                    break;
                }
                case 0xE1: {  // GALAXY_SIMILARITY
                    float entry_index_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, entry_index_scalar, error_code)) break;
                    int entry_index = rounded_index(entry_index_scalar);
                    if (g_galaxy_entries_ptr == 0ULL || !galaxy_index_valid(entry_index) || g_query_embedding_ptr == 0ULL) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    float similarity = galaxy_cosine_similarity(instance_id, entry_index);
                    push(stack, stack_size, make_scalar(similarity), error_code);
                    break;
                }
                case 0xE2: {  // GALAXY_SCAN
                    float requested_k_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, requested_k_scalar, error_code)) break;
                    if (g_galaxy_entries_ptr == 0ULL || g_query_embedding_ptr == 0ULL) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    int requested_k = rounded_index(requested_k_scalar);
                    if (requested_k <= 0) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    int available_stack = static_cast<int>(kStackCapacity - stack_size - 1);
                    if (available_stack <= 0) {
                        error_code = kErrorStackOverflow;
                        break;
                    }
                    int k = requested_k;
                    if (k > static_cast<int>(g_galaxy_entry_count)) k = static_cast<int>(g_galaxy_entry_count);
                    if (k > kGalaxyTopKMax) k = kGalaxyTopKMax;
                    if (k > available_stack) k = available_stack;
                    if (k <= 0) {
                        push(stack, stack_size, make_scalar(0.0f), error_code);
                        break;
                    }

                    float top_scores[kGalaxyTopKMax];
                    int top_indices[kGalaxyTopKMax];
                    for (int pos = 0; pos < kGalaxyTopKMax; ++pos) {
                        top_scores[pos] = -2.0f;
                        top_indices[pos] = -1;
                    }

                    for (int entry_index = 0; entry_index < static_cast<int>(g_galaxy_entry_count); ++entry_index) {
                        float score = galaxy_cosine_similarity(instance_id, entry_index);
                        for (int pos = 0; pos < k; ++pos) {
                            if (score > top_scores[pos]) {
                                for (int shift = k - 1; shift > pos; --shift) {
                                    top_scores[shift] = top_scores[shift - 1];
                                    top_indices[shift] = top_indices[shift - 1];
                                }
                                top_scores[pos] = score;
                                top_indices[pos] = entry_index;
                                break;
                            }
                        }
                    }

                    int actual_count = 0;
                    for (int pos = 0; pos < k; ++pos) {
                        if (top_indices[pos] >= 0) {
                            actual_count += 1;
                        }
                    }
                    // Push worst-to-best so the best candidate sits on top after dropping count.
                    for (int pos = actual_count - 1; pos >= 0; --pos) {
                        push(stack, stack_size, make_scalar(static_cast<float>(top_indices[pos])), error_code);
                        if (error_code != kErrorNone) break;
                    }
                    if (error_code != kErrorNone) break;
                    push(stack, stack_size, make_scalar(static_cast<float>(actual_count)), error_code);
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
                        push(stack, stack_size, make_vector(sum.x, sum.y, sum.z), error_code);
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
                    push(stack, stack_size, make_vector(entangled.x, entangled.y, entangled.z), error_code);
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
                    push(stack, stack_size, make_vector(state.x, state.y, state.z), error_code);
                    break;
                }
                case 0x70: {  // OP_TADD
                    float b = 0.0f, a = 0.0f;
                    if (!pop_scalar(stack, stack_size, b, error_code)) break;
                    if (!pop_scalar(stack, stack_size, a, error_code)) break;
                    int8_t t = clamp_trit(a + b);
                    push(stack, stack_size, make_scalar(trit_to_scalar(t)), error_code);
                    break;
                }
                case 0x71: {  // OP_TMUL
                    float b = 0.0f, a = 0.0f;
                    if (!pop_scalar(stack, stack_size, b, error_code)) break;
                    if (!pop_scalar(stack, stack_size, a, error_code)) break;
                    int8_t t = clamp_trit(a * b);
                    push(stack, stack_size, make_scalar(trit_to_scalar(t)), error_code);
                    break;
                }
                case 0x72: {  // OP_TNOT (negate trit)
                    float a = 0.0f;
                    if (!pop_scalar(stack, stack_size, a, error_code)) break;
                    int8_t t = clamp_trit(-a);
                    push(stack, stack_size, make_scalar(trit_to_scalar(t)), error_code);
                    break;
                }
                case 0x73: {  // OP_TCOMP (sign(a-b))
                    float b = 0.0f, a = 0.0f;
                    if (!pop_scalar(stack, stack_size, b, error_code)) break;
                    if (!pop_scalar(stack, stack_size, a, error_code)) break;
                    int8_t t = clamp_trit(a - b, 1e-6f);
                    push(stack, stack_size, make_scalar(trit_to_scalar(t)), error_code);
                    break;
                }
                case 0x74: {  // OP_TQUANT (quantize float to trit with 0.33 threshold)
                    float a = 0.0f;
                    if (!pop_scalar(stack, stack_size, a, error_code)) break;
                    int8_t t = clamp_trit(a);
                    push(stack, stack_size, make_scalar(trit_to_scalar(t)), error_code);
                    break;
                }
                case 0x75: {  // OP_TPACK (pack two trits into scalar bits)
                    float b = 0.0f, a = 0.0f;
                    if (!pop_scalar(stack, stack_size, b, error_code)) break;
                    if (!pop_scalar(stack, stack_size, a, error_code)) break;
                    uint32_t ea = static_cast<uint32_t>(clamp_trit(a) == 1 ? 2 : (clamp_trit(a) == 0 ? 1 : 0));
                    uint32_t eb = static_cast<uint32_t>(clamp_trit(b) == 1 ? 2 : (clamp_trit(b) == 0 ? 1 : 0));
                    uint32_t packed = (ea & 0x3u) | ((eb & 0x3u) << 2);
                    float as_float = __uint_as_float(packed);
                    push(stack, stack_size, make_scalar(as_float), error_code);
                    break;
                }
                case 0x76: {  // OP_TUNPACK (unpack two trits from scalar bits)
                    float packed_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, packed_scalar, error_code)) break;
                    uint32_t bits = __float_as_uint(packed_scalar);
                    uint32_t a_bits = bits & 0x3u;
                    uint32_t b_bits = (bits >> 2) & 0x3u;
                    int8_t t0 = (a_bits == 2 ? 1 : (a_bits == 1 ? 0 : -1));
                    int8_t t1 = (b_bits == 2 ? 1 : (b_bits == 1 ? 0 : -1));
                    push(stack, stack_size, make_scalar(trit_to_scalar(t0)), error_code);
                    push(stack, stack_size, make_scalar(trit_to_scalar(t1)), error_code);
                    break;
                }
                case 0x60: {  // OP_CHECKPOINT — see rpn_opcodes.py
                    checkpoint_size = stack_size;
                    for (uint32_t idx = 0; idx < stack_size && idx < kStackCapacity; ++idx) {
                        checkpoint_stack[idx] = stack[idx];
                    }
                    checkpoint_valid = 1;
                    break;
                }
                case 0x61: {  // OP_ROLLBACK — see rpn_opcodes.py
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
                case 0x62: {  // OP_VERIFY — see rpn_opcodes.py
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
                case 0xEA: {  // OP_VAR_X
                    push(stack, stack_size, make_scalar(static_cast<float>(SYM_X)), error_code);
                    break;
                }
                case 0xEB: {  // OP_VAR_Y
                    push(stack, stack_size, make_scalar(static_cast<float>(SYM_Y)), error_code);
                    break;
                }
                case 0xEC: {  // OP_VAR_Z
                    push(stack, stack_size, make_scalar(static_cast<float>(SYM_Z)), error_code);
                    break;
                }
                case 0xED: {  // OP_VAR_W
                    push(stack, stack_size, make_scalar(static_cast<float>(SYM_W)), error_code);
                    break;
                }
                case 0xEE: {  // OP_CONST
                    float value = scalars ? scalars[scalar_index] : 0.0f;
                    scalar_index += 1;
                    push(stack, stack_size, make_scalar(value), error_code);
                    break;
                }
                case 0x220: {  // OP_POLY_COEFF
                    float coeff = 0.0f;
                    if (!pop_scalar(stack, stack_size, coeff, error_code)) break;
                    if (cas_poly_coeff_count >= kCasPolyCoeffScratchMax) {
                        error_code = kErrorStackOverflow;
                        break;
                    }
                    cas_poly_coeff_scratch[cas_poly_coeff_count++] = coeff;
                    break;
                }
                case 0x221: {  // OP_POLY_BUILD
                    float symbol_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, symbol_scalar, error_code)) break;
                    if (cas_poly_coeff_count == 0u) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const uint32_t symbol_id = static_cast<uint32_t>(max(0.0f, floorf(symbol_scalar + 0.5f)));
                    const uint32_t node_idx = cas_make_poly(
                        symbol_id,
                        cas_poly_coeff_scratch,
                        cas_poly_coeff_count,
                        error_code);
                    if (error_code != kErrorNone) break;
                    cas_poly_coeff_count = 0u;
                    push(stack, stack_size, make_scalar(static_cast<float>(node_idx)), error_code);
                    break;
                }
                case 0x222: {  // OP_POLY_ADD
                    float rhs_scalar = 0.0f;
                    float lhs_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, rhs_scalar, error_code)) break;
                    if (!pop_scalar(stack, stack_size, lhs_scalar, error_code)) break;
                    const uint32_t rhs_idx = cas_index_from_scalar(rhs_scalar);
                    const uint32_t lhs_idx = cas_index_from_scalar(lhs_scalar);
                    if (lhs_idx == CAS_NULL_IDX || rhs_idx == CAS_NULL_IDX) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const StarNode& lhs_node = g_cas_pool[lhs_idx];
                    const StarNode& rhs_node = g_cas_pool[rhs_idx];
                    if (STAR_TAG(lhs_node.flags) != TAG_POLY || STAR_TAG(rhs_node.flags) != TAG_POLY ||
                        cas_poly_symbol(lhs_node) != cas_poly_symbol(rhs_node)) {
                        error_code = kErrorTypeMismatch;
                        break;
                    }
                    const uint32_t degree = max(cas_poly_degree(lhs_node), cas_poly_degree(rhs_node));
                    if (degree + 1u > kCasPolyCoeffScratchMax) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    float coeffs[kCasPolyCoeffScratchMax];
                    for (uint32_t idx = 0; idx <= degree; ++idx) {
                        const uint32_t power = degree - idx;
                        float lhs_coeff = 0.0f;
                        float rhs_coeff = 0.0f;
                        const uint32_t lhs_degree = cas_poly_degree(lhs_node);
                        const uint32_t rhs_degree = cas_poly_degree(rhs_node);
                        if (power <= lhs_degree) {
                            lhs_coeff = g_cas_coeffs[lhs_node.data.payload + (lhs_degree - power)];
                        }
                        if (power <= rhs_degree) {
                            rhs_coeff = g_cas_coeffs[rhs_node.data.payload + (rhs_degree - power)];
                        }
                        coeffs[idx] = lhs_coeff + rhs_coeff;
                    }
                    const uint32_t out_idx = cas_make_poly(cas_poly_symbol(lhs_node), coeffs, degree + 1u, error_code);
                    if (error_code != kErrorNone) break;
                    push(stack, stack_size, make_scalar(static_cast<float>(out_idx)), error_code);
                    break;
                }
                case 0x223: {  // OP_POLY_MUL
                    float rhs_scalar = 0.0f;
                    float lhs_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, rhs_scalar, error_code)) break;
                    if (!pop_scalar(stack, stack_size, lhs_scalar, error_code)) break;
                    const uint32_t rhs_idx = cas_index_from_scalar(rhs_scalar);
                    const uint32_t lhs_idx = cas_index_from_scalar(lhs_scalar);
                    if (lhs_idx == CAS_NULL_IDX || rhs_idx == CAS_NULL_IDX) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const StarNode& lhs_node = g_cas_pool[lhs_idx];
                    const StarNode& rhs_node = g_cas_pool[rhs_idx];
                    if (STAR_TAG(lhs_node.flags) != TAG_POLY || STAR_TAG(rhs_node.flags) != TAG_POLY ||
                        cas_poly_symbol(lhs_node) != cas_poly_symbol(rhs_node)) {
                        error_code = kErrorTypeMismatch;
                        break;
                    }
                    const uint32_t lhs_degree = cas_poly_degree(lhs_node);
                    const uint32_t rhs_degree = cas_poly_degree(rhs_node);
                    const uint32_t coeff_count = lhs_degree + rhs_degree + 1u;
                    if (coeff_count > kCasPolyCoeffScratchMax) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    float coeffs[kCasPolyCoeffScratchMax];
                    for (uint32_t idx = 0; idx < coeff_count; ++idx) coeffs[idx] = 0.0f;
                    for (uint32_t lhs_coeff_idx = 0; lhs_coeff_idx <= lhs_degree; ++lhs_coeff_idx) {
                        for (uint32_t rhs_coeff_idx = 0; rhs_coeff_idx <= rhs_degree; ++rhs_coeff_idx) {
                            coeffs[lhs_coeff_idx + rhs_coeff_idx] +=
                                g_cas_coeffs[lhs_node.data.payload + lhs_coeff_idx] *
                                g_cas_coeffs[rhs_node.data.payload + rhs_coeff_idx];
                        }
                    }
                    const uint32_t out_idx = cas_make_poly(cas_poly_symbol(lhs_node), coeffs, coeff_count, error_code);
                    if (error_code != kErrorNone) break;
                    push(stack, stack_size, make_scalar(static_cast<float>(out_idx)), error_code);
                    break;
                }
                case 0x224:  // OP_POLY_DIV
                case 0x225:  // OP_POLY_REM
                case 0x226:  // OP_POLY_GCD
                case 0x227: {  // OP_POLY_FACTOR
                    float maybe_rhs = 0.0f;
                    float maybe_lhs = 0.0f;
                    if (!pop_scalar(stack, stack_size, maybe_rhs, error_code)) break;
                    if ((opcode == 0x224 || opcode == 0x225 || opcode == 0x226) &&
                        !pop_scalar(stack, stack_size, maybe_lhs, error_code)) break;
                    push(stack, stack_size, make_scalar(0.0f), error_code);
                    break;
                }
                case 0x228: {  // OP_SIMPLIFY
                    float expr_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, expr_scalar, error_code)) break;
                    const uint32_t expr_idx = cas_index_from_scalar(expr_scalar);
                    const uint32_t simplified = cas_simple_simplify(expr_idx, error_code);
                    if (error_code != kErrorNone) break;
                    push(stack, stack_size, make_scalar(static_cast<float>(simplified)), error_code);
                    break;
                }
                case 0x229: {  // OP_SUBSTITUTE
                    float value_scalar = 0.0f;
                    float symbol_scalar = 0.0f;
                    float expr_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, value_scalar, error_code)) break;
                    if (!pop_scalar(stack, stack_size, symbol_scalar, error_code)) break;
                    if (!pop_scalar(stack, stack_size, expr_scalar, error_code)) break;
                    const uint32_t expr_idx = cas_index_from_scalar(expr_scalar);
                    const uint32_t symbol_id = static_cast<uint32_t>(max(0.0f, floorf(symbol_scalar + 0.5f)));
                    uint32_t expr_symbol = 0u;
                    uint32_t out_idx = expr_idx;
                    if (cas_node_is_symbol(expr_idx, &expr_symbol) && expr_symbol == symbol_id) {
                        out_idx = cas_make_const(value_scalar, error_code);
                    }
                    if (error_code != kErrorNone) break;
                    push(stack, stack_size, make_scalar(static_cast<float>(out_idx)), error_code);
                    break;
                }
                case 0x22A:  // OP_COLLECT
                case 0x22B:  // OP_RATIONALIZE
                case 0x22C:  // OP_TRIG_SIMPLIFY
                case 0x22D: {  // OP_LOG_SIMPLIFY
                    float maybe_var = 0.0f;
                    float expr_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, maybe_var, error_code)) break;
                    if ((opcode == 0x22A || opcode == 0x22B) &&
                        !pop_scalar(stack, stack_size, expr_scalar, error_code)) break;
                    const float out_value = (opcode == 0x22C || opcode == 0x22D) ? maybe_var : expr_scalar;
                    push(stack, stack_size, make_scalar(out_value), error_code);
                    break;
                }
                case 0x22E: {  // OP_SOLVE_LINEAR
                    float b = 0.0f;
                    float a = 0.0f;
                    if (!pop_scalar(stack, stack_size, b, error_code)) break;
                    if (!pop_scalar(stack, stack_size, a, error_code)) break;
                    const float root = fabsf(a) > 1.0e-8f ? (-b / a) : 0.0f;
                    push(stack, stack_size, make_scalar(root), error_code);
                    break;
                }
                case 0x22F: {  // OP_SOLVE_QUADRATIC
                    float c = 0.0f;
                    float b = 0.0f;
                    float a = 0.0f;
                    if (!pop_scalar(stack, stack_size, c, error_code)) break;
                    if (!pop_scalar(stack, stack_size, b, error_code)) break;
                    if (!pop_scalar(stack, stack_size, a, error_code)) break;
                    if (fabsf(a) <= 1.0e-8f) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const float disc = b * b - 4.0f * a * c;
                    const float sqrt_disc = disc >= 0.0f ? sqrtf(disc) : 0.0f;
                    push(stack, stack_size, make_scalar((-b - sqrt_disc) / (2.0f * a)), error_code);
                    if (error_code != kErrorNone) break;
                    push(stack, stack_size, make_scalar((-b + sqrt_disc) / (2.0f * a)), error_code);
                    break;
                }
                case 0x230: {  // OP_LINSOLVE
                    float matrix_ptr_scalar = 0.0f;
                    float n_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, matrix_ptr_scalar, error_code)) break;
                    if (!pop_scalar(stack, stack_size, n_scalar, error_code)) break;
                    (void)matrix_ptr_scalar;
                    (void)n_scalar;
                    push(stack, stack_size, make_scalar(0.0f), error_code);
                    break;
                }
                case 0x231: {  // OP_PATTERN_MATCH
                    float pattern_scalar = 0.0f;
                    float expr_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, pattern_scalar, error_code)) break;
                    if (!pop_scalar(stack, stack_size, expr_scalar, error_code)) break;
                    const uint32_t pattern_idx = cas_index_from_scalar(pattern_scalar);
                    const uint32_t expr_idx = cas_index_from_scalar(expr_scalar);
                    push(stack, stack_size, make_scalar(cas_same_expr(expr_idx, pattern_idx) ? 1.0f : 0.0f), error_code);
                    break;
                }
                case 0x232: {  // OP_RULE_APPLY
                    float rule_scalar = 0.0f;
                    float expr_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, rule_scalar, error_code)) break;
                    if (!pop_scalar(stack, stack_size, expr_scalar, error_code)) break;
                    (void)rule_scalar;
                    push(stack, stack_size, make_scalar(expr_scalar), error_code);
                    break;
                }
                case 0x233: {  // OP_COEFF_EXTRACT
                    float power_scalar = 0.0f;
                    float symbol_scalar = 0.0f;
                    float poly_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, power_scalar, error_code)) break;
                    if (!pop_scalar(stack, stack_size, symbol_scalar, error_code)) break;
                    if (!pop_scalar(stack, stack_size, poly_scalar, error_code)) break;
                    const uint32_t poly_idx = cas_index_from_scalar(poly_scalar);
                    if (poly_idx == CAS_NULL_IDX || STAR_TAG(g_cas_pool[poly_idx].flags) != TAG_POLY) {
                        error_code = kErrorTypeMismatch;
                        break;
                    }
                    const StarNode& poly_node = g_cas_pool[poly_idx];
                    const uint32_t symbol_id = static_cast<uint32_t>(max(0.0f, floorf(symbol_scalar + 0.5f)));
                    const uint32_t power = static_cast<uint32_t>(max(0.0f, floorf(power_scalar + 0.5f)));
                    float coeff = 0.0f;
                    const uint32_t degree = cas_poly_degree(poly_node);
                    if (cas_poly_symbol(poly_node) == symbol_id && power <= degree) {
                        coeff = g_cas_coeffs[poly_node.data.payload + (degree - power)];
                    }
                    push(stack, stack_size, make_scalar(coeff), error_code);
                    break;
                }
                case 0x234: {  // OP_CAS_PUSH_SYM
                    float symbol_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, symbol_scalar, error_code)) break;
                    const uint32_t symbol_id = static_cast<uint32_t>(max(0.0f, floorf(symbol_scalar + 0.5f)));
                    const uint32_t node_idx = cas_make_symbol(symbol_id, error_code);
                    if (error_code != kErrorNone) break;
                    push(stack, stack_size, make_scalar(static_cast<float>(node_idx)), error_code);
                    break;
                }
                case 0x235: {  // OP_CAS_PUSH_CONST
                    float const_value = 0.0f;
                    if (!pop_scalar(stack, stack_size, const_value, error_code)) break;
                    const uint32_t node_idx = cas_make_const(const_value, error_code);
                    if (error_code != kErrorNone) break;
                    push(stack, stack_size, make_scalar(static_cast<float>(node_idx)), error_code);
                    break;
                }
                case 0x236: {  // OP_CAS_BUILD
                    float arity_scalar = 0.0f;
                    float opcode_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, arity_scalar, error_code)) break;
                    if (!pop_scalar(stack, stack_size, opcode_scalar, error_code)) break;
                    const uint32_t arity = static_cast<uint32_t>(max(0.0f, floorf(arity_scalar + 0.5f)));
                    const uint32_t cas_opcode = static_cast<uint32_t>(max(0.0f, floorf(opcode_scalar + 0.5f)));
                    uint32_t node_idx = CAS_NULL_IDX;
                    if (arity == 1u) {
                        float child_scalar = 0.0f;
                        if (!pop_scalar(stack, stack_size, child_scalar, error_code)) break;
                        node_idx = cas_make_unary(cas_opcode, cas_index_from_scalar(child_scalar), error_code);
                    } else if (arity == 2u) {
                        float rhs_scalar = 0.0f;
                        float lhs_scalar = 0.0f;
                        if (!pop_scalar(stack, stack_size, rhs_scalar, error_code)) break;
                        if (!pop_scalar(stack, stack_size, lhs_scalar, error_code)) break;
                        node_idx = cas_make_binary(
                            cas_opcode,
                            cas_index_from_scalar(lhs_scalar),
                            cas_index_from_scalar(rhs_scalar),
                            error_code);
                    } else {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    if (error_code != kErrorNone) break;
                    push(stack, stack_size, make_scalar(static_cast<float>(node_idx)), error_code);
                    break;
                }
                case 0x237: {  // OP_CAS_EVAL
                    float value_scalar = 0.0f;
                    float symbol_scalar = 0.0f;
                    float expr_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, value_scalar, error_code)) break;
                    if (!pop_scalar(stack, stack_size, symbol_scalar, error_code)) break;
                    if (!pop_scalar(stack, stack_size, expr_scalar, error_code)) break;
                    const uint32_t expr_idx = cas_index_from_scalar(expr_scalar);
                    const uint32_t symbol_id = static_cast<uint32_t>(max(0.0f, floorf(symbol_scalar + 0.5f)));
                    push(stack, stack_size, make_scalar(cas_eval_node(expr_idx, symbol_id, value_scalar)), error_code);
                    break;
                }
                case 0x238: {  // OP_CANONICALIZE
                    float expr_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, expr_scalar, error_code)) break;
                    const uint32_t expr_idx = cas_index_from_scalar(expr_scalar);
                    const uint32_t canonical_idx = sas_canonicalize_root(expr_idx, error_code);
                    if (error_code != kErrorNone) break;
                    push(stack, stack_size, make_scalar(static_cast<float>(canonical_idx)), error_code);
                    break;
                }
                case 0x239: {  // OP_CAS_HASH
                    float expr_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, expr_scalar, error_code)) break;
                    const uint32_t expr_idx = cas_index_from_scalar(expr_scalar);
                    uint32_t hash_value = 0u;
                    if (expr_idx < CAS_POOL_SIZE) {
                        const StarNode& node = g_cas_pool[expr_idx];
                        const uint32_t child0 = (STAR_ARITY(node.flags) == 0u && STAR_TAG(node.flags) == TAG_FLOAT)
                            ? __float_as_uint(node.data.immf32)
                            : ((STAR_ARITY(node.flags) == 0u) ? node.data.payload : STAR_CHILD0(node));
                        const uint32_t child1 = (STAR_ARITY(node.flags) >= 2u) ? STAR_CHILD1(node) : node.next;
                        hash_value = hashcons_slot(node.opcode, child0, child1, node.flags);
                    }
                    push(stack, stack_size, make_scalar(__uint_as_float(hash_value)), error_code);
                    break;
                }
                case 0x23A: {  // OP_SEMANTIC_RESOLVE
                    float symbol_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, symbol_scalar, error_code)) break;
                    const uint32_t symbol_id = static_cast<uint32_t>(max(0.0f, floorf(symbol_scalar + 0.5f)));
                    const float resolved = (symbol_id < 256u) ? g_sas_symbol_values[symbol_id] : 0.0f;
                    push(stack, stack_size, make_scalar(resolved), error_code);
                    break;
                }
                case 0x23B: {  // OP_RULE_SELECT
                    float n_candidates_scalar = 0.0f;
                    float candidate_base_scalar = 0.0f;
                    float subject_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, n_candidates_scalar, error_code)) break;
                    if (!pop_scalar(stack, stack_size, candidate_base_scalar, error_code)) break;
                    if (!pop_scalar(stack, stack_size, subject_scalar, error_code)) break;
                    const uint32_t n_candidates = static_cast<uint32_t>(max(0.0f, floorf(n_candidates_scalar + 0.5f)));
                    const uint32_t candidate_base = cas_index_from_scalar(candidate_base_scalar);
                    const uint32_t subject_root = cas_index_from_scalar(subject_scalar);
                    sas_binding_count = 0u;
                    sas_last_subject_root = subject_root;
                    sas_last_rule_template = CAS_NULL_IDX;
                    sas_last_rule_strength = 1u;
                    sas_last_rule_matched = 0u;
                    for (uint32_t idx = 0u; idx < kSasBindingCap; ++idx) {
                        sas_binding_var_ids[idx] = kSasEmptyBinding;
                        sas_binding_subj_idxs[idx] = CAS_NULL_IDX;
                    }
                    for (uint32_t candidate_idx = 0u; candidate_idx < n_candidates; ++candidate_idx) {
                        const uint32_t pattern_root = candidate_base + candidate_idx * 2u;
                        const uint32_t replacement_root = pattern_root + 1u;
                        if (replacement_root >= CAS_POOL_SIZE) {
                            break;
                        }
                        uint32_t local_binding_vars[kSasBindingCap];
                        uint32_t local_binding_subjects[kSasBindingCap];
                        uint32_t local_binding_count = 0u;
                        for (uint32_t bind_idx = 0u; bind_idx < kSasBindingCap; ++bind_idx) {
                            local_binding_vars[bind_idx] = kSasEmptyBinding;
                            local_binding_subjects[bind_idx] = CAS_NULL_IDX;
                        }
                        if (!sas_match_pattern(
                                pattern_root,
                                subject_root,
                                local_binding_vars,
                                local_binding_subjects,
                                &local_binding_count)) {
                            continue;
                        }
                        sas_binding_count = local_binding_count;
                        for (uint32_t bind_idx = 0u; bind_idx < local_binding_count; ++bind_idx) {
                            sas_binding_var_ids[bind_idx] = local_binding_vars[bind_idx];
                            sas_binding_subj_idxs[bind_idx] = local_binding_subjects[bind_idx];
                        }
                        sas_last_rule_template = replacement_root;
                        sas_last_rule_matched = 1u;
                        break;
                    }
                    const float matched_root = sas_last_rule_matched != 0u
                        ? static_cast<float>(sas_last_rule_template)
                        : -1.0f;
                    push(stack, stack_size, make_scalar(matched_root), error_code);
                    break;
                }
                case 0x23C: {  // OP_CONTEXTUAL_REWRITE
                    float template_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, template_scalar, error_code)) break;
                    uint32_t template_root = cas_index_from_scalar(template_scalar);
                    if (template_root == CAS_NULL_IDX) {
                        template_root = sas_last_rule_template;
                    }
                    if (template_root == CAS_NULL_IDX || sas_last_rule_matched == 0u) {
                        push(
                            stack,
                            stack_size,
                            make_scalar(
                                sas_last_subject_root == CAS_NULL_IDX
                                    ? -1.0f
                                    : static_cast<float>(sas_last_subject_root)),
                            error_code);
                        break;
                    }
                    uint32_t rewritten_root = sas_materialize_template(
                        template_root,
                        sas_binding_var_ids,
                        sas_binding_subj_idxs,
                        sas_binding_count,
                        error_code);
                    if (error_code != kErrorNone || rewritten_root == CAS_NULL_IDX) break;
                    rewritten_root = sas_canonicalize_root(rewritten_root, error_code);
                    if (error_code != kErrorNone) break;
                    push(stack, stack_size, make_scalar(static_cast<float>(rewritten_root)), error_code);
                    break;
                }
                case 0x23D: {  // OP_SEMANTIC_EQUIV
                    float rhs_scalar = 0.0f;
                    float lhs_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, rhs_scalar, error_code)) break;
                    if (!pop_scalar(stack, stack_size, lhs_scalar, error_code)) break;
                    const uint32_t rhs_idx = cas_index_from_scalar(rhs_scalar);
                    const uint32_t lhs_idx = cas_index_from_scalar(lhs_scalar);
                    push(stack, stack_size, make_scalar(lhs_idx == rhs_idx ? 1.0f : 0.0f), error_code);
                    break;
                }
                case 0x150: {  // PH_BROAD_PHASE
                    float body_count_scalar = 0.0f;
                    float dt_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, body_count_scalar, error_code)) break;
                    if (!pop_scalar(stack, stack_size, dt_scalar, error_code)) break;
                    PhysicsBodySOA* bodies = physics_bodies();
                    ContactManifoldSOA* manifold = physics_manifold();
                    if (bodies == nullptr || manifold == nullptr || manifold->capacity == 0u) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }

                    const uint32_t requested = static_cast<uint32_t>(max(0.0f, floorf(body_count_scalar + 0.5f)));
                    const uint32_t body_count = min(requested, bodies->body_count);
                    (void)dt_scalar;
                    manifold->write_head = 0u;

                    uint32_t pair_count = 0u;
                    for (uint32_t a = 0; a < body_count; ++a) {
                        if (physics_body_is_static(bodies, a)) continue;
                        const float3 pos_a = physics_position(bodies, a);
                        const float radius_a = physics_bound_radius(bodies, a);
                        for (uint32_t b = a + 1; b < body_count; ++b) {
                            if (physics_body_is_sleeping(bodies, a) && physics_body_is_sleeping(bodies, b)) continue;
                            const float3 pos_b = physics_position(bodies, b);
                            const float radius_b = physics_bound_radius(bodies, b);
                            const float overlap = radius_a + radius_b;
                            if (fabsf(pos_b.x - pos_a.x) > overlap ||
                                fabsf(pos_b.y - pos_a.y) > overlap ||
                                fabsf(pos_b.z - pos_a.z) > overlap) {
                                continue;
                            }
                            if (pair_count >= manifold->capacity) break;
                            manifold->body_a_id[pair_count] = a;
                            manifold->body_b_id[pair_count] = b;
                            pair_count += 1u;
                        }
                        if (pair_count >= manifold->capacity) break;
                    }
                    manifold->write_head = pair_count;
                    push(stack, stack_size, make_scalar(static_cast<float>(pair_count)), error_code);
                    break;
                }
                case 0x151: {  // PH_NARROW_PHASE
                    float pair_count_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, pair_count_scalar, error_code)) break;
                    PhysicsBodySOA* bodies = physics_bodies();
                    ContactManifoldSOA* manifold = physics_manifold();
                    if (bodies == nullptr || manifold == nullptr) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const uint32_t pair_count = min(static_cast<uint32_t>(max(0.0f, floorf(pair_count_scalar + 0.5f))), manifold->write_head);
                    uint32_t contact_count = 0u;
                    for (uint32_t idx = 0; idx < pair_count; ++idx) {
                        const uint32_t body_a = manifold->body_a_id[idx];
                        const uint32_t body_b = manifold->body_b_id[idx];
                        if (!physics_body_valid(bodies, body_a) || !physics_body_valid(bodies, body_b)) continue;
                        const float3 delta = physics_vec_sub(physics_position(bodies, body_b), physics_position(bodies, body_a));
                        const float distance = physics_vec_len(delta);
                        const float combined_radius = physics_bound_radius(bodies, body_a) + physics_bound_radius(bodies, body_b);
                        if (distance >= combined_radius) continue;
                        const float3 normal = distance > 1e-6f ? physics_vec_scale(delta, 1.0f / distance) : make_float3(0.0f, 1.0f, 0.0f);
                        const float penetration = combined_radius - distance;
                        const float3 contact = physics_vec_add(
                            physics_position(bodies, body_a),
                            physics_vec_scale(normal, physics_bound_radius(bodies, body_a) - 0.5f * penetration));
                        manifold->contact_x[idx] = contact.x;
                        manifold->contact_y[idx] = contact.y;
                        manifold->contact_z[idx] = contact.z;
                        manifold->normal_x[idx] = normal.x;
                        manifold->normal_y[idx] = normal.y;
                        manifold->normal_z[idx] = normal.z;
                        manifold->penetration_depth[idx] = penetration;
                        contact_count += 1u;
                    }
                    push(stack, stack_size, make_scalar(static_cast<float>(contact_count)), error_code);
                    break;
                }
                case 0x152: {  // PH_CONSTRAINT_GENERATE
                    float contact_count_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, contact_count_scalar, error_code)) break;
                    PhysicsBodySOA* bodies = physics_bodies();
                    ContactManifoldSOA* manifold = physics_manifold();
                    if (bodies == nullptr || manifold == nullptr) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const uint32_t contact_count = min(static_cast<uint32_t>(max(0.0f, floorf(contact_count_scalar + 0.5f))), manifold->write_head);
                    uint32_t constraint_count = 0u;
                    for (uint32_t idx = 0; idx < contact_count; ++idx) {
                        const uint32_t body_a = manifold->body_a_id[idx];
                        const uint32_t body_b = manifold->body_b_id[idx];
                        if (!physics_body_valid(bodies, body_a) || !physics_body_valid(bodies, body_b)) continue;
                        if (manifold->persistent_id[idx] == 0u) {
                            manifold->persistent_id[idx] = manifold->persistent_counter + 1u;
                            manifold->persistent_counter += 1u;
                            manifold->lambda_normal[idx] = 0.0f;
                            manifold->lambda_tangent0[idx] = 0.0f;
                            manifold->lambda_tangent1[idx] = 0.0f;
                        } else {
                            manifold->lambda_normal[idx] *= 0.85f;
                            manifold->lambda_tangent0[idx] *= 0.85f;
                            manifold->lambda_tangent1[idx] *= 0.85f;
                        }
                        const float friction = 0.5f * (physics_friction(bodies, body_a) + physics_friction(bodies, body_b));
                        const float restitution = 0.5f * (physics_restitution(bodies, body_a) + physics_restitution(bodies, body_b));
                        manifold->compliance_normal[idx] = restitution > 0.75f ? 1.0e-5f : 0.0f;
                        manifold->lambda_tangent0[idx] = friction;
                        manifold->lambda_tangent1[idx] = -friction;
                        constraint_count += 1u;
                    }
                    push(stack, stack_size, make_scalar(static_cast<float>(constraint_count)), error_code);
                    break;
                }
                case 0x153: {  // PH_XPBD_SOLVE
                    float iter_count_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, iter_count_scalar, error_code)) break;
                    PhysicsBodySOA* bodies = physics_bodies();
                    ContactManifoldSOA* manifold = physics_manifold();
                    PhysicsPredictedSOA* predicted = physics_predicted();
                    if (bodies == nullptr || manifold == nullptr) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const uint32_t iter_count = max(1u, static_cast<uint32_t>(floorf(iter_count_scalar + 0.5f)));
                    uint8_t max_color = 0u;
                    for (uint32_t idx = 0; idx < manifold->write_head; ++idx) {
                        if (manifold->color_id[idx] > max_color) max_color = manifold->color_id[idx];
                    }
                    float accumulated_error = 0.0f;
                    for (uint32_t iter = 0; iter < iter_count; ++iter) {
                        for (uint8_t color = 0u; color <= max_color; ++color) {
                            for (uint32_t idx = 0; idx < manifold->write_head; ++idx) {
                                if (manifold->color_id[idx] != color) continue;
                                const uint32_t body_a = manifold->body_a_id[idx];
                                const uint32_t body_b = manifold->body_b_id[idx];
                                float3 pos_a = predicted && body_a < predicted->capacity
                                    ? make_float3(predicted->predicted_pos_inv[body_a].x, predicted->predicted_pos_inv[body_a].y, predicted->predicted_pos_inv[body_a].z)
                                    : physics_position(bodies, body_a);
                                float3 pos_b = predicted && body_b < predicted->capacity
                                    ? make_float3(predicted->predicted_pos_inv[body_b].x, predicted->predicted_pos_inv[body_b].y, predicted->predicted_pos_inv[body_b].z)
                                    : physics_position(bodies, body_b);
                                const float3 normal = make_float3(manifold->normal_x[idx], manifold->normal_y[idx], manifold->normal_z[idx]);
                                const float c = manifold->penetration_depth[idx];
                                const float denom = physics_inv_mass(bodies, body_a) + physics_inv_mass(bodies, body_b) + manifold->compliance_normal[idx];
                                if (denom <= 1e-8f || c <= 0.0f) continue;
                                const float delta_lambda = (-c - manifold->compliance_normal[idx] * manifold->lambda_normal[idx]) / denom;
                                manifold->lambda_normal[idx] += delta_lambda;
                                const float3 correction = physics_vec_scale(normal, delta_lambda);
                                if (physics_inv_mass(bodies, body_a) > 0.0f) {
                                    pos_a = physics_vec_sub(pos_a, physics_vec_scale(correction, physics_inv_mass(bodies, body_a)));
                                }
                                if (physics_inv_mass(bodies, body_b) > 0.0f) {
                                    pos_b = physics_vec_add(pos_b, physics_vec_scale(correction, physics_inv_mass(bodies, body_b)));
                                }
                                if (predicted != nullptr) {
                                    if (body_a < predicted->capacity) {
                                        physics_store_predicted_pose(predicted, body_a, pos_a, physics_inv_mass(bodies, body_a), physics_orientation(bodies, body_a));
                                    }
                                    if (body_b < predicted->capacity) {
                                        physics_store_predicted_pose(predicted, body_b, pos_b, physics_inv_mass(bodies, body_b), physics_orientation(bodies, body_b));
                                    }
                                } else {
                                    physics_store_position(bodies, body_a, pos_a);
                                    physics_store_position(bodies, body_b, pos_b);
                                }
                                accumulated_error += fabsf(c);
                            }
                        }
                    }
                    push(stack, stack_size, make_scalar(accumulated_error), error_code);
                    break;
                }
                case 0x154: {  // PH_INTEGRATE
                    float dt = 0.0f;
                    if (!pop_scalar(stack, stack_size, dt, error_code)) break;
                    PhysicsBodySOA* bodies = physics_bodies();
                    PhysicsPredictedSOA* predicted = physics_predicted();
                    if (bodies == nullptr) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    for (uint32_t body_id = 0; body_id < bodies->body_count; ++body_id) {
                        if (physics_body_is_static(bodies, body_id)) continue;
                        float3 velocity = physics_velocity(bodies, body_id);
                        if (physics_inv_mass(bodies, body_id) > 0.0f) {
                            velocity.y += dt * physics_gravity_y;
                        }
                        physics_store_velocity(bodies, body_id, velocity);
                        if (predicted != nullptr && body_id < predicted->capacity) {
                            const float4 predicted_pos = predicted->predicted_pos_inv[body_id];
                            physics_store_position(bodies, body_id, make_float3(predicted_pos.x, predicted_pos.y, predicted_pos.z));
                            physics_store_orientation(bodies, body_id, physics_quat_normalize(predicted->predicted_orientation[body_id]));
                        } else {
                            const float3 pos = physics_position(bodies, body_id);
                            physics_store_position(
                                bodies,
                                body_id,
                                make_float3(pos.x + velocity.x * dt, pos.y + velocity.y * dt, pos.z + velocity.z * dt));
                            const float3 ang = physics_angular_velocity(bodies, body_id);
                            const float4 dq = make_float4(0.5f * dt * ang.x, 0.5f * dt * ang.y, 0.5f * dt * ang.z, 1.0f);
                            physics_store_orientation(
                                bodies,
                                body_id,
                                physics_quat_normalize(physics_quat_mul(dq, physics_orientation(bodies, body_id))));
                        }
                        physics_mark_dirty(bodies, body_id);
                    }
                    break;
                }
                case 0x155: {  // PH_SLEEP_CHECK
                    float threshold = 0.0f;
                    if (!pop_scalar(stack, stack_size, threshold, error_code)) break;
                    PhysicsBodySOA* bodies = physics_bodies();
                    if (bodies == nullptr) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    uint32_t island_count = 0u;
                    uint32_t last_island = 0xffffffffu;
                    for (uint32_t body_id = 0; body_id < bodies->body_count; ++body_id) {
                        const float3 v = physics_velocity(bodies, body_id);
                        const float3 w = physics_angular_velocity(bodies, body_id);
                        const float energy = 0.5f * (physics_vec_dot(v, v) + physics_vec_dot(w, w));
                        physics_store_sleep_energy(bodies, body_id, energy);
                        const bool sleep = energy < threshold;
                        physics_set_sleeping(bodies, body_id, sleep);
                        const uint32_t island_id = physics_island_id(bodies, body_id);
                        if (sleep && island_id != last_island) {
                            island_count += 1u;
                            last_island = island_id;
                        }
                    }
                    push(stack, stack_size, make_scalar(static_cast<float>(island_count)), error_code);
                    break;
                }
                case 0x156: {  // PH_GALAXY_WRITE
                    PhysicsBodySOA* bodies = physics_bodies();
                    ContactManifoldSOA* manifold = physics_manifold();
                    CollisionEventQueue* events = physics_event_queue();
                    if (bodies == nullptr || manifold == nullptr || events == nullptr || events->capacity == 0u) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    events->write_head = 0u;
                    uint32_t edge_count = 0u;
                    for (uint32_t idx = 0; idx < manifold->write_head && idx < events->capacity; ++idx) {
                        if (manifold->penetration_depth[idx] <= 0.0f) continue;
                        events->body_a_id[edge_count] = manifold->body_a_id[idx];
                        events->body_b_id[edge_count] = manifold->body_b_id[idx];
                        events->material_a_star_id[edge_count] = physics_material_star_id(bodies, manifold->body_a_id[idx]);
                        events->material_b_star_id[edge_count] = physics_material_star_id(bodies, manifold->body_b_id[idx]);
                        events->impulse_magnitude[edge_count] = fabsf(manifold->lambda_normal[idx]);
                        events->normal_x[edge_count] = manifold->normal_x[idx];
                        events->normal_y[edge_count] = manifold->normal_y[idx];
                        events->normal_z[edge_count] = manifold->normal_z[idx];
                        edge_count += 1u;
                    }
                    events->write_head = edge_count;
                    push(stack, stack_size, make_scalar(static_cast<float>(edge_count)), error_code);
                    break;
                }
                case 0x157: {  // PH_MATERIAL_FETCH
                    float star_id_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, star_id_scalar, error_code)) break;
                    const uint32_t target_id = static_cast<uint32_t>(max(0.0f, floorf(star_id_scalar + 0.5f)));

                    float friction = 0.5f;
                    float restitution = 0.3f;
                    float density = 1000.0f;

                    if (g_physics_material_table_ptr != 0ULL && g_physics_material_table_count > 0u) {
                        const PhysicsMaterialEntry* table =
                            reinterpret_cast<const PhysicsMaterialEntry*>(g_physics_material_table_ptr);
                        for (uint32_t i = 0; i < g_physics_material_table_count; ++i) {
                            if (table[i].star_id == target_id) {
                                friction = table[i].friction;
                                restitution = table[i].restitution;
                                density = table[i].density;
                                break;
                            }
                        }
                    }

                    push(stack, stack_size, make_scalar(friction), error_code);
                    if (error_code != kErrorNone) break;
                    push(stack, stack_size, make_scalar(restitution), error_code);
                    if (error_code != kErrorNone) break;
                    push(stack, stack_size, make_scalar(density), error_code);
                    break;
                }
                case 0x158: {  // PH_PREDICT_POS
                    float dt = 0.0f;
                    if (!pop_scalar(stack, stack_size, dt, error_code)) break;
                    PhysicsBodySOA* bodies = physics_bodies();
                    PhysicsPredictedSOA* predicted = physics_predicted();
                    if (bodies == nullptr || predicted == nullptr) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    for (uint32_t body_id = 0; body_id < bodies->body_count && body_id < predicted->capacity; ++body_id) {
                        const float3 pos = physics_position(bodies, body_id);
                        const float3 vel = physics_velocity(bodies, body_id);
                        const float3 accel = physics_inv_mass(bodies, body_id) > 0.0f
                            ? make_float3(0.0f, physics_gravity_y, 0.0f)
                            : make_float3(0.0f, 0.0f, 0.0f);
                        const float3 ang = physics_angular_velocity(bodies, body_id);
                        const float4 dq = make_float4(0.5f * dt * ang.x, 0.5f * dt * ang.y, 0.5f * dt * ang.z, 1.0f);
                        physics_store_predicted_pose(
                            predicted,
                            body_id,
                            make_float3(
                                pos.x + vel.x * dt + 0.5f * accel.x * dt * dt,
                                pos.y + vel.y * dt + 0.5f * accel.y * dt * dt,
                                pos.z + vel.z * dt + 0.5f * accel.z * dt * dt),
                            physics_inv_mass(bodies, body_id),
                            physics_quat_normalize(physics_quat_mul(dq, physics_orientation(bodies, body_id))));
                    }
                    break;
                }
                case 0x159: {  // PH_CONSTRAINT_COLOR
                    float constraint_count_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, constraint_count_scalar, error_code)) break;
                    ContactManifoldSOA* manifold = physics_manifold();
                    if (manifold == nullptr) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const uint32_t constraint_count = min(static_cast<uint32_t>(max(0.0f, floorf(constraint_count_scalar + 0.5f))), manifold->write_head);
                    uint8_t max_color = 0u;
                    for (uint32_t idx = 0; idx < constraint_count; ++idx) {
                        uint8_t chosen = 0u;
                        bool assigned = false;
                        while (!assigned) {
                            bool conflicts = false;
                            for (uint32_t prev = 0; prev < idx; ++prev) {
                                if (manifold->color_id[prev] != chosen) continue;
                                if (manifold->body_a_id[idx] == manifold->body_a_id[prev] ||
                                    manifold->body_a_id[idx] == manifold->body_b_id[prev] ||
                                    manifold->body_b_id[idx] == manifold->body_a_id[prev] ||
                                    manifold->body_b_id[idx] == manifold->body_b_id[prev]) {
                                    conflicts = true;
                                    break;
                                }
                            }
                            if (!conflicts) {
                                manifold->color_id[idx] = chosen;
                                if (chosen > max_color) max_color = chosen;
                                assigned = true;
                            } else {
                                chosen = static_cast<uint8_t>(chosen + 1u);
                            }
                        }
                    }
                    push(stack, stack_size, make_scalar(static_cast<float>(max_color + 1u)), error_code);
                    break;
                }
                case 0x15A: {  // PH_IMPULSE_PROPAGATE
                    float island_id_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, island_id_scalar, error_code)) break;
                    (void)island_id_scalar;
                    break;
                }
                case 0x15B: {  // PH_RESTITUTION_APPLY
                    float contact_count_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, contact_count_scalar, error_code)) break;
                    ContactManifoldSOA* manifold = physics_manifold();
                    if (manifold == nullptr) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const uint32_t count = min(static_cast<uint32_t>(max(0.0f, floorf(contact_count_scalar + 0.5f))), manifold->write_head);
                    for (uint32_t idx = 0; idx < count; ++idx) {
                        manifold->lambda_normal[idx] = fabsf(manifold->lambda_normal[idx]);
                    }
                    break;
                }
                case 0x15C: {  // PH_FRICTION_APPLY
                    float contact_count_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, contact_count_scalar, error_code)) break;
                    ContactManifoldSOA* manifold = physics_manifold();
                    if (manifold == nullptr) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const uint32_t count = min(static_cast<uint32_t>(max(0.0f, floorf(contact_count_scalar + 0.5f))), manifold->write_head);
                    for (uint32_t idx = 0; idx < count; ++idx) {
                        const float max_friction = fabsf(manifold->lambda_tangent0[idx]);
                        manifold->lambda_tangent0[idx] = max(-max_friction, min(max_friction, manifold->lambda_tangent0[idx]));
                        manifold->lambda_tangent1[idx] = max(-max_friction, min(max_friction, manifold->lambda_tangent1[idx]));
                    }
                    break;
                }
                case 0x15D: {  // PH_ISLAND_WAKE
                    float trigger_star_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, trigger_star_scalar, error_code)) break;
                    PhysicsBodySOA* bodies = physics_bodies();
                    if (bodies == nullptr) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const uint32_t trigger_star = static_cast<uint32_t>(max(0.0f, floorf(trigger_star_scalar + 0.5f)));
                    uint32_t woken_count = 0u;
                    for (uint32_t body_id = 0; body_id < bodies->body_count; ++body_id) {
                        if (physics_material_star_id(bodies, body_id) == trigger_star && physics_body_is_sleeping(bodies, body_id)) {
                            physics_set_sleeping(bodies, body_id, false);
                            woken_count += 1u;
                        }
                    }
                    push(stack, stack_size, make_scalar(static_cast<float>(woken_count)), error_code);
                    break;
                }
                case 0x15E: {  // PH_BODY_SPAWN
                    float3 velocity{};
                    float3 position{};
                    float star_id_scalar = 0.0f;
                    if (!pop_vector(stack, stack_size, velocity, error_code)) break;
                    if (!pop_vector(stack, stack_size, position, error_code)) break;
                    if (!pop_scalar(stack, stack_size, star_id_scalar, error_code)) break;
                    PhysicsBodySOA* bodies = physics_bodies();
                    if (bodies == nullptr || bodies->body_count >= bodies->capacity) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const uint32_t body_id = bodies->body_count;
                    bodies->body_count += 1u;
                    bodies->pos_inv[body_id] = make_float4(position.x, position.y, position.z, 1.0f);
                    bodies->vel_sleep[body_id] = make_float4(velocity.x, velocity.y, velocity.z, 0.0f);
                    bodies->orientation[body_id] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
                    bodies->ang_vel_damp[body_id] = make_float4(0.0f, 0.0f, 0.0f, 0.01f);
                    bodies->inv_inertia_rest[body_id] = make_float4(1.0f, 1.0f, 1.0f, 0.25f);
                    bodies->galaxy_handles[body_id] = make_uint2(static_cast<uint32_t>(rounded_index(star_id_scalar)), 0u);
                    bodies->island_flags[body_id] = PHYSICS_FLAG_DIRTY;
                    bodies->bound_friction[body_id] = make_float2(0.5f, 0.5f);
                    push(stack, stack_size, make_scalar(static_cast<float>(body_id)), error_code);
                    break;
                }
                case 0x15F: {  // PH_BODY_DESPAWN
                    float body_idx_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, body_idx_scalar, error_code)) break;
                    PhysicsBodySOA* bodies = physics_bodies();
                    if (bodies == nullptr) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const uint32_t body_id = static_cast<uint32_t>(max(0.0f, floorf(body_idx_scalar + 0.5f)));
                    if (!physics_body_valid(bodies, body_id)) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    bodies->pos_inv[body_id] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    bodies->vel_sleep[body_id] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    bodies->orientation[body_id] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
                    bodies->ang_vel_damp[body_id] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    bodies->galaxy_handles[body_id] = make_uint2(0u, 0u);
                    bodies->island_flags[body_id] = 0u;
                    bodies->bound_friction[body_id] = make_float2(0.0f, 0.0f);
                    break;
                }
                case 0x160: {  // PH_GRAVITY_APPLY
                    float gravity_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, gravity_scalar, error_code)) break;
                    physics_gravity_y = gravity_scalar;
                    break;
                }
                case 0x161: {  // PH_COLLISION_QUERY
                    float3 ray_dir{};
                    float3 ray_origin{};
                    if (!pop_vector(stack, stack_size, ray_dir, error_code)) break;
                    if (!pop_vector(stack, stack_size, ray_origin, error_code)) break;
                    PhysicsBodySOA* bodies = physics_bodies();
                    if (bodies == nullptr) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    uint32_t best_body = 0xffffffffu;
                    float best_t = 1.0e30f;
                    for (uint32_t body_id = 0; body_id < bodies->body_count; ++body_id) {
                        const float3 center = physics_position(bodies, body_id);
                        const float radius = physics_bound_radius(bodies, body_id);
                        const float3 oc = physics_vec_sub(ray_origin, center);
                        const float a = physics_vec_dot(ray_dir, ray_dir);
                        const float b = 2.0f * physics_vec_dot(oc, ray_dir);
                        const float c = physics_vec_dot(oc, oc) - radius * radius;
                        const float discriminant = b * b - 4.0f * a * c;
                        if (discriminant < 0.0f) continue;
                        const float t = (-b - sqrtf(discriminant)) / (2.0f * a);
                        if (t >= 0.0f && t < best_t) {
                            best_t = t;
                            best_body = body_id;
                        }
                    }
                    push(stack, stack_size, make_scalar(static_cast<float>(best_body == 0xffffffffu ? -1.0f : static_cast<float>(best_body))), error_code);
                    if (error_code != kErrorNone) break;
                    push(stack, stack_size, make_scalar(best_t == 1.0e30f ? -1.0f : best_t), error_code);
                    break;
                }
                case 0x162: {  // PH_TERNARY_CLASSIFY
                    float body_idx_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, body_idx_scalar, error_code)) break;
                    PhysicsBodySOA* bodies = physics_bodies();
                    if (bodies == nullptr) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const uint32_t body_id = static_cast<uint32_t>(max(0.0f, floorf(body_idx_scalar + 0.5f)));
                    if (!physics_body_valid(bodies, body_id)) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const float energy = physics_sleep_energy(bodies, body_id);
                    float ternary = 0.0f;
                    if (energy > 1.0e-2f) ternary = 1.0f;
                    else if (energy < 1.0e-4f) ternary = -1.0f;
                    push(stack, stack_size, make_scalar(ternary), error_code);
                    break;
                }
                case 0x180: {  // BH_PERCEIVE
                    float radius = 0.0f;
                    if (!pop_scalar(stack, stack_size, radius, error_code)) break;
                    EntityHotPath* entities = entity_hot_paths();
                    const uint32_t self_idx = bh_self_index(instance_id, g_entity_count);
                    float nearest_dist = 999.0f;
                    const uint32_t count = bh_perceive_count(
                        fabsf(radius),
                        entities,
                        g_entity_count,
                        self_idx,
                        entity_query_scratch,
                        257u,
                        &nearest_dist);
                    (void)nearest_dist;
                    push(stack, stack_size, make_scalar(static_cast<float>(count)), error_code);
                    break;
                }
                case 0x181: {  // BH_SEEK
                    float target_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, target_scalar, error_code)) break;
                    EntityHotPath* entities = entity_hot_paths();
                    PhysicsBodySOA* bodies = physics_bodies();
                    if (entities == nullptr || g_entity_count == 0u) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const uint32_t self_idx = bh_self_index(instance_id, g_entity_count);
                    const uint32_t target_idx = min(
                        static_cast<uint32_t>(max(0.0f, floorf(target_scalar + 0.5f))),
                        g_entity_count - 1u);
                    const float3 force = bh_seek_force(entities, bodies, self_idx, target_idx, 3.0f, 0.0f);
                    push(stack, stack_size, make_vector(force.x, force.y, force.z), error_code);
                    break;
                }
                case 0x182: {  // BH_FLEE
                    float target_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, target_scalar, error_code)) break;
                    EntityHotPath* entities = entity_hot_paths();
                    PhysicsBodySOA* bodies = physics_bodies();
                    if (entities == nullptr || g_entity_count == 0u) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const uint32_t self_idx = bh_self_index(instance_id, g_entity_count);
                    const uint32_t target_idx = min(
                        static_cast<uint32_t>(max(0.0f, floorf(target_scalar + 0.5f))),
                        g_entity_count - 1u);
                    const float3 seek = bh_seek_force(entities, bodies, self_idx, target_idx, 4.0f, 0.0f);
                    const float3 flee = physics_vec_scale(seek, -1.0f);
                    push(stack, stack_size, make_vector(flee.x, flee.y, flee.z), error_code);
                    break;
                }
                case 0x183: {  // BH_ARRIVE
                    float target_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, target_scalar, error_code)) break;
                    EntityHotPath* entities = entity_hot_paths();
                    PhysicsBodySOA* bodies = physics_bodies();
                    if (entities == nullptr || g_entity_count == 0u) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const uint32_t self_idx = bh_self_index(instance_id, g_entity_count);
                    const uint32_t target_idx = min(
                        static_cast<uint32_t>(max(0.0f, floorf(target_scalar + 0.5f))),
                        g_entity_count - 1u);
                    const float3 force = bh_seek_force(entities, bodies, self_idx, target_idx, 3.0f, 10.0f);
                    push(stack, stack_size, make_vector(force.x, force.y, force.z), error_code);
                    break;
                }
                case 0x184: {  // BH_SEPARATE
                    float radius = 0.0f;
                    if (!pop_scalar(stack, stack_size, radius, error_code)) break;
                    EntityHotPath* entities = entity_hot_paths();
                    if (entities == nullptr || g_entity_count == 0u) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const uint32_t self_idx = bh_self_index(instance_id, g_entity_count);
                    const float3 force = bh_separation_force(entities, g_entity_count, self_idx, fmaxf(1.0f, fabsf(radius)));
                    push(stack, stack_size, make_vector(force.x, force.y, force.z), error_code);
                    break;
                }
                case 0x185: {  // BH_APPLY_FORCE
                    float3 force{};
                    if (!pop_vector(stack, stack_size, force, error_code)) break;
                    EntityHotPath* entities = entity_hot_paths();
                    PhysicsBodySOA* bodies = physics_bodies();
                    if (entities == nullptr || bodies == nullptr || g_entity_count == 0u) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const uint32_t self_idx = bh_self_index(instance_id, g_entity_count);
                    const uint32_t body_id = entities[self_idx].physics_body_id;
                    if (!physics_body_valid(bodies, body_id)) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const float3 velocity = physics_velocity(bodies, body_id);
                    const float inv_mass = fmaxf(physics_inv_mass(bodies, body_id), 1.0f);
                    physics_store_velocity(
                        bodies,
                        body_id,
                        make_float3(
                            velocity.x + force.x * inv_mass,
                            velocity.y + force.y * inv_mass,
                            velocity.z + force.z * inv_mass));
                    push(stack, stack_size, make_scalar(1.0f), error_code);
                    break;
                }
                case 0x186: {  // BH_BT_TICK
                    float program_addr_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, program_addr_scalar, error_code)) break;
                    const float status = fabsf(program_addr_scalar) > 0.0f ? 2.0f : 0.0f;
                    push(stack, stack_size, make_scalar(status), error_code);
                    break;
                }
                case 0x188: {  // BH_GOAP_PLAN
                    push(stack, stack_size, make_scalar(0.0f), error_code);
                    break;
                }
                case 0x189: {  // BH_SLEEP_CHECK
                    EntityHotPath* entities = entity_hot_paths();
                    if (entities == nullptr || g_entity_count == 0u) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const uint32_t self_idx = bh_self_index(instance_id, g_entity_count);
                    float dist = entities[self_idx].last_player_dist;
                    if (!(dist >= 0.0f)) {
                        dist = 999.0f;
                    }
                    uint8_t state = 2u;
                    if (dist < 50.0f) state = 0u;
                    else if (dist < 200.0f) state = 1u;
                    entities[self_idx].sleep_state = state;
                    push(stack, stack_size, make_scalar(static_cast<float>(state)), error_code);
                    break;
                }
                case 0x18A: {  // BH_BLACKBOARD_READ
                    EntityHotPath* entities = entity_hot_paths();
                    if (entities == nullptr || g_entity_count == 0u) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const uint32_t self_idx = bh_self_index(instance_id, g_entity_count);
                    push(stack, stack_size, make_scalar(static_cast<float>(entities[self_idx].blackboard_star_id)), error_code);
                    break;
                }
                case 0x18B: {  // BH_BLACKBOARD_WRITE
                    float value_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, value_scalar, error_code)) break;
                    EntityHotPath* entities = entity_hot_paths();
                    if (entities == nullptr || g_entity_count == 0u) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const uint32_t self_idx = bh_self_index(instance_id, g_entity_count);
                    entities[self_idx].blackboard_star_id =
                        static_cast<uint32_t>(max(0.0f, floorf(value_scalar + 0.5f)));
                    push(stack, stack_size, make_scalar(value_scalar), error_code);
                    break;
                }
                case 0x18C: {  // BH_PATHFIND
                    push(stack, stack_size, make_scalar(0.0f), error_code);
                    break;
                }
                case 0x1C0: {  // TEX_PERLIN_NOISE
                    float persistence = 0.0f;
                    float amplitude = 0.0f;
                    float frequency = 0.0f;
                    float octaves_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, persistence, error_code)) break;
                    if (!pop_scalar(stack, stack_size, amplitude, error_code)) break;
                    if (!pop_scalar(stack, stack_size, frequency, error_code)) break;
                    if (!pop_scalar(stack, stack_size, octaves_scalar, error_code)) break;
                    TextureHandlePool* pool = texture_pool();
                    const uint32_t handle = texture_alloc_slot(pool);
                    const uint32_t slot = handle == 0u ? kInvalidTextureSlot : handle - 1u;
                    if (!texture_slot_valid(pool, slot)) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    texture_clear_slot(pool, slot);
                    const int width = static_cast<int>(pool->width[slot]);
                    const int height = static_cast<int>(pool->height[slot]);
                    const int octave_count = max(1, min(8, rounded_index(octaves_scalar)));
                    const int maskx = (width - 1) & 4095;
                    const int masky = (height - 1) & 4095;
                    float* out = texture_slot_ptr(pool, slot);
                    for (int y = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x) {
                            float sum = 0.0f;
                            float amp = amplitude;
                            float freq = frequency;
                            float norm = 0.0f;
                            for (int octave = 0; octave < octave_count; ++octave) {
                                const float nx = static_cast<float>(x) / max(1.0f, static_cast<float>(width));
                                const float ny = static_cast<float>(y) / max(1.0f, static_cast<float>(height));
                                const int fx = static_cast<int>(nx * freq * 65536.0f);
                                const int fy = static_cast<int>(ny * freq * 65536.0f);
                                sum += texture_perlin_noise2(fx, fy, maskx, masky, octave * 17) * amp;
                                norm += fabsf(amp);
                                amp *= persistence;
                                freq *= 2.0f;
                            }
                            out[y * width + x] = norm > 1.0e-6f ? sum / norm : 0.0f;
                        }
                    }
                    push(stack, stack_size, make_scalar(static_cast<float>(handle)), error_code);
                    break;
                }
                case 0x1C1: {  // TEX_VORONOI
                    float jitter = 0.0f;
                    float cell_count = 0.0f;
                    if (!pop_scalar(stack, stack_size, jitter, error_code)) break;
                    if (!pop_scalar(stack, stack_size, cell_count, error_code)) break;
                    TextureHandlePool* pool = texture_pool();
                    const uint32_t handle = texture_alloc_slot(pool);
                    const uint32_t slot = handle == 0u ? kInvalidTextureSlot : handle - 1u;
                    if (!texture_slot_valid(pool, slot)) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const int width = static_cast<int>(pool->width[slot]);
                    const int height = static_cast<int>(pool->height[slot]);
                    float* out = texture_slot_ptr(pool, slot);
                    for (int y = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x) {
                            const float u = static_cast<float>(x) / max(1.0f, static_cast<float>(width));
                            const float v = static_cast<float>(y) / max(1.0f, static_cast<float>(height));
                            out[y * width + x] = texture_voronoi_f1(u, v, cell_count, jitter);
                        }
                    }
                    push(stack, stack_size, make_scalar(static_cast<float>(handle)), error_code);
                    break;
                }
                case 0x1C2: {  // TEX_VALUE_NOISE
                    float octaves_scalar = 0.0f;
                    float frequency = 0.0f;
                    if (!pop_scalar(stack, stack_size, octaves_scalar, error_code)) break;
                    if (!pop_scalar(stack, stack_size, frequency, error_code)) break;
                    TextureHandlePool* pool = texture_pool();
                    const uint32_t handle = texture_alloc_slot(pool);
                    const uint32_t slot = handle == 0u ? kInvalidTextureSlot : handle - 1u;
                    if (!texture_slot_valid(pool, slot)) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const int width = static_cast<int>(pool->width[slot]);
                    const int height = static_cast<int>(pool->height[slot]);
                    const int octave_count = max(1, min(8, rounded_index(octaves_scalar)));
                    const int maskx = (width - 1) & 4095;
                    const int masky = (height - 1) & 4095;
                    float* out = texture_slot_ptr(pool, slot);
                    for (int y = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x) {
                            float sum = 0.0f;
                            float amp = 1.0f;
                            float freq = frequency;
                            float norm = 0.0f;
                            for (int octave = 0; octave < octave_count; ++octave) {
                                const float nx = static_cast<float>(x) / max(1.0f, static_cast<float>(width));
                                const float ny = static_cast<float>(y) / max(1.0f, static_cast<float>(height));
                                const int fx = static_cast<int>(nx * freq * 65536.0f);
                                const int fy = static_cast<int>(ny * freq * 65536.0f);
                                sum += texture_value_noise2(fx, fy, maskx, masky, octave * 11) * amp;
                                norm += amp;
                                amp *= 0.5f;
                                freq *= 2.0f;
                            }
                            out[y * width + x] = norm > 1.0e-6f ? sum / norm : 0.0f;
                        }
                    }
                    push(stack, stack_size, make_scalar(static_cast<float>(handle)), error_code);
                    break;
                }
                case 0x1C3: {  // TEX_GRID_NOISE
                    float falloff = 0.0f;
                    float scale = 0.0f;
                    if (!pop_scalar(stack, stack_size, falloff, error_code)) break;
                    if (!pop_scalar(stack, stack_size, scale, error_code)) break;
                    TextureHandlePool* pool = texture_pool();
                    const uint32_t handle = texture_alloc_slot(pool);
                    const uint32_t slot = handle == 0u ? kInvalidTextureSlot : handle - 1u;
                    if (!texture_slot_valid(pool, slot)) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    const int width = static_cast<int>(pool->width[slot]);
                    const int height = static_cast<int>(pool->height[slot]);
                    const int maskx = (width - 1) & 4095;
                    const int masky = (height - 1) & 4095;
                    float* out = texture_slot_ptr(pool, slot);
                    for (int y = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x) {
                            const float nx = static_cast<float>(x) / max(1.0f, static_cast<float>(width));
                            const float ny = static_cast<float>(y) / max(1.0f, static_cast<float>(height));
                            const int fx = static_cast<int>(nx * scale * 65536.0f);
                            const int fy = static_cast<int>(ny * scale * 65536.0f);
                            out[y * width + x] = texture_grid_noise2(fx, fy, maskx, masky, 29, falloff);
                        }
                    }
                    push(stack, stack_size, make_scalar(static_cast<float>(handle)), error_code);
                    break;
                }
                case 0x1C4: {  // TEX_FFT_BLUR
                    float sigma = 0.0f;
                    float handle_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, sigma, error_code)) break;
                    if (!pop_scalar(stack, stack_size, handle_scalar, error_code)) break;
                    TextureHandlePool* pool = texture_pool();
                    const uint32_t src_slot = texture_handle_to_slot(handle_scalar);
                    const uint32_t handle = texture_alloc_slot(pool);
                    const uint32_t dst_slot = handle == 0u ? kInvalidTextureSlot : handle - 1u;
                    if (!texture_slot_valid(pool, src_slot) || !texture_slot_valid(pool, dst_slot)) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    pool->width[dst_slot] = pool->width[src_slot];
                    pool->height[dst_slot] = pool->height[src_slot];
                    texture_blur_into(pool, src_slot, dst_slot, sigma);
                    push(stack, stack_size, make_scalar(static_cast<float>(handle)), error_code);
                    break;
                }
                case 0x1C5: {  // TEX_WARP
                    float intensity = 0.0f;
                    float warp_handle_scalar = 0.0f;
                    float base_handle_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, intensity, error_code)) break;
                    if (!pop_scalar(stack, stack_size, warp_handle_scalar, error_code)) break;
                    if (!pop_scalar(stack, stack_size, base_handle_scalar, error_code)) break;
                    TextureHandlePool* pool = texture_pool();
                    const uint32_t base_slot = texture_handle_to_slot(base_handle_scalar);
                    const uint32_t warp_slot = texture_handle_to_slot(warp_handle_scalar);
                    const uint32_t handle = texture_alloc_slot(pool);
                    const uint32_t dst_slot = handle == 0u ? kInvalidTextureSlot : handle - 1u;
                    if (!texture_slot_valid(pool, base_slot) || !texture_slot_valid(pool, warp_slot) || !texture_slot_valid(pool, dst_slot)) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    pool->width[dst_slot] = pool->width[base_slot];
                    pool->height[dst_slot] = pool->height[base_slot];
                    texture_warp_into(pool, base_slot, warp_slot, dst_slot, intensity);
                    push(stack, stack_size, make_scalar(static_cast<float>(handle)), error_code);
                    break;
                }
                case 0x1C6: {  // TEX_BLEND
                    float mode_scalar = 0.0f;
                    float alpha = 0.0f;
                    float tex_b_scalar = 0.0f;
                    float tex_a_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, mode_scalar, error_code)) break;
                    if (!pop_scalar(stack, stack_size, alpha, error_code)) break;
                    if (!pop_scalar(stack, stack_size, tex_b_scalar, error_code)) break;
                    if (!pop_scalar(stack, stack_size, tex_a_scalar, error_code)) break;
                    TextureHandlePool* pool = texture_pool();
                    const uint32_t slot_a = texture_handle_to_slot(tex_a_scalar);
                    const uint32_t slot_b = texture_handle_to_slot(tex_b_scalar);
                    const uint32_t handle = texture_alloc_slot(pool);
                    const uint32_t dst_slot = handle == 0u ? kInvalidTextureSlot : handle - 1u;
                    if (!texture_slot_valid(pool, slot_a) || !texture_slot_valid(pool, slot_b) || !texture_slot_valid(pool, dst_slot)) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    pool->width[dst_slot] = pool->width[slot_a];
                    pool->height[dst_slot] = pool->height[slot_a];
                    texture_blend_into(pool, slot_a, slot_b, dst_slot, alpha, static_cast<uint32_t>(max(0, rounded_index(mode_scalar))));
                    push(stack, stack_size, make_scalar(static_cast<float>(handle)), error_code);
                    break;
                }
                case 0x1C7: {  // TEX_NORMAL_MAP
                    float strength = 0.0f;
                    float handle_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, strength, error_code)) break;
                    if (!pop_scalar(stack, stack_size, handle_scalar, error_code)) break;
                    TextureHandlePool* pool = texture_pool();
                    const uint32_t src_slot = texture_handle_to_slot(handle_scalar);
                    const uint32_t handle = texture_alloc_slot(pool);
                    const uint32_t dst_slot = handle == 0u ? kInvalidTextureSlot : handle - 1u;
                    if (!texture_slot_valid(pool, src_slot) || !texture_slot_valid(pool, dst_slot)) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    pool->width[dst_slot] = pool->width[src_slot];
                    pool->height[dst_slot] = pool->height[src_slot];
                    texture_normal_map_into(pool, src_slot, dst_slot, strength);
                    push(stack, stack_size, make_scalar(static_cast<float>(handle)), error_code);
                    break;
                }
                case 0x1C8: {  // TEX_COLOR_RAMP
                    float ramp_addr_scalar = 0.0f;
                    float handle_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, ramp_addr_scalar, error_code)) break;
                    if (!pop_scalar(stack, stack_size, handle_scalar, error_code)) break;
                    (void)ramp_addr_scalar;
                    TextureHandlePool* pool = texture_pool();
                    const uint32_t src_slot = texture_handle_to_slot(handle_scalar);
                    const uint32_t handle = texture_alloc_slot(pool);
                    const uint32_t dst_slot = handle == 0u ? kInvalidTextureSlot : handle - 1u;
                    if (!texture_slot_valid(pool, src_slot) || !texture_slot_valid(pool, dst_slot)) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    pool->width[dst_slot] = pool->width[src_slot];
                    pool->height[dst_slot] = pool->height[src_slot];
                    float* out = texture_slot_ptr(pool, dst_slot);
                    float* src = texture_slot_ptr(pool, src_slot);
                    const uint32_t count = pool->width[src_slot] * pool->height[src_slot];
                    for (uint32_t idx = 0; idx < count; ++idx) {
                        out[idx] = src[idx];
                    }
                    push(stack, stack_size, make_scalar(static_cast<float>(handle)), error_code);
                    break;
                }
                case 0x1C9: {  // TEX_TURBULENCE
                    float octaves_scalar = 0.0f;
                    float handle_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, octaves_scalar, error_code)) break;
                    if (!pop_scalar(stack, stack_size, handle_scalar, error_code)) break;
                    TextureHandlePool* pool = texture_pool();
                    const uint32_t src_slot = texture_handle_to_slot(handle_scalar);
                    const uint32_t handle = texture_alloc_slot(pool);
                    const uint32_t dst_slot = handle == 0u ? kInvalidTextureSlot : handle - 1u;
                    if (!texture_slot_valid(pool, src_slot) || !texture_slot_valid(pool, dst_slot)) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    pool->width[dst_slot] = pool->width[src_slot];
                    pool->height[dst_slot] = pool->height[src_slot];
                    texture_turbulence_into(pool, src_slot, dst_slot, octaves_scalar);
                    push(stack, stack_size, make_scalar(static_cast<float>(handle)), error_code);
                    break;
                }
                case 0x1CA: {  // TEX_MARBLE
                    float turbulence = 0.0f;
                    float vein_scale = 0.0f;
                    if (!pop_scalar(stack, stack_size, turbulence, error_code)) break;
                    if (!pop_scalar(stack, stack_size, vein_scale, error_code)) break;
                    TextureHandlePool* pool = texture_pool();
                    const uint32_t handle = texture_alloc_slot(pool);
                    const uint32_t dst_slot = handle == 0u ? kInvalidTextureSlot : handle - 1u;
                    if (!texture_slot_valid(pool, dst_slot)) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    texture_marble_into(pool, dst_slot, vein_scale, turbulence);
                    push(stack, stack_size, make_scalar(static_cast<float>(handle)), error_code);
                    break;
                }
                case 0x1CB: {  // TEX_TRANSFORM
                    float rot = 0.0f;
                    float sy = 0.0f;
                    float sx = 0.0f;
                    float handle_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, rot, error_code)) break;
                    if (!pop_scalar(stack, stack_size, sy, error_code)) break;
                    if (!pop_scalar(stack, stack_size, sx, error_code)) break;
                    if (!pop_scalar(stack, stack_size, handle_scalar, error_code)) break;
                    TextureHandlePool* pool = texture_pool();
                    const uint32_t src_slot = texture_handle_to_slot(handle_scalar);
                    const uint32_t handle = texture_alloc_slot(pool);
                    const uint32_t dst_slot = handle == 0u ? kInvalidTextureSlot : handle - 1u;
                    if (!texture_slot_valid(pool, src_slot) || !texture_slot_valid(pool, dst_slot)) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    pool->width[dst_slot] = pool->width[src_slot];
                    pool->height[dst_slot] = pool->height[src_slot];
                    texture_transform_into(pool, src_slot, dst_slot, sx, sy, rot);
                    push(stack, stack_size, make_scalar(static_cast<float>(handle)), error_code);
                    break;
                }
                case 0x1CF: {  // TEX_BAKE
                    float height_scalar = 0.0f;
                    float width_scalar = 0.0f;
                    float handle_scalar = 0.0f;
                    if (!pop_scalar(stack, stack_size, height_scalar, error_code)) break;
                    if (!pop_scalar(stack, stack_size, width_scalar, error_code)) break;
                    if (!pop_scalar(stack, stack_size, handle_scalar, error_code)) break;
                    TextureHandlePool* pool = texture_pool();
                    const uint32_t baked_id = texture_bake_handle(
                        pool,
                        handle_scalar,
                        static_cast<uint32_t>(max(1, rounded_index(width_scalar))),
                        static_cast<uint32_t>(max(1, rounded_index(height_scalar))));
                    if (baked_id == 0u) {
                        error_code = kErrorInvalidArgument;
                        break;
                    }
                    push(stack, stack_size, make_scalar(static_cast<float>(baked_id)), error_code);
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

// Extract top-of-stack scalar from an instance state into a provided output buffer (device → device)
extern "C" __global__ void modular_rpn_extract_top(
    uint32_t instance_id,
    const InstanceState* __restrict__ states,
    float* __restrict__ out,
    uint32_t out_index) {
    if (threadIdx.x != 0) return;
    const InstanceState* state = reinterpret_cast<const InstanceState*>(
        reinterpret_cast<const uint8_t*>(states) + instance_id * sizeof(InstanceState));
    if (state->error == kErrorNone && state->size > 0) {
        uint32_t top = (state->head + state->size - 1) & 63u;
        out[out_index] = state->stack[top].x;
    } else {
        out[out_index] = __int_as_float(0x7fc00000);  // NaN marker on error/underflow
    }
}
