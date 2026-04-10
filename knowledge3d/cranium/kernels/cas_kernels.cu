#include "cas_star_node.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

extern "C" {
__device__ StarNode g_cas_pool[CAS_POOL_SIZE];
__device__ float g_cas_coeffs[CAS_COEFF_SIZE];
__device__ uint32_t g_cas_pool_top;
__device__ uint32_t g_cas_coeff_top;
}

namespace {
constexpr uint32_t OP_ADD = 0x0Au;
constexpr uint32_t OP_SUB = 0x0Bu;
constexpr uint32_t OP_MUL = 0x0Cu;
constexpr uint32_t OP_DIV = 0x0Du;
constexpr uint32_t OP_POWER = 0x0Eu;
constexpr uint32_t OP_EXP = 0x15u;
constexpr uint32_t OP_LOG = 0x16u;
constexpr uint32_t OP_SIN = 0x18u;
constexpr uint32_t OP_COS = 0x19u;
constexpr uint32_t OP_NEGATE = 0xDBu;
constexpr uint32_t OP_POLY_BUILD = 0x221u;
constexpr uint32_t OP_POLY_ADD = 0x222u;
constexpr uint32_t OP_POLY_MUL = 0x223u;
constexpr uint32_t OP_CAS_PUSH_SYM = 0x234u;
constexpr uint32_t OP_CAS_PUSH_CONST = 0x235u;

constexpr uint32_t kCasTraversalCap = 128u;
constexpr uint32_t kCasCoeffLiteralCap = 1024u;

__device__ inline uint32_t cas_pack_poly_meta(uint32_t symbol_id, uint32_t degree) {
    return ((degree & 0xFFFFu) << 16) | (symbol_id & 0xFFFFu);
}

__device__ inline uint32_t cas_poly_symbol(const StarNode& node) {
    return node.next & 0xFFFFu;
}

__device__ inline uint32_t cas_poly_degree(const StarNode& node) {
    return (node.next >> 16) & 0xFFFFu;
}

__device__ inline uint32_t cas_alloc_node() {
    const uint32_t idx = atomicAdd(&g_cas_pool_top, 1u);
    if (idx >= CAS_POOL_SIZE) {
        return CAS_NULL_IDX;
    }
    return idx;
}

__device__ inline uint32_t cas_alloc_coeffs(uint32_t count) {
    const uint32_t offset = atomicAdd(&g_cas_coeff_top, count);
    if (offset >= CAS_COEFF_SIZE || count > (CAS_COEFF_SIZE - offset)) {
        return CAS_NULL_IDX;
    }
    return offset;
}

__device__ inline uint32_t cas_make_const(float value) {
    const uint32_t idx = cas_alloc_node();
    if (idx == CAS_NULL_IDX) {
        return idx;
    }
    g_cas_pool[idx].opcode = OP_CAS_PUSH_CONST;
    g_cas_pool[idx].flags = STAR_FLAGS(0, TAG_FLOAT, 0);
    g_cas_pool[idx].data.immf32 = value;
    g_cas_pool[idx].next = CAS_NULL_IDX;
    return idx;
}

__device__ inline uint32_t cas_make_symbol(uint32_t symbol_id) {
    const uint32_t idx = cas_alloc_node();
    if (idx == CAS_NULL_IDX) {
        return idx;
    }
    g_cas_pool[idx].opcode = OP_CAS_PUSH_SYM;
    g_cas_pool[idx].flags = STAR_FLAGS(0, TAG_SYMBOL, 0);
    g_cas_pool[idx].data.payload = symbol_id;
    g_cas_pool[idx].next = CAS_NULL_IDX;
    return idx;
}

__device__ inline uint32_t cas_make_unary(uint32_t opcode, uint32_t child) {
    const uint32_t idx = cas_alloc_node();
    if (idx == CAS_NULL_IDX) {
        return idx;
    }
    g_cas_pool[idx].opcode = opcode;
    g_cas_pool[idx].flags = STAR_FLAGS(1, TAG_FLOAT, 0);
    STAR_CHILD0(g_cas_pool[idx]) = child;
    STAR_CHILD1(g_cas_pool[idx]) = CAS_NULL_IDX;
    return idx;
}

__device__ inline uint32_t cas_make_binary(uint32_t opcode, uint32_t left, uint32_t right) {
    const uint32_t idx = cas_alloc_node();
    if (idx == CAS_NULL_IDX) {
        return idx;
    }
    g_cas_pool[idx].opcode = opcode;
    g_cas_pool[idx].flags = STAR_FLAGS(2, TAG_FLOAT, 0);
    STAR_CHILD0(g_cas_pool[idx]) = left;
    STAR_CHILD1(g_cas_pool[idx]) = right;
    return idx;
}

__device__ inline uint32_t cas_make_poly(uint32_t symbol_id, const float* coeffs, uint32_t coeff_count) {
    if (coeff_count == 0u) {
        return cas_make_const(0.0f);
    }
    const uint32_t coeff_offset = cas_alloc_coeffs(coeff_count);
    if (coeff_offset == CAS_NULL_IDX) {
        return CAS_NULL_IDX;
    }
    for (uint32_t idx = 0; idx < coeff_count; ++idx) {
        g_cas_coeffs[coeff_offset + idx] = coeffs[idx];
    }
    const uint32_t node_idx = cas_alloc_node();
    if (node_idx == CAS_NULL_IDX) {
        return node_idx;
    }
    g_cas_pool[node_idx].opcode = OP_POLY_BUILD;
    g_cas_pool[node_idx].flags = STAR_FLAGS(0, TAG_POLY, 0);
    g_cas_pool[node_idx].data.payload = coeff_offset;
    g_cas_pool[node_idx].next = cas_pack_poly_meta(symbol_id, coeff_count - 1u);
    return node_idx;
}

__device__ inline bool cas_same_poly(const StarNode& a, const StarNode& b) {
    if (cas_poly_symbol(a) != cas_poly_symbol(b)) {
        return false;
    }
    const uint32_t degree_a = cas_poly_degree(a);
    const uint32_t degree_b = cas_poly_degree(b);
    if (degree_a != degree_b) {
        return false;
    }
    const uint32_t coeff_count = degree_a + 1u;
    for (uint32_t idx = 0u; idx < coeff_count; ++idx) {
        if (fabsf(g_cas_coeffs[a.data.payload + idx] - g_cas_coeffs[b.data.payload + idx]) > 1.0e-6f) {
            return false;
        }
    }
    return true;
}

__device__ inline bool cas_node_is_const(uint32_t idx, float* value_out = nullptr) {
    if (idx == CAS_NULL_IDX) {
        return false;
    }
    const StarNode& node = g_cas_pool[idx];
    if (STAR_ARITY(node.flags) != 0u || STAR_TAG(node.flags) != TAG_FLOAT) {
        return false;
    }
    if (node.opcode != OP_CAS_PUSH_CONST) {
        return false;
    }
    if (value_out != nullptr) {
        *value_out = node.data.immf32;
    }
    return true;
}

__device__ inline bool cas_node_is_symbol(uint32_t idx, uint32_t* symbol_out = nullptr) {
    if (idx == CAS_NULL_IDX) {
        return false;
    }
    const StarNode& node = g_cas_pool[idx];
    if (STAR_ARITY(node.flags) != 0u || STAR_TAG(node.flags) != TAG_SYMBOL) {
        return false;
    }
    if (node.opcode != OP_CAS_PUSH_SYM) {
        return false;
    }
    if (symbol_out != nullptr) {
        *symbol_out = node.data.payload;
    }
    return true;
}

__device__ inline bool cas_same_expr(uint32_t lhs, uint32_t rhs) {
    if (lhs == rhs) {
        return true;
    }
    if (lhs == CAS_NULL_IDX || rhs == CAS_NULL_IDX) {
        return false;
    }
    const StarNode& a = g_cas_pool[lhs];
    const StarNode& b = g_cas_pool[rhs];
    if (a.opcode != b.opcode || a.flags != b.flags || a.next != b.next) {
        return false;
    }
    if (STAR_TAG(a.flags) == TAG_POLY) {
        return cas_same_poly(a, b);
    }
    const uint32_t arity = STAR_ARITY(a.flags);
    if (arity == 0u) {
        if (STAR_TAG(a.flags) == TAG_FLOAT) {
            return fabsf(a.data.immf32 - b.data.immf32) < 1.0e-6f;
        }
        return a.data.payload == b.data.payload;
    }
    if (arity == 1u) {
        return cas_same_expr(STAR_CHILD0(a), STAR_CHILD0(b));
    }
    return cas_same_expr(STAR_CHILD0(a), STAR_CHILD0(b)) &&
           cas_same_expr(STAR_CHILD1(a), STAR_CHILD1(b));
}

__device__ inline float cas_eval_node(uint32_t idx, uint32_t symbol_id, float symbol_value) {
    if (idx == CAS_NULL_IDX) {
        return 0.0f;
    }
    const StarNode& node = g_cas_pool[idx];
    if (STAR_ARITY(node.flags) == 0u) {
        if (STAR_TAG(node.flags) == TAG_FLOAT) {
            return node.data.immf32;
        }
        if (STAR_TAG(node.flags) == TAG_SYMBOL) {
            if (node.data.payload == symbol_id) {
                return symbol_value;
            }
            if (node.data.payload == SYM_PI) {
                return 3.1415926535f;
            }
            if (node.data.payload == SYM_E) {
                return 2.7182818284f;
            }
            return 0.0f;
        }
    }
    if (STAR_TAG(node.flags) == TAG_POLY) {
        const uint32_t degree = cas_poly_degree(node);
        const uint32_t coeff_offset = node.data.payload;
        float acc = 0.0f;
        for (uint32_t idx_coeff = 0; idx_coeff <= degree; ++idx_coeff) {
            acc = acc * symbol_value + g_cas_coeffs[coeff_offset + idx_coeff];
        }
        return acc;
    }
    const uint32_t child0 = STAR_CHILD0(node);
    const float lhs = cas_eval_node(child0, symbol_id, symbol_value);
    if (STAR_ARITY(node.flags) == 1u) {
        switch (node.opcode) {
            case OP_NEGATE: return -lhs;
            case OP_SIN: return sinf(lhs);
            case OP_COS: return cosf(lhs);
            case OP_EXP: return expf(lhs);
            case OP_LOG: return lhs > 0.0f ? logf(lhs) : 0.0f;
            default: return lhs;
        }
    }
    const float rhs = cas_eval_node(STAR_CHILD1(node), symbol_id, symbol_value);
    switch (node.opcode) {
        case OP_ADD: return lhs + rhs;
        case OP_SUB: return lhs - rhs;
        case OP_MUL: return lhs * rhs;
        case OP_DIV: return fabsf(rhs) > 1.0e-8f ? lhs / rhs : 0.0f;
        case OP_POWER: return powf(lhs, rhs);
        default: return 0.0f;
    }
}

__device__ inline uint32_t cas_find_result(uint32_t key, const uint32_t* keys, const uint32_t* values, uint32_t count) {
    for (uint32_t idx = 0; idx < count; ++idx) {
        if (keys[idx] == key) {
            return values[idx];
        }
    }
    return CAS_NULL_IDX;
}

__device__ inline bool cas_match_power_of_trig(uint32_t idx, uint32_t trig_opcode, uint32_t* arg_out) {
    if (idx == CAS_NULL_IDX) {
        return false;
    }
    const StarNode& power = g_cas_pool[idx];
    if (power.opcode != OP_POWER || STAR_ARITY(power.flags) != 2u) {
        return false;
    }
    float exponent = 0.0f;
    if (!cas_node_is_const(STAR_CHILD1(power), &exponent) || fabsf(exponent - 2.0f) > 1.0e-6f) {
        return false;
    }
    const StarNode& trig = g_cas_pool[STAR_CHILD0(power)];
    if (trig.opcode != trig_opcode || STAR_ARITY(trig.flags) != 1u) {
        return false;
    }
    if (arg_out != nullptr) {
        *arg_out = STAR_CHILD0(trig);
    }
    return true;
}

__device__ inline uint32_t cas_simple_simplify(uint32_t root_idx) {
    if (root_idx == CAS_NULL_IDX) {
        return root_idx;
    }
    const StarNode& root = g_cas_pool[root_idx];
    const uint32_t arity = STAR_ARITY(root.flags);
    if (arity == 0u || STAR_TAG(root.flags) == TAG_POLY) {
        return root_idx;
    }
    if (arity == 1u && root.opcode == OP_NEGATE) {
        const uint32_t child_idx = STAR_CHILD0(root);
        if (child_idx != CAS_NULL_IDX && g_cas_pool[child_idx].opcode == OP_NEGATE) {
            return STAR_CHILD0(g_cas_pool[child_idx]);
        }
        float child_const = 0.0f;
        if (cas_node_is_const(child_idx, &child_const)) {
            return cas_make_const(-child_const);
        }
        return root_idx;
    }
    if (arity != 2u) {
        return root_idx;
    }

    const uint32_t lhs_idx = STAR_CHILD0(root);
    const uint32_t rhs_idx = STAR_CHILD1(root);
    float lhs_const = 0.0f;
    float rhs_const = 0.0f;
    const bool lhs_is_const = cas_node_is_const(lhs_idx, &lhs_const);
    const bool rhs_is_const = cas_node_is_const(rhs_idx, &rhs_const);

    if (lhs_is_const && rhs_is_const) {
        switch (root.opcode) {
            case OP_ADD: return cas_make_const(lhs_const + rhs_const);
            case OP_SUB: return cas_make_const(lhs_const - rhs_const);
            case OP_MUL: return cas_make_const(lhs_const * rhs_const);
            case OP_DIV: return cas_make_const(fabsf(rhs_const) > 1.0e-8f ? lhs_const / rhs_const : 0.0f);
            case OP_POWER: return cas_make_const(powf(lhs_const, rhs_const));
            default: break;
        }
    }

    switch (root.opcode) {
        case OP_ADD:
            if (lhs_is_const && fabsf(lhs_const) < 1.0e-6f) return rhs_idx;
            if (rhs_is_const && fabsf(rhs_const) < 1.0e-6f) return lhs_idx;
            {
                uint32_t sin_arg = CAS_NULL_IDX;
                uint32_t cos_arg = CAS_NULL_IDX;
                if (cas_match_power_of_trig(lhs_idx, OP_SIN, &sin_arg) &&
                    cas_match_power_of_trig(rhs_idx, OP_COS, &cos_arg) &&
                    cas_same_expr(sin_arg, cos_arg)) {
                    return cas_make_const(1.0f);
                }
                if (cas_match_power_of_trig(lhs_idx, OP_COS, &cos_arg) &&
                    cas_match_power_of_trig(rhs_idx, OP_SIN, &sin_arg) &&
                    cas_same_expr(sin_arg, cos_arg)) {
                    return cas_make_const(1.0f);
                }
            }
            break;
        case OP_SUB:
            if (rhs_is_const && fabsf(rhs_const) < 1.0e-6f) return lhs_idx;
            break;
        case OP_MUL:
            if ((lhs_is_const && fabsf(lhs_const) < 1.0e-6f) || (rhs_is_const && fabsf(rhs_const) < 1.0e-6f)) {
                return cas_make_const(0.0f);
            }
            if (lhs_is_const && fabsf(lhs_const - 1.0f) < 1.0e-6f) return rhs_idx;
            if (rhs_is_const && fabsf(rhs_const - 1.0f) < 1.0e-6f) return lhs_idx;
            break;
        case OP_DIV:
            if (lhs_is_const && fabsf(lhs_const) < 1.0e-6f) return cas_make_const(0.0f);
            if (rhs_is_const && fabsf(rhs_const - 1.0f) < 1.0e-6f) return lhs_idx;
            break;
        case OP_POWER:
            if (rhs_is_const && fabsf(rhs_const - 1.0f) < 1.0e-6f) return lhs_idx;
            if (rhs_is_const && fabsf(rhs_const) < 1.0e-6f) return cas_make_const(1.0f);
            break;
        default:
            break;
    }
    return root_idx;
}

}  // namespace

extern "C" __global__ void k3d_expr_build(
    const uint32_t* __restrict__ program,
    uint32_t program_len,
    uint32_t* out_root_idx
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    uint32_t eval_stack[kCasTraversalCap];
    uint32_t sp = 0u;
    uint32_t cursor = 0u;

    while (cursor < program_len) {
        const uint32_t opcode = program[cursor++];
        if (opcode == OP_CAS_PUSH_CONST) {
            if (cursor >= program_len || sp >= kCasTraversalCap) {
                *out_root_idx = CAS_NULL_IDX;
                return;
            }
            const float value = __uint_as_float(program[cursor++]);
            eval_stack[sp++] = cas_make_const(value);
            continue;
        }
        if (opcode == OP_CAS_PUSH_SYM) {
            if (cursor >= program_len || sp >= kCasTraversalCap) {
                *out_root_idx = CAS_NULL_IDX;
                return;
            }
            eval_stack[sp++] = cas_make_symbol(program[cursor++]);
            continue;
        }
        if (opcode == OP_POLY_BUILD) {
            if ((cursor + 1u) >= program_len || sp >= kCasTraversalCap) {
                *out_root_idx = CAS_NULL_IDX;
                return;
            }
            const uint32_t symbol_id = program[cursor++];
            const uint32_t coeff_count = program[cursor++];
            if (coeff_count == 0u || coeff_count > kCasCoeffLiteralCap || (cursor + coeff_count) > program_len) {
                *out_root_idx = CAS_NULL_IDX;
                return;
            }
            float coeffs[kCasCoeffLiteralCap];
            for (uint32_t idx = 0u; idx < coeff_count; ++idx) {
                coeffs[idx] = __uint_as_float(program[cursor++]);
            }
            eval_stack[sp++] = cas_make_poly(symbol_id, coeffs, coeff_count);
            continue;
        }
        if (sp == 0u) {
            *out_root_idx = CAS_NULL_IDX;
            return;
        }
        if (opcode == OP_NEGATE || opcode == OP_SIN || opcode == OP_COS || opcode == OP_EXP || opcode == OP_LOG) {
            const uint32_t child = eval_stack[sp - 1u];
            eval_stack[sp - 1u] = cas_make_unary(opcode, child);
            continue;
        }
        if (sp < 2u) {
            *out_root_idx = CAS_NULL_IDX;
            return;
        }
        const uint32_t rhs = eval_stack[--sp];
        const uint32_t lhs = eval_stack[--sp];
        eval_stack[sp++] = cas_make_binary(opcode, lhs, rhs);
    }

    *out_root_idx = sp == 0u ? CAS_NULL_IDX : eval_stack[sp - 1u];
}

extern "C" __global__ void k3d_diff(
    uint32_t root_idx,
    uint32_t var_sym_id,
    uint32_t* out_root_idx
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (root_idx == CAS_NULL_IDX) {
        *out_root_idx = CAS_NULL_IDX;
        return;
    }

    uint32_t node_stack[kCasTraversalCap];
    uint8_t state_stack[kCasTraversalCap];
    uint32_t postorder[kCasTraversalCap];
    uint32_t result_keys[kCasTraversalCap];
    uint32_t result_values[kCasTraversalCap];
    uint32_t post_count = 0u;
    int32_t sp = 0;
    node_stack[0] = root_idx;
    state_stack[0] = 0u;

    while (sp >= 0) {
        const uint32_t node_idx = node_stack[sp];
        if (node_idx == CAS_NULL_IDX) {
            sp -= 1;
            continue;
        }
        const StarNode& node = g_cas_pool[node_idx];
        const uint32_t arity = STAR_ARITY(node.flags);
        if (arity == 0u || STAR_TAG(node.flags) == TAG_POLY) {
            postorder[post_count++] = node_idx;
            sp -= 1;
            continue;
        }
        if (state_stack[sp] == 0u) {
            state_stack[sp] = 1u;
            node_stack[++sp] = STAR_CHILD0(node);
            state_stack[sp] = 0u;
            continue;
        }
        if (arity > 1u && state_stack[sp] == 1u) {
            state_stack[sp] = 2u;
            node_stack[++sp] = STAR_CHILD1(node);
            state_stack[sp] = 0u;
            continue;
        }
        postorder[post_count++] = node_idx;
        sp -= 1;
    }

    uint32_t result_count = 0u;
    for (uint32_t idx = 0; idx < post_count; ++idx) {
        const uint32_t node_idx = postorder[idx];
        const StarNode& node = g_cas_pool[node_idx];
        uint32_t deriv_idx = CAS_NULL_IDX;

        if (STAR_TAG(node.flags) == TAG_POLY) {
            const uint32_t degree = cas_poly_degree(node);
            const uint32_t symbol_id = cas_poly_symbol(node);
            if (degree == 0u || symbol_id != var_sym_id) {
                deriv_idx = cas_make_const(0.0f);
            } else if (degree > 63u) {
                deriv_idx = CAS_NULL_IDX;
            } else {
                float coeffs[64];
                for (uint32_t c = 0; c < degree; ++c) {
                    const uint32_t power = degree - c;
                    coeffs[c] = g_cas_coeffs[node.data.payload + c] * static_cast<float>(power);
                }
                deriv_idx = cas_make_poly(symbol_id, coeffs, degree);
            }
        } else if (STAR_ARITY(node.flags) == 0u) {
            uint32_t symbol_id = 0u;
            if (cas_node_is_symbol(node_idx, &symbol_id)) {
                deriv_idx = cas_make_const(symbol_id == var_sym_id ? 1.0f : 0.0f);
            } else {
                deriv_idx = cas_make_const(0.0f);
            }
        } else if (STAR_ARITY(node.flags) == 1u) {
            const uint32_t child = STAR_CHILD0(node);
            const uint32_t dchild = cas_find_result(child, result_keys, result_values, result_count);
            switch (node.opcode) {
                case OP_NEGATE:
                    deriv_idx = cas_make_unary(OP_NEGATE, dchild);
                    break;
                case OP_SIN:
                    deriv_idx = cas_make_binary(OP_MUL, cas_make_unary(OP_COS, child), dchild);
                    break;
                case OP_COS:
                    deriv_idx = cas_make_binary(OP_MUL, cas_make_unary(OP_NEGATE, cas_make_unary(OP_SIN, child)), dchild);
                    break;
                case OP_EXP:
                    deriv_idx = cas_make_binary(OP_MUL, cas_make_unary(OP_EXP, child), dchild);
                    break;
                case OP_LOG:
                    deriv_idx = cas_make_binary(OP_DIV, dchild, child);
                    break;
                default:
                    deriv_idx = cas_make_const(0.0f);
                    break;
            }
        } else {
            const uint32_t lhs = STAR_CHILD0(node);
            const uint32_t rhs = STAR_CHILD1(node);
            const uint32_t dlhs = cas_find_result(lhs, result_keys, result_values, result_count);
            const uint32_t drhs = cas_find_result(rhs, result_keys, result_values, result_count);
            switch (node.opcode) {
                case OP_ADD:
                    deriv_idx = cas_make_binary(OP_ADD, dlhs, drhs);
                    break;
                case OP_SUB:
                    deriv_idx = cas_make_binary(OP_SUB, dlhs, drhs);
                    break;
                case OP_MUL: {
                    const uint32_t term0 = cas_make_binary(OP_MUL, dlhs, rhs);
                    const uint32_t term1 = cas_make_binary(OP_MUL, lhs, drhs);
                    deriv_idx = cas_make_binary(OP_ADD, term0, term1);
                    break;
                }
                case OP_DIV: {
                    const uint32_t num0 = cas_make_binary(OP_MUL, dlhs, rhs);
                    const uint32_t num1 = cas_make_binary(OP_MUL, lhs, drhs);
                    const uint32_t numerator = cas_make_binary(OP_SUB, num0, num1);
                    const uint32_t denominator = cas_make_binary(OP_POWER, rhs, cas_make_const(2.0f));
                    deriv_idx = cas_make_binary(OP_DIV, numerator, denominator);
                    break;
                }
                case OP_POWER: {
                    float exponent = 0.0f;
                    if (cas_node_is_const(rhs, &exponent)) {
                        const uint32_t scale = cas_make_const(exponent);
                        const uint32_t reduced_exp = cas_make_const(exponent - 1.0f);
                        const uint32_t power = cas_make_binary(OP_POWER, lhs, reduced_exp);
                        deriv_idx = cas_make_binary(OP_MUL, cas_make_binary(OP_MUL, scale, power), dlhs);
                    } else {
                        deriv_idx = cas_make_const(0.0f);
                    }
                    break;
                }
                default:
                    deriv_idx = cas_make_const(0.0f);
                    break;
            }
        }

        result_keys[result_count] = node_idx;
        result_values[result_count] = cas_simple_simplify(deriv_idx);
        result_count += 1u;
    }

    *out_root_idx = result_count == 0u ? CAS_NULL_IDX : result_values[result_count - 1u];
}

extern "C" __global__ void k3d_poly_mul(
    uint32_t poly_a_idx,
    uint32_t poly_b_idx,
    uint32_t* out_root_idx
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (poly_a_idx == CAS_NULL_IDX || poly_b_idx == CAS_NULL_IDX) {
        *out_root_idx = CAS_NULL_IDX;
        return;
    }
    const StarNode& a = g_cas_pool[poly_a_idx];
    const StarNode& b = g_cas_pool[poly_b_idx];
    if (STAR_TAG(a.flags) != TAG_POLY || STAR_TAG(b.flags) != TAG_POLY) {
        *out_root_idx = CAS_NULL_IDX;
        return;
    }
    if (cas_poly_symbol(a) != cas_poly_symbol(b)) {
        *out_root_idx = CAS_NULL_IDX;
        return;
    }
    const uint32_t degree_a = cas_poly_degree(a);
    const uint32_t degree_b = cas_poly_degree(b);
    const uint32_t coeff_count = degree_a + degree_b + 1u;
    if (coeff_count > 64u) {
        *out_root_idx = CAS_NULL_IDX;
        return;
    }
    float coeffs[64];
    for (uint32_t idx = 0; idx < coeff_count; ++idx) {
        coeffs[idx] = 0.0f;
    }
    for (uint32_t ia = 0; ia <= degree_a; ++ia) {
        for (uint32_t ib = 0; ib <= degree_b; ++ib) {
            coeffs[ia + ib] += g_cas_coeffs[a.data.payload + ia] * g_cas_coeffs[b.data.payload + ib];
        }
    }
    *out_root_idx = cas_make_poly(cas_poly_symbol(a), coeffs, coeff_count);
}

extern "C" __global__ void k3d_simplify(
    uint32_t root_idx,
    uint32_t* out_root_idx
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    *out_root_idx = cas_simple_simplify(root_idx);
}
