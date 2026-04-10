#include "cas_star_node.h"
#include "sas_hashcons.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

extern __device__ StarNode g_cas_pool[];
extern __device__ float g_cas_coeffs[];
extern __device__ uint32_t g_cas_pool_top;
extern __device__ uint32_t g_cas_coeff_top;

extern "C" {
__device__ HashconsSlot g_sas_hashcons[SAS_HASHCONS_SIZE];
__constant__ float g_sas_symbol_values[256];
__constant__ uint32_t g_sas_symbol_star_ids[256];
}

namespace {
constexpr uint32_t SAS_OP_ADD = 0x0Au;
constexpr uint32_t SAS_OP_SUB = 0x0Bu;
constexpr uint32_t SAS_OP_MUL = 0x0Cu;
constexpr uint32_t SAS_OP_DIV = 0x0Du;
constexpr uint32_t SAS_OP_POWER = 0x0Eu;
constexpr uint32_t SAS_OP_NEGATE = 0xDBu;
constexpr uint32_t SAS_OP_CAS_PUSH_SYM = 0x234u;
constexpr uint32_t SAS_OP_CAS_PUSH_CONST = 0x235u;

constexpr uint32_t kTraversalCap = 128u;
constexpr uint32_t kBindingCap = 16u;
constexpr uint32_t kAssocCap = 16u;
constexpr uint32_t kEmptyBinding = 0xFFFFFFFFu;

__device__ inline uint32_t sas_alloc_node() {
    const uint32_t idx = atomicAdd(&g_cas_pool_top, 1u);
    if (idx >= CAS_POOL_SIZE) {
        return CAS_NULL_IDX;
    }
    return idx;
}

__device__ inline uint32_t sas_make_const(float value) {
    const uint32_t idx = sas_alloc_node();
    if (idx == CAS_NULL_IDX) {
        return idx;
    }
    g_cas_pool[idx].opcode = SAS_OP_CAS_PUSH_CONST;
    g_cas_pool[idx].flags = STAR_FLAGS(0, TAG_FLOAT, 0);
    g_cas_pool[idx].data.immf32 = value;
    g_cas_pool[idx].next = CAS_NULL_IDX;
    return idx;
}

__device__ inline uint32_t sas_make_symbol(uint32_t symbol_id) {
    const uint32_t idx = sas_alloc_node();
    if (idx == CAS_NULL_IDX) {
        return idx;
    }
    g_cas_pool[idx].opcode = SAS_OP_CAS_PUSH_SYM;
    g_cas_pool[idx].flags = STAR_FLAGS(0, TAG_SYMBOL, 0);
    g_cas_pool[idx].data.payload = symbol_id;
    g_cas_pool[idx].next = CAS_NULL_IDX;
    return idx;
}

__device__ inline uint32_t sas_make_unary(uint32_t opcode, uint32_t child) {
    const uint32_t idx = sas_alloc_node();
    if (idx == CAS_NULL_IDX) {
        return idx;
    }
    g_cas_pool[idx].opcode = opcode;
    g_cas_pool[idx].flags = STAR_FLAGS(1, TAG_FLOAT, 0);
    STAR_CHILD0(g_cas_pool[idx]) = child;
    STAR_CHILD1(g_cas_pool[idx]) = CAS_NULL_IDX;
    return idx;
}

__device__ inline uint32_t sas_make_binary(uint32_t opcode, uint32_t lhs, uint32_t rhs) {
    const uint32_t idx = sas_alloc_node();
    if (idx == CAS_NULL_IDX) {
        return idx;
    }
    g_cas_pool[idx].opcode = opcode;
    g_cas_pool[idx].flags = STAR_FLAGS(2, TAG_FLOAT, 0);
    STAR_CHILD0(g_cas_pool[idx]) = lhs;
    STAR_CHILD1(g_cas_pool[idx]) = rhs;
    return idx;
}

__device__ inline bool sas_node_is_const(uint32_t idx, float* value_out = nullptr) {
    if (idx == CAS_NULL_IDX) {
        return false;
    }
    const StarNode& node = g_cas_pool[idx];
    if (STAR_ARITY(node.flags) != 0u || STAR_TAG(node.flags) != TAG_FLOAT || node.opcode != SAS_OP_CAS_PUSH_CONST) {
        return false;
    }
    if (value_out != nullptr) {
        *value_out = node.data.immf32;
    }
    return true;
}

__device__ inline bool sas_node_is_symbol(uint32_t idx, uint32_t* symbol_out = nullptr) {
    if (idx == CAS_NULL_IDX) {
        return false;
    }
    const StarNode& node = g_cas_pool[idx];
    if (STAR_ARITY(node.flags) != 0u || STAR_TAG(node.flags) != TAG_SYMBOL || node.opcode != SAS_OP_CAS_PUSH_SYM) {
        return false;
    }
    if (symbol_out != nullptr) {
        *symbol_out = node.data.payload;
    }
    return true;
}

__device__ inline uint32_t sas_find_result(uint32_t key, const uint32_t* keys, const uint32_t* values, uint32_t count) {
    for (uint32_t idx = 0; idx < count; ++idx) {
        if (keys[idx] == key) {
            return values[idx];
        }
    }
    return CAS_NULL_IDX;
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
    for (uint32_t i = 0; i < count; ++i) {
        for (uint32_t j = i + 1; j < count; ++j) {
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

__device__ inline uint32_t sas_rebuild_commutative_chain(uint32_t root_idx, uint32_t opcode) {
    uint32_t pending[kAssocCap];
    uint32_t operands[kAssocCap];
    uint32_t pending_top = 0u;
    uint32_t operand_count = 0u;
    pending[pending_top++] = root_idx;
    while (pending_top > 0u && operand_count < kAssocCap) {
        const uint32_t current = pending[--pending_top];
        if (current == CAS_NULL_IDX) {
            continue;
        }
        const StarNode& node = g_cas_pool[current];
        if (node.opcode == opcode && STAR_ARITY(node.flags) == 2u) {
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
        current = sas_make_binary(opcode, current, operands[idx]);
        current = sas_hashcons_node(current);
    }
    return current;
}

__device__ inline uint32_t sas_simple_simplify(uint32_t root_idx) {
    if (root_idx == CAS_NULL_IDX) {
        return root_idx;
    }
    const StarNode& root = g_cas_pool[root_idx];
    const uint32_t arity = STAR_ARITY(root.flags);
    if (arity == 0u || STAR_TAG(root.flags) == TAG_POLY) {
        return root_idx;
    }
    if (arity == 1u && root.opcode == SAS_OP_NEGATE) {
        const uint32_t child_idx = STAR_CHILD0(root);
        if (child_idx != CAS_NULL_IDX && g_cas_pool[child_idx].opcode == SAS_OP_NEGATE) {
            return STAR_CHILD0(g_cas_pool[child_idx]);
        }
        float child_const = 0.0f;
        if (sas_node_is_const(child_idx, &child_const)) {
            return sas_make_const(-child_const);
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
    const bool lhs_is_const = sas_node_is_const(lhs_idx, &lhs_const);
    const bool rhs_is_const = sas_node_is_const(rhs_idx, &rhs_const);

    if (lhs_is_const && rhs_is_const) {
        switch (root.opcode) {
            case SAS_OP_ADD: return sas_make_const(lhs_const + rhs_const);
            case SAS_OP_SUB: return sas_make_const(lhs_const - rhs_const);
            case SAS_OP_MUL: return sas_make_const(lhs_const * rhs_const);
            case SAS_OP_DIV: return sas_make_const(fabsf(rhs_const) > 1.0e-8f ? lhs_const / rhs_const : 0.0f);
            case SAS_OP_POWER: return sas_make_const(powf(lhs_const, rhs_const));
            default: break;
        }
    }

    switch (root.opcode) {
        case SAS_OP_ADD:
            if (lhs_is_const && fabsf(lhs_const) < 1.0e-6f) return rhs_idx;
            if (rhs_is_const && fabsf(rhs_const) < 1.0e-6f) return lhs_idx;
            break;
        case SAS_OP_SUB:
            if (rhs_is_const && fabsf(rhs_const) < 1.0e-6f) return lhs_idx;
            break;
        case SAS_OP_MUL:
            if ((lhs_is_const && fabsf(lhs_const) < 1.0e-6f) || (rhs_is_const && fabsf(rhs_const) < 1.0e-6f)) {
                return sas_make_const(0.0f);
            }
            if (lhs_is_const && fabsf(lhs_const - 1.0f) < 1.0e-6f) return rhs_idx;
            if (rhs_is_const && fabsf(rhs_const - 1.0f) < 1.0e-6f) return lhs_idx;
            break;
        case SAS_OP_DIV:
            if (lhs_is_const && fabsf(lhs_const) < 1.0e-6f) return sas_make_const(0.0f);
            if (rhs_is_const && fabsf(rhs_const - 1.0f) < 1.0e-6f) return lhs_idx;
            break;
        case SAS_OP_POWER:
            if (rhs_is_const && fabsf(rhs_const) < 1.0e-6f) return sas_make_const(1.0f);
            if (rhs_is_const && fabsf(rhs_const - 1.0f) < 1.0e-6f) return lhs_idx;
            break;
        default:
            break;
    }
    return root_idx;
}

__device__ inline uint32_t sas_canonicalize_root(uint32_t root_idx) {
    if (root_idx == CAS_NULL_IDX) {
        return root_idx;
    }
    uint32_t stack[kTraversalCap];
    uint8_t state[kTraversalCap];
    uint32_t result_keys[kTraversalCap];
    uint32_t result_values[kTraversalCap];
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
            if (arity >= 2u && top < kTraversalCap) {
                stack[top] = STAR_CHILD1(node);
                state[top++] = 0u;
            }
            if (top < kTraversalCap) {
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

        uint32_t canonical = sas_simple_simplify(idx);
        if (canonical != CAS_NULL_IDX) {
            const StarNode& canonical_node = g_cas_pool[canonical];
            if (STAR_ARITY(canonical_node.flags) == 2u &&
                (canonical_node.opcode == SAS_OP_ADD || canonical_node.opcode == SAS_OP_MUL)) {
                canonical = sas_rebuild_commutative_chain(canonical, canonical_node.opcode);
            }
            canonical = sas_simple_simplify(canonical);
            canonical = sas_hashcons_node(canonical);
        }

        if (result_count < kTraversalCap) {
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

__device__ inline bool sas_same_expr(uint32_t lhs, uint32_t rhs) {
    uint32_t lhs_stack[kTraversalCap];
    uint32_t rhs_stack[kTraversalCap];
    uint32_t top = 0u;
    lhs_stack[top] = lhs;
    rhs_stack[top++] = rhs;
    while (top > 0u) {
        const uint32_t a_idx = lhs_stack[--top];
        const uint32_t b_idx = rhs_stack[top];
        if (a_idx == b_idx) {
            continue;
        }
        if (a_idx == CAS_NULL_IDX || b_idx == CAS_NULL_IDX) {
            return false;
        }
        const StarNode& a = g_cas_pool[a_idx];
        const StarNode& b = g_cas_pool[b_idx];
        if (a.opcode != b.opcode || a.flags != b.flags) {
            return false;
        }
        if (STAR_ARITY(a.flags) == 0u) {
            if (STAR_TAG(a.flags) == TAG_FLOAT) {
                if (fabsf(a.data.immf32 - b.data.immf32) > 1.0e-6f) {
                    return false;
                }
            } else if (STAR_TAG(a.flags) == TAG_POLY) {
                if (a.next != b.next) {
                    return false;
                }
                const uint32_t coeff_count = (((a.next >> 16) & 0xFFFFu) + 1u);
                for (uint32_t idx = 0u; idx < coeff_count; ++idx) {
                    if (fabsf(g_cas_coeffs[a.data.payload + idx] - g_cas_coeffs[b.data.payload + idx]) > 1.0e-6f) {
                        return false;
                    }
                }
            } else if (a.data.payload != b.data.payload) {
                return false;
            }
            continue;
        }
        lhs_stack[top] = STAR_CHILD0(a);
        rhs_stack[top++] = STAR_CHILD0(b);
        if (STAR_ARITY(a.flags) >= 2u) {
            lhs_stack[top] = STAR_CHILD1(a);
            rhs_stack[top++] = STAR_CHILD1(b);
        }
    }
    return true;
}

__device__ inline bool sas_match_pattern(
    uint32_t pattern_root_idx,
    uint32_t subject_root_idx,
    uint32_t* binding_var_ids,
    uint32_t* binding_subj_idxs,
    uint32_t* binding_count)
{
    uint32_t pattern_stack[kTraversalCap];
    uint32_t subject_stack[kTraversalCap];
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
        if (sas_node_is_symbol(pattern_idx, &symbol_id)) {
            const int bound_idx = sas_binding_index(binding_var_ids, *binding_count, symbol_id);
            if (bound_idx >= 0) {
                if (!sas_same_expr(binding_subj_idxs[bound_idx], subject_idx)) {
                    return false;
                }
                continue;
            }
            if (*binding_count >= kBindingCap) {
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
    uint32_t binding_count)
{
    uint32_t stack[kTraversalCap];
    uint8_t state[kTraversalCap];
    uint32_t result_keys[kTraversalCap];
    uint32_t result_values[kTraversalCap];
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
            if (arity >= 2u && top < kTraversalCap) {
                stack[top] = STAR_CHILD1(node);
                state[top++] = 0u;
            }
            if (top < kTraversalCap) {
                stack[top] = STAR_CHILD0(node);
                state[top++] = 0u;
            }
            continue;
        }

        uint32_t materialized = idx;
        uint32_t symbol_id = 0u;
        if (sas_node_is_symbol(idx, &symbol_id)) {
            materialized = sas_lookup_binding(symbol_id, binding_var_ids, binding_subj_idxs, binding_count);
            if (materialized == CAS_NULL_IDX) {
                materialized = sas_make_symbol(symbol_id);
            }
        } else if (sas_node_is_const(idx)) {
            materialized = sas_make_const(node.data.immf32);
        } else if (arity == 1u) {
            const uint32_t child0 = sas_find_result(STAR_CHILD0(node), result_keys, result_values, result_count);
            materialized = sas_make_unary(node.opcode, child0 == CAS_NULL_IDX ? STAR_CHILD0(node) : child0);
        } else if (arity >= 2u) {
            const uint32_t child0 = sas_find_result(STAR_CHILD0(node), result_keys, result_values, result_count);
            const uint32_t child1 = sas_find_result(STAR_CHILD1(node), result_keys, result_values, result_count);
            materialized = sas_make_binary(
                node.opcode,
                child0 == CAS_NULL_IDX ? STAR_CHILD0(node) : child0,
                child1 == CAS_NULL_IDX ? STAR_CHILD1(node) : child1);
        }

        if (result_count < kTraversalCap) {
            result_keys[result_count] = idx;
            result_values[result_count] = materialized;
            result_count += 1u;
        }
    }

    const uint32_t mapped_root = sas_find_result(root_idx, result_keys, result_values, result_count);
    return mapped_root == CAS_NULL_IDX ? root_idx : mapped_root;
}
}  // namespace

extern "C" __global__ void k3d_canonicalize(uint32_t root_idx, uint32_t* out_canon_idx) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    *out_canon_idx = sas_canonicalize_root(root_idx);
}

extern "C" __global__ void k3d_pattern_match(
    uint32_t pattern_root_idx,
    uint32_t subject_root_idx,
    uint32_t* out_binding_var_ids,
    uint32_t* out_binding_subj_idxs,
    uint32_t* out_binding_count,
    uint32_t* out_matched)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    for (uint32_t idx = 0u; idx < kBindingCap; ++idx) {
        out_binding_var_ids[idx] = kEmptyBinding;
        out_binding_subj_idxs[idx] = CAS_NULL_IDX;
    }
    *out_binding_count = 0u;
    *out_matched = sas_match_pattern(
        pattern_root_idx,
        subject_root_idx,
        out_binding_var_ids,
        out_binding_subj_idxs,
        out_binding_count) ? 1u : 0u;
}

extern "C" __global__ void k3d_rule_apply(
    uint32_t replacement_template_idx,
    const uint32_t* binding_var_ids,
    const uint32_t* binding_subj_idxs,
    uint32_t binding_count,
    uint32_t* out_result_idx)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    const uint32_t materialized = sas_materialize_template(
        replacement_template_idx,
        binding_var_ids,
        binding_subj_idxs,
        binding_count);
    *out_result_idx = sas_canonicalize_root(materialized);
}
