from __future__ import annotations

import ctypes


REASONING_SLOT_NONE = 0
REASONING_SLOT_CBR = 1
REASONING_SLOT_SUPERPOS = 2
REASONING_SLOT_BIDUCE = 3
REASONING_SLOT_EBELIEF = 4
REASONING_SLOT_RETE = 5
REASONING_SLOT_TABLEAUX = 6
REASONING_SLOT_RESOLUTION = 7
REASONING_SLOT_ALPCHAIN = 8
REASONING_SLOT_DPLL = 9
REASONING_SLOT_CTX_SWITCH = 10
REASONING_SLOT_SUBSUME = 11
REASONING_SLOT_UNIFY = 12


def pack_case(case_id: int, anchor: int, context_id: int, ethical_code: int, flags: int = 0) -> int:
    handle = case_id & 0x3F
    handle |= (anchor & 0xFF) << 6
    handle |= (context_id & 0x3F) << 14
    handle |= (ethical_code & 0x3) << 20
    handle |= (flags & 0x3) << 22
    return handle


def pack_rule(lhs: int, rhs: int) -> int:
    return (lhs & 0xFFF) | ((rhs & 0xFFF) << 12)


def pack_opinion(belief: int, disbelief: int, uncertainty: int, status: int = 0) -> int:
    handle = belief & 0x7F
    handle |= (disbelief & 0x7F) << 7
    handle |= (uncertainty & 0x7F) << 14
    handle |= (status & 0x3) << 21
    return handle


def pack_fact(predicate_mask: int, context_id: int, cluster_id: int, ethical_code: int) -> int:
    handle = predicate_mask & 0xFF
    handle |= (context_id & 0xFF) << 8
    handle |= (cluster_id & 0xF) << 16
    handle |= (ethical_code & 0x3) << 20
    return handle


def pack_alpha(required_mask: int, required_context: int, required_cluster: int, ethical_policy: int, heuristic_floor: int = 0) -> int:
    handle = pack_fact(required_mask, required_context, required_cluster, ethical_policy)
    handle |= (heuristic_floor & 0x3) << 22
    return handle


def pack_branch(node_id: int, concept_mask: int) -> int:
    return (node_id & 0xFF) | ((concept_mask & 0xFFFF) << 8)


def pack_horn_rule(head_symbol: int, body_mask: int, ic_mask: int = 0) -> int:
    return (head_symbol & 0xFF) | ((body_mask & 0xFF) << 8) | ((ic_mask & 0xFF) << 16)


def pack_clause(positive_mask: int, negative_mask: int) -> int:
    return (positive_mask & 0xFFFF) | ((negative_mask & 0xFFFF) << 16)


def pack_trail(true_mask: int, false_mask: int = 0) -> int:
    return (true_mask & 0xFFFF) | ((false_mask & 0xFFFF) << 16)


def pack_ctx_view(context_id: int, include_global: int, ethical_trit: int) -> int:
    return (context_id & 0xFFFF) | ((include_global & 0x1) << 16) | ((ctypes.c_uint8(ethical_trit).value) << 24)


def atlas_words(*words: int, halt_after: int = 1, context_id: int = 0, ethical_trit: int = 0) -> bytes:
    payload = [0] * 18
    for index, word in enumerate(words):
        payload[index] = int(word) & 0xFFFFFFFF
    payload[15] = int(halt_after) & 0xFFFFFFFF
    payload[16] = int(context_id) & 0xFFFFFFFF
    payload[17] = ctypes.c_int32(int(ethical_trit)).value & 0xFFFFFFFF
    array = (ctypes.c_uint32 * len(payload))(*payload)
    return bytes(array)


def reference_assign(mask: int, phys_lane_id: int) -> int:
    active_bits = [bit for bit in range(16) if (mask >> bit) & 1]
    if not active_bits:
        return REASONING_SLOT_NONE
    return active_bits[phys_lane_id % len(active_bits)]
