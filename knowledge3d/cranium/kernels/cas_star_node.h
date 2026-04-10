#pragma once

#include <stdint.h>

// ──────────────────────────────────────────────────────────────
//  StarNode — 16-byte aligned, zero pointers, pool-indexed
//  All references are uint32_t pool indices (never raw pointers)
// ──────────────────────────────────────────────────────────────
struct StarNode {
    uint32_t opcode;      // K3D opcode — OP_ADD/OP_MUL/OP_SIN/OP_VAR_X/OP_CONST etc.
    uint32_t flags;       // Bits 0-7: arity, Bits 8-15: type_tag, Bits 16-23: refcount
    union {
        float immf32;         // Immediate float (arity == 0, type_tag == TAG_FLOAT)
        int32_t immi32;       // Immediate integer (arity == 0, type_tag == TAG_INT)
        uint32_t payload;     // symbol_id / coeff_buf_offset / child0 (binary and unary)
        uint32_t child0;      // left child pool index when arity >= 1
    } data;
    uint32_t next;            // child1 / free-pool chain / metadata chain
} __attribute__((aligned(16)));

static_assert(sizeof(StarNode) == 16, "StarNode must be 16 bytes");

// Type tags (flags bits 8-15)
#define TAG_FLOAT   0x01
#define TAG_INT     0x02
#define TAG_SYMBOL  0x03
#define TAG_POLY    0x04    // payload = offset into polynomial coefficient buffer

// Flags macros
#define STAR_FLAGS(arity, type_tag, refcount) \
    (((uint32_t)(arity) & 0xFFu) | (((uint32_t)(type_tag) & 0xFFu) << 8) | (((uint32_t)(refcount) & 0xFFu) << 16))
#define STAR_ARITY(flags)    ((flags) & 0xFFu)
#define STAR_TAG(flags)      (((flags) >> 8) & 0xFFu)
#define STAR_REFCOUNT(flags) (((flags) >> 16) & 0xFFu)
#define STAR_CHILD0(node)    ((node).data.child0)
#define STAR_CHILD1(node)    ((node).next)

// Pool constants
#define CAS_POOL_SIZE    (1u << 20)   // 1M nodes = 16MB VRAM
#define CAS_COEFF_SIZE   (1u << 18)   // 256K float32 coefficients = 1MB VRAM
#define CAS_NULL_IDX     0xFFFFFFFFu  // null/invalid index

// Symbol IDs (interned on CPU, stored in __constant__ memory)
#define SYM_X  0x01u
#define SYM_Y  0x02u
#define SYM_Z  0x03u
#define SYM_W  0x04u
#define SYM_PI 0x10u
#define SYM_E  0x11u
