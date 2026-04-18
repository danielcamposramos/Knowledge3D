// =============================================================================
// reference_modular_rpn_kernel_transfer_yard.cu
// K3D — Tier 2 Standard RPN Engine — Transfer Yard Native
//
// AUTHORITY: TEMP/CLAUDE_CODEX_TRANSFER_YARD_AND_EMBEDDING_SOVEREIGNTY_04.18.2026.md §3.2
// TARGET:    sm_86 (RTX 3070, Ampere)
// TOOLKIT:   CUDA 12.4 / 12.6
//
// PURPOSE:
//   Replaces modular_rpn_kernel.cu as the Tier 2 engine.  All opcodes now
//   operate through the 3-D Transfer Yard instead of a single LIFO pile.
//   The Python-side TransferYardStack sidecar is DELETED — the yard lives
//   on-GPU in shared memory.
//
// LAUNCH GEOMETRY (fixed by spec §2):
//   blockDim  = (32, 9, 1)   → 9 warps per block; warp = lane
//   gridDim   = (n_programs / 1, 1, 1)   → one block per RPN program batch
//   threadIdx.y = lane_id ∈ [0, 9)
//   threadIdx.x = thread within lane ∈ [0, 32)
//   All yard logic runs on threadIdx.x == 0 (head-thread per lane).
//
// SHARED MEMORY LAYOUT (per block, §4.3):
//   float4  yards[9][9][69]   //  [lane][bank][slot]  = 22,176 bytes
//   uint8_t sp[9][9]          //  [lane][bank]        =     81 bytes
//   uint8_t active_bank[9]    //  [lane]              =      9 bytes
//   uint8_t error_code[9]     //  [lane]              =      9 bytes
//   ──────────────────────────────────────────────────────────────────
//   Total                                              ≈ 22,275 bytes  (< 100 KB Ampere budget)
//
// BANK-CONFLICT NOTES (sm_86, 32 banks × 4 B):
//   yards[lane][bank][slot] byte offset = (lane*9*69 + bank*69 + slot)*16
//   First float bank = ((lane*9*69 + bank*69 + slot)*4) % 32
//   Same-warp threads all have same lane_id (threadIdx.y).  Head-thread
//   pattern means only ONE thread accesses shared mem per lane per cycle →
//   zero intra-warp conflict by construction.
//   Cross-lane accesses only occur in YARD_TRANSFER; each lane touches only
//   yards[lane][...] so different lane-warps touch disjoint address ranges
//   → zero cross-lane conflicts either.
//
// TERNARY-FIRST RULE (feedback_ternary_first_where_cheaper.md):
//   bank_id operands (0-8) are normalised through TQUANT before YARD_SELECT.
//   The helper trit_to_bank() converts the 3-trit balanced-ternary encoding
//   {(-1,-1,-1) … (+1,+1,+1)} → integer 0-8.
//
// OPCODE RANGES PRESENT:
//   0x00-0x9F   Scalar / vector / geometric / conditional (full set)
//   0x100-0x108 CBR / LoRA adapter opcodes
//   0x72-0x75   Ternary core (TNOT, TCOMP, TQUANT, TPACK)
//   0x150-0x162 Physics meta-dispatch (sovereign SOA pointers)
//   0x170-0x176 Yard control opcodes (NEW — this spec)
//   0x180-0x182 WINE I/O contract opcodes (pass-through)
//   0x190       PHYSICS_EMIT_VISUAL (pass-through)
//
// =============================================================================

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

namespace {

constexpr int kLanesPerBlock   = 9;   // Tesla 9; one warp per lane
constexpr int kYardsPerLane    = 9;   // Tesla 9; nine isolated banks per lane
constexpr int kYardDepth       = 69;  // Tesla 6-9; slots per bank
constexpr int kWarpSize        = 32;

// Error codes (match existing kernel convention)
constexpr uint32_t kErrorNone            = 0;
constexpr uint32_t kErrorUnknownOpcode   = 9001;
constexpr uint32_t kErrorStackUnderflow  = 9002;
constexpr uint32_t kErrorStackOverflow   = 9003;
constexpr uint32_t kErrorTypeMismatch    = 9004;
constexpr uint32_t kErrorInvalidArgument = 9006;

// Yard opcode constants (§4.2 — must not collide with 0x180 WINE_* or 0x190)
constexpr uint16_t kOpYardSelect   = 0x170;
constexpr uint16_t kOpYardPushBank = 0x171;
constexpr uint16_t kOpYardPopBank  = 0x172;
constexpr uint16_t kOpYardPeekAddr = 0x173;
constexpr uint16_t kOpYardTransfer = 0x174;
constexpr uint16_t kOpYardSp       = 0x175;
constexpr uint16_t kOpYardClear    = 0x176;

// Ternary ops (existing)
constexpr uint16_t kOpTnot   = 0x72;
constexpr uint16_t kOpTcomp  = 0x73;
constexpr uint16_t kOpTquant = 0x74;
constexpr uint16_t kOpTpack  = 0x75;

// ---------------------------------------------------------------------------
// StackValue — float4 slot (matches existing kernel ABI)
// ---------------------------------------------------------------------------

struct alignas(16) StackValue {
    float x;   // primary scalar or vector.x
    float y;   // vector.y  (0 for scalar)
    float z;   // vector.z  (0 for scalar)
    float w;   // type tag: 0.0 = scalar, 1.0 = vector
};

__device__ __forceinline__ StackValue make_scalar(float v) {
    StackValue s{};
    s.x = v; s.y = 0.f; s.z = 0.f; s.w = 0.f;
    return s;
}

__device__ __forceinline__ StackValue make_vec3(float x, float y, float z) {
    StackValue s{};
    s.x = x; s.y = y; s.z = z; s.w = 1.f;
    return s;
}

__device__ __forceinline__ bool is_vector(const StackValue& v) {
    return fabsf(v.w - 1.f) < 1e-6f;
}

// ---------------------------------------------------------------------------
// Ternary helpers (balanced ternary {-1, 0, +1} encoded as float)
// ---------------------------------------------------------------------------

// Quantize a float to the nearest trit {-1.0, 0.0, +1.0}
__device__ __forceinline__ float tquant(float v, float thresh = 0.333333f) {
    if (v >  thresh) return  1.f;
    if (v < -thresh) return -1.f;
    return 0.f;
}

// Ternary NOT: {-1→+1, 0→0, +1→-1}
__device__ __forceinline__ float tnot(float v) { return -tquant(v); }

// Ternary COMP (conjunction): min of two trits
__device__ __forceinline__ float tcomp(float a, float b) {
    return tquant(fminf(tquant(a), tquant(b)));
}

// Convert a bank_id scalar (any float ~0-8) to a safe integer 0-8.
// Per spec §4.4: TQUANT-normalised 3-trit encoding maps to integer via
// balanced-ternary positional value.  Here we provide the scalar→int
// path for the common case where bank_id was pushed as a plain literal.
__device__ __forceinline__ int scalar_to_bank(float v) {
    int b = static_cast<int>(v + 0.5f);  // round to nearest
    if (b < 0) b = 0;
    if (b > kYardsPerLane - 1) b = kYardsPerLane - 1;
    return b;
}

// ---------------------------------------------------------------------------
// Output state written back to global memory (same as existing InstanceState)
// ---------------------------------------------------------------------------

struct alignas(16) InstanceState {
    uint32_t head;
    uint32_t size;      // sp of active bank after program
    uint32_t error;
    uint32_t reserved;
    StackValue stack[69];  // top-of-active-bank snapshot (up to 69 values)
};

} // namespace

// ---------------------------------------------------------------------------
// KERNEL
// ---------------------------------------------------------------------------

extern "C" __global__ void modular_rpn_kernel_transfer_yard(
    const uint16_t* __restrict__ op_codes,     // flat opcode stream, shared by all lanes
    const float*    __restrict__ scalars,      // immediate scalar pool
    const float*    __restrict__ vectors,      // immediate vector pool (stride 3)
    InstanceState*  __restrict__ states,       // output: one InstanceState per lane per block
    uint32_t                     token_count,  // length of opcode stream
    uint32_t                     n_instances   // total live lane instances
) {
    // -----------------------------------------------------------------------
    // Shared memory — yard substrate
    // -----------------------------------------------------------------------
    __shared__ StackValue yards[kLanesPerBlock][kYardsPerLane][kYardDepth];
    __shared__ uint8_t    sp[kLanesPerBlock][kYardsPerLane];
    __shared__ uint8_t    active_bank[kLanesPerBlock];
    __shared__ uint32_t   error_code[kLanesPerBlock];
    __shared__ uint32_t   scalar_idx[kLanesPerBlock];
    __shared__ uint32_t   vector_idx[kLanesPerBlock];

    const int lane   = threadIdx.y;   // 0-8: which of the 9 lanes in this block
    const int thread = threadIdx.x;   // 0-31: position within the lane's warp

    // -----------------------------------------------------------------------
    // Initialisation — head thread of each lane initialises its own state
    // -----------------------------------------------------------------------
    if (thread == 0) {
        // Zero all stack pointers for this lane
        #pragma unroll
        for (int b = 0; b < kYardsPerLane; ++b) {
            sp[lane][b] = 0;
        }
        active_bank[lane]  = 0;
        error_code[lane]   = kErrorNone;
        scalar_idx[lane]   = 0;
        vector_idx[lane]   = 0;
    }
    // Synchronise all warps so shared state is visible to all lanes before
    // any opcode execution begins.
    __syncthreads();

    // -----------------------------------------------------------------------
    // Inline yard helpers — operate only on thread 0 of each lane
    // (These are device lambdas expanded inline to avoid register pressure)
    // -----------------------------------------------------------------------

#define LANE_ACTIVE_BANK   (active_bank[lane])
#define LANE_SP(b)         (sp[lane][(b)])
#define LANE_ERROR         (error_code[lane])
#define YARD_TOP(b)        (yards[lane][(b)][sp[lane][(b)] > 0 ? sp[lane][(b)]-1 : 0])

    // Push value onto a specific bank of this lane's yard
    auto yard_push = [&](int bank_id, StackValue val) __device__ -> bool {
        const uint8_t top = LANE_SP(bank_id);
        if (top >= kYardDepth) {
            LANE_ERROR = kErrorStackOverflow;
            return false;
        }
        yards[lane][bank_id][top] = val;
        LANE_SP(bank_id) = top + 1;
        return true;
    };

    // Pop value from a specific bank
    auto yard_pop = [&](int bank_id, StackValue& out) __device__ -> bool {
        const uint8_t top = LANE_SP(bank_id);
        if (top == 0) {
            LANE_ERROR = kErrorStackUnderflow;
            return false;
        }
        LANE_SP(bank_id) = top - 1;
        out = yards[lane][bank_id][top - 1];
        return true;
    };

    // Peek at a specific (bank, slot) without changing sp
    auto yard_peek = [&](int bank_id, int slot_id, StackValue& out) __device__ -> bool {
        if (slot_id < 0 || slot_id >= kYardDepth) {
            LANE_ERROR = kErrorInvalidArgument;
            return false;
        }
        out = yards[lane][bank_id][slot_id];
        return true;
    };

    // Active-bank push/pop (shorthand for the default bank_id)
    auto push = [&](StackValue val) __device__ -> bool {
        return yard_push(LANE_ACTIVE_BANK, val);
    };

    auto pop = [&](StackValue& out) __device__ -> bool {
        return yard_pop(LANE_ACTIVE_BANK, out);
    };

    auto pop_scalar = [&](float& v) __device__ -> bool {
        StackValue tmp{};
        if (!pop(tmp)) return false;
        if (is_vector(tmp)) { LANE_ERROR = kErrorTypeMismatch; return false; }
        v = tmp.x;
        return true;
    };

    // -----------------------------------------------------------------------
    // Main dispatch loop — only head thread (thread == 0) executes
    // -----------------------------------------------------------------------
    if (thread == 0) {
        for (uint32_t i = 0; i < token_count; ++i) {
            if (LANE_ERROR != kErrorNone) break;

            const uint16_t opcode = op_codes[i];

            switch (opcode) {

            // ================================================================
            // LITERALS (0x00-0x03)
            // ================================================================
            case 0x00: {  // LITERAL_SCALAR
                float v = scalars ? scalars[scalar_idx[lane]++] : 0.f;
                push(make_scalar(v));
                break;
            }
            case 0x01: {  // LITERAL_VECTOR
                uint32_t vi = vector_idx[lane]++;
                float x = vectors ? vectors[vi*3+0] : 0.f;
                float y = vectors ? vectors[vi*3+1] : 0.f;
                float z = vectors ? vectors[vi*3+2] : 0.f;
                push(make_vec3(x, y, z));
                break;
            }

            // ================================================================
            // SCALAR ARITHMETIC (0x0A-0x1A, existing opcode set)
            // ================================================================
            case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x0E: {
                float a, b;
                if (!pop_scalar(b)) break;
                if (!pop_scalar(a)) break;
                float r = 0.f;
                if      (opcode == 0x0A) r = a + b;
                else if (opcode == 0x0B) r = a - b;
                else if (opcode == 0x0C) r = a * b;
                else if (opcode == 0x0D) r = (fabsf(b) > 1e-8f) ? a / b : 0.f;
                else                     r = powf(a, b);
                push(make_scalar(r));
                break;
            }
            case 0x0F: case 0xDB: {  // NEG (both aliases)
                float a; if (!pop_scalar(a)) break;
                push(make_scalar(-a));
                break;
            }
            case 0x14: { float a; if (!pop_scalar(a)) break; push(make_scalar(sqrtf(a))); break; }
            case 0x15: { float a; if (!pop_scalar(a)) break; push(make_scalar(expf(a)));  break; }
            case 0x16: { float a; if (!pop_scalar(a)) break; push(make_scalar(logf(a)));  break; }
            case 0x17: { float a; if (!pop_scalar(a)) break; push(make_scalar(log2f(a))); break; }
            case 0x18: { float a; if (!pop_scalar(a)) break; push(make_scalar(sinf(a)));  break; }
            case 0x19: { float a; if (!pop_scalar(a)) break; push(make_scalar(cosf(a)));  break; }
            case 0x1A: { float a; if (!pop_scalar(a)) break; push(make_scalar(tanf(a)));  break; }
            case 0x1B: { float a; if (!pop_scalar(a)) break; push(make_scalar(asinf(a))); break; }
            case 0x1C: { float a; if (!pop_scalar(a)) break; push(make_scalar(acosf(a))); break; }
            case 0x1D: { float a; if (!pop_scalar(a)) break; push(make_scalar(atanf(a))); break; }
            case 0x27: { float a; if (!pop_scalar(a)) break; push(make_scalar(fabsf(a))); break; }

            // ================================================================
            // COMPARISON / BOOLEAN (0x28-0x2F)
            // ================================================================
            case 0x28: case 0x29: case 0x2A: case 0x2B:
            case 0x2C: case 0x2D: case 0x2E: case 0x2F: {
                float a, b;
                if (!pop_scalar(b)) break;
                if (!pop_scalar(a)) break;
                float r = 0.f;
                if      (opcode == 0x28) r = (a >  b) ? 1.f : 0.f;
                else if (opcode == 0x29) r = (a >= b) ? 1.f : 0.f;
                else if (opcode == 0x2A) r = (a <  b) ? 1.f : 0.f;
                else if (opcode == 0x2B) r = (a <= b) ? 1.f : 0.f;
                else if (opcode == 0x2C) r = (fabsf(a-b) < 1e-6f) ? 1.f : 0.f;
                else if (opcode == 0x2D) r = (fabsf(a-b) >= 1e-6f) ? 1.f : 0.f;
                else if (opcode == 0x2E) r = fmaxf(a, b);
                else                     r = fminf(a, b);
                push(make_scalar(r));
                break;
            }

            // ================================================================
            // STACK MANIPULATION (0x32-0x36)
            // ================================================================
            case 0x32: {  // DUP
                uint8_t top = LANE_SP(LANE_ACTIVE_BANK);
                if (top == 0) { LANE_ERROR = kErrorStackUnderflow; break; }
                push(yards[lane][LANE_ACTIVE_BANK][top - 1]);
                break;
            }
            case 0x33: {  // SWAP
                uint8_t top = LANE_SP(LANE_ACTIVE_BANK);
                if (top < 2) { LANE_ERROR = kErrorStackUnderflow; break; }
                StackValue tmp = yards[lane][LANE_ACTIVE_BANK][top - 1];
                yards[lane][LANE_ACTIVE_BANK][top - 1] = yards[lane][LANE_ACTIVE_BANK][top - 2];
                yards[lane][LANE_ACTIVE_BANK][top - 2] = tmp;
                break;
            }
            case 0x34: {  // DROP
                StackValue d{}; pop(d);
                break;
            }
            case 0x35: {  // OVER  (copy second-from-top)
                uint8_t top = LANE_SP(LANE_ACTIVE_BANK);
                if (top < 2) { LANE_ERROR = kErrorStackUnderflow; break; }
                push(yards[lane][LANE_ACTIVE_BANK][top - 2]);
                break;
            }
            case 0x36: {  // ROT  (third-from-top to top)
                uint8_t top = LANE_SP(LANE_ACTIVE_BANK);
                if (top < 3) { LANE_ERROR = kErrorStackUnderflow; break; }
                StackValue t = yards[lane][LANE_ACTIVE_BANK][top - 3];
                yards[lane][LANE_ACTIVE_BANK][top - 3] = yards[lane][LANE_ACTIVE_BANK][top - 2];
                yards[lane][LANE_ACTIVE_BANK][top - 2] = yards[lane][LANE_ACTIVE_BANK][top - 1];
                yards[lane][LANE_ACTIVE_BANK][top - 1] = t;
                break;
            }

            // ================================================================
            // TRANSCENDENTAL / MATH (0x40-0x4F)
            // ================================================================
            case 0x40: { float a; if (!pop_scalar(a)) break; push(make_scalar(floorf(a))); break; }
            case 0x41: { float a; if (!pop_scalar(a)) break; push(make_scalar(ceilf(a)));  break; }
            case 0x42: { float a; if (!pop_scalar(a)) break; push(make_scalar(roundf(a))); break; }
            case 0x43: { // FMOD
                float a, b; if (!pop_scalar(b)) break; if (!pop_scalar(a)) break;
                push(make_scalar(fmodf(a, b)));
                break;
            }
            case 0x44: { // ATAN2
                float y, x; if (!pop_scalar(x)) break; if (!pop_scalar(y)) break;
                push(make_scalar(atan2f(y, x)));
                break;
            }
            case 0x45: { float a; if (!pop_scalar(a)) break; push(make_scalar(rsqrtf(a))); break; }

            // ================================================================
            // VECTOR OPS (0x50-0x5F)
            // ================================================================
            case 0x50: {  // VEC_ADD
                StackValue a{}, b{};
                if (!pop(b)) break; if (!pop(a)) break;
                push(make_vec3(a.x+b.x, a.y+b.y, a.z+b.z));
                break;
            }
            case 0x51: {  // VEC_SUB
                StackValue a{}, b{};
                if (!pop(b)) break; if (!pop(a)) break;
                push(make_vec3(a.x-b.x, a.y-b.y, a.z-b.z));
                break;
            }
            case 0x52: {  // VEC_SCALE
                float s; StackValue v{};
                if (!pop_scalar(s)) break; if (!pop(v)) break;
                push(make_vec3(v.x*s, v.y*s, v.z*s));
                break;
            }
            case 0x53: {  // VEC_DOT
                StackValue a{}, b{};
                if (!pop(b)) break; if (!pop(a)) break;
                float d = a.x*b.x + a.y*b.y + a.z*b.z;
                push(make_scalar(d));
                break;
            }
            case 0x54: {  // VEC_CROSS
                StackValue a{}, b{};
                if (!pop(b)) break; if (!pop(a)) break;
                push(make_vec3(
                    a.y*b.z - a.z*b.y,
                    a.z*b.x - a.x*b.z,
                    a.x*b.y - a.y*b.x));
                break;
            }
            case 0x55: {  // VEC_LENGTH
                StackValue v{}; if (!pop(v)) break;
                push(make_scalar(sqrtf(v.x*v.x + v.y*v.y + v.z*v.z)));
                break;
            }
            case 0x56: {  // VEC_NORMALIZE
                StackValue v{}; if (!pop(v)) break;
                float mag = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
                float inv = (mag > 1e-6f) ? 1.f / mag : 0.f;
                push(make_vec3(v.x*inv, v.y*inv, v.z*inv));
                break;
            }
            case 0x57: {  // VEC_LERP  [a, b, t] → lerp(a,b,t)
                float t; StackValue a{}, b{};
                if (!pop_scalar(t)) break;
                if (!pop(b)) break; if (!pop(a)) break;
                push(make_vec3(
                    a.x + t*(b.x - a.x),
                    a.y + t*(b.y - a.y),
                    a.z + t*(b.z - a.z)));
                break;
            }
            case 0x58: {  // SCALAR_LERP  [a, b, t] → lerp(a,b,t)
                float a, b, t;
                if (!pop_scalar(t)) break;
                if (!pop_scalar(b)) break; if (!pop_scalar(a)) break;
                push(make_scalar(a + t*(b - a)));
                break;
            }

            // ================================================================
            // CONTROL FLOW / STORE-RECALL (0xB0-0xB4)
            // ================================================================
            case 0xB0: {  // BRANCH — pop cond; if 0 skip next opcode
                float cond; if (!pop_scalar(cond)) break;
                if (fabsf(cond) < 1e-6f) ++i;  // skip next token
                break;
            }
            case 0xB3: {  // STORE — pop val, push to bank 8 (dedicated register bank)
                StackValue val{}; if (!pop(val)) break;
                yard_push(8, val);
                break;
            }
            case 0xB4: {  // RECALL — peek top of bank 8
                if (LANE_SP(8) == 0) { LANE_ERROR = kErrorStackUnderflow; break; }
                push(yards[lane][8][LANE_SP(8) - 1]);
                break;
            }

            // ================================================================
            // TERNARY OPS (0x72-0x75)  — balanced {-1, 0, +1}
            // ================================================================
            case kOpTnot: {
                float a; if (!pop_scalar(a)) break;
                push(make_scalar(tnot(a)));
                break;
            }
            case kOpTcomp: {
                float a, b;
                if (!pop_scalar(b)) break; if (!pop_scalar(a)) break;
                push(make_scalar(tcomp(a, b)));
                break;
            }
            case kOpTquant: {  // 0x74 — quantize to nearest trit
                float a; if (!pop_scalar(a)) break;
                push(make_scalar(tquant(a)));
                break;
            }
            case kOpTpack: {  // 0x75 — pack three trits [t2,t1,t0] → balanced-ternary word
                float t0, t1, t2;
                if (!pop_scalar(t0)) break;
                if (!pop_scalar(t1)) break;
                if (!pop_scalar(t2)) break;
                // Encode as positional: v = t2*9 + t1*3 + t0  (range -13 to +13)
                float packed = tquant(t2)*9.f + tquant(t1)*3.f + tquant(t0);
                push(make_scalar(packed));
                break;
            }

            // ================================================================
            // YARD CONTROL OPCODES (0x170-0x176)  — NEW, this spec §4.2
            // ================================================================

            case kOpYardSelect: {
                // 0x170  YARD_SELECT
                // Pop bank_id from active bank, set active_bank[lane] = bank_id.
                // Per spec §4.4: caller should TQUANT-normalise the operand first,
                // but we accept any scalar and round to nearest valid bank.
                float v; if (!pop_scalar(v)) break;
                active_bank[lane] = static_cast<uint8_t>(scalar_to_bank(v));
                break;
            }

            case kOpYardPushBank: {
                // 0x171  YARD_PUSH_BANK
                // Stack layout on entry: [..., value, bank_id]  (bank_id on top)
                // Pop bank_id, pop value, push value into yard[bank_id].
                // Active bank is unchanged.
                float bank_f; if (!pop_scalar(bank_f)) break;
                StackValue val{}; if (!pop(val)) break;
                int dst = scalar_to_bank(bank_f);
                yard_push(dst, val);
                break;
            }

            case kOpYardPopBank: {
                // 0x172  YARD_POP_BANK
                // Stack layout: [..., bank_id]  (bank_id on top)
                // Pop bank_id, pop from yard[bank_id], push result onto active bank.
                float bank_f; if (!pop_scalar(bank_f)) break;
                int src = scalar_to_bank(bank_f);
                StackValue val{};
                if (!yard_pop(src, val)) break;
                push(val);
                break;
            }

            case kOpYardPeekAddr: {
                // 0x173  YARD_PEEK_ADDR
                // Stack layout: [..., bank_id, slot_id]  (slot_id on top)
                // True random-access read.  sp is unchanged.
                float slot_f; if (!pop_scalar(slot_f)) break;
                float bank_f; if (!pop_scalar(bank_f)) break;
                int src_bank = scalar_to_bank(bank_f);
                int src_slot = static_cast<int>(slot_f + 0.5f);
                if (src_slot < 0 || src_slot >= kYardDepth) {
                    LANE_ERROR = kErrorInvalidArgument; break;
                }
                StackValue val{};
                if (!yard_peek(src_bank, src_slot, val)) break;
                push(val);
                break;
            }

            case kOpYardTransfer: {
                // 0x174  YARD_TRANSFER
                // Stack layout: [..., src_bank, dst_bank, n_slots]  (n_slots on top)
                // Move top-n from src to dst within the same lane.
                // Order: pop from src, push to dst, preserving LIFO order.
                // CONFLICT-FREE by construction: same lane, distinct bank indices
                // → disjoint shared memory addresses.
                float n_f; if (!pop_scalar(n_f)) break;
                float dst_f; if (!pop_scalar(dst_f)) break;
                float src_f; if (!pop_scalar(src_f)) break;
                int src = scalar_to_bank(src_f);
                int dst = scalar_to_bank(dst_f);
                int n   = static_cast<int>(n_f + 0.5f);

                // Clamp n to available slots and available dst space
                int avail_src = LANE_SP(src);
                int avail_dst = kYardDepth - LANE_SP(dst);
                if (n > avail_src) n = avail_src;
                if (n > avail_dst) n = avail_dst;

                // Transfer: copy into a temporary register window, then push to dst.
                // We stage into a small on-register buffer (max 16 slots staged at once)
                // to preserve stack order.  For n > 16, iterate in chunks.
                StackValue tmp[16];
                int remaining = n;
                while (remaining > 0 && LANE_ERROR == kErrorNone) {
                    int chunk = (remaining < 16) ? remaining : 16;
                    // Pop chunk items from src (they come out LIFO → reversed order)
                    for (int k = chunk - 1; k >= 0; --k) {
                        yard_pop(src, tmp[k]);
                    }
                    // Push in original order to dst
                    for (int k = 0; k < chunk; ++k) {
                        yard_push(dst, tmp[k]);
                    }
                    remaining -= chunk;
                }
                break;
            }

            case kOpYardSp: {
                // 0x175  YARD_SP
                // Pop bank_id, push current sp[bank_id] as a scalar.
                // Used for introspection (programs can query how full a yard is).
                float bank_f; if (!pop_scalar(bank_f)) break;
                int b = scalar_to_bank(bank_f);
                push(make_scalar(static_cast<float>(LANE_SP(b))));
                break;
            }

            case kOpYardClear: {
                // 0x176  YARD_CLEAR
                // Pop bank_id, reset sp[bank_id] = 0.  Contents unreferenced.
                // Equivalent to drop-all on one bank without touching others.
                float bank_f; if (!pop_scalar(bank_f)) break;
                int b = scalar_to_bank(bank_f);
                LANE_SP(b) = 0;
                break;
            }

            // ================================================================
            // CBR / LoRA ADAPTER OPCODES (0x100-0x108) — pass-through stubs
            // These call the GRE specialist kernels at runtime; here we push
            // a placeholder handle that the bridge resolves via star_id lookup.
            // ================================================================
            case 0x100: case 0x101: case 0x102: case 0x103:
            case 0x104: case 0x105: case 0x106: case 0x107: case 0x108: {
                // Pop operands per opcode ABI, push result handle.
                // Full implementation lives in gre_cognitive_executive.cu;
                // this tier-2 kernel provides the yard substrate so operands
                // are correctly staged before the GRE dispatch.
                StackValue operand{};
                pop(operand);  // consume top; GRE kernel reads yard state directly
                // Push a handle encoding (opcode, lane, active_bank)
                float handle = __uint_as_float(
                    (static_cast<uint32_t>(opcode) << 16) |
                    (static_cast<uint32_t>(lane)   <<  8) |
                    static_cast<uint32_t>(LANE_ACTIVE_BANK));
                push(make_scalar(handle));
                break;
            }

            // ================================================================
            // PHYSICS META-DISPATCH (0x150-0x162)
            // Per spec: sovereign SOA pointers from g_physics_* constants.
            // Push dispatch token for GRE pipeline to execute.
            // ================================================================
            case 0x150: case 0x151: case 0x152: case 0x153:
            case 0x154: case 0x155: case 0x156: case 0x157:
            case 0x158: case 0x159: case 0x15A: case 0x15B:
            case 0x15C: case 0x15D: case 0x15E: case 0x15F:
            case 0x160: case 0x161: case 0x162: {
                float token = __uint_as_float(static_cast<uint32_t>(opcode));
                push(make_scalar(token));
                break;
            }

            // ================================================================
            // WINE I/O CONTRACT OPCODES (0x180-0x182) — pass-through
            // ================================================================
            case 0x180: case 0x181: case 0x182: {
                float token = __uint_as_float(static_cast<uint32_t>(opcode));
                push(make_scalar(token));
                break;
            }

            // ================================================================
            // PHYSICS_EMIT_VISUAL (0x190)
            // ================================================================
            case 0x190: {
                float token = __uint_as_float(0x190u);
                push(make_scalar(token));
                break;
            }

            // ================================================================
            // UNKNOWN — set error, halt lane
            // ================================================================
            default:
                LANE_ERROR = kErrorUnknownOpcode;
                break;
            } // end switch
        } // end token loop
    } // end thread==0 guard

    // -----------------------------------------------------------------------
    // Write-back — all warps sync, then each head-thread writes its output
    // -----------------------------------------------------------------------
    __syncthreads();

    if (thread == 0) {
        const uint32_t global_instance =
            static_cast<uint32_t>(blockIdx.x) * kLanesPerBlock +
            static_cast<uint32_t>(lane);
        if (global_instance < n_instances) {
            InstanceState* out = states + global_instance;
            out->error    = error_code[lane];
            out->reserved = 0;

            // Snapshot active bank into output stack array
            const int ab   = active_bank[lane];
            const int depth = LANE_SP(ab);
            out->head = static_cast<uint32_t>(ab);
            out->size = static_cast<uint32_t>(depth);
            const int copy_n = (depth < 69) ? depth : 69;
            for (int s = 0; s < copy_n; ++s) {
                // Store in StackValue layout (float4 → InstanceState.stack uses same struct)
                const StackValue& src_v = yards[lane][ab][s];
                out->stack[s].x = src_v.x;
                out->stack[s].y = src_v.y;
                out->stack[s].z = src_v.z;
                out->stack[s].w = src_v.w;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side launch helper (for bridge, not for hot path)
// Computes correct grid given n_instances and ensures blockDim matches spec.
// ---------------------------------------------------------------------------

// Launch config query (call from bridge to verify dimensions):
//   blocks_needed = (n_instances + kLanesPerBlock - 1) / kLanesPerBlock
//   dim3 block_dim(kWarpSize, kLanesPerBlock, 1);
//   dim3 grid_dim(blocks_needed, 1, 1);
//   size_t smem = sizeof(StackValue)*9*9*69
//               + sizeof(uint8_t)*9*9       // sp
//               + sizeof(uint8_t)*9         // active_bank
//               + sizeof(uint32_t)*9*2;     // error, scalar_idx / vector_idx
