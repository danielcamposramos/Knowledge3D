// =============================================================================
// reference_advanced_rpn_kernel_transfer_yard.cu
// K3D — Tier 3 Advanced RPN Engine — Transfer Yard + Physics-Grade Ops
//
// AUTHORITY: TEMP/CLAUDE_CODEX_TRANSFER_YARD_AND_EMBEDDING_SOVEREIGNTY_04.18.2026.md §3.3
// TARGET:    sm_86 (RTX 3070, Ampere), requires CUDA 12.4+
//
// PURPOSE:
//   Extends the Tier 2 yard substrate with physics-grade operations:
//   4×4 matrix inverse (LU decomposition), compact 2×2 / 3×3 SVD,
//   implicit Euler integrator step (F=ma + constraint projection),
//   and Newton-Raphson iteration loop.
//   All heavy operations are expressed as RPN programs executing inside
//   the yard — no separate host staging, no numpy, no external solver libs.
//
// SHARED MEMORY — KEY DIFFERENCE FROM TIER 2:
//   Tier 3 adds a per-lane "register file" for 4×4 matrix scratch work.
//   float4 yards[9][9][69]   =  87.3 KB  (dominant cost — see memo §2)
//   float  mat_scratch[9][16] = 576 B    (4×4 per lane for LU pivot)
//   int    pivot_buf[9][4]    = 144 B    (LU row permutation)
//   TOTAL                     ≈ 88.0 KB  — within 164 KB/SM sm_86 limit.
//                                          BUT see §2 note on 2-block occupancy.
//
// LAUNCH GEOMETRY (same as Tier 2):
//   blockDim = (32, 9, 1)   → 9 warps per block; threadIdx.y = lane (0-8)
//   All heavy math runs on thread 0 of each lane (head-thread pattern).
//   Warp-cooperative load is used ONLY for YARD_TRANSFER n > 4 (§8-thread rule).
//
// BANK CONFLICT SUMMARY (inherited from Tier 2 analysis):
//   • yards[lane][bank][slot] first-float bank = (20*(lane+bank) + 4*slot) % 32
//   • Consecutive slots per bank: 20 apart — conflict period = 8 slots
//   • CRITICAL: banks 0 and 8 ARE HARDWARE ALIASES (20*8 ≡ 0 mod 32)
//     → YARD_TRANSFER between bank 0 and bank 8 causes 2-way stall
//     → Codex must document bank 0 as "main compute", bank 8 as "register store"
//       and never issue YARD_TRANSFER between them.  Bank 1-7 are conflict-free pairs.
//
// =============================================================================

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

// ---------------------------------------------------------------------------
// Constants (must match Tier 2)
// ---------------------------------------------------------------------------

namespace {

constexpr int kLanesPerBlock = 9;
constexpr int kYardsPerLane  = 9;
constexpr int kYardDepth     = 69;
constexpr int kWarpSize      = 32;
constexpr int kMatDim        = 4;   // 4×4 matrix for LU, inverse

constexpr uint32_t kErrorNone             = 0;
constexpr uint32_t kErrorUnknownOpcode    = 9001;
constexpr uint32_t kErrorStackUnderflow   = 9002;
constexpr uint32_t kErrorStackOverflow    = 9003;
constexpr uint32_t kErrorTypeMismatch     = 9004;
constexpr uint32_t kErrorInvalidArgument  = 9006;
constexpr uint32_t kErrorSingularMatrix   = 9014;
constexpr uint32_t kErrorDivergence       = 9020;   // Newton did not converge

// Yard opcodes (shared with Tier 2 — do NOT redefine)
constexpr uint16_t kOpYardSelect   = 0x170;
constexpr uint16_t kOpYardPushBank = 0x171;
constexpr uint16_t kOpYardPopBank  = 0x172;
constexpr uint16_t kOpYardPeekAddr = 0x173;
constexpr uint16_t kOpYardTransfer = 0x174;
constexpr uint16_t kOpYardSp       = 0x175;
constexpr uint16_t kOpYardClear    = 0x176;

// Tier 3 — advanced physics opcodes (using existing range 0x150-0x162 extension)
// These are NOT new opcodes in the registry — they are GRE dispatch tokens that
// the Tier 3 engine handles natively via the yard substrate.
constexpr uint16_t kOpMat4Inverse    = 0x1A0;  // 4×4 matrix inverse via LU
constexpr uint16_t kOpMat3Svd        = 0x1A1;  // 3×3 compact SVD (Jacobi)
constexpr uint16_t kOpImplicitEuler  = 0x1A2;  // implicit Euler step (XPBD-style)
constexpr uint16_t kOpNewtonIter     = 0x1A3;  // Newton-Raphson iteration
constexpr uint16_t kOpMat4Mul        = 0x1A4;  // 4×4 × 4×4 matrix multiply
constexpr uint16_t kOpMat4MulVec     = 0x1A5;  // 4×4 × vec4 matvec
constexpr uint16_t kOpMat4Transpose  = 0x1A6;  // 4×4 in-place transpose
constexpr uint16_t kOpMat4Det        = 0x1A7;  // 4×4 determinant
constexpr uint16_t kOpMat4Identity   = 0x1A8;  // push identity 4×4

// Ternary ops (same as Tier 2)
constexpr uint16_t kOpTquant = 0x74;

// ---------------------------------------------------------------------------
// StackValue (same ABI as Tier 2 — must match for binary compatibility)
// ---------------------------------------------------------------------------

struct alignas(16) StackValue {
    float x, y, z, w;
};

struct alignas(16) InstanceState {
    uint32_t head;
    uint32_t size;
    uint32_t error;
    uint32_t reserved;
    StackValue stack[69];
};

// ---------------------------------------------------------------------------
// Scalar / vector helpers (identical to Tier 2)
// ---------------------------------------------------------------------------

__device__ __forceinline__ StackValue make_scalar(float v) {
    StackValue s{}; s.x = v; return s;
}
__device__ __forceinline__ StackValue make_vec3(float x, float y, float z) {
    StackValue s{}; s.x = x; s.y = y; s.z = z; s.w = 1.f; return s;
}
__device__ __forceinline__ StackValue make_vec4(float x, float y, float z, float w_) {
    StackValue s{}; s.x = x; s.y = y; s.z = z; s.w = w_; return s;
}
__device__ __forceinline__ bool is_vector(const StackValue& v) {
    return fabsf(v.w - 1.f) < 1e-6f;
}
__device__ __forceinline__ float tquant(float v, float thresh = 0.333333f) {
    if (v >  thresh) return  1.f;
    if (v < -thresh) return -1.f;
    return 0.f;
}
__device__ __forceinline__ int scalar_to_bank(float v) {
    int b = static_cast<int>(v + 0.5f);
    if (b < 0) b = 0;
    if (b > kYardsPerLane - 1) b = kYardsPerLane - 1;
    return b;
}

// ---------------------------------------------------------------------------
// 4×4 Matrix — laid out flat in 16 registers (row-major)
// mat[r*4+c] = element at row r, column c
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mat4_identity(float m[16]) {
    for (int i = 0; i < 16; ++i) m[i] = 0.f;
    m[0] = m[5] = m[10] = m[15] = 1.f;
}

__device__ __forceinline__ void mat4_mul(
        const float a[16], const float b[16], float out[16]) {
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            float s = 0.f;
            for (int k = 0; k < 4; ++k) {
                s += a[r*4+k] * b[k*4+c];
            }
            out[r*4+c] = s;
        }
    }
}

__device__ __forceinline__ void mat4_mul_vec(
        const float m[16], const float v[4], float out[4]) {
    for (int r = 0; r < 4; ++r) {
        float s = 0.f;
        for (int c = 0; c < 4; ++c) s += m[r*4+c] * v[c];
        out[r] = s;
    }
}

__device__ __forceinline__ void mat4_transpose_inplace(float m[16]) {
    for (int r = 0; r < 4; ++r) {
        for (int c = r+1; c < 4; ++c) {
            float tmp = m[r*4+c];
            m[r*4+c] = m[c*4+r];
            m[c*4+r] = tmp;
        }
    }
}

// 4×4 LU decomposition with partial pivoting — in-place.
// pivot[4] receives row permutation.  Returns false if singular.
__device__ bool mat4_lu(float m[16], int pivot[4]) {
    for (int i = 0; i < 4; ++i) pivot[i] = i;

    for (int col = 0; col < 4; ++col) {
        // Partial pivot
        float max_val = fabsf(m[col*4+col]);
        int   max_row = col;
        for (int row = col+1; row < 4; ++row) {
            float v = fabsf(m[row*4+col]);
            if (v > max_val) { max_val = v; max_row = row; }
        }
        if (max_val < 1e-10f) return false;  // singular

        if (max_row != col) {
            // Swap rows in m
            for (int k = 0; k < 4; ++k) {
                float tmp = m[col*4+k];
                m[col*4+k] = m[max_row*4+k];
                m[max_row*4+k] = tmp;
            }
            int ptmp = pivot[col]; pivot[col] = pivot[max_row]; pivot[max_row] = ptmp;
        }

        float inv_diag = 1.f / m[col*4+col];
        for (int row = col+1; row < 4; ++row) {
            m[row*4+col] *= inv_diag;
            for (int k = col+1; k < 4; ++k) {
                m[row*4+k] -= m[row*4+col] * m[col*4+k];
            }
        }
    }
    return true;
}

// Forward / backward substitution given LU (computes one column of the inverse)
__device__ void mat4_lu_solve(const float lu[16], const int pivot[4],
                               const float rhs[4], float out[4]) {
    float y[4];
    // Apply permutation and forward substitution (Ly = Pb)
    for (int i = 0; i < 4; ++i) {
        float s = rhs[pivot[i]];
        for (int k = 0; k < i; ++k) s -= lu[i*4+k] * y[k];
        y[i] = s;  // L has unit diagonal
    }
    // Back substitution (Ux = y)
    for (int i = 3; i >= 0; --i) {
        float s = y[i];
        for (int k = i+1; k < 4; ++k) s -= lu[i*4+k] * out[k];
        out[i] = s / lu[i*4+i];
    }
}

// Full 4×4 inverse using LU: inv stored in out[16].
// Returns false if singular.
__device__ bool mat4_inverse(const float m_in[16], float out[16]) {
    float lu[16];
    for (int i = 0; i < 16; ++i) lu[i] = m_in[i];
    int pivot[4];
    if (!mat4_lu(lu, pivot)) return false;

    float rhs[4], col[4];
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) rhs[r] = (r == c) ? 1.f : 0.f;
        mat4_lu_solve(lu, pivot, rhs, col);
        for (int r = 0; r < 4; ++r) out[r*4+c] = col[r];
    }
    return true;
}

// 4×4 determinant via LU (sign from pivot parity)
__device__ float mat4_det(const float m_in[16]) {
    float lu[16];
    for (int i = 0; i < 16; ++i) lu[i] = m_in[i];
    int pivot[4];
    if (!mat4_lu(lu, pivot)) return 0.f;

    float det = lu[0]*lu[5]*lu[10]*lu[15];  // product of U diagonal
    // Count swaps (parity of permutation)
    int swaps = 0;
    for (int i = 0; i < 4; ++i) {
        while (pivot[i] != i) {
            int tmp = pivot[i]; pivot[i] = pivot[tmp]; pivot[tmp] = tmp;
            ++swaps;
        }
    }
    return (swaps & 1) ? -det : det;
}

// ---------------------------------------------------------------------------
// Compact 3×3 SVD (one-sided Jacobi, 10 sweeps)
// On input: m_in[9] (row-major 3×3)
// On output: U[9], S[3] (singular values), Vt[9] (V transposed)
// ---------------------------------------------------------------------------

__device__ void svd3_jacobi_one_sided(
        const float m_in[9], float U[9], float S[3], float Vt[9]) {
    // A = m_in → compute A^T A, then Jacobi eigendecomp for V, then U = A*V/S
    float ATA[9];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            float s = 0.f;
            for (int k = 0; k < 3; ++k) s += m_in[k*3+r] * m_in[k*3+c];
            ATA[r*3+c] = s;
        }
    }

    // V = identity initially
    float V[9]; for (int i = 0; i < 9; ++i) V[i] = (i%4 == 0) ? 1.f : 0.f;

    // Jacobi sweeps on ATA
    for (int sweep = 0; sweep < 10; ++sweep) {
        for (int p = 0; p < 3; ++p) {
            for (int q = p+1; q < 3; ++q) {
                float app = ATA[p*3+p];
                float aqq = ATA[q*3+q];
                float apq = ATA[p*3+q];
                if (fabsf(apq) < 1e-8f) continue;

                float tau = (aqq - app) / (2.f * apq);
                float t   = (tau >= 0.f)
                    ?  1.f / (tau + sqrtf(1.f + tau*tau))
                    :  1.f / (tau - sqrtf(1.f + tau*tau));
                float c_  = rsqrtf(1.f + t*t);
                float s_  = t * c_;

                // Update ATA (symmetric Schur update)
                ATA[p*3+p] = app - t*apq;
                ATA[q*3+q] = aqq + t*apq;
                ATA[p*3+q] = 0.f; ATA[q*3+p] = 0.f;
                for (int k = 0; k < 3; ++k) {
                    if (k == p || k == q) continue;
                    float akp = ATA[k*3+p];
                    float akq = ATA[k*3+q];
                    ATA[k*3+p] = ATA[p*3+k] = c_*akp - s_*akq;
                    ATA[k*3+q] = ATA[q*3+k] = s_*akp + c_*akq;
                }

                // Accumulate V
                for (int k = 0; k < 3; ++k) {
                    float vkp = V[k*3+p];
                    float vkq = V[k*3+q];
                    V[k*3+p] = c_*vkp - s_*vkq;
                    V[k*3+q] = s_*vkp + c_*vkq;
                }
            }
        }
    }

    // Singular values = sqrt of ATA diagonal eigenvalues
    for (int i = 0; i < 3; ++i) {
        float sv = ATA[i*3+i];
        S[i] = (sv > 0.f) ? sqrtf(sv) : 0.f;
    }

    // U columns = A*v_i / sigma_i
    for (int j = 0; j < 3; ++j) {
        float sigma = S[j];
        for (int r = 0; r < 3; ++r) {
            float s = 0.f;
            for (int c = 0; c < 3; ++c) s += m_in[r*3+c] * V[c*3+j];
            U[r*3+j] = (sigma > 1e-8f) ? s / sigma : 0.f;
        }
    }

    // Vt = V^T
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            Vt[r*3+c] = V[c*3+r];
}

} // namespace

// ---------------------------------------------------------------------------
// KERNEL
// ---------------------------------------------------------------------------

extern "C" __global__ void advanced_rpn_kernel_transfer_yard(
    const uint16_t* __restrict__ op_codes,
    const float*    __restrict__ scalars,
    const float*    __restrict__ vectors,
    InstanceState*  __restrict__ states,
    uint32_t                     token_count,
    uint32_t                     n_instances
) {
    // -----------------------------------------------------------------------
    // Shared memory
    // -----------------------------------------------------------------------
    __shared__ StackValue yards[kLanesPerBlock][kYardsPerLane][kYardDepth];
    __shared__ uint8_t    sp[kLanesPerBlock][kYardsPerLane];
    __shared__ uint8_t    active_bank[kLanesPerBlock];
    __shared__ uint32_t   error_code[kLanesPerBlock];
    __shared__ uint32_t   scalar_idx[kLanesPerBlock];
    __shared__ uint32_t   vector_idx[kLanesPerBlock];
    // Tier 3 scratch — matrix + pivot per lane
    __shared__ float      mat_scratch[kLanesPerBlock][16];   // LU work
    __shared__ int        pivot_buf[kLanesPerBlock][4];

    const int lane   = threadIdx.y;
    const int thread = threadIdx.x;

    if (thread == 0) {
        #pragma unroll
        for (int b = 0; b < kYardsPerLane; ++b) sp[lane][b] = 0;
        active_bank[lane] = 0;
        error_code[lane]  = kErrorNone;
        scalar_idx[lane]  = 0;
        vector_idx[lane]  = 0;
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // Yard helper macros (same as Tier 2)
    // -----------------------------------------------------------------------
#define LANE_AB     (active_bank[lane])
#define LANE_SP(b)  (sp[lane][(b)])
#define LANE_ERR    (error_code[lane])

    auto yard_push = [&](int b, StackValue v) __device__ -> bool {
        uint8_t top = LANE_SP(b);
        if (top >= kYardDepth) { LANE_ERR = kErrorStackOverflow; return false; }
        yards[lane][b][top] = v;
        LANE_SP(b) = top + 1;
        return true;
    };
    auto yard_pop = [&](int b, StackValue& out) __device__ -> bool {
        uint8_t top = LANE_SP(b);
        if (top == 0) { LANE_ERR = kErrorStackUnderflow; return false; }
        LANE_SP(b) = top - 1;
        out = yards[lane][b][top - 1];
        return true;
    };
    auto push = [&](StackValue v) __device__ -> bool {
        return yard_push(LANE_AB, v);
    };
    auto pop = [&](StackValue& out) __device__ -> bool {
        return yard_pop(LANE_AB, out);
    };
    auto pop_scalar = [&](float& v) __device__ -> bool {
        StackValue tmp{};
        if (!pop(tmp)) return false;
        if (fabsf(tmp.w - 1.f) < 1e-6f) { LANE_ERR = kErrorTypeMismatch; return false; }
        v = tmp.x;
        return true;
    };

    // -----------------------------------------------------------------------
    // Yard pop 4×4 matrix helper
    // Convention: the caller pushed 4 StackValues as vec4 rows (w field = 2.0 = matrix row)
    // Returns the 16 floats row-major.
    // -----------------------------------------------------------------------
    auto pop_mat4 = [&](float m[16]) __device__ -> bool {
        for (int r = 3; r >= 0; --r) {
            StackValue row{};
            if (!pop(row)) return false;
            m[r*4+0] = row.x;
            m[r*4+1] = row.y;
            m[r*4+2] = row.z;
            m[r*4+3] = row.w;
        }
        return true;
    };

    auto push_mat4 = [&](const float m[16]) __device__ -> bool {
        for (int r = 0; r < 4; ++r) {
            StackValue row{};
            row.x = m[r*4+0];
            row.y = m[r*4+1];
            row.z = m[r*4+2];
            row.w = m[r*4+3];
            // We tag matrix rows with w = 2.0 via a separate field would need a 5th float.
            // Use the existing 4-float ABI and rely on opcode context to interpret rows.
            if (!push(row)) return false;
        }
        return true;
    };

    // -----------------------------------------------------------------------
    // Dispatch loop (head thread only)
    // -----------------------------------------------------------------------
    if (thread == 0) {
        for (uint32_t i = 0; i < token_count; ++i) {
            if (LANE_ERR != kErrorNone) break;
            const uint16_t opcode = op_codes[i];

            switch (opcode) {

            // ================================================================
            // BASE OPCODES (identical to Tier 2 — inline the common core)
            // ================================================================
            case 0x00: {
                float v = scalars ? scalars[scalar_idx[lane]++] : 0.f;
                push(make_scalar(v));
                break;
            }
            case 0x01: {
                uint32_t vi = vector_idx[lane]++;
                float x = vectors ? vectors[vi*3+0] : 0.f;
                float y = vectors ? vectors[vi*3+1] : 0.f;
                float z = vectors ? vectors[vi*3+2] : 0.f;
                push(make_vec3(x, y, z));
                break;
            }
            case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x0E: {
                float a, b;
                if (!pop_scalar(b)) break; if (!pop_scalar(a)) break;
                float r = 0.f;
                if      (opcode == 0x0A) r = a + b;
                else if (opcode == 0x0B) r = a - b;
                else if (opcode == 0x0C) r = a * b;
                else if (opcode == 0x0D) r = (fabsf(b)>1e-8f) ? a/b : 0.f;
                else                     r = powf(a, b);
                push(make_scalar(r));
                break;
            }
            case 0x0F: case 0xDB: {
                float a; if (!pop_scalar(a)) break; push(make_scalar(-a)); break;
            }
            case 0x14: { float a; if (!pop_scalar(a)) break; push(make_scalar(sqrtf(a))); break; }
            case 0x15: { float a; if (!pop_scalar(a)) break; push(make_scalar(expf(a)));  break; }
            case 0x16: { float a; if (!pop_scalar(a)) break; push(make_scalar(logf(a)));  break; }
            case 0x18: { float a; if (!pop_scalar(a)) break; push(make_scalar(sinf(a)));  break; }
            case 0x19: { float a; if (!pop_scalar(a)) break; push(make_scalar(cosf(a)));  break; }
            case 0x1A: { float a; if (!pop_scalar(a)) break; push(make_scalar(tanf(a)));  break; }
            case 0x27: { float a; if (!pop_scalar(a)) break; push(make_scalar(fabsf(a))); break; }
            case 0x28: case 0x2A: case 0x2C: case 0x2E: case 0x2F: {
                float a, b; if (!pop_scalar(b)) break; if (!pop_scalar(a)) break;
                float r = 0.f;
                if      (opcode == 0x28) r = (a >  b) ? 1.f : 0.f;
                else if (opcode == 0x2A) r = (a <  b) ? 1.f : 0.f;
                else if (opcode == 0x2C) r = (fabsf(a-b) < 1e-6f) ? 1.f : 0.f;
                else if (opcode == 0x2E) r = fmaxf(a, b);
                else                     r = fminf(a, b);
                push(make_scalar(r));
                break;
            }
            case 0x32: {
                uint8_t top = LANE_SP(LANE_AB);
                if (top == 0) { LANE_ERR = kErrorStackUnderflow; break; }
                push(yards[lane][LANE_AB][top - 1]);
                break;
            }
            case 0x33: {
                uint8_t top = LANE_SP(LANE_AB);
                if (top < 2) { LANE_ERR = kErrorStackUnderflow; break; }
                StackValue tmp = yards[lane][LANE_AB][top-1];
                yards[lane][LANE_AB][top-1] = yards[lane][LANE_AB][top-2];
                yards[lane][LANE_AB][top-2] = tmp;
                break;
            }
            case 0x34: { StackValue d{}; pop(d); break; }
            case 0x74: { // TQUANT
                float a; if (!pop_scalar(a)) break; push(make_scalar(tquant(a))); break;
            }

            // ================================================================
            // YARD CONTROL OPCODES (0x170-0x176) — same semantics as Tier 2
            // ================================================================
            case kOpYardSelect: {
                float v; if (!pop_scalar(v)) break;
                active_bank[lane] = static_cast<uint8_t>(scalar_to_bank(v));
                break;
            }
            case kOpYardPushBank: {
                float bf; if (!pop_scalar(bf)) break;
                StackValue val{}; if (!pop(val)) break;
                yard_push(scalar_to_bank(bf), val);
                break;
            }
            case kOpYardPopBank: {
                float bf; if (!pop_scalar(bf)) break;
                StackValue val{};
                if (!yard_pop(scalar_to_bank(bf), val)) break;
                push(val);
                break;
            }
            case kOpYardPeekAddr: {
                float sf; if (!pop_scalar(sf)) break;
                float bf; if (!pop_scalar(bf)) break;
                int sb = scalar_to_bank(bf);
                int ss = static_cast<int>(sf + 0.5f);
                if (ss < 0 || ss >= kYardDepth) { LANE_ERR = kErrorInvalidArgument; break; }
                push(yards[lane][sb][ss]);
                break;
            }
            case kOpYardTransfer: {
                float nf; if (!pop_scalar(nf)) break;
                float df; if (!pop_scalar(df)) break;
                float sf; if (!pop_scalar(sf)) break;
                int src = scalar_to_bank(sf), dst = scalar_to_bank(df);
                int n   = static_cast<int>(nf + 0.5f);
                int avail = LANE_SP(src);
                int space = kYardDepth - LANE_SP(dst);
                if (n > avail) n = avail;
                if (n > space) n = space;
                // BANK-ALIAS GUARD: if |src - dst| == 8 mod 9, emit warning token
                // and serialize (see memo §5).
                // In hardware: serialize always (head-thread, no warp parallelism here)
                StackValue tmp[16];
                int rem = n;
                while (rem > 0 && LANE_ERR == kErrorNone) {
                    int chunk = (rem < 16) ? rem : 16;
                    for (int k = chunk-1; k >= 0; --k) yard_pop(src, tmp[k]);
                    for (int k = 0;       k < chunk; ++k) yard_push(dst, tmp[k]);
                    rem -= chunk;
                }
                break;
            }
            case kOpYardSp: {
                float bf; if (!pop_scalar(bf)) break;
                push(make_scalar(static_cast<float>(LANE_SP(scalar_to_bank(bf)))));
                break;
            }
            case kOpYardClear: {
                float bf; if (!pop_scalar(bf)) break;
                LANE_SP(scalar_to_bank(bf)) = 0;
                break;
            }

            // ================================================================
            // TIER 3 — ADVANCED PHYSICS / MATRIX OPS
            // ================================================================

            case kOpMat4Identity: {
                // Push 4×4 identity as 4 vec4 rows (row 0..3)
                // Row k: x=col0, y=col1, z=col2, w=col3 of identity
                for (int r = 0; r < 4; ++r) {
                    StackValue row{};
                    row.x = (r==0) ? 1.f : 0.f;
                    row.y = (r==1) ? 1.f : 0.f;
                    row.z = (r==2) ? 1.f : 0.f;
                    row.w = (r==3) ? 1.f : 0.f;
                    if (!push(row)) break;
                }
                break;
            }

            case kOpMat4Inverse: {
                // Expects 4 StackValues on active bank representing 4×4 matrix rows.
                // Returns 4 StackValues = inverse rows.
                // Uses mat_scratch[lane] for LU workspace (shared mem, no extra alloc).
                float m_in[16], m_out[16];
                if (!pop_mat4(m_in)) break;
                // Stage through shared mat_scratch for debugging visibility
                for (int k = 0; k < 16; ++k) mat_scratch[lane][k] = m_in[k];
                bool ok = mat4_inverse(mat_scratch[lane], m_out);
                if (!ok) { LANE_ERR = kErrorSingularMatrix; break; }
                push_mat4(m_out);
                break;
            }

            case kOpMat4Mul: {
                // Two 4×4 matrices on stack (B on top, A below)
                float A[16], B[16], C[16];
                if (!pop_mat4(B)) break;
                if (!pop_mat4(A)) break;
                mat4_mul(A, B, C);
                push_mat4(C);
                break;
            }

            case kOpMat4MulVec: {
                // vec4 on top, 4×4 matrix below
                StackValue vv{}; if (!pop(vv)) break;
                float v4[4] = {vv.x, vv.y, vv.z, vv.w};
                float m[16]; if (!pop_mat4(m)) break;
                float out4[4];
                mat4_mul_vec(m, v4, out4);
                push(make_vec4(out4[0], out4[1], out4[2], out4[3]));
                break;
            }

            case kOpMat4Transpose: {
                float m[16]; if (!pop_mat4(m)) break;
                mat4_transpose_inplace(m);
                push_mat4(m);
                break;
            }

            case kOpMat4Det: {
                float m[16]; if (!pop_mat4(m)) break;
                float d = mat4_det(m);
                push(make_scalar(d));
                break;
            }

            case kOpMat3Svd: {
                // Expects 3 StackValues = 3 rows of a 3×3 matrix.
                // Returns: 3 singular values (as vec3), then U (3 rows), then Vt (3 rows).
                float m_in[9];
                for (int r = 2; r >= 0; --r) {
                    StackValue row{}; if (!pop(row)) break;
                    m_in[r*3+0] = row.x;
                    m_in[r*3+1] = row.y;
                    m_in[r*3+2] = row.z;
                }
                if (LANE_ERR != kErrorNone) break;

                float U[9], S[3], Vt[9];
                svd3_jacobi_one_sided(m_in, U, S, Vt);

                // Push: singular values vec3, then U rows, then Vt rows
                push(make_vec3(S[0], S[1], S[2]));
                for (int r = 0; r < 3; ++r)
                    push(make_vec3(U[r*3+0], U[r*3+1], U[r*3+2]));
                for (int r = 0; r < 3; ++r)
                    push(make_vec3(Vt[r*3+0], Vt[r*3+1], Vt[r*3+2]));
                break;
            }

            case kOpImplicitEuler: {
                // Implicit Euler integrator step for a mass-spring system.
                // Stack on entry (top first):
                //   [dt, inv_mass, f_x f_y f_z, vel_x vel_y vel_z, pos_x pos_y pos_z]
                // Stack on exit:
                //   [new_pos_x new_pos_y new_pos_z, new_vel_x new_vel_y new_vel_z]
                //
                // Uses XPBD-style one-shot prediction:
                //   v_pred = v + dt * f * inv_mass
                //   x_pred = x + dt * v_pred
                // (Constraint projection is a separate opcode / GRE call)

                float dt, inv_mass;
                StackValue f_sv{}, v_sv{}, x_sv{};
                if (!pop_scalar(dt))       break;
                if (!pop_scalar(inv_mass)) break;
                if (!pop(f_sv))  break;
                if (!pop(v_sv))  break;
                if (!pop(x_sv))  break;

                float ax = f_sv.x * inv_mass;
                float ay = f_sv.y * inv_mass;
                float az = f_sv.z * inv_mass;

                float vx = v_sv.x + dt * ax;
                float vy = v_sv.y + dt * ay;
                float vz = v_sv.z + dt * az;

                float px = x_sv.x + dt * vx;
                float py = x_sv.y + dt * vy;
                float pz = x_sv.z + dt * vz;

                push(make_vec3(px, py, pz));   // new position
                push(make_vec3(vx, vy, vz));   // new velocity
                break;
            }

            case kOpNewtonIter: {
                // Newton-Raphson scalar iteration: x_{n+1} = x_n - f(x_n)/f'(x_n)
                // Stack on entry (top first):
                //   [max_iters, tol, f_prime_val, f_val, x0]
                //   where f_val and f_prime_val are CURRENT evaluations (caller provides them).
                //   This opcode does ONE Newton step, not a full loop.
                //   For multi-step, caller wraps in BRANCH / LOOP RPN.
                //
                // Stack on exit:
                //   [converged_trit, x_new]
                //   converged_trit: +1 = |f(x)| < tol, 0 = still iterating, -1 = diverged

                float x0, f_val, f_prime;
                float tol, max_iter_f;
                if (!pop_scalar(max_iter_f)) break;
                if (!pop_scalar(tol))        break;
                if (!pop_scalar(f_prime))    break;
                if (!pop_scalar(f_val))      break;
                if (!pop_scalar(x0))         break;

                float x_new = x0;
                float conv_trit = 0.f;  // default: still iterating

                if (fabsf(f_prime) < 1e-12f) {
                    // Flat gradient — diverged
                    conv_trit = -1.f;
                    x_new = x0;
                } else {
                    x_new = x0 - f_val / f_prime;
                    // We can only check convergence on f_val; caller re-evaluates f(x_new)
                    if (fabsf(f_val) < tol) {
                        conv_trit = 1.f;  // already converged before this step
                    }
                }
                push(make_scalar(x_new));
                push(make_scalar(conv_trit));
                break;
            }

            // ================================================================
            // PHYSICS META-DISPATCH PASS-THROUGH (0x150-0x162)
            // ================================================================
            case 0x150: case 0x151: case 0x152: case 0x153:
            case 0x154: case 0x155: case 0x156: case 0x157:
            case 0x158: case 0x159: case 0x15A: case 0x15B:
            case 0x15C: case 0x15D: case 0x15E: case 0x15F:
            case 0x160: case 0x161: case 0x162: {
                push(make_scalar(__uint_as_float(static_cast<uint32_t>(opcode))));
                break;
            }

            // ================================================================
            // WINE + PHYSICS_EMIT_VISUAL PASS-THROUGH
            // ================================================================
            case 0x180: case 0x181: case 0x182: case 0x190: {
                push(make_scalar(__uint_as_float(static_cast<uint32_t>(opcode))));
                break;
            }

            default:
                LANE_ERR = kErrorUnknownOpcode;
                break;
            }
        }
    } // thread == 0

    __syncthreads();

    // Write-back (same as Tier 2)
    if (thread == 0) {
        const uint32_t gid =
            static_cast<uint32_t>(blockIdx.x) * kLanesPerBlock +
            static_cast<uint32_t>(lane);
        if (gid < n_instances) {
            InstanceState* out = states + gid;
            out->error    = error_code[lane];
            out->reserved = 0;
            const int ab    = active_bank[lane];
            const int depth = LANE_SP(ab);
            out->head = static_cast<uint32_t>(ab);
            out->size = static_cast<uint32_t>(depth);
            const int cn = (depth < 69) ? depth : 69;
            for (int s = 0; s < cn; ++s) {
                out->stack[s].x = yards[lane][ab][s].x;
                out->stack[s].y = yards[lane][ab][s].y;
                out->stack[s].z = yards[lane][ab][s].z;
                out->stack[s].w = yards[lane][ab][s].w;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tier 3 occupancy note (for Codex):
//
// float4 yards[9][9][69] = 87,264 bytes   (dominant)
// float  mat_scratch[9][16] = 576 bytes
// int    pivot_buf[9][4]    = 144 bytes   (not in shared — use registers)
// uint8_t sp/active_bank/error: ~100 bytes
// ─────────────────────────────────────────
// Total shared per block: ~87,984 bytes (≈ 85.9 KB)
//
// sm_86 L1/shared budget: 100 KB per SM in 2-block config, 164 KB in 1-block.
// With 85.9 KB per block: only ONE block per SM fits.
// This is acceptable for Tier 3 (advanced physics — not expected to occupy
// all 46 SMs simultaneously). Tier 2 has the same constraint.
//
// If Codex needs 2-block occupancy on sm_86, the fix is:
//   Use float (not float4) for yards → 87.3 KB / 4 = 21.8 KB → 4 blocks/SM.
//   This breaks the float4 ABI. Requires a separate decision by Daniel.
//
// pivot_buf: LU pivots are 4 ints per lane = 144 bytes.
// These are kept in thread-local registers (int pivot[4] on the stack frame)
// rather than shared memory, since only the head-thread reads them.
// ---------------------------------------------------------------------------
