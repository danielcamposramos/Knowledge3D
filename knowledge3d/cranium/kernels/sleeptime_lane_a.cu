// sleeptime_lane_a.cu — Sleeptime Lane A: temporary-star promote / merge / discard
//
// Opcode: SLEEPTIME_LANE_A_EVAL  0x300  (RPN_DOMAIN_OPCODE_REGISTRY §11, reserved 2026-04-21)
//
// This kernel classifies each candidate star from the temporary-ingest region
// against the existing House Galaxy embedding table.  One thread per candidate
// star (grid-stride).  Reuses the semantic-gravity formula and the defeasibility
// ternary quantisation already proven in gre_defeasible_resolver.cu.
//
// ─── Algorithm (one thread per candidate) ─────────────────────────────────────
//
//  Step 1 — Gravity probe
//    For each candidate embedding (dim=16 float32, "tier_64" prefix of Matryoshka),
//    scan house_embeddings[H × DIM] and compute cosine-similarity-based ternary
//    gravity:
//
//      T(s_cand, s_house_i) = +1 if sim >= ATTRACT_THRESH
//                              0  if sim in (REPEL_THRESH, ATTRACT_THRESH)
//                             -1  if sim <= REPEL_THRESH
//
//      F_i = T(s_cand, s_house_i)     (scaled mass / d² approximated by similarity)
//
//    Accumulate:
//      attract_count, repel_count, best_sim, best_house_idx
//
//  Step 2 — Defeasibility trit
//    The Grammar-Galaxy rule buffer carries per-rule strengths (+1 strict,
//    0 defeasible, -1 defeater).  We call gre_defeasible_resolver::quantize_trit()
//    to classify rule strengths — the input is the candidate's domain_hash matched
//    against rule domain_hashes.
//
//    If a strict defeater fires against the candidate's domain → trit = -1.
//    If no rules fire → trit mirrors the gravity verdict.
//
//  Step 3 — Decision
//    Combining gravity and defeasibility:
//      gravity_trit = +1  if attract_count > repel_count
//                      0  if equal (or no neighbors)
//                     -1  if repel_count > attract_count
//
//      final_trit = clamp(gravity_trit + defeasibility_trit)
//                   where strict_defeat overrides to -1 unconditionally.
//
//  Outputs (per candidate):
//    out_trits[c]           int8_t  — {-1 discard, 0 pending, +1 promote}
//    out_target_galaxy[c]   int32_t — house_galaxy_id for the best match
//                                     (meaningful only if out_trits[c] == +1)
//    out_best_house_idx[c]  int32_t — index into house_embeddings of best match
//                                     (-1 if trit != +1)
//
// ─── Launch parameters ────────────────────────────────────────────────────────
//   block_size = 128 (compile-time; tunable)
//   grid_size  = ceil(N_candidates / block_size)
//   Grid-stride loop handles N > block_size * max_grid naturally.
//   Shared memory: none required (each thread works independently).
//
// ─── Sovereignty ──────────────────────────────────────────────────────────────
//   Pure PTX kernel — no external libraries, no CPU fallback.
//   Python reads out_trits[] and out_target_galaxy[] after synchronisation.
//   All decisions are made here; Python is I/O only (JSONL append / tombstone).

#include <math.h>
#include <stdint.h>
#include "gre_defeasible_resolver.cu"

#define LANE_A_DIM              16
#define LANE_A_ATTRACT_THRESH   0.72f
#define LANE_A_REPEL_THRESH     0.20f
#define LANE_A_MAX_RULES        64

__device__ __forceinline__ float _cosine_sim_16(
    const float* __restrict__ a,
    const float* __restrict__ b
) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int d = 0; d < LANE_A_DIM; ++d) {
        dot += a[d] * b[d];
        na  += a[d] * a[d];
        nb  += b[d] * b[d];
    }
    float denom = sqrtf(na * nb);
    return (denom > 1e-8f) ? (dot / denom) : 0.0f;
}

extern "C" __global__ void sleeptime_lane_a_eval(
    // --- candidate inputs (N candidates) ---
    const float*    __restrict__ cand_embeddings,   // [N x DIM] float32
    const uint32_t* __restrict__ cand_domain_hash,  // [N] domain identifier
    int N,                                          // number of candidates

    // --- house galaxy entries (H existing stars) ---
    const float*    __restrict__ house_embeddings,  // [H x DIM] float32
    const uint32_t* __restrict__ house_domain_hash, // [H]
    const int32_t*  __restrict__ house_galaxy_ids,  // [H] galaxy id per star
    int H,                                          // number of house stars

    // --- grammar-galaxy rule table (R rules) ---
    const uint32_t* __restrict__ rule_domain_hash,  // [R]
    const int8_t*   __restrict__ rule_strength,      // [R] +1 strict, 0 def, -1 defeater
    int R,                                           // number of rules

    // --- per-candidate outputs ---
    int8_t*  __restrict__ out_trits,           // [N] {-1, 0, +1}
    int32_t* __restrict__ out_target_galaxy,   // [N] target galaxy id
    int32_t* __restrict__ out_best_house_idx   // [N] best house star index
) {
    // grid-stride loop — one logical thread per candidate
    for (int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
         c < N;
         c += (int)(gridDim.x * blockDim.x))
    {
        const float* cand = cand_embeddings + (c * LANE_A_DIM);
        const uint32_t c_domain = cand_domain_hash[c];

        // ── Step 1: gravity probe ──────────────────────────────────────────
        int attract = 0, repel = 0;
        float best_sim = -2.0f;
        int   best_idx = -1;
        int32_t best_gal = -1;

        for (int h = 0; h < H; ++h) {
            float sim = _cosine_sim_16(cand, house_embeddings + (h * LANE_A_DIM));
            if (sim >= LANE_A_ATTRACT_THRESH) {
                attract++;
                if (sim > best_sim) {
                    best_sim = sim;
                    best_idx = h;
                    best_gal = house_galaxy_ids[h];
                }
            } else if (sim <= LANE_A_REPEL_THRESH) {
                repel++;
            }
        }

        // If no house entries exist at all, candidate is orphan → pending (0)
        const int8_t gravity_trit = (H == 0) ? (int8_t)0
                                              : clamp_trit_int(attract - repel);

        // ── Step 2: defeasibility check ────────────────────────────────────
        int def_sum = 0;
        int8_t strict_defeat = 0;

        for (int r = 0; r < R && r < LANE_A_MAX_RULES; ++r) {
            if (rule_domain_hash[r] != c_domain) continue;
            const int8_t strength = quantize_trit(static_cast<float>(rule_strength[r]));
            if (strength > 0) {
                // strict rule for this domain — check if it defeats candidate
                // A strict rule with no supporting gravity contradicts promotion
                if (gravity_trit < 0) {
                    strict_defeat = -1;
                    break;
                }
            } else if (strength == 0) {
                def_sum += (int)gravity_trit; // defeasible: weigh gravity
            } else {
                // defeater rule fires for this domain
                strict_defeat = -1;
                break;
            }
        }

        // ── Step 3: combine ───────────────────────────────────────────────
        int8_t final_trit;
        if (strict_defeat < 0) {
            final_trit = -1;
        } else if (R == 0) {
            // no grammar rules known for this domain — rely on gravity alone
            final_trit = gravity_trit;
        } else {
            final_trit = clamp_trit_int((int)gravity_trit + clamp_trit_int(def_sum));
        }

        out_trits[c]          = final_trit;
        out_target_galaxy[c]  = (final_trit == 1) ? best_gal  : (int32_t)-1;
        out_best_house_idx[c] = (final_trit == 1) ? best_idx  : (int32_t)-1;
    }
}
