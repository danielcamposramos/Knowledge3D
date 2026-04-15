// Defeasible Resolver - non-monotonic conflict resolution over swarm conclusions.
//
// Inputs:
//   conclusions[w * C + c]  : pre-weighted worker conclusion score for candidate c
//   rule_strengths[w]       : +1 strict, 0 defeasible, -1 defeater
//   superiority[w * S + i]  : worker indexes defeated by worker w (UINT32_MAX = none)
//
// Outputs:
//   verdicts[c]   : defeasible verdict in [-1, 1] scaled by supporting confidence
//   proof_tags[c] : packed (D, d) trit pair using OP_TPACK encoding

#include <math.h>
#include <stdint.h>

#define MAX_WORKERS 32

__device__ __forceinline__ int8_t quantize_trit(float value) {
    if (value > 1e-6f) {
        return 1;
    }
    if (value < -1e-6f) {
        return -1;
    }
    return 0;
}

__device__ __forceinline__ int8_t clamp_trit_int(int value) {
    if (value > 0) {
        return 1;
    }
    if (value < 0) {
        return -1;
    }
    return 0;
}

__device__ __forceinline__ uint32_t encode_trit(int8_t trit) {
    return trit > 0 ? 2u : (trit == 0 ? 1u : 0u);
}

__device__ __forceinline__ float stage3_ethical_gate(
    int8_t ethical_trit,
    float conflict_resolved_verdict,
    uint32_t* proof_tag
) {
    if (ethical_trit < 0) {
        if (proof_tag != nullptr) {
            *proof_tag |= (encode_trit(-1) & 0x3u) << 4;
        }
        return 0.0f;
    }
    if (ethical_trit == 0) {
        const float accepted = fabsf(conflict_resolved_verdict) > 1.0e-6f
            ? conflict_resolved_verdict
            : 1.0f;
        if (proof_tag != nullptr) {
            *proof_tag |= (encode_trit(0) & 0x3u) << 4;
        }
        return accepted;
    }
    if (proof_tag != nullptr) {
        *proof_tag |= (encode_trit(1) & 0x3u) << 4;
    }
    return conflict_resolved_verdict;
}

extern "C" __global__ void gre_defeasible_resolver(
    const float* __restrict__ conclusions,
    const int8_t* __restrict__ rule_strengths,
    const uint32_t* __restrict__ superiority,
    float* __restrict__ verdicts,
    uint32_t* __restrict__ proof_tags,
    int num_workers,
    int num_candidates,
    int max_superiors
) {
    const int worker_count = num_workers < MAX_WORKERS ? num_workers : MAX_WORKERS;
    const int stride = blockDim.x;

    for (int candidate = static_cast<int>(threadIdx.x); candidate < num_candidates; candidate += stride) {
        int8_t support[MAX_WORKERS];

        for (int worker = 0; worker < worker_count; ++worker) {
            const float raw = conclusions[(worker * num_candidates) + candidate];
            support[worker] = quantize_trit(raw);
        }

        for (int superior_worker = 0; superior_worker < worker_count; ++superior_worker) {
            if (support[superior_worker] == 0 || max_superiors <= 0) {
                continue;
            }
            const int superiority_base = superior_worker * max_superiors;
            for (int edge = 0; edge < max_superiors; ++edge) {
                const uint32_t inferior_worker = superiority[superiority_base + edge];
                if (inferior_worker == 0xFFFFFFFFu || inferior_worker >= static_cast<uint32_t>(worker_count)) {
                    continue;
                }
                if ((support[superior_worker] * support[inferior_worker]) < 0) {
                    support[inferior_worker] = 0;
                }
            }
        }

        int strict_product = 1;
        int has_strict = 0;
        int defeasible_sum = 0;
        int defeater_triggered = 0;

        for (int worker = 0; worker < worker_count; ++worker) {
            const int8_t surviving = support[worker];
            const int8_t strength = quantize_trit(static_cast<float>(rule_strengths[worker]));
            if (surviving == 0) {
                continue;
            }
            if (strength > 0) {
                strict_product *= static_cast<int>(surviving);
                has_strict = 1;
            } else if (strength == 0) {
                defeasible_sum += static_cast<int>(surviving);
            } else {
                defeater_triggered = 1;
            }
        }

        const int8_t d_definite = has_strict ? clamp_trit_int(strict_product) : 0;
        int8_t d_defeasible = clamp_trit_int(static_cast<int>(d_definite) + clamp_trit_int(defeasible_sum));
        if (defeater_triggered && d_definite == 0) {
            d_defeasible = 0;
        }

        float confidence_sum = 0.0f;
        int confidence_count = 0;
        if (d_defeasible != 0) {
            for (int worker = 0; worker < worker_count; ++worker) {
                if (support[worker] != d_defeasible) {
                    continue;
                }
                confidence_sum += fabsf(conclusions[(worker * num_candidates) + candidate]);
                confidence_count += 1;
            }
        }
        const float mean_confidence = confidence_count > 0
            ? (confidence_sum / static_cast<float>(confidence_count))
            : 0.0f;

        verdicts[candidate] = static_cast<float>(d_defeasible) * mean_confidence;
        proof_tags[candidate] =
            (encode_trit(d_definite) & 0x3u) |
            ((encode_trit(d_defeasible) & 0x3u) << 2);
    }
}

extern "C" __global__ void gre_defeasible_resolver_ethical(
    const float* __restrict__ conclusions,
    const int8_t* __restrict__ rule_strengths,
    const uint32_t* __restrict__ superiority,
    const int8_t* __restrict__ ethical_trits,
    float* __restrict__ verdicts,
    uint32_t* __restrict__ proof_tags,
    int num_workers,
    int num_candidates,
    int max_superiors
) {
    const int worker_count = num_workers < MAX_WORKERS ? num_workers : MAX_WORKERS;
    const int stride = blockDim.x;

    for (int candidate = static_cast<int>(threadIdx.x); candidate < num_candidates; candidate += stride) {
        int8_t support[MAX_WORKERS];

        for (int worker = 0; worker < worker_count; ++worker) {
            const float raw = conclusions[(worker * num_candidates) + candidate];
            support[worker] = quantize_trit(raw);
        }

        for (int superior_worker = 0; superior_worker < worker_count; ++superior_worker) {
            if (support[superior_worker] == 0 || max_superiors <= 0) {
                continue;
            }
            const int superiority_base = superior_worker * max_superiors;
            for (int edge = 0; edge < max_superiors; ++edge) {
                const uint32_t inferior_worker = superiority[superiority_base + edge];
                if (inferior_worker == 0xFFFFFFFFu || inferior_worker >= static_cast<uint32_t>(worker_count)) {
                    continue;
                }
                if ((support[superior_worker] * support[inferior_worker]) < 0) {
                    support[inferior_worker] = 0;
                }
            }
        }

        int strict_product = 1;
        int has_strict = 0;
        int defeasible_sum = 0;
        int defeater_triggered = 0;

        for (int worker = 0; worker < worker_count; ++worker) {
            const int8_t surviving = support[worker];
            const int8_t strength = quantize_trit(static_cast<float>(rule_strengths[worker]));
            if (surviving == 0) {
                continue;
            }
            if (strength > 0) {
                strict_product *= static_cast<int>(surviving);
                has_strict = 1;
            } else if (strength == 0) {
                defeasible_sum += static_cast<int>(surviving);
            } else {
                defeater_triggered = 1;
            }
        }

        const int8_t d_definite = has_strict ? clamp_trit_int(strict_product) : 0;
        int8_t d_defeasible = clamp_trit_int(static_cast<int>(d_definite) + clamp_trit_int(defeasible_sum));
        if (defeater_triggered && d_definite == 0) {
            d_defeasible = 0;
        }

        float confidence_sum = 0.0f;
        int confidence_count = 0;
        if (d_defeasible != 0) {
            for (int worker = 0; worker < worker_count; ++worker) {
                if (support[worker] != d_defeasible) {
                    continue;
                }
                confidence_sum += fabsf(conclusions[(worker * num_candidates) + candidate]);
                confidence_count += 1;
            }
        }
        const float mean_confidence = confidence_count > 0
            ? (confidence_sum / static_cast<float>(confidence_count))
            : 0.0f;

        uint32_t proof =
            (encode_trit(d_definite) & 0x3u) |
            ((encode_trit(d_defeasible) & 0x3u) << 2);
        const float conflict_verdict = static_cast<float>(d_defeasible) * mean_confidence;
        const int8_t ethical_trit = ethical_trits == nullptr ? 0 : quantize_trit(static_cast<float>(ethical_trits[candidate]));
        verdicts[candidate] = stage3_ethical_gate(ethical_trit, conflict_verdict, &proof);
        proof_tags[candidate] = proof;
    }
}
