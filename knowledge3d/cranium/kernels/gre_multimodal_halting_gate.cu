// Multimodal Halting Gate - raw-score convergence analysis
// Consumes raw path scores plus candidate identity hashes and computes:
//   - top score
//   - winner gap over the second grouped score
//   - agreement count (largest group size)
//   - four halt flags [minimum, gap, agreement, converged]

extern "C" __global__ void gre_multimodal_halting_gate(
    const float* __restrict__ scores_ptr,
    const unsigned int* __restrict__ candidate_hash_ptr,
    unsigned int* __restrict__ flags_ptr,
    float* __restrict__ metrics_ptr,
    unsigned int length,
    float minimum_threshold,
    float gap_threshold,
    float agreement_threshold
)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    if (length == 0) {
        for (int idx = 0; idx < 4; ++idx) {
            flags_ptr[idx] = 0;
        }
        metrics_ptr[0] = 0.0f;
        metrics_ptr[1] = 0.0f;
        metrics_ptr[2] = 0.0f;
        return;
    }

    const unsigned int max_groups = 64;
    unsigned int group_hashes[max_groups];
    float group_best_scores[max_groups];
    unsigned int group_counts[max_groups];
    unsigned int group_count = 0;

    for (unsigned int idx = 0; idx < length && group_count < max_groups; ++idx) {
        unsigned int candidate_hash = candidate_hash_ptr[idx];
        float score = scores_ptr[idx];
        int found = -1;
        for (unsigned int group_idx = 0; group_idx < group_count; ++group_idx) {
            if (group_hashes[group_idx] == candidate_hash) {
                found = (int)group_idx;
                break;
            }
        }
        if (found < 0) {
            group_hashes[group_count] = candidate_hash;
            group_best_scores[group_count] = score;
            group_counts[group_count] = 1;
            ++group_count;
        } else {
            unsigned int group_idx = (unsigned int)found;
            if (score > group_best_scores[group_idx]) {
                group_best_scores[group_idx] = score;
            }
            group_counts[group_idx] += 1;
        }
    }

    float top_score = 0.0f;
    float second_score = 0.0f;
    float agreement_count = 0.0f;
    for (unsigned int idx = 0; idx < group_count; ++idx) {
        float grouped_score = group_best_scores[idx] + (0.02f * (float)(group_counts[idx] - 1));
        if (grouped_score >= top_score) {
            second_score = top_score;
            top_score = grouped_score;
        } else if (grouped_score > second_score) {
            second_score = grouped_score;
        }
        if ((float)group_counts[idx] > agreement_count) {
            agreement_count = (float)group_counts[idx];
        }
    }

    const float gap_score = top_score - second_score;
    const float epsilon = 1.0e-6f;
    unsigned int minimum_flag = (minimum_threshold <= 0.0f || (top_score + epsilon) >= minimum_threshold) ? 1u : 0u;
    unsigned int gap_flag = (gap_threshold <= 0.0f || (gap_score + epsilon) >= gap_threshold) ? 1u : 0u;
    unsigned int agreement_flag = (agreement_threshold <= 0.0f || (agreement_count + epsilon) >= agreement_threshold) ? 1u : 0u;
    unsigned int converged_flag = (minimum_flag && gap_flag && agreement_flag) ? 1u : 0u;

    flags_ptr[0] = minimum_flag;
    flags_ptr[1] = gap_flag;
    flags_ptr[2] = agreement_flag;
    flags_ptr[3] = converged_flag;

    metrics_ptr[0] = top_score;
    metrics_ptr[1] = gap_score;
    metrics_ptr[2] = agreement_count;
}
