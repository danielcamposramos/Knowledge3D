// Temporal Reasoning - ordered sequence pattern extraction.

#include <math.h>

extern "C" __global__ void gre_temporal_reasoning(
    const float* __restrict__ sequence,
    float* __restrict__ patterns,
    int T,
    int D
)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    const float eps = 1e-12f;
    const int max_lag = 4;
    const float dim_norm = sqrtf((float)(D > 0 ? D : 1));

    for (int i = 0; i < 24; ++i) {
        patterns[i] = 0.0f;
    }
    if (T <= 0 || D <= 0) {
        return;
    }
    if (T == 1) {
        patterns[12] = 1.0f;
        patterns[13] = 1.0f;
        patterns[18] = 1.0f;
        patterns[20] = 1.0f;
        patterns[21] = 1.0f;
        return;
    }

    float delta_norm_sum = 0.0f;
    float delta_norm_sq = 0.0f;
    float delta_norm_max = 0.0f;
    float delta_norm_min = 1e30f;
    int signed_increasing_steps = 0;
    float prev_signed_delta = 0.0f;
    int signed_delta_initialized = 0;
    int longest_monotone_run = 1;
    int current_monotone_run = 1;

    float accel_norm_sum = 0.0f;
    float accel_norm_sq = 0.0f;
    int accel_sign_changes = 0;
    int accel_zero_crossings = 0;
    float prev_signed_accel = 0.0f;
    int signed_accel_initialized = 0;

    const int steps = T - 1;
    for (int t = 0; t < steps; ++t) {
        float delta_sq = 0.0f;
        float signed_delta = 0.0f;
        for (int d = 0; d < D; ++d) {
            float delta = sequence[(t + 1) * D + d] - sequence[t * D + d];
            delta_sq += delta * delta;
            signed_delta += delta;
        }
        float delta_norm = sqrtf(delta_sq) / (dim_norm + eps);
        delta_norm_sum += delta_norm;
        delta_norm_sq += delta_norm * delta_norm;
        delta_norm_max = fmaxf(delta_norm_max, delta_norm);
        delta_norm_min = fminf(delta_norm_min, delta_norm);
        if (signed_delta > 0.0f) {
            signed_increasing_steps += 1;
        }
        if (signed_delta_initialized) {
            bool monotone = ((signed_delta >= 0.0f) == (prev_signed_delta >= 0.0f));
            current_monotone_run = monotone ? (current_monotone_run + 1) : 1;
            longest_monotone_run = max(longest_monotone_run, current_monotone_run);
        }
        prev_signed_delta = signed_delta;
        signed_delta_initialized = 1;
    }

    float delta_mean = delta_norm_sum / (float)steps;
    float delta_var = fmaxf((delta_norm_sq / (float)steps) - (delta_mean * delta_mean), 0.0f);
    patterns[0] = delta_mean;
    patterns[1] = sqrtf(delta_var);
    patterns[2] = delta_norm_max;
    patterns[3] = (delta_norm_min == 1e30f) ? 0.0f : delta_norm_min;

    const int accel_steps = max(T - 2, 0);
    for (int t = 0; t < accel_steps; ++t) {
        float accel_sq = 0.0f;
        float signed_accel = 0.0f;
        for (int d = 0; d < D; ++d) {
            float prev_delta = sequence[(t + 1) * D + d] - sequence[t * D + d];
            float next_delta = sequence[(t + 2) * D + d] - sequence[(t + 1) * D + d];
            float accel = next_delta - prev_delta;
            accel_sq += accel * accel;
            signed_accel += accel;
        }
        float accel_norm = sqrtf(accel_sq) / (dim_norm + eps);
        accel_norm_sum += accel_norm;
        accel_norm_sq += accel_norm * accel_norm;
        if (signed_accel_initialized) {
            if ((signed_accel > 0.0f && prev_signed_accel < 0.0f) || (signed_accel < 0.0f && prev_signed_accel > 0.0f)) {
                accel_sign_changes += 1;
            }
            if ((signed_accel == 0.0f && prev_signed_accel != 0.0f) || (signed_accel != 0.0f && prev_signed_accel == 0.0f)) {
                accel_zero_crossings += 1;
            }
        }
        prev_signed_accel = signed_accel;
        signed_accel_initialized = 1;
    }
    if (accel_steps > 0) {
        float accel_mean = accel_norm_sum / (float)accel_steps;
        float accel_var = fmaxf((accel_norm_sq / (float)accel_steps) - (accel_mean * accel_mean), 0.0f);
        patterns[4] = accel_mean;
        patterns[5] = sqrtf(accel_var);
        patterns[6] = (float)accel_sign_changes / (float)max(accel_steps - 1, 1);
        patterns[7] = (float)accel_zero_crossings / (float)max(accel_steps - 1, 1);
    }

    for (int lag = 1; lag <= max_lag; ++lag) {
        if (lag >= T) {
            patterns[7 + lag] = 0.0f;
            continue;
        }
        float corr_sum = 0.0f;
        int corr_count = 0;
        for (int t = 0; t + lag < T; ++t) {
            float dot = 0.0f;
            float norm_a = 0.0f;
            float norm_b = 0.0f;
            for (int d = 0; d < D; ++d) {
                float a = sequence[t * D + d];
                float b = sequence[(t + lag) * D + d];
                dot += a * b;
                norm_a += a * a;
                norm_b += b * b;
            }
            corr_sum += dot * rsqrtf((norm_a + eps) * (norm_b + eps));
            corr_count += 1;
        }
        patterns[7 + lag] = corr_count > 0 ? (corr_sum / (float)corr_count) : 0.0f;
    }

    patterns[12] = (float)signed_increasing_steps / (float)steps;
    patterns[13] = (float)longest_monotone_run / (float)max(T, 1);

    int recurrence_pairs = 0;
    float recurrence_interval_sum = 0.0f;
    float recurrence_best = -1.0f;
    float recurrence_positive_sum = 0.0f;
    int recurrence_positive_count = 0;
    for (int t0 = 0; t0 < T; ++t0) {
        for (int t1 = t0 + 1; t1 < T; ++t1) {
            float dot = 0.0f;
            float norm_a = 0.0f;
            float norm_b = 0.0f;
            for (int d = 0; d < D; ++d) {
                float a = sequence[t0 * D + d];
                float b = sequence[t1 * D + d];
                dot += a * b;
                norm_a += a * a;
                norm_b += b * b;
            }
            float cosine = dot * rsqrtf((norm_a + eps) * (norm_b + eps));
            recurrence_best = fmaxf(recurrence_best, cosine);
            if (cosine > 0.0f) {
                recurrence_positive_sum += cosine;
                recurrence_positive_count += 1;
            }
            if (cosine >= 0.75f) {
                recurrence_pairs += 1;
                recurrence_interval_sum += (float)(t1 - t0);
            }
        }
    }
    int pair_total = max((T * (T - 1)) / 2, 1);
    patterns[14] = (float)recurrence_pairs / (float)pair_total;
    patterns[15] = recurrence_pairs > 0 ? (recurrence_interval_sum / (float)recurrence_pairs) / (float)max(T - 1, 1) : 0.0f;
    patterns[16] = recurrence_best > -1.0f ? ((recurrence_best + 1.0f) * 0.5f) : 0.0f;
    patterns[17] = recurrence_positive_count > 0 ? (recurrence_positive_sum / (float)recurrence_positive_count) : 0.0f;

    if (T >= 3) {
        float forward_error_sum = 0.0f;
        float backward_error_sum = 0.0f;
        int error_count = 0;
        for (int t = 1; t < T - 1; ++t) {
            float forward_error_sq = 0.0f;
            float backward_error_sq = 0.0f;
            for (int d = 0; d < D; ++d) {
                float prev = sequence[(t - 1) * D + d];
                float curr = sequence[t * D + d];
                float next = sequence[(t + 1) * D + d];
                float pred_forward = curr + (curr - prev);
                float pred_backward = curr + (curr - next);
                float fe = next - pred_forward;
                float be = prev - pred_backward;
                forward_error_sq += fe * fe;
                backward_error_sq += be * be;
            }
            forward_error_sum += sqrtf(forward_error_sq) / (dim_norm + eps);
            backward_error_sum += sqrtf(backward_error_sq) / (dim_norm + eps);
            error_count += 1;
        }
        float forward_error = forward_error_sum / (float)max(error_count, 1);
        float backward_error = backward_error_sum / (float)max(error_count, 1);
        patterns[18] = 1.0f / (1.0f + forward_error);
        patterns[19] = (backward_error - forward_error) / (backward_error + forward_error + eps);
    }

    float previous_tail_distance = 0.0f;
    int tail_distance_initialized = 0;
    int convergence_steps = 0;
    float first_tail_distance = 0.0f;
    float tail_distance_sum = 0.0f;
    int tail_count = 0;
    for (int t = 0; t < T; ++t) {
        float dist_sq = 0.0f;
        for (int d = 0; d < D; ++d) {
            float delta = sequence[t * D + d] - sequence[(T - 1) * D + d];
            dist_sq += delta * delta;
        }
        float tail_distance = sqrtf(dist_sq) / (dim_norm + eps);
        if (t == 0) {
            first_tail_distance = tail_distance;
        }
        if (tail_distance_initialized && tail_distance <= previous_tail_distance) {
            convergence_steps += 1;
        }
        previous_tail_distance = tail_distance;
        tail_distance_initialized = 1;
        if (t >= max(0, T - 3)) {
            tail_distance_sum += tail_distance;
            tail_count += 1;
        }
    }
    patterns[20] = (float)convergence_steps / (float)max(T - 1, 1);
    patterns[21] = first_tail_distance > 0.0f ? fmaxf(first_tail_distance - previous_tail_distance, 0.0f) / (first_tail_distance + eps) : 0.0f;
    patterns[22] = tail_distance_sum / (float)max(tail_count, 1);

    float entropy = 0.0f;
    if (delta_norm_sum > eps) {
        for (int t = 0; t < steps; ++t) {
            float delta_sq = 0.0f;
            for (int d = 0; d < D; ++d) {
                float delta = sequence[(t + 1) * D + d] - sequence[t * D + d];
                delta_sq += delta * delta;
            }
            float delta_norm = sqrtf(delta_sq) / (dim_norm + eps);
            float p = delta_norm / (delta_norm_sum + eps);
            if (p > eps) {
                entropy -= p * logf(p);
            }
        }
        entropy /= logf((float)max(steps, 2));
    }
    patterns[23] = entropy;
}
