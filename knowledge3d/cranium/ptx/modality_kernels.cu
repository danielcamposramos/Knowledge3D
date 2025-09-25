#define MODALITY_MAX_THREADS 256

extern "C" __device__ inline void atomicMinFloat(float* address, float value) {
    int* addr_as_i = reinterpret_cast<int*>(address);
    int old = *addr_as_i;
    int assumed;
    while (__int_as_float(old) > value) {
        assumed = old;
        int desired = __float_as_int(value);
        int prior = atomicCAS(addr_as_i, assumed, desired);
        if (prior == assumed) {
            break;
        }
        old = prior;
    }
}

extern "C" __device__ inline void atomicMaxFloat(float* address, float value) {
    int* addr_as_i = reinterpret_cast<int*>(address);
    int old = *addr_as_i;
    int assumed;
    while (__int_as_float(old) < value) {
        assumed = old;
        int desired = __float_as_int(value);
        int prior = atomicCAS(addr_as_i, assumed, desired);
        if (prior == assumed) {
            break;
        }
        old = prior;
    }
}

extern "C" __global__ void encode_text(
    const unsigned char* __restrict__ text,
    int length,
    float* __restrict__ output
) {
    __shared__ float sum;
    __shared__ float sumsq;
    __shared__ float uppercase;
    __shared__ float digits;
    __shared__ float vowels;
    __shared__ float hist[8];

    if (threadIdx.x == 0) {
        sum = 0.0f;
        sumsq = 0.0f;
        uppercase = 0.0f;
        digits = 0.0f;
        vowels = 0.0f;
    }
    if (threadIdx.x < 8) {
        hist[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    float local_uppercase = 0.0f;
    float local_digits = 0.0f;
    float local_vowels = 0.0f;
    float local_hist[8];
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        local_hist[b] = 0.0f;
    }

    for (int idx = threadIdx.x; idx < length; idx += blockDim.x) {
        unsigned char ch = text[idx];
        float cf = static_cast<float>(ch);
        local_sum += cf;
        local_sumsq += cf * cf;
        if (ch >= 'A' && ch <= 'Z') {
            local_uppercase += 1.0f;
        }
        if (ch >= '0' && ch <= '9') {
            local_digits += 1.0f;
        }
        unsigned char lower = ch | 32u;
        if (lower == 'a' || lower == 'e' || lower == 'i' || lower == 'o' || lower == 'u') {
            local_vowels += 1.0f;
        }
        int bin = ch >> 5;
        if (bin > 7) {
            bin = 7;
        }
        local_hist[bin] += 1.0f;
    }

    atomicAdd(&sum, local_sum);
    atomicAdd(&sumsq, local_sumsq);
    atomicAdd(&uppercase, local_uppercase);
    atomicAdd(&digits, local_digits);
    atomicAdd(&vowels, local_vowels);
    for (int b = 0; b < 8; ++b) {
        atomicAdd(&hist[b], local_hist[b]);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float len = static_cast<float>(length);
        float inv_len = len > 0.0f ? 1.0f / len : 0.0f;
        output[0] = len / 512.0f;
        float mean = sum * inv_len;
        output[1] = mean / 255.0f;
        float mean_sq = sumsq * inv_len;
        float variance = 0.0f;
        if (len > 0.0f) {
            variance = fmaxf(mean_sq - mean * mean, 0.0f);
        }
        output[2] = sqrtf(variance) / 255.0f;
        output[3] = uppercase * inv_len;
        output[4] = digits * inv_len;
        output[5] = vowels * inv_len;
        float trig_cos = 0.0f;
        float trig_sin = 0.0f;
        for (int b = 0; b < 8; ++b) {
            float bin_ratio = len > 0.0f ? hist[b] * inv_len : 0.0f;
            output[8 + b] = bin_ratio;
            float angle = 0.3926990817f * static_cast<float>(b);
            trig_cos += bin_ratio * cosf(angle);
            trig_sin += bin_ratio * sinf(angle);
        }
        output[6] = trig_cos;
        output[7] = trig_sin;
    }
}

extern "C" __global__ void encode_audio(
    const float* __restrict__ audio,
    int samples,
    float sample_rate,
    float* __restrict__ output
) {
    __shared__ float sum;
    __shared__ float sumsq;
    __shared__ float abs_sum;
    __shared__ float zero_cross;
    __shared__ float hist[16];
    __shared__ float first_samples[MODALITY_MAX_THREADS];
    __shared__ float last_samples[MODALITY_MAX_THREADS];
    __shared__ int chunk_has_data[MODALITY_MAX_THREADS];

    if (threadIdx.x == 0) {
        sum = 0.0f;
        sumsq = 0.0f;
        abs_sum = 0.0f;
        zero_cross = 0.0f;
    }
    if (threadIdx.x < 16) {
        hist[threadIdx.x] = 0.0f;
    }
    if (threadIdx.x < MODALITY_MAX_THREADS) {
        first_samples[threadIdx.x] = 0.0f;
        last_samples[threadIdx.x] = 0.0f;
        chunk_has_data[threadIdx.x] = 0;
    }
    __syncthreads();

    int tid = threadIdx.x;
    int chunk_size = (samples + blockDim.x - 1) / blockDim.x;
    int start = tid * chunk_size;
    int end = start + chunk_size;
    if (end > samples) {
        end = samples;
    }

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    float local_abs = 0.0f;
    float local_zc = 0.0f;
    float local_hist[16];
    #pragma unroll
    for (int b = 0; b < 16; ++b) {
        local_hist[b] = 0.0f;
    }

    bool has_prev = false;
    float prev = 0.0f;

    if (start < end) {
        float first = audio[start];
        first_samples[tid] = first;
        chunk_has_data[tid] = 1;
        local_sum += first;
        local_sumsq += first * first;
        local_abs += fabsf(first);
        prev = first;
        has_prev = true;
        int bucket = samples > 0 ? (start * 16) / samples : 0;
        if (bucket > 15) {
            bucket = 15;
        }
        local_hist[bucket] += first * first;
    }

    for (int idx = start + 1; idx < end; ++idx) {
        float sample = audio[idx];
        local_sum += sample;
        local_sumsq += sample * sample;
        local_abs += fabsf(sample);
        if (has_prev) {
            if ((sample > 0.0f && prev <= 0.0f) || (sample < 0.0f && prev >= 0.0f)) {
                local_zc += 1.0f;
            }
        }
        prev = sample;
        has_prev = true;
        int bucket = samples > 0 ? (idx * 16) / samples : 0;
        if (bucket > 15) {
            bucket = 15;
        }
        local_hist[bucket] += sample * sample;
    }

    if (start < end) {
        last_samples[tid] = prev;
    }

    atomicAdd(&sum, local_sum);
    atomicAdd(&sumsq, local_sumsq);
    atomicAdd(&abs_sum, local_abs);
    atomicAdd(&zero_cross, local_zc);
    for (int b = 0; b < 16; ++b) {
        atomicAdd(&hist[b], local_hist[b]);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int t = 0; t < blockDim.x - 1; ++t) {
            if (chunk_has_data[t] && chunk_has_data[t + 1]) {
                float a = last_samples[t];
                float b = first_samples[t + 1];
                if ((b > 0.0f && a <= 0.0f) || (b < 0.0f && a >= 0.0f)) {
                    zero_cross += 1.0f;
                }
            }
        }
        float N = samples > 0 ? static_cast<float>(samples) : 1.0f;
        float invN = 1.0f / N;
        output[0] = sample_rate > 0.0f ? (static_cast<float>(samples) / sample_rate) : 0.0f;
        float mean = sum * invN;
        output[1] = mean;
        float var = fmaxf(sumsq * invN - mean * mean, 0.0f);
        float rms = sqrtf(var);
        output[2] = rms;
        output[3] = abs_sum * invN;
        output[4] = zero_cross * invN;
        float energy = sumsq * invN;
        output[5] = energy;
        float sum_hist = 0.0f;
        float max_bucket = 0.0f;
        float min_bucket = 1e9f;
        for (int b = 0; b < 16; ++b) {
            float val = hist[b];
            sum_hist += val;
            if (val > max_bucket) {
                max_bucket = val;
            }
            if (val < min_bucket) {
                min_bucket = val;
            }
        }
        sum_hist = fmaxf(sum_hist, 1e-6f);
        output[6] = max_bucket / sum_hist;
        output[7] = min_bucket / sum_hist;
        for (int b = 0; b < 16; ++b) {
            output[8 + b] = hist[b] / sum_hist;
        }
    }
}

extern "C" __global__ void encode_image(
    const float* __restrict__ image,
    int width,
    int height,
    int channels,
    float* __restrict__ output
) {
    __shared__ float sum_r;
    __shared__ float sum_g;
    __shared__ float sum_b;
    __shared__ float sq_r;
    __shared__ float sq_g;
    __shared__ float sq_b;
    __shared__ float sum_brightness;
    __shared__ float sq_brightness;
    __shared__ float sum_sat;
    __shared__ float sq_sat;
    __shared__ float hist[8];
    __shared__ float min_brightness;
    __shared__ float max_brightness;

    if (threadIdx.x == 0) {
        sum_r = sum_g = sum_b = 0.0f;
        sq_r = sq_g = sq_b = 0.0f;
        sum_brightness = sq_brightness = 0.0f;
        sum_sat = sq_sat = 0.0f;
        min_brightness = 1e9f;
        max_brightness = -1e9f;
    }
    if (threadIdx.x < 8) {
        hist[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    float local_hist[8];
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        local_hist[b] = 0.0f;
    }
    float local_sum_r = 0.0f;
    float local_sum_g = 0.0f;
    float local_sum_b = 0.0f;
    float local_sq_r = 0.0f;
    float local_sq_g = 0.0f;
    float local_sq_b = 0.0f;
    float local_sum_brightness = 0.0f;
    float local_sq_brightness = 0.0f;
    float local_sum_sat = 0.0f;
    float local_sq_sat = 0.0f;
    float local_min = 1e9f;
    float local_max = -1e9f;

    int pixels = width * height;
    for (int idx = threadIdx.x; idx < pixels; idx += blockDim.x) {
        int base = idx * channels;
        float r = image[base + 0];
        float g = channels > 1 ? image[base + 1] : r;
        float b = channels > 2 ? image[base + 2] : r;
        local_sum_r += r;
        local_sum_g += g;
        local_sum_b += b;
        local_sq_r += r * r;
        local_sq_g += g * g;
        local_sq_b += b * b;
        float brightness = (r + g + b) / 3.0f;
        local_sum_brightness += brightness;
        local_sq_brightness += brightness * brightness;
        float max_c = fmaxf(r, fmaxf(g, b));
        float min_c = fminf(r, fminf(g, b));
        float sat = max_c - min_c;
        local_sum_sat += sat;
        local_sq_sat += sat * sat;
        int bin = static_cast<int>(brightness * 8.0f);
        if (bin < 0) {
            bin = 0;
        } else if (bin > 7) {
            bin = 7;
        }
        local_hist[bin] += 1.0f;
        if (brightness < local_min) {
            local_min = brightness;
        }
        if (brightness > local_max) {
            local_max = brightness;
        }
    }

    atomicAdd(&sum_r, local_sum_r);
    atomicAdd(&sum_g, local_sum_g);
    atomicAdd(&sum_b, local_sum_b);
    atomicAdd(&sq_r, local_sq_r);
    atomicAdd(&sq_g, local_sq_g);
    atomicAdd(&sq_b, local_sq_b);
    atomicAdd(&sum_brightness, local_sum_brightness);
    atomicAdd(&sq_brightness, local_sq_brightness);
    atomicAdd(&sum_sat, local_sum_sat);
    atomicAdd(&sq_sat, local_sq_sat);
    for (int b = 0; b < 8; ++b) {
        atomicAdd(&hist[b], local_hist[b]);
    }
    atomicMinFloat(&min_brightness, local_min);
    atomicMaxFloat(&max_brightness, local_max);
    __syncthreads();

    if (threadIdx.x == 0) {
        float pixels_f = pixels > 0 ? static_cast<float>(pixels) : 1.0f;
        float inv_pixels = 1.0f / pixels_f;
        float mean_r = sum_r * inv_pixels;
        float mean_g = sum_g * inv_pixels;
        float mean_b = sum_b * inv_pixels;
        float std_r = sqrtf(fmaxf(sq_r * inv_pixels - mean_r * mean_r, 0.0f));
        float std_g = sqrtf(fmaxf(sq_g * inv_pixels - mean_g * mean_g, 0.0f));
        float std_b = sqrtf(fmaxf(sq_b * inv_pixels - mean_b * mean_b, 0.0f));
        float mean_brightness = sum_brightness * inv_pixels;
        float std_brightness = sqrtf(fmaxf(sq_brightness * inv_pixels - mean_brightness * mean_brightness, 0.0f));
        float mean_sat = sum_sat * inv_pixels;
        float std_sat = sqrtf(fmaxf(sq_sat * inv_pixels - mean_sat * mean_sat, 0.0f));
        float hist_sum = 0.0f;
        for (int b = 0; b < 8; ++b) {
            hist_sum += hist[b];
        }
        hist_sum = fmaxf(hist_sum, 1e-6f);

        output[0] = static_cast<float>(width) / 512.0f;
        output[1] = static_cast<float>(height) / 512.0f;
        output[2] = mean_r;
        output[3] = mean_g;
        output[4] = mean_b;
        output[5] = std_r;
        output[6] = std_g;
        output[7] = std_b;
        output[8] = mean_brightness;
        output[9] = std_brightness;
        output[10] = mean_sat;
        output[11] = std_sat;
        for (int b = 0; b < 8; ++b) {
            output[12 + b] = hist[b] / hist_sum;
        }
        output[20] = min_brightness;
        output[21] = max_brightness;
        float colorfulness = sqrtf(std_r * std_r + std_g * std_g + std_b * std_b);
        output[22] = colorfulness;
        float dynamic_range = max_brightness - min_brightness;
        output[23] = dynamic_range;
    }
}

extern "C" __global__ void encode_video(
    const float* __restrict__ video,
    int frames,
    int width,
    int height,
    int channels,
    float* __restrict__ output
) {
    __shared__ float sum_r;
    __shared__ float sum_g;
    __shared__ float sum_b;
    __shared__ float sq_r;
    __shared__ float sq_g;
    __shared__ float sq_b;
    __shared__ float sum_brightness;
    __shared__ float sq_brightness;
    __shared__ float sum_sat;
    __shared__ float sq_sat;
    __shared__ float hist[8];
    __shared__ float min_brightness;
    __shared__ float max_brightness;
    __shared__ float motion_sum;
    __shared__ float motion_sq_sum;

    if (threadIdx.x == 0) {
        sum_r = sum_g = sum_b = 0.0f;
        sq_r = sq_g = sq_b = 0.0f;
        sum_brightness = sq_brightness = 0.0f;
        sum_sat = sq_sat = 0.0f;
        motion_sum = motion_sq_sum = 0.0f;
        min_brightness = 1e9f;
        max_brightness = -1e9f;
    }
    if (threadIdx.x < 8) {
        hist[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    float local_hist[8];
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        local_hist[b] = 0.0f;
    }
    float local_sum_r = 0.0f;
    float local_sum_g = 0.0f;
    float local_sum_b = 0.0f;
    float local_sq_r = 0.0f;
    float local_sq_g = 0.0f;
    float local_sq_b = 0.0f;
    float local_sum_brightness = 0.0f;
    float local_sq_brightness = 0.0f;
    float local_sum_sat = 0.0f;
    float local_sq_sat = 0.0f;
    float local_motion_sum = 0.0f;
    float local_motion_sq_sum = 0.0f;
    float local_min = 1e9f;
    float local_max = -1e9f;

    int pixels = width * height;
    int total = frames * pixels;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int frame = idx / pixels;
        int pixel = idx % pixels;
        int base = (frame * pixels + pixel) * channels;
        float r = video[base + 0];
        float g = channels > 1 ? video[base + 1] : r;
        float b = channels > 2 ? video[base + 2] : r;
        local_sum_r += r;
        local_sum_g += g;
        local_sum_b += b;
        local_sq_r += r * r;
        local_sq_g += g * g;
        local_sq_b += b * b;
        float brightness = (r + g + b) / 3.0f;
        local_sum_brightness += brightness;
        local_sq_brightness += brightness * brightness;
        float max_c = fmaxf(r, fmaxf(g, b));
        float min_c = fminf(r, fminf(g, b));
        float sat = max_c - min_c;
        local_sum_sat += sat;
        local_sq_sat += sat * sat;
        int bin = static_cast<int>(brightness * 8.0f);
        if (bin < 0) {
            bin = 0;
        } else if (bin > 7) {
            bin = 7;
        }
        local_hist[bin] += 1.0f;
        if (brightness < local_min) {
            local_min = brightness;
        }
        if (brightness > local_max) {
            local_max = brightness;
        }
        if (frame > 0) {
            int prev_base = ((frame - 1) * pixels + pixel) * channels;
            float pr = video[prev_base + 0];
            float pg = channels > 1 ? video[prev_base + 1] : pr;
            float pb = channels > 2 ? video[prev_base + 2] : pr;
            float prev_brightness = (pr + pg + pb) / 3.0f;
            float diff = brightness - prev_brightness;
            float abs_diff = fabsf(diff);
            local_motion_sum += abs_diff;
            local_motion_sq_sum += abs_diff * abs_diff;
        }
    }

    atomicAdd(&sum_r, local_sum_r);
    atomicAdd(&sum_g, local_sum_g);
    atomicAdd(&sum_b, local_sum_b);
    atomicAdd(&sq_r, local_sq_r);
    atomicAdd(&sq_g, local_sq_g);
    atomicAdd(&sq_b, local_sq_b);
    atomicAdd(&sum_brightness, local_sum_brightness);
    atomicAdd(&sq_brightness, local_sq_brightness);
    atomicAdd(&sum_sat, local_sum_sat);
    atomicAdd(&sq_sat, local_sq_sat);
    atomicAdd(&motion_sum, local_motion_sum);
    atomicAdd(&motion_sq_sum, local_motion_sq_sum);
    for (int b = 0; b < 8; ++b) {
        atomicAdd(&hist[b], local_hist[b]);
    }
    atomicMinFloat(&min_brightness, local_min);
    atomicMaxFloat(&max_brightness, local_max);
    __syncthreads();

    if (threadIdx.x == 0) {
        float total_samples = total > 0 ? static_cast<float>(total) : 1.0f;
        float inv_total = 1.0f / total_samples;
        float mean_r = sum_r * inv_total;
        float mean_g = sum_g * inv_total;
        float mean_b = sum_b * inv_total;
        float std_r = sqrtf(fmaxf(sq_r * inv_total - mean_r * mean_r, 0.0f));
        float std_g = sqrtf(fmaxf(sq_g * inv_total - mean_g * mean_g, 0.0f));
        float std_b = sqrtf(fmaxf(sq_b * inv_total - mean_b * mean_b, 0.0f));
        float mean_brightness = sum_brightness * inv_total;
        float std_brightness = sqrtf(fmaxf(sq_brightness * inv_total - mean_brightness * mean_brightness, 0.0f));
        float mean_sat = sum_sat * inv_total;
        float std_sat = sqrtf(fmaxf(sq_sat * inv_total - mean_sat * mean_sat, 0.0f));
        float hist_sum = 0.0f;
        for (int b = 0; b < 8; ++b) {
            hist_sum += hist[b];
        }
        hist_sum = fmaxf(hist_sum, 1e-6f);
        float motion_samples = (frames > 1) ? static_cast<float>((frames - 1) * pixels) : 1.0f;
        float motion_mean = motion_sum / motion_samples;
        float motion_var = fmaxf(motion_sq_sum / motion_samples - motion_mean * motion_mean, 0.0f);
        float motion_std = sqrtf(motion_var);

        output[0] = frames / 64.0f;
        output[1] = mean_r;
        output[2] = mean_g;
        output[3] = mean_b;
        output[4] = std_r;
        output[5] = std_g;
        output[6] = std_b;
        output[7] = mean_brightness;
        output[8] = std_brightness;
        output[9] = motion_mean;
        output[10] = motion_std;
        output[11] = mean_sat;
        output[12] = std_sat;
        for (int b = 0; b < 8; ++b) {
            output[13 + b] = hist[b] / hist_sum;
        }
        output[21] = min_brightness;
        output[22] = max_brightness;
        float colorfulness = sqrtf(std_r * std_r + std_g * std_g + std_b * std_b);
        output[23] = colorfulness;
    }
}
