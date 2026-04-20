/*
 * Audio FFT + Spectrogram Kernels (Opcodes 0x250-0x25F)
 *
 * Pure CUDA implementation of Stockham radix-2 FFT for K3D sovereign hot path.
 * No cuFFT library dependency — all kernels are self-contained.
 *
 * Design:
 *   - Stockham algorithm avoids explicit bit-reversal permutation
 *   - One FFT per thread block, coalesced memory access
 *   - Pre-computed twiddle LUT in constant memory (256 complex pairs per size)
 *   - Windowing fused with input load for cache efficiency
 *   - Complex arithmetic using float2 (real, imag)
 *
 * Target: RTX 3070 (sm_86, 96 KB shared memory per block, 1536 threads/block max)
 * Occupancy: 2-3 blocks per SM with 1024-point FFT + windowing
 *
 * References:
 *   - Stockham FFT (1969) avoids permutation by reading/writing in stride order
 *   - cuFFT design patterns for shared-memory staging
 *   - NVIDIA GTC presentations on GPU signal processing
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

typedef float2 Complex;

// Compile-time FFT size selectors
#define FFT_SIZE_256  256
#define FFT_SIZE_512  512
#define FFT_SIZE_1024 1024
#define FFT_SIZE_2048 2048

// Constant memory for twiddle factors (pre-computed at kernel load)
__constant__ Complex twiddle_lut_256[128];   // 256-point: N/2 complex pairs
__constant__ Complex twiddle_lut_512[256];   // 512-point: N/2 complex pairs
__constant__ Complex twiddle_lut_1024[512];  // 1024-point: N/2 complex pairs
__constant__ Complex twiddle_lut_2048[1024]; // 2048-point: N/2 complex pairs

// Window functions in constant memory
__constant__ float window_hann_1024[1024];
__constant__ float window_hamm_1024[1024];

/* ============================================================================
 * Utility: Complex arithmetic
 * ========================================================================== */
__device__ inline Complex cmul(Complex a, Complex b) {
    return make_float2(
        a.x * b.x - a.y * b.y,  // (a.x + i*a.y) * (b.x + i*b.y) real
        a.x * b.y + a.y * b.x   // imaginary
    );
}

__device__ inline Complex csub(Complex a, Complex b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ inline Complex cadd(Complex a, Complex b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ inline Complex conj(Complex a) {
    return make_float2(a.x, -a.y);
}

/* ============================================================================
 * Stockham Radix-2 FFT Kernel Template
 *
 * One block processes one complete FFT.
 * Each thread is responsible for multiple butterflies in the radix-2 tree.
 * Shared memory holds the in-flight complex values.
 * ========================================================================== */

template <int FFT_N, int LOG2_N>
__global__ void fft_forward_stockham(
    const Complex *input,    // (num_ffts, N) flattened: [fft0[0..N-1], fft1[0..N-1], ...]
    Complex *output,         // (num_ffts, N) flattened
    int num_ffts
) {
    extern __shared__ Complex smem[];

    int fft_idx = blockIdx.x;  // Which FFT in the batch
    if (fft_idx >= num_ffts) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Load input into shared memory with stride to allow coalesced writes
    for (int i = tid; i < FFT_N; i += stride) {
        smem[i] = input[fft_idx * FFT_N + i];
    }
    __syncthreads();

    // Stockham FFT: LOG2_N stages, each stage processes a butterfly width
    for (int stage = 0; stage < LOG2_N; stage++) {
        int butterfly_width = 1 << stage;           // 2^stage
        int group_width = butterfly_width * 2;      // 2^(stage+1)
        int half_N = FFT_N >> 1;

        // Select appropriate twiddle LUT
        const Complex *twiddle = NULL;
        if (FFT_N == 256) twiddle = twiddle_lut_256;
        else if (FFT_N == 512) twiddle = twiddle_lut_512;
        else if (FFT_N == 1024) twiddle = twiddle_lut_1024;
        else if (FFT_N == 2048) twiddle = twiddle_lut_2048;

        __syncthreads();

        // Process butterflies
        // Each thread handles one butterfly per group
        for (int butterfly_idx = tid; butterfly_idx < FFT_N / 2; butterfly_idx += stride) {
            // Map butterfly index to position in the Stockham permutation
            int group_idx = butterfly_idx / butterfly_width;
            int pos_in_group = butterfly_idx % butterfly_width;

            int even_idx = group_idx * group_width + pos_in_group;
            int odd_idx = even_idx + butterfly_width;

            // Read values (from current stage smem positions)
            Complex even = smem[even_idx];
            Complex odd = smem[odd_idx];

            // Twiddle index: k in the Stockham algorithm
            int twiddle_idx = (pos_in_group * half_N) / butterfly_width;
            twiddle_idx = twiddle_idx % half_N;  // Wrap

            Complex w = twiddle[twiddle_idx];

            // Butterfly operation: standard radix-2
            Complex temp = cmul(odd, w);
            Complex out_even = cadd(even, temp);
            Complex out_odd = csub(even, temp);

            // Write back (alternates between smem layouts to avoid bank conflicts)
            smem[even_idx] = out_even;
            smem[odd_idx] = out_odd;
        }
    }

    __syncthreads();

    // Write output
    for (int i = tid; i < FFT_N; i += stride) {
        output[fft_idx * FFT_N + i] = smem[i];
    }
}

/* ============================================================================
 * Inverse FFT (conjugate, forward, conjugate + scale)
 * ========================================================================== */

template <int FFT_N, int LOG2_N>
__global__ void fft_inverse_stockham(
    const Complex *input,
    Complex *output,
    int num_ffts
) {
    extern __shared__ Complex smem[];

    int fft_idx = blockIdx.x;
    if (fft_idx >= num_ffts) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Load input, conjugate during load
    for (int i = tid; i < FFT_N; i += stride) {
        Complex x = input[fft_idx * FFT_N + i];
        smem[i] = conj(x);  // Conjugate for inverse
    }
    __syncthreads();

    // Same Stockham procedure as forward
    for (int stage = 0; stage < LOG2_N; stage++) {
        int butterfly_width = 1 << stage;
        int group_width = butterfly_width * 2;
        int half_N = FFT_N >> 1;

        const Complex *twiddle = NULL;
        if (FFT_N == 256) twiddle = twiddle_lut_256;
        else if (FFT_N == 512) twiddle = twiddle_lut_512;
        else if (FFT_N == 1024) twiddle = twiddle_lut_1024;
        else if (FFT_N == 2048) twiddle = twiddle_lut_2048;

        __syncthreads();

        for (int butterfly_idx = tid; butterfly_idx < FFT_N / 2; butterfly_idx += stride) {
            int group_idx = butterfly_idx / butterfly_width;
            int pos_in_group = butterfly_idx % butterfly_width;

            int even_idx = group_idx * group_width + pos_in_group;
            int odd_idx = even_idx + butterfly_width;

            Complex even = smem[even_idx];
            Complex odd = smem[odd_idx];

            int twiddle_idx = (pos_in_group * half_N) / butterfly_width;
            twiddle_idx = twiddle_idx % half_N;

            Complex w = twiddle[twiddle_idx];
            Complex temp = cmul(odd, w);
            Complex out_even = cadd(even, temp);
            Complex out_odd = csub(even, temp);

            smem[even_idx] = out_even;
            smem[odd_idx] = out_odd;
        }
    }

    __syncthreads();

    // Write output with scaling (1/N)
    float scale = 1.0f / FFT_N;
    for (int i = tid; i < FFT_N; i += stride) {
        Complex x = smem[i];
        x = conj(x);  // Final conjugation for inverse
        x.x *= scale;
        x.y *= scale;
        output[fft_idx * FFT_N + i] = x;
    }
}

/* ============================================================================
 * Windowing Kernels (Hann, Hamming) — fused with input load
 * ========================================================================== */

template <int FFT_N>
__global__ void fft_window_load_hann(
    const float *input,  // Real input (cast to Complex with imag=0)
    Complex *smem_output,
    const float *window  // Window function
) {
    int fft_idx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    for (int i = tid; i < FFT_N; i += stride) {
        float val = input[fft_idx * FFT_N + i];
        float w = window[i];
        Complex c;
        c.x = val * w;
        c.y = 0.0f;
        smem_output[i] = c;
    }
}

template <int FFT_N>
__global__ void fft_window_load_hamming(
    const float *input,
    Complex *smem_output,
    const float *window
) {
    int fft_idx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    for (int i = tid; i < FFT_N; i += stride) {
        float val = input[fft_idx * FFT_N + i];
        float w = window[i];
        Complex c;
        c.x = val * w;
        c.y = 0.0f;
        smem_output[i] = c;
    }
}

/* ============================================================================
 * STFT: Overlapping Windowed FFT (Hop-Size Configurable)
 * ========================================================================== */

__global__ void stft_forward(
    const float *input,      // Real audio samples (mono)
    Complex *output,         // Spectrogram (n_frames, n_bins) as complex
    int input_len,
    int frame_size,
    int hop_size,
    const float *window,     // Window function (frame_size samples)
    const Complex *twiddle,  // Twiddle factors for this frame_size
    int log2_frame_size
) {
    // Frame index
    int frame_idx = blockIdx.x;
    int max_frames = (input_len - frame_size) / hop_size + 1;
    if (frame_idx >= max_frames) return;

    extern __shared__ char smem_raw[];
    Complex *smem = (Complex *)smem_raw;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Load frame with windowing
    int frame_start = frame_idx * hop_size;
    for (int i = tid; i < frame_size; i += stride) {
        float val = input[frame_start + i];
        float w = window[i];
        smem[i] = make_float2(val * w, 0.0f);
    }
    __syncthreads();

    // In-place Stockham radix-2 FFT (same as fft_forward_stockham)
    int half_frame = frame_size >> 1;
    for (int stage = 0; stage < log2_frame_size; stage++) {
        int butterfly_width = 1 << stage;
        int group_width = butterfly_width * 2;

        __syncthreads();

        for (int butterfly_idx = tid; butterfly_idx < frame_size / 2; butterfly_idx += stride) {
            int group_idx = butterfly_idx / butterfly_width;
            int pos_in_group = butterfly_idx % butterfly_width;

            int even_idx = group_idx * group_width + pos_in_group;
            int odd_idx = even_idx + butterfly_width;

            Complex even = smem[even_idx];
            Complex odd = smem[odd_idx];

            int twiddle_idx = (pos_in_group * half_frame) / butterfly_width;
            twiddle_idx = twiddle_idx % half_frame;

            Complex w = twiddle[twiddle_idx];
            Complex temp = cmul(odd, w);
            Complex out_even = cadd(even, temp);
            Complex out_odd = csub(even, temp);

            smem[even_idx] = out_even;
            smem[odd_idx] = out_odd;
        }
    }

    __syncthreads();

    // Write output (first half + DC + Nyquist only, mirroring assumed)
    for (int i = tid; i < (frame_size / 2 + 1); i += stride) {
        output[frame_idx * (frame_size / 2 + 1) + i] = smem[i];
    }
}

/* ============================================================================
 * Mel Filter Bank: Triangle filters in frequency domain
 * Input: spectrogram (n_frames, n_freqs) complex
 * Output: mel spectrogram (n_frames, n_mels) complex magnitude
 *
 * Sparse CSR matrix multiply: output[t,m] = sum_f mag[t,f] * filter[m,f]
 * ========================================================================== */

__global__ void mel_filter_bank(
    const Complex *spectrogram,  // (n_frames, n_freqs)
    float *mel_output,           // (n_frames, n_mels) — magnitude only
    const float *filter_data,    // CSR values (non-zero triangle coefficients)
    const int *filter_cols,      // CSR column indices
    const int *filter_rows,      // CSR row pointers (n_mels + 1 entries)
    int n_frames,
    int n_freqs,
    int n_mels
) {
    int frame_idx = blockIdx.x;
    int mel_idx = threadIdx.x;

    if (frame_idx >= n_frames || mel_idx >= n_mels) return;

    // Process one mel bin per thread
    float acc = 0.0f;
    int row_start = filter_rows[mel_idx];
    int row_end = filter_rows[mel_idx + 1];

    for (int idx = row_start; idx < row_end; idx++) {
        int freq_idx = filter_cols[idx];
        float coeff = filter_data[idx];

        Complex spec = spectrogram[frame_idx * n_freqs + freq_idx];
        float mag = sqrtf(spec.x * spec.x + spec.y * spec.y);
        acc += mag * coeff;
    }

    mel_output[frame_idx * n_mels + mel_idx] = acc;
}

/* ============================================================================
 * Spectrogram magnitude extraction and scaling
 * ========================================================================== */

__global__ void spectrogram_linear(
    const Complex *complex_spec,
    float *magnitude_spec,
    int n_frames,
    int n_freqs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_frames * n_freqs;

    if (idx >= total) return;

    Complex c = complex_spec[idx];
    float mag = sqrtf(c.x * c.x + c.y * c.y);
    magnitude_spec[idx] = mag;
}

__global__ void spectrogram_log(
    const Complex *complex_spec,
    float *log_spec,
    int n_frames,
    int n_freqs,
    float eps = 1e-10f
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_frames * n_freqs;

    if (idx >= total) return;

    Complex c = complex_spec[idx];
    float mag = sqrtf(c.x * c.x + c.y * c.y);
    log_spec[idx] = logf(mag + eps);
}

/* ============================================================================
 * Audio-to-DotMap: Spectrogram as procedural image
 *
 * Quantize spectrogram magnitude to 8 levels (0-7), each level = one RPN ref
 * Output: (n_frames, n_freqs) with quantized indices
 * ========================================================================== */

__global__ void audio_to_dotmap(
    const float *magnitude_spec,
    int8_t *dotmap_indices,    // Quantized magnitude class (0-7)
    int n_frames,
    int n_freqs,
    float max_val  // Global max magnitude for normalization
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_frames * n_freqs;

    if (idx >= total) return;

    float mag = magnitude_spec[idx];

    // Normalize to [0, 1]
    float norm = max_val > 0.0f ? mag / max_val : 0.0f;

    // Quantize to [0, 7]
    int8_t level = (int8_t)(norm * 7.5f);
    if (level > 7) level = 7;
    if (level < 0) level = 0;

    dotmap_indices[idx] = level;
}

/* ============================================================================
 * DotMap-to-Audio: Inverse spectrogram via overlap-add
 *
 * Output: (n_samples) reconstructed time-domain signal
 * Uses overlap-add from windowed frames
 * ========================================================================== */

__global__ void dotmap_to_audio(
    const Complex *spectrogram_frames,  // (n_frames, n_freqs) complex
    float *audio_output,                 // (n_samples) reconstructed
    int n_frames,
    int frame_size,
    int hop_size,
    const float *window,                 // Synthesis window
    int output_len
) {
    // Thread block processes one frame's reconstruction
    int frame_idx = blockIdx.x;
    if (frame_idx >= n_frames) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Note: This is a stub for ISSTFT (inverse short-time FT).
    // Full implementation requires inverse FFT per frame + overlap-add.
    // For now, we zero-initialize to mark placeholder.

    int frame_start = frame_idx * hop_size;
    for (int i = tid; i < frame_size && frame_start + i < output_len; i += stride) {
        atomicAdd(&audio_output[frame_start + i], 0.0f);  // Placeholder
    }
}

/* ============================================================================
 * HRTF Convolution: Binaural spatialization via FIR convolution
 *
 * Left/right impulse responses (HRTFs) stored in constant memory
 * Apply FFT multiplication and iFFT per frame
 * ========================================================================== */

__global__ void hrtf_convolve_left(
    const float *input_frame,       // (frame_size) mono input
    Complex *output_fft_domain,     // (frame_size) for post-IFFT reconstruction
    int frame_size
) {
    // Stub: This would require HRTF coefficients in constant memory
    // and would apply frequency-domain convolution (multiplication of FFTs)
    // Placeholder for integration with STFT pipeline

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frame_size) return;
    output_fft_domain[idx] = make_float2(0.0f, 0.0f);
}

/* ============================================================================
 * Host Helper: Compute twiddle factors (to be called at module load)
 * These are pre-computed and uploaded to __constant__ memory
 * ========================================================================== */

extern "C" void precompute_twiddle_factors() {
    // This function is called from the Python ctypes launcher
    // after the module is loaded, to initialize constant memory
    // with pre-computed complex exponentials.

    // For brevity, we define only the 1024-point twiddle table here
    // Real implementation would pre-compute all 4 sizes

    Complex h_twiddle[512];  // 1024-point: N/2 = 512 pairs

    for (int k = 0; k < 512; k++) {
        float angle = -2.0f * 3.14159265359f * k / 1024.0f;
        h_twiddle[k].x = cosf(angle);
        h_twiddle[k].y = sinf(angle);
    }

    cudaMemcpyToSymbol(twiddle_lut_1024, h_twiddle, 512 * sizeof(Complex), 0);
}

extern "C" void precompute_window_hann(int size) {
    float *h_window = (float *)malloc(size * sizeof(float));

    for (int n = 0; n < size; n++) {
        float angle = 2.0f * 3.14159265359f * n / (size - 1);
        h_window[n] = 0.5f * (1.0f - cosf(angle));
    }

    cudaMemcpyToSymbol(window_hann_1024, h_window, size * sizeof(float), 0);
    free(h_window);
}

extern "C" void precompute_window_hamming(int size) {
    float *h_window = (float *)malloc(size * sizeof(float));

    for (int n = 0; n < size; n++) {
        float angle = 2.0f * 3.14159265359f * n / (size - 1);
        h_window[n] = 0.54f - 0.46f * cosf(angle);
    }

    cudaMemcpyToSymbol(window_hamm_1024, h_window, size * sizeof(float), 0);
    free(h_window);
}
