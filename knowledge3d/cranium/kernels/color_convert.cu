/**
 * Sovereign color space conversions — GPU-native, no external libs.
 *
 * Supported spaces:
 *   sRGB ↔ Linear RGB ↔ CIE XYZ ↔ CIE Lab
 *   sRGB ↔ CMYK (GCR)
 *   sRGB ↔ HSV
 *
 * Based on IEC 61966-2-1 (sRGB) and CIE standards (public domain math).
 */

extern "C" {

__device__ __forceinline__ float srgb_to_linear(float v) {
    return (v <= 0.04045f) ? (v / 12.92f) : powf((v + 0.055f) / 1.055f, 2.4f);
}

__device__ __forceinline__ float linear_to_srgb(float v) {
    return (v <= 0.0031308f) ? (12.92f * v) : (1.055f * powf(v, 1.0f / 2.4f) - 0.055f);
}

__device__ __forceinline__ float lab_f(float t) {
    const float delta = 6.0f / 29.0f;
    if (t > delta * delta * delta) {
        return cbrtf(t);
    }
    return t / (3.0f * delta * delta) + 4.0f / 29.0f;
}

__device__ __forceinline__ float lab_f_inv(float t) {
    const float delta = 6.0f / 29.0f;
    if (t > delta) {
        return t * t * t;
    }
    return 3.0f * delta * delta * (t - 4.0f / 29.0f);
}

// D65 white point reference
__device__ __constant__ float D65_X = 0.95047f;
__device__ __constant__ float D65_Y = 1.00000f;
__device__ __constant__ float D65_Z = 1.08883f;

// sRGB → XYZ matrix (D65)
__device__ __constant__ float SRGB_TO_XYZ[9] = {
    0.4124564f, 0.3575761f, 0.1804375f,
    0.2126729f, 0.7151522f, 0.0721750f,
    0.0193339f, 0.1191920f, 0.9503041f
};

// XYZ → sRGB matrix (D65)
__device__ __constant__ float XYZ_TO_SRGB[9] = {
     3.2404542f, -1.5371385f, -0.4985314f,
    -0.9692660f,  1.8760108f,  0.0415560f,
     0.0556434f, -0.2040259f,  1.0572252f
};

// Batch RGB → Lab conversion (rgb in [0,1])
__global__ void rgb_to_lab_batch(const float* rgb, float* lab, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float r = srgb_to_linear(rgb[i * 3 + 0]);
    float g = srgb_to_linear(rgb[i * 3 + 1]);
    float b = srgb_to_linear(rgb[i * 3 + 2]);

    // RGB → XYZ
    float x = SRGB_TO_XYZ[0] * r + SRGB_TO_XYZ[1] * g + SRGB_TO_XYZ[2] * b;
    float y = SRGB_TO_XYZ[3] * r + SRGB_TO_XYZ[4] * g + SRGB_TO_XYZ[5] * b;
    float z = SRGB_TO_XYZ[6] * r + SRGB_TO_XYZ[7] * g + SRGB_TO_XYZ[8] * b;

    // XYZ → Lab
    float fx = lab_f(x / D65_X);
    float fy = lab_f(y / D65_Y);
    float fz = lab_f(z / D65_Z);

    lab[i * 3 + 0] = 116.0f * fy - 16.0f;  // L*
    lab[i * 3 + 1] = 500.0f * (fx - fy);   // a*
    lab[i * 3 + 2] = 200.0f * (fy - fz);   // b*
}

// Batch Lab → RGB (outputs sRGB [0,1])
__global__ void lab_to_rgb_batch(const float* lab, float* rgb, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float L = lab[i * 3 + 0];
    float a = lab[i * 3 + 1];
    float b_ = lab[i * 3 + 2];

    float fy = (L + 16.0f) / 116.0f;
    float fx = a / 500.0f + fy;
    float fz = fy - b_ / 200.0f;

    float x = D65_X * lab_f_inv(fx);
    float y = D65_Y * lab_f_inv(fy);
    float z = D65_Z * lab_f_inv(fz);

    float r = XYZ_TO_SRGB[0] * x + XYZ_TO_SRGB[1] * y + XYZ_TO_SRGB[2] * z;
    float g = XYZ_TO_SRGB[3] * x + XYZ_TO_SRGB[4] * y + XYZ_TO_SRGB[5] * z;
    float b = XYZ_TO_SRGB[6] * x + XYZ_TO_SRGB[7] * y + XYZ_TO_SRGB[8] * z;

    rgb[i * 3 + 0] = fmaxf(0.0f, fminf(1.0f, linear_to_srgb(r)));
    rgb[i * 3 + 1] = fmaxf(0.0f, fminf(1.0f, linear_to_srgb(g)));
    rgb[i * 3 + 2] = fmaxf(0.0f, fminf(1.0f, linear_to_srgb(b)));
}

// RGB → CMYK with Gray Component Replacement (gcr_level 0..1)
__global__ void rgb_to_cmyk_batch(const float* rgb, float* cmyk, float gcr_level, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float r = rgb[i * 3 + 0];
    float g = rgb[i * 3 + 1];
    float b = rgb[i * 3 + 2];

    float c = 1.0f - r;
    float m = 1.0f - g;
    float y = 1.0f - b;

    float k = fminf(c, fminf(m, y)) * gcr_level;
    if (k > 0.0f && k < 1.0f) {
        float denom = 1.0f - k;
        c = (c - k) / denom;
        m = (m - k) / denom;
        y = (y - k) / denom;
    }

    cmyk[i * 4 + 0] = fmaxf(0.0f, fminf(1.0f, c));
    cmyk[i * 4 + 1] = fmaxf(0.0f, fminf(1.0f, m));
    cmyk[i * 4 + 2] = fmaxf(0.0f, fminf(1.0f, y));
    cmyk[i * 4 + 3] = fmaxf(0.0f, fminf(1.0f, k));
}

// RGB → HSV batch
__global__ void rgb_to_hsv_batch(const float* rgb, float* hsv, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float r = rgb[i * 3 + 0];
    float g = rgb[i * 3 + 1];
    float b = rgb[i * 3 + 2];

    float max_val = fmaxf(r, fmaxf(g, b));
    float min_val = fminf(r, fminf(g, b));
    float delta = max_val - min_val;

    float v = max_val;
    float s = (max_val > 0.0f) ? (delta / max_val) : 0.0f;

    float h = 0.0f;
    if (delta > 0.0f) {
        if (max_val == r) {
            h = 60.0f * fmodf((g - b) / delta + 6.0f, 6.0f);
        } else if (max_val == g) {
            h = 60.0f * ((b - r) / delta + 2.0f);
        } else {
            h = 60.0f * ((r - g) / delta + 4.0f);
        }
    }

    hsv[i * 3 + 0] = h;
    hsv[i * 3 + 1] = s;
    hsv[i * 3 + 2] = v;
}

// Delta E (CIE76) for perceptual color difference
__global__ void delta_e_batch(const float* lab1, const float* lab2, float* delta_e, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float dL = lab1[i * 3 + 0] - lab2[i * 3 + 0];
    float da = lab1[i * 3 + 1] - lab2[i * 3 + 1];
    float db = lab1[i * 3 + 2] - lab2[i * 3 + 2];
    delta_e[i] = sqrtf(dL * dL + da * da + db * db);
}

}  // extern "C"
