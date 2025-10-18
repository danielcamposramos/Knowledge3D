/*
 * Glyph Resonator - Phase C1 stub.
 *
 * Placeholder OCR kernel that currently writes zero objects. The full OCR
 * resonance matching lands in Phase C3.
 */

#include <cuda_runtime.h>

extern "C" __global__ void glyph_resonator(
    float* output_text_objects,
    const unsigned char* image_data,
    int image_width,
    int image_height,
    const float* learned_glyphs,
    int glyph_count,
    int max_chars
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        // Zero output to signal "no OCR results".
        for (int i = 0; i < max_chars * 8; ++i) {
            output_text_objects[i] = 0.0f;
        }
    }
    (void)image_data;
    (void)image_width;
    (void)image_height;
    (void)learned_glyphs;
    (void)glyph_count;
}
