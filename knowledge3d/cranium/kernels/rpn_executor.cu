// Minimal RPN bytecode executor on GPU for procedural drawing bytecodes.
// Decodes MOVE/LINE/QUAD/CUBIC/ARC + style ops into segment buffer.

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

// Opcode constants matching ProceduralDrawingBridge bytecode compiler (0x64+)
#define OP_MOVE   0x64
#define OP_LINE   0x65
#define OP_QUAD   0x66
#define OP_CUBIC  0x67
#define OP_ARC    0x68
#define OP_CLOSE  0x69
#define OP_STROKE 0x6A
#define OP_FILL   0x6B
#define OP_SET_COLOR 0x75
#define OP_SET_FILL_COLOR 0x76
#define OP_SET_LINE_WIDTH 0x77

__device__ inline float apply_ternary_stroke_width(int8_t ternary_weight, float base_width) {
    if (ternary_weight == -1) return base_width * 0.7f;  // Thin
    if (ternary_weight == 1) return base_width * 1.5f;   // Bold
    return base_width;
}

__device__ inline void emit_segment(
    float* segments,
    uint32_t off,
    float x0, float y0,
    float x1, float y1,
    float r, float g, float b, float a,
    float width
) {
    segments[off + 0] = x0;
    segments[off + 1] = y0;
    segments[off + 2] = x1;
    segments[off + 3] = y1;
    segments[off + 4] = r;
    segments[off + 5] = g;
    segments[off + 6] = b;
    segments[off + 7] = a;
    segments[off + 8] = width;
}

extern "C" __global__ void execute_rpn_bytecode(
    const uint8_t* __restrict__ bytecode,
    uint32_t bytecode_len,
    float* __restrict__ segments,   // stride 9: x0,y0,x1,y1,r,g,b,a,w
    uint32_t* __restrict__ seg_count_out,
    uint32_t max_segments,
    const int8_t* __restrict__ ternary_meta // optional: [0]=weight ternary
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float curx = 0.0f, cury = 0.0f;
    float startx = 0.0f, starty = 0.0f;
    float r = 1.0f, g = 1.0f, b = 1.0f, a = 1.0f;
    float width = 1.0f;
    int8_t tern_weight = 0;
    if (ternary_meta) tern_weight = ternary_meta[0];
    uint32_t seg_count = 0;
    uint32_t idx = 0;
    const int QUAD_SEGMENTS = 16;
    const int CUBIC_SEGMENTS = 16;
    const int ARC_SEGMENTS = 16;

    while (idx + 4 <= bytecode_len) {
        // bytecode is little-endian u32 opcodes and float operands
        uint32_t opcode = *((const uint32_t*)(bytecode + idx));
        idx += 4;
        if (opcode == OP_MOVE) {
            if (idx + 8 > bytecode_len) break;
            curx = *((const float*)(bytecode + idx));
            cury = *((const float*)(bytecode + idx + 4));
            startx = curx;
            starty = cury;
            idx += 8;
        } else if (opcode == OP_LINE) {
            if (idx + 8 > bytecode_len) break;
            float x1 = *((const float*)(bytecode + idx));
            float y1 = *((const float*)(bytecode + idx + 4));
            idx += 8;
            if (seg_count < max_segments) {
                uint32_t off = seg_count * 9;
                emit_segment(segments, off, curx, cury, x1, y1, r, g, b, a, apply_ternary_stroke_width(tern_weight, width));
                seg_count += 1;
            }
            curx = x1;
            cury = y1;
        } else if (opcode == OP_QUAD) {
            if (idx + 16 > bytecode_len) break;
            float cx = *((const float*)(bytecode + idx + 0));
            float cy = *((const float*)(bytecode + idx + 4));
            float x1 = *((const float*)(bytecode + idx + 8));
            float y1 = *((const float*)(bytecode + idx + 12));
            idx += 16;
            float x0 = curx, y0 = cury;
            for (int i = 1; i <= QUAD_SEGMENTS; ++i) {
                float t = (float)i / QUAD_SEGMENTS;
                float mt = 1.0f - t;
                float x = mt*mt*x0 + 2*mt*t*cx + t*t*x1;
                float y = mt*mt*y0 + 2*mt*t*cy + t*t*y1;
                if (seg_count < max_segments) {
                    uint32_t off = seg_count * 9;
                    emit_segment(segments, off, curx, cury, x, y, r, g, b, a, apply_ternary_stroke_width(tern_weight, width));
                    seg_count += 1;
                }
                curx = x;
                cury = y;
            }
        } else if (opcode == OP_CUBIC) {
            if (idx + 32 > bytecode_len) break;
            float cx1 = *((const float*)(bytecode + idx + 0));
            float cy1 = *((const float*)(bytecode + idx + 4));
            float cx2 = *((const float*)(bytecode + idx + 8));
            float cy2 = *((const float*)(bytecode + idx + 12));
            float x1  = *((const float*)(bytecode + idx + 16));
            float y1  = *((const float*)(bytecode + idx + 20));
            idx += 32;
            float x0 = curx, y0 = cury;
            for (int i = 1; i <= CUBIC_SEGMENTS; ++i) {
                float t = (float)i / CUBIC_SEGMENTS;
                float mt = 1.0f - t;
                float x = mt*mt*mt*x0 + 3*mt*mt*t*cx1 + 3*mt*t*t*cx2 + t*t*t*x1;
                float y = mt*mt*mt*y0 + 3*mt*mt*t*cy1 + 3*mt*t*t*cy2 + t*t*t*y1;
                if (seg_count < max_segments) {
                    uint32_t off = seg_count * 9;
                    emit_segment(segments, off, curx, cury, x, y, r, g, b, a, apply_ternary_stroke_width(tern_weight, width));
                    seg_count += 1;
                }
                curx = x;
                cury = y;
            }
        } else if (opcode == OP_ARC) {
            // operands: rx ry start sweep cx cy (matches host parser order)
            if (idx + 24 > bytecode_len) break;
            float rx = *((const float*)(bytecode + idx + 0));
            float ry = *((const float*)(bytecode + idx + 4));
            float start = *((const float*)(bytecode + idx + 8));
            float sweep = *((const float*)(bytecode + idx + 12));
            float cx = *((const float*)(bytecode + idx + 16));
            float cy = *((const float*)(bytecode + idx + 20));
            idx += 24;
            for (int i = 1; i <= ARC_SEGMENTS; ++i) {
                float t = (float)i / ARC_SEGMENTS;
                float ang = start + sweep * t;
                float x = cx + rx * cosf(ang);
                float y = cy + ry * sinf(ang);
                if (seg_count < max_segments) {
                    uint32_t off = seg_count * 9;
                    emit_segment(segments, off, curx, cury, x, y, r, g, b, a, apply_ternary_stroke_width(tern_weight, width));
                    seg_count += 1;
                }
                curx = x;
                cury = y;
            }
        } else if (opcode == OP_CLOSE) {
            if (seg_count < max_segments) {
                uint32_t off = seg_count * 9;
                emit_segment(segments, off, curx, cury, startx, starty, r, g, b, a, apply_ternary_stroke_width(tern_weight, width));
                seg_count += 1;
            }
            curx = startx;
            cury = starty;
        } else if (opcode == OP_SET_COLOR || opcode == OP_SET_FILL_COLOR) {
            if (idx + 16 > bytecode_len) break;
            r = *((const float*)(bytecode + idx));
            g = *((const float*)(bytecode + idx + 4));
            b = *((const float*)(bytecode + idx + 8));
            a = *((const float*)(bytecode + idx + 12));
            idx += 16;
        } else if (opcode == OP_SET_LINE_WIDTH) {
            if (idx + 4 > bytecode_len) break;
            float w = *((const float*)(bytecode + idx));
            width = apply_ternary_stroke_width(tern_weight, w);
            idx += 4;
        } else if (opcode == OP_STROKE || opcode == OP_FILL) {
            // no-op for geometry emission
        } else {
            // unknown opcode; stop to avoid mis-read
            break;
        }
    }
    *seg_count_out = seg_count;
}
