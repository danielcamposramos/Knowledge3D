// Minimal RPN bytecode executor on GPU for procedural drawing bytecodes.
// Decodes MOVE/LINE + optional SET_COLOR/SET_LINE_WIDTH into segment buffer.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void execute_rpn_bytecode(
    const uint8_t* __restrict__ bytecode,
    uint32_t bytecode_len,
    float* __restrict__ segments,   // stride 9: x0,y0,x1,y1,r,g,b,a,w
    uint32_t* __restrict__ seg_count_out,
    uint32_t max_segments
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float curx = 0.0f, cury = 0.0f;
    float startx = 0.0f, starty = 0.0f;
    float r = 1.0f, g = 1.0f, b = 1.0f, a = 1.0f;
    float width = 1.0f;
    uint32_t seg_count = 0;
    uint32_t idx = 0;
    while (idx + 4 <= bytecode_len) {
        // bytecode is little-endian u32 opcodes and float operands
        uint32_t opcode = *((const uint32_t*)(bytecode + idx));
        idx += 4;
        if (opcode == 0x64) { // MOVE
            if (idx + 8 > bytecode_len) break;
            curx = *((const float*)(bytecode + idx));
            cury = *((const float*)(bytecode + idx + 4));
            startx = curx;
            starty = cury;
            idx += 8;
        } else if (opcode == 0x65) { // LINE
            if (idx + 8 > bytecode_len) break;
            float x1 = *((const float*)(bytecode + idx));
            float y1 = *((const float*)(bytecode + idx + 4));
            idx += 8;
            if (seg_count < max_segments) {
                uint32_t off = seg_count * 9;
                segments[off + 0] = curx;
                segments[off + 1] = cury;
                segments[off + 2] = x1;
                segments[off + 3] = y1;
                segments[off + 4] = r;
                segments[off + 5] = g;
                segments[off + 6] = b;
                segments[off + 7] = a;
                segments[off + 8] = width;
                seg_count += 1;
            }
            curx = x1;
            cury = y1;
        } else if (opcode == 0x75 || opcode == 0x76) { // SET_COLOR / SET_FILL_COLOR
            if (idx + 16 > bytecode_len) break;
            r = *((const float*)(bytecode + idx));
            g = *((const float*)(bytecode + idx + 4));
            b = *((const float*)(bytecode + idx + 8));
            a = *((const float*)(bytecode + idx + 12));
            idx += 16;
        } else if (opcode == 0x77) { // STROKE_WIDTH
            if (idx + 4 > bytecode_len) break;
            width = *((const float*)(bytecode + idx));
            idx += 4;
        } else if (opcode == 0x69 || opcode == 0x6A || opcode == 0x6B) { // CLOSE/STROKE/FILL
            // no-op for geometry emission
        } else {
            // unknown opcode; stop to avoid mis-read
            break;
        }
    }
    *seg_count_out = seg_count;
}
