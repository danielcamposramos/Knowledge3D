// Sovereign ARC grid operations (rotate/flip/translate/recolor) on GPU.
//
// Input and output grids are uint8 color indices (0-255).
// Supported ops:
//   0: rotate 90° CW
//   1: rotate 180°
//   2: rotate 270° CW
//   3: flip horizontally
//   4: flip vertically
//   5: translate (p1=dx, p2=dy, fill=0)
//   6: recolor (p1=src, p2=dst)

#include <cuda_runtime.h>

extern "C" __global__
void arc_grid_op(const unsigned char* input,
                 unsigned char* output,
                 int src_w,
                 int src_h,
                 int dst_w,
                 int dst_h,
                 int op,
                 int p1,
                 int p2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) {
        return;
    }

    int src_x = x;
    int src_y = y;

    switch (op) {
        case 0: { // rotate 90 CW
            // dst_w = src_h, dst_h = src_w
            src_x = y;
            src_y = src_h - 1 - x;
            break;
        }
        case 1: { // rotate 180
            src_x = src_w - 1 - x;
            src_y = src_h - 1 - y;
            break;
        }
        case 2: { // rotate 270 CW (90 CCW)
            // dst_w = src_h, dst_h = src_w
            src_x = src_w - 1 - y;
            src_y = x;
            break;
        }
        case 3: { // flip horizontal
            src_x = src_w - 1 - x;
            src_y = y;
            break;
        }
        case 4: { // flip vertical
            src_x = x;
            src_y = src_h - 1 - y;
            break;
        }
        case 5: { // translate (dx=p1, dy=p2)
            src_x = x - p1;
            src_y = y - p2;
            break;
        }
        case 6: { // recolor (src=p1, dst=p2)
            int idx = y * src_w + x;
            unsigned char v = input[idx];
            output[y * dst_w + x] = (v == (unsigned char)p1) ? (unsigned char)p2 : v;
            return;
        }
        default:
            // Unsupported op; write zero
            output[y * dst_w + x] = 0;
            return;
    }

    // Bounds check for operations that access source differently
    if (src_x < 0 || src_x >= src_w || src_y < 0 || src_y >= src_h) {
        output[y * dst_w + x] = 0;
        return;
    }

    int src_idx = src_y * src_w + src_x;
    int dst_idx = y * dst_w + x;
    output[dst_idx] = input[src_idx];
}
