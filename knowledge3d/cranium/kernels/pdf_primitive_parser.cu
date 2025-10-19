/*
 * PDF Primitive Parser - Phase C2 bootstrap implementation.
 *
 * Scans the PDF byte buffer for BT ... ET blocks and records literal text
 * strings encountered in Tj operators. This lightweight interpreter provides
 * offsets back into the original byte stream so the Python bridge can decode
 * content on the host while we iterate towards a full GPU-native parser.
 */

#include <cuda_runtime.h>

#define OBJ_TYPE_TEXT 1.0f
#define OBJ_STRIDE    8

__device__ inline bool is_whitespace(char c) {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\f';
}

extern "C" __global__ void pdf_primitive_parser(
    float* output_objects,
    const char* pdf_buffer,
    int buffer_size,
    int page_num,
    int max_objects,
    int* metadata
) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id != 0) {
        return;
    }

    (void)page_num;  // Phase C2.1 treats the buffer as a flat stream.

    int object_count = 0;
    metadata[0] = 0;
    metadata[1] = 0;
    metadata[2] = 0;
    metadata[3] = 0;

    for (int i = 0; i < buffer_size - 2 && object_count < max_objects; ++i) {
        if (pdf_buffer[i] == 'B' && pdf_buffer[i + 1] == 'T' && is_whitespace(pdf_buffer[i + 2])) {
            int cursor = i + 2;

            while (cursor < buffer_size - 1 && object_count < max_objects) {
                if (pdf_buffer[cursor] == '(') {
                    int text_start = cursor + 1;
                    int text_length = 0;
                    bool escape = false;

                    while (text_start + text_length < buffer_size) {
                        char ch = pdf_buffer[text_start + text_length];
                        if (ch == '\\' && !escape) {
                            escape = true;
                            ++text_length;
                            continue;
                        }
                        if (ch == ')' && !escape) {
                            break;
                        }
                        escape = false;
                        ++text_length;
                    }

                    if (text_length > 0 && object_count < max_objects) {
                        const int base = object_count * OBJ_STRIDE;
                        output_objects[base + 0] = 0.0f;  // x placeholder
                        output_objects[base + 1] = 0.0f;  // y placeholder
                        output_objects[base + 2] = 0.0f;  // width placeholder
                        output_objects[base + 3] = 0.0f;  // height placeholder
                        output_objects[base + 4] = OBJ_TYPE_TEXT;
                        output_objects[base + 5] = static_cast<float>(text_start);
                        output_objects[base + 6] = static_cast<float>(text_length);
                        output_objects[base + 7] = 0.8f;  // default importance
                        ++object_count;
                    }

                    cursor = text_start + text_length;
                    continue;
                }

                if (pdf_buffer[cursor] == 'E' && pdf_buffer[cursor + 1] == 'T') {
                    break;
                }

                ++cursor;
            }

            i = cursor;
        }
    }

    metadata[0] = object_count;
    metadata[2] = (object_count < 2) ? 1 : 0;
}

