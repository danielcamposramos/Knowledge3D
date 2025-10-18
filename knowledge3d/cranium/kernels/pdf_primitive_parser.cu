/*
 * PDF Primitive Parser - Phase C1 stub implementation.
 *
 * The kernel simulates two primitives so the Python bridge can exercise the
 * ingestion flow without relying on a full PDF bytecode interpreter yet.
 */

#include <cuda_runtime.h>

#define OBJ_TYPE_TEXT  1.0f
#define OBJ_TYPE_IMAGE 2.0f
#define OBJ_STRIDE     8

extern "C" __global__ void pdf_primitive_parser(
    float* output_objects,
    const char* pdf_buffer,
    int buffer_size,
    int page_num,
    int max_objects,
    int* metadata
) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx != 0) {
        return;
    }

    // Basic metadata
    metadata[0] = 2;   // object_count
    metadata[1] = 0;   // processing_time_us (placeholder)
    metadata[2] = 0;   // is_scanned = false
    metadata[3] = 0;   // reserved

    if (max_objects < 2) {
        metadata[0] = 0;
        return;
    }

    // Dummy text object
    output_objects[0 * OBJ_STRIDE + 0] = 100.0f;
    output_objects[0 * OBJ_STRIDE + 1] = 700.0f;
    output_objects[0 * OBJ_STRIDE + 2] = 420.0f;
    output_objects[0 * OBJ_STRIDE + 3] = 24.0f;
    output_objects[0 * OBJ_STRIDE + 4] = OBJ_TYPE_TEXT;
    output_objects[0 * OBJ_STRIDE + 5] = 0.0f;
    output_objects[0 * OBJ_STRIDE + 6] = 11.0f;
    output_objects[0 * OBJ_STRIDE + 7] = 0.9f;

    // Dummy image object
    output_objects[1 * OBJ_STRIDE + 0] = 150.0f;
    output_objects[1 * OBJ_STRIDE + 1] = 500.0f;
    output_objects[1 * OBJ_STRIDE + 2] = 200.0f;
    output_objects[1 * OBJ_STRIDE + 3] = 150.0f;
    output_objects[1 * OBJ_STRIDE + 4] = OBJ_TYPE_IMAGE;
    output_objects[1 * OBJ_STRIDE + 5] = 0.0f;
    output_objects[1 * OBJ_STRIDE + 6] = 12345.0f;
    output_objects[1 * OBJ_STRIDE + 7] = 0.8f;
}
