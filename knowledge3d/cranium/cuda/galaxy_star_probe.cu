#include "device_functions.cuh"

extern "C" __global__ void galaxy_star_probe_records(
    const unsigned char* __restrict__ star_table,
    const unsigned int* __restrict__ star_indices,
    unsigned char* __restrict__ out_records,
    unsigned int probe_count
) {
    unsigned int probe_idx = blockIdx.x;
    unsigned int lane = threadIdx.x;
    if (probe_idx >= probe_count) {
        return;
    }

    const unsigned int star_index = star_indices[probe_idx];
    const unsigned char* src = star_table + (static_cast<unsigned long long>(star_index) * GALAXY_STAR_RECORD_BYTES);
    unsigned char* dst = out_records + (static_cast<unsigned long long>(probe_idx) * GALAXY_STAR_RECORD_BYTES);

    for (unsigned int offset = lane; offset < GALAXY_STAR_RECORD_BYTES; offset += blockDim.x) {
        dst[offset] = src[offset];
    }
}
