/**
 * K3D JPEG Huffman decode — opcodes 0x24B HUFF_ENCODE_RUN / 0x24C HUFF_DECODE_RUN.
 *
 * Honest tradeoff (Klein/Shapira DCC'13 lists the alternatives):
 *   - Huffman decode is inherently serial per bitstream. We segment by JPEG
 *     restart markers (RST0-7) and launch one warp per segment. CPU pre-scans
 *     RST offsets — microseconds for a 256×256 image. Each segment decodes in
 *     a single thread inside the warp (other threads help with coalesced
 *     output stores in a v2 follow-up).
 *   - Speculative parallel decode is 10-20× faster but non-deterministic in
 *     intermediate state. Rejected: sovereignty requires determinism.
 *
 * Author: two-level lookup tables (2 × 256-entry) go in __constant__ memory
 * (2 KiB per table, 4 KiB total, well under the 64 KiB cap). Caller builds
 * the LUTs on CPU from the JPEG DHT tables and cuMemcpyToSymbol's them in.
 */

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" {

struct HuffLUT {
    uint16_t code;   // canonical code (MSB-aligned in a 16-bit register)
    uint8_t  len;    // 1..16 — bit count consumed
    uint8_t  sym;    // decoded symbol byte
};

__constant__ HuffLUT c_dc_lut[256];
__constant__ HuffLUT c_ac_lut[256];

// 0x24C HUFF_DECODE_RUN — one segment per block; thread 0 decodes serially.
// grid=(n_segments), block=(32).
__global__ void huff_decode_run_kernel(
    const uint8_t* __restrict__ bitstream,          // packed scan data
    const int*     __restrict__ segment_offsets,    // [n_segments]
    const int*     __restrict__ segment_lengths,    // [n_segments]
    int16_t*       __restrict__ out_coeffs,          // [n_segments*blocks_per_seg*64]
    int blocks_per_seg,
    int n_segments)
{
    int seg = blockIdx.x;
    if (threadIdx.x != 0 || seg >= n_segments) return;

    const uint8_t* p   = bitstream + segment_offsets[seg];
    const uint8_t* end = p + segment_lengths[seg];
    uint32_t buf = 0;
    int nbits = 0;
    int16_t prev_dc = 0;

    for (int b = 0; b < blocks_per_seg; ++b) {
        int16_t block[64];
        #pragma unroll
        for (int i = 0; i < 64; ++i) block[i] = 0;

        // --- DC coefficient ---
        while (nbits < 16 && p < end) {
            uint8_t byte = *p++;
            // JPEG byte-stuffing: 0xFF00 means literal 0xFF.
            if (byte == 0xFF && p < end && *p == 0x00) p++;
            buf = (buf << 8) | byte;
            nbits += 8;
        }
        int idx = (buf >> (nbits - 8)) & 0xFF;
        HuffLUT h = c_dc_lut[idx];
        if (h.len == 0) break;   // malformed — bail safely
        nbits -= h.len;
        int dc_bits = h.sym;
        int dc_val = 0;
        if (dc_bits > 0) {
            dc_val = (buf >> (nbits - dc_bits)) & ((1 << dc_bits) - 1);
            nbits -= dc_bits;
            if (dc_val < (1 << (dc_bits - 1))) dc_val -= (1 << dc_bits) - 1;
        }
        prev_dc = (int16_t)(prev_dc + dc_val);
        block[0] = prev_dc;

        // --- AC coefficients (zigzag index 1..63 with run-length) ---
        int k = 1;
        while (k < 64) {
            while (nbits < 16 && p < end) {
                uint8_t byte = *p++;
                if (byte == 0xFF && p < end && *p == 0x00) p++;
                buf = (buf << 8) | byte;
                nbits += 8;
            }
            int aidx = (buf >> (nbits - 8)) & 0xFF;
            HuffLUT ah = c_ac_lut[aidx];
            if (ah.len == 0) break;
            nbits -= ah.len;
            int run  = ah.sym >> 4;
            int size = ah.sym & 0x0F;
            if (run == 0 && size == 0) break;   // EOB
            if (size == 0 && run == 15) {
                k += 16; continue;               // ZRL (zero run 16)
            }
            k += run;
            if (k >= 64) break;
            int ac_val = (buf >> (nbits - size)) & ((1 << size) - 1);
            nbits -= size;
            if (ac_val < (1 << (size - 1))) ac_val -= (1 << size) - 1;
            block[k++] = (int16_t)ac_val;
        }

        int out_base = (seg * blocks_per_seg + b) * 64;
        #pragma unroll
        for (int i = 0; i < 64; ++i) out_coeffs[out_base + i] = block[i];
    }
}

// 0x24B HUFF_ENCODE_RUN — forward encode is fully parallel per block because
// each block writes into a precomputed byte-offset region. CPU walks blocks
// once to compute the per-block bit budget (tight), then this kernel writes.
// Here we expose a tiny per-block serial encoder; throughput is limited by
// the CPU pre-scan, which is fine for offline emission.
__global__ void huff_encode_run_kernel(
    const int16_t* __restrict__ zz_coeffs,       // [N,64]
    const uint16_t* __restrict__ dc_code,         // [256] code bits
    const uint8_t*  __restrict__ dc_len,          // [256] bit-count
    const uint16_t* __restrict__ ac_code,         // [256]
    const uint8_t*  __restrict__ ac_len,          // [256]
    uint8_t*       __restrict__ out_bytes,        // [total_bytes]
    const int*     __restrict__ block_byte_off,   // [N+1] prefix-summed
    int n_blocks)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;

    // Bit writer state (per-block local).
    uint64_t bitbuf = 0;
    int bitlen = 0;
    int boff = block_byte_off[b];
    int bcap = block_byte_off[b + 1] - boff;
    int written = 0;

    auto emit_bits = [&] (uint32_t code, int nbits) {
        bitbuf = (bitbuf << nbits) | (code & ((1u << nbits) - 1u));
        bitlen += nbits;
        while (bitlen >= 8 && written < bcap) {
            uint8_t byte = (uint8_t)((bitbuf >> (bitlen - 8)) & 0xFFu);
            out_bytes[boff + written++] = byte;
            if (byte == 0xFF && written < bcap) {
                out_bytes[boff + written++] = 0x00;   // byte-stuff per JPEG
            }
            bitlen -= 8;
        }
    };

    int dc = zz_coeffs[b * 64 + 0];
    int mag = dc >= 0 ? dc : -dc;
    int size = 0; int m = mag;
    while (m) { ++size; m >>= 1; }
    emit_bits(dc_code[size], dc_len[size]);
    if (size > 0) {
        int v = dc >= 0 ? dc : (dc + (1 << size) - 1);
        emit_bits((uint32_t)v, size);
    }

    int run = 0;
    for (int k = 1; k < 64; ++k) {
        int v = zz_coeffs[b * 64 + k];
        if (v == 0) { ++run; continue; }
        while (run >= 16) { emit_bits(ac_code[0xF0], ac_len[0xF0]); run -= 16; }
        int amag = v >= 0 ? v : -v;
        int sz = 0; int mm = amag;
        while (mm) { ++sz; mm >>= 1; }
        int rs = (run << 4) | sz;
        emit_bits(ac_code[rs], ac_len[rs]);
        int av = v >= 0 ? v : (v + (1 << sz) - 1);
        emit_bits((uint32_t)av, sz);
        run = 0;
    }
    if (run > 0) emit_bits(ac_code[0x00], ac_len[0x00]);   // EOB

    // Flush remaining bits padded with 1s per JPEG.
    if (bitlen > 0 && written < bcap) {
        uint8_t byte = (uint8_t)((bitbuf << (8 - bitlen)) | ((1u << (8 - bitlen)) - 1u));
        out_bytes[boff + written++] = byte;
    }
}

}  // extern "C"
