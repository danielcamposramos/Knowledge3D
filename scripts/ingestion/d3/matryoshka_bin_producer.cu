/*
 * Sovereign standalone producer for matryoshka_embeddings.bin.
 *
 * Reads merged_stars.jsonl, extracts the first non-empty RPN-family program
 * from every Word/Character-galaxy star that lacks a matryoshka payload,
 * derives ternary basis + token ids on GPU via procedural_basis.cu, runs
 * the matryoshka_accumulator kernel, and writes the .bin directly.
 *
 * Zero Python. Zero numpy/cupy/scipy/sympy/torch. STL only for containers
 * and file I/O (no math libraries). All arithmetic on GPU.
 *
 * Build:
 *   bash scripts/ingestion/d3/build_matryoshka_producer.sh
 *
 * Run:
 *   scripts/ingestion/d3/build/matryoshka_bin_producer \
 *     --input  scripts/ingestion/staging/D3_dedup/merged_stars.jsonl \
 *     --output scripts/ingestion/staging/D3_dedup/matryoshka_embeddings.bin
 *
 * Flags:
 *   --verbose           prints vocab/row counts to stderr
 *   --force-regenerate  ignores the `"matryoshka":{...}` skip-filter, so rows
 *                       that already carry a payload (post-B4 merged_stars.jsonl)
 *                       are re-embedded. Required for byte-stable cross-check
 *                       against the Python-driven .bin without pre-stripping
 *                       matryoshka objects from the JSONL.
 *
 * Determinism: same input bytes -> same output bytes. Use two runs + sha256
 * for cross-check against the Python-driven path in galaxy_d3.py.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#define MATRYOSHKA_DIM       2048
#define PACKED_TRITS_STRIDE  410

/* Kernels in sibling translation units. */
extern "C" __global__ void procedural_basis_fingerprint(
    const uint8_t*  token_bytes,
    const uint32_t* token_offsets,
    int             vocab_count,
    uint8_t*        basis_packed);

extern "C" __global__ void procedural_token_id(
    const uint8_t*  token_bytes,
    const uint32_t* token_offsets,
    int             vocab_count,
    uint32_t*       token_ids);

extern "C" __global__ void matryoshka_accumulator(
    const uint8_t*  basis,
    int             basis_stride,
    const uint32_t* token_basis_indices,
    const uint32_t* token_ids,
    const uint8_t*  masses,
    int             token_count,
    int8_t*         output);

/* Canonical opcode registry override. Mirrors KNOWN_OPCODE_MAP in
 * scripts/ingestion/d3/galaxy_d3.py so output bytes stay aligned with the
 * Python-driven path for cross-check. If the Python map changes, update
 * both tables together. */
struct CanonicalOpcode { const char* name; uint32_t code; };
static const CanonicalOpcode CANONICAL_OPCODES[] = {
    {"+",0x0A},{"-",0x0B},{"*",0x0C},{"/",0x0D},{"^",0x0E},
    {"sqrt",0x14},{"exp",0x15},{"log",0x16},
    {"sin",0x18},{"cos",0x19},{"tan",0x1A},
    {"asin",0x1B},{"acos",0x1C},{"atan",0x1D},
    {"abs",0x27},{"gt",0x28},{"ceil",0x29},
    {"lt",0x2A},{"floor",0x2B},{"eq",0x2C},
    {"round",0x2D},{"max",0x2E},{"min",0x2F},
    {"dup",0x32},{"swap",0x33},{"drop",0x34},
    {"mod",0x38},{"log2",0x39},{"log10",0x3A},
    {"store",0xB3},{"recall",0xB4},{"gradient",0xB6},
    {"branch",0xB0},{"loop",0xB1},{"next",0xB2},
    {"normalize",0xC1},
    {"TERNARY_QUANT",0x170},{"TERNARY_DEQUANT",0x171},
    {"BATCH_DCT",0x176},{"BATCH_MDCT",0x177},
    {"RESHAPE_TO_BLOCKS",0x178},{"BLOCKS_TO_GRID",0x179},
    {nullptr,0}
};

static int canonical_opcode_for(const std::string& token) {
    for (int i = 0; CANONICAL_OPCODES[i].name; ++i) {
        if (token == CANONICAL_OPCODES[i].name) return (int)CANONICAL_OPCODES[i].code;
    }
    std::string lower = token;
    for (char& c : lower) c = (char)std::tolower((unsigned char)c);
    for (int i = 0; CANONICAL_OPCODES[i].name; ++i) {
        if (lower == CANONICAL_OPCODES[i].name) return (int)CANONICAL_OPCODES[i].code;
    }
    return -1;
}

/* Tiny key-aware JSON string scanner. Finds `"key":"..."` whose opening
 * quote sits at brace-depth 1 (top-level field of the line's object), then
 * reads until an unescaped closing quote. Handles \" and \\ escapes. */
static std::string extract_top_string_field(const std::string& line, const std::string& key) {
    const std::string pattern = "\"" + key + "\":";
    size_t search = 0;
    while (search < line.size()) {
        size_t found = line.find(pattern, search);
        if (found == std::string::npos) return "";
        int depth = 0; bool in_str = false; bool esc = false;
        for (size_t i = 0; i < found; ++i) {
            char c = line[i];
            if (esc) { esc = false; continue; }
            if (c == '\\') { esc = true; continue; }
            if (c == '"') { in_str = !in_str; continue; }
            if (in_str) continue;
            if (c == '{' || c == '[') ++depth;
            if (c == '}' || c == ']') --depth;
        }
        if (depth != 1) { search = found + 1; continue; }
        size_t i = found + pattern.size();
        while (i < line.size() && std::isspace((unsigned char)line[i])) ++i;
        if (i >= line.size() || line[i] != '"') return "";
        ++i;
        std::string out; out.reserve(64);
        bool e = false;
        for (; i < line.size(); ++i) {
            char c = line[i];
            if (e) {
                switch (c) {
                    case '\\': out.push_back('\\'); break;
                    case '"':  out.push_back('"');  break;
                    case 'n':  out.push_back('\n'); break;
                    case 't':  out.push_back('\t'); break;
                    case 'r':  out.push_back('\r'); break;
                    case '/':  out.push_back('/');  break;
                    default:   out.push_back(c);    break;  /* \uXXXX unhandled; RPN tokens are ASCII-dominant */
                }
                e = false;
            } else if (c == '\\') {
                e = true;
            } else if (c == '"') {
                return out;
            } else {
                out.push_back(c);
            }
        }
        return "";
    }
    return "";
}

/* Detects presence of a top-level key whose value is an object or array
 * (used for the "matryoshka":{...} skip-filter). */
static bool has_top_object_key(const std::string& line, const std::string& key) {
    const std::string pattern = "\"" + key + "\":";
    size_t search = 0;
    while (search < line.size()) {
        size_t found = line.find(pattern, search);
        if (found == std::string::npos) return false;
        int depth = 0; bool in_str = false; bool esc = false;
        for (size_t i = 0; i < found; ++i) {
            char c = line[i];
            if (esc) { esc = false; continue; }
            if (c == '\\') { esc = true; continue; }
            if (c == '"') { in_str = !in_str; continue; }
            if (in_str) continue;
            if (c == '{' || c == '[') ++depth;
            if (c == '}' || c == ']') --depth;
        }
        if (depth == 1) return true;
        search = found + 1;
    }
    return false;
}

/* Replicates galaxy_d3._tokenize_for_embedding: × -> *, ÷ -> /, ASCII whitespace split. */
static std::vector<std::string> tokenize_program(const std::string& program) {
    std::string norm; norm.reserve(program.size());
    for (size_t i = 0; i < program.size(); ++i) {
        uint8_t b0 = (uint8_t)program[i];
        if (b0 == 0xC3 && i + 1 < program.size()) {
            uint8_t b1 = (uint8_t)program[i + 1];
            if (b1 == 0x97) { norm.push_back('*'); ++i; continue; }   /* × U+00D7 */
            if (b1 == 0xB7) { norm.push_back('/'); ++i; continue; }   /* ÷ U+00F7 */
        }
        norm.push_back((char)b0);
    }
    std::vector<std::string> tokens;
    std::string current;
    for (char c : norm) {
        if (c == ' '||c == '\t'||c == '\n'||c == '\r'||c == '\f'||c == '\v') {
            if (!current.empty()) { tokens.push_back(std::move(current)); current.clear(); }
        } else {
            current.push_back(c);
        }
    }
    if (!current.empty()) tokens.push_back(std::move(current));
    return tokens;
}

struct PendingRow {
    std::string row_id;
    std::string galaxy;
    std::vector<uint32_t> token_vocab_indices;
};

#define CK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); } } while (0)

int main(int argc, char** argv) {
    const char* input_path  = nullptr;
    const char* output_path = nullptr;
    bool verbose = false;
    bool force_regenerate = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) input_path  = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (std::strcmp(argv[i], "--verbose") == 0) verbose = true;
        else if (std::strcmp(argv[i], "--force-regenerate") == 0) force_regenerate = true;
        else {
            std::fprintf(stderr,
                "usage: %s --input MERGED.jsonl --output EMB.bin "
                "[--verbose] [--force-regenerate]\n",
                argv[0]);
            return 2;
        }
    }
    if (!input_path || !output_path) {
        std::fprintf(stderr, "missing --input or --output\n");
        return 2;
    }

    /* Step 1: stream JSONL, filter rows, build unordered insertion vocab. */
    std::ifstream in(input_path);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", input_path); return 1; }

    std::unordered_map<std::string, uint32_t> insert_index;
    std::vector<std::string> insert_vocab;
    std::vector<PendingRow> pending;
    std::string line;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::string galaxy = extract_top_string_field(line, "galaxy");
        if (galaxy != "Word" && galaxy != "Character") continue;
        if (!force_regenerate && has_top_object_key(line, "matryoshka")) continue;

        static const char* RPN_FIELDS[] = {
            "rpn_program", "meaning_rpn", "visual_rpn",
            "audio_rpn",   "behavior_rpn", nullptr
        };
        std::string program;
        for (int i = 0; RPN_FIELDS[i]; ++i) {
            program = extract_top_string_field(line, RPN_FIELDS[i]);
            if (!program.empty()) break;
        }
        if (program.empty()) continue;

        std::string row_id = extract_top_string_field(line, "id");
        if (row_id.empty()) row_id = extract_top_string_field(line, "star_id");

        PendingRow row;
        row.row_id = std::move(row_id);
        row.galaxy = std::move(galaxy);
        for (auto& tok : tokenize_program(program)) {
            auto it = insert_index.find(tok);
            uint32_t idx;
            if (it == insert_index.end()) {
                idx = (uint32_t)insert_vocab.size();
                insert_index.emplace(tok, idx);
                insert_vocab.push_back(std::move(tok));
            } else {
                idx = it->second;
            }
            row.token_vocab_indices.push_back(idx);
        }
        pending.push_back(std::move(row));
    }
    in.close();

    /* Step 1b: sort vocab alphabetically (matches Python `sorted({...})`). */
    std::vector<std::string> vocab = insert_vocab;
    std::sort(vocab.begin(), vocab.end());
    std::unordered_map<std::string, uint32_t> sorted_index;
    sorted_index.reserve(vocab.size());
    for (uint32_t i = 0; i < vocab.size(); ++i) sorted_index.emplace(vocab[i], i);
    for (auto& row : pending) {
        for (auto& idx : row.token_vocab_indices) {
            idx = sorted_index[insert_vocab[idx]];
        }
    }
    /* Step 1c: emission order matches Python ORDER BY galaxy, row_id. */
    std::sort(pending.begin(), pending.end(), [](const PendingRow& a, const PendingRow& b) {
        if (a.galaxy != b.galaxy) return a.galaxy < b.galaxy;
        return a.row_id < b.row_id;
    });

    if (verbose) {
        std::fprintf(stderr, "rows=%zu vocab=%zu force_regenerate=%d\n",
                     pending.size(), vocab.size(), (int)force_regenerate);
    }

    /* Step 2: pack vocab bytes + offsets for the basis kernel. */
    uint32_t vocab_count = (uint32_t)vocab.size();
    std::vector<uint32_t> offsets; offsets.reserve(vocab_count + 1);
    offsets.push_back(0);
    std::string byte_blob;
    for (auto& tok : vocab) {
        byte_blob.append(tok);
        offsets.push_back((uint32_t)byte_blob.size());
    }

    /* Step 3: device allocations + uploads. */
    uint8_t*  d_bytes     = nullptr;
    uint32_t* d_offsets   = nullptr;
    uint8_t*  d_basis     = nullptr;
    uint32_t* d_vocab_ids = nullptr;
    CK(cudaMalloc(&d_bytes,     std::max<size_t>(byte_blob.size(), 1)));
    CK(cudaMalloc(&d_offsets,   offsets.size() * sizeof(uint32_t)));
    CK(cudaMalloc(&d_basis,     (size_t)vocab_count * PACKED_TRITS_STRIDE));
    CK(cudaMalloc(&d_vocab_ids, (size_t)vocab_count * sizeof(uint32_t)));
    if (!byte_blob.empty()) {
        CK(cudaMemcpy(d_bytes, byte_blob.data(), byte_blob.size(), cudaMemcpyHostToDevice));
    }
    CK(cudaMemcpy(d_offsets, offsets.data(),
                  offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    /* Step 4: derive basis + ids on GPU. */
    if (vocab_count > 0) {
        procedural_basis_fingerprint<<<vocab_count, 64>>>(
            d_bytes, d_offsets, (int)vocab_count, d_basis);
        CK(cudaGetLastError());
        const int id_threads = 256;
        const int id_blocks  = ((int)vocab_count + id_threads - 1) / id_threads;
        procedural_token_id<<<id_blocks, id_threads>>>(
            d_bytes, d_offsets, (int)vocab_count, d_vocab_ids);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
    }

    /* Step 4b: overlay canonical opcode values for tokens hitting the
     * canonical registry. This is the ONLY place where a host-derived
     * value overrides the GPU derivation, and it is sovereign because the
     * canonical opcode map IS the authoritative registry. */
    std::vector<uint32_t> ids_host(vocab_count);
    if (vocab_count > 0) {
        CK(cudaMemcpy(ids_host.data(), d_vocab_ids,
                      (size_t)vocab_count * sizeof(uint32_t),
                      cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < vocab_count; ++i) {
            int opcode = canonical_opcode_for(vocab[i]);
            if (opcode >= 0) ids_host[i] = (uint32_t)opcode;
        }
        CK(cudaMemcpy(d_vocab_ids, ids_host.data(),
                      (size_t)vocab_count * sizeof(uint32_t),
                      cudaMemcpyHostToDevice));
    }

    /* Step 5: per-row accumulator launches + direct .bin write. */
    size_t max_tokens = 1;
    for (auto& row : pending) {
        if (row.token_vocab_indices.size() > max_tokens) {
            max_tokens = row.token_vocab_indices.size();
        }
    }

    uint32_t* d_token_indices = nullptr;
    uint32_t* d_token_ids     = nullptr;
    uint8_t*  d_masses        = nullptr;
    int8_t*   d_output        = nullptr;
    CK(cudaMalloc(&d_token_indices, max_tokens * sizeof(uint32_t)));
    CK(cudaMalloc(&d_token_ids,     max_tokens * sizeof(uint32_t)));
    CK(cudaMalloc(&d_masses,        max_tokens * sizeof(uint8_t)));
    CK(cudaMalloc(&d_output,        MATRYOSHKA_DIM * sizeof(int8_t)));

    std::ofstream out(output_path, std::ios::binary);
    if (!out) { std::fprintf(stderr, "cannot open %s for write\n", output_path); return 1; }

    std::vector<int8_t>   host_output(MATRYOSHKA_DIM);
    std::vector<uint32_t> h_indices(max_tokens);
    std::vector<uint32_t> h_ids(max_tokens);
    std::vector<uint8_t>  h_masses(max_tokens);
    size_t emitted = 0;

    for (auto& row : pending) {
        uint32_t nt = (uint32_t)row.token_vocab_indices.size();
        if (nt == 0) {
            std::vector<int8_t> zeros(MATRYOSHKA_DIM, 0);
            out.write((const char*)zeros.data(), MATRYOSHKA_DIM);
            ++emitted;
            continue;
        }
        for (uint32_t i = 0; i < nt; ++i) {
            uint32_t vi = row.token_vocab_indices[i];
            h_indices[i] = vi;
            uint32_t id  = ids_host[vi];
            h_ids[i]     = id;
            bool canonical = (id & 0x80000000u) == 0;
            h_masses[i]  = canonical ? (uint8_t)64 : (uint8_t)32;
        }
        CK(cudaMemcpy(d_token_indices, h_indices.data(), nt * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_token_ids,     h_ids.data(),     nt * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_masses,        h_masses.data(),  nt * sizeof(uint8_t),  cudaMemcpyHostToDevice));
        matryoshka_accumulator<<<1, 256>>>(
            d_basis, PACKED_TRITS_STRIDE,
            d_token_indices, d_token_ids, d_masses, (int)nt,
            d_output);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(host_output.data(), d_output,
                      MATRYOSHKA_DIM * sizeof(int8_t),
                      cudaMemcpyDeviceToHost));
        out.write((const char*)host_output.data(), MATRYOSHKA_DIM);
        ++emitted;
    }
    out.close();

    std::fprintf(stderr,
        "emitted %zu rows * %d bytes = %zu bytes -> %s\n",
        emitted, MATRYOSHKA_DIM, emitted * (size_t)MATRYOSHKA_DIM, output_path);

    cudaFree(d_bytes); cudaFree(d_offsets); cudaFree(d_basis); cudaFree(d_vocab_ids);
    cudaFree(d_token_indices); cudaFree(d_token_ids); cudaFree(d_masses); cudaFree(d_output);
    return 0;
}
