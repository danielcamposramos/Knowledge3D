# CLAUDE → CODEX — Qdrant MCP Fix + Phase 6.C Sovereign RPN Execution — 2026-04-11

Two deliverables: one infrastructure fix, one sovereign execution slice. Independent. Can be done in parallel (MCP fix first, 6.C is the real work).

---

## Part 1 — Qdrant MCP Collection Fix

### Root cause

`mcp-server-qdrant v0.8.1`'s `FastEmbedProvider.get_vector_name()` returns `"fast-all-minilm-l6-v2"` and passes it as the `using=` argument to `qdrant_client.query_points()`. But the `k3d_specifications` collection was created with an **unnamed default vector** (`VectorParams(size=384, distance=Cosine)`) — no named vectors.

That is why every `qdrant-find` call fails:
```
400: Not existing vector name error: fast-all-minilm-l6-v2
```

Both Claude and Codex configs are correct — `k3d-knowledge` at `http://localhost:8501/mcp/` is wired identically. The bug is server-side: collection schema vs. embedding provider vector name mismatch.

### Fix

Re-create the collection with a **named vector** matching what the MCP server requests, then re-ingest the specs.

```python
#!/usr/bin/env python3
"""Re-create k3d_specifications with the named vector fastembed expects."""

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from pathlib import Path
import hashlib

QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = "@20Cooool58"
COLLECTION = "k3d_specifications"
VECTOR_NAME = "fast-all-minilm-l6-v2"  # must match FastEmbedProvider.get_vector_name()
SPECS_DIR = Path("docs/vocabulary")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

# 1. Delete and re-create with NAMED vector
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)

client.create_collection(
    COLLECTION,
    vectors_config={
        VECTOR_NAME: models.VectorParams(size=384, distance=models.Distance.COSINE)
    },
)

# 2. Chunk and ingest all spec .md files
points = []
point_id = 0
for md_file in sorted(SPECS_DIR.glob("*.md")):
    text = md_file.read_text(encoding="utf-8")
    spec_name = md_file.stem
    # Split by markdown section headers
    sections = []
    current_section = ""
    current_title = spec_name
    for line in text.split("\n"):
        if line.startswith("## ") or line.startswith("# "):
            if current_section.strip():
                sections.append((current_title, current_section.strip()))
            current_title = line.lstrip("#").strip()
            current_section = line + "\n"
        else:
            current_section += line + "\n"
    if current_section.strip():
        sections.append((current_title, current_section.strip()))

    for section_title, section_text in sections:
        chunk = section_text[:2000]  # cap chunk size for embedding quality
        embedding = list(embedder.embed([chunk]))[0].tolist()
        point_id += 1
        points.append(models.PointStruct(
            id=point_id,
            vector={VECTOR_NAME: embedding},
            payload={
                "document": chunk,
                "metadata": {
                    "spec_name": spec_name,
                    "section": section_title,
                    "source": str(md_file),
                },
                "content_type": "specification",
                "spec_name": spec_name,
            },
        ))

# 3. Upsert in batches of 100
for i in range(0, len(points), 100):
    client.upsert(COLLECTION, points[i:i+100])

print(f"Ingested {len(points)} chunks from {len(list(SPECS_DIR.glob('*.md')))} spec files")
print(f"Collection: {COLLECTION}, Vector: {VECTOR_NAME}")
```

Run from the repo root with any Python env that has `qdrant-client` and `fastembed`. Could also run inside the docker container — the deps are already there.

### Verification

After re-ingestion:
```bash
# From host:
curl -sS -H "api-key: @20Cooool58" http://localhost:6333/collections/k3d_specifications | python3 -m json.tool | grep -A5 '"vectors"'
```
Should show `"fast-all-minilm-l6-v2": {"size": 384, "distance": "Cosine"}` — a named vector, not a bare one.

Then test through MCP:
```bash
curl -sS -X POST http://localhost:8501/mcp/ \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","clientInfo":{"name":"test","version":"0"},"capabilities":{}}}'
# grab session-id from response, then:
curl -sS -X POST http://localhost:8501/mcp/ \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -H 'mcp-session-id: <SESSION_ID>' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"qdrant-find","arguments":{"query":"meaning-centric star schema"}}}'
```
Expected: returns spec excerpts from `MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md`. No error.

---

## Part 2 — Phase 6.C: Sovereign RPN Execution in HANDLING_QUERY

### Context

Phase 6.B.3 greenlit the substrate:
- 15 meaning-centric math stars in the 41,164-star VRAM table
- Operator stars carry `meta_rule_addr`, `program_length`, `program_opcode_count` in the 400-byte native record
- Program table in VRAM with 5 bytecodes (ADD/SUB/MUL/DIV/POW)
- Matryoshka RPN embedder: `"plus"` → `math_operator_addition` at rank 1, `"two"` → `concept_digit_two` at rank 2
- All fields survive translation and materialization: probe confirmed `meta_rule_addr=8`, `program_length=5`, `opcode_count=5`

6.C wires the execution path: Tablet operand packing → fused tick HANDLING_QUERY reads top star's program → device-local RPN executor runs it → result written to ActionBuffer → Tablet reads result.

### Authoritative spec anchors

- [MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md §2.3](../docs/vocabulary/MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md) — symlinks as dispatch: Jarvis follows star's symlinks → dispatches specialist → specialist assembles RPN chain → GPU execution → result
- [DUAL_CLIENT_CONTRACT_SPECIFICATION.md](../docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md) — ActionBuffer 288-byte layout. §4.2 tablet mutation slots
- [RPN_DOMAIN_OPCODE_REGISTRY.md](../docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md) — "programs before opcodes." `OP_STORE`, `OP_RECALL`, arithmetic ops. §2 Math tier
- [FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md](../docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md) — Layer 2 Meaning is the center; Layer 3 Rules transform; Layer 4 Meta-Rules orchestrate
- [galaxy_vram_table.py](../knowledge3d/knowledgeverse/galaxy_vram_table.py) lines 39-42 — native field offsets: `STAR_META_RULE_ADDR_OFFSET=384`, `STAR_PROGRAM_FLAGS_OFFSET=388`, `STAR_PROGRAM_LENGTH_OFFSET=392`, `STAR_PROGRAM_OPCODE_COUNT_OFFSET=396`
- [device_functions.cuh](../knowledge3d/cranium/cuda/device_functions.cuh) lines 83-86 — GPU-side matching offsets
- [action_types.py](../knowledge3d/cranium/actions/action_types.py) lines 43-78 — `ACTION_BUFFER_DTYPE` 288-byte struct. `tablet_data` is 6 × uint32 at word offset 62-67
- [trm_step_fused.cu](../knowledge3d/cranium/ptx/trm_step_fused.cu) line 966 — `case TRM_STATE_HANDLING_QUERY:` landing zone
- [galaxy_answer_decode.cu](../knowledge3d/cranium/cuda/galaxy_answer_decode.cu) — existing cosine decode kernel, already reads from 400-byte records, already extracts `selection_role`, `star_hash`

### 6.C is three slices

#### 6.C.1 — Tablet operand packing (boundary Python, legitimate)

**File:** [headless_tablet.py](../knowledge3d/bridge/headless_tablet.py)

**Current state:** `TabletEnvelope.to_action_buffer()` at line 350 packs `tablet_data[0..5]` with `task_lo, task_hi, query_lo, query_hi, query_len, specialist_code`. All 6 slots are taken. `TabletIngest.math_task()` at line 447 passes through to `to_action_buffer()` with no operand extraction.

**Problem:** there is no room in the current `tablet_data[6]` for operands. The Dual Client Contract defines 6 uint32 tablet slots. But look at the *full* ActionBuffer: there are 4 reserved uint32s at `tablet_reserved[4]` (word offset 68-71) that are currently always zero. That is the operand space.

**Fix:** extend `TabletEnvelope.to_action_buffer()` with a `math_operands` code path:

```python
# For math tasks, pack operands into tablet_reserved[] (the currently-zero tail)
if self.surface_kind == SURFACE_KIND_MATH:
    operands = _extract_math_operands(self.query)  # boundary parsing
    buf.buffer["tablet_reserved"][0][0] = np.uint32(operands.left)
    buf.buffer["tablet_reserved"][0][1] = np.uint32(operands.right)
    buf.buffer["tablet_reserved"][0][2] = np.uint32(operands.count)
    buf.buffer["tablet_reserved"][0][3] = np.uint32(operands.operator_hint)
```

`_extract_math_operands(query: str)` is a simple regex parser — this is the I/O boundary, NOT the reasoning path. Returns `(left=2, right=3, count=2, operator_hint=0x2B)` for `"2+3?"`. Per [MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md §2.3](../docs/vocabulary/MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md): "Python NEVER decides 'this is a math problem, route to math.'" — that is correct here because the Tablet is not routing; it's parsing I/O into the universal ActionBuffer contract. The Galaxy's cosine navigation decides what operator star to select. The Tablet merely supplies the operand values.

Supported patterns for Phase 6.C (extend later via Grammar Galaxy):
- `"2+3?"`, `"2 + 3"`, `"7-4"`, `"6*8"`, `"15/5"`, `"2^10"`
- Digits only (no word-form: `"two plus three"` is Phase 6.C+ via Grammar Galaxy parsing)

**Verify:** `_extract_math_operands("2+3?")` returns `left=2, right=3, count=2, op_hint=0x2B`. `_extract_math_operands("15/5")` returns `left=15, right=5, count=2, op_hint=0x2F`.

#### 6.C.2 — Device-local RPN executor (`__device__` function)

**New file:** `knowledge3d/cranium/cuda/rpn_execute_device.cuh`

A `__device__` function callable from the fused tick. Not a kernel — no host launch overhead. The fused tick calls it inline.

```cuda
#ifndef K3D_RPN_EXECUTE_DEVICE_CUH
#define K3D_RPN_EXECUTE_DEVICE_CUH

#include <stdint.h>

// Opcodes — must match star_crafter.py _PROGRAM_SPECS bytecodes
#define RPN_OP_PUSH_OPERAND_0  0x10
#define RPN_OP_PUSH_OPERAND_1  0x11
#define RPN_OP_ADD             0x20
#define RPN_OP_SUB             0x21
#define RPN_OP_MUL             0x22
#define RPN_OP_DIV             0x23
#define RPN_OP_POW             0x24
#define RPN_OP_STORE_RESULT    0x30
#define RPN_OP_RET             0xFF

#define RPN_STACK_DEPTH        16

__device__ int rpn_execute_device(
    const unsigned char* __restrict__ program_table,
    unsigned int          program_offset,
    unsigned int          program_length,
    int                   operand_0,
    int                   operand_1,
    int*                  result_out
) {
    float stack[RPN_STACK_DEPTH];
    int sp = 0;
    float result = 0.0f;

    for (unsigned int pc = 0; pc < program_length; ++pc) {
        unsigned char op = program_table[program_offset + pc];
        switch (op) {
            case RPN_OP_PUSH_OPERAND_0:
                if (sp < RPN_STACK_DEPTH) stack[sp++] = (float)operand_0;
                break;
            case RPN_OP_PUSH_OPERAND_1:
                if (sp < RPN_STACK_DEPTH) stack[sp++] = (float)operand_1;
                break;
            case RPN_OP_ADD:
                if (sp >= 2) { float b = stack[--sp]; float a = stack[--sp]; stack[sp++] = a + b; }
                break;
            case RPN_OP_SUB:
                if (sp >= 2) { float b = stack[--sp]; float a = stack[--sp]; stack[sp++] = a - b; }
                break;
            case RPN_OP_MUL:
                if (sp >= 2) { float b = stack[--sp]; float a = stack[--sp]; stack[sp++] = a * b; }
                break;
            case RPN_OP_DIV:
                if (sp >= 2) { float b = stack[--sp]; float a = stack[--sp];
                    stack[sp++] = (b != 0.0f) ? a / b : 0.0f;
                }
                break;
            case RPN_OP_POW:
                if (sp >= 2) { float b = stack[--sp]; float a = stack[--sp]; stack[sp++] = powf(a, b); }
                break;
            case RPN_OP_STORE_RESULT:
                if (sp > 0) result = stack[sp - 1];
                break;
            case RPN_OP_RET:
                *result_out = (int)result;
                return 1;  // success
            default:
                break;
        }
    }
    *result_out = (int)result;
    return (sp > 0) ? 1 : 0;
}

#endif // K3D_RPN_EXECUTE_DEVICE_CUH
```

The opcode values **must** match what `star_crafter.py` emits in `_PROGRAM_SPECS` (line 162-193). Confirm with a grep after landing.

This covers Math tier only. Future phases extend with the full `OP_STORE`/`OP_RECALL`/`OP_BRANCH`/`OP_LOOP` set from [RPN_DOMAIN_OPCODE_REGISTRY.md](../docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md) §2. The API (`program_table`, `program_offset`, `program_length`, `operands`, `result_out`) is stable across that extension — more opcodes in the switch, same interface.

#### 6.C.3 — HANDLING_QUERY wiring in the fused tick

**File:** [trm_step_fused.cu](../knowledge3d/cranium/ptx/trm_step_fused.cu) line 966

**Current state:** `HANDLING_QUERY` calls `trm_query_fast_lane_phase()` which runs the TRM recursive core on the query embedding to produce `y_new`. After that, thread 0 pops the query state and the tick ends. There is no cosine decode, no star selection, no RPN execution.

**What to add after `trm_query_fast_lane_phase()` returns:**

```
// Phase 6.C: sovereign answer path
__syncthreads();

if (tid == 0 && galaxy_table != nullptr && galaxy_star_count > 0 &&
    program_table != nullptr) {

    // 1. Cosine decode: find the top answer-eligible star
    //    (inline reduction — galaxy_answer_decode already exists as a kernel,
    //     but for the fused tick we need a device-local version. Either:
    //     (a) extract the cosine loop from galaxy_answer_decode.cu into a
    //         __device__ helper and call it here, OR
    //     (b) run a single-thread linear scan over the answer-eligible stars.
    //     For 41K stars, single-thread is ~0.1ms — acceptable for Phase 6.C.
    //     Phase 6.D replaces with the composed-head pipeline.)

    unsigned int top_index = K3D_INVALID_STAR_INDEX;
    float top_score = -1.0e30f;
    // ... cosine scan over galaxy_table using y_new[0..63] ...

    if (top_index != K3D_INVALID_STAR_INDEX) {
        unsigned int base = top_index * K3D_STAR_RECORD_BYTES;
        unsigned int role = read_u32(galaxy_table, base + K3D_STAR_SELECTION_ROLE_OFFSET);
        unsigned int meta_rule_addr = read_u32(galaxy_table, base + GALAXY_STAR_META_RULE_ADDR_OFFSET);
        unsigned int program_length = read_u32(galaxy_table, base + GALAXY_STAR_PROGRAM_LENGTH_OFFSET);

        // 2. If top star is an executor with a program, run it
        if (role == ROLE_EXECUTOR_ID && meta_rule_addr > 0 && program_length > 0) {
            // Read operands from incoming ActionBuffer tablet_reserved[]
            int op0 = (int)action_buffer_in_tablet_reserved_0;
            int op1 = (int)action_buffer_in_tablet_reserved_1;
            int result = 0;
            int ok = rpn_execute_device(
                program_table, meta_rule_addr, program_length,
                op0, op1, &result
            );
            if (ok) {
                // 3. Write result to outgoing ActionBuffer
                //    tablet_data[0] = result (integer)
                //    action_type = UPDATE_TABLET
                action_buffer_out->tablet_data[0] = (unsigned int)result;
                action_buffer_out->action_type = 0x04; // UPDATE_TABLET
                action_buffer_out->star_hash = read_u64(galaxy_table, base + K3D_STAR_STAR_HASH_OFFSET);
            }
        }
    }
}
```

**Kernel parameters added to `trm_step_fused`:**
- `const unsigned char* __restrict__ program_table` — VRAM program table base pointer
- `const unsigned int* __restrict__ action_buffer_in` — incoming ActionBuffer (query's operand slots)

Both bound through `trm_step_fused_bridge.py` which already has `bind_galaxy_table()` and `bind_program_table()`. The incoming ActionBuffer is the one constructed by `TabletEnvelope.to_action_buffer()` and passed through `bridge.submit_query()`.

**Key design decisions:**

1. **Single-thread cosine scan at tid==0.** Yes, this is O(N) in star_count. For 41K × 64D it's ~0.1ms wall time. The composed-head pipeline (Morton → LED-A* → Frustum → LOD → Swarm → Halting) already exists and will replace this in Phase 6.D. For 6.C the goal is correctness, not latency.

2. **Operand source is the incoming ActionBuffer.** The Tablet packed them at the I/O boundary. The fused tick reads them from VRAM. No Python in the loop.

3. **Result destination is the outgoing ActionBuffer.** `tablet_data[0]` carries the integer result. `action_type = UPDATE_TABLET`. The existing `HeadlessTabletMPC` readback path (`read_action_buffers_words()`) sees this.

4. **Star selection by cosine only.** TRM `y_new` after recursive core is the query vector in the same Matryoshka tier_64 space as the Galaxy embeddings (guaranteed by 6.B.3's singleton embedder). Cosine against `answer_eligible` stars selects the operator star. TRM navigation quality improves with training — but the substrate is proven correct at rank 1.

**Bridge parameter threading:**

`trm_step_fused_bridge.py` already has:
- `_galaxy_table_ptr` and `_galaxy_star_count` (Phase 5)
- `_program_table_ptr` and `_program_table_size` (Phase 6.B.2)

Add:
- Pass the incoming ActionBuffer pointer to the kernel launch. The bridge already constructs the ActionBuffer from `TabletEnvelope.to_action_buffer()` and copies it to VRAM via `submit_query()`. Thread the device pointer into the kernel args.

### Tablet readback

**File:** [headless_tablet.py](../knowledge3d/bridge/headless_tablet.py)

After the fused tick returns:
1. `HeadlessTabletMPC` calls `bridge.read_action_buffers_words()` (already exists from Phase 2.5).
2. Checks `action_type == UPDATE_TABLET`.
3. Reads `tablet_data[0]` as the integer result.
4. Formats and returns the answer string: `str(int(tablet_data[0]))`.

No Python arithmetic. The Tablet is a presenter, not a calculator. The GPU did the math.

### Validation

**`tests/test_phase6c_sovereign_math_2_plus_3.py`** under `K3D_PYTEST_PROBE_CUDA=1`:

1. Boot full Knowledgeverse (no shortcuts).
2. Submit `"2+3?"` through `HeadlessTabletMPC`.
3. Assert returned answer string is `"5"`.
4. Assert via `read_action_buffers_words()` that `action_type == UPDATE_TABLET` and `tablet_data[0] == 5`.
5. Assert resolved star hash corresponds to `math_operator_addition`.

**`tests/test_phase6c_sovereign_math_grid.py`** under `K3D_PYTEST_PROBE_CUDA=1`:

| Query | Expected `tablet_data[0]` | Expected answer | Expected star |
|-------|--------------------------|-----------------|---------------|
| `"2+3?"` | 5 | `"5"` | `math_operator_addition` |
| `"7-4"` | 3 | `"3"` | `math_operator_subtraction` |
| `"6*8"` | 48 | `"48"` | `math_operator_multiplication` |
| `"15/5"` | 3 | `"3"` | `math_operator_division` |

**Regression:** Phase 1–5 + 6.B suite stays green. Sovereignty grep clean on all touched Cranium files.

### What this does NOT do

- No word-form parsing (`"two plus three"`) — Grammar Galaxy parsing is Phase 6.C+.
- No multi-step (`"2+3*4"`) — requires precedence/composition, which is Grammar Galaxy + RPN chaining.
- No TRM training — random-init TRM `y_new` happens to work because the Matryoshka embedder already puts query embeddings into the right cosine neighborhood. Learning refines this; it's not required for correctness.
- No composed-head pipeline in the cosine pass — single-thread linear scan is sufficient for 41K stars. Phase 6.D replaces this with the full Morton → LED-A* → Frustum → LOD → Swarm → Halting path.

### Order of operations

1. **6.C.1** — `_extract_math_operands` + operand packing into `tablet_reserved`. Unit test with no GPU.
2. **6.C.2** — `rpn_execute_device.cuh`. Compile test via `nvcc`. Unit test via a tiny test kernel that calls it.
3. **6.C.3** — Wire into `HANDLING_QUERY`: cosine scan + RPN execution + ActionBuffer write. Thread parameters through bridge. End-to-end test: `"2+3?"` → `"5"`.
4. **Tablet readback** — `HeadlessTabletMPC` reads `tablet_data[0]` and returns the string.
5. **Grid test** — all four arithmetic operations pass.
6. **Regression** — full Phase 1–6.B suite green.

### One principle to hold

*TRM navigates. Galaxy carries the programs. RPN core runs them. Tablet presents the result.* No Python arithmetic anywhere. The `"5"` that comes back to the user was computed by `rpn_execute_device` on the GPU, from bytecode stored on an operator star placed by the meaning-centric Star Crafter, selected by cosine navigation through the Matryoshka embedding space, with operands packed by the Tablet at the I/O boundary. That is the sovereign path.

Report per slice. The smoke test after 6.C.3 is the moment we've been building toward: `"2+3?"` → `"5"`, sovereign, end-to-end, on GPU.
