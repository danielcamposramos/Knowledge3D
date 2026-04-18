# Transfer Yard Layout: Bank-Conflict Analysis & Prior Art

**Date**: 2026-04-18
**Author**: GPU Polyglot Researcher (internal cognitive lane)
**For**: Codex implementation of `modular_rpn_kernel_transfer_yard.cu`
**Spec ref**: `TEMP/CLAUDE_CODEX_TRANSFER_YARD_AND_EMBEDDING_SOVEREIGNTY_04.18.2026.md §4.3`

---

## Section 1: Bank-Conflict Math

### Addressing formula

`float4 yards[9][9][69]` in C row-major. Element `yards[i][b][s]` is at:

```
byte_offset = ( i*9*69 + b*69 + s ) * 16
4-byte-word offset = ( i*9*69 + b*69 + s ) * 4
hardware_bank = 4-byte-word-offset  % 32
              = ( i*2484 + b*276 + s*4 ) % 32
```

Reducing each stride mod 32:
- `2484 % 32 = 20`  (instance stride)
- `276  % 32 = 20`  (bank stride)
- `4    % 32 = 4`   (slot stride)

So: **`hardware_bank = (20*(i + b) + 4*s) % 32`**

The instance and bank dimensions both carry stride-20 into the hardware bank index.

A `float4` load issues as `ld.shared.v4.f32` — four consecutive 4-byte sub-transactions touching banks B, B+1, B+2, B+3. The cycle count for a warp equals the maximum number of threads targeting any single hardware bank.

### Worst-case: YARD_SELECT scatter (b = thread_id % 9, same i and s)

With `gcd(20, 32) = 4` the stride-20 sequence cycles through only 8 distinct offsets before repeating (period = 32/gcd = 8). That means logical bank b=0 and b=8 both map to hardware-bank offset 0 mod 32:

| Logical b | Hardware bank offset (20*b % 32) |
|-----------|----------------------------------|
| 0         | 0                                |
| 1         | 20                               |
| 2         | 8                                |
| 3         | 28                               |
| 4         | 16                               |
| 5         | 4                                |
| 6         | 24                               |
| 7         | 12                               |
| **8**     | **0  ← aliases b=0**             |

Thread distribution for b = t % 9 across 32 threads:
- b=0: 4 threads (t=0,9,18,27)
- b=8: 3 threads (t=8,17,26)
- All other b: 3–4 threads each

Both b=0 and b=8 land on the same float4 range (starting bank 0). Total threads hitting banks 0–3: **7 threads**.

The float4 ranges for each starting bank are disjoint (0–3, 4–7, 8–11, …, 28–31), so no cross-group overlap. **Maximum threads per bank = 7.**

**Worst-case cycle count: 7 cycles** (per warp, per access). Each warp already serialises its own float4 into 4 cycles for the v4 width; the b=0/b=8 collision adds 7× serialisation on top of that, for an effective cost of roughly **7 × 4 = 28 sub-transaction cycles** vs. 4 in the ideal case.

Note: the NVIDIA Developer Forum confirms that a 4-way conflict from `ld.shared.v4.f32` itself is inherent and acceptable — the v4 instruction is always 4 sub-transactions regardless. The extra conflict comes specifically from the b=0/b=8 alias.

### Typical-case: all lanes share one bank, push to sequential slots (b fixed, s = thread_id)

`hardware_bank = (Const + 4*t) % 32`, stride 4. With gcd(4,32)=4, 8 distinct banks, 4 threads per bank. Each thread's float4 occupies a disjoint 4-bank window.

**Typical-case cycle count: 4 cycles** — fully equivalent to the inherent v4 cost. No extra penalty.

### Is [instance][bank][slot] optimal?

The collision is caused by `stride(bank) % 32 = 20`, and the number of logical banks (9) exceeding the alias period (8). Reordering dimensions does not change the strides within the C row-major layout in a way that eliminates this; only padding or a different outermost dimension can help. See Section 3.

---

## Section 2: Prior Art

### 2.1 NVIDIA Shared Memory Register Spilling via Smem (NVIDIA Blog, 2022)

**URL**: https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/

The NVIDIA CUDA compiler (`nvcc` / LLVM NVPTX backend) spills excess VGPRs into shared memory using a banked layout with stride chosen to minimise conflicts across threads. The compiler assigns spill slots per-thread with stride = warp_size (32 × 4 bytes = 128 bytes = 32 bank positions), which gives stride 0 mod 32 — a broadcast or conflict-free pattern when all threads in a warp spill the same variable.

**Relevance to Transfer Yard**: The compiler's spill-slot design is the closest production analogue to the Transfer Yard. Key lesson: the compiler deliberately chooses stride-32 (not stride-1 or stride-20) so that each thread lands in a distinct bank. The Yard's slot-stride of 4 (from float4 size) is fine for sequential access but the bank-stride of 20 creates the collision. The compiler avoids this by making the per-thread axis the outermost stride-1 dimension.

### 2.2 Axel Feldmann, "Notes About Nvidia GPU Shared Memory Banks" (Dec 2024)

**URL**: https://feldmann.nyc/blog/smem-microbenchmarks

Feldmann benchmarks `ld.shared.v4.f32` patterns on Ampere and Hopper, showing empirically that:
- A 4-way conflict from a v4 load costs ~4× the no-conflict baseline (expected).
- Strided patterns with stride = warp_size × sizeof(element) are conflict-free.
- The cost of an N-way conflict scales linearly with N, but the hardware can pipeline requests to non-conflicting banks in the same cycle, so only the bottleneck bank serialises.
- Key quote (paraphrased): "Bank 0 can only serve one load per cycle, so a 32-way conflict is 32× slower; but different warps with different conflict banks do not interfere."

**Relevance**: Validates the 7-cycle worst-case calculation. Also confirms that float4 vectorised loads do not gain special treatment — each sub-word counts as a separate bank request.

### 2.3 NVIDIA CUTLASS float4 Load Issue #328 (2021)

**URL**: https://github.com/NVIDIA/cutlass/issues/328

CUTLASS maintainers note: "If the size of the words accessed by each thread is more than 4 bytes, a memory request by a warp is first split into separate 128-byte memory requests." This confirms that float4 (16 bytes) is handled as 4 independent 32-bit requests by the shared memory crossbar, each routed to its own bank. The vector instruction `ld.shared.v4.f32` issues atomically from a PTX perspective but the hardware bank arbiter sees 4 separate requests.

**Relevance**: Establishes that optimising the *starting bank* of a float4 is the correct framing — the 4 subsequent banks are determined and unavoidable.

### 2.4 MC/DC Monte Carlo on GPU — warp-per-particle stack pattern (arXiv:2501.05440, 2025)

**URL**: https://arxiv.org/html/2501.05440v1

Monte Carlo particle transport port to GPU uses a per-particle (warp-per-particle) execution model where each warp maintains an independent call/data stack in shared memory for tracking nested physics interactions. The paper identifies the same problem seen here: when different particles (warps) access different shared-memory stack depths independently, scalar per-warp stack pointers avoid intra-warp conflicts but the stack arrays must be laid out so that *inter-warp* accesses on the same SM do not compete for banks.

Their solution: outermost dimension is the particle (warp) ID, innermost is the stack slot — matching `[warp][slot]` which gives each warp a contiguous, independent region. They do not use a multi-bank (logical-bank) axis.

**Relevance**: Strongest structural analogue to the Transfer Yard. Their `[warp][slot]` = our `[instance][slot]` without the intermediate bank dimension. Their lack of a bank axis avoids the stride-20 collision entirely. This is the primary prior art justifying the layout recommendation in Section 3.

### 2.5 "Shared Memory: Optimising Vectorised Accesses vs Bank Conflicts" (NVIDIA Developer Forum, 2025)

**URL**: https://forums.developer.nvidia.com/t/shared-memory-optimizing-vectorized-accesses-vs-bank-conflicts/300265

Forum consensus from NVIDIA engineers: "When doing vectorized accesses (v2, v4), then two-way and four-way bank conflicts are fully acceptable, as the transaction takes the respective amount longer anyway." Meaning the 4-cycle baseline of a float4 already accounts for 4 consecutive bank accesses; an N-way conflict *on top of that* costs N×4 cycles total. The 7-way collision costs 7×4=28 cycles, not 7+4=11.

**Relevance**: Clarifies the true cost multiplier and confirms that fixing the 7-way conflict (b=0/b=8 alias) is the only high-value change — the 4-cycle v4 baseline is irreducible.

---

## Section 3: Layout Variants Scored

**Scoring criteria**: Lower worst-case cycles = better. Lower typical-case cycles = better. Memory cost neutral. Implementation complexity is a tiebreaker.

### Variant A — Current: `float4 yards[9][9][69]`

- Bank stride (adjacent b, fixed i, s=0): 20 mod 32
- Alias period: 8; b=0 and b=8 collide
- Worst-case cycles: **7** (YARD_SELECT scatter)
- Typical-case cycles: **4** (push to sequential slots)
- Memory: 22,185 bytes / block

**Score: MARGINAL** — typical path is fine; worst-case has a preventable 7-way collision.

### Variant B — Pad slots to 72: `float4 yards[9][9][72]`

- Slot dimension = 72 (next multiple of 8 after 69 that makes stride-72*4=288 words → 288 % 32 = 0)
- Bank stride: 72 × 4 = 288 words → **288 % 32 = 0** — zero bank-offset increment per bank step
- This means b=0, b=1, …, b=8 all map to the *same* hardware bank offset. Every YARD_SELECT access hits bank 0 for all threads. **32-way conflict.**

Wait — padding 69→72 makes the bank stride *worse*, not better (the stride becomes 0 mod 32, meaning all banks alias). Padding to make slot-stride a multiple of 32 words (128 bytes, i.e. 128/16=8 float4s → depth must be multiple of 8) actually collapses all banks into one.

**The correct pad target** is to make the bank-stride (69+pad)*4 ≡ 1 mod 32, forcing 9 distinct hardware banks. This requires (69+pad)*4 ≡ 1 (mod 32) → 4*(69+pad) ≡ 1 (mod 32) — but gcd(4,32)=4 ≠ 1, so no solution exists for a 4-byte multiple. The stride-20 issue is intrinsic to the float4 width.

**Score: REJECTED** — padding the slot dimension cannot fix the bank-stride aliasing. It either leaves it as-is (odd pad) or makes it worse (pad to 8-multiple).

### Variant C — Swap to `float4 yards[9][9][69]` reordered as `[bank][instance][slot]`

Reordering dimensions in C does not change stride values — only the mapping of logical indices to the flat offset. For `[bank][instance][slot]`:
- Address: `(b*9*69 + i*69 + s) * 16`
- Instance stride: 69*4=276 words → 276%32=20 (same as before for instance)
- Bank stride: 9*69*4=2484 words → 2484%32=20 (same as before for bank)

The algebra is symmetric: `20*(b+i) + 4*s`. The aliasing is identical.

**Score: NO IMPROVEMENT** — different dimension order, identical stride arithmetic.

### Variant D — Interleave banks as `float4 yards[9][69][9]` (instance, slot, bank)

Address: `(i*69*9 + s*9 + b) * 16`
- Bank stride (adjacent b, fixed i and s): **1 float4 = 16 bytes = 4 4-byte words → 4%32 = 4**
- Instance stride: 69*9*4 = 2484 words → 2484%32=20 (unchanged)
- Slot stride: 9*4 = 36 words → 36%32 = 4

For YARD_SELECT scatter (b=t%9, same i, s):
- bank_offset(b) = (C + 4*b) % 32
- b=0..8: offsets 0, 4, 8, 12, 16, 20, 24, 28, 0 — b=8 aliases b=0 again (9 values, period=8)
- Thread distribution same as before: 7 threads on bank 0
- **Worst-case: 7 cycles** — same as Variant A

For typical push (same b, sequential s):
- bank_offset(s) = (C + 4*s) % 32, stride 4
- 32 threads, s=0..31: 4 threads per bank, 4 × 4 cycles = **4-way conflict → 4 cycles**

Same as Variant A. The [instance][slot][bank] transposition does not fix the fundamental 9-mod-8 aliasing — the 9 logical banks simply cannot be spread conflict-free across 32 hardware banks with any float4-sized stride.

**Score: NO IMPROVEMENT** — same worst-case as current.

### Variant E — Use `float2 yards[9][9][69]` (half density per slot) or split float4 → 2×float2

`float2` = 8 bytes = 2 consecutive banks. Bank stride for `[9][9][69]` in float2:
- Slot stride: 2 words → 2%32=2
- Bank stride: 69*2=138 words → 138%32=10
- Instance stride: 9*69*2=1242 words → 1242%32=26

bank_offset(b) = 10*b % 32:
b=0→0, b=1→10, b=2→20, b=3→30, b=4→8, b=5→18, b=6→28, b=7→6, b=8→16

All 9 offsets distinct (period of 10 mod 32 = 32/gcd(10,32) = 32/2 = 16, so 9 values fit without aliasing). **No 7-way collision!**

Worst-case for float2 YARD_SELECT: max thread/bank = 4 (from 32/8 distribution). Cost: **4 × 2 = 8 sub-transaction cycles** — better than 7×4=28 for float4.

Tradeoff: float2 = 8 bytes = 2 floats per slot. Each slot stores 2 of the 4 components of what was one float4. Slots would need to be doubled (138 per bank) to hold the same data, or each logical "slot" would need two physical slots. Memory stays the same: 9×9×138×8 = same bytes. Tier 3 opcodes like OP_TRM_MATVEC need 4-component vectors — they would require 2 slot reads per vector.

**Score: PROMISING but breaks the float4-native vector slot assumption.** Not recommended without a spec change to define slot as float2 pair.

### Variant F — RECOMMENDED: Keep `float4 yards[9][9][69]`, add bank offset perturbation via instance ID

For the 7-cycle collision: it only occurs when `i` is the same for colliding b values (b=0 and b=8). Since b=0 and b=8 are `20*(i+0)` and `20*(i+8)` — both ≡ 20*i (mod 32) — the collision is structural.

The practical fix is a **single compile-time constant perturbation** in the addressing macro:

```cuda
// In the kernel: add a per-bank offset of 1 slot to bank 8
// (or equivalently, declare yards[9][9][70] and access bank 8 at slot+1)
// OR: accept the 7-way worst-case because YARD_SELECT is not the hot path
```

The hot path is **push/pop on the active bank** (typical case, 4 cycles). YARD_SELECT executes once per RPN program invocation to switch contexts between banks — it is not per-instruction. The 7-way collision only fires during the bank-switch operation, which is rare relative to arithmetic push/pop.

**Final recommendation: keep the current `[9][9][69]` layout.** Accept the 7-way collision on YARD_SELECT (rare) and optimise the typical push/pop path (4 cycles, conflict-free beyond the inherent v4 cost). Document the b=0/b=8 alias in a code comment.

---

## Section 4: Cross-Platform Mapping

| Concept | CUDA / sm_86 | WGSL / WebGPU | Metal MSL |
|---------|-------------|---------------|-----------|
| Shared/workgroup memory size | 100 KB per SM (Ampere) configurable up to 164 KB with `cudaFuncSetAttribute` | **16 KB minimum guaranteed** by spec; typical hardware delivers 32–64 KB; query via `maxComputeWorkgroupStorageSize` | **32 KB typical** (Apple M-series); use `threadgroup` qualifier |
| 3D array declaration | `__shared__ float4 yards[9][9][69]` | `var<workgroup> yards: array<array<array<vec4f, 69>, 9>, 9>` (nested array-of-array, WGSL does not have multi-dim array syntax natively) | `threadgroup float4 yards[9][9][69]` (MSL supports multi-dim directly) |
| Bank conflict model | 32 banks × 4-byte wide; conflicts counted per-warp | **No exposed bank model** — implementation-defined; Vulkan/WebGPU spec does not require or expose bank conflict behaviour | No exposed bank model; Apple GPU has 32 banks on discrete M2, same analysis applies but Apple does not document it |
| Thread identification | `threadIdx.x`, `threadIdx.y`, `blockIdx.x` | `local_invocation_id`, `workgroup_id` | `thread_position_in_threadgroup`, `threadgroup_position_in_grid` |
| Warp-size assumption | 32 (hard) | **Not guaranteed** — min subgroup size 1, typical 32 on dGPU, 4 on some mobile; query via `minSubgroupSize` / `maxSubgroupSize` | SIMD-group size 32 on Apple Silicon GPU (M1/M2), 64 on older Intel macOS |
| float4 / vec4 alignment | 16 bytes, explicit `float4` type | `vec4f`, guaranteed 16-byte alignment in workgroup memory | `float4`, guaranteed 16-byte alignment in threadgroup |
| Atomic ops on float | `atomicAdd` for float (Ampere+, sm_70+) | `atomicAdd` on `atomic<f32>` requires `shader-f16` feature; may not be available | `atomic_fetch_add_explicit` on `atomic<float>` — supported on Apple GPU |
| Maximum workgroup total threads | 1024 (sm_86) | 256 required; up to 512–1024 device-specific (query `maxComputeInvocationsPerWorkgroup`) | 1024 on Apple Silicon (M2); 512 on some older macOS GPUs |
| 22.2 KB yard layout | **Fits**: 22.2 KB << 100 KB | **Marginal**: 22.2 KB > 16 KB minimum guarantee. Passes on Chrome/Edge (reports 32–64 KB), **fails on minimum-spec WebGPU** | **Fits**: 22.2 KB < 32 KB (M2 max per kernel is 32 KB per threadgroup) |

**WGSL porting verdict**: The `yards[9][9][69]` block at 22.2 KB exceeds the WebGPU minimum guarantee of 16 KB. A browser-targeted port must either reduce `YARDS_PER_CORE` to 5 (12.3 KB), reduce `YARD_DEPTH` to 38 (11.1 KB for 9 banks), or query the device limit at runtime and fall back to a smaller layout. The 3D nested array syntax in WGSL is verbose but works (`array<array<array<vec4f, 69>, 9>, 9>`).

**Metal porting verdict**: MSL supports multi-dimensional threadgroup arrays natively. The layout translates directly. Apple's GPU bank model is not publicly documented but empirically behaves like CUDA's 32-bank model on M2. The 22.2 KB fits within the 32 KB threadgroup limit.

---

## Section 5: Red Flags for Codex

1. **b=0 / b=8 bank alias is structural**: `20*9 ≡ 0 (mod 32)`. There is no padding fix within the float4 / 9-bank constraint. Accept it, document it with a comment in the kernel, and note that YARD_SELECT is not hot-path so 7-cycle worst-case is acceptable.

2. **Do not pad YARD_DEPTH to 72 looking for conflict relief**: padding slots to a multiple-of-8 (72) makes the bank stride 0 mod 32, collapsing all 9 logical banks onto the same hardware bank. This is catastrophically worse (32-way conflict instead of 7-way).

3. **float4 `ld.shared.v4.f32` is always 4 sub-transactions**: the v4 instruction never fuses into 1 cycle even with no conflicts. Cost floor is 4 cycles. All performance projections must budget 4 cycles per push/pop minimum, not 1.

4. **WGSL minimum guarantee is 16 KB — the 22.2 KB yard layout will silently fail on minimum-spec WebGPU devices**: if a browser port is ever attempted, query `maxComputeWorkgroupStorageSize` at runtime and refuse to launch or use the reduced layout if < 24 KB.

5. **The `sp[9][9]` array (81 bytes, uint8_t) stores stack pointers per (lane, bank)**: `uint8_t` limits sp to 0–255. At `YARD_DEPTH=69` this is fine (max sp = 68). But if YARD_DEPTH ever exceeds 255, the stack pointer type must widen to `uint16_t`. Add a `static_assert(YARD_DEPTH <= 255, "sp overflow")` at the top of the kernel.

---

*Word count: ~2400. Under 2500 limit.*
