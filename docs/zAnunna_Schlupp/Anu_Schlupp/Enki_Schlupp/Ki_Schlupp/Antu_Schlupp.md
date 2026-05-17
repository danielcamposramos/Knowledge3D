# X64-BT: Hybrid Binary + Ternary Compatibility Architecture Specification

**Version:** 0.2 DRAFT — Defensive Publication (Hybrid Logic Clarification)
**Date:** 2026-03-19
**Authors:** Daniel Campos Ramos (PM-KR Chair), Christoph Dorn (PM-KR Contributor), Milton Ponson (PM-KR Co-Chair)
**Organization:** PM-KR Community Group
**License:** W3C Royalty-Free — published as prior art under the W3C Patent Policy
**Companion Specification:** `Anu_Schlupp.md`
**Reference Software Surface:** K3D Knowledgeverse ternary opcode set `TADD/TMUL/TNOT/TCOMP/TQUANT/TPACK/TUNPACK`

---

## Notice of Defensive Publication

This document is a defensive publication for a hybrid binary + ternary execution architecture layered on top of a binary x86-64 substrate. It is intended to establish prior art for:

- dual-logic chips containing both binary and ternary execution fabrics
- ternary overlays on binary fetch/decode pipelines
- ternary register and flag models hosted by x86-64-compatible systems
- ternary prefix and micro-op translation schemes
- ternary ALU, memory, and vector execution blocks attached to a binary ISA underlay

This document does **not** claim ownership of the x86-64 ISA. It defines an open ternary compatibility architecture that can be implemented by:

- licensed x86-64-compatible vendors
- binary translation engines
- FPGA research platforms
- open verification models
- future open ternary processors that choose to host an x86-64 compatibility layer

The strategic purpose is clear: **RISC-T is the native open ternary ISA; X64-BT is the bridge that lets ternary computing land inside the existing server and workstation world without waiting for the installed base to disappear.**

The key clarification in version 0.2 is explicit: the target chip contains **both binary and ternary logic**. The goal is not to replace binary everywhere. The goal is to route each class of computation to the cheaper or semantically correct substrate.

---

## Table of Contents

1. [Purpose and Positioning](#1-purpose-and-positioning)
2. [Foundational Principle: Ternary Above Binary](#2-foundational-principle-ternary-above-binary)
3. [Three-Way Relay to Silicon Language](#3-three-way-relay-to-silicon-language)
4. [Execution Modes](#4-execution-modes)
5. [Trit Representation on x86-64 Underlay](#5-trit-representation-on-x86-64-underlay)
6. [Architectural State](#6-architectural-state)
7. [Prefix and Encoding Model](#7-prefix-and-encoding-model)
8. [Instruction Families](#8-instruction-families)
9. [Ternary Flags and Branching](#9-ternary-flags-and-branching)
10. [Microarchitecture](#10-microarchitecture)
11. [Memory and Cache Architecture](#11-memory-and-cache-architecture)
12. [Vector, SIMD, and GPU Bridge](#12-vector-simd-and-gpu-bridge)
13. [K3D Opcode Mapping](#13-k3d-opcode-mapping)
14. [OS, ABI, and Toolchain Contract](#14-os-abi-and-toolchain-contract)
15. [Manufacturing and Prototyping Paths](#15-manufacturing-and-prototyping-paths)
16. [Reference Roadmap](#16-reference-roadmap)
17. [References](#17-references)

---

## 1. Purpose and Positioning

### 1.1 Why an x64 ternary document exists at all

The ternary future needs **two architectural tracks**:

1. **Native ternary track**: an open ISA designed from first principles around balanced ternary. That is the job of **RISC-T**.
2. **Installed-base bridge track**: a way to bring ternary execution into the dominant binary server/workstation world **now**, without waiting for a clean-slate industry reboot. That is the job of **X64-BT**.

If RISC-T is the clean constitutional architecture, X64-BT is the strategic migration layer.

### 1.2 What X64-BT is

X64-BT is a **hybrid binary + ternary execution architecture** hosted by an x86-64-compatible binary substrate. It preserves the binary underlay where the market is entrenched:

- boot firmware
- operating systems
- page tables
- PCIe and I/O stacks
- compilers and debuggers
- datacenter deployment assumptions

But it adds ternary execution where balanced ternary actually matters:

- arithmetic
- comparison
- uncertainty
- ternary routing
- packed knowledge representations
- vector predicates
- K3D neurosymbolic execution

The resulting chip contains:

- binary execution blocks for workloads that remain cheaper in binary
- ternary execution blocks for workloads that gain meaning or efficiency from `-1 / 0 / +1`
- explicit routing logic that decides where a computation should execute

### 1.3 What X64-BT is not

X64-BT is **not**:

- a replacement name for x86-64
- a claim that x86-64 becomes open
- a proposal to redesign every legacy instruction
- a requirement that ternary logic must wait for native ternary fabrication

Instead, X64-BT states: **keep the binary underlay stable, keep binary execution first-class, add ternary execution islands where they win, and expose a disciplined hardware/software contract for routing between them.**

### 1.4 Core design correction: both logics live on the same chip

The intended chip model is:

```text
one chip = binary logic + ternary logic + explicit dispatch policy
```

This is the correct engineering stance.

1. Some workloads are structurally cheaper in binary:
   - fetch/decode bookkeeping
   - legacy OS execution
   - byte-addressed protocol handling
   - address generation and page-walk machinery
   - bitwise compatibility-heavy code

2. Some workloads are better in ternary:
   - uncertain / neutral / positive reasoning
   - compare-and-branch with three semantic outcomes
   - quantization and confidence projection
   - consensus and defeasible logic
   - compact semantic state for K3D-style knowledge execution

3. Some workloads are effectively neutral:
   - ordinary arithmetic blocks whose cost is dominated by data movement
   - vector reductions where locality matters more than radix
   - mixed kernels where representation conversion dominates execution cost

For neutral cases, the chip should choose based on locality, throughput, energy, and already-resident data format.

---

## 2. Foundational Principle: Ternary Above Binary

### 2.1 The layering rule

X64-BT follows a hard layering rule:

```
Legacy x86-64 fetch / decode / privilege / MMU / I/O
                +
Ternary prefix / ternary micro-op translation / ternary execution blocks
                +
Ternary-aware registers / flags / vector predicates / packed memory forms
```

This means existing binary infrastructure is not discarded. It becomes the **substrate** beneath ternary execution.

### 2.2 Hybrid routing rule

X64-BT defines three execution classes:

#### Binary-preferred

Use the binary fabric when the computation is already structurally optimal there.

Examples:

- page translation
- cache tags and coherence bookkeeping
- conventional scalar control flow
- byte/bit protocol parsing
- legacy instruction decode

#### Ternary-preferred

Use the ternary fabric when the computation benefits from balanced ternary semantics or ternary density.

Examples:

- `less / equal / greater` compare chains
- `false / unknown / true` style logic
- attract / neutral / repel field operations
- ternary quantization and gating
- K3D `TADD/TMUL/TNOT/TCOMP/TQUANT/TPACK/TUNPACK`

#### Neutral / dispatchable

Use whichever fabric is cheaper in the moment.

Examples:

- arithmetic kernels whose cost is dominated by memory movement
- vector lanes with already-packed data on one side
- mixed kernels where control wants ternary but bulk math is binary

### 2.3 Why this is strategically valuable

- It lowers adoption friction in servers and workstations.
- It lets ternary hardware arrive incrementally instead of all-at-once.
- It creates a patent-defensive public design for ternary extensions in the largest installed binary ecosystem.
- It provides a direct path from K3D's current ternary software surface to physical CPUs.

### 2.4 Relation to K3D

K3D already uses balanced ternary semantics at the software and kernel level:

- `TADD (0x70)`
- `TMUL (0x71)`
- `TNOT (0x72)`
- `TCOMP (0x73)`
- `TQUANT (0x74)`
- `TPACK (0x75)`
- `TUNPACK (0x76)`

X64-BT makes these operations architecturally native on an x86-64-class host. The goal is not to put Python in the hot path. The goal is the opposite: **to give ternary kernels a host CPU architecture that speaks their logic directly, while keeping binary execution alive for the parts that remain cheaper there.**

---

## 3. Three-Way Relay to Silicon Language

### 3.1 Daniel's relay intuition

Daniel's relay model is the correct physical intuition:

- rest position = `0`
- positive contact = `+1`
- negative contact = `-1`

Electrically, this means the neutral state is not a special case or exception. It is the **natural resting state**.

This document follows the same distinction as the pure RISC-T spec:
- the primitive is a native three-position state element
- arithmetic is derived from that primitive
- the architecture does not depend on a unary plus-one/minus-one gate family

For compatibility or teaching material, the same three states may also be labeled `0, 1, 2` as long as the mapping is explicit:
- `0` = rest
- `1` = side_a
- `2` = side_b

### 3.2 Silicon translation

In silicon language, the three-way relay becomes one of three implementation families:

#### Family A: Encoded ternary on binary CMOS

Used for early X64-BT implementations.

- ternary state represented by binary rail encoding
- arithmetic performed in ternary ALU slices or lookup networks
- storage can remain binary while execution semantics are ternary

This is the **earliest practical path** and the default X64-BT assumption.

#### Family B: Multi-threshold ternary cells

Used when process support allows stable three-state switching.

- distinct threshold windows for `-1`, `0`, `+1`
- ternary sense amplifiers
- ternary SRAM / register cells
- ternary compare and branch without binary expansion

The exact transistor topology is not normative. X64-BT standardizes the three-state semantics, not a specific increment/decrement ternary gate layout.

#### Family C: Emerging device families

- CNTFET
- memristor-CMOS hybrids
- 2D stacked-material devices

These are future-native paths, but X64-BT does not depend on them. X64-BT is deliberately designed to land **before** mature native ternary foundry support exists.

### 3.3 Practical design principle

For X64-BT the priority order is:

1. preserve ternary semantics
2. preserve x86-64 deployment compatibility
3. use native ternary storage only when it becomes practical

That means **encoded ternary execution today, native ternary storage tomorrow**.

---

## 4. Execution Modes

X64-BT defines four execution modes.

### 4.1 B-mode: legacy binary

The processor behaves like a conventional x86-64-compatible machine.

### 4.2 T-scalar mode

Scalar ternary operations execute on ternary overlay registers and update ternary flags.

### 4.3 T-vector mode

Packed ternary lanes execute on vector-backed ternary registers using XMM/YMM/ZMM substrate resources or their architectural successors.

### 4.4 T-mixed mode

Binary and ternary instructions interleave in one process and one thread. This is the most important migration mode:

- OS remains binary
- most application logic can remain binary
- specific kernels, libraries, and DSLs switch into ternary regions

This is the mode most relevant to K3D, inference runtimes, codecs, search, routing, and field reasoning.

---

## 5. Trit Representation on x86-64 Underlay

X64-BT requires two canonical trit encodings.

### 5.1 Lane encoding: 2-bit relay-safe form

Used inside registers, ALUs, rename queues, and bypass paths.

| Bits | Trit |
|------|------|
| `10` | `+1` |
| `00` | `0` |
| `01` | `-1` |
| `11` | reserved / poison / trap |

Why this form:

- directly compatible with current binary storage elements
- easy to decode in parallel
- matches K3D's current pack/unpack habit
- leaves one code for error detection or speculative poison tracking

### 5.2 Dense memory form: 5 trits per byte

Used in cache lines, file formats, and memory bandwidth optimization.

Because `3^5 = 243 < 256`, five balanced trits fit in one byte with minimal waste.

This gives two valid architectural views:

- **lane view** for execution
- **dense view** for storage and transport

### 5.3 Register projection table

Under the lane encoding, the binary substrate exposes ternary capacity approximately as follows:

| Binary substrate register | Physical bits | Ternary lane capacity |
|---------------------------|---------------|------------------------|
| 64-bit GPR | 64 | 32 trits |
| 128-bit XMM | 128 | 64 trits |
| 256-bit YMM | 256 | 128 trits |
| 512-bit ZMM | 512 | 256 trits |

This is enough to make x64 vector hardware an immediately useful ternary host.

### 5.4 Architectural rule

Execution units operate on lane form. Memory hierarchy may store either lane form or dense form. `TTPACK` and `TTUNPACK` move between them.

---

## 6. Architectural State

### 6.1 Ternary scalar register file

X64-BT defines sixteen logical scalar ternary registers:

`tt0` ... `tt15`

Recommended implementation:

- architecturally separate from the classic GPR namespace
- physically banked over existing scalar or vector storage resources
- optimized for 32-trit or 64-trit ternary scalar operations

### 6.2 Ternary vector register file

X64-BT defines sixteen logical ternary vector registers:

`tv0` ... `tv15`

Suggested physical backing:

- XMM/YMM/ZMM class register files
- future widened vector banks
- optional aliasing with legacy SIMD state under OS save/restore policy

### 6.3 TFLAGS register

X64-BT replaces the binary condition-code mindset with a ternary result register:

| TFLAGS.trit | Meaning |
|-------------|---------|
| `-1` | less / false / repel / negative |
| `0` | equal / unknown / neutral |
| `+1` | greater / true / attract / positive |

This one trit is the center of ternary branching.

### 6.4 TMASK register

For vector mode, each lane has a ternary mask:

| TMASK lane value | Meaning |
|------------------|---------|
| `+1` | execute normally |
| `0` | mask out |
| `-1` | execute inverse / complement / opposite polarity form |

This is more expressive than binary predicate masks and lines up naturally with K3D's attract/neutral/repel semantics.

---

## 7. Prefix and Encoding Model

### 7.1 Why X64-BT uses a prefix model

x86-64 already normalized the idea that instruction meaning can be extended by prefixes:

- legacy prefixes
- REX
- VEX
- EVEX

X64-BT follows that strategic lesson. It does **not** redesign the whole binary instruction stream. It inserts a ternary mode marker that says:

> decode the following operation under ternary semantics.

### 7.2 TEX prefix

This specification defines a conceptual **TEX** (Ternary Extension) prefix class.

TEX carries:

- ternary enable bit
- scalar/vector width selection
- lane-vs-dense format selection
- register extension bits
- optional predicate mode bits

In an x86-64-compatible implementation, TEX may be realized as:

- a real binary prefix sequence
- a decoder escape map
- a microcode-assisted escape
- a translated internal format after frontend decode

The exact byte assignment is implementation-specific in this draft. The prior-art claim is the architecture: **x64 ternary mode via prefix-activated decode path**.

### 7.3 Canonical decoded form

After binary decode, all X64-BT operations normalize to this internal form:

```
T-op | width | dst | src1 | src2 | imm | predicate | packing
```

This internal form is intentionally close to:

- RISC-style execution scheduling
- vector micro-op scheduling
- K3D ternary kernel semantics

### 7.4 Why this matters

The installation base gets to keep its binary frontend, but the backend sees **regular ternary micro-ops**. That is the cleanest bridge between x86-64 history and ternary future.

---

## 8. Instruction Families

X64-BT instruction mnemonics are shown here in abstract assembly form.

### 8.1 Data movement

- `TTMOV dst, src`
- `TTLOAD dst, [base + offset]`
- `TTSTORE [base + offset], src`

### 8.2 Arithmetic

- `TTADD dst, a, b`
- `TTSUB dst, a, b`
- `TTMUL dst, a, b`
- `TTNEG dst, a`
- `TTABS dst, a`
- `TTMIN dst, a, b`
- `TTMAX dst, a, b`

### 8.3 Logical / consensus family

- `TTAND dst, a, b`
- `TTOR dst, a, b`
- `TTXOR dst, a, b`
- `TTNOT dst, a`

For balanced ternary, `TTAND` and `TTOR` are best interpreted as consensus/min and acceptance/max operations rather than boolean gate clones.

### 8.4 Compare and select

- `TTCOMP a, b` -> `TFLAGS`
- `TTSEL dst, neg_src, zero_src, pos_src`

`TTSEL` is particularly important. It turns ternary comparison directly into dataflow without binary branch splitting.

### 8.5 Quantize and bridge

- `TTQUANT dst, src, low_thr, high_thr`
- `TTDEQUANT dst, src`
- `TTPACK dst_dense, src_lane`
- `TTUNPACK dst_lane, src_dense`

### 8.6 Branch and control

- `TTBRNEG label`
- `TTBRZERO label`
- `TTBRPOS label`
- `TTBR3 neg_label, zero_label, pos_label`

`TTBR3` is the key x64 ternary control-flow primitive. One compare, one branch instruction, three outcomes.

---

## 9. Ternary Flags and Branching

### 9.1 Binary condition codes are not enough

Traditional x86 condition codes are decomposed bits:

- zero
- sign
- carry
- overflow

They are powerful for binary arithmetic, but they fragment ternary meaning.

### 9.2 X64-BT branch contract

The canonical ternary branch contract is:

1. `TTCOMP a, b`
2. write one trit into `TFLAGS`
3. branch on one of three semantic outcomes

### 9.3 Branch truth model

| Relation | TFLAGS |
|----------|--------|
| `a < b` | `-1` |
| `a = b` | `0` |
| `a > b` | `+1` |

### 9.4 Hardware benefit

This removes the binary pattern:

- compare
- branch-if-less
- branch-if-equal
- fall-through or second branch

and replaces it with:

- compare
- `TTBR3`

That simplifies control logic exactly where uncertain / equal / opposite outcomes matter.

---

## 10. Microarchitecture

### 10.1 High-level block diagram

```
x86-64 Frontend
  -> legacy decode / TEX detect
  -> ternary micro-op translator
  -> rename / schedule
  -> hybrid dispatch unit
  -> binary execution cluster
  -> ternary execution cluster
  -> pack/unpack bridge
  -> load/store + cache hierarchy
```

### 10.1a Hybrid Dispatch Unit (HDU)

The **Hybrid Dispatch Unit** is the concrete embodiment of the architecture.

Its job is to decide, for each decoded micro-op or translated micro-op bundle:

- send to binary cluster
- send to ternary cluster
- split into cooperating binary and ternary sub-ops

Inputs to the decision:

- opcode class
- operand packing format
- data residency
- energy policy
- latency policy
- compiler or runtime hints

### 10.2 Ternary execution cluster

The ternary cluster contains:

- scalar ternary ALUs
- vector ternary ALUs
- ternary compare/branch unit
- pack/unpack unit
- ternary mask unit
- ternary LUT or PLA slices for compact truth-table execution

### 10.3 Binary execution cluster

The binary cluster remains fully first-class. It should host:

- legacy scalar integer work
- legacy vector work
- address generation units
- page-walk and privilege-adjacent fast paths
- binary-heavy OS/runtime services

X64-BT does not demote binary logic. It preserves binary where binary is efficient and historically optimized.

### 10.4 Recommended ALU primitive set

The execution cluster should implement these as first-class hardware operations:

- add
- multiply
- negate
- compare
- min/max
- quantize
- pack/unpack

This is sufficient to host the current K3D ternary surface and a large fraction of ternary inference/control workloads.

### 10.5 Cross-cluster cooperation

The architecture should allow mixed kernels such as:

1. binary address generation
2. ternary quantization
3. binary vector memory movement
4. ternary compare/select

without forcing full round-trips through software-visible state after every step.

This implies:

- shared rename visibility or fast bridge registers
- low-latency pack/unpack paths
- scheduler awareness of binary/ternary data dependencies

### 10.6 Implementation styles

Three practical implementation styles are valid:

#### Style A: lookup-slice ALU

Use small per-lane lookup tables for trit operations. Good for FPGA and early silicon.

#### Style B: arithmetic ternary slice

Use dedicated ternary full-adder and multiplier cells for high-throughput implementations.

#### Style C: hybrid translated backend

Decode ternary ops, run some on dedicated ternary slices, and lower rarer operations to binary micro-op sequences internally. This is a practical stepping stone for first-generation x64-compatible ternary hosts.

---

## 11. Memory and Cache Architecture

### 11.1 Dual representation rule

Memory hierarchy must support:

- **lane form** for execution locality
- **dense form** for bandwidth efficiency

### 11.2 Cache policy

Recommended policy:

- L1: lane-optimized for direct execution
- L2/L3: optionally dense-packed for capacity
- memory controller: converts between the two under cache-fill / eviction policy

### 11.3 Page attribute

X64-BT may define a page attribute indicating preferred storage interpretation:

- binary page
- ternary-lane page
- ternary-dense page
- mixed page

This is useful for:

- inference buffers
- knowledge graphs
- codec data
- packed ternary state arrays

### 11.4 Error handling

The reserved 2-bit pattern `11` in lane form must be trapped or sanitized by policy. This makes X64-BT more debuggable than an unguarded ternary emulator.

---

## 12. Vector, SIMD, and GPU Bridge

### 12.1 Why vector hardware is the natural ternary host

x86-64 already invested decades of silicon into vector datapaths. X64-BT should exploit that instead of fighting it.

The vector register file is the natural substrate for:

- packed trit lanes
- ternary masks
- ternary reductions
- ternary quantization
- future SIMT offload

### 12.2 Shared design language with open GPU efforts

The x64 ternary overlay should align with the open GPU line Claude is driving:

- lane groups instead of scalar-only thinking
- ternary masks instead of binary predicates
- pack/unpack units that match GPU memory flows
- compatible attract/neutral/repel semantics across CPU and GPU

This makes X64-BT the **host-side CPU companion** to open ternary GPU work based on ideas explored in MIAOW, Nyuzi, and Vortex-class architectures.

### 12.3 Recommended offload boundary

CPU-side X64-BT should own:

- control-heavy ternary work
- branch-rich logic
- quantization
- pack/unpack
- pointer chasing
- compact knowledge routing

GPU-side ternary designs should own:

- large field operations
- swarm execution
- wide reductions
- graph traversal at scale
- multi-star scoring

That split matches K3D's real workload.

---

## 13. K3D Opcode Mapping

X64-BT should treat K3D's current ternary opcodes as the first software contract it must satisfy.

| K3D opcode | Semantic role | X64-BT primitive |
|------------|---------------|-----------------|
| `TADD` | ternary add / saturating-balanced composition | `TTADD` |
| `TMUL` | ternary multiply / polarity interaction | `TTMUL` |
| `TNOT` | polarity inversion | `TTNOT` / `TTNEG` |
| `TCOMP` | compare to `-1 / 0 / +1` | `TTCOMP` |
| `TQUANT` | thresholded ternary projection | `TTQUANT` |
| `TPACK` | dense bridge | `TTPACK` |
| `TUNPACK` | lane bridge | `TTUNPACK` |

### 13.1 K3D-specific consequence

Once these are real host instructions, the CPU side of the K3D stack can:

- preprocess ternary control state without semantic loss
- maintain exact parity with GPU ternary kernels
- avoid lossy boolean collapse of ternary meaning

This is architecturally important for neurosymbolic systems where `0` means "unknown / undecided / neutral", not "false".

---

## 14. OS, ABI, and Toolchain Contract

### 14.1 CPUID exposure

X64-BT should expose capability bits for:

- scalar ternary support
- vector ternary support
- dense pack support
- ternary page support
- ternary branch support

### 14.2 Context switch contract

The OS must save and restore:

- `tt0` ... `tt15`
- `tv0` ... `tv15`
- `TFLAGS`
- `TMASK`
- ternary control register set

### 14.3 Calling convention

Recommended early ABI:

- `tt0` ... `tt5` caller-saved
- `tt6` ... `tt11` callee-saved
- `tt12` ... `tt15` argument / return ternary registers

Vector ternary ABI may follow existing SIMD save classes where practical.

### 14.4 Compiler surface

Toolchains should expose:

- intrinsic functions
- inline assembly mnemonics
- packed ternary vector types
- ternary comparison/select builtins

This is enough to start practical adoption before full high-level-language syntax emerges.

---

## 15. Manufacturing and Prototyping Paths

### 15.1 Immediate path

The first X64-BT demonstrator does **not** need native ternary transistors.

It can be:

- FPGA-hosted
- binary encoded
- microcoded
- vector-backed

What matters is that the architecture contract is real and open.

### 15.2 First silicon target

The first serious silicon target is a binary-compatible CPU tile with:

- x86-64 frontend compatibility
- ternary execution island
- vector-backed ternary register file
- dense pack/unpack bridge

### 15.3 Native ternary migration

As multi-threshold or CNTFET-class fabrication matures, the exact same X64-BT software contract can be backed by:

- native ternary register cells
- native ternary SRAM slices
- native ternary comparator and branch units

The architecture survives the process transition unchanged.

---

## 16. Reference Roadmap

| Phase | Deliverable | Purpose |
|------|-------------|---------|
| 1 | X64-BT defensive publication | Prior art for ternary-on-x64 bridge |
| 2 | FPGA proof with TEX decode, binary cluster, and ternary ALU slice | Validate hybrid execution model |
| 3 | LLVM/assembler intrinsic surface | Let software target the model |
| 4 | Binary-compatible silicon prototype with hybrid dispatcher | Datacenter migration story |
| 5 | Native ternary backing for same contract | Long-term efficiency win |

### 16.1 Strategic role in PM-KR

RISC-T says what a clean open ternary machine is.

X64-BT says how ternary enters the current world without asking permission from that world's inertia, while explicitly preserving binary execution where it remains the cheaper substrate.

Both must exist.

---

## 17. References

### Open GPU reference line

- [MIAOW GPU](https://miaowgpu.org/)
- [MIAOW: An Open Source RTL Implementation of a GPGPU](https://old.hotchips.org/wp-content/uploads/hc_archives/hc27/HC27.25-Tuesday-Epub/HC27.25.50-GPU-Epub/HC27.25.512-MIAOW-Balasubramaniam-UWisc-v1.2.pdf)
- [NyuziProcessor](https://github.com/jbush001/NyuziProcessor)
- [Vortex: OpenCL Compatible RISC-V GPGPU](https://vortex.cc.gatech.edu/)
- [Vortex GitHub](https://github.com/vortexgpgpu/vortex)

### RISC-V reference line

- [RISC-V Ratified Specifications](https://riscv.org/specifications/ratified/)
- [The RISC-V Instruction Set Manual, Volume I](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-118.pdf)
- [RISC-V ISA Manual GitHub](https://github.com/riscv/riscv-isa-manual)
- [RISC-V Unprivileged Architecture (2024)](https://courses.grainger.illinois.edu/ece391/sp2025/docs/unpriv-isa-20240411.pdf)

### x86-64 encoding reference line

- [x86-64 Instruction Encoding — OSDev Wiki](http://wiki.osdev.org/X86-64_Instruction_Encoding)
- [VEX Prefix](https://en.wikipedia.org/wiki/VEX_prefix)
- [sandpile.org opcode encodings](https://www.sandpile.org/x86/opc_enc.htm)
- [Intel XED reference manual](https://intelxed.github.io/ref-manual/)

### Balanced ternary / silicon implementation reference line

- [Balanced Memristor-CMOS ternary logic family](https://arxiv.org/pdf/2309.01615)
- [Efficient Ternary Logic Circuits Optimized by Ternary Devices](https://hajim.rochester.edu/ece/sites/friedman/papers/TEmerging_24.pdf)
- [Binary and ternary convertible CMOS inverter using stacked MoS2/WSe2 FETs](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adfm.202510164)
- [Balanced ternary addition using a gated silicon nanowire](https://www.researchgate.net/publication/51933395_Balanced_ternary_addition_using_a_gated_silicon_nanowire)
- [Design implementations of ternary logic systems: A critical review](https://www.sciencedirect.com/science/article/pii/S2590123024010168)

### Balanced ternary arithmetic reference line

- [Douglas W. Jones on Ternary Arithmetic](https://homepage.divms.uiowa.edu/~jones/ternary/arith.shtml)
- [Truth table of ternary half-adder](https://www.researchgate.net/figure/Truth-table-of-ternary-half-adder_tbl1_290219775)

### K3D internal references

- `docs/vocabulary/SOVEREIGN_NSI_SPECIFICATION.md`
- `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md`
- `docs/vocabulary/MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md`
- `docs/W3C/K3D_VS_STATE_OF_THE_ART_2026.md`
- `Anu_Schlupp.md`
