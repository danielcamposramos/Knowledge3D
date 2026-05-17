# RISC-BT: Open Hybrid Binary-Ternary Chip Architecture Specification

**Version:** 0.1 DRAFT — Defensive Publication
**Date:** 2026-03-19
**Authors:** Daniel Campos Ramos (PM-KR Chair), Christoph Dorn (PM-KR Contributor), Milton Ponson (PM-KR Co-Chair)
**Organization:** PM-KR Community Group
**License:** W3C Royalty-Free — This document is published as prior art under the W3C Patent Policy. All architectures described herein enter the public domain upon publication. No party may patent any design disclosed in this specification.
**Reference Implementation:** K3D Knowledgeverse (sovereign GPU pipeline, 7 ternary opcodes operational since March 2026)
**Prior Art Items:** 40 (see Appendix A)
**Companion Specifications:** [Anu_Schlupp.md](Anu_Schlupp.md) - Pure ternary ISA design; [Enlil_Schlupp.md](Enlil_Schlupp.md) - One true RISC core covering RV32, RV64, RV32T, and RV64T

---

## NOTICE OF DEFENSIVE PUBLICATION

This specification constitutes a **defensive publication** under international patent law. By publishing detailed, enabling descriptions of hybrid binary-ternary chip architectures, the authors establish **prior art** that prevents any party — including the authors themselves — from obtaining patents on the designs described herein.

This document is intentionally published openly, timestamped, and indexed to maximize its effectiveness as prior art. It is designed to be **enabling** — detailed enough that a person skilled in the art of chip design could implement the described architectures.

**Motivation:** A hybrid binary-ternary chip provides the pragmatic migration path from today's binary world to tomorrow's ternary future. Rather than requiring an all-or-nothing transition, this design places BOTH logic systems on a single die, routing each computation to whichever system is cheaper. This technology must remain open and royalty-free to prevent monopolization.

The ternary side of this hybrid follows the same foundational rule as RISC-T: the primitive is a native three-state relay cell with `0` as rest and two side states. Arithmetic is derived from that primitive; no unary increment/decrement gate family is normative.

---

## Table of Contents

1. [The Hybrid Rationale](#1-the-hybrid-rationale)
2. [Cost Analysis: Binary vs Ternary per Operation](#2-cost-analysis-binary-vs-ternary-per-operation)
3. [Architecture Overview](#3-architecture-overview)
4. [Dual-Domain Register File](#4-dual-domain-register-file)
5. [The Cost Router — Automatic Dispatch](#5-the-cost-router--automatic-dispatch)
6. [Binary Execution Domain (BED)](#6-binary-execution-domain-bed)
7. [Ternary Execution Domain (TED)](#7-ternary-execution-domain-ted)
8. [Shared Memory Subsystem](#8-shared-memory-subsystem)
9. [Instruction Encoding — Unified Format](#9-instruction-encoding--unified-format)
10. [Domain Crossing Instructions](#10-domain-crossing-instructions)
11. [Hybrid Pipeline Specification](#11-hybrid-pipeline-specification)
12. [Hybrid GPU Architecture (BT-G)](#12-hybrid-gpu-architecture-bt-g)
13. [Hybrid Cache Coherence (BT-MESI)](#13-hybrid-cache-coherence-bt-mesi)
14. [I/O and Protocol Handling](#14-io-and-protocol-handling)
15. [Power Management — Asymmetric Domain Gating](#15-power-management--asymmetric-domain-gating)
16. [K3D Integration (BT-K Extension)](#16-k3d-integration-bt-k-extension)
17. [Compiler Model — Domain-Aware Compilation](#17-compiler-model--domain-aware-compilation)
18. [HDL Reference Implementation (Pseudocode)](#18-hdl-reference-implementation-pseudocode)
19. [Reference Designs](#19-reference-designs)
20. [Manufacturing Targets](#20-manufacturing-targets)
21. [Migration Path: Binary → Hybrid → Pure Ternary](#21-migration-path-binary--hybrid--pure-ternary)

**Appendices:** A (Prior Art Items 1-40), B (Operation Cost Comparison Table), C (Economic Model)

---

## 1. The Hybrid Rationale

### 1.1 Why Not Just Ternary?

The companion specification (RISC-T) defines a pure ternary architecture. It is the long-term target. But the world is binary TODAY:

- All existing software is binary
- All existing DRAM, flash, SSD storage is binary
- All I/O protocols (PCIe, USB, HDMI, Ethernet) are binary
- All EDA tools are binary-native
- All programming languages emit binary machine code

A pure ternary chip MUST translate at every boundary — memory, I/O, software. This translation has a cost.

### 1.2 Why Not Just Binary with Ternary Extensions?

The separate `x86_x64T` family keeps x86 compatibility as the center of gravity and adds ternary as a hybrid extension path. That is valid for the x86 ecosystem, but it is not the third RISC document. Inside the RISC family, the missing third companion spec is the unified `RV32/RV64/RV32T/RV64T` core family. A binary-primary extension model still wastes ternary advantages:

- Ternary operations still go through binary register files
- Three-way branching still decomposes into two binary branches
- Negation still costs a subtraction instead of a wire swap
- The ternary path is always a second-class citizen

### 1.3 The Hybrid Insight

**Daniel's principle: "some computations are simply cheaper on binary while others are the same on ternary."**

The hybrid approach places BOTH logic systems as first-class citizens on the same die:

```
┌──────────────────────────────────────────────────────────┐
│                    RISC-BT Hybrid Core                    │
│                                                           │
│   ┌─────────────────┐    ┌─────────────────┐             │
│   │  Binary Domain   │    │ Ternary Domain   │             │
│   │  (BED)           │    │ (TED)            │             │
│   │                  │    │                  │             │
│   │  Binary ALU      │◄──►│  Ternary ALU     │             │
│   │  Binary FPU      │    │  Ternary FPU     │             │
│   │  Binary regs     │    │  Ternary regs    │             │
│   │  Bit manipulation│    │  3-way branch    │             │
│   │  Boolean logic   │    │  Free negation   │             │
│   │  I/O protocols   │    │  Knowledge ops   │             │
│   └────────┬─────────┘    └────────┬─────────┘             │
│            │                       │                       │
│            └───────────┬───────────┘                       │
│                        │                                   │
│              ┌─────────┴──────────┐                        │
│              │   COST ROUTER      │                        │
│              │   (routes each op  │                        │
│              │    to cheaper      │                        │
│              │    domain)         │                        │
│              └─────────┬──────────┘                        │
│                        │                                   │
│              ┌─────────┴──────────┐                        │
│              │  SHARED MEMORY     │                        │
│              │  (unified L1/L2,   │                        │
│              │   binary DRAM      │                        │
│              │   interface)       │                        │
│              └────────────────────┘                        │
└──────────────────────────────────────────────────────────┘
```

Each instruction is dispatched to the domain where it executes **cheapest** — fewer gates, less energy, fewer cycles. The chip doesn't care about ideology (binary vs ternary); it cares about cost per operation.

### 1.4 Relationship to Companion Specs

| Specification | Domain | Purpose |
|--------------|--------|---------|
| **RISC-T** (pure ternary) | Ternary only | Long-term target: native ternary hardware |
| **RISC-BT** (this document) | Binary + Ternary | Migration path: hybrid chip with both domains |
| **RV32/RV64T True Hybrid Core** | Unified RISC product core | One core family spanning RV32, RV64, RV32T, RV64T |
| **X64-BT** (external companion family) | Binary primary, ternary extension | Compatibility path for the x86/x64 ecosystem |

**RISC-BT is the bridge.** It runs binary software at full speed (no translation penalty) AND runs ternary workloads at full speed (no second-class status). As the ternary ecosystem matures and more software becomes ternary-native, the binary domain can be powered down — eventually yielding a pure RISC-T chip.

---

## 2. Cost Analysis: Binary vs Ternary per Operation

This is the foundational table that drives the Cost Router (§5). Each operation is analyzed for gate count, energy, and cycle count in both domains.

### 2.1 Operations Cheaper in Binary

| Operation | Binary Cost | Ternary Cost | Winner | Why |
|-----------|------------|-------------|--------|-----|
| **Boolean AND** | 2 transistors (NAND) | 6 transistors (TAND = MIN over 3 levels) | Binary 3× | Boolean AND on 2 states is the simplest possible gate |
| **Boolean OR** | 2 transistors (NOR) | 6 transistors (TOR = MAX over 3 levels) | Binary 3× | Same — 2-state logic is simpler for pure boolean |
| **Boolean XOR** | 4 transistors | 8 transistors (TXOR over 3 levels) | Binary 2× | XOR is inherently a 2-state parity operation |
| **Bit shift** | Wire routing (0 gates) | Trit shift + 2-bit-per-trit repacking | Binary ∞× | Binary shifts are free wire routing; trit shifts need re-encoding |
| **Bit masking** | AND gate per bit | TAND per trit + domain awareness | Binary ~2× | Masking is a binary-native concept (on/off) |
| **Bit counting (popcount)** | Dedicated binary tree | Convert to binary first, then count | Binary ~3× | Counting set bits is binary by definition |
| **CRC/hash** | Galois field GF(2) arithmetic | No ternary GF(3) hardware exists yet | Binary ~5× | CRC polynomials are defined over GF(2) |
| **AES/crypto** | Dedicated binary AES-NI | Ternary S-box TBD | Binary ~10× | All existing crypto is binary-defined |
| **I/O protocol** | Native (PCIe, USB, HDMI) | Must convert to binary at boundary | Binary ∞× | All existing protocols are binary |
| **Memory addressing** | Native binary DRAM | Trit-to-bit address translation | Binary ~1.5× | DRAM is binary; ternary addresses need conversion |

### 2.2 Operations Cheaper in Ternary

| Operation | Binary Cost | Ternary Cost | Winner | Why |
|-----------|------------|-------------|--------|-----|
| **Negation** | Subtraction circuit (N gates) | Wire swap (0 gates, 0 delay) | Ternary ∞× | Balanced ternary: swap -1↔+1 lines. FREE. |
| **Three-way comparison** | CMP + 2 branches (2 cycles) | TCOMP: 1 instruction, 1 trit result | Ternary 2× | Result IS a trit: +1/0/-1 = greater/equal/less |
| **Three-way branch** | 2 conditional jumps (2 cycles) | TB3: 1 instruction, 3 targets (1 cycle) | Ternary 2× | Single ternary MUX vs cascaded binary MUXes |
| **Sign detection** | Extract sign bit + compare | Read MST (most significant trit) | Ternary ~1.5× | Sign is inherent in every trit, no extraction needed |
| **Absolute value** | Conditional negate (branch + SUB) | TCOMP sign trit + conditional wire swap | Ternary ~2× | Wire swap is free, conditional is 1 trit test |
| **Rounding** | Biased (complex correction) | Unbiased (natural in balanced ternary) | Ternary ~3× | Balanced ternary rounds to nearest without bias |
| **Knowledge state** | 2 bits (true/false + valid/invalid) | 1 trit (true/unknown/false) | Ternary 2× | Unknown is a native state, not a separate flag |
| **SIMT divergence** | 2 passes (if/else) | 1 pass (3-way predication) | Ternary 2× | Active/masked/inverted in single pass |
| **Semantic gravity** | Fixed-point multiply + table lookup | Native TGRAV instruction | Ternary ~3× | Ternary force model maps directly to hardware |
| **Frustum culling** | Float compare → bool | TCOMP → trit (+1 inside, 0 edge, -1 outside) | Ternary ~2× | Three-way result without epsilon tolerance |

### 2.3 Operations at Roughly Equal Cost

| Operation | Binary Cost | Ternary Cost | Notes |
|-----------|------------|-------------|-------|
| **Integer addition** | Full adder (6 transistors/bit) | Ternary full adder (28 transistors/trit, but fewer trits needed) | ~Equal when measured per information bit |
| **Integer multiplication** | O(N²) binary multiplier | O(N²) ternary multiplier | Same algorithmic complexity, different base |
| **Floating-point add** | IEEE 754 pipeline (5-6 stages) | TFloat pipeline (5-6 stages) | Same pipeline depth |
| **Floating-point multiply** | IEEE 754 multiplier | TFloat multiplier | Same complexity class |
| **Division** | Iterative (same iterations) | Iterative (same iterations) | Same algorithmic approach |
| **Memory load/store** | 1 cycle (cache hit) | 1 cycle (cache hit) + conversion overhead | Nearly equal with unified cache |

### 2.4 The Crossover Principle

**Key insight:** The winner depends on the WORKLOAD, not the instruction:

| Workload Type | Dominant Operations | Preferred Domain |
|--------------|-------------------|-----------------|
| **Crypto / hashing** | XOR, shift, boolean AND | Binary |
| **I/O drivers** | Protocol framing, CRC | Binary |
| **Bit manipulation** | Shift, mask, popcount | Binary |
| **Knowledge reasoning** | Comparison, branching, negation | Ternary |
| **3D spatial** | Frustum, Morton, gravity | Ternary |
| **AI inference** | Matrix multiply, attention | Ternary (with TMU) |
| **GPU shading** | Rasterization, 3-way tests | Ternary |
| **Scientific computing** | Arithmetic, FP | Either (roughly equal) |
| **General application** | Mixed | Both (cost router decides per-instruction) |

---

## 3. Architecture Overview

### 3.1 Core Block Diagram

```
RISC-BT Hybrid Core — Full Block Diagram
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                         UNIFIED FETCH + DECODE                          │  ║
║  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐  │  ║
║  │  │ I-Cache      │  │ Instruction  │  │ Cost Router                  │  │  ║
║  │  │ (binary +    │  │ Decoder      │  │ (domain assignment per op)   │  │  ║
║  │  │  ternary     │  │ (unified     │  │ Static: opcode table lookup  │  │  ║
║  │  │  lines)      │  │  format)     │  │ Dynamic: workload profiling  │  │  ║
║  │  └──────────────┘  └──────────────┘  └──────────────────────────────┘  │  ║
║  └──────────────────────────────┬──────────────────────────────────────────┘  ║
║                     ┌───────────┴───────────┐                                 ║
║                     ↓ Binary                ↓ Ternary                         ║
║  ┌──────────────────────────┐  ┌──────────────────────────┐                  ║
║  │  BINARY EXECUTION        │  │  TERNARY EXECUTION        │                  ║
║  │  DOMAIN (BED)            │  │  DOMAIN (TED)             │                  ║
║  │                          │  │                           │                  ║
║  │  ┌────────────────────┐  │  │  ┌─────────────────────┐  │                  ║
║  │  │ Binary ALU         │  │  │  │ Ternary ALU          │  │                  ║
║  │  │ ADD, SUB, MUL, DIV │  │  │  │ TADD,TSUB,TMUL,TDIV │  │                  ║
║  │  │ AND, OR, XOR, NOT  │  │  │  │ TAND,TOR,TXOR,TNOT  │  │                  ║
║  │  │ SHL, SHR, ROT      │  │  │  │ TSLL, TSRL, TSRA    │  │                  ║
║  │  │ CMP, branch        │  │  │  │ TCOMP, TB3 (3-way)  │  │                  ║
║  │  └────────────────────┘  │  │  └─────────────────────┘  │                  ║
║  │  ┌────────────────────┐  │  │  ┌─────────────────────┐  │                  ║
║  │  │ Binary FPU (IEEE)  │  │  │  │ Ternary FPU (TFloat) │  │                  ║
║  │  │ FP32, FP64         │  │  │  │ TFloat20, TFloat40   │  │                  ║
║  │  └────────────────────┘  │  │  └─────────────────────┘  │                  ║
║  │  ┌────────────────────┐  │  │  ┌─────────────────────┐  │                  ║
║  │  │ Binary Registers   │  │  │  │ Ternary Registers    │  │                  ║
║  │  │ 32 × 64-bit        │  │  │  │ 27 × 40-trit         │  │                  ║
║  │  └────────────────────┘  │  │  └─────────────────────┘  │                  ║
║  │  ┌────────────────────┐  │  │  ┌─────────────────────┐  │                  ║
║  │  │ Crypto Unit        │  │  │  │ Knowledge Unit       │  │                  ║
║  │  │ AES-NI, SHA, CRC   │  │  │  │ Morton, Frustum,     │  │                  ║
║  │  └────────────────────┘  │  │  │ Gravity, Swarm Vote  │  │                  ║
║  │  ┌────────────────────┐  │  │  └─────────────────────┘  │                  ║
║  │  │ I/O Protocol Unit  │  │  │  ┌─────────────────────┐  │                  ║
║  │  │ PCIe, USB, HDMI    │  │  │  │ Galaxy Nav Unit      │  │                  ║
║  │  │ Ethernet MAC       │  │  │  │ Star sampler, LOD    │  │                  ║
║  │  └────────────────────┘  │  │  │ LED-A*, Halting Gate │  │                  ║
║  └──────────────────────────┘  │  └─────────────────────┘  │                  ║
║                                └──────────────────────────┘                   ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                    DOMAIN CROSSING BRIDGE (DCB)                          │  ║
║  │  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────────────┐ │  ║
║  │  │ B2T converter│  │ T2B converter │  │ Format negotiator            │ │  ║
║  │  │ (binary →    │  │ (ternary →    │  │ (selects packed/unpacked,    │ │  ║
║  │  │  balanced    │  │  binary       │  │  handles sign, width match)  │ │  ║
║  │  │  ternary)    │  │  value)       │  │                              │ │  ║
║  │  └──────────────┘  └───────────────┘  └──────────────────────────────┘ │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                    UNIFIED MEMORY SUBSYSTEM                              │  ║
║  │  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────────────┐ │  ║
║  │  │ Unified L1   │  │ Unified L2    │  │ Memory Controller            │ │  ║
║  │  │ (dual-format │  │ (shared,      │  │ (binary DRAM interface,      │ │  ║
║  │  │  cache lines)│  │  BT-MESI)     │  │  ternary packing/unpacking)  │ │  ║
║  │  └──────────────┘  └───────────────┘  └──────────────────────────────┘ │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### 3.2 Key Design Principles

1. **Both domains are first-class.** Neither binary nor ternary is the "extension." Both have dedicated ALUs, FPUs, and register files.

2. **Unified fetch and decode.** A single instruction stream contains both binary and ternary instructions. The decoder identifies each instruction's domain and dispatches accordingly.

3. **Cost Router decides at decode time.** For instructions that can execute in EITHER domain (e.g., addition), the Cost Router picks the cheaper path based on a static cost table and optional dynamic profiling.

4. **Domain crossing is explicit.** Moving data between binary registers and ternary registers uses dedicated B2T/T2B instructions through the Domain Crossing Bridge. This is NOT implicit — the programmer (or compiler) sees both register files.

5. **Memory is unified.** One cache hierarchy serves both domains. Cache lines can contain binary or ternary data (tagged). External memory is binary DRAM with transparent packing for ternary data.

6. **Asymmetric power gating.** Either domain can be fully powered down when idle. A pure binary workload runs on BED only; a pure ternary workload runs on TED only. Mixed workloads use both.

---

## 4. Dual-Domain Register File

### 4.1 Binary Register File

Standard RISC-V compatible register file:

```
Binary Register File:
├── 32 registers × 64 bits (x0-x31, RISC-V ABI compatible)
├── x0 hardwired to zero
├── Read ports: 2 (rs1, rs2)
├── Write ports: 1 (rd)
├── Total storage: 2,048 bits
└── Access: 1 cycle
```

### 4.2 Ternary Register File

RISC-T compatible register file:

```
Ternary Register File:
├── 27 registers × 40 trits (t0-t26, RISC-T ABI compatible)
├── t0 hardwired to zero
├── Read ports: 2 (ts1, ts2)
├── Write ports: 1 (td)
├── Total storage: 1,080 trits (≈1,712 bits equivalent)
└── Access: 1 cycle
```

### 4.3 Cross-Domain Visibility

Binary code sees ONLY the binary registers (x0-x31). Ternary code sees ONLY the ternary registers (t0-t26). Cross-domain access requires explicit B2T or T2B instructions through the Domain Crossing Bridge (§10).

```
Binary code perspective:           Ternary code perspective:
┌──────────────────────┐          ┌──────────────────────┐
│ x0  = 0 (hardwired)  │          │ t0  = 0 (hardwired)  │
│ x1  (ra)              │          │ t1  (ra)              │
│ x2  (sp)              │          │ t2  (sp)              │
│ ...                   │          │ ...                   │
│ x31                   │          │ t26                   │
│                       │          │                       │
│ Ternary regs:         │          │ Binary regs:          │
│   NOT VISIBLE         │          │   NOT VISIBLE         │
│   (use B2T/T2B)       │          │   (use T2B/B2T)       │
└──────────────────────┘          └──────────────────────┘
```

### 4.4 Shadow Registers for Fast Crossing

For workloads that frequently cross domains, the architecture provides **3 shadow register pairs** — binary registers that mirror their ternary counterparts automatically:

```
Shadow Register Pairs:
├── x28 ↔ t24  (shadow pair 0) — auto-synchronized
├── x29 ↔ t25  (shadow pair 1) — auto-synchronized
├── x30 ↔ t26  (shadow pair 2) — auto-synchronized
└── Synchronization: hardware B2T/T2B on every write to either side
    Cost: ~3 extra gate delays per write (hidden in pipeline)
```

When binary code writes to x28, the hardware automatically converts and writes to t24. When ternary code writes to t24, the hardware automatically converts and writes to x28. This eliminates explicit B2T/T2B instructions for the most frequently crossed values.

---

## 5. The Cost Router — Automatic Dispatch

### 5.1 Static Cost Table

The Cost Router uses a compile-time or decode-time lookup table to assign each instruction to its optimal domain:

```
Cost Router Static Table:
┌────────────────────┬──────────┬──────────┬──────────┐
│ Instruction Class  │ Binary   │ Ternary  │ Assigned │
│                    │ Cost     │ Cost     │ Domain   │
├────────────────────┼──────────┼──────────┼──────────┤
│ Boolean AND/OR/XOR │ 1 gate   │ 3 gates  │ BED      │
│ Bit shift/rotate   │ 0 gates  │ N/A      │ BED      │
│ Bit masking        │ 1 gate   │ 2 gates  │ BED      │
│ Popcount           │ log₂(N)  │ convert  │ BED      │
│ CRC                │ 1 gate   │ N/A      │ BED      │
│ AES round          │ 1 cycle  │ N/A      │ BED      │
│ I/O protocol       │ native   │ convert  │ BED      │
├────────────────────┼──────────┼──────────┼──────────┤
│ Negation           │ N gates  │ 0 gates  │ TED      │
│ 3-way comparison   │ 2 instr  │ 1 instr  │ TED      │
│ 3-way branch       │ 2 instr  │ 1 instr  │ TED      │
│ Sign detection     │ extract  │ read MST │ TED      │
│ Absolute value     │ branch   │ wire swap│ TED      │
│ Knowledge state    │ 2 bits   │ 1 trit   │ TED      │
│ SIMT predication   │ 2 pass   │ 1 pass   │ TED      │
│ Frustum cull       │ FP cmp   │ TCOMP    │ TED      │
│ Morton encode      │ interl.  │ wiring   │ TED      │
│ Semantic gravity   │ table    │ TGRAV    │ TED      │
│ Swarm vote         │ reduce   │ TVOTE    │ TED      │
├────────────────────┼──────────┼──────────┼──────────┤
│ Integer addition   │ 6T/bit   │ 28T/trit │ EITHER   │
│ Integer multiply   │ O(N²)    │ O(N²)    │ EITHER   │
│ FP add/mul         │ 5 stage  │ 5 stage  │ EITHER   │
│ Division           │ iter.    │ iter.    │ EITHER   │
│ Memory load/store  │ 1 cycle  │ 1 cycle  │ EITHER   │
└────────────────────┴──────────┴──────────┴──────────┘
```

For **EITHER** operations, the router assigns based on where the operands already reside. If both source registers are binary → execute in BED. If ternary → execute in TED. This avoids unnecessary domain crossings.

### 5.2 Dynamic Cost Profiling (Optional)

For advanced implementations, the Cost Router can track per-domain utilization and route **EITHER** operations to the less busy domain:

```
Dynamic Profiling Registers (CSRs):
├── bt_bed_util: Binary domain utilization (0-100%, 7-bit counter)
├── bt_ted_util: Ternary domain utilization (0-100%, 2-tryte counter)
├── bt_dcb_stalls: Domain crossing stalls (saturation counter)
├── bt_route_overrides: Dynamic overrides this epoch (counter)
└── bt_epoch: Profiling epoch length (configurable, default 1024 cycles)

Routing rule for EITHER instructions:
  IF operands in binary regs → BED
  ELIF operands in ternary regs → TED
  ELIF bt_bed_util > bt_ted_util + threshold → TED (load balance)
  ELIF bt_ted_util > bt_bed_util + threshold → BED (load balance)
  ELSE → BED (default to binary for EITHER, since memory is binary)
```

### 5.3 Compiler Hints

The instruction encoding includes a 1-bit **domain hint** that the compiler can set:

```
Domain hint (1 bit in instruction encoding):
  0 = let hardware decide (Cost Router)
  1 = force specified domain

For EITHER instructions with hint=0, the Cost Router decides.
For EITHER instructions with hint=1, the compiler overrides.
BED-only and TED-only instructions ignore the hint.
```

This gives the compiler the ability to override the Cost Router when it has global knowledge (e.g., it knows a long sequence of operations should all stay in one domain to avoid crossing overhead).

---

## 6. Binary Execution Domain (BED)

### 6.1 Design Lineage

The BED is a standard RISC-V RV64GC core. This is intentional — RISC-BT can execute unmodified RISC-V binaries on the BED at full speed, with zero ternary awareness required.

```
Binary Execution Domain (BED):
├── ISA: RV64GC (RISC-V 64-bit, General + Compressed)
├── Pipeline: 5-stage in-order (Fetch → Decode → Execute → Memory → Writeback)
│   Advanced: 7-stage out-of-order (optional for higher-performance targets)
├── ALU: 64-bit binary ADD, SUB, MUL, DIV, AND, OR, XOR, SHL, SHR, SRA
├── FPU: IEEE 754 FP32 + FP64 (single + double precision)
├── Registers: 32 × 64-bit integer + 32 × 64-bit FP
├── Extensions: M (multiply), A (atomics), F (float), D (double), C (compressed)
├── Crypto: Optional AES-NI equivalent (BED-only, ternary has no equivalent)
├── Existing software: Runs unmodified RISC-V Linux, applications, drivers
└── I/O: All protocol controllers (PCIe, USB, Ethernet) connect to BED natively
```

### 6.2 Why RISC-V for the Binary Domain

| Alternative | Why NOT |
|------------|---------|
| x86-64 | Complex decode, variable-length instructions, proprietary legacy |
| ARM | Licensing costs (the problem we're trying to avoid) |
| MIPS | Declining ecosystem |
| **RISC-V** | Open ISA, no licensing, massive ecosystem, proven in silicon |

RISC-V is the binary RISC-T — an open ISA that anyone can implement royalty-free. Using RISC-V for the BED means RISC-BT inherits the entire RISC-V software ecosystem: Linux kernel, GCC/LLVM toolchains, application libraries.

### 6.3 BED-Only Instructions

These instructions execute ONLY in the binary domain (no ternary equivalent exists or makes sense):

| Instruction | Why BED-Only |
|------------|-------------|
| `SLL, SRL, SRA` | Binary bit shifts — ternary has trit shifts but semantics differ |
| `ANDI, ORI, XORI` | Boolean immediate operations — 2-state logic |
| `SLLI, SRLI, SRAI` | Shift by immediate — binary concept |
| `LUI` | Load upper immediate — binary address construction |
| `AUIPC` | Add upper immediate to PC — binary addressing |
| `FENCE` | Memory fence — binary memory model |
| `CSRRW/S/C` | Binary CSR access — BED-specific control registers |
| `ECALL/EBREAK` | Binary system calls — OS interface (when running RISC-V OS) |
| `AES*` | Crypto — all algorithms defined for binary |
| `CRC32*` | Error detection — GF(2) polynomials |

---

## 7. Ternary Execution Domain (TED)

### 7.1 Design Lineage

The TED is a RISC-T core (from the companion spec). It implements the full RISC-T ISA with all extensions.

```
Ternary Execution Domain (TED):
├── ISA: RISC-T RT40T (40-trit ternary, base + extensions)
├── Pipeline: 5-stage (Fetch → Decode → Execute → Memory → Writeback)
├── ALU: 40-trit TADD, TSUB, TMUL, TDIV, TAND, TOR, TXOR, TNOT, TCOMP, TB3
├── FPU: TFloat20 + TFloat40 (ternary floating-point)
├── Registers: 27 × 40-trit integer + 27 × 40-trit TFP
├── Extensions: RT-M, RT-F, RT-A, RT-K (knowledge), RT-G (GPU)
├── Knowledge Unit: Morton encoder, frustum culler, LOD, LED-A*, swarm vote
├── Galaxy Navigation: Star cache, semantic gravity, halting gate
└── Free negation: TNOT = wire swap, 0 gates, 0 cycles
```

### 7.2 TED-Only Instructions

These instructions execute ONLY in the ternary domain (no binary equivalent is efficient):

| Instruction | Why TED-Only |
|------------|-------------|
| `TNOT` | Free negation (wire swap) — binary NOT is a different operation |
| `TB3` | Three-way branch — binary has no single-instruction 3-way branch |
| `TCOMP` | Three-way compare (returns trit) — binary CMP returns flags |
| `TMORTON` | Ternary Morton encode — ternary interleaving |
| `TFRUSTUM` | Ternary frustum cull — returns trit (+1/0/-1) |
| `TGRAV` | Semantic gravity — ternary force model |
| `TVOTE` | 9-chain swarm vote — ternary consensus |
| `THALT` | Halting gate — ternary convergence check |
| `TDEFEAT` | Defeasible logic — ternary strength model |
| `TATTN` | Ternary attention — attract/neutral/repel |
| All RT-K | Knowledge operations — ternary-native by design |

---

## 8. Shared Memory Subsystem

### 8.1 Unified Cache with Dual-Format Lines

The cache is shared between both domains. Each cache line is tagged with its format:

```
Cache Line Format (unified L1):
┌──────┬───────┬───────────────────────────────────────────┐
│ Tag  │ Fmt   │ Data                                       │
│ (addr│ (1 bit│ (512 bits physical storage)                │
│  tag)│  0=bin│                                            │
│      │  1=ter│                                            │
└──────┴───────┴───────────────────────────────────────────┘

When Fmt=0 (binary):
  Data = 512 bits = 64 bytes (standard cache line)
  Accessed by BED directly, by TED via B2T conversion

When Fmt=1 (ternary):
  Data = 512 bits = 256 trits (packed 2 bits per trit)
  OR: 512 bits = 320 trits (packed 5 trits per 8 bits)
  Accessed by TED directly, by BED via T2B conversion
```

### 8.2 Cache Hierarchy

```
Unified Cache Hierarchy:
├── L1I (Instruction): 32 KB, 4-way, shared format
│   ├── Binary instructions: standard 32-bit RISC-V encoding
│   └── Ternary instructions: 12-trit RISC-T encoding (24 bits packed)
├── L1D (Data): 32 KB, 4-way, dual-format lines
│   ├── Format bit per line (binary or ternary)
│   └── Cross-domain access triggers automatic conversion
├── L2 (Unified): 256 KB - 1 MB, 8-way, dual-format
│   ├── Inclusive of L1
│   └── BT-MESI coherence (§13)
└── DRAM Interface: Binary DDR4/DDR5 (all data stored as binary in DRAM)
    ├── Binary data: read/write directly
    └── Ternary data: packed (5T → 8 bits) on write, unpacked on read
```

### 8.3 Address Space

Both domains share a single physical address space. Virtual memory is managed by the binary domain (RISC-V page tables), with ternary data pages tagged via a page table extension:

```
Page Table Entry extension for RISC-BT:
Standard RISC-V PTE + 2 new bits:
  Bit 62: T-page (0 = binary data, 1 = ternary-packed data)
  Bit 61: T-exec (0 = binary instructions, 1 = ternary instructions)

This allows the OS to track which pages contain ternary data/code,
enabling the memory controller to apply the correct packing/unpacking.
```

---

## 9. Instruction Encoding — Unified Format

### 9.1 Encoding Strategy

RISC-BT uses the RISC-V 32-bit instruction encoding as the base, with ternary instructions encoded in a reserved major opcode space:

```
Instruction Format:
┌──────────────────────────────────────────────────────┐
│ Standard RISC-V 32-bit encoding                       │
│ Bits [6:0] = opcode                                   │
│                                                       │
│ Binary instructions: standard RISC-V opcodes          │
│   0000011 = LOAD                                      │
│   0100011 = STORE                                     │
│   0110011 = OP (arithmetic)                           │
│   1100011 = BRANCH                                    │
│   ...                                                 │
│                                                       │
│ Ternary instructions: CUSTOM-3 opcode space (1111011) │
│   Bits [14:12] = ternary opcode group (3 trits via    │
│                   binary encoding)                     │
│   Bits [31:15] = ternary operands (packed)            │
│                                                       │
│ Domain crossing: CUSTOM-2 opcode space (1011011)      │
│   B2T, T2B, TSYNC instructions                        │
└──────────────────────────────────────────────────────┘
```

### 9.2 Why 32-bit Binary Encoding for Both

Using RISC-V's 32-bit binary encoding even for ternary instructions means:

1. **Existing binary toolchains work.** Assemblers, linkers, and loaders handle the instruction stream without modification. Ternary instructions look like "custom" RISC-V instructions.

2. **Instruction memory is binary.** No need for ternary instruction memory or ternary I-cache. The ternary operations are encoded in binary for storage/fetch, then decoded into ternary domain operations.

3. **Mixed code is seamless.** Binary and ternary instructions interleave freely in the same instruction stream. No mode switches, no domain transitions for fetch/decode.

4. **RISC-V software compatibility.** Existing RISC-V operating systems can load and manage RISC-BT executables without kernel changes — ternary instructions are just "custom" opcodes.

### 9.3 Ternary Instruction Encoding Detail

```
Ternary instruction (encoded in RISC-V CUSTOM-3, 32 bits):

 31      25 24   20 19   15 14  12 11    7 6     0
┌─────────┬───────┬───────┬──────┬───────┬────────┐
│ funct7   │  ts2  │  ts1  │funct3│  td   │1111011 │
│ (7 bits) │(5 bit)│(5 bit)│(3 b) │(5 bit)│CUSTOM-3│
└─────────┴───────┴───────┴──────┴───────┴────────┘

Fields:
  opcode [6:0]   = 1111011 (CUSTOM-3, reserved for RISC-BT ternary ops)
  td [11:7]      = destination ternary register index (t0-t26, 5 bits for 27 regs)
  funct3 [14:12] = ternary operation group:
                    000 = TARITH (TADD, TSUB, TMUL, TDIV)
                    001 = TLOGIC (TAND, TOR, TXOR, TNOT)
                    010 = TCMP   (TCOMP, TB3)
                    011 = TMEM   (TLT, TST — ternary load/store)
                    100 = TKNOW  (TMORTON, TFRUSTUM, TGRAV, TVOTE)
                    101 = TFLOAT (TFADD, TFMUL, TFDIV)
                    110 = TCONV  (B2T, T2B, TPACK, TUNPACK)
                    111 = TMISC  (THALT, TDEFEAT, TATTN)
  ts1 [19:15]    = source ternary register 1
  ts2 [24:20]    = source ternary register 2
  funct7 [31:25] = specific operation within group
```

---

## 10. Domain Crossing Instructions

### 10.1 Explicit Crossing

```
Domain Crossing Instructions (CUSTOM-2 opcode space):

B2T  td, xs      ; Binary-to-Ternary: convert binary register xs to ternary register td
                  ; Interprets 64-bit binary value as integer, converts to 40-trit balanced ternary
                  ; Latency: 3 cycles (conversion logic)

T2B  xd, ts      ; Ternary-to-Binary: convert ternary register ts to binary register xd
                  ; Converts 40-trit balanced ternary to 64-bit binary integer
                  ; Latency: 3 cycles (conversion logic)

B2TF td, fs      ; Binary-to-Ternary Float: IEEE FP64 → TFloat40
                  ; Latency: 5 cycles (exponent + significand conversion)

T2BF fd, ts      ; Ternary-to-Binary Float: TFloat40 → IEEE FP64
                  ; Latency: 5 cycles (exponent + significand conversion)

TSYNC             ; Synchronize all shadow register pairs (§4.4)
                  ; Forces all 3 shadow pairs to reflect latest writes
                  ; Used as a fence between mixed binary/ternary code
```

### 10.2 Crossing Cost Budget

| Crossing Type | Latency | Energy | When to Use |
|--------------|---------|--------|-------------|
| Shadow register (auto) | 0 extra cycles (pipelined) | ~3 gate delays | Frequently crossed values (loop counters, addresses) |
| Explicit B2T/T2B | 3 cycles | ~50 gates | Occasional integer transfers |
| Float B2TF/T2BF | 5 cycles | ~120 gates | Float format conversion (rare) |
| Memory (read in other format) | 1 extra cycle | ~30 gates | Cross-domain memory access via cache |

**Design goal:** Minimize crossings. The compiler should keep computation in one domain as long as possible. Shadow registers handle the common case (a few shared values). Explicit crossings handle the rest.

---

## 11. Hybrid Pipeline Specification

### 11.1 5-Stage Hybrid Pipeline

```
Stage 1: FETCH (shared)
  ├── Read instruction from unified I-cache (binary encoding)
  └── Output: 32-bit instruction word

Stage 2: DECODE + ROUTE (shared)
  ├── Decode instruction fields (opcode, registers, immediates)
  ├── Cost Router: determine domain (BED or TED)
  │   ├── If opcode ∈ BED-only → dispatch to BED
  │   ├── If opcode ∈ TED-only → dispatch to TED
  │   └── If opcode ∈ EITHER → consult cost table + operand location
  └── Output: decoded operation + domain assignment

Stage 3: EXECUTE (split)
  ├── BED path: binary ALU/FPU/crypto
  │   ├── Read binary registers (x0-x31)
  │   ├── Execute binary operation
  │   └── Generate binary result
  └── TED path: ternary ALU/FPU/knowledge
      ├── Read ternary registers (t0-t26)
      ├── Execute ternary operation
      └── Generate ternary result

Stage 4: MEMORY (shared)
  ├── Unified data cache access
  ├── Format-aware: reads binary or ternary cache lines
  └── Cross-domain memory access: automatic conversion

Stage 5: WRITEBACK (split)
  ├── BED path: write to binary register file (x0-x31)
  └── TED path: write to ternary register file (t0-t26)
  └── Shadow sync: if destination is shadow pair, update counterpart
```

### 11.2 Dual-Issue (Advanced)

For higher-performance implementations, the hybrid pipeline can dual-issue — one BED instruction and one TED instruction per cycle, simultaneously:

```
Dual-Issue Hybrid Pipeline:
  Cycle N:
    BED: executes binary instruction I_b
    TED: executes ternary instruction I_t (simultaneously)

  Requirements:
    - No data dependency between I_b and I_t
    - No shadow register conflict
    - Both I_b and I_t are ready (operands available)

  Throughput: 2 instructions per cycle (1 binary + 1 ternary)
  This is FREE parallelism — binary and ternary hardware are independent.
```

This is a key advantage of the hybrid design: the BED and TED are physically separate execution units. Dual-issuing binary + ternary costs nothing extra — the hardware is already there for both domains.

### 11.3 Branch Prediction — Dual-Domain

```
Branch Prediction:
├── Binary branches (BED): standard 2-bit saturating counter BTB
│   ├── Predicted: taken / not-taken
│   └── Standard RISC-V branch prediction
├── Ternary branches (TED): 3-trit saturating counter BTB
│   ├── Predicted: positive-target / neutral-target / negative-target
│   ├── 3 targets stored per BTB entry (vs binary's 1 target)
│   └── Misprediction: 1/3 vs binary's 1/2 (33% vs 50% random miss)
└── Cross-domain branches: predict based on source domain
    └── B→T transition: use ternary predictor from crossing point
    └── T→B transition: use binary predictor from crossing point
```

---

## 12. Hybrid GPU Architecture (BT-G)

### 12.1 Hybrid Compute Unit

The GPU version of RISC-BT places binary and ternary SIMT lanes side-by-side:

```
Hybrid Compute Unit (HCU):
├── Binary SIMT Block:
│   ├── 32 binary lanes (standard GPU warp)
│   ├── Binary ALU per lane (INT32 + FP32)
│   ├── Binary register file (256 × 32-bit per lane)
│   └── Use for: texture sampling, standard shader math, I/O
├── Ternary SIMT Block:
│   ├── 9 ternary lanes (3² warp, from RISC-T spec)
│   ├── Ternary ALU per lane (40-trit + TFloat)
│   ├── Ternary register file (27 × 40-trit per lane)
│   └── Use for: knowledge ops, 3-way branching, Galaxy navigation
├── Shared Resources:
│   ├── LDS/LTS (dual-format local memory)
│   ├── L1 cache (dual-format lines)
│   ├── Wavefront scheduler (manages both warp types)
│   └── Domain Crossing Unit (in-CU B2T/T2B)
└── Dispatch Logic:
    ├── Binary shader → binary SIMT block
    ├── Ternary shader → ternary SIMT block
    └── Hybrid shader → split instructions across blocks
```

### 12.2 Hybrid Shader Model

Shaders in RISC-BT can contain both binary and ternary instructions. The GPU scheduler dispatches each instruction to the appropriate SIMT block:

```
Hybrid shader example (knowledge-enhanced rendering):

; Binary domain: standard vertex transform
LW    x5, vertex_position(x3)    ; Binary: load vertex (binary memory)
FMUL  f1, f2, f3                 ; Binary: matrix multiply (IEEE FP)
FSW   f1, transformed_pos(x3)    ; Binary: store result

; Ternary domain: knowledge-enhanced lighting
B2T   t5, x5                     ; Cross: convert position to ternary
TMORTON t6, t5, t7, t8           ; Ternary: spatial index
TFRUSTUM t9, t6, t10             ; Ternary: visibility test (3-way)
TB3   t9, .visible, .edge, .hidden ; Ternary: 3-way branch (1 pass!)
.visible:
  TGRAV t11, t12, t13, t14       ; Ternary: semantic gravity lookup
  T2B   x10, t11                 ; Cross: convert result to binary
  FMUL  f4, f5, f10              ; Binary: apply to lighting equation
```

### 12.3 Rasterizer — Binary Geometry, Ternary Tests

The rasterizer can be hybrid: standard binary triangle setup with ternary edge function evaluation:

```
Hybrid Rasterizer:
├── Triangle Setup: Binary (standard FP32 math — well-optimized, proven)
│   ├── Edge equation coefficients computed in binary FPU
│   └── Bounding box computed in binary integer
├── Edge Evaluation: Ternary (native +1/0/-1 result)
│   ├── Convert edge coefficients: B2TF (one-time per triangle)
│   ├── Evaluate per pixel: ternary TCOMP → trit result
│   ├── Inside/edge/outside in 1 instruction (no epsilon)
│   └── 3×3 tile = 9 pixels = 1 ternary warp
└── Fragment Processing: domain depends on shader
    ├── Standard PBR lighting → binary (well-optimized shaders)
    └── Knowledge-enhanced → ternary (Galaxy lookups, semantic)
```

---

## 13. Hybrid Cache Coherence (BT-MESI)

### 13.1 Extended MESI with Domain Bits

Standard MESI has 4 states. BT-MESI extends each state with a format bit:

```
BT-MESI Cache Line States:

Standard MESI states:
  M (Modified):  line is dirty, only copy
  E (Exclusive): line is clean, only copy
  S (Shared):    line is clean, multiple copies
  I (Invalid):   line is not valid

Format extension (1 bit per line):
  B (Binary):    data is in binary format
  T (Ternary):   data is in ternary-packed format

Combined states: M-B, M-T, E-B, E-T, S-B, S-T, I
  Total: 7 states (vs MESI's 4)

State transitions on cross-domain access:
  BED reads a T-format line → convert T→B, cache in B format (or dual-cache)
  TED reads a B-format line → convert B→T, cache in T format (or dual-cache)
  Both domains read same line → promote to S state with both formats cached
```

### 13.2 Format Conversion at Cache Boundary

When the BED needs to read a ternary cache line (or vice versa), conversion happens in the cache controller — NOT in the pipeline:

```
Cross-domain cache access:
1. BED issues load (binary address)
2. L1 lookup → hit on T-format line
3. Cache controller activates T2B converter
4. Converted binary data forwarded to BED pipeline
5. If line frequently accessed cross-domain:
   → Cache stores BOTH formats (double-caching, trades space for speed)
   → Coherence maintained: write to one format invalidates the other

Cost: 1 extra cycle for first cross-domain access
      0 extra cycles for double-cached lines (common case)
```

---

## 14. I/O and Protocol Handling

### 14.1 All I/O Through Binary Domain

All external I/O protocols are binary. The BED handles all protocol logic natively:

```
I/O Architecture:
├── PCIe Controller → BED (binary, no conversion)
├── USB Controller → BED
├── Ethernet MAC → BED
├── HDMI/DP → BED (frame buffer read from unified memory)
├── SPI/I2C/UART → BED
├── DDR Controller → BED (binary DRAM interface)
└── GPIO → BED

Ternary domain I/O access:
  TED code does NOT directly access I/O.
  Instead: TED writes result to shared memory →
           BED reads from shared memory → BED sends via I/O protocol.
  Or: TED writes to shadow register → BED reads from shadow pair.

This is the CORRECT design: I/O protocols are binary by definition.
The TED should not waste transistors reimplementing binary protocols.
```

### 14.2 Future: Ternary I/O (PAM-3)

When native ternary I/O becomes available (PAM-3 signaling, from the RISC-T networking spec §22), the TED can drive ternary links directly:

```
Future Ternary I/O:
├── TLink (ternary chip-to-chip) → TED native
├── TNoC (ternary network-on-chip) → TED native
├── Ternary display (3-level per subpixel) → TED native
└── PAM-3 Ethernet (ternary-native networking) → TED native

The hybrid architecture is ready for this: when ternary I/O exists,
it connects directly to the TED without going through BED.
```

---

## 15. Power Management — Asymmetric Domain Gating

### 15.1 Independent Power Domains

The BED and TED are in separate power domains with independent gating:

```
Power States:
┌──────────────────────────────────────┐
│            BED Power     TED Power   │
│ Mode       Domain        Domain      │
├──────────────────────────────────────┤
│ Full       ON             ON         │  Mixed workload
│ Binary     ON             OFF        │  Pure binary (RISC-V code)
│ Ternary    OFF            ON         │  Pure ternary (RISC-T code)
│ Idle       RETENTION      RETENTION  │  Low power, registers preserved
│ Sleep      OFF            OFF        │  Deep sleep, state lost
└──────────────────────────────────────┘

Transition latency:
  ON → OFF:      1 cycle (gate clock)
  OFF → ON:      ~100 cycles (power-up + pipeline flush)
  ON → RETENTION: 10 cycles (save critical state)
  RETENTION → ON: 20 cycles (restore state)
```

### 15.2 Workload-Adaptive Gating

The Cost Router tracks domain utilization (§5.2) and can trigger power gating:

```
Auto-gating logic:
  IF bt_ted_util == 0 for bt_gate_threshold cycles → TED → RETENTION
  IF bt_bed_util == 0 for bt_gate_threshold cycles → BED → RETENTION
  IF ternary instruction decoded while TED in RETENTION → wake TED (20 cycles)
  IF binary instruction decoded while BED in RETENTION → wake BED (20 cycles)

  bt_gate_threshold: configurable, default 4096 cycles (~2µs at 2GHz)
```

### 15.3 Energy Comparison

```
Workload Energy Analysis:

Pure binary workload (e.g., compiling code):
  RISC-BT: BED ON, TED OFF → same energy as pure RISC-V chip
  No penalty for having the ternary domain — it's gated off

Pure ternary workload (e.g., K3D Galaxy reasoning):
  RISC-BT: BED OFF, TED ON → same energy as pure RISC-T chip
  No penalty for having the binary domain — it's gated off

Mixed workload (e.g., K3D on Linux with I/O):
  RISC-BT: BED ON (I/O, OS), TED ON (reasoning, knowledge)
  Both domains active, but each operation runs on cheaper hardware
  Net energy: LOWER than either pure domain doing both tasks

This is the hybrid advantage: you never pay for the domain you're not using.
```

---

## 16. K3D Integration (BT-K Extension)

### 16.1 K3D Workload Split

K3D's composed head pipeline naturally splits across domains:

```
K3D Pipeline on RISC-BT:

Python boot + I/O (BED — binary domain):
  ├── System startup, file I/O, network
  ├── Keyboard/display handling
  ├── OS-level resource management
  └── ~200 lines of Python → compiled RISC-V binary

Galaxy navigation (TED — ternary domain):
  ├── Morton encode/decode (TMORTON — free wiring)
  ├── LED-A* pathfinding (ternary edge weights)
  ├── Frustum culling (TFRUSTUM — trit result)
  ├── LOD selection (TLOD)
  ├── Semantic gravity (TGRAV)
  ├── Nine-chain swarm vote (TVOTE)
  ├── Halting gate (THALT)
  └── Star creation/lookup (Galaxy ops)

Binary Bridge (DCB — domain crossing):
  ├── User input: binary keyboard → B2T → ternary query vector
  ├── Answer output: ternary result → T2B → binary display text
  └── Network I/O: binary packets → B2T → ternary knowledge → T2B → binary response
```

### 16.2 Composed Head Pipeline — Hybrid Assembly

```asm
; K3D Composed Head Pipeline on RISC-BT Hybrid
; Binary domain handles I/O, ternary domain handles reasoning

; === BINARY DOMAIN: receive query ===
LW      x5, keyboard_buffer(x3)    ; BED: read keyboard input (binary I/O)
CALL    tokenize                    ; BED: tokenize string (binary string ops)

; === DOMAIN CROSSING: binary → ternary ===
B2T     t5, x5                     ; Convert query token to ternary register

; === TERNARY DOMAIN: composed head pipeline ===
TMORTON t22, t5, t6, t7            ; TED: spatial index
TLT     t8, t22, 0(t3)             ; TED: load Galaxy neighborhood
TCOMP   t26, t8, t9                ; TED: navigate
TB3     t26, .attract, .neutral, .repel  ; TED: 3-way branch (1 cycle!)
TFRUSTUM t10, t8, t11              ; TED: frustum cull
TLOD    t13, t14, t15              ; TED: detail level
TVOTE   t16, t5, t6, t7, t8, t9, t10, t11, t12, t13  ; TED: swarm vote
THALT   t17, t16, 121              ; TED: convergence check
TBNE    t17, t0, .converged        ; TED: loop if not converged

.converged:
; === DOMAIN CROSSING: ternary → binary ===
T2B     x10, t17                   ; Convert ternary answer to binary

; === BINARY DOMAIN: emit result ===
SW      x10, display_buffer(x3)    ; BED: write to display (binary I/O)
CALL    render_text                 ; BED: render answer (binary display driver)
```

**This is the hybrid advantage in action:** The I/O is binary (keyboard, display, network — where binary is cheaper). The reasoning is ternary (navigation, comparison, voting — where ternary is cheaper). Domain crossings happen at the natural boundaries, not in the hot loop.

---

## 17. Compiler Model — Domain-Aware Compilation

### 17.1 Language Extensions

```c
// C language extension for RISC-BT

// Ternary type qualifiers
_Ternary int    t_value;    // 40-trit balanced ternary integer
_Ternary float  t_float;    // TFloat40 ternary floating-point
_Ternary _Bool  t_flag;     // 1 trit: +1, 0, -1

// Domain annotation (optional — compiler can infer)
__attribute__((domain(binary)))   void io_handler(void);     // Force BED
__attribute__((domain(ternary)))  void galaxy_navigate(void); // Force TED
__attribute__((domain(auto)))     int  compute(int a, int b); // Cost Router decides

// Three-way branch (language-level support for TB3)
switch_ternary(condition) {
    case +1: /* positive path */   break;
    case  0: /* neutral path */    break;
    case -1: /* negative path */   break;
}

// Implicit domain crossing
int binary_val = 42;
_Ternary int ternary_val = (_Ternary int)binary_val;  // Compiler emits B2T
int back = (int)ternary_val;                           // Compiler emits T2B
```

### 17.2 Compiler Domain Assignment Algorithm

```
Domain Assignment (compile-time):

1. Build data flow graph of the function
2. For each operation node:
   a. If BED-only → assign to BED
   b. If TED-only → assign to TED
   c. If EITHER → assign to domain of majority of input operands
3. Minimize domain crossings:
   a. Count crossings in current assignment
   b. For each EITHER node adjacent to a crossing:
      Try moving it to the other domain
      If crossings decrease → keep the move
   c. Repeat until no improvement
4. Insert B2T/T2B at every remaining crossing edge
5. Assign shadow registers (§4.4) to the 3 most-crossed values
6. Emit unified instruction stream (binary + ternary interleaved)
```

### 17.3 ABI for Mixed Functions

```
RISC-BT Calling Convention:

Binary function → binary arguments in x10-x17 (a0-a7), return in x10
Ternary function → ternary arguments in t5-t13 (ta0-ta8), return in t5
Mixed function → first 4 args in x10-x13, remaining in t5-t8, return based on type

Stack: binary stack (sp = x2), shared between domains
       Ternary values on stack are stored packed (5T → 8 bits)

Cross-domain call convention:
  Binary calling ternary: caller saves binary regs, callee uses ternary regs
  Ternary calling binary: caller saves ternary regs, callee uses binary regs
  Shadow registers (x28-x30 ↔ t24-t26) are caller-saved on both sides
```

---

## 18. HDL Reference Implementation (Pseudocode)

### 18.1 Cost Router

```verilog
// RISC-BT Cost Router — Domain dispatch logic
// Prior art: Automatic binary/ternary dispatch based on per-instruction cost table.

module cost_router (
    input  [31:0] instruction,        // Decoded instruction
    input  [6:0]  opcode,             // RISC-V opcode field
    input  [2:0]  funct3,             // Function select
    input  [6:0]  funct7,             // Extended function
    input  [6:0]  bed_util,           // Binary domain utilization (0-100)
    input  [6:0]  ted_util,           // Ternary domain utilization (0-100)
    output reg     domain,            // 0 = BED, 1 = TED
    output reg     valid_bed,         // Instruction valid for BED
    output reg     valid_ted          // Instruction valid for TED
);
    parameter CUSTOM_3 = 7'b1111011;  // Ternary opcode space
    parameter CUSTOM_2 = 7'b1011011;  // Domain crossing space
    parameter UTIL_THRESHOLD = 7'd20; // Load-balance threshold

    always @(*) begin
        // Default: binary domain
        domain = 1'b0;
        valid_bed = 1'b0;
        valid_ted = 1'b0;

        case (opcode)
            // Pure ternary instructions → TED only
            CUSTOM_3: begin
                domain = 1'b1;
                valid_ted = 1'b1;
            end

            // Domain crossing instructions → special handling
            CUSTOM_2: begin
                // B2T: source is BED, dest is TED — execute in DCB
                valid_bed = 1'b1;
                valid_ted = 1'b1;
            end

            // Standard RISC-V binary instructions → BED only
            7'b0110011, // OP (binary arithmetic)
            7'b0010011, // OP-IMM
            7'b0000011, // LOAD
            7'b0100011, // STORE
            7'b1100011, // BRANCH
            7'b1101111, // JAL
            7'b1100111, // JALR
            7'b0110111, // LUI
            7'b0010111: // AUIPC
            begin
                domain = 1'b0;
                valid_bed = 1'b1;
            end

            // EITHER instructions: consult utilization
            default: begin
                valid_bed = 1'b1;
                valid_ted = 1'b1;
                // Route to less busy domain
                if (ted_util + UTIL_THRESHOLD < bed_util)
                    domain = 1'b1;  // TED is less busy
                else
                    domain = 1'b0;  // Default to BED
            end
        endcase
    end

    // Gate count: ~200 MOSFETs (opcode decode + comparator)
    // Delay: 2 gate levels (fits in decode stage)
    // Power: negligible (combinational, no state)
endmodule
```

### 18.2 Domain Crossing Bridge (B2T)

```verilog
// RISC-BT Binary-to-Ternary Converter
// Prior art: Hardware conversion from 64-bit two's complement to 40-trit balanced ternary.

module binary_to_ternary (
    input  [63:0] binary_in,       // 64-bit two's complement integer
    output trit [39:0] ternary_out  // 40-trit balanced ternary
);
    // Algorithm: repeated division by 3 with balanced remainder
    // remainder ∈ {-1, 0, +1} instead of {0, 1, 2}
    //
    // For each trit position i (LST to MST):
    //   r = value mod 3
    //   if r == 0 → trit[i] = 0,  value = value / 3
    //   if r == 1 → trit[i] = +1, value = (value - 1) / 3
    //   if r == 2 → trit[i] = -1, value = (value + 1) / 3  (borrow)

    reg signed [63:0] remaining;
    reg [1:0] remainder;
    integer i;

    always @(*) begin
        remaining = $signed(binary_in);
        for (i = 0; i < 40; i = i + 1) begin
            remainder = remaining % 3;
            case (remainder)
                2'd0: begin
                    ternary_out[i] = TRIT_ZERO;
                    remaining = remaining / 3;
                end
                2'd1: begin
                    ternary_out[i] = TRIT_POS;
                    remaining = (remaining - 1) / 3;
                end
                2'd2: begin
                    ternary_out[i] = TRIT_NEG;
                    remaining = (remaining + 1) / 3;
                end
                default: begin
                    ternary_out[i] = TRIT_ZERO;
                    remaining = remaining / 3;
                end
            endcase
        end
    end

    // Pipelined implementation: 3 cycles (13-14 trits per cycle)
    // Gate count: ~800 MOSFETs (40 dividers-by-3 chained)
    // Combinational delay: ~120 gate levels (too deep for 1 cycle → pipeline)
endmodule
```

### 18.3 Shadow Register Pair

```verilog
// RISC-BT Shadow Register Pair — Auto-synchronized binary ↔ ternary
// Prior art: Hardware-synchronized dual-domain register with automatic format conversion.

module shadow_register_pair (
    input  clk,
    input  rst,

    // Binary side
    input  [63:0] bin_write_data,
    input         bin_write_en,
    output [63:0] bin_read_data,

    // Ternary side
    input  trit [39:0] ter_write_data,
    input              ter_write_en,
    output trit [39:0] ter_read_data
);
    reg [63:0] bin_reg;
    reg trit [39:0] ter_reg;

    // Binary-to-ternary converter
    wire trit [39:0] bin_to_ter;
    binary_to_ternary b2t (.binary_in(bin_write_data), .ternary_out(bin_to_ter));

    // Ternary-to-binary converter
    wire [63:0] ter_to_bin;
    ternary_to_binary t2b (.ternary_in(ter_write_data), .binary_out(ter_to_bin));

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            bin_reg <= 64'b0;
            ter_reg <= {40{TRIT_ZERO}};
        end else begin
            if (bin_write_en) begin
                bin_reg <= bin_write_data;       // Write binary side
                ter_reg <= bin_to_ter;           // Auto-sync ternary side
            end else if (ter_write_en) begin
                ter_reg <= ter_write_data;       // Write ternary side
                bin_reg <= ter_to_bin;           // Auto-sync binary side
            end
            // If both write simultaneously: binary wins (arbitrary tie-break)
        end
    end

    assign bin_read_data = bin_reg;
    assign ter_read_data = ter_reg;

    // Gate count: ~1,800 MOSFETs (2 converters + 2 register files + MUX)
    // Delay: converter is pipelined — sync latency hidden in writeback stage
    // Power: converters only active when write_en asserted (clock-gated)
endmodule
```

### 18.4 Hybrid Pipeline Dispatch

```verilog
// RISC-BT Hybrid Pipeline Dispatch — Dual-issue binary + ternary
// Prior art: Simultaneous binary and ternary instruction issue from unified instruction stream.

module hybrid_dispatch (
    input  clk,
    input  [31:0] instr_0,      // First instruction from fetch
    input  [31:0] instr_1,      // Second instruction (for dual-issue)
    input         dual_issue_en, // Dual-issue capable?

    // Domain assignments from Cost Router
    input         domain_0,      // 0=BED, 1=TED for instr_0
    input         domain_1,      // 0=BED, 1=TED for instr_1

    // Outputs to execution domains
    output reg [31:0] bed_instr,
    output reg        bed_valid,
    output reg [31:0] ted_instr,
    output reg        ted_valid
);
    always @(posedge clk) begin
        bed_valid <= 1'b0;
        ted_valid <= 1'b0;

        if (dual_issue_en && domain_0 != domain_1) begin
            // Dual-issue: one binary + one ternary, simultaneously
            if (domain_0 == 1'b0) begin
                bed_instr <= instr_0; bed_valid <= 1'b1;
                ted_instr <= instr_1; ted_valid <= 1'b1;
            end else begin
                bed_instr <= instr_1; bed_valid <= 1'b1;
                ted_instr <= instr_0; ted_valid <= 1'b1;
            end
        end else begin
            // Single-issue: dispatch to assigned domain
            if (domain_0 == 1'b0) begin
                bed_instr <= instr_0; bed_valid <= 1'b1;
            end else begin
                ted_instr <= instr_0; ted_valid <= 1'b1;
            end
        end
    end

    // Key property: when workload is mixed binary+ternary, throughput doubles
    // compared to either pure RISC-V or pure RISC-T executing the same mix.
    // The binary and ternary execution units operate in parallel — no contention.
endmodule
```

---

## 19. Reference Designs

### 19.1 RISC-BT Minimal (BT-Mini)

Target: FPGA prototyping. Minimal viable hybrid processor.

```
BT-Mini Specifications:
├── BED: RV32I (32-bit RISC-V, integer only)
│   ├── 32 × 32-bit registers
│   ├── 3-stage pipeline
│   └── ~15K gates
├── TED: RT20T-Mini (20-trit ternary, base ISA)
│   ├── 9 × 20-trit registers
│   ├── 3-stage pipeline
│   └── ~15K gates (ternary logic cells)
├── DCB: Minimal B2T/T2B pair
│   └── ~2K gates
├── Cache: Unified 4 KB L1 (dual-format)
│   └── ~5K gates
├── Total: ~37K gates
├── FPGA: Lattice ECP5 (open toolchain)
├── Clock: ~25 MHz
├── Estimated power: ~100 mW
└── Purpose: prove hybrid concept, ISA validation
```

### 19.2 RISC-BT Application (BT-App)

Target: 65nm ASIC. Application processor with full hybrid capabilities.

```
BT-App Specifications:
├── BED: RV64GC (64-bit RISC-V, full standard extensions)
│   ├── 32 × 64-bit integer + 32 × 64-bit FP registers
│   ├── 5-stage pipeline, in-order
│   ├── IEEE 754 FP32 + FP64
│   └── ~250K gates
├── TED: RT40T-K3D (40-trit ternary, knowledge extensions)
│   ├── 27 × 40-trit integer + 27 × 40-trit TFP registers
│   ├── 5-stage pipeline
│   ├── TFloat20 + TFloat40
│   ├── Knowledge unit (Morton, Frustum, Gravity, Vote, Halt)
│   └── ~300K gates
├── DCB: Full bridge with 3 shadow pairs
│   └── ~10K gates
├── Cache: 32 KB L1I + 32 KB L1D + 256 KB L2 (all dual-format)
├── Memory: DDR4 interface (binary, ternary packing in controller)
├── I/O: UART, SPI, I2C, GPIO (BED-native)
├── Total: ~800K gates
├── Clock: ~500 MHz (65nm)
├── Estimated area: ~15 mm² at 65nm
├── Estimated power: ~3W
└── Purpose: first hybrid silicon, Linux-capable, K3D-capable
```

### 19.3 RISC-BT Galaxy (BT-Galaxy)

Target: 28nm. Multi-core hybrid with GPU capabilities.

```
BT-Galaxy Specifications:
├── Cores: 9 BT-App cores (1 binary + 1 ternary domain each)
│   ├── Total: 9 BED + 9 TED execution domains
│   ├── Each core: dual-issue (1 binary + 1 ternary per cycle)
│   └── Peak: 18 instructions/cycle across all cores
├── GPU: 1 Hybrid Compute Unit (HCU, §12)
│   ├── 32 binary SIMT lanes
│   ├── 9 ternary SIMT lanes
│   ├── Shared LDS/LTS
│   └── Knowledge Star Sampler
├── Shared: 1 MB L2 (dual-format, BT-MESI)
├── Memory: DDR5 interface (binary, ternary packing)
├── Network: Ternary NoC (TNoC) connecting cores
├── I/O: PCIe Gen4, USB 3.0, Gigabit Ethernet (all BED-native)
├── Total: ~15M gates
├── Clock: ~1-2 GHz
├── Estimated area: ~60 mm² at 28nm
├── Estimated power: ~20W
└── Purpose: full K3D House + Galaxy system, workstation-class
```

---

## 20. Manufacturing Targets

### 20.1 FPGA Prototyping (NOW)

Same as RISC-T spec — all FPGA targets support RISC-BT. The binary domain uses standard RISC-V soft cores (well-proven), the ternary domain uses 2-bit-per-trit emulation.

| FPGA | Binary Domain | Ternary Domain | Total | Cost |
|------|--------------|----------------|-------|------|
| Lattice ECP5 | RV32I (proven) | RT20T-Mini (emulated) | BT-Mini | $30-$80 |
| Xilinx Artix-7 | RV64I | RT20T | BT-Embedded | $100-$300 |
| Xilinx Kintex-7 | RV64GC | RT40T-K3D | BT-App | $500-$2K |

### 20.2 ASIC (6-24 months)

| Target | PDK | Binary Domain | Ternary Domain | Cost |
|--------|-----|--------------|----------------|------|
| SkyWater 130nm | SKY130 (FREE) | RV32I | RT20T-Mini | ~$1K (packaging only) |
| GF 180nm | GF180MCU (FREE) | RV32I | RT20T-Mini | ~$1K (packaging only) |
| IHP 130nm | SG13G2 (subsidized) | RV64I | RT20T | ~$5K |
| Tiny Tapeout | Various | RV32I | RT20T-Mini | $300-$10K |

### 20.3 Advantage Over Pure Ternary Manufacturing

A hybrid chip is EASIER to manufacture and verify than a pure ternary chip:

1. **The binary domain uses proven RISC-V cores.** Open-source RTL exists (PicoRV32, VexRiscv, BOOM). No new validation needed for the binary half.

2. **The ternary domain is smaller.** In a hybrid chip, the TED can start minimal (just TADD, TNOT, TCOMP, TB3) and expand in future revisions. The BED handles everything the TED can't yet.

3. **Testing is incremental.** Test the BED first (it's a known-good RISC-V core). Then test the TED in isolation. Then test domain crossing. This is three simple test campaigns vs one complex full-chip test.

4. **Risk is lower.** If the ternary domain has a bug, the chip still works as a RISC-V processor. A pure ternary chip with a bug is a paperweight.

---

## 21. Migration Path: Binary → Hybrid → Pure Ternary

### 21.1 The Three Generations

```
Generation 1: RISC-BT Hybrid (this spec)
  ├── Binary domain: RISC-V (full ecosystem, runs Linux)
  ├── Ternary domain: RISC-T (knowledge, comparison, negation)
  ├── Both first-class, Cost Router dispatches
  └── PURPOSE: prove hybrid works, build ternary software ecosystem

Generation 2: RISC-BT Ternary-Primary
  ├── Ternary domain: expanded (more hardware, more instructions)
  ├── Binary domain: reduced to I/O controller + compatibility shim
  ├── Most computation in ternary, binary only for I/O and legacy
  └── PURPOSE: ternary software ecosystem mature, hardware proven

Generation 3: RISC-T Pure Ternary (companion spec)
  ├── Ternary domain: sole execution engine
  ├── Binary domain: eliminated (replaced by binary bridge for I/O only)
  ├── Ternary I/O (PAM-3, TLink) replaces binary protocols
  └── PURPOSE: full ternary hardware stack, no binary legacy
```

### 21.2 Software Migration

```
Phase 1 (Hybrid, Gen 1):
  ├── OS: RISC-V Linux (runs on BED, unmodified)
  ├── K3D: Galaxy reasoning on TED, I/O on BED
  ├── Libraries: math on EITHER (Cost Router assigns)
  ├── Applications: binary (existing RISC-V apps work)
  └── New ternary code: knowledge functions, 3-way logic

Phase 2 (Ternary-Primary, Gen 2):
  ├── OS: Hybrid kernel (scheduler aware of both domains)
  ├── K3D: fully ternary except display output
  ├── Libraries: ternary-native math, string, containers
  ├── Applications: new apps in ternary, old apps on BED compatibility
  └── Compilers: domain-aware by default

Phase 3 (Pure Ternary, Gen 3):
  ├── OS: Ternary-native kernel
  ├── K3D: pure ternary, zero domain crossings
  ├── Everything ternary-native
  └── Binary Bridge: used only for external I/O to legacy devices
```

### 21.3 Economic Path

| Generation | Ternary Transistors | Binary Transistors | Ternary Software | Binary Software |
|-----------|--------------------|--------------------|------------------|-----------------|
| Gen 1 (Hybrid) | ~30% of die | ~70% of die | ~5% of code | ~95% of code |
| Gen 2 (Ternary-Primary) | ~70% of die | ~30% of die | ~60% of code | ~40% of code |
| Gen 3 (Pure Ternary) | ~95% of die | ~5% (bridge) | ~99% of code | ~1% (legacy I/O) |

**Each generation is a viable product.** Gen 1 is a better RISC-V chip (with ternary acceleration). Gen 2 is a ternary chip with binary compatibility. Gen 3 is the end state. No generation requires the next to be commercially successful.

---

## Appendix A: Prior Art Inventory

This specification establishes prior art for the following designs. No patents may be granted on:

1. Hybrid binary-ternary processor with both execution domains on a single die
2. Cost Router hardware for automatic dispatch of instructions to cheaper execution domain
3. Static cost table mapping instructions to binary or ternary domain by gate/energy cost
4. Dynamic workload profiling for domain dispatch load balancing
5. Compiler domain hint bit in instruction encoding for cost override
6. Dual-domain register file with separate binary (64-bit) and ternary (40-trit) register banks
7. Shadow register pairs with automatic hardware B2T/T2B synchronization
8. Domain Crossing Bridge (DCB) for explicit binary-ternary format conversion
9. Binary-to-balanced-ternary hardware converter using iterative division-by-3 with balanced remainder
10. Ternary-to-binary hardware converter using weighted trit summation
11. Unified instruction encoding using RISC-V 32-bit format with CUSTOM-3 opcode space for ternary
12. Unified cache with dual-format lines tagged as binary or ternary data
13. BT-MESI cache coherence protocol extending MESI with format bit (7 states)
14. Cross-domain cache access with automatic format conversion in cache controller
15. Double-caching optimization for frequently cross-domain-accessed cache lines
16. Page table extension with T-page and T-exec bits for ternary data/code pages
17. Hybrid 5-stage pipeline with shared Fetch/Decode and split Execute/Writeback
18. Dual-issue pipeline issuing one binary + one ternary instruction per cycle simultaneously
19. Cross-domain branch prediction with separate binary (2-bit) and ternary (3-trit) BTBs
20. Asymmetric power domain gating (independently power-gate BED or TED)
21. Workload-adaptive domain gating based on utilization counters
22. Hybrid GPU Compute Unit with both binary and ternary SIMT lane blocks
23. Hybrid shader model with interleaved binary and ternary instructions in single shader
24. Hybrid rasterizer with binary triangle setup and ternary edge function evaluation
25. All I/O through binary domain with ternary access via shared memory or shadow registers
26. Future ternary I/O via PAM-3 connecting directly to TED
27. Domain-aware compiler with automatic instruction-to-domain assignment algorithm
28. Domain crossing minimization via graph-based assignment optimization
29. C language _Ternary type qualifier and switch_ternary control flow extension
30. Mixed calling convention with binary and ternary register partitions
31. Shadow register caller-save convention for cross-domain function calls
32. Three-generation migration path: hybrid → ternary-primary → pure ternary
33. Generation 1 as commercially viable RISC-V chip with ternary acceleration
34. Generation 2 with reduced binary domain serving only I/O and compatibility
35. Incremental testing strategy: BED first, TED isolated, then integration
36. Hybrid chip as lower-risk manufacturing path vs pure ternary
37. K3D composed head pipeline split across BED (I/O) and TED (reasoning) domains
38. RISC-V RV64GC as binary domain ISA for maximum ecosystem compatibility
39. Ternary-packed memory pages in binary DRAM with OS-level page table awareness
40. Utilization-based EITHER instruction routing with configurable threshold

**Publication date: 2026-03-19. Timestamped, indexed, publicly available.**

---

## Appendix B: Operation Cost Comparison (Full Table)

| # | Operation | Binary Gates | Binary Cycles | Ternary Gates | Ternary Cycles | Winner | Ratio |
|---|-----------|-------------|---------------|---------------|----------------|--------|-------|
| 1 | AND (2-input) | 2 | 1 | 6 | 1 | Binary | 3× |
| 2 | OR (2-input) | 2 | 1 | 6 | 1 | Binary | 3× |
| 3 | XOR (2-input) | 4 | 1 | 8 | 1 | Binary | 2× |
| 4 | NOT (1-input) | 2 | 1 | 0 (wire) | 0 | Ternary | ∞ |
| 5 | Bit/trit shift | 0 (wire) | 0 | 12 | 1 | Binary | ∞ |
| 6 | Popcount (N-bit) | ~2N | 1 | convert+2N | 2 | Binary | 2× |
| 7 | CRC-32 | 32 | 1 | N/A | N/A | Binary | — |
| 8 | AES round | ~5000 | 1 | N/A | N/A | Binary | — |
| 9 | Comparison (3-way) | 6 + branch | 2 | 14 | 1 | Ternary | 2× |
| 10 | Branch (3-way) | 2 × branch | 2 | 40 MOSFET | 1 | Ternary | 2× |
| 11 | Negation (N-wide) | N × 2 | 1 | 0 (wire) | 0 | Ternary | ∞ |
| 12 | Absolute value | ~2N + branch | 2 | ~N MUX | 1 | Ternary | 2× |
| 13 | Sign detect | extract | 1 | read MST | 0 | Ternary | ~1.5× |
| 14 | Knowledge state | 2 bits + flag | 1 | 1 trit | 1 | Ternary | 2× |
| 15 | Frustum test | FP cmp × 6 | 6 | TCOMP × 3 | 1 | Ternary | 6× |
| 16 | Morton encode | bit interleave | 1 | wire route | 0 | Ternary | ∞ |
| 17 | Addition (32b/20T) | 192 | 1 | 560 | 1 | ~Equal | ~1× |
| 18 | Multiply (32b/20T) | ~10K | 3 | ~8K | 3 | ~Equal | ~1× |
| 19 | FP add | ~2K | 5 | ~2K | 5 | Equal | 1× |
| 20 | SIMT divergence | 2 passes | 2× | 1 pass (3-way) | 1× | Ternary | 2× |

---

## Appendix C: Economic Model

| Stakeholder | Gen 1 (Hybrid) Value | Gen 2 (Ternary-Primary) Value | Gen 3 (Pure Ternary) Value |
|------------|---------------------|------------------------------|---------------------------|
| **Chip designers** | Reuse RISC-V cores + add ternary | Expand ternary, shrink binary | Full ternary design |
| **Software developers** | Write binary, profile to identify ternary candidates | Migrate hot paths to ternary | Native ternary development |
| **Cloud providers** | Ternary acceleration for AI (cost savings) | Majority ternary inference (60% power saving) | Full ternary data centers |
| **K3D / PM-KR** | Reference hybrid implementation | Reference ternary-primary implementation | Target platform |
| **Foundries** | Standard CMOS (existing process) | Multi-Vth CMOS (minor process change) | Ternary-native cells (new PDK) |
| **OS vendors** | Minimal kernel changes (page table bits) | Domain-aware scheduler | Ternary-native kernel |

**The hybrid path de-risks ternary adoption.** Each generation is profitable on its own. No generation depends on the next. The binary domain provides a safety net — if ternary hardware has issues, the chip still functions as a RISC-V processor.

**This is the Red Hat model applied to silicon, with a migration plan built in.**

---

## Sources & References

### Open ISA
- [RISC-V Specifications — RISC-V International](https://riscv.org/specifications/ratified/)
- [RISC-V ISA Manual — GitHub](https://github.com/riscv/riscv-isa-manual)
- [PicoRV32 — Minimal RISC-V Core](https://github.com/YosysHQ/picorv32)
- [VexRiscv — Configurable RISC-V](https://github.com/SpinalHDL/VexRiscv)
- [BOOM — RISC-V Out-of-Order Core](https://github.com/riscv-boom/riscv-boom)

### Open GPU
- [MIAOW GPU — University of Wisconsin](https://miaowgpu.org/)
- [Vortex RISC-V GPGPU — Georgia Tech](https://vortex.cc.gatech.edu/)
- [Nyuzi GPGPU Processor — GitHub](https://github.com/jbush001/NyuziProcessor)

### Ternary Computing
- [Douglas W. Jones on Ternary Arithmetic](https://homepage.divms.uiowa.edu/~jones/ternary/arith.shtml)
- [Ternary ALU Design](https://louis-dr.github.io/ternalu3.html)
- [Ternary Computing Overview](https://www.ternary-computing.com/)
- [Ternary RISC Processor on FPGA — Hackaday](https://hackaday.com/2026/03/16/ternary-risc-processor-achieves-non-binary-computing-via-fpga/)

### Hybrid Computing
- [Huawei Ternary Logic Chip](https://meta-quantum.today/?p=7960)
- [MoS₂/WSe₂ Binary/Ternary Convertible Inverter](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adfm.202510164)
- [Memristor-CMOS Hybrid Ternary Logic](https://arxiv.org/abs/2309.01615)

### Silicon Fabrication
- [Google/SkyWater Open Program](https://www.theregister.com/2020/07/03/open_chip_hardware/)
- [Libre Silicon](https://libresilicon.com/)
- [CHIPS Alliance](https://www.chipsalliance.org/)
- [Open-Source Chips for Europe](https://open-source-chips.eu/)

### Defensive Publication
- [Defensive Publication — Wikipedia](https://en.wikipedia.org/wiki/Defensive_publication)
- [Defensive Publication Strategy — PatentPC](https://patentpc.com/blog/how-to-conduct-a-defensive-publication-to-prevent-patent-infringement)
- [Open Source Hardware Prior Art — FAS](https://fas.org/publication/open-source-hardware-uspto/)

### K3D Reference
- K3D RPN Domain Opcode Registry (docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md)
- K3D Sovereign NSI Specification §9 (docs/vocabulary/SOVEREIGN_NSI_SPECIFICATION.md)
- RISC-T Open Ternary ISA Specification (companion, this directory)
