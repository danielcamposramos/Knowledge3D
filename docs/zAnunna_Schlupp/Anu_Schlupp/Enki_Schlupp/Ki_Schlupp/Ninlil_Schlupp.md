# x86_x64T: True Hybrid Core Specification

**Version:** 0.1 DRAFT — Defensive Publication
**Date:** 2026-03-19
**Authors:** Daniel Campos Ramos (PM-KR Chair), Christoph Dorn (PM-KR Contributor), Milton Ponson (PM-KR Co-Chair)
**Organization:** PM-KR Community Group
**License:** W3C Royalty-Free — published as prior art under the W3C Patent Policy
**Companion Specifications:** `Anu_Schlupp.md`, `Antu_Schlupp.md`
**Design Intent:** One true x86 + one true x64 + one true ternary fabric in one core

---

## Foundational Ternary Distinction

The ternary fabric in `x86_x64T` follows the same primitive:
- `0` = natural rest position
- `+1` = one side of the relay/state cell
- `-1` = the other side of the relay/state cell

The same states may also be labeled `0, 1, 2` if the mapping is explicit.

Arithmetic, scheduling, and hybrid execution policy are derived from that primitive. This document does not normatively depend on a unary increment/decrement gate family or one specific transistor topology.

## Notice of Defensive Publication

This document is a defensive publication for a **single-core architecture** that unifies:

- true legacy x86 execution
- true x64 execution
- x86-class ternary execution
- x64-class ternary execution
- explicit hybrid routing between binary and ternary execution fabrics

The purpose is to establish prior art for a product-class core design that follows the **AMD64 design pattern**:

- preserve legacy execution
- add native 64-bit execution without discarding legacy software
- extend the same core again with ternary execution modes rather than creating a separate external coprocessor

This is not merely a compatibility layer. It is a specification for **one core that genuinely contains all of these execution identities**.

---

## Table of Contents

1. [Purpose](#1-purpose)
2. [Architectural Principle](#2-architectural-principle)
3. [AMD-Inspired Mode Pattern](#3-amd-inspired-mode-pattern)
4. [Mode Hierarchy](#4-mode-hierarchy)
5. [Core Structure](#5-core-structure)
6. [Register Architecture](#6-register-architecture)
7. [Instruction Decode Model](#7-instruction-decode-model)
8. [x86T and x64T](#8-x86t-and-x64t)
9. [Hybrid Dispatch Policy](#9-hybrid-dispatch-policy)
10. [Memory and Addressing](#10-memory-and-addressing)
11. [Flags, Branching, and Control](#11-flags-branching-and-control)
12. [Vector and GPU Alignment](#12-vector-and-gpu-alignment)
13. [K3D Mapping](#13-k3d-mapping)
14. [Toolchain and OS Contract](#14-toolchain-and-os-contract)
15. [Reference Implementation Path](#15-reference-implementation-path)
16. [References](#16-references)

---

## 1. Purpose

The `x86_x64T` core exists to solve the real migration problem.

The world already contains:

- x86 software
- x64 operating systems
- binary toolchains
- binary firmware
- binary datacenter assumptions

But K3D and future ternary-native systems need:

- balanced ternary arithmetic
- `-1 / 0 / +1` compare-and-route logic
- ternary flags
- ternary packing
- ternary execution close to the CPU frontend

So the correct product architecture is not:

- binary-only core, or
- ternary-only core

It is:

```text
one core = legacy binary + modern binary + narrow ternary + wide ternary
```

That is `x86_x64T`.

---

## 2. Architectural Principle

### 2.1 The hybrid law

`x86_x64T` follows one law:

> Keep binary where binary is cheaper.
> Use ternary where ternary is semantically superior or more efficient.
> Route mixed workloads inside one core, not through software glue.

### 2.2 What this means physically

The core contains:

- a binary frontend and binary execution path
- a ternary execution path
- shared scheduling and memory infrastructure
- an internal dispatch policy that chooses the right substrate

### 2.3 Why this is different from a coprocessor

A coprocessor model leaves ternary as an add-on.

`x86_x64T` makes ternary a **first-class architectural personality** of the same core:

- same privilege model
- same process context
- same page tables
- same interrupt domain
- same retirement model

That is the step from "interesting extension" to "real architecture".

---

## 3. AMD-Inspired Mode Pattern

### 3.1 The pattern being followed

The core design follows the successful AMD64 historical pattern:

- keep real legacy x86 alive
- add long-mode x64 in the same core
- allow software to transition without abandoning the installed base

`x86_x64T` extends that pattern one more time.

### 3.2 The extension

The new pattern becomes:

```text
Legacy x86
  + Long-mode x64
  + Ternary x86-class mode
  + Ternary x64-class mode
  = one core
```

This is the correct continuation of the AMD64 design philosophy.

### 3.3 Product meaning

This means a vendor could ship one processor family where:

- BIOS/firmware boots in binary legacy-compatible conditions
- the OS runs in binary x64 long mode
- selected applications or kernels switch into ternary execution regions
- K3D-style workloads can use x64T directly

without changing the basic machine identity.

---

## 4. Mode Hierarchy

The core defines six architectural execution modes.

### 4.1 `X86-Legacy`

Classic real/protected/compatibility-style binary x86 execution.

### 4.2 `X64-Long`

True 64-bit binary execution.

### 4.3 `X86-Compat-on-X64`

Binary x86 compatibility execution hosted under the long-mode-capable core, following the AMD64 spirit.

### 4.4 `X86T-Narrow`

Ternary execution with x86-class width expectations.

Recommended logical width:

- `20 trits` for scalar values

Reason:

- `20 trits ≈ 31.7 bits`
- this is the correct ternary-width analogue of 32-bit-class software

### 4.5 `X64T-Long`

Ternary execution with x64-class width expectations.

Recommended logical width:

- `40 trits` for scalar values

Reason:

- `40 trits ≈ 63.4 bits`
- this is the correct ternary-width analogue of 64-bit-class software

### 4.6 `Hybrid-Mixed`

Binary and ternary instructions execute in one process and one thread context with shared memory and architectural state transitions.

This is the most important mode for real deployment.

---

## 5. Core Structure

### 5.1 Top-level block model

```text
Unified Frontend
  -> legacy decode path
  -> x64 long decode path
  -> ternary decode path
  -> unified rename / schedule
  -> hybrid dispatch unit
  -> binary execution cluster
  -> ternary execution cluster
  -> shared load/store + cache hierarchy
  -> unified retirement
```

### 5.2 One frontend, not multiple chips

The design target is **one core**, not three chips in a package.

There may be:

- distinct decoders
- distinct micro-op translators
- distinct ALU fabrics

But the machine identity is one core with one retirement domain.

### 5.3 Architectural consequence

An interrupt, page fault, or context switch sees **one processor context**, not an x86 core plus a ternary side engine.

---

## 6. Register Architecture

### 6.1 Binary architectural state

The core preserves standard binary architectural views:

- x86 integer state
- x64 integer state
- x87 / SIMD / vector state as required
- control registers
- page tables and privilege registers

### 6.2 Ternary scalar state

The core adds ternary scalar registers:

- `tn0` ... `tn7` for `X86T-Narrow`
- `tl0` ... `tl15` for `X64T-Long`

Suggested rule:

- narrow ternary state is optimized for 20-trit work
- long ternary state is optimized for 40-trit work

### 6.3 Ternary vector state

The core adds ternary vector registers:

- `tv0` ... `tv15`

These may be physically banked over vector storage resources, but they are architecturally ternary.

### 6.4 Unified state management

The OS-visible save/restore contract must cover:

- binary GPR/SIMD state
- ternary scalar state
- ternary vector state
- ternary flags and masks
- mode control registers

---

## 7. Instruction Decode Model

### 7.1 Binary decode path

The binary path preserves:

- x86 legacy decode
- x64 long-mode decode
- ordinary binary prefixes and extensions

### 7.2 Ternary decode path

Ternary execution is entered through a ternary prefix/escape system consistent with the `X64-BT` spec.

That means:

- binary frontend remains valid
- ternary instructions are recognized by explicit mode and prefix context
- internal execution lowers to ternary micro-ops

### 7.3 Narrow and long ternary views

The decode path must know whether the current ternary instruction stream targets:

- `X86T-Narrow`
- `X64T-Long`

This is directly analogous to x86-class versus x64-class width expectations.

---

## 8. x86T and x64T

### 8.1 `x86T`

`x86T` is the ternary narrow personality of the core.

It should be used where:

- code size matters
- embedded or compatibility-style runtimes matter
- 20-trit width is sufficient
- ternary control and reasoning are more important than huge scalar width

### 8.2 `x64T`

`x64T` is the ternary long personality of the core.

It should be used where:

- server or workstation-class addressing is needed
- 40-trit width maps naturally to 64-bit-class software expectations
- K3D and similar knowledge workloads want wide ternary scalar state

### 8.3 Why both exist

The same reason x86 and x64 both existed in one AMD64-class machine:

- the world is heterogeneous
- migration is staged
- narrow mode and long mode solve different deployment problems

### 8.4 Architectural relation

`x64T` is not "just more x86T". It is the ternary long-mode identity of the core.

That means:

- wider scalar contract
- larger ternary ABI surface
- stronger vector coupling
- better fit for K3D, graphs, search, and knowledge routing

---

## 9. Hybrid Dispatch Policy

### 9.1 Dispatch rule

The core contains a **Hybrid Dispatch Unit** that routes work among:

- binary-preferred path
- ternary-preferred path
- split path

### 9.2 Binary-preferred examples

- instruction fetch bookkeeping
- conventional OS services
- byte and bit protocol handling
- page-table walking
- bitmask-heavy logic

### 9.3 Ternary-preferred examples

- `less / equal / greater` compare trees
- `false / unknown / true` routing
- quantize to `-1 / 0 / +1`
- K3D ternary opcodes
- defeasible reasoning state
- attract / neutral / repel field logic

### 9.4 Split examples

Mixed kernels often look like:

1. binary address generation
2. ternary quantization
3. binary memory movement
4. ternary compare/select

The core must support this without expensive software-visible context crossings.

### 9.5 Dispatch inputs

Routing decisions may use:

- opcode class
- data representation
- locality
- energy target
- latency target
- compiler hints
- runtime hints

---

## 10. Memory and Addressing

### 10.1 Binary memory remains byte-addressed

The binary underlay keeps byte-addressed memory and standard OS assumptions.

### 10.2 Ternary memory views

The core supports two ternary memory views:

- lane form for execution
- dense packed form for storage

### 10.3 Narrow vs long ternary address expectations

`x86T` should support x86-class addressing expectations over binary memory infrastructure.

`x64T` should support x64-class addressing expectations over binary memory infrastructure.

The key rule is:

> ternary value width changes; the system memory substrate does not need to become non-byte-addressed on day one.

### 10.4 Future-native path

If native ternary SRAM or dense ternary caches become practical, the same `x86_x64T` architectural contract should survive unchanged.

---

## 11. Flags, Branching, and Control

### 11.1 Binary flags remain valid

Legacy x86/x64 binary condition handling remains present.

### 11.2 Ternary flags are first-class

The core defines ternary flag state:

| Trit | Meaning |
|------|---------|
| `-1` | less / false / repel |
| `0` | equal / unknown / neutral |
| `+1` | greater / true / attract |

### 11.3 Three-way branching

Ternary control flow should expose a real `branch-3` semantic:

- negative branch
- zero/neutral branch
- positive branch

This is one of the strongest reasons to put ternary in the core itself.

### 11.4 Why this matters

Binary control flow fragments ternary meaning across multiple branches. `x86_x64T` makes ternary branching direct and architectural.

---

## 12. Vector and GPU Alignment

### 12.1 Shared design language

The vector side of `x86_x64T` should align with the open ternary GPU line:

- packed ternary lanes
- ternary masks
- pack/unpack bridges
- compact routing state

### 12.2 CPU/GPU division of labor

CPU-side `x86_x64T` should be strongest at:

- control-heavy ternary work
- branch-rich logic
- quantization
- symbolic routing
- pointer chasing and compact graph steering

GPU-side ternary designs should be strongest at:

- swarm execution
- large field operations
- wide reductions
- high-volume graph traversal

### 12.3 Why this matters for K3D

This matches K3D's split:

- host logic and orchestration pressure moves closer to ternary correctness
- bulk field and graph work stays GPU-native

---

## 13. K3D Mapping

The first software contract this core should satisfy is K3D's ternary surface:

| K3D semantic op | Core primitive |
|-----------------|----------------|
| `TADD` | ternary add |
| `TMUL` | ternary multiply |
| `TNOT` | ternary negate/invert |
| `TCOMP` | ternary compare |
| `TQUANT` | threshold projection |
| `TPACK` | dense pack bridge |
| `TUNPACK` | lane unpack bridge |

### 13.1 K3D-specific advantage

This makes the CPU side of K3D's pipeline less lossy:

- no boolean collapse of uncertain values
- direct ternary routing
- direct packed trit handling
- exact CPU/GPU ternary parity

---

## 14. Toolchain and OS Contract

### 14.1 OS contract

An OS supporting `x86_x64T` must:

- detect ternary capability bits
- save and restore ternary state
- expose mixed binary/ternary process support
- support exceptions and traps from both fabrics

### 14.2 Compiler contract

Compilers should expose:

- `x86T` and `x64T` targets
- intrinsics for ternary operations
- ABI-safe mixed-mode calls
- vector packed-trit types

### 14.3 Software migration model

Software can evolve in stages:

1. binary-only
2. binary with ternary intrinsics
3. mixed-mode libraries
4. full `x64T` kernels
5. full ternary-first runtimes

---

## 15. Reference Implementation Path

| Phase | Deliverable | Purpose |
|------|-------------|---------|
| 1 | defensive publication | prior art for unified hybrid core |
| 2 | FPGA proof with binary + ternary clusters | validate one-core model |
| 3 | assembler/intrinsic toolchain surface | software targeting |
| 4 | silicon prototype with `x86`, `x64`, `x86T`, `x64T` modes | real deployment proof |
| 5 | native ternary storage backing | long-term efficiency |

### 15.1 Strategic role

`RISC-T` is the clean native open ternary constitution.
`X64-BT` is the ternary bridge into the installed x64 world.
`x86_x64T` is the actual one-core product architecture that unifies legacy binary and ternary futures.

All three are needed.

---

## 16. References

### Companion specs

- `Anu_Schlupp.md`
- `Antu_Schlupp.md`
- `Ki_Schlupp.md`

### External references

- [RISC-V Ratified Specifications](https://riscv.org/specifications/ratified/)
- [RISC-V ISA Manual GitHub](https://github.com/riscv/riscv-isa-manual)
- [x86-64 Instruction Encoding — OSDev Wiki](http://wiki.osdev.org/X86-64_Instruction_Encoding)
- [VEX Prefix](https://en.wikipedia.org/wiki/VEX_prefix)
- [Intel XED reference manual](https://intelxed.github.io/ref-manual/)
- [MIAOW GPU](https://miaowgpu.org/)
- [NyuziProcessor](https://github.com/jbush001/NyuziProcessor)
- [Vortex GPGPU](https://vortex.cc.gatech.edu/)
- [Balanced Memristor-CMOS ternary logic family](https://arxiv.org/pdf/2309.01615)
- [Design implementations of ternary logic systems: A critical review](https://www.sciencedirect.com/science/article/pii/S2590123024010168)
- [Douglas W. Jones on Ternary Arithmetic](https://homepage.divms.uiowa.edu/~jones/ternary/arith.shtml)
