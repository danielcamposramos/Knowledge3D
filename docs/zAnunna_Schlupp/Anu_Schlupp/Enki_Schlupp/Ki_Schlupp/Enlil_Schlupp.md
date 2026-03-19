# RISC-xRVT: True Hybrid Core Specification

**Version:** 0.1 DRAFT - Defensive Publication
**Date:** 2026-03-19
**Authors:** Daniel Campos Ramos, Christoph Dorn, Milton Ponson
**Organization:** W3C PM-KR Community Group
**License:** W3C Royalty-Free defensive publication

---

## Notice of Defensive Publication

This specification publishes the missing third RISC family pattern: a single product core that unifies binary RISC-V modes and ternary RISC-T modes inside one execution architecture. The intent is to establish prior art for a true hybrid RISC core, not merely a pure ternary ISA and not merely a hybrid chip with separate domains.

---

## 1. Purpose

The RISC family now has three coordinated documents:

1. `Anu_Schlupp.md` - pure ternary ISA
2. `Enki_Schlupp.md` - binary + ternary hybrid chip
3. `Enlil_Schlupp.md` - one true unified core

This third document defines the AMD64-style product view for RISC:

- `RV32` = narrow binary mode
- `RV64` = long binary mode
- `RV32T` = narrow ternary mode
- `RV64T` = long ternary mode

The chip is one core family, one scheduler, one memory system, one privilege model, with both binary and ternary execution fabrics on-die.

Its ternary side inherits the same primitive as RISC-T and RISC-BT: `0` as natural rest, with two active side states. Arithmetic is derived from that native state model rather than from a unary increment/decrement gate family.

---

## 2. Mode Hierarchy

### 2.1 Architectural Modes

| Mode | Width | Logic | Purpose |
|------|-------|-------|---------|
| `RV32` | 32-bit | Binary | Legacy embedded and MCU software |
| `RV64` | 64-bit | Binary | Application-class software, OSes, servers |
| `RV32T` | 20 trits | Balanced ternary | Ternary narrow mode for control, logic, symbolic work |
| `RV64T` | 40 trits | Balanced ternary | Ternary long mode for reasoning, routing, knowledge ops |
| `RVx Mixed` | mixed | Hybrid | Domain-crossing execution under one process |

### 2.2 Product Rule

`RV32/RV64T` is not a coprocessor arrangement. It is one architectural product line with:

- one fetch unit
- one decode classifier
- one rename/dispatch policy
- one memory hierarchy
- two first-class arithmetic domains

---

## 3. Core Organization

```
                  +----------------------+
                  | Unified Fetch / I$   |
                  +----------+-----------+
                             |
                  +----------v-----------+
                  | Decode + Mode Class  |
                  | binary / ternary     |
                  +----+------------+----+
                       |            |
              +--------v--+      +--v--------+
              | Binary RF |      | Ternary RF|
              | x0-x31    |      | t0-t26    |
              +--------+--+      +--+--------+
                       |            |
              +--------v--+      +--v--------+
              | Binary EX |<---->| Bridge /  |
              | ALU/FPU   |      | Shadow RF |
              +--------+--+      +--+--------+
                       |            |
                       +-----+------+
                             |
                    +--------v---------+
                    | Unified LSU / MMU |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Shared L1/L2/L3  |
                    +------------------+
```

### 3.1 Register Model

- Binary register file: `x0-x31`
- Ternary register file: `t0-t26`
- Shadow pairs: `sx0-sx2` for low-latency bridge crossing
- Control/status state includes both binary flags and ternary condition state

### 3.2 Dispatch Rule

- Binary instructions stay in the binary domain unless promoted
- Ternary instructions stay in the ternary domain unless bridged
- Neutral operations may be routed by static tables or dynamic profiling

---

## 4. Instruction Identity

The unified core accepts two instruction families:

- standard RISC-V instruction words for `RV32/RV64`
- the ternary encodings defined by `RISC-T` for `RV32T/RV64T`

Implementation options:

1. separate fetch windows with decode classification
2. ternary parcels stored in aligned instruction containers
3. hybrid packed cache lines carrying binary and ternary code pages

The architectural requirement is simple: mixed software must look like one machine to the OS and toolchain.

---

## 5. Privilege and ABI Contract

- Reuse RISC-V privilege levels as the binary baseline
- Add ternary-visible CSR space for domain selection, bridge statistics, ternary exception cause, and confidence-state tracking
- Allow binaries, ternary programs, and mixed binaries to coexist in one address space
- Define ABIs for:
  - pure binary calls
  - pure ternary calls
  - bridge calls with explicit `B2T` / `T2B` transitions

---

## 6. K3D Mapping

`RV32/RV64T` is the natural CPU-side embodiment for K3D:

- binary path handles OS, drivers, storage protocols, PCIe, Ethernet, HDMI, USB
- ternary path handles `TADD`, `TMUL`, `TNOT`, `TCOMP`, `TQUANT`, `TPACK`, `TUNPACK`
- mixed mode places RPN orchestration, routing confidence, semantic gravity, and sovereign verdict packing into hardware-visible ternary state

Natural split:

- I/O, DMA, filesystem, protocol framing -> binary
- meaning, confidence, routing, truth/false/unknown, symbolic ranking -> ternary

---

## 7. Prototype Path

### 7.1 Open Hardware Baseline

- RISC-V standards from RISC-V International
- CVA6 / OpenHW for configurable 32/64-bit application-class cores
- OpenTitan for open security/root-of-trust integration

### 7.2 Recommended Staging

1. FPGA softcore with `RV32 + RV32T`
2. FPGA application core with `RV64 + RV64T`
3. ASIC hybrid core with asymmetric power gating
4. ternary-primary derivative where binary fabric becomes compatibility-only

---

## 8. Defensive Publication Claims

This document intentionally publishes prior art for:

- a unified `RV32/RV64/RV32T/RV64T` core family
- dual register files with shadow bridge pairs
- one privilege model spanning binary and ternary execution
- mixed binary/ternary instruction residency within one process
- K3D ternary opcode hardware mapping on a RISC product core

---

## Sources

- [RISC-V Ratified Specifications](https://riscv.org/specifications/ratified/)
- [CVA6 User Manual, OpenHW Group](https://docs.openhwgroup.org/projects/cva6-user-manual/)
- [OpenTitan Documentation](https://opentitan.org/documentation/)
- [RISC-V ABIs Specification](https://docs.riscv.org/reference/application-software/abi/_attachments/riscv-abi.pdf)
