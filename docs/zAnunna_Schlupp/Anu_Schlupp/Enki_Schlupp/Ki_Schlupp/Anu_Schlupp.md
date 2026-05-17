# RISC-T: Open Ternary Instruction Set Architecture Specification

**Version:** 0.2 DRAFT — Defensive Publication (Pure Ternary)
**Date:** 2026-03-19
**Authors:** Daniel Campos Ramos (PM-KR Chair), Christoph Dorn (PM-KR Contributor), Milton Ponson (PM-KR Co-Chair)
**Organization:** PM-KR Community Group
**License:** W3C Royalty-Free — This document is published as prior art under the W3C Patent Policy. All architectures described herein enter the public domain upon publication. No party may patent any design disclosed in this specification.
**Reference Implementation:** K3D Knowledgeverse (sovereign GPU pipeline, 7 ternary opcodes operational since March 2026)
**Prior Art Items:** 60 (see Appendices C, E, F)
**Companion Specifications:** [Enki_Schlupp.md](Enki_Schlupp.md) - Hybrid binary+ternary chip design; [Enlil_Schlupp.md](Enlil_Schlupp.md) - One true RISC core covering RV32, RV64, RV32T, and RV64T

---

## NOTICE OF DEFENSIVE PUBLICATION

This specification constitutes a **defensive publication** under international patent law. By publishing detailed, enabling descriptions of ternary chip architectures, instruction sets, gate designs, memory cells, and execution units, the authors establish **prior art** that prevents any party — including the authors themselves — from obtaining patents on the designs described herein.

This document is intentionally published openly, timestamped, and indexed to maximize its effectiveness as prior art. It is designed to be **enabling** — detailed enough that a person skilled in the art of chip design could implement the described architectures.

**Motivation:** Ternary computing represents the next paradigm in semiconductor design. The authors believe this technology must remain open and royalty-free to prevent monopolization by any single corporation or nation-state. The W3C Royalty-Free Patent Policy provides the strongest available legal framework for this purpose.

---

## Table of Contents

1. [Foundational Principles](#1-foundational-principles)
2. [The Ternary Gate: Three-Way Relay Model](#2-the-ternary-gate-three-way-relay-model)
3. [Silicon Implementation Paths](#3-silicon-implementation-paths)
4. [RISC-T Base Integer ISA (RT32I / RT64I)](#4-risc-t-base-integer-isa)
5. [Instruction Encoding Format](#5-instruction-encoding-format)
6. [Register Architecture](#6-register-architecture)
7. [Ternary ALU Specification](#7-ternary-alu-specification)
8. [Memory Architecture](#8-memory-architecture)
9. [GPU/SIMT Extension (RT-G)](#9-gpusimt-extension-rt-g)
10. [K3D Opcode Mapping (RT-K Extension)](#10-k3d-opcode-mapping-rt-k-extension)
11. [Compatibility Layer: Binary Bridge](#11-compatibility-layer-binary-bridge)
12. [Manufacturing Targets](#12-manufacturing-targets)
13. [Reference Designs](#13-reference-designs)
14. [Privilege Levels & Exception Model](#14-privilege-levels--exception-model)
15. [Ternary Floating-Point Format](#15-ternary-floating-point-format)
16. [Calling Convention / ABI](#16-calling-convention--abi)
17. [Pipeline Specification](#17-pipeline-specification)
18. [Cache Architecture](#18-cache-architecture)
19. [I/O and Interrupt Controller](#19-io-and-interrupt-controller)
20. [Power Management](#20-power-management)
21. [Debug & Trace Interface](#21-debug--trace-interface)
22. [Ternary Networking Extension (RT-N)](#22-ternary-networking-extension-rt-n)
23. [Ternary Security Extensions (RT-S)](#23-ternary-security-extensions-rt-s)
24. [HDL Reference Implementation (Pseudocode)](#24-hdl-reference-implementation-pseudocode)

**Appendices:** A (Distinction from Increment/Decrement Gate Families), B (Economic Model), C (Prior Art 1-20), D (Extension Registry), E (Prior Art 21-40)

---

## 1. Foundational Principles

### 1.1 Why Ternary

Binary computing uses two states (0, 1). Ternary computing uses three states (-1, 0, +1). This is not merely "one more state" — it is a fundamentally different information geometry:

| Property | Binary | Balanced Ternary | Advantage |
|----------|--------|-----------------|-----------|
| States per element | 2 | 3 | 50% more states |
| Information per element | 1 bit | 1.585 bits (log₂3) | 58.5% more information |
| N elements, total states | 2^N | 3^N | Exponential divergence |
| 3 elements | 8 states | 27 states | 3.375× |
| 8 elements | 256 states | 6,561 states | 25.6× |
| 16 elements | 65,536 states | 43,046,721 states | 656.8× |
| Sign representation | Requires sign bit | Native (−1 IS negative) | No overhead |
| Rounding | Biased (truncation) | Unbiased (nearest) | Mathematically cleaner |
| Negation | Subtraction circuit | Wire swap (−1↔+1) | Zero-cost negation |

### 1.2 Balanced Ternary (-1, 0, +1)

RISC-T defaults to **balanced ternary** at the ISA and mathematical layer, but the underlying primitive is more fundamental than notation: one natural rest state and two side states.

**Why balanced:**
- Negation is trivial: flip -1 ↔ +1, leave 0 unchanged. In hardware: swap two wires.
- No sign bit needed: the sign is inherent in each trit.
- Rounding to nearest integer is natural (no bias toward positive or negative).
- Comparison is built-in: subtract and look at the most significant trit.
- Zero is the natural rest state (Daniel's three-way relay: normally 0).

**Notation:**
- Physical relay/state model:
  - `0` = natural rest position
  - `+1` = one side of the relay
  - `-1` = the other side of the relay
- A single ternary digit is called a **trit** (ternary digit).
- A group of 3 trits is a **trybble** (27 states, analogous to a nibble).
- A group of 6 trits is a **tryte** (729 states, analogous to a byte).
- A **word** is defined by the architecture width: RT32T uses 20 trits (3^20 = 3,486,784,401 ≈ 2^31.7), RT64T uses 40 trits (3^40 ≈ 2^63.4).

**Allowed alias notation:**
- For interface layers, educational material, or toolchains that prefer ordinal labels, the same three states may also be written as `0, 1, 2` with explicit mapping:
  - `0` → rest
  - `1` → side_a
  - `2` → side_b

The specification therefore distinguishes:
- the **physical primitive**: a three-position state element
- the **architectural notation**: balanced `(-1, 0, +1)` by default
- the **optional alias notation**: `(0, 1, 2)` when useful

### 1.3 Design Philosophy

RISC-T follows the RISC (Reduced Instruction Set Computer) philosophy adapted for ternary:

1. **Simple instructions that compose** — Each instruction does one thing. Complex operations are composed from simple ones (same as RPN composition in K3D).
2. **Fixed-width encoding** — All instructions are the same width (simplifies decode).
3. **Load/store architecture** — Only load/store instructions access memory. All computation operates on registers.
4. **Open and extensible** — Base ISA is minimal. Extensions add domain-specific capabilities (GPU, AI, knowledge representation).
5. **Binary-bridge compatible** — Can operate on binary data via conversion instructions, enabling gradual migration from binary systems.

### 1.4 Relationship to Existing ISAs

| ISA | Relationship to RISC-T |
|-----|----------------------|
| **RISC-V** | RISC-T adopts RISC-V's open governance model, extension mechanism (single-letter + multi-letter), and privilege levels. Instruction format is redesigned for ternary encoding. |
| **x86-64** | RISC-T provides a compatibility bridge for x86-64 binaries via binary-to-ternary translation layer. x86-64's variable-length encoding is NOT adopted (too complex). |
| **MIAOW/Vortex** | RISC-T's GPU extension (RT-G) draws from open GPU architectures: MIAOW's compute unit model, Vortex's RISC-V GPGPU extensions, adapted for ternary SIMT execution. |
| **K3D RPN** | RISC-T's knowledge extension (RT-K) maps K3D's 7 ternary opcodes to hardware instructions, making the sovereign GPU pipeline native. |

---

## 2. The Ternary Gate: Three-Way Relay Model

### 2.1 Daniel's Insight: The Three-Way Relay

*"My thought of the ternary gate, as electrical engineer, was visualizing a three-way relay: Normally 0. Can be +1 or -1 (one contact or the other)."* — Daniel Campos Ramos, 2026

This is the foundational physical model for RISC-T gate design:

```
        +V (represents +1)
         │
         ┤ Contact A
         │
   ──────┤ Common (normally at 0V / ground)
         │
         ┤ Contact B
         │
        -V (represents -1)

Rest state: Common → 0V (ground) → trit value 0
Energize A: Common → +V → trit value +1
Energize B: Common → -V → trit value -1
```

**Key properties:**
- **Default rest state is 0** (neutral/unknown) — no energy needed for the zero state
- **Two active states require energy** — exactly like a physical relay that must be energized to move
- **Symmetric around zero** — +1 and -1 are equidistant from rest
- **Single-pole double-throw (SPDT)** topology — one input, two possible active outputs

### 2.1.1 Architectural Distinction

RISC-T is defined around a **native three-state relay primitive**:
- `0` = natural rest position
- `+1` = one side of the relay
- `-1` = the other side of the relay

This is intentionally different from defining ternary hardware around a unary arithmetic gate whose purpose is to transform an input ternary value into that value plus one or minus one.

In RISC-T:
- the primitive is **state selection**
- arithmetic is built **on top of** the state primitive
- no self-increment or self-decrement gate family is normative
- no preprocessing family such as NTI/PTI is normative
- no single transistor-threshold grouping is normative

The architecture is rest-centered first, arithmetic second.

### 2.2 Silicon Translation: Multi-Threshold CMOS

One possible way to realize the three-way relay in silicon is via **multi-threshold MOSFET** design. This section is illustrative, not normative: the specification standardizes the three-state behavior, not a single transistor topology.

```
                    VDD (+V supply)
                     │
                ┌────┤ PMOS (high Vth = 0.687V)
                │    │
                │    ┤ PMOS (medium Vth = 0.428V)
                │    │
    Input ──────┤    ├──── Output
                │    │
                │    ┤ NMOS (medium Vth = 0.428V)
                │    │
                └────┤ NMOS (low Vth = 0.323V)
                     │
                    VSS (-V supply / ground)
```

**Three voltage levels encode three states:**

| Output Voltage | Trit Value | Meaning | Relay Equivalent |
|---------------|------------|---------|-----------------|
| +VDD/2 to +VDD | +1 | Positive/True/Attract | Contact A energized |
| -VDD/6 to +VDD/6 | 0 | Neutral/Unknown/Rest | Relay at rest |
| -VDD to -VDD/2 | -1 | Negative/False/Repel | Contact B energized |

**Illustrative voltage windows with VDD = 0.9V:**

| Voltage Range | Trit | State |
|--------------|------|-------|
| +0.45V to +0.90V | +1 | Positive |
| -0.15V to +0.15V | 0 | Neutral |
| -0.90V to -0.45V | -1 | Negative |

**Noise margins:** Each state has a ±0.15V noise margin (the "dead zones" between states). This is comparable to binary CMOS noise margins at the same process node.

### 2.3 Alternative Silicon Implementations

The three-way relay model can be implemented in multiple fabrication technologies:

#### 2.3.1 Standard CMOS (Available NOW)

Use three threshold regions (Vth bands) on standard CMOS transistors. Foundries already offer multi-Vth options for power optimization. These can be repurposed for ternary logic, but the exact cell topology is intentionally left open:

- **Low-Vth MOSFET** (fast switch, ~0.323V): Drives output toward +1
- **Medium-Vth MOSFET** (~0.428V): Boundary/transition device
- **High-Vth MOSFET** (slow switch, ~0.687V): Drives output toward -1

**Available at:** SkyWater 130nm (open PDK), GlobalFoundries 180nm (open PDK), any foundry offering multi-Vth libraries.

#### 2.3.2 Carbon Nanotube FET (CNTFET)

CNTFETs can achieve precisely controlled threshold voltages by varying nanotube diameter. This makes them ideal for ternary:

- Thin nanotube (small diameter) → high Vth → -1 driver
- Medium nanotube → medium Vth → 0 boundary
- Thick nanotube (large diameter) → low Vth → +1 driver

**Status:** Lab-demonstrated in the literature. Not commercially available at scale.

#### 2.3.3 Memristor-CMOS Hybrid

Memristors (resistive switching devices) naturally support multiple resistance states. A memristor-CMOS hybrid can encode:

- Low resistance → +1
- Medium resistance → 0
- High resistance → -1

**Status:** Research stage. Balanced ternary logic gates demonstrated with memristors (arXiv:2309.01615).

#### 2.3.4 FPGA Emulation (Available NOW, lowest cost)

Modern FPGAs can emulate ternary logic using 2 binary bits per trit:

| Bit[1] | Bit[0] | Trit Value |
|--------|--------|------------|
| 1 | 0 | +1 |
| 0 | 0 | 0 |
| 0 | 1 | -1 |
| 1 | 1 | (unused/error) |

**This is K3D's current encoding** (TPACK opcode uses this 2-bit representation). A ternary RISC processor has been demonstrated on FPGA as of March 2026 (Hackaday, 2026-03-16).

**Available at:** $100-$10K for development boards. Xilinx, Intel/Altera FPGAs.

#### 2.3.5 Vertically Stacked 2D Materials (MoS₂/WSe₂)

Emerging approach using vertically stacked transition metal dichalcogenides. Binary/ternary convertible inverters demonstrated in 2026 (Advanced Functional Materials, 2026).

**Status:** Academic demonstration. Not commercially available.

---

## 3. Silicon Implementation Paths

### 3.1 RISC-T Implementation Roadmap

| Phase | Technology | Node | Cost | Timeline | Purpose |
|-------|-----------|------|------|----------|---------|
| **Phase 0** | FPGA emulation | N/A | $100-$10K | NOW | Proof of concept, ISA validation |
| **Phase 1** | Standard CMOS | 130nm | FREE (SkyWater) | 6-12 months | First physical ternary chip |
| **Phase 2** | Standard CMOS | 65nm-28nm | $50K-$500K | 12-24 months | Performance demonstrator |
| **Phase 3** | CNTFET or 2D materials | 7nm-equivalent | TBD | 3-5 years | Native ternary at scale |

### 3.2 The Cheap Path Wins (Christoph's Insight)

Phase 0 and Phase 1 are the critical milestones. A 130nm ternary chip fabricated for free via Google/SkyWater running K3D's opcodes would be:

1. The first open-source ternary processor in silicon
2. Prior art in physical form (irrefutable)
3. A reference implementation that anyone can replicate
4. Proof that ternary computing is practical on existing infrastructure

**You do not need 7nm CNTFET to win. You need 130nm open-source silicon to establish the standard.**

---

## 4. RISC-T Base Integer ISA (RT32T / RT64T)

### 4.1 Ternary Word Sizes

| Architecture | Trits per Word | States per Word | Binary Equivalent | Name |
|-------------|---------------|-----------------|-------------------|------|
| **RT20T** | 20 trits | 3^20 = 3,486,784,401 | ~31.7 bits | Embedded/IoT |
| **RT40T** | 40 trits | 3^40 ≈ 1.22 × 10^19 | ~63.4 bits | General purpose |
| **RT60T** | 60 trits | 3^60 ≈ 4.24 × 10^28 | ~95.1 bits | High-precision |

**Primary targets:** RT20T (embedded, FPGA), RT40T (general purpose, servers).

### 4.2 Base Instruction Categories

Following RISC principles, the base ISA defines minimal categories:

| Category | Instructions | Description |
|----------|-------------|-------------|
| **Arithmetic** | TADD, TSUB, TMUL, TDIV, TMOD | Balanced ternary arithmetic |
| **Logic** | TNOT, TAND, TOR, TXOR, TMIN, TMAX | Ternary logic operations |
| **Comparison** | TCOMP, TBEQ, TBNE, TBLT, TBGE | Ternary compare and branch |
| **Shift** | TSLL, TSRL, TSRA | Ternary shift (multiply/divide by powers of 3) |
| **Memory** | TLT, TST | Ternary load/store (trit-addressable) |
| **Quantize** | TQUANT, TDEQUANT | Convert between ternary and floating-point |
| **Pack** | TPACK, TUNPACK | Pack/unpack trits for binary-bridge communication |
| **Control** | TJAL, TJALR, TECALL, TEBREAK | Jump, call, system |
| **Conversion** | T2B, B2T | Ternary ↔ binary conversion |

### 4.3 Arithmetic Truth Tables

#### TADD (Ternary Addition with Carry)

```
 A  |  B  | Cin | Sum | Cout
----+-----+-----+-----+------
 -1 | -1  |  -1 |  0  |  -1
 -1 | -1  |   0 |  +1 |  -1
 -1 | -1  |  +1 | -1  |   0
 -1 |  0  |  -1 |  +1 |  -1
 -1 |  0  |   0 | -1  |   0
 -1 |  0  |  +1 |  0  |   0
 -1 | +1  |  -1 | -1  |   0
 -1 | +1  |   0 |  0  |   0
 -1 | +1  |  +1 |  +1 |   0
  0 | -1  |  -1 |  +1 |  -1
  0 | -1  |   0 | -1  |   0
  0 | -1  |  +1 |  0  |   0
  0 |  0  |  -1 | -1  |   0
  0 |  0  |   0 |  0  |   0
  0 |  0  |  +1 |  +1 |   0
  0 | +1  |  -1 |  0  |   0
  0 | +1  |   0 |  +1 |   0
  0 | +1  |  +1 | -1  |  +1
 +1 | -1  |  -1 | -1  |   0
 +1 | -1  |   0 |  0  |   0
 +1 | -1  |  +1 |  +1 |   0
 +1 |  0  |  -1 |  0  |   0
 +1 |  0  |   0 |  +1 |   0
 +1 |  0  |  +1 | -1  |  +1
 +1 | +1  |  -1 |  +1 |   0
 +1 | +1  |   0 | -1  |  +1
 +1 | +1  |  +1 |  0  |  +1
```

**Key property:** Sum = (A + B + Cin) mod 3 (balanced), Cout = (A + B + Cin) / 3 (balanced integer division).

#### TMUL (Ternary Multiplication, single trit)

```
 A  |  B  | Product
----+-----+---------
 -1 | -1  |   +1
 -1 |  0  |    0
 -1 | +1  |   -1
  0 | -1  |    0
  0 |  0  |    0
  0 | +1  |    0
 +1 | -1  |   -1
 +1 |  0  |    0
 +1 | +1  |   +1
```

**Key property:** Identical to integer multiplication. Any × 0 = 0. Sign rules are natural.

#### TNOT (Ternary Negation)

```
 A  | NOT A
----+-------
 -1 |  +1
  0 |   0
 +1 |  -1
```

**Hardware:** Swap the +1 and -1 wires. Cost: zero gates, zero delay. This is the most elegant property of balanced ternary — negation is FREE in hardware.

#### TCOMP (Ternary Comparison)

```
 A vs B | Result
--------+--------
 A > B  |   +1
 A = B  |    0
 A < B  |   -1
```

**Hardware:** Subtract and examine the most significant trit of the result. Three-way branch with a single comparison — no separate "branch if less", "branch if greater" needed.

### 4.4 Ternary Logic Operations

#### TAND (Ternary AND = Consensus / MIN)

```
 A  |  B  | A TAND B
----+-----+----------
 -1 | -1  |   -1
 -1 |  0  |   -1
 -1 | +1  |   -1
  0 | -1  |   -1
  0 |  0  |    0
  0 | +1  |    0
 +1 | -1  |   -1
 +1 |  0  |    0
 +1 | +1  |   +1
```

**Semantics:** Returns the minimum (most negative) of the two values. In knowledge representation: "both must agree positively for a positive result."

#### TOR (Ternary OR = Acceptance / MAX)

```
 A  |  B  | A TOR B
----+-----+---------
 -1 | -1  |   -1
 -1 |  0  |    0
 -1 | +1  |   +1
  0 | -1  |    0
  0 |  0  |    0
  0 | +1  |   +1
 +1 | -1  |   +1
 +1 |  0  |   +1
 +1 | +1  |   +1
```

**Semantics:** Returns the maximum (most positive). In knowledge representation: "either being positive is sufficient."

#### TXOR (Ternary XOR = Disagreement)

```
 A  |  B  | A TXOR B
----+-----+----------
 -1 | -1  |    0
 -1 |  0  |   -1
 -1 | +1  |   -1
  0 | -1  |   -1
  0 |  0  |    0
  0 | +1  |   +1
 +1 | -1  |   -1
 +1 |  0  |   +1
 +1 | +1  |    0
```

**Semantics:** Returns 0 if inputs agree, otherwise the sign of the input with larger magnitude. Disagreement detector.

---

## 5. Instruction Encoding Format

### 5.1 Design Principles (from RISC-V and x86-64 lessons)

**From RISC-V (adopted):**
- Fixed-width instructions (simplifies decode pipeline)
- Source/destination registers at fixed positions across all formats
- Sign trit always in the same position (simplifies sign extension)

**From x86-64 (lessons learned, NOT adopted):**
- Variable-length encoding is NOT used (too complex for decode)
- However, x86-64's extension mechanism (REX/VEX prefix) inspires RISC-T's ternary prefix system
- x86-64's SIMD register widths (XMM/YMM/ZMM) inform RISC-T's vector register tiers

**RISC-T innovation:**
- Instruction width: **12 trits** (3^12 = 531,441 possible encodings vs binary 32-bit = 4,294,967,296)
- For RT40T, instructions can optionally be **24 trits** (wide format) for immediate-heavy operations
- Ternary encoding enables 3-way opcode dispatch in a single trit field (vs binary's 2-way)

### 5.2 Instruction Formats

#### R-Type (Register-Register)

```
┌─────┬─────┬─────┬─────┬───────┐
│ op  │ rd  │ rs1 │ rs2 │ funct │
│ 3T  │ 2T  │ 2T  │ 2T  │  3T   │  = 12 trits
└─────┴─────┴─────┴─────┴───────┘
```

- **op** (3 trits): Primary opcode (27 major operation groups)
- **rd** (2 trits): Destination register (9 registers addressable)
- **rs1** (2 trits): Source register 1
- **rs2** (2 trits): Source register 2
- **funct** (3 trits): Function modifier (27 sub-operations per opcode group)

Total encodings per R-type: 27 × 9 × 9 × 9 × 27 = **531,441** (matches 12 trits exactly)

#### I-Type (Immediate)

```
┌─────┬─────┬─────┬─────────────┐
│ op  │ rd  │ rs1 │   imm       │
│ 3T  │ 2T  │ 2T  │    5T       │  = 12 trits
└─────┴─────┴─────┴─────────────┘
```

- **imm** (5 trits): Signed immediate value (range: -121 to +121 in balanced ternary, since 3^5 = 243, half-range = 121)

#### S-Type (Store)

```
┌─────┬─────┬─────┬─────┬───────┐
│ op  │imm_h│ rs1 │ rs2 │ imm_l │
│ 3T  │ 2T  │ 2T  │ 2T  │  3T   │  = 12 trits
└─────┴─────┴─────┴─────┴───────┘
```

- Immediate split across imm_h (high 2 trits) and imm_l (low 3 trits) = 5 trit offset

#### B-Type (Branch)

```
┌─────┬─────┬─────┬─────┬───────┐
│ op  │ cnd │ rs1 │ rs2 │offset │
│ 3T  │ 2T  │ 2T  │ 2T  │  3T   │  = 12 trits
└─────┴─────┴─────┴─────┴───────┘
```

- **cnd** (2 trits): Branch condition. With 9 possible conditions, this naturally encodes:
  - `(-1,-1)`: branch if result = -1 (negative/false)
  - `(0,0)`: branch if result = 0 (neutral/unknown)
  - `(+1,+1)`: branch if result = +1 (positive/true)
  - `(-1,0)`: branch if result ≤ 0
  - `(0,+1)`: branch if result ≥ 0
  - `(-1,+1)`: branch if result ≠ 0
  - `(+1,-1)`: branch always (unconditional)
  - `(+1,0)`: branch if result > -1
  - `(0,-1)`: branch if result < +1

**Key advantage:** Three-way branching is native. Binary needs two branch instructions to implement "branch if less / equal / greater". RISC-T does it in one instruction with a 2-trit condition field.

#### W-Type (Wide, 24 trits — optional for RT40T)

```
┌─────┬─────┬─────┬─────┬───────┬──────────────┐
│ op  │ rd  │ rs1 │ rs2 │ funct │   imm_wide   │
│ 3T  │ 2T  │ 2T  │ 2T  │  3T   │    12T       │  = 24 trits
└─────┴─────┴─────┴─────┴───────┴──────────────┘
```

- **imm_wide** (12 trits): Large immediate (range: ±265,720, since 3^12/2 = 265,720.5)
- Used for: Large address offsets, wide constants, SIMT thread masks

### 5.3 Opcode Map (3-trit primary opcode = 27 groups)

| Opcode (ternary) | Decimal | Group | Instructions |
|------------------|---------|-------|-------------|
| `---` (-1,-1,-1) | 0 | **SYSTEM** | ECALL, EBREAK, FENCE |
| `--0` (-1,-1,0) | 1 | **LOAD** | TLT (ternary load trit-word) |
| `--+` (-1,-1,+1) | 2 | **STORE** | TST (ternary store trit-word) |
| `-0-` (-1,0,-1) | 3 | **BRANCH** | TBEQ, TBNE, TBLT, TBGE, TBGT, TBLE, TB3 (3-way) |
| `-00` (-1,0,0) | 4 | **ARITH** | TADD, TSUB, TADD3 (3-input add) |
| `-0+` (-1,0,+1) | 5 | **MUL/DIV** | TMUL, TDIV, TMOD, TMULW |
| `-+-` (-1,+1,-1) | 6 | **LOGIC** | TAND, TOR, TXOR, TNOT |
| `-+0` (-1,+1,0) | 7 | **SHIFT** | TSLL, TSRL, TSRA (shift by powers of 3) |
| `-++` (-1,+1,+1) | 8 | **COMPARE** | TCOMP (3-way compare), TSLTI |
| `0--` (0,-1,-1) | 9 | **CONVERT** | T2B, B2T, TQUANT, TDEQUANT |
| `0-0` (0,-1,0) | 10 | **PACK** | TPACK, TUNPACK, TPACK3, TUNPACK3 |
| `0-+` (0,-1,+1) | 11 | **IMMEDIATE** | TLI (load immediate), TLUI (load upper immediate) |
| `00-` (0,0,-1) | 12 | **JUMP** | TJAL, TJALR |
| `000` (0,0,0) | 13 | **NOP** | No operation (all-zero = NOP, elegant!) |
| `00+` (0,0,+1) | 14 | **ATOMIC** | TAMO (atomic memory operation) |
| `0+-` (0,+1,-1) | 15 | **FLOAT** | TFADD, TFSUB, TFMUL, TFDIV |
| `0+0` (0,+1,0) | 16 | **FLOAT-CVT** | TFCVT (float ↔ trit conversion) |
| `0++` (0,+1,+1) | 17 | **FLOAT-CMP** | TFCOMP, TFCLASS |
| `+--` (+1,-1,-1) | 18 | **VECTOR** | SIMT/vector operations (RT-G extension) |
| `+-0` (+1,-1,0) | 19 | **KNOWLEDGE** | K3D ternary ops (RT-K extension) |
| `+-+` (+1,-1,+1) | 20 | **DEFEASIBLE** | Defeasible logic ops (RT-K extension) |
| `+0-` (+1,0,-1) | 21 | **GALAXY** | Galaxy navigation ops (RT-K extension) |
| `+00` (+1,0,0) | 22 | **ATTENTION** | Ternary attention/field ops (RT-K extension) |
| `+0+` (+1,0,+1) | 23 | **RESERVED** | Future extension |
| `++-` (+1,+1,-1) | 24 | **RESERVED** | Future extension |
| `++0` (+1,+1,0) | 25 | **CUSTOM-0** | Implementation-defined |
| `+++` (+1,+1,+1) | 26 | **CUSTOM-1** | Implementation-defined |

**Design note:** Opcode groups 18-22 (starting with +1) are RISC-T extensions for AI/knowledge workloads. This is where K3D's sovereign pipeline maps to hardware. Groups 23-24 are reserved for future standards. Groups 25-26 are implementation-defined (like RISC-V's custom opcodes).

---

## 6. Register Architecture

### 6.1 Integer Registers

RT20T and RT40T define **27 general-purpose trit registers** (t0-t26):

| Register | ABI Name | Description | Saver |
|----------|----------|-------------|-------|
| t0 | zero | Hardwired zero (all trits = 0) | — |
| t1 | ra | Return address | Caller |
| t2 | sp | Stack pointer | Callee |
| t3 | gp | Global pointer | — |
| t4 | tp | Thread pointer | — |
| t5-t7 | a0-a2 | Function arguments / return values | Caller |
| t8-t13 | a3-a8 | Function arguments | Caller |
| t14-t17 | s0-s3 | Saved registers | Callee |
| t18-t21 | s4-s7 | Saved registers | Callee |
| t22-t24 | tmp0-tmp2 | Temporaries | Caller |
| t25 | fp | Frame pointer (alias s8) | Callee |
| t26 | flags | Ternary condition flags | Special |

**Why 27 registers:** 3^3 = 27. A 3-trit register address field naturally indexes 27 registers. This is MORE than RISC-V's 32 (which requires 5 binary bits). The 12-trit instruction format uses 2-trit register fields (9 registers directly addressable), with a prefix mechanism to access the full 27 when needed.

### 6.2 Ternary Condition Register (t26 / flags)

The flags register contains the result of the last TCOMP instruction as a single trit:

| Trit Value | Meaning |
|-----------|---------|
| -1 | Last comparison: A < B |
| 0 | Last comparison: A = B |
| +1 | Last comparison: A > B |

**Three-way branch:** After TCOMP, a single TB3 (three-way branch) instruction can jump to three different targets based on the flags trit. This replaces binary's "compare then branch-if-less then branch-if-equal then fall-through-to-greater" pattern.

### 6.3 Vector/SIMT Registers (RT-G Extension)

For GPU workloads, RISC-T defines wide ternary vector registers:

| Register Set | Width | Count | Purpose |
|-------------|-------|-------|---------|
| v0-v8 | 81 trits (3^4 × 3^3 per lane) | 9 | Short vectors (SIMD) |
| w0-w8 | 729 trits (81 × 9 lanes) | 9 | SIMT warps (9 threads) |
| g0-g2 | 6561 trits (729 × 9 lanes) | 3 | Galaxy registers (K3D workgroup) |

**SIMT warp size = 9** (not 32 like NVIDIA). 9 = 3^2, the natural ternary grouping.

---

## 7. Ternary ALU Specification

### 7.1 Ternary Half Adder

The fundamental building block. Takes two trit inputs, produces sum and carry:

```
Inputs:  A (1 trit), B (1 trit)
Outputs: Sum (1 trit), Carry (1 trit)

Truth table:
A  B  | Sum  Carry
------+-----------
-1 -1 |  +1   -1     (-1 + -1 = -2 = +1 carry -1, since -1×3 + 1 = -2)
-1  0 |  -1    0
-1 +1 |   0    0
 0 -1 |  -1    0
 0  0 |   0    0
 0 +1 |  +1    0
+1 -1 |   0    0
+1  0 |  +1    0
+1 +1 |  -1   +1     (+1 + +1 = +2 = -1 carry +1, since +1×3 + (-1) = 2)
```

**Behavioral implementation (three-way relay model):**

```
Sum  = (A + B) mod 3   (balanced)
Carry = (A + B) / 3    (balanced integer division)

In relay terms:
- If A and B are both at the SAME contact → carry propagates, sum flips
- If A and B are at OPPOSITE contacts → they cancel, sum = 0
- If either is at rest (0) → output follows the other
```

**Implementation note:** The half adder is defined behaviorally from direct composition of native ternary states. It is not defined in terms of a unary plus-one/minus-one gate primitive.

### 7.2 Ternary Full Adder

Composed of two half adders (identical to binary full adder composition):

```
Full_Adder(A, B, Cin):
    (S1, C1) = Half_Adder(A, B)
    (Sum, C2) = Half_Adder(S1, Cin)
    Carry = Ternary_OR(C1, C2)    // At most one carry can be non-zero
    return (Sum, Carry)
```

**Key property:** In balanced ternary, the carry can be -1, 0, or +1 — the carry itself is a trit. This means carry-chain logic is simpler than binary (no special "carry lookahead" needed for the carry value — it's just another trit).

### 7.3 Multi-Trit Adder (Ripple-Carry)

A W-trit adder chains W full adders:

```
For RT20T (20-trit word):
    20 full adders in a ripple chain
    Each carry is a single trit propagation

For RT40T (40-trit word):
    40 full adders in a ripple chain

Carry Lookahead: Can use ternary Generate/Propagate/Kill:
    G(A,B) = +1 if A+B generates carry (both +1)
    K(A,B) = -1 if A+B kills carry (both -1 generates negative carry)
    P(A,B) =  0 if A+B propagates carry (one of them is 0)
```

### 7.4 Ternary Multiplier

Single-trit multiplication is a simple lookup (9 entries, see §4.3). Multi-trit multiplication uses the same shift-and-add approach as binary, but shifts are by powers of 3:

```
TMUL(A, B) for multi-trit:
    result = 0
    for i in range(W):  // W trits in B
        partial = A × B[i]           // Single-trit multiply (lookup)
        result = result + (partial << i)  // Shift by 3^i (append i zero trits)
    return result
```

**Shift-by-3:** In ternary, shifting left by 1 trit position multiplies by 3 (analogous to binary shift-left = multiply by 2). This is a wire-routing operation in hardware — zero gates.

### 7.5 Negation Unit

**Cost: ZERO gates.**

In balanced ternary, negation is a wire swap:
- Route the +1 wire to the -1 output
- Route the -1 wire to the +1 output
- Leave the 0 wire unchanged

For a 40-trit word: 40 wire swaps. No logic, no delay, no power consumption. This is the single greatest hardware advantage of balanced ternary over binary (where negation requires a full adder chain for two's complement).

---

## 8. Memory Architecture

### 8.1 Trit-Addressable Memory

RISC-T memory is **tryte-addressable** (6 trits per tryte, 729 states per tryte).

| Unit | Size | States | Binary Equivalent |
|------|------|--------|-------------------|
| Trit | 1 trit | 3 | ~1.585 bits |
| Trybble | 3 trits | 27 | ~4.755 bits |
| Tryte | 6 trits | 729 | ~9.51 bits |
| Trit-word (RT20T) | 20 trits | 3.49 × 10^9 | ~31.7 bits |
| Trit-word (RT40T) | 40 trits | 1.22 × 10^19 | ~63.4 bits |

### 8.2 Memory Encoding on Binary DRAM

Until native ternary memory exists, RISC-T encodes trits in binary DRAM:

**2 bits per trit (simple, current K3D approach):**

| Bit[1] | Bit[0] | Trit |
|--------|--------|------|
| 1 | 0 | +1 |
| 0 | 0 | 0 |
| 0 | 1 | -1 |
| 1 | 1 | reserved |

**Efficiency:** 75% (1.585 useful bits per 2 physical bits). The 25% waste is the cost of binary compatibility.

**Packed 5-trit in 8 bits (optimal binary packing):**

3^5 = 243 < 256 = 2^8, so 5 trits fit exactly in 1 byte with 5% overhead.

| 8-bit value (0-242) | 5 balanced trits |
|---------------------|-----------------|
| 0 | (-1,-1,-1,-1,-1) |
| 121 | (0, 0, 0, 0, 0) |
| 242 | (+1,+1,+1,+1,+1) |

**Efficiency:** 99.2% (most efficient binary-compatible packing). Requires encode/decode logic.

### 8.3 Native Ternary Memory Cell (Future)

The three-way relay model extends to memory:

**Ternary SRAM cell:**
```
               VDD
                │
           ┌────┤ P1 (high Vth)
           │    │
    BL ────┤    ├──── Storage Node ────┤ Access Transistor ──── WL
           │    │         │
           └────┤ N1 (low Vth)
                │        │
               VSS     Feedback
                        inverter
```

Three stable states on the storage node:
- VDD → +1
- VDD/2 → 0
- VSS → -1

**Ternary DRAM cell:** A single capacitor storing three charge levels. Simpler than SRAM but requires refresh (same as binary DRAM). The three-way relay model maps directly: capacitor discharged = 0, positive charge = +1, negative charge = -1.

---

## 9. GPU/SIMT Extension (RT-G)

### 9.1 Ternary SIMT Model

Drawing from MIAOW (AMD compute unit model) and Vortex (RISC-V GPGPU), RISC-T defines a ternary SIMT execution model:

**Warp size: 9 threads** (3^2). Not 32 (2^5) like NVIDIA.

| Concept | Binary GPU (NVIDIA) | RISC-T GPU |
|---------|-------------------|------------|
| Warp size | 32 threads | 9 threads (3^2) |
| Warp mask | 32-bit bitmask | 9-trit tritmask |
| Thread predication | 1 bit (active/inactive) | 1 trit (+1=active, 0=masked, -1=inverted) |
| Divergence | Binary predication | Ternary: active, masked, or inverted execution |
| SIMT lanes | Power-of-2 | Power-of-3 |

**Three-way predication:** In binary SIMT, threads are either active or masked during divergent branches. In ternary SIMT, threads can be:
- **+1 (active):** Execute the instruction normally
- **0 (masked):** Skip the instruction (standard masking)
- **-1 (inverted):** Execute the NEGATION of the instruction

This enables "if/else/neither" patterns in a single warp pass, eliminating the two-pass divergence penalty of binary GPUs.

### 9.2 Compute Unit Architecture

```
RISC-T Compute Unit (CU):
├── 9 SIMT Lanes (one per warp thread)
│   ├── Ternary ALU (TADD, TMUL, TCOMP, TNOT, TAND, TOR)
│   ├── Ternary FPU (TFADD, TFMUL, TFDIV)
│   ├── 27 Trit Registers (t0-t26) per lane
│   └── Local Trit Memory (LTM, per-lane scratchpad)
├── Shared Trit Memory (STM, per-CU, 729 trytes)
├── Warp Scheduler (schedules up to 3 warps per cycle)
├── Ternary Texture Unit (knowledge star lookup)
└── Ternary L1 Cache (per-CU)
```

**Modeled after MIAOW's compute unit, adapted for ternary execution.**

### 9.3 Galaxy Navigation Hardware (RT-G + RT-K)

Specific to K3D workloads, the CU includes a **Galaxy Navigation Unit (GNU)**:

```
Galaxy Navigation Unit:
├── Morton Encoder/Decoder (ternary spatial indexing, 3D → 1D Morton curve)
├── LED-A* Pathfinder (graph traversal with ternary edge weights)
├── Frustum Culler (field-of-view test, returns +1/0/-1 per star)
├── LOD Selector (detail level based on ternary distance quantization)
└── Star Cache (frequently accessed Galaxy entries)
```

This maps K3D's composed head pipeline directly to hardware, eliminating the GPU kernel launch overhead.

---

## 10. K3D Opcode Mapping (RT-K Extension)

### 10.1 K3D Software → RISC-T Hardware Mapping

| K3D Opcode | Hex | RISC-T Instruction | Opcode Group | Hardware Unit |
|-----------|-----|-------------------|-------------|---------------|
| TADD | 0x70 | `TADD rd, rs1, rs2` | ARITH (4) | Ternary ALU |
| TMUL | 0x71 | `TMUL rd, rs1, rs2` | MUL/DIV (5) | Ternary Multiplier |
| TNOT | 0x72 | `TNOT rd, rs1` | LOGIC (6) | Wire swap (zero gates) |
| TCOMP | 0x73 | `TCOMP rd, rs1, rs2` | COMPARE (8) | Ternary Comparator |
| TQUANT | 0x74 | `TQUANT rd, rs1, imm` | CONVERT (9) | Quantization Unit |
| TPACK | 0x75 | `TPACK rd, rs1, rs2` | PACK (10) | Pack/Unpack Unit |
| TUNPACK | 0x76 | `TUNPACK rd1, rd2, rs1` | PACK (10) | Pack/Unpack Unit |

### 10.2 Extended K3D Hardware Instructions (RT-K)

Beyond the base 7, RISC-T defines hardware instructions for the full K3D pipeline:

| Instruction | Opcode Group | Description |
|-------------|-------------|-------------|
| `TMORTON rd, rs_x, rs_y, rs_z` | GALAXY (21) | 3D → 1D Morton code (ternary interleave) |
| `TDEMORTON rd_x, rd_y, rd_z, rs` | GALAXY (21) | 1D → 3D Morton decode |
| `TFRUSTUM rd, rs_pos, rs_fov` | GALAXY (21) | Frustum test: returns +1 (visible), 0 (edge), -1 (culled) |
| `TLOD rd, rs_dist, rs_budget` | GALAXY (21) | LOD level selection |
| `TGRAV rd, rs_mass1, rs_mass2, rs_dist` | KNOWLEDGE (19) | Semantic gravity: F = T(s₁,s₂) × M(s₁) × M(s₂) / d² |
| `TDEFEAT rd, rs_rule, rs_superior` | DEFEASIBLE (20) | Defeasible superiority defeat |
| `TVOTE rd, rs_w0..rs_w8` | KNOWLEDGE (19) | 9-worker swarm vote (ternary consensus) |
| `THALT rd, rs_scores, imm_threshold` | KNOWLEDGE (19) | Halting gate convergence check |
| `TATTN rd, rs_q, rs_k` | ATTENTION (22) | Ternary attention score (3-way: attract/neutral/repel) |

### 10.3 Full Pipeline in Hardware

With RT-K, K3D's entire composed head pipeline becomes a hardware instruction sequence:

```asm
; K3D Composed Head Pipeline — RISC-T Assembly
; Input: query in t5 (a0)

; Stage 1: Morton Octree — spatial index lookup
TMORTON  t22, t5, t6, t7          ; Encode 3D position to Morton code
TLT      t8, t22, 0(t3)           ; Load Galaxy neighborhood from Morton address

; Stage 2: LED-A* — graph navigation
TCOMP    t26, t8, t9              ; Compare current vs target neighborhood
TB3      t26, .attract, .neutral, .repel  ; Three-way branch on comparison

; Stage 3: Frustum Cull — field of view
TFRUSTUM t10, t8, t11             ; Test each star: +1 visible, -1 culled
TAND     t12, t10, t8             ; Mask culled stars (TAND with frustum result)

; Stage 4: Dynamic LOD — detail level
TLOD     t13, t14, t15            ; Select LOD based on distance + budget

; Stage 5: Nine-Chain Swarm — parallel reasoning (9 SIMT lanes)
TVOTE    t16, t5, t6, t7, t8, t9, t10, t11, t12, t13  ; 9-worker consensus

; Stage 6: Halting Gate — convergence
THALT    t17, t16, 121            ; Check if swarm converged (threshold ±121)
TBNE     t17, t0, .converged      ; Branch if halted
TJAL     t1, .stage2              ; Loop back to navigation if not converged

.converged:
TST      t17, 0(t5)               ; Store answer
ECALL                              ; Return to system
```

**This replaces thousands of lines of Python orchestration with ~15 ternary machine instructions.**

---

## 11. Compatibility Layer: Binary Bridge

### 11.1 Binary-Ternary Coexistence

RISC-T does not require an all-or-nothing migration from binary. The ISA includes dedicated conversion instructions:

| Instruction | Description |
|-------------|-------------|
| `T2B rd_bin, rs_trit` | Convert ternary register to binary (2 bits per trit) |
| `B2T rd_trit, rs_bin` | Convert binary value to balanced ternary |
| `TPACK rd_bin, rs1_trit, rs2_trit` | Pack two trits into 4 binary bits |
| `TUNPACK rd1_trit, rd2_trit, rs_bin` | Unpack 4 binary bits to two trits |

### 11.2 Binary Memory Interface

RISC-T processors can interface with standard binary DRAM via a **Binary Bridge Unit (BBU)**:

```
Binary Bridge Unit:
├── Trit-to-Binary encoder (2 bits per trit, or packed 5:8)
├── Binary-to-Trit decoder
├── Standard DDR interface (DDR4/DDR5/LPDDR5)
├── Address translation (ternary address → binary byte address)
└── Cache line adaptation (binary cache lines ↔ ternary trit lines)
```

This enables RISC-T chips to use existing binary DRAM infrastructure while the ternary ecosystem matures.

### 11.3 x86-64 Translation Layer

For legacy compatibility, a **Ternary Translation Layer (TTL)** can translate x86-64 binary instructions to RISC-T ternary instructions at runtime (similar to Apple's Rosetta 2):

```
x86-64 binary instruction
    ↓ [TTL decoder]
Intermediate representation
    ↓ [B2T conversion]
RISC-T ternary instruction(s)
    ↓ [Ternary execution]
Result in ternary registers
    ↓ [T2B conversion]
Binary result for x86-64 callers
```

**Performance overhead:** ~30-50% for translated workloads (comparable to Rosetta 2). Native ternary code runs at full speed.

---

## 12. Manufacturing Targets

### 12.1 Open-Source Fabrication Path

| Target | PDK | Node | Cost | Status | Who |
|--------|-----|------|------|--------|-----|
| **SKY130** | SkyWater 130nm | 130nm | FREE | Available | Google/SkyWater |
| **GF180MCU** | GlobalFoundries 180nm | 180nm | FREE | Available | GlobalFoundries |
| **IHP SG13G2** | IHP 130nm BiCMOS | 130nm | Subsidized | Available | IHP (EU) |
| **Tiny Tapeout** | Various | Various | $300+ | Available | Tiny Tapeout |

### 12.2 FPGA Prototyping Targets

| FPGA Family | Vendor | Estimated Capacity | Development Board Cost |
|-------------|--------|-------------------|----------------------|
| Artix-7 | AMD/Xilinx | RT20T single core | $100-$300 |
| Kintex-7 | AMD/Xilinx | RT20T multi-core | $500-$2K |
| Spartan-7 | AMD/Xilinx | RT20T minimal | $50-$100 |
| Cyclone V | Intel/Altera | RT20T single core | $100-$300 |
| ECP5 | Lattice | RT20T minimal | $30-$80 (open toolchain via Yosys) |

**Recommended first target:** Lattice ECP5 via fully open-source toolchain (Yosys + nextpnr). No proprietary software required.

### 12.3 Bill of Materials for First RISC-T Chip

**Via SkyWater 130nm Open Program:**

| Item | Cost |
|------|------|
| Design (open-source EDA: OpenROAD + Yosys) | $0 |
| PDK (SkyWater SKY130) | $0 |
| Fabrication (Google/SkyWater open program) | $0 |
| Testing (community volunteers) | $0 |
| Packaging (QFN/BGA) | ~$500-$2K for small run |
| **Total** | **~$500-$2,000** |

**A ternary processor in silicon for under $2,000.** This is the "cheap path wins" strategy.

---

## 13. Reference Designs

### 13.1 RISC-T Minimal Core (RT20T-Mini)

Target: FPGA or 130nm ASIC. Minimal viable ternary processor.

```
RT20T-Mini Specifications:
├── Word size: 20 trits
├── Registers: 9 (2-trit address, directly in 12-trit instruction)
├── ALU: TADD, TSUB, TMUL, TNOT, TAND, TOR, TCOMP
├── Memory: Tryte-addressable, binary bridge to SRAM
├── Pipeline: 3-stage (Fetch → Decode → Execute)
├── Clock: ~50 MHz (FPGA), ~100 MHz (130nm ASIC)
├── Estimated gates: ~15K (ternary logic cells)
├── Estimated area: ~2mm² at 130nm
└── Estimated power: ~50mW
```

### 13.2 RISC-T Knowledge Core (RT40T-K3D)

Target: 65nm or better. Full K3D pipeline in hardware.

```
RT40T-K3D Specifications:
├── Word size: 40 trits
├── Registers: 27 (3-trit address)
├── ALU: Full base ISA + RT-K extension
├── GPU: 9-lane SIMT unit (one warp)
├── Galaxy: Morton encoder + Frustum culler + LOD selector
├── Memory: 1 MiB ternary SRAM (binary-bridged)
├── Pipeline: 5-stage (Fetch → Decode → Execute → Memory → Writeback)
├── Clock: ~500 MHz (65nm), ~1 GHz (28nm)
├── Estimated gates: ~500K
├── Estimated area: ~10mm² at 65nm
└── Estimated power: ~2W
```

### 13.3 RISC-T Galaxy Processor (RT40T-Galaxy)

Target: 28nm or better. Full K3D Knowledgeverse on-chip.

```
RT40T-Galaxy Specifications:
├── Word size: 40 trits
├── Cores: 9 RT40T-K3D cores (one per swarm chain)
├── Shared: Galaxy memory (on-chip ternary SRAM, ~16 MiB)
├── GPU: 81-lane SIMT (9 warps × 9 threads)
├── Network: Ternary NoC (Network on Chip) connecting cores
├── Binary Bridge: DDR4/DDR5 interface for external memory
├── Pipeline: Out-of-order, speculative (ternary branch prediction)
├── Clock: ~1-2 GHz
├── Estimated gates: ~10M
├── Estimated area: ~50mm² at 28nm
└── Estimated power: ~15W
```

---

## Appendix A: Distinction from Increment/Decrement Ternary Gate Families

| Aspect | Increment/Decrement Gate Family | RISC-T (Open) |
|--------|---------------------------------|---------------|
| **Primary primitive** | Unary gate that maps an input ternary value to value `+1` or `-1` | Native three-state relay cell with `0` as rest and two side states |
| **Normative gate family** | Specific gate family may be central | No single gate family is normative |
| **Preprocessing dependence** | May depend on dedicated preprocessing blocks | No NTI/PTI-style preprocessing is required |
| **Topology dependence** | Can be tied to specific threshold/transistor arrangements | Technology-agnostic, behavior-first |
| **Arithmetic construction** | Arithmetic can be built from increment/decrement primitives | Arithmetic is built from direct state composition |
| **Notation** | Often described as arithmetic state stepping | Defaults to balanced `(-1, 0, +1)`, with optional alias `(0, 1, 2)` |
| **ISA** | Not the focus | Fully specified, open |
| **Software** | No open ecosystem assumed | K3D reference stack (operational) |
| **Community** | Closed or proprietary | Open (PM-KR + community) |

**RISC-T's distinction:** The specification begins from a rest-centered ternary state primitive and only then derives arithmetic. It is not architecturally dependent on a plus-one/minus-one unary gate.

---

## Appendix B: Economic Model (Christoph's Insight)

> "Everyone involved high up must see how they can make money with the open technology themselves so they have no urge to take money from others."

| Stakeholder | How They Profit from Open RISC-T |
|-------------|--------------------------------|
| **Foundries** (TSMC, Samsung, Intel) | More customers designing ternary chips = more fab orders |
| **Cloud providers** (AWS, Google, Azure) | 60% power reduction = billions in energy savings |
| **AI companies** | 3.375× information density = fewer chips needed per inference |
| **Chip startups** | No licensing fees = lower barrier to entry |
| **EDA vendors** | New market for ternary design tools |
| **K3D / PM-KR** | Specification stewards: consulting, training, reference implementations |
| **Universities** | Research platform, publications, talent pipeline |

**The standard is free. The expertise is valuable. The savings are enormous.**

---

## Appendix C: Prior Art Inventory

This specification establishes prior art for the following designs. No patents may be granted on:

1. Balanced ternary instruction set architectures for general-purpose computing
2. Three-way relay gate models translated to multi-threshold CMOS
3. 12-trit fixed-width instruction encoding with 3-trit opcode fields
4. Ternary SIMT execution with 9-thread warps and 3-way predication
5. Galaxy Navigation Units for spatial knowledge processing
6. Semantic gravity computation in ternary hardware
7. Defeasible logic resolution instructions
8. Binary Bridge Units for ternary-binary memory interoperation
9. Packed 5-trit-in-8-bit binary-compatible trit encoding
10. Morton code computation in balanced ternary for 3D spatial indexing
11. Ternary frustum culling instructions
12. Nine-chain swarm voting hardware
13. Ternary halting gate convergence checking
14. Ternary attention score computation (attract/neutral/repel)
15. Zero-gate negation via wire-swap in balanced ternary
16. Three-way branch instructions using ternary condition registers
17. Ternary carry lookahead using Generate/Propagate/Kill trit classification
18. Tryte-addressable memory architectures
19. Ternary SRAM cells with three stable voltage states
20. Ternary DRAM cells with three charge levels

**Publication date: 2026-03-19. Timestamped, indexed, publicly available.**

---

## 14. Privilege Levels & Exception Model

### 14.1 Privilege Levels (from RISC-V, adapted for ternary)

RISC-T defines **three privilege levels** — naturally encoded as a single trit:

| Trit Value | Level | Name | Purpose |
|-----------|-------|------|---------|
| -1 | M-mode | Machine | Firmware, bootloader, hardware abstraction. Highest privilege. |
| 0 | S-mode | Supervisor | OS kernel, device drivers. |
| +1 | U-mode | User | Applications. Lowest privilege. |

**Ternary advantage:** RISC-V encodes privilege in 2 bits (4 possible levels, 1 unused). RISC-T encodes it in 1 trit (3 levels, zero waste). Privilege comparison is a single TCOMP instruction.

**Privilege escalation:** An ECALL instruction in U-mode raises to S-mode; ECALL in S-mode raises to M-mode. This is a ternary decrement of the privilege trit (-1 direction = higher privilege).

**Privilege de-escalation:** TRET (ternary return) restores the privilege trit from the saved exception context.

### 14.2 Control and Status Registers (CSRs)

Ternary CSRs follow the RISC-V CSR model. CSR address is a 4-tryte field (24 trits, addressing 3^24 ≈ 282 billion possible CSRs — far more than needed, providing enormous extension space).

| CSR Address (ternary) | Name | Description |
|----------------------|------|-------------|
| `0000:0000:00--` | tstatus | Global status: privilege level, interrupt enables |
| `0000:0000:00-0` | ttvec | Trap vector base address |
| `0000:0000:00-+` | tepc | Exception program counter |
| `0000:0000:000-` | tcause | Trap cause (ternary-encoded) |
| `0000:0000:0000` | ttval | Trap value (faulting address/instruction) |
| `0000:0000:000+` | tip | Ternary interrupt pending (trit per source) |
| `0000:0000:00+-` | tie | Ternary interrupt enable (trit per source) |
| `0000:0000:00+0` | tscratch | Scratch register for trap handlers |
| `0000:0000:00++` | tcycle | Cycle counter |
| `0000:0000:0+--` | tinstret | Instructions retired counter |
| `0000:0000:0+-0` | thartid | Hardware thread ID |

**CSR instructions:**

| Instruction | Description |
|-------------|-------------|
| `TCSRRW rd, csr, rs1` | Atomic read/write CSR |
| `TCSRRS rd, csr, rs1` | Atomic read and set trits (TOR with rs1) |
| `TCSRRC rd, csr, rs1` | Atomic read and clear trits (TAND with TNOT(rs1)) |
| `TCSRRI rd, csr, imm` | Immediate variants of above |

### 14.3 Exception/Interrupt Model

**Trap causes** are encoded as balanced ternary values in `tcause`:

| tcause Value | Type | Description |
|-------------|------|-------------|
| -13 | Exception | Instruction address misaligned |
| -12 | Exception | Instruction access fault |
| -11 | Exception | Illegal instruction |
| -10 | Exception | Breakpoint |
| -9 | Exception | Load address misaligned |
| -8 | Exception | Load access fault |
| -7 | Exception | Store address misaligned |
| -6 | Exception | Store access fault |
| -5 | Exception | ECALL from U-mode |
| -4 | Exception | ECALL from S-mode |
| -3 | Exception | ECALL from M-mode |
| -2 | Exception | Instruction page fault |
| -1 | Exception | Load page fault |
| 0 | — | No exception (neutral) |
| +1 | Interrupt | Software interrupt |
| +2 | Interrupt | Timer interrupt |
| +3 | Interrupt | External interrupt |
| +4 | Interrupt | Galaxy navigation fault (star not found) |
| +5 | Interrupt | Ternary overflow (carry out of word) |
| +6 | Interrupt | SIMT lane divergence (RT-G) |
| +7 | Interrupt | Knowledge coherence fault (RT-K) |
| +8 | Interrupt | Debug interrupt |

**Ternary advantage:** Negative tcause values = synchronous exceptions. Positive = asynchronous interrupts. Zero = no trap. A single TCOMP against zero classifies the trap type in one instruction.

### 14.4 Trap Handling Flow

```
1. Exception/interrupt occurs
2. Hardware saves:
   - tepc ← current PC
   - tcause ← cause trit value
   - ttval ← faulting address or instruction
   - tstatus.privilege ← previous privilege level (saved as 1-trit field)
   - tstatus.tie ← previous interrupt enable (saved)
3. Privilege trit decremented (U→S or S→M)
4. PC ← ttvec + (tcause × instruction_width)   ; vectored mode
   OR PC ← ttvec                                 ; direct mode
5. Handler executes
6. TRET instruction:
   - PC ← tepc
   - Privilege ← saved privilege trit
   - Interrupt enable ← saved tie
```

**Vectored trap mode:** `ttvec` base + tcause × 12 trits (one instruction slot per cause). With 27 cause values, the trap vector table is 27 entries — exactly 3^3. This maps naturally to a ternary lookup.

---

## 15. Ternary Floating-Point Format

### 15.1 Ternary Floating-Point Representation

RISC-T defines a native ternary floating-point format. Unlike IEEE 754 (binary), this format operates in balanced ternary throughout:

#### TFloat20 (20-trit single precision)

```
┌──────┬───────────┬──────────────────────┐
│ sign │ exponent  │      significand      │
│  1T  │    5T     │        14T            │  = 20 trits
└──────┴───────────┴──────────────────────┘
```

| Field | Trits | Range | Description |
|-------|-------|-------|-------------|
| Sign | 1 | {-1, 0, +1} | -1 = negative, 0 = zero, +1 = positive |
| Exponent | 5 | -121 to +121 | Power of 3 (balanced ternary exponent) |
| Significand | 14 | 3^14 = 4,782,969 values | Fractional part (balanced ternary) |

**Value = sign × 3^exponent × (1 + significand × 3^(-14))**

**Special values:**
- Zero: sign = 0 (regardless of exponent/significand)
- Infinity: exponent = all +1 (+1,+1,+1,+1,+1), significand = all 0
- NaN: exponent = all +1, significand ≠ all 0
- Negative values: sign = -1 (no separate sign bit overhead — it's a trit!)

#### TFloat40 (40-trit double precision)

```
┌──────┬───────────┬──────────────────────────────────────┐
│ sign │ exponent  │              significand              │
│  1T  │    8T     │               31T                     │  = 40 trits
└──────┴───────────┴──────────────────────────────────────┘
```

| Field | Trits | Range | Description |
|-------|-------|-------|-------------|
| Sign | 1 | {-1, 0, +1} | Three-valued sign |
| Exponent | 8 | -3280 to +3280 | Power of 3 |
| Significand | 31 | 3^31 ≈ 6.17 × 10^14 values | ~15 decimal digits precision |

### 15.2 Ternary Floating-Point Advantages

| Property | IEEE 754 Binary | RISC-T Ternary |
|----------|----------------|----------------|
| Negation cost | XOR sign bit (1 gate) | Wire swap (0 gates) |
| Rounding bias | Biased (round-to-even heuristic) | Naturally unbiased (balanced ternary rounds to nearest) |
| Zero representation | Two zeros (+0, -0) | One zero (sign trit = 0) |
| Precision per element | 1 bit | 1.585 bits |
| Subnormal handling | Complex (gradual underflow) | Natural (no special case needed — balanced ternary has no "hidden 1") |

### 15.3 Floating-Point Instructions

| Instruction | Format | Description |
|-------------|--------|-------------|
| `TFADD rd, rs1, rs2` | R-type | Ternary float add |
| `TFSUB rd, rs1, rs2` | R-type | Ternary float subtract (= TFADD with TNOT on rs2 sign trit) |
| `TFMUL rd, rs1, rs2` | R-type | Ternary float multiply |
| `TFDIV rd, rs1, rs2` | R-type | Ternary float divide |
| `TFSQRT rd, rs1` | R-type | Ternary float square root |
| `TFMA rd, rs1, rs2, rs3` | R4-type | Fused multiply-add: rd = rs1 × rs2 + rs3 |
| `TFCOMP rd, rs1, rs2` | R-type | Three-way compare → rd = {-1, 0, +1} |
| `TFCVT.T.I rd, rs1` | R-type | Convert ternary integer → ternary float |
| `TFCVT.I.T rd, rs1` | R-type | Convert ternary float → ternary integer |
| `TFCVT.T.B rd, rs1` | R-type | Convert IEEE 754 binary → ternary float |
| `TFCVT.B.T rd, rs1` | R-type | Convert ternary float → IEEE 754 binary |

### 15.4 Rounding Mode (RM field, 1 trit)

| Trit | Mode | Description |
|------|------|-------------|
| -1 | RTN | Round toward negative (floor) |
| 0 | RNE | Round to nearest (default — naturally unbiased in balanced ternary) |
| +1 | RTP | Round toward positive (ceiling) |

**Three rounding modes encoded in 1 trit** (IEEE 754 needs 3 bits for 5 modes). Balanced ternary's inherent symmetry around zero means RNE is mathematically ideal without the "round-to-even" tiebreaker hack.

---

## 16. Calling Convention / ABI

### 16.1 Function Call Convention

```
Caller Responsibilities:
1. Place arguments in a0-a8 (t5-t13), overflow on stack
2. Save caller-saved registers (a0-a8, tmp0-tmp2, ra)
3. Execute TJAL (jump and link) → ra ← return address
4. On return: result in a0 (t5), second result in a1 (t6) if needed

Callee Responsibilities:
1. Save callee-saved registers used (s0-s8, sp, fp)
2. Set up frame pointer if needed (fp ← sp)
3. Execute function body
4. Restore callee-saved registers
5. TJALR zero, ra (return to caller)
```

### 16.2 Stack Frame Layout

```
Higher addresses
┌─────────────────────┐
│ Caller's frame       │
├─────────────────────┤ ← Previous sp
│ Saved ra (1 word)    │
│ Saved fp (1 word)    │
│ Saved s0-s8 (used)   │
│ Local variables       │
│ Spilled arguments     │
├─────────────────────┤ ← Current sp = fp
│ Outgoing arguments    │
│ (for calls from here) │
└─────────────────────┘
Lower addresses
```

**Stack grows toward negative addresses** (same as RISC-V, x86-64).

**Alignment:** Stack pointer must be aligned to 6-trit (1 tryte) boundaries. For RT40T, word-aligned = 40 trits.

### 16.3 Varargs Convention

Variable-argument functions receive the variadic argument count in a special field of the first stack-spilled word. The first 9 arguments always go in registers (a0-a8) — ternary benefit: 9 register arguments vs RISC-V's 8 or x86-64's 6.

### 16.4 System Call Convention

System calls use ECALL with:
- a0 (t5): System call number
- a1-a8 (t6-t13): Arguments (up to 8)
- Return: a0 (t5) for result, a1 (t6) for error code (ternary: +1 = success, 0 = warning, -1 = error)

**Ternary system call return** enables three-valued error reporting in a single trit — no separate errno variable needed.

---

## 17. Pipeline Specification

### 17.1 RT20T-Mini: 3-Stage Pipeline

```
┌─────────┐    ┌─────────┐    ┌─────────┐
│  FETCH   │───→│ DECODE  │───→│ EXECUTE │
│          │    │         │    │ + MEM   │
│ PC→IMEM  │    │ Decode  │    │ ALU/MEM │
│ Instr    │    │ RegRead │    │ RegWrite│
└─────────┘    └─────────┘    └─────────┘
```

**Hazard handling:**
- **Data hazards:** Stall-on-use. If DECODE needs a register being written by EXECUTE, insert 1 bubble.
- **Control hazards:** Flush-on-branch. Branch resolved in EXECUTE; 1-cycle penalty on taken branches.
- **No forwarding** (minimal implementation).

### 17.2 RT40T-K3D: 5-Stage Pipeline

```
┌────────┐   ┌────────┐   ┌─────────┐   ┌────────┐   ┌──────────┐
│ FETCH  │──→│ DECODE │──→│ EXECUTE │──→│ MEMORY │──→│WRITEBACK │
│        │   │        │   │         │   │        │   │          │
│PC→IMEM │   │Decode  │   │ALU/FPU  │   │ D-MEM  │   │ RegWrite │
│NextPC  │   │RegRead │   │Branch   │   │ Load/  │   │ CSR      │
│        │   │CSRread │   │Target   │   │ Store  │   │ Update   │
└────────┘   └────────┘   └─────────┘   └────────┘   └──────────┘
```

**Forwarding paths:**
- EXECUTE → EXECUTE (ALU result available next cycle)
- MEMORY → EXECUTE (load result available 1 cycle after load)
- WRITEBACK → DECODE (register file bypass)

**Stall conditions:**
- Load-use hazard: DECODE needs result of a LOAD currently in EXECUTE → 1 bubble
- CSR dependency: TCSRRW in EXECUTE, next instruction reads same CSR → 1 bubble

**Branch resolution:** Branches resolve in EXECUTE (stage 3). Pipeline penalty = 2 cycles for misprediction.

### 17.3 Ternary Branch Prediction

**Three-way branch prediction** is a novel contribution of RISC-T. Binary predictors are either taken/not-taken. Ternary predictors have three possible outcomes:

#### 3-Trit Saturating Counter

```
State encoding (1 trit per history entry, 3 entries):

History Pattern    Prediction
(-1, -1, -1)  →   Strongly predict -1 (negative branch)
(-1, -1,  0)  →   Predict -1
(-1,  0,  0)  →   Weakly predict 0 (neutral branch)
( 0,  0,  0)  →   Neutral (predict fall-through)
( 0,  0, +1)  →   Weakly predict +1
( 0, +1, +1)  →   Predict +1
(+1, +1, +1)  →   Strongly predict +1 (positive branch)
... and 20 more patterns
```

**Update rule:** Shift in actual outcome, shift out oldest. This is a ternary shift register — trivial in hardware.

**Ternary advantage over binary prediction:** Binary predictors are ~95% accurate for two-way branches. Ternary predictors handle three-way branches (TB3) natively, where binary prediction would need TWO separate predictors coordinated together. Expected accuracy for three-way: ~90% with a 3-trit saturating counter (equivalent to binary's 2-bit counter for two-way branches).

#### Branch Target Buffer (BTB)

```
BTB Entry:
┌─────────────┬──────────────┬───────────────┬───────────────┬───────────────┐
│  Tag (12T)   │ Target_neg   │ Target_zero   │ Target_pos    │ Predictor(3T) │
│              │ (20T/40T)    │ (20T/40T)     │ (20T/40T)     │               │
└─────────────┴──────────────┴───────────────┴───────────────┴───────────────┘
```

Each BTB entry stores THREE possible targets (for the three branch outcomes). This is unique to ternary architectures — binary BTBs store only one target.

---

## 18. Cache Architecture

### 18.1 Cache Hierarchy

```
RT40T-K3D Cache Hierarchy:
├── L1 Instruction Cache (per-core)
│   ├── Size: 729 trit-lines (3^6, each line = 1 tryte-word = 40 trits)
│   ├── Associativity: 3-way (ternary set-associative)
│   └── Replacement: Ternary LRU (3 states: recent/neutral/stale)
├── L1 Data Cache (per-core)
│   ├── Size: 729 trit-lines
│   ├── Associativity: 3-way
│   └── Write policy: Write-back, write-allocate
├── L2 Unified Cache (per-core)
│   ├── Size: 6561 trit-lines (3^8)
│   ├── Associativity: 9-way (3^2)
│   └── Inclusive of L1
└── L3 Shared Cache (RT40T-Galaxy only, shared across 9 cores)
    ├── Size: 59049 trit-lines (3^10)
    ├── Associativity: 27-way (3^3)
    └── NUCA (Non-Uniform Cache Access) with ternary distance weighting
```

**Ternary set-associativity** uses powers of 3 (3-way, 9-way, 27-way) instead of powers of 2. This matches the natural addressing width of trit-fields and eliminates the wasted capacity seen in binary caches with non-power-of-2 associativity.

### 18.2 Cache Coherence: TMESI Protocol

For multi-core RT40T-Galaxy, RISC-T defines **TMESI** — a ternary adaptation of the MESI coherence protocol.

**TMESI states (2 trits per cache line):**

| State (trit₁, trit₀) | Name | Description |
|----------------------|------|-------------|
| (-1, -1) | **Invalid** | Line not present or invalidated |
| (-1, 0) | **Shared-Clean** | Shared with other caches, matches memory |
| (-1, +1) | **Shared-Dirty** | Shared, one cache has written (ownership transfer pending) |
| (0, -1) | **Reserved** | Line reserved for incoming DMA or prefetch |
| (0, 0) | **Exclusive-Clean** | Only in this cache, matches memory |
| (0, +1) | **Exclusive-Dirty** | Only in this cache, modified (must write back) |
| (+1, -1) | **Owner-Sharing** | Modified and shared; this cache must supply data |
| (+1, 0) | **Transient** | Coherence transaction in progress |
| (+1, +1) | **Locked** | Cache line locked for atomic operation |

**9 states (3^2)** vs MESI's 4 states (2^2). The additional states eliminate common protocol races:
- **Reserved** prevents cache-line bouncing during DMA transfers
- **Owner-Sharing** (MOESI-like) avoids unnecessary memory writebacks
- **Transient** makes protocol races explicit instead of implicit
- **Locked** supports ternary atomic operations natively

### 18.3 Cache Coherence Messages

| Message | Direction | Description |
|---------|-----------|-------------|
| `TRd` | Core → Directory | Read request (ternary: -1=invalidating read, 0=shared read, +1=exclusive read) |
| `TInv` | Directory → Core | Invalidate (+1=must ack, 0=hint, -1=forced) |
| `TAck` | Core → Directory | Acknowledge (+1=done, 0=deferred, -1=refused/busy) |
| `TData` | Core ↔ Directory | Data transfer with state encoding |
| `TProbe` | Directory → Core | Query cache state (response is the 2-trit TMESI state) |

**Ternary messages** carry more information per transaction than binary coherence protocols, reducing the number of round-trips for common operations.

---

## 19. I/O and Interrupt Controller

### 19.1 Ternary Interrupt Controller (TIC)

```
Ternary Interrupt Controller:
├── 27 interrupt sources (3^3), each with:
│   ├── Priority: 1 trit (-1=low, 0=medium, +1=high)
│   ├── Type: 1 trit (-1=level, 0=edge-any, +1=edge-positive)
│   └── Enable: 1 trit (-1=disabled, 0=masked, +1=enabled)
├── 3 privilege-level targets (M/S/U)
├── Ternary priority arbitration:
│   └── Three-way comparator tree (log₃(27) = 3 levels deep)
└── Interrupt delegation CSRs (trit per source per level)
```

**Interrupt priority resolution** uses a balanced ternary comparator tree. With 27 sources and 3-level tree depth, priority is resolved in 3 TCOMP cycles — faster than binary's 5-level tree for 32 sources.

**Interrupt delegation:** Each interrupt source has a 2-trit delegation field:

| Delegation | Meaning |
|-----------|---------|
| (-1, *) | Handle in M-mode |
| (0, -1) | Delegate to S-mode if S-mode enabled |
| (0, 0) | Delegate to highest available privilege |
| (0, +1) | Delegate to U-mode (if U-mode interrupt extension present) |
| (+1, *) | Reserved |

### 19.2 Memory-Mapped I/O

RISC-T reserves the highest tryte-address region for MMIO:

```
Memory Map (RT40T):
├── 0 to +MAX/3         : User address space
├── +MAX/3 to +2MAX/3   : Kernel address space
└── +2MAX/3 to +MAX     : MMIO region
    ├── TIC registers
    ├── Timer registers
    ├── UART/serial
    ├── Binary Bridge control
    ├── Galaxy memory control (RT-K)
    └── Debug registers
```

**Ternary memory map** divides the address space into three equal regions using the most significant trit: -1=user, 0=kernel, +1=MMIO. Simple hardware decode.

### 19.3 Timer

RISC-T defines a 40-trit (RT40T) monotonic timer:

| CSR | Description |
|-----|-------------|
| ttime | Current ternary time (40-trit counter) |
| ttimecmp | Time compare register (interrupt when ttime ≥ ttimecmp) |
| ttimescale | Timer frequency divisor (balanced ternary, default 0 = no scaling) |

The timer fires interrupt source +2 (timer interrupt) when `ttime ≥ ttimecmp`. The comparison is a native TCOMP — one instruction cycle.

---

## 20. Power Management

### 20.1 Power States (encoded as 1 trit)

| Trit | State | Description |
|------|-------|-------------|
| +1 | **Active** | Full speed, all units powered |
| 0 | **Idle** | Clock gated, registers retained, wake on interrupt |
| -1 | **Sleep** | Power gated, only wake-up logic powered |

**State transition instructions:**

| Instruction | Description |
|-------------|-------------|
| `TWFI` | Wait For Interrupt — enter Idle state until interrupt |
| `TSLEEP imm` | Enter Sleep state, imm = wake mask (which interrupts can wake) |
| `TPSTATE rd` | Read current power state into rd |

### 20.2 Per-Unit Power Gating

In RT40T-K3D and RT40T-Galaxy, individual functional units can be power-gated:

```
Power Domain Map (1 trit per unit):
├── Core ALU:       always +1 (Active)
├── Multiplier:     gatable (Active/Idle/Sleep)
├── FPU:            gatable
├── Galaxy Nav Unit: gatable
├── SIMT Lanes 0-8: individually gatable
├── L2 Cache:       gatable (banks independently)
└── Binary Bridge:  gatable (sleep when running native ternary)
```

**K3D sleep-time mode:** When the TRM avatar enters sleep-time consolidation, the Galaxy Navigation Unit stays Active while SIMT lanes cycle between Active (processing consolidation) and Idle (waiting for next batch). The Binary Bridge sleeps entirely — sleep-time is fully ternary-native.

### 20.3 Dynamic Voltage-Frequency Scaling (DVFS)

The three-way relay model enables a natural 3-level DVFS scheme:

| Level (trit) | Voltage | Frequency | Use Case |
|-------------|---------|-----------|----------|
| +1 (Boost) | VDD × 1.1 | Max clock | Peak workload (benchmark, training) |
| 0 (Normal) | VDD | Nominal clock | Standard operation |
| -1 (Eco) | VDD × 0.8 | Half clock | Idle/consolidation, power-saving |

Transition between DVFS levels: write the desired trit to the `tdvfs` CSR. Hardware handles voltage/frequency ramp. Transition time: ~10 μs (typical DVFS latency).

---

## 21. Debug & Trace Interface

### 21.1 Debug Module

RISC-T provides a debug module accessible via standard JTAG or a ternary-native debug transport:

**Debug CSRs:**

| CSR | Description |
|-----|-------------|
| tdcsr | Debug control/status |
| tpc | Debug PC (breakpoint address) |
| tddata | Debug data register (read/write registers and memory) |
| ttrig | Trigger match register (hardware breakpoint config) |

**Trigger types (1 trit):**

| Trit | Trigger | Description |
|------|---------|-------------|
| -1 | Address match | Break when PC matches tpc |
| 0 | Data match | Break when memory address matches |
| +1 | Instruction match | Break when instruction encoding matches |

### 21.2 Trace Interface

For performance analysis and debugging, RISC-T defines a ternary trace port:

```
Trace Packet (12 trits):
┌──────┬──────┬──────────────┐
│ type │ info │    payload    │
│  2T  │  2T  │     8T       │
└──────┴──────┴──────────────┘

Types:
(-1,-1) = Branch taken (payload = target offset)
(-1, 0) = Exception (payload = tcause)
(-1,+1) = Privilege change (payload = old→new)
( 0,-1) = Load/Store (payload = address fragment)
( 0, 0) = Instruction retired (payload = cycle count delta)
( 0,+1) = Pipeline stall (payload = stall reason)
(+1,-1) = SIMT divergence (payload = lane mask)
(+1, 0) = Galaxy access (payload = star ID fragment)
(+1,+1) = Custom trace event
```

**Trace bandwidth:** 12 trits × clock rate. At 1 GHz = 12 Gtrits/s ≈ 19 Gbits/s trace bandwidth. Sufficient for full instruction trace.

---

## 22. Ternary Networking Extension (RT-N)

*This section directly addresses the Nvidia networking discussion. Nvidia's $11B/quarter networking division (NVLink, InfiniBand, Spectrum-X) is entirely binary. RISC-T defines open ternary networking to prevent proprietary lock-in of the interconnect layer.*

### 22.1 Motivation

Nvidia's networking stack (NVLink 6.0: 1.8 TB/s per GPU, InfiniBand 400G, Spectrum-X Ethernet) creates vendor lock-in BENEATH the compute layer. Even if ternary compute exists, binary networking forces data conversion at every chip boundary.

**RISC-T RT-N defines ternary-native networking** to eliminate this bottleneck.

### 22.2 Ternary Link Layer

#### Physical Layer: Ternary PAM-3

Modern Ethernet already uses PAM-4 (4 voltage levels) and PAM-3 has been used in 1000BASE-T. RISC-T's ternary link uses PAM-3 signaling:

| Voltage | Trit |
|---------|------|
| +V | +1 |
| 0 | 0 |
| -V | -1 |

**This is STANDARD ELECTRICAL SIGNALING.** PAM-3 is proven technology in Gigabit Ethernet. The physical layer requires ZERO new silicon — existing Ethernet PHY transceivers already handle three voltage levels.

**Data rate:** Each symbol = 1 trit = 1.585 bits of information. At 10 Gsymbol/s (standard SerDes rate): 15.85 Gbit/s per lane. With 4 lanes (standard QSFP): 63.4 Gbit/s — comparable to 100GbE but with ternary-native data.

#### Link Frame Format

```
Ternary Link Frame:
┌──────────┬──────────┬───────────┬────────────┬────────────┬──────────┐
│ Preamble │ Src Addr │ Dst Addr  │ Length/Type│  Payload   │  TCRC    │
│  9T      │  27T     │  27T      │   6T       │  Variable  │  12T     │
└──────────┴──────────┴───────────┴────────────┴────────────┴──────────┘
```

- **Preamble** (9T): Synchronization pattern (+1,0,-1,+1,0,-1,+1,0,-1)
- **Addresses** (27T each): 3^27 ≈ 7.6 trillion unique addresses (vs Ethernet's 2^48 = 281 trillion — smaller but sufficient for datacenter scale; extendable to 40T for global addressing)
- **Length/Type** (6T): 729 possible frame types
- **TCRC** (12T): Ternary Cyclic Redundancy Check

#### Ternary CRC (TCRC)

```
TCRC-12 polynomial (balanced ternary):
g(x) = x^12 + x^8 + x^5 + x^4 + x^0

Computed over the frame using ternary polynomial division:
- Coefficients are trits (-1, 0, +1)
- Addition modulo 3 (balanced)
- Multiplication as per single-trit TMUL truth table

Error detection: detects all 1-trit errors, all 2-trit errors,
and all burst errors up to 12 trits. Probability of undetected
error for random corruption: 1/3^12 ≈ 1.88 × 10^-6.
```

### 22.3 Ternary Network-on-Chip (TNoC)

For multi-core RT40T-Galaxy, an on-chip ternary network connects the 9 cores:

```
TNoC Topology: 3×3 Ternary Mesh

    Core[0,0] ─── Core[0,1] ─── Core[0,2]
        │              │              │
    Core[1,0] ─── Core[1,1] ─── Core[1,2]
        │              │              │
    Core[2,0] ─── Core[2,1] ─── Core[2,2]

Routing: Balanced ternary dimension-order routing
- X-coordinate: trit (-1, 0, +1)
- Y-coordinate: trit (-1, 0, +1)
- Address = (x_trit, y_trit) → 9 positions in 2 trits

Packet format (on-chip):
┌──────┬──────┬──────┬──────────┐
│ Dst  │ Src  │ Type │ Payload  │
│  2T  │  2T  │  2T  │  6T-40T  │
└──────┴──────┴──────┴──────────┘
```

**Routing decision:** At each router, compare destination trit with current position trit (1 TCOMP per dimension). If -1: route west/south. If 0: arrived. If +1: route east/north. **Routing is a single ternary comparison — no routing tables needed for the mesh.**

### 22.4 Inter-Chip Link: TLink

For multi-chip systems (cluster of RT40T-Galaxy processors), RISC-T defines TLink:

| Property | NVLink 6.0 (Nvidia) | TLink (RISC-T, open) |
|----------|-------------------|---------------------|
| Signaling | PAM-4 (proprietary encoding) | PAM-3 (ternary-native) |
| Lanes per link | 20 | 9 (3^2) |
| Bandwidth per link | 1.8 TB/s | ~143 Gtrit/s (≈ 227 Gbit/s at 10 Gsym/s × 9 lanes) |
| Topology | Star (GPU↔switch) | Mesh/torus (direct peer, no switch required) |
| Licensing | Proprietary (Nvidia ecosystem only) | Open (W3C RF) |
| Data format | Binary (requires conversion for ternary data) | Native ternary (zero conversion overhead) |

**Key advantage:** TLink carries ternary data without conversion. In a ternary compute cluster, NVLink would waste ~30% bandwidth on binary-ternary conversion overhead at every link crossing. TLink: zero conversion overhead.

### 22.5 Ternary Routing Protocol: TRP

For multi-chip ternary networks, RISC-T defines a minimal routing protocol:

```
TRP Address Space:
├── Chip ID:   9T (3^9 = 19,683 chips per cluster)
├── Core ID:   2T (9 cores per chip)
├── Thread ID: 2T (9 threads per warp)
└── Total:    13T per endpoint

TRP Packet:
┌──────────┬──────────┬──────┬────────────┐
│ Dst(13T) │ Src(13T) │Flags │ Payload    │
│          │          │ (3T) │ (variable) │
└──────────┴──────────┴──────┴────────────┘

Flags (3 trits):
├── Priority:  (-1=low, 0=normal, +1=high)
├── Type:      (-1=data, 0=control, +1=coherence)
└── Multicast: (-1=unicast, 0=row-broadcast, +1=all-broadcast)
```

**Multicast in 1 trit:** Where binary networks need separate multicast group tables, ternary networks encode three useful multicast modes in a single trit. Row-broadcast (trit=0) is particularly useful for K3D's nine-chain swarm — it broadcasts to all 9 SIMT lanes simultaneously.

---

## 23. Ternary Security Extensions (RT-S)

### 23.1 Ternary Memory Protection

**Ternary page table entries** use 3-way permissions:

| Permission Trit | Meaning |
|----------------|---------|
| -1 | Forbidden (access fault) |
| 0 | Read-only |
| +1 | Read-write |

**Three permissions** (execute, read, write) × **three levels** = 3^3 = 27 permission combinations in 3 trits. Binary needs 3 bits for the same but wastes 5 of the 8 possible bit patterns. Ternary: zero waste.

#### Page Table Format

```
Ternary Page Table Entry (40T for RT40T):
┌──────────────────────┬─────┬─────┬─────┬─────┬──────────────┐
│   Physical Page (20T) │ Exec│ Read│Write│Priv │  Flags (9T)  │
│                       │ 1T  │ 1T  │ 1T  │ 1T  │              │
└──────────────────────┴─────┴─────┴─────┴─────┴──────────────┘

Flags:
├── Valid:    (-1=invalid, 0=valid-not-accessed, +1=valid-accessed)
├── Dirty:   (-1=clean, 0=not-applicable, +1=dirty)
├── Global:  (-1=local, 0=shared-readonly, +1=global-shared)
├── Cached:  (-1=uncacheable, 0=write-through, +1=write-back)
├── User:    (-1=kernel-only, 0=supervisor, +1=user-accessible)
├── Galaxy:  (-1=not-galaxy, 0=galaxy-readonly, +1=galaxy-read-write)
├── Encrypt: (-1=plaintext, 0=integrity-only, +1=encrypted)
├── Trust:   (-1=untrusted, 0=measured, +1=trusted)
└── Reserved
```

### 23.2 Ternary Trust Model

RISC-T defines a **three-way trust model** for hardware security:

| Trust Trit | Meaning | Boot Verification |
|-----------|---------|-------------------|
| -1 | **Untrusted** | No verification; code runs in sandbox |
| 0 | **Measured** | Hash recorded in TPM-equivalent; can be verified post-hoc |
| +1 | **Trusted** | Cryptographic signature verified before execution |

**Ternary advantage:** Binary systems use a single "secure boot" bit — all or nothing. RISC-T's three-way trust allows a MEASURED state where code runs but its hash is recorded, enabling post-hoc auditing without blocking execution. This is critical for open-source ecosystems where not all code is signed but auditability is desired.

### 23.3 Ternary Encryption Support

```
Instructions:
TAES rd, rs1, rs2     ; Ternary AES-equivalent block cipher (3^6 = 729-state S-box)
THASH rd, rs1         ; Ternary hash (3-way Merkle-Damgård construction)
TRAND rd              ; Ternary random number (hardware RNG, outputs -1/0/+1 per trit)
TPUF rd               ; Physically Unclonable Function (ternary PUF — 3× entropy per element)
```

**Ternary PUF (Physically Unclonable Function):** Each PUF element produces a trit instead of a bit, providing 58.5% more entropy per physical element. A 40-trit PUF response has 3^40 ≈ 1.22 × 10^19 possible values (equivalent to ~63.4 bits of entropy).

---

## 24. HDL Reference Implementation (Pseudocode)

*This section provides hardware description pseudocode for key RISC-T components. Written in a Verilog-like syntax with ternary extensions. This constitutes enabling description for silicon implementation and strengthens the prior art claim.*

### 24.1 Ternary Inverter (TNOT Gate)

```verilog
// RISC-T Ternary Inverter — Zero gates, wire swap only
// This is the fundamental elegance of balanced ternary in hardware.
//
// Prior art: This wire-swap implementation of balanced ternary negation
// is hereby published as open prior art under W3C RF license.

module ternary_inverter (
    input  trit a,      // 2-wire encoding: {wire_pos, wire_neg}
    output trit result   // 2-wire encoding: {wire_pos, wire_neg}
);
    // Negation IS the wire swap. No logic gates.
    assign result.wire_pos = a.wire_neg;
    assign result.wire_neg = a.wire_pos;
    // When a = +1: wire_pos=1, wire_neg=0 → result wire_pos=0, wire_neg=1 → result = -1
    // When a =  0: wire_pos=0, wire_neg=0 → result wire_pos=0, wire_neg=0 → result =  0
    // When a = -1: wire_pos=0, wire_neg=1 → result wire_pos=1, wire_neg=0 → result = +1
    // Gate count: 0. Delay: 0. Power: 0.
endmodule
```

### 24.2 Ternary Half Adder

```verilog
// RISC-T Ternary Half Adder
// Based on Daniel's three-way relay model:
// Rest = 0, Contact A = +1, Contact B = -1
//
// Prior art: Relay-state ternary half adder implementation.
// This is specified behaviorally from the native 3-state primitive.
// It does not require a unary increment/decrement gate family.

module ternary_half_adder (
    input  trit a,
    input  trit b,
    output trit sum,
    output trit carry
);
    // Internal 2-bit encoding: 10 = +1, 00 = 0, 01 = -1
    // (Same as K3D TPACK encoding)

    wire [1:0] a_bin = trit_to_binary(a);  // Convert trit to 2-bit
    wire [1:0] b_bin = trit_to_binary(b);

    // Ternary addition: compute (a + b) as integer, then extract sum and carry
    // a, b ∈ {-1, 0, +1}, so a+b ∈ {-2, -1, 0, +1, +2}
    // In balanced ternary:
    //   -2 = -1×3¹ + 1×3⁰ → carry=-1, sum=+1
    //   -1 = 0×3¹ + (-1)×3⁰ → carry=0, sum=-1
    //    0 = 0×3¹ + 0×3⁰ → carry=0, sum=0
    //   +1 = 0×3¹ + 1×3⁰ → carry=0, sum=+1
    //   +2 = 1×3¹ + (-1)×3⁰ → carry=+1, sum=-1

    // Sum logic (behavioral relay-state implementation):
    // sum = (a + b) mod 3 (balanced)
    // Using three-way relay model:
    //   If a and b same non-zero → sum flips sign, carry propagates
    //   If a and b opposite → sum = 0, carry = 0
    //   If either zero → sum = the other, carry = 0

    wire both_positive = (a == +1) & (b == +1);
    wire both_negative = (a == -1) & (b == -1);
    wire cancel        = (a == +1 & b == -1) | (a == -1 & b == +1);

    assign sum   = both_positive ? TRIT_NEG :    // +1 + +1 → sum = -1
                   both_negative ? TRIT_POS :    // -1 + -1 → sum = +1
                   cancel        ? TRIT_ZERO :   // +1 + -1 → sum = 0
                   /* one is zero */ (a | b);    // 0 + x → sum = x

    assign carry = both_positive ? TRIT_POS :    // +1 + +1 → carry = +1
                   both_negative ? TRIT_NEG :    // -1 + -1 → carry = -1
                   /* else */      TRIT_ZERO;    // no carry

    // Gate count / transistor topology intentionally unspecified.
    // The architecture standardizes behavior, not one cell layout.
endmodule
```

### 24.3 Ternary Full Adder

```verilog
// RISC-T Ternary Full Adder — composed of two half adders
// Prior art: Ternary full adder with single-trit carry propagation.

module ternary_full_adder (
    input  trit a,
    input  trit b,
    input  trit cin,
    output trit sum,
    output trit cout
);
    trit s1, c1, c2;

    ternary_half_adder ha1 (.a(a), .b(b), .sum(s1), .carry(c1));
    ternary_half_adder ha2 (.a(s1), .b(cin), .sum(sum), .carry(c2));

    // At most one carry can be non-zero (c1 and c2 cannot both be non-zero)
    // Therefore TOR (max) gives the correct carry output
    assign cout = ternary_or(c1, c2);

    // Gate count / transistor topology intentionally unspecified.
    // Carry is a single trit — no special carry format needed.
endmodule
```

### 24.4 20-Trit Ripple-Carry Adder

```verilog
// RISC-T 20-Trit Adder for RT20T
// Prior art: Multi-trit balanced ternary ripple-carry adder.

module ternary_adder_20t (
    input  trit [19:0] a,
    input  trit [19:0] b,
    input  trit cin,
    output trit [19:0] sum,
    output trit cout,
    output trit overflow
);
    trit carry [20:0];
    assign carry[0] = cin;

    genvar i;
    generate
        for (i = 0; i < 20; i = i + 1) begin : adder_chain
            ternary_full_adder fa (
                .a(a[i]), .b(b[i]), .cin(carry[i]),
                .sum(sum[i]), .cout(carry[i+1])
            );
        end
    endgenerate

    assign cout = carry[20];

    // Overflow: carry into MST differs from carry out of MST
    assign overflow = (carry[19] != carry[20]) ? TRIT_POS : TRIT_ZERO;

    // Total gate count: ~560 MOSFETs (20 × 28)
    // Worst-case delay: 80 gate levels (20 × 4 per full adder)
    // With ternary carry lookahead: ~20 gate levels
endmodule
```

### 24.5 Ternary Comparator (TCOMP)

```verilog
// RISC-T Three-Way Comparator
// Returns +1 if a > b, 0 if a == b, -1 if a < b
// Prior art: Single-instruction three-way comparison in balanced ternary.

module ternary_comparator (
    input  trit [WIDTH-1:0] a,
    input  trit [WIDTH-1:0] b,
    output trit result
);
    parameter WIDTH = 20;

    // Compare from most significant trit downward
    // First non-zero difference determines result
    trit diff [WIDTH-1:0];

    genvar i;
    generate
        for (i = 0; i < WIDTH; i = i + 1) begin : diff_chain
            // Per-trit difference: simple subtraction
            ternary_half_adder sub (
                .a(a[i]),
                .b(ternary_not(b[i])),  // Subtract = add negation (FREE via wire swap)
                .sum(diff[i]),
                .carry()                 // Ignore carry for comparison
            );
        end
    endgenerate

    // Priority encoder: find most significant non-zero diff
    // This is a ternary cascading comparison
    reg trit result_reg;
    integer j;
    always @(*) begin
        result_reg = TRIT_ZERO;  // Default: equal
        for (j = WIDTH-1; j >= 0; j = j - 1) begin
            if (diff[j] != TRIT_ZERO && result_reg == TRIT_ZERO)
                result_reg = diff[j];  // First non-zero diff = result
        end
    end
    assign result = result_reg;

    // Key property: Comparison result IS a trit. Three-way branch follows naturally.
    // No separate "less than", "equal", "greater than" flags — one trit encodes all three.
endmodule
```

### 24.6 Three-Way Branch Unit

```verilog
// RISC-T Three-Way Branch — TB3 instruction hardware
// Single instruction dispatches to three possible targets
// Prior art: Three-way branch instruction using ternary condition trit.

module three_way_branch (
    input  trit condition,           // From TCOMP result or flags register
    input  trit_word target_neg,     // Jump target if condition = -1
    input  trit_word target_zero,    // Jump target if condition = 0
    input  trit_word target_pos,     // Jump target if condition = +1
    input  trit_word pc_current,     // Current program counter
    output trit_word pc_next,        // Next program counter
    output trit branch_taken         // +1=taken_pos, 0=fall_through, -1=taken_neg
);
    // Ternary MUX: select target based on condition trit
    // This is a single multiplexer layer — NOT two cascaded binary MUXes

    assign pc_next = (condition == TRIT_POS)  ? target_pos :
                     (condition == TRIT_ZERO) ? target_zero :
                     /* TRIT_NEG */             target_neg;

    assign branch_taken = condition;

    // Hardware: 3:1 ternary multiplexer (2 multi-threshold MOSFETs per trit of word width)
    // For RT20T: 40 MOSFETs total (20 trits × 2)
    // For RT40T: 80 MOSFETs total
    // Delay: 1 gate level
    //
    // Compare to binary: two cascaded 2:1 MUXes = 2 gate levels, and only handles 2 outcomes
    // RISC-T handles 3 outcomes in 1 gate level — 2× faster branch resolution
endmodule
```

### 24.7 SIMT Lane with Three-Way Predication

```verilog
// RISC-T SIMT Lane — Three-way predicated execution
// Prior art: Ternary SIMT predication (active/masked/inverted).

module simt_lane (
    input  trit predicate,           // +1=active, 0=masked, -1=inverted
    input  trit_word operand_a,
    input  trit_word operand_b,
    input  [2:0] alu_op,            // Operation selector
    output trit_word result,
    output trit lane_active
);
    trit_word alu_result;
    trit_word inverted_result;

    // Normal ALU execution
    ternary_alu alu (
        .a(operand_a), .b(operand_b), .op(alu_op), .result(alu_result)
    );

    // Inverted execution: negate the result (FREE — wire swap)
    ternary_inverter_word inv (.a(alu_result), .result(inverted_result));

    // Three-way predication MUX
    assign result = (predicate == TRIT_POS)  ? alu_result :       // Active: normal
                    (predicate == TRIT_ZERO) ? operand_a :         // Masked: pass-through (no-op)
                    /* TRIT_NEG */             inverted_result;    // Inverted: negate

    assign lane_active = predicate;

    // Binary SIMT: if/else requires TWO passes (one for if-lanes, one for else-lanes)
    // Ternary SIMT: if/else/neither in ONE pass with three-way predication
    // Speedup: up to 2× for divergent code vs binary SIMT
endmodule
```

### 24.8 Morton Encoder (Ternary 3D Spatial Index)

```verilog
// RISC-T Morton Encoder — TMORTON instruction hardware
// Interleaves three ternary coordinates into a single Morton code
// Used by K3D Galaxy navigation for spatial indexing
// Prior art: Balanced ternary Morton code for 3D spatial indexing.

module ternary_morton_encoder (
    input  trit [N-1:0] x,    // X coordinate (N trits)
    input  trit [N-1:0] y,    // Y coordinate
    input  trit [N-1:0] z,    // Z coordinate
    output trit [3*N-1:0] morton  // Morton code (3N trits)
);
    parameter N = 7;  // 7 trits per axis → 21-trit Morton code (fits RT20T)

    // Interleave: Z₆Y₆X₆ Z₅Y₅X₅ ... Z₀Y₀X₀
    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : interleave
            assign morton[3*i + 0] = x[i];
            assign morton[3*i + 1] = y[i];
            assign morton[3*i + 2] = z[i];
        end
    endgenerate

    // Gate count: 0 (pure wiring, like negation)
    // Delay: 0
    // Power: 0
    //
    // This is another case where ternary hardware shines:
    // Morton interleaving is free because it's just wire routing.
    // The ternary Morton code provides 3^7 = 2187 positions per axis,
    // giving 3^21 ≈ 10 billion unique 3D cells — enough for K3D Galaxy resolution.
endmodule
```

### 24.9 Nine-Vote Swarm Consensus (TVOTE)

```verilog
// RISC-T Nine-Vote Swarm Consensus — TVOTE instruction hardware
// K3D's nine-chain swarm voting: 9 SIMT lanes vote on an answer
// Prior art: Ternary 9-lane swarm consensus hardware with three-way majority.

module ternary_swarm_vote (
    input  trit [8:0] votes,     // 9 votes, one per swarm chain
    output trit consensus,       // +1 = positive majority, -1 = negative majority, 0 = no consensus
    output trit [3:0] strength   // Margin: how many agree (0 to 9, encoded in 4 trits)
);
    // Count positives, negatives, and neutrals
    // Each is 0..9, which fits in 2 trits (3^2 = 9)
    trit [1:0] count_pos, count_neg, count_zero;

    integer i;
    always @(*) begin
        count_pos = 0; count_neg = 0; count_zero = 0;
        for (i = 0; i < 9; i = i + 1) begin
            case (votes[i])
                TRIT_POS:  count_pos = count_pos + 1;
                TRIT_NEG:  count_neg = count_neg + 1;
                TRIT_ZERO: count_zero = count_zero + 1;
            endcase
        end
    end

    // Three-way majority:
    // If positives > negatives AND positives > neutrals → positive consensus
    // If negatives > positives AND negatives > neutrals → negative consensus
    // Otherwise → no consensus (0)
    trit pos_vs_neg, pos_vs_zero, neg_vs_zero;

    ternary_comparator #(.WIDTH(2)) cmp1 (.a(count_pos), .b(count_neg),  .result(pos_vs_neg));
    ternary_comparator #(.WIDTH(2)) cmp2 (.a(count_pos), .b(count_zero), .result(pos_vs_zero));
    ternary_comparator #(.WIDTH(2)) cmp3 (.a(count_neg), .b(count_zero), .result(neg_vs_zero));

    assign consensus = (pos_vs_neg == TRIT_POS && pos_vs_zero == TRIT_POS) ? TRIT_POS :
                       (pos_vs_neg == TRIT_NEG && neg_vs_zero == TRIT_POS) ? TRIT_NEG :
                       TRIT_ZERO;

    // Strength = count of winning trit
    assign strength = (consensus == TRIT_POS)  ? count_pos :
                      (consensus == TRIT_NEG)  ? count_neg :
                      /* no consensus */         count_zero;

    // Gate count: ~90 MOSFETs (3 counters + 3 comparators + 2 MUXes)
    // Delay: ~8 gate levels
    // Cycles: 1 (fully combinational — all 9 votes resolved in 1 clock)
    //
    // Binary equivalent would need 32-thread warp reduction → 5 levels of binary comparison
    // RISC-T: 9 votes → 2 levels of ternary comparison (log₃(9) = 2)
endmodule
```

---

## Appendix D: Extension Registry

RISC-T follows RISC-V's extension naming convention. Extensions use a single-letter (standard) or multi-letter (custom) prefix:

| Extension | Name | Status | Description |
|-----------|------|--------|-------------|
| RT-I | Integer Base | Required | Base integer ISA (§4) |
| RT-M | Multiply/Divide | Standard | Hardware multiply/divide unit |
| RT-F | Floating-Point | Standard | Ternary floating-point (§15) |
| RT-A | Atomics | Standard | Ternary atomic memory operations |
| RT-C | Compressed | Standard | 6-trit compressed instructions (future) |
| RT-V | Vector | Standard | Ternary vector/SIMD operations |
| RT-G | GPU/SIMT | Standard | Ternary SIMT execution model (§9) |
| RT-K | Knowledge | Standard | K3D ternary knowledge ops (§10) |
| RT-N | Networking | Standard | Ternary networking (§22) |
| RT-S | Security | Standard | Ternary security extensions (§23) |
| RT-D | Debug | Standard | Debug/trace interface (§21) |
| RT-P | Power | Standard | Power management (§20) |

**Conformance levels:**

| Level | Required Extensions | Target |
|-------|-------------------|--------|
| RT-Minimal | RT-I | FPGA prototype, education |
| RT-Embedded | RT-I + RT-M + RT-A | IoT, sensors |
| RT-General | RT-I + RT-M + RT-F + RT-A + RT-S | Workstations, servers |
| RT-Knowledge | RT-General + RT-G + RT-K | K3D native hardware |
| RT-Galaxy | RT-Knowledge + RT-N + RT-P | Multi-chip K3D cluster |

---

## 25. Open Ternary GPU Architecture (RT-G Full Specification)

*This section expands §9 into a complete GPU microarchitecture specification, drawing from MIAOW (AMD GCN model), Vortex (RISC-V GPGPU), Nyuzi (vector GPGPU), RV64X (RISC-V graphics extensions), and the Vulkan/SPIR-V compute model. All designs are adapted from binary to ternary, with ternary-specific innovations published as prior art.*

### 25.1 Design Lineage and Rationale

| Source Architecture | What RISC-T Adopts | What RISC-T Changes |
|--------------------|--------------------|---------------------|
| **MIAOW** (AMD GCN RTL) | Compute Unit model, wavefront pool, vector/scalar split, LDS | Wavefront=9 threads (not 64), ternary ALUs, 3-way predication |
| **Vortex** (RISC-V GPGPU) | RISC ISA base + SIMT overlay, coherent cache hierarchy, Vulkan pipeline | Base ISA = RISC-T (not RISC-V), ternary-native caches, 9-thread warps |
| **Nyuzi** (vector GPGPU) | Unified scalar/vector ISA, per-lane predication, fine-grained multithreading | Predication is ternary (+1/0/-1), vector width = 9 lanes (not 16) |
| **AMD GCN/RDNA** (proprietary) | Dual-issue scalar+vector, LDS banking, wavefront scheduling | Open specification, ternary-native, royalty-free |
| **NVIDIA CUDA** (proprietary) | Warp scheduling, register file sizing, shared memory model | Open, 9-thread warps (not 32), 3-way divergence (not 2-way) |
| **Intel Xe** (proprietary) | EU pairing, SLM architecture, matrix engines | Open, ternary-native matrix unit |
| **Vulkan/SPIR-V** (open standard) | Workgroup/subgroup model, memory scopes, subgroup operations | Extended with ternary subgroup ops (3-way ballot, ternary shuffle) |

### 25.2 Ternary Compute Unit (TCU) Microarchitecture

```
Ternary Compute Unit (TCU) — RT-G Full Model
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║  ┌──────────────────────────────────────────────────────────────┐    ║
║  │                    INSTRUCTION FETCH                         │    ║
║  │  ┌────────────┐  ┌──────────────────────────────────────┐   │    ║
║  │  │ I-Cache    │  │ Wavefront Pool (27 wavefront slots)  │   │    ║
║  │  │ 729 tryte- │  │ Per-slot: PC(40T) + mask(9T) +      │   │    ║
║  │  │ lines      │  │ state(3T) + priority(1T)             │   │    ║
║  │  │ 3-way      │  │ Total: 27 slots × 53T = 1431 trits  │   │    ║
║  │  └────────────┘  └──────────────────────────────────────┘   │    ║
║  └───────────────────────────┬──────────────────────────────────┘    ║
║                              ↓                                       ║
║  ┌──────────────────────────────────────────────────────────────┐    ║
║  │                    DECODE + SCHEDULE                          │    ║
║  │  ┌─────────────────┐  ┌──────────────────────────────────┐  │    ║
║  │  │ Ternary Decoder │  │ Dual-Issue Scheduler              │  │    ║
║  │  │ (12T or 24T     │  │ Issue 1: Scalar instruction       │  │    ║
║  │  │  instruction)   │  │ Issue 2: Vector instruction       │  │    ║
║  │  │                 │  │ 3-way priority: -1=low,0=med,+1=hi│  │    ║
║  │  └─────────────────┘  └──────────────────────────────────┘  │    ║
║  └──────────────┬─────────────────────────┬─────────────────────┘    ║
║                 ↓ scalar                  ↓ vector                   ║
║  ┌──────────────────────┐  ┌────────────────────────────────────┐   ║
║  │  SCALAR UNIT          │  │  VECTOR UNIT (3 × SIMD3)          │   ║
║  │  ┌──────────────────┐│  │  ┌────────────────────────────┐   │   ║
║  │  │ Scalar ALU       ││  │  │ SIMD3-A (lanes 0,1,2)     │   │   ║
║  │  │ (TADD,TCOMP,     ││  │  │ 3 ternary ALUs            │   │   ║
║  │  │  branch logic)   ││  │  └────────────────────────────┘   │   ║
║  │  ├──────────────────┤│  │  ┌────────────────────────────┐   │   ║
║  │  │ Scalar Reg File  ││  │  │ SIMD3-B (lanes 3,4,5)     │   │   ║
║  │  │ 27 × 40T regs   ││  │  │ 3 ternary ALUs            │   │   ║
║  │  ├──────────────────┤│  │  └────────────────────────────┘   │   ║
║  │  │ Branch Unit      ││  │  ┌────────────────────────────┐   │   ║
║  │  │ (3-way TB3)      ││  │  │ SIMD3-C (lanes 6,7,8)     │   │   ║
║  │  └──────────────────┘│  │  │ 3 ternary ALUs            │   │   ║
║  └──────────────────────┘  │  └────────────────────────────┘   │   ║
║                            │  Total: 9 ALUs = 1 full warp/cycle │   ║
║                            │  ┌────────────────────────────┐   │   ║
║                            │  │ Vector Reg File             │   │   ║
║                            │  │ 27 regs × 9 lanes × 40T    │   │   ║
║                            │  │ = 9,720 trits per wavefront │   │   ║
║                            │  │ × 27 wavefronts = banked    │   │   ║
║                            │  └────────────────────────────┘   │   ║
║                            └────────────────────────────────────┘   ║
║                                                                      ║
║  ┌──────────────────────────────────────────────────────────────┐    ║
║  │  SPECIAL FUNCTION UNITS                                       │    ║
║  │  ┌───────────────┐ ┌───────────────┐ ┌────────────────────┐ │    ║
║  │  │ Ternary FPU   │ │ Galaxy Nav    │ │ Ternary Matrix     │ │    ║
║  │  │ (TFloat20/40) │ │ Unit (GNU)    │ │ Unit (TMU)         │ │    ║
║  │  │ 5-stage pipe  │ │ Morton+Frustum│ │ 3×3 trit-matrix    │ │    ║
║  │  │ 3 units       │ │ +LOD+A*       │ │ multiply-accumulate│ │    ║
║  │  └───────────────┘ └───────────────┘ └────────────────────┘ │    ║
║  └──────────────────────────────────────────────────────────────┘    ║
║                                                                      ║
║  ┌──────────────────────────────────────────────────────────────┐    ║
║  │  MEMORY SUBSYSTEM                                             │    ║
║  │  ┌────────────────┐  ┌──────────────────────────────────┐   │    ║
║  │  │ L1 Data Cache  │  │ Local Trit Share (LTS)           │   │    ║
║  │  │ 729 lines      │  │ 6561 trytes (3^8)               │   │    ║
║  │  │ 3-way assoc    │  │ 27 banks × 243 trytes/bank      │   │    ║
║  │  │ Write-back     │  │ All-to-all ternary crossbar      │   │    ║
║  │  │ TMESI coherent │  │ 27 atomic trit units             │   │    ║
║  │  └────────────────┘  └──────────────────────────────────┘   │    ║
║  └──────────────────────────────────────────────────────────────┘    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

### 25.3 Wavefront Organization (Ternary SIMT)

**Critical design: 9-thread warps executed on 3×SIMD3 units.**

| Property | AMD GCN | NVIDIA CUDA | Intel Xe | **RISC-T GPU** |
|----------|---------|-------------|----------|----------------|
| Warp/wavefront size | 64 | 32 | 8 (EU) | **9 (3²)** |
| SIMD width | 16 | 32 | 8 | **3 (SIMD3)** |
| Cycles to execute 1 wavefront | 4 | 1 | 1 | **1** (3×SIMD3 parallel) |
| Max wavefronts per CU | 40 | 64 | 56 | **27 (3³)** |
| Max threads per CU | 2560 | 2048 | 448 | **243 (27×9)** |
| Predication | 1-bit (on/off) | 1-bit | 1-bit | **1-trit (+1/0/-1)** |
| Branch divergence penalty | 2× (if/else) | 2× | 2× | **1× (three-way)** |

**Three-way warp execution model:**

```
Binary GPU divergence (2 passes):
  if (condition)     → Pass 1: execute if-branch (mask else-threads)
  else               → Pass 2: execute else-branch (mask if-threads)
  Total: 2 passes for if/else

Ternary GPU divergence (1 pass):
  TB3 condition      → All 9 lanes execute simultaneously:
                        Lanes with condition=+1 → execute positive path
                        Lanes with condition=0  → execute neutral path (NOP or fall-through)
                        Lanes with condition=-1 → execute INVERTED path (via wire swap)
  Total: 1 pass for if/else/neither

  Speedup: Up to 2× for divergent code
  Mechanism: Three-way predication (§24.7 HDL pseudocode)
```

### 25.4 Wavefront Scheduling

Drawing from MIAOW's 40-wavefront pool and Vortex's RISC-V scheduling:

**Ternary Round-Robin with Priority (TRR-P):**

```
27 wavefront slots organized as 3 priority tiers:
  Tier +1 (high):   9 slots — currently executing, data available
  Tier  0 (normal): 9 slots — ready, waiting for schedule
  Tier -1 (low):    9 slots — stalled on memory or synchronization

Scheduling algorithm:
1. Select highest non-empty tier (TCOMP against zero: 1 cycle)
2. Within tier, round-robin across 9 slots (modulo-3 counter: 0 gates)
3. Issue selected wavefront's next instruction to decode
4. If wavefront stalls (memory miss), demote to tier -1
5. If stalled wavefront's data arrives, promote to tier 0
6. Currently executing wavefront occupies tier +1

Promotion/demotion: single-trit write to slot's priority field
Selection: 2-level ternary comparator tree (3 cycles)
```

**Ternary advantage:** Priority is a single trit (not multi-bit). Tier selection is one TCOMP. Binary schedulers need a priority encoder over N-bit scores — ternary needs a single trit comparison.

### 25.5 Register File Architecture

Adapted from GCN's split scalar/vector model and Nyuzi's unified approach:

**Scalar Register File (SRF):**

```
27 scalar registers × 40 trits (RT40T) per register = 1,080 trits
Shared across all wavefronts in the CU
Used for: branch conditions, loop counters, constants, addresses
Access: 1 read + 1 write per cycle (single-port)
```

**Vector Register File (VRF):**

```
Per-wavefront allocation:
  27 vector registers × 9 lanes × 40 trits = 9,720 trits per wavefront
  27 wavefronts × 9,720 = 262,440 trits total VRF per TCU

Banking: 27 banks (3³), one per register index
  Read ports: 3 (one per SIMD3 unit)
  Write ports: 1 (shared, arbitrated)
  Bank conflicts: resolved by ternary rotation (shift-by-trit)

Physical layout:
  ┌─────────────────────────────────────────────────────┐
  │ Bank 0: r0[lane0..8] for all 27 wavefronts         │
  │ Bank 1: r1[lane0..8] for all 27 wavefronts         │
  │ ...                                                  │
  │ Bank 26: r26[lane0..8] for all 27 wavefronts       │
  └─────────────────────────────────────────────────────┘

Binary comparison:
  AMD GCN: 256 VGPRs × 64 lanes × 32 bits = 524,288 bits per CU
  RISC-T:  27 VTRs × 9 lanes × 40 trits ≈ 15,390 bits equivalent per wavefront
           × 27 wavefronts ≈ 415,530 bits equivalent per TCU
  Similar density, but ternary trits carry 58.5% more information per element
```

### 25.6 Local Trit Share (LTS) — Ternary Scratchpad

Adapted from AMD's LDS (Local Data Share) and Intel's SLM (Shared Local Memory):

```
Local Trit Share (LTS) Specification:
├── Total size: 6561 trytes (3^8 × 6 trits/tryte = 39,366 trits)
├── Banking: 27 banks (3³)
│   ├── Each bank: 243 trytes (1,458 trits)
│   ├── Bank width: 1 tryte per cycle per bank
│   └── Access latency: 1 cycle (same as register)
├── Crossbar: 27×27 ternary crossbar
│   ├── Full all-to-all connectivity
│   └── Bank conflict resolution: ternary address rotation
├── Atomic units: 27 ternary atomic operators
│   ├── TADD atomic (trit-word add)
│   ├── TMAX/TMIN atomic (ternary max/min)
│   ├── TCAS atomic (ternary compare-and-swap)
│   └── TXCHG atomic (ternary exchange)
└── Access model: Shared by all wavefronts in the CU
    └── Allocation: per-workgroup (like CUDA shared memory)

Binary comparison:
  AMD GCN LDS: 64 KB, 32 banks
  RISC-T LTS:  6561 trytes ≈ 62,275 bits equivalent, 27 banks
  Comparable capacity with natural ternary banking (no waste)
```

### 25.7 Ternary Texture Unit (TTU) — Knowledge Star Sampler

Binary GPUs have texture units for sampling images. RISC-T repurposes this as a **Knowledge Star Sampler** for K3D Galaxy lookups:

```
Ternary Texture Unit (TTU):
├── Star Lookup Pipeline (4 stages):
│   Stage 1: Morton address decode (trit-interleave → 3D coordinates)
│   Stage 2: Neighborhood fetch (3^3 = 27 neighboring stars)
│   Stage 3: Ternary interpolation (semantic gravity weighting)
│   Stage 4: Result formatting (star → vector register)
├── Star Cache: 729 entries (3^6)
│   ├── 3-way set-associative
│   ├── Per-entry: star_id(27T) + position(60T) + meaning_hash(40T)
│   │             + domain(6T) + rpn_ptr(20T) = 153 trits
│   └── Total: 729 × 153 = 111,537 trits ≈ 13.7 KB equivalent
├── Ternary Filtering Modes:
│   ├── Nearest:  pick closest star (1 lookup)
│   ├── Trilinear: interpolate 3 closest (3 lookups + TADD)
│   └── Semantic:  gravity-weighted blend (up to 27 lookups + TGRAV)
└── Throughput: 1 star lookup per cycle per TTU
```

**This replaces traditional texture sampling with knowledge retrieval.** The same hardware pipeline that a GPU uses to sample pixels from a texture, RISC-T uses to sample knowledge stars from the Galaxy. The 4-stage pipeline is identical in structure — only the data being sampled changes.

### 25.8 Ternary Matrix Unit (TMU)

Adapted from Intel Xe's matrix engines and NVIDIA's tensor cores for AI workloads:

```
Ternary Matrix Unit (TMU):
├── Matrix size: 3×3 trit-matrices (natural ternary dimension)
├── Operation: Ternary Matrix Multiply-Accumulate (TMMA)
│   Result[3×3] = A[3×3] × B[3×3] + C[3×3]
│   All operations in balanced ternary
├── Per-cycle throughput: 1 TMMA (3×3 × 3×3 = 27 TMADs per cycle)
│   Each TMAD = 1 TMUL + 1 TADD = 2 ternary operations
│   Total: 54 ternary ops per cycle per TMU
├── Trit width: 40T per matrix element (RT40T precision)
├── Pipeline: 3 stages (load A, multiply, accumulate)
└── Use cases:
    ├── TRM attention: 3×3 query-key dot products
    ├── Galaxy navigation: 3D rotation matrices
    ├── Semantic gravity: pairwise distance matrices
    └── General AI: ternary neural network layers

Binary comparison:
  NVIDIA Tensor Core: 4×4 FP16 matrix multiply = 128 FP16 ops/cycle
  Intel AMX:          16×16 INT8 tiles
  RISC-T TMU:         3×3 ternary TMMA = 54 ternary ops/cycle
                      (each trit carries 1.585× info of a bit)
                      Effective: ~85 binary-equivalent ops/cycle
```

### 25.9 Ternary Graphics Pipeline (RT-G Graphics Extension)

Adapted from Vortex/Skybox's Vulkan pipeline and the Vulkan/SPIR-V standard:

```
RISC-T Graphics Pipeline:
┌──────────────┐
│ Vertex Input  │ Vertex buffer → position(3×40T) + attributes
├──────────────┤
│ Vertex Shader │ Runs on TCU (programmable, RISC-T ISA)
│  ternary      │ 3D transforms via TMU (3×3 matrix multiply)
│  transform    │ Output: clip-space position (4×40T homogeneous)
├──────────────┤
│ Primitive     │ Triangle assembly
│ Assembly      │ Ternary winding order: +1=CW, 0=degenerate, -1=CCW
├──────────────┤
│ Rasterizer    │ Hardware fixed-function
│  (ternary)    │ Edge function evaluation in balanced ternary
│               │ Inside/outside/edge → natural trit result (+1/0/-1)
│               │ Sub-trit precision for anti-aliasing
├──────────────┤
│ Fragment      │ Runs on TCU (programmable, RISC-T ISA)
│ Shader        │ Galaxy star sampling via TTU
│  (ternary)    │ Ternary blending: attract(+1), neutral(0), repel(-1)
├──────────────┤
│ Output Merger │ Depth test: ternary comparison (1 TCOMP)
│  & Blending   │ Three-way stencil: pass(+1), clip(0), fail(-1)
└──────────────┘
```

**Ternary rasterization advantage:**

The fundamental operation in rasterization is the edge function: "is this pixel inside, outside, or ON the edge of the triangle?" Binary rasterizers compute a float and check the sign (positive/negative/zero). This requires floating-point comparison with an epsilon tolerance for the "on edge" case.

Ternary rasterizers produce a trit directly:
- +1 = inside triangle
- 0 = on edge (exact, not approximate — balanced ternary has a native zero)
- -1 = outside triangle

**No epsilon tolerance needed.** This eliminates an entire class of rasterization bugs (T-junctions, edge-case artifacts) that binary rasterizers have fought since the 1990s.

### 25.10 Workgroup / Subgroup Model (Vulkan-Compatible)

Following the Vulkan/SPIR-V compute model, adapted for ternary:

```
RISC-T Compute Hierarchy:
├── Grid (dispatch): N × M × P workgroups
│   ├── Dimensions encoded in balanced ternary
│   └── Max grid size: 3^13 per axis ≈ 1.6 million
├── Workgroup: Up to 243 threads (27 wavefronts × 9 threads)
│   ├── Shared LTS (6561 trytes)
│   ├── Barrier synchronization (all wavefronts in workgroup)
│   └── Workgroup ID: 3 × 40T (one per grid dimension)
├── Subgroup = Wavefront = 9 threads
│   ├── Lock-step execution (SIMT)
│   ├── Subgroup operations (below)
│   └── Subgroup ID: 2T (0..8 within workgroup)
└── Thread (invocation): 1 lane in a wavefront
    ├── Thread ID: local(2T within subgroup) + subgroup(2T) + workgroup(3×40T)
    └── Private registers: 27 × 40T = 1,080 trits
```

**Ternary Subgroup Operations:**

| Operation | Binary GPU | RISC-T GPU | Advantage |
|-----------|-----------|------------|-----------|
| **Ballot** | 32-bit mask (which lanes are true) | 9-trit mask (which lanes are +1/0/-1) | 3 states per lane, not 2 |
| **Shuffle** | move data between lanes (any-to-any) | TSHUFFLE rd, rs, lane_id(2T) | 9 lanes indexed by 2 trits |
| **Reduce** | sum/min/max across lanes | TREDUCE rd, rs, op(2T) | 3-valued reduction (consensus) |
| **Broadcast** | one lane to all | TBCAST rd, rs, lane(2T) | Same, but lane ID is 2T |
| **Vote** | all/any/none (binary) | TVOTE: positive/negative/neutral majority | Three-way consensus |
| **Quad ops** | 2×2 quad (4 lanes) | **Triad ops**: 3 lanes (natural ternary group) | 3-lane triad replaces 4-lane quad |

**Ternary Ballot:**

```
Binary ballot(predicate):  returns 32-bit mask — bit[i] = 1 if lane[i] is true
Ternary ballot(predicate): returns 9-trit mask — trit[i] = predicate of lane[i]

Example (9 lanes voting on a condition):
  Lane:      0    1    2    3    4    5    6    7    8
  Predicate: +1   +1   -1   0    +1   -1   0    +1   -1

  Binary ballot: 110010010 (only knows true/false, loses the 0 info)
  Ternary ballot: (+1,+1,-1,0,+1,-1,0,+1,-1) = full 3-way information

  TVOTE on this: count_pos=4, count_neg=3, count_zero=2 → consensus=+1 (positive majority)
```

### 25.11 Ternary Rasterizer Hardware

```
Ternary Rasterizer (fixed-function unit):
├── Input: Triangle vertices (3 × position[3×40T])
├── Edge Function Evaluator:
│   For each edge E(x,y) = (v1.y - v0.y)(x - v0.x) - (v1.x - v0.x)(y - v0.y)
│   Result is a balanced ternary value:
│     Positive (+1..+MAX) → inside (encode as +1)
│     Zero                → on edge (encode as 0)
│     Negative (-1..-MAX) → outside (encode as -1)
│   Three edges evaluated in parallel → 3 trits per pixel
│   Fragment generated when all 3 trits ≥ 0 (TAND ≥ 0)
├── Tile-Based Traversal:
│   Tiles = 3×3 pixels (9 pixels = 1 wavefront worth!)
│   Evaluate 9 edge functions per tile in 1 cycle (9 SIMT lanes)
│   Hierarchical culling: test tile corners first
│     If all 3 edges yield -1 at all 4 corners → entire tile culled
│     If all 3 edges yield +1 at all 4 corners → entire tile inside
│     Otherwise → per-pixel evaluation
├── Depth Test:
│   TCOMP fragment_z, buffer_z → result is 1 trit
│   Three-way: closer(+1), same(0), farther(-1)
│   Pass if result ∈ {+1, 0} (configurable via depth function trit)
└── Output: fragment position + attributes → fragment shader
```

**Natural 3×3 tile = 9 pixels = 1 wavefront.** Binary GPUs use 8×8 tiles (64 pixels = 1 wavefront of 64 for AMD, or 2 warps of 32 for NVIDIA). RISC-T's 3×3 tiles match the wavefront size exactly — no waste, no partial warps. This is the cleanest mapping of rasterization work to SIMT execution in any GPU architecture.

### 25.12 Ternary Shader ISA Extensions

Beyond the base RISC-T ISA, the graphics pipeline adds shader-specific instructions:

| Instruction | Description | Pipeline Stage |
|-------------|-------------|---------------|
| `TVERTEX rd, rs_attr, imm_idx` | Load vertex attribute by index | Vertex shader |
| `TFRAG rd, rs_bary, rs_attr` | Interpolate attribute at barycentric coords | Fragment shader |
| `TEXPORT rs_color, rs_depth` | Export fragment color and depth | Fragment output |
| `TDDX rd, rs` | Partial derivative in screen-X (finite difference across triad) | Fragment shader |
| `TDDY rd, rs` | Partial derivative in screen-Y (finite difference across triad) | Fragment shader |
| `TTEXSAMPLE rd, rs_coord, imm_unit` | Sample TTU (Knowledge Star or traditional texture) | Any shader |
| `TDISCARD imm_condition` | Discard fragment if condition trit matches | Fragment shader |
| `TBLEND rd, rs_src, rs_dst, rs_factor` | Ternary blend: factor = +1(src), 0(lerp), -1(dst) | Output merger |
| `TCLIP rd, rs_pos, rs_plane` | Clip test against plane: inside(+1), on(0), outside(-1) | Vertex output |

### 25.13 Ternary Display Output

```
Display Controller:
├── Frame Buffer Format:
│   Per-pixel: R(6T) + G(6T) + B(6T) + A(3T) = 21 trits
│   Each channel: 6 trits = 729 levels (vs binary 8-bit = 256 levels)
│   2.85× color precision per channel over 8-bit binary
│   Alpha: 3 trits = 27 levels (sufficient for standard blending)
├── Resolution support:
│   Max: 3^7 × 3^7 = 2187 × 2187 (≈2K, suitable for first-gen)
│   With extended addressing: 3^8 × 3^8 = 6561 × 6561 (≈6.5K)
├── Refresh: Via binary HDMI/DP interface (Binary Bridge handles conversion)
├── Native ternary display (future):
│   3-level per subpixel (e.g., e-ink with -V/0/+V per pixel)
│   Perfect for balanced ternary — no DAC needed, just 3 voltage levels
└── HDR support: TFloat20 per channel (20-trit floating-point color)
    Exceeds binary FP16 HDR precision
```

### 25.14 GPU Memory Map

```
Ternary GPU Memory Map (per-TCU view):
┌────────────────────────────────────────────────┐
│  Private (per-thread)                           │
│  ├── Scalar registers: 27 × 40T                │  Address: 0 to +MAX/27
│  └── Vector registers: 27 × 9 × 40T            │
├────────────────────────────────────────────────┤
│  LTS (per-workgroup shared)                     │  Address: +MAX/27 to +2MAX/27
│  └── 6561 trytes, 27 banks                     │
├────────────────────────────────────────────────┤
│  Global (all CUs share)                         │  Address: +2MAX/27 to +MAX
│  ├── Galaxy memory (star database, read-mostly) │
│  ├── Frame buffer (graphics output)             │
│  ├── Uniform buffers (constants)                │
│  └── Storage buffers (read-write)               │
├────────────────────────────────────────────────┤
│  MMIO (GPU control registers)                   │  Top of address space
│  ├── TCU status/control                         │
│  ├── Display controller                         │
│  ├── DMA engine                                 │
│  └── Binary Bridge control                      │
└────────────────────────────────────────────────┘
```

### 25.15 Host Interface & DMA

```
Host-GPU Interface:
├── Command Processor:
│   ├── Command queue: Ring buffer of ternary command packets
│   ├── Packet format: header(6T) + opcode(3T) + payload(variable)
│   ├── Opcodes (3T = 27 possible):
│   │   ├── DISPATCH_COMPUTE(000): launch compute workgroup
│   │   ├── DISPATCH_DRAW(00+): launch graphics draw call
│   │   ├── DMA_COPY(0-0): host↔device memory copy
│   │   ├── BARRIER(0+0): pipeline barrier
│   │   ├── GALAXY_LOAD(+-0): load Galaxy region into star cache
│   │   └── ... (22 more commands available)
│   └── Doorbell: Write to MMIO register triggers command processing
├── DMA Engine:
│   ├── Ternary-native DMA (TCU ↔ main memory, via TLink or Binary Bridge)
│   ├── Scatter-gather: ternary address list (3^N entries per descriptor)
│   ├── Compression: trit packing during transfer (5T → 8 bits → 5T)
│   └── Priority: 3-trit priority field (-1..+1)
└── Interrupt: GPU→host interrupt via TIC (§19)
    ├── Completion interrupts (compute/draw done)
    ├── Page fault interrupts (virtual memory)
    └── Error interrupts (shader exception, memory fault)
```

---

## Appendix F: GPU Prior Art Inventory (Items 41-60)

41. Ternary Compute Unit (TCU) with 3×SIMD3 vector execution and dual-issue scalar/vector decode
42. 27-wavefront pool with single-trit priority scheduling (high/normal/stalled)
43. Ternary round-robin wavefront scheduler with trit-level tier promotion/demotion
44. 27-bank vector register file with ternary rotation for bank conflict resolution
45. Local Trit Share (LTS) scratchpad with 27 banks and ternary crossbar
46. Ternary Texture Unit repurposed as Knowledge Star Sampler with Galaxy lookups
47. 3×3 Ternary Matrix Multiply-Accumulate (TMMA) unit for AI and spatial transforms
48. Ternary rasterizer with native +1/0/-1 edge function classification (no epsilon)
49. 3×3 pixel tiles matching 9-thread wavefront size for zero-waste rasterization
50. Three-way subgroup ballot returning 9-trit mask with per-lane ternary state
51. Ternary subgroup vote with three-way consensus (positive/negative/neutral majority)
52. Triad operations replacing quad operations (3-lane groups instead of 4-lane)
53. Ternary blend instruction with factor trit (+1=source, 0=lerp, -1=destination)
54. Ternary fragment discard based on condition trit
55. Ternary depth test yielding single trit (closer/same/farther)
56. 6-trit-per-channel color format (729 levels, 2.85× precision over 8-bit binary)
57. Ternary display output with 3-voltage-level native addressing
58. GPU command processor with 3-trit opcode encoding (27 command types)
59. Ternary DMA with trit-packing compression during host-device transfer
60. Unified Knowledge Star Sampler / texture pipeline (same hardware for Galaxy and graphics)

**Updated total prior art items: 60 (Appendices C + E + F).**

---

## Appendix E: Updated Prior Art Inventory

Items 21-40 (extending Appendix C):

21. Ternary privilege levels encoded as single trit (-1=Machine, 0=Supervisor, +1=User)
22. Ternary exception cause encoding (negative = synchronous, positive = asynchronous, zero = none)
23. Vectored trap table with 27 entries (3^3 causes, 1 instruction slot each)
24. Ternary floating-point format with three-valued sign trit (positive/zero/negative)
25. Unbiased ternary rounding modes encoded in 1 trit
26. TMESI cache coherence protocol with 9 states (3^2) for ternary multi-core systems
27. Three-way branch prediction using ternary saturating counters
28. Branch Target Buffer with three target addresses per entry
29. Ternary Network-on-Chip with dimension-order routing using single-trit comparison
30. PAM-3 signaling for ternary-native chip-to-chip links (TLink)
31. Ternary CRC (TCRC-12) polynomial for error detection on ternary links
32. Ternary routing protocol (TRP) with 1-trit multicast mode selection
33. Three-way SIMT predication (active/masked/inverted) for single-pass divergent execution
34. Ternary page table entries with three-valued permission trits
35. Ternary trust model (untrusted/measured/trusted) for hardware security
36. Nine-vote swarm consensus hardware with three-way majority detection
37. Ternary power management states encoded as single trit (active/idle/sleep)
38. Per-unit ternary power gating with independent trit-level control
39. 3-level DVFS using three-way relay voltage model
40. Ternary Physically Unclonable Functions (PUFs) with 58.5% more entropy per element

**Updated publication date: 2026-03-19. Revision 0.2 — extended specification with privilege, float, networking, security, cache coherence, power management, and HDL pseudocode.**

---

## Sources & References

### Open ISA
- [RISC-V Specifications — RISC-V International](https://riscv.org/specifications/ratified/)
- [RISC-V ISA Manual — GitHub](https://github.com/riscv/riscv-isa-manual)
- [x86-64 Instruction Encoding — OSDev Wiki](http://wiki.osdev.org/X86-64_Instruction_Encoding)

### Open GPU
- [MIAOW GPU — University of Wisconsin](https://miaowgpu.org/)
- [Vortex RISC-V GPGPU — Georgia Tech](https://vortex.cc.gatech.edu/)
- [Nyuzi GPGPU Processor — GitHub](https://github.com/jbush001/NyuziProcessor)

### Ternary Computing
- [Douglas W. Jones on Ternary Arithmetic](https://homepage.divms.uiowa.edu/~jones/ternary/arith.shtml)
- [Ternary ALU Design](https://louis-dr.github.io/ternalu3.html)
- [Ternary Computing Overview](https://www.ternary-computing.com/)
- [Ternary RISC Processor on FPGA — Hackaday](https://hackaday.com/2026/03/16/ternary-risc-processor-achieves-non-binary-computing-via-fpga/)

### Silicon Fabrication
- [Huawei Ternary Logic Chip](https://meta-quantum.today/?p=7960)
- [Huawei Patent — SCMP](https://www.scmp.com/tech/big-tech/article/3305201/tech-war-huaweis-ternary-logic-patent-could-solve-problem-power-hungry-ai-chips)
- [Carbon Nanotube Ternary Circuits — Science Advances](https://www.science.org/doi/10.1126/sciadv.adt1909)
- [Balanced Ternary CMOS — academia.edu](https://www.academia.edu/28854666)
- [MoS₂/WSe₂ Binary/Ternary Convertible — Wiley](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adfm.202510164)

### Open Fabrication
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
- K3D Hyper-Parallel Processing Specification §6 (docs/vocabulary/HYPER_PARALLEL_PROCESSING.md)
- K3D Ternary Contrastive Learning Specification (docs/vocabulary/TERNARY_CONTRASTIVE_LEARNING_SPECIFICATION.md)
