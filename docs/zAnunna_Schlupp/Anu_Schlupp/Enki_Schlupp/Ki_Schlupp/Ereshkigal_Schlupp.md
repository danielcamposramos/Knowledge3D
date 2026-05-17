# AC-T: Pure AC Adiabatic Ternary Computing Specification

**Version:** 0.1 DRAFT — Defensive Publication (Pure AC Ternary)
**Date:** 2026-03-20
**Authors:** Daniel Campos Ramos (PM-KR Chair), Christoph Dorn (PM-KR Contributor), Milton Ponson (PM-KR Co-Chair)
**Organization:** PM-KR Community Group
**License:** W3C Royalty-Free — This document is published as prior art under the W3C Patent Policy. All architectures described herein enter the public domain upon publication. No party may patent any design disclosed in this specification.
**Reference Implementations:** K3D Knowledgeverse sovereign ternary pipeline; RISC-T baseline in [Anu_Schlupp.md](Anu_Schlupp.md)
**Companion Specifications:** [Marduk_Schlupp.md](Marduk_Schlupp.md), [Nabu_Schlupp.md](Nabu_Schlupp.md), [Ningishzida_Schlupp.md](Ningishzida_Schlupp.md), [Ninazu_Schlupp.md](Ninazu_Schlupp.md), [Gugalanna_Schlupp.md](Gugalanna_Schlupp.md), [Ki_Schlupp.md](Ki_Schlupp.md)

---

## NOTICE OF DEFENSIVE PUBLICATION

This specification is an enabling defensive publication for **pure AC-driven adiabatic ternary chips**. It describes the power delivery, logic style, timing model, memory behavior, interface boundaries, and manufacturing path sufficiently to establish prior art against later patent claims on AC-native balanced ternary processors, power-clock trees, energy-recovering ternary gates, and AC-synchronized ternary sensor/compute fabrics.

The intent is explicit: AC ternary computing must remain open, royalty-free, and standards-governed rather than captured by a single chip vendor, nation-state, or patent pool.

---

## 1. Foundational Ternary Distinction

The primitive is a **three-state rest-centered element**, not an increment/decrement gate family:

- `0` = natural rest state
- `+1` = side A activated
- `-1` = side B activated

Optional alias notation `0/1/2` is allowed for tooling, but the physical model remains rest-centered. Arithmetic is derived from the primitive; it is not the primitive.

---

## 2. The AC Paradigm

AC-T replaces fixed DC rails with **multiphase resonant power clocks** that both power and time the logic. The objective is not just lower voltage. The objective is:

- slow charge and slow recovery instead of abrupt dump-to-ground switching
- reuse of charge through LC resonance
- tighter alignment between balanced ternary states and the phase geometry of the waveform

The governing resonance relation remains:

`f_res = 1 / (2π√(LC))`

For AC-T, each logic island is synchronized to a four-phase or six-phase quasi-trapezoidal power clock. Those phases define:

- evaluate
- hold
- recover
- wait

Balanced ternary maps naturally onto the waveform:

- `+1` = positive excursion / positive routed phase
- `0` = zero-crossing or neutral hold region
- `-1` = negative excursion / negative routed phase

In AC-T, the zero state is not merely a reduced voltage. It is the **energetically natural state**.

---

## 3. Architecture Overview

AC-T is the end-state architecture where all internal computation is adiabatic and ternary-native.

```text
        AC Resonant Source
                |
      +---------+---------+
      | Phase Manager     |
      | phi0 phi1 phi2... |
      +----+---------+----+
           |         |
     +-----+--+  +---+------+
     | Ternary |  | Ternary |
     | Logic   |  | Memory  |
     | Islands |  | Islands |
     +-----+---+  +---+-----+
           |          |
      +----+----------+----+
      | AC Interconnect /  |
      | Wave Pipelining    |
      +----+----------+----+
           |          |
     +-----+--+  +----+-----+
     | AC I/O  |  | AC Sensor|
     | Edge    |  | Fabric   |
     +---------+  +----------+
```

Core blocks:

- resonant power-clock generator
- phase-aligned ternary logic islands
- AC-native ternary register and memory blocks
- pipeline stages synchronized to wavefront phase
- chip-boundary AC/DC conversion blocks for external ecosystems

---

## 4. Adiabatic Ternary Logic Style

AC-T assumes ternary gate families built around energy recovery, not static CMOS rails.

Representative gates:

- ternary sum / merge gate
- ternary consensus gate
- ternary inverter with rest-preserving neutral pass-through
- ternary mux with `+1 / 0 / -1` routing

The preferred implementation style is cross-coupled adiabatic logic or equivalent energy-recovering families where the output node is:

1. charged gradually during evaluate
2. held while the next stage samples
3. returned to the resonant source during recover

The architecture is compatible with ECRL-style and quasi-static adiabatic families, but AC-T standardizes the behavior, not a single transistor family.

---

## 5. Register, Memory, and Pipeline Model

### 5.1 Register File

AC-T uses ternary registers refreshed by phased AC rails. Viable cells include:

- cross-coupled ternary SRAM variants
- stack-channel ternary storage
- dual-gate IGZO or CNT-derived multistate cells

The register file is phase-owned rather than permanently rail-held. A register bank is valid in one phase window, sampled in the next, and recovered afterward.

### 5.2 Pipeline

AC-T pipelines computation as a **wave machine**:

- Stage A evaluates on `phi0`
- Stage B evaluates on `phi1`
- Stage C evaluates on `phi2`
- recovery is staggered behind evaluation

This means clocking and energy delivery are the same network. Pipeline hazards are therefore power-distribution hazards as much as logic hazards.

### 5.3 Memory Boundary

External DRAM, flash, and HBM remain binary in the near term. AC-T therefore permits chip-edge or package-edge interface blocks that:

- convert AC ternary addresses/data to binary bursts
- isolate resonant domains from conventional I/O noise
- preserve AC purity inside the main compute die

---

## 6. ALU and Arithmetic

AC-T adopts balanced ternary arithmetic because it is symmetry-friendly under AC drive.

Important consequences:

- sign is intrinsic
- negation is wire permutation, not subtractive overhead
- comparison is naturally three-valued
- three-way branching is native

The AC-T ALU prioritizes:

- ternary add / subtract
- ternary compare
- carry-minimized multiply
- ternary MAC and reduction units for AI inference

The K3D-facing subset maps directly to:

- `TADD`
- `TMUL`
- `TNOT`
- `TCOMP`
- `TQUANT`
- `TPACK`
- `TUNPACK`

---

## 7. K3D Integration

K3D already supplies the software and GPU-side ternary semantics. AC-T is the hardware destination for that semantic model.

Mapping:

- K3D ternary opcodes become native AC-T instructions
- TRM swarm routing can use ternary compare and ternary branch without binary decomposition
- semantic gravity, frustum reasoning, and uncertainty scoring map cleanly to `-1/0/+1`
- sleep-time consolidation is a strong fit for energy-recovery compute because it is latency-tolerant and repeat-heavy

AC-T is therefore not a generic curiosity. It is a direct hardware target for the sovereign K3D path.

---

## 8. Manufacturing Path

AC-T does not require an immediate leap to bleeding-edge foundries for proof-of-concept.

### 8.1 Proof Stage

- FPGA or mixed-signal board emulation for phase/pipeline behavior
- SKY130 / GF180 open-PDK support blocks for AC/DC interfaces, controllers, and verification harnesses

### 8.2 Transitional Silicon

- hybrid integration of ternary device islands with binary periphery
- organic or oxide multistate devices above conventional CMOS
- monolithic 3D integration where logic and power-clock structures are vertically separated

### 8.3 Advanced Path

- High-NA EUV for dense resonant routing and heterogeneous stacks
- CNT, IGZO, DNTT, or 2D-material ternary devices
- package-level resonant passives or on-chip inductors with adequate Q

---

## 9. Open Prototyping Targets

Recommended path:

1. emulate AC timing and ternary ISA behavior on FPGA
2. tape out binary support logic plus interface bridges on SKY130/GF180
3. integrate experimental ternary islands through M3D or chiplet-style research vehicles
4. advance toward 22FDX-class or equivalent mixed-signal production paths

This staged path matters because it defeats the argument that AC ternary is only theoretical.

---

## 10. Prior Art and Sources

1. `/media/Arquivos/Engenharia/Ternary Chips/Research/Ternary-Binary Hybrid AC Semiconductor Lithography.md`
2. `/media/Arquivos/Engenharia/Ternary Chips/Research/Ternary-Binary Hybrid Semiconductor Lithography.md`
3. `/media/Arquivos/Engenharia/Ternary Chips/Research/Tri-State Transistor Design Feasibility Study.md`
4. Scalable Resonant Power Clock Generation for Adiabatic Logic Design — NSF/IEEE, https://par.nsf.gov/servlets/purl/10295943
5. AC Computing Methodology for RF Powered IoT Devices — CPS-VO PDF, https://cps-vo.org/sites/cps-vo.org/files/cpsvo_file_nodes/AC_Computing_Methodology_for_RF-Powered_IoT_Devices.pdf
6. Reconfigurable binary and ternary logic devices enabling logic state modulation — Nature Communications, https://www.nature.com/articles/s41467-025-62116-y
7. Demonstration of p-type stack-channel ternary logic device using scalable DNTT patterning process — PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC9998751/
8. Heterogeneous and Monolithic 3D Integration Technology for Mixed-Signal ICs — Electronics, https://doi.org/10.3390/electronics11193013
9. ASML and imec open joint High NA EUV Lithography Lab — ASML, https://www.asml.com/en/news/press-releases/2024/asml-imec-opening-high-na-euv-lithography-lab
10. Open source PDKs joining the Linux Foundation’s CHIPS Alliance — Google Open Source Blog, https://opensource.googleblog.com/2023/11/open-source-pdks-joining-linux-foundation-chips-alliance.html
11. Google SkyWater PDK repository, https://github.com/google/skywater-pdk
12. Open_PDKs reference, https://opencircuitdesign.com/open_pdks/reference.html
13. SKY130 open-source PDK note — SkyWater, https://www.skywatertechnology.com/sky130-open-source-pdk/
14. GLOBALFOUNDRIES 22FDX OpenAccess PDK release — Cadence, https://www.cadence.com/en_US/home/company/newsroom/press-releases/pr/2020/globalfoundries-collaborates-with-cadence-on-availability-of-mix.html
15. GlobalFoundries 22FDX interoperable PDK release — GF, https://gf.com/press-release/globalfoundries-partners-synopsys-mentor-and-keysight-interoperable-process-design
16. US11671054B2 — Oscillator for adiabatic computational circuitry, https://patents.google.com/patent/US11671054B2/en
17. US10868534B2 — Adiabatic logic-in-memory architecture, https://patents.google.com/patent/US10868534
18. WO2016199157A1 — Ternary arithmetic and logic unit and ternary logic circuits, https://patents.google.com/patent/WO2016199157A1/en
19. EP2549654A1 — Low-consumption logic circuit / energy recovery logic, https://patents.google.com/patent/EP2549654A1/en
20. RISC-T baseline spec — [Anu_Schlupp.md](Anu_Schlupp.md)
21. RISC-BT hybrid spec — [Enki_Schlupp.md](Enki_Schlupp.md)
22. x86/x64 hybrid core spec — [Ninlil_Schlupp.md](Ninlil_Schlupp.md)

---

## 11. Companion Specs

- [Marduk_Schlupp.md](Marduk_Schlupp.md) — AC-BT practical hybrid chip
- [Nabu_Schlupp.md](Nabu_Schlupp.md) — AC/DC true hybrid power plane
- [Ningishzida_Schlupp.md](Ningishzida_Schlupp.md) — ballistic ternary device layer
- [Ninazu_Schlupp.md](Ninazu_Schlupp.md) — fabrication and lithography
- [Gugalanna_Schlupp.md](Gugalanna_Schlupp.md) — AC-native sensor fabric
- [Ki_Schlupp.md](Ki_Schlupp.md) — family strategy hub
