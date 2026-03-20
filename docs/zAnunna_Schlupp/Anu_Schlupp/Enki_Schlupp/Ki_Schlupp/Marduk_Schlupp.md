# AC-BT: Hybrid DC Binary + AC Adiabatic Ternary Specification

**Version:** 0.1 DRAFT (Defensive Publication)
**Date:** March 20, 2026
**Authors:** Daniel Campos Ramos (PM-KR Chair), Christoph Dorn, Milton Ponson
**Organization:** W3C PM-KR Community Group
**License:** W3C Royalty-Free (Defensive Publication under W3C Patent Policy Section 3)

**Companion Specifications:**
- [Marduk_Schlupp.md](Marduk_Schlupp.md)
- [Nabu_Schlupp.md](Nabu_Schlupp.md)
- [Ereshkigal_Schlupp.md](Ereshkigal_Schlupp.md)
- [Ningishzida_Schlupp.md](Ningishzida_Schlupp.md)
- [Ninazu_Schlupp.md](Ninazu_Schlupp.md)
- [Gugalanna_Schlupp.md](Gugalanna_Schlupp.md)
- [Ki_Schlupp.md](Ki_Schlupp.md)

**Reference Implementations:**
- K3D ternary GPU pipeline (sovereign PTX kernels)
- RISC-T baseline in [Anu_Schlupp.md](Anu_Schlupp.md)
- RISC-BT baseline in [Enki_Schlupp.md](Enki_Schlupp.md)
- AC-T end-state in [Ereshkigal_Schlupp.md](Ereshkigal_Schlupp.md)

---

## 1. Defensive Publication Notice

This document establishes prior art for **hybrid chips that combine conventional DC binary compute with AC adiabatic ternary accelerator regions on the same die or package**. It covers the compartmentalized power network, binary-to-ternary and ternary-to-binary interface logic, thermal partitioning, shared memory behavior, and workload routing required to implement such systems. The disclosure is intended to block later patent monopolies on AC-assisted ternary accelerators as a category.

---

## 2. Foundational Ternary Distinction

K3D's ternary primitive is rest-centered:

- `0` = natural rest position
- `+1` = side A activated
- `-1` = side B activated

Optional alias notation `0/1/2` is permitted for teaching, software serialization, and DAC table generation, but the physical model remains rest-centered. This specification does **not** rely on increment/decrement gate families as its foundation.

---

## 3. The AC Paradigm

AC-BT exists because some operations stay cheaper on binary hardware while others become materially better when AC adiabatic energy recovery and ternary branching are available together.

Binary DC remains strong for:

- bit manipulation
- protocol framing
- crypto and checksum logic
- legacy software and OS services

AC ternary becomes attractive for:

- multiply-accumulate heavy AI kernels
- three-way confidence tests
- spatial reasoning
- ternary uncertainty propagation
- low-heat edge inference

The AC side uses resonant or quasi-resonant power clocks rather than abrupt rail switching. The result is lower dissipative loss in the accelerator while binary control stays simple and compatible.

---

## 4. Architecture Overview

```text
             AC-BT Hybrid Compute Die

   +------------------+      +--------------------------+
   | DC Binary Region |      | AC Ternary Region        |
   | CPU / OS / I/O   |<---->| Accelerator / TMU / VPU |
   | VDD / VSS rails  | ILB  | Resonant power clocks    |
   +--------+---------+      +-------------+------------+
            |                                  |
            +---------------+  +---------------+
                            |  |
                    +-------v--v--------+
                    | Shared Cache +    |
                    | Memory Controller |
                    +-------+-----------+
                            |
                      DRAM / HBM / CXL
```

Core blocks:

- DC binary cores for operating system, I/O, crypto, and compatibility paths
- AC ternary accelerator islands for AI, ranking, routing, and semantic math
- Interface Logic Blocks (ILBs) for `B2T` and `T2B`
- shared memory subsystem with domain-specific cache agents
- thermal manager that exploits lower AC heat density

---

## 5. Power and Interconnect Topology

### 5.1 DC Domain

The DC domain uses conventional rails, PLL clocking, and binary standard cells. It is responsible for:

- firmware boot
- device management
- storage and network protocol endpoints
- external DDR, LPDDR, HBM, and PCIe control

### 5.2 AC Domain

The AC domain is supplied by resonant drivers that feed multiphase power clocks into ternary accelerator fabrics. Preferred waveforms are quasi-trapezoidal or sinusoid-assisted trapezoidal because they support gradual charge and recovery.

### 5.3 PDN Partitioning

AC-BT standardizes a compartmentalized PDN:

- DC mesh for general compute
- AC clock plane for adiabatic islands
- guard bands and shielding between the domains
- optional monolithic-3D stacking so AC islands sit above DC control

---

## 6. Interface Logic Blocks

ILBs bridge the two domains without pretending that the representations are identical.

### 6.1 B2T

Binary-to-ternary conversion is used when the DC side dispatches work into the accelerator. Examples:

- integer or tensor data packed into ternary lanes
- binary confidence flags expanded into `-1/0/+1`
- binary branch states lifted to ternary branch predicates

### 6.2 T2B

Ternary-to-binary conversion is used on the return path:

- final classification results
- reduced scores
- storage serialization
- protocol-safe export

### 6.3 Isolation Rules

ILBs must isolate:

- clock domain crossings
- voltage and waveform differences
- error detection and representation mismatch

---

## 7. Shared Memory Interface

AC-BT does not require ternary external DRAM on day one. It instead standardizes:

- binary external memory with ternary packing support
- shared LLC or scratchpad visible to both domains
- cache agents aware of ternary block formats
- DMA engines that can hand data directly to AC accelerator queues

A minimal deployment can therefore use standard DDR or HBM while keeping ternary semantics internal to the AC accelerator.

---

## 8. Thermal and Cooling Model

Because adiabatic recovery reduces dissipation, AC blocks can run cooler per useful operation than comparable DC logic. AC-BT therefore permits asymmetric cooling:

- high-throughput heatsinking over binary cores
- thinner heat spreaders over AC islands
- package zoning for hotspots
- scheduler awareness of thermal headroom

This enables chips where the cool domain performs the large inference pass while the hot domain handles legacy orchestration.

---

## 9. Use Case: ML Inference Accelerator

Canonical AC-BT deployment:

- binary CPU: tokenizer, drivers, storage, networking, orchestration
- AC ternary accelerator: MAC arrays, ranking kernels, uncertainty heads, semantic routing
- ILB boundary: tensors, control tokens, confidence states

This maps directly onto K3D:

- House I/O and OS surface stay binary
- Galaxy math, uncertainty, routing, and semantic gravity migrate into the AC ternary side

---

## 10. Cost Analysis

| Operation Class | DC Binary | AC Ternary | Preferred Domain |
|---|---|---|---|
| Bitwise mask / shift | Excellent | Poor | DC binary |
| CRC / AES / SHA | Excellent | Weak | DC binary |
| Three-way compare | Emulated | Native | AC ternary |
| Confidence trit propagation | Emulated | Native | AC ternary |
| Spatial branch / cull | Good | Better | AC ternary |
| Tensor MAC with energy recovery | Good | Better when resonant | AC ternary |
| External protocol framing | Native | Requires bridge | DC binary |
| Mixed inference + I/O workload | Fragmented | Balanced via ILB | AC-BT |

AC-BT is therefore a practical migration architecture, not just an academic midpoint.

---

## 11. Migration from DC Hybrid to AC Hybrid

Starting point:

- [Enki_Schlupp.md](Enki_Schlupp.md) defines hybrid binary + ternary without AC power specialization

AC-BT migration steps:

1. replace selected ternary accelerators with resonant power-clocked islands
2. preserve binary host cores and external memory controllers
3. introduce ILBs and AC-aware caches
4. expand AC share as toolchains and device physics mature

This permits a marketable product long before full AC-T purity is practical.

---

## 12. K3D Integration

AC-BT is the most realistic near-term hardware target for K3D:

- binary side runs the shell, viewers, bridges, and device control
- AC ternary side runs galaxy scoring, semantic routing, trit confidence, and selected math kernels
- sleep-time batch reinforcement becomes an ideal AC domain workload because it is repetitive and energy-sensitive

The result preserves sovereignty direction while remaining commercially manufacturable.

---

## 13. Manufacturing Path

Recommended path:

1. FPGA emulation of ILBs and workload routing
2. open-PDK proof parts for control plane and measurement harnesses
3. research chips with AC ternary islands over or beside binary logic
4. 22FDX-class or equivalent mixed-signal path for production
5. High-NA EUV and M3D for dense AC/DC partitioning when mature

---

## 14. Prior Art and Sources

1. `/media/Arquivos/Engenharia/Ternary Chips/Research/Ternary-Binary Hybrid AC Semiconductor Lithography.md`
2. `/media/Arquivos/Engenharia/Ternary Chips/Research/Ternary-Binary Hybrid Semiconductor Lithography.md`
3. `/media/Arquivos/Engenharia/Ternary Chips/Research/Tri-State Transistor Design Feasibility Study.md`
4. Scalable Resonant Power Clock Generation for Adiabatic Logic Design — NSF/IEEE, https://par.nsf.gov/servlets/purl/10295943
5. AC Computing Methodology for RF Powered IoT Devices — CPS-VO PDF, https://cps-vo.org/sites/cps-vo.org/files/cpsvo_file_nodes/AC_Computing_Methodology_for_RF-Powered_IoT_Devices.pdf
6. Reconfigurable binary and ternary logic devices enabling logic state modulation — Nature Communications, https://www.nature.com/articles/s41467-025-62116-y
7. Demonstration of p-type stack-channel ternary logic device using scalable DNTT patterning process — PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC9998751/
8. Heterogeneous and Monolithic 3D Integration Technology for Mixed-Signal ICs — Electronics, https://doi.org/10.3390/electronics11193013
9. ASML and imec open joint High-NA EUV lithography lab — ASML, https://www.asml.com/en/news/press-releases/2024/asml-imec-opening-high-na-euv-lithography-lab
10. Open source PDKs joining the Linux Foundation's CHIPS Alliance — Google Open Source Blog, https://opensource.googleblog.com/2023/11/open-source-pdks-joining-linux-foundation-chips-alliance.html
11. Google SkyWater PDK repository, https://github.com/google/skywater-pdk
12. open_pdks reference, https://opencircuitdesign.com/open_pdks/reference.html
13. SKY130 open-source PDK note — SkyWater, https://www.skywatertechnology.com/sky130-open-source-pdk/
14. GLOBALFOUNDRIES 22FDX OpenAccess PDK release — Cadence, https://www.cadence.com/en_US/home/company/newsroom/press-releases/pr/2020/globalfoundries-collaborates-with-cadence-on-availability-of-mix.html
15. GlobalFoundries interoperable PDK release — GF, https://gf.com/press-release/globalfoundries-partners-synopsys-mentor-and-keysight-interoperable-process-design
16. US11671054B2 — Oscillator for adiabatic computational circuitry, https://patents.google.com/patent/US11671054B2/en
17. US10868534B2 — Adiabatic logic-in-memory architecture, https://patents.google.com/patent/US10868534
18. WO2016199157A1 — Ternary arithmetic and logic unit and ternary logic circuits, https://patents.google.com/patent/WO2016199157A1/en
19. EP2549654A1 — Low-consumption logic circuit / energy-recovery logic, https://patents.google.com/patent/EP2549654A1/en
20. RISC-T baseline — [Anu_Schlupp.md](Anu_Schlupp.md)
21. RISC-BT hybrid baseline — [Enki_Schlupp.md](Enki_Schlupp.md)
22. x86/x64 hybrid baseline — [Ninlil_Schlupp.md](Ninlil_Schlupp.md)
23. AC-T end-state — [Ereshkigal_Schlupp.md](Ereshkigal_Schlupp.md)

---

## 15. Companion Specs

- [Nabu_Schlupp.md](Nabu_Schlupp.md)
- [Ereshkigal_Schlupp.md](Ereshkigal_Schlupp.md)
- [Ningishzida_Schlupp.md](Ningishzida_Schlupp.md)
- [Ninazu_Schlupp.md](Ninazu_Schlupp.md)
- [Gugalanna_Schlupp.md](Gugalanna_Schlupp.md)
- [Ki_Schlupp.md](Ki_Schlupp.md)
