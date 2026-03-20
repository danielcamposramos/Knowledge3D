# AC-Litho: AC Ternary Fabrication and Lithography Specification

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
- K3D ternary semantics and sovereign PTX kernels
- fabrication target for [Marduk_Schlupp.md](Marduk_Schlupp.md), [Nabu_Schlupp.md](Nabu_Schlupp.md), and [Ereshkigal_Schlupp.md](Ereshkigal_Schlupp.md)

---

## 1. Defensive Publication Notice

This document establishes prior art for **fabrication flows and lithography stacks capable of producing AC adiabatic ternary chips and their heterogeneous control structures**. The disclosure covers High-NA EUV, CNT, IGZO, DNTT, 2D-material, spiral-inductor, SAM, and monolithic-3D integration paths.

---

## 2. Foundational Ternary Distinction

The fabrication target remains a rest-centered three-state device family:

- `0` = natural position
- `+1` = side A routed or activated
- `-1` = side B routed or activated

Process choices must preserve that physical distinction instead of collapsing ternary into binary-plus-overhead.

---

## 3. The AC Paradigm

AC ternary fabrication differs from ordinary digital scaling because it must co-optimize:

- multi-valued devices
- resonant passives
- low-loss interconnect
- mixed DC/AC or fully AC timing islands
- heterogeneous materials on one die or in one stack

This means process definition is itself a critical part of the patent-defense surface.

---

## 4. Fabrication Overview

```text
       Lithography Stack for AC Ternary

  Logic devices: CNT / IGZO / DNTT / 2D materials
  Power devices: resonant drivers / switches / damping
  Passives: spiral inductors / capacitors / shields
  Interconnect: low-loss routing / guard structures
  Integration: monolithic 3D or heterogeneous packaging
```

---

## 5. High-NA EUV

High-NA EUV is important for:

- tight resonant routing geometries
- precise multi-gate overlap
- dense heterogeneous integration
- alignment of ternary and binary blocks in hybrid chips

It is not mandatory for proof chips, but it is a strong long-term manufacturing path.

---

## 6. CNT Processing

CNT-based ternary devices require:

- chirality-controlled growth or selection
- diameter control for threshold targeting
- placement or transfer precision
- contact engineering for low-resistance terminals

CNT paths are especially attractive for multithreshold and source-gated ternary elements.

---

## 7. IGZO Dual-Gate Processing

IGZO provides a practical reconfigurable path because dual-gate asymmetry can support:

- binary mode
- ternary mode
- mode-selectable logic blocks

Key fabrication needs:

- oxide deposition uniformity
- threshold-switch integration
- back-gate and front-gate alignment
- thermal budget compatible with stacked integration

---

## 8. Organic DNTT Stack-Channel Processing

DNTT and related organic channels matter because they enable:

- low-temperature deposition
- upper-tier integration over conventional CMOS
- compact ternary storage and logic islands

This makes them attractive for M3D experiments where the base die carries binary support logic and the upper tiers carry ternary AC blocks.

---

## 9. 2D Materials

Promising materials include:

- MoS2
- WSe2
- black phosphorus
- mixed 2D heterostructures

Required patterning methods include:

- e-beam for research
- nanoimprint and transfer methods for scaling
- interface engineering to preserve mobility and threshold stability

---

## 10. Monolithic 3D Integration

M3D is central because it supports:

- vertical separation of DC and AC domains
- upper-tier ternary islands
- sensor tiers close to compute tiers
- shorter interconnect for resonant and mixed-signal paths

The AC family should therefore be treated as a natural consumer of M3D process research.

---

## 11. On-Chip Inductor and Passive Fabrication

AC chips require passives to be first-class citizens, not leftovers. Process requirements include:

- spiral inductor metal thickness
- Q-aware substrate isolation
- MIM capacitors for resonant tuning
- shielding to reduce coupling into logic and sensor blocks

This is a key distinction from ordinary digital-only nodes.

---

## 12. SAMs and Surface Engineering

Self-assembled monolayers and related surface treatments matter for:

- organic transistor alignment
- selective deposition
- interface control in heterogeneous stacks
- low-temperature process compatibility

---

## 13. 300mm Compatibility and Yield

Production viability requires:

- defect budgets compatible with mixed-material stacks
- wafer-level metrology for multistate thresholds
- parametric test structures for resonant blocks
- repair and redundancy strategies for ternary arrays

Yield risk is expected to be higher than ordinary CMOS at first, so the spec includes staged process ramps rather than pretending direct mass production.

---

## 14. Open PDK Targets

Recommended staged targets:

1. SKY130 and GF180 for binary control logic, interfaces, and proof harnesses
2. research add-ons for ternary device islands
3. 22FDX-class mixed-signal or equivalent production path
4. High-NA EUV and advanced heterogeneous flows for dense AC products

---

## 15. Process Flow Families

### 15.1 CNT Flow

Base wafer -> dielectric stack -> CNT growth / transfer -> gate pattern -> contact formation -> passivation -> resonant interconnect

### 15.2 IGZO Flow

Base CMOS or glass-compatible tier -> IGZO deposition -> dual-gate definition -> threshold-switch add-on -> metallization

### 15.3 Organic / DNTT Flow

Base binary tier -> low-temperature dielectric -> organic channel deposition -> stacked gate patterning -> upper-tier interconnect

---

## 16. K3D Integration

K3D needs the fabrication story because sovereignty is only complete when the ternary semantics can leave the GPU substrate and become device-native. Ninazu is the foundry-side map from K3D semantics to actual silicon or post-silicon materials research.

---

## 17. Manufacturing Path

1. emulate AC semantics on FPGA
2. tape out binary support and test harness blocks on open nodes
3. integrate experimental ternary islands using research materials
4. advance to M3D or mixed-signal production nodes
5. use High-NA EUV where density and alignment justify it

---

## 18. Prior Art and Sources

1. `/media/Arquivos/Engenharia/Ternary Chips/Research/Ternary-Binary Hybrid AC Semiconductor Lithography.md`
2. `/media/Arquivos/Engenharia/Ternary Chips/Research/Ternary-Binary Hybrid Semiconductor Lithography.md`
3. `/media/Arquivos/Engenharia/Ternary Chips/Research/Tri-State Transistor Design Feasibility Study.md`
4. ASML and imec open joint High-NA EUV lithography lab — ASML, https://www.asml.com/en/news/press-releases/2024/asml-imec-opening-high-na-euv-lithography-lab
5. Heterogeneous and Monolithic 3D Integration Technology for Mixed-Signal ICs — Electronics, https://doi.org/10.3390/electronics11193013
6. Open source PDKs joining the Linux Foundation's CHIPS Alliance — Google Open Source Blog, https://opensource.googleblog.com/2023/11/open-source-pdks-joining-linux-foundation-chips-alliance.html
7. Google SkyWater PDK repository, https://github.com/google/skywater-pdk
8. open_pdks reference, https://opencircuitdesign.com/open_pdks/reference.html
9. SKY130 open-source PDK note — SkyWater, https://www.skywatertechnology.com/sky130-open-source-pdk/
10. GLOBALFOUNDRIES 22FDX OpenAccess PDK release — Cadence, https://www.cadence.com/en_US/home/company/newsroom/press-releases/pr/2020/globalfoundries-collaborates-with-cadence-on-availability-of-mix.html
11. GlobalFoundries interoperable PDK release — GF, https://gf.com/press-release/globalfoundries-partners-synopsys-mentor-and-keysight-interoperable-process-design
12. Demonstration of p-type stack-channel ternary logic device using scalable DNTT patterning process — PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC9998751/
13. Reconfigurable binary and ternary logic devices enabling logic state modulation — Nature Communications, https://www.nature.com/articles/s41467-025-62116-y
14. Advances in 2D-material based device technology — imec, https://www.imec-int.com/en/press/imec-advances-2d-material-based-device-technology-beyond-state-art-support-future-logic
15. Ternary logic design based on novel tunneling-drift-diffusion field-effect transistors — MDPI, https://www.mdpi.com/2079-4991/15/16/1240
16. Tunneling transistors based on graphene and 2-D crystals — Cornell, https://djena.engineering.cornell.edu/papers-new/2013/jena2013tunneling.pdf
17. Scalable Resonant Power Clock Generation for Adiabatic Logic Design — NSF/IEEE, https://par.nsf.gov/servlets/purl/10295943
18. AC Computing Methodology for RF Powered IoT Devices — CPS-VO PDF, https://cps-vo.org/sites/cps-vo.org/files/cpsvo_file_nodes/AC_Computing_Methodology_for_RF-Powered_IoT_Devices.pdf
19. AC-BT system target — [Marduk_Schlupp.md](Marduk_Schlupp.md)
20. AC-xPowerT system target — [Nabu_Schlupp.md](Nabu_Schlupp.md)
21. AC-T end-state — [Ereshkigal_Schlupp.md](Ereshkigal_Schlupp.md)
22. AC-BDT device family — [Ningishzida_Schlupp.md](Ningishzida_Schlupp.md)
23. SoC integration baseline — [Ninhursag_Schlupp.md](Ninhursag_Schlupp.md)

---

## 19. Companion Specs

- [Marduk_Schlupp.md](Marduk_Schlupp.md)
- [Nabu_Schlupp.md](Nabu_Schlupp.md)
- [Ereshkigal_Schlupp.md](Ereshkigal_Schlupp.md)
- [Ningishzida_Schlupp.md](Ningishzida_Schlupp.md)
- [Gugalanna_Schlupp.md](Gugalanna_Schlupp.md)
- [Ki_Schlupp.md](Ki_Schlupp.md)
