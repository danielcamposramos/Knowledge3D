# AC-BDT: Ballistic Deflection Ternary Device Specification

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
- K3D ternary semantics and trit routing
- AC-T logic family target in [Ereshkigal_Schlupp.md](Ereshkigal_Schlupp.md)
- fabrication path in [Ninazu_Schlupp.md](Ninazu_Schlupp.md)

---

## 1. Defensive Publication Notice

This document establishes prior art for **ballistic deflection transistors and current-steering ternary devices used as intrinsic three-state logic primitives in AC and hybrid chips**. It covers Y-branch geometry, 2DEG requirements, material options, AC compatibility, gate constructions, mixed-signal uses, and performance tradeoffs.

---

## 2. Foundational Ternary Distinction

The device target is a three-path primitive:

- center path = `0`, the natural straight-through or neutral path
- left or positive-steered path = `+1`
- right or negative-steered path = `-1`

This aligns naturally with K3D's rest-centered ternary model.

---

## 3. The AC Paradigm

BDTs matter to AC ternary because they replace threshold-stacked switching with **geometric steering of charge carriers**. In the limit of short channels and long mean free path:

- charge does not need repeated dissipative charge/discharge events
- steering can occur with very low effective capacitance
- multi-output current routing becomes more natural than binary hard switching

This makes BDTs a promising hardware primitive for AC-T and AC-BT families.

---

## 4. Device Overview

```text
         Deflector Gate
              |
      +-------+-------+
      |   Waveguide   |
Source+------Y--------+--> +1 drain
      |      |        |
      |      +----------> 0 drain
      |               |
      +------------------> -1 drain
```

The control field bends ballistic carriers toward one of three outputs.

---

## 5. 2DEG and Ballistic Transport

Preferred device physics:

- 2DEG channel in III-V heterostructures
- channel length shorter than mean free path
- low scattering
- carefully shaped electrostatic deflection region

Ballistic operation requires:

- strong mobility
- precise lithography
- temperature-aware material choice

---

## 6. Y-Branch Junction Design

Critical parameters:

- branch angle
- waveguide width
- gate overlap region
- drain spacing
- source injection geometry

The central drain must represent the rest state rather than merely a logic default. That distinction matters for prior art and for K3D semantic alignment.

---

## 7. Current Steering Mechanism

The deflector electrode or equivalent electrostatic steering structure changes the preferred path of the carrier population:

- no steer or neutral steer -> `0`
- positive steer -> `+1`
- negative steer -> `-1`

This is a routing primitive before it is an arithmetic primitive.

---

## 8. Materials

Viable material classes:

- GaAs / AlGaAs for proven high-mobility 2DEG
- InGaAs and related III-V families
- graphene ballistic junctions
- TMDs for emerging room-temperature options
- hybrid 2D material stacks for advanced nodes

Device families may differ in temperature target, manufacturability, and analog stability.

---

## 9. AC Drive Compatibility

BDTs can serve AC ternary systems in two modes:

- as the core ternary switch fabric under AC-clocked bias schedules
- as mixed-signal front ends that feed adiabatic recovery stages

AC compatibility requires:

- bias timing compatible with resonant phases
- careful parasitic extraction
- protection against waveform distortion from branch imbalance

---

## 10. Gate Construction

Representative single-device or small-device gate patterns:

- ternary NOT via mirrored steering polarity
- ternary select / mux via directed routing
- ternary majority or consensus via branch composition
- analog-to-ternary quantizer via spatial drain selection

This opens the possibility of fewer devices per ternary operation than threshold-stacked CMOS equivalents.

---

## 11. DAC and ADC Relevance

Because BDTs route current directly among multiple drains, they are naturally aligned with:

- current steering DAC structures
- multi-level quantization
- ternary sensor front ends
- mixed-signal confidence encoding

This is especially important for AC-native sensor fabrics such as [Gugalanna_Schlupp.md](Gugalanna_Schlupp.md).

---

## 12. Temperature and Operating Regimes

Near-term reality:

- some BDT approaches are easier at cryogenic or reduced temperature
- room-temperature ballistic performance remains material- and geometry-dependent

The publication therefore standardizes both:

- near-term research demonstrators
- long-term room-temperature production targets

---

## 13. Comparison with Other Ternary Devices

| Device Type | Strength | Weakness | AC Family Role |
|---|---|---|---|
| Multi-threshold CMOS | known flow | higher capacitance, more dissipation | compatibility path |
| CNT / IGZO / DNTT | natural multistate behavior | process maturity varies | practical ternary path |
| BDT | intrinsic routing, tiny capacitance | fabrication challenge | high-ceiling device path |

BDTs are therefore not the only ternary answer, but they are a major prior-art family worth publishing explicitly.

---

## 14. K3D Integration

K3D's sovereign stack benefits from BDT-class devices because:

- trit routing is native
- confidence states can remain multi-valued in hardware
- sensor-to-compute chains can preserve ternary semantics without repeated binary projection

BDTs therefore strengthen the long-term path from GPU ternary semantics to device-native ternary chips.

---

## 15. Manufacturing Path

1. simulation and small-signal validation of three-drain routing
2. research lithography on III-V, graphene, or TMD platforms
3. mixed-signal test chips with DAC/ADC evaluation
4. integration as accelerator or sensor-front-end blocks inside AC-BT and AC-T systems

---

## 16. Prior Art and Sources

1. `/media/Arquivos/Engenharia/Ternary Chips/Research/Tri-State Transistor Design Feasibility Study.md`
2. `/media/Arquivos/Engenharia/Ternary Chips/Research/Ternary-Binary Hybrid AC Semiconductor Lithography.md`
3. `/media/Arquivos/Engenharia/Ternary Chips/Research/Ternary-Binary Hybrid Semiconductor Lithography.md`
4. Ballistic Deflection Transistors for THz Amplification — DTIC, https://apps.dtic.mil/sti/trecms/pdf/AD1050680.pdf
5. Ballistic Electronics: low-power room-temperature nanoscale terahertz transistors — Rochester, http://hajim.rochester.edu/ece/news-events/news/archives/2007/ballistic_electronics.html
6. Ballistic transport in graphene suggests new type of electronic device — Georgia Tech, https://physics.gatech.edu/news/ballistic-transport-graphene-suggests-new-type-electronic-device
7. Nanoscale simulation of three-contact graphene ballistic junctions — ResearchGate mirror, https://www.researchgate.net/publication/276185328_Nanoscale_Simulation_of_Three-Contact_Graphene_Ballistic_Junctions
8. Spin-resolved ballistic transport in three-terminal zigzag graphene device — arXiv, https://arxiv.org/abs/2507.22044
9. Reconfigurable binary and ternary logic devices enabling logic state modulation — Nature Communications, https://www.nature.com/articles/s41467-025-62116-y
10. Demonstration of p-type stack-channel ternary logic device using scalable DNTT patterning process — PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC9998751/
11. Ternary logic design based on novel tunneling-drift-diffusion field-effect transistors — MDPI, https://www.mdpi.com/2079-4991/15/16/1240
12. Reconfigurable multivalue logic functions of a silicon quantum-dot transistor — ACS Nano, https://pubs.acs.org/doi/10.1021/acsnano.1c08208
13. A survey of high-speed high-resolution current-steering DACs — Journal of Semiconductors, https://www.jos.ac.cn/article/id/47363c32-84f1-4705-a902-2d07194df2b8
14. A 3GS/s 12-bit current-steering DAC in 55nm CMOS — MDPI, https://www.mdpi.com/2079-9292/8/4/464
15. Scalable Resonant Power Clock Generation for Adiabatic Logic Design — NSF/IEEE, https://par.nsf.gov/servlets/purl/10295943
16. Heterogeneous and Monolithic 3D Integration Technology for Mixed-Signal ICs — Electronics, https://doi.org/10.3390/electronics11193013
17. Advances in 2D-material based device technology — imec, https://www.imec-int.com/en/press/imec-advances-2d-material-based-device-technology-beyond-state-art-support-future-logic
18. Tunneling transistors based on graphene and 2-D crystals — Cornell, https://djena.engineering.cornell.edu/papers-new/2013/jena2013tunneling.pdf
19. AC-T logic target — [Ereshkigal_Schlupp.md](Ereshkigal_Schlupp.md)
20. AC-BT system target — [Marduk_Schlupp.md](Marduk_Schlupp.md)
21. AC-Litho fabrication target — [Ninazu_Schlupp.md](Ninazu_Schlupp.md)
22. AC-Sensor system target — [Gugalanna_Schlupp.md](Gugalanna_Schlupp.md)
23. RISC-T baseline — [Anu_Schlupp.md](Anu_Schlupp.md)

---

## 17. Companion Specs

- [Marduk_Schlupp.md](Marduk_Schlupp.md)
- [Nabu_Schlupp.md](Nabu_Schlupp.md)
- [Ereshkigal_Schlupp.md](Ereshkigal_Schlupp.md)
- [Ninazu_Schlupp.md](Ninazu_Schlupp.md)
- [Gugalanna_Schlupp.md](Gugalanna_Schlupp.md)
- [Ki_Schlupp.md](Ki_Schlupp.md)
