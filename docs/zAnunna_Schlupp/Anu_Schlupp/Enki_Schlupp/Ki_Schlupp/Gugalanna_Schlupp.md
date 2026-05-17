# AC-Sensor: AC-Native Ternary Sensor Fabric Specification

**Version:** 0.1 DRAFT (Defensive Publication)
**Date:** March 20, 2026
**Authors:** Daniel Campos Ramos (PM-KR Chair), Christoph Dorn, Milton Ponson
**Organization:** PM-KR Community Group
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
- K3D multimodal sovereign stack
- sensor-domain baseline in [Ninurta_Schlupp.md](Ninurta_Schlupp.md), [Ishkur_Schlupp.md](Ishkur_Schlupp.md), and [Nergal_Schlupp.md](Nergal_Schlupp.md)
- AC platform baselines in [Marduk_Schlupp.md](Marduk_Schlupp.md) and [Ereshkigal_Schlupp.md](Ereshkigal_Schlupp.md)

---

## 1. Defensive Publication Notice

This document establishes prior art for **sensor architectures that natively sample, encode, and transport ternary states under AC-synchronized power and timing regimes**. It covers ToF, BioFET, ISFET, multiband imaging, SPAD/dToF, sparse-data transmission, local ternary inference, and AC phase-locked sensor fabrics.

---

## 2. Foundational Ternary Distinction

The sensor primitive is rest-centered:

- `0` = baseline / neutral / no-event / nominal band
- `+1` = positive event class
- `-1` = negative event class or opposite direction

The exact semantic labels differ per sensor, but the three-state physical model remains consistent.

---

## 3. The AC Paradigm

AC-native sensors matter because the zero state is common in real-world sensing:

- most pixels are not active at once
- most chemical receptors are neutral most of the time
- most depth rays are background or uncertain

A sensor fabric synchronized to AC ternary infrastructure can:

- quantize directly to three levels
- preserve sparse neutral states
- reduce downstream conversion overhead
- feed ternary inference blocks without a forced binary detour

---

## 4. Architecture Overview

```text
      AC Sensor Cell -> Local Ternary Quantizer -> Edge Ternary FSM
                |                    |                    |
                +--------------------+--------------------+
                                     |
                           Sparse Ternary Fabric
                                     |
                       AC-BT / AC-T Processing Core
```

---

## 5. Time-of-Flight Depth Sensing

Canonical ternary depth mapping:

- `+1` = near
- `0` = nominal range or unresolved baseline
- `-1` = far / invalid / noisy return

This enables:

- low-cost confidence-aware depth grids
- bandwidth reduction when large regions stay neutral
- early culling for spatial reasoning

---

## 6. BioFET and ISFET Sensing

Chemical and biosensing map cleanly:

- `+1` = analyte present or elevated binding
- `0` = neutral baseline
- `-1` = inhibitory, opposite, or depleted condition

Direct ternary output is useful for edge health, environmental, and lab-on-chip devices.

---

## 7. Multiband Imaging

A ternary sensor can encode band-state relations such as:

- `+1` = visible active
- `0` = dark / neutral
- `-1` = NIR or alternate channel emphasis

Equivalent mappings are also possible for motion, confidence, or change detection.

---

## 8. SPAD and dToF

SPAD and direct ToF systems can use ternary event output:

- `+1` = confident photon event
- `0` = no-event / baseline
- `-1` = noise, backscatter, or invalid tail

This improves front-end filtering before full binary packetization.

---

## 9. Sparse Data Representation

Most sensor fabrics are sparse. AC-Sensor standardizes sparse ternary transport:

- do not transmit neutral blocks at full cost when they dominate
- keep `0` as a first-class low-entropy state
- compress by region, tile, or event stream

This is especially valuable in always-on sensor networks and embodied AI.

---

## 10. AC Synchronization

Sensor sampling can be phase-locked to the AC power clock so that:

- sampling windows align with low-noise phases
- local ternary quantization aligns with compute availability
- energy-recovery behavior and sensing cadence can be co-designed

This is one of the distinguishing contributions of the AC family.

---

## 11. Analog-to-Ternary Conversion

Instead of analog-to-binary-to-ternary conversion, AC-Sensor standardizes direct three-level quantization:

- upper threshold -> `+1`
- middle band -> `0`
- lower threshold -> `-1`

The implementation may be based on:

- multithreshold comparators
- BDT-style current steering
- IGZO or DNTT multistate devices

---

## 12. Local Edge Processing

Each sensor block may include:

- ternary finite-state filters
- event confirmation
- confidence tagging
- lightweight local inference

This reduces transport bandwidth and aligns with K3D's preference for semantic structure close to the source.

---

## 13. Integration with AC-BT and AC-T

AC-Sensor is designed to plug directly into:

- [Marduk_Schlupp.md](Marduk_Schlupp.md) for near-term hybrid systems
- [Ereshkigal_Schlupp.md](Ereshkigal_Schlupp.md) for long-term AC-native platforms

It also connects cleanly to [Ningishzida_Schlupp.md](Ningishzida_Schlupp.md) when BDT-style current steering is used in the front end.

---

## 14. K3D Integration

This spec matters to K3D because the platform is fundamentally multimodal. AC-native ternary sensors preserve:

- uncertainty
- sparse neutral states
- semantic pre-quantization

before the signal ever reaches the GPU or chip fabric.

---

## 15. Manufacturing Path

1. ternary sensor emulation and DAC-based prototypes
2. mixed-signal proof chips with direct three-level readout
3. integration with AC-BT edge compute parts
4. progression toward AC-T native sensor-compute fabrics

---

## 16. Prior Art and Sources

1. `/media/Arquivos/Engenharia/Ternary Chips/Research/Ternary-Binary Hybrid AC Semiconductor Lithography.md`
2. `/media/Arquivos/Engenharia/Ternary Chips/Research/Ternary-Binary Hybrid Semiconductor Lithography.md`
3. `/media/Arquivos/Engenharia/Ternary Chips/Research/Tri-State Transistor Design Feasibility Study.md`
4. AC Computing Methodology for RF Powered IoT Devices — CPS-VO PDF, https://cps-vo.org/sites/cps-vo.org/files/cpsvo_file_nodes/AC_Computing_Methodology_for_RF-Powered_IoT_Devices.pdf
5. Reconfigurable binary and ternary logic devices enabling logic state modulation — Nature Communications, https://www.nature.com/articles/s41467-025-62116-y
6. Demonstration of p-type stack-channel ternary logic device using scalable DNTT patterning process — PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC9998751/
7. Heterogeneous and Monolithic 3D Integration Technology for Mixed-Signal ICs — Electronics, https://doi.org/10.3390/electronics11193013
8. ASML and imec open joint High-NA EUV lithography lab — ASML, https://www.asml.com/en/news/press-releases/2024/asml-imec-opening-high-na-euv-lithography-lab
9. Scalable Resonant Power Clock Generation for Adiabatic Logic Design — NSF/IEEE, https://par.nsf.gov/servlets/purl/10295943
10. Ballistic Deflection Transistors for THz Amplification — DTIC, https://apps.dtic.mil/sti/trecms/pdf/AD1050680.pdf
11. Ballistic transport in graphene suggests new type of electronic device — Georgia Tech, https://physics.gatech.edu/news/ballistic-transport-graphene-suggests-new-type-electronic-device
12. Ternary logic design based on novel tunneling-drift-diffusion field-effect transistors — MDPI, https://www.mdpi.com/2079-4991/15/16/1240
13. Reconfigurable multivalue logic functions of a silicon quantum-dot transistor — ACS Nano, https://pubs.acs.org/doi/10.1021/acsnano.1c08208
14. Advances in 2D-material based device technology — imec, https://www.imec-int.com/en/press/imec-advances-2d-material-based-device-technology-beyond-state-art-support-future-logic
15. Sensor-BT baseline — [Ninurta_Schlupp.md](Ninurta_Schlupp.md)
16. Sensor-xSenseT baseline — [Ishkur_Schlupp.md](Ishkur_Schlupp.md)
17. Sensor-T baseline — [Nergal_Schlupp.md](Nergal_Schlupp.md)
18. AC-BT host platform — [Marduk_Schlupp.md](Marduk_Schlupp.md)
19. AC-xPowerT host power plane — [Nabu_Schlupp.md](Nabu_Schlupp.md)
20. AC-T host platform — [Ereshkigal_Schlupp.md](Ereshkigal_Schlupp.md)
21. AC-BDT front-end option — [Ningishzida_Schlupp.md](Ningishzida_Schlupp.md)
22. AC-Litho fabrication path — [Ninazu_Schlupp.md](Ninazu_Schlupp.md)
23. SoC integration baseline — [Ninhursag_Schlupp.md](Ninhursag_Schlupp.md)

---

## 17. Companion Specs

- [Marduk_Schlupp.md](Marduk_Schlupp.md)
- [Nabu_Schlupp.md](Nabu_Schlupp.md)
- [Ereshkigal_Schlupp.md](Ereshkigal_Schlupp.md)
- [Ningishzida_Schlupp.md](Ningishzida_Schlupp.md)
- [Ninazu_Schlupp.md](Ninazu_Schlupp.md)
- [Ki_Schlupp.md](Ki_Schlupp.md)
