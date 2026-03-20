# AC-xPowerT: True Hybrid AC/DC Power Plane Specification

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
- K3D ternary GPU pipeline
- RISC-BT power-domain baseline in [Enki_Schlupp.md](Enki_Schlupp.md)
- AC-BT deployment baseline in [Marduk_Schlupp.md](Marduk_Schlupp.md)
- AC-T end-state in [Ereshkigal_Schlupp.md](Ereshkigal_Schlupp.md)

---

## 1. Defensive Publication Notice

This document establishes prior art for **chips where AC resonant power planes and conventional DC rails coexist as first-class, dynamically managed power domains**. The disclosure covers dual-plane topology, dynamic mode switching, resonant driver behavior, AC/DC synchronization, gating, damping, metering, and EDA requirements.

---

## 2. Foundational Ternary Distinction

The ternary states remain:

- `0` = natural rest
- `+1` = side A active
- `-1` = side B active

The power-plane architecture does not redefine the ternary primitive. It only changes how energy is supplied and recovered.

---

## 3. The AC Paradigm

AC-xPowerT does not treat AC logic as a bolt-on accelerator. It treats **power delivery itself as hybrid infrastructure**. The chip can therefore:

- keep binary or latency-sensitive blocks on DC
- move energy-sensitive or ternary-native workloads to AC
- switch selected blocks between modes where the device family permits it

This is a power-architecture document before it is a logic document.

---

## 4. Architecture Overview

```text
            AC-xPowerT Power Architecture

       +-------------------------------+
       | Power Supervisor + Telemetry  |
       +--------+-------------+--------+
                |             |
         +------v----+   +----v------+
         | DC Plane  |   | AC Plane   |
         | VDD / VSS |   | phi0..phi3 |
         +------+----+   +----+-------+
                |             |
      +---------+--+     +----+---------+
      | DC Blocks  |     | AC Blocks     |
      | CPU, I/O   |     | Ternary ALU   |
      +------------+     +---------------+
                \          /
                 \        /
             +----v------v----+
             | Mode Switch +   |
             | Clock Boundary  |
             +-----------------+
```

---

## 5. Dual Power Plane Topology

### 5.1 DC Plane

The DC plane provides:

- standard always-on logic
- retention state
- external interface compatibility
- emergency fallback operation

### 5.2 AC Plane

The AC plane provides:

- energy-recovering execution
- multiphase waveform distribution
- wave-synchronous ternary evaluation
- reduced switching heat for repetitive workloads

### 5.3 Physical Separation

Recommended implementations:

- lateral zoning on the die
- vertical M3D separation
- dedicated return paths and shielding
- resonance-aware floorplanning

---

## 6. Dynamic Mode Switching

Where device families permit, a logic island may switch between DC static mode and AC adiabatic mode.

Required sequence:

1. quiesce outstanding traffic
2. checkpoint or flush transient state
3. disconnect previous supply topology
4. ramp target domain
5. validate waveform stability or rail stability
6. resume execution

This supports:

- power-sensitive mobile modes
- thermal emergency response
- characterization and manufacturing bring-up

---

## 7. Resonant Driver Design

AC-xPowerT assumes on-chip or in-package LC resonant drivers with:

- selectable frequency bands
- quality-factor measurement
- phase alignment
- damping control for shutdown and faults

The core resonance relation remains:

`f = 1 / (2π√(LC))`

Driver design must balance:

- inductor area
- Q factor
- frequency stability
- coupling noise into mixed-signal regions

---

## 8. Multiphase AC Clocking

The preferred baseline is a four-phase quasi-trapezoidal clock:

- `phi0` evaluate
- `phi1` hold
- `phi2` recover
- `phi3` neutral / handoff

Higher phase counts are allowed for pipeline smoothing, but the key requirement is overlap-free switching and predictable recovery windows.

---

## 9. Energy Recovery Accounting

AC-xPowerT standardizes telemetry for:

- delivered AC energy
- recovered AC energy
- unrecovered loss
- DC rail draw
- per-domain energy per useful operation

Representative accounting table:

| Workload | DC Mode | AC Mode | Expected Better Mode |
|---|---|---|---|
| SPI / Ethernet framing | low complexity | unnecessary overhead | DC |
| Dense matrix tile | high switching loss | recoverable and repetitive | AC |
| Branch-heavy semantic routing | emulated ternary | native ternary + recovery | AC |
| OS scheduler tick | simple | not worth resonance setup | DC |
| Sleep-time batch reinforcement | high repetition | strong AC fit | AC |

---

## 10. Gating and Damping

DC-xPowerT combines:

- conventional power gating for DC regions
- resonance damping and bleed-off for AC regions

AC domains cannot simply be "powered off" the same way as DC domains. They must be safely de-energized so residual oscillation does not corrupt neighboring blocks.

---

## 11. Clock Distribution and Synchronization

Two trees coexist:

- binary clock tree or PLL-derived tree
- AC resonant phase tree

Synchronization boundaries require:

- elastic buffers
- ILBs
- phase alignment monitors
- error reporting when AC and DC timing assumptions diverge

---

## 12. EDA and Verification Requirements

Existing tools must be extended for:

- AC waveform-aware timing signoff
- mixed resonant and static IR-drop analysis
- LC structure extraction
- resonance-domain thermal estimation
- mode-switch state validation

The spec therefore doubles as a requirement document for open and proprietary EDA ecosystems.

---

## 13. K3D Integration

AC-xPowerT is a strong fit for K3D because the platform already has a natural split:

- binary planes for system integration
- ternary planes for knowledge routing and uncertainty-aware compute

The scheduler can assign workloads according to:

- latency sensitivity
- ternary affinity
- energy-recovery potential
- thermal budget

---

## 14. Manufacturing Path

1. emulate AC and DC planes separately on FPGA and mixed-signal boards
2. prototype DC control on open PDKs
3. characterize resonant islands using research silicon or chiplets
4. integrate into mixed-signal flows with M3D and advanced lithography

---

## 15. Prior Art and Sources

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
18. EP2549654A1 — Low-consumption logic circuit / energy-recovery logic, https://patents.google.com/patent/EP2549654A1/en
19. AC-BT deployment baseline — [Marduk_Schlupp.md](Marduk_Schlupp.md)
20. AC-T end-state — [Ereshkigal_Schlupp.md](Ereshkigal_Schlupp.md)
21. RISC-BT hybrid baseline — [Enki_Schlupp.md](Enki_Schlupp.md)
22. x86/x64 hybrid baseline — [Ninlil_Schlupp.md](Ninlil_Schlupp.md)
23. SoC integration baseline — [Ninhursag_Schlupp.md](Ninhursag_Schlupp.md)

---

## 16. Companion Specs

- [Marduk_Schlupp.md](Marduk_Schlupp.md)
- [Ereshkigal_Schlupp.md](Ereshkigal_Schlupp.md)
- [Ningishzida_Schlupp.md](Ningishzida_Schlupp.md)
- [Ninazu_Schlupp.md](Ninazu_Schlupp.md)
- [Gugalanna_Schlupp.md](Gugalanna_Schlupp.md)
- [Ki_Schlupp.md](Ki_Schlupp.md)
