# x86_x64T Ternary Universal SoC Specification

**Version:** 0.1 DRAFT — Defensive Publication
**Date:** 2026-03-19
**Authors:** Daniel Campos Ramos (PM-KR Chair), Christoph Dorn (PM-KR Contributor), Milton Ponson (PM-KR Co-Chair)
**Organization:** PM-KR Community Group
**License:** W3C Royalty-Free — published as prior art under the W3C Patent Policy
**Companion Specifications:** `Anu_Schlupp.md`, `Antu_Schlupp.md`, `Ninlil_Schlupp.md`

---

## Foundational Ternary Distinction

The ternary parts of this SoC follow the same primitive:
- `0` = natural rest position
- `+1` = one side of the relay/state cell
- `-1` = the other side of the relay/state cell

The same states may also be labeled `0, 1, 2` if the mapping is explicit.

Sensor state, security policy, power orchestration, and interconnect semantics are derived from that primitive. This document does not normatively depend on a unary increment/decrement gate family or one specific transistor topology.

## Notice of Defensive Publication

This document publishes prior art for a **single-chip open ternary system-on-chip** built on top of the `x86_x64T` hybrid core family. The purpose is to move beyond CPU-only thinking and define a complete chip architecture including:

- compute
- secure root of trust
- always-on sensor and power domains
- IoT and embedded peripheral fabrics
- camera, audio, and actuator interfaces
- vector/GPU acceleration
- reconfigurable fabric
- future chiplet expansion

The design intent is to open ternary hardware to the broadest possible implementation surface: cloud, edge, embedded, industrial, robotics, and sensor-rich always-on systems.

---

## Table of Contents

1. [Purpose](#1-purpose)
2. [Why a Universal SoC Layer Is Needed](#2-why-a-universal-soc-layer-is-needed)
3. [Source-Derived Open Building Blocks](#3-source-derived-open-building-blocks)
4. [Universal SoC Architecture](#4-universal-soc-architecture)
5. [Domain-by-Domain Ternary Enhancements](#5-domain-by-domain-ternary-enhancements)
6. [Sensors and IoT Expansion Surface](#6-sensors-and-iot-expansion-surface)
7. [Security, Safety, and Trust](#7-security-safety-and-trust)
8. [Implementation Tiers](#8-implementation-tiers)
9. [Next Specs to Publish](#9-next-specs-to-publish)
10. [References](#10-references)

---

## 1. Purpose

The earlier documents define:

- a native ternary ISA (`RISC-T`)
- a ternary bridge for x64 systems (`X64-BT`)
- a one-core hybrid product architecture (`x86_x64T`)

This document goes one level higher:

> what should the full chip contain if we want to open ternary hardware as widely as possible?

The answer is not "just another CPU." The answer is a universal SoC with:

- binary + ternary compute on one die
- sensor-first and always-on operation
- open security blocks
- edge/IoT integration
- reconfigurable space for experimentation
- an upgrade path to chiplets and richer accelerators

---

## 2. Why a Universal SoC Layer Is Needed

If ternary remains only a CPU or ALU topic, it will be too easy for the market to contain it. The architectural opening has to happen at the SoC level.

That means ternary should be specified not only for:

- arithmetic
- compare-and-branch
- K3D opcodes

but also for:

- sensor classification
- alert routing
- power state control
- trusted/untrusted/unknown security state
- camera and audio wake logic
- IoT sideband and control paths
- heterogeneous accelerator cooperation

This is where ternary can become a system principle, not a niche arithmetic curiosity.

---

## 3. Source-Derived Open Building Blocks

The following open ecosystems provide concrete design patterns worth absorbing into the ternary SoC line.

### 3.1 Open secure root and always-on security domains

From **OpenTitan** and **Caliptra**:

- always-on domain
- sensor control and alert handling
- ADC controller and analog sensor top
- life-cycle and OTP/fuse control
- entropy source, CSRNG, and entropy distribution
- open root-of-trust macro with auditable RTL

This gives the ternary SoC a credible security and always-on foundation instead of leaving security as an afterthought.

### 3.2 Open MCU / edge peripheral substrate

From **CORE-V MCU**:

- UART
- QSPI
- I2C master + I2C slave
- SDIO
- camera interface
- GPIO
- PWM / advanced timer
- IO muxing
- interleaved SRAM and flat memory map

From **PULPissimo / PULP**:

- uDMA-centered I/O subsystem
- I2S and camera-oriented edge I/O
- tightly coupled memory
- APB/AXI split fabrics
- always-on and timer/power controller ideas
- eFPGA integration for post-silicon accelerators

These are exactly the kinds of blocks needed for a serious ternary edge/IoT chip.

### 3.3 Sensor bus modernization

From **MIPI I3C / I3C Basic / I3C HCI**:

- low-power sensor/actuator command and data transport
- always-on imaging support
- server system management potential
- in-band interrupt model
- host-controller standardization
- I2C coexistence and migration path

This is the best open-facing bus surface for a sensor-rich ternary SoC.

### 3.4 Open memory safety and IoT resilience

From **Ibex / CHERIoT-Ibex**:

- memory-safety assistance at the hardware level
- explicitly low-cost embedded/IoT orientation
- open verification and evaluation paths

This should influence the narrow ternary edge domain.

### 3.5 Open application-class core patterns

From **CVA6**:

- application-processor-class open CPU documentation
- configurable 32/64-bit class processor pattern
- clearer separation of requirements spec and design spec

This is useful when defining the higher-performance binary management and host domain around the ternary blocks.

### 3.6 Open GPU and accelerator patterns

From **MIAOW**, **Nyuzi**, and **Vortex**:

- open GPGPU execution models
- lane-first thinking
- research-friendly open RTL
- vector/SIMT architectural discipline

These are the right reference points for a ternary vector/GPU subsystem.

### 3.7 Open chiplet and packaging path

From **UCIe**:

- die-to-die interconnect
- multi-vendor chiplet assembly
- customized SoC construction beyond reticle limits

This matters because ternary may first win as a specialized tile before it wins as an entire monolithic CPU market.

### 3.8 Open implementation stack

From **Yosys**, **OpenROAD**, **open_pdks**, **OpenLane**, **Sky130/chipIgnite**:

- synthesis
- place-and-route
- open PDK normalization
- affordable or free open tapeout path
- reproducible educational and production-adjacent flows

Without this stack, the SoC spec would be strategically incomplete.

---

## 4. Universal SoC Architecture

The proposed SoC is organized into eight major domains.

### 4.1 Domain A — `x86_x64T` compute complex

Contains:

- legacy x86 execution
- x64 long mode
- x86T narrow ternary mode
- x64T long ternary mode
- hybrid dispatch unit

This is the main application processor complex.

### 4.2 Domain B — always-on sensor and power island

Contains:

- always-on timers
- power manager
- clock/reset manager
- sensor controller
- ADC controller
- analog sensor top
- wake event concentrator

This domain stays alive when the main compute complex sleeps.

### 4.3 Domain C — secure root-of-trust island

Contains:

- boot ROM
- OTP / fuse / life-cycle state
- key manager
- entropy source
- cryptographic root services
- measured boot and recovery services

This block should follow the OpenTitan/Caliptra model of open, auditable, synthesizable root-of-trust IP.

### 4.4 Domain D — IoT and edge peripheral fabric

Contains:

- I3C / I3C Basic host controller
- I2C master/slave
- SPI / QSPI
- UART
- GPIO
- PWM / timers
- SDIO
- camera control / camera input
- audio control interfaces
- debug sideband

This domain is what makes the chip relevant to embedded systems, robotics, appliances, industrial systems, wearables, and sensor hubs.

### 4.5 Domain E — vector / GPU / accelerator fabric

Contains:

- packed ternary vector unit
- binary vector unit
- open GPGPU-style compute subsystem
- DMA engines
- accelerator control plane

This domain is where MIAOW / Nyuzi / Vortex-inspired designs plug into the ternary system strategy.

### 4.6 Domain F — reconfigurable fabric

Contains:

- small open eFPGA region
- APB configuration plane
- direct memory or tightly coupled memory attachment

Purpose:

- post-silicon experimentation
- domain-specific kernels
- sensor and control adapters
- new ternary co-process elements before full hardening

### 4.7 Domain G — memory hierarchy

Contains:

- boot ROM
- retention SRAM
- interleaved SRAM
- shared L2 or tightly coupled banks
- packed ternary storage support
- lane-form ternary execution support

### 4.8 Domain H — chiplet expansion edge

Contains:

- UCIe-ready boundary
- chiplet management logic
- external accelerator tile attachment path

This is optional for first silicon but should be part of the spec from day one.

---

## 5. Domain-by-Domain Ternary Enhancements

### 5.1 Sensor ternary classification

Sensor blocks should not reduce everything to binary alarms.

For many sensor classes, the natural early hardware classification is:

- `-1` = below / falling / depleted / unsafe-low
- `0` = nominal / unknown / stable / indifferent
- `+1` = above / rising / overloaded / unsafe-high

This is a more useful primitive than a single threshold interrupt.

### 5.2 Alert ternary routing

Alert routing should support:

- `-1` = suppress / quarantine / reverse action
- `0` = observe / log / defer
- `+1` = escalate / wake / act

This is especially relevant for always-on domains and low-power wake logic.

### 5.3 Power ternary policy

Power control naturally admits ternary states:

- `-1` = low-power / retention / minimum
- `0` = nominal / steady state
- `+1` = boost / urgent / burst mode

This can be used in DVFS policy and sensor-driven wake decisions.

### 5.4 Trust ternary state

Security state should not collapse to trusted/untrusted only.

Useful hardware trust trit:

- `-1` = explicitly untrusted / revoked / compromised
- `0` = unknown / not attested / pending
- `+1` = attested / trusted / verified

That maps cleanly to boot and attestation workflows.

### 5.5 Network and fabric congestion trits

On-chip or chiplet fabrics can use:

- `-1` = congested / backpressure
- `0` = normal
- `+1` = free / priority-fast path available

This is particularly interesting for sensor bursts and accelerator sharing.

---

## 6. Sensors and IoT Expansion Surface

### 6.1 Minimum sensor/IoT surface

The SoC should spec, at minimum:

- I3C / I3C Basic
- I2C
- SPI / QSPI
- UART
- GPIO
- PWM
- SDIO
- camera control / camera input
- audio input/output control

### 6.2 Recommended always-on sensor classes

The spec should assume direct support for:

- motion sensors
- biometric sensors
- environmental sensors
- microphones
- simple image / presence sensors
- power and thermal monitors
- actuator controls

### 6.3 Sensor hub behavior

The always-on island should be able to:

- ingest low-rate sensor streams
- classify them into ternary event state
- wake the main compute complex only when needed
- log and rate-limit noisy devices
- hand off dense or packed data via DMA

### 6.4 IoT edge identity

This SoC should be valid for:

- wearables
- robotics
- cameras and smart sensors
- industrial control nodes
- gateways
- secure edge inference boxes
- K3D-oriented embodied agents

---

## 7. Security, Safety, and Trust

### 7.1 Security baseline

Every serious ternary SoC should now include:

- open root of trust
- entropy and key management
- secure boot and measured boot
- recovery path
- life-cycle and ownership transfer logic

### 7.2 Memory-safety direction

The edge and IoT variants should take explicit inspiration from CHERIoT-Ibex-style memory-safety ideas.

The ternary extension opportunity here is:

- trusted / unknown / untrusted memory region state
- capability verdict trits
- permission evaluation without collapsing undecided cases to a dangerous default

### 7.3 Safety path

Always-on sensor and industrial deployments should publish:

- safe-state behavior
- wake/suppress/escalate ternary decisions
- watchdog and recovery transitions

This is especially important if PM-KR wants the hardware story to extend beyond AI and into cyber-physical systems.

---

## 8. Implementation Tiers

### 8.1 Tier A — Minimal open ternary edge demonstrator

Recommended components:

- one `x86T/x64T`-inspired ternary-capable compute tile or emulated control tile
- always-on sensor island
- I3C + I2C + SPI + UART + GPIO + PWM
- secure boot micro-root
- OpenLane / Sky130 / chipIgnite-class flow

Goal:

- prove ternary SoC, not just ternary ALU

### 8.2 Tier B — Rich IoT / robotics ternary SoC

Recommended additions:

- camera interface
- audio interface
- eFPGA fabric
- richer DMA
- secure root-of-trust subsystem
- ternary sensor fusion path

Goal:

- prove practical sensor/actuator and embodied-agent usage

### 8.3 Tier C — Workstation / server ternary platform SoC

Recommended additions:

- stronger x64 host cluster
- larger vector / GPU complex
- UCIe boundary
- advanced memory hierarchy
- system management channel

Goal:

- bridge into datacenter and workstation adoption

---

## 9. Next Specs to Publish

To maximize the opening, the following specs should be written next:

1. **Ternary Sensor Controller Specification**
2. **Ternary Always-On Event Router Specification**
3. **Ternary I3C Host Controller Specification**
4. **Ternary Root-of-Trust Integration Specification**
5. **Ternary Packed-Memory and Cache Specification**
6. **Ternary eFPGA Programming Model Specification**
7. **Ternary UCIe Tile / Chiplet Interface Specification**
8. **Ternary Edge Agent SoC Reference Design**

These are the places where open ternary hardware can spread beyond a CPU niche.

---

## 10. References

### Security / always-on / sensor-domain references

- [OpenTitan Documentation](https://opentitan.org/documentation/)
- [OpenTitan Sensor Control Theory of Operation](https://opentitan.org/book/hw/top_earlgrey/ip/sensor_ctrl/doc/theory_of_operation.html)
- [OpenTitan Entropy Source](https://opentitan.org/book/hw/ip/entropy_src/)
- [Caliptra joins CHIPS Alliance](https://www.chipsalliance.org/news/chips-alliance-welcomes-the-caliptra-open-source-root-of-trust-project/)
- [Caliptra 2.1 RTL release](https://www.chipsalliance.org/news/caliptra2-1/)

### Edge / MCU / IoT references

- [CORE-V MCU DevKit Hardware Description](https://docs.openhwgroup.org/projects/core-v-mcu-devkit-user-manual/en/latest/doc-src/hardware-description.html)
- [CORE-V MCU High Level Architecture](https://docs.openhwgroup.org/projects/core-v-mcu/doc-src/high_level_architecture.html)
- [PULP platforms overview](https://pulp-platform.org/docs/hipeac/2020/PULP_WRC_2020.pdf)
- [Ibex CPU and CHERIoT-Ibex](https://lowrisc.org/ibex/)
- [CVA6 documentation](https://docs.openhwgroup.org/projects/cva6-user-manual/)

### Sensor bus and host interface references

- [MIPI I3C / I3C Basic](https://www.mipi.org/specifications/i3c-sensor-specification)
- [MIPI I3C Host Controller Interface](https://www.mipi.org/specifications/i3c-hci)

### GPU / accelerator / chiplet references

- [MIAOW GPU](https://miaowgpu.org/)
- [NyuziProcessor](https://github.com/jbush001/NyuziProcessor)
- [Vortex GPGPU](https://vortex.cc.gatech.edu/)
- [UCIe Specifications](https://www.uciexpress.org/specifications)

### Open implementation references

- [OpenROAD documentation](https://openroad.readthedocs.io/en/latest/)
- [Yosys documentation](https://yosyshq.net/yosys/documentation.html)
- [open_pdks reference](https://opencircuitdesign.com/open_pdks/reference.html)
- [Efabless Open MPW Program](https://efabless.com/open_shuttle_program)
- [OpenLane](https://efabless.com/openlane)
- [chipIgnite](https://efabless.com/news/press-release-efabless-launches-chipignite-with-skywater-to-bring-chip-creation-to-the-masses)

### Companion specs

- `Anu_Schlupp.md`
- `Antu_Schlupp.md`
- `Ninlil_Schlupp.md`
