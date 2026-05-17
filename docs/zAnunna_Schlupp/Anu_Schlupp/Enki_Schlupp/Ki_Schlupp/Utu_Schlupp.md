# Ethernet-xNetT: True Hybrid Network Core Specification

**Version:** 0.1 DRAFT — Defensive Publication
**Date:** 2026-03-19
**Authors:** Daniel Campos Ramos (PM-KR Chair), Christoph Dorn (PM-KR Contributor), Milton Ponson (PM-KR Co-Chair)
**Organization:** PM-KR Community Group
**License:** W3C Royalty-Free — published as prior art under the W3C Patent Policy
**Companion Specification:** `Nanna_Schlupp.md`

---

## Foundational Ternary Distinction

The ternary side of this true hybrid network core uses the same primitive everywhere:
- `0` = natural rest position
- `+1` = one side of the relay/state cell
- `-1` = the other side of the relay/state cell

An optional alias notation of `0, 1, 2` is acceptable when the mapping is explicit.

Scheduling, trust, congestion, and semantic routing are derived from that primitive. This document does not normatively depend on a unary increment/decrement gate family or one specific transistor topology.

## Purpose

This is the **AMD-style one-core network document**.

The design pattern is:

- preserve the current network personality
- add a true ternary network personality in the same core/ASIC pipeline
- let both coexist inside one silicon identity rather than treating ternary as an external side engine

This is to networking what `x86_x64T` is to compute.

---

## Core Principle

One true network core contains:

- one true current binary network pipeline
- one true ternary network pipeline
- one shared parser, scheduler, memory system, and management plane
- one dispatch policy that routes packets, flows, and events between them

This is **not** just "binary with ternary metadata." It is a real unified network core with multiple architectural personalities.

---

## Mode Hierarchy

### 1. `NET-Legacy`

Ordinary Ethernet/IP/current network behavior.

### 2. `NET-Programmable`

Binary programmable packet path, including P4-class or SmartNIC-style match/action behavior.

### 3. `NETT-Narrow`

Narrow ternary network mode optimized for:

- edge decisions
- queueing
- trust
- IoT border routing
- sensor and event gating

### 4. `NETT-Wide`

Wide ternary network mode optimized for:

- large policy tables
- flow correlation
- telemetry grading
- route-field computation
- ternary network analytics

### 5. `NET-Mixed`

Packets and flows pass through both binary and ternary stages in a single pipeline context.

This is the most important production mode.

---

## Unified Network Core Structure

```text
Ingress MAC / PHY / SerDes
  -> shared parser
  -> binary decode path
  -> ternary decode path
  -> unified flow metadata builder
  -> hybrid dispatch unit
  -> binary packet-processing cluster
  -> ternary packet-processing cluster
  -> shared queue manager
  -> shared buffer manager
  -> shared scheduler
  -> egress formatter / MAC / PHY
```

---

## Hybrid Dispatch Unit

The network core contains a **Hybrid Dispatch Unit (HDU)** for packet/flow/event routing.

Its decisions include:

- keep on binary path
- move to ternary path
- split across binary and ternary stages

Inputs include:

- packet class
- flow class
- queue pressure
- trust state
- telemetry state
- energy target
- latency target
- programmable policy hints

---

## Binary Network Cluster

This cluster remains first-class and hosts:

- byte- and bit-oriented parsing
- conventional header rewrite
- checksum/update logic
- compatibility packet formats
- standard DMA and host interface work
- established SmartNIC and switch fast paths

It is not demoted. It stays where binary remains the cheapest substrate.

---

## Ternary Network Cluster

This cluster hosts:

- ternary compare/select
- route trit generation
- congestion trit generation
- trust trit generation
- ternary queue admission policy
- ternary scheduler hints
- ternary correlation for sensor/IoT traffic

Examples:

- `-1 / 0 / +1` queue state
- `-1 / 0 / +1` trust state
- `-1 / 0 / +1` route preference
- `-1 / 0 / +1` telemetry severity

---

## Shared State

### Flow state

Each flow may carry:

- binary identifiers
- ternary policy trits
- ternary trust state
- ternary congestion state
- ternary urgency state

### Queue state

Each queue may carry:

- occupancy counters
- backpressure counters
- ternary congestion grade

### Event state

IoT and sensor-originated events may carry:

- invalid / suppress = `-1`
- uncertain / defer = `0`
- escalate / wake = `+1`

---

## Scheduling Model

Scheduler decisions should be able to express:

- deprioritize
- keep steady
- promote

directly as ternary state.

That gives a cleaner hardware model for:

- congestion
- differentiated service
- anomaly handling
- observability pipelines

than a purely binary admission policy.

---

## Memory and Buffers

Binary payload remains binary.

The unified core should support ternary metadata in:

- descriptor sideband
- queue state SRAM
- policy SRAM
- telemetry registers

Optional later path:

- dense ternary metadata packing in policy SRAM / tables

---

## IoT / Border Router Role

The unified network core should directly support an edge-border role:

- current network uplink
- low-power edge/IoT downstream
- ternary policy on sensor-derived traffic
- border-router wake/defer/escalate behavior

This is one of the fastest routes to real adoption because IoT and edge networking naturally benefit from an explicit uncertain state.

---

## Security Model

Security policy in the unified core should treat ternary trust as first-class:

- `-1` = block / quarantine
- `0` = inspect / mirror / observe
- `+1` = allow / fast path

This is a more realistic silicon security model than pass/drop alone.

---

## Implementation Path

### Phase 1

Unified FPGA data-plane proof on Corundum / OpenNIC / NetFPGA-class platforms.

### Phase 2

Programmable hybrid packet core with binary and ternary stages.

### Phase 3

Silicon SmartNIC / switch core with:

- current binary packet personality
- ternary packet personality
- unified dispatch, queue, and schedule plane

---

## Strategic Role

`Ethernet-BT` is the compatibility bridge.

`Ethernet-xNetT` is the actual one-core hybrid network architecture.

This is the document that says:

> one current network core + one ternary network core = one real product core

---

## References

- [Corundum](https://github.com/corundum/corundum)
- [AMD OpenNIC Shell](https://github.com/Xilinx/open-nic-shell)
- [NetFPGA-SUME](https://netfpga.org/NetFPGA-SUME.html)
- [P4 Specifications](https://p4.org/specifications/)
- [OpenThread Border Router](https://openthread.io/reference/group/api-border-router)
