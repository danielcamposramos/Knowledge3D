# Ethernet-BT: Hybrid Binary + Ternary Network Compatibility Specification

**Version:** 0.1 DRAFT — Defensive Publication
**Date:** 2026-03-19
**Authors:** Daniel Campos Ramos (PM-KR Chair), Christoph Dorn (PM-KR Contributor), Milton Ponson (PM-KR Co-Chair)
**Organization:** PM-KR Community Group
**License:** W3C Royalty-Free — published as prior art under the W3C Patent Policy

---

## Foundational Ternary Distinction

Where this document refers to ternary hardware, the primitive is a native three-state element:
- `0` = natural rest position
- `+1` = one side of the relay/state cell
- `-1` = the other side of the relay/state cell

If an interface, toolchain, or teaching layer prefers ordinal labels, the same three states may also be written as `0, 1, 2` with an explicit mapping.

Packet semantics, routing state, and network policy are derived from that primitive. This document does not normatively depend on a unary increment/decrement gate family or one specific transistor topology.

## Purpose

`Ethernet-BT` is the network-hardware equivalent of the hybrid CPU bridge strategy:

- preserve current binary network infrastructure
- add ternary semantics where they improve control, routing, congestion handling, trust, telemetry, and smart edge decisions
- keep deployment friction low enough that ternary networking can land inside today's NICs, SmartNICs, switches, gateways, and border routers

This is the **current + ternary compatibility** document.

---

## Architectural Position

`Ethernet-BT` does **not** replace:

- Ethernet framing
- IPv4/IPv6
- PCIe host attachment
- DMA descriptor rings
- switch buffers
- existing NIC and switch software stacks

Instead, it adds ternary meaning above them:

- `-1` = negative / congested / deny / reverse / unsafe / below-threshold
- `0` = neutral / unknown / defer / stable / hold
- `+1` = positive / free / allow / promote / above-threshold

This keeps the current network world intact while opening ternary semantics inside the data plane and control plane.

---

## Open Reference Line

The compatibility architecture is grounded in open and primary-source networking hardware ecosystems:

- **Corundum**: open high-performance FPGA NIC and in-network compute platform
- **OpenNIC Shell**: open FPGA NIC shell and plugin boxes
- **NetFPGA-SUME**: open PCIe/network research board supporting NIC, switch, firewall, and measurement roles
- **P4 + PNA/PSA**: open programmable packet-processing language and architecture surfaces
- **OpenThread Border Router**: open edge/border-router pattern for IoT network integration

These are the correct open anchors for a patent-defensive network hardware specification.

---

## Hybrid Compatibility Model

### 1. Binary-preferred path

Keep binary for:

- standard Ethernet frame encoding
- MAC/PCS/PHY logic
- DMA and PCIe transactions
- descriptor ring management
- packet buffer indexing
- conventional checksum and byte-oriented protocol parsing

### 2. Ternary-preferred path

Use ternary for:

- route verdicts
- ACL verdicts
- congestion grading
- telemetry classification
- queue admission and scheduling state
- trust state
- packet fate decisions
- IoT edge wake/defer/escalate policy

### 3. Neutral / dispatchable path

Choose the cheaper substrate for:

- parser-side classification metadata
- policy-table lookups
- vector packet scoring
- event correlation

---

## Packet and Metadata Model

### 1. Packet payload remains binary

Current Ethernet/IP/TCP/UDP payloads remain binary. This is the compatibility guarantee.

### 2. Ternary sideband metadata

Each packet or packet batch gains optional ternary sideband fields:

- `route_trit`
- `congestion_trit`
- `trust_trit`
- `urgency_trit`
- `telemetry_trit`
- `sensor_event_trit`

These can travel:

- in descriptor sideband
- in metadata buses
- in programmable parser/extractor state
- in internal switch/NIC pipeline registers

### 3. Ternary action semantics

A single decision stage may now produce:

- `-1` = drop / quarantine / reroute-away / deprioritize
- `0` = hold / mirror / observe / defer
- `+1` = forward / admit / promote / accelerate

That is strictly richer than a binary pass/drop worldview.

---

## SmartNIC / Switch Pipeline Enhancements

### 1. Parser stage

Parser remains binary-header aware but emits ternary classification results.

Examples:

- packet clearly trusted -> `+1`
- packet suspicious but not proven malicious -> `0`
- packet explicitly denied or malformed -> `-1`

### 2. Match/action stage

P4-style tables or fixed-function classifiers may return ternary verdicts instead of boolean-only outcomes.

This is especially relevant for:

- policy engines
- ACLs
- QoS gates
- telemetry thresholds
- in-network compute triggers

### 3. Queue manager

Queue state should support ternary decisions:

- `-1` = congested / throttle / backpressure
- `0` = normal / stable
- `+1` = available / fast path

### 4. Egress scheduler

Scheduler can interpret ternary urgency:

- `-1` = background / delay
- `0` = ordinary service
- `+1` = priority boost

---

## IoT and Edge Networking Surface

### 1. Border router compatibility

The compatibility design should include an edge network block inspired by **OpenThread Border Router**:

- route registration
- external route advertisement
- local network data management
- network-data-full handling
- secure low-power edge gateway behavior

### 2. IoT ternary routing

For IoT edge gateways and sensor-border nodes, ternary state is immediately useful:

- `-1` = suppress / unsafe / invalid edge event
- `0` = uncertain / wait for another sensor / low-confidence
- `+1` = trigger / publish / wake host

### 3. Sensor-network compatibility

`Ethernet-BT` should assume attachment to:

- sensor hubs
- cameras
- microphone arrays
- low-power border routers
- industrial edge boxes

without requiring the network fabric itself to become fully ternary-native on day one.

---

## Security and Trust

### 1. Trust trit

Networking hardware should not collapse trust to yes/no only.

Recommended trust state:

- `-1` = explicitly untrusted / blocked
- `0` = unknown / not attested / under observation
- `+1` = trusted / attested / allowed

### 2. Security actions

This enables:

- quarantine instead of immediate drop
- staged inspection instead of blind allow
- safe defer in low-confidence situations

That is a stronger match for real security workflows.

---

## Implementation Path

### Phase 1

FPGA-based compatibility demonstrations on:

- Corundum-style NICs
- OpenNIC-style shells
- NetFPGA-style data-plane boards

### Phase 2

P4/PNA-style programmable packet pipeline with ternary sideband semantics.

### Phase 3

Silicon SmartNIC / switch ASIC with:

- binary packet substrate
- ternary control/routing sideband
- hybrid dispatch in queueing and policy stages

---

## References

- [Corundum](https://github.com/corundum/corundum)
- [NetFPGA-SUME](https://netfpga.org/NetFPGA-SUME.html)
- [AMD OpenNIC Shell](https://github.com/Xilinx/open-nic-shell)
- [P4 Language Consortium](https://p4.org/)
- [P4 Specifications](https://p4.org/specifications/)
- [OpenThread Border Router](https://openthread.io/reference/group/api-border-router)
