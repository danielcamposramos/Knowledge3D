# Net-T: Open Ternary Network Fabric Specification

**Version:** 0.1 DRAFT — Defensive Publication  
**Date:** 2026-03-19  
**Authors:** Daniel Campos Ramos (PM-KR Chair), Christoph Dorn (PM-KR Contributor), Milton Ponson (PM-KR Co-Chair)  
**Organization:** W3C PM-KR Community Group  
**License:** W3C Royalty-Free — published as prior art under the W3C Patent Policy  

---

## Foundational Ternary Distinction

`Net-T` is built on a native three-state primitive:
- `0` = natural rest position
- `+1` = one side of the relay/state cell
- `-1` = the other side of the relay/state cell

An optional alias notation of `0, 1, 2` may be used when the mapping is explicit.

Networking arithmetic, trust state, and routing semantics are derived from that primitive. This document does not normatively depend on a unary increment/decrement gate family or one specific transistor topology.

## Purpose

`Net-T` is the **full ternary** network-hardware specification.

If `Ethernet-BT` is the compatibility bridge and `Ethernet-xNetT` is the one-core hybrid ASIC story, then `Net-T` is the clean native network-fabric architecture designed around ternary principles from the start.

This document exists so that the network future is not locked into binary assumptions forever.

---

## Foundational Model

Binary networking tends to collapse many meaningful system states into two buckets.

`Net-T` instead treats ternary state as native:

- `-1` = negative / reverse / congested / deny / unsafe
- `0` = neutral / unknown / observe / steady
- `+1` = positive / forward / free / allow / promote

The result is a network fabric that reasons more naturally about:

- routing
- congestion
- trust
- telemetry
- QoS
- IoT and sensor event propagation

---

## Native Ternary Fabric Elements

### 1. Ternary route field

A route field should not just encode a next hop. It should carry ternary route preference:

- `-1` = route-away / penalize / reverse preference
- `0` = no preference / unresolved / equal candidate
- `+1` = prefer / accelerate / direct

### 2. Ternary congestion field

Congestion should be natively graded:

- `-1` = congested / backpressure
- `0` = stable / nominal
- `+1` = free / underutilized

### 3. Ternary trust field

Trust should be native:

- `-1` = untrusted
- `0` = unknown / not attested
- `+1` = trusted

### 4. Ternary telemetry field

Telemetry severity should be native:

- `-1` = problem state
- `0` = informational / unresolved
- `+1` = good / healthy / promoted

---

## Packet / Cell Model

`Net-T` allows data transport units to be represented in ternary-first form.

This does **not** require application payloads to be ternary. It means the network's control and transport fabric is ternary-native.

The native fabric unit may carry:

- binary payload region
- ternary control region
- ternary route metadata
- ternary trust and congestion metadata

In a pure native future, even packet headers and queue metadata can be ternary-packed natively. In early systems, payload may still remain binary while fabric state is ternary.

---

## Ternary Switching Fabric

### 1. Crossbar or NoC state

Each output arbitration decision can be:

- `-1` = deny / defer / reroute
- `0` = hold / undecided / equal-weight
- `+1` = grant / promote

### 2. Buffer state

Buffers carry native ternary occupancy state:

- `-1` = full-pressure / danger
- `0` = nominal
- `+1` = available

### 3. Arbitration

A ternary arbitration stage can naturally encode:

- reject
- wait
- admit

without converting everything into chained binary comparators.

---

## Native Ternary Routing

`Net-T` routing should support:

- multi-path route preference as ternary field
- local/unknown/remote quality trits
- ternary policy at each hop
- route confidence and trust carried as state, not external annotations only

This is useful for:

- software-defined fabrics
- data center fabrics
- edge meshes
- robotics interconnects
- sensor swarms

---

## IoT and Sensor Networking

This is where `Net-T` becomes especially powerful.

Sensor and IoT traffic often has an irreducible third state:

- invalid / contradictory
- uncertain / waiting
- valid / trigger-worthy

The native network fabric should support that directly for:

- sensor event propagation
- low-power wake policies
- edge correlation
- border routing
- distributed control systems

---

## Fabric Security

In `Net-T`, security policy should be native to the fabric, not only host-side software.

Every link, packet, or flow may carry a trust trit:

- `-1` = revoked / unsafe
- `0` = unknown / inspect
- `+1` = attested / allowed

This makes quarantine, mirror, defer, and allow all native fabric behaviors.

---

## Native Ternary NIC / Switch / Router Roles

The same `Net-T` primitives should support:

- NICs
- SmartNICs
- switch ASICs
- border routers
- industrial edge switches
- robotic coordination networks

That is the key advantage of publishing the fabric as an open ternary standard instead of a single product.

---

## Binary Bridge

Even the full ternary fabric requires bridge logic to today's world.

That bridge should:

- map binary Ethernet/IP traffic into ternary fabric state
- export ternary trust/congestion/routing state as ordinary metadata where needed
- terminate or gateway native ternary fabrics to binary external networks

This keeps `Net-T` from becoming isolated.

---

## Implementation Path

### Phase 1

Emulate native ternary network-state handling on FPGA with open NIC and switch platforms.

### Phase 2

Implement native ternary queue, scheduler, and route metadata fabrics.

### Phase 3

Tape out a native ternary network element or tile:

- NIC tile
- switch tile
- border-router tile

### Phase 4

Scale to chiplet or board-level native ternary interconnect.

---

## Strategic Role

`Net-T` is the clean constitutional architecture for ternary networking.

`Ethernet-BT` is the compatibility bridge.  
`Ethernet-xNetT` is the one-core hybrid deployment architecture.  
`Net-T` is the open native future.

All three are needed to fully pre-empt patent capture of ternary networking ideas.

---

## References

- [Corundum](https://github.com/corundum/corundum)
- [AMD OpenNIC Shell](https://github.com/Xilinx/open-nic-shell)
- [NetFPGA-SUME](https://netfpga.org/NetFPGA-SUME.html)
- [P4 Language Consortium](https://p4.org/)
- [P4 Specifications](https://p4.org/specifications/)
- [OpenThread Border Router](https://openthread.io/reference/group/api-border-router)
