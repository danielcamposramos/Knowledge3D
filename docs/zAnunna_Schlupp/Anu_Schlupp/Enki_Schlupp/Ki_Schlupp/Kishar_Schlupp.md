# RF-xRFT True Hybrid Wireless Core Specification

**Version:** 0.1 DRAFT - Defensive Publication
**Date:** 2026-03-19
**Organization:** PM-KR Community Group

---

## Foundational Ternary Distinction

The ternary side of this true hybrid wireless core follows the same primitive:
- `0` = natural rest position
- `+1` = one side of the relay/state cell
- `-1` = the other side of the relay/state cell

The same states may also be labeled `0, 1, 2` if the mapping is explicit.

Policy, scheduling, and wireless semantic state are derived from that primitive. This document does not normatively depend on a unary increment/decrement gate family or one specific transistor topology.

## 1. Purpose

This is the second wireless pattern: one true hybrid wireless core with a standard binary RF chain and a ternary policy/scheduling/meaning engine on the same chip.

It targets:

- IoT gateways
- robotics edge nodes
- industrial controllers
- mobile/embedded communication SoCs

---

## 2. Core Topology

```
Binary RF PHYs (Wi-Fi / BLE / LoRa / Thread)
                |
       +--------v--------+
       | MAC / Packet RX |
       +---+--------+----+
           |        |
   +-------v--+  +--v----------------+
   | Binary   |  | Ternary Session   |
   | MAC / QoS|  | Engine            |
   | codecs   |  | trust / salience  |
   | DMA      |  | route / urgency   |
   +-------+--+  +--------+----------+
           |              |
           +------v-------+
                  |
         Hybrid Router / Export
```

### 2.1 Binary Domain

- certified RF modulation and demodulation
- standard MAC timing
- encryption/authentication transport primitives
- legacy device interoperability

### 2.2 Ternary Domain

- trust state
- event salience
- edge route priority
- local keep/drop/fuse decisions for multimodal systems

---

## 3. Power and Deployment

The binary RF path stays awake only for standards-required radio work. The ternary domain may sleep independently or remain active as an always-on semantic router.

This matches low-power IoT deployment better than forcing ternary into the RF waveform immediately.

---

## Sources

- [Bluetooth Core Resources](https://www.bluetooth.com/specifications/specs/)
- [LoRa Alliance](https://lora-alliance.org/)
- [OpenThread Border Router](https://openthread.io/guides/border-router)
- [Arduino Docs](https://docs.arduino.cc/)
