# RF-BT Hybrid Wireless Compatibility Specification

**Version:** 0.1 DRAFT - Defensive Publication
**Date:** 2026-03-19
**Organization:** PM-KR Community Group

---

## Foundational Ternary Distinction

Where this document refers to ternary wireless logic, the primitive is a native three-state element:
- `0` = natural rest position
- `+1` = one side of the relay/state cell
- `-1` = the other side of the relay/state cell

The same states may also be labeled `0, 1, 2` if the mapping is explicit.

Session trust, routing policy, and wireless meaning layers are derived from that primitive. This document does not normatively depend on a unary increment/decrement gate family or one specific transistor topology.

## 1. Purpose

This document defines the first wireless pattern: preserve current wireless PHYs and certifications, while adding ternary-aware session, routing, confidence, and IoT semantics on top.

Covered families:

- Wi-Fi
- Bluetooth / BLE
- LoRa / LoRaWAN
- Thread / low-power mesh

---

## 2. Compatibility Rule

Current binary radios remain valid:

- modulation stays standard-compliant
- packet framing stays standards-compliant
- certification path stays standards-compliant

Ternary additions live in control, metadata, and gateway logic:

- `-1` reject / deprioritize / low-confidence
- `0` neutral / unknown / maintain
- `+1` prioritize / accept / high-confidence

Examples:

- ternary QoS for sensor traffic
- ternary trust/provenance state for device onboarding
- ternary salience for edge event forwarding

---

## 3. IoT Gateway Role

Hybrid wireless is most practical first as a gateway architecture:

- binary radios on the air
- ternary session engine in the gateway
- ternary routing between sensor, edge compute, and knowledge system

That lets current IoT ecosystems participate immediately without waiting for a new PHY.

---

## 4. K3D and Sensor Relevance

For K3D-like embodied systems:

- Bluetooth and Wi-Fi handle conventional device interoperability
- LoRa handles long-range low-power telemetry
- Thread handles mesh devices
- ternary side-state carries confidence, urgency, and world-model relevance

---

## Sources

- [Bluetooth Specifications and Resources](https://www.bluetooth.com/specifications/specs/)
- [LoRa Alliance](https://lora-alliance.org/)
- [OpenThread](https://openthread.io/)
- [Arduino IoT Documentation](https://docs.arduino.cc/)
