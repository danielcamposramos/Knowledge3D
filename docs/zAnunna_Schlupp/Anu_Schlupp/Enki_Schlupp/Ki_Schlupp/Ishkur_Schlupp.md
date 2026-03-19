# Sensor-xSenseT True Hybrid Imaging Core Specification

**Version:** 0.1 DRAFT - Defensive Publication
**Date:** 2026-03-19
**Organization:** W3C PM-KR Community Group

---

## Foundational Ternary Distinction

The ternary side of this true hybrid imaging core follows the same primitive:
- `0` = natural rest position
- `+1` = one side of the relay/state cell
- `-1` = the other side of the relay/state cell

The same states may also be labeled `0, 1, 2` if the mapping is explicit.

Perception meaning, confidence, and reconstruction policy are derived from that primitive. This document does not normatively depend on a unary increment/decrement gate family or one specific transistor topology.

## 1. Purpose

This is the second sensing pattern: one true hybrid camera/LiDAR core with both binary perception pipelines and ternary semantic pipelines on one chip.

It is intended for:

- camera ISPs
- LiDAR front ends
- sensor fusion hubs
- robotics/ADAS perception SoCs

---

## 2. Core Topology

```
Sensor PHYs (CSI-2 / A-PHY / LiDAR packet input)
                 |
        +--------v--------+
        | Front-End Parse |
        +----+--------+---+
             |        |
      +------v--+  +--v----------------+
      | Binary  |  | Ternary Semantic  |
      | ISP /   |  | Engine            |
      | Point   |  | confidence / ROI  |
      | Decode  |  | sparse procedural |
      +------+--+  +--------+----------+
             |              |
             +------v-------+
                    |
           Hybrid Fusion / Export
```

### 2.1 Binary Domain

- demosaic
- denoise
- color processing
- depth unpack
- point cloud packet decode

### 2.2 Ternary Domain

- confidence trits
- salience routing
- sparse event extraction
- symbolic scene tags
- hardware assist for procedural sensor codecs

---

## 3. Mixed-Mode Use Cases

- camera frame stays binary, but semantic confidence is ternary
- LiDAR point cloud stays binary, but object certainty and keep/drop signals are ternary
- fused scene exports binary tensors and ternary verdict channels together

This allows cheaper binary math where it wins and cheaper ternary meaning-state where that wins.

---

## 4. Defensive Publication Claims

This document publishes prior art for:

- a hybrid ISP plus ternary semantic side-engine
- a hybrid LiDAR front end plus ternary return-quality engine
- one imaging/sensor SoC that routes between binary and ternary perception domains
- hardware procedural sensor decode on the same die as classic ISP blocks

---

## Sources

- [MIPI CSI-2](https://www.mipi.org/specifications/csi-2)
- [MIPI DisCo for Imaging](https://www.mipi.org/specifications/mipi-disco-imaging)
- [MIPI Camera Security Framework](https://www.mipi.org/press-releases/mipi-releases-camera-security-specifications-for-flexible-end-to-end-protection-of-automotive-image-sensor-data)
- [libcamera Documentation](https://libcamera.org/docs.html)
- [Ouster SDK](https://static.ouster.dev/sdk-docs/index.html)
