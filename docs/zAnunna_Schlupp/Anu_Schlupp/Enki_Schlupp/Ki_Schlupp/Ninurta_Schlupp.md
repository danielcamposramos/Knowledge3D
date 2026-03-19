# Sensor-BT Hybrid Camera and LiDAR Compatibility Specification

**Version:** 0.1 DRAFT - Defensive Publication
**Date:** 2026-03-19
**Organization:** W3C PM-KR Community Group

---

## Foundational Ternary Distinction

Where this document refers to ternary sensing logic, the primitive is a native three-state element:
- `0` = natural rest position
- `+1` = one side of the relay/state cell
- `-1` = the other side of the relay/state cell

The same states may also be labeled `0, 1, 2` if the mapping is explicit.

Confidence, salience, and procedural scene semantics are derived from that primitive. This document does not normatively depend on a unary increment/decrement gate family or one specific transistor topology.

## 1. Purpose

This document defines the first imaging/sensing pattern: keep current camera and LiDAR transport standards, but add ternary-aware extensions for confidence, salience, classification, and procedural scene reconstruction.

Current standards remain valid:

- MIPI CSI-2 for camera transport
- MIPI A-PHY for long-reach sensor transport
- current Ethernet/packetized LiDAR transport
- current ISP and point-cloud toolchains

---

## 2. Compatibility Rule

Binary sensor payload remains the baseline:

- raw Bayer / YUV / RGB frames
- depth maps
- point clouds
- timestamps and calibration

The ternary extension adds sideband meaning:

- `-1` = reject / occluded / low-trust
- `0` = unknown / neutral / unchanged
- `+1` = accept / salient / high-trust

These trits can annotate:

- regions of interest
- object confidence
- motion certainty
- LiDAR return quality
- fusion confidence

---

## 3. Procedural Sensor Codec Extension

This specification assumes a companion software-side extension exists and standardizes the hardware side:

- procedural depth-field compression
- procedural radiance-field tiles
- procedural point-cloud block encoding
- event-style sparse update emission

Instead of only transporting dense frames, sensors may emit compact coefficient programs plus a confidence map.

---

## 4. Reference Hybrid Sensor Bridge

| Plane | Role |
|------|------|
| Binary plane | CSI-2 / packetized raster or point-cloud data |
| Ternary sideband | confidence, accept/reject, salience |
| Procedural plane | sparse field or point program payload |
| Security plane | end-to-end provenance and camera security |

This works for:

- embedded cameras
- automotive camera stacks
- drones
- AR/VR perception stacks
- robotics LiDAR + camera fusion

---

## 5. K3D Relevance

K3D benefits directly because sensor feeds can become:

- meaning-aware observations instead of only pixels
- confidence-labeled stars or House events
- procedural scene updates instead of endlessly dense frames

---

## Sources

- [MIPI CSI-2](https://www.mipi.org/specifications/csi-2)
- [MIPI Camera Security Framework Press Release](https://www.mipi.org/press-releases/mipi-releases-camera-security-specifications-for-flexible-end-to-end-protection-of-automotive-image-sensor-data)
- [libcamera Documentation](https://libcamera.org/docs.html)
- [Ouster SDK Documentation](https://static.ouster.dev/sdk-docs/index.html)
- [Raspberry Pi Camera Software Overview](https://www.raspberrypi.com/documentation/computers/camera_software.html)
