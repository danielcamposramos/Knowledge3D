# Sensor-T Open Ternary Sensor Fabric Specification

**Version:** 0.1 DRAFT - Defensive Publication
**Date:** 2026-03-19
**Organization:** W3C PM-KR Community Group

---

## 1. Purpose

This document defines the pure ternary sensor end-state: a sensing fabric where semantic state, confidence, and sparse world structure are native, and dense binary payloads are compatibility artifacts.

---

## 2. Native Ternary Sensor Objects

`Sensor-T` transports:

- object hypotheses
- sparse geometry blocks
- procedural depth/radiance tiles
- confidence, uncertainty, and salience trits
- keep/drop/refine control signals

Camera and LiDAR become native world-model feeds, not only raw stream devices.

---

## 3. Native Fabric Rules

- binary pixels and points may still be bridged in and out
- the preferred internal representation is sparse and procedural
- semantic trits are first-class transport members
- camera and LiDAR are designed as one fused perception fabric

---

## 4. K3D Alignment

`Sensor-T` matches K3D's worldview:

- perception feeds stars and structures directly
- uncertainty is explicit
- spatial knowledge updates arrive as procedural scene changes
- the House/Galaxy system can reason over native confidence, not inferred confidence

---

## Sources

- [MIPI CSI-2](https://www.mipi.org/specifications/csi-2)
- [MIPI A-PHY Overview via CSI-2 page](https://www.mipi.org/specifications/csi-2)
- [libcamera](https://libcamera.org/docs.html)
- [Ouster SDK](https://static.ouster.dev/sdk-docs/index.html)
