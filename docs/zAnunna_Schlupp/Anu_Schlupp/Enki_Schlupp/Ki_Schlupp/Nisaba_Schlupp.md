# Display-BT Hybrid Display Compatibility Specification

**Version:** 0.1 DRAFT - Defensive Publication
**Date:** 2026-03-19
**Organization:** PM-KR Community Group

---

## Foundational Ternary Distinction

Where this document refers to ternary display logic, the primitive is a native three-state element:
- `0` = natural rest position
- `+1` = one side of the relay/state cell
- `-1` = the other side of the relay/state cell

The same states may also be labeled `0, 1, 2` if the mapping is explicit.

Display confidence, procedural decode state, and rendering policy are derived from that primitive. This document does not normatively depend on a unary increment/decrement gate family or one specific transistor topology.

## 1. Purpose

This document defines the first display-side pattern: keep current display infrastructure intact, but add ternary-compatible extensions on top of HDMI, DisplayPort, and MIPI DSI so no future hybrid display controller or monitor topology can be patented as a closed invention.

Current standards remain valid:

- HDMI 2.1b class links
- DisplayPort 2.1 class links
- MIPI DSI-2 display links
- VESA DSC transport compression

The ternary additions live above or beside those standards.

---

## 2. Compatibility Model

Binary raster transport remains the normative fallback. Every compliant controller or monitor may still consume:

- pixel raster
- timing
- HDR metadata
- audio/control channels
- EDID/DDC style display capability exchange

The ternary extension adds three new concepts:

1. **Ternary semantic sideband**
   States like `-1 / 0 / +1` represent suppress, neutral, or amplify for quality-of-service, region-of-interest, and confidence.

2. **Procedural display payloads**
   Instead of sending only raster frames, the source may send coefficient programs for procedural reconstruction.

3. **Hybrid surface composition**
   One monitor can show binary raster surfaces and ternary procedural surfaces in the same frame budget.

---

## 3. Procedural Codec Decoder

This specification explicitly adds a hardware decoder class for K3D procedural visuals:

- `VectorDotMap`
- procedural image codec programs
- procedural text/glyph programs
- resolution-independent coefficient fields

The monitor/controller side reconstructs images at panel-native resolution from coefficient payloads. This is the display-hardware equivalent of K3D's claim that the same representation can render at 1080p, 4K, or 8K without storing separate bitmaps.

### 3.1 Infinite Resolution Rule

The display controller is allowed to scale procedural payloads to the panel's native grid at render time. Storage size is therefore bounded by program complexity, not by raster area.

---

## 4. Hybrid Link Structure

| Plane | Role |
|------|------|
| Binary plane | HDMI/DP/DSI-compatible raster fallback |
| Ternary sideband | confidence, intent, LOD, semantic emphasis |
| Procedural plane | VectorDotMap / procedural image programs |
| Composition plane | on-monitor merge of raster + procedural surfaces |

Examples:

- UI chrome in raster, infinite-resolution diagrams in procedural form
- video in binary stream, captions/annotations in ternary procedural form
- CAD overlays, knowledge maps, and tablet content rendered at panel-native sharpness

---

## 5. Monitor Architecture

A compliant monitor may contain:

- binary link PHY and scaler
- ternary sideband decoder
- procedural codec decoder
- on-panel compositor
- local cache for procedural coefficient pages

This architecture applies to:

- desktop monitors
- TVs
- cockpit displays
- AR/VR microdisplay bridges
- digital signage

---

## 6. K3D Integration

This document assumes the software-side extension exists in K3D:

- `VectorDotMap` encode/decode
- procedural image payload generation
- sovereign codec and RPN-driven scene composition

The hardware claim published here is the monitor/controller half of that stack.

---

## Sources

- [HDMI 2.1b Overview, HDMI Licensing Administrator](https://www.hdmi.org/spec/hdmi2_1/index.aspx)
- [VESA DisplayPort 2.1 Release](https://vesa.org/featured-articles/vesa-releases-displayport-2-1-specification/)
- [MIPI DSI-2](https://www.mipi.org/specifications/dsi-2)
- [VESA DSC](https://vesa.org/vesa-display-compression-codecs/dsc/)
- [LiteVideo](https://github.com/litex-hub/litevideo)
- [Antmicro Video Overlays](https://github.com/antmicro/video-overlays)
- [K3D Procedural Visual Specification](../../../../vocabulary/PROCEDURAL_VISUAL_SPECIFICATION.md)
