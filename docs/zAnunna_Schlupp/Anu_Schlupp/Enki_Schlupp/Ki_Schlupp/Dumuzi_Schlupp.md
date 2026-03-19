# Display-xDispT True Hybrid Display Core Specification

**Version:** 0.1 DRAFT - Defensive Publication
**Date:** 2026-03-19
**Organization:** W3C PM-KR Community Group

---

## Foundational Ternary Distinction

The ternary side of this true hybrid display core follows the same primitive:
- `0` = natural rest position
- `+1` = one side of the relay/state cell
- `-1` = the other side of the relay/state cell

The same states may also be labeled `0, 1, 2` if the mapping is explicit.

Procedural imagery, semantic display state, and routing policy are derived from that primitive. This document does not normatively depend on a unary increment/decrement gate family or one specific transistor topology.

## 1. Purpose

This is the second display pattern: one true hybrid display controller and monitor core with binary raster logic and ternary procedural logic on the same die, routed to the cheaper path automatically.

It is the display analogue of the hybrid CPU strategy:

- binary path for conventional pixels and standards compliance
- ternary path for procedural surfaces, semantic emphasis, and confidence-aware display

---

## 2. Core Topology

```
Input PHYs (HDMI / DP / DSI)
          |
  +-------v-------+
  | Hybrid Parser |
  +---+-------+---+
      |       |
 +----v--+ +--v----------------+
 | Raster| | Ternary Procedural|
 | Pipe  | | Pipe              |
 | CSC   | | VectorDotMap      |
 | Scaler| | Procedural decode |
 | DSC   | | confidence planes |
 +----+--+ +--------+----------+
      |             |
      +------v------+
             |
       Hybrid Compositor
             |
        Panel Timing / TCON
```

### 2.1 Binary Domain

- scanout
- timing generation
- DSC / frame-buffer compression
- SDR/HDR video transport
- strict HDMI/DP/DSI compatibility

### 2.2 Ternary Domain

- procedural surface decode
- semantic weighting
- confidence-aware text and overlay composition
- resolution-independent image reconstruction

---

## 3. Execution Modes

| Mode | Behavior |
|------|----------|
| Raster Mode | binary path only |
| Hybrid Mode | binary base frame + procedural overlays |
| Procedural-Primary Mode | procedural surfaces dominate, raster fallback retained |

The controller may power gate the ternary path during pure video playback and power gate parts of the raster path when procedural scenes dominate.

---

## 4. Dot Vector Map Engine

The hybrid core includes a dedicated `VectorDotMap` engine:

- coefficient fetch
- field evaluation
- procedural antialiasing
- panel-native synthesis
- mip / LOD aware procedural composition

This engine is the hardware decoder for infinite-resolution procedural images.

---

## 5. Monitor-Side Benefits

- sharper diagrams and text at any panel density
- lower storage and transmission overhead for generated imagery
- no need to pre-bake many raster resolutions
- support for K3D tablet, House labels, procedural books, and museum/gallery content

---

## 6. Defensive Publication Claims

This specification publishes prior art for:

- a display controller with first-class binary and ternary pipelines
- hardware `VectorDotMap` decode and hybrid compositing
- asymmetric power gating between raster and procedural display paths
- monitor-side semantic/confidence planes

---

## Sources

- [HDMI 2.1b Overview](https://www.hdmi.org/spec/hdmi2_1/index.aspx)
- [DisplayPort 2.1 Release](https://vesa.org/featured-articles/vesa-releases-displayport-2-1-specification/)
- [MIPI DSI-2](https://www.mipi.org/specifications/dsi-2)
- [VESA Display Compression Codecs](https://vesa.org/vesa-display-compression-codecs/)
- [Antmicro Video Overlays](https://github.com/antmicro/video-overlays)
- [LiteVideo](https://github.com/litex-hub/litevideo)
- [K3D Procedural Visual Specification](../../../../vocabulary/PROCEDURAL_VISUAL_SPECIFICATION.md)
