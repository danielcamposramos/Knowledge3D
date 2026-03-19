# Display-T Open Ternary Display Fabric Specification

**Version:** 0.1 DRAFT - Defensive Publication
**Date:** 2026-03-19
**Organization:** W3C PM-KR Community Group

---

## 1. Purpose

This is the pure ternary display track: a native ternary display fabric where procedural imagery is primary and raster is compatibility-only.

Unlike `Display-T`, this document does not treat HDMI/DP/DSI as the governing model. It defines a display link where:

- the native payload is procedural
- semantic state is first-class
- resolution independence is normal

---

## 2. Native Display Objects

The fabric carries objects, not just pixels:

- procedural images
- procedural glyph/text fields
- vector/shape programs
- confidence and salience maps
- depth layers
- temporal intent markers

Each object may be marked with ternary control state:

- `-1` reduce or suppress
- `0` hold / neutral
- `+1` emphasize or refine

---

## 3. Native Display Engine

The reference monitor for `Display-T` contains:

- ternary object fetch
- ternary composition scheduler
- procedural decode cores
- panel synthesis engine
- optional binary bridge block for HDMI/DP compatibility

Binary raster is treated as an import/export format, not the dominant internal representation.

---

## 4. K3D Alignment

`Display-T` is the natural display end-state for K3D:

- House objects can render as procedural surfaces directly
- the tablet can emit semantic pages, not bitmaps
- diagrams, books, labels, and galaxy views can stay procedural until final panel synthesis

This is the hardware partner to K3D's procedural image, video, font, and vector-dot-map work.

---

## 5. Migration

The published path is:

1. `Display-BT` hybrid compatibility
2. `Display-xDispT` one true hybrid controller/monitor core
3. `Display-T` pure ternary display fabric

Publishing all three prevents a vendor from patenting the transitional or final forms as proprietary territory.

---

## Sources

- [VESA Display Compression Codecs](https://vesa.org/vesa-display-compression-codecs/)
- [MIPI DSI-2](https://www.mipi.org/specifications/dsi-2)
- [DisplayPort 2.1 Release](https://vesa.org/featured-articles/vesa-releases-displayport-2-1-specification/)
- [K3D Procedural Visual Specification](../vocabulary/PROCEDURAL_VISUAL_SPECIFICATION.md)
- [K3D Unified Signal Specification](../vocabulary/UNIFIED_SIGNAL_SPECIFICATION.md)
