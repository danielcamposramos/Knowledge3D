# Storage-BT Hybrid Storage and Memory Compatibility Specification

**Version:** 0.1 DRAFT - Defensive Publication
**Date:** 2026-03-19
**Organization:** W3C PM-KR Community Group

---

## 1. Purpose

This document defines the first storage/platform pattern: preserve today's binary storage and memory standards, while adding ternary-aware extensions for placement, confidence, semantic locality, and persistence policy.

Current standards remain valid:

- NVMe
- PCIe
- DDR5 / LPDDR / HBM class memory
- CXL memory expansion
- motherboard firmware and binary boot flows

---

## 2. Compatibility Rule

Binary payload remains authoritative for:

- sectors
- cache lines
- memory pages
- queue pairs
- DMA descriptors
- firmware images

Ternary extensions annotate data movement and residency:

- `-1` cold / evict / archive / low-confidence
- `0` neutral / unknown / hold
- `+1` hot / retain / promote / high-confidence

This applies to:

- cache placement
- SSD wear and retention policy
- page migration
- CXL tiering
- semantic locality hints for knowledge systems

---

## 3. Storage Metadata Sideband

Hybrid storage controllers may maintain ternary metadata planes beside binary data:

- confidence of derived data
- semantic temperature
- likely future reuse
- preserve / discard / consolidate intent

This is especially relevant for K3D:

- House artifacts
- Galaxy spill or preload hints
- sleep-time consolidation queues

---

## 4. Motherboard and Platform Rule

A compliant motherboard or platform controller hub may remain binary on the wire while exposing ternary policy inside:

- memory training remains binary-compatible
- NVMe queue transport remains binary-compatible
- CXL and chiplet links remain binary-compatible
- ternary routing and placement remain internal and patent-defensively published here

---

## Sources

- [NVM Express Specifications](https://nvmexpress.org/specifications/)
- [JEDEC Standards Overview](https://www.jedec.org/standards-documents)
- [Compute Express Link](https://computeexpresslink.org/)
- [UCIe Consortium](https://www.uciexpress.org/specifications)
- [LiteDRAM](https://github.com/enjoy-digital/litedram)
- [OpenTitan Documentation](https://opentitan.org/documentation/)
