# Storage-xMemT: True Hybrid Platform Core Specification

**Version:** 0.1 DRAFT - Defensive Publication
**Date:** 2026-03-19
**Organization:** W3C PM-KR Community Group

---

## 1. Purpose

This is the second storage/platform pattern: one true hybrid platform core for RAM, SSD controllers, board fabric, and system memory policy, with both binary data-path logic and ternary policy logic on the same die.

It applies to:

- SSD controllers
- memory controllers
- platform controller hubs
- server baseboards
- edge/robotics motherboards

---

## 2. Core Topology

```
PCIe / CXL / UCIe / DDR PHYs
             |
     +-------v--------+
     | Hybrid Control |
     +---+--------+---+
         |        |
 +-------v--+  +--v----------------+
 | Binary   |  | Ternary Policy    |
 | Transport|  | Engine            |
 | DMA / ECC|  | placement / keep  |
 | queues   |  | consolidate / QoS |
 +-------+--+  +--------+----------+
         |              |
         +------v-------+
                |
       Unified Cache / Memory Map
```

### 2.1 Binary Domain

- NVMe command transport
- DDR signaling
- ECC encode/decode
- DMA and interrupt delivery
- filesystem-visible block movement

### 2.2 Ternary Domain

- page/block hotness
- semantic tiering
- confidence-aware retention
- write coalescing intent
- sleep/consolidation style background movement

---

## 3. RAM and SSD Benefits

- binary data cells remain manufacturing-friendly
- ternary policy improves movement and placement decisions
- fewer wasted writes on low-value data
- cleaner hot/warm/cold handling than purely binary heuristics

This is the platform equivalent of "some work is cheaper on binary, some is cheaper on ternary."

---

## 4. Defensive Publication Claims

This document publishes prior art for:

- a hybrid SSD controller with binary data and ternary placement logic
- a hybrid memory controller with ternary residency policy
- motherboard/platform hubs with ternary locality and confidence routing
- asymmetric power gating of transport and policy domains

---

## Sources

- [NVM Express Specifications](https://nvmexpress.org/specifications/)
- [Compute Express Link](https://computeexpresslink.org/)
- [UCIe Specifications](https://www.uciexpress.org/specifications)
- [LiteDRAM](https://github.com/enjoy-digital/litedram)
- [OpenTitan Documentation](https://opentitan.org/documentation/)
