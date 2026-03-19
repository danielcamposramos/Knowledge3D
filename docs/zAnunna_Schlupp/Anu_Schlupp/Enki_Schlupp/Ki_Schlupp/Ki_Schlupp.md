# Ternary Hardware Strategy Research — PM-KR Discussion

**Date:** 2026-03-19
**Context:** Daniel (Chair), Milton (Co-Chair), Christoph (K3D contributor) discussing strategy after Nvidia networking article.
**Trigger:** Nvidia's $11B/quarter networking division — still binary. Christoph asked how to pre-empt patents, keep architectures open, and get ternary hardware moving.

---

## 1. The Nvidia Context

Nvidia's networking division (Mellanox acquisition, 2020, $7B) now generates **$11 billion per quarter** — $31B full year. NVLink, InfiniBand, Spectrum-X. All binary. This is the infrastructure layer BENEATH the GPU compute layer.

**Daniel's reaction:** "still binary design — I will be concerned when I see ternary hardware outside our control"

**Key insight:** Nvidia is building the plumbing (networking) to lock in the GPU ecosystem. The entire stack — compute + networking + software (CUDA) — is binary and proprietary. K3D's ternary architecture operates on TOP of this binary substrate but the real sovereignty comes when ternary hardware exists natively.

---

## 2. Huawei's Ternary Chip — The Wake-Up Call

**Huawei officially unveiled the world's first ternary logic chip in 2025.**

Specifications:
- Three states: -1, 0, +1
- CNTFET technology (Carbon Nanotube Field-Effect Transistors)
- 40% fewer transistors than binary equivalent
- 60% power reduction
- Energy efficiency: 1.8 TOPS/W under 7nm process (3× traditional binary GPUs)
- Binary GPU: 200W for 30 TFLOPS → Ternary: 75W for same performance
- 3 ternary bits = 27 states vs 3 binary bits = 8 states (3.375× information density)

**Status:** Patent filed, prototype unveiled, but CNTFET technology still immature and far from mass commercialization. This is a PATENT LAND GRAB, not a product yet.

**Critical concern:** Huawei has PATENTED their ternary logic gate design. If ternary hardware becomes the future (and the physics says it will), Huawei could control the foundational patents. This is EXACTLY what Christoph is worried about.

---

## 3. Christoph's Questions — Answered

### Q: How can we best approach this?

**Strategy: Publish ternary chip architectures as defensive prior art via W3C/PM-KR.**

K3D already has ternary computing at the SOFTWARE level:
- Ternary opcodes: TADD(0x70), TMUL(0x71), TNOT(0x72), TCOMP(0x73), TQUANT(0x74), TPACK(0x75), TUNPACK(0x76)
- Balanced ternary throughout the Galaxy Universe
- Ternary video compression (documented as 7 years ahead)

The next step is publishing ternary HARDWARE architectures — instruction set, logic gate arrangements, memory cell designs — as open specifications through PM-KR. This creates prior art that prevents anyone from patenting what K3D has already described.

**Important distinction:** K3D's foundational primitive is not "a gate that adds or subtracts 1 from a ternary value." It is a native three-state relay model:
- `0` = natural rest position
- `+1` = one side of the relay
- `-1` = the other side of the relay

Arithmetic comes after that primitive. If useful for tooling or education, the same states may also be named `0, 1, 2`, but the physical model stays rest-centered.

### Q: We need to talk about ternary chips to get people moving there

**Action:** W3C PM-KR should publish a ternary computing specification that:
1. Defines a ternary instruction set architecture (like RISC-V but ternary)
2. Specifies ternary memory cell designs (balanced ternary, -1/0/+1)
3. Documents ternary-native data paths, ALUs, and register files
4. Maps K3D's existing ternary opcodes to hardware instructions

The hardware language should explicitly say:
- state primitive first
- arithmetic derived second
- no normative dependence on one increment/decrement gate family

This gives chip designers a TARGET to build toward — just as RISC-V gave designers a binary ISA to implement.

### Q: We need to keep the architectures open

**Model: RISC-V's approach, but for ternary.**

RISC-V (binary, open ISA) has captured ~25% of the global processor market by 2026. Key lessons:
- Open ISA maintained by neutral Swiss foundation (not a corporation)
- No licensing fees (vs ARM's multi-million dollar upfront fees)
- Custom instructions allowed (modularity)
- Any foundry can manufacture (no vendor lock-in)

A "RISC-T" (Ternary RISC) specification published through W3C/PM-KR would:
- Be royalty-free (W3C patent policy)
- Be internationally neutral (W3C is not a US/China entity)
- Allow custom extensions (K3D's ternary opcodes as a reference implementation)
- Prevent any single company from owning the ternary ISA

### Q: Is there multiple manufacturing processes available at low enough cost?

**YES. Several paths exist:**

| Path | Cost | Status | Notes |
|------|------|--------|-------|
| **FPGA prototyping** | $100-$10K | Available NOW | Ternary RISC processor already demonstrated on FPGA (Hackaday, March 2026) |
| **Google/SkyWater 130nm** | FREE | Available | Open-source PDK, free fabrication for open-source designs |
| **GlobalFoundries 180nm** | FREE | Available | Open-source PDK for open-source chip innovators |
| **ChipIgnite/Tiny Tapeout** | $300-$10K | Available | Small-batch ASIC fabrication, accessible to anyone |
| **Libre Silicon** | Open | In development | Fully open-source semiconductor manufacturing process |
| **CNTFET (Huawei)** | Unknown | Lab stage | Carbon nanotube — most promising for ternary but immature |
| **IHP open program** | Subsidized | Available | European open-source chip fabrication |

**Key insight from Christoph: "the cheaper approaches may actually win out"** — This is correct. FPGA prototyping and 130nm/180nm open PDKs are available TODAY at near-zero cost. You don't need 7nm to prove ternary computing works. A 130nm ternary chip running K3D's opcodes would be a historic demonstrator.

### Q: How can we pre-empt patents without being in the chip business?

**Three-layer defensive strategy:**

**Layer 1: Defensive Publication (immediate)**
- Publish ternary chip architecture specifications through PM-KR/W3C
- Include: ISA definition, logic gate designs, memory cell layouts, ALU architectures
- Once published, this becomes PRIOR ART — nobody can patent it
- Cost: $0 (just documentation work)
- Timeline: Can be done NOW

**Layer 2: Open Standard via W3C (medium-term)**
- Submit ternary computing specification as W3C Community Group Report
- W3C Royalty-Free Patent Policy means all contributors grant royalty-free licenses
- This is the nuclear option against patent trolls — anything in a W3C spec is RF-licensed
- Timeline: 6-12 months

**Layer 3: Reference Implementation (longer-term)**
- Publish an open-source ternary processor design (HDL/Verilog)
- Fabricate via Google/SkyWater open program (free)
- Physical chip = irrefutable prior art
- Timeline: 12-24 months

### Q: Can we publish in the area of chip design without implementation as a strategy?

**YES. This is exactly what defensive publication is.**

A defensive publication is an IP strategy where you disclose an enabling description of an invention so it enters the public domain as prior art. Once published, nobody can patent it — including you.

**Key facts:**
- Up to 2/3 of patenting organizations actively use defensive publication
- W3C standards are one of the strongest forms of defensive publication (royalty-free patent policy)
- Publication must be "enabling" — detailed enough that someone skilled in the art could implement it
- K3D's ternary opcodes + architecture specs are ALREADY enabling descriptions
- Publishing the HARDWARE mapping of these opcodes = defensive publication for chip designs

**Databases for defensive publication:**
- IP.com Prior Art Database
- Research Disclosure (Questel)
- arXiv.org (free, timestamped, widely indexed)
- W3C Technical Reports (strongest for standards-essential patents)

### Q: What must be open to allow this to happen?

**Three things must be open:**

1. **Instruction Set Architecture (ISA)** — The ternary opcode definitions, register spec, memory model. This is the "RISC-V equivalent" for ternary. K3D already has this at the software level.

2. **Process Design Kit (PDK)** — The manufacturing rules that tell you how to lay out transistors. Google/SkyWater and GlobalFoundries already offer open PDKs for binary. The ternary-specific additions (three-threshold voltage cells, balanced ternary memory) need to be published openly.

3. **EDA Tools** — Electronic Design Automation software to simulate and verify ternary circuits. Open-source EDA exists (OpenROAD, Yosys, KLayout) but needs ternary extensions.

**If all three are open, anyone can design + fabricate ternary chips without paying royalties to anyone.**

---

## 4. Christoph's Economic Insight

> "Everyone involved high up must see how they can make money with the open technology themselves so they have no urge to take money from others."

This is the RISC-V lesson crystallized. RISC-V succeeded because:
- **Chip designers** save millions in licensing fees → they support the open ISA
- **Cloud providers** (Google, Alibaba) get custom silicon without ARM/Intel lock-in → they fund RISC-V
- **Foundries** (TSMC, Samsung, Intel Foundry) get MORE customers → they support open PDKs
- **Startups** can enter the market without upfront licensing → they innovate on top

**For ternary, the economic incentives are even stronger:**
- 60% power reduction = massive data center energy savings
- 40% fewer transistors = cheaper chips
- 3.375× information density = fewer chips needed
- Anyone who builds ternary SAVES money vs binary

**The W3C angle:** If PM-KR publishes the ternary computing specification as a W3C standard:
- Chip manufacturers can build ternary chips royalty-free
- Software companies can target ternary ISA without vendor lock-in
- K3D becomes the reference software stack for ternary hardware
- Daniel/Milton/Christoph's organizations benefit from being the standard-setters (consulting, training, implementation) without needing to manufacture chips

**This is the Red Hat model applied to silicon:** The standard is free. The expertise is valuable.

---

## 5. Immediate W3C Actions (Christoph's "Critical questions")

1. **Publish K3D's ternary opcode specification as a PM-KR Community Group Report**
   - TADD, TMUL, TNOT, TCOMP, TQUANT, TPACK, TUNPACK already defined
   - Add hardware-level ISA mapping (register widths, memory model, I/O)
   - This creates prior art for ternary instruction sets

2. **Draft a "Ternary Computing for Knowledge Representation" W3C Note**
   - Why ternary matters for knowledge systems (true/false/unknown)
   - How ternary reduces AI inference energy by 60%
   - Reference K3D's sovereign ternary pipeline as proof of concept
   - Include chip architecture sketches (defensive publication)

3. **Propose a W3C Community Group for Open Ternary Computing**
   - Broader than PM-KR — invites chip designers, foundries, EDA tool makers
   - Goal: ternary ISA standard (like RISC-V but for three-valued logic)
   - W3C RF patent policy protects everything published

4. **Submit K3D's ternary video compression spec as prior art**
   - Already documented as 7 years ahead of industry
   - Ternary video codec architecture = defensive publication against codec patents

---

## 6. Timeline Recommendation

| When | What | Who |
|------|------|-----|
| Now | Publish ternary opcode spec as PM-KR CG Report | Daniel + Christoph |
| Q2 2026 | Draft "Ternary Computing for KR" W3C Note | PM-KR group |
| Q3 2026 | Propose Open Ternary Computing CG | Daniel (Chair) |
| Q4 2026 | FPGA reference implementation of ternary ISA | Community |
| 2027 | Free fabrication via Google/SkyWater open program | Community |

---

## Naming Grammar

- `*-T` = pure ternary or native ternary end-state
- `*-BT` = hybrid binary + ternary compatibility bridge
- `x*T` = true unified hybrid core or product line

This is the normalization rule used across the companion family below.

---

## Companion Specifications

### RISC Family

- `Anu_Schlupp.md` - `RISC-T`, pure ternary ISA
- `Enki_Schlupp.md` - `RISC-BT`, hybrid binary + ternary RISC chip
- `Enlil_Schlupp.md` - `RISC-xRVT`, one true RISC hybrid core

### x86/x64 Family

- `Antu_Schlupp.md` - `X64-BT`, hybrid binary + ternary overlay on binary x86-64 substrate
- `Ninlil_Schlupp.md` - `x86_x64T`, one true x86/x64 hybrid core
- `Ninhursag_Schlupp.md` - full single-chip SoC layer with sensors, IoT, security, accelerators, and chiplet path

### Network Family

- `Nanna_Schlupp.md` - `Ethernet-BT`, current network hardware + ternary compatibility bridge
- `Utu_Schlupp.md` - `Ethernet-xNetT`, one true hybrid network core
- `Inanna_Schlupp.md` - `Net-T`, full native ternary network fabric

### Display Family

- `Nisaba_Schlupp.md` - `Display-BT`, HDMI/DP/DSI compatibility plus ternary semantic and procedural display extensions
- `Dumuzi_Schlupp.md` - `Display-xDispT`, one true hybrid display controller/monitor core
- `Geshtinanna_Schlupp.md` - `Display-T`, pure ternary display fabric

### Sensor Family

- `Ninurta_Schlupp.md` - `Sensor-BT`, current camera/LiDAR transport plus ternary confidence and procedural extensions
- `Ishkur_Schlupp.md` - `Sensor-xSenseT`, one true hybrid imaging core
- `Nergal_Schlupp.md` - `Sensor-T`, pure ternary camera/LiDAR sensor fabric

### Storage and Memory Family

- `Abzu_Schlupp.md` - `Storage-BT`, NVMe/DDR/CXL/UCIe compatibility plus ternary policy extensions
- `Tiamat_Schlupp.md` - `Storage-xMemT`, one true hybrid platform core for SSD, RAM, and board fabrics
- `Kingu_Schlupp.md` - `Storage-T`, pure ternary memory and storage fabric

### Wireless Family

- `Anshar_Schlupp.md` - `RF-BT`, Wi-Fi/Bluetooth/LoRa/Thread compatibility plus ternary routing/state extensions
- `Kishar_Schlupp.md` - `RF-xRFT`, one true hybrid wireless core
- `Lahmu_Schlupp.md` - `RF-T`, pure ternary wireless fabric

### Community Guidance

- `Lahamu_Schlupp.md` - maker and education path for open ternary experimentation on Arduino-class hardware

---

## Sources

- [Nvidia networking division — TechCrunch](https://techcrunch.com/2026/03/18/nvidia-networking-division-building-a-multibillion-dollar-behemoth-to-rival-its-chips-business/)
- [Huawei ternary logic chip — Meta Quantum](https://meta-quantum.today/?p=7960)
- [Huawei ternary patent — South China Morning Post](https://www.scmp.com/tech/big-tech/article/3305201/tech-war-huaweis-ternary-logic-patent-could-solve-problem-power-hungry-ai-chips)
- [Huawei ternary patent details — Huawei Central](https://www.huaweicentral.com/huawei-patents-ternary-logic-to-develop-energy-efficient-ai-chips/)
- [Carbon nanotube ternary circuits — Science Advances](https://www.science.org/doi/10.1126/sciadv.adt1909)
- [Ternary RISC processor on FPGA — Hackaday](https://hackaday.com/2026/03/16/ternary-risc-processor-achieves-non-binary-computing-via-fpga/)
- [RISC-V Silicon Sovereignty — Financial Content](https://www.financialcontent.com/article/tokenring-2026-1-8-silicon-sovereignty-how-risc-vs-open-source-revolution-is-dismantling-the-arm-and-x86-duopoly)
- [RISC-V open architecture — IEEE Spectrum](https://spectrum.ieee.org/riscvs-opensource-architecture-shakes-up-chip-design)
- [Google open-source chip fabrication — The Register](https://www.theregister.com/2020/07/03/open_chip_hardware/)
- [Libre Silicon](https://libresilicon.com/)
- [CHIPS Alliance](https://www.chipsalliance.org/)
- [Open-source chips for Europe](https://open-source-chips.eu/)
- [Defensive publication — Wikipedia](https://en.wikipedia.org/wiki/Defensive_publication)
- [Defensive publication strategy — PatentPC](https://patentpc.com/blog/how-to-conduct-a-defensive-publication-to-prevent-patent-infringement)
- [Open source hardware prior art — Federation of American Scientists](https://fas.org/publication/open-source-hardware-uspto/)
- [Ternary computing overview — ternary-computing.com](https://www.ternary-computing.com/)
