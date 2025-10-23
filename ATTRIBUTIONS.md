# Attributions & Acknowledgments

**Knowledge3D** stands on the shoulders of giants. This document provides proper attribution to the research, projects, and techniques we leverage and adapt. Our contribution lies in the novel synthesis and adaptation of these ideas for sovereign, GPU-native multi-modal AI — not in claiming to have invented the foundational concepts.

---

## Core Philosophy: Adaptation, Not Invention

**What We Did NOT Invent**:
- Reverse Polish Notation (RPN)
- Field of View (FOV) and Level of Detail (LOD) systems
- GPU-native programming
- Spatial indexing
- Multi-modal fusion
- Thinking tags / chain-of-thought
- Text compression techniques

**What We DID Contribute**:
- Adaptation of game industry LOD/FOV for *cognitive workload management*
- RPN as a *neural execution engine* for GPU-native AI reasoning
- Spatial memory consolidation for *semantic knowledge* (not just 3D rendering)
- Dual-texture paradigm for *human-AI cohabitation*
- GPU-batched RLWHF pipeline leveraging *tiny model parallelization*
- Integration of DeepSeek-OCR techniques with *sovereign PTX kernels*

---

## 1. Research Foundations

### 1.1 DeepSeek-OCR: Contexts Optical Compression

**Source**: [DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
**Paper**: DeepSeek-OCR: Efficient Vision-Language Models for Document Understanding

**What We Adapted**:
- Two-stage vision encoder (SAM-base + CLIP-large)
- 16× convolutional compressor for token reduction
- Multi-resolution processing modes
- Text-as-image compression paradigm

**Our Contribution**:
- Mapped DeepSeek architecture to K3D's sovereign PTX stack
- `LocalPerceptionEncoder` (SAM-base equivalent, Phase E: CPU stub, Phase F: PTX kernels)
- `ConvolutionalCompressor` (16× compression with PTX strided conv)
- `GlobalContextEncoder` (CLIP-large equivalent using GalaxyResonanceEngine)
- `MultiResolutionController` (token budget management)
- **Dual-texture GLB folios**: Human (512×512) + AI (256×256) textures on same 3D object

**Credit**: DeepSeek AI Team for pioneering the "contexts optical compression" approach. We honor their research by properly implementing their techniques within K3D's architecture.

---

### 1.2 AI-RLWHF: Reinforced Learning with Honesty and Feedback

**Source**: [AI-RLWHF GitHub](https://github.com/danielcamposramos/AI-RLWHF)
**Author**: Daniel Campos Ramos (K3D Project Lead)

**What We Adapted**:
- Teacher-student evaluation paradigm
- 5-tier reward system (-2 to +2)
- Thinking tag harvesting from teacher models
- Honesty scoring and feedback loops

**Our Contribution**:
- Integration with K3D's TRM (Tiny Recursive Model) architecture
- GPU-batched student attempts (32-128× parallelization)
- Sequential teacher evaluation with context cleaning
- Reward-weighted training for semantic reasoning
- Paradigm shift: Train on *reasoning patterns*, not data storage

**Credit**: This is our own research project, but the RLWHF concept builds on reinforcement learning from human feedback (RLHF) literature.

---

### 1.3 ARC-AGI: Abstraction and Reasoning Corpus

**Source**: [ARC-AGI GitHub](https://github.com/fchollet/ARC-AGI)
**Author**: François Chollet

**What We Use**:
- 1,302 grid transformation tasks for training TRM
- Validation benchmark for abstract reasoning

**Our Contribution**:
- Proved K3D paradigm: Knowledge in embeddings, TRM learns reasoning patterns
- Achieved 62,000× improvement (MSE 274 → 0.004) on ARC tasks
- Validated domain specificity: ARC training ≠ semantic training

**Credit**: François Chollet for creating the ARC-AGI benchmark and advancing the field of AI reasoning.

---

## 2. Game Industry Techniques (Repurposed)

### 2.1 Level of Detail (LOD)

**Original Use**: 3D graphics optimization (reduce polygon count for distant objects)

**Papers/Sources**:
- Luebke, D. et al. (2002). "Level of Detail for 3D Graphics"
- Clark, J. H. (1976). "Hierarchical Geometric Models for Visible Surface Algorithms"

**Our Adaptation**:
- **Cognitive LOD**: Reduce *reasoning precision* for non-salient knowledge, not polygon count
- **Morton Curve Saliency**: Z-order curve traversal for spatial-semantic importance ranking
- **Dynamic LOD in ThinkingTagBridge**: Adjust inference depth based on query context

**Credit**: Game industry pioneers for LOD concepts. We repurposed them for AI workload management.

---

### 2.2 Field of View (FOV)

**Original Use**: Camera frustum culling (only render what's visible)

**Papers/Sources**:
- Sutherland, I. E. (1974). "Ten Unsolved Problems in Computer Graphics"
- Various game engine implementations (Unity, Unreal Engine)

**Our Adaptation**:
- **Cognitive FOV**: Attention mechanism that *focuses reasoning* on relevant knowledge
- **Spatial Memory Queries**: FOV-based retrieval from Galaxy/House 3D embeddings
- **Attention Budget**: Similar to draw call budgets in games, but for neural operations

**Credit**: Real-time graphics community for FOV and frustum culling. We adapted it for attention mechanisms.

---

### 2.3 Spatial Indexing (Octrees, Z-Order Curves)

**Original Use**: 3D scene acceleration structures

**Papers/Sources**:
- Meagher, D. (1982). "Geometric Modeling Using Octree Encoding"
- Morton, G. M. (1966). "A Computer Oriented Geodetic Data Base and a New Technique in File Sequencing"

**Our Adaptation**:
- **Galaxy 3D Embeddings**: Knowledge positioned in 3D space via crystallization
- **Morton Curve Traversal**: Z-order curve for cache-efficient semantic access
- **House/Galaxy/Museum**: Spatial memory hierarchy (RAM → Disk → Cold storage)

**Credit**: Computer graphics researchers for spatial data structures. We applied them to semantic memory.

---

## 3. AI/ML Foundations

### 3.1 Reverse Polish Notation (RPN)

**Original Source**: Jan Łukasiewicz (1920s), Charles Hamblin (1962)

**Papers/Sources**:
- Hamblin, C. L. (1962). "Translation to and from Polish notation"
- HP calculator documentation (1970s)

**Our Adaptation**:
- **RPN as Neural Engine**: Not just postfix notation, but a *GPU-native execution stack*
- **15 Inter-Referrable Stacks**: Parallel execution contexts for batched inference
- **Trigram Embeddings**: Character-level RPN with 128-dim learned representations

**Credit**: Historical computer science for RPN. We transformed it into a neural computation paradigm.

---

### 3.2 Thinking Tags / Chain-of-Thought

**Original Research**:
- Wei, J. et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
- DeepSeek-R1 (2024): Thinking-enabled models with `<think>` tags

**Our Adaptation**:
- **Thinking Tag Harvesting**: Extract reasoning patterns from teacher models (deepseek-r1)
- **ThinkingTagBridge**: GPU-native inference engine with sub-35µs latency
- **ActionBuffer Integration**: Every inference emits 288-byte action for execution

**Credit**: Chain-of-thought researchers and DeepSeek team for thinking-enabled models.

---

### 3.3 Multi-Modal Fusion

**Original Research**:
- Baltrusaitis, T. et al. (2019). "Multimodal Machine Learning: A Survey and Taxonomy"
- ViLBERT, CLIP, Flamingo, and other vision-language models

**Our Adaptation**:
- **AtomicFissionFusion**: Multi-modal fusion with PTX kernels (not transformers)
- **Dual-Texture Paradigm**: Visual encoding for both human and AI clients
- **GraphCrystallizer**: Fuses text/audio/visual into unified 3D Galaxy positions

**Credit**: Multi-modal ML community for vision-language fusion techniques.

---

## 4. Software & Tools

### 4.1 CUDA & PTX

**Source**: NVIDIA Corporation
**Documentation**: [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)

**What We Use**:
- PTX assembly for GPU-native kernels
- ctypes + libcuda.so for zero-dependency execution
- CUDA memory management (malloc, memcpy, etc.)

**Our Contribution**:
- **Sovereign Architecture**: No PyTorch/TensorFlow dependencies
- **Direct PTX Compilation**: Custom kernels for RPN, TRM, fusion
- **Sub-35µs Latency**: Optimized for real-time cognitive inference

**Credit**: NVIDIA for CUDA platform and extensive documentation.

---

### 4.2 Ollama

**Source**: [Ollama GitHub](https://github.com/ollama/ollama)
**Use**: Local LLM inference for question generation and teacher evaluation

**What We Use**:
- exaone3.5 for question generation
- deepseek-r1 for teacher evaluation with thinking tags
- qwen2.5 for alternative teacher evaluation

**Our Contribution**:
- Integration with K3D's RLWHF pipeline
- Sequential processing with context cleaning (`keep_alive=0s`)
- 600s timeout handling for thinking models

**Credit**: Ollama team for making local LLM inference accessible.

---

### 4.3 PyMuPDF (fitz)

**Source**: [PyMuPDF GitHub](https://github.com/pymupdf/PyMuPDF)
**Use**: PDF text extraction (structured documents)

**What We Use**:
- Text block extraction with bounding boxes
- Image extraction from PDF pages
- Multi-modal content parsing

**Our Contribution**:
- Integration with DeepSeek-OCR pipeline
- Dual-texture folio generation from PDFs
- Phase C: 15× speedup (300ms → 20ms per page)

**Credit**: PyMuPDF maintainers for excellent PDF parsing library.

---

### 4.4 Tesseract OCR

**Source**: [Tesseract GitHub](https://github.com/tesseract-ocr/tesseract)
**Use**: Fallback OCR for scanned documents

**What We Use**:
- Text extraction from scanned PDFs (~1% of corpus)
- Temporary bridge until sovereign OCR in Phase E/F

**Our Contribution**:
- Graceful fallback architecture (DeepSeek → Tesseract → Fail)
- Integration with sovereign PTX pipeline

**Credit**: Tesseract community for open-source OCR.

---

## 5. Datasets & Corpora

### 5.1 WordNet

**Source**: [Princeton WordNet](https://wordnet.princeton.edu/)
**License**: WordNet 3.0 License

**What We Use**:
- 117,659 English synsets for lexical knowledge
- Multi-lingual WordNet variants (PT-BR, ES, JP, ZH)

**Our Contribution**:
- RPN-native embeddings (no GloVe bootstrap)
- 3D Galaxy positioning via semantic crystallization
- Parallel ingestion (8 workers, 821 synsets/s)

**Credit**: Princeton Cognitive Science Laboratory for WordNet.

---

### 5.2 Font Libraries

**Sources**: System fonts from `/usr/share/fonts` (Linux)

**What We Use**:
- 2,713 fonts → 168,206 glyph-text pairs
- HOG (Histogram of Oriented Gradients) visual features
- Per-font character renderings (16×16 pixels)

**Our Contribution**:
- Visual-text grounding for OCR
- RPN fusion of visual + semantic features
- Phase B: 750 glyphs/s parallel harvesting

**Credit**: Font designers and open-source font communities.

---

## 6. K3D's Novel Contributions

To clearly delineate our work from prior art:

### 6.1 Architectural Innovations

1. **Sovereign GPU-Native AI Stack**
   - Zero external ML dependencies (no PyTorch/TensorFlow)
   - Direct PTX kernel execution via ctypes
   - Sub-35µs cognitive inference latency

2. **Dual-Texture Paradigm for Human-AI Cohabitation**
   - Same 3D object, two visual languages
   - Human texture: 512×512 (pretty, game-style)
   - AI texture: 256×256 (compressed text-as-image, 7-20×)
   - Both clients see the same knowledge in different encodings

3. **Spatial Memory Consolidation**
   - Knowledge lives in embeddings (Galaxy/House), not model weights
   - TRM learns reasoning patterns, not data
   - Sleep-time clustering (290K trigrams → 256 clusters)

4. **GPU-Batched RLWHF with Tiny Models**
   - Student: 2.1M params, batches 128× in parallel
   - Teacher: 70B+ params, sequential with thinking tags
   - 20-40× speedup on student attempts (Phase E.5)

5. **Game Industry Techniques for AI**
   - LOD for cognitive workload management
   - FOV for attention mechanisms
   - Morton curves for semantic traversal

### 6.2 Integration Contributions

1. **DeepSeek-OCR → PTX Kernels**
   - Mapped SAM-base, 16× conv, CLIP-large to K3D's sovereign stack
   - Phase E: CPU stubs, Phase F: Full PTX implementation

2. **RLWHF for Semantic Reasoning**
   - Train on reasoning patterns (thinking tags), not data
   - Reward-weighted training with 5-tier system
   - Honesty scoring and feedback loops

3. **Multi-Resolution OCR**
   - Tiny/Small/Base/Large/Gundam modes
   - Token budget management
   - 7-20× compression with 97% fidelity

---

## 7. Paper Preparation

This document is structured to support academic publication. Key sections for a paper:

### Abstract Elements
- **Problem**: Current LLMs are large, opaque, and data-dependent
- **Solution**: K3D's sovereign, GPU-native architecture with spatial memory
- **Results**: 2.1M param TRM, 128× GPU efficiency, 7-20× OCR compression
- **Contribution**: Novel synthesis of game industry + ML + sovereign computing

### Related Work
- DeepSeek-OCR (vision-language compression)
- ARC-AGI (abstract reasoning)
- RLHF/RLWHF (reinforcement learning from feedback)
- Multi-modal fusion (CLIP, ViLBERT, Flamingo)
- Spatial indexing (octrees, Morton curves)
- Game industry LOD/FOV systems

### Novel Contributions
- Dual-texture paradigm for human-AI cohabitation
- RPN as neural execution engine
- Spatial memory consolidation (sleep-time clustering)
- GPU-batched RLWHF with tiny models
- Sovereign PTX architecture (zero ML dependencies)

---

## 8. Citation Guidelines

When citing Knowledge3D in academic work:

```bibtex
@software{knowledge3d2025,
  author = {Ramos, Daniel Campos and Contributors},
  title = {Knowledge3D: Sovereign GPU-Native Multi-Modal AI with Spatial Memory},
  year = {2025},
  url = {https://github.com/danielcamposramos/Knowledge3D},
  note = {Open-source spatial AI operating system}
}
```

When citing specific components:

**DeepSeek-OCR Integration**:
```bibtex
@misc{knowledge3d_deepseek2025,
  author = {Ramos, Daniel Campos},
  title = {Adaptation of DeepSeek-OCR for Sovereign PTX Kernels},
  year = {2025},
  howpublished = {Knowledge3D Phase E},
  url = {https://github.com/danielcamposramos/Knowledge3D/blob/main/docs/DEEPSEEK_OCR_INTEGRATION.md}
}
```

**GPU-Batched RLWHF**:
```bibtex
@misc{knowledge3d_rlwhf2025,
  author = {Ramos, Daniel Campos},
  title = {GPU-Batched RLWHF for Tiny Model Parallelization},
  year = {2025},
  howpublished = {Knowledge3D Phase E.5},
  url = {https://github.com/danielcamposramos/Knowledge3D/blob/main/TEMP/CODEX_GPU_BATCHING_ADDENDUM.md}
}
```

---

## 9. Contact & Collaboration

For research collaboration, licensing inquiries, or attribution questions:

- **Project Lead**: Daniel Campos Ramos
- **Repository**: https://github.com/danielcamposramos/Knowledge3D
- **Documentation**: See `docs/Jules_K3D_Whitepaper.md`
- **NotebookLM Research Space**: https://notebooklm.google.com/notebook/1bd10bda-8900-4c41-931e-c9ec67ac865f

---

## 10. License & Legal

Knowledge3D is licensed under Apache 2.0 (see `LICENSE`).

**Upstream Licenses**:
- DeepSeek-OCR: MIT License
- Ollama: MIT License
- PyMuPDF: GNU AGPL v3
- Tesseract OCR: Apache 2.0
- WordNet: WordNet 3.0 License
- CUDA: NVIDIA CUDA EULA

All third-party code and data are used in compliance with their respective licenses.

---

## Acknowledgments

We stand on the shoulders of:
- **NVIDIA** for CUDA/PTX platform
- **DeepSeek AI** for OCR research
- **François Chollet** for ARC-AGI
- **Ollama team** for local LLM inference
- **Game industry pioneers** for LOD/FOV systems
- **Open-source ML community** for foundational research
- **Historical CS giants** for RPN, spatial indexing, and core algorithms

**Thank you** for advancing the field and making your work accessible. K3D would not exist without your contributions.

---

**Last Updated**: October 22, 2025
**Version**: Phase E.5 (DeepSeek-OCR + GPU-Batched RLWHF)
