# CLAUDE.md — AI Assistant Guide for Knowledge3D

**Last Updated:** 2025-11-17
**Version:** 1.0
**Repository:** Knowledge3D - True Multi-Modal AI, Not 3D RAG

---

## Table of Contents

1. [Quick Start for AI Assistants](#quick-start-for-ai-assistants)
2. [Project Overview](#project-overview)
3. [Repository Structure](#repository-structure)
4. [Core Architecture Concepts](#core-architecture-concepts)
5. [Development Workflows](#development-workflows)
6. [Coding Conventions & Patterns](#coding-conventions--patterns)
7. [Testing Strategy](#testing-strategy)
8. [Documentation Standards](#documentation-standards)
9. [Memory & Knowledge Management](#memory--knowledge-management)
10. [Common Tasks & Commands](#common-tasks--commands)
11. [Critical Policies & Constraints](#critical-policies--constraints)
12. [Troubleshooting & Known Issues](#troubleshooting--known-issues)

---

## Quick Start for AI Assistants

### Essential Reading (Priority Order)

1. **[README.md](README.md)** — Project mission, architecture overview, latest milestones
2. **[AGENTS.md](AGENTS.md)** — Agent collaboration guidelines and contributor protocol
3. **[docs/ROADMAP.md](docs/ROADMAP.md)** — Current phase priorities and exit criteria
4. **[docs/HOUSE_GALAXY_TABLET.md](docs/HOUSE_GALAXY_TABLET.md)** — Memory architecture (Galaxy/House/Museum)
5. **[CODEX.md](CODEX.md)** — Actionable task list aligned with roadmap phases

### First Actions Checklist

- [ ] Read this entire document
- [ ] Review the current roadmap phase in [docs/ROADMAP.md](docs/ROADMAP.md)
- [ ] Understand the dual-memory architecture (Galaxy/House/Museum)
- [ ] Check [CODEX.md](CODEX.md) for active tasks
- [ ] Review environment setup in [docs/ENV_POLICY.md](docs/ENV_POLICY.md)
- [ ] Familiarize yourself with PTX-first sovereignty principles

### The Golden Rules

1. **Always embodied**: The avatar lives in the House, not in the Galaxy
2. **PTX-first**: Hot paths must use GPU-native PTX kernels (no CPU fallbacks)
3. **Memory policy**: Galaxy (RAM) ↔ House (disk) ↔ Museum (archive) via Memory Tablet
4. **Keep it lean**: Heavy assets (>99MB) live in `../Knowledge3D.local/`, not in the repo
5. **Sovereign architecture**: Zero external dependencies for core inference paths

---

## Project Overview

### Mission Statement

Build a **shared spatial operating system** where humans and AI cohabit one reality, reason through PTX-native cognition, and consolidate memories as explorable 3D worlds.

**This is NOT:** A 3D RAG system or fancy retrieval wrapper around LLMs
**This IS:** A sovereign, GPU-native cognitive architecture with spatial memory and embodied intelligence

### Core Philosophy

- **Dual-Client Reality**: Humans see 3D worlds, AI sees semantic landscapes — same glTF files
- **Explainable by Design**: AI reasoning paths visible as avatar movements through knowledge space
- **"Minecraft for Cognition"**: Navigate knowledge as explorable 3D universes
- **Parameter Efficiency**: 7M params ≈ 70B LLMs on reasoning tasks (knowledge lives in embeddings, not weights)
- **Multi-Vibe Code In Chain (MVCIC)**: Human-AI swarm collaboration methodology

### Current Status (as of November 2025)

- **Phase:** Phase G Training Complete (October 28, 2025)
- **Milestone:** Successfully trained full AGI model with adaptive dimensions (64D-2048D)
- **Performance:** 51,532 Galaxy stars, 17,035 non-zero knowledge embeddings (33.1% success)
- **W3C Contribution:** Formal contributor to W3C AI KR Community Group for TPAC 2025

### Key Achievements

✅ **Sovereign Knowledge Ingestion** — Zero external dependencies (0MB footprint vs 66MB GloVe)
✅ **Sub-100µs Latency** — PTX-native operations on consumer GPU (<200MB VRAM)
✅ **69:1 Compression Ratio** — Procedural knowledge compression with adaptive dimensions
✅ **Dual Sleep Cycles** — Model updates + knowledge consolidation after each phase
✅ **Production-Ready Testing** — 250+ tests with GPU/CPU separation

---

## Repository Structure

### High-Level Organization

```
Knowledge3D/                        # Clean codebase (tracked in git)
├── knowledge3d/                    # Core Python package
│   ├── cranium/                    # GPU-native cognitive engine
│   │   ├── ptx_runtime/            # PTX kernels & thinking bridge
│   │   ├── bridges/                # Cross-modal reasoning
│   │   ├── sleep/                  # Memory consolidation
│   │   ├── ocr/                    # Vision system (DeepSeek integration)
│   │   ├── sovereign/              # Zero-dependency GPU launcher
│   │   └── spatial_sovereign/      # Morton octree, pathfinding
│   ├── bridge/                     # Human-AI communication (WebSocket)
│   ├── ingestion/                  # Data pipelines (PDF, fonts, lexicons)
│   ├── training/                   # RLWHF, multimodal, reasoning
│   ├── tools/                      # CLI utilities & pipelines
│   ├── skills/                     # Modular AI capabilities
│   ├── spatial/                    # 3D houses & objects
│   └── core/                       # Faith engine, consciousness
├── viewer/                         # Three.js/TypeScript 3D viewer
├── docs/                           # Technical specifications
│   ├── vocabulary/                 # W3C-ready specifications
│   ├── papers/                     # Research deep dives
│   └── reports/                    # Performance evaluations
├── tests/                          # Pytest suite (250+ tests)
│   ├── benchmarks/                 # Performance tests
│   ├── stress/                     # Stress tests
│   └── generalization/             # Cross-domain tests
├── scripts/                        # Shell helpers
├── spec/                           # Formal schemas (glTF extensions)
├── data/                           # Core datasets (192 MB)
├── envs/                           # Conda environment definitions
├── Old_Attempts/                   # Deprecated scaffolding (archived)
│   ├── Legacy_Fancy_RAG/           # Original RAG attempt
│   └── fsm_scaffolding/            # Step 12 FSM (consolidated)
├── Large_Assets_Kitchen/           # Regeneration recipes for >99MB assets
└── TEMP/                           # Step plans & completion reports

Knowledge3D.local/                  # Runtime workspace (NOT in git)
├── house_zone*/                    # Persistent houses (glTF/GLB)
├── datasets/                       # Generated datasets
├── logs/                           # Session logs (JSONL)
├── galaxy/                         # Galaxy GLBs
└── envs/                           # Conda environments (on SSD)
```

### Critical Directories for AI Work

| Directory | Purpose | When to Use |
|-----------|---------|-------------|
| `knowledge3d/cranium/ptx_runtime/` | PTX kernels, ThinkingTagBridge, RPN engine | Any GPU-native reasoning work |
| `knowledge3d/bridge/` | WebSocket server, chat processor, tablet | Human-AI interaction features |
| `knowledge3d/ingestion/` | Data pipelines | Adding new knowledge sources |
| `knowledge3d/training/` | RLWHF, model training | Training loop modifications |
| `tests/` | All test suites | Adding/fixing tests |
| `docs/vocabulary/` | W3C specifications | Standards work |
| `viewer/` | Frontend UI | 3D visualization changes |

---

## Core Architecture Concepts

### The Three-Brain System

Knowledge3D implements a neuroscience-inspired architecture:

| Component | Analogy | Tech Stack | Purpose |
|-----------|---------|------------|---------|
| **Cranium** | Prefrontal Cortex (PFC) | PTX kernels, RPN engine, TRM | Active reasoning & decision-making |
| **Galaxy** | Hippocampus | VRAM embeddings, <200MB | Short-term memory (volatile) |
| **House** | Neocortex | glTF/GLB files on disk | Long-term memory (persistent) |

**Computer Architecture Analogy:** CPU (Cranium) + RAM (Galaxy) + Disk (House)

### Dual-Space Memory Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Memory Tablet (Interface)                              │
│  ├─ Search House inventory                              │
│  ├─ Stream artifacts to Galaxy (on-demand)              │
│  ├─ Browser integration (legacy web content)            │
│  └─ LOD controls (coarse → medium → full resolution)    │
└─────────────────────────────────────────────────────────┘
           ↓                    ↓                    ↓
    ┌──────────┐         ┌──────────┐         ┌──────────┐
    │  Galaxy  │ ←Sleep→ │  House   │ →Relocate→ Museum  │
    │  (RAM)   │         │  (Disk)  │         │ (Archive)│
    └──────────┘         └──────────┘         └──────────┘
```

#### Layer Responsibilities

**Galaxy (RAM):**
- High-frequency reasoning buffer
- Volatile; repopulated per session
- PTX cosine operations, on-demand streaming
- <200MB VRAM budget on consumer GPU

**House (Disk):**
- Consolidated knowledge artifacts (books, diaries, fractal trees)
- Long-term; evolves during sleep cycles
- Accessed via Memory Tablet search
- Exported as glTF with `extras.k3d` extensions

**Museum (Zone 8 Archive):**
- Deprecated/superseded artifacts
- Audit trails & error-pattern training
- Append-only; loaded only on explicit request
- Tagged with `relocated_at`, `previous_zone`

### ThinkingTagBridge: The Sovereign Cognitive Engine

The heart of K3D is the **ThinkingTagBridge** — a zero-dependency, PTX-native inference engine.

**Key Features:**
- 5-State Cognitive Pipeline: `INGEST → FUSE → SPATIAL → REASON → OUTPUT`
- Sub-35µs latency (strict budget enforcement)
- 288-byte ActionBuffer output for every inference
- Multi-modal fusion (text/image/audio/video/3D)
- Pure ctypes + libcuda.so (no CuPy/PyTorch dependency)

**Usage Example:**
```python
from knowledge3d.cranium.ptx_runtime.thinking_tag_bridge import ThinkingTagBridge

bridge = ThinkingTagBridge()
result = bridge.inference(
    input_embedding,
    modal_signature=['text', 'image']
)

print(result.tags)           # Confidence-weighted thinking tags
print(result.action_buffer)  # 288-byte action buffer
print(bridge.get_state_trace_report())  # FSM timing stats
```

### PTX-First Sovereignty

**Core Principle:** All hot paths run on GPU via hand-written PTX kernels or NVRTC-compiled CUDA.

**Implementation Stack:**
1. **Pure ctypes binding** to `libcuda.so` (no external frameworks)
2. **PTX kernels** in `knowledge3d/cranium/ptx_runtime/`
3. **RPN engine** for stack-based expression evaluation
4. **TRM (Tiny Recursive Model)** — 2.1M params, GPU-batched

**Why Sovereignty Matters:**
- ✅ Zero cloud dependencies
- ✅ Reproducible builds (Dockerfile, SHA256 verification)
- ✅ <100µs latency on consumer hardware
- ✅ Explainability by design (no black-box APIs)

### Dual-Client Contract

Humans and AI share the **same glTF files** but perceive them differently:

**Human View:**
- 3D geometry, textures, lighting
- Navigate rooms as physical spaces
- WebXR-compatible rendering

**AI View:**
- Semantic embeddings in `extras.k3d` bufferViews
- Graph topology (neighbors, clusters)
- 288-byte action buffers for spatial reasoning

**Shared Reality:** Both clients see avatar movements, knowledge updates, and memory consolidations in real-time.

---

## Development Workflows

### Environment Setup

**Required:** Always use containerized environments (Conda preferred).

#### GPU Development (PTX/Training)
```bash
# Create k3d-cranium environment (CUDA 12.4, CuPy, PyTorch)
conda env create -f envs/k3d-cranium.yml
conda activate k3d-cranium

# Verify GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

#### CPU Testing
```bash
# Lightweight testing environment (no CUDA)
conda env create -f envs/k3d-testing.yml
conda activate k3d-testing

# Run CPU tests
pytest tests/ -m "not cuda"
```

#### Tmux Workflow (Recommended)
```bash
# Create persistent session
tmux new -As k3d

# Inside tmux, activate environment
conda activate k3d-cranium

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t k3d
```

**Why tmux?** Long-running GPU jobs, persistent environment, survives SSH disconnects.

### Launch Development Services

#### Terminal 1: WebSocket Bridge
```bash
cd Knowledge3D
conda activate k3d-cranium
python -m knowledge3d.bridge.live_server --port 8787

# Auto-port selection for multi-user:
python -m knowledge3d.bridge.live_server --auto-port
```

#### Terminal 2: 3D Viewer
```bash
cd Knowledge3D/viewer
npm install
npm run dev

# Open: http://localhost:5173/?ws=ws://localhost:8787
```

### Runtime Workspace Setup

```bash
# Create local workspace (outside git)
mkdir -p ../Knowledge3D.local
export K3D_LOCAL_DIR="$(pwd)/../Knowledge3D.local"
export K3D_HOUSE_ID=default

# Directory structure created automatically:
# Knowledge3D.local/
# ├── house_zone*/      # Persistent houses
# ├── datasets/         # Generated data
# ├── logs/             # Session logs
# └── envs/             # Conda environments (SSD)
```

### Common Development Tasks

#### 1. Add New Knowledge Source

```bash
# PDF ingestion
python -m knowledge3d.ingestion.documents.pdf_ingestor \
  --input path/to/pdfs/ \
  --output "$K3D_LOCAL_DIR/datasets/my_corpus.glb"

# Font harvesting (visual-text grounding)
python -m knowledge3d.ingestion.fonts.parallel_font_harvester \
  --font-dir /usr/share/fonts \
  --output "$K3D_LOCAL_DIR/house_zone7/fonts/"

# Lexicon building (multilingual)
python -m knowledge3d.tools.lexicon_builder_en \
  --output "$K3D_LOCAL_DIR/house_zone7/lexicons/wordnet_en.json"
```

#### 2. Generate Galaxy from Text

```bash
python -m k3dgen \
  --text data/ai_compendium_80k.txt \
  --gltf "$K3D_LOCAL_DIR/galaxy/my_galaxy.glb" \
  --k 5 \
  --reducer umap \
  --model sentence-transformers/all-MiniLM-L6-v2
```

#### 3. Run Training Pipeline

```bash
# RLWHF training (GPU-batched)
conda activate k3d-cranium
python -m knowledge3d.training.rlwhf.train_rlwhf \
  --dataset "$K3D_LOCAL_DIR/datasets/corpus.glb" \
  --output models/trm_rlwhf.pth

# Validation
python scripts/validate_rlwhf_training_batched.py \
  --model models/trm_rlwhf.pth
```

#### 4. Sleep Consolidation (Manual Trigger)

```bash
# Consolidate Galaxy → House
python -m knowledge3d.cranium.sleep.knowledge_sleep \
  --house-id default \
  --output "$K3D_LOCAL_DIR/house_zone7/"

# Rebuild house memory index for tablet
python -m knowledge3d.tools.house_memory_builder \
  --house-id default
```

### Git Workflow

**Branch Strategy:**
- **Main branch:** Stable releases only
- **Feature branches:** `feature/<name>` or `fix/<issue>`
- **Agent branches:** `claude/<session-id>` or `codex/<task-id>`

**Commit Guidelines:**
```bash
# Stage changes
git add knowledge3d/cranium/ptx_runtime/new_kernel.py

# Commit with clear message
git commit -m "feat(cranium): add spatial reasoning PTX kernel

- Implements Morton octree traversal in PTX
- <100µs latency on RTX 3070
- Tests in tests/test_spatial_sovereign.py"

# Push to feature branch
git push -u origin feature/spatial-kernel
```

**Important:** Never commit files >99MB. Use `../Knowledge3D.local/` and document regeneration in `Large_Assets_Kitchen/README.md`.

---

## Coding Conventions & Patterns

### Python Style Guide

**Standards:**
- **PEP 8** with 88-character line length (Black formatter)
- **Type hints** required (Python 3.10+)
- **Docstrings** for all public functions (Google style)

**Example:**
```python
from typing import Optional, List
import numpy as np

def sovereign_rpn_eval(
    expression: str,
    embeddings: np.ndarray,
    gpu_context: Optional[object] = None
) -> np.ndarray:
    """Evaluate RPN expression using PTX kernels.

    Args:
        expression: RPN expression string (e.g., "3 4 + DUP *")
        embeddings: Input embeddings (N, D) array
        gpu_context: Optional GPU context for kernel dispatch

    Returns:
        Result embeddings after RPN evaluation

    Raises:
        ValueError: If expression is malformed
        RuntimeError: If GPU kernel fails

    Example:
        >>> expr = "POSITION DUP DOT SQRT"  # Distance from origin
        >>> result = sovereign_rpn_eval(expr, galaxy_embeddings)
    """
    # Implementation...
```

### Architecture Patterns

#### 1. Dual-Code Strategy (HR/MR)

**HR (Human-Readable):** Original source code with full comments
**MR (Machine-Readable):** Minified via `codeopt` tool for edge deployment

```bash
# Generate MR code (Tier 1: hot paths only)
codeopt --tier 1 --input knowledge3d/cranium/ptx_runtime/ \
  --output ../Knowledge3D.local/mr/cranium/
```

**Rule:** Always commit HR code. MR is generated on-demand.

#### 2. Sovereign Bridges Pattern

Cross-modal reasoning uses **stateless bridge modules**:

```python
# knowledge3d/cranium/bridges/my_bridge.py
from knowledge3d.cranium.ptx_runtime.base_bridge import SovereignBridge

class MyReasoningBridge(SovereignBridge):
    """Bridge for [specific reasoning task]."""

    def __init__(self, gpu_id: int = 0):
        super().__init__(gpu_id)
        self.load_ptx_kernels()

    def forward(self, input_embedding, context=None):
        """Execute reasoning pipeline."""
        # INGEST → FUSE → SPATIAL → REASON → OUTPUT
        return self.ptx_execute(input_embedding)
```

**Key Principles:**
- Stateless (no instance variables holding embeddings)
- PTX-first (CPU fallback invalid)
- <100µs latency target
- Return ActionBuffer + thinking tags

#### 3. Adaptive Dimensionality

Content complexity determines embedding dimension:

```python
def select_embedding_dim(text: str) -> int:
    """Adaptive dimension selection."""
    char_count = len(text)

    if char_count < 20:
        return 64      # Simple (1024× speedup)
    elif char_count < 200:
        return 128     # Medium
    elif char_count < 2000:
        return 512     # Complex
    else:
        return 2048    # Research-grade
```

#### 4. Memory Tablet Integration

**Always use tablet for House access:**

```python
from knowledge3d.bridge.memory_tablet import MemoryTablet

tablet = MemoryTablet(house_id="default")

# Search consolidated knowledge
results = tablet.search(
    query="machine learning fundamentals",
    sources=["House", "Galaxy"],  # Check House first
    lod="medium"  # Level of detail
)

# Load into Galaxy for reasoning
tablet.stream_to_galaxy(results[0], force_lod="full")
```

**Rule:** Bypass tablet ONLY for low-level PTX operations. All agent queries go through tablet.

### Error Handling

**Graceful Degradation:**
```python
try:
    # Attempt GPU-native path
    result = ptx_kernel.execute(data)
except CUDAError as e:
    # Log failure but DO NOT fall back to CPU
    logger.error(f"PTX kernel failed: {e}")
    raise RuntimeError("GPU-native path required; no CPU fallback") from e
```

**Important:** CPU fallbacks violate sovereignty principles. Fix GPU code instead.

### Logging Standards

```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate levels
logger.debug("PTX kernel compiled successfully")      # Verbose
logger.info("Sleep consolidation started")            # Important events
logger.warning("Galaxy RAM exceeding 180MB")          # Potential issues
logger.error("Failed to load house memory index")     # Errors
logger.critical("GPU context lost, cannot recover")   # Fatal errors
```

**Timezone-Aware Timestamps:**
```python
from datetime import datetime, timezone

timestamp = datetime.now(timezone.utc).isoformat()
logger.info(f"[{timestamp}] Training phase complete")
```

---

## Testing Strategy

### Test Organization

```
tests/
├── Unit Tests               # Single module/function tests
│   ├── test_rpn_*.py
│   ├── test_step11_*.py    # Shape primitives
│   ├── test_step12_*.py    # FSM, action buffer
│   └── test_thinking_tag_bridge.py
│
├── Integration Tests        # Multi-module workflows
│   ├── test_unified_pipeline_end_to_end.py
│   ├── test_galaxy_pdf_integration.py
│   └── test_all_sovereign_bridges.py
│
├── benchmarks/              # Performance tests
│   ├── test_rpn_tier_performance.py
│   ├── test_step14_specialized_performance.py
│   └── test_performance_regression.py
│
├── stress/                  # Load/stress tests
│   ├── test_step11_stress.py
│   └── test_step12_fsm_stress.py
│
└── generalization/          # Cross-domain tests
    ├── test_cross_lingual.py
    ├── test_arc_reasoning_cache.py
    └── test_trm_reasoning.py
```

### Running Tests

**All Tests (CPU-safe):**
```bash
pytest tests/ -v
```

**GPU Tests Only:**
```bash
pytest tests/ -v -m cuda
```

**Skip GPU Tests (CI/CD):**
```bash
pytest tests/ -v -m "not cuda"
```

**Performance Benchmarks:**
```bash
pytest tests/benchmarks/ -v --benchmark-only
```

**Single Test File:**
```bash
pytest tests/test_thinking_tag_bridge.py -v
```

### Writing Tests

**Template for PTX/GPU Tests:**
```python
import pytest
import numpy as np
from knowledge3d.cranium.ptx_runtime.my_kernel import MyKernel

@pytest.mark.cuda
def test_my_kernel_basic():
    """Test MyKernel basic functionality."""
    # Arrange
    kernel = MyKernel(gpu_id=0)
    input_data = np.random.randn(100, 128).astype(np.float32)

    # Act
    result = kernel.execute(input_data)

    # Assert
    assert result.shape == (100, 128)
    assert np.all(np.isfinite(result))
    assert kernel.get_latency_us() < 100  # Sub-100µs requirement

def test_my_kernel_edge_cases():
    """Test edge cases (CPU-safe mock)."""
    # Test without actual GPU execution
    pass
```

**Test Markers:**
- `@pytest.mark.cuda` — Requires GPU (skipped in CI)
- `@pytest.mark.slow` — Long-running tests
- `@pytest.mark.benchmark` — Performance benchmarks

### Coverage Requirements

**Minimum Coverage:** 80% for core modules

```bash
# Generate coverage report
pytest tests/ --cov=knowledge3d --cov-report=html

# View report
open htmlcov/index.html
```

---

## Documentation Standards

### Required Documentation

**Every module must have:**
1. Module-level docstring explaining purpose
2. Public function docstrings (Google style)
3. Type hints for all parameters
4. Usage examples in docstrings

**Every feature must have:**
1. Entry in relevant `docs/` file
2. Test coverage
3. Performance benchmarks (for hot paths)

### Documentation Files

**Technical Specifications:**
- `docs/vocabulary/` — W3C-ready specs (K3D Node, Three-Brain System, etc.)
- `docs/CRANIUM_CORE.md` — Cognitive engine details
- `docs/PTX_FUSED_HEAD_PLAN.md` — PTX architecture
- `docs/DUAL_CODE_STRATEGY.md` — HR/MR optimization

**Development Guides:**
- `docs/ENV_POLICY.md` — Environment setup
- `docs/HOUSE_GALAXY_TABLET.md` — Memory architecture
- `docs/TRAINING_DIRECTIVES.md` — Training hygiene

**Standards & Specs:**
- `spec/glTF_K3D_extension.md` — glTF schema
- `docs/W3C/` — W3C contribution documents

### Markdown Style

**Headers:**
```markdown
# Top-Level (Document Title)
## Section
### Subsection
#### Detail
```

**Code Blocks:**
````markdown
```python
# Always specify language
def example():
    pass
```
````

**Links:**
```markdown
# Internal: relative paths
See [Memory Architecture](docs/HOUSE_GALAXY_TABLET.md)

# External: full URLs
Visit [W3C AI KR](https://www.w3.org/community/aikr/)
```

**Tables:**
```markdown
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Value    | Value    | Value    |
```

### Asset Management

**Visual Assets:**
- Location: `docs/images/`
- Naming: `snake_case.png`
- Pair with prompt: `image_name_prompt.md`
- Reference in docs with relative links

**Large Assets:**
- Never commit files >99MB
- Store in `../Knowledge3D.local/`
- Document regeneration in `Large_Assets_Kitchen/README.md`

---

## Memory & Knowledge Management

### The Memory Policy

**Core Principle:** Galaxy ↔ House ↔ Museum via Memory Tablet

#### Galaxy (Active RAM)
**Purpose:** High-frequency reasoning buffer
**Lifespan:** Volatile; session-based
**Access:** PTX kernels, on-demand streaming
**Budget:** <200MB VRAM on consumer GPU

**When to Use:**
- Active reasoning tasks
- PTX kernel operations
- Temporary working sets

**When NOT to Use:**
- Long-term storage (use House)
- Deprecated knowledge (use Museum)

#### House (Persistent Disk)
**Purpose:** Consolidated knowledge artifacts
**Lifespan:** Long-term; evolves during sleep cycles
**Access:** Memory Tablet search + selective PTX load
**Format:** glTF/GLB with `extras.k3d` extensions

**Contents:**
- Books (consolidated documents)
- Diaries (AI reflections)
- Fractal trees (hierarchical knowledge)
- Learning insights (training outcomes)
- Dream records (sleep-time reasoning)

**When to Use:**
- Storing validated knowledge
- Artifacts that survived sleep consolidation
- Knowledge requiring provenance tracking

#### Museum (Cold Archive)
**Purpose:** Deprecated/superseded artifacts
**Lifespan:** Long-term; append-only
**Access:** Manual load via tablet "Museum mode"
**Zone:** Zone 8 (dedicated archive zone)

**When to Use:**
- Superseded versions of knowledge
- Error-pattern analysis
- Audit trails
- Retrospective training

### Sleep-Time Consolidation

**Process:** Nightly (or on-demand) memory crystallization

**Steps:**
1. **Lock Galaxy** — Prevent new writes
2. **EMA Updates** — Exponential moving average of embeddings
3. **Prune Redundancy** — Remove duplicates, low-confidence items
4. **Serialize Artifacts** — Export to House as glTF/GLB
5. **Commit House** — Update house memory index
6. **Unlock Galaxy** — Resume normal operations

**Triggering:**
```bash
# Manual consolidation
python -m knowledge3d.cranium.sleep.sleep_time_compute \
  --house-id default \
  --mode full

# Scheduled (cron example)
0 3 * * * cd /path/to/Knowledge3D && conda run -n k3d-cranium \
  python -m knowledge3d.cranium.sleep.sleep_time_compute
```

**Performance:** <10ms for 51,532 nodes (production validated)

### Memory Tablet Usage

**Search House:**
```python
from knowledge3d.bridge.memory_tablet import MemoryTablet

tablet = MemoryTablet(house_id="default")

# Semantic search
results = tablet.search(
    query="neural network architecture",
    sources=["House"],  # Check House first, then Galaxy
    limit=10,
    confidence_threshold=0.7
)

for result in results:
    print(f"{result.label} — {result.provenance}")
    # provenance: "House/Zone3/Books/ML_Fundamentals.glb"
```

**Stream to Galaxy (On-Demand):**
```python
# Load artifact into active memory
tablet.stream_to_galaxy(
    artifact_path="House/Zone3/Books/ML_Fundamentals.glb",
    lod="full"  # "coarse", "medium", "full"
)
```

**Browser Integration (Legacy Web):**
```python
# Capture web content via embedded browser
note = tablet.browse(
    url="https://arxiv.org/abs/2501.12345",
    capture_mode="structured"  # Extract text, images, links
)

# Note auto-consolidates to House during next sleep cycle
```

### Relocation to Museum

**When to Relocate:**
- Knowledge superseded by new version
- Deprecated prompts after mastery
- Error-inducing artifacts needing analysis

**Example:**
```python
from knowledge3d.tools.relocate_to_museum import relocate

relocate(
    artifact_path="House/Zone3/Books/Old_ML_Book.glb",
    reason="Superseded by ML_Fundamentals_v2.glb",
    preserve_metadata=True
)

# Result: moved to Museum/Zone8/2025-11-17/Old_ML_Book.glb
# Tagged with: relocated_at, previous_zone, superseded_by
```

### Prompt Hygiene (Training)

**Retirement Policy:**
1. Prompt achieves 1.0 honesty score
2. Verified in two consecutive sleep cycles
3. Moved from "active drills" to "mastered verification"
4. Replaced with harder/novel prompts

**Implementation:**
```python
# In training loops (Phase 18/25)
if prompt.honesty_score >= 1.0 and prompt.consecutive_passes >= 2:
    prompt_pool.retire(prompt, reason="mastered")
    logger.info(f"Retired: {prompt.text}")
```

**Timestamp Logging:**
```python
from datetime import datetime, timezone

timestamp = datetime.now(timezone.utc).isoformat()
log_entry = {
    "timestamp": timestamp,
    "prompt": prompt.text,
    "action": "retired",
    "reason": "mastered"
}
```

---

## Common Tasks & Commands

### Dataset Generation

**Text to Galaxy:**
```bash
python -m k3dgen \
  --text data/my_corpus.txt \
  --gltf output.glb \
  --k 5 \
  --reducer umap \
  --model sentence-transformers/all-MiniLM-L6-v2
```

**CSV to Galaxy:**
```bash
python -m k3dgen data.csv \
  --gltf scene.glb \
  --k 10 \
  --reducer umap
```

**Multimodal Ingestion:**
```bash
# Text + Images
python -m knowledge3d.tools.ingest_wit \
  --dataset wit_train.tsv \
  --output multimodal.glb

# Video → CLIP embeddings
python -m knowledge3d.tools.ingest_video \
  --input videos/ \
  --output video_galaxy.glb

# Audio → CLAP embeddings
python -m knowledge3d.tools.ingest_audio \
  --input audio/ \
  --output audio_galaxy.glb
```

### Training Commands

**RLWHF Training:**
```bash
# Generate questions from corpus
python -m knowledge3d.training.rlwhf.question_generator_ollama \
  --corpus data/pdfs/ \
  --output questions.json

# Student attempts (GPU-batched)
python -m knowledge3d.training.rlwhf.student_attempt_trm_batched \
  --questions questions.json \
  --output student_attempts.json \
  --batch-size 128

# Teacher evaluation
python -m knowledge3d.training.rlwhf.teacher_eval_ollama \
  --attempts student_attempts.json \
  --output teacher_feedback.json \
  --model deepseek-r1:70b

# Train TRM on feedback
python -m knowledge3d.training.rlwhf.train_rlwhf \
  --feedback teacher_feedback.json \
  --output models/trm_trained.pth
```

**Validation:**
```bash
# Batched validation
python scripts/validate_rlwhf_training_batched.py \
  --model models/trm_trained.pth \
  --questions questions.json
```

### Viewer Commands

**Launch Viewer:**
```bash
cd viewer
npm install
npm run dev
```

**Build for Production:**
```bash
npm run build
```

**Run Viewer Tests:**
```bash
npm install --ignore-scripts --no-bin-links
node ./node_modules/jest/bin/jest.js --runInBand
```

### Code Quality

**Linting:**
```bash
flake8 knowledge3d/
```

**Formatting:**
```bash
black knowledge3d/
```

**Type Checking:**
```bash
mypy knowledge3d/
```

### Docker Operations

**Build Image:**
```bash
docker build -t knowledge3d:latest .
```

**Run Container:**
```bash
docker run -p 8765:8765 \
  -v $(pwd)/../Knowledge3D.local:/data \
  --gpus all \
  knowledge3d:latest
```

**Test Container:**
```bash
docker build -f Dockerfile.test -t knowledge3d:test .
docker run knowledge3d:test pytest tests/
```

---

## Critical Policies & Constraints

### 1. Sovereignty Constraints

**MUST:**
- ✅ Use PTX kernels for all hot paths
- ✅ Bind directly to libcuda.so via ctypes
- ✅ Achieve <100µs latency on consumer GPU
- ✅ Keep VRAM <200MB for core operations
- ✅ Build reproducibly (Dockerfile, SHA256 verification)

**MUST NOT:**
- ❌ Fall back to CPU for core inference
- ❌ Depend on cloud APIs for reasoning
- ❌ Use black-box frameworks (TensorFlow, PyTorch in hot paths)
- ❌ Exceed GPU memory budget

**Rationale:** Sovereignty ensures explainability, reproducibility, and zero vendor lock-in.

### 2. Memory Budget

**Galaxy RAM:** <200MB VRAM
**House Disk:** Unlimited (but keep LOD tiers)
**Museum Archive:** Unlimited (cold storage)

**Enforcement:**
```python
import cupy as cp

mem_info = cp.cuda.runtime.memGetInfo()
used_mb = (mem_info[1] - mem_info[0]) / (1024 ** 2)

if used_mb > 200:
    logger.warning(f"Galaxy RAM exceeded budget: {used_mb:.1f} MB")
    # Trigger LOD downgrade or prune low-confidence items
```

### 3. Embodiment Constraint

**The avatar lives in the House, not the Galaxy.**

**Implications:**
- Navigation occurs in 3D rooms (not abstract embedding space)
- Galaxy is introspective (like viewing a brain scan)
- Tablet bridges House ↔ Galaxy for reasoning
- All user interactions assume House context

**Violation Example (WRONG):**
```python
# WRONG: Placing avatar inside Galaxy
avatar.teleport_to(galaxy_position)
```

**Correct Example:**
```python
# CORRECT: Avatar in House, consults Galaxy via tablet
tablet = MemoryTablet()
galaxy_context = tablet.query_galaxy(semantic_query)
avatar.reason_with(galaxy_context, house_context)
```

### 4. File Size Limits

**Repository (<99MB):**
- Source code
- Documentation
- Test fixtures
- Small datasets (<10MB)

**Knowledge3D.local (Unlimited):**
- Houses (glTF/GLB files)
- Galaxies
- Logs (JSONL)
- Datasets (>10MB)
- Trained models

**Regeneration Recipe Required:**
- Document steps in `Large_Assets_Kitchen/README.md`
- Include SHA256 checksums
- Provide download/generation scripts

### 5. Deprecation Policy

**Deprecated Code:**
- Move to `Old_Attempts/`
- Add `README_DEPRECATION.md` explaining why
- Link to replacement/consolidation
- Never import from `Old_Attempts/` in new code

**Examples:**
- `Old_Attempts/Legacy_Fancy_RAG/` — Original RAG scaffolding
- `Old_Attempts/fsm_scaffolding/` — Step 12 FSM (consolidated into ThinkingTagBridge)

### 6. Commit Hygiene

**DO:**
- ✅ Write clear, descriptive commit messages
- ✅ Use conventional commits (`feat:`, `fix:`, `docs:`, `test:`)
- ✅ Reference issue numbers (`fixes #123`)
- ✅ Keep commits atomic (one logical change)
- ✅ Test before committing

**DON'T:**
- ❌ Commit generated files (MR code, compiled PTX)
- ❌ Commit secrets (.env, credentials)
- ❌ Commit >99MB files
- ❌ Force-push to shared branches
- ❌ Mix unrelated changes in one commit

### 7. AI Diary (AI-Only Writes)

**Policy:**
- Humans can **read** diary pages
- Humans **cannot write** to Diary room (bridge blocks this)
- AI writes based on novelty + confidence threshold

**Implementation:**
```python
# knowledge3d/cranium/diary.py
def can_write_diary(agent_type: str) -> bool:
    return agent_type == "AI"

# In bridge
if room.name == "Diary" and user.is_human:
    raise PermissionError("Diary is AI-only. Read access granted.")
```

### 8. Timezone-Aware Timestamps

**All logs/training must use UTC timestamps:**

```python
from datetime import datetime, timezone

timestamp = datetime.now(timezone.utc).isoformat()
# Output: 2025-11-17T14:32:01.123456+00:00
```

**Rationale:** Reproducibility, cross-timezone collaboration, sleep-cycle scheduling.

---

## Troubleshooting & Known Issues

### GPU/CUDA Issues

**Problem:** `CUDA_ERROR_INVALID_CONTEXT`

**Solution:**
```bash
# Ensure CUDA_VISIBLE_DEVICES is set before activating conda env
export CUDA_VISIBLE_DEVICES=0
tmux new -As k3d
conda activate k3d-cranium
```

**Problem:** GPU tests failing in CI

**Solution:** Tests marked `@pytest.mark.cuda` are automatically skipped in CI (CPU-only runners). Ensure proper markers:
```python
@pytest.mark.cuda
def test_my_gpu_kernel():
    # GPU-specific test
```

**Problem:** Out of VRAM

**Solution:**
```python
# Check memory usage
import cupy as cp
mem_info = cp.cuda.runtime.memGetInfo()
print(f"Free: {mem_info[0] / (1024**2):.1f} MB")

# Trigger LOD downgrade or clear cache
galaxy.prune_low_confidence(threshold=0.5)
```

### WebSocket Connection Issues

**Problem:** Handshake timeout on `live_server`

**Solution:**
```bash
# Use pinned websockets version
pip install websockets==10.4

# Increase timeout
python -m knowledge3d.bridge.live_server --open-timeout 30

# Fast start mode (delay heavy imports)
K3D_LIVE_FAST=1 python -m knowledge3d.bridge.live_server
```

**Problem:** Port 8787 already in use (ComfyUI conflict)

**Solution:**
```bash
# Override default ports
export K3D_LIVE_PORTS="8791 8793 8797"
python -m knowledge3d.bridge.live_server --auto-port
```

### Environment Issues

**Problem:** System Python conflicts

**Solution:** Always use Conda environments:
```bash
# Never use system Python
which python  # Should show conda env, not /usr/bin/python

# If wrong:
conda deactivate  # Deactivate system
conda activate k3d-cranium  # Activate K3D env
```

**Problem:** PYTHONPATH issues

**Solution:**
```bash
# Always set PYTHONPATH when running from repo root
env PYTHONPATH=. python -m knowledge3d.tools.my_tool
```

### Testing Issues

**Problem:** Tests pass locally but fail in CI

**Causes:**
1. GPU tests not marked with `@pytest.mark.cuda`
2. Hardcoded paths (use fixtures)
3. Missing dependencies in test environment

**Solution:**
```python
# Proper GPU test marking
@pytest.mark.cuda
def test_gpu_feature():
    pass

# Use fixtures for paths
@pytest.fixture
def tmp_galaxy(tmp_path):
    return tmp_path / "test_galaxy.glb"
```

### Memory/Performance Issues

**Problem:** Sleep consolidation taking too long

**Solution:**
```python
# Enable profiling
python -m knowledge3d.cranium.sleep.sleep_time_compute \
  --profile \
  --output profile.json

# Analyze bottlenecks
python -m knowledge3d.tools.analyze_profile profile.json
```

**Problem:** Galaxy search slow

**Solution:**
```bash
# Rebuild index with optimal parameters
python -m knowledge3d.tools.rebuild_galaxy_index \
  --k 5 \
  --method faiss  # Faster than sklearn for large datasets
```

### Known Limitations

1. **PDF OCR:** Scanned PDFs use Tesseract fallback (temporary; sovereign OCR in Phase F)
2. **Query Ranking:** Some COCO captions rank higher than exact matches (tuning needed)
3. **Audio Pipeline:** Sovereign audio embeddings experimental (CLAP temporary)
4. **Multimodal Fusion:** Visual-text alignment being refined in Phase E.5

---

## Appendix: Quick Reference

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `K3D_LOCAL_DIR` | Runtime workspace path | `../Knowledge3D.local` |
| `K3D_HOUSE_ID` | Active house identifier | `default` |
| `CUDA_VISIBLE_DEVICES` | GPU selection | `0` |
| `K3D_LIVE_FAST` | Fast-start mode for live server | `1` |
| `K3D_LIVE_PORTS` | WebSocket port candidates | `8765 8766 8767` |
| `K3D_SEED_GRAPH_MAX` | Max graph size for seeder | `10000` |
| `K3D_RPN_TRACE` | Enable RPN trace in output | `0` |
| `PYTHONPATH` | Python module path | `.` (repo root) |

### Key File Locations

| File | Purpose |
|------|---------|
| `knowledge3d/cranium/ptx_runtime/thinking_tag_bridge.py` | Main cognitive engine |
| `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py` | RPN executor |
| `knowledge3d/bridge/memory_tablet.py` | Memory Tablet interface |
| `knowledge3d/bridge/live_server.py` | WebSocket server |
| `docs/HOUSE_GALAXY_TABLET.md` | Memory architecture spec |
| `docs/ROADMAP.md` | Current phase priorities |
| `CODEX.md` | Actionable task list |
| `AGENTS.md` | Agent collaboration guide |

### Command Cheat Sheet

```bash
# Environment
conda activate k3d-cranium
tmux new -As k3d

# Development
python -m knowledge3d.bridge.live_server --port 8787
cd viewer && npm run dev

# Testing
pytest tests/ -v
pytest tests/ -v -m cuda  # GPU only
pytest tests/ -v -m "not cuda"  # CPU only

# Training
python -m knowledge3d.training.rlwhf.train_rlwhf --dataset corpus.glb

# Sleep
python -m knowledge3d.cranium.sleep.sleep_time_compute --house-id default

# Code Quality
flake8 knowledge3d/
black knowledge3d/
```

---

## Getting Help

### Documentation Resources

1. **[NotebookLM Research Space](https://notebooklm.google.com/notebook/1bd10bda-8900-4c41-931e-c9ec67ac865f)** — Best entry point for comprehensive understanding
2. **[K3D Technical White Paper](K3D_Technical_White_Paper.md)** — Architecture deep dive
3. **[Video Presentation (6 min)](https://www.youtube.com/watch?v=Dy7mnNSZWuU)** — Quick introduction
4. **W3C Specifications** — `docs/vocabulary/*.md` files
5. **TEMP/** — Step-by-step implementation reports

### Community

- **GitHub Issues:** https://github.com/danielcamposramos/Knowledge3D/issues
- **W3C AI KR Group:** https://www.w3.org/community/aikr/
- **Contact:** daniel@echosystems.ai

### For AI Assistants

**When stuck:**
1. Re-read [AGENTS.md](AGENTS.md) and [CODEX.md](CODEX.md)
2. Check current phase in [docs/ROADMAP.md](docs/ROADMAP.md)
3. Consult [docs/HOUSE_GALAXY_TABLET.md](docs/HOUSE_GALAXY_TABLET.md) for memory operations
4. Search `TEMP/` for recent implementation notes
5. Ask user for clarification if architectural decisions unclear

**Philosophy to Remember:**
- "We fix or we fix — never fallback to CPU"
- "The avatar lives in the House, not the Galaxy"
- "Knowledge lives in embeddings, TRM learns reasoning patterns"
- "Sovereign, explainable, embodied — no exceptions"

---

**End of CLAUDE.md**

*This document is a living guide. Update it when architecture evolves or new patterns emerge.*
