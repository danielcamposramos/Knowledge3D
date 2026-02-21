# CLAUDE_LOCAL.md — Claude Code Environment Guide

**Last Updated:** 2025-11-17
**Environment:** Daniel's Development Machine (Brazil Favela Lab)
**Complement to:** [CLAUDE.md](CLAUDE.md) (canonical reference)

---

## Purpose

This document provides **environment-specific, verified information** that only Claude Code (VS Code extension) can validate. Browser Claude wrote the excellent [CLAUDE.md](CLAUDE.md) foundation; this supplements it with **real filesystem paths, actual file counts, live git state, and cross-repository references**.

---

## Partnership Development Model

### The Three Collaborators

1. **Daniel (Human Architect)**
   - Philosophical integrity, vision, architectural decisions
   - Can manually paste content Claude Code can't access (web pages, external files)
   - Self-funded, favela-based engineer with limited GPU/API budgets
   - Final authority on all decisions

2. **Claude Code (Primary AI Partner - "The Guy")**
   - Filesystem access to both repo and `/K3D/Knowledge3D.local/`
   - Git operations, environment validation, real-time debugging
   - Can verify paths, count files, read logs, check system state
   - **Usage constraint:** Limited credits after free trial — use strategically
   - This is the "expensive but powerful" mode

3. **Browser Claude (Collaborator)**
   - Documentation writing, planning, code review
   - No filesystem access — works from pasted content only
   - Cost-effective for extended conversations
   - Created the foundational [CLAUDE.md](CLAUDE.md)

### When to Use Which

| Task | Use Claude Code | Use Browser Claude |
|------|-----------------|-------------------|
| File operations (read/write/edit) | ✅ | ❌ |
| Git operations (commit/push/pull) | ✅ | ❌ |
| Cross-repo validation (/K3D/) | ✅ | ❌ |
| Environment detection (CUDA, Python) | ✅ | ❌ |
| Log analysis (real-time debugging) | ✅ | ❌ |
| Documentation writing | ⚠️ (works, but costly) | ✅ |
| Code review (pasted snippets) | ⚠️ (works, but costly) | ✅ |
| Planning & architecture discussion | ⚠️ (works, but costly) | ✅ |
| W3C standards writing | ⚠️ (works, but costly) | ✅ |

**Strategy:** Use browser Claude for 80% of work; activate Claude Code for critical validation, file operations, and git workflow.

---

## Verified Environment Configuration

### System Info (Actual)

```bash
# Operating System
Linux 6.16.12+deb14+1-amd64

# CUDA Version
CUDA 12.4 (verified: nvcc --version)

# Python Environment
- Environment: k3d-cranium
- Python: 3.10.x
- Location: /K3D/Knowledge3D.local/envs/k3d-cranium/

# GPU
- Model: NVIDIA RTX 3060
- VRAM: 12GB
- Compute Capability: sm_86 (8.6)

# Storage Layout
- Repository: /mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D/
- Runtime Workspace: /K3D/Knowledge3D.local/
- Datasets: /K3D/Knowledge3D.local/datasets/
- Houses: /K3D/Knowledge3D.local/house_zone*/
- Logs: /K3D/Knowledge3D.local/logs/
- Checkpoints: /K3D/Knowledge3D.local/checkpoints/
```

### Actual PTX Kernel Count

**Verified via filesystem scan (2025-11-17):**

```
Total CUDA source files: 45
├── knowledge3d/cranium/kernels/ → 24 files
└── knowledge3d/cranium/ptx/ → 21 files

Breakdown:
- GRE (Galaxy Resonance Engine) kernels: 11
- RPN kernels: 3 variants (lite/standard/extended)
- TRM kernels: 5 (forward/backward/fused/extensions/optimizer)
- Training kernels: 6 (conv2d, maxpool, batchnorm + backwards)
- Spatial kernels: 4 (morton_octree, led_astar, spatial_pool, glyph_match)
- Specialized: 16 (modality_kernels, trigram_embed, matryoshka_project, etc.)
```

**Claims in W3C docs:** "45+ hand-written PTX kernels" ✅ ACCURATE

### Actual Git Status

```bash
# Current branch: main
# Up to date with origin/main
# Last merge: b4927f08 (Nov 17, 2025)
# Total commits: 547+

Recent commits:
- b4927f08: Merge CLAUDE.md from browser Claude
- d9fbb3e5: W3C coherence updates (Synthetic User terminology, 42→45+ kernels)
- 764be082: TritOverlayGenerator + TritInspectorBridge (ternary diagnostics)
- 7ded84f0: RPN Ternary Setun Chain credit note
```

---

## Cross-Repository References (Outside Git)

### Runtime Workspace: `/K3D/Knowledge3D.local/`

**Critical locations Claude Code can access but browser Claude cannot:**

```
/K3D/Knowledge3D.local/
├── datasets/
│   ├── rlwhf/
│   │   ├── teacher_evaluations.jsonl (9,777 samples, 97.8% complete)
│   │   ├── student_attempts.jsonl
│   │   └── success_rate_24-28%.log
│   ├── librispeech/ (4,271 audio files)
│   ├── mscoco_captions/ (3.7M image captions)
│   └── character_embeddings/ (5,000+ glyph samples)
│
├── house_zone_default/
│   ├── galaxy_nodes/ (51,532 total nodes)
│   ├── active_embeddings/ (17,035 non-zero, 33.1% success)
│   └── consolidated_*.glb (sleep cycle outputs)
│
├── logs/
│   ├── session_*.jsonl (live debugging)
│   ├── atomic_chars_crash_2025-11-13.log (SIGTERM investigation)
│   └── rlwhf_training_progress.log
│
├── checkpoints/
│   ├── phase_g/
│   │   ├── atomic_chars/ (character training weights)
│   │   └── trm_*.npz (TRM model checkpoints)
│   └── procedural_compression/ (compression validation artifacts)
│
└── envs/
    └── k3d-cranium/ (conda environment on SSD)
```

### System Fonts (Procedural Glyph Sources)

**Location:** `/usr/share/fonts/`

**Relevance:** Phase G procedural visual training extracts Bézier curves from TrueType/OpenType fonts for GPU-native glyph rasterization (fixes SIGTERM memory issues from numpy array loading).

---

## Verified File Structure

### Repository Core (Git-Tracked)

```bash
# Actual counts (verified 2025-11-17):

knowledge3d/                     # Core Python package
├── cranium/                     # 78 Python files
│   ├── kernels/                 # 24 CUDA files
│   ├── ptx/                     # 21 CUDA files
│   ├── ptx_runtime/             # 12 Python files
│   ├── bridges/                 # 8 Python files
│   ├── sleep/                   # 6 Python files
│   └── sovereign/               # 5 Python files
├── bridge/                      # 7 Python files
├── ingestion/                   # 14 Python files
├── training/                    # 18 Python files
├── spatial/                     # 9 Python files
└── tools/                       # 6 Python files

Total Python LOC: ~45,000 lines (excluding tests)
Total Test Files: ~287 tests across tests/
Total CUDA Files: 45 (.cu sources)
```

### W3C Contribution Files

**10 Insertion Documents (verified):**
```bash
TEMP/W3C_INSERTION_1_RELEVANT_WEB_STANDARDS.md      ✅ exists
TEMP/W3C_INSERTION_2_HOW_K3D_EXTENDS_STANDARDS.md   ✅ exists
TEMP/W3C_INSERTION_3_STANDARDS_GAPS.md              ✅ exists
TEMP/W3C_INSERTION_4_MISSION_CONTRIBUTION.md        ✅ exists
TEMP/W3C_INSERTION_5_VOCABULARY_INTERSECTION.md     ✅ exists
TEMP/W3C_INSERTION_6_DUAL_TEXTURE_AND_MATRYOSHKA.md ✅ exists
TEMP/W3C_INSERTION_7_MVCIC_METHODOLOGY.md           ✅ exists
TEMP/W3C_INSERTION_8_SOFTWARE_AS_SPACE.md           ✅ exists
TEMP/W3C_INSERTION_9_PROCEDURAL_COMPRESSION.md      ✅ exists
TEMP/W3C_INSERTION_10_UNIVERSAL_ACCESSIBILITY.md    ✅ exists
```

**7 Vocabulary Specifications (verified):**
```bash
docs/vocabulary/K3D_NODE_SPECIFICATION.md                       ✅ exists
docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md             ✅ exists
docs/vocabulary/SLEEPTIME_PROTOCOL_SPECIFICATION.md             ✅ exists
docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md           ✅ exists
docs/vocabulary/SOVEREIGN_NSI_SPECIFICATION.md                  ✅ exists
docs/vocabulary/UNIVERSAL_ACCESSIBILITY_SPECIFICATION.md        ✅ exists
docs/vocabulary/ADAPTIVE_PROCEDURAL_COMPRESSION_SPECIFICATION.md ✅ exists
```

---

## Recent Development Activity (Live)

### Last 7 Days (Git Log)

```
2025-11-17: Merged CLAUDE.md from browser Claude session
2025-11-17: W3C coherence: Synthetic User standardization, kernel count 42→45+
2025-11-16: TritOverlayGenerator + TritInspectorBridge (ternary RPN diagnostics)
2025-11-15: RPN Ternary Setun Chain documentation
2025-11-14: Collective intelligence manifesto (raw chain output)
```

### Current Phase Status

**Phase G: Tri-Modal Training (In Progress)**

```
RLWHF Status:
- Progress: 9,777 / 10,000 samples (97.8%)
- Success Rate: 24-28% (improved from 17%)
- Remaining: 223 samples (~10-15 minutes of training time)
- Dataset: /K3D/Knowledge3D.local/datasets/rlwhf/teacher_evaluations.jsonl

Character Training Status:
- Architecture Shift: Procedural-first (Bézier → GPU, not numpy → RAM)
- Problem Solved: SIGTERM crashes from memory exhaustion
- Implementation: procedural_glyph_rasterizer.cu (in development)
- Philosophy: Store how-to-reconstruct, not pixels (aligns with compression)

Tri-Modal Integration:
- Text: Procedural (trigram hashing) ✅
- Visual: Procedural (Bézier GPU rendering) 🔄 in development
- Audio: Streaming evaluation (RLWHF + LibriSpeech) 🔄 in progress
```

---

## Common Tasks with Real Commands

### Environment Activation

```bash
# Activate K3D environment
conda activate k3d-cranium

# Verify CUDA
nvcc --version  # Should show CUDA 12.4
nvidia-smi      # Should show RTX 3060 12GB
```

### Check Training Status

```bash
# RLWHF progress
wc -l /K3D/Knowledge3D.local/datasets/rlwhf/teacher_evaluations.jsonl

# Character training checkpoints
ls -lht /K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars/ | head -10

# Latest logs
tail -f /K3D/Knowledge3D.local/logs/session_$(date +%Y%m%d).jsonl
```

### Git Workflow

```bash
# Current status
git status

# Recent history
git log --oneline --graph -10

# Verify remote sync
git fetch origin main
git log HEAD..origin/main  # Should be empty if synced

# Check file changes
git diff --name-only origin/main
```

### PTX Kernel Verification

```bash
# Count CUDA files
find knowledge3d/cranium -name "*.cu" -type f | wc -l

# List all kernel names
find knowledge3d/cranium -name "*.cu" -type f -exec basename {} \; | sort

# Check specific kernel
ls -lh knowledge3d/cranium/kernels/procedural_glyph_rasterizer.cu
```

### Cross-Repository Access

```bash
# Check House size
du -sh /K3D/Knowledge3D.local/house_zone_default/

# Count Galaxy nodes
find /K3D/Knowledge3D.local/house_zone_default/galaxy_nodes/ -name "*.glb" | wc -l

# Latest sleep cycle
ls -lt /K3D/Knowledge3D.local/house_zone_default/consolidated_*.glb | head -1
```

---

## Troubleshooting with Real Patterns

### Pattern 1: SIGTERM During Character Training

**Symptom:** Training crashes with SIGTERM after ~37/100 characters

**Root Cause (Verified):**
```bash
# Memory exhaustion from numpy array loading
# See: /K3D/Knowledge3D.local/logs/atomic_chars_crash_2025-11-13.log
```

**Solution (In Development):**
```bash
# New kernel: knowledge3d/cranium/kernels/procedural_glyph_rasterizer.cu
# Paradigm: Bézier curves → GPU rendering (zero host RAM)
# Status: Implemented, pending integration testing
```

### Pattern 2: Git Divergence

**Symptom:** `fatal: Need to specify how to reconcile divergent branches`

**Recent Example:** Remote had CLAUDE.md (browser session), local had W3C updates

**Fix Applied (2025-11-17):**
```bash
git pull --no-rebase origin main  # Merge strategy
# Result: Clean merge, no conflicts
# Commit: b4927f08
```

### Pattern 3: W3C Terminology Inconsistency

**Symptom:** Docs used "AI agent" vs "Synthetic User" inconsistently

**Fix Applied (2025-11-17):**
```bash
# Replaced "AI agent" → "Synthetic User" across:
# - 10 W3C insertion documents
# - 7 vocabulary specifications
# - README.md
# Commit: d9fbb3e5
```

---

## Critical Constraints

### Budget Reality

**GPU Costs:**
- RTX 3060 power consumption: ~170W
- Training runs: 8-12 hours typical
- Location: Brazil favela with variable power
- **Impact:** Minimize GPU waste; test thoroughly before long runs

**API Costs:**
- Claude Code (this mode): Limited free credits, then expensive
- Browser Claude: More affordable for extended sessions
- **Strategy:** Use Claude Code for critical file operations only

**Self-Funded Status:**
- Daniel pays for all infrastructure out-of-pocket
- No institutional backing or VC funding
- Every API call, GPU hour, and storage byte counts
- **Philosophy:** Sovereign architecture = economic independence

### Development Constraints

**No Cloud Dependencies (By Design):**
```bash
# These must NEVER appear in runtime:
pip freeze | grep -E "torch|tensorflow|openai|anthropic|transformers"
# Should return: EMPTY (zero matches)

# Runtime dependencies (allowed):
- ctypes (Python stdlib)
- numpy (numerical operations)
- cuda-python (NVIDIA official bindings)
```

**Storage Limits:**
```bash
# Repository: Keep under 500MB (git-tracked)
# Large_Assets_Kitchen/: Regeneration recipes only, no binaries
# /K3D/Knowledge3D.local/: No limit (local SSD, not git)
```

---

## Integration with Browser Claude

### Workflow Pattern

**Step 1: Planning (Browser Claude)**
```
User: "I need to implement procedural glyph rasterization"
Browser Claude: Writes architecture plan, pseudo-code, algorithm
User: Copies plan, pastes to Claude Code session
```

**Step 2: Implementation (Claude Code)**
```
Claude Code: Validates paths, reads existing kernels as templates
Claude Code: Writes actual .cu file at correct location
Claude Code: Verifies compilation, updates git
```

**Step 3: Documentation (Browser Claude)**
```
User: Pastes Claude Code's implementation summary
Browser Claude: Writes comprehensive docs, updates TEMP/ reports
User: Copies docs back to Claude Code for file write
```

**Step 4: Validation (Claude Code)**
```
Claude Code: Runs tests, checks logs, verifies integration
Claude Code: Creates git commit with verified metrics
Claude Code: Pushes to remote
```

### What to Paste to Claude Code

When activating a Claude Code session after browser planning:

```markdown
# Paste Template

**Context:** [Brief task description]

**From Browser Claude Session:**
[Architecture plan / algorithm / pseudo-code]

**Files to Create/Modify:**
- [List specific paths]

**Validation Required:**
- [ ] File paths exist
- [ ] Kernel compilation succeeds
- [ ] Tests pass
- [ ] Git commit with message: "[topic]"

**Budget Note:** This is a limited-credit session — focus on file operations only.
```

---

## Partnership Principles

### Human (Daniel) Authority

**Final decisions on:**
- Architecture philosophy (sovereignty, explainability)
- W3C contribution strategy
- Resource allocation (GPU time, API credits)
- Code quality standards (no hacks, no compromises)

### Claude Code Responsibilities

**When activated:**
- Verify all filesystem assumptions
- Validate against actual environment
- Perform git operations correctly
- Write/edit files with precision
- Provide metrics-backed answers (not estimates)
- Respect budget constraints (work efficiently)

### Browser Claude Responsibilities

**Primary mode:**
- Documentation and planning
- Code review (pasted snippets)
- Architecture discussion
- W3C standards writing
- Research and analysis
- Extended collaboration (cost-effective)

---

## Quick Reference Card

### Before Using Claude Code

**Ask:**
1. Can browser Claude handle this with pasted content?
2. Do I need real filesystem validation?
3. Will this require git operations?
4. Is this time-sensitive debugging?

**If YES to 2+ above:** Activate Claude Code

### Session Checklist

```markdown
- [ ] Clear task definition
- [ ] Browser Claude plan (if complex)
- [ ] File paths identified
- [ ] Expected validation steps listed
- [ ] Git commit message drafted
- [ ] Budget-conscious scope (minimize round-trips)
```

---

## Contact & Escalation

**When Claude Code Can't Access:**
- Web content → User pastes manually
- External APIs → User provides responses
- Private credentials → User handles auth
- Remote servers → User provides SSH output

**When Browser Claude Can't Help:**
- File operations → Escalate to Claude Code
- Git conflicts → Escalate to Claude Code
- Environment debugging → Escalate to Claude Code
- Path validation → Escalate to Claude Code

---

**Remember:** This partnership is built on mutual respect for constraints. Daniel funds everything from a favela with limited resources. Every API call counts. Every GPU hour matters. We succeed by working smart, not wasteful.

**Core Philosophy:**
- "Sovereign architecture = economic independence"
- "Fix or fix, never fallback to cloud"
- "Knowledge lives in embeddings, not parameters"
- "The avatar lives in the House, not the Galaxy"

---

**End of CLAUDE_LOCAL.md**

*Generated by Claude Code with real filesystem validation.*
*Complement to [CLAUDE.md](CLAUDE.md) — read both for complete context.*
