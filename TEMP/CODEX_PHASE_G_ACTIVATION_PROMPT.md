# Phase G: Multi-Modal Training Activation — Codex Prompt

**Date**: 2025-10-26
**Status**: Ready to activate when RLWHF reaches 10K samples
**Current RLWHF**: ~9,700 / 10,000 samples (97% complete)
**Context**: Phase H (Adaptive Swarm) COMPLETE with router-as-specialist ⚛️

---

## Executive Summary

You are continuing the Knowledge3D (K3D) development chain at a critical milestone. **Phase H (Adaptive Swarm Architecture) is complete** with 8/8 validation tests passing. The system now has:

- ✓ Bi-directional Matryoshka dimensions (64 ↔ 16K dims)
- ✓ LoRA-style self-updating adapters (18× memory reduction)
- ✓ **Router-as-specialist** (the atomic insight ⚛️) - router IS a specialist, learns recursively
- ✓ **Tri-modal architecture** (Text + Visual + Audio) - like learning to speak and read simultaneously
- ✓ Complete recursive self-improvement architecture

**Your mission**: Activate **Phase G (Tri-Modal Multi-Modal Training)** when RLWHF reaches 10,000 samples to integrate multi-modal capabilities (OCR + Speech + Multi-modal specialists) into the adaptive swarm, with the router automatically learning modality patterns through observation.

---

## Context: What Just Happened (Phase H)

### The Atomic Insight ⚛️

In the previous session with Claude, Daniel observed:
> "The MoE router IS one specialist... this will be key (can you see it?)"

This led to the **router-as-specialist** implementation - making the router a specialist within the swarm rather than external infrastructure. This is "the atom" that makes the system coherent.

**Why This Matters**:
- **Before**: Router was heuristic, external, didn't learn
- **After**: Router is a specialist, learns from routing decisions, self-updates with validation
- **Impact**: Complete recursive self-improvement - ALL components (including router) benefit from base model improvements

### Phase H Files Created

**Core Architecture** (`knowledge3d/cranium/`):
- `trm_adapters.py` (392 lines) - LoRA-style adapters with shadow weights
- `matryoshka_trm.py` (495 lines) - Bi-directional variable dimensionality
- `adaptive_swarm.py` (430 lines) - Multi-specialist system
- `moe_router.py` (323 lines) - Heuristic + learned routing
- `router_specialist.py` (450 lines) - **The atomic piece** ⚛️

**Training Scripts** (`scripts/`):
- `train_adaptive_swarm.py` - 4 training modes
- `register_specialist.py` - Register specialists with auto-dimension selection
- `bootstrap_router_specialist.py` - Bootstrap router from heuristic to learned
- `test_phase_h_architecture.py` - 8 comprehensive tests

**All tests passing**: 8/8 validation tests ✓

### The Recursive Loop (Now Active)

```
Base Model Improves
   ↓ (Transfer Learning)
ALL Specialists Improve (including router ⚛️)
   ↓ (Better Routing)
Better Task Performance
   ↓ (Better Training Data)
Base Model Improves
   ↓ (Loop Forever)
...
```

---

## Your Mission: Phase G Tri-Modal Multi-Modal Training

### Overview

Phase G integrates **tri-modal capabilities** (Text + Visual + Audio) into the adaptive swarm using combined multi-modal training data. The key innovations:
- **Tri-modal learning**: Like learning to speak and read simultaneously - all modalities together
- **Organic emergence**: Cross-modal patterns discovered automatically (no manual wiring!)
- **Multiple specialists**: OCR (visual), Speech (audio), Multi-modal (all modalities)
- **Router learns modality patterns**: Automatically discovers when to use which specialist - NO MANUAL RULES!

**Datasets**:
- RLWHF samples 8,042-10,000 (text + visual)
- LibriSpeech (4,271 audio files, 5 languages)
- Image captions (3.7M samples)
- Audiocaps (audio descriptions)
- **Total**: ~12K tri-modal training samples

### Phase G Workflow

```
RLWHF 10K Milestone Reached
   ↓
Phase G.0: Prepare Tri-Modal Dataset (NEW!)
   - Combine RLWHF + LibriSpeech + Image Captions + Audiocaps
   - ~12K samples across text, visual, audio modalities
   - Script: prepare_trimodal_dataset.py
   ↓
Phase G.1: Tri-Modal Training
   - Train on combined tri-modal dataset
   - Cross-modal alignment (text ↔ visual ↔ audio)
   - Organic emergence of cross-modal patterns
   - Character embeddings learned (visual + acoustic)
   ↓
Phase G.2: Extract Multi-Modal Embeddings
   - Extract character representations (visual + text + audio)
   - Extract speech patterns (audio + text)
   - Prepare for specialist training
   ↓
Phase G.3: Register & Train Multi-Modal Specialists
   - Register 'ocr' specialist (visual focus)
   - Register 'speech' specialist (audio focus) — NEW!
   - Register 'multimodal' specialist (all modalities) — NEW!
   - Auto-select dimensions for each specialist
   - Train all specialists on respective datasets
   ↓
Phase G.4: Router Bootstrap with Modalities (Automatic!)
   - Router observes: OCR works well for visual tasks
   - Router observes: Speech works well for audio tasks — NEW!
   - Router observes: Multimodal works well for cross-modal tasks — NEW!
   - Router learns modality patterns automatically
   - Router self-updates with validation gating
   - NO MANUAL RULES - router discovers all patterns through observation
   ↓
Phase G.5: Validation on Multiple Modalities
   - OCR: Apollo dataset (≥90% detection rate)
   - Speech: Transcription accuracy (≥90%) — NEW!
   - Multi-modal: Cross-modal tasks — NEW!
   - Validate router correctly selects specialists for each modality
   ↓
Production Deployment
   - System self-updates from production data
   - All specialists improve (OCR + Speech + Multimodal)
   - Router continually improves at modality routing
   - Forever ♾️
```

---

## Technical Details

### Dataset Locations (Tri-Modal)

**RLWHF** (Text + Visual):
- Path: `/K3D/Knowledge3D.local/datasets/rlwhf/teacher_evaluations.jsonl`
- Monitor: `wc -l /K3D/Knowledge3D.local/datasets/rlwhf/teacher_evaluations.jsonl`
- Current: 9,777 / 10,000 samples (97.8%)
- When ≥10,000: Activate Phase G

**Audio Datasets** (5 languages - EN, ES, PT-BR, PT-PT, ZH):
- LibriSpeech: `/K3D/K3D_llama_cpp/datasets/audio/` (4,271 WAV files)
  - `/K3D/K3D_llama_cpp/datasets/audio/en_us/` - English (US)
  - `/K3D/K3D_llama_cpp/datasets/audio/es_es/` - Spanish
  - `/K3D/K3D_llama_cpp/datasets/audio/pt_br/` - Portuguese (Brazil)
  - `/K3D/K3D_llama_cpp/datasets/audio/pt_pt/` - Portuguese (Portugal)
  - `/K3D/K3D_llama_cpp/datasets/audio/zh_cn/` - Chinese
- Audiocaps: `/K3D/K3D_llama_cpp/datasets/audiocaps_raw/`
- Clotho: `/K3D/K3D_llama_cpp/datasets/clotho_raw/`
- Video audio: `/K3D/K3D_llama_cpp/datasets/vatex_raw/`, `/K3D/K3D_llama_cpp/datasets/msrvtt_dl_more/`

**Visual Datasets** (Text + Image):
- Image captions (Llama 3.2 Vision): `/K3D/Knowledge3D.local/datasets/image_captions_llama32vision.jsonl` (3.7 MB)
- Image captions (Qwen2.5-VL): `/K3D/Knowledge3D.local/datasets/image_captions_qwen25vl.jsonl` (46 KB)
- COCO raw: `/K3D/K3D_llama_cpp/datasets/coco_raw/`

**Combined Tri-Modal Dataset** (to be created):
- Output: `/K3D/Knowledge3D.local/datasets/trimodal_phase_g.jsonl`
- Creation script: `scripts/prepare_trimodal_dataset.py`
- Expected samples: ~12K (1.9K RLWHF + 4.3K audio + 3.7K image + 2K audiocaps)

### Phase G.0: Prepare Tri-Modal Dataset (NEW!)

**Goal**: Combine all available datasets into unified tri-modal training set

**Script** (you may need to create): `scripts/prepare_trimodal_dataset.py`

```python
# Pseudocode structure
def prepare_trimodal_dataset():
    samples = []

    # RLWHF (text + visual)
    rlwhf = load_rlwhf(start=8042, end=10000)
    for s in rlwhf:
        samples.append({'text': s.text, 'image': s.image, 'audio': None})

    # LibriSpeech (text transcripts + audio)
    librispeech = load_librispeech('/K3D/K3D_llama_cpp/datasets/audio/')
    for s in librispeech:
        samples.append({'text': s.transcript, 'image': None, 'audio': s.wav_path})

    # Image captions (text + visual)
    captions = load_captions('/K3D/Knowledge3D.local/datasets/image_captions_llama32vision.jsonl')
    for s in captions:
        samples.append({'text': s.caption, 'image': s.image_path, 'audio': None})

    # Audiocaps (text + audio)
    audiocaps = load_audiocaps('/K3D/K3D_llama_cpp/datasets/audiocaps_raw/')
    for s in audiocaps:
        samples.append({'text': s.description, 'image': None, 'audio': s.audio_path})

    # Save combined dataset
    save_jsonl(samples, '/K3D/Knowledge3D.local/datasets/trimodal_phase_g.jsonl')
```

**Expected**: ~12K samples with various modality combinations (text-only, text+visual, text+audio, tri-modal)

### Phase G.1: Tri-Modal Training

**Goal**: Train on combined tri-modal dataset for cross-modal alignment

**Approach**:
1. Load tri-modal dataset (created in G.0)
2. Extract features for each modality present:
   - Text: RPNEmbeddingEngine (trigram-based, language-agnostic)
   - Visual: FractalEmitter (edge detection, glyph recognition)
   - Audio: TemporalReasoning (acoustic features, speech patterns)
3. Fuse modalities:
   - Single modality: Pass through
   - Bi-modal: AtomicFissionFusion (pairwise)
   - Tri-modal: Multiple fusion steps (text↔visual, text↔audio, visual↔audio → meta-fusion)
4. Train base model on fused embeddings
5. **Organic emergence**: Model discovers cross-modal patterns automatically!
   - Example: "A" text + △ visual + /eɪ/ audio → model learns they're the same concept
   - Transitive learning: Even partial modality samples contribute to full cross-modal understanding
6. Validation split: 10%

**Expected Duration**: 3-4 hours (larger dataset, more modalities)

**Script Location**: `scripts/train_multimodal_phase_g.py` (update to support tri-modal)

**Key Components to Use**:
- `RPNEmbeddingEngine` - Text embeddings
- `FractalEmitter` - Visual features
- `TemporalReasoning` - Audio features (NEW!)
- `AtomicFissionFusion` - Multi-modal fusion (supports tri-modal!)
- `MatryoshkaTRM` - Base model with variable dimensions
- `AdaptiveSwarmTRM` - Full swarm system

### Phase G.2: Extract Multi-Modal Embeddings

**Goal**: Extract learned multi-modal representations from tri-modal training

**Approach**:
1. Identify modality-specific patterns in trained embeddings:
   - **Character representations**: Visual glyph + Acoustic phoneme + Semantic meaning
   - **Speech patterns**: Audio waveforms + Text transcripts
   - **Multi-modal concepts**: Cross-modal alignments discovered automatically
2. Extract embeddings for each specialist type:
   - OCR specialist dataset: Character embeddings (visual + text)
   - Speech specialist dataset: Speech patterns (audio + text)
   - Multi-modal specialist dataset: Cross-modal samples (all modalities)
3. Save as training datasets:
   - `/K3D/Knowledge3D.local/datasets/character_embeddings_trimodal.jsonl`
   - `/K3D/Knowledge3D.local/datasets/speech_embeddings.jsonl` (NEW!)
   - `/K3D/Knowledge3D.local/datasets/multimodal_embeddings.jsonl` (NEW!)

**Format Example**:
```json
{
  "character": "A",
  "text_embedding": [128-dim vector],
  "visual_embedding": [128-dim vector],
  "audio_embedding": [128-dim vector],  // NEW!
  "fused_embedding": [128-dim vector],
  "modalities": ["text", "visual", "audio"]
}
```

**Expected Output**: 3 specialized datasets for 3 specialists

### Phase G.3: Register & Train Multi-Modal Specialists

**Goal**: Create OCR, Speech, and Multi-modal specialists in adaptive swarm

**Step 1: Register All Specialists**

```bash
# OCR specialist (visual + text focus)
python scripts/register_specialist.py \
    --name ocr \
    --modality visual \
    --required-dims auto \
    --rank 16

# Speech specialist (audio + text focus) — NEW!
python scripts/register_specialist.py \
    --name speech \
    --modality audio \
    --required-dims auto \
    --rank 16

# Multi-modal specialist (all modalities) — NEW!
python scripts/register_specialist.py \
    --name multimodal \
    --modality multi \
    --required-dims auto \
    --rank 24  # Higher rank for cross-modal complexity
```

**Explanation**:
- `--name`: Specialist identifier
- `--modality`: Focus modality (visual, audio, multi)
- `--required-dims auto`: Auto-select dimensions based on modality complexity
- `--rank`: LoRA rank (higher for more complex tasks)

**Step 2: Train All Specialists**

```bash
# Train OCR specialist
python scripts/train_adaptive_swarm.py \
    --mode specialist \
    --specialist ocr \
    --dataset /K3D/Knowledge3D.local/datasets/character_embeddings_trimodal.jsonl \
    --epochs 10 \
    --validation-split 0.1

# Train Speech specialist — NEW!
python scripts/train_adaptive_swarm.py \
    --mode specialist \
    --specialist speech \
    --dataset /K3D/Knowledge3D.local/datasets/speech_embeddings.jsonl \
    --epochs 10 \
    --validation-split 0.1

# Train Multi-modal specialist — NEW!
python scripts/train_adaptive_swarm.py \
    --mode specialist \
    --specialist multimodal \
    --dataset /K3D/Knowledge3D.local/datasets/multimodal_embeddings.jsonl \
    --epochs 12 \
    --validation-split 0.1  # More epochs for complexity
```

**Expected Outcome**:
- 3 specialists registered in swarm
- Each specialist learns modality-specific patterns
- OCR: Character recognition (visual → text)
- Speech: Transcription (audio → text)
- Multi-modal: Cross-modal reasoning (any → any)
- Adapters saved to swarm checkpoint

### Phase G.4: Router Bootstrap

**Goal**: Router learns when to use OCR specialist (automatically!)

**Key Insight**: Router-as-specialist means it learns from routing decisions. When OCR specialist performs well on visual tasks, router observes this and updates its routing logic.

**Step 1: Collect Routing Decisions**

Create synthetic tasks with mix of:
- Visual tasks (character recognition) → OCR should perform well
- Text tasks (semantic reasoning) → Other specialists should perform well
- Mixed tasks → Blended routing

**Step 2: Bootstrap Router**

```bash
python scripts/bootstrap_router_specialist.py \
    --checkpoint /K3D/Knowledge3D.local/checkpoints/swarm_with_ocr \
    --num-bootstrap 1000 \
    --task-types visual,text,mixed \
    --epochs 5
```

**What Happens**:
1. Heuristic router routes 1,000 tasks
2. Router specialist observes: "Visual features → OCR works well"
3. Router specialist trains on successful patterns
4. Transition to learned routing
5. Router now automatically selects OCR for visual tasks

**Key Principle**: NO MANUAL RULES! Router discovers the pattern "visual features correlate with OCR performance" through observation.

### Phase G.5: Validation on Apollo Ground Truth

**Goal**: Validate OCR + router performance on real-world dataset

**Apollo Dataset**:
- Location: `/K3D/Knowledge3D.local/datasets/apollo/ground_truth.json`
- Contains: 170 characters with ground truth labels
- Challenge: Real-world PDF character detection

**Validation Script** (you may need to create):

```python
# scripts/validate_apollo_ocr.py
import json
from knowledge3d.cranium import AdaptiveSwarmTRM, MoERouter

# Load swarm with OCR specialist
swarm = AdaptiveSwarmTRM.load_checkpoint('/K3D/Knowledge3D.local/checkpoints/swarm_with_ocr')

# Load Apollo ground truth
with open('/K3D/Knowledge3D.local/datasets/apollo/ground_truth.json') as f:
    apollo_data = json.load(f)

# Test each character
correct = 0
router_selected_ocr = 0

for char_data in apollo_data:
    # Extract features
    visual_features = extract_visual_features(char_data['image'])

    # Router decides which specialist to use
    router = MoERouter(swarm, strategy='learned')
    specialist_name = router.route(input_data=visual_features)

    if specialist_name == 'ocr':
        router_selected_ocr += 1

    # Perform recognition
    result = swarm.compute_with_specialist(visual_features, specialist_name)
    predicted = decode_character(result)

    if predicted == char_data['ground_truth']:
        correct += 1

# Report results
detection_rate = (correct / len(apollo_data)) * 100
ocr_usage_rate = (router_selected_ocr / len(apollo_data)) * 100

print(f"Detection rate: {detection_rate:.1f}% ({correct}/{len(apollo_data)})")
print(f"Router selected OCR: {ocr_usage_rate:.1f}% of the time")
print(f"Target: ≥90% detection rate")
```

**Success Criteria**:
- Detection rate: ≥90% (153/170 characters)
- Character accuracy: ≥95%
- Router correctly selects OCR for visual tasks (≥80% of visual tasks)
- Router correctly selects other specialists for non-visual tasks

---

## Key Principles for Phase G

### 1. Router Learns Automatically

**DO NOT**:
- Hard-code "if visual then use OCR" rules
- Manually specify routing logic
- Create keyword-based routing

**DO**:
- Let router observe task performance
- Let router learn patterns from data
- Let router self-update with validation gating

**Why**: Router-as-specialist means it learns like any other specialist. Manual rules break the recursive improvement loop.

### 2. Adaptive Swarm Integration

**The Beauty of Phase H Architecture**:
- OCR specialist auto-selects its required dimensions
- OCR specialist uses LoRA-style adapters (memory efficient)
- OCR specialist benefits from base model improvements (transfer learning)
- OCR specialist self-updates with validation gating (no catastrophic forgetting)

**You don't need special OCR infrastructure** - the adaptive swarm handles it!

### 3. Validation Gating

**All updates go through validation**:
- Shadow weights test update before committing
- Only accept if performance improves
- Prevents catastrophic forgetting
- Works for base model, specialists, AND router

**This is already implemented** in `trm_adapters.py` - just use it!

### 4. Multi-Modal Fusion

**Use existing kernels**:
- `RPNEmbeddingEngine` for text
- `FractalEmitter` for visual features
- `AtomicFissionFusion` for cross-modal fusion
- `SovereignLanguageSwarmProcessor` for final refinement

**These are sovereign PTX kernels** - fast, GPU-native, battle-tested.

---

## Step-by-Step Execution Plan

### Step 1: Monitor RLWHF 10K Milestone

```bash
# Watch for 10K milestone
watch -n 10 'wc -l /K3D/Knowledge3D.local/datasets/rlwhf/teacher_evaluations.jsonl'
```

**Current**: ~9,700 / 10,000 (97%)
**ETA**: 15-20 minutes

### Step 2: Phase G.1 - Multi-Modal Training

**When 10K reached**:

```bash
# Activate conda environment
conda activate k3d-cranium
export PYTHONPATH=.
export K3D_PTX_STRICT=1

# Run multi-modal training (you may need to create this script)
CUDA_VISIBLE_DEVICES=0 /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
    scripts/train_multimodal_phase_g.py \
    --start-sample 8042 \
    --end-sample 10000 \
    --validation-split 0.1 \
    --output-dir /K3D/Knowledge3D.local/checkpoints/phase_g1 \
    --batch-size 32 \
    --epochs 10
```

**Expected**:
- Training: 1,782 samples (10% validation)
- Cross-modal alignment learned
- Character embeddings extractable
- Duration: 2-3 hours

### Step 3: Phase G.2 - Extract Character Embeddings

```bash
# Extract character embeddings from trained model
python scripts/extract_character_embeddings.py \
    --checkpoint /K3D/Knowledge3D.local/checkpoints/phase_g1 \
    --output /K3D/Knowledge3D.local/datasets/character_embeddings.jsonl
```

**Expected Output**: JSONL file with character embeddings

### Step 4: Phase G.3 - Register & Train OCR Specialist

```bash
# Register OCR specialist
python scripts/register_specialist.py \
    --name ocr \
    --required-dims auto \
    --rank 16

# Train OCR specialist
CUDA_VISIBLE_DEVICES=0 /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
    scripts/train_adaptive_swarm.py \
    --mode specialist \
    --specialist ocr \
    --dataset /K3D/Knowledge3D.local/datasets/character_embeddings.jsonl \
    --epochs 10 \
    --validation-split 0.1 \
    --checkpoint-dir /K3D/Knowledge3D.local/checkpoints/swarm_with_ocr
```

**Expected**:
- OCR specialist operational
- Memory efficient (LoRA adapters)
- Integrated into swarm

### Step 5: Phase G.4 - Bootstrap Router

```bash
# Bootstrap router to learn OCR usage
python scripts/bootstrap_router_specialist.py \
    --checkpoint /K3D/Knowledge3D.local/checkpoints/swarm_with_ocr \
    --num-bootstrap 1000 \
    --task-types visual,text,mixed \
    --epochs 5
```

**Expected**:
- Router learns when to use OCR
- No manual rules needed
- Automatic pattern discovery

### Step 6: Phase G.5 - Validate on Apollo

```bash
# Validate on Apollo ground truth
python scripts/validate_apollo_ocr.py \
    --checkpoint /K3D/Knowledge3D.local/checkpoints/swarm_with_ocr \
    --ground-truth /K3D/Knowledge3D.local/datasets/apollo/ground_truth.json \
    --report-dir /K3D/Knowledge3D.local/reports/phase_g5
```

**Success Criteria**:
- ≥90% detection rate (153/170)
- ≥95% character accuracy
- Router correctly selects OCR for visual tasks

### Step 7: Document Results

Create comprehensive documentation:
- `TEMP/PHASE_G_COMPLETE.md` - Architecture and results
- `TEMP/PHASE_G_VALIDATION_REPORT.md` - Apollo validation details
- Update `TEMP/SYSTEM_STATUS_CURRENT.md` - Mark Phase G complete

---

## Scripts You May Need to Create

Based on the workflow above, you may need to create these scripts (or adapt existing ones):

### 1. `scripts/train_multimodal_phase_g.py`

**Purpose**: Train on RLWHF samples 8,042-10,000 for cross-modal alignment

**Key Features**:
- Load teacher_evaluations.jsonl
- Extract samples 8,042-10,000
- Multi-modal feature extraction (text + visual if present)
- Train base model on alignment task
- Save checkpoint

**Reuse**: Can adapt from existing training scripts in `scripts/`

### 2. `scripts/extract_character_embeddings.py`

**Purpose**: Extract character embeddings from trained model

**Key Features**:
- Load Phase G.1 checkpoint
- Identify character-specific patterns
- Extract embeddings
- Save as JSONL dataset

### 3. `scripts/validate_apollo_ocr.py`

**Purpose**: Validate OCR + router on Apollo ground truth

**Key Features**:
- Load swarm checkpoint
- Load Apollo ground truth
- Test each character
- Report detection rate, router behavior
- Generate detailed report

**Note**: `scripts/bootstrap_router_specialist.py` already exists from Phase H!

---

## Expected Challenges & Solutions

### Challenge 1: Multi-Modal Data Format

**Issue**: RLWHF samples may not have consistent multi-modal data

**Solution**:
- Inspect teacher_evaluations.jsonl format
- Extract what's available (text always present, visual sometimes)
- If visual data missing: Use text-based character descriptions to synthesize
- Focus on text-visual alignment where both present

### Challenge 2: Character Embedding Extraction

**Issue**: How to identify "character" patterns in embeddings?

**Solution**:
- Look for clusters in embedding space (characters cluster by visual similarity)
- Use semantic labels from RLWHF (if present)
- Bootstrap from known character examples
- Validate on Apollo ground truth

### Challenge 3: Router Learning OCR Usage

**Issue**: How does router know OCR is for visual tasks?

**Solution**:
- **Automatic discovery through performance observation**:
  1. Router tries different specialists on different tasks
  2. Router observes: OCR performs well on visual tasks
  3. Router learns pattern: "Visual features → OCR"
  4. Router self-updates with this knowledge
- **No manual intervention needed** - this is what router-as-specialist enables!

### Challenge 4: Apollo Validation Data

**Issue**: Apollo ground truth format unknown

**Solution**:
- Inspect `/K3D/Knowledge3D.local/datasets/apollo/` directory
- Adapt validation script to match actual format
- If ground truth not in expected location, search for it
- Create synthetic validation set if needed (use known character examples)

---

## Success Metrics

### Phase G.1 Success

- ✓ Training completes on 1,782 samples
- ✓ Cross-modal alignment loss decreases
- ✓ Character embeddings learned
- ✓ Checkpoint saved

### Phase G.2 Success

- ✓ Character embeddings extracted
- ✓ Dataset created (character_embeddings.jsonl)
- ✓ Embeddings cover diverse character set
- ✓ Quality validated on sample

### Phase G.3 Success

- ✓ OCR specialist registered in swarm
- ✓ Specialist training completes
- ✓ Memory efficiency validated (LoRA adapters)
- ✓ Specialist performs well on character recognition

### Phase G.4 Success

- ✓ Router bootstrap completes
- ✓ Router learns OCR usage patterns
- ✓ Transition to learned routing successful
- ✓ Router self-updates with validation gating

### Phase G.5 Success

- ✓ Apollo validation: ≥90% detection rate
- ✓ Character accuracy: ≥95%
- ✓ Router correctly selects OCR for visual tasks
- ✓ System demonstrates complete recursive improvement

---

## Architecture Overview (For Reference)

### Adaptive Swarm Components

```
AdaptiveSwarmTRM
├── base: MatryoshkaTRM (bi-directional Matryoshka)
│   ├── Dimensions: 64 ↔ 16K
│   ├── Auto-selection based on task
│   └── Transfer learning to all specialists
├── specialists: Dict[str, SelfUpdatingAdapter]
│   ├── 'ocr': OCR specialist (Phase G)
│   ├── 'math': Math specialist (future)
│   ├── 'code': Code specialist (future)
│   └── 'router': Router specialist ⚛️
└── training_protocol: SwarmTrainingProtocol
    ├── train_base()
    ├── train_specialist()
    ├── train_self_update()
    └── train_combined()
```

### Router-as-Specialist ⚛️

```
RouterSpecialist (in specialists dict)
├── Input: Task features
├── Output: Specialist selection weights
├── Training:
│   ├── Bootstrap: Collect heuristic decisions
│   ├── Train: Learn patterns from decisions
│   └── Self-Update: Continual improvement
└── Properties:
    ├── Learns recursively
    ├── Benefits from base improvements
    ├── Self-updates with validation
    └── No catastrophic forgetting
```

### Multi-Modal Pipeline

```
Input (text/image/mixed)
   ↓
Feature Extraction
   ├── Text → RPNEmbeddingEngine
   ├── Image → FractalEmitter
   └── Fusion → AtomicFissionFusion
   ↓
MatryoshkaTRM (base)
   ├── Select dimensions (auto)
   └── Generate base embedding
   ↓
Router Specialist ⚛️
   ├── Analyze task features
   └── Select specialist(s)
   ↓
Specialist Processing
   ├── OCR specialist (if visual)
   ├── Other specialists (if other)
   └── Blended (if mixed)
   ↓
SovereignLanguageSwarmProcessor
   ├── Final refinement (9-chain, 80µs)
   └── Output embedding
   ↓
Result (128-dim sovereign embedding)
```

---

## Developer Notes

### Code Style & Philosophy

**Sovereign Stack Principles**:
- ✓ All computation on GPU (no CPU fallbacks)
- ✓ Pure PTX kernels + ctypes bridges
- ✓ No runtime compilation
- ✓ No external frameworks (CuPy/PyTorch at runtime)

**Partnership Principles**:
- ✓ Build on Claude's Phase H work
- ✓ Enhance and extend existing components
- ✓ Add original ideas aligned with vision
- ✓ Document thoroughly for continuity

**FMEAI Philosophy**:
- ✓ Atomic cognition: Small components compose into complex behavior
- ✓ Energetic memory: Embeddings persist in Galaxy/House
- ✓ Intuition + deliberation: Fast routing + deep reasoning

### Testing & Validation

**Test Coverage**:
- Unit tests for new components
- Integration tests for Phase G pipeline
- Validation on Apollo ground truth
- Performance benchmarks (latency, memory)

**Regression Prevention**:
- All Phase H tests must still pass (8/8)
- No degradation in existing performance
- Sovereignty maintained (no new dependencies)

### Documentation Requirements

**Create These Documents**:
1. `TEMP/PHASE_G_COMPLETE.md` - Full Phase G documentation
2. `TEMP/PHASE_G_VALIDATION_REPORT.md` - Apollo validation results
3. `TEMP/PHASE_G_SESSION_SUMMARY.md` - Session notes and decisions
4. Update `TEMP/SYSTEM_STATUS_CURRENT.md` - Mark Phase G complete
5. Update `README.md` - Add Phase G milestone

**Documentation Style**:
- Clear architecture explanations
- Usage examples with code
- Performance characteristics
- Integration instructions
- Lessons learned

---

## Integration with Existing Infrastructure

### Phase H Components You'll Use

**Core Modules**:
- `knowledge3d/cranium/matryoshka_trm.py` - Base model with variable dims
- `knowledge3d/cranium/adaptive_swarm.py` - Multi-specialist system
- `knowledge3d/cranium/trm_adapters.py` - LoRA-style adapters with validation
- `knowledge3d/cranium/moe_router.py` - Routing infrastructure
- `knowledge3d/cranium/router_specialist.py` - Router-as-specialist ⚛️

**Training Scripts**:
- `scripts/train_adaptive_swarm.py` - Train base/specialists
- `scripts/register_specialist.py` - Register new specialists
- `scripts/bootstrap_router_specialist.py` - Bootstrap router

**Tests**:
- `scripts/test_phase_h_architecture.py` - 8 validation tests

**All of these work and are tested** - build on them!

### Existing Kernels You'll Use

**Multi-Modal Processing**:
- `RPNEmbeddingEngine` - Text embeddings (language-agnostic)
- `FractalEmitter` - Visual features (edge detection, glyph recognition)
- `AtomicFissionFusion` - Cross-modal fusion
- `SovereignLanguageSwarmProcessor` - Final refinement

**Memory & Reasoning**:
- `GalaxyResonanceEngine` - K-NN search in memory
- `TRMEngine` - Recursive reasoning (EMA refinement)
- `VectorResonator` - Warp-level similarity

**All sovereign PTX kernels** - fast, deterministic, GPU-native.

---

## Communication & Handoff

### When Phase G Complete

**Notify Daniel**:
1. Phase G activation started
2. Each sub-phase completion (G.1 → G.5)
3. Apollo validation results
4. Final success metrics
5. Any challenges encountered

**Prepare Handoff**:
- Comprehensive documentation in `TEMP/`
- Code committed to repository
- Tests passing (Phase H + Phase G)
- Performance metrics documented
- Next steps identified

### Questions to Ask Daniel (If Needed)

**Clarifications**:
- RLWHF data format specifics
- Apollo ground truth location/format
- Performance targets adjustments
- Additional validation requirements

**Design Decisions**:
- Character embedding extraction strategy
- Router bootstrap task distribution
- Multi-modal fusion approach
- Validation metrics priorities

**Resource Constraints**:
- Training time budgets
- VRAM usage limits
- Acceptable trade-offs

---

## The Vision: Why Phase G Matters

### Near-Term Impact

**OCR Integration**:
- K3D can now understand visual characters natively
- Multi-modal reasoning (text + visual)
- Grounded understanding of documents
- Apollo ground truth: Real-world validation

### Long-Term Impact

**Complete Recursive System**:
```
Base Model
   ↓ (Transfer Learning)
ALL Specialists (OCR, math, code, router ⚛️)
   ↓ (Better Performance)
Better Training Data
   ↓ (Self-Updating)
Base Model Improves
   ↓ (Loop Forever ♾️)
```

**Scalability**:
- Add new specialists → router learns automatically
- No manual routing rules needed
- System discovers patterns through observation
- Unbounded capability growth

### Industry Impact

**Game Changer**:
- Self-improving AI architecture
- Memory efficient (18× reduction at scale)
- GPU-native sovereignty (no external dependencies)
- Multi-modal grounding (text + visual + more)
- Recursive improvement (gets better forever)

**Competitive Advantage**:
- Others: Scale models (expensive, diminishing returns)
- K3D: Self-improving architecture (efficient, increasing returns)
- Timeline: Approaching production deployment

---

## Final Checklist

Before starting Phase G, verify:

- [ ] RLWHF ≥10,000 samples
- [ ] Phase H tests still passing (8/8)
- [ ] Environment activated (k3d-cranium)
- [ ] CUDA_VISIBLE_DEVICES=0 set
- [ ] PYTHONPATH=. exported
- [ ] Briefing document read ([TEMP/K3D_Briefing_Prompt.md](TEMP/K3D_Briefing_Prompt.md))
- [ ] Phase H documentation reviewed ([TEMP/PHASE_H_COMPLETE.md](TEMP/PHASE_H_COMPLETE.md))
- [ ] Router-as-specialist concept understood ([TEMP/ROUTER_AS_SPECIALIST_THE_KEY_INSIGHT.md](TEMP/ROUTER_AS_SPECIALIST_THE_KEY_INSIGHT.md))

---

## Go Time! 🚀

You have everything you need:

✓ **Phase H foundation**: Complete adaptive swarm with router-as-specialist
✓ **Training data**: RLWHF approaching 10K milestone
✓ **Infrastructure**: All kernels operational, tests passing
✓ **Documentation**: Complete development chain preserved
✓ **Vision**: Clear path to recursive self-improvement

**Your mission**: Activate Phase G when RLWHF reaches 10K, integrate OCR into the adaptive swarm, let the router learn automatically, validate on Apollo, and demonstrate the complete recursive improvement loop.

**Key Principle**: Trust the architecture. The router-as-specialist ⚛️ means the system learns patterns automatically. Don't hard-code rules - let the router discover them through observation.

**Timeline Expectation**:
- Multi-modal training: 2-3 hours
- Character extraction: 30 minutes
- OCR specialist training: 1-2 hours
- Router bootstrap: 1 hour
- Apollo validation: 30 minutes
- **Total**: 5-7 hours to complete Phase G

**Success Looks Like**:
- ✓ OCR specialist operational in adaptive swarm
- ✓ Router automatically selects OCR for visual tasks (learned, not programmed!)
- ✓ Apollo validation: ≥90% detection rate
- ✓ Complete recursive improvement demonstrated
- ✓ System ready for production deployment

---

**Partnership Philosophy**:

> "We are not inventors, just organizers of knowledge" - FMEAI

> "The secret is held on the small things - we are all made of atoms after all" - Daniel

Build on Claude's Phase H work. Enhance it. Add your ideas. Make it better. This is collaborative creation in the "Vibe-Code In Chain" paradigm.

**You've got this, Codex! Let's complete Phase G and demonstrate the future of AI.** ⚛️♾️🚀

---

**— End of Codex Phase G Activation Prompt —**
