# Codex GPU Orchestration Brief – K3D Environment Pattern

**Date**: 2025-10-16
**From**: Claude (Architect) + Daniel (Systems Engineer)
**To**: Codex (Implementation Specialist)
**Context**: Phase B RPN embeddings delivered → Now run ingestion pipelines with GPU access

---

## Critical Environment Pattern

### Why Standard pytest/python Won't Work

**Problem**: The K3D Conda environment at `/K3D/Knowledge3D.local/envs/k3d-cranium` has GPU access **only** when launched with specific orchestration.

**Root Cause**:
- The environment binary at `/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3` is configured with CUDA libraries
- GPU visibility requires `CUDA_VISIBLE_DEVICES=0` to be set **before** process launch
- Interactive shell sessions may not inherit the correct CUDA environment variables
- Direct `pytest` or `python` commands often fail with "CUDA not available" or "no GPU detected"

**Solution**: Use **tmux orchestration** pattern that Daniel has validated for all GPU workflows.

---

## The Correct GPU Launch Pattern

### Pattern 1: Single GPU Test/Script

**For tests** (`pytest`):
```bash
# Create tmux session with GPU access
tmux new-session -d -s test_rpn_gpu "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 -m pytest tests/test_rpn_embeddings.py -xvs; exec bash'"

# Attach to watch live output
tmux attach -t test_rpn_gpu
```

**For scripts** (batch ingestion):
```bash
# Create tmux session for long-running ingestion
tmux new-session -d -s corpus_ingest "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 scripts/ingest_full_corpus.py 2>&1 | tee /K3D/Knowledge3D.local/logs/corpus_ingest.log; exec bash'"

# Check progress (detached)
tmux capture-pane -t corpus_ingest -p

# Attach to interact
tmux attach -t corpus_ingest
```

### Pattern 2: Interactive GPU Session

**For debugging/exploration**:
```bash
# Launch interactive Python REPL with GPU
tmux new-session -d -s gpu_repl "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3; exec bash'"

tmux attach -t gpu_repl
```

**For Jupyter notebooks** (if needed):
```bash
tmux new-session -d -s jupyter_gpu "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser; exec bash'"

tmux attach -t jupyter_gpu
```

---

## Why Each Component Matters

### 1. `CUDA_VISIBLE_DEVICES=0`
- **What**: Environment variable that tells CUDA runtime which GPU(s) to use
- **Why**: System may have multiple GPUs or need explicit device selection
- **Value**: `0` = first GPU (RTX 3060 in this system)

### 2. Full Python Path: `/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3`
- **What**: Direct path to Conda environment Python binary
- **Why**: Ensures CUDA libraries and K3D packages are in the environment
- **Don't use**: `python3` (system), `python` (ambiguous), or `conda run` (adds overhead)

### 3. `export PYTHONPATH=.`
- **What**: Adds current directory to Python module search path
- **Why**: Allows `import knowledge3d` from repo root
- **Critical**: Must be set **before** running Python

### 4. `tmux` Session Management
- **What**: Terminal multiplexer for persistent sessions
- **Why**:
  - Sessions survive SSH disconnects
  - Logs persist even if connection drops
  - Can monitor long-running jobs remotely
  - GPU context remains active throughout session
- **Usage**:
  - `tmux new-session -d -s <name>` = create detached session
  - `tmux attach -t <name>` = attach to session
  - `tmux capture-pane -t <name> -p` = view output without attaching
  - `tmux ls` = list active sessions
  - `Ctrl+B, D` = detach (keep session running)
  - `tmux kill-session -t <name>` = terminate session

### 5. `exec bash` at End
- **What**: Keeps tmux session alive after command completes
- **Why**: Allows inspection of final state, error messages, or manual debugging
- **Alternative**: Omit to auto-close on completion (for automated workflows)

---

## Common Workflows for Phase B

### Workflow 1: Test RPN Embeddings (GPU Required)

```bash
# Single test
tmux new-session -d -s test_rpn "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 -m pytest tests/test_rpn_embeddings.py -xvs; exec bash'"
tmux attach -t test_rpn
```

**Expected Output**:
- RPN embeddings initialized
- Trigram vocab built
- Semantic clustering validated
- All tests pass

### Workflow 2: Test Lexicon Ingestion (GPU Required)

```bash
# WordNet sample ingestion
tmux new-session -d -s test_lexicon "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 -m pytest tests/test_lexicon_ingestion.py -xvs --tb=short; exec bash'"
tmux attach -t test_lexicon
```

**Expected Output**:
- WordNet synsets loaded
- RPN embeddings generated
- Swarm refinement at 80µs
- JSON artifacts saved to House

### Workflow 3: Test PDF Ingestion (GPU Required)

```bash
# "How to think" folder ingestion
tmux new-session -d -s test_pdf "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 -m pytest tests/test_pdf_ingestion.py -xvs --tb=short; exec bash'"
tmux attach -t test_pdf
```

**Expected Output**:
- PDFs extracted (PyPDF2)
- Sentences embedded (RPN)
- Swarm processing (<5s per document)
- Summaries saved to House

### Workflow 4: Font Glyph Harvesting (GPU Required)

```bash
# DejaVu fonts sample
tmux new-session -d -s test_fonts "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 -m pytest tests/test_font_harvester.py -xvs --tb=short; exec bash'"
tmux attach -t test_fonts
```

**Expected Output**:
- Fonts rendered (PIL)
- Glyphs processed (FractalEmitter)
- Multi-modal fusion (AtomicFissionFusion)
- Visual-text pairs saved

### Workflow 5: Full Corpus Ingestion (Long-Running)

```bash
# Background batch job with logging
tmux new-session -d -s full_corpus "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 scripts/ingest_full_corpus.py 2>&1 | tee /K3D/Knowledge3D.local/logs/full_corpus_$(date +%Y%m%d_%H%M%S).log; exec bash'"

# Monitor progress (without attaching)
tmux capture-pane -t full_corpus -p | tail -20

# Check GPU usage
nvidia-smi

# Attach to interact
tmux attach -t full_corpus
```

**Expected Output**:
- Priority folders processed sequentially
- RPN embeddings saved after each folder
- 327 PDFs → Galaxy positions
- Total time, VRAM logged

---

## Monitoring GPU During Ingestion

### Real-Time GPU Stats

**Watch GPU usage** (separate terminal):
```bash
watch -n 1 nvidia-smi
```

**Key metrics**:
- **GPU Util**: Should be >80% during swarm processing
- **Memory-Usage**: Target <8GB (RTX 3060 limit)
- **Temperature**: <85°C is healthy
- **Power**: ~170W typical for RTX 3060

### Log Output Inspection

**Tail live logs**:
```bash
tail -f /K3D/Knowledge3D.local/logs/full_corpus_*.log
```

**Search for errors**:
```bash
grep -i "error\|fail\|cuda" /K3D/Knowledge3D.local/logs/full_corpus_*.log
```

**Check VRAM spikes**:
```bash
grep -i "vram\|memory" /K3D/Knowledge3D.local/logs/full_corpus_*.log
```

---

## Troubleshooting Common Issues

### Issue 1: "CUDA not available" Error

**Symptoms**:
```
RuntimeError: CUDA not available
```

**Cause**: Missing `CUDA_VISIBLE_DEVICES=0` or wrong Python binary

**Fix**:
```bash
# Verify CUDA is visible in environment
tmux new-session -d -s check_cuda "bash -c 'export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 -c \"import cupy; print(cupy.cuda.runtime.getDeviceCount())\"; exec bash'"
tmux attach -t check_cuda
# Expected: 1
```

### Issue 2: "Module not found: knowledge3d"

**Symptoms**:
```
ModuleNotFoundError: No module named 'knowledge3d'
```

**Cause**: Missing `PYTHONPATH=.` or wrong working directory

**Fix**:
```bash
# Verify PYTHONPATH is set correctly
cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D
export PYTHONPATH=.
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 -c "import knowledge3d; print(knowledge3d.__file__)"
# Expected: /mnt/arquivos/.../Knowledge3D/knowledge3d/__init__.py
```

### Issue 3: OOM (Out of Memory) on GPU

**Symptoms**:
```
cupy.cuda.memory.OutOfMemoryError: Out of memory allocating 2,147,483,648 bytes
```

**Cause**: Processing too many items in parallel, or VRAM leak

**Fix**:
```python
# In ingestion scripts, add batch limits:
max_sentences_per_pdf = 100  # Reduce if OOM
batch_size = 10  # Process PDFs in smaller batches

# Use OOMSpillManager (already in swarm)
from knowledge3d.cranium.resource_safety import OOMSpillManager
oom_guard = OOMSpillManager(max_vram_gb=8.0)
```

### Issue 4: tmux Session Not Starting

**Symptoms**:
```
no server running on /tmp/tmux-1000/default
```

**Cause**: tmux server not initialized

**Fix**:
```bash
# Start tmux server
tmux start-server

# Verify
tmux ls
# Expected: (empty list) or list of active sessions
```

---

## Phase B Execution Checklist

### Pre-Flight Checks
- [ ] CUDA visible: `nvidia-smi` shows RTX 3060
- [ ] Conda env active: `/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 --version`
- [ ] tmux installed: `tmux -V` (should be ≥3.0)
- [ ] Working directory: `/mnt/arquivos/.../GitHub/Knowledge3D`
- [ ] Logs directory exists: `mkdir -p /K3D/Knowledge3D.local/logs/`

### Test Sequence (GPU Required)
1. **RPN Embeddings** (foundation):
   ```bash
   tmux new-session -d -s test_rpn "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 -m pytest tests/test_rpn_embeddings.py -xvs; exec bash'"
   ```

2. **Lexicon Ingestion** (WordNet sample):
   ```bash
   tmux new-session -d -s test_lexicon "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 -m pytest tests/test_lexicon_ingestion.py -xvs; exec bash'"
   ```

3. **PDF Ingestion** ("How to think"):
   ```bash
   tmux new-session -d -s test_pdf "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 -m pytest tests/test_pdf_ingestion.py -xvs; exec bash'"
   ```

4. **Font Harvesting** (DejaVu sample):
   ```bash
   tmux new-session -d -s test_fonts "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 -m pytest tests/test_font_harvester.py -xvs; exec bash'"
   ```

### Full Corpus Ingestion (After Tests Pass)
```bash
# Launch background job with logging
tmux new-session -d -s full_corpus "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 scripts/ingest_full_corpus.py 2>&1 | tee /K3D/Knowledge3D.local/logs/full_corpus_$(date +%Y%m%d_%H%M%S).log; exec bash'"

# Monitor without blocking
watch -n 5 "tmux capture-pane -t full_corpus -p | tail -20"
```

---

## Success Metrics

**After full ingestion**:
- [ ] 327 PDFs processed
- [ ] RPN vocab: ~100K trigrams learned
- [ ] VRAM peak: <8GB
- [ ] Average latency: <5s per document
- [ ] House artifacts:
  - `/K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl`
  - `/K3D/Knowledge3D.local/house_zone7/lexicons/wordnet_en.json`
  - `/K3D/Knowledge3D.local/house_zone7/documents/{folder}_summary.json`
  - `/K3D/Knowledge3D.local/house_zone7/fonts/font_glyphs.json`

---

## Final Notes

**Why This Pattern Matters**:
- GPU access is **not automatic** in K3D environment
- tmux orchestration ensures CUDA context survives disconnects
- `CUDA_VISIBLE_DEVICES=0` is the gateway to RTX 3060
- Logs persist even if SSH session dies

**Best Practices**:
1. **Always use tmux** for GPU jobs (even short tests)
2. **Always set CUDA_VISIBLE_DEVICES=0** before Python launch
3. **Always use full Python path** (`/K3D/.../bin/python3`)
4. **Always log to /K3D/Knowledge3D.local/logs/** for persistence
5. **Monitor GPU** with `nvidia-smi` during ingestion

**Codex, you now have**:
- ✅ Complete RPN embedding sovereignty
- ✅ Lexicon/PDF/font ingestors ready
- ✅ Batch corpus pipeline implemented
- ✅ GPU orchestration pattern documented

**Next step**: Run the test sequence with tmux orchestration, validate GPU access, then launch full corpus ingestion.

**Go feed the genius mind, Codex. The sovereign substrate awaits.** 🚀

---

**Signed**:
Claude (Architect) + Daniel (Systems Engineer)
2025-10-16
