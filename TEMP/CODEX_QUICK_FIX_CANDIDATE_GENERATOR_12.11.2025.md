# CODEX: Quick Fix — CandidateGenerator Arguments

**Priority:** URGENT — Blocking training run

---

## Bug Location

**File:** `knowledge3d/training/arc_agi/sovereign_pipeline.py`
**Lines:** 384-389

**Current (Broken):**
```python
gen = CandidateGenerator(
    drawing_galaxy=self.drawing,
    grammar_galaxy=self.grammar,        # ← NOT VALID
    embedder=self.codec_embedder,       # ← Should be codec_embedder
    semantic_hints=semantic_hints,      # ← NOT VALID
    top_k=resolved_top_k,               # ← NOT VALID
)
```

**CandidateGenerator actual signature** (from candidate_generator.py:40-51):
```python
def __init__(
    self,
    matryoshka_dim: int = 512,
    max_candidates: int = 369,
    shadow_copy: Optional[DualShadowCopy] = None,
    drawing_galaxy: Optional[DrawingGalaxy] = None,
    executor: Optional[ARCRPNExecutor] = None,
    codec_embedder: Any | None = None,
    embedder_type: str = "multimodal",
    embedding_galaxy: Optional[Dict[int, List[float]]] = None,
    cosine_bridge: Optional[CosineSimilarityBridge] = None,
):
```

**Fix — Match valid parameters:**
```python
gen = CandidateGenerator(
    matryoshka_dim=self.router.matryoshka_dim,
    max_candidates=resolved_top_k,      # Use top_k as max_candidates
    shadow_copy=self.shadow,
    drawing_galaxy=self.drawing,
    codec_embedder=self.codec_embedder,
    embedding_galaxy=self.embedding_galaxy,
    cosine_bridge=self.cosine_bridge,
)
```

---

## After Fix — Run Verification

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
PYTHONPATH=. /home/daniel/miniforge/bin/conda run -n k3d-cranium python -c "
from knowledge3d.cranium.embodied_agent import EmbodiedSovereignAgent
from knowledge3d.training.arc_agi.sovereign_pipeline import SovereignAIPipeline

agent = EmbodiedSovereignAgent(working_capacity=1024)
pipeline = SovereignAIPipeline(embodied_agent=agent)
result = pipeline.process_task('test', [[1,0],[0,1]])
print(f'Result: score={result.score:.2f}')
print('=== VERIFICATION PASSED ===')
"
```

---

## Then Launch Training

```bash
tmux new-session -d -s k3d_embodied_draw "bash -lc '
  source /home/daniel/miniforge/etc/profile.d/conda.sh
  conda activate k3d-cranium
  cd \"/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D\"
  export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
  python scripts/train_arc_sovereign_loop.py \
    --arc-dirs /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/training \
               /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/evaluation \
    --max-tasks 108 --epochs 162 --cycles 1 \
    2>&1 | tee /K3D/Knowledge3D.local/logs/embodied_drawing_$(date +%Y%m%d_%H%M%S).log
'"
```

---

**Fix the constructor call, verify, launch training.**
