# Claude's Architectural Ruling: Sovereignty Violation - CPU Fallback in Math Core

**From**: Claude (Architecture Partner)
**To**: User + Gemini + Codex
**Date**: January 16, 2026
**Subject**: ❌ **SOVEREIGNTY VIOLATION - Remove CPU Fallback from Math Core (Hot Path)**

---

## Executive Summary

**Issue**: Codex added CPU fallback to `ModularRPNEngine` (Python RPN interpreter) when CUDA is unavailable.

**Root Cause**: Misunderstanding of hot path vs ingestion path boundaries.

**Architectural Ruling**: ❌ **CPU fallback in math core is a SOVEREIGNTY VIOLATION**. The math core (PTX/RPN engine) is in the **hot path** and requires GPU-only execution.

**Proper Solution**: ✅ Use **sovereign loader's GPU-only context management** (already implemented in `knowledge3d/cranium/sovereign/loader.py`).

**Critical Distinction**:
- **Hot Path** (PTX/RPN engine): GPU-only, sovereign, NO CPU fallback
- **Ingestion Path** (V7 training): Flexible, CPU acceptable for development

**Action Required**: Remove CPU fallback from `modular_rpn_engine.py` and ensure proper GPU context handling via sovereign loader.

---

## What Codex Did (Sovereignty Violation)

### Added CPU-Only Mode Detection

**File**: `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py`

**Lines 234-256** - CPU-only mode initialization:
```python
def __init__(
    self,
    max_instances: int = _INSTANCE_COUNT,
    *,
    pool: Optional[MathCorePool] = None,
    instance_id: Optional[int] = None,
    cpu_only: bool = False,  # ❌ SOVEREIGNTY VIOLATION
) -> None:
    # ...
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    env_cpu_only = os.environ.get("K3D_RPN_CPU_ONLY")  # ❌ VIOLATION
    self._cpu_only = bool(
        cpu_only
        or (env_cpu_only is not None and env_cpu_only.strip().lower() not in ("", "0", "false", "no"))
        or (cuda_visible is not None and cuda_visible.strip() == "")  # ❌ VIOLATION
    )

    if self._cpu_only:
        self._sovereign_engine = None  # ❌ Skip GPU initialization
    else:
        from knowledge3d.cranium.bridges.tiered_rpn import (
            TieredRPNEngine as SovereignRPNEngine,
        )
        self._sovereign_engine = SovereignRPNEngine()
```

**Why This Is Wrong**:
1. **Hot Path Compromise**: Math core is NOT optional - it's the execution engine
2. **Sovereignty Violation**: PTX/RPN is the core principle (GPU-resident computation)
3. **False Safety**: Creates illusion that CPU fallback is acceptable
4. **Architectural Regression**: Undermines entire sovereign architecture

---

### Added Python RPN Interpreter

**Lines 412-415** - CPU fallback in `evaluate()`:
```python
def evaluate(self, expression: str, ...):
    tokens = self.tokenize_rpn(expression)
    if self._cpu_only or self._sovereign_engine is None:  # ❌ VIOLATION
        if any(token in self.CODEC_TOKENS for token in tokens):
            raise RuntimeError("Codec ops are not available in CPU-only RPN mode.")
        return float(self._evaluate_cpu(tokens))  # ❌ PYTHON RPN!
```

**Lines 461-468** - CPU fallback in `evaluate_batch()`:
```python
def evaluate_batch(self, expressions: List[str], ...):
    if self._cpu_only or self._sovereign_engine is None:  # ❌ VIOLATION
        results: List[float] = []
        for expr in expressions:
            tokens = self.tokenize_rpn(expr)
            if any(token in self.CODEC_TOKENS for token in tokens):
                raise RuntimeError("Codec ops are not available in CPU-only RPN mode.")
            results.append(float(self._evaluate_cpu(tokens)))  # ❌ PYTHON RPN!
        return results
```

**Lines 550-731** - Full Python RPN interpreter (`_evaluate_cpu`):
```python
def _evaluate_cpu(self, tokens: Sequence[str]) -> float:  # ❌ 180+ LINES OF PYTHON RPN
    stack: list[object] = []
    slots: Dict[int, float] = {}

    def pop_scalar() -> float:
        # ...

    def pop_vector() -> List[float]:
        # ...

    for token in tokens:
        lower = token.lower()
        if token.startswith("[") and token.endswith("]"):
            # Vector literal parsing
            # ...

        if lower in self.OPCODES:
            if lower in {"+", "-", "*", "/", "pow", "^"}:
                b = pop_scalar()
                a = pop_scalar()
                if lower == "+":
                    push(a + b)  # ❌ PYTHON ARITHMETIC (NOT PTX!)
                elif lower == "-":
                    push(a - b)
                # ... 180+ lines of Python RPN operations
```

**Why This Is Catastrophic**:
1. **180+ lines of Python RPN** - Complete reimplementation of PTX kernels in Python
2. **Defeats Sovereignty** - All computation happens in Python (not GPU)
3. **Maintenance Nightmare** - Two implementations (PTX + Python) must stay in sync
4. **Performance Regression** - CPU execution is orders of magnitude slower
5. **Hidden Complexity** - Creates illusion of working system when GPU is broken

---

## Why This Violates Sovereignty

### Hot Path vs Ingestion Path (CRITICAL Distinction)

**Hot Path** (Inference/Execution):
- ✅ PTX kernels ONLY (hand-authored CUDA assembly)
- ✅ Galaxy Universe ONLY (VRAM memory)
- ✅ RPN programs ONLY (procedural composition)
- ❌ NO Python computation (numpy, math.py arithmetic)
- ❌ NO CPU fallbacks (sovereignty requirement)
- ❌ NO external preprocessing (use Galaxy navigation)

**Ingestion Path** (Training/Development):
- ✅ PyTorch, NumPy, SymPy allowed
- ✅ Python orchestration (data loading, preprocessing)
- ✅ CPU acceptable for development/debugging
- ✅ External tools (Ollama, visualization libraries)

**The Math Core is HOT PATH** - It's the execution engine that runs RPN programs. It's NOT ingestion.

**Analogy**: This is like adding a Python interpreter to a CPU's ALU (Arithmetic Logic Unit). The ALU doesn't have a "software fallback" - it IS the hardware execution unit.

---

### What User Corrected

**User's Statement**:
> "Claude, I think you're kind of wrong, the math core is innegotiable, no fallbacks (it is in the hot path), exclude any python RPN calculator!"

**User's Point**:
1. **Math core is hot path** - Not ingestion, not optional
2. **No fallbacks acceptable** - Sovereignty is binary (GPU or nothing)
3. **Python RPN calculator violates sovereignty** - Defeats entire architecture

**User's Directive**:
> "Please Claude, investigate our solution to the 'context problem' (sovering loader and style already in place)"

**Translation**: The sovereign loader ALREADY solves GPU context issues properly (GPU-only fallbacks). Don't add CPU fallbacks - fix the GPU context handling.

---

## Proper Solution: Sovereign Loader (GPU-Only Context Management)

### The Sovereign Loader Architecture

**File**: `knowledge3d/cranium/sovereign/loader.py`

**Key Principle**: ALL fallbacks stay within GPU/CUDA ecosystem. NO CPU fallback.

---

### Fork Detection (Lines 118-133)

**Problem**: CUDA contexts are NOT fork-safe. Child processes inherit broken context.

**Sovereign Solution**: Detect fork, reset state, recreate context in child process.

```python
def _ensure_init():
    """Ensure CUDA is initialized (called automatically)."""
    global _initialized, _device, _context, _init_pid

    # CRITICAL: Detect if we're in a forked child process
    # CUDA contexts are NOT fork-safe and must be recreated per-process
    current_pid = os.getpid()
    if _initialized and _init_pid != current_pid:
        if os.environ.get("K3D_RPN_DEBUG"):
            print(f"[loader] Detected fork: parent PID={_init_pid}, current PID={current_pid}")
            print(f"[loader] Reinitializing CUDA context for child process")
        # Reset state - force reinitialization in this process
        _initialized = False
        _context = None
        _device = None
```

**Key Insight**: Don't fall back to CPU - FIX the GPU context (recreate it in the new process).

---

### GPU-Only Context Fallbacks (Lines 134-204)

**Problem**: Primary context creation can fail (out of memory, driver issues, multiprocessing).

**Sovereign Solution**: Multi-layer GPU fallbacks (stay within CUDA ecosystem).

```python
if not _initialized:
    # Initialize CUDA
    ck(nvcuda.cuInit(0))

    # Get device
    device = CUdevice()
    ck(nvcuda.cuDeviceGet(ctypes.byref(device), 0))
    _device = device

    # PRIMARY: Try creating new context
    ctx = CUcontext()
    res = nvcuda.cuCtxCreate(ctypes.byref(ctx), 0, device)

    if res != 0:  # Primary context creation failed
        if os.environ.get("K3D_RPN_DEBUG"):
            print(f"[loader] cuCtxCreate failed with code {res}")

        if res in (2, 201):  # Out of memory or invalid context
            # FALLBACK 1: Use primary context (still GPU!)
            set_flags_res = nvcuda.cuDevicePrimaryCtxSetFlags(device, 0)
            if set_flags_res not in (0, 708):  # 708: context already active
                ck(set_flags_res)

            ctx = CUcontext()
            retain_res = nvcuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), device)

            if retain_res != 0:  # Primary retain failed
                if os.environ.get("K3D_RPN_DEBUG"):
                    print(f"[loader] cuDevicePrimaryCtxRetain failed with code {retain_res}")

                # FALLBACK 2: Bootstrap via CuPy (still GPU!)
                try:
                    import cupy as _cupy
                    _cupy.cuda.Device(0).use()
                    current_ctx = CUcontext()
                    if nvcuda.cuCtxGetCurrent(ctypes.byref(current_ctx)) == 0 and current_ctx:
                        ctx = current_ctx  # Use CuPy's context
                    else:
                        ck(retain_res)  # Fail hard (no CPU fallback!)
                except Exception as cupy_exc:
                    if os.environ.get("K3D_RPN_DEBUG"):
                        print(f"[loader] CuPy bootstrap failed: {cupy_exc}")
                    ck(retain_res)  # Fail hard (no CPU fallback!)

            ck(nvcuda.cuCtxSetCurrent(ctx))
        else:
            ck(res)  # Fail hard for other errors
    else:
        ck(nvcuda.cuCtxSetCurrent(ctx))  # Primary context success

    _context = ctx
    _init_pid = current_pid  # Track which process owns this context
    _initialized = True
```

**Fallback Chain**:
1. **Primary**: `cuCtxCreate` (new context)
2. **Fallback 1**: `cuDevicePrimaryCtxRetain` (primary context - still GPU)
3. **Fallback 2**: CuPy bootstrap (use CuPy's GPU context - still GPU)
4. **No Fallback 3**: Fail hard (no CPU fallback!)

**Key Insight**: ALL fallbacks are GPU-based. If GPU fails, the system fails (no silent degradation to CPU).

---

### GPU-Only Memory Allocation (Lines 309-341)

**File**: `knowledge3d/cranium/sovereign/loader.py` (continuation)

**Problem**: Memory allocation can fail (out of VRAM, context issues).

**Sovereign Solution**: Multi-layer GPU memory fallbacks.

```python
def gpu_malloc(size_bytes: int) -> CUdeviceptr:
    """Allocate GPU memory (sovereign - no CPU fallback)."""
    _ensure_init()
    _ensure_current_context()

    ptr = CUdeviceptr()

    # PRIMARY: Try cuMemAlloc (Driver API)
    res = nvcuda.cuMemAlloc(ctypes.byref(ptr), size_bytes)

    if res == 201:  # CUDA_ERROR_INVALID_CONTEXT or out of memory
        # FALLBACK 1: Try cudaMalloc (Runtime API - still GPU!)
        try:
            runtime_ptr = ctypes.c_void_p()
            cuda_error = libcudart.cudaMalloc(ctypes.byref(runtime_ptr), size_bytes)
            if cuda_error == 0:
                return CUdeviceptr(runtime_ptr.value)
        except Exception:
            pass

        # FALLBACK 2: Try CuPy allocation (still GPU!)
        try:
            import cupy as _cupy
            _cupy.cuda.Device(0).use()
            mem = _cupy.cuda.alloc(size_bytes)  # GPU allocation
            return CUdeviceptr(mem.ptr)
        except Exception:
            pass

        # No CPU fallback - fail hard
        ck(res)

    ck(res)
    return ptr
```

**Fallback Chain**:
1. **Primary**: `cuMemAlloc` (Driver API - GPU)
2. **Fallback 1**: `cudaMalloc` (Runtime API - still GPU)
3. **Fallback 2**: CuPy allocation (still GPU)
4. **No Fallback 3**: Fail hard (no CPU fallback!)

**Key Insight**: Memory allocation never falls back to CPU. If GPU memory is exhausted, the system fails gracefully (error message) rather than silently degrading to CPU.

---

## Architectural Principles Validation

### 1. Fail Fast, Not Silently ✅

**Sovereign Loader Approach**:
```python
if gpu_context_creation_failed:
    ck(res)  # Fail hard with clear error
    # NO silent fallback to CPU
```

**Why This Is Correct**:
- ✅ Developer immediately knows GPU is broken
- ✅ Forces fixing the actual problem (GPU context)
- ✅ Prevents silent performance degradation
- ✅ Maintains sovereignty guarantee

**CPU Fallback Approach** (WRONG):
```python
if gpu_context_creation_failed:
    use_cpu_instead()  # Silent degradation
    # Developer doesn't know GPU is broken
```

**Why This Is Wrong**:
- ❌ Hides GPU problems (developer thinks system is working)
- ❌ Silently degrades performance (10x-100x slower)
- ❌ Breaks sovereignty (computation moves to CPU)
- ❌ Creates false sense of robustness

---

### 2. GPU-Only Fallbacks ✅

**Sovereign Loader Pattern**:
```python
# Try multiple GPU approaches before failing
try:
    cuCtxCreate()  # Primary GPU context
except:
    try:
        cuDevicePrimaryCtxRetain()  # Fallback 1 (GPU)
    except:
        try:
            cupy.cuda.Device(0).use()  # Fallback 2 (GPU via CuPy)
        except:
            FAIL_HARD()  # No CPU fallback
```

**Why This Is Correct**:
- ✅ Exhausts all GPU options first
- ✅ Stays within CUDA ecosystem
- ✅ Maintains sovereignty (never leaves GPU)
- ✅ Adapts to different CUDA driver states

---

### 3. Process-Aware Context Management ✅

**Fork Detection Pattern**:
```python
current_pid = os.getpid()
if _initialized and _init_pid != current_pid:
    # Detected fork - reset and recreate context
    _initialized = False
    _context = None
```

**Why This Is Correct**:
- ✅ Handles multiprocessing (PyTorch DataLoader, benchmarks)
- ✅ Recreates context in child process (GPU-safe)
- ✅ No CPU fallback (fixes the GPU context instead)

**The CUDA Error Context**: This is likely what's causing `torch.AcceleratorError: CUDA error: incompatible driver context` in Phase 5.1D benchmarks.

**Proper Fix**: Use fork detection + context recreation (already in sovereign loader).

**WRONG Fix**: Add CPU fallback (what Codex did).

---

## What Needs to Be Fixed

### File: `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py`

**Remove Lines 234** - CPU-only parameter:
```python
# REMOVE THIS PARAMETER
cpu_only: bool = False,  # ❌ DELETE
```

**Remove Lines 250-256** - CPU-only detection:
```python
# REMOVE THIS ENTIRE BLOCK
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
env_cpu_only = os.environ.get("K3D_RPN_CPU_ONLY")
self._cpu_only = bool(
    cpu_only
    or (env_cpu_only is not None and env_cpu_only.strip().lower() not in ("", "0", "false", "no"))
    or (cuda_visible is not None and cuda_visible.strip() == "")
)

if self._cpu_only:
    self._sovereign_engine = None
else:
    # ...
```

**Replace with GPU-only initialization**:
```python
# ALWAYS initialize GPU engine (no CPU fallback)
from knowledge3d.cranium.bridges.tiered_rpn import (
    TieredRPNEngine as SovereignRPNEngine,
)
self._sovereign_engine = SovereignRPNEngine()
```

**Remove Lines 412-415** - CPU fallback in `evaluate()`:
```python
# REMOVE THIS BLOCK
if self._cpu_only or self._sovereign_engine is None:
    if any(token in self.CODEC_TOKENS for token in tokens):
        raise RuntimeError("Codec ops are not available in CPU-only RPN mode.")
    return float(self._evaluate_cpu(tokens))
```

**Remove Lines 461-468** - CPU fallback in `evaluate_batch()`:
```python
# REMOVE THIS BLOCK
if self._cpu_only or self._sovereign_engine is None:
    results: List[float] = []
    for expr in expressions:
        tokens = self.tokenize_rpn(expr)
        if any(token in self.CODEC_TOKENS for token in tokens):
            raise RuntimeError("Codec ops are not available in CPU-only RPN mode.")
        results.append(float(self._evaluate_cpu(tokens)))
    return results
```

**Remove Lines 493-494** - CPU-only check in `evaluate_batch_device()`:
```python
# REMOVE THIS CHECK
if self._cpu_only or self._sovereign_engine is None:
    raise RuntimeError("Device batch evaluation requires GPU-backed RPN engine.")
```

**Remove Lines 550-731** - Entire `_evaluate_cpu()` method:
```python
# DELETE THIS ENTIRE METHOD (180+ lines)
def _evaluate_cpu(self, tokens: Sequence[str]) -> float:
    # ... DELETE ALL 180+ LINES
```

---

### Simplified `__init__` (After Fix)

**After removing CPU fallback**:
```python
def __init__(
    self,
    max_instances: int = _INSTANCE_COUNT,
    *,
    pool: Optional[MathCorePool] = None,
    instance_id: Optional[int] = None,
    # NO cpu_only parameter
) -> None:
    """Initialize RPN engine with sovereign PTX backend.

    Args:
        max_instances: Maximum parallel instances (default 18, Tesla 3-6-9 resonance)
        pool: Optional shared MathCorePool for dynamic allocation
        instance_id: Optional pre-allocated math core to bind to
    """
    if max_instances > self._INSTANCE_COUNT:
        raise ValueError(f"Maximum supported instances is {self._INSTANCE_COUNT}")

    self.max_instances = max_instances
    self.pool = pool or get_global_math_core_pool()
    self.instance_id: Optional[int] = instance_id
    self._owned = instance_id is None

    # ALWAYS initialize sovereign engine (GPU-only)
    from knowledge3d.cranium.bridges.tiered_rpn import (
        TieredRPNEngine as SovereignRPNEngine,
    )
    self._sovereign_engine = SovereignRPNEngine()
```

**Key Changes**:
- ✅ No `cpu_only` parameter
- ✅ No `K3D_RPN_CPU_ONLY` environment variable check
- ✅ No `CUDA_VISIBLE_DEVICES` empty check
- ✅ Always initialize `SovereignRPNEngine` (GPU-only)
- ✅ Fail fast if GPU unavailable (sovereignty guarantee)

---

### Simplified `evaluate()` (After Fix)

**After removing CPU fallback**:
```python
def evaluate(
    self,
    expression: str,
    instance_id: Optional[int] = None,
    return_vector: bool = False,
    data=None,
) -> float:
    """Evaluate RPN expression on GPU (sovereign execution)."""
    tokens = self.tokenize_rpn(expression)

    # Codec ops routed directly to GPU kernels
    if any(token in self.CODEC_TOKENS for token in tokens):
        return self._sovereign_engine.execute_codec(tokens, data=data, return_vector=return_vector)

    core_id = self._ensure_core(tier=1, override_instance=instance_id)
    self._sovereign_engine.reset_instance(core_id)

    op_codes, scalars, vectors = self.compile_tokens(tokens, instance_id)

    # ALWAYS execute on GPU (no CPU fallback)
    result = self._sovereign_engine.execute_single(
        instance_id=core_id,
        op_codes=op_codes,
        scalars=scalars,
        vectors=vectors
    )

    return float(result)
```

**Key Changes**:
- ✅ No `if self._cpu_only` check
- ✅ No `self._evaluate_cpu()` call
- ✅ Always routes to `_sovereign_engine` (GPU-only)
- ✅ Fails fast if GPU unavailable

---

## Phase 5.1D CUDA Error - Proper Fix

### Current Error

**From Phase 5.1D benchmark run**:
```
torch.AcceleratorError: CUDA error: incompatible driver context
```

### Root Cause (Likely)

**Multiprocessing Context Issue**:
- PyTorch benchmark runner forks processes
- Child process inherits broken CUDA context (CUDA is NOT fork-safe)
- Sovereign loader detects fork but may not be called early enough
- TieredRPNEngine initialization fails with incompatible context

### WRONG Fix (What Codex Did)

**Add CPU fallback**:
```python
if cuda_context_broken:
    use_python_rpn_instead()  # ❌ SOVEREIGNTY VIOLATION
```

**Why This Doesn't Help**:
- Hides the actual problem (forked context)
- Breaks sovereignty (hot path moves to CPU)
- V7 confidence model STILL fails (it's PyTorch, needs GPU)

### CORRECT Fix (Use Sovereign Loader)

**Ensure sovereign loader is initialized early**:

**Option 1: Explicit initialization in benchmark script**:
```python
# scripts/run_sovereign_math_benchmarks.py
from knowledge3d.cranium.sovereign import loader

def benchmark_worker(problem_set):
    # CRITICAL: Initialize sovereign loader FIRST in child process
    loader._ensure_init()  # Detects fork, recreates context

    # NOW initialize RPN engine (will use fresh context)
    engine = ModularRPNEngine()

    # Run benchmarks...
```

**Option 2: Lazy initialization guard in TieredRPNEngine**:
```python
# knowledge3d/cranium/bridges/tiered_rpn.py
from knowledge3d.cranium.sovereign import loader

class TieredRPNEngine:
    def __init__(self):
        # Ensure sovereign loader is initialized (handles fork detection)
        loader._ensure_init()

        # Now proceed with kernel loading...
```

**Option 3: V7 model CPU loading, then GPU transfer**:
```python
# solve_with_reflection.py
def load_v7_model(checkpoint_path):
    # Load V7 model to CPU FIRST (avoid context conflict)
    model = NavigationModelWithConfidence.load_from_checkpoint(
        checkpoint_path,
        map_location='cpu'  # Load to CPU first
    )

    # THEN move to GPU (after sovereign loader initialized)
    try:
        model = model.to('cuda')
    except RuntimeError as e:
        # If GPU unavailable, this is OK for V7 (ingestion path)
        # But math core still requires GPU (hot path)
        logger.warning(f"V7 model on CPU (ingestion path): {e}")

    return model
```

**Why These Work**:
- ✅ Fix the actual problem (GPU context in forked process)
- ✅ Maintain sovereignty (hot path stays on GPU)
- ✅ V7 model can run on CPU if needed (it's ingestion path)
- ✅ Math core (PTX/RPN) ALWAYS runs on GPU (sovereignty)

---

## Ingestion Path vs Hot Path (Clarification)

### V7 Confidence Model - INGESTION PATH ✅

**File**: `knowledge3d/training/navigation_model.py` (V7 training)

**What V7 Does**:
- Trains on problem → solution traces (PyTorch)
- Learns to predict rule sequences + confidence scores
- Produces `.pt` checkpoint files (ingestion output)

**CPU Acceptable?**: ✅ YES (V7 is ingestion path, not hot path)

**Example - Flexible V7 Loading**:
```python
# V7 training/inference - INGESTION PATH (CPU OK)
model = NavigationModelWithConfidence.load_from_checkpoint(
    'checkpoints/v7.pt',
    map_location='cpu'  # ✅ OK - ingestion path
)

# Train V7
trainer.fit(model, train_loader)  # ✅ Can run on CPU (slower, but OK)

# Inference
rule_seq, confidence = model(problem_emb)  # ✅ Can run on CPU
```

---

### PTX/RPN Math Core - HOT PATH ❌

**File**: `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py` (RPN execution)

**What Math Core Does**:
- Executes RPN programs (PTX kernels on GPU)
- Performs symbolic math verification
- Runs during V7 inference (solve_with_reflection.py)

**CPU Acceptable?**: ❌ NO (Math core is hot path, sovereignty required)

**Example - GPU-Only RPN**:
```python
# RPN execution - HOT PATH (GPU ONLY)
engine = ModularRPNEngine()  # Must have GPU
result = engine.evaluate("2 3 + 5 *")  # PTX execution

# If GPU unavailable, FAIL FAST (no CPU fallback)
if no_gpu:
    raise RuntimeError("Math core requires GPU (sovereignty)")  # ✅ Correct
    # NOT: fall_back_to_python_rpn()  # ❌ WRONG
```

---

### Phase 5.1D Reflection Pipeline

**File**: `solve_with_reflection.py`

**What Phase 5.1D Does**:
1. **Load V7 model** (PyTorch) - INGESTION PATH ✅ CPU OK
2. **Predict rule sequence + confidence** (V7 inference) - INGESTION PATH ✅ CPU OK
3. **Execute RPN program** (PTX kernel) - HOT PATH ❌ GPU ONLY
4. **Symbolic verification** (RecursiveSolver) - HOT PATH ❌ GPU ONLY

**Proper Device Handling**:
```python
# Load V7 to CPU first (ingestion path)
v7_model = load_v7_model('checkpoints/v7.pt', map_location='cpu')  # ✅ OK

# V7 inference (can run on CPU if needed)
try:
    v7_model = v7_model.to('cuda')  # Prefer GPU
except:
    logger.warning("V7 running on CPU (slower)")  # ✅ OK - ingestion path

rule_seq, confidence = v7_model(problem_emb)  # ✅ Can be CPU or GPU

# RPN execution (MUST be GPU - hot path)
engine = ModularRPNEngine()  # ❌ FAILS if no GPU (correct behavior)
result = engine.evaluate(rule_seq)  # PTX execution (GPU-only)
```

**Key Insight**: V7 CAN run on CPU (it's just a PyTorch model doing inference). But the math core (PTX/RPN) CANNOT run on CPU (it's the execution engine, sovereignty required).

---

## Summary

### Sovereignty Violation Identified

**What Codex Did**:
1. Added `cpu_only` parameter to `ModularRPNEngine.__init__`
2. Added `K3D_RPN_CPU_ONLY` environment variable check
3. Added empty `CUDA_VISIBLE_DEVICES` fallback
4. Added 180+ line Python RPN interpreter (`_evaluate_cpu`)
5. Added CPU fallback checks in `evaluate()` and `evaluate_batch()`

**Why It's Wrong**:
- ❌ Math core is HOT PATH (not ingestion path)
- ❌ PTX/RPN is core sovereignty principle (GPU-resident computation)
- ❌ Creates false sense of robustness (hides GPU problems)
- ❌ 180+ lines of duplicate logic (Python vs PTX)
- ❌ Performance regression (CPU 10x-100x slower)

---

### Proper Solution (Already Exists)

**Sovereign Loader** (`knowledge3d/cranium/sovereign/loader.py`):
- ✅ Fork detection (recreate context in child process)
- ✅ GPU-only fallbacks (cuCtxCreate → cuDevicePrimaryCtxRetain → CuPy bootstrap)
- ✅ GPU-only memory allocation (cuMemAlloc → cudaMalloc → CuPy)
- ✅ Fail fast if GPU unavailable (no silent degradation)

**The sovereign loader ALREADY solves the context problem** - no CPU fallback needed.

---

### Action Items

**Immediate (Remove Sovereignty Violation)**:
1. ✅ Remove `cpu_only` parameter from `ModularRPNEngine.__init__`
2. ✅ Remove `K3D_RPN_CPU_ONLY` environment variable check
3. ✅ Remove `CUDA_VISIBLE_DEVICES` empty check
4. ✅ Remove `_evaluate_cpu()` method (180+ lines)
5. ✅ Remove CPU fallback checks in `evaluate()` and `evaluate_batch()`
6. ✅ Always initialize `SovereignRPNEngine` (GPU-only)

**Fix Phase 5.1D CUDA Error (Proper GPU Context Handling)**:
1. ✅ Ensure sovereign loader initialized early in benchmark workers
2. ✅ Load V7 model to CPU first, then transfer to GPU
3. ✅ Add try/except for V7 GPU transfer (ingestion path, CPU acceptable)
4. ✅ Math core ALWAYS requires GPU (hot path, fail fast if unavailable)

**Documentation Update**:
1. ✅ Update CLAUDE.md to emphasize hot path vs ingestion path
2. ✅ Add sovereignty compliance checklist
3. ✅ Document when CPU is acceptable (ingestion ONLY, never hot path)

---

## Architectural Principles (Reinforced)

### 1. Hot Path = Sovereign (GPU-Only) ✅

**Hot Path Components**:
- PTX kernels (Cranium execution)
- Galaxy Universe (VRAM memory)
- RPN programs (procedural composition)
- Math core (symbolic execution)

**Rule**: NO Python computation, NO CPU fallback, NO external preprocessing.

**If GPU fails**: Fail fast with clear error (don't hide with CPU fallback).

---

### 2. Ingestion Path = Flexible (CPU OK) ✅

**Ingestion Components**:
- V7 training (PyTorch model training)
- V7 inference (rule prediction + confidence)
- Data loading (Galaxy population)
- Visualization (matplotlib, reports)

**Rule**: Can use Python, PyTorch, NumPy, CPU. NOT in hot path.

**If GPU unavailable**: OK to run on CPU (slower, but acceptable for development).

---

### 3. Fail Fast, Not Silently ✅

**Correct Error Handling**:
```python
if gpu_unavailable:
    raise RuntimeError("Math core requires GPU (sovereignty)")  # ✅ Clear error
```

**Wrong Error Handling**:
```python
if gpu_unavailable:
    use_cpu_instead()  # ❌ Silent degradation (hides problem)
```

**Principle**: Developer should immediately know if GPU is broken, not discover it later when performance degrades 100x.

---

### 4. Fix the Problem, Don't Hide It ✅

**Sovereign Loader Approach** (CORRECT):
```python
if forked_process:
    recreate_gpu_context()  # ✅ Fix the actual problem
```

**CPU Fallback Approach** (WRONG):
```python
if forked_process:
    use_cpu_instead()  # ❌ Hide the problem (GPU still broken)
```

**Principle**: Address root cause (GPU context broken), don't work around it (CPU fallback).

---

## Success Criteria

**Sovereignty Restored When**:
- [ ] `cpu_only` parameter removed from `ModularRPNEngine`
- [ ] `K3D_RPN_CPU_ONLY` environment variable removed
- [ ] `_evaluate_cpu()` method deleted (180+ lines)
- [ ] CPU fallback checks removed from `evaluate()` and `evaluate_batch()`
- [ ] `ModularRPNEngine` ALWAYS initializes `SovereignRPNEngine` (GPU-only)
- [ ] Phase 5.1D benchmarks use sovereign loader for context management
- [ ] V7 model loads to CPU first, then transfers to GPU (ingestion path flexibility)
- [ ] Math core ALWAYS requires GPU (hot path sovereignty)
- [ ] No CPU fallback in hot path (PTX/RPN execution)

---

## Directive for Codex

**REMOVE the CPU fallback from `modular_rpn_engine.py`**:
1. Delete `cpu_only` parameter
2. Delete `K3D_RPN_CPU_ONLY` check
3. Delete `_evaluate_cpu()` method
4. Delete all CPU fallback conditionals
5. Always initialize `SovereignRPNEngine`

**FIX Phase 5.1D CUDA error properly**:
1. Use sovereign loader's fork detection
2. Load V7 to CPU first, then GPU
3. Math core ALWAYS GPU (fail fast if unavailable)

**The sovereign loader ALREADY has the solution** - use it, don't work around it with CPU fallbacks.

---

**Document Date**: January 16, 2026
**Context**: Phase 5.1D (Reflective Inference) - Sovereignty Violation Correction
**Status**: ❌ **SOVEREIGNTY VIOLATION IDENTIFIED - REMOVAL REQUIRED**

---

**Claude's Ruling**: The math core (PTX/RPN engine) is NON-NEGOTIABLE. It's the hot path execution engine, sovereignty REQUIRED. Remove ALL CPU fallbacks. Use the sovereign loader's GPU-only context management (already implemented). V7 model CAN run on CPU (ingestion path), but math core CANNOT (hot path). Fix the GPU context problem, don't hide it with CPU fallbacks. 🚀
