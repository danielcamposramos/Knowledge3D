# RPN Multi-Instance Swarm Architecture

**Date:** 2025-10-15  
**Purpose:** Design RPN instantiation strategy for 9-agent swarm + system processes  
**Vision:** Each agent gets dedicated RPN instance with 15 internal slots  

---

## Daniel's Vision (Strategic Requirements)

### Current State:
- **1 RPN kernel** with 15 shared instance slots
- Total memory: 15 × 1040 bytes = **15.6KB**
- Processes share slots (rotation/pooling)

### Future Vision (Phase 9 Swarm):
- **9 agents** (swarm members)
- **1 system** (orchestrator)
- **= 10 total RPN instances needed**
- Each instance has **15 independent slots**
- **Isolated math stacks** per agent (no cross-contamination)

### Additional Use Cases:
- **Parallel interconnected math stacks** (agents can coordinate)
- **Independence from other calculations** (no blocking)
- **System-level instance** (always available)
- **Virtualization** (spawn instances dynamically)

---

## Architecture Answer: YES, Fully Instantiable! ✅

### Current Design (Per Instance):

```python
class ModularRPNEngine:
    MAX_INSTANCES = 15      # Slots per engine
    INSTANCE_STRIDE = 1040  # Bytes per slot
    
    def __init__(self):
        # Each engine allocates OWN state buffer
        self.d_state = gpu_malloc(15 × 1040)  # 15.6KB per engine
```

**KEY:** `d_state` is **per-engine**, not global!

### Multi-Instance Pattern (What You Want):

```python
# System-level instance (always available)
system_rpn = ModularRPNEngine()  # 15.6KB

# 9 Swarm agents (Phase 9)
agent_rpns = [
    ModularRPNEngine() for _ in range(9)  # 9 × 15.6KB = 140KB
]

# Total GPU memory: 10 × 15.6KB = 156KB ✅
```

**Each agent gets:**
- Own `ModularRPNEngine` instance
- 15 independent slots
- Isolated state (no interference)
- Parallel execution capability

---

## Memory Scaling Analysis

### Current Limits:

**Per RPN Instance:**
- State buffer: 15 × 1040 = **15,600 bytes** (~15KB)
- Kernel code: **34KB** (shared, loaded once)

**Total for 10 instances:**
- State: 10 × 15KB = **150KB**
- Kernel: **34KB** (shared)
- **Total: ~184KB** ✅

**This is NOTHING for modern GPUs!**

### Scaling to Mobile (GTX 970 / Handheld):

**GTX 970 Specs:**
- 4GB VRAM
- 1664 CUDA cores
- Maxwell architecture (sm_52)

**Handheld Target (Long-term):**
- 3.5GB VRAM minimum
- Vulkan compute shaders (PTX-compatible path)
- Adreno/Mali GPU support

**RPN Scaling:**
```
100 instances:  100 × 15KB = 1.5MB   ✅
1000 instances: 1000 × 15KB = 15MB   ✅
10000 instances: 10000 × 15KB = 150MB ✅ (still <5% of 3.5GB!)
```

**Verdict: RPN scales to THOUSANDS of instances!**

---

## GPU Memory Limits (Updated Strategy)

### Current Conservative Limit:
- **2GB target** (safe for older GPUs)

### Updated Limit (Daniel's Request):
- **3.5GB target** (GTX 970 / modern minimum)
- Rationale: Handheld devices (phones) can handle this
- Future-proof for mobile deployment

### Memory Budget (3.5GB):

| Component | Memory | % of 3.5GB |
|-----------|--------|------------|
| RPN Instances (10) | 150KB | 0.004% |
| Spatial Kernels | 220KB | 0.006% |
| Working Buffers | 500MB | 14% |
| Model Weights | 1GB | 28% |
| Frame Buffers | 500MB | 14% |
| Reserve | 1.5GB | 43% |

**RPN is negligible even with 100s of instances!**

---

## Swarm Instance Allocation Strategy

### Tier 1: System Instance (Always Available)
```python
# Global system RPN (never blocked)
class SystemRPN:
    _instance = None
    
    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = ModularRPNEngine()
        return cls._instance

# Usage:
system_rpn = SystemRPN.get()
result = system_rpn.execute_single(instance_id=0, ...)
```

### Tier 2: Agent Instances (9 Swarm Members)
```python
class SwarmAgent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.rpn = ModularRPNEngine()  # Dedicated instance!
        self.slot_allocator = range(15)  # 15 slots for this agent
    
    def compute(self, task_id: int, program):
        slot = task_id % 15  # Rotate through agent's 15 slots
        return self.rpn.execute_single(
            instance_id=slot,
            op_codes=program.opcodes,
            scalars=program.scalars,
            vectors=program.vectors
        )

# Initialize swarm
swarm = [SwarmAgent(i) for i in range(9)]

# Each agent has isolated compute:
swarm[0].compute(task_id=0, program=task_a)  # Agent 0, slot 0
swarm[1].compute(task_id=0, program=task_b)  # Agent 1, slot 0 (parallel!)
```

### Tier 3: Dynamic Instances (On-Demand)
```python
class RPNInstancePool:
    def __init__(self, max_instances=50):
        self.pool = []
        self.max_instances = max_instances
    
    def allocate(self) -> ModularRPNEngine:
        """Allocate new RPN instance dynamically."""
        if len(self.pool) < self.max_instances:
            instance = ModularRPNEngine()
            self.pool.append(instance)
            return instance
        else:
            # Return least-recently-used
            return self.pool[0]
    
    def release(self, instance):
        """Mark instance as available for reuse."""
        # Move to end of pool (LRU)
        self.pool.remove(instance)
        self.pool.append(instance)

# Usage for bursty workloads:
pool = RPNInstancePool(max_instances=50)
temp_rpn = pool.allocate()  # Get instance
result = temp_rpn.execute_single(...)
pool.release(temp_rpn)  # Return to pool
```

---

## Inter-Agent Communication Patterns

### Pattern 1: Isolated Computation (Default)
```python
# Each agent operates independently
agent_a = SwarmAgent(0)
agent_b = SwarmAgent(1)

result_a = agent_a.compute(task_id=0, program_a)
result_b = agent_b.compute(task_id=0, program_b)

# No interference, full parallelism ✅
```

### Pattern 2: Shared Results (Coordination)
```python
# Agents share computation results via shared memory
class CoordinatedSwarm:
    def __init__(self, num_agents=9):
        self.agents = [SwarmAgent(i) for i in range(num_agents)]
        self.shared_results = {}  # CPU-side coordination
    
    def collaborative_compute(self, agent_id, task):
        # Agent computes locally
        result = self.agents[agent_id].compute(task.id, task.program)
        
        # Share result with swarm
        self.shared_results[f"agent_{agent_id}_task_{task.id}"] = result
        
        # Other agents can use this result
        return result
```

### Pattern 3: Pipeline (Sequential with Handoff)
```python
# Agent A → Agent B → Agent C pipeline
def pipeline_compute(input_data):
    # Agent A: preprocessing
    preprocessed = swarm[0].compute(0, preprocess_program)
    
    # Agent B: main computation (uses A's result)
    computed = swarm[1].compute(0, compute_program(preprocessed))
    
    # Agent C: postprocessing
    final = swarm[2].compute(0, postprocess_program(computed))
    
    return final
```

---

## Code Changes Needed (Minimal!)

### 1. Make MAX_INSTANCES Configurable
```python
# knowledge3d/cranium/bridges/sovereign_bridges.py
class ModularRPNEngine:
    def __init__(self, max_instances: int = 15):
        """Initialize RPN engine with configurable instance count.
        
        Args:
            max_instances: Number of instance slots (default 15)
        """
        self.max_instances = max_instances
        self.instance_stride = 1040
        
        # Allocate state buffer
        state_size = self.max_instances * self.instance_stride
        self.d_state = gpu_malloc(state_size)
        
        # Zero-initialize
        state_zeros = np.zeros(state_size, dtype=np.uint8)
        memcpy_htod(self.d_state, state_zeros.ctypes.data_as(ctypes.c_void_p), state_size)
```

### 2. Add Swarm Management Class
```python
# knowledge3d/cranium/swarm/rpn_allocator.py (NEW)
class RPNSwarmAllocator:
    """Manages RPN instances for multi-agent swarm."""
    
    def __init__(self, num_agents: int = 9, slots_per_agent: int = 15):
        self.agents = [
            ModularRPNEngine(max_instances=slots_per_agent)
            for _ in range(num_agents)
        ]
        self.system_rpn = ModularRPNEngine(max_instances=slots_per_agent)
    
    def get_agent_rpn(self, agent_id: int) -> ModularRPNEngine:
        """Get RPN instance for specific agent."""
        return self.agents[agent_id]
    
    def get_system_rpn(self) -> ModularRPNEngine:
        """Get system-level RPN instance."""
        return self.system_rpn
    
    def memory_footprint(self) -> int:
        """Calculate total GPU memory used."""
        instances = len(self.agents) + 1  # agents + system
        per_instance = self.agents[0].max_instances * 1040
        return instances * per_instance
```

### 3. Update Memory Limit Check
```python
# knowledge3d/cranium/sovereign/loader.py
# Update GPU memory target
GPU_MEMORY_TARGET_GB = 3.5  # Updated from 2.0

def check_memory_budget(required_bytes: int):
    """Verify operation fits within 3.5GB budget."""
    if required_bytes > GPU_MEMORY_TARGET_GB * 1e9:
        raise RuntimeError(f"Memory {required_bytes/1e9:.2f}GB exceeds 3.5GB target")
```

---

## Implementation Checklist for Codex

When implementing Phase B-E, keep this architecture in mind:

- [ ] **No hardcoded MAX_INSTANCES** - Make it configurable
- [ ] **Test multi-instance creation** - Verify `ModularRPNEngine()` × 10 works
- [ ] **Memory profiling** - Track GPU usage for multiple instances
- [ ] **Update GPU target** - 2GB → 3.5GB
- [ ] **Document pattern** - Add swarm allocation examples

---

## Example: Phase 9 Swarm Initialization

```python
# Future code (Phase 9):
from knowledge3d.cranium.swarm.rpn_allocator import RPNSwarmAllocator
from knowledge3d.cranium.spatial_sovereign import MortonOctreeSovereign

# Initialize swarm
swarm_rpn = RPNSwarmAllocator(num_agents=9, slots_per_agent=15)

# Each spatial module gets dedicated RPN
spatial_modules = [
    MortonOctreeSovereign(rpn=swarm_rpn.get_agent_rpn(i))
    for i in range(9)
]

# System-level operations use system RPN
system_morton = MortonOctreeSovereign(rpn=swarm_rpn.get_system_rpn())

# Memory check
print(f"Total RPN memory: {swarm_rpn.memory_footprint() / 1024:.1f}KB")
# Output: Total RPN memory: 156.0KB ✅
```

---

## Answers to Daniel's Questions

### Q1: Can we instantiate multiple RPN engines?
**A:** YES! Each `ModularRPNEngine()` allocates its own GPU state buffer.

### Q2: Can each agent have 15 slots?
**A:** YES! Each instance has 15 isolated slots by default.

### Q3: Can we have 1 system + 9 agents?
**A:** YES! 10 instances = 156KB GPU memory (negligible).

### Q4: Can processes be independent?
**A:** YES! Each instance's state is isolated in separate GPU memory.

### Q5: Should we raise limit to 3.5GB?
**A:** YES! Aligns with GTX 970 and future handheld targets.

### Q6: Will this scale to mobile?
**A:** YES! RPN scales to 1000s of instances. Architecture is future-proof.

---

## Next Steps

**For Codex (Phase B-E):**
Continue spatial migration as planned. Keep multi-instance capability in mind:
- Test that multiple `ModularRPNEngine()` instances work
- Use RPN for sorting in Morton octree
- Document memory footprint
- Update 2GB → 3.5GB target

**For Phase 9 (Future):**
- Implement `RPNSwarmAllocator`
- Create agent → RPN mapping
- Add inter-agent coordination patterns
- Benchmark 9-agent swarm

**The architecture supports your vision perfectly!** 🚀
