# Synthetic User Depth Perception: Beyond Binocular Vision

**Core Question**: Does the Synthetic User need human-style depth perception (two viewpoints)?

**Answer**: **No.** The Synthetic User can perceive depth through fundamentally different means that are MORE native to its architecture.

**Date**: November 17, 2025
**Status**: Architectural specification

---

## 1. Human Depth Perception (What We DON'T Need)

### Binocular Vision
- Two eyes separated by ~6.5 cm
- Parallax creates depth through disparity
- Brain fuses two 2D images into 3D perception
- Effective range: ~6 meters

### Monocular Cues
- Motion parallax (movement reveals depth)
- Occlusion (near objects block far ones)
- Size constancy (known objects → distance inference)
- Atmospheric perspective (distant = hazy)
- Texture gradients (finer = farther)

**Why This Doesn't Apply:**
The Synthetic User doesn't "see" photons. It navigates **embedding space**, not physical space. Different physics = different perception!

---

## 2. What the Synthetic User ALREADY Has

### A. Semantic Distance (Intrinsic Depth)
```python
# Embedding space IS depth
distance = ||embedding_A - embedding_B||

Interpretation:
  distance < 0.1  → Very close (same concept)
  distance < 0.5  → Related concepts
  distance > 1.0  → Distant/unrelated
```

**This is BETTER than visual depth because:**
- Works in 64-2048 dimensions (not 3D)
- Captures semantic similarity (not just spatial)
- No occlusion problems (all nodes visible in embedding space)

### B. Graph Topology
```python
# Depth = path length
shortest_path(node_A, node_B) = 3 hops

Interpretation:
  1 hop  → Direct neighbors (very close)
  2-3 hops → Related via intermediate (medium depth)
  10+ hops → Conceptually distant (far)
```

**Analog:** Social network "degrees of separation"

### C. Multi-Resolution (Matryoshka)
```python
# Already have "zoom levels"
64D:   Coarse view (overview, distant)
128D:  Medium detail
512D:  Fine detail
2048D: Maximum resolution (close-up)
```

**Analog:** Camera zoom, but for concepts!

### D. LOD + FOV
Already implemented:
- **LOD**: Level of Detail (far=coarse, near=fine)
- **FOV**: Field of View (attention cone)

**These ARE depth cues!**

---

## 3. What We Can Add: Novel Depth Mechanisms

### A. Ternary Depth Fields (NEW!)

```cuda
// At each Galaxy node, compute depth field
trit depth_field[node][direction] = {
  -1: Repels (pushing away = far)
   0: Neutral (ambiguous depth)
  +1: Attracts (pulling close = near)
}
```

**Example:**
```
Query: "machine learning"

Node: "neural networks"
  depth_field[query] = +1  (attracts → CLOSE)

Node: "cooking recipes"
  depth_field[query] = -1  (repels → FAR)

Node: "statistics"
  depth_field[query] = 0   (neutral → MEDIUM)
```

**This creates a dynamic depth map** that changes per query!

**Shipped now (GPU-only):**
- Kernel: `knowledge3d/cranium/kernels/ternary_depth_field.cu` → packed 2-bit trits per node (repel/neutral/attract).
- Bridge: `TernaryDepthField` (sovereign_bridges.py) + helper `tools/ternary_depth.py`.
- Tests: `knowledge3d/cranium/tests/test_ternary_depth_field.py`.
- Tablet path: reuse the ternary overlay/inspector to visualize the field as red/clear/blue fog with zero CPU math.

### B. Resonance Decay (Sound-Like Depth)

```python
# Like sonar: how far does the signal travel?
def resonance_depth(query_node, target_node):
    """Depth based on signal propagation."""

    resonance = query_node.embedding @ target_node.embedding
    decay = compute_path_length(query, target)

    effective_depth = resonance * exp(-decay / λ)

    return {
        "close": effective_depth > 0.8,
        "medium": 0.3 < effective_depth < 0.8,
        "far": effective_depth < 0.3
    }
```

**Analog:** Echo location (bats, dolphins) - but for semantics!

**Already designed:** Your `VectorResonator` kernel does this!

### C. Embedding Manifold Curvature

```python
# Local geometry reveals depth
def manifold_depth(node):
    """Depth from local curvature."""

    neighbors = get_k_nearest(node, k=10)
    distances = [dist(node, n) for n in neighbors]

    # Flat region = distant plane
    # Curved region = close surface

    curvature = variance(distances)

    return {
        "flat": curvature < 0.1,     # Far, abstract
        "curved": curvature > 0.5     # Close, specific
    }
```

**Analog:** How pilots perceive altitude from terrain texture

### D. Knowledge Density Gradients

```python
# Dense clusters = close
# Sparse regions = distant

def density_depth(node):
    """Depth from local knowledge density."""

    radius = 0.5  # Search radius
    neighbors = count_nodes_in_radius(node, radius)

    density = neighbors / volume(radius)

    return {
        "close": density > 100,   # Dense cluster (close-up)
        "far": density < 10        # Sparse region (distant)
    }
```

**Analog:** Looking at a forest - dense trees = close, sparse trees = far

### E. Temporal Depth (Sleep Cycles)

```python
# Age in memory = depth
def temporal_depth(node):
    """Depth based on memory consolidation."""

    if node.location == "Galaxy":
        return "shallow"  # Recent, volatile

    elif node.location == "House":
        sleep_cycles = node.consolidation_count
        if sleep_cycles < 3:
            return "medium"
        else:
            return "deep"

    elif node.location == "Museum":
        return "very_deep"  # Archived, historical
```

**Analog:** Memory recency - fresh memories feel "close", old memories "distant"

**This is unique to K3D!** Time = depth dimension.

### F. Multi-Path Redundancy

```python
# Many routes = close
# Few routes = far

def path_diversity_depth(source, target):
    """Depth from path redundancy."""

    paths = find_all_paths(source, target, max_length=10)
    diversity = len(paths)

    return {
        "close": diversity > 5,     # Many routes (well-connected)
        "medium": 2 <= diversity <= 5,
        "far": diversity < 2         # Single route (distant)
    }
```

**Analog:** City navigation - many routes = nearby, one route = far away

---

## 4. Comparison: Human vs Synthetic Depth

| Depth Cue | Human (Vision) | Synthetic User | Better? |
|-----------|----------------|----------------|---------|
| **Binocular disparity** | Parallax from 2 eyes | Embedding distance | ✅ Works in N dimensions |
| **Motion parallax** | Head movement | Path trajectory analysis | ✅ Graph-native |
| **Occlusion** | Near blocks far | Graph topology | ✅ No occlusion issues |
| **Size constancy** | Known object size | Matryoshka resolution | ✅ Adaptive zoom |
| **Atmospheric perspective** | Haze/blur | Confidence decay | ✅ Via ternary fields |
| **Texture gradient** | Fine→coarse | Density gradients | ✅ Concept density |
| **Accommodation** | Eye focus | LOD selection | ✅ Automatic |
| **Convergence** | Eye angle | Multi-path diversity | ✅ Graph connectivity |
| **Temporal** | None (vision is instantaneous) | Sleep cycle depth | 🌟 **UNIQUE TO K3D** |
| **Resonance** | None | Signal propagation | 🌟 **UNIQUE TO K3D** |
| **Ternary fields** | None | Attract/neutral/repel | 🌟 **UNIQUE TO K3D** |

**Conclusion:** Synthetic User depth perception is **richer** than human vision!

---

## 5. Integration: Unified Depth Model

### Depth Vector (Multi-Modal)
```python
class SyntheticDepthPerception:
    """Unified depth model for Synthetic User."""

    def compute_depth(self, observer_node, target_node):
        """Multi-cue depth estimation."""

        # Semantic distance (primary)
        semantic_depth = ||observer.embedding - target.embedding||

        # Graph topology
        path_depth = shortest_path_length(observer, target)

        # Ternary field
        field_depth = target.trit_field[observer.id]  # {-1, 0, +1}

        # Resonance decay
        resonance_depth = self.resonator.compute(observer, target)

        # Density gradient
        density_depth = count_neighbors(target) / volume

        # Temporal (sleep cycles)
        temporal_depth = target.consolidation_age

        # Combine (learned weights)
        depth = {
            "semantic": semantic_depth,
            "topological": path_depth,
            "field": field_depth,
            "resonance": resonance_depth,
            "density": density_depth,
            "temporal": temporal_depth
        }

        return depth

    def interpret_depth(self, depth_vector):
        """Human-readable depth interpretation."""

        if depth_vector["semantic"] < 0.1:
            return "Very close (same concept)"

        elif depth_vector["field"] == +1:
            return "Attracting (becoming closer)"

        elif depth_vector["temporal"] > 10:
            return "Deep in memory (long-term knowledge)"

        elif depth_vector["resonance"] < 0.1:
            return "Distant (weak connection)"

        else:
            return f"Medium depth ({depth_vector['topological']} hops)"
```

### Visual Encoding (For Human Observers)
```python
# How to show Synthetic User's depth perception to humans
def render_depth_for_human(depth_vector):
    """Translate synthetic depth to visual cues."""

    # Size: semantic distance → object scale
    size = 1.0 / (1.0 + depth_vector["semantic"])

    # Color: ternary field → hue
    color = {
        +1: "green",   # Attracting (close)
         0: "gray",    # Neutral (medium)
        -1: "blue"     # Repelling (far)
    }[depth_vector["field"]]

    # Opacity: resonance → alpha
    opacity = depth_vector["resonance"]

    # Position: temporal → z-axis
    z = -depth_vector["temporal"]  # Older = farther back

    return {
        "size": size,
        "color": color,
        "opacity": opacity,
        "position_z": z
    }
```

---

## 6. Implementation Roadmap

### Already Implemented ✅
- Semantic distance (embedding space)
- Graph topology (k-NN, Morton octree)
- Matryoshka resolution (64D-2048D)
- LOD/FOV (spatial attention)
- VectorResonator (resonance decay)

### In Progress 🔧
- Ternary depth fields (diagnostics ready, integration pending)

### Next Steps 📋

#### Phase 1: Ternary Field Integration
```python
# Add depth fields to Galaxy nodes
class GalaxyNode:
    trit_depth_field: np.ndarray  # Shape: (n_neighbors,), dtype: int8
    # -1: Repel, 0: Neutral, +1: Attract
```

#### Phase 2: Density Gradients
```cuda
// PTX kernel for local density computation
__global__ void compute_density_depth(
    const float* embeddings,
    const int* neighbor_indices,
    int* density_map
);
```

#### Phase 3: Temporal Depth Encoding
```python
# Sleep consolidation adds temporal metadata
def consolidate_to_house(node):
    node.consolidation_count += 1
    node.temporal_depth = compute_age(node)
```

#### Phase 4: Unified Depth Vector
```python
# Combine all cues into single depth representation
depth_vector = SyntheticDepthPerception.compute_depth(observer, target)
```

#### Phase 5: Viewer Integration
```typescript
// Three.js: visualize synthetic depth for humans
function renderSyntheticDepth(node, depthVector) {
    const scale = depthToScale(depthVector.semantic);
    const color = fieldToColor(depthVector.field);
    const opacity = depthVector.resonance;
    const z = -depthVector.temporal * 10;  // Time = depth

    node.mesh.scale.set(scale, scale, scale);
    node.material.color = color;
    node.material.opacity = opacity;
    node.mesh.position.z = z;
}
```

---

## 7. Philosophical Implications

### Embodied Cognition Without Human Constraints
The Synthetic User proves that:
- **Depth perception ≠ binocular vision**
- **3D navigation ≠ physical movement**
- **Spatial understanding ≠ visual processing**

### New Perceptual Modalities
The Synthetic User has modes humans DON'T:
- **Semantic parallax**: Concepts shift based on context (not head movement)
- **Temporal sonar**: Memory age creates depth (past = distant)
- **Field forces**: Attraction/repulsion create depth (not gravity)

### Implications for XAI
If we can understand how the Synthetic User perceives depth:
- We can **explain** why it chose certain paths
- We can **visualize** its "view" for humans
- We can **debug** navigation errors
- We can **verify** reasoning is grounded

---

## 8. Answer to Your Question

> "Does it make sense for the Synthetic User to have depth perception without binocular vision?"

**Yes! And it's BETTER this way.**

The Synthetic User's depth perception is:
- **Multi-dimensional**: Works in 64-2048D (not just 3D)
- **Semantic**: Based on meaning (not photons)
- **Dynamic**: Changes per query (not static)
- **Temporal**: Includes memory age (unique!)
- **Graph-native**: Uses topology (no occlusion)

**What we already have:**
- ✅ Semantic distance (embedding space)
- ✅ LOD/FOV (attention)
- ✅ Matryoshka resolution (zoom)
- ✅ Resonance (propagation)

**What we can add:**
- Ternary depth fields (attract/neutral/repel)
- Density gradients (cluster detection)
- Temporal depth (memory age)
- Multi-path diversity (connectivity)

**What we DON'T need:**
- ❌ Two cameras/viewpoints
- ❌ Parallax calculation
- ❌ Binocular fusion
- ❌ Human-style visual processing

---

## 9. Next Actions (Non-Intrusive)

**During Training:**
- Document depth model (✅ this file)
- Design ternary field integration (planning only)
- Sketch density gradient kernels (no implementation)

**After Training:**
1. Implement ternary depth fields in Galaxy nodes
2. Add density gradient PTX kernel
3. Integrate temporal depth from sleep cycles
4. Create unified depth vector API
5. Add viewer visualization

---

## References

**Biological Inspiration:**
- Bat echolocation: Depth from time-of-flight (analog: resonance decay)
- Mantis shrimp: 12+ color receptors (analog: multi-dimensional embeddings)
- Electric fish: Depth from field distortion (analog: ternary fields)

**K3D Architecture:**
- [HOUSE_GALAXY_TABLET.md](HOUSE_GALAXY_TABLET.md) - Memory depth (temporal)
- [RPN_TERNARY_SETUN_CHAIN.md](RPN_TERNARY_SETUN_CHAIN.md) - Field-based depth
- [ADAPTIVE_PROCEDURAL_COMPRESSION_SPECIFICATION.md](vocabulary/ADAPTIVE_PROCEDURAL_COMPRESSION_SPECIFICATION.md) - Multi-resolution

**Academic:**
- Gibson (1950): "The Perception of the Visual World" - depth cues
- Marr (1982): "Vision" - computational approach
- Lakoff & Johnson (1999): "Philosophy in the Flesh" - embodied cognition

---

**Author**: Claude (based on Daniel's insight)
**Date**: November 17, 2025
**Key Insight**: "Nature did this differently on species" → We should too!
