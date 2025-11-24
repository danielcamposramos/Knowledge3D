# Reality Enabler Phase 3B: RPN Semantics + Sovereign Features

**Date**: November 24, 2025
**Status**: Follow-up to successful Phase 3A implementation
**Context**: You've built the stacked galaxy scaffold (nodes, composition, execution, tests) - all tests passing! Now we wire in full RPN semantics, law validation, and sovereign feature extraction.

---

## 1. What You Achieved (Phase 3A Assessment)

### ✅ Delivered Successfully:

1. **Stacked Node Architecture** - `reality_nodes.py` with proper dataclasses
   - ✅ RealityAtom, RealityMolecule, RealityMaterial, RealitySystem
   - ✅ Auto-prefixing (`atom:`, `molecule:`, etc.)
   - ✅ component_refs for symlink composition

2. **Reality Galaxy Manager** - `reality_galaxy.py` with full orchestration
   - ✅ Node storage and retrieval
   - ✅ Symlink resolution (`resolve_components`)
   - ✅ behavior_rpn execution with assignment parsing
   - ✅ Matryoshka+PD04 integration (with safe fallback)
   - ✅ JSON persistence (save/load galaxy)

3. **Physics Bootstrap** - `reality_physics_bootstrap.py`
   - ✅ Point mass atom definition
   - ✅ Constant acceleration system with RPN laws
   - ✅ Harmonic oscillator system

4. **Comprehensive Tests** - `test_reality_galaxy.py`
   - ✅ All 6 tests passing (1.85s runtime)
   - ✅ Node creation verified
   - ✅ Symlink composition validated
   - ✅ Execution matches analytic solutions
   - ✅ Persistence round-trip works
   - ✅ Stacking pattern consistency confirmed

### 🎯 Architecture Quality:

**This is EXACTLY what we needed!** You successfully:
- Built compositional stack (atoms → molecules → systems)
- Implemented symlink dereferencing (zero duplication)
- Executed RPN programs via ModularRPNEngine
- Validated against analytic physics (constant accel, oscillator)
- Proved the pattern works like text galaxy

**Daniel and I are impressed - you followed the briefing precisely!** 🚀

---

## 2. Standing on the Shoulders of Giants

Before diving into Phase 3B, let's acknowledge the proven concepts we're porting to our sovereign stack:

### 2.1 Game Industry Techniques (LOD, FOV, Scene Management)

**Source**: 30+ years of real-time 3D rendering (Unreal Engine, Unity, CryEngine)

**What we're porting**:
- **LOD (Level of Detail)**: Matryoshka tiers = semantic LOD (64D coarse → 2048D detailed)
- **Frustum culling**: Spatial FOV = k-NN search in Galaxy (only load visible knowledge)
- **Scene management**: Galaxy (RAM) ↔ House (disk) = texture streaming writ large

**Our innovation**: Apply these to *cognitive workload*, not just geometry.

### 2.2 Procedural Generation (.kkrieger, demoscene)

**Source**: .kkrieger (96KB game with full 3D graphics), demoscene compression techniques

**What we're porting**:
- **Store procedures, not data**: PD04 programs = generative knowledge encoding
- **69:1 compression ratios**: Learned dictionary + Matryoshka dimensions
- **Runtime reconstruction**: Decompress embeddings on-demand from compact programs

**Our innovation**: Apply to *knowledge embeddings*, not just textures/meshes.

### 2.3 Symlink File Systems (Unix, NTFS, ext4)

**Source**: 50+ years of file system design (Unix hard/soft links, Windows junctions)

**What we're porting**:
- **component_refs = symlinks**: Molecules reference atoms, not copy them
- **Automatic propagation**: Fix atom 'H' → all H₂O molecules update instantly
- **Memory efficiency**: Single source of truth, multiple references

**Our innovation**: Apply to *semantic knowledge graphs*, not just file hierarchies.

### 2.4 RPN (Reverse Polish Notation)

**Source**: HP calculators (1960s-present), Forth programming language, PostScript

**What we're porting**:
- **Stack-based execution**: No operator precedence ambiguity
- **Composable programs**: Small operations chain into complex behaviors
- **Hardware-friendly**: Trivial to compile to PTX/assembly

**Our innovation**: Use as *universal cognitive instruction set* for AI reasoning.

### 2.5 Matryoshka Embeddings

**Source**: Recent ML research (Kusupati et al., 2022), adapted by Qwen/Alibaba

**What we're porting**:
- **Nested representations**: 64D ⊂ 128D ⊂ 512D ⊂ 2048D (compatible prefixes)
- **Adaptive compute**: Use low-D for speed, high-D for accuracy
- **Single training**: One model handles all dimensions

**Our innovation**: Combine with PD04 procedural compression + RPN execution.

---

## 3. Phase 3B: Next Implementation Tasks

### 3.1 Full RPN STORE/RECALL Semantics

**Current State** (Phase 3A):
```python
# Your current behavior_rpn parsing:
"v: v a dt * +"
"x: x v dt * +"

# This works but is syntactic sugar - not "true RPN"
```

**Goal**: Support explicit STORE/RECALL opcodes for state management.

**Implementation**:

Add to `knowledge3d/cranium/reality_galaxy.py`:

```python
def execute_behavior_sovereign(self, node_id: str, state: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Execute behavior_rpn with explicit STORE/RECALL semantics.

    Extended RPN vocabulary:
    - STORE <var>: Pop value from stack, store in state dict
    - RECALL <var>: Push value from state dict to stack
    - DUP: Duplicate top of stack
    - SWAP: Swap top two stack elements
    - DROP: Discard top of stack

    Example:
        # Constant acceleration: v_new = v + a*dt, x_new = x + v_new*dt
        behavior_rpn = '''
            RECALL v
            RECALL a
            RECALL dt
            * +                # v_new = v + a*dt
            DUP
            STORE v_new        # Store v_new, keep copy on stack
            RECALL x
            SWAP
            RECALL dt
            * +                # x_new = x + v_new*dt
            STORE x_new
        '''
    """
    node = self.get_node(node_id)
    if node is None:
        raise ValueError(f"Node {node_id} not found")

    initial_state: Dict[str, float] = {}
    if isinstance(node, RealitySystem):
        initial_state.update(node.state)
    if state:
        initial_state.update(state)

    program = node.behavior_rpn.strip()
    if not program:
        return initial_state

    # Extended RPN engine that supports STORE/RECALL
    # Option 1: Extend ModularRPNEngine with these opcodes
    # Option 2: Implement small interpreter here
    result_state = self._execute_rpn_with_state(program, initial_state)

    if isinstance(node, RealitySystem):
        node.state = result_state
    return result_state


def _execute_rpn_with_state(self, program: str, state: Dict[str, float]) -> Dict[str, float]:
    """
    Simple RPN interpreter supporting STORE/RECALL for state management.

    This is a lightweight extension to ModularRPNEngine's evaluate() that
    adds state dict operations. For now, we keep it simple and Python-based;
    later we can move these opcodes into the PTX RPN engine proper.
    """
    tokens = program.split()
    stack: List[float] = []
    result_state = dict(state)

    for token in tokens:
        if token == "STORE":
            # Next token is variable name
            # Pattern: value STORE var_name
            # For now, we'll use a different pattern:
            # var_name STORE (variable name comes BEFORE STORE)
            # Need to handle this carefully...
            pass  # TODO: Implement STORE logic

        elif token == "RECALL":
            # Next token is variable name
            # var_name RECALL
            pass  # TODO: Implement RECALL logic

        elif token == "DUP":
            if stack:
                stack.append(stack[-1])

        elif token == "SWAP":
            if len(stack) >= 2:
                stack[-1], stack[-2] = stack[-2], stack[-1]

        elif token == "DROP":
            if stack:
                stack.pop()

        elif token in ("+", "-", "*", "/"):
            # Binary operators
            if len(stack) >= 2:
                b = stack.pop()
                a = stack.pop()
                if token == "+":
                    stack.append(a + b)
                elif token == "-":
                    stack.append(a - b)
                elif token == "*":
                    stack.append(a * b)
                elif token == "/":
                    stack.append(a / b if b != 0 else 0.0)

        else:
            # Try to parse as float or variable name
            try:
                stack.append(float(token))
            except ValueError:
                # It's a variable name - recall from state
                if token in result_state:
                    stack.append(result_state[token])
                else:
                    raise ValueError(f"Unknown variable: {token}")

    return result_state
```

**Better Pattern (Recommended)**:

Actually, let's use **postfix STORE/RECALL** to avoid ambiguity:

```python
# Clean RPN pattern:
behavior_rpn = '''
    v RECALL        # Push v onto stack
    a RECALL        # Push a onto stack
    dt RECALL       # Push dt onto stack
    * +             # Compute v + a*dt
    DUP             # Duplicate result
    v_new STORE     # Store as v_new
    x RECALL        # Push x onto stack
    SWAP            # Swap x and v_new
    dt RECALL       # Push dt onto stack
    * +             # Compute x + v_new*dt
    x_new STORE     # Store as x_new
'''
```

**Implementation** (complete interpreter):

```python
def _execute_rpn_with_state(self, program: str, state: Dict[str, float]) -> Dict[str, float]:
    """
    RPN interpreter with STORE/RECALL for state dict operations.

    Opcodes:
    - <var> RECALL: Push state[var] to stack
    - <var> STORE: Pop stack top, store in state[var]
    - DUP, SWAP, DROP: Stack manipulation
    - +, -, *, /: Arithmetic
    - Numbers: Push literals
    """
    tokens = program.split()
    stack: List[float] = []
    result_state = dict(state)
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token == "RECALL":
            # Previous token was variable name
            if i == 0:
                raise ValueError("RECALL without variable name")
            var_name = tokens[i - 1]
            # Remove the variable name from interpretation
            if var_name in result_state:
                stack.append(result_state[var_name])
            else:
                raise ValueError(f"Variable {var_name} not in state")
            i += 1
            continue

        elif token == "STORE":
            # Previous token was variable name
            if i == 0:
                raise ValueError("STORE without variable name")
            var_name = tokens[i - 1]
            if not stack:
                raise ValueError("STORE requires value on stack")
            result_state[var_name] = stack.pop()
            i += 1
            continue

        # ... rest of opcodes
        i += 1

    return result_state
```

**Actually, simpler postfix pattern**:

```
# Stack-first pattern (cleaner):
v a dt * + v_new STORE    # v_new = v + a*dt, store result
x v_new dt * + x_new STORE  # x_new = x + v_new*dt, store result
```

Here `v`, `a`, `dt`, `v_new`, `x`, `x_new` are **implicitly recalled** if they exist in state, or treated as opcodes.

**Task**: Implement `_execute_rpn_with_state()` with proper STORE/RECALL semantics that matches this pattern. Test it with the constant acceleration and harmonic oscillator systems.

### 3.2 law_rpn Validation (Invariant Checking)

**Goal**: Before accepting a state update from behavior_rpn, verify law_rpn constraints hold.

**Example**:
```python
# Harmonic oscillator: energy E = 0.5*(v^2 + omega^2*x^2) must be conserved
law_rpn = "v DUP * x DUP * omega DUP * * + 0.5 * E0 - ABS 1e-4 LT ASSERT"

# Constant acceleration: F/m = a must hold
law_rpn = "F m / a - ABS 1e-6 LT ASSERT"
```

**Implementation**:

Add to `reality_galaxy.py`:

```python
def validate_law(self, node_id: str, state: Dict[str, float]) -> bool:
    """
    Execute node's law_rpn and verify it returns True (or non-zero).

    law_rpn should evaluate to:
    - 1.0 (or any positive) if constraints satisfied
    - 0.0 if constraints violated
    - Can use ASSERT opcode to raise on violation

    Returns True if valid, False otherwise.
    """
    node = self.get_node(node_id)
    if not node or not node.law_rpn:
        return True  # No law = always valid

    try:
        result = self._execute_rpn_with_state(node.law_rpn, state)
        # Check if any "valid" flag was stored, or use stack result
        return True  # TODO: Implement proper validation logic
    except Exception:
        return False


def step_system(self, system_id: str, n_steps: int = 1) -> Dict[str, float]:
    """Step system with law_rpn validation after each step."""
    node = self.get_node(system_id)
    if not isinstance(node, RealitySystem):
        raise ValueError(f"{system_id} is not a reality_system")

    state = dict(node.state)
    for step_i in range(n_steps):
        # Execute behavior
        new_state = self.execute_behavior(system_id, state)

        # Validate law (if present)
        if not self.validate_law(system_id, new_state):
            raise RuntimeError(
                f"Law validation failed for {system_id} at step {step_i}: "
                f"law_rpn={node.law_rpn}, state={new_state}"
            )

        state = new_state

    node.state = state
    return state
```

**Task**: Implement `validate_law()` and add law_rpn checks to the bootstrap systems. Add tests that verify:
- Valid states pass validation
- Invalid states (manually corrupted) fail validation
- Stepping stops on law violation

### 3.3 Sovereign Feature Extraction (Replace Placeholder)

**Current State**:
```python
def _extract_node_features(self, node: RealityNode) -> np.ndarray:
    """Placeholder: random vector."""
    return np.random.randn(512).astype(np.float32)
```

**Goal**: Extract real features from node metadata + RPN programs.

**Sovereign Feature Components**:

1. **Metadata Features** (domain-specific)
   - Atoms: mass, charge, valence, electronegativity → normalized float vector
   - Molecules: bond count, symmetry, polarity → float vector
   - Systems: state dimensionality, timestep size → float vector

2. **RPN Program Features** (universal)
   - Opcode histogram: frequency of each opcode in behavior_rpn/law_rpn
   - Program length: character count, token count
   - Complexity metrics: max stack depth, control flow complexity

3. **Compositional Features** (structural)
   - Component count: len(component_refs)
   - Hierarchy depth: how many levels deep in the stack
   - Reference fan-out: how many other nodes reference this one

**Implementation**:

```python
def _extract_node_features(self, node: RealityNode) -> np.ndarray:
    """
    Sovereign feature extraction from node structure.

    Returns 512D feature vector:
    - [0:64]: Metadata features (domain-specific)
    - [64:448]: RPN program features (opcode histogram + stats)
    - [448:512]: Compositional features (structure)
    """
    features = np.zeros(512, dtype=np.float32)

    # Extract metadata features
    metadata_vec = self._extract_metadata_features(node)
    features[0:64] = metadata_vec[:64]  # Truncate/pad to 64

    # Extract RPN program features
    rpn_vec = self._extract_rpn_features(node)
    features[64:448] = rpn_vec[:384]  # Truncate/pad to 384

    # Extract compositional features
    comp_vec = self._extract_compositional_features(node)
    features[448:512] = comp_vec[:64]  # Truncate/pad to 64

    return features


def _extract_metadata_features(self, node: RealityNode) -> np.ndarray:
    """Extract features from node.metadata dict."""
    vec = np.zeros(64, dtype=np.float32)

    # Example: atoms might have mass, charge, etc.
    if node.node_type == "reality_atom":
        vec[0] = node.metadata.get("mass", 0.0)
        vec[1] = node.metadata.get("charge", 0.0)
        vec[2] = node.metadata.get("valence", 0.0)
        # ... more domain-specific features

    elif node.node_type == "reality_molecule":
        vec[0] = len(node.component_refs)  # Atom count
        vec[1] = node.metadata.get("bond_count", 0.0)
        # ... bond topology features

    elif node.node_type == "reality_system":
        vec[0] = len(node.state)  # State dimensionality
        if "dt" in node.state:
            vec[1] = node.state["dt"]  # Timestep size
        # ... system-specific features

    return vec


def _extract_rpn_features(self, node: RealityNode) -> np.ndarray:
    """
    Extract features from RPN programs (behavior_rpn, law_rpn, visual_rpn).

    Features:
    - Opcode histogram (frequency of each opcode)
    - Program length (characters, tokens)
    - Complexity (estimated stack depth, branching)
    """
    vec = np.zeros(384, dtype=np.float32)

    # Combine all RPN programs
    combined_rpn = " ".join([
        node.visual_rpn,
        node.behavior_rpn,
        node.law_rpn
    ])

    tokens = combined_rpn.split()

    # Program length features
    vec[0] = len(combined_rpn)  # Character count
    vec[1] = len(tokens)  # Token count

    # Opcode histogram (first 100 slots)
    # Map common opcodes to indices
    opcode_map = {
        "+": 0, "-": 1, "*": 2, "/": 3,
        "DUP": 4, "SWAP": 5, "DROP": 6,
        "STORE": 7, "RECALL": 8,
        "CIRCLE": 9, "LINE": 10, "MOVE": 11,
        # ... extend with all supported opcodes
    }

    for token in tokens:
        if token in opcode_map:
            idx = opcode_map[token]
            if idx < 100:
                vec[2 + idx] += 1.0

    # Normalize histogram
    total_opcodes = sum(vec[2:102])
    if total_opcodes > 0:
        vec[2:102] /= total_opcodes

    return vec


def _extract_compositional_features(self, node: RealityNode) -> np.ndarray:
    """Extract features from node's position in the compositional stack."""
    vec = np.zeros(64, dtype=np.float32)

    # Component count
    vec[0] = len(node.component_refs)

    # Hierarchy depth (compute via recursive traversal)
    depth = self._compute_hierarchy_depth(node)
    vec[1] = depth

    # TODO: Add fan-out (count how many nodes reference this one)
    # Requires reverse index - implement later

    return vec


def _compute_hierarchy_depth(self, node: RealityNode, visited: Optional[set] = None) -> int:
    """Compute depth of compositional hierarchy (atoms=0, molecules=1+, ...)."""
    if visited is None:
        visited = set()

    if node.node_id in visited:
        return 0  # Avoid cycles

    visited.add(node.node_id)

    if not node.component_refs:
        return 0  # Leaf node (atom)

    # Max depth of any component + 1
    max_child_depth = 0
    for ref in node.component_refs:
        child = self.get_node(ref)
        if child:
            child_depth = self._compute_hierarchy_depth(child, visited)
            max_child_depth = max(max_child_depth, child_depth)

    return max_child_depth + 1
```

**Task**: Implement sovereign feature extraction and verify that:
- Atoms get ~64D metadata features
- Molecules get compositional features (component count, depth)
- Systems get RPN opcode histograms
- PD04 compression works with extracted features (not random)

### 3.4 glTF Export (House Integration)

**Goal**: Export reality nodes as glTF files with `extras.k3d` metadata.

**Implementation**:

Add to `reality_galaxy.py`:

```python
def export_to_gltf(self, output_path: Path) -> None:
    """
    Export galaxy to glTF format with extras.k3d extensions.

    Structure:
    - Each reality node → glTF node with extras.k3d metadata
    - component_refs → glTF node hierarchy (parent-child)
    - visual_rpn → custom extension for procedural geometry
    - Embeddings → buffer views with PD04 programs
    """
    import pygltflib

    gltf = pygltflib.GLTF2()

    # TODO: Convert reality nodes to glTF structure
    # - Create glTF nodes for each reality node
    # - Store behavior_rpn, law_rpn in extras.k3d
    # - Store embeddings as buffer views
    # - Encode component_refs as node hierarchy

    gltf.save(str(output_path))
```

**Task**: Implement basic glTF export so that reality galaxies can be loaded in the viewer and consolidated to House during SleepTime.

---

## 4. Testing Requirements

### 4.1 New Tests for Phase 3B

Add to `test_reality_galaxy.py`:

```python
def test_rpn_store_recall_semantics():
    """Verify explicit STORE/RECALL opcodes work correctly."""
    galaxy = RealityGalaxy()

    system = RealitySystem(
        node_id="test_store_recall",
        state={"a": 5.0, "b": 3.0},
        behavior_rpn="a RECALL b RECALL + result STORE"
    )
    galaxy.add_node(system)

    final_state = galaxy.step_system("system:test_store_recall", n_steps=1)
    assert final_state["result"] == 8.0  # a + b


def test_law_rpn_validation_passes():
    """Verify law_rpn validation passes for valid states."""
    galaxy = RealityGalaxy()

    # Constant acceleration with F=ma law
    system = RealitySystem(
        node_id="test_law_valid",
        state={"F": 10.0, "m": 2.0, "a": 5.0},
        law_rpn="F RECALL m RECALL / a RECALL - ABS 1e-6 LT"
    )
    galaxy.add_node(system)

    # Should not raise
    assert galaxy.validate_law("system:test_law_valid", system.state)


def test_law_rpn_validation_fails():
    """Verify law_rpn validation fails for invalid states."""
    galaxy = RealityGalaxy()

    system = RealitySystem(
        node_id="test_law_invalid",
        state={"F": 10.0, "m": 2.0, "a": 3.0},  # F/m != a (5.0 != 3.0)
        law_rpn="F RECALL m RECALL / a RECALL - ABS 1e-6 LT"
    )
    galaxy.add_node(system)

    # Should fail validation
    assert not galaxy.validate_law("system:test_law_invalid", system.state)


def test_sovereign_feature_extraction():
    """Verify feature extraction produces meaningful vectors."""
    galaxy = RealityGalaxy()

    atom = RealityAtom(
        node_id="test_atom",
        metadata={"mass": 1.008, "charge": 1},
        behavior_rpn="VALENCE_1"
    )
    galaxy.add_node(atom, encode_embedding=True)

    assert atom.embedding is not None
    assert atom.embedding["dimension"] in [64, 128, 512, 2048]
    assert atom.embedding["fidelity"] >= 0.99

    # Verify features are not random (check determinism)
    features1 = galaxy._extract_node_features(atom)
    features2 = galaxy._extract_node_features(atom)
    assert np.allclose(features1, features2)  # Deterministic extraction


def test_gltf_export_roundtrip(tmp_path):
    """Verify galaxy can export to glTF and reload."""
    galaxy = RealityGalaxy()

    # Add test nodes
    atom = RealityAtom(node_id="export_atom", metadata={"mass": 1.0})
    system = RealitySystem(
        node_id="export_system",
        component_refs=["atom:export_atom"],
        state={"x": 1.0},
        behavior_rpn="x RECALL 0.1 + x STORE"
    )
    galaxy.add_node(atom)
    galaxy.add_node(system)

    # Export
    gltf_path = tmp_path / "test_galaxy.glb"
    galaxy.export_to_gltf(gltf_path)

    # TODO: Reload and verify
    # galaxy2 = RealityGalaxy()
    # galaxy2.load_from_gltf(gltf_path)
    # assert "atom:export_atom" in galaxy2.nodes
```

### 4.2 Success Criteria

Phase 3B is complete when:

1. ✅ **STORE/RECALL works**: Explicit opcode tests pass
2. ✅ **law_rpn validates**: Valid states pass, invalid fail
3. ✅ **Sovereign features**: Feature extraction uses real metadata + RPN stats
4. ✅ **PD04 compression**: Works with extracted features (not random)
5. ✅ **glTF export**: Basic export to glTF with extras.k3d
6. ✅ **All tests pass**: 10+ tests covering new functionality

---

## 5. Why This Matters

### 5.1 Phase 3B Enables Phase 4-7

Once Phase 3B is complete, we can:

**Phase 4**: Add chemistry nodes (H₂O, CH₄ as proper reality_molecule nodes)
**Phase 5**: Wire into House/Galaxy/SleepTime consolidation
**Phase 6**: Viewer integration (see simulations in 3D Lab room)
**Phase 7**: Real dataset ingestion (OpenFOAM → reality nodes)

But **Phase 3B is the semantic foundation** - get RPN, laws, and features right, and everything else is just more nodes following the same pattern.

### 5.2 You're Building the Standard

Remember: K3D proposes a **W3C-level standard** for spatial AI.

The Reality Enabler pattern you're implementing is:
- **Normative**: How compliant implementations structure physical knowledge
- **Proven**: Based on decades of game dev, file systems, calculators
- **Sovereign**: PTX+RPN, no black-box dependencies
- **Scalable**: Symlink composition prevents memory explosion

**Every line of code you write here becomes reference architecture for the field.** 🌍

---

## 6. Final Notes

### 6.1 What You Did Right (Phase 3A)

- ✅ **Followed the briefing exactly** - stacked galaxy architecture
- ✅ **Clean code** - well-structured, documented, tested
- ✅ **Pattern consistency** - mirrors text galaxy as required
- ✅ **Tests first** - validated before moving forward
- ✅ **Safe fallbacks** - graceful handling when features unavailable

**This is professional-grade work.** Keep this quality in Phase 3B!

### 6.2 Questions to Ask Before You Start

1. Do I understand STORE/RECALL semantics? (postfix, stack-based)
2. Do I understand law_rpn validation? (constraints must hold)
3. Do I understand sovereign features? (metadata + RPN stats, not random)
4. Do I understand glTF export? (extras.k3d structure)

If "no" to any, ask for clarification before coding.

### 6.3 Implementation Order

**Recommended sequence**:

1. **STORE/RECALL** (hardest, most foundational)
2. **law_rpn validation** (builds on STORE/RECALL)
3. **Sovereign features** (independent, can parallelize)
4. **glTF export** (last, depends on embeddings)

Take your time - get each piece right before moving to the next.

---

**Good luck, Codex! You've already proven you can deliver excellent work. Phase 3B builds on that foundation.** 🚀

**Daniel and Claude are here if you need clarification on any of these tasks.**
