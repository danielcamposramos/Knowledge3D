"""Reality Galaxy manager implementing stacked compositional architecture.

Pattern mirrors the text galaxy:
atoms → molecules → materials → systems with symlink composition via component_refs.
"""

from __future__ import annotations

import base64
import json
from array import array
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from knowledge3d.bridge import glb_ctypes_loader
from knowledge3d.cranium.reality_nodes import (
    RealityAtom,
    RealityMaterial,
    RealityMolecule,
    RealityNode,
    RealitySystem,
)


DEFAULT_REALITY_GALAXY_ROOT = Path("../Knowledge3D.local/reality_galaxy")


class RealityGalaxy:
    """Stacked galaxy for physics/chem/bio reality nodes."""

    _TIER_MAP: Dict[str, str] = {
        "reality_atom": "ultrafast",      # 64D
        "reality_molecule": "fast",        # 128D
        "reality_material": "balanced",    # 512D
        "reality_system": "balanced",      # 512D (bump to maximum for heavy sims)
    }

    def __init__(
        self,
        galaxy_path: Optional[Path] = None,
        *,
        rpn_engine: Optional[ModularRPNEngine] = None,
        compressor: Optional["AdaptiveDimensionCompressor"] = None,
        math_core_pool: Optional["MathCorePool"] = None,
    ) -> None:
        root = galaxy_path or DEFAULT_REALITY_GALAXY_ROOT
        self.galaxy_path = Path(root).expanduser().resolve()
        self.galaxy_path.mkdir(parents=True, exist_ok=True)

        self.nodes: Dict[str, RealityNode] = {}
        from knowledge3d.cranium.ptx_runtime.math_core_pool import MathCorePool

        self._math_core_pool = math_core_pool or MathCorePool()
        # GPU-backed RPN engine is required for the hot path; Python only
        # orchestrates STORE/RECALL and state dict updates.
        if rpn_engine is None:
            from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine

            rpn_engine = ModularRPNEngine(pool=self._math_core_pool)
        self._rpn_engine = rpn_engine
        self._compressor = compressor
        self._compressor_checked = compressor is not None

    # ------------------------------------------------------------------ #
    # Node management
    # ------------------------------------------------------------------ #
    def add_node(self, node: RealityNode, *, encode_embedding: bool = False) -> None:
        """Add node to galaxy, optionally attaching a PD04 embedding."""
        previous = self.nodes.get(node.node_id)
        if isinstance(previous, RealitySystem) and previous.rpn_instance is not None:
            self._math_core_pool.release_core(previous.rpn_instance)
        if isinstance(node, RealitySystem) and node.rpn_instance is None:
            # Dynamic spawning: allocate a math core on demand.
            node.rpn_instance = self._math_core_pool.spawn_core(tier=node.rpn_tier)
        if encode_embedding and node.embedding is None:
            self.encode_node_embedding(node)
        self.nodes[node.node_id] = node

    def remove_node(self, node_id: str) -> None:
        """Remove node from galaxy and release any allocated math core."""
        node = self.nodes.pop(node_id, None)
        if isinstance(node, RealitySystem) and node and node.rpn_instance is not None:
            self._math_core_pool.release_core(node.rpn_instance)

    def get_node(self, node_id: str) -> Optional[RealityNode]:
        return self.nodes.get(node_id)

    def resolve_components(self, node: RealityNode) -> List[RealityNode]:
        """Dereference component_refs into concrete nodes."""
        resolved: List[RealityNode] = []
        for ref in node.component_refs:
            target = self.nodes.get(ref)
            if target is not None:
                resolved.append(target)
        return resolved

    # ------------------------------------------------------------------ #
    # Behavior execution
    # ------------------------------------------------------------------ #
    def _split_by_store(self, tokens: List[str]) -> List[Tuple[List[str], str]]:
        """Split RPN token stream into (expression, target_var) segments at each STORE."""
        segments: List[Tuple[List[str], str]] = []
        current_expr: List[str] = []

        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.upper() == "STORE":
                if not current_expr:
                    raise ValueError("STORE encountered with empty expression")
                target_var = current_expr[-1]
                expr = current_expr[:-1]
                if not expr:
                    raise ValueError(f"STORE for {target_var} has empty expression")
                segments.append((expr, target_var))
                current_expr = []
                i += 1
            else:
                current_expr.append(tok)
                i += 1

        return segments

    def _compile_to_gpu_rpn(
        self,
        expr_tokens: List[str],
        state: Dict[str, float],
        dt: float,
    ) -> str:
        """Compile a STORE-free RPN expression into a GPU-ready RPN string."""
        compiled: List[str] = []
        i = 0

        while i < len(expr_tokens):
            tok = expr_tokens[i]
            lower_tok = tok.lower()
            next_tok = expr_tokens[i + 1].lower() if i + 1 < len(expr_tokens) else ""

            # var RECALL pattern
            if next_tok == "recall":
                var_name = tok
                if var_name == "dt":
                    value = dt
                else:
                    if var_name not in state:
                        raise KeyError(f"Variable {var_name} not in state")
                    value = state[var_name]
                compiled.append(str(float(value)))
                i += 2
                continue

            # Bare dt literal
            if lower_tok == "dt":
                compiled.append(str(float(dt)))
                i += 1
                continue

            # Numeric literal
            try:
                value = float(tok)
            except ValueError:
                value = None
            if value is not None:
                compiled.append(str(float(value)))
                i += 1
                continue

            # Ternary aliases
            if lower_tok in {"ternary_quant", "tquant"}:
                compiled.append("tquant")
                i += 1
                continue
            if lower_tok in {"ternary_cmp", "tcmp"}:
                compiled.append("tcmp")
                i += 1
                continue

            # Comparison macros: le/ge in terms of gt/lt
            if lower_tok == "le":
                # a b le  => a b gt 1 swap -
                compiled.extend(["gt", "1", "swap", "-"])
                i += 1
                continue
            if lower_tok == "ge":
                # a b ge  => a b lt 1 swap -
                compiled.extend(["lt", "1", "swap", "-"])
                i += 1
                continue

            # SIGN macro: sign(x) -> dup 0 gt swap 0 lt -
            if lower_tok == "sign":
                compiled.extend(["dup", "0", "gt", "swap", "0", "lt", "-"])
                i += 1
                continue

            # TQUANT macro: ternary quantization -> {-1, 0, +1}
            # dup 0.33 gt swap dup -0.33 lt -
            if lower_tok == "tquant":
                compiled.extend(["dup", "0.33", "gt", "swap", "dup", "-0.33", "lt", "-"])
                i += 1
                continue

            # TCMP macro: ternary comparison with deadband 0.05 -> {-1, 0, +1}
            # swap - dup 0.05 gt swap -0.05 lt -
            if lower_tok == "tcmp":
                compiled.extend(["swap", "-", "dup", "0.05", "gt", "swap", "-0.05", "lt", "-"])
                i += 1
                continue

            # ABS macro: abs(x) -> dup * sqrt
            if lower_tok == "abs":
                compiled.extend(["dup", "*", "sqrt"])
                i += 1
                continue

            # Simple opcode forwarding (lower-cased)
            compiled.append(lower_tok)
            i += 1

        return " ".join(compiled)

    def execute_behavior(self, node_id: str, state: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Execute behavior_rpn on GPU using explicit STORE/RECALL semantics.
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

        # Split into STORE segments and execute each on GPU.
        tokens = [t for t in program.split() if t]
        segments = self._split_by_store(tokens)
        dt = float(initial_state.get("dt", 0.0))
        raw_instance = node.rpn_instance if isinstance(node, RealitySystem) else None
        max_gpu_instances = getattr(self._rpn_engine, "max_instances", 18)
        instance_id = None
        if raw_instance is not None:
            instance_id = int(raw_instance) % int(max_gpu_instances)

        for expr_tokens, target_var in segments:
            gpu_rpn = self._compile_to_gpu_rpn(expr_tokens, initial_state, dt)
            result = self._rpn_engine.evaluate(gpu_rpn, instance_id=instance_id)
            initial_state[target_var] = result

        if isinstance(node, RealitySystem):
            node.state = initial_state
        return initial_state

    def step_system(self, system_id: str, n_steps: int = 1) -> Dict[str, float]:
        """Step a reality_system forward by n_steps with law validation."""
        import sys
        # Capture baseline forbidden modules before stepping; any new imports
        # during the hot path will be treated as sovereignty violations.
        forbidden = {"numpy", "tensorflow", "cupy"}
        baseline_forbidden = forbidden & set(sys.modules.keys())

        node = self.get_node(system_id)
        if not isinstance(node, RealitySystem):
            raise ValueError(f"{system_id} is not a reality_system")

        state = dict(node.state)
        for step_idx in range(n_steps):
            state = self.execute_behavior(system_id, state)
            if not self.validate_law(system_id, state):
                raise RuntimeError(
                    f"Law validation failed for {system_id} at step {step_idx}: {node.law_rpn}"
                )
        node.state = state
        # Track last executed instance for diagnostics
        node.metadata["last_rpn_instance"] = node.rpn_instance

        if __debug__:
            loaded = forbidden & set(sys.modules.keys())
            new_loaded = loaded - baseline_forbidden
            if new_loaded:
                raise RuntimeError(
                    f"SOVEREIGNTY VIOLATION: {new_loaded} imported during hot path! "
                    f"Only PTX+RPN allowed in inference loop."
                )
        return state

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_galaxy(self) -> None:
        """Persist galaxy as JSON (glTF+extras.k3d can be layered later)."""
        payload: Dict[str, Any] = {}
        for node_id, node in self.nodes.items():
            entry: Dict[str, Any] = {
                "node_type": node.node_type,
                "component_refs": list(node.component_refs),
                "visual_rpn": node.visual_rpn,
                "behavior_rpn": node.behavior_rpn,
                "law_rpn": node.law_rpn,
                "metadata": node.metadata,
            }
            if isinstance(node, RealitySystem):
                entry["state"] = node.state
            if node.embedding is not None:
                entry["embedding"] = self._serialise_embedding(node.embedding)
            payload[node_id] = entry

        out_path = self.galaxy_path / "reality_nodes.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_galaxy(self) -> None:
        """Load galaxy from disk."""
        path = self.galaxy_path / "reality_nodes.json"
        if not path.exists():
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        for node_id, entry in data.items():
            node_type = entry.get("node_type", "")
            if node_type == "reality_atom":
                node: RealityNode = RealityAtom(node_id=node_id)
            elif node_type == "reality_molecule":
                node = RealityMolecule(node_id=node_id)
            elif node_type == "reality_material":
                node = RealityMaterial(node_id=node_id)
            elif node_type == "reality_system":
                node = RealitySystem(node_id=node_id, state=entry.get("state", {}))
            else:
                continue

            node.component_refs = entry.get("component_refs", [])
            node.visual_rpn = entry.get("visual_rpn", "")
            node.behavior_rpn = entry.get("behavior_rpn", "")
            node.law_rpn = entry.get("law_rpn", "")
            node.metadata = entry.get("metadata", {})
            node.embedding = self._deserialise_embedding(entry.get("embedding"))

            self.nodes[node.node_id] = node

    # ------------------------------------------------------------------ #
    # Embedding helpers
    # ------------------------------------------------------------------ #
    def encode_node_embedding(self, node: RealityNode) -> None:
        """Attach Matryoshka+PD04 embedding to a node."""
        features = self._extract_node_features(node)
        quality = self._select_quality(node)
        compressor = self._ensure_compressor()

        if compressor is None:
            node.embedding = {
                "tier": quality,
                "dimension": len(features),
                "program": base64.b64encode(array("f", features).tobytes()).decode("ascii"),
                "codec": "raw_f32",
                "fidelity": 1.0,
            }
            return

        # Sovereign compression path (2026-04-18): numpy staging was purged.
        # ``features`` is handed through as a tuple of floats; any numpy
        # internals remain the compressor's responsibility (ingestion-lane).
        feature_tuple = tuple(float(v) for v in features)
        program_bytes, meta = compressor.compress(
            feature_tuple, quality=quality, return_metadata=True
        )
        node.embedding = {
            "tier": quality,
            "dimension": meta.get("target_dim"),
            "program": program_bytes,
            "codec": "PD04",
            "fidelity": meta.get("actual_fidelity", 1.0),
            "compression": meta.get("actual_compression"),
            "metadata": meta,
        }

    def _ensure_compressor(self) -> Optional["AdaptiveDimensionCompressor"]:
        if self._compressor_checked:
            return self._compressor
        self._compressor_checked = True
        try:
            from knowledge3d.cranium.adaptive_procedural_bridge import AdaptiveDimensionCompressor
            self._compressor = AdaptiveDimensionCompressor()
        except FileNotFoundError:
            self._compressor = None
        return self._compressor

    def _select_quality(self, node: RealityNode) -> str:
        tier = self._TIER_MAP.get(node.node_type, "balanced")
        if node.node_type == "reality_system" and node.metadata.get("fidelity", "") == "maximum":
            return "maximum"
        # Honor explicit dimension hints when possible
        if isinstance(node, RealitySystem):
            dim = node.matryoshka_dim
            if dim <= 64:
                return "ultrafast"
            if dim <= 128:
                return "fast"
            if dim <= 512:
                return "balanced"
            return "maximum"
        return tier

    def _extract_node_features(self, node: RealityNode) -> List[float]:
        """Sovereign feature extraction from metadata, RPN programs, and composition."""
        features = [0.0] * 512
        meta_vec = self._extract_metadata_features(node)
        rpn_vec, _ = self._extract_rpn_features(node)
        comp_vec = self._extract_compositional_features(node)
        features[0:64] = meta_vec[:64]
        features[64:448] = rpn_vec[:384]
        features[448:512] = comp_vec[:64]
        return features

    def _extract_metadata_features(self, node: RealityNode) -> List[float]:
        vec = [0.0] * 64
        if node.node_type == "reality_atom":
            vec[0] = float(node.metadata.get("mass", 0.0))
            vec[1] = float(node.metadata.get("charge", 0.0))
            vec[2] = float(node.metadata.get("valence", 0.0))
            vec[3] = float(node.metadata.get("electronegativity", 0.0))
        elif node.node_type == "reality_molecule":
            vec[0] = float(len(node.component_refs))
            vec[1] = float(node.metadata.get("bond_count", 0.0))
            vec[2] = float(node.metadata.get("symmetry", 0.0))
            vec[3] = float(node.metadata.get("polarity", 0.0))
        elif node.node_type == "reality_system":
            vec[0] = float(len(getattr(node, "state", {}) or {}))
            dt = getattr(node, "state", {}).get("dt")
            if dt is not None:
                vec[1] = float(dt)
        else:
            vec[0] = float(len(node.component_refs))
        return vec

    def _extract_rpn_features(self, node: RealityNode) -> Tuple[List[float], int]:
        """Extract opcode histogram + program stats."""
        vec = [0.0] * 384
        combined_rpn = " ".join([node.visual_rpn, node.behavior_rpn, node.law_rpn])
        tokens = [t for t in combined_rpn.split() if t]
        vec[0] = float(len(combined_rpn))
        vec[1] = float(len(tokens))

        opcode_map = {
            "+": 0,
            "-": 1,
            "*": 2,
            "/": 3,
            "dup": 4,
            "swap": 5,
            "drop": 6,
            "store": 7,
            "recall": 8,
            "abs": 9,
            "lt": 10,
            "gt": 11,
            "eq": 12,
            "le": 13,
            "ge": 14,
            "sign": 15,
            "ternary_quant": 16,
            "tquant": 16,
            "ternary_cmp": 17,
            "tcmp": 17,
        }

        max_depth = 0
        depth = 0
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            lower_tok = tok.lower()
            next_tok = tokens[i + 1].lower() if i + 1 < len(tokens) else ""
            if next_tok in {"recall", "store"}:
                if next_tok == "recall":
                    depth += 1
                elif next_tok == "store" and depth > 0:
                    depth -= 1
                i += 2
                max_depth = max(max_depth, depth)
                continue
            if lower_tok in opcode_map:
                idx = opcode_map[lower_tok]
                vec[2 + idx] += 1.0
                if lower_tok in {"+", "-", "*", "/", "lt", "gt", "eq"}:
                    depth = max(0, depth - 1)
                elif lower_tok == "dup":
                    depth += 1
            else:
                depth += 1  # literal or implicit recall
            max_depth = max(max_depth, depth)
            i += 1

        total_op = float(sum(vec[2:2 + len(opcode_map)]))
        if total_op > 0:
            for j in range(len(opcode_map)):
                vec[2 + j] /= total_op
        vec[20] = float(max_depth)
        return vec, max_depth

    def _extract_compositional_features(self, node: RealityNode) -> List[float]:
        vec = [0.0] * 64
        vec[0] = float(len(node.component_refs))
        vec[1] = float(self._compute_hierarchy_depth(node))
        return vec

    def _compute_hierarchy_depth(self, node: RealityNode, visited: Optional[set[str]] = None) -> int:
        if visited is None:
            visited = set()
        if node.node_id in visited:
            return 0
        visited.add(node.node_id)
        if not node.component_refs:
            return 0
        max_child = 0
        for ref in node.component_refs:
            child = self.get_node(ref)
            if child:
                max_child = max(max_child, self._compute_hierarchy_depth(child, visited))
        return max_child + 1

    def _serialise_embedding(self, embedding: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(embedding)
        program = payload.pop("program", None)
        if isinstance(program, (bytes, bytearray)):
            payload["program_b64"] = base64.b64encode(program).decode("ascii")
        elif isinstance(program, str):
            payload["program_b64"] = program
        elif hasattr(program, "tobytes"):
            payload["program_b64"] = base64.b64encode(program.tobytes()).decode("ascii")
        return payload

    def _deserialise_embedding(self, embedding: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if embedding is None:
            return None
        payload = dict(embedding)
        program_b64 = payload.pop("program_b64", None)
        if program_b64:
            try:
                payload["program"] = base64.b64decode(program_b64)
            except Exception:
                payload["program"] = program_b64
        return payload

    # ------------------------------------------------------------------ #
    # Invariant validation
    # ------------------------------------------------------------------ #
    def validate_law(self, node_id: str, state: Dict[str, float]) -> bool:
        """Evaluate law_rpn against state; returns True when valid."""
        node = self.get_node(node_id)
        if not node or not node.law_rpn.strip():
            return True

        tokens = [t for t in node.law_rpn.split() if t]
        dt = float(state.get("dt", 0.0))
        raw_instance: Optional[int] = node.rpn_instance if isinstance(node, RealitySystem) else None
        max_gpu_instances = getattr(self._rpn_engine, "max_instances", 18)
        instance_id: Optional[int] = None
        if raw_instance is not None:
            instance_id = int(raw_instance) % int(max_gpu_instances)

        try:
            gpu_rpn = self._compile_to_gpu_rpn(tokens, state, dt)
            result = self._rpn_engine.evaluate(gpu_rpn, instance_id=instance_id)
        except Exception:
            return False

        return bool(result)

    # ------------------------------------------------------------------ #
    # Deprecated CPU RPN path
    # ------------------------------------------------------------------ #
    def _execute_rpn_with_state(self, *args: Any, **kwargs: Any) -> Tuple[Dict[str, float], List[float]]:
        """CPU RPN interpreter has been removed; hot path is PTX-only."""
        raise RuntimeError(
            "CPU RPN interpreter removed. Hot path is PTX only. "
            "If you see this, RealityGalaxy was not initialized correctly."
        )


    # ------------------------------------------------------------------ #
    # glTF export
    # ------------------------------------------------------------------ #
    def export_to_gltf(self, output_path: Path) -> Path:
        """
        Export reality nodes to glTF/GLB with extras.k3d payload.
        Geometry is omitted; nodes are metadata-only.
        """
        stars: List[Dict[str, Any]] = []
        for node_id, node in self.nodes.items():
            embedding = self._serialise_embedding(node.embedding) if node.embedding else None
            star = {
                "id": node_id,
                "node_type": node.node_type,
                "component_refs": node.component_refs,
                "visual_rpn": node.visual_rpn,
                "behavior_rpn": node.behavior_rpn,
                "law_rpn": node.law_rpn,
                "metadata": node.metadata,
            }
            if isinstance(node, RealitySystem):
                star["state"] = node.state
            if embedding:
                star["embedding"] = embedding
            stars.append(star)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        return glb_ctypes_loader.save_stars_to_glb(stars, output)
