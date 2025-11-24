"""Reality Galaxy manager implementing stacked compositional architecture.

Pattern mirrors the text galaxy:
atoms → molecules → materials → systems with symlink composition via component_refs.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from knowledge3d.bridge import glb_ctypes_loader
from knowledge3d.cranium.adaptive_procedural_bridge import AdaptiveDimensionCompressor
from knowledge3d.cranium.reality_nodes import (
    RealityAtom,
    RealityMaterial,
    RealityMolecule,
    RealityNode,
    RealitySystem,
)
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine


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
        compressor: Optional[AdaptiveDimensionCompressor] = None,
        math_core_pool: Optional["MathCorePool"] = None,
    ) -> None:
        root = galaxy_path or DEFAULT_REALITY_GALAXY_ROOT
        self.galaxy_path = Path(root).expanduser().resolve()
        self.galaxy_path.mkdir(parents=True, exist_ok=True)

        self.nodes: Dict[str, RealityNode] = {}
        from knowledge3d.cranium.ptx_runtime.math_core_pool import MathCorePool

        self._math_core_pool = math_core_pool or MathCorePool()
        self._rpn = rpn_engine or ModularRPNEngine(pool=self._math_core_pool)
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
    def execute_behavior(self, node_id: str, state: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Execute behavior_rpn using explicit STORE/RECALL stack semantics.

        Example (postfix):
            v RECALL a RECALL dt RECALL * + v STORE
            x RECALL v RECALL dt RECALL * + x STORE
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

        result_state, _ = self._execute_rpn_with_state(
            program,
            initial_state,
            instance_id=node.rpn_instance if isinstance(node, RealitySystem) else 0,
        )
        if isinstance(node, RealitySystem):
            node.state = result_state
        return result_state

    def step_system(self, system_id: str, n_steps: int = 1) -> Dict[str, float]:
        """Step a reality_system forward by n_steps with law validation."""
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
                "dimension": features.size,
                "program": base64.b64encode(features.tobytes()).decode("ascii"),
                "codec": "raw_f32",
                "fidelity": 1.0,
            }
            return

        program_bytes, meta = compressor.compress(features, quality=quality, return_metadata=True)
        node.embedding = {
            "tier": quality,
            "dimension": meta.get("target_dim"),
            "program": program_bytes,
            "codec": "PD04",
            "fidelity": meta.get("actual_fidelity", 1.0),
            "compression": meta.get("actual_compression"),
            "metadata": meta,
        }

    def _ensure_compressor(self) -> Optional[AdaptiveDimensionCompressor]:
        if self._compressor_checked:
            return self._compressor
        self._compressor_checked = True
        try:
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

    def _extract_node_features(self, node: RealityNode) -> np.ndarray:
        """Sovereign feature extraction from metadata, RPN programs, and composition."""
        features = np.zeros(512, dtype=np.float32)
        meta_vec = self._extract_metadata_features(node)
        rpn_vec, _ = self._extract_rpn_features(node)
        comp_vec = self._extract_compositional_features(node)
        features[0:64] = meta_vec[:64]
        features[64:448] = rpn_vec[:384]
        features[448:512] = comp_vec[:64]
        return features

    def _extract_metadata_features(self, node: RealityNode) -> np.ndarray:
        vec = np.zeros(64, dtype=np.float32)
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

    def _extract_rpn_features(self, node: RealityNode) -> Tuple[np.ndarray, int]:
        """Extract opcode histogram + program stats."""
        vec = np.zeros(384, dtype=np.float32)
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

        total_op = float(np.sum(vec[2:2 + len(opcode_map)]))
        if total_op > 0:
            vec[2:2 + len(opcode_map)] /= total_op
        vec[20] = float(max_depth)
        return vec, max_depth

    def _extract_compositional_features(self, node: RealityNode) -> np.ndarray:
        vec = np.zeros(64, dtype=np.float32)
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
        elif isinstance(program, np.ndarray):
            payload["program_b64"] = base64.b64encode(program.tobytes()).decode("ascii")
        elif isinstance(program, str):
            payload["program_b64"] = program
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
        try:
            _, stack = self._execute_rpn_with_state(node.law_rpn, state, return_stack=True)
        except Exception:
            return False
        if stack:
            return bool(stack[-1])
        return True

    # ------------------------------------------------------------------ #
    # RPN with STORE/RECALL
    # ------------------------------------------------------------------ #
    def _execute_rpn_with_state(
        self,
        program: str,
        state: Dict[str, float],
        *,
        return_stack: bool = False,
        instance_id: int = 0,
    ) -> Tuple[Dict[str, float], List[float]]:
        tokens = [t for t in program.split() if t]
        stack: List[float] = []
        result_state = dict(state)
        i = 0

        def pop2() -> Tuple[float, float]:
            if len(stack) < 2:
                raise ValueError("Insufficient operands")
            b = stack.pop()
            a = stack.pop()
            return a, b

        while i < len(tokens):
            tok = tokens[i]
            next_tok = tokens[i + 1].lower() if i + 1 < len(tokens) else ""
            lower_tok = tok.lower()

            if next_tok == "recall":
                if tok not in result_state:
                    raise ValueError(f"Variable {tok} not in state")
                stack.append(float(result_state[tok]))
                i += 2
                continue

            if next_tok == "store":
                if not stack:
                    raise ValueError("STORE requires value on stack")
                result_state[tok] = float(stack.pop())
                i += 2
                continue

            if lower_tok in {"+", "-", "*", "/"}:
                a, b = pop2()
                if lower_tok == "+":
                    stack.append(a + b)
                elif lower_tok == "-":
                    stack.append(a - b)
                elif lower_tok == "*":
                    stack.append(a * b)
                elif lower_tok == "/":
                    stack.append(a / b if b != 0 else 0.0)
                i += 1
                continue

            if lower_tok == "neg":
                if not stack:
                    raise ValueError("NEG requires value on stack")
                stack.append(-stack.pop())
                i += 1
                continue

            if lower_tok == "dup":
                if stack:
                    stack.append(stack[-1])
                i += 1
                continue

            if lower_tok == "swap":
                if len(stack) >= 2:
                    stack[-1], stack[-2] = stack[-2], stack[-1]
                i += 1
                continue

            if lower_tok == "drop":
                if stack:
                    stack.pop()
                i += 1
                continue

            if lower_tok == "abs":
                if not stack:
                    raise ValueError("ABS requires value on stack")
                stack.append(abs(stack.pop()))
                i += 1
                continue

            if lower_tok == "sqrt":
                if not stack:
                    raise ValueError("SQRT requires value on stack")
                val = stack.pop()
                stack.append(float(np.sqrt(val)))
                i += 1
                continue

            if lower_tok == "sign":
                if not stack:
                    raise ValueError("SIGN requires value on stack")
                val = stack.pop()
                stack.append(-1.0 if val < 0 else (1.0 if val > 0 else 0.0))
                i += 1
                continue

            if lower_tok in {"ternary_quant", "tquant"}:
                if len(stack) < 2:
                    raise ValueError("TERNARY_QUANT requires value and threshold on stack")
                thresh = stack.pop()
                val = stack.pop()
                if val > thresh:
                    stack.append(1.0)
                elif val < -thresh:
                    stack.append(-1.0)
                else:
                    stack.append(0.0)
                i += 1
                continue

            if lower_tok in {"ternary_cmp", "tcmp"}:
                a, b = pop2()
                diff = a - b
                stack.append(-1.0 if diff < 0 else (1.0 if diff > 0 else 0.0))
                i += 1
                continue

            if lower_tok in {"lt", "gt", "eq", "le", "ge"}:
                a, b = pop2()
                if lower_tok == "lt":
                    stack.append(1.0 if a < b else 0.0)
                elif lower_tok == "gt":
                    stack.append(1.0 if a > b else 0.0)
                elif lower_tok == "eq":
                    stack.append(1.0 if a == b else 0.0)
                elif lower_tok == "le":
                    stack.append(1.0 if a <= b else 0.0)
                elif lower_tok == "ge":
                    stack.append(1.0 if a >= b else 0.0)
                i += 1
                continue

            if lower_tok == "assert":
                if not stack:
                    raise ValueError("ASSERT requires value on stack")
                val = stack.pop()
                if not val:
                    raise AssertionError("Law assertion failed")
                i += 1
                continue

            # literal or implicit recall
            try:
                stack.append(float(tok))
            except ValueError:
                if tok in result_state:
                    stack.append(float(result_state[tok]))
                else:
                    raise ValueError(f"Unknown token: {tok}")
            i += 1

        return (result_state, stack if return_stack else [])

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
