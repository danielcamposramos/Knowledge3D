"""
Semantic Bridge Renderer - GLM's visualization system.

Converts abstract bridge metadata into visual elements for human navigation.
Bridges appear as glowing portals connecting semantic domains in the House.

Architecture:
- Bridge strength → visual intensity (brightness)
- Domain crossing → hue (color based on source/destination domains)
- Usage frequency → saturation (frequently used bridges glow brighter)

Authors:
- GLM 4.6: Semantic bridge rendering concept
- Qwen-Max: Integration with House GLB export
"""

import cupy as cp
import numpy as np
from typing import List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BridgeVisual:
    """Visual properties for a semantic bridge."""
    src_node: int
    dst_node: int
    src_domain: int
    dst_domain: int
    strength: float  # Semantic similarity
    intensity: float  # Visual brightness [0, 1]
    hue: float  # Color hue [0, 360]
    saturation: float  # Color saturation [0, 1]


class SemanticBridgeRenderer:
    """
    Renders semantic bridges as visual elements for human perception.

    GLM's insight: Abstract bridges must be spatially embodied as glowing
    portals that humans can see and navigate through.
    """

    def __init__(self):
        """Initialize bridge renderer."""
        self.bridge_visuals: List[BridgeVisual] = []
        self.usage_stats: Optional[cp.ndarray] = None  # Bridge usage frequency

    def render_bridges(
        self,
        bridges: cp.ndarray,  # (M, 2) edge indices
        domain_ids: cp.ndarray,  # (N,) node→domain mapping
        embeddings_gpu: cp.ndarray,  # (N, D) embeddings
        positions_gpu: cp.ndarray,  # (N, 3) positions
        usage_stats: Optional[cp.ndarray] = None  # (M,) usage counts
    ) -> List[BridgeVisual]:
        """
        Convert bridge metadata into visual properties.

        Args:
            bridges: Cross-domain edge indices
            domain_ids: Node to domain assignments
            embeddings_gpu: Node embeddings (for strength calculation)
            positions_gpu: Node positions (for portal placement)
            usage_stats: Optional usage frequency per bridge

        Returns:
            List of BridgeVisual objects with rendering properties
        """
        logger.info("Rendering semantic bridges for visualization...")

        n_bridges = len(bridges)
        if n_bridges == 0:
            logger.warning("No bridges to render")
            return []

        # Transfer to CPU for processing
        bridges_cpu = bridges.get()
        domain_ids_cpu = domain_ids.get()
        embeddings_cpu = embeddings_gpu.get()

        bridge_visuals = []

        for i, (src, dst) in enumerate(bridges_cpu):
            # Get domain info
            src_domain = int(domain_ids_cpu[src])
            dst_domain = int(domain_ids_cpu[dst])

            # Compute semantic strength (cosine similarity)
            src_emb = embeddings_cpu[src]
            dst_emb = embeddings_cpu[dst]
            strength = float(np.dot(src_emb, dst_emb) / (np.linalg.norm(src_emb) * np.linalg.norm(dst_emb) + 1e-8))

            # Map strength to visual intensity
            # Stronger bridges (>0.85) glow brighter
            intensity = min(1.0, strength * 0.8 + 0.2)  # Range [0.2, 1.0]

            # Map domain crossing to hue
            hue = self._domain_crossing_to_hue(src_domain, dst_domain)

            # Saturation from usage (if available)
            if usage_stats is not None:
                usage = float(usage_stats[i].get()) if hasattr(usage_stats[i], 'get') else float(usage_stats[i])
                # Log scale: frequently used → high saturation
                saturation = min(1.0, 0.5 + 0.1 * np.log1p(usage))
            else:
                saturation = 0.8  # Default saturation

            visual = BridgeVisual(
                src_node=int(src),
                dst_node=int(dst),
                src_domain=src_domain,
                dst_domain=dst_domain,
                strength=float(strength),
                intensity=float(intensity),
                hue=float(hue),
                saturation=float(saturation)
            )
            bridge_visuals.append(visual)

        logger.info(f"✓ Rendered {len(bridge_visuals)} semantic bridges")
        logger.info(f"  Strength range: {min(v.strength for v in bridge_visuals):.2f} - {max(v.strength for v in bridge_visuals):.2f}")
        logger.info(f"  Intensity range: {min(v.intensity for v in bridge_visuals):.2f} - {max(v.intensity for v in bridge_visuals):.2f}")

        self.bridge_visuals = bridge_visuals
        return bridge_visuals

    def _domain_crossing_to_hue(self, src_domain: int, dst_domain: int) -> float:
        """
        Map domain crossing to color hue.

        Different domain transitions get different colors to help humans
        visually distinguish different types of conceptual jumps.

        Args:
            src_domain: Source domain ID
            dst_domain: Destination domain ID

        Returns:
            Hue in degrees [0, 360]
        """
        # Hash the domain pair to a consistent hue
        # Use golden ratio for even distribution on color wheel
        golden_ratio = 0.618033988749895
        hash_val = (src_domain * 73 + dst_domain * 137) % 360
        hue = (hash_val * golden_ratio * 360) % 360

        return hue

    def export_to_glb_metadata(self, positions_gpu: cp.ndarray) -> dict:
        """
        Export bridge visuals as GLB metadata for the House.

        Creates line segments with color/intensity for WebGL rendering.

        Args:
            positions_gpu: Node positions for computing bridge endpoints

        Returns:
            Dictionary with bridge visual data for GLB export
        """
        if not self.bridge_visuals:
            return {"bridges": [], "bridge_count": 0}

        positions_cpu = positions_gpu.get()

        bridge_metadata = []
        for visual in self.bridge_visuals:
            src_pos = positions_cpu[visual.src_node].tolist()
            dst_pos = positions_cpu[visual.dst_node].tolist()

            bridge_metadata.append({
                "src_pos": src_pos,
                "dst_pos": dst_pos,
                "src_node": visual.src_node,
                "dst_node": visual.dst_node,
                "src_domain": visual.src_domain,
                "dst_domain": visual.dst_domain,
                "strength": visual.strength,
                "visual": {
                    "intensity": visual.intensity,
                    "hue": visual.hue,
                    "saturation": visual.saturation,
                    # Convert HSV to RGB for rendering
                    "rgb": self._hsv_to_rgb(visual.hue, visual.saturation, visual.intensity)
                }
            })

        logger.info(f"Exported {len(bridge_metadata)} bridges to GLB metadata")

        return {
            "bridges": bridge_metadata,
            "bridge_count": len(bridge_metadata),
            "rendering_mode": "semantic_portals"
        }

    def _hsv_to_rgb(self, h: float, s: float, v: float) -> List[float]:
        """
        Convert HSV to RGB for rendering.

        Args:
            h: Hue [0, 360]
            s: Saturation [0, 1]
            v: Value/brightness [0, 1]

        Returns:
            [r, g, b] in range [0, 1]
        """
        h = h / 60.0
        c = v * s
        x = c * (1 - abs(h % 2 - 1))
        m = v - c

        if 0 <= h < 1:
            r, g, b = c, x, 0
        elif 1 <= h < 2:
            r, g, b = x, c, 0
        elif 2 <= h < 3:
            r, g, b = 0, c, x
        elif 3 <= h < 4:
            r, g, b = 0, x, c
        elif 4 <= h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        return [r + m, g + m, b + m]

    def visualize_cross_domain_path(
        self,
        path_indices: List[int],
        domain_ids: cp.ndarray,
        positions_gpu: cp.ndarray
    ) -> dict:
        """
        Visualize a cross-domain reasoning path with highlighted transitions.

        GLM's insight: Show humans where the AI's reasoning crosses conceptual boundaries.

        Args:
            path_indices: Sequence of node indices in the path
            domain_ids: Node to domain mapping
            positions_gpu: Node positions

        Returns:
            Path visualization metadata
        """
        if len(path_indices) < 2:
            return {"segments": [], "transitions": []}

        domain_ids_cpu = domain_ids.get()
        positions_cpu = positions_gpu.get()

        segments = []
        transitions = []

        for i in range(len(path_indices) - 1):
            src_idx = path_indices[i]
            dst_idx = path_indices[i + 1]

            src_domain = int(domain_ids_cpu[src_idx])
            dst_domain = int(domain_ids_cpu[dst_idx])

            # Check if this is a domain transition
            is_transition = (src_domain != dst_domain)

            segment = {
                "src_node": src_idx,
                "dst_node": dst_idx,
                "src_pos": positions_cpu[src_idx].tolist(),
                "dst_pos": positions_cpu[dst_idx].tolist(),
                "src_domain": src_domain,
                "dst_domain": dst_domain,
                "is_transition": is_transition,
                "intensity": 1.0 if is_transition else 0.5  # Highlight transitions
            }
            segments.append(segment)

            if is_transition:
                transitions.append({
                    "index": i,
                    "from_domain": src_domain,
                    "to_domain": dst_domain,
                    "position": positions_cpu[src_idx].tolist()
                })

        logger.info(f"Visualized path: {len(segments)} segments, {len(transitions)} domain transitions")

        return {
            "segments": segments,
            "transitions": transitions,
            "total_domains": len(set(int(domain_ids_cpu[i]) for i in path_indices))
        }
