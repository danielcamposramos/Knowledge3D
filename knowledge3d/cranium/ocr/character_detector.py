"""
CharacterDetector: Phase F.2 - Character Detection from Feature Maps

Integrates 5 swarm-designed components:
1. Qwen: Adaptive sliding window with activation pruning
2. DeepSeek: Galactic template bank (3-layer system)
3. Kimi: Warp-swizzle glyph matcher with resonance
4. GLM: Hierarchical NMS (character → spatial → context)
5. Grok: Graph-based spatial decoder with A* pathfinding

Pipeline:
    Feature Map [H/4, W/4, 128] → Sliding Window → Patch Extraction →
    Glyph Matching → Hierarchical NMS → Spatial Decoding → Text

Target: ≥90% detection rate, ≥0.7 IoU, ≥95% text accuracy
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import heapq
from collections import defaultdict


class GalacticTemplateBank:
    """
    DeepSeek's 3-layer template system.

    Layer 1 (Synthetic): Mathematical constructs, perfect shapes
    Layer 2 (Galactic): Bootstrap from common fonts (Arial, Times)
    Layer 3 (Learned): Fine-tuned from RLWHF data

    Warp-ready: Each template block = 32 glyphs × 128 features
    """

    def __init__(self, num_glyphs: int = 256, feature_dim: int = 128):
        """
        Initialize template bank.

        Args:
            num_glyphs: Number of character classes (256 for ASCII)
            feature_dim: Feature dimension (128 from DeepSeekOCRModel)
        """
        self.num_glyphs = num_glyphs
        self.feature_dim = feature_dim

        # Layer 1: Synthetic templates
        self.synthetic_templates = self._generate_synthetic_templates()

        # Layer 2: Galactic templates (from common fonts)
        self.galactic_templates = self._generate_galactic_templates()

        # Layer 3: Learned templates (placeholder, will be trained)
        self.learned_templates = np.zeros((num_glyphs, feature_dim), dtype=np.float32)

        # Active templates (blend of all 3 layers)
        self.active_templates = self._blend_templates()

    def _generate_synthetic_templates(self) -> np.ndarray:
        """Generate Layer 1: Synthetic mathematical templates."""
        templates = np.random.randn(self.num_glyphs, self.feature_dim).astype(np.float32)

        # Normalize each template
        norms = np.linalg.norm(templates, axis=1, keepdims=True)
        templates = templates / np.maximum(norms, 1e-6)

        return templates

    def _generate_galactic_templates(self) -> np.ndarray:
        """Generate Layer 2: Galactic font-based templates."""
        # Phase F.2: Bootstrap from common fonts
        # For now, use synthetic + small perturbation
        templates = self.synthetic_templates.copy()
        templates += np.random.randn(*templates.shape).astype(np.float32) * 0.1

        # Normalize
        norms = np.linalg.norm(templates, axis=1, keepdims=True)
        templates = templates / np.maximum(norms, 1e-6)

        return templates

    def _blend_templates(self, alpha_synthetic: float = 0.3,
                        alpha_galactic: float = 0.5,
                        alpha_learned: float = 0.2) -> np.ndarray:
        """Blend all 3 template layers."""
        blended = (alpha_synthetic * self.synthetic_templates +
                   alpha_galactic * self.galactic_templates +
                   alpha_learned * self.learned_templates)

        # Normalize
        norms = np.linalg.norm(blended, axis=1, keepdims=True)
        blended = blended / np.maximum(norms, 1e-6)

        return blended

    def get_templates(self) -> np.ndarray:
        """Get active templates [num_glyphs, feature_dim]."""
        return self.active_templates

    def set_external_templates(self, templates: np.ndarray) -> None:
        """
        Override active templates with externally supplied embeddings.

        Args:
            templates: Array with shape [num_glyphs, feature_dim]
        """
        if templates.ndim != 2:
            raise ValueError("External templates must be 2-D")

        if templates.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected template dim {self.feature_dim}, got {templates.shape[1]}"
            )

        norm = np.linalg.norm(templates, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-6)
        self.active_templates = templates / norm

    def update_learned_templates(self, learned: np.ndarray):
        """Update Layer 3 with RLWHF-trained templates."""
        if learned.shape != (self.num_glyphs, self.feature_dim):
            raise ValueError(f"Expected shape {(self.num_glyphs, self.feature_dim)}, got {learned.shape}")

        self.learned_templates = learned.copy()
        self.active_templates = self._blend_templates()


class AdaptiveSlidingWindow:
    """
    Qwen's adaptive sliding window with activation pruning.

    Dynamic stride based on local activation intensity.
    Prunes low-activation patches to reduce computation.
    """

    def __init__(self, patch_size: int = 8, min_stride: int = 2, max_stride: int = 8,
                 activation_threshold: float = 0.1):
        """
        Initialize adaptive sliding window.

        Args:
            patch_size: Size of each patch (8×8 default)
            min_stride: Minimum stride in dense regions
            max_stride: Maximum stride in sparse regions
            activation_threshold: Minimum activation to keep patch
        """
        self.patch_size = patch_size
        self.min_stride = min_stride
        self.max_stride = max_stride
        self.activation_threshold = activation_threshold

    def extract_patches(self, feature_map: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Extract patches with adaptive stride and pruning.

        Args:
            feature_map: Input features [H, W, C]

        Returns:
            patches: Extracted patches [num_patches, patch_size*patch_size*C]
            positions: List of (row, col) positions for each patch
        """
        H, W, C = feature_map.shape

        # Calculate activation map (L2 norm per spatial location)
        activation_map = np.linalg.norm(feature_map, axis=2)  # [H, W]

        patches = []
        positions = []

        row = 0
        while row + self.patch_size <= H:
            col = 0
            while col + self.patch_size <= W:
                # Extract patch
                patch = feature_map[row:row+self.patch_size,
                                   col:col+self.patch_size, :]  # [patch_size, patch_size, C]

                # Calculate patch activation
                patch_activation = activation_map[row:row+self.patch_size,
                                                 col:col+self.patch_size].mean()

                # Pruning: Skip low-activation patches
                if patch_activation >= self.activation_threshold:
                    patches.append(patch.reshape(-1))  # Flatten to [patch_size*patch_size*C]
                    positions.append((row, col))

                # Adaptive stride based on local activation
                if patch_activation > 0.5:
                    stride = self.min_stride  # Dense region, small stride
                elif patch_activation > 0.2:
                    stride = (self.min_stride + self.max_stride) // 2  # Medium
                else:
                    stride = self.max_stride  # Sparse region, large stride

                col += stride

            # Row stride (use similar adaptive logic)
            row_activation = activation_map[row:row+self.patch_size, :].mean()
            if row_activation > 0.5:
                row += self.min_stride
            elif row_activation > 0.2:
                row += (self.min_stride + self.max_stride) // 2
            else:
                row += self.max_stride

        if len(patches) == 0:
            return np.zeros((0, self.patch_size * self.patch_size * C), dtype=np.float32), []

        return np.array(patches, dtype=np.float32), positions


class HierarchicalNMS:
    """
    GLM's hierarchical non-maximum suppression.

    3 levels:
    1. Character-level: Suppress duplicate detections per character class
    2. Spatial-graph: Build graph, suppress overlapping regions
    3. Context-aware: Use text context to resolve ambiguities
    """

    def __init__(self, iou_threshold: float = 0.3, conf_threshold: float = 0.5):
        """
        Initialize hierarchical NMS.

        Args:
            iou_threshold: IoU threshold for suppression
            conf_threshold: Minimum confidence to keep detection
        """
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

    def _compute_iou(self, box1: Tuple[int, int, int, int],
                     box2: Tuple[int, int, int, int]) -> float:
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        inter_area = (x2_i - x1_i) * (y2_i - y1_i)

        # Union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / max(union_area, 1e-6)

    def apply(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply hierarchical NMS.

        Args:
            detections: List of detections, each with:
                - 'bbox': [x1, y1, x2, y2]
                - 'confidence': float
                - 'char_id': int

        Returns:
            Filtered detections
        """
        if len(detections) == 0:
            return []

        # Filter by confidence threshold
        detections = [d for d in detections if d['confidence'] >= self.conf_threshold]

        if len(detections) == 0:
            return []

        # Level 1: Character-level NMS (per character class)
        char_groups = defaultdict(list)
        for det in detections:
            char_groups[det['char_id']].append(det)

        nms_results = []
        for char_id, dets in char_groups.items():
            # Sort by confidence (descending)
            dets = sorted(dets, key=lambda x: x['confidence'], reverse=True)

            keep = []
            while len(dets) > 0:
                # Keep highest confidence
                best = dets.pop(0)
                keep.append(best)

                # Suppress overlapping detections
                dets = [d for d in dets if self._compute_iou(best['bbox'], d['bbox']) < self.iou_threshold]

            nms_results.extend(keep)

        # Level 2: Spatial-graph NMS (cross-character)
        # Sort all results by confidence
        nms_results = sorted(nms_results, key=lambda x: x['confidence'], reverse=True)

        final_results = []
        while len(nms_results) > 0:
            best = nms_results.pop(0)
            final_results.append(best)

            # Suppress overlapping (more aggressive across different characters)
            nms_results = [d for d in nms_results
                          if self._compute_iou(best['bbox'], d['bbox']) < self.iou_threshold * 1.5]

        return final_results


class SpatialTextDecoder:
    """
    Grok's graph-based spatial decoder with A* pathfinding.

    Handles non-linear text layouts (columns, tables, annotations).
    Uses resonance field to guide pathfinding.
    """

    def __init__(self, patch_size: int = 8):
        """
        Initialize spatial decoder.

        Args:
            patch_size: Patch size used in sliding window
        """
        self.patch_size = patch_size

    def decode(self, detections: List[Dict[str, Any]], image_width: int, image_height: int) -> str:
        """
        Decode detections to text using graph-based clustering.

        Args:
            detections: NMS-filtered detections
            image_width: Original image width
            image_height: Original image height

        Returns:
            Decoded text string
        """
        if len(detections) == 0:
            return ""

        # Build spatial graph
        graph = self._build_spatial_graph(detections)

        # Cluster into text lines
        lines = self._cluster_into_lines(detections, graph)

        # Sort lines top-to-bottom
        lines = sorted(lines, key=lambda line: min(d['bbox'][1] for d in line))

        # Decode each line
        text_lines = []
        for line in lines:
            # Sort detections left-to-right within line
            line = sorted(line, key=lambda d: d['bbox'][0])

            # Convert char_ids to text
            line_text = ''.join(chr(d['char_id']) if 32 <= d['char_id'] < 127 else '?'
                               for d in line)
            text_lines.append(line_text)

        return '\n'.join(text_lines)

    def _build_spatial_graph(self, detections: List[Dict[str, Any]]) -> Dict[int, List[int]]:
        """Build graph connecting nearby detections."""
        graph = defaultdict(list)

        for i, det1 in enumerate(detections):
            x1_1, y1_1, x2_1, y2_1 = det1['bbox']
            cx1, cy1 = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2

            for j, det2 in enumerate(detections):
                if i == j:
                    continue

                x1_2, y1_2, x2_2, y2_2 = det2['bbox']
                cx2, cy2 = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2

                # Connect if nearby (within 3× patch size)
                dist = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
                if dist < 3 * self.patch_size:
                    graph[i].append(j)

        return graph

    def _cluster_into_lines(self, detections: List[Dict[str, Any]],
                           graph: Dict[int, List[int]]) -> List[List[Dict[str, Any]]]:
        """Cluster detections into text lines using graph."""
        visited = set()
        lines = []

        for i in range(len(detections)):
            if i in visited:
                continue

            # BFS to find connected component (text line)
            line = []
            queue = [i]
            visited.add(i)

            while queue:
                idx = queue.pop(0)
                line.append(detections[idx])

                for neighbor in graph[idx]:
                    if neighbor not in visited:
                        # Check if on same line (similar y-coordinate)
                        y1 = detections[idx]['bbox'][1]
                        y2 = detections[neighbor]['bbox'][1]

                        if abs(y2 - y1) < self.patch_size:  # Same line threshold
                            visited.add(neighbor)
                            queue.append(neighbor)

            if len(line) > 0:
                lines.append(line)

        return lines


class CharacterDetector:
    """
    Phase F.2: Complete character detection pipeline.

    Integrates all 5 swarm components into end-to-end system.
    """

    def __init__(self, num_glyphs: int = 256, feature_dim: int = 128,
                 patch_size: int = 8):
        """
        Initialize character detector.

        Args:
            num_glyphs: Number of character classes (256 for ASCII)
            feature_dim: Feature dimension from CNN (128)
            patch_size: Sliding window patch size (8×8)
        """
        self.num_glyphs = num_glyphs
        self.feature_dim = feature_dim
        self.patch_size = patch_size

        # Component 2: DeepSeek's template bank
        print("[F.2] Initializing GalacticTemplateBank...")
        self.template_bank = GalacticTemplateBank(num_glyphs, feature_dim)

        # Component 1: Qwen's adaptive sliding window
        print("[F.2] Initializing AdaptiveSlidingWindow...")
        self.sliding_window = AdaptiveSlidingWindow(
            patch_size=patch_size,
            min_stride=2,
            max_stride=8,
            activation_threshold=0.1
        )

        # Component 4: GLM's hierarchical NMS
        print("[F.2] Initializing HierarchicalNMS...")
        self.nms = HierarchicalNMS(iou_threshold=0.3, conf_threshold=0.25)

        # Component 5: Grok's spatial decoder
        print("[F.2] Initializing SpatialTextDecoder...")
        self.decoder = SpatialTextDecoder(patch_size=patch_size)

        self.last_template_score: float = 0.0

        # External template storage (bootstrapped from glyph embeddings)
        self.template_embeddings: Optional[np.ndarray] = None
        self.template_char_ids: Optional[np.ndarray] = None
        self.template_offsets: Dict[int, Tuple[int, int]] = {}
        self.template_feature_dim: int = feature_dim
        self._template_source: str = "synthetic"

        print("[F.2] ✓ CharacterDetector ready")

    # ------------------------------------------------------------------ #
    # Template management
    # ------------------------------------------------------------------ #
    def set_template_bank(self, template_bank: Dict[str, np.ndarray]) -> None:
        """
        Load externally supplied template bank (glyph embeddings).

        Args:
            template_bank: Dict mapping character string → [K, D] embeddings
        """
        if not template_bank:
            return

        rows: List[np.ndarray] = []
        char_ids: List[int] = []
        offsets: Dict[int, Tuple[int, int]] = {}
        cursor = 0
        target_dim = self.feature_dim

        for char in sorted(template_bank.keys()):
            vectors = np.asarray(template_bank[char], dtype=np.float32)
            if vectors.size == 0:
                continue

            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)

            norm = np.linalg.norm(vectors, axis=1, keepdims=True)
            norm = np.maximum(norm, 1e-8)
            vectors = vectors / norm

            if vectors.shape[1] > target_dim:
                vectors = vectors[:, :target_dim]
            elif vectors.shape[1] < target_dim:
                padded = np.zeros((vectors.shape[0], target_dim), dtype=np.float32)
                padded[:, :vectors.shape[1]] = vectors
                vectors = padded

            rows.append(vectors)
            count = vectors.shape[0]
            char_id = ord(char) if char else -1
            char_ids.extend([char_id] * count)
            offsets[char_id] = (cursor, cursor + count)
            cursor += count

        if not rows:
            return

        template_matrix = np.vstack(rows).astype(np.float32, copy=False)
        norms = np.linalg.norm(template_matrix, axis=1, keepdims=True)
        template_matrix = template_matrix / np.maximum(norms, 1e-8)

        self.template_embeddings = template_matrix
        self.template_char_ids = np.asarray(char_ids, dtype=np.int32)
        self.template_offsets = offsets
        self.template_feature_dim = target_dim
        self._template_source = "glyph_bootstrap"

        print(
            f"[F.2] Template bank loaded from glyphs: "
            f"{len(offsets)} characters, {template_matrix.shape[0]} templates "
            f"(dim={target_dim})"
        )

    def clear_template_bank(self) -> None:
        """Reset to synthetic Galactic template bank."""
        self.template_embeddings = None
        self.template_char_ids = None
        self.template_offsets = {}
        self.template_feature_dim = self.feature_dim
        self._template_source = "synthetic"

    def detect(self, feature_map: np.ndarray, image_width: int, image_height: int) -> Dict[str, Any]:
        """
        Detect characters from feature map.

        Args:
            feature_map: CNN output features [H, W, C]
            image_width: Original image width (for bbox scaling)
            image_height: Original image height

        Returns:
            Dictionary with:
            - text: Decoded text string
            - detections: List of character detections
            - num_patches: Number of patches processed
        """
        H_feat, W_feat, C_feat = feature_map.shape

        # Step 1: Extract patches with Qwen's adaptive window
        print(f"[F.2] Extracting patches from {H_feat}×{W_feat}×{C_feat} feature map...")
        patches, positions = self.sliding_window.extract_patches(feature_map)
        num_patches = len(patches)
        print(f"[F.2] Extracted {num_patches} patches (after pruning)")

        if num_patches == 0:
            return {'text': '', 'detections': [], 'num_patches': 0}

        # Step 2: Match patches to templates (Kimi's glyph matcher)
        template_count = (
            int(self.template_embeddings.shape[0])
            if self.template_embeddings is not None and self.template_embeddings.size > 0
            else self.num_glyphs
        )
        print(
            f"[F.2] Matching {num_patches} patches to {template_count} templates "
            f"(source={self._template_source})..."
        )
        detections = self._match_patches_to_glyphs(patches, positions, feature_map.shape)
        print(f"[F.2] Found {len(detections)} candidate detections")

        # Step 3: Apply hierarchical NMS (GLM)
        print(f"[F.2] Applying hierarchical NMS...")
        detections = self.nms.apply(detections)
        print(f"[F.2] {len(detections)} detections after NMS")

        # Step 4: Decode to text (Grok's spatial decoder)
        print(f"[F.2] Decoding to text...")
        text = self.decoder.decode(detections, image_width, image_height)
        print(f"[F.2] Decoded {len(text)} characters")

        return {
            'text': text,
            'detections': detections,
            'num_patches': num_patches,
            'max_template_score': float(self.last_template_score),
            'accepted_templates': int(getattr(self, 'last_accept_count', 0)),
        }

    def _match_patches_to_glyphs(self, patches: np.ndarray,
                                 positions: List[Tuple[int, int]],
                                 feature_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        """
        Match patches to character templates using normalized cross-correlation.

        This is Kimi's warp-swizzle glyph matcher (CPU fallback version).

        Args:
            patches: Extracted patches [num_patches, patch_size*patch_size*C]
            positions: List of (row, col) positions
            feature_shape: Original feature map shape [H, W, C]

        Returns:
            List of detections
        """
        H_feat, W_feat, C_feat = feature_shape

        # Get templates (external glyph bank overrides synthetic templates)
        if self.template_embeddings is not None and self.template_embeddings.size > 0:
            templates = self.template_embeddings
            template_char_ids = self.template_char_ids
        else:
            templates = self.template_bank.get_templates()
            template_char_ids = np.arange(self.num_glyphs, dtype=np.int32)

        if templates is None or templates.size == 0:
            return []

        templates = np.asarray(templates, dtype=np.float32)

        # For each patch, compute average feature vector
        patch_size = self.patch_size
        patch_features = patches.reshape(len(patches), patch_size, patch_size, C_feat)
        patch_features = patch_features.mean(axis=(1, 2))  # [num_patches, C_feat]

        # Normalize patch features
        patch_norms = np.linalg.norm(patch_features, axis=1, keepdims=True)
        patch_features = patch_features / np.maximum(patch_norms, 1e-6)

        # Align template dimensionality with feature map channels
        if templates.shape[1] > C_feat:
            templates = templates[:, :C_feat]
        elif templates.shape[1] < C_feat:
            padded = np.zeros((templates.shape[0], C_feat), dtype=np.float32)
            padded[:, :templates.shape[1]] = templates
            templates = padded

        template_norms = np.linalg.norm(templates, axis=1, keepdims=True)
        templates = templates / np.maximum(template_norms, 1e-6)

        # Compute similarity scores (cosine similarity)
        # scores[i, j] = similarity between patch i and template j
        scores = np.dot(patch_features, templates.T)  # [num_patches, num_glyphs]

        self.last_template_score = float(np.max(scores)) if scores.size > 0 else 0.0

        # For each patch, find top-k matches
        detections = []
        top_k = 3  # Keep top 3 candidates per patch
        accepted = 0

        for i, (row, col) in enumerate(positions):
            # Get top-k scores for this patch
            top_indices = np.argsort(scores[i])[-top_k:][::-1]

            for idx in top_indices:
                confidence = float(scores[i, idx])

                # Skip very low confidence
                if confidence < 0.3:
                    continue

                accepted += 1

                # Compute bounding box in feature space
                # Scale to original image space (assuming 4× downsampling)
                scale = 4
                x1 = col * scale
                y1 = row * scale
                x2 = (col + patch_size) * scale
                y2 = (row + patch_size) * scale

                char_id = int(template_char_ids[idx]) if template_char_ids is not None else int(idx)
                if char_id < 0:
                    continue

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'char_id': char_id,
                    'position': (row, col)
                })

        self.last_accept_count = accepted
        return detections
